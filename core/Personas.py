import os
import json
import cv2
import numpy as np
from core.Identidades import Registro_Identidades
from core.Modelo_Cara import Modelo_Cara

VERSION = 1

# Frames que se analizan: FPS por segundo, como maximo MAX_FRAMES repartidos en todo el video (un video de
# 30 minutos se analiza a 1 frame cada 6 segundos, asi el analisis no pasa de minuto y medio)
FPS = 1.0
MAX_FRAMES = 300
# Limites de lo que se puede elegir en la interfaz y la CLI
FPS_RANGO = (0.1, 10.0)
MAX_FRAMES_RANGO = (10, 3000)

# Similitud minima para que una cara sea de una persona ya vista (la misma del modo identidad)
UMBRAL_MISMA = 0.35
# Dos grupos se unen si sus promedios se parecen al menos esto y nunca aparecen en el mismo frame
UMBRAL_UNIR = 0.45
# Tomas por persona ademas del promedio, y cuanto pueden parecerse entre si (para que sean variadas)
TOMAS = 5
TOMAS_PARECIDO_MAX = 0.8


def validar_fps(valor):
    '''Frames por segundo del analisis (numero entre FPS_RANGO) o ValueError'''
    try:
        fps = float(str(valor).replace(",", "."))
    except ValueError:
        raise ValueError(f"los frames por segundo del analisis deben ser un numero: {valor}")
    if not FPS_RANGO[0] <= fps <= FPS_RANGO[1]:
        raise ValueError(f"los frames por segundo del analisis deben estar entre {FPS_RANGO[0]:g} y {FPS_RANGO[1]:g}")
    return fps


def validar_maximo(valor):
    '''Maximo de frames del analisis (entero entre MAX_FRAMES_RANGO) o ValueError'''
    try:
        maximo = int(str(valor).strip())
    except ValueError:
        raise ValueError(f"el maximo de frames del analisis debe ser un numero entero: {valor}")
    if not MAX_FRAMES_RANGO[0] <= maximo <= MAX_FRAMES_RANGO[1]:
        raise ValueError(f"el maximo de frames del analisis debe estar entre {MAX_FRAMES_RANGO[0]} y "
                         f"{MAX_FRAMES_RANGO[1]}")
    return maximo


class Personas:

    '''
     Analisis previo del video para el modo identidad: quienes aparecen, numerados de la persona que MAS aparece
    (0) a la que menos, con varias referencias cada una: el promedio de todas sus caras y hasta TOMAS tomas
    buenas y distintas entre si (de frente, de lado, con otra luz). Asi se las reconoce desde el primer frame y
    en mas angulos que con una sola referencia, y los extras (que aparecen poco) quedan con numeros altos y no
    le quitan el numero a nadie.

     El resultado se guarda en <carpeta de salida>/<nombre del video>_personas/ (personas.json y una miniatura
    persona_N.jpg por persona) y se reutiliza mientras el video y el filtro de genero no cambien.
    '''

    def __init__(self, lista):
        self.lista = lista  # [{"refs": matriz (k, 512), "apariciones": int, "desde": segundos, "miniatura": img}]

    def __len__(self):
        return len(self.lista)

    def refs(self):
        return [p["refs"] for p in self.lista]

    def resumen(self, carpeta=None):
        lineas = []
        for i, p in enumerate(self.lista):
            desde = f"{int(p['desde'] // 60)}:{int(p['desde'] % 60):02d}"
            linea = f"  persona {i}   aparece {p['apariciones']:4d} veces   desde {desde}"
            if carpeta:
                linea += f"   -> {os.path.join(carpeta, f'persona_{i}.jpg')}"
            lineas.append(linea)
        return lineas


    @staticmethod
    def analizar(ruta, detectar, progreso=None, cancelado=None, fps=FPS, maximo=MAX_FRAMES):

        '''
         Recorre el video (fps frames por segundo, como maximo "maximo" repartidos en todo el video) y agrupa las
        caras por persona. detectar(frame) devuelve las caras de un frame (con normed_embedding). Devuelve
        Personas, o None si cancelado() pidio parar.
        '''

        calidad_de = Registro_Identidades(max_ids=1).calidad_cara
        grupos = []  # {"suma", "frames": set, "desde", "tomas": [(calidad, emb)], "mejor": (calidad, miniatura)}
        _, _, duracion = Modelo_Cara.info_video(ruta)
        total = Modelo_Cara.frames_a_extraer(duracion, fps, maximo)

        for numero, (frame, segundo) in enumerate(Modelo_Cara.frames_video(ruta, fps, maximo)):
            if cancelado is not None and cancelado():
                return None
            caras = detectar(frame) or []
            if caras and grupos:
                centros = np.stack([g["suma"] / np.linalg.norm(g["suma"]) for g in grupos])
                sim = np.stack([c.normed_embedding for c in caras]) @ centros.T
            else:
                sim = np.zeros((len(caras), len(grupos)))

            # Cada cara con su grupo mas parecido, sin repetir grupo en el mismo frame (dos caras de un frame
            # son dos personas distintas)
            asignadas, usados = {}, set()
            for idx in np.argsort(-sim, axis=None) if sim.size else []:
                i, k = np.unravel_index(idx, sim.shape)
                if sim[i, k] < UMBRAL_MISMA:
                    break
                if i not in asignadas and k not in usados:
                    asignadas[i] = k
                    usados.add(k)

            for i, cara in enumerate(caras):
                calidad = calidad_de(cara)
                k = asignadas.get(i)
                if k is None:
                    if calidad == 0:
                        continue  # una cara borrosa o de perfil no crea una persona nueva
                    grupos.append({"suma": np.zeros(512), "frames": set(), "desde": segundo, "tomas": [],
                                   "mejor": (0.0, None)})
                    k = len(grupos) - 1
                grupo = grupos[k]
                grupo["frames"].add(numero)
                if calidad > 0:
                    grupo["suma"] += cara.normed_embedding
                    grupo["tomas"].append((calidad, cara.normed_embedding.astype(np.float32)))
                    if calidad > grupo["mejor"][0]:
                        grupo["mejor"] = (calidad, Modelo_Cara._miniatura(frame, cara))
            if progreso is not None:
                progreso(numero + 1, total)

        Personas._unir(grupos)
        grupos.sort(key=lambda g: (-len(g["frames"]), g["desde"]))
        return Personas([{"refs": Personas._referencias(g), "apariciones": len(g["frames"]), "desde": g["desde"],
                          "miniatura": g["mejor"][1]} for g in grupos])


    @staticmethod
    def _unir(grupos):

        '''Une grupos que son la misma persona (promedios parecidos y nunca juntos en un frame)'''

        unido = True
        while unido and len(grupos) > 1:
            unido = False
            centros = np.stack([g["suma"] / np.linalg.norm(g["suma"]) for g in grupos])
            sim = centros @ centros.T
            np.fill_diagonal(sim, -1)
            for idx in np.argsort(-sim, axis=None):
                a, b = np.unravel_index(idx, sim.shape)
                if sim[a, b] < UMBRAL_UNIR:
                    break
                if a < b and not (grupos[a]["frames"] & grupos[b]["frames"]):
                    g, h = grupos[a], grupos.pop(b)
                    g["suma"] += h["suma"]
                    g["frames"] |= h["frames"]
                    g["desde"] = min(g["desde"], h["desde"])
                    g["tomas"] += h["tomas"]
                    g["mejor"] = max(g["mejor"], h["mejor"], key=lambda m: m[0])
                    unido = True
                    break


    @staticmethod
    def _referencias(grupo):

        '''Promedio de todas las caras + hasta TOMAS tomas de buena calidad y distintas entre si'''

        promedio = grupo["suma"] / np.linalg.norm(grupo["suma"])
        elegidas = []
        for _, emb in sorted(grupo["tomas"], key=lambda t: -t[0]):
            if len(elegidas) == TOMAS:
                break
            if all(float(emb @ e) < TOMAS_PARECIDO_MAX for e in elegidas):
                elegidas.append(emb)
        return np.stack([promedio.astype(np.float32)] + elegidas)


    @staticmethod
    def carpeta(video, salida_base):
        nombre = os.path.splitext(os.path.basename(video))[0]
        return os.path.join(os.path.abspath(salida_base), nombre + "_personas")


    @staticmethod
    def _clave(video, genero, fps, maximo):
        info = os.stat(video)
        return {"version": VERSION, "video": os.path.abspath(video), "tamano": info.st_size,
                "modificado": int(info.st_mtime), "genero": genero, "fps": fps, "maximo": maximo}


    def guardar(self, carpeta, clave):
        os.makedirs(carpeta, exist_ok=True)
        for nombre in os.listdir(carpeta):  # miniaturas de un analisis anterior
            if nombre.startswith("persona_") and nombre.endswith(".jpg"):
                os.remove(os.path.join(carpeta, nombre))
        for i, p in enumerate(self.lista):
            if p["miniatura"] is not None:
                cv2.imwrite(os.path.join(carpeta, f"persona_{i}.jpg"), p["miniatura"])
        datos = {"clave": clave, "personas": [{"apariciones": p["apariciones"], "desde": round(p["desde"], 2),
                                               "refs": np.round(p["refs"], 6).tolist()} for p in self.lista]}
        with open(os.path.join(carpeta, "personas.json"), "w", encoding="utf-8") as archivo:
            json.dump(datos, archivo, ensure_ascii=False)


    @staticmethod
    def cargar(carpeta, clave):

        '''El analisis guardado, o None si no hay o si es de otro video, otro genero u otra version'''

        ruta = os.path.join(carpeta, "personas.json")
        try:
            with open(ruta, encoding="utf-8") as archivo:
                datos = json.load(archivo)
        except (OSError, ValueError):
            return None
        if datos.get("clave") != clave:
            return None
        lista = []
        for i, p in enumerate(datos.get("personas", [])):
            mini = os.path.join(carpeta, f"persona_{i}.jpg")
            lista.append({"refs": np.asarray(p["refs"], dtype=np.float32), "apariciones": int(p["apariciones"]),
                          "desde": float(p["desde"]), "miniatura": cv2.imread(mini) if os.path.isfile(mini) else None})
        return Personas(lista)


    @staticmethod
    def obtener(video, salida_base, detectar, genero="todos", progreso=None, cancelado=None, fps=FPS,
                maximo=MAX_FRAMES):

        '''
         El analisis del video: el guardado si sigue valiendo, si no lo hace y lo guarda. Devuelve
        (Personas, carpeta, reutilizado) o (None, carpeta, False) si se cancelo.
        '''

        carpeta = Personas.carpeta(video, salida_base)
        clave = Personas._clave(video, genero, fps, maximo)
        personas = Personas.cargar(carpeta, clave)
        if personas is not None:
            return personas, carpeta, True
        personas = Personas.analizar(video, detectar, progreso, cancelado, fps, maximo)
        if personas is None:
            return None, carpeta, False
        personas.guardar(carpeta, clave)
        return personas, carpeta, False
