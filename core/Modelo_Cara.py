import os
import math
import json
import cv2
import numpy as np
from insightface.app.common import Face
from core.Identidades import Registro_Identidades

EXTENSIONES_IMAGEN = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
EXTENSIONES_VIDEO = (".mp4", ".avi", ".mkv", ".mov", ".webm")
CARPETA_MODELOS = os.path.join("src", "modelos_caras")

# Modelo de reconocimiento que genera el vector; si se cambia, los .npy viejos dejan de ser validos
RECONOCIMIENTO = "buffalo_l/w600k_r50"
DIMENSION = 512

# Similitud minima (coseno) con la cara de referencia para considerar que es la misma persona
UMBRAL_MISMA_PERSONA = 0.35

# Tamaño de deteccion al crear el modelo (mayor que el normal para encontrar caras pequeñas en collages)
DET_SIZE_CREACION = 640

# Frames que se sacan de un video por segundo (por defecto) y como maximo por video. Si el video es largo, se
# baja el fps para repartir el maximo en todo el video en lugar de quedarse solo con el principio
FPS_VIDEO = 1.0
MAX_FRAMES_VIDEO = 200


class Modelo_Cara:

    '''
     Un "modelo de cara" es la identidad de una persona guardada como vector (.npy): el promedio de los
    vectores de identidad (normed_embedding) de varias fotos o de un collage de la misma persona.
    inswapper solo usa ese vector de la cara fuente, asi que se puede usar en lugar de una foto.

     Junto al .npy se guardan:
        <nombre>.json   datos del modelo (fotos usadas, caras descartadas, modelo de reconocimiento)
        <nombre>.jpg    miniatura de la mejor cara, para reconocer de quien es
    '''

    @staticmethod
    def rutas_imagenes(rutas):

        '''Expande archivos y carpetas a la lista de imagenes que contienen (sin entrar en subcarpetas)'''

        imagenes = []
        for ruta in rutas:
            if os.path.isdir(ruta):
                for nombre in sorted(os.listdir(ruta)):
                    if nombre.lower().endswith(EXTENSIONES_IMAGEN):
                        imagenes.append(os.path.join(ruta, nombre))
            elif ruta.lower().endswith(EXTENSIONES_IMAGEN):
                imagenes.append(ruta)
        return imagenes


    @staticmethod
    def es_video(ruta):
        return ruta.lower().endswith(EXTENSIONES_VIDEO)


    @staticmethod
    def info_video(ruta):

        '''(fps, cantidad de frames, duracion en segundos) del video'''

        captura = cv2.VideoCapture(ruta)
        try:
            if not captura.isOpened():
                raise ValueError(f"No se pudo abrir el video: {ruta}")
            fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
            return fps, total, total / fps
        finally:
            captura.release()


    @staticmethod
    def fps_efectivo(duracion, fps=FPS_VIDEO, maximo=MAX_FRAMES_VIDEO):

        '''fps con que se sacan los frames: el pedido, o menos si con el pedido se pasaria del maximo'''

        return min(fps, maximo / duracion) if duracion > 0 else fps


    @staticmethod
    def frames_a_extraer(duracion, fps=FPS_VIDEO, maximo=MAX_FRAMES_VIDEO):

        '''Cuantos frames saldran de un video de esa duracion (aproximado, para avisar antes de empezar)'''

        return max(1, min(maximo, math.ceil(duracion * Modelo_Cara.fps_efectivo(duracion, fps, maximo) - 1e-6)))


    @staticmethod
    def frames_video(ruta, fps=FPS_VIDEO, maximo=MAX_FRAMES_VIDEO):

        '''
         Saca frames del video a "fps" por segundo (0.5 = uno cada 2 segundos), en memoria y sin guardarlos
        en disco. Devuelve (imagen, segundo) de cada uno. Se recorre el video con grab(), que no convierte
        el frame a imagen; solo se convierte (retrieve) el frame que toca.
        '''

        fps_video, _, duracion = Modelo_Cara.info_video(ruta)
        intervalo = 1.0 / Modelo_Cara.fps_efectivo(duracion, fps, maximo)
        captura = cv2.VideoCapture(ruta)
        siguiente, indice, sacados = 0.0, 0, 0
        try:
            while sacados < maximo and captura.grab():
                segundo = indice / fps_video
                indice += 1
                if segundo + 1e-6 < siguiente:
                    continue
                while siguiente <= segundo + 1e-6:
                    siguiente += intervalo
                ok, imagen = captura.retrieve()
                if ok:
                    sacados += 1
                    yield imagen, segundo
        finally:
            captura.release()


    @staticmethod
    def crear(rutas, analizador, salida, progreso=print, fps_video=FPS_VIDEO, max_frames=MAX_FRAMES_VIDEO):

        '''
         Detecta todas las caras de las imagenes (un collage puede tener varias) y de los frames de los
        videos (fps_video frames por segundo, como maximo max_frames por video), descarta las que no son
        de la misma persona, promedia sus vectores y guarda <salida>.npy, .json y .jpg.
        "progreso" recibe mensajes de texto (print en la CLI, la barra de estado en la interfaz).
        Devuelve el diccionario de datos guardado en el .json.
        '''

        imagenes = Modelo_Cara.rutas_imagenes(rutas)
        videos = [r for r in rutas if os.path.isfile(r) and Modelo_Cara.es_video(r)]
        if not imagenes and not videos:
            raise ValueError("No se encontro ninguna imagen (.jpg, .png, .bmp, .webp) ni video (.mp4, .avi, .mkv, .mov, .webm)")

        registro = Registro_Identidades(max_ids=1)
        # (cara, miniatura, origen, nombre del video o None). Se guarda la miniatura y no la imagen entera:
        # con cientos de frames de video la memoria se iria a varios GB
        caras = []

        def agregar(imagen, origen, video=None):
            encontradas = [c for c in analizador.obtener_caras(imagen) if registro.calidad_cara(c) > 0]
            caras.extend((c, Modelo_Cara._miniatura(imagen, c), origen, video) for c in encontradas)
            return len(encontradas)

        analizador.preparar_modelo(dimension=DET_SIZE_CREACION)
        try:
            for i, ruta in enumerate(imagenes, 1):
                imagen = cv2.imread(ruta)
                if imagen is None:
                    progreso(f"Foto {i}/{len(imagenes)}: no se pudo leer, omitida")
                    continue
                progreso(f"Foto {i}/{len(imagenes)}: {agregar(imagen, os.path.basename(ruta))} cara(s) util(es)")

            for ruta in videos:
                nombre = os.path.basename(ruta)
                _, _, duracion = Modelo_Cara.info_video(ruta)
                total = Modelo_Cara.frames_a_extraer(duracion, fps_video, max_frames)
                efectivo = Modelo_Cara.fps_efectivo(duracion, fps_video, max_frames)
                if efectivo < fps_video:
                    progreso(f"{nombre}: dura {duracion:.0f} s; para no pasar de {max_frames} frames se usa "
                             f"{efectivo:.2f} fps en lugar de {fps_video:g}")
                for j, (imagen, segundo) in enumerate(Modelo_Cara.frames_video(ruta, fps_video, max_frames), 1):
                    encontradas = agregar(imagen, f"{nombre} {segundo:.1f}s", nombre)
                    progreso(f"{nombre}: frame {j}/~{total} ({segundo:.1f} s): {encontradas} cara(s) util(es)")
        finally:
            analizador.preparar_modelo()

        if not caras:
            raise ValueError("No se detecto ninguna cara util (nitida, de frente y de buen tamaño)")

        # Referencia = la cara que mas se parece a todas las demas; asi un intruso no arrastra el promedio
        vectores = np.array([c.normed_embedding for c, _, _, _ in caras], dtype=np.float32)
        similitudes = vectores @ vectores.T
        referencia = int(np.argmax(similitudes.mean(axis=1)))

        usadas, descartadas = [], []
        for i, (cara, _, origen, _) in enumerate(caras):
            if similitudes[referencia, i] >= UMBRAL_MISMA_PERSONA:
                usadas.append(i)
            else:
                descartadas.append(i)
                progreso(f"Cara descartada ({origen}): parece otra persona "
                         f"(similitud {similitudes[referencia, i]:.2f})")

        if descartadas:
            progreso(f"Aviso: las caras eran de mas de una persona; se uso la que mas se repite "
                     f"({len(usadas)} de {len(caras)} caras)")

        promedio = vectores[usadas].mean(axis=0)
        promedio /= np.linalg.norm(promedio)

        sim_promedio = float(np.mean([similitudes[referencia, i] for i in usadas if i != referencia])) if len(usadas) > 1 else 1.0

        # Miniatura: la cara usada de mejor calidad (frontal, nitida y grande)
        mejor = max(usadas, key=lambda i: registro.calidad_cara(caras[i][0]))
        miniatura = caras[mejor][1]

        base, _ = os.path.splitext(salida)
        os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
        np.save(base + ".npy", promedio.astype(np.float32))
        cv2.imwrite(base + ".jpg", miniatura)

        datos = {
            "nombre": os.path.basename(base),
            "reconocimiento": RECONOCIMIENTO,
            "caras_usadas": len(usadas),
            "caras_descartadas": len(descartadas),
            "fotos": sorted({caras[i][2] for i in usadas if caras[i][3] is None}),
            "similitud_promedio": round(sim_promedio, 3),
        }
        # Por video: cuantos frames aportaron al menos una cara usada
        frames_video = {}
        for video, _ in {(caras[i][3], caras[i][2]) for i in usadas if caras[i][3] is not None}:
            frames_video[video] = frames_video.get(video, 0) + 1
        if frames_video:
            datos["videos"] = [{"nombre": v, "frames": n} for v, n in sorted(frames_video.items())]
            datos["fps_video"] = fps_video
        with open(base + ".json", "w", encoding="utf-8") as archivo:
            json.dump(datos, archivo, ensure_ascii=False, indent=2)

        return datos


    @staticmethod
    def combinar(rutas, salida, progreso=print):

        '''
         Une varios modelos de la MISMA persona en uno solo. Cada modelo pesa segun cuantas caras tiene
        (caras_usadas de su .json; 1 si no tiene .json), asi el resultado equivale a haber creado un modelo
        con todas las fotos juntas. Si algun modelo no se parece a los demas, no combina y avisa.
        Los modelos originales no se modifican. Devuelve el diccionario guardado en el .json.
        '''

        rutas = [os.path.abspath(r) for r in rutas]
        if len(set(rutas)) < 2:
            raise ValueError("Elige al menos dos modelos distintos para combinar")
        base, _ = os.path.splitext(os.path.abspath(salida))
        if base + ".npy" in rutas:
            raise ValueError("El modelo combinado debe tener un nombre distinto a los modelos que se combinan")

        nombres = [os.path.splitext(os.path.basename(r))[0] for r in rutas]
        vectores = np.array([Modelo_Cara.cargar(r).normed_embedding for r in rutas], dtype=np.float32)
        datos_modelos = [Modelo_Cara.datos(r) for r in rutas]
        pesos = np.array([max(1, int(d.get("caras_usadas", 1))) for d in datos_modelos], dtype=np.float32)

        # Misma persona: cada modelo debe parecerse al promedio de los demas
        for i, nombre in enumerate(nombres):
            otros = np.delete(np.arange(len(rutas)), i)
            resto = (vectores[otros] * pesos[otros, None]).sum(axis=0)
            similitud = float(vectores[i] @ (resto / np.linalg.norm(resto)))
            if similitud < UMBRAL_MISMA_PERSONA:
                raise ValueError(f"«{nombre}» no parece la misma persona que los demas modelos "
                                 f"(similitud {similitud:.2f}); no se combino nada")

        promedio = (vectores * pesos[:, None]).sum(axis=0)
        promedio /= np.linalg.norm(promedio)

        similitudes = vectores @ vectores.T
        pares = [float(similitudes[i, j]) for i in range(len(rutas)) for j in range(i + 1, len(rutas))]

        os.makedirs(os.path.dirname(base), exist_ok=True)
        np.save(base + ".npy", promedio.astype(np.float32))

        # Miniatura: la del modelo con mas caras que tenga una
        for i in np.argsort(-pesos):
            miniatura = Modelo_Cara.miniatura(rutas[i])
            if miniatura is not None:
                cv2.imwrite(base + ".jpg", miniatura)
                break

        datos = {
            "nombre": os.path.basename(base),
            "reconocimiento": RECONOCIMIENTO,
            "caras_usadas": int(pesos.sum()),
            "caras_descartadas": int(sum(d.get("caras_descartadas", 0) for d in datos_modelos)),
            "fotos": sorted({f for d in datos_modelos for f in d.get("fotos", [])}),
            "similitud_promedio": round(float(np.mean(pares)), 3),
            "combinado_de": [
                {"nombre": n, "caras": int(p), "similitud_con_resultado": round(float(v @ promedio), 3)}
                for n, p, v in zip(nombres, pesos, vectores)
            ],
        }
        with open(base + ".json", "w", encoding="utf-8") as archivo:
            json.dump(datos, archivo, ensure_ascii=False, indent=2)

        for m in datos["combinado_de"]:
            progreso(f"{m['nombre']}: {m['caras']} cara(s), similitud con el resultado {m['similitud_con_resultado']:.2f}")
        return datos


    @staticmethod
    def _miniatura(imagen, cara, lado=256):

        '''Recorte cuadrado de la cara con margen, a lado x lado'''

        x1, y1, x2, y2 = cara.bbox.astype(int)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        mitad = int(max(x2 - x1, y2 - y1) * 0.8)
        alto, ancho = imagen.shape[:2]
        recorte = imagen[max(0, cy - mitad):min(alto, cy + mitad), max(0, cx - mitad):min(ancho, cx + mitad)]
        return cv2.resize(recorte, (lado, lado), interpolation=cv2.INTER_AREA)


    @staticmethod
    def cargar(ruta):

        '''
         Carga un .npy y lo devuelve como una cara fuente que inswapper puede usar (solo lleva el vector).
        Sin pickle: un .npy con objetos de Python se rechaza.
        '''

        if not os.path.isfile(ruta):
            raise FileNotFoundError(f"No existe el modelo de cara: {ruta}")
        try:
            vector = np.load(ruta, allow_pickle=False)
        except ValueError as error:
            raise ValueError(f"El archivo no es un modelo de cara valido: {ruta}") from error

        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != DIMENSION or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0:
            raise ValueError(f"El archivo no es un modelo de cara valido (se esperaban {DIMENSION} valores): {ruta}")

        datos = Modelo_Cara.datos(ruta)
        if datos.get("reconocimiento", RECONOCIMIENTO) != RECONOCIMIENTO:
            print(f"Aviso: {os.path.basename(ruta)} se creo con {datos['reconocimiento']} y ahora se usa "
                  f"{RECONOCIMIENTO}; conviene volver a crearlo")

        return Face(embedding=vector / np.linalg.norm(vector))


    @staticmethod
    def datos(ruta):

        '''Datos del .json que acompaña al .npy ({} si no existe)'''

        ruta_json = os.path.splitext(ruta)[0] + ".json"
        if not os.path.isfile(ruta_json):
            return {}
        try:
            with open(ruta_json, encoding="utf-8") as archivo:
                return json.load(archivo)
        except (OSError, ValueError):
            return {}


    @staticmethod
    def miniatura(ruta):

        '''Imagen de la miniatura que acompaña al .npy, o None'''

        ruta_jpg = os.path.splitext(ruta)[0] + ".jpg"
        return cv2.imread(ruta_jpg) if os.path.isfile(ruta_jpg) else None


    @staticmethod
    def listar(carpeta=CARPETA_MODELOS):

        '''Rutas de los modelos guardados en la carpeta, ordenados por nombre'''

        if not os.path.isdir(carpeta):
            return []
        return [os.path.join(carpeta, n) for n in sorted(os.listdir(carpeta)) if n.lower().endswith(".npy")]


    @staticmethod
    def eliminar(ruta):

        '''
         Borra un modelo: el .npy y lo que lo acompaña (.json con sus datos y .jpg con la miniatura).
        No se puede deshacer. Devuelve los archivos borrados.
        '''

        if not Modelo_Cara.es_modelo(ruta):
            raise ValueError(f"No es un modelo de cara (.npy): {ruta}")
        base = os.path.splitext(ruta)[0]
        borrados = []
        for extension in (".npy", ".json", ".jpg"):
            if os.path.isfile(base + extension):
                os.remove(base + extension)
                borrados.append(base + extension)
        return borrados


    @staticmethod
    def es_modelo(ruta):
        return ruta.lower().endswith(".npy")
