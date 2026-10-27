import os
import json
from datetime import datetime
from core.Mejorador import validar_minimo, MINIMO_POR_DEFECTO
from core.Proceso_Video import CONTENEDORES, CODECS
from core.Personas import FPS as FPS_PERSONAS, MAX_FRAMES as MAX_PERSONAS, validar_fps, validar_maximo

CARPETA_CONFIGURACIONES = "configuraciones"
VERSION = 1

# Todas las opciones que se guardan, con su valor por defecto (el mismo de la interfaz y de la CLI).
# No se guardan la fuente, el destino ni el fragmento: una configuracion es "como" se procesa, no "que"
DEFECTOS = {
    "asignacion": "posicion",
    "genero": "todos",
    "caras_inclinadas": True,
    "personas_fps": FPS_PERSONAS,
    "personas_max": MAX_PERSONAS,
    "indices_fuente": "0",
    "indices_destino": "0",
    "mascara": "amplia",
    "color": "ninguna",
    "ojos": False,
    "boca": False,
    "nariz": False,
    "frente": False,
    "maquillaje": 0.0,
    "oclusion": False,
    "oclusion_modo": "adaptable",
    "filtro": "None",
    "mejorar": False,
    "fuerza_mejora": 0.8,
    "minimo_mejora": MINIMO_POR_DEFECTO,
    "hilos": 2,
    "contenedor": "mkv",
    "codec": "libx264",
    "zip": False,
    "zip_conservar": False,
    "carpeta_salida": "output",
    "nombre_archivo": "resultado",
}

OPCIONES = {
    "asignacion": ("posicion", "identidad"),
    "genero": ("todos", "mujer", "hombre"),
    "mascara": ("normal", "amplia", "maxima"),
    "color": ("ninguna", "suave", "completa"),
    "oclusion_modo": ("debil", "adaptable", "fuerte"),
    "filtro": ("None", "natural", "strong", "adaptativo"),
    "contenedor": tuple(CONTENEDORES),
    "codec": tuple(CODECS),
}


def _indices(valor):
    texto = str(valor).replace(" ", "")
    if not texto or not all(p.isdigit() for p in texto.split(",")):
        raise ValueError("deben ser numeros separados por comas, como 0,1")
    return texto


def _fraccion(minimo, maximo):
    def validar(valor):
        if isinstance(valor, bool):
            raise ValueError("debe ser un numero")
        numero = float(valor)
        if not minimo <= numero <= maximo:
            raise ValueError(f"debe estar entre {minimo:g} y {maximo:g}")
        return round(numero, 3)
    return validar


def _hilos(valor):
    if isinstance(valor, bool) or int(valor) != float(valor) or not 1 <= int(valor) <= 16:
        raise ValueError("debe ser un entero de 1 a 16")
    return int(valor)


def _si_no(valor):
    if not isinstance(valor, bool):
        raise ValueError("debe ser true o false")
    return valor


def _texto(valor):
    if not isinstance(valor, str) or not valor.strip():
        raise ValueError("debe ser un texto")
    return valor.strip()


VALIDADORES = {
    "indices_fuente": _indices,
    "indices_destino": _indices,
    "zip": _si_no, "zip_conservar": _si_no, "caras_inclinadas": _si_no,
    "ojos": _si_no, "boca": _si_no, "nariz": _si_no, "frente": _si_no, "oclusion": _si_no, "mejorar": _si_no,
    "maquillaje": _fraccion(0.0, 1.0),
    "fuerza_mejora": _fraccion(0.1, 1.0),
    "minimo_mejora": validar_minimo,
    "hilos": _hilos,
    "personas_fps": lambda v: validar_fps(v) if not isinstance(v, bool) else validar_fps("x"),
    "personas_max": lambda v: validar_maximo(v) if not isinstance(v, bool) and float(v) == int(float(v))
                    else validar_maximo("x"),
    "carpeta_salida": _texto,
    "nombre_archivo": _texto,
}


def validar(ajustes):

    '''
     Revisa cada opcion y devuelve (validas, avisos). Las opciones que faltan, las desconocidas y las
    invalidas no estan en "validas" (quien carga deja su valor actual) y se explican en "avisos".
     Asi una configuracion guardada con una version anterior sigue sirviendo.
    '''

    validas, avisos = {}, []
    for clave, valor in ajustes.items():
        if clave not in DEFECTOS:
            avisos.append(f"«{clave}» no es una opcion conocida; se ignora")
            continue
        try:
            if clave in OPCIONES:
                if valor not in OPCIONES[clave]:
                    raise ValueError(f"debe ser una de: {', '.join(OPCIONES[clave])}")
                validas[clave] = valor
            else:
                validas[clave] = VALIDADORES[clave](valor)
        except (ValueError, TypeError) as error:
            avisos.append(f"«{clave}» = {valor!r}: {error}; se deja como estaba")
    faltan = [c for c in DEFECTOS if c not in ajustes]
    if faltan:
        avisos.append(f"No trae {', '.join(faltan)}; se deja como estaba")
    return validas, avisos


def nombre_valido(nombre):

    '''Nombre de archivo seguro (letras, numeros, espacios, - y _), o "" si no queda nada'''

    return "".join(c for c in str(nombre).strip() if c.isalnum() or c in "-_ ").strip()


def ruta(nombre, carpeta=CARPETA_CONFIGURACIONES):

    '''Ruta del .json de una configuracion por su nombre; si "nombre" ya es una ruta a un .json, esa'''

    if nombre.lower().endswith(".json") and (os.sep in nombre or "/" in nombre or os.path.isfile(nombre)):
        return nombre
    return os.path.join(carpeta, nombre_valido(nombre) + ".json")


def guardar(nombre, ajustes, carpeta=CARPETA_CONFIGURACIONES):

    '''Guarda las opciones validas en <carpeta>/<nombre>.json y devuelve la ruta'''

    limpio = nombre_valido(nombre)
    if not limpio:
        raise ValueError("Usa letras, numeros, guiones o espacios en el nombre")
    validas, avisos = validar(ajustes)
    invalidas = [a for a in avisos if not a.startswith("No trae")]
    if invalidas:
        raise ValueError("No se guardo:\n" + "\n".join(invalidas))
    os.makedirs(carpeta, exist_ok=True)
    destino = os.path.join(carpeta, limpio + ".json")
    datos = {"nombre": limpio, "version": VERSION, "guardado": datetime.now().isoformat(timespec="seconds"),
             "ajustes": {c: validas[c] for c in DEFECTOS if c in validas}}
    with open(destino, "w", encoding="utf-8") as archivo:
        json.dump(datos, archivo, ensure_ascii=False, indent=2)
    return destino


def cargar(nombre, carpeta=CARPETA_CONFIGURACIONES):

    '''Lee una configuracion (por nombre o ruta) y devuelve (opciones validas, avisos)'''

    origen = ruta(nombre, carpeta)
    if not os.path.isfile(origen):
        raise FileNotFoundError(f"No existe la configuracion: {origen}")
    try:
        with open(origen, encoding="utf-8") as archivo:
            datos = json.load(archivo)
    except ValueError as error:
        raise ValueError(f"La configuracion no es un JSON valido ({origen}): {error}") from error
    if not isinstance(datos, dict) or not isinstance(datos.get("ajustes"), dict):
        raise ValueError(f"La configuracion no tiene la seccion \"ajustes\": {origen}")
    return validar(datos["ajustes"])


def listar(carpeta=CARPETA_CONFIGURACIONES):

    '''Nombres de las configuraciones guardadas, en orden alfabetico'''

    if not os.path.isdir(carpeta):
        return []
    return sorted((os.path.splitext(n)[0] for n in os.listdir(carpeta) if n.lower().endswith(".json")), key=str.lower)


def eliminar(nombre, carpeta=CARPETA_CONFIGURACIONES):
    os.remove(ruta(nombre, carpeta))
