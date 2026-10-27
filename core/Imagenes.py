import os
import cv2
import numpy as np

# En Windows, cv2.imread y cv2.imwrite no aceptan rutas con tildes o ñ ("canción.jpg", "Señora/..."):
# imread devuelve None y imwrite devuelve True SIN crear el archivo. Estas funciones leen y escriben los bytes
# con numpy (que si acepta esas rutas) y los decodifican o codifican con OpenCV; el resultado es el mismo
# (imdecode tambien aplica la orientacion EXIF de las fotos de celular).


def leer(ruta, flags=cv2.IMREAD_COLOR):

    '''Como cv2.imread: la imagen, o None si no existe o no se puede leer'''

    try:
        datos = np.fromfile(ruta, dtype=np.uint8)
    except OSError:
        return None
    if datos.size == 0:
        return None
    return cv2.imdecode(datos, flags)


def guardar(ruta, imagen, parametros=()):

    '''Como cv2.imwrite (el formato sale de la extension): True si el archivo quedo escrito'''

    ok, datos = cv2.imencode(os.path.splitext(ruta)[1] or ".png", imagen, list(parametros))
    if not ok:
        return False
    try:
        datos.tofile(ruta)
    except OSError:
        return False
    return True
