import os
import cv2
from concurrent.futures import ThreadPoolExecutor
from core.Modelo_Cara import EXTENSIONES_IMAGEN

# Se agrega al nombre de la carpeta destino para formar la de salida: fotos/ -> output/fotos_output/
SUFIJO_SALIDA = "_output"


def imagenes_carpeta(carpeta):

    '''Imagenes de la carpeta, ordenadas por nombre (sin entrar en subcarpetas)'''

    if not os.path.isdir(carpeta):
        return []
    return [os.path.join(carpeta, n) for n in sorted(os.listdir(carpeta), key=str.lower)
            if n.lower().endswith(EXTENSIONES_IMAGEN) and os.path.isfile(os.path.join(carpeta, n))]


def carpeta_salida(carpeta, salida_base):

    '''Carpeta donde se guardan los resultados: <salida_base>/<nombre de la carpeta>_output'''

    nombre = os.path.basename(os.path.normpath(os.path.abspath(carpeta)))
    return os.path.join(os.path.abspath(salida_base), nombre + SUFIJO_SALIDA)


def procesar_carpeta(imagenes, destino, preparar, hilos=1, progreso=None, cancelado=None):

    '''
     Cambia las caras de cada imagen y la guarda en "destino" con el mismo nombre de archivo.

     preparar(imagen, numero) se llama en este hilo y en orden (asi el modo identidad registra a las personas
    en el orden de las imagenes) y devuelve una funcion sin argumentos que hace el cambio; esas funciones
    corren en paralelo en "hilos" hilos. Solo se cargan unas pocas imagenes a la vez en memoria.
     Una imagen que no se puede leer o que falla se anota y se sigue con las demas.
     progreso(hechas, total) se llama al terminar cada imagen; cancelado() = True detiene antes de la siguiente
    tanda (las ya guardadas quedan).

     Devuelve (cantidad guardada, [(nombre, error)], cancelado)
    '''

    os.makedirs(destino, exist_ok=True)
    hilos = max(1, int(hilos))
    guardadas, errores, total = 0, [], len(imagenes)

    with ThreadPoolExecutor(max_workers=hilos) as ejecutor:
        for inicio in range(0, total, hilos * 2):
            if cancelado is not None and cancelado():
                return guardadas, errores, True

            tanda = []
            for numero in range(inicio, min(total, inicio + hilos * 2)):
                ruta = imagenes[numero]
                imagen = cv2.imread(ruta)
                if imagen is None:
                    errores.append((os.path.basename(ruta), "no se pudo leer"))
                    continue
                try:
                    tanda.append((ruta, ejecutor.submit(preparar(imagen, numero))))
                except Exception as error:
                    errores.append((os.path.basename(ruta), str(error)))

            for ruta, futuro in tanda:
                nombre = os.path.basename(ruta)
                try:
                    if not cv2.imwrite(os.path.join(destino, nombre), futuro.result()):
                        raise RuntimeError("no se pudo escribir")
                    guardadas += 1
                except Exception as error:
                    errores.append((nombre, str(error)))
                if progreso is not None:
                    progreso(guardadas + len(errores), total)

    return guardadas, errores, False
