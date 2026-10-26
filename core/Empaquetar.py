import os
import zipfile

try:
    import pyzipper
except ImportError:  # se avisa al usarlo, no al arrancar el programa
    pyzipper = None


def disponible():
    return pyzipper is not None


def ruta_zip(salida, es_carpeta=False):

    '''El .zip que corresponde a un resultado: output/resultado.mp4 -> output/resultado.zip;
    una carpeta, output/fotos_output -> output/fotos_output.zip'''

    salida = os.path.normpath(salida)
    return (salida if es_carpeta else os.path.splitext(salida)[0]) + ".zip"


def empaquetar(archivos, destino, contrasena, conservar=False, carpeta_en_zip=""):

    '''
     Guarda los archivos en un .zip cifrado con AES-256 (se abre con 7-Zip o WinRAR; el Explorador de Windows
    no abre AES). Antes de borrar nada comprueba que el zip se lea con la contrasena y que cada archivo este
    completo. Con conservar=False borra los archivos sueltos despues. "carpeta_en_zip" es la carpeta dentro
    del zip (para una carpeta de imagenes). Devuelve la ruta del zip.
     Los videos e imagenes ya vienen comprimidos: se guardan sin volver a comprimir (mas rapido, mismo tamaño).
    '''

    if pyzipper is None:
        raise RuntimeError("Para guardar en ZIP con contraseña falta instalar pyzipper: pip install pyzipper")
    if not contrasena:
        raise ValueError("Escribe una contraseña para el ZIP")
    if not archivos:
        raise ValueError("No hay archivos para guardar en el ZIP")

    clave = contrasena.encode("utf-8")
    temporal = destino + ".tmp"
    nombres = [os.path.join(carpeta_en_zip, os.path.basename(a)) if carpeta_en_zip else os.path.basename(a)
               for a in archivos]
    try:
        with pyzipper.AESZipFile(temporal, "w", compression=zipfile.ZIP_STORED,
                                 encryption=pyzipper.WZ_AES) as zip_:
            zip_.setpassword(clave)
            zip_.setencryption(pyzipper.WZ_AES, nbits=256)
            for archivo, nombre in zip(archivos, nombres):
                zip_.write(archivo, nombre.replace(os.sep, "/"))

        # Verificacion: se abre con la contrasena, cada archivo pasa su control (CRC) y tiene su tamaño
        with pyzipper.AESZipFile(temporal) as zip_:
            zip_.setpassword(clave)
            if zip_.testzip() is not None:
                raise RuntimeError("el ZIP quedó dañado")
            tamanos = {i.filename: i.file_size for i in zip_.infolist()}
        for archivo, nombre in zip(archivos, nombres):
            if tamanos.get(nombre.replace(os.sep, "/")) != os.path.getsize(archivo):
                raise RuntimeError(f"{os.path.basename(archivo)} no quedó completo en el ZIP")
        os.replace(temporal, destino)
    finally:
        if os.path.exists(temporal):
            os.remove(temporal)

    if not conservar:
        for archivo in archivos:
            os.remove(archivo)
    return destino
