import os
import json
import time
import shutil
import subprocess
import concurrent.futures
import cv2

# Formato del video temporal mientras se procesa. Los dos aguantan cortes: si el proceso se interrumpe, lo ya
# escrito se puede reproducir y se puede continuar desde ahi (un MP4 normal quedaria inservible).
#   mkv      Matroska
#   mp4frag  MP4 fragmentado (guarda el indice por partes)
CONTENEDORES = {"mkv": ".mkv", "mp4frag": ".mp4"}

# Codificadores H.264: libx264 = CPU (mejor compresion); h264_nvenc = tarjeta NVIDIA; h264_amf = tarjeta AMD
CODECS = ("libx264", "h264_nvenc", "h264_amf")

# Calidad (menor = mejor y mas pesado). 18 se ve practicamente igual al original
CALIDAD = 18

# Cada cuantos segundos se guarda el progreso (progreso.json) para poder continuar
GUARDAR_CADA = 2.0
# Hasta cuantos frames se saltan leyendo (exacto siempre); mas alla se usa la busqueda de OpenCV (rapida)
SALTO_LEYENDO = 300

VERSION_PROGRESO = 1


class Cancelado(Exception):
    pass


class Escritor_Video:

    '''
     Manda los frames directo a ffmpeg por una tuberia (sin guardar imagenes en disco) y los comprime en H.264.
    Sin B-frames (-bf 0) y con un fotograma clave cada 2 s: asi, si el video queda cortado, cada paquete es un
    frame en orden y se puede recortar y continuar sin saltos.
    '''

    def __init__(self, ruta, ancho, alto, fps, contenedor="mkv", codec="libx264", calidad=CALIDAD):

        if contenedor not in CONTENEDORES:
            raise ValueError(f"Contenedor no valido: {contenedor} (usa {', '.join(CONTENEDORES)})")
        if codec not in CODECS:
            raise ValueError(f"Codificador no valido: {codec} (usa {', '.join(CODECS)})")

        self.ruta = ruta
        self.tamano = (alto, ancho)
        gop = max(1, int(round(fps * 2)))

        comando = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ancho}x{alto}", "-r", f"{fps:.6f}", "-i", "-",
            # yuv420p necesita ancho y alto pares
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", codec, "-pix_fmt", "yuv420p", "-g", str(gop), "-bf", "0",
        ]
        if codec == "libx264":
            comando += ["-preset", "medium", "-crf", str(calidad)]
        elif codec == "h264_nvenc":
            comando += ["-preset", "p5", "-rc", "vbr", "-cq", str(calidad), "-b:v", "0"]
        else:  # h264_amf
            comando += ["-quality", "quality", "-rc", "cqp", "-qp_i", str(calidad), "-qp_p", str(calidad)]

        if contenedor == "mkv":
            comando += ["-f", "matroska"]
        else:
            comando += ["-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-f", "mp4"]
        comando.append(ruta)

        self.proceso = subprocess.Popen(comando, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def escribir(self, frame):
        if frame.shape[:2] != self.tamano:
            frame = cv2.resize(frame, (self.tamano[1], self.tamano[0]))
        try:
            self.proceso.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            raise RuntimeError(f"ffmpeg dejo de recibir frames: {self._error()}")

    def cerrar(self):
        '''Cierra la tuberia: ffmpeg termina de escribir y el archivo queda valido'''
        try:
            self.proceso.stdin.close()
        except OSError:
            pass
        codigo = self.proceso.wait()
        if codigo != 0:
            raise RuntimeError(f"ffmpeg fallo al codificar ({codigo}): {self._error()}")

    def _error(self):
        try:
            return self.proceso.stderr.read().decode(errors="replace").strip()[-500:]
        except Exception:
            return ""


def info_video(ruta):
    '''{"fps", "frames", "ancho", "alto", "duracion"} de un video (frames segun el contenedor), o None si no abre'''
    captura = cv2.VideoCapture(ruta)
    try:
        if not captura.isOpened():
            return None
        fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
        frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
        return {"fps": fps, "frames": frames, "ancho": int(captura.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "alto": int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT)), "duracion": frames / fps}
    finally:
        captura.release()


def fragmento_en_frames(inicio, fin, fps, total):
    '''
     Pasa un fragmento en segundos (inicio y fin; None = principio / final del video) a frames [inicio, fin)
    dentro del video. Devuelve None si es el video completo.
    '''
    primero = 0 if inicio is None else int(round(inicio * fps))
    ultimo = total if fin is None else int(round(fin * fps))
    primero, ultimo = max(0, primero), min(total, ultimo)
    if ultimo - primero < 1:
        raise ValueError(f"El fragmento no tiene frames: inicio {inicio}s, fin {fin}s (el video dura {total / fps:.2f}s)")
    if primero == 0 and ultimo >= total:
        return None
    return primero, ultimo


def contar_frames(ruta):
    '''Cantidad de frames (paquetes de video) de un archivo, sin decodificarlo'''
    salida = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", ruta],
        capture_output=True, text=True)
    try:
        return int(salida.stdout.strip().split()[0])
    except (ValueError, IndexError):
        return 0


def recortar(ruta, frames):
    '''Deja solo los primeros "frames" de un video temporal (sin recomprimir). Para partes cortadas de golpe.'''
    base, extension = os.path.splitext(ruta)
    recorte = base + "_recorte" + extension
    formato = ["-f", "matroska"] if extension == ".mkv" else ["-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-f", "mp4"]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", ruta, "-map", "0:v:0", "-c", "copy",
                    "-frames:v", str(frames)] + formato + [recorte], check=True)
    os.replace(recorte, ruta)


def unir(partes, destino, salida, audio=True, tramo=None):

    '''
     Une las partes (sin recomprimir), les pone el audio del video original y guarda el resultado final.
    Si el audio original no entra en el formato de salida, se convierte a AAC.
    tramo = (inicio, duracion) en segundos: se usa solo ese pedazo del audio (video procesado por fragmento).
    '''

    carpeta = os.path.dirname(partes[0])
    lista = os.path.join(carpeta, "lista.txt")
    with open(lista, "w", encoding="utf-8") as archivo:
        for parte in partes:
            archivo.write("file '" + parte.replace("\\", "/").replace("'", "'\\''") + "'\n")

    extension = os.path.splitext(salida)[1].lower()
    rapido = ["-movflags", "+faststart"] if extension in (".mp4", ".mov", ".m4v") else []
    base = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lista]

    if not audio:
        subprocess.run(base + ["-map", "0:v:0", "-c", "copy"] + rapido + [salida], check=True)
        return

    recorte = ["-ss", f"{tramo[0]:.3f}", "-t", f"{tramo[1]:.3f}"] if tramo else []
    comando = base + recorte + ["-i", destino, "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-shortest"]
    resultado = subprocess.run(comando + ["-c:a", "copy"] + rapido + [salida], capture_output=True, text=True)
    if resultado.returncode != 0:
        subprocess.run(comando + ["-c:a", "aac", "-b:a", "192k"] + rapido + [salida], check=True)


class Trabajo_Video:

    '''
     Estado de un video en proceso, para poder continuarlo si se corta. Vive en "<salida>.partes/":
        progreso.json   destino, opciones, partes y estado del modo identidad
        parte_001.mkv   lo procesado en cada sesion (una parte nueva cada vez que se continua)
    '''

    def __init__(self, salida, destino, firma, contenedor, fragmento=None):
        self.carpeta = os.path.abspath(salida) + ".partes"
        self.ruta_progreso = os.path.join(self.carpeta, "progreso.json")
        self.destino = os.path.abspath(destino)
        self.firma = firma
        self.contenedor = contenedor
        # [inicio, fin) en frames si solo se procesa un pedazo del video; None = completo
        self.fragmento = list(fragmento) if fragmento else None
        self.datos = None

    def _huella_destino(self):
        info = os.stat(self.destino)
        return {"ruta": self.destino, "tamano": info.st_size, "modificado": int(info.st_mtime)}

    def leer(self):
        '''Datos del trabajo anterior, o None si no hay'''
        try:
            with open(self.ruta_progreso, encoding="utf-8") as archivo:
                return json.load(archivo)
        except (OSError, ValueError):
            return None

    def pendiente(self):

        '''
         None si no hay un proceso sin terminar. Si lo hay: {"coincide": bool, "hechos": int, "total": int,
        "motivo": str}. "coincide" es False si cambiaron el video destino o las opciones.
        '''

        datos = self.leer()
        if datos is None:
            return None
        hechos = sum(p.get("frames", 0) for p in datos.get("partes", []))
        motivo = ""
        if datos.get("version") != VERSION_PROGRESO:
            motivo = "el proceso anterior es de otra version del programa"
        elif datos.get("destino") != self._huella_destino():
            motivo = "el video destino cambio"
        elif datos.get("contenedor") != self.contenedor:
            motivo = "cambio el formato del video temporal"
        elif datos.get("opciones") != self.firma:
            motivo = "cambiaron las opciones (fuente, ojos, mascara, color, mejora...)"
        elif datos.get("fragmento") != self.fragmento:
            motivo = "cambio el fragmento del video (inicio o fin)"
        return {"coincide": not motivo, "hechos": hechos, "total": datos.get("total", 0), "motivo": motivo}

    def descartar(self):
        shutil.rmtree(self.carpeta, ignore_errors=True)

    def empezar(self, total, fps):
        self.descartar()
        os.makedirs(self.carpeta, exist_ok=True)
        self.datos = {"version": VERSION_PROGRESO, "destino": self._huella_destino(), "opciones": self.firma,
                      "contenedor": self.contenedor, "fragmento": self.fragmento, "total": total, "fps": fps,
                      "partes": [], "estado": None}
        self.guardar()

    def continuar(self):

        '''
         Carga el trabajo anterior y revisa sus partes: una parte que no se cerro bien (corte de luz, cierre
        forzado) se recorta a los frames que de verdad tiene, menos un margen por si el ultimo quedo a medias.
        Devuelve cuantos frames ya estan hechos.
        '''

        self.datos = self.leer()
        partes = []
        for parte in self.datos.get("partes", []):
            ruta = os.path.join(self.carpeta, parte["archivo"])
            if not os.path.isfile(ruta):
                break
            if not parte.get("cerrada"):
                frames = max(0, contar_frames(ruta) - 2)
                if frames > 0:
                    recortar(ruta, frames)
                parte["frames"], parte["cerrada"] = frames, True
            if parte["frames"] > 0:
                partes.append(parte)
            else:
                os.remove(ruta)
        self.datos["partes"] = partes
        self.guardar()
        return sum(p["frames"] for p in partes)

    def nueva_parte(self):
        nombre = f"parte_{len(self.datos['partes']) + 1:03d}{CONTENEDORES[self.contenedor]}"
        self.datos["partes"].append({"archivo": nombre, "frames": 0, "cerrada": False})
        self.guardar()
        return os.path.join(self.carpeta, nombre)

    def avance(self, frames, estado=None, cerrada=False):
        parte = self.datos["partes"][-1]
        parte["frames"], parte["cerrada"] = frames, cerrada
        if estado is not None:
            self.datos["estado"] = estado
        self.guardar()

    def estado(self):
        return self.datos.get("estado")

    def rutas_partes(self):
        return [os.path.join(self.carpeta, p["archivo"]) for p in self.datos["partes"] if p["frames"] > 0]

    def guardar(self):
        temporal = self.ruta_progreso + ".tmp"
        with open(temporal, "w", encoding="utf-8") as archivo:
            json.dump(self.datos, archivo, ensure_ascii=False, indent=1)
        os.replace(temporal, self.ruta_progreso)


def procesar_video(destino, salida, preparar, hilos=2, contenedor="mkv", codec="libx264", calidad=CALIDAD,
                   audio=True, firma=None, reanudar=False, estado=None, progreso=None, cancelado=None,
                   fragmento=None):

    '''
     Procesa un video frame por frame con varios hilos y lo guarda comprimido, pudiendo continuar si se corta.

        preparar(frame, numero)  se llama en orden en el hilo lector; devuelve una funcion sin argumentos que
                                 hace el cambio de cara y devuelve el frame (se ejecuta en un hilo)
        estado                   objeto opcional con guardar() -> dict y cargar(dict) (modo identidad)
        progreso(hechos, total)  se llama al escribir cada frame
        cancelado()              devuelve True para detener; lo hecho queda guardado para continuar
        reanudar                 True = seguir el proceso anterior de esta salida (debe coincidir)
        fragmento                (inicio, fin) en frames, fin sin incluir: solo se procesa y se guarda ese
                                 pedazo, con su audio. None = el video completo. "numero" en preparar() es
                                 el frame del video original

     Solo hay en memoria los frames que estan procesando los hilos (unos pocos), no el video entero.
    Devuelve "completo" o "cancelado". Ctrl+C tambien detiene de forma ordenada.
    '''

    captura = cv2.VideoCapture(destino)
    if not captura.isOpened():
        raise ValueError(f"No se pudo abrir el video: {destino}")

    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    if fragmento is not None:
        inicio, fin = int(fragmento[0]), min(int(fragmento[1]), total_video)
        if not 0 <= inicio < fin:
            captura.release()
            raise ValueError(f"Fragmento invalido: frames {fragmento[0]} a {fragmento[1]} (el video tiene {total_video})")
        fragmento = (inicio, fin)
    else:
        inicio, fin = 0, total_video
    total = fin - inicio
    trabajo = Trabajo_Video(salida, destino, firma, contenedor, fragmento)
    escritor = None
    detenido = False

    try:
        hechos = 0
        if reanudar:
            info = trabajo.pendiente()
            if info is None:
                raise ValueError("No hay un proceso sin terminar para esta salida")
            if not info["coincide"]:
                raise ValueError(f"No se puede continuar: {info['motivo']}")
            hechos = trabajo.continuar()
            if estado is not None and trabajo.estado() is not None:
                estado.cargar(trabajo.estado())
        else:
            trabajo.empezar(total, fps)
        _saltar(captura, inicio + hechos)

        escritor = Escritor_Video(trabajo.nueva_parte(), ancho, alto, fps, contenedor, codec, calidad)
        escritos_parte = 0
        numero = hechos
        pendientes = {}
        siguiente = hechos
        leido_todo = False
        maximo_pendientes = max(2, hilos * 2)
        ultimo_guardado = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=hilos) as ejecutor:
            try:
                while not leido_todo or pendientes:
                    if not detenido and cancelado is not None and cancelado():
                        detenido = True
                    while not leido_todo and not detenido and len(pendientes) < maximo_pendientes:
                        ok, frame = captura.read() if numero < total else (False, None)
                        if not ok:
                            leido_todo = True
                            break
                        pendientes[numero] = ejecutor.submit(preparar(frame, inicio + numero))
                        numero += 1
                    if detenido:
                        leido_todo = True
                    if siguiente not in pendientes:
                        continue

                    escritor.escribir(pendientes.pop(siguiente).result())
                    siguiente += 1
                    escritos_parte += 1
                    if progreso is not None:
                        progreso(siguiente, total)
                    if time.time() - ultimo_guardado >= GUARDAR_CADA:
                        trabajo.avance(escritos_parte, estado.guardar() if estado is not None else None)
                        ultimo_guardado = time.time()
            except KeyboardInterrupt:
                # Ctrl+C: terminar los frames que ya estan en proceso y cerrar bien el archivo
                detenido = True
                for n in sorted(pendientes):
                    if n != siguiente:
                        break
                    escritor.escribir(pendientes.pop(n).result())
                    siguiente += 1
                    escritos_parte += 1
                for futuro in pendientes.values():
                    futuro.cancel()

        escritor.cerrar()
        escritor = None
        trabajo.avance(escritos_parte, estado.guardar() if estado is not None else None, cerrada=True)

        if detenido:
            return "cancelado"
        if siguiente == 0:
            raise ValueError("El video no contiene frames legibles")

        tramo = (inicio / fps, siguiente / fps) if fragmento is not None else None
        unir(trabajo.rutas_partes(), destino, salida, audio, tramo)
        trabajo.descartar()
        return "completo"
    finally:
        captura.release()
        if escritor is not None:
            try:
                escritor.cerrar()
            except Exception:
                pass


def _saltar(captura, frame):
    '''
     Deja la captura lista para leer "frame". Cerca del principio lee y descarta (grab, sin convertir la imagen);
    mas lejos usa la busqueda de OpenCV, que va al fotograma clave anterior y decodifica hasta el pedido.
    '''
    if frame <= 0:
        return
    if frame > SALTO_LEYENDO:
        captura.set(cv2.CAP_PROP_POS_FRAMES, frame)
        if int(round(captura.get(cv2.CAP_PROP_POS_FRAMES))) == frame:
            return
        captura.set(cv2.CAP_PROP_POS_FRAMES, 0)  # la busqueda no quedo exacta: leer desde el principio
    for _ in range(frame):
        if not captura.grab():
            break
