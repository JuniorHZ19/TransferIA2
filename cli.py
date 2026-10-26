import os
import argparse
import subprocess

import cv2

from core.Face_Swaper import Swapper_Faces
from core.Identidades import Registro_Identidades
from core.Modelo_Cara import Modelo_Cara, FPS_VIDEO, MAX_FRAMES_VIDEO
from core.Mejorador import validar_minimo, MINIMO_POR_DEFECTO
from core.Proceso_Carpeta import imagenes_carpeta, carpeta_salida, procesar_carpeta, SUFIJO_SALIDA
from core import Configuraciones
from core.Empaquetar import empaquetar, ruta_zip, disponible as zip_disponible
from core.Personas import (Personas, validar_fps, validar_maximo, FPS as PERSONAS_FPS,
                           MAX_FRAMES as PERSONAS_MAX)
from core.Proceso_Video import (procesar_video, Trabajo_Video, CONTENEDORES, CODECS, CALIDAD, info_video,
                                fragmento_en_frames)


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov")
PROGRESS_BAR_WIDTH = 30


def _fuerza(valor):
    """Numero entre 0 y 1 (para --makeup y --enhance)"""
    try:
        numero = float(valor)
    except ValueError:
        raise argparse.ArgumentTypeError(f"debe ser un numero entre 0 y 1: {valor}")
    if not 0.0 <= numero <= 1.0:
        raise argparse.ArgumentTypeError(f"debe estar entre 0 y 1: {valor}")
    return numero


def _tiempo(valor):
    """Momento del video en segundos: 90, 90.5, 1:30, 1:30.5 o 0:01:30"""
    try:
        partes = [float(p) for p in valor.strip().split(":")]
        if not 1 <= len(partes) <= 3 or any(p < 0 for p in partes):
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(f"tiempo invalido (usa segundos, m:ss o h:mm:ss): {valor}")
    segundos = 0.0
    for parte in partes:
        segundos = segundos * 60 + parte
    return segundos


def _minimo_gpen(valor):
    """Tamaño minimo de cara para GPEN: pixeles (96) o porcentaje del lado mas corto de la imagen (25%)"""
    try:
        return validar_minimo(valor)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Transfer-IA: cambio facial por consola",
    )
    subparsers = parser.add_subparsers(dest="command")

    process_parser = subparsers.add_parser(
        "process",
        help="procesa una imagen o un video",
    )
    process_parser.add_argument(
        "--source",
        required=True,
        help="imagen con la cara de origen, o modelo(s) de cara .npy separados por comas "
             "(con varios .npy, --source-index 0 es el primero, 1 el segundo...)",
    )
    process_parser.add_argument(
        "--target",
        required=True,
        help="imagen o video destino, o una carpeta con imagenes (cada una se procesa por separado)",
    )
    process_parser.add_argument(
        "--output",
        required=True,
        help=f"archivo de salida; si --target es una carpeta, la carpeta donde se crea <nombre>{SUFIJO_SALIDA} "
             f"con las imagenes procesadas (mismos nombres de archivo)",
    )
    process_parser.add_argument(
        "--models-dir",
        default="models",
        help="carpeta de modelos; por defecto: models",
    )
    process_parser.add_argument(
        "--source-index",
        default="0",
        help="indices de caras origen, por ejemplo: 0,1",
    )
    process_parser.add_argument(
        "--target-index",
        default="0",
        help="indices de caras destino, por ejemplo: 0,1",
    )
    process_parser.add_argument(
        "--asignacion",
        choices=["posicion", "identidad"],
        default="posicion",
        help="como emparejar caras destino: por posicion (izquierda a derecha) o por identidad "
             "(--target-index = orden en que cada persona aparece por primera vez); por defecto: posicion",
    )
    process_parser.add_argument(
        "--personas-fps",
        type=float,
        default=PERSONAS_FPS,
        help=f"con --asignacion identidad en video: frames por segundo del analisis de personas (por defecto "
             f"{PERSONAS_FPS:g})",
    )
    process_parser.add_argument(
        "--personas-max",
        type=int,
        default=PERSONAS_MAX,
        help=f"con --asignacion identidad en video: maximo de frames del analisis, repartidos en todo el video "
             f"(por defecto {PERSONAS_MAX})",
    )
    process_parser.add_argument(
        "--genero",
        choices=["todos", "mujer", "hombre"],
        default="todos",
        help="solo cambia caras destino de este genero; por defecto: todos",
    )
    process_parser.add_argument(
        "--occlusion",
        action="store_true",
        help="activa el modelo de oclusion: el pelo, las manos y los objetos delante de la cara vuelven del destino",
    )
    process_parser.add_argument(
        "--oclusion-modo",
        choices=["debil", "adaptable", "fuerte"],
        default="adaptable",
        help="con --occlusion, cuanta cara puede quedar sobre lo que la tapa en el borde: debil (como antes), "
             "adaptable (por defecto: no pinta cara sobre el objeto) o fuerte (objeto limpio hasta su borde)",
    )
    process_parser.add_argument(
        "--nitidez", "--occlusion-mode",
        dest="occlusion_mode",
        choices=["None", "natural", "strong", "adaptativo"],
        default="None",
        help="filtro de nitidez de la cara cambiada, con o sin --occlusion (adaptativo = iguala el detalle y "
             "el grano del destino). --occlusion-mode es el nombre anterior y sigue funcionando",
    )
    process_parser.add_argument(
        "--eyes",
        action="store_true",
        help="cambia el rostro completo pero conserva los ojos originales del destino",
    )
    process_parser.add_argument(
        "--mouth",
        action="store_true",
        help="cambia el rostro completo pero conserva la boca original del destino (labios, dientes, lengua)",
    )
    process_parser.add_argument(
        "--enhance",
        type=_fuerza,
        nargs="?",
        const=0.8,
        default=0.0,
        metavar="FUERZA",
        help="mejora la cara cambiada con GPEN (models/gpen_bfr_256.onnx): mas nitidez en piel, ojos y "
             "dientes; FUERZA de 0 a 1 (sin numero: 0.8; menos = mas natural y mas parecido con la fuente); "
             "tarda mas por cada cara",
    )
    process_parser.add_argument(
        "--enhance-min-size",
        type=_minimo_gpen,
        default=MINIMO_POR_DEFECTO,
        metavar="TAMANO",
        help="con --enhance: caras mas chicas que esto no se mejoran. En pixeles de ancho (96, por defecto) o en "
             "porcentaje del lado mas corto de la imagen (25%%: util en imagenes chicas donde la cara ocupa casi "
             "todo). 0 = mejorar todas",
    )
    process_parser.add_argument(
        "--nose",
        action="store_true",
        help="cambia el rostro completo pero conserva la nariz original del destino",
    )
    process_parser.add_argument(
        "--remove-bangs",
        action="store_true",
        help="quita el cerquillo de la fuente: la frente (desde un poco arriba de las cejas) queda la del "
             "destino",
    )
    process_parser.add_argument(
        "--makeup",
        type=_fuerza,
        default=0.0,
        metavar="FUERZA",
        help="conserva el maquillaje o la pintura de cara del destino, de 0 a 1 (mas fuerza = mas "
             "maquillaje y menos parecido con la fuente); por defecto: 0 (desactivado)",
    )
    process_parser.add_argument(
        "--color",
        choices=["ninguna", "suave", "completa"],
        default="ninguna",
        help="corrige el color de la cara cambiada para igualar la piel del destino: suave (tono y luz media) "
             "o completa (tambien el contraste); por defecto: ninguna",
    )
    process_parser.add_argument(
        "--face-mask",
        choices=["normal", "amplia", "maxima"],
        default="amplia",
        help="zona de la cara cambiada que se pega: normal (la de insightface), amplia o maxima "
             "(cubren mas mejillas, menton y frente); por defecto: amplia",
    )
    process_parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="numero de hilos solicitado; por defecto: 2",
    )
    process_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="reemplaza el archivo de salida si ya existe",
    )
    process_parser.add_argument(
        "--no-audio",
        action="store_true",
        help="no copia el audio original del video",
    )
    process_parser.add_argument(
        "--container",
        choices=list(CONTENEDORES),
        default="mkv",
        help="formato del video temporal mientras se procesa (los dos aguantan cortes y permiten continuar): "
             "mkv o mp4frag (MP4 fragmentado); por defecto: mkv. El resultado final es el de --output",
    )
    process_parser.add_argument(
        "--video-codec",
        choices=list(CODECS),
        default="libx264",
        help="codificador H.264: libx264 (CPU, mejor compresion), h264_nvenc (NVIDIA) o h264_amf (AMD); "
             "por defecto: libx264",
    )
    process_parser.add_argument(
        "--crf",
        type=int,
        default=CALIDAD,
        help=f"calidad del video (menor = mejor y mas pesado; 18 = casi igual al original); por defecto: {CALIDAD}",
    )
    process_parser.add_argument(
        "--start",
        type=_tiempo,
        default=None,
        help="solo videos: procesa desde este momento (segundos, m:ss o h:mm:ss); el resultado es solo el "
             "fragmento, con su audio. Por defecto: desde el principio",
    )
    process_parser.add_argument(
        "--end",
        type=_tiempo,
        default=None,
        help="solo videos: procesa hasta este momento (segundos, m:ss o h:mm:ss). Por defecto: hasta el final",
    )
    process_parser.add_argument(
        "--resume",
        action="store_true",
        help="continua un video que se corto (Ctrl+C, cierre o apagon) desde donde quedo; "
             "deben ser el mismo destino, la misma salida y las mismas opciones",
    )
    process_parser.add_argument(
        "--restart",
        action="store_true",
        help="descarta un proceso sin terminar de esta salida y empieza de cero",
    )
    process_parser.add_argument(
        "--verbose",
        action="store_true",
        help="muestra informacion detallada",
    )
    process_parser.add_argument(
        "--zip",
        action="store_true",
        help="guarda el resultado en un .zip con contraseña (AES-256; se abre con 7-Zip o WinRAR) y borra el "
             "archivo suelto. La contraseña se pide al empezar, sin mostrarla",
    )
    process_parser.add_argument(
        "--zip-password",
        metavar="CONTRASEÑA",
        help="contraseña del --zip sin preguntarla (queda en el historial de comandos: mejor no usarla)",
    )
    process_parser.add_argument(
        "--zip-conservar",
        action="store_true",
        help="con --zip, conserva también el archivo suelto",
    )
    process_parser.add_argument(
        "--config",
        metavar="NOMBRE",
        help=f"usa una configuracion guardada ({Configuraciones.CARPETA_CONFIGURACIONES}/<NOMBRE>.json, o la ruta "
             "a un .json); las opciones escritas en el comando tienen prioridad",
    )
    process_parser.add_argument(
        "--save-config",
        metavar="NOMBRE",
        help=f"guarda las opciones de este comando como configuracion en "
             f"{Configuraciones.CARPETA_CONFIGURACIONES}/<NOMBRE>.json (la reemplaza si existe)",
    )
    parser.set_defaults(_process_parser=process_parser)

    modelo_parser = subparsers.add_parser(
        "crear-modelo",
        help="crea un modelo de cara (.npy) promediando fotos, un collage o frames de videos de la misma persona",
    )
    modelo_parser.add_argument(
        "--fotos",
        nargs="+",
        default=[],
        help="imagenes y/o carpetas con fotos de la misma persona (un collage puede tener varias caras)",
    )
    modelo_parser.add_argument(
        "--video",
        nargs="+",
        default=[],
        help="uno o varios videos de la misma persona; se sacan frames a --fps por segundo",
    )
    modelo_parser.add_argument(
        "--fps",
        type=float,
        default=FPS_VIDEO,
        help=f"frames por segundo que se sacan de cada --video (0.5 = uno cada 2 segundos); por defecto: {FPS_VIDEO:g}. "
             f"Como maximo {MAX_FRAMES_VIDEO} frames por video: si es largo, se reparten en todo el video",
    )
    modelo_parser.add_argument(
        "--output",
        required=True,
        help="archivo .npy a crear; junto a el se guardan una miniatura .jpg y un .json con los datos",
    )
    modelo_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="reemplaza el modelo si ya existe",
    )

    combinar_parser = subparsers.add_parser(
        "combinar-modelos",
        help="une varios modelos de cara (.npy) de la MISMA persona en uno solo",
    )
    combinar_parser.add_argument(
        "--modelos",
        required=True,
        nargs="+",
        help="modelos .npy de la misma persona (al menos dos); cada uno pesa segun cuantas caras tiene",
    )
    combinar_parser.add_argument(
        "--output",
        required=True,
        help="modelo .npy a crear (nombre distinto a los que se combinan); los originales no se modifican",
    )
    combinar_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="reemplaza el modelo de salida si ya existe",
    )
    return parser


def _parse_indices(value):
    try:
        indices = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise ValueError("Los indices deben ser enteros separados por comas") from error

    if not indices or any(index < 0 for index in indices):
        raise ValueError("Los indices deben ser enteros mayores o iguales que cero")
    return indices


def _swap_frame(swapper, caras_src, frame, args):
    return swapper.cambiar_Faces(
        frame_dst=frame,
        caras_src=caras_src,
        index_src=args.source_index,
        index_dst=args.target_index,
        mode=args.occlusion_mode,
        oclution=args.occlusion,
        oclusion_modo=args.oclusion_modo,
        genero=args.genero,
        ojos=args.eyes,
        boca=args.mouth,
        nariz=args.nose,
        frente=args.remove_bangs,
        mascara=args.face_mask,
        color=args.color,
        maquillaje=args.makeup,
        mejorar=args.enhance,
        mejorar_minimo=args.enhance_min_size,
    )


class _ModoIdentidad:
    """Estado del modo por identidad: caras fuente detectadas una vez, mapa destino->fuente y registro."""

    def __init__(self, swapper, caras_src, args, output_path, fps=None, personas=None):
        indices_src = _parse_indices(args.source_index)
        indices_dst = _parse_indices(args.target_index)
        if len(indices_src) != len(indices_dst):
            raise ValueError("En modo identidad --source-index y --target-index deben tener la misma cantidad de indices")
        if len(set(indices_dst)) != len(indices_dst):
            raise ValueError("En modo identidad --target-index no puede repetir identidades")

        self.swapper = swapper
        self.args = args
        self.fps = fps
        self.caras_src = caras_src
        self.mapa = dict(zip(indices_dst, indices_src))
        self.registro = Registro_Identidades(max_ids=max(indices_dst) + 1)
        if personas is not None and len(personas):
            # Personas del analisis previo del video (0 = la que mas aparece): fijas desde el primer frame
            if max(indices_dst) >= len(personas):
                raise ValueError(f"En el video hay {len(personas)} persona(s) (de 0 a {len(personas) - 1}) y "
                                 f"--target-index pide la {max(indices_dst)}")
            self.registro.fijar(personas.refs())
        base, _ = os.path.splitext(output_path)
        self.carpeta_miniaturas = base + "_identidades"

    def guardar(self):
        """Estado para continuar un video cortado (identidades ya registradas)"""
        return self.registro.guardar()

    def cargar(self, datos):
        self.registro.cargar(datos)

    def registrar(self, frame, frame_number=0):
        """Llamar en orden de frames. Devuelve las caras detectadas para no detectarlas otra vez,
        o None si ya estan todas las identidades registradas."""
        if self.registro.completo():
            return None

        caras = self.swapper.detectar_caras(frame, self.args.genero)
        for identidad, cara in self.registro.registrar(caras):
            self._informar(frame, identidad, cara, frame_number)
        return caras

    def _informar(self, frame, identidad, cara, frame_number):
        os.makedirs(self.carpeta_miniaturas, exist_ok=True)
        x1, y1, x2, y2 = cara.bbox.astype(int)
        margen = int((x2 - x1) * 0.3)
        miniatura = frame[max(0, y1 - margen):y2 + margen, max(0, x1 - margen):x2 + margen]
        ruta = os.path.join(self.carpeta_miniaturas, f"destino_{identidad}.jpg")
        cv2.imwrite(ruta, miniatura)

        momento = ""
        if self.fps:
            segundos = frame_number / self.fps
            momento = f" en {int(segundos // 60)}:{int(segundos % 60):02d} (frame {frame_number})"
        fuente = self.mapa.get(identidad)
        destino = f"recibe la fuente {fuente}" if fuente is not None else "no se cambia"
        print(f"\nIdentidad destino {identidad} registrada{momento}: {destino} -> {ruta}", flush=True)

    def cambiar(self, frame, caras_dst, refs):
        return self.swapper.cambiar_Faces_por_identidad(
            frame_dst=frame,
            caras_src=self.caras_src,
            mapa=self.mapa,
            refs=refs,
            caras_dst=caras_dst,
            mode=self.args.occlusion_mode,
            oclution=self.args.occlusion,
            oclusion_modo=self.args.oclusion_modo,
            genero=self.args.genero,
            ojos=self.args.eyes,
            boca=self.args.mouth,
            nariz=self.args.nose,
            frente=self.args.remove_bangs,
            mascara=self.args.face_mask,
            color=self.args.color,
            maquillaje=self.args.makeup,
            mejorar=self.args.enhance,
            mejorar_minimo=self.args.enhance_min_size,
        )


def _process_image(swapper, caras_src, target_path, output_path, args):
    target_img = cv2.imread(target_path)
    if target_img is None:
        raise ValueError(f"No se pudo leer la imagen destino: {target_path}")

    if args.asignacion == "identidad":
        modo = _ModoIdentidad(swapper, caras_src, args, output_path)
        caras_dst = modo.registrar(target_img)
        result = modo.cambiar(target_img, caras_dst, modo.registro.copia_refs())
    else:
        result = _swap_frame(swapper, caras_src, target_img, args)
    if not cv2.imwrite(output_path, result):
        raise RuntimeError(f"No se pudo escribir el archivo: {output_path}")


def _show_progress(current, total, label="Procesando"):
    if total > 0:
        percentage = min(current / total, 1.0)
        completed_width = int(PROGRESS_BAR_WIDTH * percentage)
        bar = "#" * completed_width + "-" * (PROGRESS_BAR_WIDTH - completed_width)
        message = f"\r{label}: [{bar}] {percentage * 100:6.2f}% ({current}/{total})"
    else:
        message = f"\r{label}: {current} frames"

    print(message, end="", flush=True)


# Opciones que no cambian el resultado: no cuentan para decidir si un video cortado se puede continuar
_OPCIONES_SIN_EFECTO = {"command", "target", "output", "overwrite", "threads", "verbose", "resume", "restart",
                        "no_audio", "models_dir", "start", "end", "config", "save_config", "zip", "zip_password", "zip_conservar",
                        "_process_parser"}  # el fragmento se revisa aparte (Trabajo_Video)


def _firma_opciones(args):
    """Opciones que definen el resultado (fuente, ojos, mascara, color...), para el progreso del video"""
    firma = {k: v for k, v in sorted(vars(args).items()) if k not in _OPCIONES_SIN_EFECTO}
    firma["source"] = ",".join(os.path.abspath(p.strip()) for p in args.source.split(",") if p.strip())
    if not args.occlusion or firma.get("oclusion_modo") == "debil":
        firma.pop("oclusion_modo", None)  # "debil" es el de antes: los videos empezados antes siguen
    if not args.occlusion and args.occlusion_mode != "None":
        firma["nitidez_sin_oclusion"] = True  # antes el filtro sin --occlusion no se aplicaba: no mezclar videos
    if args.asignacion == "identidad":
        firma["personas"] = "analisis"  # antes se numeraban por orden de aparicion: no mezclar videos
    else:
        firma.pop("personas_fps", None)  # el analisis solo existe con identidad
        firma.pop("personas_max", None)
    if not firma.get("remove_bangs"):
        firma.pop("remove_bangs", None)  # asi los videos empezados antes de esta opcion se pueden continuar
    return firma


def _process_video(swapper, caras_src, target_path, output_path, args):
    info = info_video(target_path)
    if info is None:
        raise ValueError(f"No se pudo abrir el video: {target_path}")
    fragmento = fragmento_en_frames(args.start, args.end, info["fps"], info["frames"])
    if fragmento is not None:
        print(f"Fragmento: {fragmento[0] / info['fps']:.2f}s a {fragmento[1] / info['fps']:.2f}s "
              f"({(fragmento[1] - fragmento[0]) / info['fps']:.2f}s, {fragmento[1] - fragmento[0]} frames)", flush=True)
    trabajo = Trabajo_Video(output_path, target_path, _firma_opciones(args), args.container, fragmento)
    pendiente = trabajo.pendiente()
    reanudar = False

    if args.resume:
        if pendiente is None:
            raise ValueError("No hay un proceso sin terminar para esta salida; quita --resume")
        if not pendiente["coincide"]:
            raise ValueError(f"No se puede continuar: {pendiente['motivo']}. Usa las mismas opciones, "
                             f"o --restart para empezar de cero")
        reanudar = True
        print(f"Continuando desde el frame {pendiente['hechos']} de {pendiente['total']}", flush=True)
    elif pendiente is not None:
        if not args.restart:
            raise ValueError(f"Hay un proceso sin terminar de esta salida ({pendiente['hechos']} de "
                             f"{pendiente['total']} frames). Usa --resume para continuar o --restart para empezar de cero")
        trabajo.descartar()

    fps = info["fps"]
    modo = None
    if args.asignacion == "identidad":
        # Analisis previo: quienes aparecen, numerados de la que mas aparece (0) a la que menos
        personas, carpeta, reutilizado = Personas.obtener(
            target_path, os.path.dirname(output_path), lambda frame: swapper.detectar_caras(frame, args.genero),
            args.genero, progreso=lambda h, t: _show_progress(h, t, "Analizando personas"),
            fps=args.personas_fps, maximo=args.personas_max)
        print("\n" + ("Personas (analisis guardado):" if reutilizado else "Personas en el video:"))
        print("\n".join(personas.resumen(carpeta)) or "  ninguna cara encontrada", flush=True)
        modo = _ModoIdentidad(swapper, caras_src, args, output_path, fps, personas)

    def preparar(frame, numero):
        if modo is not None:
            # El registro de identidades se hace aqui, en orden de lectura; los hilos solo usan una copia
            caras_dst = modo.registrar(frame, numero)
            refs = modo.registro.copia_refs()
            return lambda: modo.cambiar(frame, caras_dst, refs)
        return lambda: _swap_frame(swapper, caras_src, frame, args)

    def progreso(hechos, total):
        if args.verbose:
            print(f"\nProcesando frame {hechos}", flush=True)
        else:
            _show_progress(hechos, total)

    resultado = procesar_video(
        target_path, output_path, preparar,
        hilos=args.threads,
        contenedor=args.container,
        codec=args.video_codec,
        calidad=args.crf,
        audio=not args.no_audio,
        firma=trabajo.firma,
        reanudar=reanudar,
        estado=modo,
        progreso=progreso,
        fragmento=fragmento,
    )
    print()
    if resultado == "cancelado":
        raise _Detenido(f"Proceso detenido; lo hecho quedo guardado en {trabajo.carpeta}. "
                        f"Para continuar, repite el comando con --resume")


def _process_folder(swapper, caras_src, target_path, destino, args):
    """Cada imagen de la carpeta se procesa con las mismas opciones y se guarda con su nombre en destino"""
    imagenes = imagenes_carpeta(target_path)
    if not imagenes:
        raise ValueError(f"La carpeta no tiene imagenes (.jpg, .png, .bmp, .webp): {target_path}")
    print(f"{len(imagenes)} imagen(es) -> {destino}", flush=True)
    modo = _ModoIdentidad(swapper, caras_src, args, destino) if args.asignacion == "identidad" else None

    def preparar(imagen, numero):
        if modo is not None:
            caras_dst = modo.registrar(imagen)
            refs = modo.registro.copia_refs()
            return lambda: modo.cambiar(imagen, caras_dst, refs)
        return lambda: _swap_frame(swapper, caras_src, imagen, args)

    guardadas, errores, _ = procesar_carpeta(imagenes, destino, preparar, hilos=args.threads,
                                             progreso=lambda h, t: _show_progress(h, t, "Imagenes"))
    print()
    for nombre, error in errores:
        print(f"Aviso: {nombre}: {error}", file=os.sys.stderr)
    print(f"{guardadas} de {len(imagenes)} imagen(es) guardada(s)")
    if not guardadas:
        raise RuntimeError("No se pudo procesar ninguna imagen")


class _Detenido(Exception):
    pass


def _cargar_caras_fuente(swapper, source):
    """Caras fuente, una sola vez para todo el proceso: de uno o varios .npy, o detectadas en una imagen"""
    partes = [p.strip() for p in source.split(",") if p.strip()]
    if partes and all(Modelo_Cara.es_modelo(p) for p in partes):
        caras = [Modelo_Cara.cargar(os.path.abspath(p)) for p in partes]
        for ruta in partes:
            datos = Modelo_Cara.datos(ruta)
            detalle = f" ({datos['caras_usadas']} caras)" if "caras_usadas" in datos else ""
            print(f"Modelo de cara: {os.path.basename(ruta)}{detalle}")
        return caras

    source_path = os.path.abspath(source)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"No existe la imagen fuente: {source_path}")
    source_img = cv2.imread(source_path)
    if source_img is None:
        raise ValueError(f"No se pudo leer la imagen fuente: {source_path}")
    caras = swapper.detectar_caras(source_img)
    if not caras:
        raise ValueError("No se detecto ninguna cara en la imagen fuente")
    return caras


def run_crear_modelo(args):
    try:
        salida = os.path.abspath(args.output)
        if not salida.lower().endswith(".npy"):
            salida += ".npy"
        if os.path.exists(salida) and not args.overwrite:
            raise FileExistsError(f"El modelo ya existe; usa --overwrite: {salida}")
        if not args.fotos and not args.video:
            raise ValueError("Indica --fotos, --video o ambos")
        for ruta in args.fotos + args.video:
            if not os.path.exists(ruta):
                raise FileNotFoundError(f"No existe: {ruta}")
        for ruta in args.fotos:
            if Modelo_Cara.es_video(ruta):
                raise ValueError(f"Es un video; pasalo con --video: {ruta}")
        for ruta in args.video:
            if not Modelo_Cara.es_video(ruta):
                raise ValueError(f"No es un video (.mp4, .avi, .mkv, .mov, .webm); las fotos van con --fotos: {ruta}")
        if not 0 < args.fps <= 60:
            raise ValueError("--fps debe ser mayor que 0 y como maximo 60")

        swapper = Swapper_Faces(os.path.join(os.path.abspath("models"), "inswapper128.onnx"))
        datos = Modelo_Cara.crear(args.fotos + args.video, swapper.get_Analizador(), salida, fps_video=args.fps)

        similitud = datos["similitud_promedio"]
        calidad = "buena" if similitud >= 0.5 else "aceptable" if similitud >= 0.4 else "baja"
        origen = [f"{len(datos['fotos'])} foto(s)"] if datos["fotos"] else []
        origen += [f"{v['frames']} frame(s) de {v['nombre']}" for v in datos.get("videos", [])]
        print(f"Modelo creado con {datos['caras_usadas']} cara(s) de {' y '.join(origen)} -> {salida}")
        if datos["caras_usadas"] > 1:
            print(f"Similitud promedio entre caras: {similitud:.2f} ({calidad})")
        else:
            print("Aviso: solo se uso una cara; con 3 a 10 fotos distintas el parecido es mas estable")
        print(f"Miniatura: {os.path.splitext(salida)[0]}.jpg")
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=os.sys.stderr)
        return 1


def run_combinar_modelos(args):
    try:
        salida = os.path.abspath(args.output)
        if not salida.lower().endswith(".npy"):
            salida += ".npy"
        if os.path.exists(salida) and not args.overwrite:
            raise FileExistsError(f"El modelo ya existe; usa --overwrite: {salida}")

        datos = Modelo_Cara.combinar(args.modelos, salida)
        print(f"Modelo combinado con {datos['caras_usadas']} cara(s) de {len(datos['combinado_de'])} modelos -> {salida}")
        print(f"Similitud promedio entre los modelos: {datos['similitud_promedio']:.2f}")
        if datos["similitud_promedio"] >= 0.9:
            print("Aviso: los modelos son casi iguales; combinarlos aporta poco")
        return 0
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"Error: {error}", file=os.sys.stderr)
        return 1


def _pedir_contrasena():
    """Contraseña del ZIP, escrita dos veces y sin mostrarla en pantalla"""
    import getpass
    for _ in range(3):
        primera = getpass.getpass("Contraseña del ZIP: ")
        if not primera:
            print("La contraseña no puede estar vacía")
            continue
        if getpass.getpass("Repite la contraseña: ") == primera:
            return primera
        print("Las contraseñas no coinciden")
    raise ValueError("No se ingresó la contraseña del ZIP")


def run_cli(args):
    try:
        target_path = os.path.abspath(args.target)
        output_path = os.path.abspath(args.output)

        es_carpeta = os.path.isdir(target_path)
        if es_carpeta:
            # La salida es una carpeta nueva dentro de --output: fotos/ -> <output>/fotos_output/
            output_path = carpeta_salida(target_path, output_path)
            if os.path.normcase(output_path) == os.path.normcase(target_path):
                raise ValueError("La carpeta de salida no puede ser la misma carpeta de las imagenes")
            if os.path.isdir(output_path) and os.listdir(output_path) and not args.overwrite:
                raise FileExistsError(f"La carpeta de salida ya tiene archivos; usa --overwrite: {output_path}")
        elif not os.path.isfile(target_path):
            raise FileNotFoundError(f"No existe el destino: {target_path}")
        elif os.path.exists(output_path) and not args.overwrite and not args.resume:
            raise FileExistsError(f"El archivo ya existe; usa --overwrite: {output_path}")
        if args.asignacion == "identidad":
            validar_fps(args.personas_fps)
            validar_maximo(args.personas_max)
        if args.threads < 1:
            raise ValueError("--threads debe ser mayor o igual que 1")
        if args.start is not None and args.end is not None and args.end <= args.start:
            raise ValueError("--end debe ser mayor que --start")

        _parse_indices(args.source_index)
        _parse_indices(args.target_index)
        contrasena = None
        if args.zip:
            if not zip_disponible():
                raise RuntimeError("Para --zip falta instalar pyzipper: pip install pyzipper")
            destino_zip = ruta_zip(output_path, es_carpeta=es_carpeta)
            if os.path.exists(destino_zip) and not args.overwrite:
                raise FileExistsError(f"El ZIP ya existe; usa --overwrite: {destino_zip}")
            contrasena = args.zip_password or _pedir_contrasena()
        os.makedirs(output_path if es_carpeta else os.path.dirname(output_path), exist_ok=True)

        models_dir = os.path.abspath(args.models_dir)
        model_path = os.path.join(models_dir, "inswapper128.onnx")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No existe el modelo requerido: {model_path}")

        swapper = Swapper_Faces(model_path)
        caras_src = _cargar_caras_fuente(swapper, args.source)
        if es_carpeta:
            if args.start is not None or args.end is not None:
                print("Aviso: --start y --end son solo para videos; en imagenes se ignoran", flush=True)
            _process_folder(swapper, caras_src, target_path, output_path, args)
        elif os.path.splitext(target_path)[1].lower() in VIDEO_EXTENSIONS:
            _process_video(swapper, caras_src, target_path, output_path, args)
        else:
            if args.start is not None or args.end is not None:
                print("Aviso: --start y --end son solo para videos; en imagenes se ignoran", flush=True)
            _process_image(swapper, caras_src, target_path, output_path, args)

        if contrasena:
            if es_carpeta:
                archivos = [os.path.join(output_path, n) for n in sorted(os.listdir(output_path))
                            if os.path.isfile(os.path.join(output_path, n))]
                empaquetar(archivos, destino_zip, contrasena, conservar=args.zip_conservar,
                           carpeta_en_zip=os.path.basename(output_path))
                if not args.zip_conservar:
                    try:
                        os.rmdir(output_path)
                    except OSError:
                        pass
            else:
                empaquetar([output_path], destino_zip, contrasena, conservar=args.zip_conservar)
            print(f"Resultado guardado en: {destino_zip} (con contraseña; se abre con 7-Zip o WinRAR)")
            if args.zip_conservar:
                print(f"Tambien se conservo: {output_path}")
            return 0
        print(f"Resultado guardado en: {output_path}")
        return 0
    except _Detenido as aviso:
        print(aviso, file=os.sys.stderr)
        return 130
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=os.sys.stderr)
        return 1


# Clave de core/Configuraciones.py -> opcion de "process". "mejorar" y "fuerza_mejora" van juntas en --enhance;
# carpeta_salida y nombre_archivo son de la interfaz (en la CLI la salida es --output)
_CONFIG_A_OPCION = {
    "asignacion": "asignacion", "genero": "genero",
    "personas_fps": "personas_fps", "personas_max": "personas_max",
    "indices_fuente": "source_index", "indices_destino": "target_index",
    "mascara": "face_mask", "color": "color",
    "ojos": "eyes", "boca": "mouth", "nariz": "nose", "frente": "remove_bangs",
    "zip": "zip", "zip_conservar": "zip_conservar",
    "maquillaje": "makeup", "oclusion": "occlusion", "oclusion_modo": "oclusion_modo", "filtro": "occlusion_mode",
    "minimo_mejora": "enhance_min_size", "hilos": "threads",
    "contenedor": "container", "codec": "video_codec",
}


def _opciones_de_config(nombre):
    """Valores por defecto de "process" sacados de una configuracion guardada"""
    ajustes, avisos = Configuraciones.cargar(nombre)
    for aviso in avisos:
        if not aviso.startswith("No trae"):
            print(f"Aviso ({nombre}): {aviso}", file=os.sys.stderr)
    opciones = {_CONFIG_A_OPCION[c]: v for c, v in ajustes.items() if c in _CONFIG_A_OPCION}
    if "mejorar" in ajustes:
        opciones["enhance"] = ajustes.get("fuerza_mejora", 0.8) if ajustes["mejorar"] else 0.0
    return opciones


def _config_de_opciones(args):
    """Opciones de "process" con las claves de core/Configuraciones.py (para --save-config)"""
    ajustes = {c: getattr(args, o) for c, o in _CONFIG_A_OPCION.items()}
    ajustes["mejorar"] = args.enhance > 0
    ajustes["fuerza_mejora"] = args.enhance if args.enhance >= 0.1 else Configuraciones.DEFECTOS["fuerza_mejora"]
    return ajustes


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "process":
        if args.config:
            # La configuracion pasa a ser el valor por defecto; lo escrito en el comando la reemplaza
            try:
                args._process_parser.set_defaults(**_opciones_de_config(args.config))
            except (FileNotFoundError, ValueError) as error:
                print(f"Error: {error}", file=os.sys.stderr)
                return 1
            args = parser.parse_args()
            print(f"Configuracion: {args.config}", flush=True)
        if args.save_config:
            try:
                print(f"Configuracion guardada en {Configuraciones.guardar(args.save_config, _config_de_opciones(args))}")
            except (OSError, ValueError) as error:
                print(f"Error: {error}", file=os.sys.stderr)
                return 1
        return run_cli(args)
    if args.command == "crear-modelo":
        return run_crear_modelo(args)
    if args.command == "combinar-modelos":
        return run_combinar_modelos(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())