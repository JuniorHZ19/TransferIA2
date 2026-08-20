import os
import argparse
import subprocess
import tempfile

import cv2

from core.Face_Swaper import Swapper_Faces


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov")


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
        help="imagen con la cara de origen",
    )
    process_parser.add_argument(
        "--target",
        required=True,
        help="imagen o video destino",
    )
    process_parser.add_argument(
        "--output",
        required=True,
        help="archivo de salida",
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
        "--occlusion",
        action="store_true",
        help="activa el modelo de oclusion",
    )
    process_parser.add_argument(
        "--occlusion-mode",
        choices=["None", "natural", "strong"],
        default="None",
        help="filtro del modelo de oclusion",
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
        "--verbose",
        action="store_true",
        help="muestra informacion detallada",
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


def _swap_frame(swapper, source_img, frame, args):
    return swapper.cambiar_Faces(
        frame_dst=frame,
        frame_src=source_img,
        index_src=args.source_index,
        index_dst=args.target_index,
        mode=args.occlusion_mode if args.occlusion else "None",
        oclution=args.occlusion,
    )


def _process_image(swapper, source_img, target_path, output_path, args):
    target_img = cv2.imread(target_path)
    if target_img is None:
        raise ValueError(f"No se pudo leer la imagen destino: {target_path}")

    result = _swap_frame(swapper, source_img, target_img, args)
    if not cv2.imwrite(output_path, result):
        raise RuntimeError(f"No se pudo escribir el archivo: {output_path}")


def _process_video(swapper, source_img, target_path, output_path, args):
    capture = cv2.VideoCapture(target_path)
    if not capture.isOpened():
        raise ValueError(f"No se pudo abrir el video: {target_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_video = None

    try:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", dir=output_dir, delete=False).name
        writer = cv2.VideoWriter(
            temp_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("No se pudo crear el video temporal")

        frame_number = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            writer.write(_swap_frame(swapper, source_img, frame, args))
            frame_number += 1
            if args.verbose:
                print(f"Procesando frame {frame_number}", flush=True)

        writer.release()
        if frame_number == 0:
            raise ValueError("El video no contiene frames legibles")

        if args.no_audio:
            os.replace(temp_video, output_path)
            temp_video = None
            return

        ffmpeg_command = [
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", target_path,
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "copy",
            "-c:a", "copy",
            "-shortest",
            output_path,
        ]
        subprocess.run(ffmpeg_command, check=True)
    finally:
        capture.release()
        if temp_video and os.path.exists(temp_video):
            os.remove(temp_video)


def run_cli(args):
    try:
        source_path = os.path.abspath(args.source)
        target_path = os.path.abspath(args.target)
        output_path = os.path.abspath(args.output)

        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"No existe la imagen fuente: {source_path}")
        if not os.path.isfile(target_path):
            raise FileNotFoundError(f"No existe el destino: {target_path}")
        if os.path.exists(output_path) and not args.overwrite:
            raise FileExistsError(f"El archivo ya existe; usa --overwrite: {output_path}")
        if args.threads < 1:
            raise ValueError("--threads debe ser mayor o igual que 1")

        _parse_indices(args.source_index)
        _parse_indices(args.target_index)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        source_img = cv2.imread(source_path)
        if source_img is None:
            raise ValueError(f"No se pudo leer la imagen fuente: {source_path}")

        models_dir = os.path.abspath(args.models_dir)
        model_path = os.path.join(models_dir, "inswapper128.onnx")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No existe el modelo requerido: {model_path}")

        swapper = Swapper_Faces(model_path)
        if os.path.splitext(target_path)[1].lower() in VIDEO_EXTENSIONS:
            _process_video(swapper, source_img, target_path, output_path, args)
        else:
            _process_image(swapper, source_img, target_path, output_path, args)

        print(f"Resultado guardado en: {output_path}")
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=os.sys.stderr)
        return 1


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command != "process":
        parser.print_help()
        return 2

    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())