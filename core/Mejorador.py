import os
import threading
import cv2
import numpy as np
import onnxruntime as ort
from core.xseg import FFHQ_TEMPLATE

# Mejorador de caras GPEN (version ligera de 256 px). Reconstruye piel, ojos y dientes que se pierden porque
# inswapper genera la cara a 128 px. Modelo: gpen_bfr_256.onnx de facefusion-assets (models-3.0.0).
RUTA_GPEN = os.path.join("models", "gpen_bfr_256.onnx")
TAMANO = 256

# Cuanto de la cara mejorada se usa (0 a 1). 1 = solo GPEN; menos deja algo de la cara original, que se ve
# mas natural (GPEN tiende a dejar la piel muy lisa).
MEZCLA = 0.8

# Caras mas angostas que esto (px) no se mejoran: inswapper ya las genera a 128 px, mas de lo que ocupan en la
# imagen, asi que GPEN no agrega nada y al reducirlas deja detalle exagerado y baja el parecido
ANCHO_MINIMO = 96

# Tamaño minimo de cara para usar GPEN, como texto: "96" = 96 px de ancho; "25%" = el 25 % del lado mas corto
# de la imagen (para imagenes chicas donde la cara ocupa casi todo, aunque mida pocos px); "0" = siempre
MINIMO_POR_DEFECTO = str(ANCHO_MINIMO)


def validar_minimo(valor):
    '''Normaliza el tamaño minimo ("96", "96px", "25%", "0.25" no) o lanza ValueError'''
    texto = str(valor).strip().lower().replace(" ", "")
    porcentaje = texto.endswith("%")
    numero = texto[:-1] if porcentaje else texto[:-2] if texto.endswith("px") else texto
    try:
        numero = float(numero)
    except ValueError:
        raise ValueError(f"tamano minimo invalido: {valor} (usa pixeles, como 96, o porcentaje, como 25%)")
    if numero < 0 or (porcentaje and numero > 100):
        raise ValueError(f"tamano minimo fuera de rango: {valor} (pixeles >= 0, o porcentaje de 0% a 100%)")
    return f"{numero:g}%" if porcentaje else f"{numero:g}"


def ancho_minimo(minimo, alto, ancho):
    '''Ancho minimo de cara en px para una imagen de alto x ancho, segun el texto de validar_minimo'''
    minimo = validar_minimo(minimo)
    if minimo.endswith("%"):
        return float(minimo[:-1]) / 100.0 * min(alto, ancho)
    return float(minimo)

# Difuminado del borde del recorte que se pega, en fraccion del tamaño (como la mascara de caja de facefusion)
DIFUMINADO = 0.3


class Mejorador_Cara:

    '''
     Mejora una cara ya cambiada: la alinea con sus 5 puntos (plantilla FFHQ) a 256 px, la pasa por GPEN y
    la devuelve a su lugar con una mascara de caja de bordes difuminados. El modelo se carga una sola vez y
    solo cuando se usa por primera vez.
    '''

    _SESION = None
    _LOCK = threading.Lock()

    def __init__(self, ruta=RUTA_GPEN):
        if self.__class__._SESION is None:
            with self.__class__._LOCK:
                if self.__class__._SESION is None:
                    if not os.path.isfile(ruta):
                        raise FileNotFoundError(
                            f"No se encontro el modelo del mejorador: {ruta}. Descargalo de "
                            "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gpen_bfr_256.onnx "
                            "y guardalo en la carpeta models/")
                    self.__class__._SESION = ort.InferenceSession(
                        ruta, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.entrada = self.__class__._SESION.get_inputs()[0].name


    def mejorar(self, frame, cara, mezcla=MEZCLA, devolver_mascara=False, minimo=MINIMO_POR_DEFECTO):

        '''
         Devuelve el frame con la cara (cara.kps) mejorada por GPEN, mezclada con la original segun "mezcla".
        Con devolver_mascara=True devuelve (frame, mascara): cuanto de cada pixel viene de GPEN (0 a 1, del
        tamaño del frame; 0.0 si no se cambio nada).
        minimo: caras mas chicas que esto no se mejoran ("96" px, o "25%" del lado mas corto de la imagen)
        '''

        resultado, mascara = self._mejorar(frame, cara, mezcla, minimo)
        return (resultado, mascara) if devolver_mascara else resultado

    def _mejorar(self, frame, cara, mezcla, minimo):

        kps = getattr(cara, "kps", None)
        if kps is None or np.linalg.norm(kps[1] - kps[0]) < 2:
            return frame, 0.0
        bbox = getattr(cara, "bbox", None)
        if bbox is not None and bbox[2] - bbox[0] < ancho_minimo(minimo, *frame.shape[:2]):
            return frame, 0.0

        plantilla = (FFHQ_TEMPLATE * TAMANO).astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), plantilla, method=cv2.LMEDS)
        if M is None:
            return frame, 0.0
        recorte = cv2.warpAffine(frame, M, (TAMANO, TAMANO), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)

        # BGR 0-255 -> RGB -1..1, NCHW
        x = recorte[:, :, ::-1].astype(np.float32) / 255.0
        x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
        y = self.__class__._SESION.run(None, {self.entrada: x})[0][0]
        y = (np.clip(y, -1, 1) + 1) / 2
        mejorada = (y.transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(np.float32)

        # Mascara de caja con bordes difuminados
        difuminado = int(TAMANO * 0.5 * DIFUMINADO)
        area = max(difuminado // 2, 1)
        mascara = np.ones((TAMANO, TAMANO), dtype=np.float32)
        mascara[:area, :] = mascara[-area:, :] = 0
        mascara[:, :area] = mascara[:, -area:] = 0
        if difuminado > 0:
            mascara = cv2.GaussianBlur(mascara, (0, 0), difuminado * 0.25)

        # Volver al frame solo en el rectangulo que ocupa la cara
        alto, ancho = frame.shape[:2]
        IM = cv2.invertAffineTransform(M)
        esquinas = cv2.transform(np.array([[[0, 0], [TAMANO, 0], [0, TAMANO], [TAMANO, TAMANO]]], np.float32), IM)[0]
        x1, y1 = np.floor(esquinas.min(0)).astype(int)
        x2, y2 = np.ceil(esquinas.max(0)).astype(int)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(ancho, x2), min(alto, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return frame, 0.0
        IM[:, 2] -= (x1, y1)
        tam = (x2 - x1, y2 - y1)

        # Si la cara en el frame es mas chica que 256 px, suavizar antes de reducir: sin esto la reduccion
        # deja picos (aliasing) que se ven como detalle exagerado y bajan el parecido en caras pequeñas
        escala = float(np.sqrt(abs(np.linalg.det(IM[:, :2]))))
        if escala < 1.0:
            mejorada = cv2.GaussianBlur(mejorada, (0, 0), max(0.3, (1.0 / escala - 1.0) / 2.0))
        cara_vuelta = cv2.warpAffine(mejorada, IM, tam, flags=cv2.INTER_LINEAR, borderValue=0.0)
        mascara = cv2.warpAffine(mascara, IM, tam, flags=cv2.INTER_LINEAR, borderValue=0.0)[..., None] * float(mezcla)

        resultado = frame.copy()
        zona = resultado[y1:y2, x1:x2].astype(np.float32)
        resultado[y1:y2, x1:x2] = np.clip(zona * (1 - mascara) + cara_vuelta * mascara, 0, 255).astype(np.uint8)
        mascara_frame = np.zeros((alto, ancho), dtype=np.float32)
        mascara_frame[y1:y2, x1:x2] = mascara[..., 0]
        return resultado, mascara_frame
