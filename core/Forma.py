"""Forma de la cara de la fuente (--forma-cara).

inswapper cambia los rasgos pero dibuja la cara con el contorno del destino: una cara ancha sobre una delgada sale
delgada. Aqui, antes de cambiar la cara, se deforma el frame (como "licuar") para que la mandibula y los cachetes del
destino tomen el ancho de la fuente. Ojos, cejas, nariz y boca no se mueven, y el fondo lejos de la cara tampoco.

La forma se guarda como el ancho de la cara a distintas alturas (FORMA_PARES), en distancias entre ojos: al girar la
cabeza se achican igual el ancho y la distancia entre ojos, asi la medida casi no depende del giro.
"""
import cv2
import numpy as np

# Contorno de los 106 puntos de buffalo_l: 0 es el menton; cada punto del lado A (1-16) tiene enfrente el i+16.
# De arriba (altura de los ojos) hacia abajo (cerca del menton):
LADO_A = (1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8)
FORMA_PARES = tuple((a, a + 16) for a in LADO_A)
CONTORNO = (0,) + tuple(i for par in FORMA_PARES for i in par)
FIJOS = tuple(range(33, 106))  # ojos, cejas, nariz y boca

# Cuanto puede ensanchar o angostar como mucho (1.4 = 40 %), para no deformar de mas con una medida mala
CAMBIO_MAXIMO = 1.4
# Solo de los ojos para abajo: el cambio va de 0 a pleno entre estas alturas (en distancias entre ojos)
DESDE = (0.15, 0.6)
# Cara girada: el lado de atras casi no se ve y su contorno es poco confiable. Desde GIRO[0] grados la fuerza
# baja y desde GIRO[1] no se deforma. Para medir la forma de la fuente, como mucho GIRO_MEDIR grados
GIRO = (20.0, 35.0)
GIRO_MEDIR = 25.0
GRADOS_POR_ASIMETRIA = 45.0  # sin "pose" (yaw) se estima con la asimetria: ~0.32 a 14 grados
# La deformacion se apaga del todo a RADIO veces el tamaño de la cara desde el centro (el fondo lejano no se mueve)
RADIO = 1.7
PASO_GRILLA = 4  # el desplazamiento se calcula cada PASO_GRILLA pixeles y se interpola (es suave)


def _ejes(cara):
    izq, der = cara.kps[0].astype(np.float64), cara.kps[1].astype(np.float64)
    d = float(np.linalg.norm(der - izq))
    u = (der - izq) / max(d, 1e-6)
    return (izq + der) / 2, u, np.array([-u[1], u[0]]), d


def _normalizar(puntos, cara):
    centro, u, v, d = _ejes(cara)
    q = puntos.astype(np.float64) - centro
    return np.stack([q @ u, q @ v], 1) / d


def _giro(cara, n=None):
    """Grados que la cabeza esta girada a un lado (yaw); de "pose" si el detector la dio, si no por la asimetria"""
    pose = cara.get("pose")
    if pose is not None and len(pose) == 3 and np.isfinite(pose[1]):
        return abs(float(pose[1]))
    if n is None:
        n = _normalizar(cara.landmark_2d_106, cara)
    return _asimetria(n, _normalizar(cara.kps[2:3], cara)[0, 0]) * GRADOS_POR_ASIMETRIA


def _asimetria(n, nariz_u):
    """0 de frente; crece al girar la cabeza (un lado de la cara se ve mucho mas angosto que el otro)"""
    a = np.array([nariz_u - n[i, 0] for i, _ in FORMA_PARES])
    b = np.array([n[j, 0] - nariz_u for _, j in FORMA_PARES])
    a, b = np.clip(a, 1e-3, None).mean(), np.clip(b, 1e-3, None).mean()
    return float(abs(a - b) / (a + b))


def medir(cara):
    """Forma de una cara: ancho del contorno en cada par de FORMA_PARES, en distancias entre ojos.
    None si la cara no tiene los 106 puntos o esta muy girada para medirla bien"""
    puntos = cara.get("landmark_2d_106")
    if puntos is None or getattr(cara, "kps", None) is None or _ejes(cara)[3] < 2:
        return None
    n = _normalizar(puntos, cara)
    if _giro(cara, n) > GIRO_MEDIR:
        return None
    return np.array([n[j, 0] - n[i, 0] for i, j in FORMA_PARES], np.float32)


def combinar(formas, pesos=None):
    """Forma de varias caras de la misma persona: mediana (o promedio pesado), sin las que no se pudieron medir"""
    validas = [(np.asarray(f, np.float32), p) for f, p in zip(formas, pesos or [1] * len(formas))
               if f is not None and len(f) == len(FORMA_PARES)]
    if not validas:
        return None
    if pesos is None:
        return np.median([f for f, _ in validas], axis=0).astype(np.float32)
    return np.average([f for f, _ in validas], axis=0, weights=[p for _, p in validas]).astype(np.float32)


def valida(forma):
    forma = np.asarray(forma, np.float32) if forma is not None else None
    return forma is not None and forma.shape == (len(FORMA_PARES),) and np.all(np.isfinite(forma)) and np.all(forma > 0)


def _tps(origen, destino):
    """Thin-plate spline que lleva cada punto de "origen" a "destino" (devuelve una funcion para muchos puntos)"""
    n = len(origen)
    def kernel(r2):
        with np.errstate(divide="ignore", invalid="ignore"):
            k = r2 * np.log(r2)
        return np.nan_to_num(k)
    r2 = ((origen[:, None] - origen[None]) ** 2).sum(-1)
    P = np.hstack([np.ones((n, 1)), origen])
    A = np.zeros((n + 3, n + 3))
    A[:n, :n] = kernel(r2) + np.eye(n) * 1e-3  # un poco de suavizado: puntos muy juntos no explotan
    A[:n, n:], A[n:, :n] = P, P.T
    b = np.zeros((n + 3, 2))
    b[:n] = destino
    w = np.linalg.solve(A, b)

    def aplicar(x):
        k = kernel(((x[:, None] - origen[None]) ** 2).sum(-1))
        return k @ w[:n] + w[n] + x @ w[n + 1:]
    return aplicar


def ajustar(frame, forma_src, cara_dst, fuerza=1.0, devolver_mascara=False):
    """Deforma el frame para que el contorno de cara_dst tome el ancho de forma_src (ver medir).
    Devuelve (frame_deformado, cara_deformada[, mascara]): la cara con sus puntos del contorno en el lugar nuevo
    (asi el resto del proceso no vuelve a detectar). La mascara (0 a 1) marca los pixeles que se movieron.
    Si no se puede (sin puntos, cara muy girada, fuerza 0) devuelve el frame y la cara sin cambios"""
    sin_cambio = (frame, cara_dst, None) if devolver_mascara else (frame, cara_dst)
    puntos = cara_dst.get("landmark_2d_106")
    if not fuerza or not valida(forma_src) or puntos is None or getattr(cara_dst, "kps", None) is None:
        return sin_cambio
    centro, u, v, d = _ejes(cara_dst)
    if d < 2:
        return sin_cambio
    n = _normalizar(puntos, cara_dst)
    fuerza = float(fuerza) * float(np.clip((GIRO[1] - _giro(cara_dst, n)) / (GIRO[1] - GIRO[0]), 0, 1))
    if fuerza <= 0:
        return sin_cambio

    ancho_dst = np.array([n[j, 0] - n[i, 0] for i, j in FORMA_PARES])
    cambio = np.asarray(forma_src, np.float64) / np.clip(ancho_dst, 1e-3, None)
    cambio = np.convolve(np.pad(cambio, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")  # sin saltos entre alturas
    cambio = np.clip(cambio, 1 / CAMBIO_MAXIMO, CAMBIO_MAXIMO)
    nuevo = n.copy()
    for k, (i, j) in enumerate(FORMA_PARES):
        altura = (n[i, 1] + n[j, 1]) / 2
        peso = fuerza * np.clip((altura - DESDE[0]) / (DESDE[1] - DESDE[0]), 0, 1)
        factor = 1 + peso * (cambio[k] - 1)
        medio = (n[i, 0] + n[j, 0]) / 2  # cada altura se ensancha desde su propio centro (sirve con la cara girada)
        nuevo[i, 0] = medio + (n[i, 0] - medio) * factor
        nuevo[j, 0] = medio + (n[j, 0] - medio) * factor
    a_frame = lambda q: centro + (q[:, :1] * u + q[:, 1:] * v) * d
    contorno_nuevo = a_frame(nuevo[list(CONTORNO)])
    if np.abs(contorno_nuevo - puntos[list(CONTORNO)]).max() < 0.5:
        return sin_cambio

    # Mapeo inverso (para cada pixel del resultado, de donde sale): del contorno nuevo al de antes; los rasgos y un
    # anillo lejano quedan quietos
    tam = float(np.abs(n[list(CONTORNO)]).max()) * d
    radio = RADIO * max(tam, d)
    angulos = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    anillo = centro + np.stack([np.cos(angulos), np.sin(angulos)], 1) * radio
    fijos = puntos[list(FIJOS)].astype(np.float64)
    llegada = np.vstack([contorno_nuevo, fijos, anillo])
    salida = np.vstack([puntos[list(CONTORNO)].astype(np.float64), fijos, anillo])
    escala = max(radio, 1.0)  # el TPS trabaja en coordenadas chicas (mejor condicionado)
    mover = _tps((llegada - centro) / escala, (salida - llegada) / escala)

    alto, ancho = frame.shape[:2]
    x1, y1 = max(0, int(centro[0] - radio)), max(0, int(centro[1] - radio))
    x2, y2 = min(ancho, int(np.ceil(centro[0] + radio)) + 1), min(alto, int(np.ceil(centro[1] + radio)) + 1)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return sin_cambio
    gy, gx = np.mgrid[y1:y2 + PASO_GRILLA:PASO_GRILLA, x1:x2 + PASO_GRILLA:PASO_GRILLA]
    grilla = (np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64) - centro) / escala
    desp = (mover(grilla) * escala).reshape(gx.shape + (2,)).astype(np.float32)
    desp = cv2.resize(desp, ((gx.shape[1] - 1) * PASO_GRILLA + 1, (gx.shape[0] - 1) * PASO_GRILLA + 1),
                      interpolation=cv2.INTER_LINEAR)[:y2 - y1, :x2 - x1]
    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    lejania = np.hypot(xx - centro[0], yy - centro[1]) / radio
    desp *= np.clip((1 - lejania) / 0.3, 0, 1)[..., None]  # llega a 0 antes del borde del recuadro

    resultado = frame.copy()
    resultado[y1:y2, x1:x2] = cv2.remap(frame, xx + desp[..., 0], yy + desp[..., 1], cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)
    cara = type(cara_dst)(dict(cara_dst))
    nuevos = puntos.astype(np.float32).copy()
    nuevos[list(CONTORNO)] = contorno_nuevo.astype(np.float32)
    cara["landmark_2d_106"] = nuevos
    if not devolver_mascara:
        return resultado, cara
    mascara = np.zeros((alto, ancho), np.float32)
    mascara[y1:y2, x1:x2] = np.clip(np.hypot(desp[..., 0], desp[..., 1]) / 1.0, 0, 1)
    return resultado, cara, mascara
