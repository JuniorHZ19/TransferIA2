import cv2
import numpy as np

# Detalle de la piel de la fuente (--detalle-piel): solo rasgos (pecas, lunares, arrugas finas, poros), no sombras
# (el costado de la nariz, los surcos de la sonrisa, las ojeras) ni barba. Todo es pixel / cierre morfologico por
# canal: el cierre rellena lo oscuro mas angosto que un disco con la piel de alrededor, asi el lunar conserva su
# marron y no depende de la luz de la foto. Una sombra es un escalon ancho y el cierre no la toca.
#  RASGO_FINO: lo mas fino (poros, pecas chicas, arrugas finas) pasa siempre
#  RASGO_GRANDE: hasta el tamaño de un lunar, pero solo lo que tiene forma de mancha: la curvatura (hessiana) en las
#  dos direcciones es parecida (FORMA_MANCHA, cociente entre las dos); un surco curva solo hacia los costados. Se mide
#  en cada pixel, a ESCALAS_CHICAS (pecas) y ESCALAS_GRANDES (lunares): pecas juntas siguen siendo pecas
#  FONDO_CHICO / FONDO_GRANDE: de cada mancha se aplica solo lo que sobresale de lo que tiene debajo (un poro sobre
#  el surco no se lleva el surco), y si sobresale al menos MANCHA_CONTRASTE
#  ESQUINAS_106: en las comisuras de los ojos y las aletas de la nariz (ESQUINA_RADIO) lo oscuro es sombra
#  BARBA: en un hombre, desde la base de la nariz hacia abajo, donde los puntos oscuros cubren mas de esa fraccion
#  de la piel es pelo (barba, bigote) y no se lleva
# (antes era pixel / desenfoque, que llevaba tambien las sombras; en 2 caras de IA con 158 rasgos pintados: pecas
# tenues 77 -> 76 %, lunares 65 -> 79 %, costado de la nariz 3,1 -> 1,0, ojeras 2,8 -> 1,6, barba 3,9 -> 1,1)
RASGO_FINO = 0.015
RASGO_GRANDE = 0.09
FORMA_MANCHA = (0.15, 0.45)
ESCALAS_CHICAS = (0.008, 0.016)
ESCALAS_GRANDES = (0.03,)
FONDO_CHICO = 0.045
FONDO_GRANDE = 0.1
MANCHA_CONTRASTE = (0.035, 0.06)
ESQUINAS_106 = (35, 39, 89, 93, 77, 83)
ESQUINA_RADIO = 0.08
BARBA = (0.15, 0.3)
# Proporcion minima (0,6 = un pixel puede oscurecerse hasta 40 %); solo se oscurece (el maximo no se usa)
DETALLE_LIMITE = (0.6, 1.4)
# Frente: los 106 puntos no la cubren; se agregan puntos sobre las cejas, FRENTE_ALTO distancias entre ojos mas arriba
FRENTE_ALTO = 0.55
# Contorno de la cara (de la sien izquierda al menton y a la sien derecha), cejas, ojos, nariz (fosas) y boca
CONTORNO_106 = (1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0, 24, 23, 22, 21, 20, 19, 18, 32, 31, 30, 29,
                28, 27, 26, 25, 17)
CEJAS_106 = (range(43, 52), range(97, 106))
OJOS_106 = ((35, 41, 40, 42, 39, 37, 33, 36), (93, 96, 94, 95, 89, 90, 87, 91))
FOSAS_106 = (77, 78, 79, 80, 85, 84, 83)
BOCA_106 = range(52, 64)
PUPILAS_106 = (34, 38, 88, 92)  # se mueven con la mirada: no sirven para deformar la piel
# Cuanto se agranda cada parte que no es piel (en distancias entre ojos), y el difuminado del borde
EXCLUIR_OJOS = 0.08
EXCLUIR_CEJAS = 0.05
EXCLUIR_BOCA = 0.05
EXCLUIR_FOSAS = 0.05
EXCLUIR_CONTORNO = 0.05  # hacia adentro del contorno: el borde de la cara contra el fondo no es detalle de piel
DIFUMINADO = 0.06
# Un triangulo que en el destino queda mucho mas grande o mas chico que en la fuente (relativo a toda la cara) o dado
# vuelta no lleva detalle: estirado se ve como manchas. Amplio: en una cara girada la mejilla de adelante crece y la
# de atras se achica (con 0,4-2,5 se perdia la mitad de la mejilla que se ve)
TRIANGULO_ESCALA = (0.15, 4.0)


def puntos(cara):

    '''Puntos para deformar la piel (N, 2): los 106 sin las pupilas y FRENTE_ALTO por encima de cada ceja y de las
       sienes. None si la cara no tiene 106 puntos.'''

    p = cara.get("landmark_2d_106")
    if p is None:
        return None
    p = np.asarray(p, np.float32)
    ojos = (cara.kps[0] + cara.kps[1]) / 2
    boca = (cara.kps[3] + cara.kps[4]) / 2
    d = float(np.linalg.norm(cara.kps[1] - cara.kps[0]))
    arriba = (ojos - boca) / max(float(np.linalg.norm(ojos - boca)), 1e-6)
    cejas = [i for r in CEJAS_106 for i in r]
    frente = p[cejas] + arriba * FRENTE_ALTO * d
    sienes = p[[1, 17]] + arriba * FRENTE_ALTO * 0.6 * d
    usados = [i for i in range(len(p)) if i not in PUPILAS_106]
    return np.vstack([p[usados], frente, sienes]).astype(np.float32)


def _triangulos(pts):

    '''Triangulos de Delaunay (T, 3) con indices de pts'''

    x1, y1 = pts.min(0) - 10
    x2, y2 = pts.max(0) + 10
    sub = cv2.Subdiv2D((int(x1), int(y1), int(x2 - x1) + 1, int(y2 - y1) + 1))
    for q in pts:
        sub.insert((float(q[0]), float(q[1])))
    tri = []
    for t in sub.getTriangleList().reshape(-1, 3, 2):
        idx = [int(np.argmin(np.linalg.norm(pts - v, axis=1))) for v in t]
        if len(set(idx)) == 3 and all(np.linalg.norm(pts[i] - v) < 1 for i, v in zip(idx, t)):
            tri.append(idx)
    return np.array(tri, np.int32)


def mascara_piel(tamano, cara, desplazamiento=(0, 0)):

    '''Piel de la cara (0 a 1, float32) en un recuadro de "tamano" (alto, ancho) que empieza en desplazamiento: dentro
       del contorno y la frente, sin ojos, cejas, boca ni fosas nasales, con el borde difuminado'''

    p = cara.get("landmark_2d_106")
    pts = puntos(cara)
    d = float(np.linalg.norm(cara.kps[1] - cara.kps[0]))
    o = np.asarray(desplazamiento, np.float32)
    m = np.zeros(tamano, np.float32)
    cv2.fillConvexPoly(m, np.round(cv2.convexHull(pts) - o).astype(np.int32), 1.0)
    e = max(1, int(d * EXCLUIR_CONTORNO))
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1)))

    def quitar(indices, agrandar):
        zona = p[list(indices)].astype(np.float32)
        centro = zona.mean(0)
        radio = np.linalg.norm(zona - centro, axis=1).mean()
        zona = centro + (zona - centro) * (1 + agrandar * d / max(radio, 1e-6))
        cv2.fillConvexPoly(m, np.round(cv2.convexHull(zona) - o).astype(np.int32), 0.0)

    for ojo in OJOS_106:
        quitar(ojo, EXCLUIR_OJOS)
    for ceja in CEJAS_106:
        quitar(ceja, EXCLUIR_CEJAS)
    quitar(BOCA_106, EXCLUIR_BOCA)
    quitar(FOSAS_106, EXCLUIR_FOSAS)
    k = max(3, int(d * DIFUMINADO) | 1)
    return cv2.GaussianBlur(m, (k, k), 0)


def _disco(tam, d):
    k = max(3, int(round(tam * d)) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _forma_mancha(oscuro, escalas, d):

    '''0 a 1: cuanto se parece cada pixel de "oscuro" (>= 0) a una mancha redonda y no a una linea, con la escala
       donde la curvatura es mas fuerte; se extiende a toda la mancha (su borde curva menos que el centro)'''

    forma = np.zeros_like(oscuro)
    fuerza = np.zeros_like(oscuro)
    for s in escalas:
        g = cv2.GaussianBlur(oscuro, (0, 0), max(0.7, s * d))
        gxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        gyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        gxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        raiz = np.sqrt(np.maximum((gxx - gyy) ** 2 + 4 * gxy ** 2, 0))
        mayor, menor = -(gxx + gyy - raiz) / 2, -(gxx + gyy + raiz) / 2  # curvaturas hacia abajo
        cociente = np.where(mayor > 1e-6, np.clip(menor / np.maximum(mayor, 1e-6), 0, 1), 0)
        f = np.maximum(mayor, 0) * (s * d) ** 2
        forma = np.where(f > fuerza, cociente, forma)
        fuerza = np.maximum(fuerza, f)
    w = np.clip((forma - FORMA_MANCHA[0]) / (FORMA_MANCHA[1] - FORMA_MANCHA[0]), 0, 1)
    k = _disco(0.02, d)
    w = cv2.dilate(w, k)
    return cv2.GaussianBlur(w, k.shape, 0)


def preparar_fuente(imagen, cara, mascara_cara=None):

    '''
     Detalle de la piel de la cara fuente, para llevarlo despues a otras caras: dict con "detalle" (proporcion - 1 por
    canal, float16, <= 0, 0 fuera de la piel), "puntos" y "triangulos", en un recorte de la foto que empieza en
    "origen". mascara_cara (XSeg, 0 a 255, del tamaño de la imagen) quita lo que tape la piel en la fuente (pelo,
    manos). None si la cara no tiene 106 puntos.
    '''

    pts = puntos(cara)
    if pts is None:
        return None
    d = float(np.linalg.norm(cara.kps[1] - cara.kps[0]))
    alto, ancho = imagen.shape[:2]
    margen = int(d * 0.2) + 4
    x1, y1 = np.maximum(0, np.floor(pts.min(0)).astype(int) - margen)
    x2, y2 = np.minimum([ancho, alto], np.ceil(pts.max(0)).astype(int) + margen)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    recorte = imagen[y1:y2, x1:x2].astype(np.float32) + 1.0
    piel = mascara_piel(recorte.shape[:2], cara, (x1, y1))
    if mascara_cara is not None:
        piel = piel * (mascara_cara[y1:y2, x1:x2].astype(np.float32) / 255)
    lim = DETALLE_LIMITE[0]
    fino = np.clip(recorte / cv2.morphologyEx(recorte, cv2.MORPH_CLOSE, _disco(RASGO_FINO, d)), lim, 1)
    grande = np.clip(recorte / cv2.morphologyEx(recorte, cv2.MORPH_CLOSE, _disco(RASGO_GRANDE, d)), lim, 1)
    oscuro = (1 - grande.mean(2)) * piel

    # Manchas: lo que sobresale de su fondo (opening: la oscuridad sin lo mas chico que el disco), con forma de mancha
    sobre = np.maximum(
        np.clip(oscuro - cv2.morphologyEx(oscuro, cv2.MORPH_OPEN, _disco(FONDO_CHICO, d)), 0, None)
        * _forma_mancha(oscuro, ESCALAS_CHICAS, d),
        np.clip(oscuro - cv2.morphologyEx(oscuro, cv2.MORPH_OPEN, _disco(FONDO_GRANDE, d)), 0, None)
        * _forma_mancha(oscuro, ESCALAS_GRANDES, d))
    pico = cv2.dilate(sobre, _disco(0.03, d))  # el contraste de cada mancha, en toda ella
    sobre = sobre * np.clip((pico - MANCHA_CONTRASTE[0]) / (MANCHA_CONTRASTE[1] - MANCHA_CONTRASTE[0]), 0, 1)
    p = cara.get("landmark_2d_106")
    esquinas = np.zeros(recorte.shape[:2], np.float32)
    for i in ESQUINAS_106:
        cv2.circle(esquinas, tuple(np.round(p[i] - [x1, y1]).astype(int)), int(ESQUINA_RADIO * d), 1.0, -1)
    sobre = sobre * (1 - np.clip(cv2.GaussianBlur(esquinas, (0, 0), 0.02 * d), 0, 1))
    # Por canal y con el color de la mancha: lo oscuro del cierre grande, escalado a lo que sobresale
    escala = np.where(oscuro > 1e-4, sobre / np.maximum(oscuro, 1e-4), 0)[..., None]
    proporcion = np.minimum(fino, 1 - (1 - grande) * escala)

    if getattr(cara, "sex", None) == "M":
        # Barba y bigote: muchos puntos oscuros juntos, desde la base de la nariz hacia abajo
        puntos_osc = ((1 - proporcion.mean(2)) * piel > 0.06).astype(np.float32)
        densidad = cv2.GaussianBlur(puntos_osc, (0, 0), 0.05 * d)
        quita = np.clip((densidad - BARBA[0]) / (BARBA[1] - BARBA[0]), 0, 1)
        abajo = (p[80] - p[72]) / max(float(np.linalg.norm(p[80] - p[72])), 1e-6)
        yy, xx = np.mgrid[0:recorte.shape[0], 0:recorte.shape[1]].astype(np.float32)
        altura = ((xx + x1 - p[80][0]) * abajo[0] + (yy + y1 - p[80][1]) * abajo[1]) / d
        quita = cv2.GaussianBlur(quita * np.clip((altura + 0.05) / 0.1, 0, 1), (0, 0), 0.03 * d)
        proporcion = 1 - (1 - proporcion) * (1 - quita)[..., None]
    detalle = ((proporcion - 1) * piel[..., None]).astype(np.float16)
    return {"detalle": detalle, "origen": np.array([x1, y1], np.float32), "puntos": pts,
            "triangulos": _triangulos(pts)}


def aplicar(frame, cara, piel, fuerza, mascara_cara=None, devolver_mascara=False):

    '''
     Lleva el detalle de piel de la fuente (preparar_fuente) a la cara del destino en "frame" (la cara ya cambiada):
    cada triangulo de puntos del destino toma el detalle del mismo triangulo de la fuente, y cada pixel de piel se
    multiplica por 1 + fuerza * detalle. Solo dentro de la piel del destino (y de mascara_cara, XSeg 0 a 255, si se da).
    Con devolver_mascara=True devuelve (frame, mascara): cuanto se cambio cada pixel (0 a 1), o None si nada.
    '''

    pts = puntos(cara)
    if piel is None or pts is None or fuerza <= 0 or len(pts) != len(piel["puntos"]):
        return (frame, None) if devolver_mascara else frame
    alto, ancho = frame.shape[:2]
    d = float(np.linalg.norm(cara.kps[1] - cara.kps[0]))
    x1, y1 = np.maximum(0, np.floor(pts.min(0)).astype(int) - 2)
    x2, y2 = np.minimum([ancho, alto], np.ceil(pts.max(0)).astype(int) + 3)
    if x2 - x1 < 8 or y2 - y1 < 8 or d < 8:
        return (frame, None) if devolver_mascara else frame

    # Mapa de cada pixel del destino al pixel de la fuente: el triangulo donde cae y su afin
    fuente = piel["puntos"] - piel["origen"]
    destino = pts - [x1, y1]
    def area(q):
        return (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1]) - (q[:, 2, 0] - q[:, 0, 0]) * (q[:, 1, 1] - q[:, 0, 1])

    tri = piel["triangulos"]
    area_f = area(fuente[tri])
    escala = area(destino[tri]) / np.where(np.abs(area_f) > 1e-6, area_f, np.nan)
    # Relativa a la escala de toda la cara (el destino puede ser mas chico o mas grande que la fuente)
    escala = escala / max(float(np.nanmedian(np.abs(escala))), 1e-6)
    ids = np.full((y2 - y1, x2 - x1), -1, np.int32)
    afines = []
    for t, e in zip(tri, escala):
        if not TRIANGULO_ESCALA[0] <= e <= TRIANGULO_ESCALA[1]:
            continue  # estirado, aplastado o dado vuelta
        a, b = destino[t], fuente[t]
        cv2.fillConvexPoly(ids, np.round(a).astype(np.int32), len(afines))
        afines.append(cv2.getAffineTransform(a.astype(np.float32), b.astype(np.float32)))
    if not afines:
        return (frame, None) if devolver_mascara else frame
    afines = np.array(afines, np.float32)
    yy, xx = np.mgrid[0:y2 - y1, 0:x2 - x1].astype(np.float32)
    dentro = ids >= 0
    A = afines[np.maximum(ids, 0)]
    mapa_x = A[..., 0, 0] * xx + A[..., 0, 1] * yy + A[..., 0, 2]
    mapa_y = A[..., 1, 0] * xx + A[..., 1, 1] * yy + A[..., 1, 2]
    mapa_x[~dentro] = -1
    mapa_y[~dentro] = -1
    detalle = piel["detalle"].astype(np.float32)
    # Si el destino es mas chico, se suaviza antes de achicar: sin eso las pecas chicas caen entre pixeles y
    # aparecen o desaparecen al azar (y en video titilan)
    achica = float(np.linalg.norm(np.ptp(piel["puntos"], 0)) / max(float(np.linalg.norm(np.ptp(pts, 0))), 1e-6))
    if achica > 1.5:
        detalle = cv2.GaussianBlur(detalle, (0, 0), 0.45 * achica)
    detalle = cv2.remap(detalle, mapa_x, mapa_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    peso = mascara_piel((y2 - y1, x2 - x1), cara, (x1, y1)) * cv2.GaussianBlur(dentro.astype(np.float32), (0, 0), 2)
    if mascara_cara is not None:
        peso = peso * (mascara_cara[y1:y2, x1:x2].astype(np.float32) / 255)
    peso = peso * float(fuerza)
    region = frame[y1:y2, x1:x2].astype(np.float32)
    nuevo = region * (1 + detalle * peso[..., None])
    resultado = frame.copy()
    resultado[y1:y2, x1:x2] = np.clip(nuevo + 0.5, 0, 255).astype(np.uint8)
    if not devolver_mascara:
        return resultado
    completa = np.zeros((alto, ancho), np.float32)
    completa[y1:y2, x1:x2] = np.clip(np.abs(detalle).max(2) * peso * 4, 0, 1)
    return resultado, completa
