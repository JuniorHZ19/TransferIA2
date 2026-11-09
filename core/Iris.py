import cv2
import numpy as np

# Color de ojos (--color-ojos): el iris es un circulo centrado en la pupila (punto 38 / 88 de los 106) y cortado por el contorno
# del ojo (parpados, de la comisura de afuera a la de adentro y vuelta por abajo). El radio se busca entre
# IRIS_RADIO anchos del ojo, donde mas sube la luz del iris a la esclera. La pupila (hasta IRIS_PUPILA radios)
# y los reflejos no se tiñen. Un ojo mas cerrado que IRIS_ABIERTO (alto / ancho) se deja como esta (parpadeo)
IRIS_106 = ((38, (35, 41, 40, 42, 39, 37, 33, 36)), (88, (93, 96, 94, 95, 89, 90, 87, 91)))
IRIS_RADIO = (0.15, 0.3)
IRIS_PUPILA = (0.25, 0.45)
IRIS_ABIERTO = 0.18
# Salto de luz entre el iris y la esclera (gris de 0 a 255, del lado del iris donde mas sube): por debajo del
# primero no hay iris a la vista y no se tiñe, desde el segundo se tiñe entero, y en medio de a poco (en video no
# se prende y apaga de golpe). Los 106 puntos dibujan un ojo abierto aunque este cerrado o tapado (gorra, visera,
# la nariz de perfil); en 100 caras sinteticas, los ojos tapados daban 8 o menos y casi todos los visibles mas de 10
IRIS_CONTRASTE = (6.0, 14.0)
# Giro de la cara (grados): hasta el primero se tiñe entero y desde el segundo nada. Casi de perfil el iris apenas
# se ve y los 106 puntos del ojo fallan (en las caras sinteticas de mas de 60 grados, la mitad del color caia fuera)
IRIS_GIRO = (50.0, 65.0)
FUERZA = 0.85
IRIS_MANCHAS = 0.4
# Colores con nombre, en LAB (luz de 0 a 100): el color medio de un iris de ese tipo, no el de la pintura
COLORES = {"azul": (40, -4, -16), "verde": (38, -10, 14), "miel": (38, 8, 26), "gris": (44, -1, -5),
           "marron": (24, 9, 15)}


def validar(valor):

    '''Valor de --color-ojos ya normalizado: "ninguno", "fuente", un nombre de COLORES o "#rrggbb"'''

    texto = str(valor).strip().lower()
    if texto in ("ninguno", "fuente") or texto in COLORES:
        return texto
    hexa = texto.lstrip("#")
    if len(hexa) == 6 and all(c in "0123456789abcdef" for c in hexa):
        return "#" + hexa
    raise ValueError(f"debe ser ninguno, fuente, {', '.join(COLORES)} o un color #RRGGBB")


def color_objetivo(valor, cara_src=None):

    '''
     Color al que se llevan los iris (L, a, b de LAB, L de 0 a 100) o None si no hay que cambiarlo. "valor"
    puede ser "fuente" (el iris de la foto fuente, medido al cargarla: cara_src["color_iris"]), un nombre de
    COLORES o un color #RRGGBB.
    '''

    if not valor or valor == "ninguno":
        return None
    if valor == "fuente":
        color = cara_src.get("color_iris") if cara_src is not None else None
        return None if color is None else np.asarray(color, np.float32)
    if valor in COLORES:
        return np.asarray(COLORES[valor], np.float32)
    try:
        texto = validar(valor).lstrip("#")
    except ValueError:
        return None  # p. ej. «elegir color...» de la interfaz mientras el selector esta abierto
    rgb = np.array([[[int(texto[i:i + 2], 16) for i in (4, 2, 0)]]], np.float32) / 255  # BGR
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)[0, 0]


def zonas_iris(imagen, cara):

    '''
     Iris de cada ojo abierto: lista de (x1, y1, mascara), la mascara (0 a 1) en un recuadro del frame que
    empieza en (x1, y1). Vacia si la cara no tiene 106 puntos o los dos ojos estan cerrados.
    '''

    puntos = cara.get("landmark_2d_106")
    if puntos is None:
        return []
    pose = cara.get("pose")
    giro = 1.0 if pose is None else float(np.clip((IRIS_GIRO[1] - abs(float(pose[1]))) / (IRIS_GIRO[1] - IRIS_GIRO[0]), 0, 1))
    if giro <= 0:
        return []
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY).astype(np.float32)
    alto, ancho_img = gris.shape
    zonas = []
    for pupila, indices in IRIS_106:
        centro = puntos[pupila].astype(np.float32)
        contorno = puntos[list(indices)].astype(np.float32)
        ancho = float(np.linalg.norm(contorno[0] - contorno[4]))
        if ancho < 12 or np.linalg.norm(contorno[2] - contorno[6]) < IRIS_ABIERTO * ancho:
            continue
        m = int(ancho * 0.5) + 2
        x1, y1 = max(0, int(centro[0]) - m), max(0, int(centro[1]) - m)
        x2, y2 = min(ancho_img, int(centro[0]) + m + 1), min(alto, int(centro[1]) + m + 1)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        centro = centro - [x1, y1]
        ojo = np.zeros((y2 - y1, x2 - x1), np.float32)
        cv2.fillPoly(ojo, [np.round(contorno - [x1, y1]).astype(np.int32)], 1.0)
        # Sin la linea de las pestañas ni el borde del parpado
        e = max(1, int(ancho * 0.04))
        ojo = cv2.erode(ojo, np.ones((2 * e + 1, 2 * e + 1), np.uint8))
        radio, contraste = _radio(gris[y1:y2, x1:x2], centro, ancho, ojo)
        visible = giro * float(np.clip((contraste - IRIS_CONTRASTE[0]) / (IRIS_CONTRASTE[1] - IRIS_CONTRASTE[0]), 0, 1))
        if visible <= 0:
            continue
        yy, xx = np.mgrid[0:y2 - y1, 0:x2 - x1]
        r = np.hypot(xx - centro[0], yy - centro[1]) / radio
        disco = np.clip((1 - r) / 0.12 + 0.5, 0, 1)
        sin_pupila = np.clip((r - IRIS_PUPILA[0]) / (IRIS_PUPILA[1] - IRIS_PUPILA[0]), 0, 1)
        k = max(3, int(ancho * 0.06) | 1)
        mascara = disco * sin_pupila * cv2.GaussianBlur(ojo, (k, k), 0) * visible
        if mascara.max() > 0.05:
            zonas.append((x1, y1, mascara.astype(np.float32)))
    return zonas


def _radio(gris, centro, ancho, ojo):

    '''(radio, contraste): radio del iris en pixeles, el circulo donde mas sube la luz de adentro (iris) hacia
       afuera (esclera), mirando solo a los costados (arriba y abajo lo suelen tapar los parpados) y solo dentro
       del ojo; contraste es cuanto sube en ese circulo del lado que mas sube (si el ojo mira a un costado, del
       otro lado casi no hay esclera), gris de 0 a 255, 0 si no se pudo medir'''

    angulos = np.radians(np.r_[np.linspace(-40, 40, 9), np.linspace(140, 220, 9)])
    direcciones = np.stack([np.cos(angulos), np.sin(angulos)], 1)
    derecha = np.arange(len(angulos)) < 9
    alto, ancho_rec = gris.shape

    def muestras(r):
        p = np.rint(centro + direcciones * r).astype(int)
        dentro = (p[:, 0] >= 0) & (p[:, 0] < ancho_rec) & (p[:, 1] >= 0) & (p[:, 1] < alto)
        p = np.clip(p, 0, [ancho_rec - 1, alto - 1])
        return gris[p[:, 1], p[:, 0]], dentro & (ojo[p[:, 1], p[:, 0]] > 0.5)

    mejor, radio, contraste = 0.0, 0.22 * ancho, 0.0
    for f in np.linspace(IRIS_RADIO[0], IRIS_RADIO[1], 16):
        adentro, ok_adentro = muestras(f * ancho * 0.85)
        afuera, ok_afuera = muestras(f * ancho * 1.15)
        ok = ok_adentro & ok_afuera
        if ok.sum() < 3:
            continue
        sube = afuera - adentro
        salto = float(np.mean(sube[ok]))
        if salto > mejor:
            lados = [float(np.mean(sube[ok & lado])) for lado in (derecha, ~derecha) if (ok & lado).sum() >= 2]
            mejor, radio, contraste = salto, f * ancho, max(lados, default=salto)
    return max(radio, 2.0), contraste


def _lab(bgr):
    return cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)


def color_iris(imagen, cara, zonas=None):

    '''Color tipico del iris (L, a, b de LAB, L de 0 a 100) de los dos ojos juntos, o None si no se ve: la
       mediana, sin el 10 % mas oscuro (sombra de las pestañas, pupila) ni el 10 % mas claro (reflejos)'''

    zonas = zonas_iris(imagen, cara) if zonas is None else zonas
    muestras = []
    for x1, y1, mascara in zonas:
        h, w = mascara.shape
        lab = _lab(imagen[y1:y1 + h, x1:x1 + w])[mascara > 0.6]
        if len(lab) < 10:
            continue
        bajo, alto = np.percentile(lab[:, 0], [10, 90])
        muestras.append(lab[(lab[:, 0] >= bajo) & (lab[:, 0] <= alto)])
    return np.median(np.concatenate(muestras), 0) if muestras else None


def cambiar_color(frame, cara, objetivo, fuerza=FUERZA, mascara_cara=None, devolver_mascara=False, original=None):

    '''
     Lleva los iris de la cara (en frame, que ya es la cara cambiada) al color "objetivo" (L, a, b). Se mueve
    el color medio y se conserva la textura del iris: la luz por proporcion (el borde oscuro y las vetas siguen
    mas oscuros) y el color (a, b) pasa a ser el nuevo, con algo de las manchas del iris. Los reflejos y la
    esclera quedan blancos. Con mascara_cara (XSeg, 0 a 255) no se tiñe lo que tape el ojo (anteojos de sol,
    una mano).
    "original": el destino antes del cambio, donde se busca el iris (la cara que dibuja inswapper sale de un
    recorte de 128 px y su iris es borroso: el borde se encontraba mal y el color se salia a la esclera).
    Con devolver_mascara=True devuelve (frame, mascara): cuanto se cambio cada pixel (0 a 1).
    '''

    zonas = zonas_iris(frame if original is None else original, cara) if objetivo is not None else []
    actual = color_iris(frame, cara, zonas) if zonas else None
    if actual is None:
        return (frame, None) if devolver_mascara else frame
    proporcion = float(np.clip(objetivo[0] / max(float(actual[0]), 1.0), 0.6, 1.7))
    resultado = frame.copy()
    completa = np.zeros(frame.shape[:2], np.float32) if devolver_mascara else None
    for x1, y1, mascara in zonas:
        h, w = mascara.shape
        lab = _lab(frame[y1:y1 + h, x1:x1 + w])
        # El color nuevo va en proporcion a la luz del pixel (el anillo oscuro del borde y las vetas no se
        # saturan) y de las manchas de color del iris queda IRIS_MANCHAS: sumando la diferencia a todos, una
        # mancha ya azulada en un ojo marron quedaba como un aro azul chillon
        luz = np.clip(lab[..., 0] / max(float(actual[0]), 1.0), 0.3, 1.2)[..., None]
        nuevo = np.dstack([lab[..., 0] * proporcion,
                           objetivo[1:] * luz + (lab[..., 1:] - actual[1:]) * IRIS_MANCHAS])
        # Lo mucho mas claro que el iris (reflejos y la esclera que toca el borde del circulo) no se tiñe
        peso = mascara * fuerza * (1 - np.clip((lab[..., 0] - actual[0] - 25) / 15, 0, 1))
        if mascara_cara is not None:
            peso = peso * mascara_cara[y1:y1 + h, x1:x1 + w].astype(np.float32) / 255
        lab = lab + (nuevo - lab) * peso[..., None]
        lab[..., 0] = np.clip(lab[..., 0], 0, 100)
        bgr = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2BGR)
        region = resultado[y1:y1 + h, x1:x1 + w]
        cambia = peso > 0.004
        region[cambia] = np.clip(bgr * 255 + 0.5, 0, 255).astype(np.uint8)[cambia]
        if devolver_mascara:
            completa[y1:y1 + h, x1:x1 + w] = np.maximum(completa[y1:y1 + h, x1:x1 + w], peso)
    return (resultado, completa) if devolver_mascara else resultado
