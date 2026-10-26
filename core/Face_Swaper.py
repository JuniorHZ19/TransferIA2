import onnxruntime as ort
import cv2
import numpy as np
import insightface
from insightface.utils import face_align
import array
from core.Face_Analyzer import Analyzer_Faces
from core.xseg import DFL_XSeg_Model
from core.Identidades import Registro_Identidades
from core.Mejorador import Mejorador_Cara, MEZCLA, MINIMO_POR_DEFECTO
import threading

# Indices de los ojos, la boca (labios) y la nariz (puente y fosas) en los 106 puntos (landmark_2d_106) de buffalo_l
OJOS_106 = (range(33, 43), range(87, 97))
BOCA_106 = range(52, 72)
NARIZ_106 = range(72, 87)
CEJAS_106 = (range(43, 52), range(97, 106))

# Filtros de nitidez de la cara pegada (ver xseg.filtro_image); "None" = sin filtro
FILTROS_NITIDEZ = ("natural", "strong", "adaptativo")

# Frente del destino: en distancias entre ojos, desde el borde de arriba de las cejas.
# La mezcla empieza en 0 a FRENTE_INICIO por encima de la ceja y llega a 1 a FRENTE_COMPLETA (las cejas quedan
# de la cara cambiada, que pesan en el parecido); a los lados llega hasta FRENTE_ANCHO del centro (sienes)
FRENTE_INICIO = 0.06
FRENTE_COMPLETA = 0.32
FRENTE_ANCHO = 0.95
# Tono de la frente del destino: se iguala al de la cara cambiada justo debajo del borde, en el entrecejo y el
# puente de la nariz (sin cerquillo ni ojos): hasta REFERENCIA_TONO_X a cada lado del centro, desde la ceja
# hasta REFERENCIA_TONO_Y por debajo de los ojos (en distancias entre ojos). Las mejillas no sirven: su tono
# natural es otro que el de la frente y la correccion quedaba peor que sin corregir
REFERENCIA_TONO_X = 0.13  # una franja angosta: mas ancha toma el borde de los ojos y las cejas
REFERENCIA_TONO_Y = 0.35

# Conservar maquillaje: radio del desenfoque que separa los colores grandes (maquillaje, del destino) del
# detalle fino (rasgos, de la fuente), en fraccion del recorte alineado (128 px -> ~4 px)
SIGMA_MAQUILLAJE = 0.03

# Correccion de color de la cara cambiada para igualar la piel del destino (transferencia en LAB):
# "suave" iguala el tono (A, B) y la luz media; "completa" ademas ajusta el contraste de la luz.
COLORES = ("ninguna", "suave", "completa")

# Zona de la cara cambiada que se pega sobre el destino: (recorte del borde, difuminado), en fraccion del
# recorte alineado del modelo. "normal" es el pegado original de insightface (deja mas contorno del destino);
# "amplia" y "maxima" cubren mas mejillas, menton y frente, asi el resultado se parece mas a la fuente.
MASCARAS = {"normal": (0.10, 0.05), "amplia": (0.05, 0.035), "maxima": (0.02, 0.025)}

class Swapper_Faces:

    '''

     Esta clase se encargara de manejar el modelo inswapper128.onnx , con este modelo y ayuda de otros modelos que hemos
    definido tambien sus clase de manejo como XSeg_model(para recortar el rostro y poner en el area extacta del rostro)
    y tambien Analyzer_faces que es para analizar y obtener las caras de las imanges, en este caso esta clase aplicara cada uno de 
    estos modelos  para tener listos los parametros necesarios de .get(parametros) y obtener la iamgen  cual se cambio la cara , y leugo aplicarameos oclucion con
    frame_with_occlusion, para no tener un cuadrado alredor de la cara  y sea mas fino ponr el rostro fake en la cara oriignal.


     '''
    _XSEG_MODEL = DFL_XSeg_Model("models/face_occluder.onnx")
    _ANALIZADOR = Analyzer_Faces()
    _LOCK = threading.Lock()
    _SWAPPER_MODEL = None  # atributo de clase (constante/singleton)
    _MEJORADOR = None  # GPEN, se carga solo si se usa --enhance



    def __init__(self,path):

        '''Creamos en el constructor o incializamos el modelo Analyzer_Faces  solo una vez'''
       

        if self.__class__._SWAPPER_MODEL is None:
           self.__class__._SWAPPER_MODEL=insightface.model_zoo.get_model(path, providers = ["CUDAExecutionProvider",
                                                                                                "CPUExecutionProvider"
                                                                                            ])
      
    def get_Swapper_Model(self):
        return  self.__class__._SWAPPER_MODEL
    

    def cambiar_Faces(self,frame_dst,frame_src=None,index_src="0",index_dst="0",oclution=True,mode="None",genero="todos",ojos=False,mascara="amplia",caras_src=None,boca=False,color="ninguna",nariz=False,maquillaje=0.0,mejorar=0.0,devolver_mascara=False,mejorar_minimo=MINIMO_POR_DEFECTO,frente=False,oclusion_modo="adaptable"):

        '''
         caras_src: caras fuente ya preparadas (de una foto o de modelos .npy); si es None se detectan en frame_src.
        Con devolver_mascara=True devuelve (frame, mascara): la zona que de verdad cambio, de 0 a 1 (ver cambiar_una)
        '''
     
        indices_src = [int(i) for i in index_src.split(',')]

        indices_dst = [int(i) for i in index_dst.split(',')]


        if caras_src is None:
            caras_src=self.__class__._ANALIZADOR.obtener_caras(frame_src)

        caras_dst=self.__class__._ANALIZADOR.obtener_caras(frame_dst,genero=genero)

        fake_frame=frame_dst
        mascara_total = np.zeros(frame_dst.shape[:2], dtype=np.float32) if devolver_mascara else None


        # Cada par (fuente, destino) por separado: una misma fuente puede ir a varias caras (0,0 -> 0,1). Antes el
        # total se limitaba a la cantidad de fuentes y con una sola fuente solo cambiaba la primera cara.
        # Si en este frame no esta la cara pedida (o la fuente no existe), ese par se salta
        for indice_src, indice_dst in zip(indices_src, indices_dst):

         if not (0 <= indice_src < len(caras_src) and 0 <= indice_dst < len(caras_dst)):
            continue

         frame_dst=fake_frame

         fake_frame, alfa = self.cambiar_una(frame_dst, caras_src[indice_src], caras_dst[indice_dst], oclution, mode,
                                             ojos, mascara, boca, color, nariz, maquillaje, mejorar, devolver_mascara,
                                             mejorar_minimo, frente, oclusion_modo)
         if devolver_mascara:
            mascara_total = np.maximum(mascara_total, alfa)


        return (fake_frame, mascara_total) if devolver_mascara else fake_frame


    def cambiar_una(self, frame, cara_src, cara_dst, oclution=True, mode="None", ojos=False, mascara="amplia",
                    boca=False, color="ninguna", nariz=False, maquillaje=0.0, mejorar=0.0, devolver_mascara=False,
                    mejorar_minimo=MINIMO_POR_DEFECTO, frente=False, oclusion_modo="adaptable"):

        '''
         Cambia una sola cara: pegado (+ color + maquillaje) -> GPEN -> oclusion -> partes del destino.
        Devuelve (frame, alfa). Con devolver_mascara=True, alfa es cuanto cambio cada pixel respecto de "frame"
        (0 = queda el destino, 1 = todo nuevo), combinando las mascaras de cada paso; si no, alfa es None.
        '''

        # Las mascaras solo se calculan si se piden (el preview); al procesar no se pierde tiempo en ellas
        d = devolver_mascara
        alfa = None
        # Filtro de nitidez sin oclusion: se aplica dentro de la zona pegada (con oclusion, dentro de su mascara)
        solo_nitidez = not oclution and mode in FILTROS_NITIDEZ

        fake_frame = self.pegar_cara(frame, cara_src, cara_dst, mascara, color, maquillaje,
                                     devolver_zona=d or solo_nitidez)
        if d or solo_nitidez:
            fake_frame, zona = fake_frame
            alfa = zona if d else None

        if mejorar:
            fake_frame = self.mejorar_cara(fake_frame, cara_dst, mejorar, devolver_mascara=d, minimo=mejorar_minimo)
            if d:
                # GPEN retoca tambien un poco alrededor de la cara (su recorte es mas grande que la zona pegada)
                fake_frame, g = fake_frame
                alfa = alfa + (1 - alfa) * g

        if oclution == True:
            with self.__class__._LOCK:
                fake_frame = self.__class__._XSEG_MODEL.frame_with_occlusion(frame=frame, fake_frame=fake_frame,
                                                                              face=cara_dst, mode=mode,
                                                                              devolver_mascara=d,
                                                                              modo_oclusion=oclusion_modo)
            if d:
                fake_frame, m = fake_frame
                alfa = alfa * m
        elif solo_nitidez:
            fake_frame = self.aplicar_nitidez(frame, fake_frame, cara_dst, mode, zona)

        if ojos or boca or nariz or frente:
            fake_frame = self.conservar_partes(frame, fake_frame, cara_dst, ojos, boca, nariz, devolver_mascara=d,
                                               frente=frente)
            if d:
                fake_frame, p = fake_frame
                alfa = alfa * (1 - p)

        return fake_frame, (np.clip(alfa, 0, 1).astype(np.float32) if d else None)


    def pegar_cara(self, frame, cara_src, cara_dst, mascara="amplia", color="ninguna", maquillaje=0.0, devolver_zona=False):

        '''
         Cambia una cara y la pega sobre el frame. Con "normal" (sin correccion de color ni maquillaje) se usa
        el pegado de insightface; si no, la mascara se arma en el recorte alineado (128 px) con menos recorte
        de borde, la cara se corrige de color y/o se le devuelve el maquillaje del destino, y se lleva al frame.
        Con devolver_zona=True devuelve (frame, zona): la mascara de pegado (0 a 1) del tamaño del frame.
        '''

        modelo = self.__class__._SWAPPER_MODEL
        if mascara not in MASCARAS:
            mascara = "amplia"
        maquillaje = float(np.clip(maquillaje or 0.0, 0.0, 1.0))
        if mascara == "normal" and color not in ("suave", "completa") and maquillaje == 0:
            resultado = modelo.get(frame, source_face=cara_src, target_face=cara_dst, paste_back=True)
            if not devolver_zona:
                return resultado
            return resultado, self._zona_insightface(frame.shape[:2], cara_dst, modelo.input_size[0])

        cara_fake, M = modelo.get(frame, source_face=cara_src, target_face=cara_dst, paste_back=False)
        lado = cara_fake.shape[0]
        recorte, difuminado = MASCARAS[mascara]

        if color in ("suave", "completa") or maquillaje > 0:
            original = cv2.warpAffine(frame, M, (lado, lado), borderValue=0.0)
            if color in ("suave", "completa"):
                cara_fake = self.corregir_color(cara_fake, original, cara_dst, M, completa=(color == "completa"))
            if maquillaje > 0:
                cara_fake = self.conservar_maquillaje(cara_fake, original, maquillaje)

        borde = max(2, int(lado * recorte))
        zona = np.zeros((lado, lado), dtype=np.float32)
        zona[borde:-borde, borde:-borde] = 1.0
        k = max(3, int(lado * difuminado) * 2 + 1)
        zona = cv2.GaussianBlur(zona, (k, k), 0)
        zona[:1, :] = zona[-1:, :] = 0
        zona[:, :1] = zona[:, -1:] = 0

        alto, ancho = frame.shape[:2]
        IM = cv2.invertAffineTransform(M)
        cara_fake = cv2.warpAffine(cara_fake, IM, (ancho, alto), borderValue=0.0)
        zona = cv2.warpAffine(zona, IM, (ancho, alto), borderValue=0.0)[..., None]

        resultado = zona * cara_fake.astype(np.float32) + (1 - zona) * frame.astype(np.float32)
        return (resultado.astype(np.uint8), zona[..., 0]) if devolver_zona else resultado.astype(np.uint8)


    def aplicar_nitidez(self, frame, fake_frame, cara, modo, zona):

        '''
         Filtro de nitidez sin oclusion (natural, strong o adaptativo, ver xseg.filtro_image) sobre la cara
        recien pegada. "zona" es la mascara de pegado (0 a 1): el filtro solo cambia la cara nueva, nunca el
        fondo del destino que la rodea. "frame" es el destino original, que el modo adaptativo usa de referencia.
        '''

        mascara = np.clip(zona * 255.0, 0, 255).astype(np.uint8)
        x, y, w, h = cv2.boundingRect((mascara > 0).astype(np.uint8))
        if w == 0 or h == 0:
            return fake_frame
        nitido = self.__class__._XSEG_MODEL.filtro_image(fake_frame, modo, face=cara, frame=frame, mascara=mascara)
        # Mezcla solo en el rectangulo de la zona: afuera la zona es 0 y queda fake_frame
        z = zona[y:y + h, x:x + w, None]
        resultado = fake_frame.copy()
        resultado[y:y + h, x:x + w] = (nitido[y:y + h, x:x + w].astype(np.float32) * z
                                       + fake_frame[y:y + h, x:x + w].astype(np.float32) * (1 - z)).astype(np.uint8)
        return resultado


    @staticmethod
    def _zona_insightface(tamano, cara, lado):

        '''
         La mascara con la que insightface pega la cara (inswapper.get con paste_back): el cuadrado alineado
        llevado al frame, erosionado 1/10 y difuminado 1/20 de su tamaño. Solo se usa para mostrarla.
        '''

        _, M = face_align.norm_crop2(np.zeros((8, 8, 3), np.uint8), cara.kps, lado)
        alto, ancho = tamano
        blanco = cv2.warpAffine(np.full((lado, lado), 255, np.float32), cv2.invertAffineTransform(M), (ancho, alto),
                                borderValue=0.0)
        blanco[blanco > 20] = 255
        filas, columnas = np.where(blanco == 255)
        if len(filas) == 0:
            return np.zeros(tamano, dtype=np.float32)
        medida = int(np.sqrt((filas.max() - filas.min()) * (columnas.max() - columnas.min())))
        k = max(medida // 10, 10)
        blanco = cv2.erode(blanco, np.ones((k, k), np.uint8), iterations=1)
        k = max(medida // 20, 5)
        blanco = cv2.GaussianBlur(blanco, (2 * k + 1, 2 * k + 1), 0)
        return blanco / 255.0


    def mejorar_cara(self, frame, cara, fuerza=True, devolver_mascara=False, minimo=MINIMO_POR_DEFECTO):

        '''
         Pasa la cara recien cambiada por GPEN (core/Mejorador.py) antes de la oclusion, asi el pelo, las manos
        y los objetos que tapan la cara siguen saliendo del destino sin pasar por el mejorador.
        "fuerza" (0 a 1) es cuanto de la cara mejorada se usa; True usa la de Mejorador.MEZCLA (0.8).
        "minimo": caras mas chicas no se mejoran ("96" px, o "25%" del lado mas corto de la imagen).
        '''

        if self.__class__._MEJORADOR is None:
            self.__class__._MEJORADOR = Mejorador_Cara()
        mezcla = MEZCLA if fuerza is True else float(min(1.0, max(0.0, fuerza)))
        return self.__class__._MEJORADOR.mejorar(frame, cara, mezcla=mezcla, devolver_mascara=devolver_mascara,
                                                 minimo=minimo)


    def get_Analizador(self):
        return self.__class__._ANALIZADOR


    def detectar_caras(self, frame, genero="todos"):

        '''Detecta las caras de un frame (filtradas por genero) con el analizador compartido'''
        return self.__class__._ANALIZADOR.obtener_caras(frame, genero=genero)


    def cambiar_Faces_por_identidad(self, frame_dst, caras_src, mapa, refs, caras_dst=None,
                                    oclution=True, mode="None", genero="todos", ojos=False, mascara="amplia",
                                    boca=False, color="ninguna", nariz=False, maquillaje=0.0, mejorar=0.0,
                                    devolver_mascara=False, mejorar_minimo=MINIMO_POR_DEFECTO, frente=False,
                                    oclusion_modo="adaptable"):

        '''
         Cambia las caras emparejando por identidad y no por posicion. "mapa" es {identidad_destino: indice_fuente},
        donde la identidad destino es el orden en que esa persona aparecio por primera vez (ver Registro_Identidades),
        y "refs" son los embeddings de esas identidades. caras_src se detecta una sola vez fuera de esta funcion.
        Con devolver_mascara=True devuelve (frame, mascara), como cambiar_Faces.
        '''

        if caras_dst is None:
            caras_dst = self.detectar_caras(frame_dst, genero)

        fake_frame = frame_dst
        mascara_total = np.zeros(frame_dst.shape[:2], dtype=np.float32) if devolver_mascara else None

        for cara_dst, identidad in Registro_Identidades.asignar(caras_dst, refs):

            indice_src = mapa.get(identidad)
            if indice_src is None or indice_src >= len(caras_src):
                continue

            fake_frame, alfa = self.cambiar_una(fake_frame, caras_src[indice_src], cara_dst, oclution, mode, ojos,
                                                mascara, boca, color, nariz, maquillaje, mejorar, devolver_mascara,
                                                mejorar_minimo, frente, oclusion_modo)
            if devolver_mascara:
                mascara_total = np.maximum(mascara_total, alfa)

        return (fake_frame, mascara_total) if devolver_mascara else fake_frame


    @staticmethod
    def corregir_color(cara_fake, original, cara, M, completa=False):

        '''
         Iguala el color de la cara generada (recorte alineado) al de la cara original del destino en el mismo
        recorte, para que no se note el "parche". Se trabaja en LAB (L = luz, A/B = tono) y se mide solo sobre
        la piel: el centro del recorte sin ojos ni boca. Por canal: (x - media_fake) / desv_fake * desv_dst + media_dst.
        El tono se iguala completo; en la luz se iguala la media y, solo en "completa", el contraste (limitado),
        porque igualarlo del todo borraria las sombras propias de la cara fuente y se perderia parecido.
        '''

        lado = cara_fake.shape[0]
        piel = np.zeros((lado, lado), dtype=np.uint8)
        borde = int(lado * 0.18)
        piel[borde:-borde, borde:-borde] = 1

        puntos = cara.get("landmark_2d_106")
        if puntos is not None:
            alineados = cv2.transform(puntos[None].astype(np.float32), M)[0]
            for indices in (*OJOS_106, BOCA_106):
                zona = alineados[list(indices)]
                centro = zona.mean(0)
                zona = centro + (zona - centro) * 1.4
                cv2.fillConvexPoly(piel, cv2.convexHull(zona.astype(np.int32)), 0)

        piel = piel.astype(bool)
        if piel.sum() < lado:  # cara demasiado tapada o puntos fuera del recorte: no se corrige
            return cara_fake

        fake = cv2.cvtColor(cara_fake, cv2.COLOR_BGR2LAB).astype(np.float32)
        dst = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)

        for canal in range(3):
            media_f, desv_f = fake[..., canal][piel].mean(), fake[..., canal][piel].std() + 1e-6
            media_d, desv_d = dst[..., canal][piel].mean(), dst[..., canal][piel].std() + 1e-6
            if canal == 0:
                escala = float(np.clip(desv_d / desv_f, 0.8, 1.25)) if completa else 1.0
            else:
                escala = float(np.clip(desv_d / desv_f, 0.5, 2.0))
            fake[..., canal] = (fake[..., canal] - media_f) * escala + media_d

        fake = np.clip(fake, 0, 255).astype(np.uint8)
        return cv2.cvtColor(fake, cv2.COLOR_LAB2BGR)


    @staticmethod
    def conservar_maquillaje(cara_fake, original, fuerza):

        '''
         Devuelve a la cara generada el maquillaje o la pintura del destino (separacion de frecuencias).
        inswapper dibuja la piel de la fuente desde cero, asi que la pintura del destino se pierde. Se separa:
            base    = desenfoque(original)               colores grandes: maquillaje, pintura, sombras
            detalle = cara_fake - desenfoque(cara_fake)  detalle fino: bordes de ojos, nariz, labios (la fuente)
        y se juntan (base + detalle). "fuerza" (0 a 1) mezcla ese resultado con la cara generada sin tocar:
        mas fuerza = mas maquillaje del destino y menos parecido con la fuente.
        '''

        lado = cara_fake.shape[0]
        sigma = max(1.0, lado * SIGMA_MAQUILLAJE)
        fake = cara_fake.astype(np.float32)
        base = cv2.GaussianBlur(original.astype(np.float32), (0, 0), sigma)
        detalle = fake - cv2.GaussianBlur(fake, (0, 0), sigma)
        con_maquillaje = base + detalle
        resultado = fake * (1 - fuerza) + con_maquillaje * fuerza
        return np.clip(resultado, 0, 255).astype(np.uint8)


    @staticmethod
    def conservar_partes(frame_original, fake_frame, cara, ojos=True, boca=False, nariz=False, devolver_mascara=False,
                         frente=False):

        '''
         Devuelve fake_frame con los ojos, la boca y/o la nariz de frame_original (el destino): la cara se cambia
        completa pero la mirada, el parpadeo y el color de ojos (ojos), los labios, dientes y lengua al hablar (boca)
        o la forma de la nariz (nariz) siguen siendo los del destino. La mascara se arma con el contorno de cada
        parte (106 puntos), se agranda un poco (pestañas y parpados, borde de labios, aletas de la nariz) y se
        difumina el borde para que no se note el corte. Si no hay 106 puntos se usan elipses sobre los 5 puntos
        (kps: ojos, punta de la nariz y comisuras de la boca).
        "frente": la frente del destino, desde un poco arriba de las cejas (ver mascara_frente). Sirve cuando la
        fuente tiene cerquillo: inswapper dibuja algo de ese pelo sobre la frente y aqui se quita.
        Con devolver_mascara=True devuelve (frame, mascara): cuanto de cada pixel vuelve del destino (0 a 1).
        '''

        dist_ojos = float(np.linalg.norm(cara.kps[1] - cara.kps[0]))
        if dist_ojos < 2 or not (ojos or boca or nariz or frente):
            return (fake_frame, 0.0) if devolver_mascara else fake_frame

        # Formas a pintar en la mascara: ("poligono", puntos) o ("elipse", centro, ejes, angulo), en pixeles del frame
        formas = []
        puntos = cara.get("landmark_2d_106")
        if puntos is not None:
            partes = (([(i, 1.35) for i in OJOS_106] if ojos else []) + ([(BOCA_106, 1.2)] if boca else [])
                      + ([(NARIZ_106, 1.15)] if nariz else []))
            for indices, agrandar in partes:
                zona = puntos[list(indices)].astype(np.float32)
                centro = zona.mean(0)
                # Agrandar el contorno desde el centro para tomar pestañas, parpados, borde de labios y aletas
                zona = centro + (zona - centro) * agrandar
                formas.append(("poligono", cv2.convexHull(zona.astype(np.int32))))
        else:
            angulo = float(np.degrees(np.arctan2(*(cara.kps[1] - cara.kps[0])[::-1])))
            if ojos:
                ejes = (int(dist_ojos * 0.28), int(dist_ojos * 0.16))
                for centro in cara.kps[:2]:
                    formas.append(("elipse", tuple(int(v) for v in centro), ejes, angulo))
            if boca:
                centro = (cara.kps[3] + cara.kps[4]) / 2
                ancho = float(np.linalg.norm(cara.kps[4] - cara.kps[3]))
                ejes = (int(ancho * 0.65), int(dist_ojos * 0.25))
                formas.append(("elipse", tuple(int(v) for v in centro), ejes, angulo))
            if nariz:
                # Desde el entrecejo hasta la punta: centro un poco por encima de la punta (kps[2])
                centro = (cara.kps[0] + cara.kps[1]) / 2 * 0.35 + cara.kps[2] * 0.65
                ejes = (int(dist_ojos * 0.25), int(dist_ojos * 0.42))
                formas.append(("elipse", tuple(int(v) for v in centro), ejes, angulo))

        # Difuminado proporcional al tamaño de la cara (impar para GaussianBlur)
        k = max(3, int(dist_ojos * 0.12) | 1)

        # Todo se calcula solo en un recuadro (x1, y1, x2, y2) que contiene las formas y la frente: afuera la mascara
        # es 0 y el resultado es fake_frame tal cual, igual que si se calculara el frame entero (que es mucho mas
        # lento: en 1080p la cara es ~10 % del frame). El margen (2 radios del difuminado) hace que el borde del
        # recuadro quede lejos de las formas, asi el difuminado da exactamente los mismos valores
        alto, ancho = fake_frame.shape[:2]
        margen = 2 * (k // 2) + 2
        cajas = []
        for forma in formas:
            if forma[0] == "poligono":
                xy = forma[1].reshape(-1, 2)
                cajas.append((*xy.min(0), *xy.max(0)))
            else:
                (cx, cy), radio = forma[1], max(forma[2]) + 1
                cajas.append((cx - radio, cy - radio, cx + radio, cy + radio))
        cajas = [(x1 - margen, y1 - margen, x2 + margen + 1, y2 + margen + 1) for x1, y1, x2, y2 in cajas]
        rectangulo = None
        if frente:
            mascara_frente, rectangulo = Swapper_Faces.mascara_frente((alto, ancho), cara, devolver_rectangulo=True)
            if rectangulo is not None:
                cajas.append(rectangulo)
        x1 = max(0, min(c[0] for c in cajas)) if cajas else 0
        y1 = max(0, min(c[1] for c in cajas)) if cajas else 0
        x2 = min(ancho, max(c[2] for c in cajas)) if cajas else 0
        y2 = min(alto, max(c[3] for c in cajas)) if cajas else 0
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x2 > x1 and y2 > y1 and (x1 == 0 or x2 == ancho):
            # Si el recuadro toca el borde izquierdo o derecho, filas enteras: OpenCV difumina el borde de la fila
            # por otro camino segun el ancho, y la mascara cambiaba en el ultimo decimal (1e-8)
            x1, x2 = 0, ancho
        if x2 <= x1 or y2 <= y1:  # nada cae dentro del frame
            resultado = fake_frame.copy()
            return (resultado, np.zeros((alto, ancho), np.float32)) if devolver_mascara else resultado

        mascara = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
        desplazamiento = np.array([x1, y1], dtype=np.int32)
        for forma in formas:
            if forma[0] == "poligono":
                cv2.fillConvexPoly(mascara, forma[1] - desplazamiento, 1.0)
            else:
                centro = (forma[1][0] - x1, forma[1][1] - y1)
                cv2.ellipse(mascara, centro, forma[2], forma[3], 0, 360, 1.0, -1)
        mascara = cv2.GaussianBlur(mascara, (k, k), 0)

        if frente:
            # La frente del destino casi nunca tiene el tono de la cara que dibuja inswapper: se corrige antes
            # de mezclarla para que no se note la mancha (fuera de la frente frame_original queda igual)
            frame_original = Swapper_Faces.igualar_tono_frente(frame_original, fake_frame, cara, rectangulo)
            mascara = np.maximum(mascara, mascara_frente[y1:y2, x1:x2])
        mascara = mascara[..., None]

        resultado = fake_frame.copy()
        resultado[y1:y2, x1:x2] = (frame_original[y1:y2, x1:x2].astype(np.float32) * mascara
                                   + fake_frame[y1:y2, x1:x2].astype(np.float32) * (1 - mascara)).astype(np.uint8)
        if not devolver_mascara:
            return resultado
        completa = np.zeros((alto, ancho), np.float32)
        completa[y1:y2, x1:x2] = mascara[..., 0]
        return resultado, completa


    @staticmethod
    def mascara_frente(tamano, cara, devolver_rectangulo=False):

        '''
         Mascara (0 a 1, del tamaño del frame) de la frente: todo lo que esta por encima de las cejas, entre las
        sienes. Se mide en el sistema de la cara (eje x = de un ojo al otro, eje y = hacia el menton, en distancias
        entre ojos), asi sigue a la cara si esta inclinada. Abajo empieza suave (0 justo arriba de la ceja, 1 a
        FRENTE_COMPLETA) y a los lados se apaga en las sienes. Hacia arriba no tiene limite: la zona pegada ya
        termina en la raiz del pelo.
        Con devolver_rectangulo=True devuelve (mascara, (x1, y1, x2, y2)): el rectangulo fuera del cual es 0.
        '''

        alto, ancho = tamano
        centro, u, v, d = Swapper_Faces._ejes_cara(cara)

        ceja = Swapper_Faces._borde_cejas(cara)

        # Solo se calcula en el rectangulo donde puede haber frente (la mascara pegada no pasa de ~1.6 d arriba)
        esquinas = np.array([[sx * 1.2, sy] for sx in (-1, 1) for sy in (ceja, ceja - 1.8)], np.float32)
        esquinas = centro + d * (esquinas[:, :1] * u + esquinas[:, 1:] * v)
        x1, y1 = np.floor(esquinas.min(0)).astype(int)
        x2, y2 = np.ceil(esquinas.max(0)).astype(int) + 1
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(ancho, x2), min(alto, y2)
        mascara = np.zeros(tamano, dtype=np.float32)
        if x2 <= x1 or y2 <= y1:
            return (mascara, None) if devolver_rectangulo else mascara

        rel_x, rel_y = Swapper_Faces._en_ejes_cara(cara, x1, y1, x2, y2)

        def subida(x):  # 0 -> 1 suave (smoothstep)
            x = np.clip(x, 0, 1)
            return x * x * (3 - 2 * x)

        vertical = subida(((ceja - FRENTE_INICIO) - rel_y) / (FRENTE_COMPLETA - FRENTE_INICIO))
        lados = subida((FRENTE_ANCHO - np.abs(rel_x)) / 0.25)
        mascara[y1:y2, x1:x2] = vertical * lados
        return (mascara, (x1, y1, x2, y2)) if devolver_rectangulo else mascara


    @staticmethod
    def igualar_tono_frente(frame_original, fake_frame, cara, rectangulo):

        '''
         Devuelve frame_original con la frente (el rectangulo x1, y1, x2, y2 de mascara_frente) llevada al tono de la cara cambiada. La diferencia
        de tono se mide en el entrecejo y el puente de la nariz, junto al borde y sin cerquillo: por canal LAB,
        mediana(cara cambiada) - mediana(destino), y se suma a la frente del destino. Asi se conserva la textura y
        las sombras propias de la frente del destino, solo cambia el tono. La mediana no se deja llevar por lo que
        tape esa zona (una mano que la oclusion devuelve del destino da diferencia 0 en esos pixeles).
        '''

        alto, ancho = frame_original.shape[:2]
        centro, u, v, d = Swapper_Faces._ejes_cara(cara)
        y_min, y_max = Swapper_Faces._borde_cejas(cara) + 0.02, REFERENCIA_TONO_Y
        esquinas = np.array([[sx * REFERENCIA_TONO_X, sy] for sx in (-1, 1) for sy in (y_min, y_max)], np.float32)
        esquinas = centro + d * (esquinas[:, :1] * u + esquinas[:, 1:] * v)
        x1, y1 = np.maximum(np.floor(esquinas.min(0)).astype(int), 0)
        x2, y2 = np.ceil(esquinas.max(0)).astype(int) + 1
        x2, y2 = min(ancho, x2), min(alto, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return frame_original
        rel_x, rel_y = Swapper_Faces._en_ejes_cara(cara, x1, y1, x2, y2)
        zona = (np.abs(rel_x) < REFERENCIA_TONO_X) & (rel_y > y_min) & (rel_y < y_max)
        if zona.sum() < 50:
            return frame_original
        lab_fake = cv2.cvtColor(fake_frame[y1:y2, x1:x2], cv2.COLOR_BGR2LAB)[zona].astype(np.float32)
        lab_dst = cv2.cvtColor(frame_original[y1:y2, x1:x2], cv2.COLOR_BGR2LAB)[zona].astype(np.float32)
        diferencia = np.median(lab_fake, 0) - np.median(lab_dst, 0)

        if rectangulo is None or not np.any(np.abs(diferencia) >= 0.5):
            return frame_original
        fx1, fy1, fx2, fy2 = rectangulo
        lab = cv2.cvtColor(frame_original[fy1:fy2, fx1:fx2], cv2.COLOR_BGR2LAB).astype(np.float32) + diferencia
        resultado = frame_original.copy()
        resultado[fy1:fy2, fx1:fx2] = cv2.cvtColor(np.clip(np.rint(lab), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return resultado


    @staticmethod
    def _borde_cejas(cara):

        '''Altura del borde de arriba de las cejas en el sistema de la cara (negativa: arriba de los ojos)'''

        puntos = cara.get("landmark_2d_106")
        if puntos is None:
            return -0.40  # donde suelen estar
        centro, _, v, d = Swapper_Faces._ejes_cara(cara)
        cejas = np.concatenate([puntos[list(r)] for r in CEJAS_106]).astype(np.float32)
        return float(np.percentile((cejas - centro) @ v, 20)) / d


    @staticmethod
    def _ejes_cara(cara):

        '''Sistema de la cara: centro entre los ojos, u hacia el otro ojo, v hacia el menton, d = distancia entre ojos'''

        izq, der = cara.kps[0].astype(np.float32), cara.kps[1].astype(np.float32)
        d = float(np.linalg.norm(der - izq))
        u = (der - izq) / d
        return (izq + der) / 2, u, np.array([-u[1], u[0]]), d


    @staticmethod
    def _en_ejes_cara(cara, x1, y1, x2, y2):

        '''Coordenadas (en distancias entre ojos) de cada pixel del rectangulo, en el sistema de la cara'''

        centro, u, v, d = Swapper_Faces._ejes_cara(cara)
        ys, xs = np.mgrid[y1:y2, x1:x2].astype(np.float32)
        xs, ys = xs - centro[0], ys - centro[1]
        return (xs * u[0] + ys * u[1]) / d, (xs * v[0] + ys * v[1]) / d


    @staticmethod
    def conservar_ojos(frame_original, fake_frame, cara):
        return Swapper_Faces.conservar_partes(frame_original, fake_frame, cara, ojos=True)
