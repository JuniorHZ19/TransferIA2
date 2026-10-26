import onnxruntime as ort
import cv2
import numpy as np
import insightface

# Angulos en los que se busca caras si no se encuentra ninguna en el frame original.
# Poner (0,) para volver al comportamiento anterior (solo deteccion normal).
ANGULOS_DETECCION = (0, 90, 180, 270)

# Si una cara esta inclinada (a la izquierda o derecha) mas de estos grados, se endereza la imagen
# y se vuelve a detectar para obtener puntos clave mas precisos. Poner None para desactivarlo.
UMBRAL_REFINAR_INCLINACION = 30

class Analyzer_Faces:

    '''

        FaceAnalysis de la biblioteca insightface proporciona una interfaz de alto nivel para realizar análisis faciales, incluyendo detección de rostros, 
        extracción de características,estimación de edad y género, y alineación facial. A continuación, vamos a manejar lo que son los "faces" ,cual teine siguieten atributos
       
        
                bbox: Coordenadas de la caja delimitadora del rostro.

                landmark: Coordenadas de los puntos clave faciales (ojos, nariz, boca, etc.).

                embedding: Vector de características del rostro para tareas de reconocimiento.

                normed_embedding: Versión normalizada del vector de características.

                gender: Predicción del género (0 para femenino, 1 para masculino).

                age: Estimación de la edad.


     '''
    
    _ANALIZADOR_CARA = None  # atributo de clase (constante/singleton)

    def __init__(self):

        '''Creamos en el constructor o incializamos el modelo Analyzer_Faces  solo una vez'''

        if Analyzer_Faces._ANALIZADOR_CARA is None:
            Analyzer_Faces._ANALIZADOR_CARA = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers = ["CUDAExecutionProvider",
                                "CPUExecutionProvider"
                            ]#providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            Analyzer_Faces._ANALIZADOR_CARA.prepare(ctx_id=0,det_size=(480, 480))  # preparar el modelo

    def preparar_modelo(self,ctx_id=0,dimension=480):
       
       Analyzer_Faces._ANALIZADOR_CARA.prepare(ctx_id, det_size=(dimension, dimension))


    def obtener_caras(self, frame , ordenamiento="iz-de", genero="todos"):

        '''Retornaremos el conjunto de caras que detecta nuestro analizador  usaremos get(img) de Anylzer_Faces cual devuelve un arrray de Faces.
           "genero" puede ser "todos", "mujer" o "hombre" para quedarse solo con esas caras'''
        caras=self.detectar_con_rotacion(frame)

        caras=self.filtrar_genero(caras,genero)

        caras_ordenadas=self.ordenamiento_caras(caras,ordenamiento)

        return caras_ordenadas



    def _rotar(self, frame, angulo):
        '''Rota el frame sin recortar las esquinas y devuelve la matriz inversa para volver al original'''
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angulo, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += nw / 2 - w / 2
        M[1, 2] += nh / 2 - h / 2
        return cv2.warpAffine(frame, M, (nw, nh)), cv2.invertAffineTransform(M)

    def _mapear(self, pts, M_inv):
        '''Aplica la matriz afin a un arreglo de puntos (N,2)'''
        return pts @ M_inv[:, :2].T + M_inv[:, 2]

    def detectar_con_rotacion(self, frame, angulos=None):
        '''Detecta caras probando cada angulo; las coordenadas (bbox, kps, landmarks) se devuelven en el
           sistema del frame original. Permite caras inclinadas a la izquierda, derecha o boca abajo.
           Una deteccion solo se da por buena si la cara queda casi derecha (+-45) en ese giro; si no,
           se sigue probando y por cada cara se elige la deteccion con mayor det_score.'''
        if angulos is None:
            angulos = ANGULOS_DETECCION

        candidatas = []
        for ang in angulos:
            if ang == 0:
                rot, M_inv = frame, None
            else:
                rot, M_inv = self._rotar(frame, ang)
            caras = Analyzer_Faces._ANALIZADOR_CARA.get(rot)
            confiables = all(abs(self.inclinacion(c)) <= 45 for c in caras)
            if M_inv is not None:
                for c in caras:
                    self._mapear_cara(c, M_inv)
            candidatas.extend(caras)
            if len(caras) > 0 and confiables:
                break

        caras = self._unir_duplicadas(candidatas)

        if UMBRAL_REFINAR_INCLINACION is not None:
            caras = [self._refinar_cara_inclinada(frame, c) for c in caras]

        return caras

    def _unir_duplicadas(self, caras):
        '''Si la misma cara se detecto en varios giros, se queda con la de mayor det_score'''
        elegidas = []
        for c in sorted(caras, key=lambda c: c.det_score, reverse=True):
            cx, cy = (c.bbox[:2] + c.bbox[2:]) / 2
            duplicada = False
            for e in elegidas:
                ex, ey = (e.bbox[:2] + e.bbox[2:]) / 2
                tam = max(e.bbox[2] - e.bbox[0], e.bbox[3] - e.bbox[1])
                if np.hypot(cx - ex, cy - ey) < tam * 0.5:
                    duplicada = True
                    break
            if not duplicada:
                elegidas.append(c)
        return elegidas

    def _mapear_cara(self, c, M_inv):
        '''Pasa bbox, kps y landmarks de una cara detectada en un frame rotado al frame original'''
        c.kps = self._mapear(c.kps, M_inv).astype(np.float32)
        x1, y1, x2, y2 = c.bbox
        esquinas = self._mapear(np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float32), M_inv)
        c.bbox = np.array([*esquinas.min(0), *esquinas.max(0)], dtype=np.float32)
        if c.get("landmark_2d_106") is not None:
            c.landmark_2d_106 = self._mapear(c.landmark_2d_106, M_inv).astype(np.float32)
        return c

    def inclinacion(self, cara):
        '''Angulo (grados) de la linea de los ojos: positivo = inclinada a un lado, negativo = al otro'''
        vx, vy = cara.kps[1] - cara.kps[0]
        return float(np.degrees(np.arctan2(vy, vx)))

    def _refinar_cara_inclinada(self, frame, cara):
        '''Si la cara esta muy inclinada, endereza la imagen segun la linea de los ojos, vuelve a detectar
           y se queda con la cara mas cercana a la original. Los puntos clave salen mas precisos.'''
        angulo = self.inclinacion(cara)
        if abs(angulo) <= UMBRAL_REFINAR_INCLINACION:
            return cara

        rot, M_inv = self._rotar(frame, angulo)
        candidatas = Analyzer_Faces._ANALIZADOR_CARA.get(rot)
        if not candidatas:
            return cara

        centro = cara.kps.mean(0)
        candidatas = [self._mapear_cara(c, M_inv) for c in candidatas]
        mejor = min(candidatas, key=lambda c: np.linalg.norm(c.kps.mean(0) - centro))

        # Aceptar solo si es la misma cara (centro cercano en relacion a la distancia entre ojos)
        dist_ojos = np.linalg.norm(cara.kps[1] - cara.kps[0])
        if np.linalg.norm(mejor.kps.mean(0) - centro) > dist_ojos or mejor.det_score < cara.det_score * 0.9:
            return cara
        return mejor

    def filtrar_genero(self, caras, genero="todos"):

        '''Filtra las caras por el genero que estima buffalo_l (face.gender: 0 = mujer, 1 = hombre)'''
        if genero == "mujer":
            return [c for c in caras if c.get("gender") == 0]
        if genero == "hombre":
            return [c for c in caras if c.get("gender") == 1]
        return caras

    def obtener_cara(self, frame, posicion):

        '''Retornaremos la posicion de una cara de un conjuto de Faces'''

        caras=self.obtener_caras(frame)

        return caras[posicion]
    

    def ordenamiento_caras(self,faces,ordenamiento="iz-de"):

        ''' Ordenar de izquierda a derecha (menor x del bounding box) o derecha izquierda de las caras'''
         
        faces_ordenadas=faces

        if ordenamiento == "iz-de":
         
         faces_ordenadas = sorted(faces, key=lambda face: face.bbox[0])
        
        elif ordenamiento == "de-iz":
         
         faces_ordenadas = sorted(faces, key=lambda face: face.bbox[0], reverse=True)

        return faces_ordenadas 
    

    def print_info_faces(self,img):

        '''Funcion que usa draw_on(img, faces) cual dibuja los resultados del análisis facial (como cajas delimitadoras y puntos clave)
           sobre la imagen original.Este devolvera una iamgen con su respectivo dibujos (cajas y puntos claves) en el'''
        
        faces=self.obtener_caras(img)  
        
        return Analyzer_Faces._ANALIZADOR_CARA.draw_on(img,faces)
        
         
