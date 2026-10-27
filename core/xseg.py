import onnxruntime as ort
import cv2
import numpy as np
import threading

# Interruptor: True = recorte alineado con los 5 puntos clave (soporta caras inclinadas
# y boca abajo). False = metodo anterior (recorte rectangular del bbox).
USAR_ALINEACION = True

# Posicion estandar de ojos, nariz y comisuras de la boca (normalizada 0-1)
FFHQ_TEMPLATE = np.array([
    [0.37691676, 0.46864664],  # ojo izq
    [0.62285697, 0.46912813],  # ojo der
    [0.50123859, 0.61331904],  # nariz
    [0.39308822, 0.72541100],  # boca izq
    [0.61150205, 0.72490465],  # boca der
], dtype=np.float32)

# Nitidez de la cara pegada. La cara de inswapper sale de un recorte de 128 px, asi que su detalle esta a una
# escala de (ancho de la cara / 128) px: el radio del enfoque se calcula con esa escala.
# natural / strong: fuerza fija del enfoque (mascara de enfoque sobre la luz).
NITIDEZ_FIJA = {"natural": 0.8, "strong": 1.6}

# Modo de la oclusion: cuanto se respeta lo que tapa la cara (pelo, manos, objetos) en el borde difuminado.
#   debil      la mascara difuminada tal cual (como antes): en el borde de un objeto queda hasta ~46 % de cara
#              y dentro del objeto un poco (7-16 %), porque el difuminado "derrama" cara hacia el objeto
#   adaptable  nunca mas cara de la que marca el modelo: el difuminado solo suaviza hacia la piel, no tapa
#              el objeto (dentro 0 %, en el borde exacto ~30 %); la transicion del lado de la piel no cambia
#   fuerte     ademas contrae la mascara (lo que queda bajo 50 % pasa a 0): el objeto queda limpio hasta su
#              borde, pero la cara nueva retrocede unos pixeles junto al objeto
MODOS_OCLUSION = ("debil", "adaptable", "fuerte")
# adaptativo: la fuerza se busca para que el detalle de la cara llegue a esta fraccion del detalle del destino
# (sin pasar de NITIDEZ_MAXIMA), y se agrega grano hasta GRANO_OBJETIVO del ruido del destino.
NITIDEZ_OBJETIVO = 0.9
NITIDEZ_MAXIMA = 3.0
GRANO_OBJETIVO = 0.85

class DFL_XSeg_Model:
    _XSEG_SESSION = None
    _LOCK = threading.Lock()
    def __init__(self, model_path):
        """
         Inicializamos la session de inferenica del modelo que hara la oclucion , 
         xseg.onnx 
           
        """
        
        if self.__class__._XSEG_SESSION is None:
            with self.__class__._LOCK:
                if self.__class__._XSEG_SESSION is None:
                    self.__class__._XSEG_SESSION = ort.InferenceSession(
                        model_path,
                        providers = ["CUDAExecutionProvider",
                                        "CPUExecutionProvider"
                                    ]#providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                                                    )
         # Almacenar nombres de entrada y salida una sola vez
        session = self.__class__._XSEG_SESSION
        self.input_names = [input.name for input in session.get_inputs()]
        self.output_names = [output.name for output in session.get_outputs()]
            

    def run_inference(self, img_input):
        """
        Ejecuta la inferencia sobre una imagen con exclusión mutua.
        """
        with self.__class__._LOCK:
            outputs = self.__class__._XSEG_SESSION.run(
                self.output_names,
                {self.input_names[0]: img_input}
            )
        return outputs
         
  

    def preprocess(self, img, input_size=(256, 256)):
        """
        Preprocesa la imagen para que sea compatible con el modelo de xseg.onnx.
        Cambia el tamaño de la imagen, normaliza, y convierte a formato de tensor.
        Las entradas del modelo son una imagen de (1,256,256,3) y la salida sera 
        una imagen binaria (1,256,256,1) cual es una maskara binaria del rostro
        """
        # Redimensionar la imagen
        img_resized = cv2.resize(img, input_size)
        
        # Normalización si es necesario (suponiendo que el rango de valores sea [0, 255])
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Si el modelo espera un formato específico, como RGB
        img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB)
        
        # Agregar una dimensión de batch
        img_input = np.expand_dims(img_rgb, axis=0)
        
        # Tendremos para devolver "img_input" con  el formato [batch_size, channels, height, width] que sera entrada para el modelo
               
        return img_input

    def crop_face(self,frame,face,expansion_factor=0.55):
       """
        Podemos hacer un recorte para que solo se muestre el rectangulo (bbox) de la cara, y es muy importante
        tener un recuador que cubra amplimante el rostro el "expasion_factor" nos ayuda a aumtenar las dimension del recuadro
        para que se localize bien el rostro,  mayor factor mayor agrandamiento del recorte .
       """
       try:
           x1, y1, x2, y2 = face.bbox 
        
           # Calcular el ancho y alto del recorte original
           ancho = x2 - x1
           alto = y2 - y1
           #landmarks_crop = face.landmark_2d_5 - np.array([x1, y1])
           # Expande el área del recorte de manera más controlada
           self.x1_exp = max(0, int(x1 - ancho * expansion_factor))
           self.y1_exp = max(0, int(y1 - alto * expansion_factor))
           self.x2_exp = min(frame.shape[1], int(x2 + ancho * expansion_factor))
           self.y2_exp = min(frame.shape[0], int(y2 + alto * expansion_factor))

           # Validar que el recorte es válido
           if self.x2_exp <= self.x1_exp or self.y2_exp <= self.y1_exp:
               print("Warning: Recorte inválido detectado")
               return None

           cropped_face = frame[int(self.y1_exp):int(self.y2_exp), int(self.x1_exp):int(self.x2_exp)]
           
           return cropped_face
       except Exception as e:
           print(f"Error en crop_face: {str(e)}")
           return None
       

    '''
        def alignt_face(self ,img,landmarks):
            
            src_pts = np.array(landmarks, dtype=np.float32)

            # Puntos destino normalizados
            dst_pts = np.float32([
                [70, 100],   # ojo izq
                [186, 100],  # ojo der
                [128, 160],  # nariz
            ])

            # Solo usamos ojos y nariz para transformación afín
            M, _ = cv2.estimateAffinePartial2D(src_pts[:3], dst_pts)

            # Aplicar transformación a la imagen original
            aligned_img = cv2.warpAffine(img, M, output_size)

            # Matriz inversa
            M_inv = cv2.invertAffineTransform(M)
            
            return aligned_img, M, M_inv
    '''

    def align_face(self, frame, face, size=256, margen=1.0):
        """
        Alinea la cara usando los 5 puntos clave (face.kps) para que el modelo xseg siempre
        la reciba derecha, sin importar la inclinacion. "margen" < 1 aleja la cara (mas contexto).
        Devuelve el recorte alineado y la matriz afin usada.
        """
        dst = ((FFHQ_TEMPLATE - 0.5) * margen + 0.5) * size
        M, _ = cv2.estimateAffinePartial2D(face.kps.astype(np.float32), dst,
                                           method=cv2.RANSAC, ransacReprojThreshold=100)
        crop = cv2.warpAffine(frame, M, (size, size),
                              borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
        return crop, M

    def _inferir(self, crop):
        """Mascara cruda (256x256, 0-1) de un recorte alineado"""
        return np.array(self.run_inference(self.preprocess(crop))).reshape((256, 256)).astype(np.float32)

    def get_complete_mask_alineada(self, img, face):
        """
        Genera la mascara sobre la cara alineada y la devuelve a su posicion/rotacion original
        con la matriz inversa.
        """
        try:
            h, w = img.shape[:2]
            crop, M = self.align_face(img, face)
            mask = self._inferir(crop)

            # Cara en primer plano: parte del recorte cae fuera del cuadro y se rellena repitiendo el borde. Esas
            # rayas el modelo las toma como un objeto que tapa la cara y corta la piel visible (entre nariz y boca).
            # Se repite con dos rellenos sin rayas (espejo y difuso) y se toma el mayor: un falso objeto que sale de
            # un relleno no sale en el otro, mientras que un objeto real esta dentro del cuadro y lo marcan ambos.
            valido = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (256, 256), borderValue=0) > 127
            if not valido.all():
                espejo = cv2.warpAffine(img, M, (256, 256), borderMode=cv2.BORDER_REFLECT_101, flags=cv2.INTER_AREA)
                difuso = crop.copy()
                difuso[~valido] = cv2.GaussianBlur(crop, (0, 0), 12)[~valido]
                mask = np.maximum(self._inferir(espejo), self._inferir(difuso))
            mask = cv2.GaussianBlur(mask, (7, 7), 3)
            mask = np.clip(np.power(np.clip(mask, 0, 1), 1.5), 0, 1)

            M_inv = cv2.invertAffineTransform(M)
            full = cv2.warpAffine(mask, M_inv, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
            full = np.uint8(np.clip(full, 0, 1) * 255)
            full = cv2.morphologyEx(full, cv2.MORPH_CLOSE, (3, 3), iterations=2)

            return cv2.GaussianBlur(full, (5, 5), 2)

        except Exception as e:
            print(f"Error generating mask: {str(e)}")
            return None

    def get_complete_mask(self, img, face):
        """
        Genera una máscara más suave y precisa, con manejo robusto de dimensiones
        """
        if USAR_ALINEACION and getattr(face, "kps", None) is not None:
            return self.get_complete_mask_alineada(img, face)

        try:
            img_blank = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

            crop_cara = self.crop_face(img, face)
            if crop_cara is None or crop_cara.size == 0:
                return None

            crop_height = int(self.y2_exp - self.y1_exp)
            crop_width = int(self.x2_exp - self.x1_exp)

            img_input = self.preprocess(crop_cara)
           
            outputs =self.run_inference(img_input) 

            outputs = np.array(outputs).reshape((256, 256, 1))
            outputs = cv2.resize(outputs, (crop_width, crop_height), interpolation=cv2.INTER_CUBIC)
            outputs = cv2.GaussianBlur(outputs, (7, 7), 3)
            outputs = np.power(outputs, 1.5)
            outputs = np.clip(outputs, 0, 1)

            mask_uint8 = np.uint8(outputs * 255)
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, (3, 3), iterations=2)

            mh, mw = mask_uint8.shape  # dimensiones reales de la máscara
            x1, y1 = int(self.x1_exp), int(self.y1_exp)
            x2, y2 = x1 + mw, y1 + mh

            # Prevenir que x2/y2 se salgan de la imagen
            x2 = min(x2, img_blank.shape[1])
            y2 = min(y2, img_blank.shape[0])

            # Ajustar máscara si se sale de los bordes
            mask_uint8 = mask_uint8[:(y2 - y1), :(x2 - x1)]
           
            
            if img_blank[y1:y2, x1:x2].shape != mask_uint8.shape:
                print(f"Dimension mismatch after adjustment: blank {img_blank[y1:y2, x1:x2].shape}, mask {mask_uint8.shape}")
                return None

            img_blank[y1:y2, x1:x2] = mask_uint8
            img_blank = cv2.GaussianBlur(img_blank, (5, 5), 2)
       
            return img_blank

        except Exception as e:
            print(f"Error generating mask: {str(e)}")
            return None
        

    def frame_with_occlusion(self, frame, fake_frame, face, mode="natural", devolver_mascara=False,
                             modo_oclusion="debil"):
        """
        Proceso mejorado para una fusión suave entre la cara fake y el fondo,
        aplicando nitidez solo a la cara fake.
        Con devolver_mascara=True devuelve (resultado, mascara): cuanto de cada pixel viene de fake_frame
        (0 a 1, del tamaño del frame), o 0.0 / 1.0 si se devolvio todo el frame o todo el fake
        modo_oclusion: debil, adaptable o fuerte (ver MODOS_OCLUSION)
        """
        resultado, mascara = self._oclusion(frame, fake_frame, face, mode, modo_oclusion)
        return (resultado, mascara) if devolver_mascara else resultado

    def _oclusion(self, frame, fake_frame, face, mode, modo_oclusion="debil"):
        try:
            # Asegurar que los frames son arrays de numpy con el tipo correcto
            frame = np.asarray(frame).astype(np.uint8)
            fake_frame = np.asarray(fake_frame).astype(np.uint8)
            
            if frame.shape != fake_frame.shape:
                print(f"Error: Shape mismatch - frame: {frame.shape}, fake_frame: {fake_frame.shape}")
                return frame, 0.0
            
            # 1. Obtener la máscara base
            mascara = self.get_complete_mask(frame, face)
            if mascara is None:
                print("Warning: No se pudo generar la máscara")
                return self.filtro_image(fake_frame, mode, face=face), 1.0

            # Validar que la máscara no esté vacía y tenga valores válidos
            if np.max(mascara) == 0:
                print("Warning: Máscara vacía detectada")
                return frame, 0.0

            # 2. Aplicar nitidez solo a la cara fake (el modo adaptativo la mide dentro de la mascara)
            fake_frame_nitido = self.filtro_image(fake_frame, mode, face=face, frame=frame, mascara=mascara)

            # 3. Difuminado del borde proporcional al tamaño de la cara (antes fijo: 7 y 15 px de sigma,
            # que en caras pequeñas borraba el borde y en caras grandes lo dejaba marcado)
            dist_ojos = float(np.linalg.norm(face.kps[1] - face.kps[0])) if getattr(face, "kps", None) is not None else 80.0
            sigma_suave = max(1.0, dist_ojos * 0.09)
            sigma_bordes = max(2.0, dist_ojos * 0.19)
            mascara_suave = cv2.GaussianBlur(mascara, (0, 0), sigma_suave)

            # 4. Convertir la máscara a float y normalizar con validación
            mascara_norm = np.clip(mascara_suave.astype(float) / 255.0, 0, 1)

            # 5. Crear una versión más suave para los bordes
            mascara_bordes = cv2.GaussianBlur(mascara, (0, 0), sigma_bordes)
            mascara_bordes = np.clip(mascara_bordes.astype(float) / 255.0, 0, 1)
            
            # 6. Combinar las máscaras para una mejor transición
            mascara_final = np.clip(mascara_norm * 0.8 + mascara_bordes * 0.2, 0, 1)

            # 6b. Que el difuminado no pinte cara sobre lo que la tapa (ver MODOS_OCLUSION)
            if modo_oclusion in ("adaptable", "fuerte"):
                mascara_final = np.minimum(mascara_final, mascara.astype(float) / 255.0)
            if modo_oclusion == "fuerte":
                mascara_final = (mascara_final.clip(0.5, 1) - 0.5) * 2
            
            # 7. Expandir la máscara para manejar los canales de color
            if len(mascara_final.shape) == 2:
                mascara_final = np.expand_dims(mascara_final, axis=2)
            
            # Validar la forma final de la máscara
            if mascara_final.shape[:2] != frame.shape[:2]:
                print("Error: Dimensiones de máscara incorrectas")
                return frame, 0.0
            
            #cv2.imshow("final",mascara_final)
            
            # 8. Realizar la fusión suave usando la cara fake con nitidez
            result = (frame * (1 - mascara_final) + fake_frame_nitido * mascara_final)
            
            # 9. Asegurar que los valores estén en el rango correcto
            # (ya no se desenfoca el fotograma completo al final: ablandaba el fondo y quitaba la nitidez)
            result = np.clip(result, 0, 255).astype(np.uint8)

            return result, mascara_final[..., 0]
            
        except Exception as e:
            print(f"Error detallado en frame_with_occlusion: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame, 0.0
    
    
    def filtro_image(self, image, mode="None", face=None, frame=None, mascara=None):

      """
       Aumenta la nitidez de la cara fake. Modos:
          None        sin filtro
          natural     enfoque suave (fuerza fija)
          strong      enfoque fuerte (fuerza fija)
          adaptativo  iguala el detalle y el grano de la cara original del destino (necesita frame y mascara)
       Se enfoca solo la luz (canal L de LAB, sin halos de color) con una mascara de enfoque
       L + fuerza * (L - desenfoque(L)), cuyo radio sigue el tamaño de la cara (detalle a escala ancho/128).
       Solo se procesa el rectangulo de la cara, no el fotograma entero.
      """

      if image is None or image.size == 0 or mode not in ("natural", "strong", "adaptativo"):
        return image
      if mode == "adaptativo" and (frame is None or mascara is None):
        mode = "natural"

      # Rectangulo de trabajo: la mascara si existe, si no el bbox de la cara agrandado
      alto, ancho = image.shape[:2]
      if mascara is not None and np.any(mascara > 25):
        ys, xs = np.where(mascara > 25)
        x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
      elif face is not None:
        bx1, by1, bx2, by2 = face.bbox
        margen = (bx2 - bx1) * 0.3
        x1, y1 = int(max(0, bx1 - margen)), int(max(0, by1 - margen))
        x2, y2 = int(min(ancho, bx2 + margen)), int(min(alto, by2 + margen))
      else:
        x1, y1, x2, y2 = 0, 0, ancho, alto
      if x2 - x1 < 4 or y2 - y1 < 4:
        return image

      ancho_cara = float(face.bbox[2] - face.bbox[0]) if face is not None else 128.0
      sigma = max(0.6, 0.8 * ancho_cara / 128.0)

      recorte = image[y1:y2, x1:x2]
      lab = cv2.cvtColor(recorte, cv2.COLOR_BGR2LAB).astype(np.float32)
      luz = lab[..., 0]
      detalle = luz - cv2.GaussianBlur(luz, (0, 0), sigma)

      if mode in NITIDEZ_FIJA:
        nueva = luz + NITIDEZ_FIJA[mode] * detalle
      else:
        zona = self._zona_piel(mascara[y1:y2, x1:x2], face, x1, y1)
        if zona is None:
          nueva = luz + NITIDEZ_FIJA["natural"] * detalle
        else:
          luz_dst = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
          objetivo = NITIDEZ_OBJETIVO * self._detalle(luz_dst, zona)
          fuerza = self._buscar_fuerza(luz, detalle, zona, objetivo)
          nueva = luz + fuerza * detalle

          # Grano: la cara generada sale mas limpia que la foto; se agrega ruido hasta parecerse al destino
          grano_dst = self._grano(luz_dst, zona)
          grano_fake = self._grano(nueva, zona)
          falta = GRANO_OBJETIVO ** 2 * grano_dst ** 2 - grano_fake ** 2
          if falta > 0:
            nueva = nueva + np.random.normal(0.0, np.sqrt(falta), nueva.shape).astype(np.float32)

      lab[..., 0] = np.clip(nueva, 0, 255)
      resultado = image.copy()
      resultado[y1:y2, x1:x2] = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
      return resultado

    @staticmethod
    def _zona_piel(mascara, face, x1, y1):
      """Pixeles de la cara donde medir detalle y grano: mascara de oclusion firme, sin el borde"""
      zona = (mascara > 200).astype(np.uint8)
      if face is not None and getattr(face, "kps", None) is not None:
        k = max(3, int(np.linalg.norm(face.kps[1] - face.kps[0]) * 0.15) | 1)
        zona = cv2.erode(zona, np.ones((k, k), np.uint8))
      zona = zona.astype(bool)
      return zona if zona.sum() >= 64 else None

    @staticmethod
    def _detalle(luz, zona):
      """Cantidad de detalle fino: varianza del laplaciano dentro de la zona"""
      return float(cv2.Laplacian(luz, cv2.CV_32F)[zona].var())

    @staticmethod
    def _grano(luz, zona):
      """Ruido o grano: desviacion de lo que queda al quitar un desenfoque de 1 px"""
      return float((luz - cv2.GaussianBlur(luz, (0, 0), 1.0))[zona].std())

    @staticmethod
    def _buscar_fuerza(luz, detalle, zona, objetivo):
      """
       Fuerza del enfoque (0 a NITIDEZ_MAXIMA) con la que el detalle de la cara llega al objetivo.
      El laplaciano es lineal: Lap(luz + f*detalle) = A + f*B, asi que su varianza es una parabola en f
      (var(A) + 2f cov(A,B) + f² var(B)) y la fuerza sale directa, sin probar valores.
      """
      A = cv2.Laplacian(luz, cv2.CV_32F)[zona].astype(np.float64)
      B = cv2.Laplacian(detalle, cv2.CV_32F)[zona].astype(np.float64)
      A -= A.mean()
      B -= B.mean()
      var_a, cov, var_b = (A * A).mean(), (A * B).mean(), (B * B).mean()
      if var_a >= objetivo:
        return 0.0
      if var_b <= 1e-9:
        return 0.0
      # f² var_b + 2f cov + (var_a - objetivo) = 0, raiz positiva
      fuerza = (-cov + np.sqrt(max(0.0, cov * cov - var_b * (var_a - objetivo)))) / var_b
      return float(np.clip(fuerza, 0.0, NITIDEZ_MAXIMA))

    def blend_with_mask(self,img1, img2, mask):
      
      """Combina dos imágenes usando una máscara suavizada"""
      # Convertir máscara a float y normalizar
      mask = mask.astype(float) / 255
        
      # Aplicar transformación gamma para ajustar los bordes
      mask = np.power(mask, 1.5)
        
      # Suavizar los bordes de la máscara
      mask = cv2.GaussianBlur(mask, (7,7), 3)
        
        # Asegurar que los valores estén en el rango correcto
      mask = np.clip(mask, 0, 1)
        
      if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=2)
        
      # Realizar la fusión con la máscara ajustada
      result = img1 * (1 - mask) + img2 * mask
    
      return result.astype(np.uint8)

    def process_mask(self,mask, kernel_size=7, iterations=2):
      """Procesa la máscara para mejorar los bordes y eliminar artefactos"""
      # Dilatar la máscara para cubrir mejor los bordes
      kernel = np.ones((kernel_size,kernel_size), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=iterations)
      
      # Suavizar los bordes
      mask = cv2.GaussianBlur(mask, (kernel_size*2+1, kernel_size*2+1), kernel_size//2)
      
      # Normalizar y ajustar contraste
      mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
      mask = np.clip(mask, 0, 255).astype(np.uint8)
      
      return mask
    


    def get_model_input_output_shapes(self,session,model_path):
     """
        Funcion general para poder ver informacion de las entradas y salidas que tiene el modelo
      """
   
    # Obtener detalles de las entradas
     input_info = {}
     for input_tensor in session.get_inputs():
        input_info[input_tensor.name] = {
            "shape": input_tensor.shape, 
            "type": input_tensor.type
        }
    
    # Obtener detalles de las salidas
     output_info = {}
     for output_tensor in session.get_outputs():
        output_info[output_tensor.name] = {
            "shape": output_tensor.shape, 
            "type": output_tensor.type
        }
    
     print("inputs:",input_info) 
     print("outpus:",output_info) 


