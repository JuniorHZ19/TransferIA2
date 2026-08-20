import onnxruntime as ort
import cv2
import numpy as np
import threading
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
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
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

    def get_complete_mask(self, img, face):
        """
        Genera una máscara más suave y precisa, con manejo robusto de dimensiones
        """
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
        

    def frame_with_occlusion(self, frame, fake_frame, face, mode="natural"):
        """
        Proceso mejorado para una fusión suave entre la cara fake y el fondo,
        aplicando nitidez solo a la cara fake
        """
        try:
            # Asegurar que los frames son arrays de numpy con el tipo correcto
            frame = np.asarray(frame).astype(np.uint8)
            fake_frame = np.asarray(fake_frame).astype(np.uint8)
            
            if frame.shape != fake_frame.shape:
                print(f"Error: Shape mismatch - frame: {frame.shape}, fake_frame: {fake_frame.shape}")
                return frame
            
            # 1. Aplicar filtro de nitidez solo a la cara fake
            fake_frame_nitido = self.filtro_image(fake_frame, mode)
            
            # 2. Obtener la máscara base
            mascara = self.get_complete_mask(frame, face)
            if mascara is None:
                print("Warning: No se pudo generar la máscara")
                return fake_frame_nitido
            
            # Validar que la máscara no esté vacía y tenga valores válidos
            if np.max(mascara) == 0:
                print("Warning: Máscara vacía detectada")
                return frame
            
            # 3. Crear una máscara suave para la transición con parámetros más conservadores
            mascara_suave = cv2.GaussianBlur(mascara, (15,15), 7)
            
            # 4. Convertir la máscara a float y normalizar con validación
            mascara_norm = np.clip(mascara_suave.astype(float) / 255.0, 0, 1)
            
            # 5. Crear una versión más suave para los bordes con parámetros más conservadores
            mascara_bordes = cv2.GaussianBlur(mascara, (31,31), 15)
            mascara_bordes = np.clip(mascara_bordes.astype(float) / 255.0, 0, 1)
            
            # 6. Combinar las máscaras para una mejor transición
            mascara_final = np.clip(mascara_norm * 0.8 + mascara_bordes * 0.2, 0, 1)
            
            # 7. Expandir la máscara para manejar los canales de color
            if len(mascara_final.shape) == 2:
                mascara_final = np.expand_dims(mascara_final, axis=2)
            
            # Validar la forma final de la máscara
            if mascara_final.shape[:2] != frame.shape[:2]:
                print("Error: Dimensiones de máscara incorrectas")
                return frame
            
            #cv2.imshow("final",mascara_final)
            
            # 8. Realizar la fusión suave usando la cara fake con nitidez
            result = (frame * (1 - mascara_final) + fake_frame_nitido * mascara_final)
            
            # 9. Asegurar que los valores estén en el rango correcto
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            # 10. Aplicar un suavizado final más sutil
            result = cv2.GaussianBlur(result, (3,3), 0.5)
            
            return result
            
        except Exception as e:
            print(f"Error detallado en frame_with_occlusion: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame
    
    
    def filtro_image(self,image,mode="None"):

      """Aumenta la nitidez de la imagen tiene 3 modos uno suave otro fuerte y otro none que no aplica ni un filtro"""

      if image is None or image.size == 0:
        return image
      
      if mode == "strong":
       kernel_nitidez = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
       
      elif  mode=="natural":
        kernel_nitidez = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
      elif mode =="None":
         return image
          
      return cv2.filter2D(image, -1, kernel_nitidez)

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


