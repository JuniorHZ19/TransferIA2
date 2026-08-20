import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import torch
from PIL import Image
import os

class FaceEnhancer:
    def __init__(self):
        """
        Inicializa el mejorador de rostros pixelados
        """
        # Inicializar el detector de rostros
        self.face_analyzer = FaceAnalysis(name='buffalo_l')
        self.face_analyzer.prepare(ctx_id=0)
        
        # Asegurar que existe el directorio de salida
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

    def enhance_face_region(self, face_img):
        """
        Mejora la calidad de una región facial específica
        Args:
            face_img: Imagen recortada del rostro
        Returns:
            Imagen mejorada del rostro
        """
        # Convertir a escala de grises para análisis de calidad
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Calcular la varianza local para detectar regiones pixeladas
        local_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Si la varianza es baja, aplicar técnicas de mejora
        if local_var < 100:  # Umbral ajustable
            # Aplicar técnicas de mejora preservando detalles
            enhanced = cv2.detailEnhance(face_img, sigma_s=10, sigma_r=0.15)
            
            # Reducir ruido mientras se preservan bordes
            enhanced = cv2.edgePreservingFilter(enhanced, flags=1, sigma_s=60, sigma_r=0.4)
            
            # Equilibrar el contraste local
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            
            return enhanced
        
        return face_img

    def process_image(self, img, output_path=None):
        """
        Procesa una imagen para mejorar los rostros pixelados
        Args:
            img: Imagen en formato OpenCV (numpy array)
            output_path: Ruta opcional para guardar la imagen
        Returns:
            Imagen procesada con rostros mejorados
        """
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("La entrada debe ser una imagen válida en formato OpenCV (numpy array)")
        
        # Hacer una copia para el resultado
        result = img.copy()
        
        # Detectar rostros
        faces = self.face_analyzer.get(img)
        
        # Procesar cada rostro
        for face in faces:
            # Obtener coordenadas del rostro
            bbox = face.bbox.astype(np.int32)
            x1, y1, x2, y2 = bbox
            
            # Extraer región del rostro con margen
            margin = 0.2  # 20% de margen
            h, w = y2 - y1, x2 - x1
            x1 = max(0, int(x1 - w * margin))
            y1 = max(0, int(y1 - h * margin))
            x2 = min(img.shape[1], int(x2 + w * margin))
            y2 = min(img.shape[0], int(y2 + h * margin))
            
            face_region = img[y1:y2, x1:x2]
            
            # Mejorar la región del rostro
            enhanced_face = self.enhance_face_region(face_region)
            
            # Fusionar suavemente el rostro mejorado
            mask = np.zeros((y2-y1, x2-x1), dtype=np.float32)
            cv2.rectangle(mask, (0, 0), (x2-x1, y2-y1), (1, 1, 1), -1)
            mask = cv2.GaussianBlur(mask, (31, 31), 0)
            mask = np.stack([mask] * 3, axis=2)
            
            result[y1:y2, x1:x2] = enhanced_face * mask + result[y1:y2, x1:x2] * (1 - mask)
        
        # Guardar resultado si se especifica una ruta
        if output_path:
            cv2.imwrite(output_path, result)
        
        return result

    def enhance_video(self, input_path, output_path):
        """
        Procesa un video para mejorar los rostros pixelados
        Args:
            input_path: Ruta del video de entrada
            output_path: Ruta para guardar el video procesado
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {input_path}")
        
        # Obtener propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Configurar el escritor de video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Procesar el frame
                enhanced_frame = self.process_image(frame)
                
                # Escribir el frame procesado
                out.write(enhanced_frame)
                
        finally:
            cap.release()
            out.release()

# Ejemplo de uso:
if __name__ == "__main__":
    enhancer = FaceEnhancer()
    
    # Ejemplo de uso con imagen ya cargada con OpenCV
    img = cv2.imread("ruta/a/imagen_con_rostros_pixelados.jpg")
    if img is not None:
        try:
            # Procesar la imagen
            result = enhancer.process_image(img, "output/imagen_mejorada.jpg")
            print("Imagen procesada guardada en: output/imagen_mejorada.jpg")
        except Exception as e:
            print(f"Error al procesar la imagen: {str(e)}")
    else:
        print("No se pudo cargar la imagen de ejemplo") 