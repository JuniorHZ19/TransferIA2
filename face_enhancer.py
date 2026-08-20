import cv2
import numpy as np
import torch
import os
import gdown
import insightface
from insightface.app import FaceAnalysis
import warnings
warnings.filterwarnings("ignore")

class GPENFaceEnhancer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Inicializa el mejorador de rostros usando GPEN
        Args:
            device: Dispositivo para inferencia ('cuda' o 'cpu')
        """
        self.device = device
        self.model_path = os.path.join("models", "GPEN-BFR-512.pth")
        self.face_analyzer = FaceAnalysis(name='buffalo_l')
        self.face_analyzer.prepare(ctx_id=0)
        self.model = self._load_model()
        print(f"Modelo GPEN inicializado en dispositivo: {device}")
        
    def _load_model(self):
        """
        Carga el modelo GPEN
        Returns:
            torch.nn.Module: Modelo GPEN cargado
        """
        try:
            from gpen_model import FullGenerator
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"No se encontró el modelo en: {self.model_path}")
                
            model = FullGenerator(512, 512, 3, style_dim=512, n_mlp=8)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['g_ema'])
            model.eval()
            model.to(self.device)
            return model
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            return None

    def enhance_face(self, face_img):
        """
        Mejora una región facial específica
        Args:
            face_img: Imagen numpy del rostro
        Returns:
            numpy.ndarray: Rostro mejorado
        """
        if self.model is None:
            return face_img
            
        with torch.no_grad():
            try:
                # Preprocesar
                face_tensor = self._preprocess_face(face_img)
                
                # Inferencia
                enhanced_face = self.model(face_tensor)
                
                # Postprocesar
                enhanced_face = enhanced_face.cpu().numpy().transpose(0, 2, 3, 1)[0]
                enhanced_face = np.clip(enhanced_face * 255, 0, 255).astype(np.uint8)
                
                return enhanced_face
            except Exception as e:
                print(f"Error en enhance_face: {str(e)}")
                return face_img

    def _preprocess_face(self, face_img):
        """
        Preprocesa la imagen facial para GPEN
        Args:
            face_img: Imagen numpy del rostro
        Returns:
            torch.Tensor: Tensor preparado para el modelo
        """
        face_img = cv2.resize(face_img, (512, 512))
        face_img = face_img.astype(np.float32) / 255.0
        face_img = torch.from_numpy(face_img.transpose(2, 0, 1)).unsqueeze(0)
        return face_img.to(self.device)

    def enhance_image(self, img, enhance_factor=1.0):
        """
        Mejora los rostros en una imagen
        Args:
            img: Imagen numpy (BGR)
            enhance_factor: Factor de mejora (0.0 a 1.0)
        Returns:
            numpy.ndarray: Imagen con rostros mejorados
        """
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("La entrada debe ser una imagen válida (numpy array)")
        
        # Hacer una copia para el resultado
        result = img.copy()
        
        try:
            # Detectar rostros
            faces = self.face_analyzer.get(img)
            
            # Procesar cada rostro
            for face in faces:
                # Obtener coordenadas del rostro
                bbox = face.bbox.astype(np.int32)
                x1, y1, x2, y2 = bbox
                
                # Añadir margen
                margin = 0.2  # 20% de margen
                h, w = y2 - y1, x2 - x1
                x1 = max(0, int(x1 - w * margin))
                y1 = max(0, int(y1 - h * margin))
                x2 = min(img.shape[1], int(x2 + w * margin))
                y2 = min(img.shape[0], int(y2 + h * margin))
                
                # Extraer región del rostro
                face_region = img[y1:y2, x1:x2]
                
                # Mejorar rostro
                enhanced_face = self.enhance_face(face_region)
                
                # Redimensionar si es necesario
                if enhanced_face.shape[:2] != (y2-y1, x2-x1):
                    enhanced_face = cv2.resize(enhanced_face, (x2-x1, y2-y1))
                
                # Crear máscara de fusión suave
                mask = np.zeros((y2-y1, x2-x1), dtype=np.float32)
                cv2.rectangle(mask, (0, 0), (x2-x1, y2-y1), (1, 1, 1), -1)
                mask = cv2.GaussianBlur(mask, (31, 31), 0)
                mask = np.stack([mask] * 3, axis=2) * enhance_factor
                
                # Fusionar con la imagen original
                result[y1:y2, x1:x2] = (
                    enhanced_face * mask + 
                    result[y1:y2, x1:x2] * (1 - mask)
                )
            
            return result
            
        except Exception as e:
            print(f"Error en enhance_image: {str(e)}")
            return img

# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar el enhancer
    enhancer = GPENFaceEnhancer()
    
    # Cargar y procesar una imagen
    img = cv2.imread("test.jpg")
    if img is not None:
        # Mejorar imagen
        result = enhancer.enhance_image(img, enhance_factor=0.8)
        
        # Mostrar resultado
        cv2.imshow("Original", img)
        cv2.imshow("Enhanced", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No se pudo cargar la imagen de prueba") 