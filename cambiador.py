import cv2
import numpy as np
import os
from face_enhancer import GPENFaceEnhancer
from core.Face_Swaper import Swapper_Faces

class FaceProcessor:
    def __init__(self, swapper_model_path="models/inswapper128.onnx"):
        """
        Inicializa el procesador de rostros con GPEN y Face Swapper
        Args:
            swapper_model_path: Ruta al modelo de face swapping
        """
        try:
            # Verificar que existe el modelo de face swapping
            if not os.path.exists(swapper_model_path):
                raise FileNotFoundError(f"No se encontró el modelo de face swapping en: {swapper_model_path}")

            # Inicializar el mejorador de rostros GPEN
            self.enhancer = GPENFaceEnhancer()
            
            # Inicializar el face swapper con el modelo específico
            self.swapper = Swapper_Faces(path=swapper_model_path)
            print("Sistema inicializado correctamente")
        except Exception as e:
            print(f"Error al inicializar: {str(e)}")
            raise e

    def process_image(self, source_img, target_img, enhance_result=True):
        """
        Procesa las imágenes aplicando primero face swap y luego mejora GPEN
        Args:
            source_img: Imagen fuente (de donde se toma el rostro)
            target_img: Imagen objetivo (donde se pondrá el rostro)
            enhance_result: Si se debe mejorar el resultado final con GPEN
        Returns:
            numpy.ndarray: Imagen resultante con el rostro swapeado y mejorado
        """
        try:
            # 1. Realizar el face swap primero
            print("Realizando face swap...")
            swapped_img = self.swapper.cambiar_Faces(target_img, source_img, index_dst="0", index_src="0")
            
            if swapped_img is None:
                print("Error: El face swap no produjo ningún resultado")
                return None

            # 2. Mejorar el resultado con GPEN si se solicita
            if enhance_result:
                print("Mejorando resultado con GPEN...")
                swapped_img = self.enhancer.enhance_image(swapped_img, enhance_factor=0.8)

            return swapped_img

        except Exception as e:
            print(f"Error en el procesamiento: {str(e)}")
            return None

def main():
    try:
        # Verificar que existen las carpetas necesarias
        if not os.path.exists("models"):
            print("Error: No se encontró la carpeta 'models'")
            return

        # Inicializar el procesador con la ruta correcta al modelo
        processor = FaceProcessor(swapper_model_path="models/inswapper128.onnx")

        # Cargar imágenes (usando imágenes que existen en el directorio)
        source_path = "cara.jpg"  # Imagen del rostro que quieres usar
        target_path = "dst.jpeg"  # Imagen donde quieres poner el rostro

        print(f"Cargando imagen fuente: {source_path}")
        source_img = cv2.imread(source_path)
        print(f"Cargando imagen objetivo: {target_path}")
        target_img = cv2.imread(target_path)

        if source_img is None:
            raise ValueError(f"No se pudo cargar la imagen fuente: {source_path}")
        if target_img is None:
            raise ValueError(f"No se pudo cargar la imagen objetivo: {target_path}")

        print("Dimensiones de imagen fuente:", source_img.shape)
        print("Dimensiones de imagen objetivo:", target_img.shape)

        # Procesar imágenes - primero swap, luego GPEN
        result = processor.process_image(
            source_img, 
            target_img,
            enhance_result=True  # True para aplicar GPEN al resultado final
        )

        if result is not None:
            # Mostrar resultados
            cv2.imshow("Fuente", source_img)
            cv2.imshow("Objetivo", target_img)
            cv2.imshow("Resultado", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Guardar resultado
            output_path = "resultado_final.jpg"
            cv2.imwrite(output_path, result)
            print(f"Resultado guardado como: {output_path}")
        else:
            print("No se pudo generar el resultado")

    except Exception as e:
        print(f"Error en la ejecución: {str(e)}")

if __name__ == "__main__":
    main()


