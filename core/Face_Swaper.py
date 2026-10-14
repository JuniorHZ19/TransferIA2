import onnxruntime as ort
import cv2
import numpy as np
import insightface
import array
from core.Face_Analyzer import Analyzer_Faces
from core.xseg import DFL_XSeg_Model
import threading
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



    def __init__(self,path):

        '''Creamos en el constructor o incializamos el modelo Analyzer_Faces  solo una vez'''
       

        if self.__class__._SWAPPER_MODEL is None:
           self.__class__._SWAPPER_MODEL=insightface.model_zoo.get_model(path, providers = ["CUDAExecutionProvider",
                                                                                                "CPUExecutionProvider"
                                                                                            ])
      
    def get_Swapper_Model(self):
        return  self.__class__._SWAPPER_MODEL
    

    def cambiar_Faces(self,frame_dst,frame_src,index_src="0",index_dst="0",oclution=True,mode="None"):

     
        indices_src = [int(i) for i in index_src.split(',')]

        indices_dst = [int(i) for i in index_dst.split(',')]


        caras_src=self.__class__._ANALIZADOR.obtener_caras(frame_src)

        caras_dst=self.__class__._ANALIZADOR.obtener_caras(frame_dst)

        fake_frame=frame_dst

        
        for i in range(min(len(caras_src), len(caras_dst),len(index_src),len(indices_dst))):
        

         frame_dst=fake_frame

         indice_src=indices_src[i]

         indice_dst=indices_dst[i]
         
         fake_frame=Swapper_Faces._SWAPPER_MODEL.get(frame_dst,source_face= caras_src[indice_src],target_face= caras_dst[indice_dst], paste_back=True)
        
         if oclution == True:
            with self.__class__._LOCK:
                fake_frame_oclucion=self.__class__._XSEG_MODEL.frame_with_occlusion(frame=frame_dst , 
                                                                                            fake_frame=fake_frame,
                                                                                            face=caras_dst[indice_dst],
                                                                                            mode=mode)
                fake_frame=fake_frame_oclucion
                    


        return fake_frame


   
