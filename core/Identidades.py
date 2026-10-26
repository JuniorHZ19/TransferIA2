import threading
import numpy as np


class Registro_Identidades:

    '''
     Registra automaticamente las caras destino en el orden en que aparecen por primera vez en el video
    (identidad 0, 1, 2 ...) usando el embedding de identidad (normed_embedding) que ya calcula buffalo_l.
    Luego empareja cada cara de un frame con su identidad, sin importar en que posicion de la imagen este.

     registrar() debe llamarse en orden de frames (hilo lector); los hilos de procesamiento solo usan
    una copia de las referencias (copia_refs) y asignar(), que no modifica nada.
    '''

    def __init__(self, max_ids, umbral_nuevo=0.35, score_min=0.5, ojos_min=12):

        self.max_ids = max_ids
        self.umbral_nuevo = umbral_nuevo  # similitud menor a esto contra todas las referencias = persona nueva
        self.score_min = score_min        # det_score minimo para usar una cara como referencia
        self.ojos_min = ojos_min          # distancia minima entre ojos (px) para usar una cara como referencia
        self.refs = []                    # embeddings normalizados de cada identidad
        self.calidad = []                 # calidad de la cara usada como referencia de cada identidad
        self._lock = threading.Lock()

    def completo(self):
        return len(self.refs) >= self.max_ids

    def calidad_cara(self, cara):

        '''0 si la cara no sirve como referencia (borrosa, pequeña o de perfil); mayor valor = mejor toma'''

        ojo_i, ojo_d, nariz = cara.kps[0], cara.kps[1], cara.kps[2]
        dist_ojos = float(np.linalg.norm(ojo_d - ojo_i))

        if cara.det_score < self.score_min or dist_ojos < self.ojos_min:
            return 0.0

        # Frontalidad: la nariz debe caer entre los ojos (0.5 = centrada)
        eje = (ojo_d - ojo_i) / dist_ojos
        pos_nariz = float(np.dot(nariz - ojo_i, eje) / dist_ojos)
        if not 0.2 <= pos_nariz <= 0.8:
            return 0.0

        return float(cara.det_score * dist_ojos * (1 - abs(pos_nariz - 0.5)))

    def registrar(self, caras):

        '''Agrega personas nuevas y mejora referencias con tomas de mejor calidad.
           Devuelve [(identidad, cara)] de las identidades registradas en esta llamada.'''

        nuevas = []
        with self._lock:
            for cara in caras:
                q = self.calidad_cara(cara)
                if q == 0:
                    continue

                emb = cara.normed_embedding
                sims = [float((np.atleast_2d(r) @ emb).max()) for r in self.refs]

                if sims and max(sims) >= self.umbral_nuevo:
                    k = int(np.argmax(sims))
                    if q > self.calidad[k]:  # misma persona con mejor toma: reemplazar referencia
                        self.refs[k], self.calidad[k] = emb, q
                elif not self.completo():
                    self.refs.append(emb)
                    self.calidad.append(q)
                    nuevas.append((len(self.refs) - 1, cara))

        return nuevas

    def copia_refs(self):
        with self._lock:
            return [r.copy() for r in self.refs]

    def guardar(self):
        '''Referencias registradas, para guardarlas y continuar un video cortado'''
        with self._lock:
            return {"refs": [r.astype(float).tolist() for r in self.refs], "calidad": list(self.calidad)}

    def cargar(self, datos):
        with self._lock:
            self.refs = [np.asarray(r, dtype=np.float32) for r in datos.get("refs", [])][:self.max_ids]
            self.calidad = [float(q) for q in datos.get("calidad", [])][:len(self.refs)]

    def fijar(self, refs):

        '''
         Usa identidades ya conocidas (del analisis previo del video, ver core/Personas.py) en lugar de registrarlas
        mientras se procesa. Cada ref puede ser un vector o una matriz (varias tomas de la persona). Quedan fijas:
        con todas las identidades puestas, registrar() ya no agrega ni reemplaza nada.
        '''

        with self._lock:
            self.refs = [np.asarray(r, dtype=np.float32) for r in refs]
            self.calidad = [1e9] * len(self.refs)
            self.max_ids = len(self.refs)

    @staticmethod
    def asignar(caras, refs, umbral=0.35):

        '''Empareja caras con identidades: [(cara, identidad)]. Voraz por mayor similitud, sin repetir
           caras ni identidades; las caras que no se parecen a ninguna referencia se dejan fuera.
           Una referencia puede ser un vector o una matriz con varias tomas de la persona (de frente, de lado,
           con otra luz): cuenta la toma que mas se parece a la cara.'''

        if not caras or not refs:
            return []

        embeddings = np.stack([c.normed_embedding for c in caras])
        sim = np.stack([(embeddings @ np.atleast_2d(r).T).max(axis=1) for r in refs], axis=1)
        pares, usadas_c, usadas_k = [], set(), set()

        for idx in np.argsort(-sim, axis=None):
            i, k = np.unravel_index(idx, sim.shape)
            if sim[i, k] < umbral:
                break
            if i not in usadas_c and k not in usadas_k:
                pares.append((caras[i], int(k)))
                usadas_c.add(i)
                usadas_k.add(k)

        return pares
