import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
from core.Imagenes import leer, guardar
import os
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import subprocess
import sys
from core.Face_Swaper import Swapper_Faces
from core.Face_Analyzer import Analyzer_Faces
from core.Identidades import Registro_Identidades
from core.Modelo_Cara import Modelo_Cara, CARPETA_MODELOS, FPS_VIDEO, MAX_FRAMES_VIDEO, CARPETA_CARAS, MARCA_CARAS
from core.Proceso_Carpeta import imagenes_carpeta, carpeta_salida, procesar_carpeta
from core import Configuraciones
from core.Empaquetar import empaquetar, ruta_zip, disponible as zip_disponible
from core.Personas import (Personas, validar_fps, validar_maximo, FPS as PERSONAS_FPS, MAX_FRAMES as PERSONAS_MAX,
                           MAX_FRAMES_RANGO as PERSONAS_MAX_RANGO)
import tkinterdnd2 as tkdnd
import shutil
from core.Proceso_Video import (procesar_video, Trabajo_Video, CONTENEDORES, CODECS, info_video, FPS_PRUEBA,
                                SEGUNDOS_PRUEBA)
from core.Mejorador import validar_minimo, MINIMO_POR_DEFECTO


# Alto maximo (px) de la lista de modelos de cara antes de mostrar la barra de desplazamiento
ALTO_LISTA_MODELOS = 380


def copiar_al_portapapeles(img):
    '''
     Copia una imagen BGR al portapapeles de Windows (formato CF_DIB), para pegarla en Paint, Word,
    WhatsApp, etc. Tkinter solo copia texto, por eso se usa la API de Windows directamente.
    '''
    if sys.platform != "win32":
        raise RuntimeError("copiar imagenes al portapapeles solo funciona en Windows")
    import ctypes
    from ctypes import wintypes
    from io import BytesIO

    buf = BytesIO()
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(buf, "BMP")
    dib = buf.getvalue()[14:]  # un BMP sin su cabecera de archivo (14 bytes) es un DIB

    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    kernel32.GlobalAlloc.argtypes = (wintypes.UINT, ctypes.c_size_t)
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = (wintypes.HGLOBAL,)
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = (wintypes.HGLOBAL,)
    kernel32.GlobalFree.argtypes = (wintypes.HGLOBAL,)
    user32.SetClipboardData.argtypes = (wintypes.UINT, wintypes.HANDLE)
    user32.SetClipboardData.restype = wintypes.HANDLE
    CF_DIB, GMEM_MOVEABLE = 8, 0x0002

    # Otro programa puede tener el portapapeles abierto un instante: reintentar un poco
    for _ in range(10):
        if user32.OpenClipboard(None):
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("el portapapeles esta ocupado por otro programa")
    try:
        user32.EmptyClipboard()
        memoria = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(dib))
        destino = kernel32.GlobalLock(memoria)
        ctypes.memmove(destino, dib, len(dib))
        kernel32.GlobalUnlock(memoria)
        if not user32.SetClipboardData(CF_DIB, memoria):
            kernel32.GlobalFree(memoria)  # si se acepto, la memoria pasa a ser de Windows
            raise RuntimeError("Windows no acepto la imagen")
    finally:
        user32.CloseClipboard()


class ModernButton(ttk.Button):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        
    def on_enter(self, e):
        self['style'] = 'Accent.TButton'
        
    def on_leave(self, e):
        self['style'] = 'TButton'

class DropFrame(ttk.LabelFrame):
    def __init__(self, master, text, callback, multi_callback=None,
                 filetypes=(("Image/Video files", "*.jpg *.jpeg *.png *.mp4 *.avi *.gif *.mkv"),), **kwargs):
        super().__init__(master, text=text, **kwargs)
        self.callback = callback
        self.multi_callback = multi_callback  # recibe la lista cuando se sueltan varios archivos o una carpeta
        self.filetypes = list(filetypes)
        
        # Recuadro de la vista previa: ocupa todo el espacio libre del panel (crece con la ventana)
        self.container = ttk.Frame(self, height=200, width=200)
        self.container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.container.pack_propagate(False)

        # Preview label
        self.preview = ttk.Label(self.container, text="Arrastra aquí o haz clic para seleccionar")
        self.preview.place(relx=0.5, rely=0.5, anchor="center")  # Centrar el texto

        # Soltar o hacer clic en cualquier parte del recuadro, no solo sobre el texto o la imagen
        for zona in (self.container, self.preview):
            zona.drop_target_register(tkdnd.DND_FILES)
            zona.dnd_bind('<<Drop>>', self.on_drop)
            zona.bind('<Button-1>', self.on_click)

        # Imagen mostrada (sin escalar) para volver a ajustarla cuando cambia el tamaño del recuadro
        self.imagen = None
        self._ajuste_pendiente = None
        self.container.bind('<Configure>', self._al_redimensionar)
        
    def on_drop(self, event):
        rutas = list(self.tk.splitlist(event.data))
        if not rutas:
            return
        if self.multi_callback and (len(rutas) > 1 or os.path.isdir(rutas[0])):
            self.multi_callback(rutas)
        else:
            self.callback(rutas[0])

    def on_click(self, event):
        file_path = filedialog.askopenfilename(filetypes=self.filetypes)
        if file_path:
            self.callback(file_path)
            
    def update_preview(self, img):
        self.imagen = img
        if img is None:
            self.preview.configure(text="Arrastra aquí o haz clic para seleccionar")
            self.preview.configure(image='')
            self.preview.image = None
            self.preview.place(relx=0.5, rely=0.5, anchor="center")
            return
        self._dibujar()

    def _al_redimensionar(self, event):
        # Reajustar la imagen al nuevo tamaño (una vez, cuando la ventana deja de cambiar)
        if self.imagen is None:
            return
        if self._ajuste_pendiente is not None:
            self.after_cancel(self._ajuste_pendiente)
        self._ajuste_pendiente = self.after(60, self._dibujar)

    def _dibujar(self):
        """
         Muestra la imagen completa lo más grande posible dentro del recuadro: se escala hasta tocar el borde
        más cercano, sin recortar nada y sin deformarla (conserva la proporción).
        """
        self._ajuste_pendiente = None
        img = self.imagen
        if img is None:
            return

        container_width = self.container.winfo_width()
        container_height = self.container.winfo_height()
        if container_width <= 1 or container_height <= 1:  # aún no está dibujado: probar cuando tenga tamaño
            self._ajuste_pendiente = self.after(60, self._dibujar)
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = min(container_width / w, container_height / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        # INTER_AREA al achicar (sin dientes de sierra), LANCZOS al agrandar (más nítido)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4)
        
        # Create photo image
        photo = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.preview.configure(image=photo, text="")
        self.preview.image = photo
        
        # Center in container
        self.preview.place(relx=0.5, rely=0.5, anchor="center")

class PreviewWindow(tk.Toplevel):
    # Formas de ver el preview: original y resultado juntos, una linea que se arrastra para comparar
    # (antes/despues sobre la misma imagen) o solo el resultado
    VISTAS = ("Deslizar", "Lado a lado", "Solo resultado")
    FONDO = (240, 240, 240)  # BGR, el mismo gris claro del recuadro
    SEPARACION = 10  # px entre original y resultado en "Lado a lado"
    # "Ver máscara": color de la zona cambiada (BGR) y cuanto tapa donde el cambio es total
    COLOR_MASCARA = (60, 220, 60)
    OPACIDAD_MASCARA = 0.45

    def __init__(self, parent, app, video_path=None, image=None, imagenes=None):
        super().__init__(parent)
        self.title("Preview - Transfer")
        self.geometry("1400x800")
        self.app = app
        self.video_path = video_path  # con video, el modo identidad usa el analisis previo de personas
        self.imagenes = imagenes or []  # destino carpeta: la barra pasa de una imagen a otra
        self._indice_mostrado = None

        # Ultimo frame mostrado: original y resultado (sin escalar), para cambiar de vista o de tamaño
        # sin volver a procesar
        self.original = None
        self.resultado = None
        self.mascara = None  # zona que cambio (0 a 1), solo si "Ver máscara" esta marcado
        self.corte = 0.5  # posicion de la linea en "Deslizar" (0 = todo resultado, 1 = todo original)
        self._escaladas = None  # (ancho, alto, original, resultado) ya ajustadas al recuadro
        self._ancho_mostrado = 0
        self._ajuste_pendiente = None
        # Procesado en segundo plano: uno a la vez; si llega otro pedido mientras tanto, solo se guarda el
        # ultimo (arrastrar la barra del video o un deslizador no encola decenas de frames)
        self._ocupado = False
        self._pendiente = None
        self._config_pendiente = None

        # Main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Selector de vista
        barra = ttk.Frame(main_frame)
        barra.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(barra, text="Vista:").pack(side=tk.LEFT, padx=(0, 5))
        self.vista_var = tk.StringVar(value=self.VISTAS[0])  # Deslizar
        for vista in self.VISTAS:
            ttk.Radiobutton(barra, text=vista, value=vista, variable=self.vista_var,
                            command=self.cambiar_vista).pack(side=tk.LEFT, padx=5)
        # Opcional: pinta sobre el resultado la zona que de verdad se cambio
        self.ver_mascara_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(barra, text="Ver máscara", variable=self.ver_mascara_var,
                        command=self.cambiar_mascara).pack(side=tk.LEFT, padx=(20, 5))
        self.ayuda_label = ttk.Label(barra, text="", foreground="gray")
        self.ayuda_label.pack(side=tk.RIGHT)
        self.estado_label = ttk.Label(barra, text="", foreground="#b05a00")
        self.estado_label.pack(side=tk.RIGHT, padx=(0, 15))

        # Preview area
        self.preview_frame = ttk.Frame(main_frame)
        self.preview_frame.pack(fill=tk.BOTH, expand=True)

        # Preview label con fondo gris claro para ver los límites
        self.preview_label = ttk.Label(self.preview_frame, background='#f0f0f0')
        self.preview_label.place(relx=0.5, rely=0.5, anchor="center")
        # En "Deslizar": clic o arrastre sobre la imagen mueve la linea
        self.preview_label.bind("<Button-1>", self.mover_corte)
        self.preview_label.bind("<B1-Motion>", self.mover_corte)
        # Clic derecho: copiar la imagen para pegarla en otro programa; Ctrl+C copia el resultado
        self.menu_copiar = tk.Menu(self, tearoff=0)
        self.menu_copiar.add_command(label="Copiar resultado", accelerator="Ctrl+C",
                                     command=lambda: self.copiar("resultado"))
        self.menu_copiar.add_command(label="Copiar original", command=lambda: self.copiar("original"))
        self.menu_copiar.add_command(label="Copiar lo que se ve (vista actual)",
                                     command=lambda: self.copiar("vista"))
        self.preview_label.bind("<Button-3>", self.abrir_menu_copiar)
        self.bind("<Control-c>", lambda e: self.copiar("resultado"))
        self.bind("<Control-C>", lambda e: self.copiar("resultado"))
        self._mostrada = None  # la imagen tal como se ve en pantalla (vista, rotulos y mascara)
        self._aviso_pendiente = None
        self.cambiar_vista()  # cursor y ayuda de la vista inicial

        # Bind resize event
        self.bind('<Configure>', self.on_resize)
        
        # Controls frame para video
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

            controls = ttk.Frame(main_frame)
            controls.pack(fill=tk.X, pady=(10, 0))
            
            # Frame counter
            self.frame_label = ttk.Label(controls, text="0/0")
            self.frame_label.pack(side=tk.RIGHT, padx=10)
            
            # Slider mejorado para video
            slider_frame = ttk.Frame(controls)
            slider_frame.pack(fill=tk.X, padx=10, expand=True)
            
            # Con "Solo un fragmento", la barra recorre solo ese pedazo
            fragmento = app.fragmento_activo()
            self.es_fragmento = fragmento is not None
            self.primer_frame, self.ultimo_frame = (fragmento[0], fragmento[1] - 1) if fragmento else (0, self.total_frames - 1)
            self.frame_slider = ttk.Scale(
                slider_frame,
                from_=self.primer_frame,
                to=self.ultimo_frame,
                orient=tk.HORIZONTAL
            )
            self.frame_slider.set(self.primer_frame)
            self.frame_slider.pack(fill=tk.X, expand=True)
            
            # Configurar eventos del slider
            self.frame_slider.bind("<Button-1>", self.on_slider_click)
            self.frame_slider.bind("<B1-Motion>", self.on_slider_drag)
            self.frame_slider.bind("<ButtonRelease-1>", self.on_slider_release)
            
            self.update_frame()
        elif self.imagenes:
            # Carpeta de imagenes: la misma barra que con video, pero cada posicion es una imagen
            self.primer_frame, self.ultimo_frame = 0, len(self.imagenes) - 1
            controls = ttk.Frame(main_frame)
            controls.pack(fill=tk.X, pady=(10, 0))
            ttk.Button(controls, text="< Anterior", command=lambda: self.mover_imagen(-1)).pack(side=tk.LEFT)
            ttk.Button(controls, text="Siguiente >", command=lambda: self.mover_imagen(1)).pack(side=tk.LEFT, padx=(2, 0))
            self.frame_label = ttk.Label(controls, text="")
            self.frame_label.pack(side=tk.RIGHT, padx=10)
            slider_frame = ttk.Frame(controls)
            slider_frame.pack(fill=tk.X, padx=10, expand=True)
            self.frame_slider = ttk.Scale(slider_frame, from_=0, to=max(1, self.ultimo_frame), orient=tk.HORIZONTAL)
            self.frame_slider.set(0)
            self.frame_slider.pack(fill=tk.X, expand=True)
            if self.ultimo_frame == 0:
                self.frame_slider.state(["disabled"])
            self.frame_slider.bind("<Button-1>", self.on_slider_click)
            self.frame_slider.bind("<B1-Motion>", self.on_slider_drag)
            self.frame_slider.bind("<ButtonRelease-1>", self.on_slider_release)
            self.bind("<Left>", lambda e: self.mover_imagen(-1))
            self.bind("<Right>", lambda e: self.mover_imagen(1))
            self.focus_set()  # las flechas del teclado funcionan sin tener que hacer clic antes
            self.update_frame()
        elif image is not None:
            self.show_image(image)

    def mover_imagen(self, paso):
        indice = min(self.ultimo_frame, max(0, int(round(self.frame_slider.get())) + paso))
        self.frame_slider.set(indice)
        self.update_frame()
            
    def on_resize(self, event):
        # Reajustar una sola vez cuando la ventana deja de cambiar de tamaño
        if self.original is None:
            return
        if self._ajuste_pendiente is not None:
            self.after_cancel(self._ajuste_pendiente)
        self._ajuste_pendiente = self.after(60, self.redibujar)

    def show_preview(self, original, resultado=None, mascara=None):
        '''Guarda el frame original, su resultado (None = no se pudo procesar) y la mascara, y los muestra'''
        if original is None:
            return
        self.original = original
        self.resultado = resultado
        self.mascara = mascara
        self._escaladas = None
        self.redibujar()

    def cambiar_vista(self):
        self.actualizar_ayuda()
        self.preview_label.configure(cursor="sb_h_double_arrow" if self.vista_var.get() == "Deslizar" else "")
        self.redibujar()

    def actualizar_ayuda(self):
        textos = []
        if self.vista_var.get() == "Deslizar":
            textos.append("Haz clic o arrastra sobre la imagen para mover la línea")
        if self.ver_mascara_var.get():
            textos.append("Verde = zona cambiada (más intenso, más cambio)")
        self.ayuda_label.configure(text="   |   ".join(textos))

    def cambiar_mascara(self):
        self.actualizar_ayuda()
        self._escaladas = None
        # La mascara se calcula al procesar el frame: si todavia no esta, volver a procesarlo pidiendola
        if self.ver_mascara_var.get() and self.mascara is None and self.original is not None:
            self.show_image(self.original)
        else:
            self.redibujar()

    def _con_mascara(self, img):
        '''El resultado con la zona cambiada pintada encima y su contorno (donde el cambio llega a la mitad)'''
        if not self.ver_mascara_var.get() or self.mascara is None or self.mascara.shape != img.shape[:2]:
            return img
        alfa = (self.mascara * self.OPACIDAD_MASCARA)[..., None]
        pintada = img.astype(np.float32) * (1 - alfa) + np.array(self.COLOR_MASCARA, np.float32) * alfa
        pintada = np.rint(pintada).astype(np.uint8)  # rint: truncar oscurece 1 nivel donde casi no hay mascara
        contornos, _ = cv2.findContours((self.mascara >= 0.5).astype(np.uint8), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        grosor = max(1, round(max(img.shape[:2]) / 500))  # que se siga viendo al achicar la imagen
        cv2.drawContours(pintada, contornos, -1, self.COLOR_MASCARA, grosor, cv2.LINE_AA)
        return pintada

    def mover_corte(self, event):
        if self.vista_var.get() != "Deslizar" or self._ancho_mostrado <= 0:
            return
        self.corte = min(1.0, max(0.0, event.x / self._ancho_mostrado))
        self.redibujar()

    def _tamano_recuadro(self):
        ancho = self.preview_frame.winfo_width() - 40  # 20px padding each side
        alto = self.preview_frame.winfo_height() - 40
        if ancho <= 0 or alto <= 0:  # Si aún no está renderizado
            ancho, alto = 1300, 650
        return ancho, alto

    @staticmethod
    def _ajustar(img, ancho, alto):
        # La imagen completa lo mas grande posible dentro de ancho x alto, sin recortar ni deformar
        h, w = img.shape[:2]
        escala = min(ancho / w, alto / h)
        nuevo = (max(1, int(w * escala)), max(1, int(h * escala)))
        return cv2.resize(img, nuevo, interpolation=cv2.INTER_AREA if escala < 1 else cv2.INTER_LANCZOS4)

    @staticmethod
    def _rotular(img, texto, x, y, derecha=False):
        # Etiqueta legible sobre cualquier imagen: texto blanco sobre un recuadro oscuro
        (tw, th), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        if derecha:
            x -= tw + 12
        cv2.rectangle(img, (x, y), (x + tw + 12, y + th + base + 10), (30, 30, 30), -1)
        cv2.putText(img, texto, (x + 6, y + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    def redibujar(self):
        self._ajuste_pendiente = None
        if self.original is None:
            return
        visible = lambda: self._con_mascara(self.resultado) if self.resultado is not None else self.original
        nombre_resultado = "Resultado" if self.resultado is not None else "Resultado (error, ver consola)"
        vista = self.vista_var.get()
        ancho, alto = self._tamano_recuadro()
        self._ancho_mostrado = 0

        if vista == "Lado a lado":
            mitad = max(1, (ancho - self.SEPARACION) // 2)
            a = self._ajustar(self.original, mitad, alto)
            b = self._ajustar(visible(), mitad, alto)
            h = max(a.shape[0], b.shape[0])
            img = np.full((h, a.shape[1] + self.SEPARACION + b.shape[1], 3), self.FONDO, dtype=np.uint8)
            ya, yb, xb = (h - a.shape[0]) // 2, (h - b.shape[0]) // 2, a.shape[1] + self.SEPARACION
            img[ya:ya + a.shape[0], :a.shape[1]] = a
            img[yb:yb + b.shape[0], xb:] = b
            self._rotular(img, "Original", 8, ya + 8)
            self._rotular(img, nombre_resultado, xb + 8, yb + 8)
        else:
            # Las dos vistas usan la imagen a tamaño completo; se guardan escaladas para que arrastrar
            # la linea no vuelva a redimensionar en cada movimiento
            if self._escaladas is None or self._escaladas[:2] != (ancho, alto):
                self._escaladas = (ancho, alto, self._ajustar(self.original, ancho, alto),
                                   self._ajustar(visible(), ancho, alto))
            a, b = self._escaladas[2:]
            if vista == "Solo resultado":
                img = b
            else:
                # Izquierda de la linea: original; derecha: resultado
                img = b.copy()
                x = int(img.shape[1] * self.corte)
                img[:, :x] = a[:, :x]
                cv2.line(img, (x, 0), (x, img.shape[0]), (255, 255, 255), 2)
                self._rotular(img, "Original", 8, 8)
                self._rotular(img, nombre_resultado, img.shape[1] - 8, 8, derecha=True)
                self._ancho_mostrado = img.shape[1]

        self._mostrada = img
        photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo

    def abrir_menu_copiar(self, event):
        hay_resultado = "normal" if self.resultado is not None else "disabled"
        hay_imagen = "normal" if self.original is not None else "disabled"
        self.menu_copiar.entryconfigure(0, state=hay_resultado)
        self.menu_copiar.entryconfigure(1, state=hay_imagen)
        self.menu_copiar.entryconfigure(2, state=hay_imagen)
        try:
            self.menu_copiar.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu_copiar.grab_release()

    def copiar(self, cual):
        '''
         Copia al portapapeles: "resultado" y "original" a tamaño completo (el resultado sin la mascara
        pintada), "vista" tal como se ve en la ventana (con rotulos, linea y mascara si estan)
        '''
        img = {"resultado": self.resultado, "original": self.original, "vista": self._mostrada}[cual]
        if img is None:
            self.avisar("Todavía no hay resultado para copiar")
            return
        try:
            copiar_al_portapapeles(img)
        except Exception as e:
            print(f"Error al copiar: {e}")
            self.avisar(f"No se pudo copiar: {e}")
            return
        nombres = {"resultado": "Resultado copiado", "original": "Original copiado", "vista": "Vista copiada"}
        self.avisar(f"{nombres[cual]} ({img.shape[1]}x{img.shape[0]}), pégalo con Ctrl+V")

    def avisar(self, texto):
        # Mensaje corto en la barra; se borra solo, salvo que mientras tanto aparezca "Procesando..." u otro
        self.estado_label.configure(text=texto)
        if self._aviso_pendiente is not None:
            self.after_cancel(self._aviso_pendiente)
        self._aviso_pendiente = self.after(
            3000, lambda: self.estado_label.cget("text") == texto and self.estado_label.configure(text=""))

    def on_slider_click(self, event):
        # Calcular la posición relativa del clic
        slider = event.widget
        width = slider.winfo_width()
        x = event.x
        value = self.primer_frame + (x / width) * (self.ultimo_frame - self.primer_frame)
        self.frame_slider.set(value)
        self.update_frame()

    def on_slider_drag(self, event):
        self.update_frame()

    def on_slider_release(self, event):
        self.update_frame()

    def update_frame(self, *args):
        if self.imagenes:
            indice = min(self.ultimo_frame, max(0, int(round(self.frame_slider.get()))))
            if indice == self._indice_mostrado:  # arrastrar dentro de la misma imagen no la vuelve a procesar
                return
            ruta = self.imagenes[indice]
            imagen = leer(ruta)
            self.frame_label.config(text=f"{indice + 1}/{len(self.imagenes)}  ·  {os.path.basename(ruta)}"
                                    + ("  (no se pudo leer)" if imagen is None else ""))
            self._indice_mostrado = indice
            if imagen is not None:
                self.show_image(imagen)
            return
        if hasattr(self, 'cap'):
            frame_no = int(self.frame_slider.get())
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = self.cap.read()
            if ret:
                self.frame_label.config(text=self.texto_posicion(frame_no))
                self.show_image(frame)

    def texto_posicion(self, frame_no):
        """
        Contador bajo la barra: con fragmento, cuenta dentro del fragmento (y dice que frame es del video);
        sin fragmento, dentro del video. Con el tiempo, para ver cuanto dura
        """
        cantidad = self.ultimo_frame - self.primer_frame + 1
        relativo = frame_no - self.primer_frame
        texto = (f"{relativo}/{cantidad - 1}  ·  {formato_tiempo(relativo / self.fps)} de "
                 f"{formato_tiempo(cantidad / self.fps)}")
        if self.es_fragmento:
            texto = f"Fragmento: {texto}  (frame {frame_no} del video)"
        return texto

    def show_image(self, img):
        '''
         Procesa el frame en segundo plano (la ventana no se congela) y lo muestra al terminar. Las opciones
        se leen aqui, en el hilo de la interfaz; el hilo solo cambia las caras.
        '''
        if img is None or not getattr(self.app, 'source_path', None):
            return
        self._pendiente = img
        if not self._ocupado:
            self._lanzar()

    def _lanzar(self):
        img, self._pendiente = self._pendiente, None
        try:
            # Caras fuente (de la foto o del modelo .npy), preparadas una sola vez
            caras_src = self.app.obtener_caras_src()
            opciones = self.app.opciones_cambio()
            identidad = self.app.asignacion_var.get() == "identidad"
        except Exception as e:
            print(f"Error en preview: {str(e)}")
            self.estado_label.configure(text=f"Error: {e}")
            self.show_preview(img)
            return

        self._ocupado = True
        crear_modo = identidad and getattr(self, "modo_identidad", None) is None
        self.estado_label.configure(text="Analizando personas del video..." if crear_modo and self.video_path
                                    else "Procesando...")
        threading.Thread(target=self._procesar,
                         args=(img, caras_src, opciones, identidad, crear_modo, self.ver_mascara_var.get()),
                         daemon=True).start()

    def _procesar(self, img, caras_src, opciones, identidad, crear_modo, con_mascara):
        resultado, mascara, error = None, None, None
        try:
            swapper = self.app.swapper
            if crear_modo:
                # Con un video, las personas salen del analisis previo (las mismas que al procesar; la primera
                # vez tarda, despues se reutiliza). Con una imagen se registran en el orden en que se ven
                personas = self.app.analizar_personas()[0] if self.video_path else None
                self.modo_identidad = self.app.crear_modo_identidad(caras_src, personas)
            modo = self.modo_identidad if identidad else None
            if modo is not None:
                del opciones["index_src"], opciones["index_dst"]
                caras_dst = swapper.detectar_caras(img, opciones["genero"])
                if not modo["registro"].completo():
                    modo["registro"].registrar(caras_dst)
                resultado = swapper.cambiar_Faces_por_identidad(
                    frame_dst=img.copy(), caras_src=modo["caras_src"], mapa=modo["mapa"],
                    refs=modo["registro"].copia_refs(), caras_dst=caras_dst, devolver_mascara=con_mascara, **opciones)
            else:
                resultado = swapper.cambiar_Faces(frame_dst=img.copy(), caras_src=caras_src,
                                                  devolver_mascara=con_mascara, **opciones)
            if con_mascara:
                resultado, mascara = resultado
        except Exception as e:
            print(f"Error en preview: {str(e)}")
            error = str(e)
        try:
            self.after(0, self._terminado, img, resultado, mascara, error)
        except Exception:
            pass  # la ventana ya se cerro

    def _terminado(self, img, resultado, mascara, error):
        self._ocupado = False
        if not self.winfo_exists():
            return
        # Si hay un pedido mas nuevo (otro frame u otra configuracion), este resultado ya no sirve
        if self._pendiente is not None:
            self._lanzar()
            return
        self.estado_label.configure(text=f"Error: {error}" if error else "")
        self.show_preview(img, resultado, mascara)

    def config_cambiada(self):
        '''Se cambio algo en la configuracion: volver a procesar el frame que se esta viendo'''
        # Esperar a que se deje de mover el deslizador o de escribir, para no procesar cada paso
        if self._config_pendiente is not None:
            self.after_cancel(self._config_pendiente)
        self._config_pendiente = self.after(400, self._aplicar_config)

    def _aplicar_config(self):
        self._config_pendiente = None
        # Los indices o la asignacion pueden haber cambiado: el modo identidad se vuelve a armar
        self.modo_identidad = None
        if self.original is not None:
            self.show_image(self.original)
        elif hasattr(self, "cap"):
            self.update_frame()

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

def formato_tiempo(segundos):
    """1:05.3 (o 1:02:05.3 si pasa de una hora)"""
    segundos = max(0.0, segundos)
    horas, resto = divmod(segundos, 3600)
    minutos, seg = divmod(resto, 60)
    if horas >= 1:
        return f"{int(horas)}:{int(minutos):02d}:{seg:04.1f}"
    return f"{int(minutos)}:{seg:04.1f}"


class VentanaFragmento(tk.Toplevel):
    """
     Elige el pedazo del video que se va a procesar: un deslizador para el inicio y otro para el fin, y un
    reproductor para ver ese pedazo y cuanto dura. Solo muestra el video original (no cambia caras).
    al_aceptar((inicio, fin)) recibe frames [inicio, fin), o None para el video completo.
    """

    TAMANO_VISTA = (860, 440)  # recuadro donde se ve el video (se ajusta sin recortar ni deformar)

    def __init__(self, parent, ruta, fragmento, al_aceptar):
        super().__init__(parent)
        self.title("Fragmento del video")
        self.geometry("920x720")
        self.transient(parent)
        self.al_aceptar = al_aceptar

        info = info_video(ruta)
        self.cap = cv2.VideoCapture(ruta)
        self.fps = info["fps"]
        self.total = max(1, info["frames"])
        inicio, fin = fragmento if fragmento else (0, self.total)
        self.pos = inicio  # frame que se esta mostrando
        self._reproduciendo = None
        self._mostrar_pendiente = None

        principal = ttk.Frame(self)
        principal.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        ttk.Label(principal, text=f"{os.path.basename(ruta)}  ·  dura {formato_tiempo(self.total / self.fps)} "
                                  f"({self.total} frames a {self.fps:.2f} fps)", foreground="gray").pack(anchor="w")
        self.vista = ttk.Label(principal, background='#f0f0f0', anchor="center")
        self.vista.pack(fill=tk.BOTH, expand=True, pady=10)

        # Deslizadores de inicio y fin
        self.inicio_var = tk.DoubleVar(value=inicio)
        self.fin_var = tk.DoubleVar(value=fin)
        deslizadores = ttk.Frame(principal)
        deslizadores.pack(fill=tk.X)
        deslizadores.columnconfigure(1, weight=1)
        self.inicio_label = ttk.Label(deslizadores, width=10)
        self.fin_label = ttk.Label(deslizadores, width=10)
        for fila, (nombre, var, desde, hasta, etiqueta) in enumerate((
                ("Inicio:", self.inicio_var, 0, self.total - 1, self.inicio_label),
                ("Fin:", self.fin_var, 1, self.total, self.fin_label))):
            ttk.Label(deslizadores, text=nombre, font=('Helvetica', 10, 'bold')).grid(row=fila, column=0, sticky="w", padx=(0, 10), pady=4)
            ttk.Scale(deslizadores, from_=desde, to=hasta, variable=var,
                      command=lambda v, cual=nombre: self.mover(cual)).grid(row=fila, column=1, sticky="ew", pady=4)
            etiqueta.grid(row=fila, column=2, sticky="e", padx=(10, 0))

        # Reproductor y duracion
        reproductor = ttk.Frame(principal)
        reproductor.pack(fill=tk.X, pady=(10, 0))
        self.boton_reproducir = ttk.Button(reproductor, text="▶ Reproducir fragmento", command=self.reproducir)
        self.boton_reproducir.pack(side=tk.LEFT)
        self.duracion_label = ttk.Label(reproductor, font=('Helvetica', 10, 'bold'))
        self.duracion_label.pack(side=tk.LEFT, padx=15)
        self.posicion_label = ttk.Label(reproductor, foreground="gray")
        self.posicion_label.pack(side=tk.LEFT)

        botones = ttk.Frame(principal)
        botones.pack(pady=(15, 0))
        ttk.Button(botones, text="Aceptar", command=self.aceptar).pack(side=tk.LEFT, padx=5)
        ttk.Button(botones, text="Video completo", command=self.video_completo).pack(side=tk.LEFT, padx=5)
        ttk.Button(botones, text="Cancelar", command=self.cerrar).pack(side=tk.LEFT, padx=5)

        self.protocol("WM_DELETE_WINDOW", self.cerrar)
        self.actualizar_textos()
        self.after(100, lambda: self.mostrar(self.pos))

    def rango(self):
        return int(round(self.inicio_var.get())), int(round(self.fin_var.get()))

    def mover(self, cual):
        """Se movio un deslizador: el inicio siempre antes que el fin, y se muestra ese frame"""
        self.pausar()
        inicio, fin = self.rango()
        if cual == "Inicio:":
            if inicio >= fin:
                self.fin_var.set(min(self.total, inicio + 1))
            self.pos = min(inicio, self.total - 1)
        else:
            if fin <= inicio:
                self.inicio_var.set(max(0, fin - 1))
            self.pos = max(0, fin - 1)  # el fin no se incluye: se muestra el ultimo frame del fragmento
        self.actualizar_textos()
        # Buscar un frame cuesta: esperar a que el deslizador se quede quieto un momento
        if self._mostrar_pendiente is not None:
            self.after_cancel(self._mostrar_pendiente)
        self._mostrar_pendiente = self.after(40, lambda: self.mostrar(self.pos))

    def actualizar_textos(self):
        inicio, fin = self.rango()
        self.inicio_label.configure(text=formato_tiempo(inicio / self.fps))
        self.fin_label.configure(text=formato_tiempo(fin / self.fps))
        self.duracion_label.configure(text=f"Duración del fragmento: {formato_tiempo((fin - inicio) / self.fps)} "
                                           f"({fin - inicio} frames)")
        self.posicion_label.configure(text=f"Viendo {formato_tiempo(self.pos / self.fps)}")

    def mostrar(self, numero, buscar=True):
        """Muestra el frame "numero" (buscar=False: el siguiente de la captura, al reproducir)"""
        self._mostrar_pendiente = None
        if buscar:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, numero)
        ok, frame = self.cap.read()
        if not ok:
            return False
        img = PreviewWindow._ajustar(frame, *self.TAMANO_VISTA)
        foto = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self.vista.configure(image=foto)
        self.vista.image = foto
        return True

    def reproducir(self):
        if self._reproduciendo is not None:
            self.pausar()
            return
        inicio, fin = self.rango()
        if not inicio <= self.pos < fin - 1:
            self.pos = inicio
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.pos)
        self.boton_reproducir.configure(text="⏸ Pausar")
        self._siguiente(time.time())

    def _siguiente(self, momento):
        inicio, fin = self.rango()
        if self.pos >= fin or not self.mostrar(self.pos, buscar=False):
            self.pausar()
            self.pos = inicio  # al terminar, la proxima vez empieza otra vez desde el inicio
            return
        self.posicion_label.configure(text=f"Viendo {formato_tiempo(self.pos / self.fps)}  "
                                           f"({formato_tiempo((self.pos - inicio) / self.fps)} de "
                                           f"{formato_tiempo((fin - inicio) / self.fps)})")
        self.pos += 1
        # A la velocidad real del video: descontar lo que tardo en leer y mostrar el frame
        momento += 1.0 / self.fps
        espera = max(1, int((momento - time.time()) * 1000))
        self._reproduciendo = self.after(espera, self._siguiente, momento)

    def pausar(self):
        if self._reproduciendo is not None:
            self.after_cancel(self._reproduciendo)
            self._reproduciendo = None
        self.boton_reproducir.configure(text="▶ Reproducir fragmento")

    def aceptar(self):
        inicio, fin = self.rango()
        self.cerrar()
        self.al_aceptar(None if inicio == 0 and fin >= self.total else (inicio, fin))

    def video_completo(self):
        self.cerrar()
        self.al_aceptar(None)

    def cerrar(self):
        self.pausar()
        self.cap.release()
        self.destroy()


class FaceSwapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Transfer-IA Studio")
        self.root.geometry("1600x900")
        self.root.minsize(1300, 760)
        try:
            self.root.state("zoomed")  # Windows: abrir maximizada
        except tk.TclError:
            pass
        
        # Configure styles
        self.setup_styles()
        
        # Initialize variables
        self.source_path = None
        self.target_path = None
        self.output_dir = "output"
        self.current_frame = None
        self.processing = False
        self.models_dir = "models"
        self.preview_window = None
        self.opciones_proceso = None  # opciones fijadas mientras se procesa (ver start_processing)
        self.fragmento_proceso = None  # fragmento fijado mientras se procesa
        
        # Procesamiento de video (core/Proceso_Video.py)
        self.detener_video = False   # True = el boton Detener pidio parar (lo hecho queda guardado)
        self.reanudar_video = False  # True = continuar el proceso sin terminar de esta salida
        self._caras_src_cache = (None, None)  # (ruta fuente, caras) para no detectar la fuente en cada frame
        self.modelos_src = []  # modelos .npy usados como fuentes; su orden es el índice fuente
        
        # Create directories if they don't exist
        for dir_path in [self.output_dir, self.models_dir, "temp_frames"]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Check for required models
        self.check_required_models()
        
        # Initialize models
        self.setup_models()
        
        # Create main frames
        if self.create_frames() is not False:
            self.vigilar_configuracion()

    def setup_styles(self):
        style = ttk.Style()
        
        # Configure main styles
        style.configure('TFrame', background="#ebf3e2")
        style.configure('TLabelframe', background="#ebf3e2")
        style.configure('TLabelframe.Label', font=('Helvetica', 10, 'bold'))
        style.configure('TButton', padding=10, font=('Helvetica', 9))
        style.configure('Accent.TButton', background="#DFEADE")
        style.configure('TLabel', background="#ebf3e2", font=('Helvetica', 9))
        style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        style.configure('TProgressbar', thickness=8)
        
        # Configure colors
        self.root.configure(bg="#B8EC9C")

    def check_required_models(self):
        required_models = {
            "xseg.onnx": "XSeg model for occlusion",
            "inswapper128.onnx": "InsightFace model for face swapping"
        }
        
        missing_models = []
        for model, description in required_models.items():
            if not os.path.exists(os.path.join(self.models_dir, model)):
                missing_models.append(f"{model} ({description})")
        
        if missing_models:
            messagebox.showerror("Missing Models", 
                               "The following required models are missing:\n\n" + 
                               "\n".join(missing_models) + 
                               "\n\nPlease place them in the 'models' directory.")
            sys.exit(1)

    def setup_models(self):
        try:
          
            # Initialize Swapper
            self.swapper = Swapper_Faces(os.path.join(self.models_dir, "inswapper128.onnx"))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
            raise

    def create_frames(self):

        if shutil.which("ffmpeg") is None:
        # Crear ventana oculta solo para mostrar el mensaje
            root = tk.Tk()
            root.withdraw()  # Oculta la ventana principal
            messagebox.showerror("FFmpeg no encontrado", 
                                 "FFmpeg no está instalado o no está en el PATH.\n\nDescárgalo desde:\nhttps://ffmpeg.org/download.html y vuevla a ejecutar el programa")
            return False

        # Dos columnas: a la izquierda fuente (arriba) y destino (abajo); a la derecha el titulo y toda la
        # configuracion. Se reparten el ancho 2 a 3 (las opciones mas anchas que las vistas previas)
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        main_container.columnconfigure(0, weight=2, uniform="columnas")
        main_container.columnconfigure(2, weight=3, uniform="columnas")
        main_container.rowconfigure(0, weight=1)

        izquierda = ttk.Frame(main_container)
        izquierda.grid(row=0, column=0, sticky="nsew")
        izquierda.columnconfigure(0, weight=1)
        izquierda.rowconfigure(0, weight=1)
        izquierda.rowconfigure(1, weight=1)

        ttk.Separator(main_container, orient=tk.VERTICAL).grid(row=0, column=1, sticky="ns", padx=10)

        derecha = ttk.Frame(main_container)
        derecha.grid(row=0, column=2, sticky="nsew")

        # Title
        title = ttk.Label(derecha, text="Transfer-IA STUDIO", style='Header.TLabel')
        title.pack(pady=(0, 10))

        # Source frame (izquierda, arriba)
        self.source_frame = DropFrame(izquierda, text="Imagen Fuente o Modelo de Cara", callback=self.set_source,
                                      multi_callback=self.soltar_varios,
                                      filetypes=[("Imagen o modelo de cara", "*.jpg *.jpeg *.png *.npy"),
                                                 ("Modelo de cara", "*.npy"),
                                                 ("Video para crear un modelo", "*.mp4 *.avi *.mkv *.mov *.webm")])
        self.source_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))

        modelos_frame = ttk.Frame(self.source_frame)
        modelos_frame.pack(fill=tk.X, padx=10)
        ttk.Label(modelos_frame, text="Modelos:").pack(side=tk.LEFT)
        self.modelos_label = ttk.Label(modelos_frame, text="(ninguno)")
        self.modelos_label.pack(side=tk.LEFT, padx=5)

        botones_modelos = ttk.Frame(self.source_frame)
        botones_modelos.pack(fill=tk.X, padx=10, pady=(5, 10))
        ModernButton(botones_modelos, text="Elegir modelos...", command=self.elegir_modelos_dialogo).pack(side=tk.LEFT)
        ModernButton(botones_modelos, text="Crear modelo de cara", command=self.crear_modelo_dialogo).pack(side=tk.LEFT, padx=5)
        ModernButton(botones_modelos, text="Desde video", command=self.crear_modelo_video_dialogo).pack(side=tk.LEFT)
        ModernButton(botones_modelos, text="Combinar modelos", command=self.combinar_modelos_dialogo).pack(side=tk.LEFT, padx=5)
        ModernButton(botones_modelos, text="Quitar", command=self.quitar_fuente).pack(side=tk.LEFT)
        
        # Target frame (izquierda, abajo)
        self.target_frame = DropFrame(izquierda, text="Imagen/Video Destino (o carpeta de imágenes)",
                                      callback=self.set_target, multi_callback=self.soltar_destino_varios)
        self.target_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))

        botones_destino = ttk.Frame(self.target_frame)
        botones_destino.pack(fill=tk.X, padx=10, pady=(0, 10))
        ModernButton(botones_destino, text="Carpeta de imágenes...", command=self.elegir_carpeta_destino).pack(side=tk.LEFT)
        self.carpeta_destino_label = ttk.Label(botones_destino, text="")
        self.carpeta_destino_label.pack(side=tk.LEFT, padx=10)

        # Preview button
        preview_frame = ttk.Frame(derecha)
        preview_frame.pack(fill=tk.X, pady=(0, 5))
        ModernButton(preview_frame, text="Abrir Preview", width=45,
                    command=self.show_preview).pack(pady=5)

        # Configuraciones guardadas: todas las opciones de abajo con un nombre, en configuraciones/<nombre>.json
        guardadas_frame = ttk.Frame(derecha)
        guardadas_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(guardadas_frame, text="Configuración guardada:", font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT, padx=(10, 5))
        self.configuracion_var = tk.StringVar()
        self.configuracion_combo = ttk.Combobox(guardadas_frame, textvariable=self.configuracion_var,
                                                state="readonly", width=25)
        self.configuracion_combo.pack(side=tk.LEFT)
        self.configuracion_combo.bind("<<ComboboxSelected>>", lambda e: self.cargar_configuracion())
        ModernButton(guardadas_frame, text="Guardar como...", command=self.guardar_configuracion).pack(side=tk.LEFT, padx=(10, 5))
        ModernButton(guardadas_frame, text="Eliminar", command=self.eliminar_configuracion).pack(side=tk.LEFT)
        self.actualizar_lista_configuraciones()

        # Controls frame: una fila por funcion, con su nombre a la izquierda (columna 0) y sus controles al lado
        controls_frame = ttk.LabelFrame(derecha, text="Configuración")
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        controls_frame.columnconfigure(1, weight=1)
        self._fila_config = 0

        def fila(nombre, pady=(0, 6)):
            """Agrega una fila a la configuracion y devuelve el frame donde van sus controles"""
            primera = self._fila_config == 0
            ttk.Label(controls_frame, text=nombre, font=('Helvetica', 10, 'bold')).grid(
                row=self._fila_config, column=0, sticky="w", padx=(10, 15), pady=(10, 6) if primera else pady)
            controles = ttk.Frame(controls_frame)
            controles.grid(row=self._fila_config, column=1, sticky="w", padx=(0, 10),
                           pady=(10, 6) if primera else pady)
            self._fila_config += 1
            return controles

        # 1. Caras: como se empareja cada cara destino con una fuente y cuales se cambian
        caras_fila = fila("Caras:")
        ttk.Label(caras_fila, text="Asignación:").pack(side=tk.LEFT, padx=(0, 5))
        self.asignacion_var = tk.StringVar(value="posicion")
        ttk.Combobox(caras_fila, textvariable=self.asignacion_var, values=["posicion", "identidad"],
                     state="readonly", width=10).pack(side=tk.LEFT)
        ttk.Label(caras_fila, text="Género destino:").pack(side=tk.LEFT, padx=(20, 5))
        self.genero_var = tk.StringVar(value="todos")
        ttk.Combobox(caras_fila, textvariable=self.genero_var, values=["todos", "mujer", "hombre"],
                     state="readonly", width=8).pack(side=tk.LEFT)
        # Deteccion de caras inclinadas o boca abajo (giros y refinado); sin ella, la deteccion original
        self.caras_inclinadas_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(caras_fila, text="Caras inclinadas", variable=self.caras_inclinadas_var).pack(side=tk.LEFT,
                                                                                                    padx=(15, 0))
        self.caras_inclinadas_var.trace_add("write", lambda *_: setattr(
            Analyzer_Faces, "CARAS_INCLINADAS", self.caras_inclinadas_var.get()))
        # Analisis previo de personas (solo con identidad en video): cuantos frames por segundo y como maximo
        self.analisis_label = ttk.Label(caras_fila, text="Análisis de personas:")
        self.analisis_label.pack(side=tk.LEFT, padx=(20, 5))
        self.personas_fps_var = tk.StringVar(value=f"{PERSONAS_FPS:g}")
        self.personas_fps_spin = ttk.Spinbox(caras_fila, textvariable=self.personas_fps_var, width=4,
                                             values=("0.25", "0.5", "1", "2", "3", "5"))
        self.personas_fps_spin.pack(side=tk.LEFT)
        ttk.Label(caras_fila, text="fps, máx.").pack(side=tk.LEFT, padx=5)
        self.personas_max_var = tk.StringVar(value=str(PERSONAS_MAX))
        self.personas_max_spin = ttk.Spinbox(caras_fila, textvariable=self.personas_max_var, width=5,
                                             from_=PERSONAS_MAX_RANGO[0], to=PERSONAS_MAX_RANGO[1], increment=50)
        self.personas_max_spin.pack(side=tk.LEFT)
        ttk.Label(caras_fila, text="frames").pack(side=tk.LEFT, padx=(5, 0))
        self.asignacion_var.trace_add("write", lambda *_: self._estado_analisis())
        self._estado_analisis()

        # 2. Indices: que cara fuente va en que cara destino
        indices_fila = fila("Índices:", pady=(0, 0))
        ttk.Label(indices_fila, text="Fuente:").pack(side=tk.LEFT, padx=(0, 5))
        self.src_indices_var = tk.StringVar(value="0")
        ttk.Entry(indices_fila, textvariable=self.src_indices_var, width=10).pack(side=tk.LEFT)
        ttk.Label(indices_fila, text="Destino:").pack(side=tk.LEFT, padx=(20, 5))
        self.dst_indices_var = tk.StringVar(value="0")
        ttk.Entry(indices_fila, textvariable=self.dst_indices_var, width=10).pack(side=tk.LEFT)
        # Modo identidad con video: muestra quien es cada numero (analisis previo, ver core/Personas.py)
        ModernButton(indices_fila, text="Ver personas...", command=self.ver_personas).pack(side=tk.LEFT, padx=(15, 0))
        ttk.Label(controls_frame, wraplength=600, justify=tk.LEFT, foreground="gray",
                  text="(identidad en video: Índices Destino = número de persona, 0 = la que más aparece; "
                       "«Ver personas» muestra quién es cada una)"
                  ).grid(row=self._fila_config, column=1, sticky="w", padx=(0, 10), pady=(2, 6))
        self._fila_config += 1

        # 3. Pegado: cuanta zona de la cara se cambia y como se iguala su color
        pegado_fila = fila("Pegado:")
        ttk.Label(pegado_fila, text="Máscara:").pack(side=tk.LEFT, padx=(0, 5))
        self.mascara_var = tk.StringVar(value="amplia")
        ttk.Combobox(pegado_fila, textvariable=self.mascara_var, values=["normal", "amplia", "maxima"],
                     state="readonly", width=8).pack(side=tk.LEFT)
        ttk.Label(pegado_fila, text="Corrección de color:").pack(side=tk.LEFT, padx=(20, 5))
        self.color_var = tk.StringVar(value="ninguna")
        ttk.Combobox(pegado_fila, textvariable=self.color_var, values=["ninguna", "suave", "completa"],
                     state="readonly", width=9).pack(side=tk.LEFT)

        # 4. Del destino: partes y maquillaje que se quedan como en el destino
        destino_fila = fila("Del destino:")
        self.var_ojos = tk.BooleanVar(value=False)
        ttk.Checkbutton(destino_fila, text="Ojos", variable=self.var_ojos).pack(side=tk.LEFT)
        self.var_boca = tk.BooleanVar(value=False)
        ttk.Checkbutton(destino_fila, text="Boca", variable=self.var_boca).pack(side=tk.LEFT, padx=(8, 0))
        self.var_nariz = tk.BooleanVar(value=False)
        ttk.Checkbutton(destino_fila, text="Nariz", variable=self.var_nariz).pack(side=tk.LEFT, padx=(8, 0))
        # La frente del destino: quita el cerquillo que inswapper dibuja cuando la fuente lo tiene
        self.var_frente = tk.BooleanVar(value=False)
        ttk.Checkbutton(destino_fila, text="Frente (quitar cerquillo)",
                        variable=self.var_frente).pack(side=tk.LEFT, padx=(8, 0))
        self.var_mejillas = tk.BooleanVar(value=False)
        ttk.Checkbutton(destino_fila, text="Mejillas y pómulos",
                        variable=self.var_mejillas).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(destino_fila, text="Maquillaje:").pack(side=tk.LEFT, padx=(20, 5))
        self.maquillaje_var = tk.DoubleVar(value=0.0)
        maquillaje_valor = ttk.Label(destino_fila, text="0.00", width=4)
        ttk.Scale(destino_fila, from_=0.0, to=1.0, variable=self.maquillaje_var, length=120).pack(side=tk.LEFT)
        # Con trace (y no con command) el numero tambien cambia al cargar una configuracion guardada
        self.maquillaje_var.trace_add("write", lambda *_: maquillaje_valor.configure(
            text=f"{self.maquillaje_var.get():.2f}"))
        maquillaje_valor.pack(side=tk.LEFT, padx=(5, 0))

        # 5. Oclusion: pelo, manos y objetos delante de la cara vuelven del destino
        oclusion_fila = fila("Oclusión:")
        self.var_oclucion = tk.BooleanVar(value=False)
        self.check_oclucion = ttk.Checkbutton(oclusion_fila, text="Activar (pelo, manos y objetos delante de la cara)",
                                              variable=self.var_oclucion)
        self.check_oclucion.pack(side=tk.LEFT)
        # Modo: cuanta cara puede quedar sobre lo que la tapa en el borde difuminado (ver xseg.MODOS_OCLUSION)
        ttk.Label(oclusion_fila, text="Modo:").pack(side=tk.LEFT, padx=(20, 5))
        self.oclusion_modo_var = tk.StringVar(value="adaptable")
        ttk.Combobox(oclusion_fila, textvariable=self.oclusion_modo_var, values=["debil", "adaptable", "fuerte"],
                     state="readonly", width=9).pack(side=tk.LEFT)

        # 6. Nitidez de la cara pegada: funciona con o sin oclusion
        nitidez_fila = fila("Nitidez:")
        ttk.Label(nitidez_fila, text="Filtro:").pack(side=tk.LEFT, padx=(0, 5))
        self.combofiltro = ttk.Combobox(nitidez_fila, values=["None", "natural", "strong", "adaptativo"],
                                        state="readonly", width=10)
        self.combofiltro.set("None")
        self.combofiltro.pack(side=tk.LEFT)
        ttk.Label(nitidez_fila, text="(adaptativo = iguala el detalle y el grano del destino)",
                  foreground="gray").pack(side=tk.LEFT, padx=(10, 0))

        # 7. Mejora: GPEN y su fuerza
        mejora_fila = fila("Mejora:", pady=(0, 10))
        self.var_mejorar = tk.BooleanVar(value=False)
        ttk.Checkbutton(mejora_fila, text="Mejorar cara (GPEN)", variable=self.var_mejorar).pack(side=tk.LEFT)
        ttk.Label(mejora_fila, text="Fuerza:").pack(side=tk.LEFT, padx=(20, 5))
        self.fuerza_gpen_var = tk.DoubleVar(value=0.8)
        fuerza_gpen_valor = ttk.Label(mejora_fila, text="0.80", width=4)
        ttk.Scale(mejora_fila, from_=0.1, to=1.0, variable=self.fuerza_gpen_var, length=120).pack(side=tk.LEFT)
        self.fuerza_gpen_var.trace_add("write", lambda *_: fuerza_gpen_valor.configure(
            text=f"{self.fuerza_gpen_var.get():.2f}"))
        fuerza_gpen_valor.pack(side=tk.LEFT, padx=(5, 0))
        # Caras mas chicas que esto no pasan por GPEN: en pixeles o en % del lado mas corto de la imagen
        ttk.Label(mejora_fila, text="Cara mínima:").pack(side=tk.LEFT, padx=(20, 5))
        self.gpen_minimo_var = tk.StringVar(value=MINIMO_POR_DEFECTO)
        ttk.Spinbox(mejora_fila, from_=0, to=4000, increment=1, textvariable=self.gpen_minimo_var,
                    width=5).pack(side=tk.LEFT)
        self.gpen_unidad_var = tk.StringVar(value="px")
        unidad = ttk.Combobox(mejora_fila, textvariable=self.gpen_unidad_var, values=["px", "% imagen"],
                              state="readonly", width=8)
        unidad.pack(side=tk.LEFT, padx=(5, 0))
        # Al cambiar de unidad, poner un valor que tenga sentido en la nueva (96 px o 25 %)
        unidad.bind("<<ComboboxSelected>>", lambda e: self.gpen_minimo_var.set(
            "25" if self.gpen_unidad_var.get() == "% imagen" else MINIMO_POR_DEFECTO))

        # Output configuration frame
        output_frame = ttk.LabelFrame(derecha, text="Configuración de Salida")
        output_frame.pack(fill=tk.X, pady=10)
        
        # Output directory
        dir_frame = ttk.Frame(output_frame)
        dir_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(dir_frame, text="Carpeta de Salida:").pack(side=tk.LEFT)
        self.output_path_var = tk.StringVar(value=self.output_dir)
        ttk.Entry(dir_frame, textvariable=self.output_path_var, width=50).pack(side=tk.LEFT, padx=5)
        ModernButton(dir_frame, text="Seleccionar", command=self.select_output_dir).pack(side=tk.LEFT)
        
        # Output filename
        file_frame = ttk.Frame(output_frame)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(file_frame, text="Nombre del Archivo:").pack(side=tk.LEFT)
        self.output_filename_var = tk.StringVar(value="resultado")
        ttk.Entry(file_frame, textvariable=self.output_filename_var, width=50).pack(side=tk.LEFT, padx=5)

        # ZIP con contraseña (AES-256): el resultado no queda suelto en la carpeta. La contraseña no se guarda
        zip_frame = ttk.Frame(output_frame)
        zip_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        self.zip_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(zip_frame, text="Guardar en ZIP con contraseña", variable=self.zip_var,
                        command=self._estado_zip).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(zip_frame, text="Contraseña:").pack(side=tk.LEFT)
        self.zip_contrasena_var = tk.StringVar()
        self.zip_contrasena_entry = ttk.Entry(zip_frame, textvariable=self.zip_contrasena_var, show="•", width=18)
        self.zip_contrasena_entry.pack(side=tk.LEFT, padx=5)
        self.zip_conservar_var = tk.BooleanVar(value=False)
        self.zip_conservar_check = ttk.Checkbutton(zip_frame, text="Conservar también el archivo suelto",
                                                   variable=self.zip_conservar_var)
        self.zip_conservar_check.pack(side=tk.LEFT, padx=(10, 0))
        self._estado_zip()
        
        # Configuración de procesamiento
        process_frame = ttk.LabelFrame(derecha, text="Configuración de Procesamiento")
        process_frame.pack(fill=tk.X, pady=10)
        
        # Control de hilos
        thread_frame = ttk.Frame(process_frame)
        thread_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(thread_frame, text="Número de Hilos:").pack(side=tk.LEFT, padx=5)
        self.thread_var = tk.StringVar(value="2")
        thread_spinbox = ttk.Spinbox(thread_frame, from_=1, to=4, 
                                   textvariable=self.thread_var, width=5)
        thread_spinbox.pack(side=tk.LEFT, padx=5)

        # Video: formato del temporal (aguanta cortes y permite continuar) y codificador, en su propia fila
        video_frame = ttk.Frame(process_frame)
        video_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        ttk.Label(video_frame, text="Video temporal:").pack(side=tk.LEFT, padx=5)
        self.contenedor_var = tk.StringVar(value="mkv")
        ttk.Combobox(video_frame, textvariable=self.contenedor_var, values=list(CONTENEDORES),
                     state="readonly", width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(video_frame, text="Códec:").pack(side=tk.LEFT, padx=(20, 5))
        self.codec_var = tk.StringVar(value="libx264")
        ttk.Combobox(video_frame, textvariable=self.codec_var, values=list(CODECS),
                     state="readonly", width=11).pack(side=tk.LEFT)

        # Fragmento: procesar solo un pedazo del video (inicio y fin elegidos en su ventana)
        fragmento_frame = ttk.Frame(process_frame)
        fragmento_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        self.fragmento = None  # (inicio, fin) en frames del video destino, o None
        self.fragmento_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(fragmento_frame, text="Solo un fragmento", variable=self.fragmento_var,
                        command=self.cambiar_fragmento_check).pack(side=tk.LEFT, padx=5)
        ModernButton(fragmento_frame, text="Elegir fragmento...", command=self.elegir_fragmento).pack(side=tk.LEFT, padx=5)
        self.fragmento_label = ttk.Label(fragmento_frame, text="(video completo)", foreground="gray")
        self.fragmento_label.pack(side=tk.LEFT, padx=5)

        # Prueba rapida: el fragmento (o los primeros segundos) a pocos fps, con las opciones actuales
        prueba_frame = ttk.Frame(process_frame)
        prueba_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        ModernButton(prueba_frame, text="Video de prueba", command=self.probar_video).pack(side=tk.LEFT, padx=5)
        ttk.Label(prueba_frame, text="a").pack(side=tk.LEFT, padx=(5, 3))
        self.prueba_fps_var = tk.StringVar(value=f"{FPS_PRUEBA:g}")
        ttk.Spinbox(prueba_frame, textvariable=self.prueba_fps_var, values=("5", "8", "10", "12", "15"),
                    width=4).pack(side=tk.LEFT)
        ttk.Label(prueba_frame, text=f"fps (el fragmento elegido, o los primeros {SEGUNDOS_PRUEBA} s)",
                  foreground="gray").pack(side=tk.LEFT, padx=5)
        self.prueba_fps = None  # fps mientras se procesa un video de prueba; None = proceso normal

        # Botones de proceso
        botones_proceso = ttk.Frame(derecha)
        botones_proceso.pack(pady=10)
        self.process_button = ModernButton(botones_proceso, text="Iniciar Procesamiento",
                                         command=self.start_processing)
        self.process_button.pack(side=tk.LEFT, padx=5)
        self.detener_button = ModernButton(botones_proceso, text="Detener", command=self.detener_proceso)
        self.detener_button.pack(side=tk.LEFT, padx=5)
        self.detener_button.state(['disabled'])
        
        # Progress frame
        progress_frame = ttk.Frame(derecha)
        progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress.pack(fill=tk.X, padx=10)
        
        self.status_label = ttk.Label(progress_frame, text="Listo")
        self.status_label.pack(pady=5)

    def fragmento_activo(self):
        """(inicio, fin) en frames si esta marcado "Solo un fragmento" y el destino es un video; si no, None"""
        if self.fragmento_var.get() and self.fragmento and self.es_video(self.target_path):
            return self.fragmento
        return None

    def cambiar_fragmento_check(self):
        # Marcar la casilla sin haber elegido un fragmento abre la ventana para elegirlo
        if self.fragmento_var.get() and self.fragmento is None:
            self.elegir_fragmento()
        self._mostrar_fragmento()

    def elegir_fragmento(self):
        if not self.es_video(self.target_path):
            messagebox.showinfo("Fragmento", "Primero elige un video destino")
            self.fragmento_var.set(False)
            return
        if info_video(self.target_path) is None:
            messagebox.showerror("Fragmento", "No se pudo abrir el video destino")
            self.fragmento_var.set(False)
            return
        VentanaFragmento(self.root, self.target_path, self.fragmento, self.fragmento_elegido)

    def fragmento_elegido(self, fragmento):
        self.fragmento = fragmento
        self.fragmento_var.set(fragmento is not None)
        self._mostrar_fragmento()
        # El preview abierto pasa a recorrer solo el fragmento
        if self.preview_window is not None and self.preview_window.winfo_exists():
            self.show_preview()

    def _mostrar_fragmento(self):
        fragmento = self.fragmento_activo()
        if fragmento is None:
            self.fragmento_label.configure(text="(video completo)")
            return
        fps = info_video(self.target_path)["fps"]
        inicio, fin = fragmento
        self.fragmento_label.configure(text=f"{formato_tiempo(inicio / fps)} → {formato_tiempo(fin / fps)}  "
                                            f"(dura {formato_tiempo((fin - inicio) / fps)})")

    def gpen_minimo(self):
        """Tamaño minimo de cara para GPEN como texto ("96" px o "25%"); si lo escrito no vale, el de por defecto"""
        porcentaje = self.gpen_unidad_var.get() == "% imagen"
        try:
            return validar_minimo(self.gpen_minimo_var.get().strip() + ("%" if porcentaje else ""))
        except ValueError:
            return "25%" if porcentaje else MINIMO_POR_DEFECTO

    def fuerza_mejora(self):
        """Fuerza de GPEN que se pasa al cambio de cara: 0 si la casilla esta apagada"""
        return self.fuerza_gpen_var.get() if self.var_mejorar.get() else 0.0

    def leer_configuracion(self):
        """Todas las opciones de la interfaz (sin fuente, destino ni fragmento), con las claves de core/Configuraciones.py"""
        def numero(texto, tipo):
            try:
                return tipo(str(texto).replace(",", "."))
            except ValueError:
                return texto  # se rechaza al validar

        hilos = numero(self.thread_var.get(), int)
        return {
            "asignacion": self.asignacion_var.get(),
            "genero": self.genero_var.get(),
            "caras_inclinadas": self.caras_inclinadas_var.get(),
            "personas_fps": numero(self.personas_fps_var.get(), float),
            "personas_max": numero(self.personas_max_var.get(), int),
            "indices_fuente": self.src_indices_var.get(),
            "indices_destino": self.dst_indices_var.get(),
            "mascara": self.mascara_var.get(),
            "color": self.color_var.get(),
            "ojos": self.var_ojos.get(),
            "boca": self.var_boca.get(),
            "nariz": self.var_nariz.get(),
            "frente": self.var_frente.get(),
            "mejillas": self.var_mejillas.get(),
            "maquillaje": round(self.maquillaje_var.get(), 3),
            "oclusion": self.var_oclucion.get(),
            "oclusion_modo": self.oclusion_modo_var.get(),
            "filtro": self.combofiltro.get(),
            "mejorar": self.var_mejorar.get(),
            "fuerza_mejora": round(self.fuerza_gpen_var.get(), 3),
            "minimo_mejora": self.gpen_minimo(),
            "hilos": hilos,
            "contenedor": self.contenedor_var.get(),
            "codec": self.codec_var.get(),
            "zip": self.zip_var.get(),
            "zip_conservar": self.zip_conservar_var.get(),
            "carpeta_salida": self.output_path_var.get(),
            "nombre_archivo": self.output_filename_var.get(),
        }

    def aplicar_configuracion(self, ajustes):
        """Pone en la interfaz las opciones dadas; las que no vienen se quedan como estaban"""
        variables = {
            "asignacion": self.asignacion_var, "genero": self.genero_var,
            "indices_fuente": self.src_indices_var, "indices_destino": self.dst_indices_var,
            "mascara": self.mascara_var, "color": self.color_var,
            "ojos": self.var_ojos, "boca": self.var_boca, "nariz": self.var_nariz, "frente": self.var_frente,
            "mejillas": self.var_mejillas, "maquillaje": self.maquillaje_var, "oclusion": self.var_oclucion, "mejorar": self.var_mejorar,
            "oclusion_modo": self.oclusion_modo_var,
            "fuerza_mejora": self.fuerza_gpen_var, "contenedor": self.contenedor_var, "codec": self.codec_var,
            "carpeta_salida": self.output_path_var, "nombre_archivo": self.output_filename_var,
            "zip": self.zip_var, "zip_conservar": self.zip_conservar_var,
            "caras_inclinadas": self.caras_inclinadas_var,
        }
        for clave, variable in variables.items():
            if clave in ajustes:
                variable.set(ajustes[clave])
        if "hilos" in ajustes:
            self.thread_var.set(str(ajustes["hilos"]))
        if "personas_fps" in ajustes:
            self.personas_fps_var.set(f"{ajustes['personas_fps']:g}")
        if "personas_max" in ajustes:
            self.personas_max_var.set(str(ajustes["personas_max"]))
        if "minimo_mejora" in ajustes:
            minimo = ajustes["minimo_mejora"]
            porcentaje = minimo.endswith("%")
            self.gpen_unidad_var.set("% imagen" if porcentaje else "px")
            self.gpen_minimo_var.set(minimo.rstrip("%"))
        if "filtro" in ajustes:
            self.combofiltro.set(ajustes["filtro"])
        self._estado_zip()
        self.config_cambiada()

    def actualizar_lista_configuraciones(self, elegida=None):
        nombres = Configuraciones.listar()
        self.configuracion_combo.configure(values=nombres)
        self.configuracion_var.set(elegida if elegida in nombres else "")

    def cargar_configuracion(self):
        nombre = self.configuracion_var.get()
        if not nombre:
            return
        try:
            ajustes, avisos = Configuraciones.cargar(nombre)
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Configuración", str(e))
            self.actualizar_lista_configuraciones()
            return
        self.aplicar_configuracion(ajustes)
        faltan = [a for a in avisos if a.startswith("No trae")]
        errores = [a for a in avisos if not a.startswith("No trae")]
        # Una configuracion de una version anterior no trae las opciones nuevas: quedan como estaban, sin ventana
        self.status_label.config(text=f"Configuración «{nombre}» cargada" + (f" ({faltan[0]})" if faltan else ""))
        if errores:
            messagebox.showwarning("Configuración", f"«{nombre}» se cargó con avisos:\n\n" + "\n".join(errores))

    def guardar_configuracion(self):
        """Guarda todas las opciones actuales con un nombre (si ya existe, pregunta antes de reemplazarla)"""
        nombre = simpledialog.askstring("Guardar configuración",
                                        "Nombre de la configuración (por ejemplo: catrina, natural):",
                                        initialvalue=self.configuracion_var.get(), parent=self.root)
        if not nombre:
            return
        limpio = Configuraciones.nombre_valido(nombre)
        if not limpio:
            messagebox.showerror("Guardar configuración", "Usa letras, números, guiones o espacios en el nombre")
            return
        if limpio in Configuraciones.listar() and not messagebox.askyesno(
                "Guardar configuración", f"Ya existe «{limpio}». ¿Reemplazarla?"):
            return
        try:
            ruta = Configuraciones.guardar(limpio, self.leer_configuracion())
        except (OSError, ValueError) as e:
            messagebox.showerror("Guardar configuración", str(e))
            return
        self.actualizar_lista_configuraciones(limpio)
        self.status_label.config(text=f"Configuración «{limpio}» guardada en {ruta}")

    def eliminar_configuracion(self):
        nombre = self.configuracion_var.get()
        if not nombre:
            messagebox.showinfo("Eliminar configuración", "Primero elige una configuración en la lista")
            return
        if not messagebox.askyesno("Eliminar configuración", f"¿Eliminar la configuración «{nombre}»?\n\n"
                                   "No se puede deshacer. Las opciones que están en pantalla no cambian.",
                                   default=messagebox.NO):
            return
        try:
            Configuraciones.eliminar(nombre)
        except OSError as e:
            messagebox.showerror("Eliminar configuración", str(e))
        self.actualizar_lista_configuraciones()
        self.status_label.config(text=f"Configuración «{nombre}» eliminada")

    def set_source(self, path):
        if os.path.isdir(path):
            self.crear_modelo_desde([path])
            return
        if Modelo_Cara.es_modelo(path):
            self.agregar_o_reemplazar_modelos([path])
            return
        if Modelo_Cara.es_video(path):
            self.crear_modelo_desde_video([path])
            return
        self.modelos_src = []
        self.source_path = path
        self._caras_src_cache = (None, None)
        self.modelos_label.config(text="(ninguno)")
        self.update_preview(path, self.source_frame)
        self.config_cambiada()

    def soltar_varios(self, rutas):
        """Varios archivos o una carpeta en la zona fuente: varios .npy = varias fuentes; fotos o videos = crear un modelo"""
        modelos = [r for r in rutas if Modelo_Cara.es_modelo(r)]
        videos = [r for r in rutas if Modelo_Cara.es_video(r)]
        if (modelos and len(modelos) != len(rutas)) or (videos and len(videos) != len(rutas)):
            messagebox.showwarning("Fuente", "Suelta solo modelos de cara (.npy), solo fotos o solo videos, sin mezclarlos")
            return
        if modelos:
            self.agregar_o_reemplazar_modelos(modelos)
        elif videos:
            self.crear_modelo_desde_video(videos)
        else:
            self.crear_modelo_desde(rutas)

    def agregar_o_reemplazar_modelos(self, rutas):
        """Si ya hay modelos cargados pregunta si agregar los nuevos al final o reemplazarlos"""
        if self.modelos_src:
            respuesta = messagebox.askyesnocancel(
                "Modelos de cara",
                f"Ya hay {len(self.modelos_src)} modelo(s) cargado(s).\n\n"
                f"Sí = agregar como fuente {len(self.modelos_src)}"
                f"{f' a {len(self.modelos_src) + len(rutas) - 1}' if len(rutas) > 1 else ''}\n"
                "No = reemplazar los actuales")
            if respuesta is None:
                return
            if respuesta:
                rutas = self.modelos_src + [r for r in map(os.path.abspath, rutas) if r not in self.modelos_src]
        self.set_modelos(rutas)

    def set_modelos(self, rutas):
        """Usa uno o varios modelos de cara (.npy) como fuentes; su orden es el índice fuente (0, 1, 2...)"""
        rutas = [os.path.abspath(r) for r in rutas]
        caras = []
        for ruta in rutas:
            try:
                caras.append(Modelo_Cara.cargar(ruta))
            except (FileNotFoundError, ValueError) as e:
                messagebox.showerror("Modelo de cara", str(e))
                return

        self.modelos_src = list(rutas)
        self.source_path = "modelos:" + "|".join(self.modelos_src)
        self._caras_src_cache = (self.source_path, caras)

        nombres = [os.path.splitext(os.path.basename(r))[0] for r in self.modelos_src]
        self.modelos_label.config(text=", ".join(f"{i}={n}" for i, n in enumerate(nombres)))
        self.source_frame.update_preview(self._miniaturas_numeradas(self.modelos_src))
        self.config_cambiada()

        detalles = []
        for i, (ruta, nombre) in enumerate(zip(self.modelos_src, nombres)):
            datos = Modelo_Cara.datos(ruta)
            detalles.append(f"{i}={nombre}" + (f" ({datos['caras_usadas']} caras)" if "caras_usadas" in datos else ""))
        self.status_label.config(text="Fuentes: " + ", ".join(detalles))

    @staticmethod
    def _miniaturas_numeradas(rutas, lado=160):
        """Miniaturas de los modelos una al lado de otra, con su índice debajo"""
        piezas = []
        for i, ruta in enumerate(rutas):
            mini = Modelo_Cara.miniatura(ruta)
            if mini is None:
                mini = np.full((lado, lado, 3), 200, dtype=np.uint8)
                cv2.putText(mini, "sin foto", (35, lado // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1)
            mini = cv2.resize(mini, (lado, lado))
            barra = np.full((32, lado, 3), 255, dtype=np.uint8)
            texto = f"[{i}]"
            ancho = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
            cv2.putText(barra, texto, ((lado - ancho) // 2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            piezas.append(np.vstack([mini, barra]))
            piezas.append(np.full((lado + 32, 8, 3), 255, dtype=np.uint8))
        return np.hstack(piezas[:-1])

    def quitar_fuente(self):
        self.modelos_src = []
        self.source_path = None
        self._caras_src_cache = (None, None)
        self.modelos_label.config(text="(ninguno)")
        self.source_frame.update_preview(None)
        self.status_label.config(text="Listo")

    def elegir_modelos_dialogo(self):
        """Ventana para marcar los modelos de src/modelos_caras/ y ordenarlos (el orden es el índice fuente)"""
        self._ventana_modelos(combinar=False)

    def combinar_modelos_dialogo(self):
        """Ventana para unir varios modelos de la misma persona en uno nuevo"""
        self._ventana_modelos(combinar=True)

    def _ventana_modelos(self, combinar):
        """Lista de modelos de src/modelos_caras/ con casillas. Para elegir fuentes se ordenan con ↑ ↓ (orden = índice);
        para combinar se marcan los de la misma persona y se escribe el nombre del modelo nuevo"""
        disponibles = Modelo_Cara.listar()
        minimo = 2 if combinar else 1
        if len(disponibles) < minimo:
            messagebox.showinfo("Modelos de cara", f"Hacen falta al menos {minimo} modelo(s) en {CARPETA_MODELOS}.\n"
                                "Crea uno con «Crear modelo de cara».")
            return

        # Para elegir fuentes: primero los ya cargados, marcados y en su orden; luego el resto.
        # Para combinar: nada marcado, para no mezclar sin querer el modelo que esta cargado
        cargados = [] if combinar else [r for r in self.modelos_src if os.path.abspath(r) in map(os.path.abspath, disponibles)]
        resto = [r for r in disponibles if os.path.abspath(r) not in map(os.path.abspath, cargados)]
        filas = [[r, tk.BooleanVar(value=True)] for r in cargados] + [[r, tk.BooleanVar(value=False)] for r in resto]

        ventana = tk.Toplevel(self.root)
        ventana.title("Combinar modelos de la misma persona" if combinar else "Elegir modelos de cara")
        ventana.transient(self.root)
        ventana.grab_set()
        if combinar:
            ttk.Label(ventana, text="Marca los modelos de la MISMA persona. Cada uno pesa según cuántas caras tiene;\n"
                                    "los modelos originales no se modifican.").pack(padx=15, pady=(10, 0), anchor="w")
        # Lista con barra de desplazamiento: con muchos modelos la ventana no crece mas alla de ALTO_LISTA
        contenedor = ttk.Frame(ventana)
        contenedor.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        lienzo = tk.Canvas(contenedor, highlightthickness=0, borderwidth=0,
                           background=ttk.Style().lookup("TFrame", "background"))
        barra = ttk.Scrollbar(contenedor, orient=tk.VERTICAL, command=lienzo.yview)
        lienzo.configure(yscrollcommand=barra.set)
        lienzo.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lista = ttk.Frame(lienzo)
        lienzo.create_window((0, 0), window=lista, anchor="nw")
        alto_maximo = min(ALTO_LISTA_MODELOS, int(self.root.winfo_screenheight() * 0.5))

        def ajustar_lista(_=None):
            alto = lista.winfo_reqheight()
            lienzo.configure(scrollregion=(0, 0, lista.winfo_reqwidth(), alto),
                             width=lista.winfo_reqwidth(), height=min(alto, alto_maximo))
            if alto > alto_maximo:
                barra.pack(side=tk.RIGHT, fill=tk.Y)
            else:
                barra.pack_forget()
                lienzo.yview_moveto(0)

        def rueda(evento):
            if lista.winfo_reqheight() > alto_maximo:
                lienzo.yview_scroll(int(-evento.delta / 120), "units")

        lista.bind("<Configure>", ajustar_lista)
        ventana.bind("<MouseWheel>", rueda)
        orden_label = ttk.Label(ventana)
        orden_label.pack(padx=15, anchor="w")
        nombre_var = tk.StringVar()
        if combinar:
            nombre_frame = ttk.Frame(ventana)
            nombre_frame.pack(fill=tk.X, padx=15, pady=(10, 0))
            ttk.Label(nombre_frame, text="Nombre del modelo combinado:").pack(side=tk.LEFT)
            ttk.Entry(nombre_frame, textvariable=nombre_var, width=25).pack(side=tk.LEFT, padx=5)

        def actualizar_orden():
            marcados = [r for r, v in filas if v.get()]
            if combinar:
                total = sum(max(1, int(Modelo_Cara.datos(r).get("caras_usadas", 1))) for r in marcados)
                orden_label.config(text=f"{len(marcados)} modelo(s) marcado(s), {total} cara(s) en total")
                return
            nombres = [os.path.splitext(os.path.basename(r))[0] for r in marcados]
            orden_label.config(text="Orden = índice: " + (", ".join(f"{n}={i}" for i, n in enumerate(nombres)) or "(ninguno)"))

        def mover(i, paso):
            j = i + paso
            if 0 <= j < len(filas):
                filas[i], filas[j] = filas[j], filas[i]
                dibujar()

        def dibujar():
            for hijo in lista.winfo_children():
                hijo.destroy()
            for i, (ruta, var) in enumerate(filas):
                datos = Modelo_Cara.datos(ruta)
                nombre = os.path.splitext(os.path.basename(ruta))[0]
                texto = nombre + (f"  ({datos['caras_usadas']} caras)" if "caras_usadas" in datos else "")
                ttk.Checkbutton(lista, text=texto, variable=var, command=actualizar_orden).grid(row=i, column=0, sticky="w", pady=2)
                if not combinar:
                    ttk.Button(lista, text="↑", width=3, command=lambda i=i: mover(i, -1)).grid(row=i, column=1, padx=(10, 2))
                    ttk.Button(lista, text="↓", width=3, command=lambda i=i: mover(i, 1)).grid(row=i, column=2)
                ttk.Button(lista, text="Eliminar", width=8, command=lambda i=i: eliminar(i)).grid(row=i, column=3, padx=(10, 0))
            actualizar_orden()

        def eliminar(i):
            """Borra el modelo de src/modelos_caras/ (.npy, .json y miniatura), después de confirmar"""
            ruta = filas[i][0]
            nombre = os.path.splitext(os.path.basename(ruta))[0]
            if not messagebox.askyesno("Eliminar modelo",
                                       f"¿Eliminar el modelo «{nombre}»?\n\nSe borran su archivo .npy, sus datos (.json) "
                                       f"y su miniatura (.jpg) de {CARPETA_MODELOS}.\nNo se puede deshacer.",
                                       icon="warning", default="no", parent=ventana):
                return
            try:
                Modelo_Cara.eliminar(ruta)
            except (OSError, ValueError) as e:
                messagebox.showerror("Eliminar modelo", f"No se pudo eliminar «{nombre}»:\n{e}", parent=ventana)
                return
            del filas[i]
            dibujar()
            # Si estaba cargado como fuente, se quita de las fuentes (los demás siguen, en su orden)
            if any(os.path.abspath(r) == os.path.abspath(ruta) for r in self.modelos_src):
                quedan = [r for r in self.modelos_src if os.path.abspath(r) != os.path.abspath(ruta)]
                if quedan:
                    self.set_modelos(quedan)
                else:
                    self.quitar_fuente()
            self.status_label.config(text=f"Modelo «{nombre}» eliminado")

        def aceptar():
            elegidos = [r for r, v in filas if v.get()]
            if len(elegidos) < minimo:
                messagebox.showwarning("Modelos de cara", f"Marca al menos {minimo} modelo(s)", parent=ventana)
                return
            if not combinar:
                ventana.destroy()
                self.set_modelos(elegidos)
                return

            nombre = Modelo_Cara.nombre_seguro(nombre_var.get())
            if not nombre:
                messagebox.showwarning("Modelos de cara", "Escribe un nombre (letras, números, guiones o espacios)",
                                       parent=ventana)
                return
            salida = os.path.join(CARPETA_MODELOS, nombre + ".npy")
            if os.path.exists(salida) and not messagebox.askyesno("Modelos de cara", f"Ya existe «{nombre}». ¿Reemplazarlo?",
                                                                  parent=ventana):
                return
            try:
                datos = Modelo_Cara.combinar(elegidos, salida, progreso=print)
            except (FileNotFoundError, ValueError) as e:
                messagebox.showerror("Combinar modelos", str(e), parent=ventana)
                return
            ventana.destroy()
            self.set_modelos([salida])
            mensaje = (f"Modelo «{datos['nombre']}» creado con {datos['caras_usadas']} cara(s) de "
                       f"{len(datos['combinado_de'])} modelos.\n\n")
            mensaje += "\n".join(f"{m['nombre']}: {m['caras']} cara(s), similitud con el resultado "
                                 f"{m['similitud_con_resultado']:.2f}" for m in datos["combinado_de"])
            if datos["similitud_promedio"] >= 0.9:
                mensaje += "\n\nLos modelos eran casi iguales: combinarlos aportó poco."
            messagebox.showinfo("Combinar modelos", mensaje + f"\n\nGuardado en {salida}")

        dibujar()
        botones = ttk.Frame(ventana)
        botones.pack(pady=10)
        ttk.Button(botones, text="Aceptar", command=aceptar).pack(side=tk.LEFT, padx=5)
        ttk.Button(botones, text="Cancelar", command=ventana.destroy).pack(side=tk.LEFT, padx=5)

    def validar_indices_fuente(self, caras_src):
        """Avisa antes de procesar si algún índice fuente no existe"""
        try:
            indices = [int(i) for i in self.src_indices_var.get().split(',')]
        except ValueError:
            raise Exception("Los Índices Fuente deben ser números separados por comas, por ejemplo 0,1")
        tipo = "modelo(s)" if self.modelos_src else "cara(s) en la imagen fuente"
        malos = [i for i in indices if i < 0 or i >= len(caras_src)]
        if malos:
            raise Exception(f"El índice fuente {malos[0]} no existe: hay {len(caras_src)} {tipo} "
                            f"(índices de 0 a {len(caras_src) - 1})")

    def obtener_caras_src(self):
        """Caras fuente para el procesamiento: de los modelos .npy o detectadas una vez en la foto fuente"""
        ruta, caras = self._caras_src_cache
        if ruta == self.source_path and caras is not None:
            return caras

        if self.modelos_src:
            caras = [Modelo_Cara.cargar(r) for r in self.modelos_src]
        else:
            source_img = leer(self.source_path)
            if source_img is None:
                raise Exception("No se pudo leer la imagen fuente")
            caras = self.swapper.detectar_caras(source_img)
            if not caras:
                raise Exception("No se detectó ninguna cara en la imagen fuente")
        self._caras_src_cache = (self.source_path, caras)
        return caras

    def crear_modelo_dialogo(self):
        rutas = filedialog.askopenfilenames(title="Fotos de la misma persona (o un collage)",
                                            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if rutas:
            self.crear_modelo_desde(list(rutas))

    def crear_modelo_desde(self, rutas):
        """Crea un modelo de cara con varias fotos, una carpeta o un collage, y lo deja como fuente"""
        if self.processing:
            messagebox.showwarning("Aviso", "Espera a que termine el procesamiento actual")
            return

        imagenes = Modelo_Cara.rutas_imagenes(rutas)
        if not imagenes:
            messagebox.showwarning("Modelo de cara", "No hay imágenes (.jpg, .png, .bmp, .webp) en lo que soltaste")
            return
        if not messagebox.askyesno("Modelo de cara", f"¿Crear un modelo de cara con {len(imagenes)} foto(s)?\n\n"
                                   "Deben ser de la misma persona (un collage puede tener varias caras)."):
            return

        nombre = simpledialog.askstring("Modelo de cara", "Nombre del modelo (por ejemplo: persona1):", parent=self.root)
        if not nombre:
            return
        nombre = "".join(c for c in nombre.strip() if c.isalnum() or c in "-_ ").strip()
        if not nombre:
            messagebox.showerror("Modelo de cara", "Usa letras, números, guiones o espacios en el nombre")
            return

        salida = os.path.join(CARPETA_MODELOS, nombre + ".npy")
        if os.path.exists(salida) and not messagebox.askyesno("Modelo de cara", f"Ya existe «{nombre}». ¿Reemplazarlo?"):
            return
        self._lanzar_creacion(imagenes, salida)

    def crear_modelo_video_dialogo(self):
        rutas = filedialog.askopenfilenames(title="Video(s) de la misma persona",
                                            filetypes=[("Videos", "*.mp4 *.avi *.mkv *.mov *.webm")])
        if rutas:
            self.crear_modelo_desde_video(list(rutas))

    def crear_modelo_desde_video(self, videos):
        """Ventana para crear un modelo con los frames de uno o varios videos: se elige cuántos frames por
        segundo sacar (menos fps = menos frames y más rápido) y el nombre del modelo"""
        if self.processing:
            messagebox.showwarning("Aviso", "Espera a que termine el procesamiento actual")
            return
        try:
            info = [Modelo_Cara.info_video(r) for r in videos]
        except ValueError as e:
            messagebox.showerror("Modelo de cara", str(e))
            return

        ventana = tk.Toplevel(self.root)
        ventana.title("Crear modelo de cara desde video")
        ventana.transient(self.root)
        ventana.grab_set()
        ventana.resizable(False, False)

        ttk.Label(ventana, text="El video debe mostrar a UNA persona (si aparece otra, se usa la que más se repite).\n"
                                "Los frames se sacan en memoria; solo se guardan imágenes si eliges revisar las caras."
                  ).pack(padx=15, pady=(10, 5), anchor="w")
        for ruta, (fps_original, total, duracion) in zip(videos, info):
            ttk.Label(ventana, text=f"• {os.path.basename(ruta)}: {duracion:.1f} s, {fps_original:.0f} fps ({total} frames)"
                      ).pack(padx=15, anchor="w")

        opciones = ttk.Frame(ventana)
        opciones.pack(fill=tk.X, padx=15, pady=(10, 0))
        ttk.Label(opciones, text="Frames por segundo a sacar:").grid(row=0, column=0, sticky="w")
        fps_var = tk.StringVar(value=f"{FPS_VIDEO:g}")
        ttk.Spinbox(opciones, textvariable=fps_var, width=6,
                    values=("0.25", "0.5", "1", "2", "3", "5", "10")).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(opciones, text="Nombre del modelo:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        sugerido = Modelo_Cara.nombre_seguro(os.path.splitext(os.path.basename(videos[0]))[0])
        nombre_var = tk.StringVar(value=sugerido)
        ttk.Entry(opciones, textvariable=nombre_var, width=25).grid(row=1, column=1, sticky="w", padx=5, pady=(8, 0))
        revisar_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opciones, text="Revisar las caras antes de crear el modelo", variable=revisar_var
                        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(opciones, text="Guarda las caras por persona (de mejor a peor) para que borres las malas;\n"
                                 "luego se crea el modelo con las que quedan.", foreground="gray"
                  ).grid(row=3, column=0, columnspan=2, sticky="w")

        estimado_label = ttk.Label(ventana, justify=tk.LEFT)
        estimado_label.pack(padx=15, pady=(10, 0), anchor="w")

        def leer_fps():
            try:
                fps = float(fps_var.get().replace(",", "."))
            except ValueError:
                return None
            return fps if 0 < fps <= 60 else None

        def actualizar(*_):
            fps = leer_fps()
            if fps is None:
                estimado_label.config(text="Escribe un número mayor que 0 (por ejemplo 0.5 = un frame cada 2 segundos)")
                return
            frames = sum(Modelo_Cara.frames_a_extraer(d, fps) for _, _, d in info)
            texto = f"Se sacarán unos {frames} frame(s)."
            if any(Modelo_Cara.fps_efectivo(d, fps) < fps for _, _, d in info):
                texto += f"\nVideo largo: se reparten como máximo {MAX_FRAMES_VIDEO} frames por video en todo el video."
            if frames < 5:
                texto += "\nPocos frames: sube los fps para que el parecido sea más estable."
            estimado_label.config(text=texto)

        fps_var.trace_add("write", actualizar)
        actualizar()

        def aceptar():
            fps = leer_fps()
            if fps is None:
                messagebox.showwarning("Modelo de cara", "Los frames por segundo deben ser un número entre 0 y 60",
                                       parent=ventana)
                return
            nombre = Modelo_Cara.nombre_seguro(nombre_var.get())
            if not nombre:
                messagebox.showwarning("Modelo de cara", "Escribe un nombre (letras, números, guiones o espacios)",
                                       parent=ventana)
                return
            salida = os.path.join(CARPETA_MODELOS, nombre + ".npy")
            if os.path.exists(salida) and not messagebox.askyesno("Modelo de cara", f"Ya existe «{nombre}». ¿Reemplazarlo?",
                                                                  parent=ventana):
                return
            if revisar_var.get():
                carpeta = os.path.join(CARPETA_CARAS, nombre)
                reemplazar = False
                if os.path.isdir(carpeta) and os.listdir(carpeta):
                    if not os.path.exists(os.path.join(carpeta, MARCA_CARAS)):
                        messagebox.showerror("Modelo de cara", f"{carpeta} ya tiene otros archivos; usa otro nombre",
                                             parent=ventana)
                        return
                    if not messagebox.askyesno("Modelo de cara", f"Ya hay caras extraídas de «{nombre}». "
                                               "¿Reemplazarlas?", parent=ventana):
                        return
                    reemplazar = True
                ventana.destroy()
                self._lanzar_extraccion(videos, carpeta, salida, fps, reemplazar)
                return
            ventana.destroy()
            self._lanzar_creacion(videos, salida, fps_video=fps)

        def texto_boton(*_):
            boton_aceptar.config(text="Extraer caras" if revisar_var.get() else "Crear modelo")

        botones = ttk.Frame(ventana)
        botones.pack(pady=10)
        boton_aceptar = ttk.Button(botones, text="Crear modelo", command=aceptar)
        boton_aceptar.pack(side=tk.LEFT, padx=5)
        revisar_var.trace_add("write", texto_boton)
        ttk.Button(botones, text="Cancelar", command=ventana.destroy).pack(side=tk.LEFT, padx=5)

    def _lanzar_creacion(self, rutas, salida, **opciones):
        """Crea el modelo en segundo plano (fotos y/o videos) mostrando el avance en la barra de estado"""
        self.processing = True
        self.process_button.state(['disabled'])

        def estado(texto):
            print(texto)
            self.root.after(0, lambda: self.status_label.config(text=texto))

        def tarea():
            try:
                datos = Modelo_Cara.crear(rutas, self.swapper.get_Analizador(), salida, progreso=estado, **opciones)
                self.root.after(0, lambda: self._modelo_creado(salida, datos))
            except Exception as e:
                error = str(e)
                self.root.after(0, lambda: messagebox.showerror("Modelo de cara", error))
            finally:
                self.processing = False
                self.root.after(0, lambda: self.process_button.state(['!disabled']))

        threading.Thread(target=tarea, daemon=True).start()

    def _lanzar_extraccion(self, videos, carpeta, salida, fps, reemplazar):
        """Extrae las caras a carpeta/persona_N en segundo plano y abre la ventana para revisarlas"""
        self.processing = True
        self.process_button.state(['disabled'])

        def estado(texto):
            print(texto)
            self.root.after(0, lambda: self.status_label.config(text=texto))

        def tarea():
            try:
                # La carpeta _en_proceso se abre al empezar: las caras van apareciendo a medida que se detectan
                datos = Modelo_Cara.extraer_caras(
                    videos, self.swapper.get_Analizador(), carpeta, progreso=estado, fps_video=fps,
                    reemplazar=reemplazar, al_empezar=lambda ruta: self.root.after(0, lambda: self._abrir_carpeta(ruta)))
                self.root.after(0, lambda: self._revisar_caras(datos, salida))
            except Exception as e:
                error = str(e)
                self.root.after(0, lambda: messagebox.showerror("Modelo de cara", error))
            finally:
                self.processing = False
                self.root.after(0, lambda: self.process_button.state(['!disabled']))

        threading.Thread(target=tarea, daemon=True).start()

    @staticmethod
    def _abrir_carpeta(carpeta):
        try:
            os.startfile(carpeta)
        except (AttributeError, OSError):  # os.startfile solo existe en Windows
            subprocess.Popen(["xdg-open", carpeta])

    def _revisar_caras(self, datos, salida):
        """Ventana (no bloquea) para elegir la persona, abrir su carpeta, borrar las caras malas en el explorador
        y crear el modelo con las que quedan"""
        personas = datos["personas"]
        self.status_label.config(text=f"{datos['caras']} cara(s) extraídas en {datos['carpeta']}")
        ventana = tk.Toplevel(self.root)
        ventana.title("Revisar caras extraídas")
        ventana.transient(self.root)
        ventana.resizable(False, False)

        ttk.Label(ventana, justify=tk.LEFT,
                  text="1. Elige la persona y abre su carpeta.\n"
                       "2. Borra las caras malas: borrosas, tapadas, con gestos raros o de otra persona\n"
                       "    (están ordenadas de mejor a peor: las malas suelen estar al final).\n"
                       "3. Pulsa «Crear modelo» con las caras que quedaron."
                  ).pack(padx=15, pady=(10, 5), anchor="w")

        persona_var = tk.IntVar(value=0)
        lista = ttk.Frame(ventana)
        lista.pack(padx=15, pady=5, anchor="w")
        ventana.fotos = []
        for k, persona in enumerate(personas[:8]):
            fila = ttk.Frame(lista)
            fila.pack(anchor="w", pady=2)
            imagenes = Modelo_Cara.rutas_imagenes([persona["carpeta"]])
            if imagenes:
                mini = leer(imagenes[0])
                if mini is not None:
                    foto = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(cv2.resize(mini, (64, 64)), cv2.COLOR_BGR2RGB)))
                    ventana.fotos.append(foto)
                    ttk.Label(fila, image=foto).pack(side=tk.LEFT, padx=(0, 6))
            ttk.Radiobutton(fila, variable=persona_var, value=k,
                            text=f"{os.path.basename(persona['carpeta'])}: {persona['caras']} cara(s)"
                            ).pack(side=tk.LEFT)
        if len(personas) > 8:
            ttk.Label(ventana, text=f"(y {len(personas) - 8} persona(s) más con pocas caras en {datos['carpeta']})",
                      foreground="gray").pack(padx=15, anchor="w")

        def carpeta_elegida():
            return personas[persona_var.get()]["carpeta"]

        def crear():
            if not os.path.isdir(carpeta_elegida()):
                messagebox.showerror("Modelo de cara", f"Ya no existe {carpeta_elegida()}", parent=ventana)
                return
            imagenes = Modelo_Cara.rutas_imagenes([carpeta_elegida()])
            if not imagenes:
                messagebox.showwarning("Modelo de cara", "La carpeta quedó sin caras", parent=ventana)
                return
            if self.processing:
                messagebox.showwarning("Aviso", "Espera a que termine el procesamiento actual", parent=ventana)
                return
            nombre = os.path.splitext(os.path.basename(salida))[0]
            if not messagebox.askyesno("Modelo de cara", f"¿Crear el modelo «{nombre}» con {len(imagenes)} cara(s) de "
                                       f"{os.path.basename(carpeta_elegida())}?", parent=ventana):
                return
            ventana.destroy()
            self._lanzar_creacion(imagenes, salida)

        botones = ttk.Frame(ventana)
        botones.pack(pady=10)
        ttk.Button(botones, text="Abrir carpeta", command=lambda: self._abrir_carpeta(carpeta_elegida())
                   ).pack(side=tk.LEFT, padx=5)
        ttk.Button(botones, text="Crear modelo", command=crear).pack(side=tk.LEFT, padx=5)
        ttk.Button(botones, text="Cerrar", command=ventana.destroy).pack(side=tk.LEFT, padx=5)
        ttk.Label(ventana, text="Si cierras, las caras quedan guardadas: puedes crear el modelo después soltando\n"
                                "la carpeta de la persona en la zona de fuente.", foreground="gray", justify=tk.LEFT
                  ).pack(padx=15, pady=(0, 10), anchor="w")
        if personas:
            self._abrir_carpeta(personas[0]["carpeta"])

    def _modelo_creado(self, salida, datos):
        self.set_modelos([salida])
        origen = [f"{len(datos['fotos'])} foto(s)"] if datos["fotos"] else []
        origen += [f"{v['frames']} frame(s) de {v['nombre']}" for v in datos.get("videos", [])]
        mensaje = f"Modelo «{datos['nombre']}» creado con {datos['caras_usadas']} cara(s) de {' y '.join(origen)}."
        if datos["caras_descartadas"]:
            mensaje += f"\n{datos['caras_descartadas']} cara(s) descartada(s) por ser de otra persona o de baja calidad."
        if datos["caras_usadas"] > 1:
            mensaje += f"\nSimilitud promedio entre caras: {datos['similitud_promedio']:.2f}"
        messagebox.showinfo("Modelo de cara", mensaje + f"\n\nGuardado en {salida}")
        
    def set_target(self, path):
        if os.path.isdir(path):
            imagenes = imagenes_carpeta(path)
            if not imagenes:
                messagebox.showwarning("Carpeta de imágenes", "La carpeta no tiene imágenes (.jpg, .png, .bmp, .webp)")
                return
            texto = (f"{len(imagenes)} imagen(es) → se guardan en "
                     f"{os.path.basename(carpeta_salida(path, self.output_path_var.get()))}")
        else:
            texto = ""
        self.target_path = path
        self.carpeta_destino_label.config(text=texto)
        self.fragmento = None
        self.fragmento_var.set(False)
        self._mostrar_fragmento()
        self.update_preview(path, self.target_frame)
        if self.preview_window is not None and self.preview_window.winfo_exists():
            self.show_preview()

    def soltar_destino_varios(self, rutas):
        """En la zona destino se puede soltar una carpeta de imágenes (cada una se procesa por separado)"""
        if len(rutas) == 1 and os.path.isdir(rutas[0]):
            self.set_target(rutas[0])
        else:
            messagebox.showwarning("Destino", "Suelta un solo archivo, o una carpeta con las imágenes a procesar")

    def elegir_carpeta_destino(self):
        carpeta = filedialog.askdirectory(title="Carpeta con las imágenes destino")
        if carpeta:
            self.set_target(carpeta)

    @staticmethod
    def es_carpeta(ruta):
        return bool(ruta) and os.path.isdir(ruta)

    def _estado_zip(self):
        """La contraseña y «conservar» solo se usan con el ZIP activado"""
        estado = "normal" if self.zip_var.get() else "disabled"
        self.zip_contrasena_entry.configure(state=estado)
        self.zip_conservar_check.configure(state=estado)

    def _guardar_zip(self, archivos, salida_zip, carpeta_en_zip=""):
        """Mete los resultados en un ZIP con contraseña (AES-256). Devuelve el texto para el mensaje final"""
        self.root.after(0, lambda: self.status_label.config(text="Guardando en ZIP con contraseña..."))
        empaquetar(archivos, salida_zip, self.zip_contrasena_var.get(), conservar=self.zip_conservar_var.get(),
                   carpeta_en_zip=carpeta_en_zip)
        texto = f"en ZIP con contraseña:\n{salida_zip}\n(se abre con 7-Zip o WinRAR)"
        if self.zip_conservar_var.get():
            texto += "\nTambién se conservó el archivo suelto."
        return texto

    def select_output_dir(self):
        dir_path = filedialog.askdirectory(initialdir=self.output_dir)
        if dir_path:
            self.output_dir = dir_path
            self.output_path_var.set(dir_path)

    def update_preview(self, path, drop_frame):
        try:
            if os.path.isdir(path):  # carpeta de imágenes: se muestra la primera
                path = imagenes_carpeta(path)[0]
            if path.lower().endswith(('.mp4', '.avi','.mkv')):
                cap = cv2.VideoCapture(path)
                ret, img = cap.read()
                cap.release()
                if not ret:
                    raise Exception("No se pudo leer el video")
            else:
                img = leer(path)
                if img is None:
                    raise Exception("No se pudo leer la imagen")
            
            drop_frame.update_preview(img)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            drop_frame.update_preview(None)  # Reset preview on error

    def opciones_cambio(self):
        """
        Las opciones de la configuracion para cambiar_Faces (con index_src / index_dst) leidas de una vez.
        Se leen en el hilo de la interfaz y se pasan a los hilos que procesan
        """
        return dict(
            index_src=self.src_indices_var.get(),
            index_dst=self.dst_indices_var.get(),
            mode=self.combofiltro.get(),
            oclution=self.var_oclucion.get(),
            oclusion_modo=self.oclusion_modo_var.get(),
            genero=self.genero_var.get(),
            ojos=self.var_ojos.get(),
            boca=self.var_boca.get(),
            nariz=self.var_nariz.get(),
            frente=self.var_frente.get(),
            mejillas=self.var_mejillas.get(),
            mascara=self.mascara_var.get(),
            color=self.color_var.get(),
            maquillaje=self.maquillaje_var.get(),
            mejorar=self.fuerza_mejora(),
            mejorar_minimo=self.gpen_minimo()
        )

    def vigilar_configuracion(self):
        """Cada cambio en la configuracion actualiza el preview abierto"""
        for var in (self.src_indices_var, self.dst_indices_var, self.asignacion_var, self.genero_var,
                    self.mascara_var, self.color_var, self.var_ojos, self.var_boca, self.var_nariz, self.var_frente,
                    self.var_mejillas,
                    self.maquillaje_var, self.var_oclucion, self.oclusion_modo_var, self.var_mejorar,
                    self.caras_inclinadas_var,
                    self.fuerza_gpen_var,
                    self.gpen_minimo_var, self.gpen_unidad_var):
            var.trace_add("write", self.config_cambiada)
        self.combofiltro.bind("<<ComboboxSelected>>", self.config_cambiada)

    def config_cambiada(self, *args):
        if self.preview_window is not None and self.preview_window.winfo_exists():
            self.preview_window.config_cambiada()

    def process_frame(self, caras_src, frame):

        try:
            # Opciones fijadas al iniciar el proceso: cambiar la configuracion a la mitad no cambia el video
            frame_fake = self.swapper.cambiar_Faces(frame_dst=frame, caras_src=caras_src,
                                                    **(self.opciones_proceso or self.opciones_cambio()))
            
            return frame_fake
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame

    def crear_modo_identidad(self, caras_src, personas=None):
        """Prepara el modo por identidad: caras fuente detectadas una vez, mapa destino->fuente y registro.
        Con "personas" (analisis previo del video, core/Personas.py) las identidades quedan fijas desde el inicio:
        0 = la persona que mas aparece. Sin el, se registran en el orden en que aparecen (imagenes y carpetas)"""
        indices_src = [int(i) for i in self.src_indices_var.get().split(',')]
        indices_dst = [int(i) for i in self.dst_indices_var.get().split(',')]
        if len(indices_src) != len(indices_dst):
            raise Exception("En modo identidad los Índices Fuente y Destino deben tener la misma cantidad")

        registro = Registro_Identidades(max_ids=max(indices_dst) + 1)
        if personas is not None and len(personas):
            if max(indices_dst) >= len(personas):
                raise Exception(f"En el video hay {len(personas)} persona(s) (de 0 a {len(personas) - 1}) y en "
                                f"Índices Destino pusiste {max(indices_dst)}. Usa «Ver personas» para ver los números.")
            registro.fijar(personas.refs())
        return {
            "caras_src": caras_src,
            "mapa": dict(zip(indices_dst, indices_src)),
            "registro": registro,
        }

    def _estado_analisis(self):
        """Las opciones del analisis de personas solo se usan con asignacion por identidad"""
        estado = "normal" if self.asignacion_var.get() == "identidad" else "disabled"
        self.personas_fps_spin.configure(state=estado)
        self.personas_max_spin.configure(state=estado)

    def opciones_analisis(self):
        """(fps, maximo) del analisis de personas, validados; Exception con el motivo si no valen"""
        try:
            return validar_fps(self.personas_fps_var.get()), validar_maximo(self.personas_max_var.get())
        except ValueError as e:
            raise Exception(f"Análisis de personas: {e}")

    def analizar_personas(self, progreso=None, cancelado=None):
        """Analisis previo del video destino para el modo identidad (se guarda en la Carpeta de Salida y se
        reutiliza). Devuelve (Personas, carpeta), o (None, None) si el destino no es un video o se cancelo"""
        if not self.es_video(self.target_path):
            return None, None
        genero = self.genero_var.get()
        fps, maximo = self.opciones_analisis()
        personas, carpeta, _ = Personas.obtener(self.target_path, self.output_path_var.get(),
                                                lambda frame: self.swapper.detectar_caras(frame, genero), genero,
                                                progreso=progreso, cancelado=cancelado, fps=fps, maximo=maximo)
        return personas, carpeta

    def ver_personas(self):
        """Botón «Ver personas»: analiza el video destino (o usa lo guardado) y muestra quién es cada número"""
        if not self.es_video(self.target_path):
            messagebox.showinfo("Ver personas", "Primero elige un video destino.\n\nEn imágenes, las personas se "
                                "numeran de izquierda a derecha.")
            return
        if self.processing:
            messagebox.showwarning("Aviso", "Espera a que termine el procesamiento actual")
            return
        self.processing = True
        self.process_button.state(['disabled'])

        def progreso(hechos, total):
            self.root.after(0, lambda: self.status_label.config(text=f"Analizando personas: frame {hechos}/~{total}"))

        def tarea():
            try:
                personas, carpeta = self.analizar_personas(progreso)
                self.root.after(0, lambda: self._mostrar_personas(personas, carpeta))
            except Exception as e:
                error = str(e)
                self.root.after(0, lambda: messagebox.showerror("Ver personas", error))
            finally:
                self.processing = False
                self.root.after(0, lambda: self.process_button.state(['!disabled']))
                self.root.after(0, lambda: self.status_label.config(text="Listo"))

        threading.Thread(target=tarea, daemon=True).start()

    def _mostrar_personas(self, personas, carpeta, maximo=12):
        if not personas:
            messagebox.showinfo("Ver personas", "No se encontró ninguna cara en el video")
            return
        ventana = tk.Toplevel(self.root)
        ventana.title("Personas en el video")
        ventana.transient(self.root)
        ttk.Label(ventana, text="Numeradas de la que más aparece (0) a la que menos. En «Índices Destino» usa "
                                "estos números.", justify=tk.LEFT).pack(padx=15, pady=(10, 5), anchor="w")
        grilla = ttk.Frame(ventana)
        grilla.pack(padx=15, pady=5)
        ventana.imagenes = []  # referencia a las PhotoImage para que no se borren
        for i, p in enumerate(personas.lista[:maximo]):
            celda = ttk.Frame(grilla)
            celda.grid(row=i // 4, column=i % 4, padx=8, pady=8)
            if p["miniatura"] is not None:
                foto = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(cv2.resize(p["miniatura"], (140, 140)),
                                                                       cv2.COLOR_BGR2RGB)))
                ventana.imagenes.append(foto)
                ttk.Label(celda, image=foto).pack()
            desde = f"{int(p['desde'] // 60)}:{int(p['desde'] % 60):02d}"
            ttk.Label(celda, text=f"Persona {i}", font=('Helvetica', 10, 'bold')).pack()
            ttk.Label(celda, text=f"aparece {p['apariciones']} veces · desde {desde}", foreground="gray").pack()
        if len(personas) > maximo:
            ttk.Label(ventana, text=f"... y {len(personas) - maximo} más con pocas apariciones (extras)",
                      foreground="gray").pack(padx=15, anchor="w")
        ttk.Label(ventana, text=f"Miniaturas guardadas en {carpeta}", foreground="gray", wraplength=600,
                  justify=tk.LEFT).pack(padx=15, pady=(5, 0), anchor="w")
        ttk.Button(ventana, text="Cerrar", command=ventana.destroy).pack(pady=10)

    def registrar_identidades(self, modo, frame, frame_number=None, fps=None):
        """Llamar en orden de frames. Devuelve las caras detectadas (o None si ya están todas registradas)"""
        registro = modo["registro"]
        if registro.completo():
            return None

        caras = self.swapper.detectar_caras(frame, self.genero_var.get())
        for identidad, cara in registro.registrar(caras):
            momento = ""
            if frame_number is not None and fps:
                seg = frame_number / fps
                momento = f" en {int(seg // 60)}:{int(seg % 60):02d}"
            fuente = modo["mapa"].get(identidad)
            destino = f"recibe la fuente {fuente}" if fuente is not None else "no se cambia"
            mensaje = f"Identidad destino {identidad} registrada{momento}: {destino}"
            print(mensaje)
            self.status_label.config(text=mensaje)
        return caras

    def cambiar_por_identidad(self, modo, frame, caras_dst="detectar", refs=None, devolver_mascara=False):
        """
        Registra (si hace falta) y cambia las caras de un frame emparejando por identidad.
        Con devolver_mascara=True devuelve (frame, mascara de la zona cambiada)
        """
        if isinstance(caras_dst, str):
            caras_dst = self.registrar_identidades(modo, frame)
        if refs is None:
            refs = modo["registro"].copia_refs()
        opciones = dict(self.opciones_proceso or self.opciones_cambio())
        del opciones["index_src"], opciones["index_dst"]
        return self.swapper.cambiar_Faces_por_identidad(
            frame_dst=frame,
            caras_src=modo["caras_src"],
            mapa=modo["mapa"],
            refs=refs,
            caras_dst=caras_dst,
            devolver_mascara=devolver_mascara,
            **opciones
        )

    def start_processing(self):
        if not self.source_path or not self.target_path:
            messagebox.showwarning("Warning", "Please select both source and target files")
            return
        
        if self.processing:
            return

        if self.zip_var.get():
            if not zip_disponible():
                messagebox.showerror("ZIP", "Para guardar en ZIP con contraseña falta instalar pyzipper:\n\n"
                                     "pip install pyzipper")
                return
            if not self.zip_contrasena_var.get():
                messagebox.showwarning("ZIP", "Escribe la contraseña del ZIP (o desmarca «Guardar en ZIP»)")
                return

        try:
            self.validar_indices_fuente(self.obtener_caras_src())
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.reanudar_video = False
        if self.es_video(self.target_path) and not self.preguntar_proceso_pendiente():
            return
        if self.es_carpeta(self.target_path):
            destino = carpeta_salida(self.target_path, self.output_path_var.get())
            if os.path.normcase(destino) == os.path.normcase(os.path.abspath(self.target_path)):
                messagebox.showerror("Error", "La carpeta de salida no puede ser la misma carpeta de las imágenes")
                return
            if (os.path.isdir(destino) and os.listdir(destino) and not messagebox.askyesno(
                    "Carpeta de salida", f"Ya existe {destino} con archivos.\n\n"
                                         "Las imágenes con el mismo nombre se reemplazan. ¿Continuar?")):
                return

        self.prueba_fps = None
        self.process_button.state(['disabled'])
        self.processing = True
        self.opciones_proceso = self.opciones_cambio()
        self.fragmento_proceso = self.fragmento_activo()
        threading.Thread(target=self.process_files, daemon=True).start()

    def probar_video(self):
        """Video de prueba: el fragmento elegido (o los primeros SEGUNDOS_PRUEBA s) a pocos fps, con las opciones
        actuales y el audio sincronizado. Se guarda como <nombre>_prueba.mp4; el resultado final no se toca"""
        if self.processing:
            return
        if not self.source_path or not self.es_video(self.target_path):
            messagebox.showinfo("Video de prueba", "Elige una fuente y un video destino")
            return
        try:
            fps = float(self.prueba_fps_var.get().replace(",", "."))
            if not 1 <= fps <= 30:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Video de prueba", "Los fps de la prueba deben ser un número de 1 a 30")
            return
        try:
            self.validar_indices_fuente(self.obtener_caras_src())
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        fragmento = self.fragmento_activo()
        if fragmento is None:
            datos = info_video(self.target_path)
            if datos is None:
                messagebox.showerror("Video de prueba", "No se pudo abrir el video")
                return
            fragmento = (0, min(datos["frames"], int(round(SEGUNDOS_PRUEBA * datos["fps"]))))
        self.prueba_fps = fps
        self.reanudar_video = False
        self.process_button.state(['disabled'])
        self.processing = True
        self.opciones_proceso = self.opciones_cambio()
        self.fragmento_proceso = fragmento
        threading.Thread(target=self.process_files, daemon=True).start()

    @staticmethod
    def es_video(ruta):
        return bool(ruta) and ruta.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))

    def ruta_salida_video(self):
        sufijo = "_prueba" if self.prueba_fps else ""
        return os.path.abspath(os.path.join(self.output_path_var.get(), f"{self.output_filename_var.get()}{sufijo}.mp4"))

    def firma_video(self):
        """Opciones que definen el resultado: si cambian, un video cortado no se puede continuar"""
        return {
            "fuente": self.source_path,
            "indices_fuente": self.src_indices_var.get(),
            "indices_destino": self.dst_indices_var.get(),
            "asignacion": self.asignacion_var.get(),
            "genero": self.genero_var.get(),
            "oclusion": self.var_oclucion.get(),
            "filtro": self.combofiltro.get(),
            "ojos": self.var_ojos.get(),
            "boca": self.var_boca.get(),
            "nariz": self.var_nariz.get(),
            "mascara": self.mascara_var.get(),
            "color": self.color_var.get(),
            "maquillaje": round(self.maquillaje_var.get(), 2),
            "mejorar": round(self.fuerza_mejora(), 2),
            "mejorar_minimo": self.gpen_minimo() if self.fuerza_mejora() else None,
            "codec": self.codec_var.get(),
            # solo si esta marcada: los videos empezados antes de esta opcion se siguen pudiendo continuar
            **({"frente": True} if self.var_frente.get() else {}),
            **({"mejillas": True} if self.var_mejillas.get() else {}),
            # 2: caras inclinadas sin saltos (mezcla gradual entre 10 y 20 grados); 3: caras volteadas
            "deteccion": 3,
            **({"sin_inclinadas": True} if not self.caras_inclinadas_var.get() else {}),
            # identidad: las personas salen del analisis previo (antes, por orden de aparicion): no mezclar
            **({"personas": "analisis", "personas_fps": self.personas_fps_var.get(),
                "personas_max": self.personas_max_var.get()} if self.asignacion_var.get() == "identidad" else {}),
            # el modo de oclusion "debil" es el de antes: los videos empezados antes de esta opcion siguen con el
            **({"oclusion_modo": self.oclusion_modo_var.get()}
               if self.var_oclucion.get() and self.oclusion_modo_var.get() != "debil" else {}),
            # antes el filtro sin oclusion no se aplicaba: un video empezado asi no se continua con el filtro nuevo
            **({"nitidez_sin_oclusion": True}
               if not self.var_oclucion.get() and self.combofiltro.get() != "None" else {}),
        }

    def preguntar_proceso_pendiente(self):
        """Si esta salida tiene un video sin terminar, pregunta si continuar. False = no procesar"""
        trabajo = Trabajo_Video(self.ruta_salida_video(), self.target_path, self.firma_video(), self.contenedor_var.get(),
                                self.fragmento_activo())
        pendiente = trabajo.pendiente()
        if pendiente is None:
            return True
        avance = f"{pendiente['hechos']} de {pendiente['total']} frames"
        if pendiente["coincide"]:
            respuesta = messagebox.askyesnocancel(
                "Proceso sin terminar",
                f"Este video tiene un proceso sin terminar ({avance}).\n\n"
                "Sí: continuar desde donde quedó\nNo: empezar de cero (se borra lo hecho)\nCancelar: no hacer nada")
            if respuesta is None:
                return False
            if respuesta:
                self.reanudar_video = True
                return True
        elif not messagebox.askokcancel(
                "Proceso sin terminar",
                f"Este video tiene un proceso sin terminar ({avance}), pero no se puede continuar porque "
                f"{pendiente['motivo']}.\n\nPara continuarlo, vuelve a poner las mismas opciones. "
                "¿Borrar lo hecho y empezar de cero?"):
            return False
        trabajo.descartar()
        return True

    def detener_proceso(self):
        """Pide parar el video: se terminan los frames en curso y lo hecho queda guardado para continuar"""
        self.detener_video = True
        self.detener_button.state(['disabled'])
        self.status_label.config(text="Deteniendo... (terminando los frames en curso)")

    def process_files(self):
        try:
            # Caras fuente (foto o modelo .npy), una sola vez para todo el proceso
            caras_src = self.obtener_caras_src()

            # Process target file
            if self.es_video(self.target_path):
                self.process_video(caras_src)
            elif self.es_carpeta(self.target_path):
                self.process_folder(caras_src)
            else:
                self.process_image(caras_src)
                
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.processing = False
            self.prueba_fps = None
            self.opciones_proceso = None
            self.process_button.state(['!disabled'])
            self.status_label.config(text="Processing complete")
            self.progress['value'] = 0

    def process_video(self, caras_src):

        """
         Procesa el video con core/Proceso_Video.py: los frames van directo a ffmpeg (sin imagenes en disco y
        con solo unos pocos en memoria) y lo hecho se guarda en "<salida>.partes" para poder continuar si se
        corta o se pulsa Detener.
        """

        self.detener_video = False
        modo = None
        if self.asignacion_var.get() == "identidad":
            # Analisis previo: quienes aparecen (0 = la que mas aparece), con varias tomas de cada uno
            self.root.after(0, lambda: self.detener_button.state(['!disabled']))

            def progreso_personas(hechos, total):
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Analizando personas: frame {hechos}/~{total}"))

            personas, carpeta = self.analizar_personas(progreso_personas, lambda: self.detener_video)
            if personas is None:
                messagebox.showinfo("Detenido", "Se detuvo el análisis de personas; no se procesó nada.")
                return
            print("Personas en el video:\n" + "\n".join(personas.resumen(carpeta)))
            modo = self.crear_modo_identidad(caras_src, personas)
        self.modo_identidad = modo
        captura = cv2.VideoCapture(self.target_path)
        if not captura.isOpened():
            raise Exception("No se pudo abrir el video")
        fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
        captura.release()

        final_output = self.ruta_salida_video()
        os.makedirs(os.path.dirname(final_output), exist_ok=True)

        def preparar(frame, numero):
            if modo is not None:
                # El registro de identidades se hace aquí, en orden de lectura; los hilos solo usan una copia
                caras_dst = self.registrar_identidades(modo, frame, numero, fps)
                refs = modo["registro"].copia_refs()
                return lambda: self.cambiar_por_identidad(modo, frame, caras_dst, refs)
            return lambda: self.process_frame(caras_src, frame)

        def progreso(hechos, total):
            self.root.after(0, self._mostrar_progreso, hechos, total)

        fragmento = self.fragmento_proceso
        self.root.after(0, lambda: self.detener_button.state(['!disabled']))
        try:
            resultado = procesar_video(
                self.target_path, final_output, preparar,
                hilos=max(1, int(self.thread_var.get())),
                contenedor=self.contenedor_var.get(),
                codec=self.codec_var.get(),
                firma=self.firma_video(),
                reanudar=self.reanudar_video,
                estado=modo["registro"] if modo is not None else None,
                progreso=progreso,
                cancelado=lambda: self.detener_video,
                fragmento=fragmento,
                avisar=lambda texto: (print(texto), self.root.after(0, lambda: self.status_label.config(text=texto))),
                fps_prueba=self.prueba_fps,
            )
        finally:
            self.root.after(0, lambda: self.detener_button.state(['disabled']))

        if resultado == "cancelado":
            messagebox.showinfo("Detenido", "Proceso detenido. Lo procesado quedó guardado.\n\n"
                                "Para continuar, vuelve a procesar el mismo video con la misma salida y las "
                                "mismas opciones, y elige «Sí» cuando pregunte.")
        elif self.prueba_fps:
            self.root.after(0, lambda: self.progress.configure(value=100))
            if messagebox.askyesno("Video de prueba", f"Video de prueba guardado en:\n{final_output}\n\n¿Abrirlo?"):
                try:
                    os.startfile(final_output)
                except (AttributeError, OSError) as e:  # os.startfile solo existe en Windows
                    messagebox.showinfo("Video de prueba", f"Ábrelo desde la carpeta de salida ({e})")
        else:
            self.root.after(0, lambda: self.progress.configure(value=100))
            if self.zip_var.get():
                try:
                    texto = self._guardar_zip([final_output], ruta_zip(final_output))
                except Exception as e:
                    messagebox.showerror("ZIP", f"El video quedó en {final_output}, pero no se pudo guardar en "
                                         f"el ZIP:\n\n{e}")
                    return
                messagebox.showinfo("Éxito", f"Video guardado {texto}")
            else:
                messagebox.showinfo("Éxito", f"Video guardado en {final_output}")

    def _mostrar_progreso(self, hechos, total):
        if total > 0:
            self.progress['value'] = min(100, hechos / total * 100)
        self.status_label.config(text=f"Procesando frames: {hechos}/{total}")

    def process_folder(self, caras_src):
        """
         Procesa cada imagen de la carpeta destino con las mismas opciones y la guarda con su mismo nombre en
        <Carpeta de Salida>/<carpeta>_output. En modo identidad las personas se reconocen entre imágenes.
        """
        imagenes = imagenes_carpeta(self.target_path)
        destino = carpeta_salida(self.target_path, self.output_path_var.get())
        modo = self.crear_modo_identidad(caras_src) if self.asignacion_var.get() == "identidad" else None

        def preparar(imagen, numero):
            if modo is not None:
                caras_dst = self.registrar_identidades(modo, imagen)
                refs = modo["registro"].copia_refs()
                return lambda: self.cambiar_por_identidad(modo, imagen, caras_dst, refs)
            return lambda: self.process_frame(caras_src, imagen)

        def progreso(hechas, total):
            self.root.after(0, self._mostrar_progreso_carpeta, hechas, total)

        self.detener_video = False
        self.root.after(0, lambda: self.detener_button.state(['!disabled']))
        try:
            guardadas, errores, detenido = procesar_carpeta(
                imagenes, destino, preparar, hilos=max(1, int(self.thread_var.get())),
                progreso=progreso, cancelado=lambda: self.detener_video)
        finally:
            self.root.after(0, lambda: self.detener_button.state(['disabled']))

        mensaje = f"{guardadas} de {len(imagenes)} imagen(es) guardada(s) en {destino}"
        if errores:
            mensaje += f"\n\nNo se pudieron procesar {len(errores)}:\n" + "\n".join(
                f"• {nombre}: {error}" for nombre, error in errores[:10])
            if len(errores) > 10:
                mensaje += f"\n... y {len(errores) - 10} más (ver la consola)"
            for nombre, error in errores:
                print(f"Error en {nombre}: {error}")
        if detenido:
            messagebox.showinfo("Detenido", "Proceso detenido.\n\n" + mensaje +
                                ("\n\nNo se creó el ZIP porque el proceso no terminó." if self.zip_var.get() else ""))
            return
        self.root.after(0, lambda: self.progress.configure(value=100))
        if self.zip_var.get() and guardadas:
            fallidas = {nombre for nombre, _ in errores}
            archivos = [os.path.join(destino, os.path.basename(r)) for r in imagenes
                        if os.path.basename(r) not in fallidas]
            try:
                texto = self._guardar_zip(archivos, ruta_zip(destino, es_carpeta=True),
                                          carpeta_en_zip=os.path.basename(destino))
                if not self.zip_conservar_var.get():
                    try:
                        os.rmdir(destino)  # solo si quedo vacia: no se borra nada que ya estuviera ahi
                    except OSError:
                        pass
                mensaje = mensaje.replace(f"guardada(s) en {destino}", f"guardada(s) {texto}")
            except Exception as e:
                mensaje += f"\n\nNo se pudo guardar en el ZIP (las imágenes quedaron sueltas): {e}"
                errores = errores or [("ZIP", str(e))]
        (messagebox.showwarning if errores else messagebox.showinfo)("Carpeta procesada", mensaje)

    def _mostrar_progreso_carpeta(self, hechas, total):
        if total > 0:
            self.progress['value'] = min(100, hechas / total * 100)
        self.status_label.config(text=f"Procesando imágenes: {hechas}/{total}")

    def process_image(self, caras_src):
        target_img = leer(self.target_path)
        if target_img is None:
            raise Exception("No se pudo leer la imagen destino")
            
        if self.asignacion_var.get() == "identidad":
            result = self.cambiar_por_identidad(self.crear_modo_identidad(caras_src), target_img)
        else:
            result = self.process_frame(caras_src, target_img)
        
        # Use custom filename
        output_filename = f"{self.output_filename_var.get()}.jpg"
        output_path = os.path.join(self.output_path_var.get(), output_filename)
        
        guardar(output_path, result)
        
        self.status_label.config(text="Procesamiento de imagen completado")
        self.progress['value'] = 100
        if self.zip_var.get():
            try:
                texto = self._guardar_zip([output_path], ruta_zip(output_path))
            except Exception as e:
                messagebox.showerror("ZIP", f"La imagen quedó en {output_path}, pero no se pudo guardar en el ZIP:"
                                     f"\n\n{e}")
                return
            messagebox.showinfo("Éxito", f"Imagen guardada {texto}")
        else:
            messagebox.showinfo("Éxito", f"Imagen guardada en {output_path}")

    def show_preview(self):
        if self.preview_window:
            self.preview_window.destroy()
        
        if not self.source_path:
            messagebox.showwarning("Aviso", "Primero selecciona una imagen fuente")
            return
            
        if self.target_path:
            if self.es_video(self.target_path):
                self.preview_window = PreviewWindow(self.root, self, video_path=self.target_path)
            elif self.es_carpeta(self.target_path):
                # Con una carpeta, la barra de abajo pasa de una imagen a otra (también con Anterior/Siguiente o las flechas)
                imagenes = imagenes_carpeta(self.target_path)
                if imagenes:
                    self.preview_window = PreviewWindow(self.root, self, imagenes=imagenes)
            else:
                img = leer(self.target_path)
                if img is not None:
                    self.preview_window = PreviewWindow(self.root, self, image=img)

if __name__ == "__main__":
    root = tkdnd.Tk()
    app = FaceSwapApp(root)
    root.mainloop() 