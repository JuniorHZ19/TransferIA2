import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
from PIL import Image, ImageTk
import threading
import subprocess
import sys
from core.Face_Swaper import Swapper_Faces
import tkinterdnd2 as tkdnd
from queue import Queue
import concurrent.futures
import queue
import shutil


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
    def __init__(self, master, text, callback, **kwargs):
        super().__init__(master, text=text, **kwargs)
        self.callback = callback
        
        # Container frame with fixed size
        self.container = ttk.Frame(self, height=300, width=300)
        self.container.pack(padx=10, pady=10)
        self.container.pack_propagate(False)  # Mantiene el tamaño fijo
        
        # Preview label
        self.preview = ttk.Label(self.container, text="Arrastra aquí o haz clic para seleccionar")
        self.preview.place(relx=0.5, rely=0.5, anchor="center")  # Centrar el texto
        
        # Bind events
        self.preview.drop_target_register(tkdnd.DND_FILES)
        self.preview.dnd_bind('<<Drop>>', self.on_drop)
        self.preview.bind('<Button-1>', self.on_click)
        
    def on_drop(self, event):
        file_path = event.data
        if file_path.startswith('{'):
            file_path = file_path[1:-1]
        self.callback(file_path)
        
    def on_click(self, event):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image/Video files", "*.jpg *.jpeg *.png *.mp4 *.avi *.gif *.mkv")])
        if file_path:
            self.callback(file_path)
            
    def update_preview(self, img):
        if img is None:
            self.preview.configure(text="Arrastra aquí o haz clic para seleccionar")
            self.preview.configure(image='')
            self.preview.place(relx=0.5, rely=0.5, anchor="center")
            return
            
        # Get container size
        container_width = self.container.winfo_width()
        container_height = self.container.winfo_height()
        
        if container_width == 0:  # Si aún no está renderizado, usar el tamaño fijo
            container_width = 300
            container_height = 300
            
        # Convert and resize for preview
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Calculate scaling factor to fit container while maintaining aspect ratio
        scale = min(container_width/w, container_height/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        img = cv2.resize(img, (new_w, new_h))
        
        # Create photo image
        photo = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.preview.configure(image=photo, text="")
        self.preview.image = photo
        
        # Center in container
        self.preview.place(relx=0.5, rely=0.5, anchor="center")

class PreviewWindow(tk.Toplevel):
    def __init__(self, parent, app, video_path=None, image=None):
        super().__init__(parent)
        self.title("Preview - Transfer")
        self.geometry("900x800")
        self.app = app
        
        # Main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Preview area - Solo resultado
        self.preview_frame = ttk.Frame(main_frame)
        self.preview_frame.pack(fill=tk.BOTH, expand=True)
        
        # Preview label con fondo gris claro para ver los límites
        self.preview_label = ttk.Label(self.preview_frame, background='#f0f0f0')
        self.preview_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Bind resize event
        self.bind('<Configure>', self.on_resize)
        
        # Controls frame para video
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            controls = ttk.Frame(main_frame)
            controls.pack(fill=tk.X, pady=(10, 0))
            
            # Frame counter
            self.frame_label = ttk.Label(controls, text="0/0")
            self.frame_label.pack(side=tk.RIGHT, padx=10)
            
            # Slider mejorado para video
            slider_frame = ttk.Frame(controls)
            slider_frame.pack(fill=tk.X, padx=10, expand=True)
            
            self.frame_slider = ttk.Scale(
                slider_frame,
                from_=0,
                to=self.total_frames-1,
                orient=tk.HORIZONTAL
            )
            self.frame_slider.pack(fill=tk.X, expand=True)
            
            # Configurar eventos del slider
            self.frame_slider.bind("<Button-1>", self.on_slider_click)
            self.frame_slider.bind("<B1-Motion>", self.on_slider_drag)
            self.frame_slider.bind("<ButtonRelease-1>", self.on_slider_release)
            
            self.update_frame()
        elif image is not None:
            self.show_image(image)
            
    def on_resize(self, event):
        # Solo actualizar si tenemos una imagen
        if hasattr(self, 'current_img') and self.current_img is not None:
            self.show_preview(self.current_img)

    def show_preview(self, img):
        if img is None:
            return
            
        self.current_img = img  # Guardar la imagen actual para resize
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get frame size (excluding padding)
        frame_width = self.preview_frame.winfo_width() - 40  # 20px padding each side
        frame_height = self.preview_frame.winfo_height() - 40
        
        if frame_width <= 0 or frame_height <= 0:  # Si aún no está renderizado
            frame_width = 800
            frame_height = 600
        
        # Get image size
        h, w = img.shape[:2]
        
        # Calculate scaling factor
        scale = min(frame_width/w, frame_height/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        img = cv2.resize(img, (new_w, new_h))
        
        # Create and update photo
        photo = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo

    def on_slider_click(self, event):
        # Calcular la posición relativa del clic
        slider = event.widget
        width = slider.winfo_width()
        x = event.x
        value = (x / width) * self.total_frames
        self.frame_slider.set(value)
        self.update_frame()

    def on_slider_drag(self, event):
        self.update_frame()

    def on_slider_release(self, event):
        self.update_frame()

    def update_frame(self, *args):
        if hasattr(self, 'cap'):
            frame_no = int(self.frame_slider.get())
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = self.cap.read()
            if ret:
                self.frame_label.config(text=f"{frame_no}/{self.total_frames-1}")
                self.show_image(frame)

    def show_image(self, img):
        if img is None or not hasattr(self.app, 'source_path'):
            return
            
        try:
            # Leer la imagen fuente
            source_img = cv2.imread(self.app.source_path)
            if source_img is None:
                return
                
            # Procesar la imagen con cambiar_Faces
            result_img = self.app.swapper.cambiar_Faces(
                frame_dst=img.copy(),
                frame_src=source_img,
                index_dst=self.app.dst_indices_var.get(),
                index_src=self.app.src_indices_var.get(),
                mode=self.app.combofiltro.get(),
                oclution=self.app.var_oclucion.get()
            )
            
            # Mostrar solo el resultado
            self.show_preview(result_img)
            
        except Exception as e:
            print(f"Error en preview: {str(e)}")
            # Si hay error, mostrar la imagen original
            self.show_preview(img)

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

class FaceSwapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Transfer-IA Studio")
        self.root.geometry("1200x900")
        
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
        
        # Variables para procesamiento multihilo
        self.frame_queue = Queue()
        self.result_queue = Queue()
        self.processing_active = False
        self.current_source_img = None
        
        # Create directories if they don't exist
        for dir_path in [self.output_dir, self.models_dir, "temp_frames"]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Check for required models
        self.check_required_models()
        
        # Initialize models
        self.setup_models()
        
        # Create main frames
        self.create_frames()

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

        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title = ttk.Label(main_container, text="Transfer-IA STUDIO", style='Header.TLabel')
        title.pack(pady=(0, 20))
        
        # Create top frame for drag & drop areas
        top_frame = ttk.Frame(main_container)
        top_frame.pack(fill=tk.X, pady=10)
        
        # Source frame (left)
        self.source_frame = DropFrame(top_frame, text="Imagen Fuente", callback=self.set_source)
        self.source_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Target frame (right)
        self.target_frame = DropFrame(top_frame, text="Imagen/Video Destino", callback=self.set_target)
        self.target_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Preview button
        preview_frame = ttk.Frame(main_container)
        preview_frame.pack(fill=tk.X, pady=10)
        ModernButton(preview_frame, text="Abrir Preview", 
                    command=self.show_preview).pack(pady=5)
        
        # Controls frame
        controls_frame = ttk.LabelFrame(main_container, text="Configuración")
        controls_frame.pack(fill=tk.X, pady=20)
        
        # Face indices frame
        indices_frame = ttk.Frame(controls_frame)
        indices_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Source indices
        src_frame = ttk.Frame(indices_frame)
        src_frame.pack(side=tk.LEFT, padx=15)
        ttk.Label(src_frame, text="Índices Fuente:").pack(side=tk.LEFT)
        self.src_indices_var = tk.StringVar(value="0")
        ttk.Entry(src_frame, textvariable=self.src_indices_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # Target indices
        dst_frame = ttk.Frame(indices_frame)
        dst_frame.pack(side=tk.LEFT, padx=25)
        ttk.Label(dst_frame, text="Índices Destino:").pack(side=tk.LEFT)
        self.dst_indices_var = tk.StringVar(value="0")
        ttk.Entry(dst_frame, textvariable=self.dst_indices_var, width=10).pack(side=tk.LEFT, padx=5)

        #Filtro :
        filtro_frame = ttk.Frame(indices_frame)
        filtro_frame.pack(pady=5, anchor="w")

        label = ttk.Label(filtro_frame, text="Filtro:")
        label.pack(side="left", padx=(0, 5))

        self.combofiltro = ttk.Combobox(filtro_frame, values=["None", "natural", "strong"], state="disabled")
        self.combofiltro.set("None")
        self.combofiltro.pack(side="left")

        #Oclucion:
        self.var_oclucion = tk.BooleanVar(value=False)
        self.check_oclucion = ttk.Checkbutton(filtro_frame, text="Activar Oclucion", variable=self.var_oclucion,command=lambda: self.combofiltro.configure(state="disabled" if self.var_oclucion.get()!=True else "readonly"))
        self.check_oclucion.pack(side="left", padx=(10, 0))
          
        # Output configuration frame
        output_frame = ttk.LabelFrame(main_container, text="Configuración de Salida")
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
        
        # Configuración de procesamiento
        process_frame = ttk.LabelFrame(main_container, text="Configuración de Procesamiento")
        process_frame.pack(fill=tk.X, pady=10)
        
        # Control de hilos
        thread_frame = ttk.Frame(process_frame)
        thread_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(thread_frame, text="Número de Hilos:").pack(side=tk.LEFT, padx=5)
        self.thread_var = tk.StringVar(value="2")
        thread_spinbox = ttk.Spinbox(thread_frame, from_=1, to=4, 
                                   textvariable=self.thread_var, width=5)
        thread_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Process button
        self.process_button = ModernButton(main_container, text="Iniciar Procesamiento", 
                                         command=self.start_processing)
        self.process_button.pack(pady=10)
        
        # Progress frame
        progress_frame = ttk.Frame(main_container)
        progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress.pack(fill=tk.X, padx=10)
        
        self.status_label = ttk.Label(progress_frame, text="Listo")
        self.status_label.pack(pady=5)

    def set_source(self, path):
        self.source_path = path
        self.update_preview(path, self.source_frame)
        
    def set_target(self, path):
        self.target_path = path
        self.update_preview(path, self.target_frame)

    def select_output_dir(self):
        dir_path = filedialog.askdirectory(initialdir=self.output_dir)
        if dir_path:
            self.output_dir = dir_path
            self.output_path_var.set(dir_path)

    def update_preview(self, path, drop_frame):
        try:
            if path.lower().endswith(('.mp4', '.avi','.mkv')):
                cap = cv2.VideoCapture(path)
                ret, img = cap.read()
                cap.release()
                if not ret:
                    raise Exception("No se pudo leer el video")
            else:
                img = cv2.imread(path)
                if img is None:
                    raise Exception("No se pudo leer la imagen")
            
            drop_frame.update_preview(img)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            drop_frame.update_preview(None)  # Reset preview on error

    def process_frame(self, source_img, frame):
      
        try:
            # Usar el método cambiar_Faces directamente
            frame_fake = self.swapper.cambiar_Faces(
                frame_dst=frame,
                frame_src=source_img,
                index_dst=self.dst_indices_var.get(),
                index_src=self.src_indices_var.get(),
                mode=self.combofiltro.get(),
                oclution=self.var_oclucion.get()
            )
            
            return frame_fake
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame

    def process_frame_thread(self):
        while self.processing_active:
            try:
                frame_data = self.frame_queue.get(timeout=1)
                if frame_data is None:
                    break
                    
                frame_number, frame = frame_data
                result_frame = self.process_frame(self.current_source_img, frame)
                self.result_queue.put((frame_number, result_frame))
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error en hilo de procesamiento: {str(e)}")
                self.frame_queue.task_done()

    def start_processing(self):
        if not self.source_path or not self.target_path:
            messagebox.showwarning("Warning", "Please select both source and target files")
            return
        
        if self.processing:
            return
        
        self.process_button.state(['disabled'])
        self.processing = True
        threading.Thread(target=self.process_files, daemon=True).start()

    def process_files(self):
        try:
            # Load source image
            source_img = cv2.imread(self.source_path)
            if source_img is None:
                raise Exception("Could not read source image")
            
            # Process target file
            if self.target_path.lower().endswith(('.mp4', '.avi','.mkv')):
                self.process_video(source_img)
            else:
                self.process_image(source_img)
                
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.processing = False
            self.process_button.state(['!disabled'])
            self.status_label.config(text="Processing complete")
            self.progress['value'] = 0

    def process_video(self, source_img):
        self.current_source_img = source_img
        cap = cv2.VideoCapture(self.target_path)
        if not cap.isOpened():
            raise Exception("No se pudo abrir el video")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Prepare output paths
        output_filename = f"{self.output_filename_var.get()}.mp4"
        temp_video = os.path.join(self.output_dir, "temp_video.mp4")
        final_output = os.path.join(self.output_path_var.get(), output_filename)
        
        # Create temporary directory for frames
        temp_dir =os.path.join(self.output_dir, "/temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Iniciar procesamiento multihilo
            self.processing_active = True
            num_threads = int(self.thread_var.get())
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Iniciar hilos de procesamiento
                futures = [executor.submit(self.process_frame_thread) for _ in range(num_threads)]
                
                # Leer y encolar frames
                frame_count = 0
                frames_read = 0
                processed_frames = {}
                total_processed = 0
                
                while frames_read < total_frames:
                    # Mantener un buffer de frames en la cola
                    while frames_read < total_frames and self.frame_queue.qsize() < num_threads * 2:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        self.frame_queue.put((frames_read, frame))
                        frames_read += 1
                        
                        # Actualizar progreso de lectura
                        self.progress['value'] = (frames_read / total_frames) * 30
                        self.status_label.config(text=f"Leyendo frames: {frames_read}/{total_frames}")
                        self.root.update_idletasks()
                    
                    # Procesar resultados mientras se leen más frames
                    try:
                        frame_number, result_frame = self.result_queue.get(timeout=0.1)
                        processed_frames[frame_number] = result_frame
                        total_processed += 1
                        
                        # Guardar frames en orden
                        next_frame = 0
                        while next_frame in processed_frames:
                            cv2.imwrite(f"{temp_dir}/frame_{next_frame:06d}.jpg", 
                                      processed_frames.pop(next_frame))
                            next_frame += 1
                            frame_count += 1
                        
                        # Actualizar progreso de procesamiento
                        progress = 30 + (total_processed / total_frames) * 45
                        self.progress['value'] = progress
                        self.status_label.config(text=f"Procesando frames: {total_processed}/{total_frames}")
                        self.root.update_idletasks()
                        
                    except queue.Empty:
                        continue
                
                # Señalar fin de frames a los hilos
                for _ in range(num_threads):
                    self.frame_queue.put(None)
                
                # Esperar y procesar los frames restantes
                while total_processed < frames_read:
                    try:
                        frame_number, result_frame = self.result_queue.get(timeout=1)
                        processed_frames[frame_number] = result_frame
                        total_processed += 1
                        
                        # Guardar frames restantes en orden
                        next_frame = frame_count
                        while next_frame in processed_frames:
                            cv2.imwrite(f"{temp_dir}/frame_{next_frame:06d}.jpg", 
                                      processed_frames.pop(next_frame))
                            next_frame += 1
                            frame_count += 1
                        
                        progress = 30 + (total_processed / total_frames) * 45
                        self.progress['value'] = progress
                        self.status_label.config(text=f"Procesando frames restantes: {total_processed}/{frames_read}")
                        self.root.update_idletasks()
                        
                    except queue.Empty:
                        continue
            
            # Desactivar procesamiento
            self.processing_active = False
            
            # Generar video con FFmpeg
            self.status_label.config(text="Generando video...")
            self.progress['value'] = 75
            
            # First FFmpeg command: Create video from frames without audio
            ffmpeg_video_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', f'{temp_dir}/frame_%06d.jpg',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-vsync', 'vfr',  # Variable frame rate
                '-start_number', '0',  # Ensure frame numbering starts at 0
                temp_video
            ]
            
            subprocess.run(ffmpeg_video_cmd, check=True, capture_output=True)
            
            # Second FFmpeg command: Copy audio from original to new video
            self.status_label.config(text="Copiando audio original...")
            self.progress['value'] = 90
            
            ffmpeg_audio_cmd = [
                'ffmpeg', '-y',
                '-i', temp_video,
                '-i', self.target_path,
                '-c:v', 'copy',
                '-c:a', 'copy',  # Copy audio codec instead of re-encoding
                '-map', '0:v:0',
                '-map', '1:a:0?',
                '-shortest',  # Match duration to shortest stream
                '-async', '1',  # Audio sync method
                '-vsync', '1',  # Video sync method
                #'-itsoffset', '-1',  # Offset audio by -1 second
                final_output
            ]
            
            subprocess.run(ffmpeg_audio_cmd, check=True, capture_output=True)
            
            self.progress['value'] = 100
            messagebox.showinfo("Éxito", f"Video guardado en {final_output}")
            
        except Exception as e:
            self.processing_active = False
            messagebox.showerror("Error", str(e))
        finally:
            cap.release()
            # Cleanup
            if os.path.exists(temp_video):
                try:
                    os.remove(temp_video)
                except:
                    pass
            for file in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, file))
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass

    def process_image(self, source_img):
        target_img = cv2.imread(self.target_path)
        if target_img is None:
            raise Exception("No se pudo leer la imagen destino")
            
        result = self.process_frame(source_img, target_img)
        
        # Use custom filename
        output_filename = f"{self.output_filename_var.get()}.jpg"
        output_path = os.path.join(self.output_path_var.get(), output_filename)
        
        cv2.imwrite(output_path, result)
        
        self.status_label.config(text="Procesamiento de imagen completado")
        self.progress['value'] = 100
        messagebox.showinfo("Éxito", f"Imagen guardada en {output_path}")

    def show_preview(self):
        if self.preview_window:
            self.preview_window.destroy()
        
        if not self.source_path:
            messagebox.showwarning("Aviso", "Primero selecciona una imagen fuente")
            return
            
        if self.target_path:
            if self.target_path.lower().endswith(('.mp4', '.avi','.mkv')):
                self.preview_window = PreviewWindow(self.root, self, video_path=self.target_path)
            else:
                img = cv2.imread(self.target_path)
                if img is not None:
                    self.preview_window = PreviewWindow(self.root, self, image=img)

if __name__ == "__main__":
    root = tkdnd.Tk()
    app = FaceSwapApp(root)
    root.mainloop() 