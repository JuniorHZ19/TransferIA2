# Transfer-IA

Transfer-IA es una aplicación de cambio facial que utiliza inteligencia artificial para transferencia facial de manera realista.

## Características

- Interfaz gráfica moderna y fácil de usar
- Soporte para imágenes y videos
- Vista previa que se actualiza sola al cambiar la configuración, con el original y el resultado lado a lado o con una línea deslizable de antes/después, y la zona cambiada (máscara) opcional
- Guardar el resultado en un ZIP con contraseña (AES-256), sin dejar el video suelto en la carpeta (ver [Guardar en ZIP con contraseña](#guardar-en-zip-con-contraseña---zip))
- Configuraciones guardadas: todas las opciones con un nombre, en JSON, para la interfaz y la CLI (ver [Configuraciones guardadas](#configuraciones-guardadas))
- Procesamiento por lotes: una carpeta de imágenes como destino da otra carpeta con los resultados (ver [Carpeta de imágenes como destino](#carpeta-de-imágenes-como-destino))
- Mejora de calidad facial integrada
- Arrastrar y soltar archivos
- Barra de progreso para seguimiento del proceso
- Detección de caras inclinadas a la izquierda, a la derecha y boca abajo
- Asignación de caras por identidad: cada persona del video conserva su cara aunque cambie de posición
- Filtro por género de las caras destino (todos, mujer u hombre)
- Opción para conservar los ojos del destino (`--eyes`)
- Opción para conservar la boca del destino (`--mouth`)
- Opción para conservar la nariz del destino (`--nose`)
- Opción para quitar el cerquillo de la fuente: la frente queda la del destino (`--remove-bangs`)
- Opción para conservar las mejillas y los pómulos del destino (`--mejillas`)
- Conservar el maquillaje o la pintura de cara del destino (`--makeup`)
- Mejorador de caras GPEN opcional para más nitidez en piel, ojos y dientes (`--enhance`)
- Videos que se pueden detener y continuar después, aunque se corte la luz (`--resume`); sin imágenes temporales en disco y con poca RAM
- Procesar solo un fragmento del video, elegido con deslizadores de inicio y fin y un reproductor para verlo (`--start` / `--end`)
- Corrección de color de la cara cambiada para igualar la piel del destino (`--color`)
- Zona de la cara configurable para más parecido a la fuente (`--face-mask`)
- Modelos de cara (`.npy`): la identidad promediada de varias fotos o un collage, reutilizable como origen; se pueden usar varios a la vez y combinar los de una misma persona

## 📋 Requisitos del Sistema

- Windows 10 o superior
- Python 3.8 o superior
- GPU compatible con CUDA (recomendado para mejor rendimiento)
- Memoria RAM: 8GB mínimo (16GB recomendado)
- Espacio en disco: 2GB mínimo para modelos e instalación

## 🔧 Instalación

1. Clone o descargue este repositorio

2. Instale las dependencias:
```bash
pip install -r requirements.txt
```

3. Asegúrese de que todas las dependencias estén correctamente instaladas:
- numpy >= 1.21.0
- opencv-python >= 4.5.0
- pillow >= 8.0.0
- insightface >= 0.7.0
- onnxruntime-gpu >= 1.8.0
- tkinter (generalmente viene con Python)

4. Modelos en la carpeta `models/` (no se suben a git):

| Archivo | Para qué | Obligatorio |
|---|---|---|
| `inswapper128.onnx` | Cambio de cara | Sí |
| `face_occluder.onnx` | Oclusión (`--occlusion`) | Sí |
| `gpen_bfr_256.onnx` | Mejorador de caras (`--enhance`) | Solo si se usa `--enhance`. Descarga: [gpen_bfr_256.onnx](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gpen_bfr_256.onnx) (76 MB, CRC32 `2189ffb5`) |

## 📖 Uso

### Google Colab

`TransferIa_Colab.ipynb` usa TransferIa en Colab con formularios, como la interfaz (por debajo corre la CLI). Ábrelo en [colab.research.google.com](https://colab.research.google.com) → **Archivo → Subir cuaderno** (o desde GitHub: pestaña **GitHub** → `JuniorHZ19/TransferIA2`), elige **Entorno de ejecución → GPU T4** y corre las celdas en orden:

| Celda | Qué hace |
|---|---|
| 1. Instalar | Descarga el programa de GitHub (la rama elegida) e instala las librerías. Avisa si no hay GPU. |
| 2. Modelos | Trae los modelos desde un zip o una carpeta de tu Drive, o subiendo el zip (`modelsTransferIa.zip`). |
| 3. Subir archivos | Opcional: sube fotos, videos o `.npy` y te da la ruta para los formularios. También sirven rutas de Drive. |
| 4. Crear modelo de cara | Con fotos y/o videos (y sus fps); por defecto lo guarda en tu Drive para otras sesiones. |
| 5. Ver personas | Para el modo identidad: analiza el video y muestra la miniatura y el número de cada persona. |
| 6. Procesar | Todas las opciones: caras, índices, pegado, partes del destino, maquillaje, oclusión y su modo, nitidez, GPEN, video (hilos, códec, calidad, fragmento, continuar si quedó a medias), ZIP con contraseña y configuraciones guardadas. |
| 7. Ver y descargar | Muestra el resultado, lo descarga a tu PC o lo copia a tu Drive. |

Notas:

- El cuaderno usa la versión del programa **que está en GitHub**: los cambios hechos en la PC tienen que estar subidos (`git push`).
- En Colab conviene el códec `h264_nvenc` (usa la GPU) y unos 4 hilos.
- La contraseña del ZIP se pide al correr la celda 6 (no se escribe en el formulario ni se guarda).
- Si se corta la sesión con un video a medias, al volver a correr la celda 6 con la misma salida continúa desde donde quedó (si la carpeta `output` sigue existiendo; al reiniciar Colab se borra, salvo que la salida esté en Drive).

### Interfaz gráfica

Ejecute la aplicación sin argumentos:
```bash
python app.py
```

La ventana se abre maximizada y tiene dos columnas: a la izquierda la **fuente** (arriba) y el **destino** (abajo), cuyos recuadros crecen con la ventana y muestran la imagen completa lo más grande posible, sin recortarla ni deformarla; a la derecha, **Abrir Preview**, la configuración, la salida, el procesamiento y el progreso. Se puede soltar o hacer clic en cualquier parte de cada recuadro.

En la interfaz principal:
   - Seleccione o arrastre una imagen de origen (rostro a transferir), o un modelo de cara `.npy` (ver [Modelos de cara](#modelos-de-cara-npy))
   - Seleccione o arrastre una imagen/video de destino, o una carpeta de imágenes (arrastrándola o con el botón **Carpeta de imágenes...**; ver [Carpeta de imágenes como destino](#carpeta-de-imágenes-como-destino))
   - Seleccione la carpeta de salida para los resultados
   - Opcional, **Guardar en ZIP con contraseña** (en "Configuración de Salida"): el resultado se guarda en un `.zip` cifrado y no queda suelto en la carpeta (ver [Guardar en ZIP con contraseña](#guardar-en-zip-con-contraseña---zip))
   - Opcional, **Configuración guardada** (debajo de "Abrir Preview"): elige una configuración guardada para poner todas las opciones de una vez, o guarda las actuales con **Guardar como...** (ver [Configuraciones guardadas](#configuraciones-guardadas))
   - Opcional, en "Configuración". Cada fila es una función, con su nombre a la izquierda, en el orden en que se aplican:

     | Fila | Opciones |
     |---|---|
     | **Caras** | **Asignación**: `posicion` (izquierda a derecha, como siempre) o `identidad` (ver [Asignación por identidad](#asignación-por-identidad---asignacion-identidad)). **Género destino**: `todos`, `mujer` u `hombre`. **Análisis de personas** (solo con `identidad`): frames por segundo (`1`) y máximo de frames (`300`) del análisis previo del video |
     | **Índices** | **Fuente** y **Destino**: qué cara fuente va en qué cara destino (`0`, `0,1`...). Una misma fuente puede ir a varias caras repitiendo su número: Fuente `0,0` y Destino `0,1` pone la cara 0 en las dos. En modo identidad con video, Destino es el número de persona (0 = la que más aparece); el botón **Ver personas...** muestra quién es cada una (ver [Asignación por identidad](#asignación-por-identidad---asignacion-identidad)) |
     | **Pegado** | **Máscara**: `normal`, `amplia` (por defecto) o `maxima`, cuánta zona de la cara se cambia (ver [Zona de la cara](#zona-de-la-cara---face-mask)). **Corrección de color**: `ninguna` (por defecto), `suave` o `completa`, iguala el color de la cara cambiada a la piel del destino (ver [Corrección de color](#corrección-de-color---color)) |
     | **Del destino** | **Ojos / Boca / Nariz**: cambia la cara pero deja esas partes originales del destino (ver [ojos](#conservar-ojos-del-destino), [boca](#conservar-boca-del-destino---mouth) y [nariz](#conservar-nariz-del-destino---nose)). **Frente (quitar cerquillo)**: la frente queda la del destino, para cuando la fuente tiene cerquillo (ver [Quitar el cerquillo](#quitar-el-cerquillo-de-la-fuente---remove-bangs)). **Mejillas y pómulos**: las mejillas quedan las del destino (ver [Conservar mejillas](#conservar-mejillas-y-pómulos-del-destino---mejillas)). **Maquillaje**: deslizador de 0 (desactivado) a 1; devuelve el maquillaje o la pintura de cara del destino (ver [Conservar maquillaje](#conservar-maquillaje---makeup)) |
     | **Oclusión** | **Activar**: deja el pelo, las manos y los objetos delante de la cara. **Modo**: `debil`, `adaptable` (por defecto) o `fuerte`, cuánta cara puede quedar sobre lo que la tapa (ver [Modo de la oclusión](#modo-de-la-oclusión---oclusion-modo)) |
     | **Nitidez** | **Filtro**: `None`, `natural`, `strong` o `adaptativo` (recomendado). Funciona con o sin la oclusión (ver [Nitidez de la cara](#nitidez-de-la-cara---nitidez)) |
     | **Mejora** | **Mejorar cara (GPEN)** + **Fuerza**: pasa la cara cambiada por el mejorador GPEN para más nitidez; la fuerza va de 0.1 a 1 (0.8 por defecto); tarda unos 0.25 s más por cara. **Cara mínima** + unidad (`px` o `% imagen`): caras más chicas no se mejoran; `96 px` por defecto, o proporcional a la imagen, como `25 % imagen` (ver [Mejorador de caras](#mejorador-de-caras-gpen---enhance)) |

   - Opcional, debajo de la imagen de origen:
     - **Elegir modelos...**: marca uno o varios modelos de cara de `src/modelos_caras/` y ordénalos; el orden es su índice fuente (0, 1, 2...). Con muchos modelos la lista tiene barra de desplazamiento (y rueda del mouse), así la ventana no crece más que la pantalla. Cada modelo tiene un botón **Eliminar** (ver [Eliminar un modelo](#eliminar-un-modelo))
     - **Crear modelo de cara**: crea un modelo con varias fotos o un collage de la misma persona (también se puede arrastrar varias fotos o una carpeta a la zona de origen)
     - **Desde video**: crea un modelo con los frames de uno o varios videos de la persona; se elige cuántos frames por segundo sacar (también se puede arrastrar el video a la zona de origen). Ver [Crear un modelo desde un video](#crear-un-modelo-desde-un-video)
     - **Combinar modelos**: une varios modelos de la misma persona en uno nuevo (ver [Combinar modelos](#combinar-modelos-de-la-misma-persona))
     - **Quitar**: vacía el origen
   - Opcional, en "Configuración de Procesamiento" (solo videos):
     - **Número de Hilos** (primera fila): 2 es lo mejor en CPU; con GPU NVIDIA, de 2 a 4
     - **Video temporal** (segunda fila): `mkv` (por defecto) o `mp4frag` (MP4 fragmentado); los dos permiten continuar si se corta (ver [Videos: detener y continuar](#videos-detener-y-continuar))
     - **Códec** (segunda fila): `libx264` (CPU, por defecto), `h264_nvenc` (tarjeta NVIDIA) o `h264_amf` (tarjeta AMD)
     - **Solo un fragmento** + **Elegir fragmento...** (tercera fila): procesa solo el pedazo del video entre un inicio y un fin elegidos con deslizadores, viendo el video y cuánto dura el fragmento (ver [Procesar solo un fragmento](#procesar-solo-un-fragmento-del-video))
   - Opcional, **Abrir Preview** para ver cómo queda antes de procesar (en videos, con una barra para elegir el frame). Arriba se elige la **Vista**:
     - **Deslizar** (por defecto, la que está seleccionada al abrir): las dos en la misma imagen, separadas por una línea; a la izquierda el original y a la derecha el resultado. Haga clic o arrastre sobre la imagen para mover la línea y ver el antes y el después de cada zona (útil para revisar el maquillaje)
     - **Lado a lado**: el original a la izquierda y el resultado a la derecha
     - **Solo resultado**: el resultado a tamaño completo

     Al lado está **Ver máscara** (opcional, desmarcada por defecto): pinta de verde sobre el resultado la zona que de verdad se cambió, con una línea donde el cambio llega a la mitad. Más intenso = más cambio; donde no hay verde, quedó el destino tal cual. Junta todo lo que decide esa zona: la **Máscara** de pegado (`normal`, `amplia`, `maxima`), lo que retoca **GPEN**, lo que devuelve la **Oclusión** (pelo, manos, objetos) y las partes que se quedan del destino (**Ojos / Boca / Nariz / Frente**). Sirve para comparar `amplia` con `maxima`, o ver si la oclusión está tapando de más. Se calcula solo con la casilla marcada; al procesar no se calcula y el resultado sale idéntico.

     **Copiar la imagen** (para pegarla en Paint, Word, WhatsApp, etc.): **clic derecho** sobre la imagen del preview abre un menú (el clic izquierdo sigue moviendo la línea de **Deslizar**):
     - **Copiar resultado** (o **Ctrl+C** con la ventana de preview activa): el resultado a **tamaño completo**, sin la máscara verde aunque **Ver máscara** esté marcada
     - **Copiar original**: el frame o la imagen de destino a tamaño completo, sin cambios
     - **Copiar lo que se ve (vista actual)**: la imagen exactamente como está en la ventana, con los rótulos, la línea de **Deslizar** o las dos de **Lado a lado**, y la máscara si está marcada. Sale al tamaño de la ventana (más chica que el original)

     Luego se pega con **Ctrl+V** en el otro programa. Arriba a la derecha aparece por 3 s el aviso, por ejemplo "Resultado copiado (640x797), pégalo con Ctrl+V". Mientras todavía se está procesando el primer frame, avisa "Todavía no hay resultado para copiar". Solo en Windows (usa el portapapeles de Windows directamente, sin instalar nada más). Copia el frame que se está viendo en el preview; para guardar como archivo, procese normalmente.

     **El preview se actualiza solo**: con la ventana de preview abierta, cualquier cambio en "Configuración" (caras, índices, pegado, partes del destino, maquillaje, oclusión y filtro, GPEN y su fuerza) vuelve a procesar el frame que se está viendo. También al cambiar la fuente; al cambiar el destino, el preview se vuelve a abrir con el nuevo. Detalles:
     - Espera 0.4 s a que se deje de mover un deslizador o de escribir los índices, y procesa una sola vez con el último valor (mover el deslizador de maquillaje 10 pasos seguidos = 1 procesado)
     - Procesa en segundo plano: la ventana no se congela y arriba a la derecha dice **Procesando...** mientras tanto. Se sigue viendo el resultado anterior hasta que llega el nuevo
     - Si mientras procesa llega otro cambio (u otro frame de la barra del video), al terminar descarta ese resultado y procesa solo el más nuevo
     - Cambiar de vista o el tamaño de la ventana no vuelve a procesar el frame
     - Si el frame no se pudo procesar (por ejemplo, índices mal escritos), arriba aparece **Error: ...**, el resultado se rotula "error, ver consola" y se muestra el original

     Al pulsar **Iniciar Procesamiento** las opciones quedan fijas hasta terminar: se puede seguir probando cambios en el preview mientras se procesa un video, sin que el video cambie de configuración a la mitad
   - Haga clic en **Iniciar Procesamiento** para empezar. En videos, **Detener** para el proceso guardando lo hecho; al volver a procesar el mismo video con la misma salida y las mismas opciones, pregunta si continuar desde donde quedó

En modo `identidad`, la vista previa registra a las personas en el orden en que se van viendo los frames, así que conviene recorrer el video desde el principio.

### Línea de comandos

El CLI está separado de la interfaz y se ejecuta desde `cli.py`. Tiene dos comandos:

| Comando | Para qué sirve |
|---|---|
| `process` | Cambia la cara en una imagen o un video. |
| `crear-modelo` | Crea un modelo de cara (`.npy`) a partir de varias fotos, un collage o videos de la misma persona (`--fotos` y/o `--video`; de los videos saca frames a `--fps` por segundo), para usarlo después como origen en `process` (ver [Modelos de cara](#modelos-de-cara-npy)). Con `--extraer-caras` solo guarda las caras por persona para revisarlas antes (ver [Revisar las caras antes de crear el modelo](#revisar-las-caras-antes-de-crear-el-modelo)). |
| `combinar-modelos` | Une varios modelos de la misma persona en uno solo, como si se hubiera creado con todas sus fotos juntas (ver [Combinar modelos](#combinar-modelos-de-la-misma-persona)). |

Ejemplo de `process` con una imagen:

```bash
python cli.py process --source "src/rostro.jpg" --target "src/imagen.jpg" --output "output/resultado.jpg"
```

Para procesar un video y conservar su audio:

```bash
python cli.py process --source "src/rostro.jpg" --target "src/video.mp4" --output "output/resultado.mp4"
```

Para procesar todas las imágenes de una carpeta (el resultado queda en `output/fotos_output/`):

```bash
python cli.py process --source "src/rostro.jpg" --target "src/fotos" --output "output"
```

#### Carpeta de imágenes como destino

El destino puede ser una carpeta: cada imagen (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) se procesa con la misma fuente y las mismas opciones, y se guarda **con su mismo nombre** en una carpeta nueva que se llama como la original más `_output`, dentro de la carpeta de salida:

```text
src/fotos/                 output/fotos_output/
├── ana.jpg        ──►     ├── ana.jpg
├── luis.png       ──►     ├── luis.png
└── grupo.jpg      ──►     └── grupo.jpg
```

- **En la interfaz:** arrastra la carpeta a la zona **Imagen/Video Destino** o usa el botón **Carpeta de imágenes...** debajo de ella. Se ve la primera imagen y, al lado del botón, cuántas imágenes hay y el nombre de la carpeta donde se guardarán. La salida usa la **Carpeta de Salida**; **Nombre del Archivo** no se usa (cada imagen conserva su nombre). **Abrir Preview** permite revisar el resultado imagen por imagen antes de procesar todas: abajo hay una barra (como la de los videos), los botones **< Anterior** / **Siguiente >** y las flechas ← → del teclado; al lado se ve la posición y el nombre (`3/25 · ana.jpg`).
- **En la CLI:** `--target` es la carpeta y `--output` la carpeta donde se crea `<nombre>_output`.
- Solo se toman las imágenes de la carpeta, no las de sus subcarpetas; los archivos se procesan por orden de nombre.
- Se procesan en paralelo según el **Número de Hilos** (`--threads`), con pocas imágenes en memoria a la vez.
- Si una imagen no se puede leer o falla, se avisa al final y se siguen las demás. Una imagen sin caras se guarda igual que estaba.
- Si la carpeta de salida ya tiene archivos, la interfaz pregunta antes de seguir (los que tengan el mismo nombre se reemplazan); en la CLI hace falta `--overwrite`.
- **Detener** para después de las imágenes en curso; las ya guardadas quedan.
- En modo `identidad`, las personas se reconocen entre una imagen y otra (la persona 0 es la primera que aparece, en orden de nombre de archivo). En la CLI, las miniaturas de cada identidad van a `output/fotos_output_identidades/`.
- `--start` y `--end` (fragmento) son solo para videos y se ignoran.

Una sola cara fuente para varias caras del destino (repitiendo el índice fuente):

```bash
python cli.py process --source "src/rostro.jpg" --target "src/foto_grupo.jpg" --output "output/resultado.jpg" --source-index 0,0,0 --target-index 0,1,2
```

Si en algún frame no está una de esas caras (por ejemplo, en el video a veces hay una sola persona), ese par se salta y las demás se cambian igual.

Opciones de `process`:

```text
--source PATH              imagen con la cara de origen, o modelo(s) de cara .npy separados por comas
                           (la cara de origen se detecta una sola vez, no en cada frame del video)
--target PATH              imagen o video destino, o una carpeta con imagenes
--output PATH              archivo de salida; con una carpeta en --target, la carpeta donde se crea
                           <nombre>_output con las imagenes procesadas
--models-dir PATH          carpeta de modelos (por defecto: models)
--source-index INDICES     indices de origen, por ejemplo 0,1; se pueden repetir para usar una misma
                           fuente en varias caras (--source-index 0,0 --target-index 0,1)
--target-index INDICES     indices de destino, por ejemplo 0,1
                           (en modo identidad con video: numero de persona, 0 = la que mas aparece)
--asignacion MODO          posicion (por defecto) o identidad
--genero GENERO            todos (por defecto), mujer u hombre
--sin-inclinadas           deteccion original: sin buscar caras inclinadas o boca abajo
--personas-fps N           con identidad en video: frames por segundo del analisis de personas (1)
--personas-max N           con identidad en video: maximo de frames del analisis, repartidos en todo
                           el video (300)
--occlusion                activa el modelo de oclusion (pelo, manos y objetos delante de la cara)
--oclusion-modo MODO       con --occlusion: debil (como antes), adaptable (por defecto) o fuerte
                           (ver Modo de la oclusion)
--nitidez MODO             nitidez de la cara pegada, con o sin --occlusion: None, natural, strong o
                           adaptativo (iguala el detalle y el grano del destino; recomendado).
                           --occlusion-mode es el nombre anterior y sigue funcionando
--eyes                     cambia el rostro pero conserva los ojos del destino
                           (para mantener la mirada, el parpadeo y el maquillaje originales)
--mouth                    cambia el rostro pero conserva la boca del destino
                           (labios, dientes y lengua originales; útil cuando la persona habla o come)
--enhance [FUERZA]         mejora la cara cambiada con GPEN (models/gpen_bfr_256.onnx): más nitidez
                           en piel, ojos y dientes; FUERZA de 0 a 1 (solo --enhance = 0.8; menos =
                           más natural y más parecido con la fuente); unos 0.25 s más por cara en CPU
--enhance-min-size TAMANO  con --enhance: caras más chicas no se mejoran. En píxeles de ancho (96, por
                           defecto) o en % del lado más corto de la imagen (25%); 0 = todas
--nose                     cambia el rostro pero conserva la nariz del destino
--remove-bangs             quita el cerquillo de la fuente: la frente queda la del destino
--mejillas                 cambia el rostro pero conserva las mejillas y los pómulos del destino
--makeup FUERZA            0 (por defecto, desactivado) a 1: conserva el maquillaje o la pintura de cara
                           del destino (Halloween, catrina, pintura artística); más fuerza = más
                           maquillaje y menos parecido con la fuente. Recomendado: 0.7 a 1
--color MODO               ninguna (por defecto), suave o completa: iguala el color de la cara
                           cambiada a la piel del destino para que no se note el "parche"
--face-mask ZONA           normal, amplia (por defecto) o maxima: cuánta zona de la cara se pega
                           (más zona = más parecido a la fuente; normal = como antes de esta opción)
--threads NUMBER           numero de hilos (2 es lo mejor en CPU; con GPU NVIDIA, de 2 a 4)
--overwrite                reemplaza un archivo existente
--no-audio                 omite el audio del video original
--container FORMATO        video temporal mientras se procesa: mkv (por defecto) o mp4frag
                           (MP4 fragmentado); los dos permiten continuar si se corta
--video-codec CODEC        libx264 (CPU, por defecto), h264_nvenc (NVIDIA) o h264_amf (AMD)
--crf NUMERO               calidad del video: menor = mejor y más pesado; 18 (por defecto) se ve
                           prácticamente igual al original
--start TIEMPO             solo videos: procesa desde este momento (90, 1:30, 1:30.5 o 0:01:30);
                           el resultado es solo el fragmento, con su audio
--end TIEMPO               solo videos: procesa hasta este momento (ver Procesar solo un fragmento)
--prueba [FPS]             video de prueba: el fragmento o los primeros 10 s, a FPS (10 por defecto),
                           guardado como <salida>_prueba.mp4
--resume                   continúa un video que se cortó, desde donde quedó
--restart                  borra un proceso sin terminar de esa salida y empieza de cero
--verbose                  muestra el progreso por frame
--zip                      guarda el resultado en un .zip con contraseña (AES-256) y borra el archivo
                           suelto; la contraseña se pide al empezar, sin mostrarla
--zip-password CLAVE       contraseña del --zip sin preguntarla (queda en el historial: mejor no usarla)
--zip-conservar            con --zip, conserva tambien el archivo suelto
--config NOMBRE            usa una configuracion guardada (configuraciones/NOMBRE.json, o la ruta a un
                           .json); lo escrito en el comando tiene prioridad (ver Configuraciones guardadas)
--save-config NOMBRE       guarda las opciones de este comando como configuracion NOMBRE
```

Opciones de `crear-modelo`:

```text
--fotos PATH [PATH ...]    imágenes y/o carpetas con fotos de la misma persona
                           (un collage puede tener varias caras; se usan todas)
--output PATH              archivo .npy a crear; al lado se guardan una miniatura .jpg y un .json
--overwrite                reemplaza el modelo si ya existe
```

Ejemplo:

```bash
python cli.py crear-modelo --fotos "src/persona1/" --output "src/modelos_caras/persona1.npy"
```

Opciones de `combinar-modelos`:

```text
--modelos PATH [PATH ...]  modelos .npy de la misma persona (al menos dos)
--output PATH              modelo .npy a crear (nombre distinto a los que se combinan)
--overwrite                reemplaza el modelo de salida si ya existe
```

Ejemplo:

```bash
python cli.py combinar-modelos --modelos "src/modelos_caras/ana_1.npy" "src/modelos_caras/ana_2.npy" --output "src/modelos_caras/ana.npy"
```

Puedes consultar la ayuda completa con:

```bash
python cli.py process --help
python cli.py crear-modelo --help
python cli.py combinar-modelos --help
```

> Las opciones van después del comando (`process`, `crear-modelo` o `combinar-modelos`). Sin comando, el programa solo muestra la ayuda y no procesa nada.

### Caras inclinadas y boca abajo

Activado por defecto: casilla **Caras inclinadas** en la fila **Caras** de la interfaz. Para desactivarlo: desmarcarla, o `--sin-inclinadas` en la CLI (vuelve a la detección original de insightface).

- **Detección:** si no encuentra caras, o si alguna está inclinada más de 45°, vuelve a buscar con la imagen girada 90°, 180° y 270°. Si la misma cara aparece en varios giros, se usa la que quedó más derecha en su giro (ahí el detector ubica mejor los puntos).
- **Refinado:** las caras inclinadas se enderezan y se detectan otra vez para ubicar mejor ojos, nariz y boca. Entre 10° y 20° de inclinación se **mezclan de a poco** los puntos normales y los refinados, para que la cara pegada no salte de un frame a otro cuando la cabeza se mueve cerca de ese ángulo.
- **Máscara de oclusión (`--occlusion`):** la cara se endereza con sus 5 puntos clave antes de generar la máscara, y luego la máscara vuelve a su ángulo original.

Medido con una cara girada grado a grado (error = distancia de los puntos a donde deberían estar, en % de la distancia entre ojos):

| | Detección original (`--sin-inclinadas`) | Antes del arreglo | Ahora |
|---|---|---|---|
| Error medio, de −60° a +60° | 5,9 % | 1,7 % | **1,0 %** |
| Error con la cabeza a 40-60° | 11 % | 1 % | **0,9 %** |
| Mayor salto entre un grado y el siguiente | 1,9 % | 4,2 % (a los 27°) | **0,7 %** |
| Error medio, de −180° a +180° (boca abajo incluido) | 24 % | — | **1,0 %** |

El salto de 4,2 % se veía como la cara "temblando" cuando la cabeza oscilaba cerca de los 30° (el corte anterior). Los videos a medias empezados antes de este cambio no se pueden continuar (la cara se ubica distinto): hay que empezarlos de cero.

Ajustes en el código: `ANGULOS_DETECCION`, `REFINAR_DESDE` (10) y `REFINAR_COMPLETO` (20) en `core/Face_Analyzer.py`; `USAR_ALINEACION` en `core/xseg.py`.

#### Caras inclinadas y tapadas (celular, mano)

Con la nariz y la boca tapadas y la cabeza inclinada, el detector a veces ubicaba los ojos pero ponía la nariz y la boca del lado de la frente: la cara se pegaba **al revés**. Ahora, si una cara es dudosa, se lee también girada 180° y se elige entre las dos lecturas:

- **Cuándo es dudosa:** puntaje del detector menor a 0.72. En videos, además, si quedó con la cabeza hacia abajo o si giró más de 60° respecto del frame anterior (una cabeza no da media vuelta de un frame a otro).
- **En videos, memoria entre frames:** gana la lectura que sigue la dirección que tenía esa misma cara en los frames anteriores. El recuerdo solo vale si es de hace 0,5 s como mucho, la cara está casi en el mismo lugar y del mismo tamaño, y es la misma persona; así un cambio de plano no arrastra la dirección de otra toma. Funciona con varios hilos.
- **Sin memoria (fotos, carpetas, o el primer frame):** gana la lectura con la cabeza hacia arriba, salvo que la otra tenga claramente más puntaje. Con la cara de costado (75-105°) se queda la del detector salvo que la otra sea claramente mejor.

Medido con caras de IA (cara girada y medio tapada por un celular, una mano, un libro o un gorro):

| Prueba | Antes | Ahora |
|---|---|---|
| Video: cabeza a 45-75° y un celular que tapa nariz y boca 3 s | 29 de 90 frames al revés | **0** |
| Video: persona boca abajo con el celular | 5 de 90 | **0** |
| Video: cambios de plano (otra persona en el mismo lugar, la misma en otro lugar) | 39 de 81 | **0** |
| 525 fotos de −100° a +100° | 410 bien | **424 bien** (ninguna peor) |

No resuelve los ojos tapados (gorro, visera): sin ojos el detector no encuentra la cara. Los videos a medias empezados antes de este cambio no se pueden continuar: hay que empezarlos de cero.

Ajustes en el código: `VERIFICAR_VOLTEO_BAJO`, `MARGEN_VOLTEO`, `MARGEN_VOLTEO_VIDEO`, `SEGUNDOS_MEMORIA`, `PARECIDO_MEMORIA` y `GIRO_MEMORIA` en `core/Face_Analyzer.py`.

### Caras en primer plano

Cuando la cara llena casi todo el cuadro (selfies, primeros planos), el detector no la encontraba: al procesar se perdía desde que la cara ocupaba un 70-80 % del ancho, y al crear modelos (que detecta a 640 px) desde el 55 %. Ahora, si no encuentra ninguna cara, vuelve a buscar con la imagen dentro de un marco (borde repetido, con la imagen a la mitad o a un tercio de tamaño), así la cara entra en el rango del detector. Funciona siempre, con o sin «Caras inclinadas»; si la cara ya se detectaba, no cambia nada.

Además, cuando el recorte de la cara que usa inswapper se sale de la imagen (primer plano o cara en el borde), lo que quedaba afuera se rellenaba con negro y aparecía un **borde oscuro** en el resultado. Ahora se trabaja con un margen de borde repetido y se recorta al final.

| La cara ocupa del ancho del cuadro | Antes, al procesar | Antes, al crear modelos | Ahora (ambos) |
|---|---|---|---|
| 40 % | 5 de 5 | 5 de 5 | 5 de 5 |
| 55 % | 5 de 5 | 2 de 5 | 5 de 5 |
| 70 % | 3 de 5 | 0 de 5 | 5 de 5 |
| 80-140 % | 0-1 de 5 | 0 de 5 | 5 de 5 (error de los puntos 2-7 %) |

Probado con 5 caras generadas por IA. En el borde de un primer plano, el brillo pasó de un 25-50 % más oscuro que el original a igual que el original; con la cara dentro de la imagen el resultado es idéntico píxel a píxel al anterior. Ajuste en el código: `BORDES_PRIMER_PLANO` en `core/Face_Analyzer.py`.

### Mejorador de caras GPEN (`--enhance`)

inswapper genera la cara a 128 px y luego se amplía, así que pierde detalle de piel, ojos, pestañas y dientes. Con `--enhance` (o la casilla **Mejorar cara (GPEN)** en la interfaz, con su deslizador **Fuerza**) cada cara cambiada pasa por GPEN, un modelo que reconstruye ese detalle a 256 px. Es opcional y viene apagado.

Cómo funciona (`core/Mejorador.py`):

1. La cara se alinea con sus 5 puntos (ojos, nariz, comisuras) a un recorte de 256 × 256.
2. GPEN la reconstruye.
3. Vuelve a su lugar con una máscara de caja de bordes difuminados y se mezcla con la cara sin mejorar según la **fuerza** (0.8 por defecto, `MEZCLA` en `core/Mejorador.py`).
4. Se aplica justo después del cambio de cara y antes de la oclusión, así el pelo, las manos y los objetos que tapan la cara salen del destino sin pasar por el mejorador.

**Fuerza** (`--enhance 0.5`, o el deslizador en la interfaz): cuánto de la cara mejorada se usa. `--enhance` sin número usa 0.8.

| Fuerza | Parecido con la fuente | Detalle | Cuándo |
|---|---|---|---|
| sin `--enhance` | 0.910 | 19 | — |
| `0.3` | 0.908 | 21 | Retoque leve, casi no cambia la cara |
| `0.5` | 0.902 | 25 | Natural: más nitidez sin perder parecido |
| `0.8` (por defecto) | 0.888 | 35 | Equilibrio |
| `1` | 0.878 | 46 | Máxima nitidez; la piel puede quedar muy lisa |

(Promedio de 4 fotos con caras de 96 px o más.) Más fuerza = más nitidez pero un poco menos de parecido, porque GPEN también cambia levemente los rasgos. El tiempo es el mismo con cualquier fuerza.

- **Caras pequeñas (tamaño mínimo):** por defecto, las caras de menos de 96 px de ancho no se mejoran. inswapper ya las genera con más resolución de la que ocupan, y GPEN solo les agregaba detalle falso y bajaba el parecido. Cuando la cara mejorada se reduce, se suaviza antes para no dejar picos. Ese mínimo se puede cambiar con `--enhance-min-size` (en la interfaz, **Cara mínima** en la fila **Mejora**):

  | Valor | Qué hace |
  |---|---|
  | `96` (por defecto) | Píxeles de ancho de la cara. Igual que antes |
  | `25%` (en la interfaz: unidad **% imagen**) | Proporcional a la imagen: se mejoran las caras que miden al menos el 25 % del **lado más corto** de la imagen o del video. Sirve para imágenes chicas donde la cara ocupa casi todo aunque mida pocos píxeles: en una imagen de 250×250, `25%` = 62 px. En un video de 1920×1080, `25%` = 270 px, así que solo se mejoran caras grandes (primeros planos) |
  | `0` | Mejora todas las caras, de cualquier tamaño |

  Prueba con una imagen de 250×233 y una cara de 78 px (34 % del lado corto): con `96` y con `40%` GPEN no se aplica; con `25%` y con `0` sí (la zona cambiada pasa del 30 % al 43 % de la imagen, porque GPEN retoca también alrededor de la cara). En caras chicas GPEN agrega nitidez pero puede bajar un poco el parecido: conviene mirarlo en el preview con **Deslizar** y bajar la fuerza si hace falta.

  En la interfaz, al cambiar la unidad el valor pasa a uno razonable (`96` px o `25` %). Si lo escrito no es un número válido, se usa ese valor por defecto. Para continuar un video cortado, el mínimo debe ser el mismo.
- **Tiempo:** unos 0.25 s por cara en CPU (en las pruebas, entre 0.21 y 0.26 s). En un video de 1 minuto a 30 fps con una cara son unos 7 a 8 minutos más. El modelo se carga solo la primera vez que se usa.
- **Parecido:** GPEN cambia un poco los rasgos; en las pruebas el parecido con la fuente bajó entre 0.01 y 0.03 (por ejemplo, de 0.909 a 0.896).
- **Con la nitidez:** combínalo con `--nitidez adaptativo` (con o sin `--occlusion`). GPEN reconstruye la piel y los ojos, y el modo adaptativo iguala el detalle y el grano del resto de la imagen. En las pruebas, una cara de 214 px pasó de 30 de detalle (sin nada) a 44 (GPEN) y a 271 (GPEN + adaptativo), frente a 326 del destino.

Si falta el modelo, el programa avisa con el enlace de descarga. Está en `models/gpen_bfr_256.onnx` (versión ligera de 256 px; `GPEN-BFR-512.pth` es la versión pesada de 512 px en formato PyTorch y este programa no la usa).

```bash
python cli.py process --source "src/cara.jpg" --target "src/foto.jpg" --output "output/resultado.jpg" --enhance --occlusion --nitidez adaptativo --overwrite
python cli.py process --source "src/cara.jpg" --target "src/foto.jpg" --output "output/resultado.jpg" --enhance 0.5 --occlusion --nitidez adaptativo --overwrite
```

### Modo de la oclusión (`--oclusion-modo`)

Para que no se vea un corte en el borde, la máscara de la oclusión se difumina. Pero el difuminado no distingue lados: además de suavizar hacia la piel, "derramaba" cara nueva **sobre** el objeto que tapa la cara (una mano, un dedo, un micrófono, un mechón), que se veía con la cara semitransparente encima. El modo decide cuánto se permite eso:

| Modo | Cara sobre el objeto | Junto al objeto, del lado de la piel |
|---|---|---|
| `debil` | En el borde ~46 %, dentro 7-16 % (lo que se veía antes) | Igual que antes |
| `adaptable` (por defecto) | Dentro **0 %**; solo una transición corta en el borde exacto (~30 %) | **Igual que antes** |
| `fuerte` | **0 %**, hasta el borde | La cara nueva retrocede unos píxeles junto al objeto (puede asomar la cara original) |

(Medido con una máscara sintética: cara con una barra de 40 px delante.)

- **`adaptable`**: la cara nunca se pinta más de lo que el modelo de oclusión marca como cara; el difuminado solo puede suavizar hacia la piel. Es el recomendado: arregla la cara sobre los objetos sin cambiar nada más.
- **`fuerte`**: además contrae la máscara (lo que queda bajo 50 % pasa a 0). Para objetos finos (dedos, cables, pelo) donde con `adaptable` todavía se note el borde.
- **`debil`**: el comportamiento anterior, por si un video ya hecho debe salir igual.
- Si con cualquier modo se sigue viendo la cara sobre un objeto, puede que el modelo no lo detecte (objetos finos, transparentes o del color de la piel). Para saberlo: **Preview** → **Ver máscara**; si el objeto aparece pintado de verde, el modelo lo tomó como cara.
- Los videos a medias hechos antes de esta opción se continúan solo con `debil`; con otro modo hay que empezarlos de cero (para que no cambien a mitad del video).

```bash
python cli.py process --source "src/cara.jpg" --target "src/video.mp4" --output "output/resultado.mp4" --occlusion --oclusion-modo adaptable
```

### Nitidez de la cara (`--nitidez`)

La cara que sale de inswapper viene de un recorte de 128 px y luego se amplía, así que queda más blanda y más limpia que el resto de la foto o el video (se ve "de plástico"). Este filtro le devuelve nitidez. En la interfaz es la fila **Nitidez** (selector **Filtro**); en la CLI, `--nitidez` (antes `--occlusion-mode`, que sigue funcionando).

**Es independiente de la oclusión** (antes solo funcionaba con la oclusión activada):

| Oclusión | Dónde se aplica el filtro |
|---|---|
| Activada | Dentro de la máscara de oclusión: la cara nueva, sin el pelo, las manos ni los objetos que la tapan. Igual que antes. |
| Desactivada | Dentro de la zona donde se pegó la cara (la de **Máscara**: `normal`, `amplia` o `maxima`). El fondo del destino alrededor de la cara no se toca. |

Con la oclusión activada el resultado es el mismo de siempre. Un video empezado sin oclusión y con un filtro elegido (que antes no se aplicaba) no se puede continuar con la versión nueva, para que no cambie de estilo a la mitad.

| Modo | Qué hace |
|---|---|
| `None` (por defecto) | Sin filtro. |
| `natural` | Enfoque suave con fuerza fija. |
| `strong` | Enfoque fuerte con fuerza fija. |
| `adaptativo` (recomendado) | Mide el detalle y el grano de la cara original del destino y deja la cara nueva igual: ni más blanda ni más dura que el resto de la imagen. |

Cómo funciona:

- **Solo la luz:** se enfoca el canal L (LAB), no el color, para no crear halos de colores en cejas, labios y pelo.
- **Radio según el tamaño de la cara:** el detalle de la cara generada está a una escala de (ancho de la cara / 128) px; el enfoque (`L + fuerza × (L − desenfoque(L))`) usa ese radio, así no sobreafila caras pequeñas ni se queda corto en caras grandes.
- **`adaptativo`, fuerza automática:** calcula la fuerza para que el detalle de la cara llegue al 90 % del detalle del destino (tope 3). El cálculo es directo (el detalle crece como una parábola con la fuerza), no prueba valores.
- **`adaptativo`, grano:** la cara generada tiene la mitad de ruido que una foto real; se le agrega grano hasta el 85 % del ruido del destino. Es lo que más ayuda a que no parezca pegada.
- Solo se procesa el rectángulo de la cara, y el borde de la máscara se difumina en proporción al tamaño de la cara.

Pruebas con 5 fotos (detalle de la cara; el destino es la referencia):

| Cara | Destino | Sin filtro | `natural` | `strong` | `adaptativo` |
|---|---|---|---|---|---|
| 136 px | 91 | 18 | 41 | 67 | 72 |
| 204 px | 235 | 24 | 63 | 105 | 155 |
| 53 px | 406 | 215 | 345 | 510 | 354 |
| 214 px | 326 | 36 | 77 | 123 | 227 |
| 208 px | 34 | 5 | 16 | 25 | 26 |

El grano de la cara con `adaptativo` quedó entre 2.6 y 6.4 frente a 2.9 y 6.6 del destino. Cuesta unos 30 ms por cara. En caras grandes no llega al 100 % porque el enfoque no puede inventar detalle que el recorte de 128 px no tiene (para eso está el mejorador de caras, [`--enhance`](#mejorador-de-caras-gpen---enhance)).

Ajustes en `core/xseg.py`: `NITIDEZ_OBJETIVO` (0.9), `NITIDEZ_MAXIMA` (3.0), `GRANO_OBJETIVO` (0.85) y `NITIDEZ_FIJA` (fuerza de `natural` y `strong`).

Cambios respecto a la versión anterior del filtro:

- `natural` y `strong` usaban un filtro fijo de 3×3 píxeles sobre los tres colores: en caras pequeñas se pasaban mucho (`strong` dejaba hasta 8 veces el detalle del destino) y en caras grandes se quedaban cortos. Ahora siguen el tamaño de la cara y enfocan solo la luz, así que se ven más suaves que antes en caras pequeñas.
- Se quitó el desenfoque final que se aplicaba a todo el fotograma: ablandaba también el fondo y le quitaba parte de la nitidez a la cara.

```bash
python cli.py process --source "src/cara.jpg" --target "src/video.mp4" --output "output/resultado.mp4" --occlusion --nitidez adaptativo --overwrite
```

### Conservar ojos del destino

Con `--eyes` (o la casilla **Ojos** de la fila «Del destino» en la interfaz) se cambia el rostro completo, pero los ojos se toman del destino: mirada, parpadeo, pestañas y maquillaje. La máscara sale del contorno de cada ojo (106 puntos), se agranda un poco para incluir los párpados y se difumina el borde para que no se note el corte. Funciona en imagen y video, en modo posición e identidad.

**Velocidad** (vale para ojos, boca, nariz y frente): todo se calcula solo en el recuadro de la cara, no en el frame entero. Por cara, en 1080p: ojos 4 ms (antes 236), ojos + boca + nariz 5 ms (antes 192), frente 15 ms (antes 178); en 4K, 11 a 77 ms (antes 620 a 780). El resultado es idéntico píxel por píxel al de antes (comprobado en 4.200 casos: 5 resoluciones, caras en el borde, inclinadas, chicas y grandes, con y sin 106 puntos, todas las combinaciones de partes).

```bash
python cli.py process --source "src/cara.jpg" --target "src/video.mp4" --output "output/resultado.mp4" --eyes --overwrite
```

### Conservar boca del destino (`--mouth`)

Con `--mouth` (o la casilla **Boca** de la fila «Del destino» en la interfaz) se cambia el rostro completo, pero la boca se toma del destino: labios, dientes y lengua. Sirve cuando la persona habla, come o sonríe, porque inswapper suele deformar los dientes y el interior de la boca. La máscara sale del contorno de los labios (puntos 52 a 71 de los 106), se agranda un poco para tomar el borde de los labios y se difumina. Si la cara no trae los 106 puntos, se usa una elipse entre las comisuras.

Se puede combinar con `--eyes` para conservar ojos y boca a la vez. Ten en cuenta que cada parte que se conserva es una parte menos de la fuente, así que el parecido baja un poco (en las pruebas, de 0.90 a 0.83 con boca y a 0.81 con ojos y boca).

```bash
python cli.py process --source "src/cara.jpg" --target "src/video.mp4" --output "output/resultado.mp4" --mouth --eyes --overwrite
```

### Conservar nariz del destino (`--nose`)

Con `--nose` (o la casilla **Nariz** de la fila «Del destino» en la interfaz) se cambia el rostro completo, pero la nariz se toma del destino: el puente desde el entrecejo, la punta y las aletas. La máscara sale de los puntos 72 a 86 de los 106, se agranda un poco para tomar las aletas y se difumina. Si la cara no trae los 106 puntos, se usa una elipse desde el entrecejo hasta la punta de la nariz.

La nariz es uno de los rasgos que más identifican a una persona, así que conservarla baja el parecido con la fuente más que los ojos o la boca (en las pruebas, de 0.89 a 0.79). Úsala cuando la nariz del destino tenga algo que quieras mantener (pintura, piercing, forma muy característica). Se puede combinar con `--eyes` y `--mouth`, pero con las tres juntas queda poco de la fuente.

```bash
python cli.py process --source "src/cara.jpg" --target "src/video.mp4" --output "output/resultado.mp4" --nose --overwrite
```

### Conservar mejillas y pómulos del destino (`--mejillas`)

Con `--mejillas` (o la casilla **Mejillas y pómulos** de la fila «Del destino» en la interfaz) se cambia el rostro completo, pero las **mejillas y los pómulos se toman del destino**: la zona entre el párpado de abajo, el costado de la nariz, la comisura de la boca y el contorno de la cara. Sirve para mantener el rubor, el contorno o el iluminador del destino, o la forma de sus pómulos.

- La zona sale de los 106 puntos de la cara (contorno, ojos, nariz y boca) y se separa un poco de cada uno, para no tocar el párpado, las aletas de la nariz, los labios ni el borde de la cara. Sin los 106 puntos se usa una elipse por lado entre el ojo y la comisura.
- El borde se difumina más que en ojos, boca y nariz (la mejilla no tiene un borde natural donde esconder el corte). Si el tono de la cara cambiada y el del destino son muy distintos, combínala con `--color completa`.
- Baja poco el parecido con la fuente: en las pruebas, de 0.91 a 0.87 (con `--nose` baja de 0.89 a 0.79).

```bash
python cli.py process --source "src/cara.jpg" --target "src/video.mp4" --output "output/resultado.mp4" --mejillas --color completa --overwrite
```

Ajustes en el código: `MEJILLAS_OJO`, `MEJILLAS_NARIZ`, `MEJILLAS_BOCA`, `MEJILLAS_CONTORNO` y `MEJILLAS_DIFUMINADO` en `core/Face_Swaper.py`.

### Quitar el cerquillo de la fuente (`--remove-bangs`)

Con `--remove-bangs` (o la casilla **Frente (quitar cerquillo)** de la fila «Del destino» en la interfaz) se cambia el rostro, pero la **frente se toma del destino**, desde un poco arriba de las cejas hasta la raíz del pelo y de sien a sien.

**Para qué sirve:** cuando la foto fuente tiene cerquillo, inswapper no copia el pelo de la foto, pero al dibujar la cara de esa persona a veces "pinta" sobre la frente del destino sombras o mechones oscuros, como si el cerquillo estuviera ahí. Con esta opción esa zona vuelve a ser la frente limpia del destino.

**Por qué no lo arregla la oclusión:** la oclusión mira solo el destino (busca pelo, manos u objetos que tapan la cara en el video). Si el destino tiene la frente despejada, la oclusión deja pasar toda la frente de la cara cambiada, con las sombras del cerquillo incluidas. Si en cambio el **destino** tiene cerquillo, eso ya lo respeta la oclusión sin esta opción.

Cómo se arma la zona:
- Abajo empieza justo arriba del borde superior de las **cejas** (puntos 43 a 51 y 97 a 105 de los 106) y se va mezclando de a poco: las cejas quedan de la cara cambiada (pesan en el parecido) y a 0.32 distancias entre ojos por encima ya es frente del destino al 100 %
- A los lados se apaga suave en las sienes; hacia arriba llega hasta donde termina la zona pegada
- Sigue la inclinación de la cara (probado con la cara girada 25°). Sin los 106 puntos se estima la ceja con la posición de los ojos
- Se aplica después de la oclusión, junto con **Ojos / Boca / Nariz**. Con **Ver máscara** en el preview, la frente sale sin verde (no cambió)

**El tono de la frente se iguala solo.** La cara que dibuja inswapper casi nunca tiene el mismo tono que la frente del destino, y sin corregir se notaba una mancha o franja de otro color sobre las cejas. Ahora, antes de mezclarla:
- Se mide la diferencia de tono (luz y color, en LAB) entre la cara cambiada y el destino en una **franja angosta del entrecejo y el puente de la nariz**: es la piel de la cara nueva que queda justo debajo del borde, y ahí nunca hay cerquillo ni ojos. Se usa la mediana, así una mano u otra cosa que tape esa zona no la desvía
- Esa diferencia se suma a la frente del destino: cambia solo el tono; las arrugas, los poros y las sombras siguen siendo los de la frente del destino
- Se probó medir en las mejillas, pero quedaba peor que sin corregir (las mejillas tienen naturalmente otro tono que la frente)

Pruebas (salto de tono en el borde, en ΔE —diferencia de color en LAB: menos de ~2 a 3 casi no se ve—, descontando el que ya tiene la cara del destino): sin igualar 3.3 a 5.2, igualado **1.0 a 3.1**, en 6 casos (3 fotos, cara inclinada 25°, destino con otra luz y máscara `maxima`). La textura de la frente sigue siendo la del destino (correlación 1.00), fuera de la frente nada cambia y sin la opción el resultado es idéntico al de antes. Cuesta unos 18 ms más por cara en una imagen de 3840×1080.

Si aun así se nota algo, probar **Corrección de color** `suave` (iguala toda la cara al destino) o la máscara de pegado `normal`.

Para continuar un video cortado hay que mantener la misma elección (marcada o no). Los videos empezados antes de que existiera esta opción se pueden seguir continuando si se deja desmarcada.

```bash
python cli.py process --source "src/cara.jpg" --target "src/video.mp4" --output "output/resultado.mp4" --remove-bangs --overwrite
```

### Conservar maquillaje (`--makeup`)

inswapper dibuja la piel de la cara fuente desde cero, así que el maquillaje fuerte o la pintura de cara del destino (calavera, catrina, Halloween, pintura de fútbol, maquillaje artístico) desaparece. Con `--makeup FUERZA` (o el deslizador **Conservar maquillaje** en la interfaz) se le devuelve a la cara cambiada.

Cómo funciona (separación de frecuencias, en el recorte alineado de 128 px):

```
base    = desenfoque(cara original del destino)        colores grandes: el maquillaje
detalle = cara generada - desenfoque(cara generada)    detalle fino: los rasgos de la fuente
cara    = base + detalle                               la fuente "maquillada" como el destino
```

La fuerza (0 a 1) mezcla ese resultado con la cara generada sin tocar:

| Fuerza | Qué se ve |
|---|---|
| `0` (por defecto) | Desactivado: igual que antes de esta opción. |
| `0.5` | Maquillaje tenue, parecido alto. |
| `0.7` a `0.85` | Recomendado: el maquillaje se ve y la cara se sigue pareciendo a la fuente. |
| `1` | Todo el maquillaje que se puede recuperar. |

Siempre hay un equilibrio: cuanto más maquillaje se conserva, menos se parece a la fuente, porque la pintura tapa justo lo que distingue a una persona (tono, sombras de ojos y nariz). En la prueba con una cara pintada de calavera, la diferencia de maquillaje con el destino bajó de 30.7 a 17.2 con fuerza 0.85 y a 16.0 con 1, y el parecido con la fuente pasó de 0.90 a 0.79 y 0.77 (sigue siendo claramente la misma persona).

- Combínalo con `--eyes` para que los ojos traigan su sombra y su delineado, y con `--mouth` si la boca está pintada (por ejemplo, la boca cosida de una calavera).
- `--color` no sirve para pintura de cara: iguala el color promedio, y con blanco y negro el promedio es gris; `--makeup` es el que conserva el dibujo.
- Con `--face-mask normal` y maquillaje, la cara se pega con la máscara propia del programa (como con `--color`). Se aplica después de `--color` y antes de la oclusión y de `--eyes` / `--mouth` / `--nose` / `--remove-bangs` / `--mejillas`.
- El radio del desenfoque se ajusta en `core/Face_Swaper.py` con `SIGMA_MAQUILLAJE` (0.03 = unos 4 px del recorte). Más alto = más parecido y menos maquillaje.

```bash
python cli.py process --source "src/cara.jpg" --target "src/halloween.jpg" --output "output/resultado.jpg" --makeup 0.85 --eyes --mouth --overwrite
```

### Corrección de color (`--color`)

Iguala el color de la cara generada al de la piel original del destino antes de pegarla, para que no se note un tono o una luz distinta. En la interfaz es el selector **Corrección de color**.

| Valor | Qué hace |
|---|---|
| `ninguna` (por defecto) | No cambia el color: igual que antes de esta opción. |
| `suave` | Iguala el tono de piel (canales A y B de LAB) y la luz media. Recomendado para empezar. |
| `completa` | Además ajusta el contraste de la luz (limitado entre 0.8 y 1.25). Úsalo cuando el destino tiene luz muy distinta a la de la fuente (muy oscura, muy dura o con sombras fuertes). |

Cómo funciona: la cara generada (128 px) y la cara original del destino se recortan alineadas igual; se mide el color solo en la piel (el centro de la cara sin ojos ni boca) y a cada canal se le aplica `(píxel - media_generada) × (desviación_destino / desviación_generada) + media_destino`. El contraste de la luz no se iguala del todo a propósito: borraría las sombras propias de la cara fuente y se perdería parecido. En las pruebas con 8 pares de caras, la diferencia de luz con el destino bajó de 3.3 a 2.1 (`suave`) y a 1.4 (`completa`) sin cambiar el parecido con la fuente. Cuesta menos de 1 ms por cara.

Con `--face-mask normal` y corrección de color, la cara se pega con la máscara propia del programa (mismo recorte que el pegado de insightface), porque el pegado de insightface no permite corregir la cara antes. Se aplica antes de la oclusión y de `--eyes` / `--mouth`.

```bash
python cli.py process --source "src/cara.jpg" --target "src/video.mp4" --output "output/resultado.mp4" --color suave --overwrite
```

### Zona de la cara (`--face-mask`)

Define cuánto de la cara cambiada se pega sobre el destino. En la interfaz es el selector **Máscara**.

| Valor | Qué cubre |
|---|---|
| `normal` | El pegado original de insightface: recorta un 10 % del borde, así que mejillas, mentón y frente quedan del destino. |
| `amplia` (por defecto) | Recorta un 5 %: el resultado se parece más a la fuente. |
| `maxima` | Recorta un 2 %: cubre casi todo el recorte del modelo. Si en algún video aparece un borde marcado, vuelve a `amplia`. |

La oclusión (`--occlusion`) se aplica después, así que el pelo, las manos y los objetos siguen quedando delante de la cara con cualquier valor.

### Filtro de género (`--genero`)

Solo cambia las caras destino del género indicado: `todos` (por defecto), `mujer` u `hombre`. Las caras de la imagen fuente no se filtran.

```bash
python cli.py process --source "src/rostro.jpg" --target "src/video.mp4" --output "output/resultado.mp4" --genero mujer
```

- En modo `posicion`, los índices cuentan solo las caras de ese género: con `--genero mujer --target-index 0`, el `0` es la primera mujer de izquierda a derecha aunque haya un hombre a su izquierda.
- En modo `identidad`, solo se registran personas de ese género.
- El género lo estima el modelo y puede fallar con caras pequeñas, de perfil o ambiguas.

### Asignación por identidad (`--asignacion identidad`)

Por defecto (`posicion`), las caras se emparejan de izquierda a derecha, así que si las personas cambian de lugar en el video, las caras se intercambian. Con `identidad`, cada persona se reconoce por su rostro y conserva siempre la misma cara fuente.

- `--source-index`: qué cara fuente usar (por posición en la imagen fuente, o el orden de los modelos `.npy`).
- `--target-index`: **número de persona del video**: `0` = la persona que **más aparece**, `1` = la segunda que más aparece, etc.
- Ambos deben tener la misma cantidad de índices y se emparejan en orden.

```bash
python cli.py process --source "src/modelos_caras/ana.npy,src/modelos_caras/luis.npy" --target "src/video.mp4" --output "output/resultado.mp4" --asignacion identidad --source-index 0,1 --target-index 0,1
```

**Antes de procesar, el video se analiza solo** (análisis previo de personas):

1. Recorre el video a 1 frame por segundo (como máximo 300 frames repartidos en todo el video: un video de 30 minutos se analiza a 1 frame cada 6 segundos) y detecta todas las caras.
2. Agrupa las caras por persona y las numera de la que más aparece (0) a la que menos. Los extras, que aparecen poco, quedan con números altos y no se tocan si no están en `--target-index`.
3. De cada persona guarda varias referencias: **el promedio de todas sus caras y hasta 5 tomas buenas y distintas entre sí** (de frente, girada, con otra luz). Al procesar, cada cara se compara con todas y cuenta la que más se le parece: así se reconoce a la persona girada o con otra luz, donde una sola referencia fallaba.
4. Las personas quedan fijas desde el primer frame (antes se iban registrando mientras se procesaba y los primeros segundos usaban la toma que hubiera).

El CLI muestra a quién encontró antes de empezar:

```text
Personas en el video:
  persona 0   aparece  107 veces   desde 0:03   -> output\video_personas\persona_0.jpg
  persona 1   aparece   93 veces   desde 0:00   -> output\video_personas\persona_1.jpg
  persona 2   aparece    6 veces   desde 0:01   -> output\video_personas\persona_2.jpg
```

Si los números no son los que esperabas, cancela (Ctrl+C), mira las miniaturas y cambia `--target-index` (por ejemplo `1,0`).

**En la interfaz**, en la fila **Índices**, el botón **Ver personas...** analiza el video destino y muestra una ventana con la miniatura de cada persona, su número, cuántas veces aparece y desde cuándo. Así sabes qué número poner en **Destino** antes de procesar. El **Preview** usa la misma numeración.

**Ajustar el análisis** (solo con `identidad`): en la interfaz, fila **Caras** → **Análisis de personas**: `fps` y `máx.` frames (se activan al elegir `identidad`); en la CLI, `--personas-fps` y `--personas-max`. También se guardan en las [configuraciones](#configuraciones-guardadas).

| Opción | Por defecto | Rango | Más alto | Más bajo |
|---|---|---|---|---|
| Frames por segundo | 1 | 0.1 a 10 | Más tomas de cada persona y detecta a quien aparece poco; tarda más | Más rápido; puede no ver a quien aparece solo unos segundos |
| Máximo de frames | 300 | 10 a 3000 | En videos largos, más tomas repartidas | El análisis de un video largo tarda menos; las personas secundarias tienen menos tomas |

El máximo solo importa en videos largos: con 1 fps, un video de menos de 5 minutos nunca llega a 300 frames. Ejemplo: `--personas-fps 2 --personas-max 600` para un video con muchos extras o cambios de luz.

**Tiempo y guardado:** el análisis tarda unos 8 a 15 s por minuto de video, y nunca más de 1 minuto y medio (por el tope de 300 frames); comparado con el proceso completo es menos del 1 %. Se guarda en `<carpeta de salida>/<nombre del video>_personas/` (`personas.json` y las miniaturas `persona_N.jpg`) y se reutiliza: procesar el mismo video otra vez (otras fuentes u opciones) o abrir **Ver personas** de nuevo no vuelve a analizarlo. Se vuelve a analizar solo si cambia el video, el **Género destino** o las opciones del análisis.

Consideraciones:

- Si `--target-index` pide un número que no existe (por ejemplo `3` cuando hay 3 personas, de 0 a 2), avisa antes de procesar.
- Una cara borrosa o de perfil no crea una persona nueva, pero sí cuenta como aparición de una persona ya vista.
- Si una misma persona quedó partida en dos grupos (por ejemplo, con dos peinados muy distintos), se unen cuando sus promedios se parecen y nunca aparecen en el mismo frame.
- Los videos a medias hechos con la numeración anterior (por orden de aparición) no se pueden continuar: hay que empezarlos de cero.
- **Con una imagen o una carpeta de imágenes** no hay análisis previo: las personas se numeran en el orden en que se ven (en una imagen, de izquierda a derecha), como antes.

### Modelos de cara (`.npy`)

Un modelo de cara guarda la identidad de una persona como un vector de 512 números: el promedio de varias fotos de esa persona. inswapper solo usa ese vector de la cara fuente, así que el modelo reemplaza a la foto, y con varias fotos el parecido es más estable que con una sola (se compensan la luz, el ángulo y la expresión de cada foto).

**Crear un modelo** (una vez por persona), con varias fotos, una carpeta o un collage:

```bash
python cli.py crear-modelo --fotos "src/persona1/" --output "src/modelos_caras/persona1.npy"
python cli.py crear-modelo --fotos "src/foto1.jpg" "src/foto2.jpg" "src/collage.jpg" --output "src/modelos_caras/persona1.npy"
```

Al crearlo:

- Se detectan todas las caras (un collage puede tener varias) y se descartan las borrosas, pequeñas o de perfil.
- Si se cuela otra persona, sus caras se descartan y se avisa: se usa la persona que más se repite.
- Se guardan tres archivos: `persona1.npy` (el vector), `persona1.jpg` (miniatura para reconocerlo) y `persona1.json` (fotos usadas, caras descartadas y similitud promedio entre caras; desde 0.5 es buena).
- Con 3 a 10 fotos nítidas y con luces o expresiones distintas alcanza.

**Qué fotos usar** (importa más la variedad y la calidad que la cantidad):

| Aporta mucho | Evitar |
|---|---|
| Luces distintas (exterior, interior, sombra) | Filtros de belleza fuertes (cambian los rasgos) |
| Ángulos ligeros (de frente y girada unos 15-20°) | Perfiles, lentes de sol, mascarillas, manos o pelo tapando ojos, nariz o boca |
| Fotos de días distintos (otro peinado o maquillaje) | Fotos borrosas, muy pequeñas o muy comprimidas |
| Expresiones suaves (neutra, sonrisa) | Muchas copias casi iguales de la misma toma |

El fondo o el lugar no importa: solo se usa la cara. Más de 20 fotos casi no mejora el resultado (el promedio ya está estable) y, si muchas son malas o parecidas entre sí, puede empeorarlo.

#### Crear un modelo desde un video

En vez de juntar fotos, se puede usar un video de la persona: se sacan frames del video y con las caras de esos frames se crea el modelo, igual que con fotos.

```bash
python cli.py crear-modelo --video "src/persona1.mp4" --fps 1 --output "src/modelos_caras/persona1.npy"
python cli.py crear-modelo --video "src/video1.mp4" "src/video2.mp4" --fotos "src/foto1.jpg" --fps 0.5 --output "src/modelos_caras/persona1.npy"
```

`--video` recibe uno o varios videos y `--fotos` imágenes o carpetas; se puede usar uno de los dos o ambos. `--fps` solo afecta a los videos. Si se pasa un video en `--fotos` (o una foto en `--video`), avisa en vez de seguir.

En la interfaz: botón **Desde video** (o arrastrar uno o varios videos a la zona de origen). Se abre una ventana con:

- La duración y los fps de cada video.
- **Frames por segundo a sacar**: `1` = un frame por segundo (por defecto), `0.5` = uno cada 2 segundos, `2` = dos por segundo. Debajo se ve cuántos frames saldrán y se actualiza al cambiar el valor.
- **Nombre del modelo**: viene con el nombre del video.
- **Crear modelo**: el avance se ve abajo en la barra de estado (`persona1.mp4: frame 12/~30 (11.0 s): 1 cara(s) util(es)`) y al terminar el modelo queda cargado como origen.

Cómo funciona:

- Los frames se sacan **en memoria**: no se guardan imágenes en disco (salvo si eliges revisar las caras, ver abajo) ni se modifica el video.
- Como máximo se sacan **200 frames por video**. Si el video es largo, se baja el fps solo lo necesario para repartir esos 200 frames en todo el video (no solo el principio), y se avisa.
- Cada frame pasa por los mismos filtros que una foto: se descartan las caras borrosas, pequeñas o de perfil, y si aparece otra persona se usa la que más se repite.
- Se pueden mezclar videos y fotos en la CLI (`--video` y `--fotos` juntos); en la interfaz, o solo videos o solo fotos.
- El `.json` del modelo indica cuántos frames de cada video aportaron una cara (`videos`) y el fps usado (`fps_video`).

¿Qué fps elegir? Los frames seguidos de un video son casi iguales, así que más fps no siempre da más parecido: lo que mejora el modelo es que en el video cambien la luz, el ángulo o la expresión. Para un video de 10 a 60 segundos, `1` o `2` alcanza; para uno de varios minutos, `0.5` o menos. Cada frame tarda en analizarse más o menos lo mismo que una foto.

#### Revisar las caras antes de crear el modelo

Para un modelo más fiel, las caras del video se pueden guardar en carpetas, revisar a mano (borrar las borrosas, tapadas, con gestos raros o de otra persona) y crear el modelo solo con las que quedan.

Se guardan en `src/caras_extraidas/<nombre>/`, una carpeta por persona:

```
src/caras_extraidas/persona1/
├── persona_1/          la persona que más aparece
│   ├── 001_persona1_12.0s.jpg    la mejor cara (nítida, de frente y grande)
│   ├── 002_persona1_3.0s.jpg
│   └── ...                       las peores quedan al final
└── persona_2/          otra persona que apareció en el video
```

El nombre de cada archivo dice de qué segundo del video salió. Cada imagen es un recorte cuadrado con la cara al centro.

Mientras se extraen, las caras se van guardando una por una en `_en_proceso/` dentro de esa carpeta (la interfaz la abre al empezar), así se ve el avance. Al terminar, se reparten en `persona_1`, `persona_2`... ordenadas, y `_en_proceso` desaparece: para separar por persona y ordenar de mejor a peor hace falta haber visto todas las caras.

**En la interfaz:** en la ventana **Desde video**, marca **Revisar las caras antes de crear el modelo**; el botón pasa a decir **Extraer caras**. Al terminar:

1. Se abre la carpeta de `persona_1` en el explorador y una ventana con cada persona (miniatura y cantidad de caras).
2. Borra las caras malas en el explorador. Para otra persona, elígela en la ventana y pulsa **Abrir carpeta**.
3. Pulsa **Crear modelo**: se crea con las caras que quedaron en la carpeta de la persona elegida y queda cargado como fuente.

Si cierras la ventana, las caras quedan guardadas: después puedes soltar la carpeta de la persona en la zona de fuente para crear el modelo.

**En la CLI** son dos pasos:

```bash
# 1. Extraer (sin CARPETA: src/caras_extraidas/<nombre del video>)
python cli.py crear-modelo --video "src/persona1.mp4" --fps 1 --extraer-caras
# 2. Revisar la carpeta y crear el modelo con las que quedaron
python cli.py crear-modelo --fotos "src/caras_extraidas/persona1/persona_1" --output "src/modelos_caras/persona1.npy"
```

- `--extraer-caras CARPETA` guarda en otra carpeta.
- Los videos bajados de internet suelen tener nombres muy largos, y Windows no acepta rutas de más de 260 caracteres. Por eso el nombre sugerido (y la carpeta por defecto) se acorta a 40 caracteres y, dentro del nombre de cada cara, el del video a 30.
- Si ya se extrajeron caras con ese nombre, avisa; con `--overwrite` las reemplaza. Una carpeta que ya tiene otros archivos (no creada por TransferIa) nunca se reemplaza.

**Usar un modelo** como fuente, con las mismas opciones de siempre:

```bash
python cli.py process --source "src/modelos_caras/persona1.npy" --target "src/video.mp4" --output "output/resultado.mp4" --occlusion --eyes
```

Con varias personas, los modelos se separan por comas y `--source-index` se refiere a ellos en ese orden (0 = el primero):

```bash
python cli.py process --source "src/modelos_caras/persona1.npy,src/modelos_caras/persona2.npy" --target "src/video.mp4" --output "output/resultado.mp4" --asignacion identidad --source-index 0,1 --target-index 0,1
```

**En la interfaz**, la zona **Imagen Fuente o Modelo de Cara** acepta:

| Qué se suelta o elige | Qué pasa |
|---|---|
| Una imagen | Igual que siempre: se usa como fuente (reemplaza los modelos cargados). |
| Un `.npy` | Se usa el modelo como fuente y se ve su miniatura. Si ya hay modelos cargados, pregunta: **Sí** = agregarlo al final, **No** = reemplazarlos. |
| Varios `.npy` juntos | Se cargan todos como fuentes, en el orden en que se soltaron (misma pregunta si ya había modelos). |
| Varias imágenes o una carpeta | Pregunta si crear un modelo, pide un nombre y lo guarda en `src/modelos_caras/`; queda cargado como fuente. |
| Uno o varios videos | Abre la ventana para crear un modelo desde video (ver [Crear un modelo desde un video](#crear-un-modelo-desde-un-video)). |
| `.npy`, imágenes y videos mezclados | No se permite: avisa que se suelten solo modelos, solo fotos o solo videos. |

**Varios modelos a la vez** (como varias caras en una imagen fuente):

- Las miniaturas se ven una al lado de otra con su índice debajo (`[0]`, `[1]`...), y junto a **Modelos:** aparece el orden (`0=ana, 1=luis`).
- **Índices Fuente** se refiere a ese orden, igual que con una imagen de varias caras: con `0,1` en Índices Fuente y `0,1` en Índices Destino, la cara 0 del destino recibe `ana` y la 1 recibe `luis`; con `1,0` se invierten.
- Funciona igual en modo `identidad`.
- Antes de procesar se valida que los índices existan: con 2 modelos, un índice `2` muestra un error en vez de fallar a mitad del video.
- **Elegir modelos...** abre una ventana con los modelos de `src/modelos_caras/`: se marcan los que se quieren usar y se ordenan con ↑ ↓; el orden es el índice.

Debajo de la zona de origen están los botones **Elegir modelos...**, **Crear modelo de cara**, **Desde video**, **Combinar modelos** y **Quitar**.

#### Eliminar un modelo

En la ventana de **Elegir modelos...** (y también en la de **Combinar modelos**) cada modelo tiene, a la derecha, un botón **Eliminar**:

1. Pregunta antes: *«¿Eliminar el modelo «ana»?»*, con **No** como opción por defecto. Con **No** no se borra nada.
2. Con **Sí** se borran sus tres archivos de `src/modelos_caras/`: `ana.npy` (la identidad), `ana.json` (sus datos) y `ana.jpg` (la miniatura). **No se puede deshacer**: para volver a tenerlo hay que crearlo otra vez con sus fotos.
3. Desaparece de la lista y, si estaba cargado como fuente, se quita de las fuentes; los demás siguen en su orden (sus índices se corren: si se elimina el 0, el que era 1 pasa a ser 0). Si era la única fuente, el origen queda vacío.

Los otros modelos y las fotos con las que se crearon no se tocan.

#### Combinar modelos de la misma persona

Si tienes varios modelos de la misma persona (por ejemplo `ana_1` hecho con unas fotos y `ana_2` con otras), puedes unirlos en uno nuevo sin volver a buscar las fotos:

```bash
python cli.py combinar-modelos --modelos "src/modelos_caras/ana_1.npy" "src/modelos_caras/ana_2.npy" --output "src/modelos_caras/ana.npy"
```

En la interfaz: botón **Combinar modelos** → marca los modelos de la misma persona (empieza sin ninguno marcado) → escribe el nombre del nuevo → **Aceptar**. Debajo de la lista se ve cuántas caras suman. El modelo creado queda cargado como origen.

Cómo funciona:

- **Cada modelo pesa según cuántas caras tiene** (dato del `.json`; si no tiene `.json` cuenta como 1). Si `ana_1` tiene 4 caras y `ana_2` 12, `ana_2` pesa el triple. El resultado es prácticamente igual a haber creado un modelo con todas las fotos juntas (en las pruebas: similitud 0.9996).
- **Comprueba que sea la misma persona:** si un modelo no se parece a los demás (similitud menor a 0.35), no combina nada y avisa cuál es.
- **Los originales no se modifican**, y el nombre del nuevo debe ser distinto al de los que se combinan.
- Se guarda como un modelo normal: `.npy`, la miniatura del modelo con más caras y un `.json` con el total de caras y de qué modelos salió (`combinado_de`, con la similitud de cada uno con el resultado).

¿Cuándo mejora? Igual que agregar fotos: ayuda si los modelos tienen pocas caras (por ejemplo 3 + 4) o si salieron de fotos de épocas, luces o estilos distintos. Si cada modelo ya tiene 15 o 20 caras, o si son casi iguales (similitud entre ellos de 0.9 o más; el programa lo avisa), aporta muy poco.

Notas:

- Solo el `.npy` es necesario: el `.json` y el `.jpg` son opcionales (sin `.jpg`, la interfaz muestra el nombre en lugar de la miniatura).
- El `.npy` no es una imagen y no se puede convertir en foto; la miniatura `.jpg` es solo para reconocerlo. Aplicar el modelo sobre una foto da una imagen parecida a esa identidad, pero no equivalente: si se usa esa imagen como origen, se pierde parte del parecido. Para cambiar caras, use siempre el `.npy`.
- Sirve mientras se use el mismo modelo de reconocimiento (`buffalo_l`); si cambia, el programa avisa y conviene volver a crearlo.
- Es un dato biométrico: guárdelo con el mismo cuidado que las fotos. Los modelos se guardan en `src/modelos_caras/` (no en `models/`, que es solo para los modelos `.onnx`); `src/` no se sube a git.

### Ejemplo combinado

Solo mujeres, por identidad, con oclusión y 4 hilos: la 2ª mujer en aparecer recibe la fuente 0 y la 1ª recibe la fuente 1.

```bash
python cli.py process --source "src/rostros.jpg" --target "src/video.mp4" --output "output/resultado.mp4" --asignacion identidad --genero mujer --source-index 0,1 --target-index 1,0 --occlusion --nitidez natural --threads 4 --overwrite
```

Con modelo de cara, ojos y boca del destino y corrección de color:

```bash
python cli.py process --source "src/modelos_caras/persona1.npy" --target "src/video.mp4" --output "output/resultado.mp4" --occlusion --eyes --mouth --color suave --overwrite
```

### Guardar en ZIP con contraseña (`--zip`)

Para que el video (o la imagen) resultante no quede suelto en la carpeta: al terminar se guarda en un `.zip` cifrado con **AES-256** y se borra el archivo suelto.

**En la interfaz**, en "Configuración de Salida":

- **Guardar en ZIP con contraseña**: actívalo y escribe la **Contraseña** (se ve con puntos). Sin contraseña no deja empezar.
- **Conservar también el archivo suelto** (opcional, desmarcado): deja el `.mp4` o `.jpg` además del ZIP.

**En la CLI**:

```bash
python cli.py process --source "src/cara.jpg" --target "src/video.mp4" --output "output/resultado.mp4" --zip
```

Pide la contraseña dos veces sin mostrarla. `--zip-password CLAVE` la pasa sin preguntar, pero queda guardada en el historial de comandos. `--zip-conservar` conserva también el archivo suelto. Si el ZIP ya existe, hace falta `--overwrite`.

| Resultado | ZIP |
|---|---|
| Video `output/resultado.mp4` | `output/resultado.zip` con `resultado.mp4` adentro |
| Imagen `output/resultado.jpg` | `output/resultado.zip` con `resultado.jpg` adentro |
| Carpeta `output/fotos_output/` | `output/fotos_output.zip` con la carpeta `fotos_output/` y todas las imágenes |

Cómo funciona y qué hay que saber:

- **Para abrirlo hace falta 7-Zip o WinRAR** (gratis). El Explorador de Windows no abre ZIP con AES-256; los que sí abre usan un cifrado débil (ZipCrypto) que se rompe con programas en minutos.
- **No se borra nada hasta comprobar el ZIP**: se vuelve a abrir con la contraseña, se revisa que cada archivo esté completo (control CRC y tamaño) y recién entonces se borra el suelto. Si algo falla, el archivo suelto queda y se avisa.
- **La contraseña no se guarda en ningún lado**, tampoco en las [configuraciones guardadas](#configuraciones-guardadas) (que sí recuerdan si el ZIP está activado). Si se olvida, el contenido no se puede recuperar.
- Sin contraseña se ven los nombres de los archivos que hay adentro, pero no su contenido.
- Los videos e imágenes ya vienen comprimidos: el ZIP no los achica (se guardan tal cual, que es más rápido).
- **Mientras se procesa**, el video se va guardando en pedazos en una carpeta temporal (para poder detener y continuar), y el `.mp4` final existe unos segundos antes de pasar al ZIP. Un archivo borrado se puede recuperar con programas especiales; para proteger todo el disco está BitLocker de Windows.
- Si se pulsa **Detener**, no se crea el ZIP (el proceso no terminó).
- Con una carpeta de imágenes se guardan en el ZIP las imágenes de este proceso, y la carpeta suelta se borra solo si queda vacía (no se borra nada que ya estuviera ahí).
- Requiere la librería `pyzipper` (`pip install pyzipper`, ya está en `requirements.txt`).

### Configuraciones guardadas

Todas las opciones se pueden guardar con un nombre (por ejemplo `catrina` o `natural`) y volver a usarlas después, sin marcar todo otra vez. Cada configuración es un archivo JSON en `configuraciones/<nombre>.json`.

**En la interfaz**, debajo de **Abrir Preview**:

- **Guardar como...**: pide un nombre y guarda todas las opciones que están en pantalla. Si el nombre ya existe, pregunta antes de reemplazarla.
- **Configuración guardada** (lista): al elegir una, se ponen todas sus opciones. Si el preview está abierto, se actualiza solo.
- **Eliminar**: borra la configuración elegida, después de confirmar (no cambia las opciones que están en pantalla).

**En la CLI**:

```bash
# Guardar las opciones de un comando (y procesar con ellas)
python cli.py process --source "src/cara.jpg" --target "src/foto.jpg" --output "output/r.jpg" --makeup 0.6 --eyes --color suave --save-config catrina

# Usar una configuracion guardada; lo que se escribe en el comando tiene prioridad
python cli.py process --source "src/cara.jpg" --target "src/video.mp4" --output "output/r.mp4" --config catrina
python cli.py process --source "src/cara.jpg" --target "src/video.mp4" --output "output/r.mp4" --config catrina --face-mask maxima
```

Las configuraciones son las mismas para la interfaz y la CLI: una guardada en una se puede usar en la otra.

**Qué se guarda:** asignación, género, índices fuente y destino, máscara, corrección de color, ojos, boca, nariz, frente, maquillaje, oclusión (y su modo), filtro de nitidez, mejora GPEN (activada, fuerza y cara mínima), número de hilos, video temporal, códec, carpeta de salida, nombre del archivo y si se guarda en ZIP (sin la contraseña). **No se guardan** la fuente, el destino ni el fragmento: una configuración dice *cómo* procesar, no *qué*. La carpeta de salida y el nombre del archivo solo los usa la interfaz (en la CLI la salida es `--output`).

Ejemplo de `configuraciones/catrina.json`:

```json
{
  "nombre": "catrina",
  "version": 1,
  "guardado": "2026-10-25T18:30:00",
  "ajustes": {
    "mascara": "amplia",
    "color": "suave",
    "ojos": true,
    "maquillaje": 0.6,
    "mejorar": true,
    "fuerza_mejora": 0.8,
    "minimo_mejora": "96",
    "...": "y el resto de las opciones"
  }
}
```

Notas:

- Se puede editar a mano. Al cargar, cada valor se revisa: si uno es inválido (por ejemplo `"mascara": "gigante"`) o falta, esa opción se deja como estaba y se avisa cuál fue; las demás se cargan igual. Así, una configuración guardada con una versión anterior sigue sirviendo.
- Al guardar, si alguna opción es inválida (por ejemplo índices con letras), no se guarda y se dice cuál corregir.
- En la CLI, las opciones de sí/no (`--eyes`, `--occlusion`...) que vienen activadas en la configuración no se pueden apagar desde el comando; para eso, guarda otra configuración.
- Cargar o guardar una configuración no afecta la opción de continuar un video cortado.

### Videos: detener y continuar

Los videos se procesan frame por frame con varios hilos y cada frame terminado va **directo a ffmpeg** por una tubería: no se guardan imágenes en disco y en memoria solo están los frames que se están procesando (unos pocos), así que un video largo no llena la RAM. ffmpeg los comprime en H.264 una sola vez (antes, la interfaz guardaba cada frame como JPG y luego los volvía a comprimir, perdiendo calidad dos veces).

Mientras se procesa, todo queda en una carpeta junto a la salida, `<salida>.partes/`:

| Archivo | Qué es |
|---|---|
| `parte_001.mkv`, `parte_002.mkv`... | Lo procesado en cada sesión (una parte nueva cada vez que se continúa) |
| `progreso.json` | Video destino, opciones, frames hechos y, en modo identidad, las personas ya registradas. Se actualiza cada 2 s |

Al terminar, las partes se unen sin recomprimir, se les pone el audio original (si el audio no entra en el formato de salida se convierte a AAC) y la carpeta se borra.

**Formato del video temporal** (`--container`, o **Video temporal** en la interfaz). Un MP4 normal guarda su índice al final: si el proceso se corta, el archivo queda inservible aunque tenga horas de video. Por eso el temporal usa un formato que aguanta cortes:

| Valor | Qué es |
|---|---|
| `mkv` (por defecto) | Matroska. Guarda el video en bloques; si se corta, se reproduce hasta el corte. |
| `mp4frag` | MP4 fragmentado: guarda el índice por partes. Útil si se quiere abrir el temporal con programas que no leen MKV. |

El resultado final es siempre el de `--output` (por ejemplo `.mp4`), con cualquiera de los dos.

**Detener y continuar:**

- **Interfaz:** botón **Detener**. Termina los frames que ya estaban en proceso, cierra bien el archivo y avisa. Al volver a procesar el mismo video con la misma salida y las mismas opciones, pregunta: **Sí** = continuar, **No** = empezar de cero, **Cancelar** = no hacer nada.
- **CLI:** `Ctrl+C` detiene de la misma forma y dice cómo continuar. Para continuar se repite el mismo comando con `--resume`:

```bash
python cli.py process --source "src/cara.jpg" --target "src/video.mp4" --output "output/resultado.mp4" --occlusion --resume
```

- **Corte brusco** (apagón, cierre forzado, cuelgue): también se puede continuar. Al continuar se cuentan los frames que de verdad tiene la última parte, se descartan los 2 últimos por si quedaron a medias, y se sigue desde ahí. Se pierde como mucho un par de segundos de trabajo.
- Se puede detener y continuar las veces que haga falta.
- **Las opciones deben ser las mismas** (fuente, índices, asignación, género, oclusión, filtro, ojos, boca, nariz, máscara, color, maquillaje, mejora, códec, calidad y el [fragmento](#procesar-solo-un-fragmento-del-video), si se usa), y el video destino no debe haber cambiado. Si algo cambió, no deja continuar y dice qué fue, para que el video no cambie de estilo a la mitad. En la CLI, `--restart` empieza de cero; en la interfaz, se ofrece borrar lo hecho. El número de hilos sí se puede cambiar.
- En la CLI, si queda un proceso sin terminar y se ejecuta sin `--resume` ni `--restart`, avisa en lugar de borrarlo.

**Codificador** (`--video-codec`, o **Códec** en la interfaz):

| Valor | Cuándo |
|---|---|
| `libx264` (por defecto) | Siempre funciona. Usa la CPU y comprime mejor (archivos más chicos a igual calidad). Codificar cuesta muy poco comparado con el cambio de caras. |
| `h264_nvenc` | PCs con tarjeta NVIDIA: casi no usa la CPU. Los archivos salen un poco más grandes (10-30 %) a igual calidad. |
| `h264_amf` | PCs con tarjeta AMD (Radeon). |

`--crf` regula la calidad (18 por defecto: prácticamente igual al original; 23 = más liviano). El video se guarda sin B-frames y con un fotograma clave cada 2 s, para que una parte cortada se pueda recortar y continuar sin saltos.

Pruebas con un video de 30 frames numerados: completo, con corte brusco + `--resume`, con dos cortes seguidos, con Detener en la interfaz, en modo identidad y con `--restart`; en todos los casos salieron 30 de 30 frames, en orden y con audio.

### Procesar solo un fragmento del video

Para no procesar un video entero cuando solo interesa una parte (por ejemplo, los segundos donde se ve bien la cara), se elige un **inicio** y un **fin**. El resultado es **solo ese fragmento**, con su audio; el resto del video no se procesa ni se guarda.

**Interfaz:** en "Configuración de Procesamiento", fila de abajo:

1. Marque **Solo un fragmento** (o pulse **Elegir fragmento...**). Se abre la ventana **Fragmento del video**, que muestra el video original sin cambiar caras:
   - Arriba, el nombre del video, cuánto dura, cuántos frames tiene y a cuántos fps.
   - **Inicio** y **Fin**: dos deslizadores. Al mover uno se ve ese frame (el del inicio, o el último del fragmento) y su momento (`1:05.3`). El inicio siempre queda antes que el fin: si se cruzan, el otro se corre solo.
   - **Duración del fragmento**: cuánto dura y cuántos frames tiene, se actualiza al mover los deslizadores.
   - **▶ Reproducir fragmento**: lo reproduce a velocidad real, de inicio a fin, mostrando en qué momento va (`Viendo 0:16.4 (0:01.8 de 0:02.5)`). El mismo botón pausa; al terminar, la próxima vez empieza otra vez desde el inicio.
   - **Aceptar** guarda el fragmento; **Video completo** lo quita; **Cancelar** no cambia nada.
2. Al lado de la casilla queda el fragmento elegido, por ejemplo `0:14.6 → 0:17.1 (dura 0:02.5)`. Desmarcar la casilla vuelve al video completo (el fragmento se recuerda por si se vuelve a marcar).
3. Con el fragmento activo, la barra del **preview** recorre solo ese pedazo, y el contador cuenta dentro del fragmento: `Fragmento: 37/74 · 0:01.2 de 0:02.5 (frame 474 del video)`. Sin fragmento cuenta el video completo: `299/599 · 0:10.0 de 0:20.0`.
4. Al cambiar el video destino, el fragmento se borra (era del video anterior).

**CLI:** `--start` y `--end`, en segundos (`90`, `90.5`), `m:ss` (`1:30`, `1:30.5`) o `h:mm:ss`. Se puede usar uno solo: sin `--start` empieza desde el principio, sin `--end` llega hasta el final. Al empezar muestra el fragmento que va a procesar. En imágenes se ignoran (con un aviso).

```bash
python cli.py process --source "src/cara.jpg" --target "src/video.mp4" --output "output/fragmento.mp4" --start 1:05 --end 1:20.5
```

Detalles:

- Para llegar al inicio, si está cerca del principio (hasta 300 frames) lee y descarta los frames anteriores; si está más lejos, salta directo (va al fotograma clave anterior y avanza hasta el frame pedido). Si el salto no queda exacto, lee desde el principio: el fragmento siempre empieza en el frame justo.
- El audio se recorta al mismo tramo.
- **Videos con ritmo de frames variable o con audio e imagen que no empiezan juntos** (frecuente en grabaciones de VLC, de pantalla o del celular): se detectan solos y se procesa una copia a ritmo fijo del tramo, con su audio, para que el resultado no quede desfasado. Se avisa al empezar; tarda un poco más y la copia se borra al terminar. En las pruebas, el desfase bajó de hasta 470 ms a menos de 25 ms. Vale también para el video completo.
- **Detener y continuar** funciona igual dentro del fragmento. Para continuar, el fragmento debe ser el mismo; si cambió el inicio o el fin, avisa *"cambió el fragmento del video (inicio o fin)"* y no deja continuar.
- Pruebas con un video numerado de 600 frames (fotograma clave cada 50): fragmentos cerca del inicio (10-40), lejos (437-512), hasta el final (550-600) y detenido + continuado (300-420); en todos salieron exactamente esos frames, en orden, y el audio dura lo mismo que el fragmento (±0.01 s).

### Video de prueba

Para ver cómo queda antes de procesar un video entero: botón **Video de prueba** (fila de abajo de "Configuración de Procesamiento"), o `--prueba` en la CLI.

- Procesa **el fragmento elegido** o, si no hay, **los primeros 10 segundos**, con las opciones actuales, a **10 fps** por defecto (se puede elegir entre 1 y 30).
- Se guarda como `<nombre>_prueba.mp4` en la carpeta de salida: el resultado final no se toca. Al terminar ofrece abrirlo.
- El audio queda sincronizado (el tramo se prepara a esos fps con la imagen y el audio alineados).
- ¿Por qué 10 fps? Es un tercio del trabajo de un video a 30 fps y alcanza para juzgar parecido, color, oclusión y ojos. Con menos se ve entrecortado: se pierden parpadeos y movimientos de boca, y cada imagen puede quedar hasta medio frame corrida respecto del sonido (63 ms a 8 fps, 50 ms a 10 fps).

```bash
python cli.py process --source "src/cara.jpg" --target "src/video.mp4" --output "output/resultado.mp4" --prueba
python cli.py process --source "src/cara.jpg" --target "src/video.mp4" --output "output/resultado.mp4" --prueba 12 --start 1:00 --end 1:20
```

## 🎯 Reglas y Consideraciones

1. **Formatos Soportados**:
   - Imágenes: JPG, JPEG, PNG
   - Videos: MP4, AVI, MKV, MOV

2. **Calidad de Entrada**:
   - Las imágenes de origen deben tener un rostro claramente visible
   - Mejor calidad de entrada = mejor resultado
   - Resolución mínima recomendada: 512x512 píxeles

3. **Limitaciones**:
   - Se pueden cambiar varios rostros a la vez indicando varios índices (por ejemplo `0,1,2`)
   - El proceso puede ser intensivo en recursos
   - Los videos largos pueden tomar tiempo considerable

4. **Buenas Prácticas**:
   - Utilice imágenes con buena iluminación
   - Las caras inclinadas o boca abajo se detectan, pero las caras de perfil siguen dando peores resultados
   - Mantenga la aplicación actualizada
   - Cierre otras aplicaciones pesadas durante el procesamiento

5. **Seguridad**:
   - No se guardan ni envían datos a servidores externos
   - Todo el procesamiento es local
   - Respete los derechos de imagen y privacidad

## ⚠️ Advertencias

- El uso de esta herramienta debe ser ético y responsable
- Respete los derechos de autor y privacidad
- No use la aplicación para crear contenido malicioso o engañoso
- El rendimiento puede variar según el hardware disponible

## 🛠️ Solución de Problemas

1. Si la aplicación no inicia:
   - Verifique que Python esté instalado correctamente
   - Confirme que todas las dependencias estén instaladas
   - Revise los permisos de ejecución

2. Si el procesamiento es lento:
   - Verifique que su GPU sea compatible y esté siendo utilizada
   - Reduzca la resolución del video de entrada
   - Cierre aplicaciones en segundo plano

3. Si hay errores en el procesamiento:
   - Asegúrese de que las imágenes contengan rostros detectables
   - Verifique que los archivos no estén corruptos
   - Intente con diferentes imágenes de origen/destino

## 📞 Soporte

Para reportar problemas o sugerir mejoras:
- Abra un issue en el repositorio
- Proporcione detalles específicos del problema
- Incluya capturas de pantalla si es posible

## 🔄 Actualizaciones

- Mantenga el software actualizado
- Revise periódicamente nuevas versiones
- Lea las notas de cambios antes de actualizar

## 📄 Licencia

Este proyecto está protegido por derechos de autor. Todos los derechos reservados.

## 🖱️ Reglas del Cursor e Interacción

1. **Arrastrar y Soltar**:
   - El cursor cambiará de forma al arrastrar archivos válidos
   - Suelte los archivos en las zonas designadas (marcadas con bordes)
   - Si el archivo no es válido, el cursor mostrará un símbolo de prohibido

2. **Zonas Interactivas**:
   - Zona de Origen: Área para la imagen del rostro a transferir
   - Zona de Destino: Área para la imagen o video donde se aplicará el rostro
   - Botones de Control: Cursor tipo "mano" al pasar sobre ellos

3. **Controles de Video**:
   - El cursor se convierte en "redimensionar" sobre la barra de progreso
   - Use el cursor para arrastrar el control deslizante del video
   - Haga clic en cualquier punto de la barra para saltar a esa posición

4. **Áreas de Vista Previa**:
   - En la ventana de preview, con la vista **Deslizar**, el cursor se convierte en flechas a los lados sobre la imagen: haga clic o arrastre para mover la línea entre el original y el resultado
   - La imagen se ajusta sola al tamaño de la ventana, completa y sin deformarse

5. **Restricciones**:
   - No se permite arrastrar elementos fuera de las zonas designadas
   - El cursor se bloqueará durante el procesamiento
   - Espere a que el cursor vuelva a su estado normal antes de nuevas interacciones 
