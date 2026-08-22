# Transfer-IA

Transfer-IA es una aplicación de cambio facial que utiliza inteligencia artificial para transferencia facial entre imágenes y videos de manera realista.

## 🚀 Características

- Interfaz gráfica moderna y fácil de usar
- Soporte para imágenes y videos
- Vista previa en tiempo real
- Procesamiento por lotes
- Mejora de calidad facial integrada
- Arrastrar y soltar archivos
- Barra de progreso para seguimiento del proceso

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

## 📖 Uso

### Interfaz gráfica

Ejecute la aplicación sin argumentos:
```bash
python app.py
```

En la interfaz principal:
   - Seleccione o arrastre una imagen de origen (rostro a transferir)
   - Seleccione o arrastre una imagen/video de destino
   - Seleccione la carpeta de salida para los resultados
   - Haga clic en "Procesar" para iniciar la transferencia

### Línea de comandos

El CLI está separado de la interfaz y se ejecuta desde `cli.py`:

```bash
python cli.py process --source "src/rostro.jpg" --target "src/imagen.jpg" --output "output/resultado.jpg"
```

Para procesar un video y conservar su audio:

```bash
python cli.py process --source "src/rostro.jpg" --target "src/video.mp4" --output "output/resultado.mp4"
```

Opciones disponibles:

```text
--source PATH              imagen con la cara de origen
--target PATH              imagen o video destino
--output PATH              archivo de salida
--models-dir PATH          carpeta de modelos (por defecto: models)
--device DEVICE            auto, cuda o cpu (por defecto: auto)
--source-index INDICES     indices de origen, por ejemplo 0,1
--target-index INDICES     indices de destino, por ejemplo 0,1
--occlusion                activa el modelo de oclusion
--occlusion-mode MODE      None, natural o strong
--threads NUMBER           numero de hilos solicitado
--overwrite                reemplaza un archivo existente
--no-audio                 omite el audio del video original
--verbose                  muestra el progreso por frame
```

Puedes consultar la ayuda completa con:

```bash
python cli.py process --help
```

Para exigir el uso de la GPU NVIDIA:

```bash
python cli.py process --device cuda --source "src/rostro.jpg" --target "src/video.mp4" --output "output/resultado.mp4" --threads 2
```

Si CUDA no esta disponible, `--device cuda` detiene el proceso y muestra un error. `--device auto` intenta usar CUDA y permite CPU como respaldo.

## 🎯 Reglas y Consideraciones

1. **Formatos Soportados**:
   - Imágenes: JPG, JPEG, PNG
   - Videos: MP4, AVI

2. **Calidad de Entrada**:
   - Las imágenes de origen deben tener un rostro claramente visible
   - Mejor calidad de entrada = mejor resultado
   - Resolución mínima recomendada: 512x512 píxeles

3. **Limitaciones**:
   - Solo se procesa un rostro a la vez
   - El proceso puede ser intensivo en recursos
   - Los videos largos pueden tomar tiempo considerable

4. **Buenas Prácticas**:
   - Utilice imágenes con buena iluminación
   - Evite ángulos extremos en las fotos
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
   - El cursor cambia al pasar sobre imágenes previsualizadas
   - Haga clic para ampliar la vista previa
   - Use la rueda del mouse para hacer zoom en la vista previa

5. **Restricciones**:
   - No se permite arrastrar elementos fuera de las zonas designadas
   - El cursor se bloqueará durante el procesamiento
   - Espere a que el cursor vuelva a su estado normal antes de nuevas interacciones 
