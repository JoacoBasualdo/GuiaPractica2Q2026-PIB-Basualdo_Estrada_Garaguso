# Pipeline de Procesamiento de Señales e Imágenes Biomédicas: Reflejo Fotomotor

Repositorio oficial del proyecto integrador correspondiente a los Módulos 0 al 4 del procesamiento de la respuesta pupilar ante estímulos luminosos.

## Descripción del Proyecto
Este pipeline automatizado en Python procesa secuencias de video de la pupila humana para:
1. **Analizar la calidad digital** (espacios de color, cuantización, filtrado espacial de ruido y dominio frecuencial).
2. **Segmentar y limpiar la máscara pupilar** mediante morfología matemática (operaciones de apertura/cierre para eliminar reflejos).
3. **Extraer características morfométricas y de textura** del estroma iridiano (descriptores de primer orden y GLCM).
4. **Cuantificar la dinámica temporal** del reflejo fotomotor (latencia, amplitud, velocidades de constricción y redilatación, validando contra mediciones manuales con MAE/RMSE).
5. **Evaluar esquemas de compresión** sin pérdida (Huffman, RLE) y con pérdida (JPEG), analizando artefactos de bloque y su impacto en la medición del diámetro.

## Uso

1. **Selección de video:** en la función main(), se debe cambiar el nombre del video a analizar.
    En la línea "video_original = 'registro_pupila.mp4'", se debe reemplazar 'registro_pupila.mp4' por el nombre del video que se desea analizar.
2. **Configuración de mediciones manuales:** también en main(), las funciones anotar_referencia y anotar_referencia_3_2 deben ponerse en "True" si es la primera vez que se analiza el video, para poder tomar registro de las mediciones manuales, y luego deben ponerse en False una vez creados los archivos .csv y .png
3. **Respuestas:** hay respuestas que varían según el video a analizar, y respuestas que se corresponden al video original analizado por el grupo. Ambas se imprimen en el pdf. 
4. **Archivos:** el script genera una serie de archivos. Para el módulo 2.3, genera la referencia_registro_pupila (cada vez que se delimita manualmente). Luego, se genera un csv con anotaciones de resultados de mediciones manuales para el módulo 3.2. La carpeta resultados contiene salidas del pipeline en los distintos módulos, principalmente del módulo 3.2.

---

## ⚙️ Requisitos e Instalación

Asegurate de tener instalado **Python 3.8 o superior**. Luego, instalá las dependencias necesarias ejecutando en tu terminal:

```bash
pip install -r requirements.txt