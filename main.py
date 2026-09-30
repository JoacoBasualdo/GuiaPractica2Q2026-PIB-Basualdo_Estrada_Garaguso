# =========================================
# CONSULTAS
# que hacer en la parte de analizar la pérdida de definición
# Que quiere que hagamos en el 1.2.4 para el análisis de ruido y filtrado espacial
# Qué se espera que observemos al aplicarle la TFD-2D
# ============================================



# ======================================================
# ESPACIO PARA LIBRERIAS
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
# IMPORTACIONES ADICIONALES PARA EL MÓDULO 2
from skimage.segmentation import active_contour
from skimage.morphology import skeletonize
from skimage import img_as_float
from skimage.filters import gaussian
# ======================================================


# ======================================================
# MODULO 1 



#Se recorre toda la imagen con un algoritmo para hallar el frame con mayor nitidez y utilizar este como frame representativo
def obtener_mejor_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    mejor_puntaje = 0
    mejor_frame = None
    
    while True:
        ret, frame = cap.read() # Si frame tiene la matriz completa, ret = true
        if not ret:
            break
            
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        puntaje_nitidez = cv2.Laplacian(gris, cv2.CV_64F).var() # Con la varianza se le da un valor. Cuanto mayor sea la varianza, más nitida es
        
        if puntaje_nitidez > mejor_puntaje:
            mejor_puntaje = puntaje_nitidez
            mejor_frame = frame.copy()
            
    cap.release() # "Suelta" el archivo de video para liberar memoria.
    
    # Abrimos una ventana para visualizar el frame ganador antes de devolverlo
    if mejor_frame is not None:
        cv2.imshow('Mejor Frame Encontrado', mejor_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return mejor_frame


# Esta función busca de forma automática la pupila asumiendo que es la zona más oscura de la imagen
def obtener_roi_automatica(frame):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Pasa la imagen a escala de grises
    blur = cv2.GaussianBlur(gris, (15, 15), 0) # Aplica un desenfoque difuminando ruidos pequeños
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blur)
    centro_x, centro_y = min_loc # Valor más oscuro donde se supone que esta la pupila
    
    alto_img, ancho_img = frame.shape[:2]
    
    lado = int(min(ancho_img, alto_img) * 0.4) # Se calcula el tamaño dinámicamente
    w, h = lado, lado
    
    x = max(0, centro_x - w // 2)
    y = max(0, centro_y - h // 2)
    
    if x + w > ancho_img:
        x = ancho_img - w
    if y + h > alto_img:
        y = alto_img - h
        
    return x, y, w, h


def procesar_modulo_1_1(frame, x, y, w, h):
    roi = frame[y:y+h, x:x+w]

    # Mostramos el recorte en pantalla para verificar el encuadre
    cv2.imshow('ROI - Ojo Centrado', roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Dimensiones y canales (Alto, Ancho, Canales):", roi.shape)
    print("Profundidad de bits:", roi.dtype)
    
    # Ahora cambiamos la cantidad de píxeles
    # Lo que hace la función resize es "achicar" el número de píxeles agarrando un grupo de píxeles vecinos, sumando sus valores de color y dividiendo para sacar un promedio, obteniendo un único pixel nuevo
    roi_50 = cv2.resize(roi, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    roi_25 = cv2.resize(roi, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    roi_5 = cv2.resize(roi, (0, 0), fx=0.05, fy=0.05, interpolation=cv2.INTER_LINEAR)

    roi_50_pix = cv2.resize(roi_50, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    roi_25_pix = cv2.resize(roi_25, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    roi_5_pix = cv2.resize(roi_5, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)

    cv2.imshow('Resolucion 50%', roi_50_pix)
    cv2.imshow('Resolucion 25%', roi_25_pix)
    cv2.imshow('Resolucion 5%', roi_5_pix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Consultar que hacer en la parte de analizar la pérdida de definición
    

    # Ahora se modifican la cantidad de bits (cantidad de tonos para cada color)
    # Se hace una división entera (No tiene en cuenta los decimales) y al multiplicar expando en todo el rango
    roi_4bit = (roi // 16) * 16
    roi_2bit = (roi // 64) * 64

    cv2.imshow('Cuantizacion 4 bits', roi_4bit)
    cv2.imshow('Cuantizacion 2 bits', roi_2bit)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Ahora vamos a trabajar con los espacios de color
    
    # Las funciones ctvColor lo que hace es trasnformar los colores de la imagen para cada espacio de color
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(roi_rgb) # La desarma en cada capa y guarda el valor en una variable distintas

    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_comp, s_comp, v_comp = cv2.split(roi_hsv)
    
    
    
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    
    # En la sección de las imagenes donde haya partes claras, indica que hay mucha cantidad de ese color en esa zona
    # Las partes negras representan la poca o nula presencia de ese color.
    axs[0, 0].imshow(r, cmap='gray')
    axs[0, 0].set_title('Canal Rojo (R)')
    axs[0, 1].imshow(g, cmap='gray')
    axs[0, 1].set_title('Canal Verde (G)')
    axs[0, 2].imshow(b, cmap='gray')
    axs[0, 2].set_title('Canal Azul (B)')
    
    axs[1, 0].imshow(h_comp, cmap='gray')
    axs[1, 0].set_title('Matiz (Hue)')
    axs[1, 1].imshow(s_comp, cmap='gray')
    axs[1, 1].set_title('Saturación (Sat)')
    axs[1, 2].imshow(v_comp, cmap='gray')
    axs[1, 2].set_title('Brillo (Value)')
    
    plt.tight_layout()
    plt.show()
    
def procesar_modulo_1_2(frame, x, y, w, h):
    roi = frame[y:y+h, x:x+w]
    
    roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Son los parámetros que controlan el contraste y el brillo de la imagen aplicando la fórmula g(x) = alpha * f(x) + beta
    alpha = 1.5 # Aumenta el contraste
    beta = 50 # Suma brillo constante
    roi_ajustada = cv2.convertScaleAbs(roi_gris, alpha=alpha, beta=beta)

    hist_original = cv2.calcHist([roi_gris], [0], None, [256], [0, 256])
    hist_ajustada = cv2.calcHist([roi_ajustada], [0], None, [256], [0, 256])

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].imshow(roi_gris, cmap='gray', vmin=0, vmax=255)
    axs[0, 0].set_title('ROI Original (Grises)')
    axs[0, 1].plot(hist_original, color='gray')
    axs[0, 1].set_title('Histograma Original')

    axs[1, 0].imshow(roi_ajustada, cmap='gray', vmin=0, vmax=255)
    axs[1, 0].set_title(f'ROI Ajustada (alpha={alpha}, beta={beta})')
    axs[1, 1].plot(hist_ajustada, color='gray')
    axs[1, 1].set_title('Histograma Ajustado')

    plt.tight_layout()
    plt.show()
    
    
    # Lo que hace el escalado por rango de interes es un ajuste lineal agarrando el valor minimo, anclandolo e 0 y el valor máximo en 255. Los píxeles intermedos se separas de manera proporcional
    # El histograma realiza un ajuste no lineal más agresivo, donde se busca cuáles son los tonos de gris más repetidos y los separa para que abarquen más niveles de luz.
    # La ecualización global divide la imagen en cuadrículas pequeñas y calcula un histograma y ecualiza cada cuadradito por separado
    roi_escalada = cv2.normalize(roi_gris, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    roi_ecualizada = cv2.equalizeHist(roi_gris)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    roi_clahe = clahe.apply(roi_gris)

    hist_escalada = cv2.calcHist([roi_escalada], [0], None, [256], [0, 256])
    hist_ecualizada = cv2.calcHist([roi_ecualizada], [0], None, [256], [0, 256])
    hist_clahe = cv2.calcHist([roi_clahe], [0], None, [256], [0, 256])

    # Aumentamos la altura de figsize (de 12 a 15)
    fig2, axs2 = plt.subplots(3, 2, figsize=(12, 15))

    axs2[0, 0].imshow(roi_escalada, cmap='gray', vmin=0, vmax=255)
    axs2[0, 0].set_title('Escalado por Rango de Interés')
    axs2[0, 1].plot(hist_escalada, color='gray')
    axs2[0, 1].set_title('Histograma Escalado')

    axs2[1, 0].imshow(roi_ecualizada, cmap='gray', vmin=0, vmax=255)
    axs2[1, 0].set_title('Ecualización de Histograma')
    axs2[1, 1].plot(hist_ecualizada, color='gray')
    axs2[1, 1].set_title('Histograma Ecualizado')

    axs2[2, 0].imshow(roi_clahe, cmap='gray', vmin=0, vmax=255)
    axs2[2, 0].set_title('Ecualización Local (CLAHE)')
    axs2[2, 1].plot(hist_clahe, color='gray')
    axs2[2, 1].set_title('Histograma CLAHE')

    # Agregamos h_pad para forzar espacio vertical entre subplots
    plt.tight_layout(h_pad=3.0)
    plt.show()

def procesar_modulo_1_3(frame, x, y, w, h):
    roi = frame[y:y+h, x:x+w]
    roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_suavizada = cv2.GaussianBlur(roi_gris, (5, 5), 0) # Le aplicamos un filtro gausseano dado que los detectores de bordes son extremadamente sensibles al ruido de alta frecuencia

    # Roberts y Prewitt no tienen funciones nativas en OpenCV, por lo tanto para implementarlas hay que pasarle manualmente los kernels.
    kernel_roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    roberts_x = cv2.filter2D(roi_suavizada, cv2.CV_64F, kernel_roberts_x)
    roberts_y = cv2.filter2D(roi_suavizada, cv2.CV_64F, kernel_roberts_y)
    roberts = cv2.convertScaleAbs(cv2.magnitude(roberts_x, roberts_y))

    kernel_prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernel_prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    prewitt_x = cv2.filter2D(roi_suavizada, cv2.CV_64F, kernel_prewitt_x)
    prewitt_y = cv2.filter2D(roi_suavizada, cv2.CV_64F, kernel_prewitt_y)
    prewitt = cv2.convertScaleAbs(cv2.magnitude(prewitt_x, prewitt_y))

    sobel_x = cv2.Sobel(roi_suavizada, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(roi_suavizada, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))

    laplaciano = cv2.convertScaleAbs(cv2.Laplacian(roi_suavizada, cv2.CV_64F))

    canny = cv2.Canny(roi_suavizada, 50, 150)

    fig, axs = plt.subplots(3, 2, figsize=(12, 16))

    axs[0, 0].imshow(roi_suavizada, cmap='gray', vmin=0, vmax=255)
    axs[0, 0].set_title('ROI Pre-procesada (Gaussiano 5x5)')

    axs[0, 1].imshow(roberts, cmap='gray', vmin=0, vmax=255)
    axs[0, 1].set_title('Bordes: Roberts')

    axs[1, 0].imshow(prewitt, cmap='gray', vmin=0, vmax=255)
    axs[1, 0].set_title('Bordes: Prewitt')

    axs[1, 1].imshow(sobel, cmap='gray', vmin=0, vmax=255)
    axs[1, 1].set_title('Bordes: Sobel')

    axs[2, 0].imshow(laplaciano, cmap='gray', vmin=0, vmax=255)
    axs[2, 0].set_title('Bordes: Laplaciano')

    axs[2, 1].imshow(canny, cmap='gray', vmin=0, vmax=255)
    axs[2, 1].set_title('Bordes: Canny (50-150)')

    plt.tight_layout(h_pad=3.0, w_pad=2.0)
    plt.show()
    
    
    # Ahora le aplicamos la TFD-2D
    f = np.fft.fft2(roi_gris)
    fshift = np.fft.fftshift(f)

    # Cálculo de Magnitud (en escala logarítmica para visualización)
    espectro_magnitud = 20 * np.log(np.abs(fshift))
    
    # Cálculo de Fase (ángulos en radianes)
    espectro_fase = np.angle(fshift)

    fig3, axs3 = plt.subplots(1, 3, figsize=(15, 5))

    axs3[0].imshow(roi_gris, cmap='gray', vmin=0, vmax=255)
    axs3[0].set_title('ROI Original')

    axs3[1].imshow(espectro_magnitud, cmap='gray')
    axs3[1].set_title('Espectro de Magnitud')

    axs3[2].imshow(espectro_fase, cmap='gray')
    axs3[2].set_title('Espectro de Fase')

    plt.tight_layout()
    plt.show()

    # Ahora realizamos la reconstrucción de la imagen
    f = np.fft.fft2(roi_gris)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    phase = np.angle(fshift)

    f_solo_mag = mag * np.exp(1j * np.zeros_like(phase))
    img_recon_mag = np.abs(np.fft.ifft2(np.fft.ifftshift(f_solo_mag)))

    f_solo_phase = np.ones_like(mag) * np.exp(1j * phase)
    img_recon_phase = np.abs(np.fft.ifft2(np.fft.ifftshift(f_solo_phase)))

    roi_suavizada = cv2.GaussianBlur(roi_gris, (5, 5), 0)
    f_suav = np.fft.fft2(roi_suavizada)
    mag_suav = np.abs(np.fft.fftshift(f_suav))
    
    f_cruzada = mag_suav * np.exp(1j * phase)
    img_recon_cruzada = np.abs(np.fft.ifft2(np.fft.ifftshift(f_cruzada)))

    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    axs[0].imshow(roi_gris, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title('Original')
    axs[1].imshow(img_recon_mag, cmap='gray')
    axs[1].set_title('Solo Magnitud')
    axs[2].imshow(img_recon_phase, cmap='gray')
    axs[2].set_title('Solo Fase')
    axs[3].imshow(img_recon_cruzada, cmap='gray')
    axs[3].set_title('Cruzada (Mag Suav + Fase Orig)')
    plt.tight_layout()
    plt.show()

    # La fase es la que conserva la información estructural

def procesar_modulo_2(frame, x, y, w, h):
    roi = frame[y:y+h, x:x+w]
    roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Preprocesamiento para reducir ruidos antes de segmentar
    blur = cv2.GaussianBlur(roi_gris, (5, 5), 0)

    # ==========================================================
    # 2.1 Algoritmos de Segmentación
    # ==========================================================

    # 1. Umbralización Automática de Otsu
    # Calcula el umbral óptimo global. Usamos THRESH_BINARY_INV para que la pupila 
    # (que es la parte más oscura) quede en blanco (255) y el fondo negro (0).
    # Lo que sea más oscuro que el umbral (pupila) va de negro, lo que sea más claro va de blanco
    ret, mascara_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Agrupamiento K-Means
    # Agrupamos los píxeles según su intensidad. Transformamos la ROI en una lista de píxeles 1D.
    # Muy oscuros --> pupila
    # medios --> iris
    # claros --> reflejos o esclerótica
    # Nos quedamos con los más oscuros (pupila)
    pixeles = roi_gris.reshape((-1, 1))
    pixeles = np.float32(pixeles)
    # Criterio de parada del algoritmo K-Means
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    K = 3  # Probamos con 3 grupos: pupila (oscuro), iris (medio), reflejos/esclerótica (claro)
    _, etiquetas, centros = cv2.kmeans(pixeles, K, None, criterio, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Reconstruimos la imagen con los centros encontrados
    centros = np.uint8(centros)
    kmeans_resultado = centros[etiquetas.flatten()]
    imagen_kmeans = kmeans_resultado.reshape((roi_gris.shape))
    
    # Aislamos el grupo más oscuro asumiendo que es la pupila
    centro_mas_oscuro = int(np.min(centros))
    mascara_kmeans = cv2.inRange(imagen_kmeans, centro_mas_oscuro, centro_mas_oscuro)

    # 3. Contornos Activos (Snakes)
    # Definimos un círculo inicial que rodee el centro de la pupila
    s = np.linspace(0, 2*np.pi, 100)
    centro_x, centro_y = w//2, h//2
    radio = min(w, h)//4
    init_x = centro_x + radio * np.cos(s)
    init_y = centro_y + radio * np.sin(s)
    contorno_inicial = np.array([init_y, init_x]).T # Formato (fila, columna)
    
    # Preparamos la imagen normalizándola para el algoritmo Snake
    roi_float = img_as_float(roi_gris)
    roi_float_blur = gaussian(roi_float, 3, preserve_range=False)
    # El algoritmo ajusta iterativamente las fuerzas para pegarse al borde de la pupila
    snake = active_contour(roi_float_blur, contorno_inicial, alpha=0.015, beta=10, gamma=0.001)

    # 4. Watershed (Cuenca Hidrográfica)
    # Aplicamos segmentación basada en inundación para separar pupila de reflejos
    kernel_ws = np.ones((3,3), np.uint8)
    # Buscamos áreas que seguramente son fondo
    fondo_seguro = cv2.dilate(mascara_otsu, kernel_ws, iterations=3)
    # Buscamos áreas que seguramente son pupila (Distance Transform)
    dist_transform = cv2.distanceTransform(mascara_otsu, cv2.DIST_L2, 5)
    _, pupila_segura = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    pupila_segura = np.uint8(pupila_segura)
    
    # Área de borde desconocida
    zona_desconocida = cv2.subtract(fondo_seguro, pupila_segura)
    
    # Etiquetamos marcadores
    _, marcadores = cv2.connectedComponents(pupila_segura)
    marcadores = marcadores + 1 # El fondo queda en 1 en vez de 0
    marcadores[zona_desconocida == 255] = 0 # La zona dudosa queda en 0
    
    roi_watershed = roi.copy()
    marcadores = cv2.watershed(roi_watershed, marcadores)
    roi_watershed[marcadores == -1] = [0, 0, 255] # Pintamos los bordes Watershed de rojo

    # --- Visualización 2.1 ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0,0].imshow(mascara_otsu, cmap='gray'); axs[0,0].set_title('1. Otsu (Global)')
    axs[0,1].imshow(mascara_kmeans, cmap='gray'); axs[0,1].set_title(f'2. K-Means (K={K})')
    
    axs[1,0].imshow(roi_gris, cmap='gray')
    axs[1,0].plot(contorno_inicial[:, 1], contorno_inicial[:, 0], '--r', lw=2, label='Inicial')
    axs[1,0].plot(snake[:, 1], snake[:, 0], '-b', lw=2, label='Snake')
    axs[1,0].legend(); axs[1,0].set_title('3. Contornos Activos (Snakes)')
    
    # OpenCV usa BGR, Matplotlib usa RGB, por eso el [...,::-1]
    axs[1,1].imshow(roi_watershed[...,::-1]); axs[1,1].set_title('4. Watershed (Borde Rojo)')
    plt.tight_layout()
    plt.show()

    # ==========================================================
    # 2.2 Morfología Matemática para Limpieza de Máscara
    # ==========================================================
    
    # Tomamos la máscara de Otsu como punto de partida para limpiar
    elemento_estructurante = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Erosión: Achica el objeto blanco, puede borrar ruido pequeño pero reduce la pupila
    erosion = cv2.erode(mascara_otsu, elemento_estructurante, iterations=1)
    
    # Dilatación: Agranda el objeto blanco, rellena huecos pero engrosa el borde
    dilatacion = cv2.dilate(mascara_otsu, elemento_estructurante, iterations=1)
    
    # Apertura (Erosión seguida de Dilatación): Excelente para eliminar pestañas o ruido externo
    apertura = cv2.morphologyEx(mascara_otsu, cv2.MORPH_OPEN, elemento_estructurante, iterations=2)
    
    # Cierre (Dilatación seguida de Erosión): Excelente para rellenar los destellos corneales (flash) dentro de la pupila
    cierre = cv2.morphologyEx(mascara_otsu, cv2.MORPH_CLOSE, elemento_estructurante, iterations=3)
    
    # Limpieza Óptima Combinada: Primero cerramos huecos (destellos) y luego limpiamos ruido externo (pestañas/sombras)
    mascara_limpia = cv2.morphologyEx(cierre, cv2.MORPH_OPEN, elemento_estructurante, iterations=2)

    # Esqueletización del contorno resultante
    # Convertimos a formato booleano (True/False) que es lo que requiere skimage
    mascara_bool = mascara_limpia > 0
    esqueleto = skeletonize(mascara_bool)

    # --- Visualización 2.2 ---
    fig2, axs2 = plt.subplots(2, 3, figsize=(15, 10))
    axs2[0,0].imshow(mascara_otsu, cmap='gray'); axs2[0,0].set_title('Máscara Base (Otsu)')
    axs2[0,1].imshow(erosion, cmap='gray'); axs2[0,1].set_title('Erosión')
    axs2[0,2].imshow(dilatacion, cmap='gray'); axs2[0,2].set_title('Dilatación')
    
    axs2[1,0].imshow(apertura, cmap='gray'); axs2[1,0].set_title('Apertura (Saca ruido externo)')
    axs2[1,1].imshow(cierre, cmap='gray'); axs2[1,1].set_title('Cierre (Rellena destello de flash)')
    axs2[1,2].imshow(esqueleto, cmap='gray'); axs2[1,2].set_title('Esqueletización de la máscara limpia')
    
    plt.tight_layout()
    plt.show()

# ==================================
# Modificando el nombre del video en el main ya está
def main():
    video_original = 'registro_pupila.mp4'
    video_trabajo = 'registro_pupila_copia.mp4'
    
    if not os.path.exists(video_trabajo):
        shutil.copy(video_original, video_trabajo)
    
    
    # -------------------Anidación de funciones---------------------        
    frame_representativo = obtener_mejor_frame(video_trabajo)
    
    if frame_representativo is not None:
        x, y, w, h = obtener_roi_automatica(frame_representativo)
        # procesar_modulo_1_1(frame_representativo, x, y, w, h)
        # procesar_modulo_1_2(frame_representativo, x, y, w, h)
        procesar_modulo_1_3(frame_representativo, x, y, w, h)
        procesar_modulo_2(frame_representativo, x, y, w, h)    
    



if __name__ == '__main__':
    main()