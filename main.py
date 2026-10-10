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
# IMPORTACIONES ADICIONALES PARA EL MÓDULO 3
from scipy.stats import skew, entropy
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from matplotlib.patches import Ellipse
# IMPORTACIONES PARA MÓDULO 4
import heapq
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
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
    
    return roi, roi_clahe

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

def procesar_modulo_2(roi):
    # Extraemos los canales y nos quedamos solo con el Rojo (R)
    b, g, r = cv2.split(roi)
    
    # Aplicamos un difuminado agresivo para borrar cualquier textura residual del iris
    blur = cv2.GaussianBlur(r, (15, 15), 0)
    
    h, w = roi.shape[:2] 

    # ==========================================================
    # 2.1 Algoritmos de Segmentación
    # ==========================================================

    # 1. Umbralización Automática de Otsu (ahora funcionará perfecto sobre el canal R)
    ret, mascara_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Agrupamiento K-Means
    pixeles = blur.reshape((-1, 1))
    pixeles = np.float32(pixeles)
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    K = 5
    _, etiquetas, centros = cv2.kmeans(pixeles, K, None, criterio, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centros = np.uint8(centros)
    imagen_kmeans = centros[etiquetas.flatten()].reshape((blur.shape))
    centro_mas_oscuro = int(np.min(centros))
    mascara_kmeans = cv2.inRange(imagen_kmeans, centro_mas_oscuro, centro_mas_oscuro)

    # 3. Contornos Activos (Snakes)
    s = np.linspace(0, 2*np.pi, 100)
    centro_x, centro_y = w//2, h//2
    # Radio minúsculo para asegurar que inicialice completamente dentro de la pupila
    radio = min(w, h) // 10 
    init_x = centro_x + radio * np.cos(s)
    init_y = centro_y + radio * np.sin(s)
    contorno_inicial = np.array([init_y, init_x]).T 
    
    # Le pasamos la imagen difuminada del canal rojo
    roi_float = img_as_float(blur) 
    snake = active_contour(roi_float, contorno_inicial, alpha=0.01, beta=10, gamma=0.001)

    # 4. Watershed (Cuenca Hidrográfica)
    kernel_ws = np.ones((3,3), np.uint8)
    
    # Usamos la máscara de K-Means en lugar de Otsu para definir las áreas seguras
    fondo_seguro = cv2.dilate(mascara_kmeans, kernel_ws, iterations=3)
    
    dist_transform = cv2.distanceTransform(mascara_kmeans, cv2.DIST_L2, 5)
    _, pupila_segura = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    pupila_segura = np.uint8(pupila_segura)
    
    zona_desconocida = cv2.subtract(fondo_seguro, pupila_segura)
    
    _, marcadores = cv2.connectedComponents(pupila_segura)
    marcadores = marcadores + 1 
    marcadores[zona_desconocida == 255] = 0 
    
    roi_watershed = roi.copy()
    marcadores = cv2.watershed(roi_watershed, marcadores)
    roi_watershed[marcadores == -1] = [0, 0, 255]

    # --- Visualización 2.1 ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0,0].imshow(mascara_otsu, cmap='gray'); axs[0,0].set_title('1. Otsu (Canal Rojo)')
    axs[0,1].imshow(mascara_kmeans, cmap='gray'); axs[0,1].set_title(f'2. K-Means (K={K})')
    
    # Mostramos el canal rojo de fondo para que veas cómo se aclaró el iris
    axs[1,0].imshow(r, cmap='gray') 
    axs[1,0].plot(contorno_inicial[:, 1], contorno_inicial[:, 0], '--r', lw=2, label='Inicial')
    axs[1,0].plot(snake[:, 1], snake[:, 0], '-b', lw=2, label='Snake')
    axs[1,0].legend(); axs[1,0].set_title('3. Contornos Activos (Snakes)')
    
    axs[1,1].imshow(roi_watershed[...,::-1]); axs[1,1].set_title('4. Watershed (Borde Rojo)')
    plt.tight_layout()
    plt.show()

    # ==========================================================
    # 2.2 Morfología Matemática para Limpieza de Máscara
    # ==========================================================
    
    # Descartamos Otsu y usamos un umbral estricto (< 45) para capturar solo el negro profundo de la pupila
    _, mascara_base = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY_INV)
    
    elemento_estructurante = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Aplicamos las operaciones sobre la nueva máscara de la pupila
    erosion = cv2.erode(mascara_base, elemento_estructurante, iterations=1)
    dilatacion = cv2.dilate(mascara_base, elemento_estructurante, iterations=1)
    
    apertura = cv2.morphologyEx(mascara_base, cv2.MORPH_OPEN, elemento_estructurante, iterations=2)
    cierre = cv2.morphologyEx(mascara_base, cv2.MORPH_CLOSE, elemento_estructurante, iterations=3)
    
    # Limpieza Óptima Combinada
    mascara_limpia = cv2.morphologyEx(cierre, cv2.MORPH_OPEN, elemento_estructurante, iterations=2)

    # Extraer exclusivamente el anillo exterior (Gradiente Morfológico)
    contorno_pupila = cv2.morphologyEx(mascara_limpia, cv2.MORPH_GRADIENT, elemento_estructurante)

    # Esqueletización sobre el anillo
    mascara_bool = contorno_pupila > 0
    esqueleto = skeletonize(mascara_bool)

    # --- Visualización 2.2 ---
    fig2, axs2 = plt.subplots(2, 3, figsize=(15, 10))
    axs2[0,0].imshow(mascara_base, cmap='gray'); axs2[0,0].set_title('Máscara Base (Umbral < 45)')
    axs2[0,1].imshow(erosion, cmap='gray'); axs2[0,1].set_title('Erosión')
    axs2[0,2].imshow(dilatacion, cmap='gray'); axs2[0,2].set_title('Dilatación')
    
    axs2[1,0].imshow(apertura, cmap='gray'); axs2[1,0].set_title('Apertura (Saca ruido externo)')
    axs2[1,1].imshow(cierre, cmap='gray'); axs2[1,1].set_title('Cierre (Rellena destello de flash)')
    axs2[1,2].imshow(esqueleto, cmap='gray'); axs2[1,2].set_title('Esqueletización del contorno')
    
    plt.tight_layout()
    plt.show()
    
    
    return mascara_limpia

def procesar_modulo_3_1(roi, mascara_limpia):
    """
    Extracción de Descriptores Morfométricos y de Textura en el frame segmentado.
    Requiere que procesar_modulo_2() retorne 'mascara_limpia'.
    """
    # 1. Descriptores Morfométricos (Forma)
    contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        print("No se detectó pupila en la máscara.")
        return
        
    c = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimetro = cv2.arcLength(c, True)
    
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
    cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
    
    # Estimación de circularidad
    circularidad = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0
    
    print("\n--- 3.1.1 Descriptores Morfométricos ---")
    print(f"Área (A): {area} px")
    print(f"Perímetro (P): {perimetro:.2f} px")
    print(f"Centroide (X_c, Y_c): ({cX}, {cY})")
    print(f"Circularidad: {circularidad:.4f}")

    # 2. Textura de Primer Orden (Estadística Global)
    b, g, r = cv2.split(roi)
    canales = {'Rojo (R)': r, 'Verde (G)': g, 'Azul (B)': b}
    
    print("\n--- 3.1.2 Textura de Primer Orden ---")
    for nombre, canal in canales.items():
        # Extraer solo píxeles dentro de la ROI segmentada (pupila/iris)
        pixeles = canal[mascara_limpia > 0]
        if len(pixeles) == 0:
            continue
            
        media = np.mean(pixeles)
        desvio = np.std(pixeles)
        asimetria = skew(pixeles)
        
        hist, _ = np.histogram(pixeles, bins=256, range=(0,256), density=True)
        ent = entropy(hist + 1e-9) # 1e-9 para evitar log(0)
        
        print(f"{nombre} -> Media: {media:.2f} | Std: {desvio:.2f} | Skewness: {asimetria:.2f} | Entropía: {ent:.2f}")

    # 3. Textura de Segundo Orden (GLCM)
    print("\n--- 3.1.3 Textura de Segundo Orden (GLCM) ---")
    for nombre, canal in canales.items():
        # Cuantización a 32 niveles de gris
        canal_cuantizado = (canal // 8).astype(np.uint8)
        
        # Calcular GLCM (distancia=1, 4 ángulos)
        glcm = graycomatrix(canal_cuantizado, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                            levels=32, symmetric=True, normed=True)
        
        contraste = graycoprops(glcm, 'contrast').mean()
        homogeneidad = graycoprops(glcm, 'homogeneity').mean()
        energia = graycoprops(glcm, 'energy').mean()
        correlacion = graycoprops(glcm, 'correlation').mean()
        
        print(f"{nombre} -> Contraste: {contraste:.4f} | Homogeneidad: {homogeneidad:.4f} | Energía: {energia:.4f} | Correlación: {correlacion:.4f}")


# ----------------------------------------------------------------------
# 1) SEGMENTACIÓN FRAME A FRAME
#    En vez de Otsu global + filtros de circularidad (que cambian de umbral
#    en cada frame cuando el flash se apaga), se busca el BORDE de la pupila
#    lanzando rayos desde su centro y ubicando el máximo gradiente
#    oscuro->claro en cada uno. Luego se ajusta una elipse robusta.
#    Es invariante al brillo/exposición, y tolera reflejos y pestañas.
# ----------------------------------------------------------------------
def _parametros_segmentacion(area_min, area_max):
    r_min = np.sqrt(area_min / np.pi)
    r_max = np.sqrt(area_max / np.pi)
    return {
        'area_min': area_min,
        'area_max': area_max,
        'r_lo': max(2.0, 0.8 * r_min),        # inicio de cada rayo (px)
        'r_hi': 1.15 * r_max,                 # fin de cada rayo (px)
        'H': int(np.ceil(3 * r_max)),         # semilado de la ventana de trabajo
        'Hv': int(np.ceil(2.5 * r_max)),      # semilado del recorte para verificación visual
        'radio_busqueda': 0.8 * r_max,        # cuánto puede moverse el centro entre frames
        'k_centro': int(2 * round(r_min)) | 1,  # lado (impar) del promedio para hallar la zona oscura
        'sigma': 1.2,                         # suavizado previo (px)
        'contraste_min': 4.0,                 # salto mínimo (niveles de gris) para aceptar un borde
        'min_puntos': 14,                     # mínimo de rayos válidos (de 48)
    }
 
 
def _rellenar_reflejos(canal_u8, umbral=235, area_max_reflejo=150):
    """Inpainting de los reflejos especulares chicos del flash (los grandes, como piel saturada, se dejan)."""
    mascara = (canal_u8 >= umbral).astype(np.uint8)
    if not mascara.any():
        return canal_u8
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    chica = np.zeros_like(mascara)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] <= area_max_reflejo:
            chica[etiquetas == i] = 1
    if not chica.any():
        return canal_u8
    chica = cv2.dilate(chica, np.ones((3, 3), np.uint8), iterations=2)
    return cv2.inpaint(canal_u8, chica, 3, cv2.INPAINT_TELEA)
 
 
def _ajustar_elipse(img, c, par, n_rayos=48):
    """Dado un centro aproximado c, busca el borde de la pupila por rayos y ajusta una elipse."""
    radios = np.arange(par['r_lo'], par['r_hi'], 0.5)
    ang = np.linspace(0, 2 * np.pi, n_rayos, endpoint=False)
    xs = (c[0] + np.outer(np.cos(ang), radios)).astype(np.float32)
    ys = (c[1] + np.outer(np.sin(ang), radios)).astype(np.float32)
 
    perfiles = cv2.remap(img, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    perfiles = gaussian_filter1d(perfiles, sigma=1.0, axis=1)
    grad = np.gradient(perfiles, axis=1)
 
    idx = np.argmax(grad, axis=1)                       # máximo salto oscuro -> claro
    ult = len(radios) - 1
    filas = np.arange(n_rayos)
    salto = perfiles[filas, np.minimum(idx + 4, ult)] - perfiles[filas, np.maximum(idx - 4, 0)]
    r_borde = radios[idx]
 
    ok = (salto >= par['contraste_min']) & (idx > 0) & (idx < ult)
    if ok.sum() < par['min_puntos']:
        return None
 
    # Descarta puntos que no son consistentes con el resto (pestañas, reflejos, párpado)
    rm = np.median(r_borde[ok])
    ok &= np.abs(r_borde - rm) <= max(1.5, 0.25 * rm)
    if ok.sum() < par['min_puntos']:
        return None
 
    # El borde debe verse alrededor de la pupila (no solo en un costado)
    if len(np.unique((ang[ok] // (np.pi / 4)).astype(int))) < 5:
        return None
 
    px = c[0] + r_borde[ok] * np.cos(ang[ok])
    py = c[1] + r_borde[ok] * np.sin(ang[ok])
    pts = np.stack([px, py], axis=1).astype(np.float32)
    (ex, ey), (d1, d2), angulo = cv2.fitEllipse(pts)
 
    if min(d1, d2) / max(d1, d2) < 0.6:                 # elipse demasiado alargada -> usar círculo
        elipse = ((float(c[0]), float(c[1])), (2 * rm, 2 * rm), 0.0)
        area = np.pi * rm ** 2
    else:
        elipse = ((ex, ey), (d1, d2), angulo)
        area = np.pi / 4.0 * d1 * d2
 
    if not (par['area_min'] <= area <= par['area_max']):
        return None
 
    return {'area': float(area), 'centro': (ex, ey) if min(d1, d2) / max(d1, d2) >= 0.6 else (c[0], c[1]),
            'elipse': elipse, 'calidad': float(ok.sum() / n_rayos)}
 
 
def _medir_pupila(c_img, centro_prev, par):
    """Mide la pupila en un frame, buscando cerca de la posición del frame anterior."""
    H = min(par['H'], min(c_img.shape) // 2)
    h_roi, w_roi = c_img.shape
    cx0, cy0 = int(round(centro_prev[0])), int(round(centro_prev[1]))
    x0 = int(np.clip(cx0 - H, 0, max(0, w_roi - 2 * H)))
    y0 = int(np.clip(cy0 - H, 0, max(0, h_roi - 2 * H)))
 
    ventana = np.ascontiguousarray(c_img[y0:y0 + 2 * H, x0:x0 + 2 * H])
    ventana = _rellenar_reflejos(ventana)
    suave = cv2.GaussianBlur(ventana.astype(np.float32), (0, 0), par['sigma'])
 
    # Centro aproximado: zona más oscura cerca del centro anterior
    k = par['k_centro']
    medio = cv2.blur(suave, (k, k))
    mascara = np.zeros(medio.shape, np.uint8)
    cv2.circle(mascara, (cx0 - x0, cy0 - y0), int(par['radio_busqueda']), 255, -1)
    _, _, min_loc, _ = cv2.minMaxLoc(medio, mascara)
    c = np.array(min_loc, dtype=float)
 
    mejor = None
    for _ in range(3):                                   # refinamiento iterativo del centro
        res = _ajustar_elipse(suave, c, par)
        if res is None:
            break
        mejor = res
        c = np.array(res['centro'], dtype=float)
    if mejor is None:
        return None
 
    # Volver a coordenadas de la ROI completa
    (ex, ey), (d1, d2), ang = mejor['elipse']
    mejor['elipse'] = ((ex + x0, ey + y0), (d1, d2), ang)
    mejor['centro'] = (mejor['centro'][0] + x0, mejor['centro'][1] + y0)
    return mejor
 
 
def _centro_inicial(c_img, par):
    suave = cv2.GaussianBlur(c_img.astype(np.float32), (0, 0), par['sigma'] + 1)
    medio = cv2.blur(suave, (par['k_centro'], par['k_centro']))
    _, _, min_loc, _ = cv2.minMaxLoc(medio)
    return np.array(min_loc, dtype=float)
 
 
def _roi_estable(video_path, fps, t_inicio, n_frames=8):
    """ROI automática usando la mediana de varios frames (más estable que usar solo el primero)."""
    cap = cv2.VideoCapture(video_path)
    rois, idx = [], -1
    while len(rois) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx / fps < t_inicio:
            continue
        rois.append(obtener_roi_automatica(frame))
    cap.release()
    if not rois:
        return None
    rois = np.array(rois)
    x, y = int(np.median(rois[:, 0])), int(np.median(rois[:, 1]))
    return x, y, int(rois[0, 2]), int(rois[0, 3])
 
 
def _recorte_visual(roi, centro, res, par):
    Hv = min(par['Hv'], min(roi.shape[:2]) // 2)
    x0 = int(np.clip(int(centro[0]) - Hv, 0, max(0, roi.shape[1] - 2 * Hv)))
    y0 = int(np.clip(int(centro[1]) - Hv, 0, max(0, roi.shape[0] - 2 * Hv)))
    recorte = roi[y0:y0 + 2 * Hv, x0:x0 + 2 * Hv].copy()
    elipse = None
    if res is not None:
        (ex, ey), ejes, ang = res['elipse']
        elipse = ((ex - x0, ey - y0), ejes, ang)
    return recorte, elipse
 
 
# ----------------------------------------------------------------------
# 2) ARTEFACTOS: detección, interpolación y suavizado de la serie A(t)
# ----------------------------------------------------------------------
def _limpiar_serie(area_cruda, fps, n_sigma=3.0):
    """
    Artefactos tratados:
      - Parpadeos / oclusión parcial / movimiento brusco: el frame no pasa los controles
        de calidad de la segmentación -> NaN.
      - Mediciones aisladas incoherentes con sus vecinas: filtro de Hampel (mediana +- 3 MAD).
      - Los huecos se rellenan con interpolación lineal.
      - Suavizado final con Savitzky-Golay (conserva la forma de la dilatación mejor que un gaussiano).
    """
    n = len(area_cruda)
    idx = np.arange(n)
    x = area_cruda.copy()
 
    valido = ~np.isnan(x)
    if valido.sum() < 5:
        return None, None, None
    x_i = np.interp(idx, idx[valido], x[valido])
 
    k = max(4, int(round(0.2 * fps)))                    # semiventana de Hampel (~0.2 s)
    atipico = np.zeros(n, dtype=bool)
    for i in range(n):
        v = x_i[max(0, i - k):min(n, i + k + 1)]
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med))
        if abs(x_i[i] - med) > n_sigma * max(mad, 0.03 * med):
            atipico[i] = True
 
    es_artefacto = np.isnan(x) | atipico
    ok = ~es_artefacto
    if ok.sum() < 5:
        return None, None, None
    x_limpia = np.interp(idx, idx[ok], x[ok])
 
    w = int(round(0.3 * fps)) | 1
    w = max(5, min(w, n if n % 2 == 1 else n - 1))
    x_suave = savgol_filter(x_limpia, w, 2)
    return x_limpia, x_suave, es_artefacto
 
 
# ----------------------------------------------------------------------
# 3) ESTÍMULO Y PARÁMETROS FISIOLÓGICOS
# ----------------------------------------------------------------------
def _detectar_estimulo(brillo, fps, idx_forzado=None):
    """Escalón de brillo: compara el promedio de ~0.15 s antes vs después de cada frame."""
    n = len(brillo)
    k = max(2, int(round(0.15 * fps)))
    score = np.zeros(n)
    for i in range(k, n - k + 1):
        score[i] = np.mean(brillo[i - k:i]) - np.mean(brillo[i:i + k])
    i = int(np.argmax(np.abs(score))) if idx_forzado is None else int(idx_forzado)
    signo = 1 if score[i] >= 0 else -1                   # brillo baja -> dilatación (+1); sube -> constricción (-1)
    return i, signo, abs(score[i])
 
 
def _cruce(t, s, i0, i1, umbral):
    """Tiempo (interpolado) en que s supera 'umbral', buscando entre i0 e i1."""
    for i in range(max(i0, 1), i1 + 1):
        if s[i] >= umbral:
            if s[i] == s[i - 1]:
                return t[i]
            f = (umbral - s[i - 1]) / (s[i] - s[i - 1])
            return t[i - 1] + f * (t[i] - t[i - 1])
    return np.nan
 
 
def _extraer_parametros(t, a, i_est, signo, fps):
    """
    Trabaja con s = signo * A(t) para que la respuesta siempre sea una "subida"
    (dilatación: signo=+1; constricción: signo=-1).
    """
    s = signo * a
    t_est = t[i_est]
 
    pre = np.where((t >= t_est - 1.0) & (t < t_est - 0.1))[0]
    if len(pre) < 3:
        pre = np.arange(0, max(3, i_est - 1))
    basal = np.median(s[pre])
    ruido = np.std(s[pre])
 
    post = np.arange(i_est, len(s))
    i_ext = int(post[np.argmax(s[post])])
    amp = s[i_ext] - basal
    if amp <= 0:
        return None
 
    umbral_lat = basal + min(max(0.10 * amp, 3 * ruido), 0.30 * amp)
    t_ini = _cruce(t, s, i_est, i_ext, umbral_lat)
    t10 = _cruce(t, s, i_est, i_ext, basal + 0.10 * amp)
    t90 = _cruce(t, s, i_est, i_ext, basal + 0.90 * amp)
 
    w = max(5, int(round(0.3 * fps)) | 1)
    w = min(w, len(s) if len(s) % 2 == 1 else len(s) - 1)
    ds = savgol_filter(s, w, 2, deriv=1, delta=1.0 / fps)
    vel_pico = float(np.max(ds[i_est:i_ext + 1]))
    vel_media = (0.8 * amp / (t90 - t10)) if (not np.isnan(t10) and not np.isnan(t90) and t90 > t10) else np.nan
 
    return {
        'basal': float(signo * basal), 'extremo': float(signo * s[i_ext]),
        'amplitud': float(amp), 'i_ext': i_ext, 't_ext': float(t[i_ext]),
        't_ini': float(t_ini), 'latencia': float(t_ini - t_est),
        't10': float(t10), 't90': float(t90),
        'vel_media': float(vel_media), 'vel_pico': vel_pico,
    }
 
 
def _metricas_error(pares):
    e = np.array([a - m for a, m in pares], dtype=float)
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))
 
 
# ----------------------------------------------------------------------
# 4) FUNCIÓN PRINCIPAL
# ----------------------------------------------------------------------
def procesar_modulo_3_2(video_path, t_inicio=0.0, t_fin=7.5, t_estimulo_manual=None,
                        area_min_px=80, area_max_px=900,
                        manual_contraida=218.0, manual_dilatada=393.0,
                        anotaciones_manuales=None, roi_manual=None,
                        canal=2, etiqueta="Registro B"):
    """
    video_path          : ruta del video.
    t_inicio, t_fin     : ventana de análisis en segundos (tiempo del video). Si el video es el completo
                          y querés saltear los primeros 2 s, usá t_inicio=2.0.
    t_estimulo_manual   : si la detección automática del flash falla, indicá el segundo aprox. (ej. 4.8).
    area_min_px/max_px  : rango físicamente plausible del área de pupila en TU video (px²).
    manual_contraida/dilatada : tus mediciones manuales (218 y 393 px²).
    anotaciones_manuales: opcional, {nro_de_frame: area_manual_px} para MAE/RMSE frame a frame.
    roi_manual          : opcional, (x, y, w, h) para forzar la ROI.
    canal               : 2 = rojo, 1 = verde, 0 = azul, None = gris.
    Devuelve un dict con los parámetros (útil para calcular variabilidad entre registros).
    """
    print(f"\n--- 3.2 Procesando video ({etiqueta}) ---")
 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0
 
    # ---- ROI ----
    if roi_manual is not None:
        x, y, w, h = roi_manual
    else:
        roi_info = _roi_estable(video_path, fps, t_inicio)
        if roi_info is None:
            print("No se pudo determinar la ROI.")
            return None
        x, y, w, h = roi_info
    par = _parametros_segmentacion(area_min_px, area_max_px)
    print(f"FPS: {fps:.2f} | ROI: x={x}, y={y}, w={w}, h={h} | rango de área válido: {area_min_px}-{area_max_px} px²")
 
    # ---- Frame a frame ----
    cap = cv2.VideoCapture(video_path)
    tiempos, brillos, areas, calidades, visuales = [], [], [], [], []
    frames_idx = []
    centro, centro_ref, fallos = None, None, 0
    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t = frame_idx / fps
        if t < t_inicio:
            continue
        if t > t_fin:
            break
 
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brillos.append(cv2.mean(gris)[0])                # brillo global -> detecta el apagado del flash
 
        roi = frame[y:y + h, x:x + w]
        c_img = np.ascontiguousarray(roi[:, :, canal]) if canal in (0, 1, 2) else np.ascontiguousarray(gris[y:y + h, x:x + w])
 
        if centro is None:
            centro = _centro_inicial(c_img, par)
            centro_ref = centro.copy()
 
        res = _medir_pupila(c_img, centro, par)
        if res is not None:
            centro = np.array(res['centro'])
            fallos = 0
            areas.append(res['area'])
            calidades.append(res['calidad'])
        else:
            fallos += 1
            areas.append(np.nan)
            calidades.append(0.0)
            if fallos >= 10:                             # se perdió el seguimiento: volver al centro inicial
                centro = centro_ref.copy()
 
        visuales.append(_recorte_visual(roi, centro, res, par))
        tiempos.append(t)
        frames_idx.append(frame_idx)
    cap.release()
 
    if len(areas) < 10:
        print("Muy pocos frames dentro de la ventana de análisis.")
        return None
 
    t = np.array(tiempos)
    a_cruda = np.array(areas, dtype=float)
    brillo = np.array(brillos, dtype=float)
 
    # ---- Limpieza de artefactos ----
    a_limpia, a_suave, es_art = _limpiar_serie(a_cruda, fps)
    if a_limpia is None:
        print("La segmentación falló en casi todos los frames. Revisá la ROI o el rango area_min_px/area_max_px.")
        return None
    n_sin_det = int(np.isnan(a_cruda).sum())
    n_hampel = int(es_art.sum() - n_sin_det)
    print(f"Frames analizados: {len(t)} | sin detección fiable (parpadeo/oclusión/baja confianza): {n_sin_det} "
          f"| atípicos (Hampel): {n_hampel} | interpolados en total: {int(es_art.sum())} "
          f"({100 * es_art.mean():.1f} %)")
 
    # ---- Estímulo ----
    idx_forzado = int(np.argmin(np.abs(t - t_estimulo_manual))) if t_estimulo_manual is not None else None
    i_est, signo, fuerza = _detectar_estimulo(brillo, fps, idx_forzado)
    t_est = t[i_est]
    tipo = "Luz a Oscuridad" if signo > 0 else "Oscuridad a Luz"
    if t_estimulo_manual is None and fuerza < 0.03 * np.mean(brillo):
        print("ADVERTENCIA: el escalón de brillo es débil; verificá el instante del estímulo "
              "(podés forzarlo con t_estimulo_manual=...).")
 
    p = _extraer_parametros(t, a_suave, i_est, signo, fps)
    if p is None:
        print("No se observó respuesta pupilar posterior al estímulo.")
        return None
 
    contraida = min(p['basal'], p['extremo'])
    dilatada = max(p['basal'], p['extremo'])
    nombre_resp = "Redilatación" if signo > 0 else "Constricción"
 
    # ---- Validación contra mediciones manuales ----
    pares = [(contraida, manual_contraida), (dilatada, manual_dilatada)]
    mae_niv, rmse_niv = _metricas_error(pares)
    mae_fr = rmse_fr = None
    if anotaciones_manuales:
        pares_fr = []
        for fr, a_man in anotaciones_manuales.items():
            if fr in frames_idx:
                pares_fr.append((a_limpia[frames_idx.index(fr)], a_man))
        if pares_fr:
            mae_fr, rmse_fr = _metricas_error(pares_fr)
 
    # ---------------- Gráfico 1: curva temporal ----------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(t, a_cruda, '.', color='tab:blue', alpha=0.45, ms=4, label='Área bruta A(t) (frames válidos)')
    if es_art.any():
        ax1.plot(t[es_art], a_limpia[es_art], 'x', color='orange', ms=6, label='Artefacto interpolado')
    ax1.plot(t, a_suave, color='red', lw=2, label='Área suavizada')
    ax1.axhline(manual_contraida, color='gray', ls=':', label=f'Manual contraída ({manual_contraida:.0f} px²)')
    ax1.axhline(manual_dilatada, color='gray', ls='-.', label=f'Manual dilatada ({manual_dilatada:.0f} px²)')
    ax1.axvline(t_est, color='black', ls='--', label='Estímulo (cambio del flash)')
    if not np.isnan(p['t_ini']):
        ax1.axvline(p['t_ini'], color='green', ls='--', label=f"Inicio {nombre_resp.lower()}")
    ax1.axvline(p['t_ext'], color='purple', ls='--', label=f"{nombre_resp} máxima")
    ax1.set_title(f'Dinámica Temporal de la Pupila - {etiqueta} ({tipo})')
    ax1.set_ylabel('Área pupilar (px²)')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True)
 
    ax2.plot(t, brillo, color='tab:green')
    ax2.axvline(t_est, color='black', ls='--')
    ax2.set_ylabel('Brillo medio\ndel frame')
    ax2.set_xlabel('Tiempo (s)')
    ax2.grid(True)
    plt.tight_layout()
 
    # ---------------- Gráfico 2: verificación visual de la segmentación ----------------
    sel = np.linspace(0, len(t) - 1, 8).astype(int)
    fig2, axs = plt.subplots(2, 4, figsize=(12, 6.5))
    for ax, i in zip(axs.ravel(), sel):
        recorte, elipse = visuales[i]
        ax.imshow(recorte[..., ::-1])
        if elipse is not None:
            (ex, ey), (d1, d2), ang = elipse
            ax.add_patch(Ellipse((ex, ey), d1, d2, angle=ang, fill=False, color='lime', lw=1.5))
            ax.set_title(f't={t[i]:.2f}s  A={a_cruda[i]:.0f}', fontsize=9)
        else:
            ax.set_title(f't={t[i]:.2f}s  (sin detección)', fontsize=9, color='red')
        ax.axis('off')
    fig2.suptitle('Verificación de la segmentación (elipse verde = pupila detectada)')
    plt.tight_layout()
    plt.show()
 
    # ---------------- Reporte ----------------
    print(f"\n--- Parámetros Fisiológicos Extraídos ({etiqueta}: {tipo}) ---")
    print(f"Tiempo de estímulo: {t_est:.2f} s")
    print(f"Área contraída (auto): {contraida:.1f} px²   | manual: {manual_contraida:.1f} px²")
    print(f"Área dilatada  (auto): {dilatada:.1f} px²   | manual: {manual_dilatada:.1f} px²")
    print(f"Amplitud de respuesta: {p['amplitud']:.1f} px²   | manual: {manual_dilatada - manual_contraida:.1f} px²")
    print(f"Latencia: {p['latencia']:.2f} s   (estímulo -> inicio de {nombre_resp.lower()})")
    print(f"Tiempo hasta {nombre_resp.lower()} máxima: {p['t_ext'] - t_est:.2f} s")
    print(f"Velocidad media de {nombre_resp.lower()} (10-90 %): {p['vel_media']:.1f} px²/s")
    print(f"Velocidad pico de {nombre_resp.lower()}: {p['vel_pico']:.1f} px²/s")
    print(f"\n--- Validación vs. medición manual ---")
    print(f"Niveles (contraída y dilatada): MAE = {mae_niv:.1f} px² | RMSE = {rmse_niv:.1f} px²")
    for nombre, auto, man in (("contraída", contraida, manual_contraida), ("dilatada", dilatada, manual_dilatada)):
        print(f"   {nombre}: error = {auto - man:+.1f} px² ({100 * (auto - man) / man:+.1f} %)")
    if mae_fr is not None:
        print(f"Frames anotados ({len(anotaciones_manuales)}): MAE = {mae_fr:.1f} px² | RMSE = {rmse_fr:.1f} px²")
 
    return {
        'etiqueta': etiqueta, 'tipo': tipo, 'fps': fps, 't_estimulo': float(t_est),
        'area_contraida': contraida, 'area_dilatada': dilatada,
        'amplitud': p['amplitud'], 'latencia': p['latencia'],
        'vel_media': p['vel_media'], 'vel_pico': p['vel_pico'],
        'mae_niveles': mae_niv, 'rmse_niveles': rmse_niv,
        'tiempos': t, 'area_cruda': a_cruda, 'area_suave': a_suave,
    }
    




# ----------------------------------------------------------------------
# 1) COMPRESIÓN SIN PÉRDIDA
# ----------------------------------------------------------------------
def _huffman_longitudes(frecuencias):
    """Longitud del código de Huffman de cada símbolo (algoritmo clásico con heap)."""
    simbolos = [(int(f), s) for s, f in enumerate(frecuencias) if f > 0]
    if len(simbolos) == 1:
        return {simbolos[0][1]: 1}
    heap = [(f, s, [s]) for f, s in simbolos]
    heapq.heapify(heap)
    largo = {s: 0 for _, s in simbolos}
    contador = 256
    while len(heap) > 1:
        f1, _, g1 = heapq.heappop(heap)
        f2, _, g2 = heapq.heappop(heap)
        for s in g1 + g2:
            largo[s] += 1
        heapq.heappush(heap, (f1 + f2, contador, g1 + g2))
        contador += 1
    return largo
 
 
def _huffman_codigos(largo):
    """Códigos canónicos: alcanza con guardar las longitudes (256 bytes) para decodificar."""
    orden = sorted(largo.items(), key=lambda kv: (kv[1], kv[0]))
    codigo, previa, tabla = 0, orden[0][1], {}
    for s, l in orden:
        codigo <<= (l - previa)
        previa = l
        tabla[s] = (l, codigo)
        codigo += 1
    return tabla
 
 
def _huffman_tamano(img):
    """Tamaño exacto en bytes = bits de los datos / 8 + 256 bytes de tabla de longitudes."""
    frec = np.bincount(img.ravel(), minlength=256)
    largo = _huffman_longitudes(frec)
    bits = sum(int(frec[s]) * l for s, l in largo.items())
    return int(np.ceil(bits / 8.0)) + 256
 
 
def _huffman_verificar(img):
    """Codifica y decodifica de verdad para comprobar que es SIN pérdida (solo imágenes chicas)."""
    if img.size > 400_000:
        return None
    flat = img.ravel()
    frec = np.bincount(flat, minlength=256)
    tabla = _huffman_codigos(_huffman_longitudes(frec))
    lon = np.zeros(256, dtype=np.int64)
    cod = np.zeros(256, dtype=np.uint64)
    for s, (l, c) in tabla.items():
        lon[s], cod[s] = l, c
    lens, vals = lon[flat], cod[flat]
    total = int(lens.sum())
    inicio = np.cumsum(lens) - lens
    sim = np.repeat(np.arange(flat.size), lens)
    pos = np.arange(total) - inicio[sim]
    desplaz = (lens[sim] - 1 - pos).astype(np.uint64)
    bits = ((vals[sim] >> desplaz) & np.uint64(1)).astype(np.uint8)
 
    inversa = {(l, c): s for s, (l, c) in tabla.items()}
    salida = np.empty(flat.size, dtype=np.uint8)
    k, val, ln = 0, 0, 0
    for b in bits.tolist():
        val = (val << 1) | b
        ln += 1
        s = inversa.get((ln, val))
        if s is not None:
            salida[k] = s
            k += 1
            val, ln = 0, 0
    return bool(np.array_equal(salida, flat))
 
 
def _rle(flat):
    """
    RLE con pares (valor, contador) de 1 byte cada uno. Las corridas de más de 255
    se parten en varias. Devuelve (tamaño en bytes, verificación de reconstrucción).
    """
    cambios = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    inicios = np.concatenate(([0], cambios))
    largos = np.diff(np.concatenate((inicios, [flat.size])))
    trozos = (largos + 254) // 255
    valores = np.repeat(flat[inicios], trozos)
    cuentas = np.full(valores.size, 255, dtype=np.int64)
    cuentas[np.cumsum(trozos) - 1] = largos - 255 * (trozos - 1)
    ok = bool(np.array_equal(np.repeat(valores, cuentas), flat))
    return 2 * int(valores.size), ok
 
 
def _rle_dos_direcciones(img):
    horizontal, ok_h = _rle(np.ascontiguousarray(img).ravel())        # recorrido fila por fila
    vertical, ok_v = _rle(np.ascontiguousarray(img.T).ravel())        # recorrido columna por columna
    return horizontal, vertical, (ok_h and ok_v)
 
 
# ----------------------------------------------------------------------
# 2) ANÁLISIS DE ARTEFACTOS DE BLOQUE
# ----------------------------------------------------------------------
def _salto_bloque(dec, ref, region=None, bloque=8):
    """
    Discontinuidad artificial que JPEG agrega en las fronteras de los bloques de 8x8, en niveles de gris.
    Se mide sobre la imagen de ERROR e = JPEG - original (así se descartan los bordes propios de la imagen):
        (salto medio de e ENTRE bloques vecinos) - (salto medio de e DENTRO de los bloques).
    ~0: no hay estructura de bloques. Valores positivos y crecientes: las fronteras 8x8 se hacen visibles.
    Si se pasa 'region' (máscara booleana) solo cuentan los vecinos que están ambos dentro de ella.
    """
    f = dec.astype(np.float32) - ref.astype(np.float32)
    dh = np.abs(f[:, 1:] - f[:, :-1])
    dv = np.abs(f[1:, :] - f[:-1, :])
    jh = (np.arange(dh.shape[1]) % bloque) == bloque - 1
    jv = (np.arange(dv.shape[0]) % bloque) == bloque - 1
    if region is None:
        rh = np.ones(dh.shape, bool)
        rv = np.ones(dv.shape, bool)
    else:
        rh = region[:, 1:] & region[:, :-1]
        rv = region[1:, :] & region[:-1, :]
    borde = np.concatenate([dh[jh[None, :] & rh], dv[jv[:, None] & rv]])
    interior = np.concatenate([dh[(~jh)[None, :] & rh], dv[(~jv)[:, None] & rv]])
    if borde.size < 50 or interior.size < 50:
        return np.nan
    return float(borde.mean() - interior.mean())
 
 
def _actividad(ref, region):
    """Salto medio entre píxeles vecinos de la imagen ORIGINAL en una región (cuánta 'textura' enmascara los artefactos)."""
    f = ref.astype(np.float32)
    dh = np.abs(f[:, 1:] - f[:, :-1])[region[:, 1:] & region[:, :-1]]
    dv = np.abs(f[1:, :] - f[:-1, :])[region[1:, :] & region[:-1, :]]
    v = np.concatenate([dh, dv])
    return float(v.mean()) if v.size else np.nan
 
 
def _regiones(mascara, roi_gris):
    """
    Regiones para comparar dónde se ven más los artefactos:
      borde   : banda de ~±3 px alrededor del contorno de la pupila (borde nítido)
      uniforme: iris/esclerótica lejos de la pupila (>10 px) con baja textura local
    """
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    borde = cv2.morphologyEx(mascara, cv2.MORPH_GRADIENT, el) > 0
    lejos = cv2.dilate(mascara, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) == 0
    f = roi_gris.astype(np.float32)
    media = cv2.blur(f, (5, 5))
    desvio = np.sqrt(np.maximum(cv2.blur(f * f, (5, 5)) - media ** 2, 0))
    umbral = np.median(desvio[lejos]) if lejos.any() else np.median(desvio)
    uniforme = lejos & (desvio <= umbral)
    return borde, uniforme
 
 
# ----------------------------------------------------------------------
# 3) FUNCIÓN PRINCIPAL
# ----------------------------------------------------------------------
def procesar_modulo_4(frame, x, y, w, h, area_min_px=250, area_max_px=2000, canal=2):
    """
    area_min_px / area_max_px: mismo rango que usás en procesar_modulo_3_2.
    canal: canal de color con el que se mide la pupila (2 = rojo, igual que el Módulo 3.2).
    """
    print("\n" + "=" * 70)
    print("--- MÓDULO 4: Compresión de la información de imagen ---")
    print("=" * 70)
 
    FACTORES_Q = [90, 50, 10, 5]                           # los pedidos por la consigna
    Q_BARRIDO = [95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5]  # barrido fino para ubicar la aparición de bloques
    UMBRAL_BLOQUES = 0.5                                   # discontinuidad añadida (niveles de gris) desde la cual se consideran visibles los bloques (criterio heurístico)
    TOL_DIAMETRO = 0.03                                    # variación relativa del diámetro que se considera "cambio"
 
    roi = np.ascontiguousarray(frame[y:y + h, x:x + w])
    roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # ---- Segmentación de la pupila en el frame original (mismo método que el Módulo 3.2) ----
    par = _parametros_segmentacion(area_min_px, area_max_px)
    c_roi = np.ascontiguousarray(roi[:, :, canal])
    res0 = _medir_pupila(c_roi, _centro_inicial(c_roi, par), par)
    mascara = None
    if res0 is not None:
        mascara = np.zeros(roi_gris.shape, np.uint8)
        cv2.ellipse(mascara, res0['elipse'], 255, -1)
        d0 = 2 * np.sqrt(res0['area'] / np.pi)
        print(f"Pupila segmentada en el frame original: área = {res0['area']:.1f} px², "
              f"diámetro equivalente = {d0:.2f} px")
    else:
        print("ADVERTENCIA: no se pudo segmentar la pupila en este frame (revisá area_min_px/area_max_px). "
              "Se omiten la ROI segmentada, el análisis por regiones y el diámetro.")
 
    imagenes = {'Frame original (gris)': frame_gris, 'ROI (gris)': roi_gris}
    if mascara is not None:
        imagenes['ROI segmentada (máscara 0/255)'] = mascara
 
    # ==========================================================
    # 1) Compresión con y sin pérdida sobre cada imagen
    # ==========================================================
    resultados = {}
    for nombre, img in imagenes.items():
        img = np.ascontiguousarray(img)
        n0 = img.size                                       # bytes sin comprimir (8 bit/píxel)
        filas = []
 
        t_huff = _huffman_tamano(img)
        ok_huff = _huffman_verificar(img)
        t_rh, t_rv, ok_rle = _rle_dos_direcciones(img)
        filas.append(dict(metodo='Huffman', bytes=t_huff, ratio=n0 / t_huff, psnr=np.inf, ssim=1.0, ok=ok_huff))
        filas.append(dict(metodo='RLE horizontal', bytes=t_rh, ratio=n0 / t_rh, psnr=np.inf, ssim=1.0, ok=ok_rle))
        filas.append(dict(metodo='RLE vertical', bytes=t_rv, ratio=n0 / t_rv, psnr=np.inf, ssim=1.0, ok=ok_rle))
 
        for q in FACTORES_Q:
            _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
            filas.append(dict(metodo=f'JPEG Q={q}', bytes=len(enc), ratio=n0 / len(enc),
                              psnr=psnr(img, dec, data_range=255), ssim=ssim(img, dec, data_range=255), ok=None))
        resultados[nombre] = filas
 
        print(f"\n>>> {nombre}  [{img.shape[1]}x{img.shape[0]} px | {n0} bytes sin comprimir]")
        print(f"    {'Método':<16}{'Bytes':>10}{'Ratio':>10}{'PSNR (dB)':>12}{'SSIM':>9}")
        for f in filas:
            p = "   inf (=)" if np.isinf(f['psnr']) else f"{f['psnr']:11.2f}"
            print(f"    {f['metodo']:<16}{f['bytes']:>10d}{f['ratio']:>9.2f}:1{p:>12}{f['ssim']:>9.4f}")
        veri = []
        if ok_huff is not None:
            veri.append(f"Huffman reconstrucción exacta: {'SÍ' if ok_huff else 'NO'}")
        veri.append(f"RLE reconstrucción exacta: {'SÍ' if ok_rle else 'NO'}")
        print("    " + " | ".join(veri))
 
    # ==========================================================
    # 2) Artefactos de bloque y pérdida en la ROI (barrido de Q)
    # ==========================================================
    barrido = {}
    if mascara is not None:
        borde, uniforme = _regiones(mascara, roi_gris)
        act_b, act_u = _actividad(roi_gris, borde), _actividad(roi_gris, uniforme)
    d_orig = 2 * np.sqrt(res0['area'] / np.pi) if res0 is not None else np.nan
 
    jpeg_gris, jpeg_color = {}, {}
    for q in Q_BARRIDO:
        _, enc = cv2.imencode('.jpg', roi_gris, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
        jpeg_gris[q] = dec
        fila = {'sb': _salto_bloque(dec, roi_gris)}
        if mascara is not None:
            err = np.abs(roi_gris.astype(np.float32) - dec.astype(np.float32))
            fila.update(mae_b=float(err[borde].mean()), mae_u=float(err[uniforme].mean()),
                        sb_b=_salto_bloque(dec, roi_gris, borde), sb_u=_salto_bloque(dec, roi_gris, uniforme))
            # Diámetro con el MISMO método del Módulo 3.2, sobre la ROI a color comprimida
            _, enc_c = cv2.imencode('.jpg', roi, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            dec_c = cv2.imdecode(enc_c, cv2.IMREAD_COLOR)
            jpeg_color[q] = dec_c
            rq = _medir_pupila(np.ascontiguousarray(dec_c[:, :, canal]), np.array(res0['centro']), par)
            fila['area'] = rq['area'] if rq is not None else np.nan
            fila['diam'] = 2 * np.sqrt(rq['area'] / np.pi) if rq is not None else np.nan
        barrido[q] = fila
 
    # ---------------- Figura 1: PSNR/SSIM vs ratio ----------------
    fig1, (axp, axs) = plt.subplots(1, 2, figsize=(14, 5.5))
    colores = ['tab:blue', 'tab:orange', 'tab:green']
    marcas = {'Huffman': 's', 'RLE horizontal': '^', 'RLE vertical': 'v'}
    psnr_fin = [f['psnr'] for fs in resultados.values() for f in fs if np.isfinite(f['psnr'])]
    tope = max(psnr_fin) + 8
    for (nombre, filas), col in zip(resultados.items(), colores):
        jp = [f for f in filas if f['metodo'].startswith('JPEG')]
        axp.plot([f['ratio'] for f in jp], [f['psnr'] for f in jp], '-o', color=col, label=f'JPEG - {nombre}')
        axs.plot([f['ratio'] for f in jp], [f['ssim'] for f in jp], '-o', color=col, label=f'JPEG - {nombre}')
        for f in jp:
            q = f['metodo'].split('=')[1]
            axp.annotate(f'Q{q}', (f['ratio'], f['psnr']), textcoords='offset points', xytext=(4, 5), fontsize=7)
            axs.annotate(f'Q{q}', (f['ratio'], f['ssim']), textcoords='offset points', xytext=(4, 5), fontsize=7)
        for f in filas:
            if f['metodo'] in marcas:
                axp.plot(f['ratio'], tope, marker=marcas[f['metodo']], color=col, mfc='none', ls='')
                axs.plot(f['ratio'], 1.0, marker=marcas[f['metodo']], color=col, mfc='none', ls='')
    for m, mk in marcas.items():
        axp.plot([], [], marker=mk, color='k', mfc='none', ls='', label=f'{m} (sin pérdida)')
    axp.axhline(tope, color='gray', ls=':')
    axp.set_ylim(top=tope + 4)
    axp.text(0.02, 0.97, 'línea punteada: métodos sin pérdida (PSNR = ∞)', transform=axp.transAxes, va='top', fontsize=8, color='gray')
    axp.set_xscale('log'); axs.set_xscale('log')
    axp.set_xlabel('Ratio de compresión (log)'); axs.set_xlabel('Ratio de compresión (log)')
    axp.set_ylabel('PSNR (dB)'); axs.set_ylabel('SSIM')
    axp.set_title('PSNR vs. ratio'); axs.set_title('SSIM vs. ratio')
    axp.grid(True, which='both', alpha=0.4); axs.grid(True, which='both', alpha=0.4)
    axp.legend(fontsize=7, loc='lower left')
    plt.tight_layout()
 
    # ---------------- Figura 2: observación visual de artefactos ----------------
    if mascara is not None:
        cy, cx = int(res0['centro'][1]), int(res0['centro'][0])
        hz = int(min(max(24, 2.4 * np.sqrt(res0['area'] / np.pi)), min(roi_gris.shape) // 2 - 1))
        y0, x0 = max(0, cy - hz), max(0, cx - hz)
        sl = (slice(y0, y0 + 2 * hz), slice(x0, x0 + 2 * hz))
    else:
        sl = (slice(None), slice(None))
    fig2, axs2 = plt.subplots(3, 5, figsize=(17, 10))
    columnas = [('Original', roi_gris)] + [(f'JPEG Q={q}', jpeg_gris[q]) for q in FACTORES_Q]
    for j, (titulo, im) in enumerate(columnas):
        axs2[0, j].imshow(im, cmap='gray', vmin=0, vmax=255); axs2[0, j].set_title(titulo)
        axs2[1, j].imshow(im[sl], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        axs2[1, j].set_title('Zoom borde pupila + iris' if j == 0 else titulo + ' (zoom)')
        if j == 0:
            if mascara is not None:
                vis = np.zeros(roi_gris.shape + (3,), np.uint8)
                vis[...] = roi_gris[..., None] // 2
                vis[borde] = (255, 0, 0)
                vis[uniforme] = (0, 120, 255)
                axs2[2, 0].imshow(vis); axs2[2, 0].set_title('Regiones (rojo: borde, azul: uniforme)', fontsize=9)
            else:
                axs2[2, 0].axis('off')
        else:
            err = np.clip(np.abs(roi_gris.astype(np.float32) - im.astype(np.float32)) * 8, 0, 255)
            axs2[2, j].imshow(err, cmap='magma', vmin=0, vmax=255)
            axs2[2, j].set_title(f'|Error| x8  (Q={FACTORES_Q[j - 1]})')
    for a in axs2.ravel():
        a.axis('off')
    plt.suptitle('Artefactos de bloque JPEG: ROI completa (arriba), zoom (centro) y mapa de error (abajo)', fontsize=13)
    plt.tight_layout()
 
    # ---------------- Figura 3: análisis cuantitativo ----------------
    qs = sorted(barrido, reverse=True)
    n_graf = 3 if mascara is not None else 1
    fig3, ax3 = plt.subplots(1, n_graf, figsize=(5.2 * n_graf, 4.6))
    ax3 = np.atleast_1d(ax3)
    ax3[0].plot(qs, [barrido[q]['sb'] for q in qs], '-o', label='ROI completa', color='k')
    if mascara is not None:
        ax3[0].plot(qs, [barrido[q]['sb_u'] for q in qs], '-o', label='Zona uniforme', color='tab:blue')
        ax3[0].plot(qs, [barrido[q]['sb_b'] for q in qs], '-o', label='Borde pupila', color='tab:red')
    ax3[0].axhline(UMBRAL_BLOQUES, color='gray', ls='--', label=f'Umbral visible ({UMBRAL_BLOQUES:g})')
    ax3[0].set_title('Salto extra entre bloques 8x8 vs. Q'); ax3[0].set_xlabel('Q'); ax3[0].invert_xaxis()
    ax3[0].set_ylabel('Salto extra (niveles de gris)'); ax3[0].legend(fontsize=8); ax3[0].grid(True)
    if mascara is not None:
        ax3[1].plot(qs, [barrido[q]['mae_b'] for q in qs], '-o', color='tab:red', label='Borde pupila')
        ax3[1].plot(qs, [barrido[q]['mae_u'] for q in qs], '-o', color='tab:blue', label='Zona uniforme')
        ax3[1].set_title('Error absoluto medio vs. Q'); ax3[1].set_xlabel('Q'); ax3[1].invert_xaxis()
        ax3[1].set_ylabel('MAE (niveles de gris)'); ax3[1].legend(fontsize=8); ax3[1].grid(True)
        ax3[2].plot(qs, [barrido[q]['diam'] for q in qs], '-o', color='tab:green', label='Diámetro con JPEG')
        ax3[2].axhline(d_orig, color='k', ls='--', label='Original')
        ax3[2].axhspan(d_orig * (1 - TOL_DIAMETRO), d_orig * (1 + TOL_DIAMETRO), color='gray', alpha=0.2,
                       label=f'±{100 * TOL_DIAMETRO:.0f} %')
        ax3[2].set_title('Diámetro pupilar (Módulo 3) vs. Q'); ax3[2].set_xlabel('Q'); ax3[2].invert_xaxis()
        ax3[2].set_ylabel('Diámetro equivalente (px)'); ax3[2].legend(fontsize=8); ax3[2].grid(True)
    plt.tight_layout()
    plt.show()
 
    # ==========================================================
    # 3) RESPUESTAS A LAS PREGUNTAS DE LA CONSIGNA
    # ==========================================================
    print("\n" + "=" * 70)
    print("RESPUESTAS A LAS PREGUNTAS DE LA CONSIGNA")
    print("=" * 70)
 
    # --- P1: ratios ---
    print("\n[1] Tasa de compresión y relación calidad/ratio")
    for nombre, filas in resultados.items():
        sin = max((f for f in filas if np.isinf(f['psnr'])), key=lambda f: f['ratio'])
        con = [f for f in filas if f['metodo'].startswith('JPEG')]
        print(f"   - {nombre}: mejor método sin pérdida = {sin['metodo']} ({sin['ratio']:.2f}:1); "
              f"JPEG va de {con[0]['ratio']:.1f}:1 (Q=90, PSNR {con[0]['psnr']:.1f} dB) "
              f"a {con[-1]['ratio']:.1f}:1 (Q=5, PSNR {con[-1]['psnr']:.1f} dB).")
        peor = [f['metodo'] for f in filas if np.isinf(f['psnr']) and f['ratio'] < 1]
        if peor:
            print(f"     * {' y '.join(peor)}: {'expanden' if len(peor) > 1 else 'expande'} la imagen (ratio < 1) porque casi no "
                  f"hay corridas de píxeles idénticos.")
    print("   Nota: RLE solo rinde donde hay corridas largas (máscara binaria). Huffman aprovecha la distribución "
          "de grises, y JPEG logra ratios mucho mayores a costa de pérdida (PSNR/SSIM bajan al subir el ratio).")
    print("   Nota: el frame viene de un video ya comprimido (H.264), así que el 'original' no es estrictamente sin pérdida.")
 
    # --- P2: aparición de bloques ---
    print(f"\n[2] ¿Desde qué factor de calidad aparecen los artefactos de bloque?")
    ref = 'sb_u' if mascara is not None else 'sb'
    donde = 'zonas uniformes' if mascara is not None else 'toda la ROI'
    q_bloque = None
    for q in qs:                                            # de mayor a menor calidad
        v = barrido[q][ref]
        if not np.isnan(v) and v >= UMBRAL_BLOQUES:
            q_bloque = q
            break
    print(f"   Discontinuidad añadida entre bloques 8x8 en {donde} (niveles de gris): " +
          " | ".join(f"Q={q}: {barrido[q][ref]:.2f}" for q in FACTORES_Q))
    if q_bloque is None:
        print(f"   -> En el barrido Q=95..5 el salto extra nunca llega a {UMBRAL_BLOQUES:g} nivel de gris: "
              f"no se detectan bloques de forma objetiva.")
    else:
        print(f"   -> Los bloques de 8x8 emergen a partir de Q ≈ {q_bloque} "
              f"(salto extra = {barrido[q_bloque][ref]:.2f} niveles; criterio: >= {UMBRAL_BLOQUES:g} nivel, "
              f"aprox. el umbral de visibilidad de escalones en zonas planas).")
        print("      Con Q más bajo se vuelven cada vez más evidentes (ver Figura 2: zoom y mapa de error).")
    print("   Es un criterio objetivo aproximado: confirmalo mirando la Figura 2.")
 
    # --- P3: bordes vs uniformes ---
    print("\n[3] ¿Los artefactos son más notorios en el borde de la pupila o en regiones uniformes?")
    if mascara is None:
        print("   (No disponible: no se pudo segmentar la pupila.)")
    else:
        print("   MAE y salto entre bloques en niveles de gris. Visibilidad relativa = salto / actividad local de la imagen original "
              f"(actividad: borde = {act_b:.1f}, uniforme = {act_u:.1f}).")
        print(f"   {'Q':>4} | {'MAE borde':>10} {'MAE unif.':>10} | {'Salto borde':>11} {'Salto unif.':>11} | {'Visib. borde':>12} {'Visib. unif.':>12}")
        for q in FACTORES_Q:
            b = barrido[q]
            print(f"   {q:>4} | {b['mae_b']:>10.2f} {b['mae_u']:>10.2f} | {b['sb_b']:>11.2f} {b['sb_u']:>11.2f} | "
                  f"{b['sb_b'] / max(act_b, 1.0):>12.2f} {b['sb_u'] / max(act_u, 1.0):>12.2f}")
        qb = 10
        b = barrido[qb]
        mae_borde_mayor = b['mae_b'] > b['mae_u']
        vis_b, vis_u = b['sb_b'] / max(act_b, 1.0), b['sb_u'] / max(act_u, 1.0)
        vis_uniforme_mayor = vis_u > vis_b
        factor = max(b['mae_b'], b['mae_u']) / max(min(b['mae_b'], b['mae_u']), 1e-6)
        if mae_borde_mayor:
            print(f"   -> Magnitud del error (a Q={qb}): MAYOR en el borde de la pupila ({factor:.1f}x el de las zonas uniformes). "
                  f"Ahí se concentran el ringing / ruido de mosquito: los bordes nítidos tienen mucha energía en altas "
                  f"frecuencias, que la cuantización destruye.")
        else:
            print(f"   -> Magnitud del error (a Q={qb}): MAYOR en las zonas uniformes ({factor:.1f}x el del borde de la pupila).")
        if vis_uniforme_mayor:
            print(f"   -> Visibilidad del efecto de bloque (a Q={qb}): MAYOR en las zonas uniformes ({vis_u:.2f} vs {vis_b:.2f}). "
                  f"Allí el bloque pierde sus detalles y queda como un parche plano, y el salto contra el vecino no tiene "
                  f"textura que lo enmascare; en el borde, el propio contraste de la pupila lo disimula.")
        else:
            print(f"   -> Visibilidad del efecto de bloque (a Q={qb}): MAYOR en el borde de la pupila ({vis_b:.2f} vs {vis_u:.2f}).")
        if mae_borde_mayor and vis_uniforme_mayor:
            print("   Conclusión: el daño numérico se concentra en el borde de la pupila (lo que podría afectar la medición del "
                  "diámetro), pero la 'cuadrícula' se percibe más en las regiones uniformes del iris/esclerótica.")
        elif (not mae_borde_mayor) and (not vis_uniforme_mayor):
            print("   Conclusión: en este frame, tanto el error como la visibilidad de bloques predominan en el borde "
                  "de la pupila.")
        else:
            print("   Conclusión: las dos medidas no coinciden en este frame; interpretá la tabla junto con la Figura 2.")
 
    # --- P4: diámetro ---
    print("\n[4] ¿La compresión a Q=50 / Q=10 cambia el diámetro pupilar estimado en el Módulo 3?")
    if mascara is None:
        print("   (No disponible: no se pudo segmentar la pupila.)")
    else:
        print(f"   Original: área = {res0['area']:.1f} px² | diámetro equivalente = {d_orig:.2f} px")
        for q in FACTORES_Q:
            a, d = barrido[q]['area'], barrido[q]['diam']
            if np.isnan(d):
                print(f"   Q={q:>2}: la segmentación FALLA (no se detecta la pupila).")
            else:
                rel = (d - d_orig) / d_orig
                veredicto = "SÍ cambia" if abs(rel) > TOL_DIAMETRO else "no cambia significativamente"
                print(f"   Q={q:>2}: área = {a:7.1f} px² | diámetro = {d:6.2f} px | "
                      f"Δ = {d - d_orig:+.2f} px ({100 * rel:+.1f} %) -> {veredicto} (umbral ±{100 * TOL_DIAMETRO:.0f} %)")
        cambian = [q for q in (50, 10) if np.isnan(barrido[q]['diam']) or
                   abs(barrido[q]['diam'] - d_orig) / d_orig > TOL_DIAMETRO]
        if cambian:
            print(f"   -> Respuesta: SÍ, a Q={' y Q='.join(map(str, cambian))} el diámetro estimado se aparta más de "
                  f"{100 * TOL_DIAMETRO:.0f} % del original.")
        else:
            print(f"   -> Respuesta: NO. A Q=50 y Q=10 el diámetro estimado queda dentro de ±{100 * TOL_DIAMETRO:.0f} % "
                  f"del original: el método basado en gradiente del borde es robusto a la compresión moderada.")
        print("   Nota: la comparación usa un único frame; para generalizar, repetilo en varios frames del video.")
 
    return {'resultados': resultados, 'barrido': barrido, 'q_bloque': q_bloque,
            'diametro_original': d_orig}
    
def exportar_frames_para_medir(video_path,
                               tiempos=(0.5, 1.5, 2.5, 3.5, 4.4, 5.2, 5.5, 5.8, 6.2, 6.8, 7.2, 7.4),
                               carpeta='frames_manual'):
    os.makedirs(carpeta, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    objetivo = {int(round(t * fps)): t for t in tiempos}
    ultimo = max(objetivo)

    idx = -1
    while idx < ultimo:                      # lectura secuencial: mismo índice que usa procesar_modulo_3_2
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx in objetivo:
            cv2.imwrite(os.path.join(carpeta, f'frame_{idx:04d}_t{objetivo[idx]:.1f}s.png'), frame)
    cap.release()

    print(f"FPS: {fps:.2f}. Frames exportados a '{carpeta}/'")
    print("Plantilla para completar con tus mediciones:")
    print("anotaciones = {" + ", ".join(f"{i}: 0.0" for i in sorted(objetivo)) + "}")
# ======================================================
# INSERCIÓN EN EL MAIN
# ======================================================

def main():
    video_original = 'registro_pupila.mp4'
    video_trabajo = 'registro_pupila_copia.mp4'

    if not os.path.exists(video_trabajo):
        shutil.copy(video_original, video_trabajo)

    frame_representativo = obtener_mejor_frame(video_trabajo)

    # ----- MÓDULOS 1, 2 y 3.1 (sobre el frame representativo) -----
    if frame_representativo is not None:
        x, y, w, h = obtener_roi_automatica(frame_representativo)

        procesar_modulo_1_1(frame_representativo, x, y, w, h)
        procesar_modulo_1_2(frame_representativo, x, y, w, h)
        procesar_modulo_1_3(frame_representativo, x, y, w, h)

        roi = frame_representativo[y:y+h, x:x+w]
        mascara_limpia = procesar_modulo_2(roi)
        procesar_modulo_3_1(roi, mascara_limpia)

    # ----- MÓDULO 3.2 (consume todo el video) -----
    anotaciones = {132: 624.0, 222: 1204.0}      # agregá todas las que tengas
    procesar_modulo_3_2(video_trabajo,
                        anotaciones_manuales=anotaciones,
                        manual_contraida=624.0,
                        manual_dilatada=1204.0,
                        area_min_px=250,
                        area_max_px=2000)

    # ----- MÓDULO 4 (sobre el frame representativo) -----
    if frame_representativo is not None:
        procesar_modulo_4(frame_representativo, x, y, w, h,
                          area_min_px=250, area_max_px=2000)


if __name__ == '__main__':
    main()