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
import zlib
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
    
def rle_compresion(canal):
    """Implementación de compresión Run-Length Encoding 1D"""
    pixeles = canal.flatten()
    cambios = np.where(pixeles[:-1] != pixeles[1:])[0] + 1
    
    if len(cambios) > 0:
        valores = np.insert(pixeles[cambios], 0, pixeles[0])
        longitudes = np.diff(np.append(np.insert(cambios, 0, 0), len(pixeles)))
    else:
        valores = [pixeles[0]]
        longitudes = [len(pixeles)]
    
    # Asumimos 1 byte para el valor del píxel y 1 byte para el contador (longitud)
    tamaño_comprimido = len(valores) * 2 
    return tamaño_comprimido

def extraer_area_comprimida(roi_bgr):
    _, _, r = cv2.split(roi_bgr)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    r_clahe = clahe.apply(r)
    blur = cv2.GaussianBlur(r_clahe, (5, 5), 0)
    _, mascara_base = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Volvemos a los parámetros exactos del Módulo 3
    elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cierre = cv2.morphologyEx(mascara_base, cv2.MORPH_CLOSE, elemento, iterations=2)
    mascara_limpia = cv2.morphologyEx(cierre, cv2.MORPH_OPEN, elemento, iterations=1)
    
    contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mejor_area = 0
    if contornos:
        for c in contornos:
            # Envoltura convexa: repara las imperfecciones o roturas causadas por reflejos
            hull = cv2.convexHull(c)
            area_c = cv2.contourArea(hull)
            perimetro = cv2.arcLength(hull, True)
            
            if perimetro > 0:
                circularidad = (4 * np.pi * area_c) / (perimetro ** 2)
                
                # Mismos umbrales estrictos validados en el Módulo 3
                if circularidad > 0.35 and 150 < area_c < 550:
                    if area_c > mejor_area:
                        mejor_area = area_c
                        
    return mejor_area

def procesar_modulo_4(frame, x, y, w, h):
    print("\n" + "="*50)
    print("--- MÓDULO 4: Compresión de Información ---")
    print("="*50)
    
    roi = frame[y:y+h, x:x+w]
    roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    tamaño_original = roi_gris.nbytes
    print(f"Tamaño Original de la ROI (Escala de grises): {tamaño_original} bytes")

    # ==========================================================
    # 1. Compresión sin pérdida (RLE y Huffman)
    # ==========================================================
    tamaño_rle_h = rle_compresion(roi_gris)
    ratio_rle_h = tamaño_original / tamaño_rle_h if tamaño_rle_h > 0 else 1
    
    tamaño_rle_v = rle_compresion(roi_gris.T)
    ratio_rle_v = tamaño_original / tamaño_rle_v if tamaño_rle_v > 0 else 1
    
    # Utilizamos zlib como proxy de compresión Huffman/LZ77 nativa en Python
    comprimido_huffman = zlib.compress(roi_gris.tobytes(), level=9)
    ratio_huffman = tamaño_original / len(comprimido_huffman)
    
    print("\n--- 1. Compresión Sin Pérdida ---")
    print(f"Ratio RLE (Barrido Horizontal): {ratio_rle_h:.2f}:1")
    print(f"Ratio RLE (Barrido Vertical):   {ratio_rle_v:.2f}:1")
    print(f"Ratio Huffman (Aprox Zlib):     {ratio_huffman:.2f}:1")

    # ==========================================================
    # 2. Compresión con pérdida (JPEG) y Evaluación
    # ==========================================================
    factores_q = [90, 50, 10, 5]
    imagenes_jpeg = []
    
    # Calculamos el área original (Ground Truth algorítmico) para comparar
    area_original = extraer_area_comprimida(roi)
    print(f"\n--- 2. Compresión Con Pérdida (JPEG) ---")
    print(f"Área Pupilar Original Detectada: {area_original} px\n")
    
    for q in factores_q:
        # Codificamos a JPEG simulado en memoria
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        _, encimg = cv2.imencode('.jpg', roi, encode_param)
        
        # Decodificamos para obtener la imagen degradada
        roi_jpeg_bgr = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        roi_jpeg_gris = cv2.cvtColor(roi_jpeg_bgr, cv2.COLOR_BGR2GRAY)
        imagenes_jpeg.append(roi_jpeg_gris)
        
        # Métricas
        tamaño_jpeg = len(encimg)
        ratio = tamaño_original / tamaño_jpeg
        valor_psnr = psnr(roi_gris, roi_jpeg_gris)
        valor_ssim = ssim(roi_gris, roi_jpeg_gris, data_range=255)
        
        # Comprobación de robustez de segmentación
        area_q = extraer_area_comprimida(roi_jpeg_bgr)
        error_area = abs(area_original - area_q)
        
        print(f"[JPEG Q={q:2d}] Ratio: {ratio:.2f}:1 | PSNR: {valor_psnr:5.2f} dB | SSIM: {valor_ssim:.4f}")
        print(f"           Área detectada: {area_q} px (Error: {error_area} px)")

    # ==========================================================
    # 3. Visualización de Artefactos de Bloque
    # ==========================================================
    fig, axs = plt.subplots(1, 5, figsize=(18, 4))
    axs[0].imshow(roi_gris, cmap='gray'); axs[0].set_title('ROI Original')
    
    for i, q in enumerate(factores_q):
        axs[i+1].imshow(imagenes_jpeg[i], cmap='gray')
        axs[i+1].set_title(f'JPEG Q={q}')
        
    plt.suptitle('Evaluación de Artefactos de Bloque (Blocking Artifacts) en JPEG', fontsize=14)
    plt.tight_layout()
    plt.show()
    
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
    video_trabajo = 'registro_pupila_copia_2.mp4'
    
    if not os.path.exists(video_trabajo):
        shutil.copy(video_original, video_trabajo)
            
    frame_representativo = obtener_mejor_frame(video_trabajo)
    
    if frame_representativo is not None:
        x, y, w, h = obtener_roi_automatica(frame_representativo)
        
        procesar_modulo_1_1(frame_representativo, x, y, w, h)
        procesar_modulo_1_2(frame_representativo, x, y, w, h)
        procesar_modulo_1_3(frame_representativo, x, y, w, h)
        
        roi = frame_representativo[y:y+h, x:x+w]
        
        # Módulo 2 (debe retornar mascara_limpia)
        mascara_limpia = procesar_modulo_2(roi)
        
        # ----- EJECUCIÓN MÓDULO 3 -----
        procesar_modulo_3_1(roi, mascara_limpia) 
        
        procesar_modulo_4(frame_representativo, x, y, w, h)
        
    # Módulo 3.2 opera independientemente de los modulos anteriores consumiendo todo el video
    anotaciones = {132: 624.0, 222: 1204.0}      # agregá todas las que tengas

    procesar_modulo_3_2(video_trabajo,
                        anotaciones_manuales=anotaciones,
                        manual_contraida=624.0,
                        manual_dilatada=1204.0,
                        area_min_px=250,
                        area_max_px=2000)
    
    exportar_frames_para_medir(video_trabajo)



if __name__ == '__main__':
    main()