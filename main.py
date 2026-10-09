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


def procesar_modulo_3_2(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 
    
    areas = []
    tiempos = []
    brillos = [] # NUEVO: Vector para detectar el instante del estímulo
    frame_idx = 0
    
    print("\n--- 3.2 Procesando Video Completo ---")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        x, y, w, h = obtener_roi_automatica(frame)
        roi = frame[y:y+h, x:x+w]
        
        # Calculamos el brillo medio del frame para detectar el apagón
        brillo_medio = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        brillos.append(brillo_medio)
        
        _, _, r = cv2.split(roi)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        r_clahe = clahe.apply(r)
        
        blur = cv2.GaussianBlur(r_clahe, (5, 5), 0)
        
        ret_otsu, mascara_base = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cierre = cv2.morphologyEx(mascara_base, cv2.MORPH_CLOSE, elemento, iterations=2)
        mascara_limpia = cv2.morphologyEx(cierre, cv2.MORPH_OPEN, elemento, iterations=1)
        
        contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        area_actual = 0
        if contornos:
            mejor_area = 0
            for c in contornos:
                area_c = cv2.contourArea(c)
                perimetro = cv2.arcLength(c, True)
                if perimetro > 0:
                    circularidad = (4 * np.pi * area_c) / (perimetro ** 2)
                    if circularidad > 0.35 and 150 < area_c < 550:
                        if area_c > mejor_area:
                            mejor_area = area_c
            area_actual = mejor_area
            
        if area_actual == 0 and len(areas) > 0:
            area_actual = areas[-1] 
            
        areas.append(area_actual)
        tiempos.append(frame_idx / fps)
        frame_idx += 1
        
    cap.release()
    
    if not areas:
        return

    areas_smooth = gaussian_filter1d(areas, sigma=2.0)
    
    # --- LÓGICA TEMPORAL CORREGIDA (ANCLAJE DIRECTO AL ESTÍMULO) ---
    
    # 1. Detectar el estímulo (caída abrupta de brillo general)
    brillos_diff = np.diff(brillos)
    indice_estimulo = np.argmin(brillos_diff) 
    tiempo_estimulo = tiempos[indice_estimulo]
    
    # Guardamos el área EXACTA que tenía la pupila en el instante del apagón
    area_en_estimulo = areas_smooth[indice_estimulo] 
    
    # 2. Encontrar el pico máximo de dilatación
    indice_maximo = indice_estimulo + np.argmax(areas_smooth[indice_estimulo:])
    area_maxima = areas_smooth[indice_maximo]
    tiempo_maximo = tiempos[indice_maximo]
    
    amplitud = area_maxima - area_en_estimulo
    
    # 3. Buscar la latencia real:
    # Apenas el área supere el tamaño del apagón por un margen mínimo (15 px), 
    # consideramos que inició fisiológicamente la redilatación.
    umbral_inicio = area_en_estimulo + 15
    
    tiempo_inicio_respuesta = tiempo_estimulo
    for i in range(indice_estimulo, indice_maximo):
        if areas_smooth[i] > umbral_inicio:
            tiempo_inicio_respuesta = tiempos[i]
            break
            
    # 4. Cálculo de parámetros fisiológicos
    latencia_verdadera = tiempo_inicio_respuesta - tiempo_estimulo
    delta_t_redilatacion = tiempo_maximo - tiempo_inicio_respuesta
    vel_redilatacion = (amplitud / delta_t_redilatacion) if delta_t_redilatacion > 0 else 0

    # Gráfico
    plt.figure(figsize=(10, 5))
    plt.plot(tiempos, areas, label='Área Bruta A(t)', alpha=0.3)
    plt.plot(tiempos, areas_smooth, label='Área Suavizada', color='red', linewidth=2)
    
    plt.axvline(x=tiempo_estimulo, color='black', linestyle='--', label='Estímulo (Flash OFF)')
    plt.axvline(x=tiempo_inicio_respuesta, color='g', linestyle='--', label='Inicio Redilatación')
    plt.axvline(x=tiempo_maximo, color='purple', linestyle='--', label='Redilatación Máxima')
    
    plt.title('Dinámica Temporal de la Pupila - Registro B (Luz a Oscuridad)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Área Pupilar (px)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\n--- Parámetros Fisiológicos Extraídos (Registro B) ---")
    print(f"Área en Estímulo (Luz): {area_en_estimulo:.2f} px")
    print(f"Amplitud de Redilatación: {amplitud:.2f} px")
    print(f"Tiempo de Estímulo (Apagón): {tiempo_estimulo:.2f} s")
    print(f"Latencia Fisiológica: {latencia_verdadera:.2f} s")
    print(f"Tiempo real de expansión (Delta T): {delta_t_redilatacion:.2f} s")
    print(f"Velocidad de Redilatación: {vel_redilatacion:.2f} px/s")
# ======================================================
# INSERCIÓN EN EL MAIN
# ======================================================

def main():
    video_original = 'registro_pupila.mp4'
    video_trabajo = 'registro_pupila_copia.mp4'
    
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
        
    # Módulo 3.2 opera independientemente de los modulos anteriores consumiendo todo el video
    procesar_modulo_3_2(video_trabajo)



if __name__ == '__main__':
    main()