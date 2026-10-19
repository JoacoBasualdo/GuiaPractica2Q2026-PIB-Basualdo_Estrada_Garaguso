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
import csv
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
from matplotlib.patches import Ellipse, Rectangle
# IMPORTACIONES PARA MÓDULO 4
import heapq
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# GENERACION DEL PDF
import sys
import contextlib
import re
import tempfile
from fpdf import FPDF
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

def mostrar_instrucciones_interactivas():
    """
    Muestra una ventana emergente al iniciar el programa con una guía clara
    sobre cómo configurar el video y las opciones de anotación en el código.
    """
    root = tk.Tk()
    root.withdraw() # Oculta la ventana principal de Tkinter para que solo salga el cartel
    
    titulo = "Guía de Configuración - Procesamiento de Pupila"
    mensaje = (
        "Los siguientes cambios se tienen que hacer en 'main()' al operar con un nuevo video:\n\n"
        "📌 1. CAMBIAR EL VIDEO A ANALIZAR:\n"
        "   - Buscá en 'main()' la línea: video_original = 'registro_pupila.mp4'\n"
        "   - Reemplazá 'registro_pupila.mp4' por el nombre del nuevo archivo de video (ej. 'video_prueba2.mp4').\n\n"
        "📌 2. CONFIGURAR ANOTACIONES MANUALES (True / False):\n"
        "   - anotar_referencia = True / False\n"
        "     (Ponelo en True la PRIMERA VEZ que analices un video nuevo para dibujar la máscara del Módulo 2.3).\n"
        "   - anotar_frames_3_2 = True / False\n"
        "     (Ponelo en True la PRIMERA VEZ que se trabaja con el nuevo video para medir con clics los frames del Módulo 3.2).\n\n"
        "ℹ️ Una vez creados los archivos .png y .csv de un video, volvé a poner ambas opciones en FALSE "
        "para que el programa corra 100% automatizado (ideal para la evaluación de la cátedra).\n"
        "El pdf generado cuenta con respuestas variables según video, y con respuestas que solo aplican al video original del grupo"
    )
    
    messagebox.showinfo(titulo, mensaje)
    root.destroy()

class CapturadorPDF:
    """
    Captura lo que se imprime en la terminal Y las figuras que se muestran (plt.show y cv2.imshow),
    en el mismo orden en que aparecen, para después armar un PDF con texto e imágenes intercalados.
    Las imágenes se guardan en una carpeta temporal que se borra una vez armado el PDF.
    """
    def __init__(self, dpi=120, mostrar_ventanas=False):
        self.consola_original = sys.stdout
        self.mostrar_ventanas = mostrar_ventanas    # False = las figuras solo se guardan, no se abren
        self.contenido = []                 # elementos: ('texto', str) o ('imagen', ruta, titulo)
        self.carpeta_imagenes = None                # se crea en activar() (carpeta temporal del sistema)
        self.dpi = dpi
        self._n_img = 0
        self._cv2_pendientes = {}           # ventana de cv2 -> última imagen mostrada (se guarda en waitKey(0))
        self._originales = None

    # ------------------------- texto -------------------------
    def write(self, mensaje):
        # Muestra el texto en la terminal normal
        self.consola_original.write(mensaje)
        # Y también lo guarda en nuestra lista
        self.contenido.append(('texto', mensaje))

    def flush(self):
        self.consola_original.flush()

    # ------------------------ imágenes ------------------------
    def _ruta_nueva(self):
        self._n_img += 1
        return os.path.join(self.carpeta_imagenes, f"figura_{self._n_img:03d}.png")

    def _guardar_figuras_plt(self):
        """Guarda todas las figuras de matplotlib abiertas que todavía no estén en el PDF."""
        for num in plt.get_fignums():
            fig = plt.figure(num)
            if getattr(fig, '_ya_en_pdf', False):
                continue
            ruta = self._ruta_nueva()
            fig.savefig(ruta, dpi=self.dpi, bbox_inches='tight')
            fig._ya_en_pdf = True
            self.contenido.append(('imagen', ruta, ''))

    def _show(self, *args, **kwargs):
        self._guardar_figuras_plt()
        if self.mostrar_ventanas:
            return self._originales['show'](*args, **kwargs)
        plt.close('all')                    # no se abren: se cierran para liberar memoria

    def _imshow(self, nombre, imagen):
        self._cv2_pendientes[nombre] = np.array(imagen, copy=True)
        if self.mostrar_ventanas:
            return self._originales['imshow'](nombre, imagen)

    def _waitkey(self, delay=0):
        # Solo se guardan las ventanas que esperan una tecla (waitKey(0)); así no se guardan los
        # cientos de cuadros de las ventanas interactivas (p. ej. la herramienta de anotación).
        if delay <= 0:
            for nombre, img in self._cv2_pendientes.items():
                if img.dtype != np.uint8:
                    img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                ruta = self._ruta_nueva()
                cv2.imwrite(ruta, img)
                self.contenido.append(('imagen', ruta, nombre))
            self._cv2_pendientes.clear()
        if self.mostrar_ventanas:
            return self._originales['waitKey'](delay)
        return -1                           # como si no se hubiera apretado ninguna tecla

    def _destroy_all(self):
        self._cv2_pendientes.clear()
        if self.mostrar_ventanas:
            return self._originales['destroyAllWindows']()

    @contextlib.contextmanager
    def ventanas_reales(self):
        """Dentro de este bloque las ventanas sí se abren (para herramientas interactivas, como anotar la pupila)."""
        previo = self.mostrar_ventanas
        self.mostrar_ventanas = True
        try:
            yield
        finally:
            self.mostrar_ventanas = previo

    # ------------------- activar / desactivar -------------------
    def activar(self):
        self.carpeta_imagenes = tempfile.mkdtemp(prefix="figuras_pdf_")
        self._originales = {'show': plt.show, 'imshow': cv2.imshow,
                            'waitKey': cv2.waitKey, 'destroyAllWindows': cv2.destroyAllWindows}
        # Se usan funciones comunes (no métodos) porque matplotlib, al cargar su backend, les agrega atributos
        def show(*args, **kwargs): return self._show(*args, **kwargs)
        def imshow(nombre, imagen): return self._imshow(nombre, imagen)
        def waitKey(delay=0): return self._waitkey(delay)
        def destroyAllWindows(): return self._destroy_all()
        plt.show = show
        cv2.imshow = imshow
        cv2.waitKey = waitKey
        cv2.destroyAllWindows = destroyAllWindows
        sys.stdout = self

    def desactivar(self):
        sys.stdout = self.consola_original
        if self._originales:
            plt.show = self._originales['show']
            cv2.imshow = self._originales['imshow']
            cv2.waitKey = self._originales['waitKey']
            cv2.destroyAllWindows = self._originales['destroyAllWindows']
            self._originales = None

    @staticmethod
    def _separar_titulos(texto):
        """
        Divide el texto impreso en bloques ('titulo' | 'subtitulo' | 'cuerpo') según cómo se imprimió:
          TÍTULO    : una línea encerrada entre dos líneas de '====='
          SUBTÍTULO : '--- texto ---', '[1] texto', '1.3.2 texto' o '>>> texto' al principio de la línea
        """
        es_regla = lambda l: len(l.strip()) >= 10 and set(l.strip()) == {'='}
        sacar_guiones = lambda l: re.sub(r'\s*-{3,}.*$', '', re.sub(r'^\s*-{3,}\s*', '', l)).strip()
        patrones_sub = [r'^-{3,}\s*(.+?)\s*-{3,}', r'^(\[\d+\]\s+\S.*)', r'^(\d+\.\d+(?:\.\d+)*\s+\S.*)', r'^>>>\s*(.+)']

        lineas = texto.split('\n')
        bloques, cuerpo = [], []

        def cerrar_cuerpo():
            if cuerpo:
                bloques.append(('cuerpo', '\n'.join(cuerpo)))
                cuerpo.clear()

        i = 0
        while i < len(lineas):
            l = lineas[i]
            if es_regla(l):
                if i + 2 < len(lineas) and lineas[i + 1].strip() and es_regla(lineas[i + 2]):
                    cerrar_cuerpo()
                    bloques.append(('titulo', sacar_guiones(lineas[i + 1])))
                    i += 3
                else:
                    i += 1                                   # línea de '=' suelta: no se escribe
                continue
            sub = next((m.group(1) for p in patrones_sub if (m := re.match(p, l.rstrip()))), None)
            if sub:
                cerrar_cuerpo()
                bloques.append(('subtitulo', sub.strip()))
            else:
                cuerpo.append(l)
            i += 1
        cerrar_cuerpo()
        return bloques

    def borrar_imagenes(self):
        """Borra la carpeta temporal con las figuras (llamar después de generar_pdf)."""
        if self.carpeta_imagenes and os.path.isdir(self.carpeta_imagenes):
            shutil.rmtree(self.carpeta_imagenes, ignore_errors=True)
        self.carpeta_imagenes = None

    # -------------------------- PDF --------------------------
    @staticmethod
    def _limpiar(texto):
        # FPDF clásico no soporta bien algunos símbolos como '²' o '±'.
        # Los reemplazamos por equivalentes para evitar errores de codificación.
        texto = texto.replace('²', '^2').replace('±', '+/-').replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('Á', 'A').replace('É', 'E').replace('Í', 'I').replace('Ó', 'O').replace('Ú', 'U').replace('ñ', 'n').replace('Ñ', 'N')
        return texto.encode('latin-1', 'ignore').decode('latin-1')

    def generar_pdf(self, nombre_archivo="Respuestas.pdf", video=None, titulo="Respuestas"):
        nombre_video = os.path.basename(video) if video else ""
        fecha = datetime.now().strftime("%d/%m/%Y %H:%M")

        class PDFConEncabezado(FPDF):
            """FPDF llama solo a header() y footer() en cada página nueva. En la portada (página 1) no se dibujan."""
            def header(self):
                if self.page_no() == 1:
                    return
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(110)
                self.set_xy(self.l_margin, 10)
                self.cell(0, 5, CapturadorPDF._limpiar(f"{titulo} - {nombre_video}" if nombre_video else titulo), align="L")
                self.set_xy(self.l_margin, 10)
                self.cell(0, 5, fecha, align="R")
                self.set_draw_color(170)
                self.line(self.l_margin, 16, self.w - self.r_margin, 16)
                self.set_text_color(0)
                self.set_y(20)

            def footer(self):
                if self.page_no() == 1:
                    return
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(110)
                self.cell(0, 10, f"Página {self.page_no()} de {{nb}}", align="C")   # {nb} = total de páginas
                self.set_text_color(0)

        COLOR_TITULO = (25, 60, 120)       # azul oscuro
        COLOR_SUBTITULO = (60, 60, 60)     # gris oscuro
        pdf = PDFConEncabezado()
        ancho_util = pdf.w - pdf.l_margin - pdf.r_margin
        alto_util = pdf.h - 20 - pdf.b_margin              # 20 mm arriba por el encabezado

        # ---------------- Portada ----------------
        pdf.add_page()
        pdf.set_y(95)
        pdf.set_font("Helvetica", "B", 28)
        pdf.cell(0, 14, self._limpiar(titulo), align="C")
        pdf.ln(20)
        pdf.set_draw_color(120)
        pdf.line(pdf.w / 2 - 40, pdf.get_y(), pdf.w / 2 + 40, pdf.get_y())
        pdf.ln(10)
        if nombre_video:
            pdf.set_font("Helvetica", "", 14)
            pdf.cell(0, 8, self._limpiar(f"Video analizado: {nombre_video}"), align="C")
            pdf.ln(10)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(90)
        pdf.cell(0, 6, f"Generado el {fecha}", align="C")
        pdf.set_text_color(0)

        # ---------------- Contenido ----------------
        pdf.add_page()

        # Junta los textos consecutivos en un solo bloque
        bloques = []
        for item in self.contenido:
            if item[0] == 'texto' and bloques and bloques[-1][0] == 'texto':
                bloques[-1] = ('texto', bloques[-1][1] + item[1])
            else:
                bloques.append(item)

        for b in bloques:
            if b[0] == 'texto':
                for tipo, contenido in self._separar_titulos(self._limpiar(b[1])):
                    pdf.set_x(pdf.l_margin)
                    if tipo == 'titulo':
                        if pdf.get_y() > pdf.h - pdf.b_margin - 35:       # que no quede un título solo al pie de la página
                            pdf.add_page()
                        pdf.ln(4)
                        pdf.set_font("Helvetica", "B", 15)
                        pdf.set_text_color(*COLOR_TITULO)
                        pdf.multi_cell(0, 8, txt=contenido)
                        pdf.set_draw_color(*COLOR_TITULO)
                        pdf.set_line_width(0.5)
                        pdf.line(pdf.l_margin, pdf.get_y() + 1, pdf.w - pdf.r_margin, pdf.get_y() + 1)
                        pdf.set_line_width(0.2)
                        pdf.ln(5)
                    elif tipo == 'subtitulo':
                        if pdf.get_y() > pdf.h - pdf.b_margin - 25:
                            pdf.add_page()
                        pdf.ln(2)
                        pdf.set_font("Helvetica", "B", 11)
                        pdf.set_text_color(*COLOR_SUBTITULO)
                        pdf.multi_cell(0, 6, txt=contenido)
                        pdf.ln(1)
                    else:
                        texto = contenido.strip('\n')
                        if not texto.strip():
                            continue
                        # Courier es clave para que las tablas del código no se deformen
                        pdf.set_font("Courier", size=9)
                        pdf.multi_cell(0, 4, txt=texto)
                        pdf.ln(2)
                    pdf.set_text_color(0)
            else:
                _, ruta, titulo_img = b
                img = cv2.imread(ruta)
                if img is None:
                    continue
                alto_px, ancho_px = img.shape[:2]
                w = ancho_util
                h = w * alto_px / ancho_px
                if h > 0.9 * alto_util:                 # figuras muy altas: se achican para que entren en una página
                    h = 0.9 * alto_util
                    w = h * ancho_px / alto_px
                alto_titulo = 5 if titulo_img else 0
                if pdf.get_y() + alto_titulo + h > pdf.h - pdf.b_margin:
                    pdf.add_page()
                if titulo_img:
                    pdf.set_font("Courier", "B", 9)
                    pdf.set_x(pdf.l_margin)
                    pdf.cell(0, 5, txt=self._limpiar(titulo_img))
                    pdf.ln(5)
                y = pdf.get_y()
                pdf.image(ruta, x=pdf.l_margin + (ancho_util - w) / 2, y=y, w=w, h=h)
                pdf.set_y(y + h + 4)

        pdf.output(nombre_archivo)










# ======================================================================
# UTILIDADES COMUNES
# ======================================================================
def _mascara_pupila_elipse(roi, area_min_px=250, area_max_px=2000, canal=2):
    """Pupila segmentada con el método del Módulo 3.2. Devuelve (máscara uint8 0/255, resultado) o (None, None)."""
    par = _parametros_segmentacion(area_min_px, area_max_px)
    c = np.ascontiguousarray(roi[:, :, canal]) if roi.ndim == 3 else np.ascontiguousarray(roi)
    res = _medir_pupila(c, _centro_inicial(c, par), par)
    if res is None:
        return None, None
    m = np.zeros(c.shape, np.uint8)
    cv2.ellipse(m, res['elipse'], 255, -1)
    return m, res
 
 
def leer_frame_en(video_path, t_seg):
    """Lee el frame del instante t_seg (en segundos)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t_seg * fps)))
    ret, f = cap.read()
    cap.release()
    return f if ret else None
 
 
def _banda_borde(mascara, ancho=5):
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ancho, ancho))
    return cv2.morphologyEx(mascara, cv2.MORPH_GRADIENT, el) > 0


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

# -------------------------------------------------------------------------------------
# MÓDULO 1.1
# -------------------------------------------------------------------------------------


def procesar_modulo_1_1(frame, x, y, w, h):
    print("\n" + "=" * 70)
    print("--- 1.1 Resolución, cuantización y espacios de color ---")
    print("=" * 70)
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
    
    print("\n" + "=" * 70)
    print("--- ANÁLISIS ESPECÍFICO: Pérdida de Definición en Nuestro Video ---")
    print("=" * 70)
            
    print("\n" + "=" * 40)
    print("Estas respuestas que aplican únicamente a nuestro video se imprimen por separado, para darles un formato más agradable.")
    print("=" * 40)
    
    print("""
    
    - Al reducir la resolución espacial del ojo analizado al 50%, 25% y 5% (y reescalarla con vecino más cercano), 
      se evidencia el fenómeno de aliasing espacial y la pérdida progresiva de altas frecuencias.
    - En el 50% los detalles finos del estroma iridiano de nuestro registro se atenúan levemente; en el 25% los bordes 
      comienzan a pixelarse; y en el 5% el contorno pupilar se vuelve completamente discontinuo e 
      impreciso, demostrando que un submuestreo excesivo destruye la geometría necesaria para la segmentación.
    """)
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
    
# -------------------------------------------------------------------------------------
# MÓDULO 1.2
# -------------------------------------------------------------------------------------

def rango_pupila_iris(roi, area_min_px=250, area_max_px=2000):
    """[f1, f2] = niveles de gris que abarcan la pupila y el iris (sin reflejos). Respaldo: percentiles 1 y 40."""
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    m, res = _mascara_pupila_elipse(roi, area_min_px, area_max_px)
    if m is None:
        return float(np.percentile(gris, 1)), float(np.percentile(gris, 40))
    cx, cy = res['centro']
    r = np.sqrt(res['area'] / np.pi)
    yy, xx = np.indices(gris.shape)
    zona = (np.hypot(xx - cx, yy - cy) <= 2.6 * r) & (gris < 235)
    v = gris[zona]
    return float(np.percentile(v, 1)), float(np.percentile(v, 99))
 
 
def escalar_rango(gris, f1, f2):
    """Transformación lineal: f1 -> 0 y f2 -> 255; lo que queda afuera se satura."""
    return (np.clip((gris.astype(np.float32) - f1) / max(f2 - f1, 1e-6), 0, 1) * 255).astype(np.uint8)
    
def procesar_modulo_1_2(frame, x, y, w, h):
    print("\n" + "=" * 70)
    print("--- 1.2 Contraste e histograma ---")
    print("=" * 70)
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
    f1, f2 = rango_pupila_iris(roi)
    roi_escalada = escalar_rango(roi_gris, f1, f2)
    print(f"Escalado por rango de interés: [f1, f2] = [{f1:.0f}, {f2:.0f}]")
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

# ======================================================================
# B) 1.2.4  Ruido y filtrado espacial 
# ======================================================================
def _agregar_ruido(gris, tipo, rng):
    g = gris.astype(np.float32)
    if tipo == 'Gaussiano':
        r = g + rng.normal(0, 20, g.shape)
    elif tipo == 'Sal y pimienta':
        r = g.copy()
        u = rng.random(g.shape)
        r[u < 0.025] = 0
        r[u > 0.975] = 255
    else:  # Speckle (multiplicativo)
        r = g + g * rng.normal(0, 0.25, g.shape)
    return np.clip(r, 0, 255).astype(np.uint8)
 
 
def procesar_modulo_1_2_4(roi, area_min_px=250, area_max_px=2000):
    print("\n" + "=" * 70)
    print("--- 1.2.4 Ruido y filtrado espacial ---")
    print("=" * 70)
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    rng = np.random.default_rng(0)
 
    filtros = {
        'Media': lambda im, k: cv2.blur(im, (k, k)),
        'Mediana': lambda im, k: cv2.medianBlur(im, k),
        'Gaussiano': lambda im, k: cv2.GaussianBlur(im, (k, k), 0),
    }
 
    # ---------- (1) Ruidos típicos y filtros de 5x5 ----------
    ruidos = ['Gaussiano', 'Sal y pimienta', 'Speckle']
    fig, axs = plt.subplots(3, 4, figsize=(15, 11))
    mejores = {}
    print("\nPSNR (dB) / SSIM de cada imagen contra la original SIN ruido (filtros 5x5):")
    print(f"{'Ruido':<16}{'Sin filtrar':>16}" + "".join(f"{n:>16}" for n in filtros))
    for i, tipo in enumerate(ruidos):
        ruidosa = _agregar_ruido(gris, tipo, rng)
        fila = [('Sin filtrar', ruidosa)] + [(n, f(ruidosa, 5)) for n, f in filtros.items()]
        textos, vals = [], {}
        for j, (nombre, im) in enumerate(fila):
            p, s = psnr(gris, im, data_range=255), ssim(gris, im, data_range=255)
            textos.append(f"{p:5.1f}/{s:.3f}")
            vals[nombre] = (p, s)
            axs[i, j].imshow(im, cmap='gray', vmin=0, vmax=255)
            axs[i, j].set_title(f"{tipo}\n{nombre}  {p:.1f} dB / {s:.3f}", fontsize=9)
            axs[i, j].axis('off')
        print(f"{tipo:<16}" + "".join(f"{t:>16}" for t in textos))
        mejores[tipo] = max(filtros, key=lambda n: vals[n][1])
    plt.tight_layout()
 
    print("\nMejor filtro (por SSIM) para cada ruido:")
    for tipo, n in mejores.items():
        print(f"   - {tipo}: {n}")
 
    # ---------- (2) Pregunta de la guía: ¿qué filtro preserva los bordes de la pupila y elimina los destellos? ----------
    mascara, res = _mascara_pupila_elipse(roi, area_min_px, area_max_px)
    sat = (gris >= 235).astype(np.uint8)
    n, et, st, cen = cv2.connectedComponentsWithStats(sat, connectivity=8)
    dest = np.zeros(gris.shape, np.uint8)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] > 400:
            continue                                         # zonas saturadas grandes (piel/esclerótica) no son destellos
        if res is not None:
            r = np.sqrt(res['area'] / np.pi)
            if np.hypot(cen[k][0] - res['centro'][0], cen[k][1] - res['centro'][1]) > 3.0 * r:
                continue                                     # solo destellos cerca de la pupila
        dest[et == k] = 1
 
    print("\n--- Pregunta: ¿qué filtro preserva los bordes de la pupila eliminando los destellos? ---")
    if not dest.any() or mascara is None:
        print("No se detectaron destellos saturados (>=235) junto a la pupila, o no se pudo segmentar la pupila: "
              "no se puede hacer esta comparación en este frame.")
        plt.show()
        return
    dest_d = cv2.dilate(dest, np.ones((3, 3), np.uint8), iterations=2)
    ref = cv2.inpaint(gris, dest_d * 255, 5, cv2.INPAINT_TELEA)    # "lo ideal": imagen sin el destello
    banda = _banda_borde(mascara)
    gx = lambda im: np.hypot(cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3))
    grad_ref = gx(ref)[banda].mean()
    exceso0 = float(gris[dest_d > 0].mean() - ref[dest_d > 0].mean())
    print(f"Destello detectado: {int(dest.sum())} px (nivel medio {gris[dest > 0].mean():.0f}; "
          f"sin destello sería ~{ref[dest > 0].mean():.0f})")
 
    kernels = [5, 11, 21]
    filas = []
    print(f"\n{'Filtro':<11}{'Kernel':>7}{'Destello eliminado':>20}{'Borde retenido':>16}{'PSNR':>9}{'SSIM':>8}")
    for nombre, f in filtros.items():
        for k in kernels:
            im = f(gris, k)
            elim = float(np.clip(100 * (1 - (float(im[dest_d > 0].mean() - ref[dest_d > 0].mean()) / exceso0)), 0, 100))
            ret = 100 * gx(im)[banda].mean() / grad_ref
            p, s = psnr(ref, im, data_range=255), ssim(ref, im, data_range=255)
            filas.append((nombre, k, elim, ret, p, s, im))
            print(f"{nombre:<11}{k:>5}x{k:<2}{elim:>18.0f} %{ret:>14.0f} %{p:>9.1f}{s:>8.3f}")
 
    cand = [f for f in filas if f[2] >= 80]
    if cand:
        mejor = max(cand, key=lambda f: f[3])
        print(f"\n-> Entre los que eliminan >= 80 % del destello, el que mejor conserva el borde pupilar es: "
              f"{mejor[0]} {mejor[1]}x{mejor[1]} (destello eliminado {mejor[2]:.0f} %, borde retenido {mejor[3]:.0f} %).")
    else:
        mejor = max(filas, key=lambda f: f[2])
        print(f"\n-> Ningún filtro llega a eliminar el 80 % del destello. El que más lo reduce es {mejor[0]} "
              f"{mejor[1]}x{mejor[1]} ({mejor[2]:.0f} %), con borde retenido {mejor[3]:.0f} %.")
    print("   (Si ninguno es satisfactorio, el destello se trata con morfología/relleno, como hace el pipeline en _rellenar_reflejos.)")
    
    print("\n" + "=" * 70)
    print("--- RESPUESTA APLICADA: Filtro Óptimo para Destellos en el Video ---")
    print("=" * 70)
            
    print("\n" + "=" * 40)
    print("Estas respuestas que aplican únicamente a nuestro video se imprimen por separado, para darles un formato más agradable.")
    print("=" * 40)
    
    
    
    print("""
    
    - El filtro de Mediana (con kernel de 11x11) es el que preserva de manera idónea el borde de 
      la pupila en nuestro registro mientras elimina los destellos especulares del flash. Esto ocurre porque la mediana 
      es un operador no lineal que descarta valores extremos (como el brillo saturado del reflejo) 
      sin promediarlos con los vecinos, evitando el efecto de difuminado típico de los filtros lineales (Media/Gaussiano).
    - La calidad de la imagen filtrada frente a la original sin ruido se cuantifica mediante:
      * PSNR (Peak Signal-to-Noise Ratio): Relación entre la potencia máxima y el ruido de reconstrucción en dB.
      * SSIM (Structural Similarity Index): Similitud estructural percibida (rango de 0 a 1).
    """)
    
    # Figura: zoom sobre la pupila para cada filtro/kernel
    cx, cy = res['centro']
    hz = int(2.6 * np.sqrt(res['area'] / np.pi))
    y0, x0 = max(0, int(cy) - hz), max(0, int(cx) - hz)
    sl = (slice(y0, y0 + 2 * hz), slice(x0, x0 + 2 * hz))
    fig2, axs2 = plt.subplots(3, 4, figsize=(14, 10))
    for i, nombre in enumerate(filtros):
        axs2[i, 0].imshow(gris[sl], cmap='gray', vmin=0, vmax=255)
        axs2[i, 0].set_title('Original (con destello)', fontsize=9)
        for j, k in enumerate(kernels):
            f = [x for x in filas if x[0] == nombre and x[1] == k][0]
            axs2[i, j + 1].imshow(f[6][sl], cmap='gray', vmin=0, vmax=255)
            axs2[i, j + 1].set_title(f"{nombre} {k}x{k}\ndestello -{f[2]:.0f}% | borde {f[3]:.0f}%", fontsize=9)
    for a in axs2.ravel():
        a.axis('off')
    plt.suptitle('Destello del flash vs. borde de la pupila según filtro y tamaño de kernel')
    plt.tight_layout()
    plt.show()

# ======================================================================
# C) 1.3  Bordes, TFD-2D, PSD y filtrado en frecuencia
# -------------------------------------------------------------------------------------

def _canny_auto(img):
    """Canny con umbrales automáticos (alto = percentil 92 del gradiente, bajo = 40 % del alto): sirve con cualquier brillo/filtrado."""
    mag = cv2.magnitude(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3))
    alto = float(np.percentile(mag, 92))
    return cv2.Canny(img, 0.4 * alto, alto, L2gradient=True)
 
 
def _operadores_borde(img):
    """Devuelve {nombre: imagen binaria de bordes} (los gradientes se binarizan con Otsu)."""
    def binarizar(mag):
        mag = cv2.convertScaleAbs(mag)
        _, b = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return b
    kr_x = np.array([[1, 0], [0, -1]], np.float32)
    kr_y = np.array([[0, 1], [-1, 0]], np.float32)
    kp_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], np.float32)
    kp_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], np.float32)
    f = lambda k: cv2.filter2D(img, cv2.CV_64F, k)
    return {
        'Roberts': binarizar(cv2.magnitude(f(kr_x), f(kr_y))),
        'Prewitt': binarizar(cv2.magnitude(f(kp_x), f(kp_y))),
        'Sobel': binarizar(cv2.magnitude(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3),
                                          cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3))),
        'Laplaciano': binarizar(cv2.Laplacian(img, cv2.CV_64F)),
        'Canny': _canny_auto(img),
    }
 
 
def _continuidad_y_espurios(bordes, mascara, n_sect=72):
    """
    Continuidad = % de los 72 sectores angulares del contorno pupilar donde hay al menos un píxel de borde.
    Espurios   = % de píxeles lejos del contorno pupilar que fueron marcados como borde (ruido/textura).
    """
    banda = _banda_borde(mascara, 7)
    ys, xs = np.nonzero(mascara)
    cy, cx = ys.mean(), xs.mean()
    by, bx = np.nonzero(banda & (bordes > 0))
    sect = ((np.arctan2(by - cy, bx - cx) + np.pi) / (2 * np.pi) * n_sect).astype(int) % n_sect
    cont = 100.0 * len(np.unique(sect)) / n_sect
    lejos = cv2.dilate(banda.astype(np.uint8), np.ones((15, 15), np.uint8)) == 0
    esp = 100.0 * ((bordes > 0) & lejos).sum() / lejos.sum()
    return cont, esp
 
 
def _tfd(img):
    F = np.fft.fft2(img.astype(np.float64))
    return np.abs(F), np.angle(F)
 
 
def _recon(mag, fase):
    return np.real(np.fft.ifft2(mag * np.exp(1j * fase)))
 
 
def _norm_vis(im):
    lo, hi = np.percentile(im, 1), np.percentile(im, 99)
    return np.clip((im - lo) / max(hi - lo, 1e-9), 0, 1)
 
 
def _corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
 
 
def _psd_radial(img):
    """PSD = |TFD|^2 (sin la componente continua) promediada en anillos de frecuencia. Devuelve (frecuencia [ciclos/px], PSD)."""
    im = img.astype(np.float64)
    P = np.abs(np.fft.fftshift(np.fft.fft2(im - im.mean()))) ** 2
    h, w = P.shape
    yy, xx = np.indices(P.shape)
    r = np.hypot(xx - w // 2, yy - h // 2).astype(int)
    n = min(h, w) // 2
    suma = np.bincount(r.ravel(), P.ravel(), minlength=n + 1)[:n]
    cuenta = np.maximum(np.bincount(r.ravel(), minlength=n + 1)[:n], 1)
    return np.arange(n) / min(h, w), suma / cuenta, P
 
 
def _filtrar_frecuencia(img, H_fun):
    """Filtra en frecuencia con H(f) (f en ciclos/píxel) usando relleno por reflejo para evitar el efecto de borde."""
    pad = min(img.shape) // 4
    im = np.pad(img.astype(np.float64), pad, mode='reflect')
    fy = np.fft.fftfreq(im.shape[0])[:, None]
    fx = np.fft.fftfreq(im.shape[1])[None, :]
    H = H_fun(np.hypot(fx, fy))
    out = np.real(np.fft.ifft2(np.fft.fft2(im) * H))
    return out[pad:-pad, pad:-pad]
 
 
def procesar_modulo_1_3(frame, x, y, w, h, frame_otro=None, sigma_filtro=2.0, area_min_px=250, area_max_px=2000):
    print("\n" + "=" * 70)
    print("--- 1.3 Bordes y dominio frecuencial ---")
    print("=" * 70)
    roi = np.ascontiguousarray(frame[y:y + h, x:x + w])
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(gris, (5, 5), 0)
 
    # ---------------- 1.3.1 Bordes: original vs. filtrada ----------------
    mascara, res = _mascara_pupila_elipse(roi, area_min_px, area_max_px)
    b_orig, b_filt = _operadores_borde(gris), _operadores_borde(suave)
    fig, axs = plt.subplots(2, 6, figsize=(20, 7.5))
    axs[0, 0].imshow(gris, cmap='gray'); axs[0, 0].set_title('ROI original')
    axs[1, 0].imshow(suave, cmap='gray'); axs[1, 0].set_title('ROI filtrada (Gaussiano 5x5)')
    print("\n1.3.1 Continuidad del contorno pupilar y bordes espurios (gradientes binarizados con Otsu; Canny con umbrales automáticos)")
    if mascara is not None:
        print(f"{'Operador':<12}{'Continuidad orig.':>19}{'filtrada':>10}{'Espurios orig.':>17}{'filtrada':>10}")
    for j, nombre in enumerate(b_orig, start=1):
        axs[0, j].imshow(b_orig[nombre], cmap='gray'); axs[1, j].imshow(b_filt[nombre], cmap='gray')
        t0, t1 = nombre, nombre
        if mascara is not None:
            c0, e0 = _continuidad_y_espurios(b_orig[nombre], mascara)
            c1, e1 = _continuidad_y_espurios(b_filt[nombre], mascara)
            print(f"{nombre:<12}{c0:>17.0f} %{c1:>8.0f} %{e0:>15.1f} %{e1:>8.1f} %")
            t0, t1 = f"{nombre} (original)\ncontinuidad {c0:.0f}%", f"{nombre} (filtrada)\ncontinuidad {c1:.0f}%"
        axs[0, j].set_title(t0, fontsize=9); axs[1, j].set_title(t1, fontsize=9)
    for a in axs.ravel():
        a.axis('off')
    plt.tight_layout()
    if mascara is None:
        print("(No se pudo segmentar la pupila: no se calcula la continuidad; mirá las figuras.)")
 
    # ---------------- 1.3.2 TFD-2D ----------------
    magA, faseA = _tfd(gris)
    fig3, axs3 = plt.subplots(1, 3, figsize=(15, 5))
    axs3[0].imshow(gris, cmap='gray'); axs3[0].set_title('ROI original')
    axs3[1].imshow(np.fft.fftshift(20 * np.log10(magA + 1)), cmap='gray'); axs3[1].set_title('Magnitud (dB, escala log)')
    axs3[2].imshow(np.fft.fftshift(faseA), cmap='gray'); axs3[2].set_title('Fase (rad)')
    for a in axs3:
        a.axis('off')
    plt.tight_layout()
 
    if frame_otro is not None:
        otra = cv2.cvtColor(np.ascontiguousarray(frame_otro[y:y + h, x:x + w]), cv2.COLOR_BGR2GRAY)
        nombre_B = 'Otro frame'
    else:
        from skimage import data
        otra = cv2.resize(data.camera(), (gris.shape[1], gris.shape[0]), interpolation=cv2.INTER_AREA)
        nombre_B = 'Imagen distinta (cameraman)'
    magB, faseB = _tfd(otra)
    r_mag = _recon(magA, np.zeros_like(faseA))
    r_fase = _recon(np.ones_like(magA), faseA)
    r_AmagBfase = _recon(magA, faseB)
    r_BmagAfase = _recon(magB, faseA)
    fig4, axs4 = plt.subplots(2, 4, figsize=(17, 8.5))
    paneles = [('A: original', gris), (f'B: {nombre_B}', otra), ('A: solo magnitud', r_mag), ('A: solo fase', r_fase),
               ('Magnitud de A + fase de B', r_AmagBfase), ('Magnitud de B + fase de A', r_BmagAfase)]
    for a, (t, im) in zip(axs4.ravel(), paneles):
        a.imshow(_norm_vis(im) if 'solo' in t or '+' in t else im, cmap='gray'); a.set_title(t); a.axis('off')
    axs4[1, 2].axis('off'); axs4[1, 3].axis('off')
    plt.tight_layout()
    grad = lambda im: cv2.magnitude(cv2.Sobel(im.astype(np.float32), cv2.CV_32F, 1, 0), cv2.Sobel(im.astype(np.float32), cv2.CV_32F, 0, 1))
    print("\n1.3.2 Correlación de cada reconstrucción con las imágenes originales (1 = idéntica):")
    print(f"   Solo magnitud de A      -> con A: {_corr(r_mag, gris):+.2f} | con los BORDES de A: {_corr(np.abs(r_mag - r_mag.mean()), grad(gris)):+.2f}")
    print(f"   Solo fase de A          -> con A: {_corr(r_fase, gris):+.2f} | con los BORDES de A: {_corr(np.abs(r_fase - r_fase.mean()), grad(gris)):+.2f}")
    print(f"   Magnitud A + fase B     -> con A: {_corr(r_AmagBfase, gris):+.2f} | con B: {_corr(r_AmagBfase, otra):+.2f}")
    print(f"   Magnitud B + fase A     -> con A: {_corr(r_BmagAfase, gris):+.2f} | con B: {_corr(r_BmagAfase, otra):+.2f}")
 
    # ---------------- 1.3.3 PSD antes y después del filtrado espacial ----------------
    imagenes = {'Original': gris, 'Media 5x5': cv2.blur(gris, (5, 5)),
                'Mediana 5x5': cv2.medianBlur(gris, 5), 'Gaussiano 5x5': suave}
    fig5, (axp, axq) = plt.subplots(1, 2, figsize=(15, 5.5))
    print("\n1.3.3 Fracción de la potencia (sin componente continua) en altas frecuencias (> 0.1 ciclos/píxel):")
    for nombre, im in imagenes.items():
        f, psd, P = _psd_radial(im)
        axp.loglog(f[1:], psd[1:], label=nombre)
        total = psd[1:] * np.arange(1, len(psd))                    # potencia por anillo (aprox.)
        alta = total[f[1:] > 0.1].sum() / total.sum()
        print(f"   {nombre:<15} {100 * alta:6.2f} %")
    axp.set_xlabel('Frecuencia (ciclos/píxel)'); axp.set_ylabel('PSD radial'); axp.grid(True, which='both', alpha=0.4)
    axp.set_title('PSD radial antes y después del filtrado espacial'); axp.legend()
    _, _, P0 = _psd_radial(gris)
    _, _, P1 = _psd_radial(suave)
    axq.imshow(np.log10(P0 + 1) - np.log10(P1 + 1), cmap='magma')
    axq.set_title('log PSD(original) - log PSD(gaussiano): dónde se perdió potencia'); axq.axis('off')
    plt.tight_layout()
 
    # ---------------- 1.3.4 Filtrado en frecuencia ----------------
    # Gaussiana de sigma_filtro en espacio <-> H(f) = exp(-2 pi^2 sigma^2 f^2) en frecuencia (par de Fourier)
    fc = np.sqrt(np.log(np.sqrt(2)) / (2 * np.pi ** 2 * sigma_filtro ** 2))   # frecuencia de corte a -3 dB
    lp_gauss = _filtrar_frecuencia(gris, lambda f: np.exp(-2 * np.pi ** 2 * sigma_filtro ** 2 * f ** 2))
    hp_gauss = _filtrar_frecuencia(gris, lambda f: 1 - np.exp(-2 * np.pi ** 2 * sigma_filtro ** 2 * f ** 2))
    lp_ideal = _filtrar_frecuencia(gris, lambda f: (f <= fc).astype(float))
    esp_gauss = cv2.GaussianBlur(gris, (0, 0), sigma_filtro, borderType=cv2.BORDER_REFLECT)
    u8 = lambda a: np.clip(a, 0, 255).astype(np.uint8)
    lp_gauss8, lp_ideal8 = u8(lp_gauss), u8(lp_ideal)
    fig6, axs6 = plt.subplots(2, 3, figsize=(15, 10))
    paneles = [('Original', gris), (f'Pasa-bajos gaussiano (frecuencia), fc={fc:.3f} c/px', lp_gauss8),
               (f'Pasa-bajos IDEAL, fc={fc:.3f} c/px', lp_ideal8),
               (f'Filtro gaussiano ESPACIAL sigma={sigma_filtro:g}', esp_gauss),
               ('Pasa-altos gaussiano (detalle/bordes)', _norm_vis(hp_gauss)),
               ('|Pasa-bajos frecuencia - espacial| x20', np.clip(np.abs(lp_gauss8.astype(float) - esp_gauss) * 20, 0, 255))]
    for a, (t, im) in zip(axs6.ravel(), paneles):
        a.imshow(im, cmap='gray'); a.set_title(t, fontsize=9); a.axis('off')
    plt.tight_layout()
    print(f"\n1.3.4 Filtros en frecuencia (corte fc = {fc:.3f} ciclos/píxel, equivalente a un gaussiano espacial de sigma = {sigma_filtro:g}):")
    print(f"   Pasa-bajos gaussiano (frecuencia) vs. gaussiano espacial: PSNR = {psnr(esp_gauss, lp_gauss8, data_range=255):.1f} dB | "
          f"SSIM = {ssim(esp_gauss, lp_gauss8, data_range=255):.4f}")
    print(f"   Contra la original -> PB gaussiano: PSNR {psnr(gris, lp_gauss8, data_range=255):.1f} dB / SSIM {ssim(gris, lp_gauss8, data_range=255):.3f} | "
          f"PB ideal: PSNR {psnr(gris, lp_ideal8, data_range=255):.1f} dB / SSIM {ssim(gris, lp_ideal8, data_range=255):.3f}")
    print("   (El pasa-bajos ideal corta de golpe y genera 'anillos' (ringing) alrededor de los bordes; el gaussiano no.)")
    
    print("\n" + "=" * 70)
    print("--- RESPUESTA APLICADA: Bordes y Fase en Nuestro Registro ---")
    print("=" * 70)
        
    print("\n" + "=" * 40)
    print("Estas respuestas que aplican únicamente a nuestro video se imprimen por separado, para darles un formato más agradable.")
    print("=" * 40)
    print("""
    
    - Operadores de bordes: Canny demuestra la mayor robustez y continuidad (100%) sobre nuestro video 
      al utilizar histéresis y doble umbral, minimizando falsos positivos tras el pre-filtrado gaussiano.
    - Reconstrucción por Transformada de Fourier 2D (TFD-2D): Al cruzar la magnitud de nuestra imagen 
      con la fase de otra, se comprueba empíricamente que la Fase concentra la información 
      estructural y geométrica de nuestro ojo (delimitando la posición de los contornos), mientras 
      que la Magnitud solo define la distribución energética global de intensidades y contrastes.
    """)
    
    plt.show()
 

    # La fase es la que conserva la información estructural
    
# -------------------------------------------------------------------------------------
# MÓDULO 2
# -------------------------------------------------------------------------------------
def _poligono_a_mascara(contorno_rc, shape):
    """Contorno del snake (filas, columnas) -> máscara 0/255."""
    m = np.zeros(shape, np.uint8)
    pts = np.round(np.asarray(contorno_rc)[:, ::-1]).astype(np.int32)
    cv2.fillPoly(m, [pts], 255)
    return m
 
 
def anotar_contorno_manual(roi, ruta_salida='referencia_pupila.png', forma='elipse', escala=2):
    """
    Herramienta para dibujar a mano el contorno de la pupila sobre la ROI:
      clic izquierdo = agregar punto (poné 12-20 puntos sobre el borde) | z = deshacer | Enter = terminar | Esc = cancelar
    forma='elipse' ajusta una elipse a tus puntos; forma='poligono' rellena el polígono que unen.
    Guarda una máscara PNG (blanco = pupila) con el tamaño exacto de la ROI.
    """
    puntos = []
    ventana = 'Anotar pupila: clic=punto | z=deshacer | Enter=terminar | Esc=cancelar'
    base = cv2.resize(roi, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
 
    def click(ev, px, py, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            puntos.append((px / escala, py / escala))
 
    cv2.namedWindow(ventana)
    cv2.setMouseCallback(ventana, click)
    while True:
        vis = base.copy()
        for (px, py) in puntos:
            cv2.circle(vis, (int(px * escala), int(py * escala)), 3, (0, 255, 0), -1)
        if len(puntos) >= 2:
            cv2.polylines(vis, [np.round(np.array(puntos) * escala).astype(np.int32)], len(puntos) >= 3, (0, 255, 0), 1)
        cv2.imshow(ventana, vis)
        k = cv2.waitKey(20) & 0xFF
        if k == 13 and len(puntos) >= 8:
            break
        if k == ord('z') and puntos:
            puntos.pop()
        if k == 27:
            cv2.destroyAllWindows()
            print("Anotación cancelada.")
            return None
    cv2.destroyAllWindows()
    m = np.zeros(roi.shape[:2], np.uint8)
    pts = np.array(puntos, np.float32)
    if forma == 'elipse':
        cv2.ellipse(m, cv2.fitEllipse(pts), 255, -1)
    else:
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 255)
    cv2.imwrite(ruta_salida, m)
    print(f"Referencia guardada en '{ruta_salida}' ({int((m > 0).sum())} px).")
    return m
 
 
def guardar_roi_para_anotar(roi, ruta='roi_para_anotar.png', ruta_ref='referencia_pupila.png'):
    """Si preferís anotar en ImageJ: guarda la ROI exacta; dibujá la pupila, Edit > Selection > Create Mask y guardá como ruta_ref."""
    cv2.imwrite(ruta, roi)
    print(f"ROI guardada en '{ruta}'. Anotala y guardá la máscara como '{ruta_ref}' (blanco = pupila, mismo tamaño).")
 
 
def _mayor_componente(mascara):
    n, et, st, _ = cv2.connectedComponentsWithStats((mascara > 0).astype(np.uint8), 8)
    if n <= 1:
        return np.zeros(mascara.shape, bool)
    return et == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
 
 
def _dice_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    s = a.sum() + b.sum()
    return (2.0 * inter / s if s else 0.0), (inter / union if union else 0.0)
 
 
def procesar_modulo_2_3(roi, mascaras, ruta_ref='referencia_pupila.png', area_min_px=250, area_max_px=2000):
    print("\n" + "=" * 70)
    print("--- 2.3 Comparación de máscaras contra la referencia manual (Dice / IoU) ---")
    print("=" * 70)
    if not os.path.exists(ruta_ref):
        print(f"No existe '{ruta_ref}'. Creala con anotar_contorno_manual(roi) o con guardar_roi_para_anotar(roi) + ImageJ.")
        return None
    ref = cv2.imread(ruta_ref, cv2.IMREAD_GRAYSCALE)
    if ref is None or ref.shape != roi.shape[:2]:
        print(f"La referencia debe tener el tamaño de la ROI {roi.shape[1]}x{roi.shape[0]} px.")
        return None
    ref = ref > 127
    if ref.mean() > 0.5:      # ImageJ a veces guarda la máscara invertida (pupila negra): se corrige sola
        ref = ~ref
        print("Aviso: la referencia venía invertida (pupila en negro); se invirtió automáticamente.")
    if ref.sum() == 0:
        print("La referencia está vacía (no hay píxeles blancos): revisá el archivo.")
        return None
    cand = dict(mascaras)
    m_el, _ = _mascara_pupila_elipse(roi, area_min_px, area_max_px)
    if m_el is not None:
        cand['Elipse por gradiente (Módulo 3.2)'] = m_el
 
    filas = []
    for nombre, m in cand.items():
        b = m > 0
        d_all, i_all = _dice_iou(b, ref)
        c = _mayor_componente(m)
        d_c, i_c = _dice_iou(c, ref)
        filas.append(dict(nombre=nombre, dice=d_all, iou=i_all, dice_c=d_c, iou_c=i_c,
                          area=int(b.sum()), err=100 * (int(b.sum()) - int(ref.sum())) / ref.sum(), masc=c))
    print(f"Área de la referencia manual: {int(ref.sum())} px")
    print(f"{'Método':<34}{'Dice':>7}{'IoU':>7}{'Dice (mayor comp.)':>20}{'IoU (mayor comp.)':>19}{'Área':>8}{'Error área':>12}")
    for f in sorted(filas, key=lambda f: -f['dice_c']):
        print(f"{f['nombre']:<34}{f['dice']:>7.3f}{f['iou']:>7.3f}{f['dice_c']:>20.3f}{f['iou_c']:>19.3f}{f['area']:>8d}{f['err']:>+11.1f}%")
 
    seg = [f for f in filas if f['nombre'] in ('Otsu', 'K-Means', 'Snake', 'Watershed')]
    mor = [f for f in filas if f['nombre'] in ('Umbral<45', 'Erosión', 'Dilatación', 'Apertura', 'Cierre', 'Cierre+Apertura (final)')]
    if seg:
        b = max(seg, key=lambda f: f['dice_c'])
        print(f"\n-> Mejor método de segmentación (2.1): {b['nombre']} (Dice {b['dice_c']:.3f}).")
    if mor:
        b = max(mor, key=lambda f: f['dice_c'])
        print(f"-> Mejor operación/combinación morfológica (2.2): {b['nombre']} (Dice {b['dice_c']:.3f}).")
        ero = [f for f in mor if f['nombre'] == 'Erosión']
        dil = [f for f in mor if f['nombre'] == 'Dilatación']
        if ero and dil:
            print(f"   La erosión cambia el área {ero[0]['err']:+.1f} % y la dilatación {dil[0]['err']:+.1f} % respecto de la referencia.")
 
    # Figura: contorno de la referencia (verde) vs. contorno de cada máscara (rojo)
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(ref)
    r = int(2.4 * np.sqrt(ref.sum() / np.pi))
    y0, x0 = max(0, int(ys.mean()) - r), max(0, int(xs.mean()) - r)
    sl = (slice(y0, y0 + 2 * r), slice(x0, x0 + 2 * r))
    n = len(filas)
    cols = 4
    fig, axs = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(4 * cols, 4 * int(np.ceil(n / cols))))
    for a in np.atleast_1d(axs).ravel():
        a.axis('off')
    for a, f in zip(np.atleast_1d(axs).ravel(), sorted(filas, key=lambda f: -f['dice_c'])):
        a.imshow(gris[sl], cmap='gray')
        a.contour(ref[sl], levels=[0.5], colors='lime', linewidths=1.5)
        if f['masc'].any():
            a.contour(f['masc'][sl], levels=[0.5], colors='red', linewidths=1.2)
        a.set_title(f"{f['nombre']}\nDice {f['dice_c']:.3f}", fontsize=9)
    plt.suptitle('Verde = referencia manual | Rojo = máscara del método (mayor componente)')
    plt.tight_layout()
    plt.show()
    return filas

def procesar_modulo_2(roi):
    print("\n" + "=" * 70)
    print("--- 2.1 y 2.2 Segmentación de la pupila y morfología ---")
    print("=" * 70)
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
    
    
    mascaras = {
               'Otsu': mascara_otsu, 'K-Means': mascara_kmeans,
               'Snake': _poligono_a_mascara(snake, roi.shape[:2]),
               'Watershed': ((marcadores > 1).astype(np.uint8) * 255),
               'Umbral<45': mascara_base, 'Erosión': erosion, 'Dilatación': dilatacion,
               'Apertura': apertura, 'Cierre': cierre, 'Cierre+Apertura (final)': mascara_limpia,
           }
    return mascara_limpia, mascaras

# -------------------------------------------------------------------------------------
# MÓDULO 3.1
# -------------------------------------------------------------------------------------

def _glcm_region(canal, region, niveles=32):
    """GLCM (distancia 1, 4 ángulos) usando solo los píxeles de 'region' (no se cuentan pares con píxeles de afuera)."""
    ys, xs = np.nonzero(region)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    q = (canal[y0:y1, x0:x1] // (256 // niveles)).astype(np.uint8)
    q[~region[y0:y1, x0:x1]] = niveles                              # nivel reservado para "fuera de la región"
    g = graycomatrix(q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                     levels=niveles + 1, symmetric=True, normed=False).astype(np.float64)
    g[niveles, :, :, :] = 0
    g[:, niveles, :, :] = 0
    g /= np.maximum(g.sum(axis=(0, 1), keepdims=True), 1)
    return {k: float(graycoprops(g, k).mean()) for k in ('contrast', 'homogeneity', 'energy', 'correlation')}


def procesar_modulo_3_1(roi, mascara_limpia, factor_iris=2.6, usar_elipse=False):
    """usar_elipse=True: en vez de la máscara del Módulo 2 usa la pupila por gradiente/elipse del Módulo 3.2
    (conviene si el Dice del 2.3 muestra que es mejor, y es más robusto con videos nuevos)."""
    print("\n" + "=" * 70)
    print("--- 3.1 Forma y textura ---")
    print("=" * 70)
    if usar_elipse:
        m_el, _ = _mascara_pupila_elipse(roi)
        if m_el is not None:
            mascara_limpia = m_el
        else:
            print("No se pudo segmentar con el método por gradiente; se usa la máscara del Módulo 2.")
    contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        print("No se detectó pupila en la máscara.")
        return
    c = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimetro = cv2.arcLength(c, True)
    M = cv2.moments(c)
    cX = M["m10"] / M["m00"] if M["m00"] else 0.0
    cY = M["m01"] / M["m00"] if M["m00"] else 0.0
    circ = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0
    excentricidad = np.nan
    if len(c) >= 5:
        (_, _), (d1, d2), _ = cv2.fitEllipse(c)
        a_, b_ = max(d1, d2) / 2, min(d1, d2) / 2
        excentricidad = float(np.sqrt(max(1 - (b_ / a_) ** 2, 0)))

    print("\n--- 3.1.1 Descriptores Morfométricos (respuestas genéricas) ---")
    print(f"Área (A): {area:.0f} px²  (diámetro equivalente {2 * np.sqrt(area / np.pi):.1f} px)")
    print(f"Perímetro (P): {perimetro:.2f} px")
    print(f"Centroide (X_c, Y_c): ({cX:.1f}, {cY:.1f})")
    print(f"Circularidad: {circ:.4f} (1 = círculo perfecto) | Excentricidad: {excentricidad:.3f} (0 = círculo)")

    # Regiones: pupila (mayor contorno) e iris (anillo, sin destellos)
    pup = np.zeros(mascara_limpia.shape, np.uint8)
    cv2.drawContours(pup, [c], -1, 255, -1)
    pup = pup > 0
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    yy, xx = np.indices(gris.shape)
    d = np.hypot(xx - cX, yy - cY)
    r_eq = np.sqrt(area / np.pi)
    iris = (d >= 1.25 * r_eq) & (d <= factor_iris * r_eq) & (gris < 235)
    regiones = {'Pupila': pup, 'Iris (anillo)': iris}

    vis = roi.copy()
    vis[iris] = (0.5 * vis[iris] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    vis[pup] = (0.5 * vis[pup] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    plt.figure(figsize=(5, 5))
    plt.imshow(vis[..., ::-1]); plt.axis('off')
    plt.title('Regiones: pupila (rojo) e iris (verde)\nSi el verde invade esclerótica/piel, bajá factor_iris')
    plt.tight_layout()
    plt.show()

    b, g, r = cv2.split(roi)
    canales = {'Rojo (R)': r, 'Verde (G)': g, 'Azul (B)': b}

    print("\n--- 3.1.2 Textura de Primer Orden (por región y canal) (respuestas genéricas) ---")
    entropia_iris = {}
    for reg_nombre, reg in regiones.items():
        print(f"[{reg_nombre}]  ({int(reg.sum())} px)")
        for nombre, canal in canales.items():
            px = canal[reg]
            if px.size == 0:
                continue
            hist, _ = np.histogram(px, bins=256, range=(0, 256), density=True)
            ent = float(entropy(hist + 1e-12, base=2))
            if reg_nombre.startswith('Iris'):
                entropia_iris[nombre] = ent
            print(f"   {nombre:<10} Media: {px.mean():6.2f} | Std: {px.std():6.2f} | "
                  f"Skewness: {skew(px):6.2f} | Entropía: {ent:5.2f} bits")

    print("\n--- 3.1.3 Textura de Segundo Orden (GLCM, 32 niveles) en el IRIS (respuestas genéricas) ---")
    glcm = {}
    for nombre, canal in canales.items():
        glcm[nombre] = _glcm_region(canal, iris)
        v = glcm[nombre]
        print(f"   {nombre:<10} Contraste: {v['contrast']:7.3f} | Homogeneidad: {v['homogeneity']:.4f} | "
              f"Energía: {v['energy']:.4f} | Correlación: {v['correlation']:.4f}")

    # Ranking de "riqueza textural": más entropía y contraste, menos homogeneidad y energía
    nombres = list(canales)
    def rango(valores, mayor_es_mejor):
        orden = sorted(nombres, key=lambda n: valores[n], reverse=mayor_es_mejor)
        return {n: orden.index(n) + 1 for n in nombres}
    rankings = {
        'entropía': rango(entropia_iris, True),
        'contraste': rango({n: glcm[n]['contrast'] for n in nombres}, True),
        'homogeneidad (menor=más textura)': rango({n: glcm[n]['homogeneity'] for n in nombres}, False),
        'energía (menor=más textura)': rango({n: glcm[n]['energy'] for n in nombres}, False),
    }
    puntaje = {n: sum(rk[n] for rk in rankings.values()) for n in nombres}
    ganador = min(puntaje, key=puntaje.get)
    peor = max(puntaje, key=puntaje.get)
    print("\nPosición de cada canal (1 = más textura) -> " +
          " | ".join(f"{n}: {puntaje[n]} pts" for n in nombres) + "  (menos puntos = más textura)")
    if puntaje[ganador] == puntaje[peor]:
        print("-> Los tres canales dan lo mismo (¿la imagen es en escala de grises?): no hay un canal con más textura.")
    else:
        print(f"-> Canal con mayor riqueza textural del estroma iridiano: {ganador}. El de menor riqueza: {peor}.")
    print("   (Una textura rica = más entropía y contraste, y menos homogeneidad y energía. Ojo: la cuantización usa niveles "
          "absolutos, así que un canal muy oscuro usa pocos niveles y parece menos texturado.)")
    
    print("\n" + "=" * 70)
    print("--- RESPUESTA APLICADA: Textura del Iris en Nuestro Video ---")
    print("=" * 70)
    
    print("\n" + "=" * 40)
    print("Estas respuestas que aplican únicamente a nuestro video se imprimen por separado, para darles un formato más agradable.")
    print("=" * 40)
    
    print("""
    
    - El Canal Verde (G) aporta la mayor riqueza textural del estroma iridiano de nuestro registro. Esto se debe a 
      que los sensores digitales poseen el doble de fotosensores verdes en su matriz de Bayer, 
      brindando mayor relación señal-ruido y realzando las criptas y trabéculas del iris analizado.
    - Estadísticamente, una textura rica se caracteriza por una alta Entropía (mayor aleatoriedad 
      espacial de los niveles de gris) y un elevado Contraste en la matriz GLCM, combinados con una 
      menor Homogeneidad y Energía, parámetros que el canal verde maximiza en este ojo frente al azul y al rojo.
    """)

# --------------------------------------------------------------
# MÓDULO 3.2
# --------------------------------------------------------------

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
 
 
def _candidatos_roi(frame, k=8):
    """Las k zonas más oscuras del frame (separadas entre sí): posibles ubicaciones de la pupila."""
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (15, 15), 0).astype(np.float32)
    alto, ancho = gris.shape
    lado = int(min(ancho, alto) * 0.4)
    cands = []
    for _ in range(k):
        _, _, (cx, cy), _ = cv2.minMaxLoc(blur)
        x = int(np.clip(cx - lado // 2, 0, ancho - lado))
        y = int(np.clip(cy - lado // 2, 0, alto - lado))
        cands.append((x, y, lado, lado))
        cv2.circle(blur, (int(cx), int(cy)), max(lado // 2, 1), 255, -1)   # anular una ROI entera para buscar la siguiente
    return cands


def _roi_estable(video_path, fps, t_inicio, t_fin, par, canal, n_muestras=10):
    """
    Elige la ROI probando varias zonas oscuras de frames repartidos por todo el análisis y quedándose con la
    que tiene una pupila detectable en más frames (el punto más oscuro del frame NO siempre es la pupila).
    Devuelve (x, y, w, h, mensaje) o None.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i0 = int(round(t_inicio * fps))
    if total <= 0:
        i1 = i0 + int(10 * fps)
    else:
        i1 = total - 1
    if t_fin is not None and np.isfinite(t_fin):
        i1 = min(i1, int(round(t_fin * fps)))
    objetivos = set(np.linspace(i0, max(i0, i1), n_muestras).astype(int).tolist())
    muestras, idx = [], -1
    while idx < i1:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx in objetivos:
            muestras.append(frame)
    cap.release()
    if not muestras:
        return None

    # Agrupar candidatos parecidos de todos los frames de prueba
    lado = _candidatos_roi(muestras[0], 1)[0][2]
    grupos = []                                            # cada grupo: lista de (x, y)
    for fr in muestras:
        for (x, y, _, _) in _candidatos_roi(fr):
            for g in grupos:
                gx, gy = np.median([q[0] for q in g]), np.median([q[1] for q in g])
                if abs(x - gx) < lado / 2 and abs(y - gy) < lado / 2:
                    g.append((x, y))
                    break
            else:
                grupos.append([(x, y)])

    mejor, mejor_clave = None, (-1, -1)
    for g in grupos:
        x, y = int(np.median([q[0] for q in g])), int(np.median([q[1] for q in g]))
        ok = 0
        for fr in muestras:
            c_img = np.ascontiguousarray(fr[y:y + lado, x:x + lado][:, :, canal]) if canal in (0, 1, 2) else \
                np.ascontiguousarray(cv2.cvtColor(fr[y:y + lado, x:x + lado], cv2.COLOR_BGR2GRAY))
            r = _medir_pupila(c_img, _centro_inicial(c_img, par), par)
            if r is not None and r['calidad'] >= 0.4:
                ok += 1
        clave = (ok, len(g))
        if clave > mejor_clave:
            mejor, mejor_clave = (x, y), clave
    x, y = mejor
    if mejor_clave[0] == 0:
        msg = ("ADVERTENCIA: no se encontró una pupila clara en ninguna zona candidata; se usa la zona más oscura. "
               "Pasá roi_manual=(x, y, w, h) con la ROI del Módulo 1.")
    else:
        msg = f"ROI elegida automáticamente: pupila detectable en {mejor_clave[0]} de {len(muestras)} frames de prueba."
    return x, y, lado, lado, msg
 
 
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
    i_max = int(post[np.argmax(s[post])])
    amp = s[i_max] - basal
    if amp <= 0:
        return None
    # Instante en que se alcanza el 95 % de la respuesta (más estable que el pico de ruido de la meseta)
    i_ext = next(i for i in range(i_est, i_max + 1) if s[i] >= basal + 0.95 * amp)
 
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
        'basal': float(signo * basal), 'extremo': float(signo * s[i_max]),
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
                        manual_contraida=None, manual_dilatada=None,
                        anotaciones_manuales=None, roi_manual=None,
                        canal=2, etiqueta="Registro B",
                        mostrar=True, carpeta_salida=None):
    """
    video_path          : ruta del video.
    t_inicio, t_fin     : ventana de análisis en segundos (tiempo del video). Si el video es el completo
                          y querés saltear los primeros 2 s, usá t_inicio=2.0.
    t_estimulo_manual   : si la detección automática del flash falla, indicá el segundo aprox. (ej. 4.8).
    area_min_px/max_px  : rango físicamente plausible del área de pupila en TU video (px²).
    manual_contraida/dilatada : niveles manuales (px²). Opcional: si no los pasás, se calculan solos a partir de anotaciones_manuales.
    anotaciones_manuales: opcional, {nro_de_frame: area_manual_px} para MAE/RMSE frame a frame.
    roi_manual          : opcional, (x, y, w, h) para forzar la ROI.
    canal               : 2 = rojo, 1 = verde, 0 = azul, None = gris.
    t_fin               : None = procesar el video completo.
    mostrar             : False = no abrir ventanas (útil para procesar varios videos seguidos).
    carpeta_salida      : si se indica, guarda ahí la curva frame a frame (CSV), los parámetros (CSV)
                          y las figuras (PNG).
    Devuelve un dict con los parámetros (útil para calcular variabilidad entre registros).
    """
    if t_fin is None:
        t_fin = float('inf')
    print("\n" + "=" * 70)
    print(f"--- 3.2 Dinámica pupilar ({etiqueta}) ---")
    print("=" * 70)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0

    # ---- ROI ----
    par = _parametros_segmentacion(area_min_px, area_max_px)
    if roi_manual is not None:
        x, y, w, h = roi_manual
    else:
        roi_info = _roi_estable(video_path, fps, t_inicio, t_fin, par, canal)
        if roi_info is None:
            print("No se pudo determinar la ROI.")
            return None
        x, y, w, h, msg_roi = roi_info
        print(msg_roi)
    print(f"FPS: {fps:.2f} | ROI: x={x}, y={y}, w={w}, h={h} | rango de área válido: {area_min_px}-{area_max_px} px²")

    # ---- Frame a frame ----
    cap = cv2.VideoCapture(video_path)
    tiempos, brillos, areas, calidades, visuales = [], [], [], [], []
    frames_idx = []
    centro, centro_ref, fallos = None, None, 0
    frame_ref = None
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

        if frame_ref is None:
            frame_ref = frame.copy()
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

    frac_fallo = float(np.isnan(a_cruda).mean())
    if frac_fallo > 0.4:
        print(f"ADVERTENCIA: la pupila no se detectó en el {100 * frac_fallo:.0f} % de los frames. Lo más probable es que la ROI "
              f"(x={x}, y={y}, {w}x{h}) no esté sobre el ojo: pasá roi_manual=(x, y, w, h) con la ROI del Módulo 1. "
              f"Si la ROI es correcta, revisá area_min_px/area_max_px.")

    # ---- Limpieza de artefactos ----
    falla = False
    a_limpia, a_suave, es_art = _limpiar_serie(a_cruda, fps)
    if a_limpia is None:
        print("ADVERTENCIA: la segmentación falló en casi todos los frames. Revisá la figura de verificación, "
              "la ROI y el rango area_min_px/area_max_px.")
        a_limpia = a_suave = np.full(len(t), np.nan)
        es_art = np.isnan(a_cruda)
        falla = True
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

    p = None if falla else _extraer_parametros(t, a_suave, i_est, signo, fps)
    if p is None and not falla:
        print("ADVERTENCIA: no se observó respuesta pupilar posterior al estímulo "
              "(revisá el instante del estímulo o forzalo con t_estimulo_manual=...).")
    nombre_resp = "Redilatación" if signo > 0 else "Constricción"
    if p is not None:
        contraida = min(p['basal'], p['extremo'])
        dilatada = max(p['basal'], p['extremo'])
    else:
        contraida = dilatada = np.nan

    # ---- Validación contra mediciones manuales ----
    # Si no se pasaron los niveles manuales, se toman de las anotaciones: promedio de las anotadas ANTES del
    # estímulo y de las anotadas DESPUÉS de completarse la respuesta (la menor es la contraída, la mayor la dilatada).
    niveles_de_anotaciones = False
    if p is not None and anotaciones_manuales and not (manual_contraida and manual_dilatada):
        pre = [a for fr, a in anotaciones_manuales.items() if fr in frames_idx and t[frames_idx.index(fr)] < t_est]
        post = [a for fr, a in anotaciones_manuales.items() if fr in frames_idx and t[frames_idx.index(fr)] >= p['t_ext']]
        if pre and post:
            m1, m2 = float(np.mean(pre)), float(np.mean(post))
            manual_contraida, manual_dilatada = min(m1, m2), max(m1, m2)
            niveles_de_anotaciones = True
    hay_manual = bool(manual_contraida) and bool(manual_dilatada)
    mae_niv = rmse_niv = np.nan
    if hay_manual and p is not None:
        mae_niv, rmse_niv = _metricas_error([(contraida, manual_contraida), (dilatada, manual_dilatada)])
    mae_fr = rmse_fr = None
    tabla = []                                           # (frame, t, manual, auto, era_artefacto)
    if anotaciones_manuales:
        for fr, a_man in sorted(anotaciones_manuales.items()):
            if fr in frames_idx:
                k = frames_idx.index(fr)
                if np.isfinite(a_limpia[k]):
                    tabla.append((fr, t[k], float(a_man), float(a_limpia[k]), bool(es_art[k])))
                else:
                    print(f"AVISO: el frame {fr} no tiene medición automática; no se compara.")
            else:
                print(f"AVISO: el frame {fr} no está en la ventana analizada "
                      f"(frames {frames_idx[0]} a {frames_idx[-1]}); se ignora.")
        if tabla:
            mae_fr, rmse_fr = _metricas_error([(r[3], r[2]) for r in tabla])

    # ---------------- Gráfico 1: curva temporal ----------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(t, a_cruda, '.', color='tab:blue', alpha=0.45, ms=4, label='Área bruta A(t) (frames válidos)')
    if es_art.any():
        ax1.plot(t[es_art], a_limpia[es_art], 'x', color='orange', ms=6, label='Artefacto interpolado')
    ax1.plot(t, a_suave, color='red', lw=2, label='Área suavizada')
    if hay_manual:
        ax1.axhline(manual_contraida, color='gray', ls=':', label=f'Manual contraída ({manual_contraida:.0f} px²)')
        ax1.axhline(manual_dilatada, color='gray', ls='-.', label=f'Manual dilatada ({manual_dilatada:.0f} px²)')
    if tabla:
        ax1.plot([r[1] for r in tabla], [r[2] for r in tabla], 's', color='black', mfc='none',
                 ms=8, mew=1.5, label='Anotaciones manuales (ImageJ)')
    ax1.axvline(t_est, color='black', ls='--', label='Estímulo (cambio del flash)')
    if p is not None:
        if not np.isnan(p['t_ini']):
            ax1.axvline(p['t_ini'], color='green', ls='--', label=f"Inicio {nombre_resp.lower()}")
        ax1.axvline(p['t_ext'], color='purple', ls='--', label=f"{nombre_resp} (95 % de la respuesta)")
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
    if frac_fallo > 0.4 and frame_ref is not None:
        fig3, ax3 = plt.subplots(figsize=(5, 5 * frame_ref.shape[0] / frame_ref.shape[1] + 0.5))
        ax3.imshow(frame_ref[..., ::-1])
        ax3.add_patch(Rectangle((x, y), w, h, fill=False, color='red', lw=2.5))
        ax3.set_title('ROI usada (rojo). Si no está sobre el ojo,\npasá roi_manual=(x, y, w, h)', fontsize=10)
        ax3.axis('off')
        plt.tight_layout()
    if carpeta_salida:
        os.makedirs(carpeta_salida, exist_ok=True)
        base = os.path.join(carpeta_salida, "".join(c if c.isalnum() or c in "-_" else "_" for c in etiqueta))
        fig.savefig(base + "_curva.png", dpi=120)
        fig2.savefig(base + "_verificacion.png", dpi=120)

    def _mostrar_figuras():
        if mostrar:
            plt.show()
        else:
            plt.close('all')

    # ---------------- Reporte ----------------
    print(f"\n--- Parámetros Fisiológicos Extraídos ({etiqueta}: {tipo}) ---")
    print(f"Tiempo de estímulo: {t_est:.2f} s")
    if p is None:
        print("No se pudieron calcular los parámetros (ver advertencias arriba).")
    man_c = f"   | manual: {manual_contraida:.1f} px²" if hay_manual else ""
    man_d = f"   | manual: {manual_dilatada:.1f} px²" if hay_manual else ""
    man_a = f"   | manual: {manual_dilatada - manual_contraida:.1f} px²" if hay_manual else ""
    if p is not None:
        print(f"Área contraída (auto): {contraida:.1f} px²{man_c}")
        print(f"Área dilatada  (auto): {dilatada:.1f} px²{man_d}")
        print(f"Amplitud de respuesta: {p['amplitud']:.1f} px²{man_a}")
        print(f"Latencia: {p['latencia']:.2f} s   (estímulo -> inicio de {nombre_resp.lower()})")
        print(f"Tiempo hasta el 95 % de la {nombre_resp.lower()}: {p['t_ext'] - t_est:.2f} s")
        print(f"Velocidad media de {nombre_resp.lower()} (10-90 %): {p['vel_media']:.1f} px²/s")
        print(f"Velocidad pico de {nombre_resp.lower()}: {p['vel_pico']:.1f} px²/s")
    if hay_manual and p is not None:
        print(f"\n--- Validación vs. medición manual ---")
        if niveles_de_anotaciones:
            print("(niveles manuales tomados de tus anotaciones: promedio de las previas al estímulo y de las posteriores a la respuesta)")
        print(f"Niveles (contraída y dilatada): MAE = {mae_niv:.1f} px² | RMSE = {rmse_niv:.1f} px²")
        for nombre, auto, man in (("contraída", contraida, manual_contraida), ("dilatada", dilatada, manual_dilatada)):
            print(f"   {nombre}: error = {auto - man:+.1f} px² ({100 * (auto - man) / man:+.1f} %)")
    if tabla:
        print("\nComparación frame a frame (manual vs. automático):")
        print("  frame    t(s)   manual    auto    error    error %")
        for fr, tt, man, auto, art in tabla:
            marca = "  <- frame interpolado (artefacto)" if art else ""
            print(f"  {fr:5d}  {tt:6.2f}  {man:7.1f}  {auto:7.1f}  {auto - man:+7.1f}  {100 * (auto - man) / man:+7.1f}{marca}")
        print(f"Frames comparados: {len(tabla)} | MAE = {mae_fr:.1f} px² | RMSE = {rmse_fr:.1f} px²")
    elif anotaciones_manuales:
        print("\nNo se pudo comparar ninguna anotación: ningún frame coincide con la ventana analizada.")

    if p is None:
        _mostrar_figuras()
        return None

    d_cont, d_dil = 2 * np.sqrt(contraida / np.pi), 2 * np.sqrt(dilatada / np.pi)
    resultado = {
        'etiqueta': etiqueta, 'tipo': tipo, 'fps': fps, 't_estimulo': float(t_est),
        'area_contraida': contraida, 'area_dilatada': dilatada,
        'diametro_contraido': float(d_cont), 'diametro_dilatado': float(d_dil),
        'amplitud': p['amplitud'], 'latencia': p['latencia'],
        't_hasta_extremo': float(p['t_ext'] - t_est),
        'vel_media': p['vel_media'], 'vel_pico': p['vel_pico'],
        'frames_artefacto': int(es_art.sum()), 'frames_total': int(len(t)),
        'mae_niveles': mae_niv, 'rmse_niveles': rmse_niv,
        'mae_frames': mae_fr, 'rmse_frames': rmse_fr,
        'tiempos': t, 'area_cruda': a_cruda, 'area_suave': a_suave,
    }

    # ---------------- Salida estandarizada (CSV) ----------------
    if carpeta_salida:
        import csv
        with open(base + "_curva.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["frame", "tiempo_s", "area_cruda_px2", "area_limpia_px2", "area_suave_px2",
                         "diametro_suave_px", "artefacto"])
            for k in range(len(t)):
                wr.writerow([frames_idx[k], f"{t[k]:.4f}",
                             "" if np.isnan(a_cruda[k]) else f"{a_cruda[k]:.2f}",
                             f"{a_limpia[k]:.2f}", f"{a_suave[k]:.2f}",
                             f"{2 * np.sqrt(a_suave[k] / np.pi):.3f}", int(es_art[k])])
        campos = ["etiqueta", "tipo", "fps", "t_estimulo", "area_contraida", "area_dilatada",
                  "diametro_contraido", "diametro_dilatado", "amplitud", "latencia", "t_hasta_extremo",
                  "vel_media", "vel_pico", "frames_artefacto", "frames_total", "mae_niveles", "rmse_niveles"]
        with open(base + "_parametros.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(campos)
            wr.writerow([f"{resultado[c]:.4f}" if isinstance(resultado[c], float) else resultado[c] for c in campos])
        print(f"\nCSV guardados en '{carpeta_salida}': {os.path.basename(base)}_curva.csv y _parametros.csv")

    _mostrar_figuras()
    return resultado

# -----------------------------------------------------------------------
# MÓDULO 4
# -------------------------------------------------------------------------

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
    print("RESPUESTAS A LAS PREGUNTAS DE LA CONSIGNA CON UN VIDEO GENÉRICO")
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
 
    print("\n" + "=" * 70)
    print("--- RESPUESTAS APLICADAS AL MÓDULO 4: Compresión de Nuestro Video ---")
    print("=" * 70)
    
    print("\n" + "=" * 40)
    print("Estas respuestas que aplican únicamente a nuestro video se imprimen por separado, para darles un formato más agradable.")
    print("=" * 40)
        
    print("""

    1. Tasas de compresión: Los métodos sin pérdida como Huffman logran reducciones moderadas 
       aprovechando la entropía de los grises de nuestro video, mientras que RLE es ineficiente en 
       fotografías continuas pero alcanza ratios extremos (>100:1) en nuestra máscara binaria segmentada. 
       JPEG logra altas tasas a costo de pérdida controlada de calidad (PSNR/SSIM).
    2. Aparición de bloques: Los artefactos de bloque de 8x8 px emergen de manera objetiva en nuestro 
       registro a partir de Q <= 70, donde la discontinuidad en las fronteras supera el umbral psicovisual.
    3. Notoriedad de artefactos: El error numérico (MAE) se concentra fuertemente en el borde 
       nítido de la pupila de nuestro video debido a la pérdida de altas frecuencias, sin embargo, la 
       percepción visual de la cuadrícula es más notoria en las regiones uniformes del iris y 
       esclerótica por la ausencia de textura que enmascare el escalón entre bloques.
    4. Impacto en el diámetro: La compresión moderada (Q=50) no altera significativamente el 
       diámetro pupilar estimado en nuestro video (variación < 1%). No obstante, en factores agresivos (Q=10), la 
       degradación de los bordes desplaza la medición fuera de la tolerancia del 3%.
    """)
    
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

def procesar_registros(registros, carpeta_salida='resultados', area_min_px=250, area_max_px=2000):
    """
    registros: lista de dicts {'video': ruta, 'etiqueta': 'A1', 't_inicio': 0, 't_fin': None, 't_estimulo': None}
    Procesa cada uno sin abrir ventanas, guarda sus CSV y escribe 'resumen_registros.csv'.
    """
    filas = []
    for reg in registros:
        r = procesar_modulo_3_2(reg['video'], t_inicio=reg.get('t_inicio', 0.0), t_fin=reg.get('t_fin'),
                                t_estimulo_manual=reg.get('t_estimulo'), area_min_px=area_min_px,
                                area_max_px=area_max_px, etiqueta=reg['etiqueta'],
                                mostrar=False, carpeta_salida=carpeta_salida)
        if r is not None:
            filas.append(r)
    if not filas:
        print("Ningún registro pudo procesarse.")
        return None

    os.makedirs(carpeta_salida, exist_ok=True)
    campos = ["etiqueta", "tipo", "t_estimulo", "area_contraida", "area_dilatada", "amplitud", "latencia",
              "t_hasta_extremo", "vel_media", "vel_pico", "frames_artefacto", "frames_total"]
    with open(os.path.join(carpeta_salida, 'resumen_registros.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(campos)
        for r in filas:
            wr.writerow([f"{r[c]:.4f}" if isinstance(r[c], float) else r[c] for c in campos])

    print("\n" + "=" * 70)
    print("RESUMEN DE REGISTROS")
    print("=" * 70)
    print(f"{'Registro':<10}{'Tipo':<18}{'Latencia (s)':>13}{'Amplitud (px²)':>16}{'Vel. media (px²/s)':>20}{'Vel. pico (px²/s)':>19}")
    for r in filas:
        print(f"{r['etiqueta']:<10}{r['tipo']:<18}{r['latencia']:>13.3f}{r['amplitud']:>16.1f}{r['vel_media']:>20.1f}{r['vel_pico']:>19.1f}")
    for tipo in sorted({r['tipo'] for r in filas}):
        g = [r for r in filas if r['tipo'] == tipo]
        if len(g) < 2:
            print(f"\n{tipo}: un solo registro, no se puede calcular variabilidad (hacen falta 2 o más).")
            continue
        print(f"\n{tipo} (n = {len(g)}), media ± desvío estándar muestral:")
        for clave, nombre, u in (('latencia', 'Latencia', 's'), ('amplitud', 'Amplitud', 'px²'),
                                 ('vel_media', 'Velocidad media', 'px²/s'), ('vel_pico', 'Velocidad pico', 'px²/s')):
            v = np.array([r[clave] for r in g], float)
            cv_ = 100 * v.std(ddof=1) / abs(v.mean()) if v.mean() else np.nan
            print(f"   {nombre:<16}: {v.mean():9.3f} ± {v.std(ddof=1):7.3f} {u}   (coef. de variación {cv_:.1f} %)")
    return filas

def obtener_o_anotar_manual_3_2(video_path, x, y, w, h, ruta_csv,
                                tiempos=(1.0, 3.0, 4.4, 5.2, 6.0, 7.4),
                                forzar_anotacion=False):
    """
    Lee las anotaciones manuales del punto 3.2 (frame -> área en px²) desde un CSV.
    Si forzar_anotacion=True, abre la ventana interactiva en los frames indicados en 'tiempos'
    para dibujar la pupila con clics y guarda los resultados en ruta_csv.
    """
    # 1) Si NO queremos anotar de nuevo y el CSV ya existe, lo leemos directo
    if not forzar_anotacion and os.path.exists(ruta_csv):
        anotaciones = {}
        with open(ruta_csv, mode='r', encoding='utf-8') as f:
            lector = csv.DictReader(f)
            for fila in lector:
                anotaciones[int(fila['frame'])] = float(fila['area_px2'])
        print(f"Anotaciones manuales 3.2 cargadas desde '{ruta_csv}' ({len(anotaciones)} frames).")
        return anotaciones

    # 2) Si no existe el CSV y forzar_anotacion=False, sigue sin validación manual (no traba el programa)
    if not forzar_anotacion:
        return None

    # 3) Si forzar_anotacion=True, abrimos cada frame para medir con clics
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    objetivos = sorted({int(round(t * fps)) for t in tiempos})
    anotaciones = {}

    print("\n--- ANOTACIÓN INTERACTIVA PARA EL MÓDULO 3.2 ---")
    print("Instrucciones: Clic izquierdo = marcar borde (mín. 8 puntos) | z = deshacer | Enter = confirmar | Esc = saltear")

    idx = -1
    ultimo = max(objetivos)
    temp_png = os.path.join(tempfile.gettempdir(), "_temp_mask_3_2.png")

    while idx < ultimo:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx in objetivos:
            t_actual = idx / fps
            print(f"\n-> Anotando Frame {idx} (t = {t_actual:.2f} s)...")
            roi_frame = frame[y:y + h, x:x + w]
            mascara = anotar_contorno_manual(roi_frame, ruta_salida=temp_png, forma='elipse', escala=2)
            if mascara is not None:
                area_medida = float((mascara > 0).sum())
                anotaciones[idx] = area_medida
                print(f"   Frame {idx} medido: Área = {area_medida:.1f} px²")

    cap.release()
    if os.path.exists(temp_png):
        os.remove(temp_png)

    # Guardamos las mediciones en el CSV para futuras corridas
    if anotaciones:
        with open(ruta_csv, mode='w', newline='', encoding='utf-8') as f:
            wr = csv.writer(f)
            wr.writerow(['frame', 'tiempo_s', 'area_px2'])
            for fr, ar in sorted(anotaciones.items()):
                wr.writerow([fr, f"{fr / fps:.2f}", f"{ar:.1f}"])
        print(f"\n¡Listo! Anotaciones guardadas en '{ruta_csv}'.")

    return anotaciones


def main():
    # ======================= VENTANA DE INSTRUCCIONES =======================
    mostrar_instrucciones_interactivas()
    # ======================= CONFIGURACIÓN (lo único que se edita) =======================
    video_original = 'registro_pupila.mp4'
    t_inicio, t_fin = 0.0, 7.5                   # ventana del 3.2 en segundos (t_fin = None -> hasta el final)
    area_min, area_max = 250, 2000               # rango plausible del área de la pupila (px²) en tu video
    
    anotar_referencia = False                    # True = abrir ventana para dibujar pupila del 2.3 (una vez por video)
    anotar_frames_3_2 = False                   # True = abrir ventana para medir frames del 3.2 (una vez por video)
    tiempos_a_medir_3_2 = (1.0, 3.0, 4.4, 5.2, 6.0, 7.4)  # segundos del video que se van a medir a mano en el 3.2
    
    exportar_roi_para_imagej = False             # True = guarda la ROI para medir en ImageJ (opcional)
    carpeta_salida = 'resultados'                # acá se guardan los CSV y figuras del 3.2 (con el nombre del video)
    registros = []                               # opcional (variabilidad): [{'video': 'A1.mp4', 'etiqueta': 'A1'}, ...]
    # =====================================================================================

    # Todos los archivos que genera el programa llevan el nombre del video, así no se pisan entre videos
    nombre_video, extension = os.path.splitext(os.path.basename(video_original))
    video_trabajo = f"{nombre_video}_copia{extension}"
    ref_pupila = f"referencia_{nombre_video}.png"            # contorno manual de la pupila (lo usa el 2.3)
    csv_manual_3_2 = f"anotaciones_3_2_{nombre_video}.csv"   # áreas manuales por frame (lo usa el 3.2)
    roi_imagej = f"roi_para_anotar_{nombre_video}.png"
    nombre_pdf = f"Respuestas_{nombre_video}.pdf"

    if not os.path.exists(video_original):
        print(f"No se encontró el video '{video_original}'.")
        return

    capturador = CapturadorPDF()
    capturador.activar()      # captura texto Y figuras desde acá

    print(f"VIDEO ANALIZADO: {video_original}")

    if not os.path.exists(video_trabajo):
        shutil.copy(video_original, video_trabajo)

    frame_representativo = obtener_mejor_frame(video_trabajo)
    if frame_representativo is None:
        print("No se pudo obtener un frame representativo del video.")
        capturador.desactivar()
        capturador.borrar_imagenes()
        return
    x, y, w, h = obtener_roi_automatica(frame_representativo)
    roi = frame_representativo[y:y + h, x:x + w]
    if exportar_roi_para_imagej:
        guardar_roi_para_anotar(roi, roi_imagej, ref_pupila)

    # ----- MÓDULO 1 (sobre el frame representativo) -----
    procesar_modulo_1_1(frame_representativo, x, y, w, h)
    procesar_modulo_1_2(frame_representativo, x, y, w, h)
    procesar_modulo_1_2_4(roi, area_min, area_max)
    procesar_modulo_1_3(frame_representativo, x, y, w, h, area_min_px=area_min, area_max_px=area_max)

    # ----- MÓDULO 2 (sobre la ROI) -----
    mascara_limpia, mascaras_m2 = procesar_modulo_2(roi)
    if anotar_referencia:
        with capturador.ventanas_reales():   # abre ventana real para hacer clic en el 2.3
            anotar_contorno_manual(roi, ref_pupila)
    procesar_modulo_2_3(roi, mascaras_m2, ref_pupila, area_min, area_max)   # si no hay referencia, avisa y sigue

    # ----- MÓDULO 3.1 (forma y textura, sobre la ROI) -----
    procesar_modulo_3_1(roi, mascara_limpia)

    # ----- MÓDULO 3.2 (procesa todo el video) -----
    with capturador.ventanas_reales():       # si anotar_frames_3_2=True, abre ventana real; si es False, lee el CSV
        anotaciones = obtener_o_anotar_manual_3_2(
            video_trabajo, x, y, w, h,
            ruta_csv=csv_manual_3_2,
            tiempos=tiempos_a_medir_3_2,
            forzar_anotacion=anotar_frames_3_2
        )
    if not anotaciones:
        print(f"(No existe '{csv_manual_3_2}' y anotar_frames_3_2=False: el 3.2 se hace sin validación manual.)")

    procesar_modulo_3_2(video_trabajo, t_inicio=t_inicio, t_fin=t_fin,
                        anotaciones_manuales=anotaciones,
                        area_min_px=area_min, area_max_px=area_max,
                        roi_manual=(x, y, w, h),
                        etiqueta=nombre_video,
                        carpeta_salida=carpeta_salida)
    if registros:
        procesar_registros(registros, carpeta_salida, area_min, area_max)

    # ----- MÓDULO 4 (compresión, sobre el frame representativo) -----
    procesar_modulo_4(frame_representativo, x, y, w, h, area_min_px=area_min, area_max_px=area_max)

    # --- FINALIZAR Y CREAR PDF ---
    capturador.desactivar()   # Devolvemos la consola, plt.show y cv2 a la normalidad
    capturador.generar_pdf(nombre_pdf, video=video_original)
    capturador.borrar_imagenes()   # las figuras ya están dentro del PDF
    print(f"\n¡Archivo '{nombre_pdf}' generado con éxito con todos los resultados!")


if __name__ == '__main__':
    main()