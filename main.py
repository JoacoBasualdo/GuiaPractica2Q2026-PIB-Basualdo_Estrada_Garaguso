# Leer el README antes de ejecutar. Las instrucciones también aparecen en una ventana al iniciar.
#
# Consultas pendientes:
#   - Qué hacer en el análisis de pérdida de definición.
#   - Qué se espera en el 1.2.4 (ruido y filtrado espacial).
#   - Qué se espera observar al aplicar la TFD-2D.

import contextlib
import csv
import heapq
import os
import re
import shutil
import sys
import tempfile
import tkinter as tk
from datetime import datetime
from tkinter import messagebox

import cv2
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from matplotlib.patches import Ellipse, Rectangle
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.stats import entropy, skew
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import skeletonize
from skimage.segmentation import active_contour

def mostrar_instrucciones_interactivas():
    """Ventana inicial con los cambios que se hacen en main() al trabajar con un video nuevo."""
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "Guía de Configuración - Procesamiento de Pupila",
        "Los siguientes cambios se tienen que hacer en 'main()' al operar con un nuevo video:\n\n"
        "📌 1. CAMBIAR EL VIDEO A ANALIZAR:\n"
        "   - Buscá en 'main()' la línea: video_original = 'registro_pupila.mp4'\n"
        "   - Reemplazá 'registro_pupila.mp4' por el nombre del nuevo archivo de video (ej. 'video_prueba2.mp4').\n\n"
        "📌 2. CONFIGURAR ANOTACIONES MANUALES (True / False):\n"
        "   - anotar_referencia = True / False (también en main())\n"
        "     (Ponelo en True la PRIMERA VEZ que analices un video nuevo para dibujar la máscara del Módulo 2.3).\n"
        "   - anotar_frames_3_2 = True / False (también en main())\n "
        "     (Ponelo en True la PRIMERA VEZ que se trabaja con el nuevo video para medir con clics los frames del Módulo 3.2).\n\n"
        "📌 3. SI LA ROI AUTOMÁTICA NO CAE SOBRE EL OJO:\n"
        "   - Indicá roi_forzada = (x, y, w, h) en main().\n\n"
        "ℹ️ Una vez creados los archivos .png y .csv de un video, volvé a poner ambas opciones en FALSE "
        "para que el programa corra 100% automatizado (ideal para la evaluación de la cátedra).\n"
        "El pdf generado cuenta con respuestas variables según video, y con respuestas que solo aplican al video original del grupo")
    root.destroy()

class CapturadorPDF:
    """Registra en orden lo que se imprime y las figuras (plt.show y cv2.imshow + waitKey(0)) para armar un PDF
    con texto e imágenes intercalados. Las imágenes viven en una carpeta temporal que borrar_imagenes() elimina."""

    def __init__(self, dpi=120, mostrar_ventanas=False):
        self.consola_original = sys.stdout
        self.mostrar_ventanas = mostrar_ventanas      # False: las figuras solo se guardan, no se abren
        self.dpi = dpi
        self.contenido = []                           # ('texto', str) o ('imagen', ruta, titulo)
        self.carpeta_imagenes = None
        self._n_img = 0
        self._cv2_pendientes = {}
        self._originales = None

    def write(self, mensaje):
        self.consola_original.write(mensaje)
        self.contenido.append(('texto', mensaje))

    def flush(self):
        self.consola_original.flush()

    def _guardar(self, imagen=None, fig=None, titulo=''):
        self._n_img += 1
        ruta = os.path.join(self.carpeta_imagenes, f"figura_{self._n_img:03d}.png")
        if fig is not None:
            fig.savefig(ruta, dpi=self.dpi, bbox_inches='tight', pil_kwargs={'compress_level': 1})
            fig._ya_en_pdf = True
        else:
            cv2.imwrite(ruta, imagen)
        self.contenido.append(('imagen', ruta, titulo))

    def _show(self, *args, **kwargs):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            if not getattr(fig, '_ya_en_pdf', False):
                self._guardar(fig=fig)
        if self.mostrar_ventanas:
            return self._originales['show'](*args, **kwargs)
        plt.close('all')

    def _imshow(self, nombre, imagen):
        self._cv2_pendientes[nombre] = np.array(imagen, copy=True)
        if self.mostrar_ventanas:
            return self._originales['imshow'](nombre, imagen)

    def _waitkey(self, delay=0):
        # Solo se guardan las ventanas que esperan tecla: así no entran los cuadros de las ventanas interactivas.
        if delay <= 0:
            for nombre, img in self._cv2_pendientes.items():
                if img.dtype != np.uint8:
                    img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                self._guardar(img, titulo=nombre)
            self._cv2_pendientes.clear()
        if self.mostrar_ventanas:
            return self._originales['waitKey'](delay)
        return -1

    def _destroy_all(self):
        self._cv2_pendientes.clear()
        if self.mostrar_ventanas:
            return self._originales['destroyAllWindows']()

    @contextlib.contextmanager
    def ventanas_reales(self):
        """Dentro del bloque las ventanas sí se abren (herramientas de anotación)."""
        previo, self.mostrar_ventanas = self.mostrar_ventanas, True
        try:
            yield
        finally:
            self.mostrar_ventanas = previo

    def activar(self):
        self.carpeta_imagenes = tempfile.mkdtemp(prefix="figuras_pdf_")
        self._originales = {'show': plt.show, 'imshow': cv2.imshow,
                            'waitKey': cv2.waitKey, 'destroyAllWindows': cv2.destroyAllWindows}
        plt.show = lambda *a, **k: self._show(*a, **k)
        cv2.imshow = lambda nombre, imagen: self._imshow(nombre, imagen)
        cv2.waitKey = lambda delay=0: self._waitkey(delay)
        cv2.destroyAllWindows = lambda: self._destroy_all()
        sys.stdout = self

    def desactivar(self):
        sys.stdout = self.consola_original
        if self._originales:
            plt.show, cv2.imshow = self._originales['show'], self._originales['imshow']
            cv2.waitKey, cv2.destroyAllWindows = self._originales['waitKey'], self._originales['destroyAllWindows']
            self._originales = None

    def borrar_imagenes(self):
        if self.carpeta_imagenes and os.path.isdir(self.carpeta_imagenes):
            shutil.rmtree(self.carpeta_imagenes, ignore_errors=True)
        self.carpeta_imagenes = None

    @staticmethod
    def _separar_titulos(texto):
        """Divide el texto en bloques ('titulo' | 'subtitulo' | 'cuerpo'). Título: línea entre dos reglas de '=====';
        subtítulo: '--- texto ---', '[1] texto', '1.3.2 texto' o '>>> texto' al comienzo de la línea."""
        es_regla = lambda l: len(l.strip()) >= 10 and set(l.strip()) == {'='}
        sacar_guiones = lambda l: re.sub(r'\s*-{3,}.*$', '', re.sub(r'^\s*-{3,}\s*', '', l)).strip()
        patrones = [r'^-{3,}\s*(.+?)\s*-{3,}', r'^(\[\d+\]\s+\S.*)', r'^(\d+\.\d+(?:\.\d+)*\s+\S.*)', r'^>>>\s*(.+)']
        lineas, bloques, cuerpo, i = texto.split('\n'), [], [], 0

        def cerrar_cuerpo():
            if cuerpo:
                bloques.append(('cuerpo', '\n'.join(cuerpo)))
                cuerpo.clear()

        while i < len(lineas):
            l = lineas[i]
            if es_regla(l):
                if i + 2 < len(lineas) and lineas[i + 1].strip() and es_regla(lineas[i + 2]):
                    cerrar_cuerpo()
                    bloques.append(('titulo', sacar_guiones(lineas[i + 1])))
                    i += 3
                else:
                    i += 1
                continue
            sub = next((m.group(1) for p in patrones if (m := re.match(p, l.rstrip()))), None)
            if sub:
                cerrar_cuerpo()
                bloques.append(('subtitulo', sub.strip()))
            else:
                cuerpo.append(l)
            i += 1
        cerrar_cuerpo()
        return bloques

    @staticmethod
    def _limpiar(texto):
        """Reemplaza los símbolos que la fuente base del PDF no soporta."""
        for a, b in (('²', '^2'), ('±', '+/-'), ('Á', 'A'), ('É', 'E'), ('Í', 'I'), ('Ó', 'O'), ('Ú', 'U'), ('Ñ', 'N'),
                     ('á', 'a'), ('é', 'e'), ('í', 'i'), ('ó', 'o'), ('ú', 'u'), ('ñ', 'n')):
            texto = texto.replace(a, b)
        return texto.encode('latin-1', 'ignore').decode('latin-1')

    @staticmethod
    def _dimensiones_png(ruta):
        """(alto, ancho) leyendo solo la cabecera si es PNG; (0, 0) si no se puede leer."""
        with open(ruta, 'rb') as f:
            cab = f.read(24)
        if cab[:8] == b'\x89PNG\r\n\x1a\n':
            return int.from_bytes(cab[20:24], 'big'), int.from_bytes(cab[16:20], 'big')
        img = cv2.imread(ruta)
        return img.shape[:2] if img is not None else (0, 0)

    def generar_pdf(self, nombre_archivo="Respuestas.pdf", video=None, titulo="Respuestas"):
        nombre_video = os.path.basename(video) if video else ""
        fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
        limpiar = self._limpiar

        class PDFConEncabezado(FPDF):
            """Encabezado y pie en todas las páginas salvo la portada."""
            def header(self):
                if self.page_no() == 1:
                    return
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(110)
                self.set_xy(self.l_margin, 10)
                self.cell(0, 5, limpiar(f"{titulo} - {nombre_video}" if nombre_video else titulo), align="L")
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
                self.cell(0, 10, f"Página {self.page_no()} de {{nb}}", align="C")
                self.set_text_color(0)

        COLOR_TITULO, COLOR_SUBTITULO = (25, 60, 120), (60, 60, 60)
        pdf = PDFConEncabezado()
        ancho_util = pdf.w - pdf.l_margin - pdf.r_margin
        alto_util = pdf.h - 20 - pdf.b_margin

        pdf.add_page()                                           # portada
        pdf.set_y(95)
        pdf.set_font("Helvetica", "B", 28)
        pdf.cell(0, 14, limpiar(titulo), align="C")
        pdf.ln(20)
        pdf.set_draw_color(120)
        pdf.line(pdf.w / 2 - 40, pdf.get_y(), pdf.w / 2 + 40, pdf.get_y())
        pdf.ln(10)
        if nombre_video:
            pdf.set_font("Helvetica", "", 14)
            pdf.cell(0, 8, limpiar(f"Video analizado: {nombre_video}"), align="C")
            pdf.ln(10)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(90)
        pdf.cell(0, 6, f"Generado el {fecha}", align="C")
        pdf.set_text_color(0)
        pdf.add_page()

        bloques = []                                             # textos consecutivos en un solo bloque
        for item in self.contenido:
            if item[0] == 'texto' and bloques and bloques[-1][0] == 'texto':
                bloques[-1] = ('texto', bloques[-1][1] + item[1])
            else:
                bloques.append(item)

        for b in bloques:
            if b[0] == 'texto':
                for tipo, contenido in self._separar_titulos(limpiar(b[1])):
                    pdf.set_x(pdf.l_margin)
                    if tipo == 'titulo':
                        if pdf.get_y() > pdf.h - pdf.b_margin - 35:
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
                        pdf.set_font("Courier", size=9)          # monoespaciada para que las tablas no se deformen
                        pdf.multi_cell(0, 4, txt=texto)
                        pdf.ln(2)
                    pdf.set_text_color(0)
            else:
                _, ruta, titulo_img = b
                alto_px, ancho_px = self._dimensiones_png(ruta)
                if not alto_px:
                    continue
                w = ancho_util
                h = w * alto_px / ancho_px
                if h > 0.9 * alto_util:                          # figuras muy altas: se achican para que entren
                    h = 0.9 * alto_util
                    w = h * ancho_px / alto_px
                alto_titulo = 5 if titulo_img else 0
                if pdf.get_y() + alto_titulo + h > pdf.h - pdf.b_margin:
                    pdf.add_page()
                if titulo_img:
                    pdf.set_font("Courier", "B", 9)
                    pdf.set_x(pdf.l_margin)
                    pdf.cell(0, 5, txt=limpiar(titulo_img))
                    pdf.ln(5)
                y = pdf.get_y()
                pdf.image(ruta, x=pdf.l_margin + (ancho_util - w) / 2, y=y, w=w, h=h)
                pdf.set_y(y + h + 4)
        pdf.output(nombre_archivo)
# Respuestas que aplican solo al video original del grupo (texto fijo).
RESPUESTAS = {
    'definicion': """
    
    - Al reducir la resolución espacial del ojo analizado al 50%, 25% y 5% (y reescalarla con vecino más cercano), 
      se evidencia el fenómeno de aliasing espacial y la pérdida progresiva de altas frecuencias.
    - En el 50% los detalles finos del estroma iridiano de nuestro registro se atenúan levemente; en el 25% los bordes 
      comienzan a pixelarse; y en el 5% el contorno pupilar se vuelve completamente discontinuo e 
      impreciso, demostrando que un submuestreo excesivo destruye la geometría necesaria para la segmentación.
    """,
    'filtro': """
    
    - El filtro de Mediana (con kernel de 11x11) es el que preserva de manera idónea el borde de 
      la pupila en nuestro registro mientras elimina los destellos especulares del flash. Esto ocurre porque la mediana 
      es un operador no lineal que descarta valores extremos (como el brillo saturado del reflejo) 
      sin promediarlos con los vecinos, evitando el efecto de difuminado típico de los filtros lineales (Media/Gaussiano).
    - La calidad de la imagen filtrada frente a la original sin ruido se cuantifica mediante:
      * PSNR (Peak Signal-to-Noise Ratio): Relación entre la potencia máxima y el ruido de reconstrucción en dB.
      * SSIM (Structural Similarity Index): Similitud estructural percibida (rango de 0 a 1).
    """,
    'bordes': """
    
    - Operadores de bordes: Canny demuestra la mayor robustez y continuidad (100%) sobre nuestro video 
      al utilizar histéresis y doble umbral, minimizando falsos positivos tras el pre-filtrado gaussiano.
    - Reconstrucción por Transformada de Fourier 2D (TFD-2D): Al cruzar la magnitud de nuestra imagen 
      con la fase de otra, se comprueba empíricamente que la Fase concentra la información 
      estructural y geométrica de nuestro ojo (delimitando la posición de los contornos), mientras 
      que la Magnitud solo define la distribución energética global de intensidades y contrastes.
    """,
    'textura': """
    
    - El Canal Verde (G) aporta la mayor riqueza textural del estroma iridiano de nuestro registro. Esto se debe a 
      que los sensores digitales poseen el doble de fotosensores verdes en su matriz de Bayer, 
      brindando mayor relación señal-ruido y realzando las criptas y trabéculas del iris analizado.
    - Estadísticamente, una textura rica se caracteriza por una alta Entropía (mayor aleatoriedad 
      espacial de los niveles de gris) y un elevado Contraste en la matriz GLCM, combinados con una 
      menor Homogeneidad y Energía, parámetros que el canal verde maximiza en este ojo frente al azul y al rojo.
    """,
    'compresion': """

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
    """,
}

def _titulo(texto):
    """Encabezado de sección: texto entre dos reglas de '='."""
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)

def _respuesta_video(titulo, clave):
    """Imprime aparte las respuestas fijas que solo aplican al video original del grupo."""
    _titulo(titulo)
    print("\n" + "=" * 40)
    print("Estas respuestas que aplican únicamente a nuestro video se imprimen por separado, para darles un formato más agradable.")
    print("=" * 40)
    print(RESPUESTAS[clave])

def _banda_borde(mascara, ancho=5):
    """Banda booleana alrededor del contorno de una máscara (gradiente morfológico)."""
    return cv2.morphologyEx(mascara, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ancho, ancho))) > 0

def _mascara_pupila_elipse(roi, area_min_px=250, area_max_px=2000, canal=2):
    """Pupila segmentada con el método del Módulo 3.2. Devuelve (máscara 0/255, resultado) o (None, None)."""
    c = np.ascontiguousarray(roi[:, :, canal]) if roi.ndim == 3 else np.ascontiguousarray(roi)
    res, _ = _medir_robusto(c, area_min_px, area_max_px)
    if res is None:
        return None, None
    m = np.zeros(c.shape, np.uint8)
    cv2.ellipse(m, res['elipse'], 255, -1)
    return m, res

# ----------------------------------------------------------------------
# Detección de pupila. Se busca un disco oscuro y compacto rodeado por un anillo claro y parejo (el iris).
# El pelo, las cejas y las pestañas pueden ser más oscuros que la pupila pero no tienen esa forma, así que
# "lo más oscuro del frame" no se usa como único criterio.
# ----------------------------------------------------------------------
_EXIGENCIA = {False: dict(puntaje=0.3, cobertura=0.7, regularidad=0.6, uniformidad=0.5, contraste=12.0),
              True: dict(puntaje=0.12, cobertura=0.6, regularidad=0.5, uniformidad=0.35, contraste=6.0)}

def _suavizar_para_pupila(gris, rmax):
    """Gris suavizado y sin reflejos del flash (la apertura de grises borra los puntos blancos chicos)."""
    k = int(max(5, round(0.5 * rmax))) | 1
    sin_reflejo = cv2.morphologyEx(gris, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.GaussianBlur(sin_reflejo.astype(np.float32), (0, 0), max(1.5, 0.08 * rmax))

def _puntuar_pupila(b, cx, cy, r, n_rayos=36, laxo=False, crudo=None):
    """
    Puntúa el círculo (cx, cy, r) sobre la imagen suave b con n_rayos perfiles radiales:
      cobertura   fracción de rayos con el anillo exterior más claro que el interior
      regularidad fracción de rayos cuyo borde cae a un radio parecido (forma redonda)
      uniformidad qué tan parejo es el anillo (con pelo se mezcla pelo oscuro y piel clara)
      contraste   salto medio de gris entre el interior y el anillo
    Si se pasa la imagen sin suavizar (crudo), se descuenta la fracción del anillo con trazos tan oscuros como el
    interior (pelo o pestañas cruzando el anillo): en una pupila el anillo es iris y nunca llega a ese nivel.
    Devuelve (puntaje entre 0 y 1, es_valida). laxo=True usa los umbrales de _EXIGENCIA[True] (ojos de bajo contraste).
    """
    ex = _EXIGENCIA[laxo]
    paso = 1.0 if r >= 8 else 0.5
    rho = np.arange(0.0, 2.4 * r + paso, paso)
    ang = np.linspace(0, 2 * np.pi, n_rayos, endpoint=False)
    xs = (cx + np.outer(np.cos(ang), rho)).astype(np.float32)
    ys = (cy + np.outer(np.sin(ang), rho)).astype(np.float32)
    p = cv2.remap(b, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    dentro = p[:, (rho >= 0.1 * r) & (rho <= 0.7 * r)].mean(axis=1)
    fuera = p[:, (rho >= 1.3 * r) & (rho <= 1.9 * r)].mean(axis=1)
    dif = fuera - dentro
    contraste = float(np.median(dif))
    cobertura = float(np.mean(dif >= (4 if laxo else 8)))
    uniformidad = 1.0 - min(float(np.median(np.abs(fuera - np.median(fuera)))) / max(contraste, 1.0), 1.0)

    sobre = (p >= ((dentro + fuera) / 2)[:, None]) & (rho[None, :] >= 0.3 * r)          # primer cruce del nivel medio
    hay = sobre.any(axis=1)
    rb = rho[np.argmax(sobre, axis=1)]
    rm = float(np.median(rb[hay])) if hay.any() else 0.0
    regularidad = float(np.mean(hay & (np.abs(rb - rm) <= 0.3 * max(rm, 1.0))))

    puntaje = cobertura * regularidad * uniformidad * min(max(contraste, 0.0) / 40.0, 1.0)
    if crudo is not None:
        pc = cv2.remap(crudo, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[:, (rho >= 1.3 * r) & (rho <= 1.9 * r)]
        oscuro = np.mean(pc < dentro.mean() + 0.3 * (np.median(fuera) - dentro.mean()))
        puntaje *= 1.0 - min(oscuro / 0.15, 1.0)
    valida = (puntaje >= ex['puntaje'] and cobertura >= ex['cobertura'] and regularidad >= ex['regularidad']
              and uniformidad >= ex['uniformidad'] and contraste >= ex['contraste'] and abs(rm - r) <= 0.4 * r)
    return puntaje, valida

def _refinar_a_pupila(planos, cx, cy, r, puntaje, laxo=True):
    """
    Un iris oscuro con esclera clara alrededor puntúa como "pupila" aunque no lo sea. Si dentro del disco (cx, cy, r)
    hay una mancha mucho más oscura que el resto, compacta y casi concéntrica, esa mancha es la pupila y el disco era
    el iris: se devuelve (cx, cy, r, puntaje) de la mancha. Si no la hay, el disco queda como está. Se prueba sobre
    el gris y sobre el canal rojo (en iris marrones es donde más se separa la pupila).
    """
    alto, ancho = planos[0].shape[:2]
    sig = max(1.0, 0.05 * r)
    y0, y1, x0, x1 = max(0, int(cy - r) - 2), min(alto, int(cy + r) + 3), max(0, int(cx - r) - 2), min(ancho, int(cx + r) + 3)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dentro = np.hypot(xx - cx, yy - cy) <= 0.9 * r
    mejor = None
    for p in planos:
        b = cv2.GaussianBlur(p, (0, 0), sig).astype(np.float32)
        vals = b[y0:y1, x0:x1][dentro]
        if vals.size < 30:
            continue
        piso, mediana = np.percentile(vals, 2), np.median(vals)
        if mediana - piso < 10:                                   # interior parejo: ya es la pupila
            continue
        for frac in (0.2, 0.3, 0.45, 0.6):
            msk = ((b[y0:y1, x0:x1] <= piso + frac * (mediana - piso)) & dentro).astype(np.uint8)
            n, _, st, cen = cv2.connectedComponentsWithStats(msk, connectivity=8)
            for i in range(1, n):
                a, w0, h0 = st[i, 4], st[i, 2], st[i, 3]
                req = np.sqrt(a / np.pi)
                bx, by = cen[i][0] + x0, cen[i][1] + y0
                if (0.12 * r <= req <= 0.7 * r and np.hypot(bx - cx, by - cy) <= 0.5 * r and min(w0, h0) >= 0.6 * max(w0, h0)
                        and a >= 0.6 * w0 * h0):
                    pt, ok = _puntuar_pupila(b, bx, by, req, laxo=laxo)
                    if ok and (mejor is None or pt > mejor[3]):
                        mejor = (bx, by, req, pt)
    return mejor if mejor is not None else (cx, cy, r, puntaje)

def _pupilas_candidatas(img, r_min=None, r_max=None, n_max=5, n_niveles=16, laxo=False):
    """
    Pupilas posibles de la imagen (gris, BGR o un canal), de mejor a peor: [{'centro': (x, y), 'radio', 'puntaje'}].
    Umbraliza a varios niveles (así separa la pupila del pelo que se le pega) y conserva las manchas redondas, de
    tamaño plausible, que no tocan el borde y pasan _puntuar_pupila.
    r_min y r_max son el radio plausible en px; por defecto r_max = lado_ROI / 6, el mismo tope del resto del programa.
    """
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    alto, ancho = gris.shape[:2]
    if r_max is None:
        r_max = 0.4 * min(alto, ancho) / 6.0
    if r_min is None:
        r_min = r_max / 8.0

    esc = max(1.0, min(alto, ancho) / 540.0)                  # se trabaja a ~540 px de alto como máximo
    if esc > 1.0:
        gris = cv2.resize(gris, None, fx=1 / esc, fy=1 / esc, interpolation=cv2.INTER_AREA)
        alto, ancho = gris.shape[:2]
    rmin, rmax = r_min / esc, r_max / esc
    b = _suavizar_para_pupila(gris, rmax)

    lo, hi = float(b.min()) + 2.0, float(np.percentile(b[::3, ::3], 60))
    if hi <= lo:
        return []
    cands = []
    for t in np.linspace(lo, hi, n_niveles):
        n, _, st, cen = cv2.connectedComponentsWithStats((b <= t).astype(np.uint8), connectivity=8)
        if n <= 1:
            continue
        x0, y0, w0, h0, a0 = st[1:].T
        req = np.sqrt(a0 / np.pi)
        ok = ((req >= rmin) & (req <= rmax)
              & (np.minimum(w0, h0) >= 0.5 * np.maximum(w0, h0))              # no es una tira larga (ceja, pestaña)
              & (a0 >= 0.6 * w0 * h0) & (a0 <= 0.95 * w0 * h0)                # llena ~pi/4 de su caja: es un disco
              & (x0 > 0) & (y0 > 0) & (x0 + w0 < ancho) & (y0 + h0 < alto))   # no toca el borde de la imagen
        cands += [(cen[i + 1][0], cen[i + 1][1], req[i]) for i in np.flatnonzero(ok)]
        # círculos inscritos máximos: encuentran la pupila aunque a este umbral siga pegada a la sombra del párpado
        dt = cv2.distanceTransform((b <= t).astype(np.uint8), cv2.DIST_L2, 5)
        k = 2 * int(max(rmin, 2)) + 1
        picos = np.argwhere((dt == cv2.dilate(dt, np.ones((k, k), np.uint8))) & (dt >= rmin) & (dt <= rmax))
        d_p = dt[picos[:, 0], picos[:, 1]]
        banda = np.digitize(d_p, [rmin + 0.4 * (rmax - rmin), rmin + 0.7 * (rmax - rmin)])      # 8 por banda de radio
        picos = np.concatenate([picos[banda == j][np.argsort(-d_p[banda == j])[:8]] for j in range(3)])
        cands += [(float(px), float(py), float(dt[py, px])) for py, px in picos]

    circ = cv2.HoughCircles(cv2.GaussianBlur(gris, (0, 0), 2), cv2.HOUGH_GRADIENT, 1, 4, param1=60, param2=12,
                            minRadius=max(int(rmin), 2), maxRadius=int(np.ceil(rmax)))
    if circ is not None:                                      # el contorno del iris o de la pupila aunque no haya mancha aislada
        cands += [tuple(map(float, c)) for c in circ[0][:400]]
    evaluadas = []
    for cx, cy, r in cands:
        puntaje, valida = _puntuar_pupila(b, cx, cy, r, laxo=laxo, crudo=gris)
        if valida:
            evaluadas.append((puntaje * (0.8 + 0.2 * min(r / (0.4 * rmax), 1.0)), cx, cy, r, puntaje))

    rojo = img[..., 2] if img.ndim == 3 else None
    if rojo is not None and esc > 1.0:
        rojo = cv2.resize(rojo, (ancho, alto), interpolation=cv2.INTER_AREA)
    planos = [p for p in (gris, rojo) if p is not None]
    salida = []
    for _, cx, cy, r, puntaje in sorted(evaluadas, reverse=True):      # el mismo disco aparece en varios umbrales: queda el mejor
        cx, cy, r, puntaje = _refinar_a_pupila(planos, cx, cy, r, puntaje)
        if all(np.hypot(cx - s['centro'][0] / esc, cy - s['centro'][1] / esc) > 0.6 * max(r, s['radio'] / esc) for s in salida):
            salida.append({'centro': (float(cx * esc), float(cy * esc)), 'radio': float(r * esc), 'puntaje': float(puntaje)})
    salida.sort(key=lambda d: -d['puntaje'] * (0.8 + 0.2 * min(d['radio'] / esc / (0.4 * rmax), 1.0)))
    return salida[:n_max]

def _detectar_pupila(img, r_min=None, r_max=None, laxo=False):
    """La mejor pupila candidata de la imagen, o None. laxo=True admite también las de bajo contraste y las ordena todas
    por puntaje (no prefiere una candidata estricta cualquiera, como una ceja, a una pupila de contraste bajo)."""
    c = _pupilas_candidatas(img, r_min, r_max, n_max=1, laxo=laxo)
    return c[0] if c else None

def _punto_oscuro(blur, centro=None, radio=None):
    """Punto más oscuro de blur. Si se conoce la pupila (centro, radio) y ese punto cae fuera de ella (pelo, ceja,
    sombra), devuelve el más oscuro dentro de la pupila."""
    pt = cv2.minMaxLoc(blur)[2]
    if centro is None or np.hypot(pt[0] - centro[0], pt[1] - centro[1]) <= 1.3 * radio:
        return pt
    m = np.zeros(blur.shape[:2], np.uint8)
    cv2.circle(m, (int(round(centro[0])), int(round(centro[1]))), max(1, int(round(1.3 * radio))), 255, -1)
    return cv2.minMaxLoc(blur, m)[2]

def _centro_pupila_frame(frame, laxo=False):
    """(cx, cy, det) de la pupila en un frame BGR. Si el detector no encuentra ninguna, det es None y (cx, cy) es la
    zona más oscura del frame."""
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (15, 15), 0)
    det = _detectar_pupila(frame, laxo=laxo)                  # a color: el canal rojo separa mejor la pupila del iris
    if det is None:
        pt = cv2.minMaxLoc(blur)[2]
        return pt[0], pt[1], None
    pt = _punto_oscuro(blur, det['centro'], det['radio'])
    return pt[0], pt[1], det

def _medir_cerca_de_deteccion(c_img, det, desfase=(0, 0), laxo=False):
    """Mide la pupila (elipse por gradiente) de un canal arrancando en la detección det, con un rango de área centrado
    en su tamaño. desfase = posición de c_img dentro de la imagen donde se detectó; laxo = umbrales de borde permisivos.
    Devuelve el resultado o None."""
    a = np.pi * det['radio'] ** 2
    tope = np.pi * (min(c_img.shape) / 6.0) ** 2
    par = _parametros_segmentacion(max(30, 0.4 * a), max(min(3 * a, tope), 60), laxo)
    return _medir_pupila(c_img, np.array(det['centro']) - np.array(desfase), par)

def _roi_alrededor(forma, cx, cy):
    """ROI cuadrada (x, y, w, h) de 0.4 * el lado menor, centrada en (cx, cy) y contenida en el frame."""
    alto_img, ancho_img = forma[:2]
    lado = int(min(ancho_img, alto_img) * 0.4)
    return min(max(0, cx - lado // 2), ancho_img - lado), min(max(0, cy - lado // 2), alto_img - lado), lado, lado

_CACHE_MUESTREO = {}

def _muestrear_pupila(video_path, n_muestras=15, laxo=False):
    """
    Mide la pupila en n_muestras frames repartidos por el video (una sola vez por video, queda en caché).
    Devuelve [{'valida', 'centro', 'area', 'area_det', 'lado'}]: valida indica que el detector vio una pupila y se
    pudo medir; area es el área (px²) de la elipse medida en el canal rojo de la ROI y area_det la del círculo
    detectado (respaldo si la medición falla).
    """
    try:
        st = os.stat(video_path)
        clave = (os.path.abspath(video_path), st.st_size, st.st_mtime, n_muestras, laxo)
    except OSError:
        clave = None
    if clave is not None and clave in _CACHE_MUESTREO:
        return _CACHE_MUESTREO[clave]

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    muestras = []
    for i in np.linspace(0, max(total - 1, 0), n_muestras).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, f = cap.read()
        if not ret:
            continue
        cx, cy, det = _centro_pupila_frame(f, laxo)
        x, y, w, h = _roi_alrededor(f.shape, cx, cy)
        res = _medir_cerca_de_deteccion(np.ascontiguousarray(f[y:y + h, x:x + w, 2]), det, (x, y)) if det else None
        muestras.append({'valida': res is not None, 'centro': (cx, cy), 'lado': w, 'area': res['area'] if res else None,
                         'area_det': np.pi * det['radio'] ** 2 if det else None})
    cap.release()
    if clave is not None:
        _CACHE_MUESTREO[clave] = muestras
    return muestras

def _consenso_pupila(video_path, n_muestras=15):
    """Centro (x, y) donde se detectó la pupila en la mayoría de los frames muestreados; None si no hay acuerdo.
    Si con los umbrales normales no alcanzan 3 frames, se reintenta con los permisivos."""
    for laxo in (False, True):
        ms = [m for m in _muestrear_pupila(video_path, n_muestras, laxo) if m['valida']]
        if len(ms) < 3:
            continue
        cs = np.array([m['centro'] for m in ms], float)
        cerca = np.hypot(cs[:, None, 0] - cs[None, :, 0], cs[:, None, 1] - cs[None, :, 1]) <= ms[0]['lado'] / 4.0
        i = int(np.argmax(cerca.sum(axis=1)))
        if cerca[i].sum() >= 3:
            return tuple(np.median(cs[cerca[i]], axis=0))
    return None

def _nitidez(frame):
    """Nitidez del frame: varianza del Laplaciano en gris."""
    return float(cv2.meanStdDev(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_16S))[1][0, 0]) ** 2

def obtener_mejor_frame(video_path, k_alternativos=20):
    """Frame más nítido del video. Si en él no se ve la pupila donde está en la mayoría de los frames (parpadeo, ojo
    cerrado, una mancha oscura cualquiera), devuelve el más nítido de los k_alternativos siguientes en nitidez que sí
    la muestren. Lo muestra en pantalla antes de devolverlo."""
    cons = _consenso_pupila(video_path)

    def con_pupila(f):
        det = _centro_pupila_frame(f, True)[2]
        return det is not None and (cons is None or np.hypot(det['centro'][0] - cons[0], det['centro'][1] - cons[1]) <= max(2 * det['radio'], 8))

    cap = cv2.VideoCapture(video_path)
    puntajes = []
    mejor_puntaje, mejor_frame, mejor_idx = 0, None, -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        p = _nitidez(frame)
        puntajes.append(p)
        if p > mejor_puntaje:
            mejor_puntaje, mejor_frame, mejor_idx = p, frame, len(puntajes) - 1   # read() entrega un array nuevo cada vez
    cap.release()

    if mejor_frame is not None and not con_pupila(mejor_frame) and len(puntajes) > 1:
        orden = [int(i) for i in np.argsort(puntajes)[::-1] if i != mejor_idx]
        pendientes, idx, validos = set(orden[:k_alternativos]), -1, []
        cap = cv2.VideoCapture(video_path)
        while pendientes and cap.grab():
            idx += 1
            if idx in pendientes:
                pendientes.discard(idx)
                ret, f = cap.retrieve()
                if ret and con_pupila(f):
                    validos.append((puntajes[idx], f))
        cap.release()
        if validos:
            mejor_frame = max(validos, key=lambda v: v[0])[1]

    if mejor_frame is not None:
        cv2.imshow('Mejor Frame Encontrado', mejor_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return mejor_frame

def obtener_roi_automatica(frame, video_path=None):
    """ROI (x, y, w, h) centrada en la pupila del frame. Con video_path se contrasta con los demás frames del video y,
    si este no muestra la pupila o discrepa de la mayoría, se usa la posición de consenso."""
    cx, cy, det = _centro_pupila_frame(frame, True)
    if video_path is not None:
        cons = _consenso_pupila(video_path)
        if cons is not None and (det is None or np.hypot(cx - cons[0], cy - cons[1]) > int(min(frame.shape[:2]) * 0.4) / 3):
            cx, cy = int(round(cons[0])), int(round(cons[1]))
    return _roi_alrededor(frame.shape, cx, cy)

# ----------------------------------------------------------------------
# MÓDULO 1.1 - 1.2.4
# ----------------------------------------------------------------------
def _paneles(axs, paneles, **kw):
    """Dibuja [(título, imagen)] en los ejes de axs."""
    for a, (t, im) in zip(np.ravel(axs), paneles):
        a.imshow(im, **kw)
        a.set_title(t)

def _fila_hist(ax_img, ax_hist, im, titulo, t_hist):
    """Imagen en gris con su histograma al lado."""
    ax_img.imshow(im, cmap='gray', vmin=0, vmax=255)
    ax_img.set_title(titulo)
    ax_hist.plot(cv2.calcHist([im], [0], None, [256], [0, 256]), color='gray')
    ax_hist.set_title(t_hist)

def procesar_modulo_1_1(frame, x, y, w, h):
    """1.1: submuestreo espacial, cuantización y canales RGB / HSV de la ROI."""
    _titulo("--- 1.1 Resolución, cuantización y espacios de color ---")
    roi = frame[y:y + h, x:x + w]
    cv2.imshow('ROI - Ojo Centrado', roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Dimensiones y canales (Alto, Ancho, Canales):", roi.shape)
    print("Profundidad de bits:", roi.dtype)

    _respuesta_video("--- ANÁLISIS ESPECÍFICO: Pérdida de Definición en Nuestro Video ---", 'definicion')
    for f in (0.5, 0.25, 0.05):         # se achica promediando vecinos y se vuelve al tamaño original con vecino más cercano
        chica = cv2.resize(roi, (0, 0), fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(f'Resolucion {f * 100:g}%', cv2.resize(chica, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('Cuantizacion 4 bits', (roi // 16) * 16)
    cv2.imshow('Cuantizacion 2 bits', (roi // 64) * 64)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    r, g, b = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    h_c, s_c, v_c = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    _paneles(axs, [('Canal Rojo (R)', r), ('Canal Verde (G)', g), ('Canal Azul (B)', b),
                   ('Matiz (Hue)', h_c), ('Saturación (Sat)', s_c), ('Brillo (Value)', v_c)], cmap='gray')
    plt.tight_layout()
    plt.show()

def rango_pupila_iris(roi, area_min_px=250, area_max_px=2000):
    """[f1, f2]: niveles de gris que abarcan pupila e iris sin reflejos. Sin pupila medible: percentiles 1 y 40."""
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    m, res = _mascara_pupila_elipse(roi, area_min_px, area_max_px)
    if m is None:
        return float(np.percentile(gris, 1)), float(np.percentile(gris, 40))
    cx, cy = res['centro']
    yy, xx = np.indices(gris.shape)
    v = gris[(np.hypot(xx - cx, yy - cy) <= 2.6 * np.sqrt(res['area'] / np.pi)) & (gris < 235)]
    return float(np.percentile(v, 1)), float(np.percentile(v, 99))

def escalar_rango(gris, f1, f2):
    """Transformación lineal f1 -> 0 y f2 -> 255; lo que queda afuera se satura."""
    return (np.clip((gris.astype(np.float32) - f1) / max(f2 - f1, 1e-6), 0, 1) * 255).astype(np.uint8)

def procesar_modulo_1_2(frame, x, y, w, h, area_min_px=250, area_max_px=2000):
    """1.2: contraste lineal, escalado por rango de interés, ecualización global y CLAHE, con sus histogramas."""
    _titulo("--- 1.2 Contraste e histograma ---")
    roi = frame[y:y + h, x:x + w]
    roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    alpha, beta = 1.5, 50                                       # g(x) = alpha * f(x) + beta
    roi_ajustada = cv2.convertScaleAbs(roi_gris, alpha=alpha, beta=beta)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    _fila_hist(axs[0, 0], axs[0, 1], roi_gris, 'ROI Original (Grises)', 'Histograma Original')
    _fila_hist(axs[1, 0], axs[1, 1], roi_ajustada, f'ROI Ajustada (alpha={alpha}, beta={beta})', 'Histograma Ajustado')
    plt.tight_layout()
    plt.show()

    f1, f2 = rango_pupila_iris(roi, area_min_px, area_max_px)
    print(f"Escalado por rango de interés: [f1, f2] = [{f1:.0f}, {f2:.0f}]")
    fig2, axs2 = plt.subplots(3, 2, figsize=(12, 15))
    for i, (im, t, th) in enumerate([
            (escalar_rango(roi_gris, f1, f2), 'Escalado por Rango de Interés', 'Histograma Escalado'),
            (cv2.equalizeHist(roi_gris), 'Ecualización de Histograma', 'Histograma Ecualizado'),
            (cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi_gris), 'Ecualización Local (CLAHE)', 'Histograma CLAHE')]):
        _fila_hist(axs2[i, 0], axs2[i, 1], im, t, th)
    plt.tight_layout(h_pad=3.0)
    plt.show()

def _agregar_ruido(gris, tipo, rng):
    """Ruido gaussiano, sal y pimienta o speckle (multiplicativo) sobre una imagen en gris."""
    g = gris.astype(np.float32)
    if tipo == 'Gaussiano':
        r = g + rng.normal(0, 20, g.shape)
    elif tipo == 'Sal y pimienta':
        r, u = g.copy(), rng.random(g.shape)
        r[u < 0.025], r[u > 0.975] = 0, 255
    else:
        r = g + g * rng.normal(0, 0.25, g.shape)
    return np.clip(r, 0, 255).astype(np.uint8)

def procesar_modulo_1_2_4(roi, area_min_px=250, area_max_px=2000):
    """1.2.4: filtros de media, mediana y gaussiano contra tres ruidos, y cuál conserva el borde pupilar eliminando el destello."""
    _titulo("--- 1.2.4 Ruido y filtrado espacial ---")
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    rng = np.random.default_rng(0)
    filtros = {'Media': lambda im, k: cv2.blur(im, (k, k)),
               'Mediana': lambda im, k: cv2.medianBlur(im, k),
               'Gaussiano': lambda im, k: cv2.GaussianBlur(im, (k, k), 0)}

    fig, axs = plt.subplots(3, 4, figsize=(15, 11))
    mejores = {}
    print("\nPSNR (dB) / SSIM de cada imagen contra la original SIN ruido (filtros 5x5):")
    print(f"{'Ruido':<16}{'Sin filtrar':>16}" + "".join(f"{n:>16}" for n in filtros))
    for i, tipo in enumerate(['Gaussiano', 'Sal y pimienta', 'Speckle']):
        ruidosa = _agregar_ruido(gris, tipo, rng)
        textos, vals = [], {}
        for j, (nombre, im) in enumerate([('Sin filtrar', ruidosa)] + [(n, f(ruidosa, 5)) for n, f in filtros.items()]):
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

    # Destellos saturados cerca de la pupila y su versión "ideal" (inpainting) como referencia
    mascara, res = _mascara_pupila_elipse(roi, area_min_px, area_max_px)
    n, et, st, cen = cv2.connectedComponentsWithStats((gris >= 235).astype(np.uint8), connectivity=8)
    dest = np.zeros(gris.shape, np.uint8)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] > 400:                       # saturación grande (piel, esclerótica): no es destello
            continue
        if res is not None and np.hypot(cen[k][0] - res['centro'][0], cen[k][1] - res['centro'][1]) > 3.0 * np.sqrt(res['area'] / np.pi):
            continue
        dest[et == k] = 1

    print("\n--- Pregunta: ¿qué filtro preserva los bordes de la pupila eliminando los destellos? ---")
    if not dest.any() or mascara is None:
        print("No se detectaron destellos saturados (>=235) junto a la pupila, o no se pudo segmentar la pupila: "
              "no se puede hacer esta comparación en este frame.")
        plt.show()
        return
    dest_d = cv2.dilate(dest, np.ones((3, 3), np.uint8), iterations=2)
    ref = cv2.inpaint(gris, dest_d * 255, 5, cv2.INPAINT_TELEA)
    banda = _banda_borde(mascara)
    grad = lambda im: np.hypot(cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3))
    grad_ref = grad(ref)[banda].mean()
    exceso0 = float(gris[dest_d > 0].mean() - ref[dest_d > 0].mean())
    print(f"Destello detectado: {int(dest.sum())} px (nivel medio {gris[dest > 0].mean():.0f}; "
          f"sin destello sería ~{ref[dest > 0].mean():.0f})")

    kernels, filas = [5, 11, 21], []
    print(f"\n{'Filtro':<11}{'Kernel':>7}{'Destello eliminado':>20}{'Borde retenido':>16}{'PSNR':>9}{'SSIM':>8}")
    for nombre, f in filtros.items():
        for k in kernels:
            im = f(gris, k)
            elim = float(np.clip(100 * (1 - (float(im[dest_d > 0].mean() - ref[dest_d > 0].mean()) / exceso0)), 0, 100))
            ret = 100 * grad(im)[banda].mean() / grad_ref
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
    _respuesta_video("--- RESPUESTA APLICADA: Filtro Óptimo para Destellos en el Video ---", 'filtro')

    cx, cy = res['centro']
    hz = int(2.6 * np.sqrt(res['area'] / np.pi))
    y0, x0 = max(0, int(cy) - hz), max(0, int(cx) - hz)
    sl = (slice(y0, y0 + 2 * hz), slice(x0, x0 + 2 * hz))
    fig2, axs2 = plt.subplots(3, 4, figsize=(14, 10))
    for i, nombre in enumerate(filtros):
        axs2[i, 0].imshow(gris[sl], cmap='gray', vmin=0, vmax=255)
        axs2[i, 0].set_title('Original (con destello)', fontsize=9)
        for j, k in enumerate(kernels):
            f = next(x for x in filas if x[0] == nombre and x[1] == k)
            axs2[i, j + 1].imshow(f[6][sl], cmap='gray', vmin=0, vmax=255)
            axs2[i, j + 1].set_title(f"{nombre} {k}x{k}\ndestello -{f[2]:.0f}% | borde {f[3]:.0f}%", fontsize=9)
    for a in axs2.ravel():
        a.axis('off')
    plt.suptitle('Destello del flash vs. borde de la pupila según filtro y tamaño de kernel')
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# MÓDULO 1.3 - Bordes, TFD-2D, PSD y filtrado en frecuencia
# ----------------------------------------------------------------------
def _canny_auto(img):
    """Canny con umbral alto = percentil 92 del gradiente y bajo = 40 % del alto."""
    mag = cv2.magnitude(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3))
    alto = float(np.percentile(mag, 92))
    return cv2.Canny(img, 0.4 * alto, alto, L2gradient=True)

def _operadores_borde(img):
    """{operador: bordes binarios}. Los gradientes se binarizan con Otsu."""
    def binarizar(mag):
        return cv2.threshold(cv2.convertScaleAbs(mag), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    k = lambda *f: np.array(f, np.float32)
    f = lambda kern: cv2.filter2D(img, cv2.CV_64F, kern)
    sobel = lambda dx, dy: cv2.Sobel(img, cv2.CV_64F, dx, dy, ksize=3)
    return {'Roberts': binarizar(cv2.magnitude(f(k([1, 0], [0, -1])), f(k([0, 1], [-1, 0])))),
            'Prewitt': binarizar(cv2.magnitude(f(k([1, 0, -1], [1, 0, -1], [1, 0, -1])), f(k([1, 1, 1], [0, 0, 0], [-1, -1, -1])))),
            'Sobel': binarizar(cv2.magnitude(sobel(1, 0), sobel(0, 1))),
            'Laplaciano': binarizar(cv2.Laplacian(img, cv2.CV_64F)),
            'Canny': _canny_auto(img)}

def _continuidad_y_espurios(bordes, mascara, n_sect=72):
    """Continuidad: % de sectores angulares del contorno pupilar con algún píxel de borde.
    Espurios: % de píxeles lejos del contorno marcados como borde."""
    banda = _banda_borde(mascara, 7)
    ys, xs = np.nonzero(mascara)
    by, bx = np.nonzero(banda & (bordes > 0))
    sect = ((np.arctan2(by - ys.mean(), bx - xs.mean()) + np.pi) / (2 * np.pi) * n_sect).astype(int) % n_sect
    lejos = cv2.dilate(banda.astype(np.uint8), np.ones((15, 15), np.uint8)) == 0
    return 100.0 * len(np.unique(sect)) / n_sect, 100.0 * ((bordes > 0) & lejos).sum() / lejos.sum()

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
    """(frecuencia [ciclos/px], PSD promediada en anillos, PSD 2D) de la imagen sin su componente continua."""
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
    """Filtra con H(f) (f en ciclos/píxel) rellenando por reflejo para evitar el efecto de borde."""
    pad = min(img.shape) // 4
    im = np.pad(img.astype(np.float64), pad, mode='reflect')
    f = np.hypot(np.fft.fftfreq(im.shape[1])[None, :], np.fft.fftfreq(im.shape[0])[:, None])
    return np.real(np.fft.ifft2(np.fft.fft2(im) * H_fun(f)))[pad:-pad, pad:-pad]

def procesar_modulo_1_3(frame, x, y, w, h, frame_otro=None, sigma_filtro=2.0, area_min_px=250, area_max_px=2000):
    """1.3: bordes sobre la imagen original y filtrada, TFD-2D (magnitud vs fase), PSD y filtros en frecuencia."""
    _titulo("--- 1.3 Bordes y dominio frecuencial ---")
    roi = np.ascontiguousarray(frame[y:y + h, x:x + w])
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(gris, (5, 5), 0)

    # 1.3.1 Bordes
    mascara, _ = _mascara_pupila_elipse(roi, area_min_px, area_max_px)
    b_orig, b_filt = _operadores_borde(gris), _operadores_borde(suave)
    fig, axs = plt.subplots(2, 6, figsize=(20, 7.5))
    axs[0, 0].imshow(gris, cmap='gray'); axs[0, 0].set_title('ROI original')
    axs[1, 0].imshow(suave, cmap='gray'); axs[1, 0].set_title('ROI filtrada (Gaussiano 5x5)')
    print("\n1.3.1 Continuidad del contorno pupilar y bordes espurios (gradientes binarizados con Otsu; Canny con umbrales automáticos)")
    if mascara is not None:
        print(f"{'Operador':<12}{'Continuidad orig.':>19}{'filtrada':>10}{'Espurios orig.':>17}{'filtrada':>10}")
    for j, nombre in enumerate(b_orig, start=1):
        axs[0, j].imshow(b_orig[nombre], cmap='gray'); axs[1, j].imshow(b_filt[nombre], cmap='gray')
        t0 = t1 = nombre
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

    # 1.3.2 TFD-2D: magnitud y fase
    magA, faseA = _tfd(gris)
    fig3, axs3 = plt.subplots(1, 3, figsize=(15, 5))
    _paneles(axs3, [('ROI original', gris), ('Magnitud (dB, escala log)', np.fft.fftshift(20 * np.log10(magA + 1))),
                    ('Fase (rad)', np.fft.fftshift(faseA))], cmap='gray')
    for a in axs3:
        a.axis('off')
    plt.tight_layout()

    if frame_otro is not None:
        otra, nombre_B = cv2.cvtColor(np.ascontiguousarray(frame_otro[y:y + h, x:x + w]), cv2.COLOR_BGR2GRAY), 'Otro frame'
    else:
        from skimage import data
        otra, nombre_B = cv2.resize(data.camera(), (gris.shape[1], gris.shape[0]), interpolation=cv2.INTER_AREA), 'Imagen distinta (cameraman)'
    magB, faseB = _tfd(otra)
    r_mag, r_fase = _recon(magA, np.zeros_like(faseA)), _recon(np.ones_like(magA), faseA)
    r_AmagBfase, r_BmagAfase = _recon(magA, faseB), _recon(magB, faseA)
    fig4, axs4 = plt.subplots(2, 4, figsize=(17, 8.5))
    for a, (t, im) in zip(axs4.ravel(), [('A: original', gris), (f'B: {nombre_B}', otra), ('A: solo magnitud', r_mag),
                                         ('A: solo fase', r_fase), ('Magnitud de A + fase de B', r_AmagBfase),
                                         ('Magnitud de B + fase de A', r_BmagAfase)]):
        a.imshow(_norm_vis(im) if 'solo' in t or '+' in t else im, cmap='gray'); a.set_title(t); a.axis('off')
    axs4[1, 2].axis('off'); axs4[1, 3].axis('off')
    plt.tight_layout()
    grad = lambda im: cv2.magnitude(cv2.Sobel(im.astype(np.float32), cv2.CV_32F, 1, 0), cv2.Sobel(im.astype(np.float32), cv2.CV_32F, 0, 1))
    print("\n1.3.2 Correlación de cada reconstrucción con las imágenes originales (1 = idéntica):")
    print(f"   Solo magnitud de A      -> con A: {_corr(r_mag, gris):+.2f} | con los BORDES de A: {_corr(np.abs(r_mag - r_mag.mean()), grad(gris)):+.2f}")
    print(f"   Solo fase de A          -> con A: {_corr(r_fase, gris):+.2f} | con los BORDES de A: {_corr(np.abs(r_fase - r_fase.mean()), grad(gris)):+.2f}")
    print(f"   Magnitud A + fase B     -> con A: {_corr(r_AmagBfase, gris):+.2f} | con B: {_corr(r_AmagBfase, otra):+.2f}")
    print(f"   Magnitud B + fase A     -> con A: {_corr(r_BmagAfase, gris):+.2f} | con B: {_corr(r_BmagAfase, otra):+.2f}")

    # 1.3.3 PSD antes y después del filtrado espacial
    imagenes = {'Original': gris, 'Media 5x5': cv2.blur(gris, (5, 5)), 'Mediana 5x5': cv2.medianBlur(gris, 5), 'Gaussiano 5x5': suave}
    fig5, (axp, axq) = plt.subplots(1, 2, figsize=(15, 5.5))
    print("\n1.3.3 Fracción de la potencia (sin componente continua) en altas frecuencias (> 0.1 ciclos/píxel):")
    for nombre, im in imagenes.items():
        f, psd, _ = _psd_radial(im)
        axp.loglog(f[1:], psd[1:], label=nombre)
        total = psd[1:] * np.arange(1, len(psd))                    # potencia por anillo (aproximada)
        print(f"   {nombre:<15} {100 * total[f[1:] > 0.1].sum() / total.sum():6.2f} %")
    axp.set_xlabel('Frecuencia (ciclos/píxel)'); axp.set_ylabel('PSD radial'); axp.grid(True, which='both', alpha=0.4)
    axp.set_title('PSD radial antes y después del filtrado espacial'); axp.legend()
    axq.imshow(np.log10(_psd_radial(gris)[2] + 1) - np.log10(_psd_radial(suave)[2] + 1), cmap='magma')
    axq.set_title('log PSD(original) - log PSD(gaussiano): dónde se perdió potencia'); axq.axis('off')
    plt.tight_layout()

    # 1.3.4 Filtros en frecuencia. Gaussiana espacial de sigma <-> H(f) = exp(-2 pi^2 sigma^2 f^2)
    fc = np.sqrt(np.log(np.sqrt(2)) / (2 * np.pi ** 2 * sigma_filtro ** 2))   # corte a -3 dB
    gauss_f = lambda f: np.exp(-2 * np.pi ** 2 * sigma_filtro ** 2 * f ** 2)
    lp_gauss = _filtrar_frecuencia(gris, gauss_f)
    hp_gauss = _filtrar_frecuencia(gris, lambda f: 1 - gauss_f(f))
    lp_ideal = _filtrar_frecuencia(gris, lambda f: (f <= fc).astype(float))
    esp_gauss = cv2.GaussianBlur(gris, (0, 0), sigma_filtro, borderType=cv2.BORDER_REFLECT)
    u8 = lambda a: np.clip(a, 0, 255).astype(np.uint8)
    lp_gauss8, lp_ideal8 = u8(lp_gauss), u8(lp_ideal)
    fig6, axs6 = plt.subplots(2, 3, figsize=(15, 10))
    for a, (t, im) in zip(axs6.ravel(), [
            ('Original', gris), (f'Pasa-bajos gaussiano (frecuencia), fc={fc:.3f} c/px', lp_gauss8),
            (f'Pasa-bajos IDEAL, fc={fc:.3f} c/px', lp_ideal8), (f'Filtro gaussiano ESPACIAL sigma={sigma_filtro:g}', esp_gauss),
            ('Pasa-altos gaussiano (detalle/bordes)', _norm_vis(hp_gauss)),
            ('|Pasa-bajos frecuencia - espacial| x20', np.clip(np.abs(lp_gauss8.astype(float) - esp_gauss) * 20, 0, 255))]):
        a.imshow(im, cmap='gray'); a.set_title(t, fontsize=9); a.axis('off')
    plt.tight_layout()
    print(f"\n1.3.4 Filtros en frecuencia (corte fc = {fc:.3f} ciclos/píxel, equivalente a un gaussiano espacial de sigma = {sigma_filtro:g}):")
    print(f"   Pasa-bajos gaussiano (frecuencia) vs. gaussiano espacial: PSNR = {psnr(esp_gauss, lp_gauss8, data_range=255):.1f} dB | "
          f"SSIM = {ssim(esp_gauss, lp_gauss8, data_range=255):.4f}")
    print(f"   Contra la original -> PB gaussiano: PSNR {psnr(gris, lp_gauss8, data_range=255):.1f} dB / SSIM {ssim(gris, lp_gauss8, data_range=255):.3f} | "
          f"PB ideal: PSNR {psnr(gris, lp_ideal8, data_range=255):.1f} dB / SSIM {ssim(gris, lp_ideal8, data_range=255):.3f}")
    print("   (El pasa-bajos ideal corta de golpe y genera 'anillos' (ringing) alrededor de los bordes; el gaussiano no.)")
    _respuesta_video("--- RESPUESTA APLICADA: Bordes y Fase en Nuestro Registro ---", 'bordes')
    plt.show()

# ----------------------------------------------------------------------
# MÓDULO 2 - Segmentación, morfología y comparación contra la referencia manual
# ----------------------------------------------------------------------
def _poligono_a_mascara(contorno_rc, shape):
    """Contorno del snake (filas, columnas) -> máscara 0/255."""
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(contorno_rc)[:, ::-1]).astype(np.int32)], 255)
    return m

def anotar_contorno_manual(roi, ruta_salida='referencia_pupila.png', forma='elipse', escala=2):
    """Dibujo manual del contorno pupilar sobre la ROI: clic = punto (12-20 sobre el borde), z = deshacer,
    Enter = terminar (mínimo 8), Esc = cancelar. Guarda una máscara PNG (blanco = pupila) del tamaño de la ROI."""
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
        for px, py in puntos:
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
    """Guarda la ROI para anotarla en ImageJ (Edit > Selection > Create Mask, guardada como ruta_ref)."""
    cv2.imwrite(ruta, roi)
    print(f"ROI guardada en '{ruta}'. Anotala y guardá la máscara como '{ruta_ref}' (blanco = pupila, mismo tamaño).")

def _mayor_componente(mascara):
    n, et, st, _ = cv2.connectedComponentsWithStats((mascara > 0).astype(np.uint8), 8)
    if n <= 1:
        return np.zeros(mascara.shape, bool)
    return et == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))

def _dice_iou(a, b):
    inter, union, s = np.logical_and(a, b).sum(), np.logical_or(a, b).sum(), a.sum() + b.sum()
    return (2.0 * inter / s if s else 0.0), (inter / union if union else 0.0)

def procesar_modulo_2_3(roi, mascaras, ruta_ref='referencia_pupila.png', area_min_px=250, area_max_px=2000):
    """2.3: Dice e IoU de cada máscara del Módulo 2 (y de la elipse por gradiente) contra la referencia manual."""
    _titulo("--- 2.3 Comparación de máscaras contra la referencia manual (Dice / IoU) ---")
    if not os.path.exists(ruta_ref):
        print(f"No existe '{ruta_ref}'. Creala con anotar_contorno_manual(roi) o con guardar_roi_para_anotar(roi) + ImageJ.")
        return None
    ref = cv2.imread(ruta_ref, cv2.IMREAD_GRAYSCALE)
    if ref is None or ref.shape != roi.shape[:2]:
        print(f"La referencia debe tener el tamaño de la ROI {roi.shape[1]}x{roi.shape[0]} px.")
        return None
    ref = ref > 127
    if ref.mean() > 0.5:                      # ImageJ a veces guarda la máscara invertida
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
        filas.append(dict(nombre=nombre, dice=d_all, iou=i_all, dice_c=d_c, iou_c=i_c, area=int(b.sum()),
                          err=100 * (int(b.sum()) - int(ref.sum())) / ref.sum(), masc=c))
    orden = sorted(filas, key=lambda f: -f['dice_c'])
    print(f"Área de la referencia manual: {int(ref.sum())} px")
    print(f"{'Método':<34}{'Dice':>7}{'IoU':>7}{'Dice (mayor comp.)':>20}{'IoU (mayor comp.)':>19}{'Área':>8}{'Error área':>12}")
    for f in orden:
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

    # Contorno de la referencia (verde) contra el de cada máscara (rojo)
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(ref)
    r = int(2.4 * np.sqrt(ref.sum() / np.pi))
    y0, x0 = max(0, int(ys.mean()) - r), max(0, int(xs.mean()) - r)
    sl = (slice(y0, y0 + 2 * r), slice(x0, x0 + 2 * r))
    cols = 4
    filas_fig = int(np.ceil(len(filas) / cols))
    fig, axs = plt.subplots(filas_fig, cols, figsize=(4 * cols, 4 * filas_fig))
    axs = np.atleast_1d(axs).ravel()
    for a in axs:
        a.axis('off')
    for a, f in zip(axs, orden):
        a.imshow(gris[sl], cmap='gray')
        a.contour(ref[sl], levels=[0.5], colors='lime', linewidths=1.5)
        if f['masc'].any():
            a.contour(f['masc'][sl], levels=[0.5], colors='red', linewidths=1.2)
        a.set_title(f"{f['nombre']}\nDice {f['dice_c']:.3f}", fontsize=9)
    plt.suptitle('Verde = referencia manual | Rojo = máscara del método (mayor componente)')
    plt.tight_layout()
    plt.show()
    return filas

def _semilla_pupila(roi):
    """(detección o None, centro, radio) aproximados de la pupila en la ROI. Sin detección: el centro de la ROI con un
    radio de lado/10."""
    det = _detectar_pupila(roi, r_max=min(roi.shape[:2]) / 6.0, laxo=True)
    if det:
        return det, det['centro'], det['radio']
    return None, (roi.shape[1] / 2.0, roi.shape[0] / 2.0), min(roi.shape[:2]) / 10.0

def _umbral_pupila_auto(blur_r, fallback=45, frac_max=np.pi / 36, pt=None):
    """Umbral (Otsu recursivo) que deja en la zona oscura central un área compatible con una pupila: no puede
    ocupar más de pi/36 de la ROI, el mismo límite r <= lado/6 que usa el Módulo 3.2."""
    if pt is None:
        pt = cv2.minMaxLoc(blur_r)[2]
    t, maxima = 255, frac_max * blur_r.size
    for _ in range(5):
        oscuros = blur_r[blur_r <= t]
        if oscuros.size < 50 or oscuros.min() == oscuros.max():
            return fallback
        t = cv2.threshold(oscuros.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        et = cv2.connectedComponents((blur_r <= t).astype(np.uint8))[1]
        if et[pt[1], pt[0]] > 0 and (et == et[pt[1], pt[0]]).sum() <= maxima:
            return t
    return t


def procesar_modulo_2(roi):
    """2.1 (Otsu, K-Means, snake, watershed) y 2.2 (morfología) sobre el canal rojo de la ROI.
    Devuelve (máscara limpia final, {nombre: máscara}) para compararlas en el 2.3."""
    _titulo("--- 2.1 y 2.2 Segmentación de la pupila y morfología ---")
    r = cv2.split(roi)[2]
    blur = cv2.GaussianBlur(r, (15, 15), 0)
    det, (cx, cy), radio = _semilla_pupila(roi)
    dist = np.hypot(*(np.indices(r.shape)[::-1] - np.array([cx, cy])[:, None, None]))
    # Otsu y K-Means se aplican a la vecindad de la pupila (3 radios): sobre la ROI entera pesan el pelo, la ceja
    # y la piel, y la pupila queda como una mancha chica dentro del grupo oscuro.
    local = dist <= 3.0 * radio if det else np.ones(r.shape, bool)
    vl = blur[local]
    umbral_otsu = cv2.threshold(vl.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[0]
    mascara_otsu = ((blur <= umbral_otsu) & local).astype(np.uint8) * 255

    K = 5
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, etiquetas, centros = cv2.kmeans(np.float32(vl.reshape((-1, 1))), K, None, criterio, 10, cv2.KMEANS_RANDOM_CENTERS)
    mascara_kmeans = np.zeros(r.shape, np.uint8)
    mascara_kmeans[local] = 255 * (etiquetas.ravel() == np.argmin(centros.ravel()))

    # Snake: arranca como un círculo algo mayor que la pupila (dentro del iris) y se contrae hasta el borde
    sin_reflejo = _rellenar_reflejos(r).astype(np.float32)
    s = np.linspace(0, 2 * np.pi, 100)
    contorno_inicial = np.array([cy + 1.5 * radio * np.sin(s), cx + 1.5 * radio * np.cos(s)]).T
    snake = active_contour(cv2.GaussianBlur(sin_reflejo, (0, 0), 2.5) / 255.0, contorno_inicial, alpha=0.02, beta=10, gamma=0.01)

    # Watershed con marcadores sobre el gradiente: núcleo de la pupila (2) y fondo lejano (1); el borde cae en la cresta
    suave = cv2.GaussianBlur(sin_reflejo, (0, 0), 1.0)
    grad = cv2.normalize(cv2.magnitude(cv2.Sobel(suave, cv2.CV_32F, 1, 0), cv2.Sobel(suave, cv2.CV_32F, 0, 1)),
                         None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    marcadores = np.zeros(r.shape, np.int32)
    marcadores[dist > 1.8 * radio] = 1
    marcadores[dist <= 0.5 * radio] = 2
    marcadores = cv2.watershed(cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR), marcadores)
    roi_watershed = roi.copy()
    roi_watershed[marcadores == -1] = [0, 0, 255]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].imshow(mascara_otsu, cmap='gray'); axs[0, 0].set_title('1. Otsu (Canal Rojo)')
    axs[0, 1].imshow(mascara_kmeans, cmap='gray'); axs[0, 1].set_title(f'2. K-Means (K={K})')
    axs[1, 0].imshow(r, cmap='gray')
    axs[1, 0].plot(contorno_inicial[:, 1], contorno_inicial[:, 0], '--r', lw=2, label='Inicial')
    axs[1, 0].plot(snake[:, 1], snake[:, 0], '-b', lw=2, label='Snake')
    axs[1, 0].legend(); axs[1, 0].set_title('3. Contornos Activos (Snakes)')
    axs[1, 1].imshow(roi_watershed[..., ::-1]); axs[1, 1].set_title('4. Watershed (Borde Rojo)')
    plt.tight_layout()
    plt.show()

    # 2.2 Morfología sobre la zona oscura central
    pt_osc = _punto_oscuro(blur, det['centro'] if det else None, det['radio'] if det else None)
    if det:                                    # punto medio entre el interior de la pupila y su entorno (el iris)
        umbral_pupila = (blur[dist <= 0.6 * radio].mean() + blur[(dist >= 1.4 * radio) & (dist <= 2.0 * radio)].mean()) / 2
    else:
        umbral_pupila = _umbral_pupila_auto(blur, pt=pt_osc)
    mascara_base = cv2.threshold(blur, umbral_pupila, 255, cv2.THRESH_BINARY_INV)[1]
    etq = cv2.connectedComponents(mascara_base)[1]
    if etq[pt_osc[1], pt_osc[0]] > 0:
        mascara_base = np.where(etq == etq[pt_osc[1], pt_osc[0]], 255, 0).astype(np.uint8)
    print(f"Umbral automático de pupila: {umbral_pupila:.0f}")

    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    erosion, dilatacion = cv2.erode(mascara_base, el), cv2.dilate(mascara_base, el)
    apertura = cv2.morphologyEx(mascara_base, cv2.MORPH_OPEN, el, iterations=2)
    cierre = cv2.morphologyEx(mascara_base, cv2.MORPH_CLOSE, el, iterations=3)
    mascara_limpia = cv2.morphologyEx(cierre, cv2.MORPH_OPEN, el, iterations=2)
    esqueleto = skeletonize(cv2.morphologyEx(mascara_limpia, cv2.MORPH_GRADIENT, el) > 0)

    fig2, axs2 = plt.subplots(2, 3, figsize=(15, 10))
    _paneles(axs2, [('Máscara Base (Umbral < 45)', mascara_base), ('Erosión', erosion), ('Dilatación', dilatacion),
                    ('Apertura (Saca ruido externo)', apertura), ('Cierre (Rellena destello de flash)', cierre),
                    ('Esqueletización del contorno', esqueleto)], cmap='gray')
    plt.tight_layout()
    plt.show()

    mascaras = {'Otsu': mascara_otsu, 'K-Means': mascara_kmeans, 'Snake': _poligono_a_mascara(snake, roi.shape[:2]),
                'Watershed': (marcadores == 2).astype(np.uint8) * 255, 'Umbral<45': mascara_base, 'Erosión': erosion,
                'Dilatación': dilatacion, 'Apertura': apertura, 'Cierre': cierre, 'Cierre+Apertura (final)': mascara_limpia}
    return mascara_limpia, mascaras

# ----------------------------------------------------------------------
# MÓDULO 3.1 - Forma y textura
# ----------------------------------------------------------------------
def _glcm_region(canal, region, niveles=32):
    """GLCM (distancia 1, 4 ángulos) con solo los píxeles de 'region'; los pares con píxeles de afuera no cuentan."""
    claves = ('contrast', 'homogeneity', 'energy', 'correlation')
    ys, xs = np.nonzero(region)
    if ys.size == 0:
        return {k: float('nan') for k in claves}
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    q = (canal[y0:y1, x0:x1] // (256 // niveles)).astype(np.uint8)
    q[~region[y0:y1, x0:x1]] = niveles                            # nivel reservado para "fuera de la región"
    g = graycomatrix(q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                     levels=niveles + 1, symmetric=True, normed=False).astype(np.float64)
    g[niveles, :, :, :] = 0
    g[:, niveles, :, :] = 0
    g /= np.maximum(g.sum(axis=(0, 1), keepdims=True), 1)
    return {k: float(graycoprops(g, k).mean()) for k in claves}

def procesar_modulo_3_1(roi, mascara_limpia, factor_iris=2.6, usar_elipse=False, area_min_px=250, area_max_px=2000):
    """3.1: descriptores de forma de la pupila y textura (primer y segundo orden) de pupila e iris por canal.
    usar_elipse=True reemplaza la máscara del Módulo 2 por la elipse por gradiente del Módulo 3.2."""
    _titulo("--- 3.1 Forma y textura ---")
    if usar_elipse:
        m_el, _ = _mascara_pupila_elipse(roi, area_min_px, area_max_px)
        if m_el is not None:
            mascara_limpia = m_el
        else:
            print("No se pudo segmentar con el método por gradiente; se usa la máscara del Módulo 2.")
    contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        print("No se detectó pupila en la máscara.")
        return
    c = max(contornos, key=cv2.contourArea)
    area, perimetro, M = cv2.contourArea(c), cv2.arcLength(c, True), cv2.moments(c)
    cX = M["m10"] / M["m00"] if M["m00"] else 0.0
    cY = M["m01"] / M["m00"] if M["m00"] else 0.0
    circ = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0
    excentricidad = np.nan
    if len(c) >= 5:
        d1, d2 = cv2.fitEllipse(c)[1]
        excentricidad = float(np.sqrt(max(1 - (min(d1, d2) / max(d1, d2)) ** 2, 0)))

    print("\n--- 3.1.1 Descriptores Morfométricos (respuestas genéricas) ---")
    print(f"Área (A): {area:.0f} px²  (diámetro equivalente {2 * np.sqrt(area / np.pi):.1f} px)")
    print(f"Perímetro (P): {perimetro:.2f} px")
    print(f"Centroide (X_c, Y_c): ({cX:.1f}, {cY:.1f})")
    print(f"Circularidad: {circ:.4f} (1 = círculo perfecto) | Excentricidad: {excentricidad:.3f} (0 = círculo)")

    pup = np.zeros(mascara_limpia.shape, np.uint8)
    cv2.drawContours(pup, [c], -1, 255, -1)
    pup = pup > 0
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    yy, xx = np.indices(gris.shape)
    d = np.hypot(xx - cX, yy - cY)
    r_eq = np.sqrt(area / np.pi)
    iris = (d >= 1.25 * r_eq) & (d <= factor_iris * r_eq) & (gris < 235)       # anillo sin destellos
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
            ent = float(entropy(np.histogram(px, bins=256, range=(0, 256), density=True)[0] + 1e-12, base=2))
            if reg_nombre.startswith('Iris'):
                entropia_iris[nombre] = ent
            print(f"   {nombre:<10} Media: {px.mean():6.2f} | Std: {px.std():6.2f} | Skewness: {skew(px):6.2f} | Entropía: {ent:5.2f} bits")

    print("\n--- 3.1.3 Textura de Segundo Orden (GLCM, 32 niveles) en el IRIS (respuestas genéricas) ---")
    glcm = {nombre: _glcm_region(canal, iris) for nombre, canal in canales.items()}
    for nombre, v in glcm.items():
        print(f"   {nombre:<10} Contraste: {v['contrast']:7.3f} | Homogeneidad: {v['homogeneity']:.4f} | "
              f"Energía: {v['energy']:.4f} | Correlación: {v['correlation']:.4f}")

    # Riqueza textural: más entropía y contraste, menos homogeneidad y energía
    nombres = list(canales)

    def posiciones(valores, mayor_es_mejor):
        orden = sorted(nombres, key=lambda n: valores[n], reverse=mayor_es_mejor)
        return {n: orden.index(n) + 1 for n in nombres}

    rankings = [posiciones(entropia_iris, True),
                posiciones({n: glcm[n]['contrast'] for n in nombres}, True),
                posiciones({n: glcm[n]['homogeneity'] for n in nombres}, False),
                posiciones({n: glcm[n]['energy'] for n in nombres}, False)]
    puntaje = {n: sum(rk[n] for rk in rankings) for n in nombres}
    ganador, peor = min(puntaje, key=puntaje.get), max(puntaje, key=puntaje.get)
    print("\nPosición de cada canal (1 = más textura) -> " + " | ".join(f"{n}: {puntaje[n]} pts" for n in nombres) + "  (menos puntos = más textura)")
    if puntaje[ganador] == puntaje[peor]:
        print("-> Los tres canales dan lo mismo (¿la imagen es en escala de grises?): no hay un canal con más textura.")
    else:
        print(f"-> Canal con mayor riqueza textural del estroma iridiano: {ganador}. El de menor riqueza: {peor}.")
    print("   (Una textura rica = más entropía y contraste, y menos homogeneidad y energía. Ojo: la cuantización usa niveles "
          "absolutos, así que un canal muy oscuro usa pocos niveles y parece menos texturado.)")
    _respuesta_video("--- RESPUESTA APLICADA: Textura del Iris en Nuestro Video ---", 'textura')

# ----------------------------------------------------------------------
# MÓDULO 3.2 - Dinámica pupilar
# Segmentación frame a frame: se lanzan rayos desde el centro de la pupila, se ubica en cada uno el máximo
# gradiente oscuro -> claro y se ajusta una elipse robusta. No depende del brillo y tolera reflejos y pestañas.
# ----------------------------------------------------------------------
def _parametros_segmentacion(area_min, area_max, laxo=False):
    """Parámetros de la medición derivados del rango de área plausible (px²). laxo=True baja los umbrales de borde
    (contraste, rayos válidos y sectores cubiertos) para ojos de bajo contraste."""
    r_min, r_max = np.sqrt(area_min / np.pi), np.sqrt(area_max / np.pi)
    return {'area_min': area_min, 'area_max': area_max,
            'r_lo': max(2.0, 0.8 * r_min),            # inicio de cada rayo (px)
            'r_hi': 1.15 * r_max,                     # fin de cada rayo (px)
            'H': int(np.ceil(3 * r_max)),             # semilado de la ventana de trabajo
            'Hv': int(np.ceil(2.5 * r_max)),          # semilado del recorte de verificación visual
            'radio_busqueda': 0.8 * r_max,            # cuánto puede moverse el centro entre frames
            'k_centro': int(2 * round(r_min)) | 1,    # lado impar del promedio que ubica la zona oscura
            'sigma': 1.2,                             # suavizado previo (px)
            'contraste_min': 2.0 if laxo else 4.0,    # salto mínimo de gris para aceptar un borde
            'min_puntos': 10 if laxo else 14,         # mínimo de rayos válidos (de 48)
            'min_sectores': 4 if laxo else 5}         # sectores de 45° donde debe verse el borde (de 8)

def _rellenar_reflejos(canal_u8, umbral=235, area_max_reflejo=150):
    """Inpainting de los reflejos del flash chicos; las zonas saturadas grandes (piel) se dejan."""
    mascara = (canal_u8 >= umbral).astype(np.uint8)
    if not mascara.any():
        return canal_u8
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    es_chica = np.zeros(n, np.uint8)
    es_chica[1:] = stats[1:, cv2.CC_STAT_AREA] <= area_max_reflejo
    chica = es_chica[etiquetas]
    if not chica.any():
        return canal_u8
    return cv2.inpaint(canal_u8, cv2.dilate(chica, np.ones((3, 3), np.uint8), iterations=2), 3, cv2.INPAINT_TELEA)

def _ajustar_elipse(img, c, par, n_rayos=48):
    """Dado un centro aproximado c, busca el borde de la pupila por rayos y ajusta una elipse. None si no es creíble."""
    radios = np.arange(par['r_lo'], par['r_hi'], 0.5)
    ang = np.linspace(0, 2 * np.pi, n_rayos, endpoint=False)
    xs = (c[0] + np.outer(np.cos(ang), radios)).astype(np.float32)
    ys = (c[1] + np.outer(np.sin(ang), radios)).astype(np.float32)
    perfiles = gaussian_filter1d(cv2.remap(img, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), sigma=1.0, axis=1)
    idx = np.argmax(np.gradient(perfiles, axis=1), axis=1)            # mayor salto oscuro -> claro
    ult, filas = len(radios) - 1, np.arange(n_rayos)
    salto = perfiles[filas, np.minimum(idx + 4, ult)] - perfiles[filas, np.maximum(idx - 4, 0)]
    r_borde = radios[idx]

    ok = (salto >= par['contraste_min']) & (idx > 0) & (idx < ult)
    if ok.sum() < par['min_puntos']:
        return None
    rm = np.median(r_borde[ok])
    ok &= np.abs(r_borde - rm) <= max(1.5, 0.25 * rm)                 # descarta pestañas, reflejos y párpado
    if ok.sum() < par['min_puntos'] or len(np.unique((ang[ok] // (np.pi / 4)).astype(int))) < par['min_sectores']:
        return None                                                   # el borde debe verse alrededor, no de un solo lado

    pts = np.stack([c[0] + r_borde[ok] * np.cos(ang[ok]), c[1] + r_borde[ok] * np.sin(ang[ok])], axis=1).astype(np.float32)
    (ex, ey), (d1, d2), angulo = cv2.fitEllipse(pts)
    if min(d1, d2) / max(d1, d2) < 0.6:                               # demasiado alargada: se usa un círculo
        elipse, area, centro = ((float(c[0]), float(c[1])), (2 * rm, 2 * rm), 0.0), np.pi * rm ** 2, (c[0], c[1])
    else:
        elipse, area, centro = ((ex, ey), (d1, d2), angulo), np.pi / 4.0 * d1 * d2, (ex, ey)
    if not (par['area_min'] <= area <= par['area_max']):
        return None
    return {'area': float(area), 'centro': centro, 'elipse': elipse, 'calidad': float(ok.sum() / n_rayos)}

def _medir_pupila(c_img, centro_prev, par):
    """Mide la pupila de un canal buscando cerca de la posición anterior. Devuelve {'area','centro','elipse','calidad'} o None."""
    H = min(par['H'], min(c_img.shape) // 2)
    h_roi, w_roi = c_img.shape
    cx0, cy0 = int(round(centro_prev[0])), int(round(centro_prev[1]))
    x0 = int(np.clip(cx0 - H, 0, max(0, w_roi - 2 * H)))
    y0 = int(np.clip(cy0 - H, 0, max(0, h_roi - 2 * H)))
    ventana = _rellenar_reflejos(np.ascontiguousarray(c_img[y0:y0 + 2 * H, x0:x0 + 2 * H]))
    suave = cv2.GaussianBlur(ventana.astype(np.float32), (0, 0), par['sigma'])

    medio = cv2.blur(suave, (par['k_centro'], par['k_centro']))       # zona más oscura cerca del centro anterior
    mascara = np.zeros(medio.shape, np.uint8)
    cv2.circle(mascara, (cx0 - x0, cy0 - y0), int(par['radio_busqueda']), 255, -1)
    c = np.array(cv2.minMaxLoc(medio, mascara)[2], dtype=float)

    mejor = None
    for _ in range(3):                                                # el centro se refina con la propia elipse
        res = _ajustar_elipse(suave, c, par)
        if res is None:
            break
        mejor, c = res, np.array(res['centro'], dtype=float)
    if mejor is None:
        return None
    (ex, ey), ejes, ang = mejor['elipse']
    mejor['elipse'] = ((ex + x0, ey + y0), ejes, ang)
    mejor['centro'] = (mejor['centro'][0] + x0, mejor['centro'][1] + y0)
    return mejor

def _parece_pupila(c_img, c, par):
    """True si medir desde c da un disco con anillo claro (pupila) y no pelo o sombra."""
    res = _medir_pupila(c_img, c, par)
    if res is None:
        return False
    b = _suavizar_para_pupila(c_img, np.sqrt(par['area_max'] / np.pi))
    return _puntuar_pupila(b, res['centro'][0], res['centro'][1], float(np.sqrt(res['area'] / np.pi)))[1]

def _centro_inicial(c_img, par):
    """Centro aproximado de la pupila: la zona más oscura si parece una pupila; si no, la mejor pupila candidata
    de la imagen; y si no hay ninguna, la zona más oscura."""
    suave = cv2.GaussianBlur(c_img.astype(np.float32), (0, 0), par['sigma'] + 1)
    c = np.array(cv2.minMaxLoc(cv2.blur(suave, (par['k_centro'], par['k_centro'])))[2], dtype=float)
    if _parece_pupila(c_img, c, par):
        return c
    det = _detectar_pupila(c_img, r_min=np.sqrt(par['area_min'] / np.pi), r_max=np.sqrt(par['area_max'] / np.pi), laxo=True)
    return np.array(det['centro'], dtype=float) if det else c

def _medir_robusto(c_img, area_min, area_max):
    """
    Mide la pupila de un canal. Si el detector ve una pupila, el rango de área se acota a 0.2-5 veces su área para que
    el ajuste no se quede con el iris, y si la medición falla se arranca desde la detección y luego con umbrales de
    borde permisivos. Devuelve (resultado o None, (area_min, area_max)): el rango acotado o, si el dado no incluía la
    pupila, ensanchado alrededor del área medida (tope pi * (lado / 6)²).
    """
    tope = np.pi * (min(c_img.shape) / 6.0) ** 2
    det = _detectar_pupila(c_img, r_max=min(c_img.shape) / 6.0, laxo=True)
    a_det = np.pi * det['radio'] ** 2 if det else None
    lo, hi = area_min, area_max
    if a_det:
        lo, hi = max(area_min, int(0.2 * a_det)), min(area_max, int(5 * a_det))
        if not lo <= a_det <= hi:
            lo, hi = max(30, int(0.2 * a_det)), int(min(tope, 5 * a_det))
    for laxo in (False, True):
        par = _parametros_segmentacion(lo, hi, laxo)
        res = _medir_pupila(c_img, _centro_inicial(c_img, par), par)
        if res is None and det:
            res = _medir_cerca_de_deteccion(c_img, det, laxo=laxo)
        if res is not None:
            if not lo <= res['area'] <= hi:
                lo, hi = int(max(30, min(lo, 0.5 * res['area']))), int(min(tope, max(hi, 2 * res['area'])))
            return res, (lo, hi)
    return None, (area_min, area_max)

def _validar_rango(roi, area_min, area_max, canal=2):
    """Ajusta el rango de área a la pupila de la ROI: lo acota si deja entrar al iris y lo ensancha si no la incluye."""
    _, (a0, a1) = _medir_robusto(np.ascontiguousarray(roi[:, :, canal]), area_min, area_max)
    if (a0, a1) != (area_min, area_max):
        print(f"Rango de área ajustado a {a0}-{a1} px² según la pupila detectada en la ROI (estimado: {area_min}-{area_max} px²).")
    return a0, a1

def _candidatos_roi(frame, k=8):
    """k ROI candidatas: primero las que contienen algo con forma de pupila y luego las zonas más oscuras del frame."""
    blur = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (15, 15), 0).astype(np.float32)
    alto, ancho = blur.shape
    lado = int(min(ancho, alto) * 0.4)
    cands = [(int(np.clip(int(c['centro'][0]) - lado // 2, 0, ancho - lado)),
              int(np.clip(int(c['centro'][1]) - lado // 2, 0, alto - lado)), lado, lado)
             for c in _pupilas_candidatas(frame, n_max=k)]
    for _ in range(k - len(cands)):
        _, _, (cx, cy), _ = cv2.minMaxLoc(blur)
        cands.append((int(np.clip(cx - lado // 2, 0, ancho - lado)), int(np.clip(cy - lado // 2, 0, alto - lado)), lado, lado))
        cv2.circle(blur, (int(cx), int(cy)), max(lado // 2, 1), 255, -1)       # se anula una ROI entera para buscar la siguiente
    return cands

def _roi_estable(video_path, fps, t_inicio, t_fin, par, canal, n_muestras=10):
    """Elige la ROI probando varias candidatas en frames repartidos por la ventana de análisis y quedándose con la
    que tiene pupila medible en más frames. Devuelve (x, y, w, h, mensaje) o None."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i0 = int(round(t_inicio * fps))
    i1 = i0 + int(10 * fps) if total <= 0 else total - 1
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

    lado = _candidatos_roi(muestras[0], 1)[0][2]
    grupos = []                                            # candidatos parecidos de todos los frames, agrupados
    for fr in muestras:
        for (x, y, _, _) in _candidatos_roi(fr):
            for g in grupos:
                if abs(x - np.median([q[0] for q in g])) < lado / 2 and abs(y - np.median([q[1] for q in g])) < lado / 2:
                    g.append((x, y))
                    break
            else:
                grupos.append([(x, y)])

    mejor, mejor_clave = None, (-1, -1)
    for g in grupos:
        x, y = int(np.median([q[0] for q in g])), int(np.median([q[1] for q in g]))
        ok = 0
        for fr in muestras:
            rec = fr[y:y + lado, x:x + lado]
            c_img = np.ascontiguousarray(rec[:, :, canal] if canal in (0, 1, 2) else cv2.cvtColor(rec, cv2.COLOR_BGR2GRAY))
            r = _medir_pupila(c_img, _centro_inicial(c_img, par), par)
            ok += r is not None and r['calidad'] >= 0.4
        if (ok, len(g)) > mejor_clave:
            mejor, mejor_clave = (x, y), (ok, len(g))
    x, y = mejor
    if mejor_clave[0] == 0:
        msg = ("ADVERTENCIA: no se encontró una pupila clara en ninguna zona candidata; se usa la zona más oscura. "
               "Pasá roi_manual=(x, y, w, h) con la ROI del Módulo 1.")
    else:
        msg = f"ROI elegida automáticamente: pupila detectable en {mejor_clave[0]} de {len(muestras)} frames de prueba."
    return x, y, lado, lado, msg

def _recorte_visual(roi, centro, res, par):
    """(recorte de la ROI alrededor de la pupila, elipse en coordenadas del recorte) para la figura de verificación."""
    Hv = min(par['Hv'], min(roi.shape[:2]) // 2)
    x0 = int(np.clip(int(centro[0]) - Hv, 0, max(0, roi.shape[1] - 2 * Hv)))
    y0 = int(np.clip(int(centro[1]) - Hv, 0, max(0, roi.shape[0] - 2 * Hv)))
    elipse = None
    if res is not None:
        (ex, ey), ejes, ang = res['elipse']
        elipse = ((ex - x0, ey - y0), ejes, ang)
    return roi[y0:y0 + 2 * Hv, x0:x0 + 2 * Hv].copy(), elipse

def _limpiar_serie(area_cruda, fps, n_sigma=3.0):
    """Limpia A(t): los frames sin medición (parpadeo, oclusión) son NaN, los valores aislados incoherentes con sus
    vecinos se descartan con un filtro de Hampel (mediana ± 3 MAD), los huecos se interpolan y se suaviza con
    Savitzky-Golay. Devuelve (serie limpia, serie suave, máscara de artefactos) o (None, None, None)."""
    n = len(area_cruda)
    idx, x = np.arange(n), area_cruda.copy()
    valido = ~np.isnan(x)
    if valido.sum() < 5:
        return None, None, None
    x_i = np.interp(idx, idx[valido], x[valido])
    k = max(4, int(round(0.2 * fps)))                                 # semiventana de Hampel (~0.2 s)
    atipico = np.zeros(n, dtype=bool)
    for i in range(n):
        v = x_i[max(0, i - k):min(n, i + k + 1)]
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med))
        atipico[i] = abs(x_i[i] - med) > n_sigma * max(mad, 0.03 * med)
    es_artefacto = np.isnan(x) | atipico
    ok = ~es_artefacto
    if ok.sum() < 5:
        return None, None, None
    x_limpia = np.interp(idx, idx[ok], x[ok])
    w = int(round(0.3 * fps)) | 1
    w = max(5, min(w, n if n % 2 == 1 else n - 1))
    return x_limpia, savgol_filter(x_limpia, w, 2), es_artefacto

def _detectar_estimulo(brillo, fps, idx_forzado=None):
    """Escalón de brillo: compara el promedio de ~0.15 s antes y después de cada frame.
    Devuelve (índice, signo, intensidad); signo +1 si el brillo baja (dilatación) y -1 si sube (constricción)."""
    n = len(brillo)
    k = max(2, int(round(0.15 * fps)))
    score = np.zeros(n)
    for i in range(k, n - k + 1):
        score[i] = np.mean(brillo[i - k:i]) - np.mean(brillo[i:i + k])
    i = int(np.argmax(np.abs(score))) if idx_forzado is None else int(idx_forzado)
    return i, (1 if score[i] >= 0 else -1), abs(score[i])

def _cruce(t, s, i0, i1, umbral):
    """Tiempo (interpolado) en que s supera 'umbral' entre los índices i0 e i1."""
    for i in range(max(i0, 1), i1 + 1):
        if s[i] >= umbral:
            if s[i] == s[i - 1]:
                return t[i]
            return t[i - 1] + (umbral - s[i - 1]) / (s[i] - s[i - 1]) * (t[i] - t[i - 1])
    return np.nan

def _extraer_parametros(t, a, i_est, signo, fps):
    """Parámetros de la respuesta pupilar sobre s = signo * A(t), así la respuesta siempre es una subida
    (dilatación: signo +1; constricción: signo -1). None si no hay respuesta posterior al estímulo."""
    s = signo * a
    t_est = t[i_est]
    pre = np.where((t >= t_est - 1.0) & (t < t_est - 0.1))[0]
    if len(pre) < 3:
        pre = np.arange(0, max(3, i_est - 1))
    basal, ruido = np.median(s[pre]), np.std(s[pre])

    post = np.arange(i_est, len(s))
    i_max = int(post[np.argmax(s[post])])
    amp = s[i_max] - basal
    if amp <= 0:
        return None
    i_ext = next(i for i in range(i_est, i_max + 1) if s[i] >= basal + 0.95 * amp)    # 95 % de la respuesta

    t_ini = _cruce(t, s, i_est, i_ext, basal + min(max(0.10 * amp, 3 * ruido), 0.30 * amp))
    t10 = _cruce(t, s, i_est, i_ext, basal + 0.10 * amp)
    t90 = _cruce(t, s, i_est, i_ext, basal + 0.90 * amp)
    w = max(5, int(round(0.3 * fps)) | 1)
    w = min(w, len(s) if len(s) % 2 == 1 else len(s) - 1)
    ds = savgol_filter(s, w, 2, deriv=1, delta=1.0 / fps)
    vel_media = (0.8 * amp / (t90 - t10)) if (not np.isnan(t10) and not np.isnan(t90) and t90 > t10) else np.nan
    return {'basal': float(signo * basal), 'extremo': float(signo * s[i_max]), 'amplitud': float(amp),
            'i_ext': i_ext, 't_ext': float(t[i_ext]), 't_ini': float(t_ini), 'latencia': float(t_ini - t_est),
            't10': float(t10), 't90': float(t90), 'vel_media': float(vel_media),
            'vel_pico': float(np.max(ds[i_est:i_ext + 1]))}

def _metricas_error(pares):
    """(MAE, RMSE) de una lista de pares (automático, manual)."""
    e = np.array([a - m for a, m in pares], dtype=float)
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))

def _escribir_csv(ruta, encabezado, filas):
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(encabezado)
        wr.writerows(filas)

def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else v

def _medir_video(video_path, par, roi_xywh, canal, fps, t_inicio, t_fin):
    """Mide el área pupilar en cada frame de la ventana [t_inicio, t_fin]. Devuelve (tiempos, brillos, áreas, recortes
    para verificación, índices de frame, primer frame)."""
    x, y, w, h = roi_xywh
    cap = cv2.VideoCapture(video_path)
    tiempos, brillos, areas, visuales, frames_idx = [], [], [], [], []
    centro, centro_ref, fallos, frame_ref, frame_idx = None, None, 0, None, -1
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
        brillos.append(cv2.mean(gris)[0])                          # el brillo global delata el cambio del flash
        roi = frame[y:y + h, x:x + w]
        c_img = np.ascontiguousarray(roi[:, :, canal]) if canal in (0, 1, 2) else np.ascontiguousarray(gris[y:y + h, x:x + w])
        if centro is None:
            centro = _centro_inicial(c_img, par)
            centro_ref = centro.copy()
        res = _medir_pupila(c_img, centro, par)
        if res is not None:
            centro, fallos = np.array(res['centro']), 0
            areas.append(res['area'])
        else:
            fallos += 1
            areas.append(np.nan)
            if fallos >= 10:                                       # se perdió el seguimiento: se vuelve al centro inicial
                centro = centro_ref.copy()
        visuales.append(_recorte_visual(roi, centro, res, par))
        tiempos.append(t)
        frames_idx.append(frame_idx)
    cap.release()
    return tiempos, brillos, areas, visuales, frames_idx, frame_ref

def procesar_modulo_3_2(video_path, t_inicio=0.0, t_fin=7.5, t_estimulo_manual=None,
                        area_min_px=80, area_max_px=900,
                        manual_contraida=None, manual_dilatada=None,
                        anotaciones_manuales=None, roi_manual=None,
                        canal=2, etiqueta="Registro B",
                        mostrar=True, carpeta_salida=None):
    """
    Mide el área pupilar en cada frame y extrae los parámetros de la respuesta al cambio del flash.
      t_inicio, t_fin      ventana de análisis en segundos (t_fin=None: hasta el final)
      t_estimulo_manual    segundo aproximado del estímulo si la detección automática del flash falla
      area_min_px/max_px   rango plausible del área pupilar en el video (px²)
      manual_contraida/dilatada  niveles manuales (px²); si no se pasan se toman de anotaciones_manuales
      anotaciones_manuales {nro_de_frame: área_manual_px} para MAE/RMSE frame a frame
      roi_manual           (x, y, w, h) para fijar la ROI; si es None se elige automáticamente
      canal                2 = rojo, 1 = verde, 0 = azul, None = gris
      mostrar              False: no abre ventanas (varios videos seguidos)
      carpeta_salida       si se indica, guarda la curva (CSV), los parámetros (CSV) y las figuras (PNG)
    Devuelve un dict con los parámetros (sirve para calcular variabilidad entre registros) o None.
    """
    if t_fin is None:
        t_fin = float('inf')
    _titulo(f"--- 3.2 Dinámica pupilar ({etiqueta}) ---")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0

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

    tiempos, brillos, areas, visuales, frames_idx, frame_ref = _medir_video(video_path, par, (x, y, w, h), canal, fps, t_inicio, t_fin)
    if len(areas) >= 10 and np.isnan(areas).mean() > 0.4:          # casi no se mide: se reintenta con umbrales de borde permisivos
        reintento = _medir_video(video_path, _parametros_segmentacion(area_min_px, area_max_px, True), (x, y, w, h), canal, fps, t_inicio, t_fin)
        if np.isnan(reintento[2]).mean() < np.isnan(areas).mean():
            tiempos, brillos, areas, visuales, frames_idx, frame_ref = reintento
            print("Medición repetida con umbrales de borde permisivos (con los normales se detectaba la pupila en pocos frames).")
    if len(areas) < 10:
        print("Muy pocos frames dentro de la ventana de análisis.")
        return None

    t, a_cruda, brillo = np.array(tiempos), np.array(areas, dtype=float), np.array(brillos, dtype=float)
    frac_fallo = float(np.isnan(a_cruda).mean())
    if frac_fallo > 0.4:
        print(f"ADVERTENCIA: la pupila no se detectó en el {100 * frac_fallo:.0f} % de los frames. Lo más probable es que la ROI "
              f"(x={x}, y={y}, {w}x{h}) no esté sobre el ojo: pasá roi_manual=(x, y, w, h) con la ROI del Módulo 1. "
              f"Si la ROI es correcta, revisá area_min_px/area_max_px.")

    # ---- Limpieza y estímulo ----
    falla = False
    a_limpia, a_suave, es_art = _limpiar_serie(a_cruda, fps)
    if a_limpia is None:
        print("ADVERTENCIA: la segmentación falló en casi todos los frames. Revisá la figura de verificación, "
              "la ROI y el rango area_min_px/area_max_px.")
        a_limpia = a_suave = np.full(len(t), np.nan)
        es_art, falla = np.isnan(a_cruda), True
    n_sin_det = int(np.isnan(a_cruda).sum())
    print(f"Frames analizados: {len(t)} | sin detección fiable (parpadeo/oclusión/baja confianza): {n_sin_det} "
          f"| atípicos (Hampel): {int(es_art.sum()) - n_sin_det} | interpolados en total: {int(es_art.sum())} "
          f"({100 * es_art.mean():.1f} %)")

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
    contraida, dilatada = (min(p['basal'], p['extremo']), max(p['basal'], p['extremo'])) if p is not None else (np.nan, np.nan)

    # ---- Validación contra mediciones manuales ----
    # Sin niveles manuales explícitos se toman de las anotaciones: promedio de las previas al estímulo y de las
    # posteriores a la respuesta (la menor es la contraída y la mayor la dilatada).
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
    tabla = []                                                     # (frame, t, manual, auto, era_artefacto)
    if anotaciones_manuales:
        for fr, a_man in sorted(anotaciones_manuales.items()):
            if fr in frames_idx:
                k = frames_idx.index(fr)
                if np.isfinite(a_limpia[k]):
                    tabla.append((fr, t[k], float(a_man), float(a_limpia[k]), bool(es_art[k])))
                else:
                    print(f"AVISO: el frame {fr} no tiene medición automática; no se compara.")
            else:
                print(f"AVISO: el frame {fr} no está en la ventana analizada (frames {frames_idx[0]} a {frames_idx[-1]}); se ignora.")
        if tabla:
            mae_fr, rmse_fr = _metricas_error([(r[3], r[2]) for r in tabla])

    # ---- Figura 1: curva temporal ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(t, a_cruda, '.', color='tab:blue', alpha=0.45, ms=4, label='Área bruta A(t) (frames válidos)')
    if es_art.any():
        ax1.plot(t[es_art], a_limpia[es_art], 'x', color='orange', ms=6, label='Artefacto interpolado')
    ax1.plot(t, a_suave, color='red', lw=2, label='Área suavizada')
    if hay_manual:
        ax1.axhline(manual_contraida, color='gray', ls=':', label=f'Manual contraída ({manual_contraida:.0f} px²)')
        ax1.axhline(manual_dilatada, color='gray', ls='-.', label=f'Manual dilatada ({manual_dilatada:.0f} px²)')
    if tabla:
        ax1.plot([r[1] for r in tabla], [r[2] for r in tabla], 's', color='black', mfc='none', ms=8, mew=1.5,
                 label='Anotaciones manuales (ImageJ)')
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

    # ---- Figura 2: verificación visual de la segmentación ----
    fig2, axs = plt.subplots(2, 4, figsize=(12, 6.5))
    for ax, i in zip(axs.ravel(), np.linspace(0, len(t) - 1, 8).astype(int)):
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

    def mostrar_figuras():
        plt.show() if mostrar else plt.close('all')

    # ---- Reporte ----
    print(f"\n--- Parámetros Fisiológicos Extraídos ({etiqueta}: {tipo}) ---")
    print(f"Tiempo de estímulo: {t_est:.2f} s")
    if p is None:
        print("No se pudieron calcular los parámetros (ver advertencias arriba).")
    else:
        man = lambda v: f"   | manual: {v:.1f} px²" if hay_manual else ""
        print(f"Área contraída (auto): {contraida:.1f} px²{man(manual_contraida) if hay_manual else ''}")
        print(f"Área dilatada  (auto): {dilatada:.1f} px²{man(manual_dilatada) if hay_manual else ''}")
        print(f"Amplitud de respuesta: {p['amplitud']:.1f} px²{man(manual_dilatada - manual_contraida) if hay_manual else ''}")
        print(f"Latencia: {p['latencia']:.2f} s   (estímulo -> inicio de {nombre_resp.lower()})")
        print(f"Tiempo hasta el 95 % de la {nombre_resp.lower()}: {p['t_ext'] - t_est:.2f} s")
        print(f"Velocidad media de {nombre_resp.lower()} (10-90 %): {p['vel_media']:.1f} px²/s")
        print(f"Velocidad pico de {nombre_resp.lower()}: {p['vel_pico']:.1f} px²/s")
    if hay_manual and p is not None:
        print("\n--- Validación vs. medición manual ---")
        if niveles_de_anotaciones:
            print("(niveles manuales tomados de tus anotaciones: promedio de las previas al estímulo y de las posteriores a la respuesta)")
        print(f"Niveles (contraída y dilatada): MAE = {mae_niv:.1f} px² | RMSE = {rmse_niv:.1f} px²")
        for nombre, auto, man_v in (("contraída", contraida, manual_contraida), ("dilatada", dilatada, manual_dilatada)):
            print(f"   {nombre}: error = {auto - man_v:+.1f} px² ({100 * (auto - man_v) / man_v:+.1f} %)")
    if tabla:
        print("\nComparación frame a frame (manual vs. automático):")
        print("  frame    t(s)   manual    auto    error    error %")
        for fr, tt, man_v, auto, art in tabla:
            marca = "  <- frame interpolado (artefacto)" if art else ""
            print(f"  {fr:5d}  {tt:6.2f}  {man_v:7.1f}  {auto:7.1f}  {auto - man_v:+7.1f}  {100 * (auto - man_v) / man_v:+7.1f}{marca}")
        print(f"Frames comparados: {len(tabla)} | MAE = {mae_fr:.1f} px² | RMSE = {rmse_fr:.1f} px²")
    elif anotaciones_manuales:
        print("\nNo se pudo comparar ninguna anotación: ningún frame coincide con la ventana analizada.")

    if p is None:
        mostrar_figuras()
        return None

    resultado = {'etiqueta': etiqueta, 'tipo': tipo, 'fps': fps, 't_estimulo': float(t_est),
                 'area_contraida': contraida, 'area_dilatada': dilatada,
                 'diametro_contraido': float(2 * np.sqrt(contraida / np.pi)), 'diametro_dilatado': float(2 * np.sqrt(dilatada / np.pi)),
                 'amplitud': p['amplitud'], 'latencia': p['latencia'], 't_hasta_extremo': float(p['t_ext'] - t_est),
                 'vel_media': p['vel_media'], 'vel_pico': p['vel_pico'],
                 'frames_artefacto': int(es_art.sum()), 'frames_total': int(len(t)),
                 'mae_niveles': mae_niv, 'rmse_niveles': rmse_niv, 'mae_frames': mae_fr, 'rmse_frames': rmse_fr,
                 'tiempos': t, 'area_cruda': a_cruda, 'area_suave': a_suave}

    if carpeta_salida:
        _escribir_csv(base + "_curva.csv",
                      ["frame", "tiempo_s", "area_cruda_px2", "area_limpia_px2", "area_suave_px2", "diametro_suave_px", "artefacto"],
                      [[frames_idx[k], f"{t[k]:.4f}", "" if np.isnan(a_cruda[k]) else f"{a_cruda[k]:.2f}", f"{a_limpia[k]:.2f}",
                        f"{a_suave[k]:.2f}", f"{2 * np.sqrt(a_suave[k] / np.pi):.3f}", int(es_art[k])] for k in range(len(t))])
        campos = ["etiqueta", "tipo", "fps", "t_estimulo", "area_contraida", "area_dilatada", "diametro_contraido",
                  "diametro_dilatado", "amplitud", "latencia", "t_hasta_extremo", "vel_media", "vel_pico",
                  "frames_artefacto", "frames_total", "mae_niveles", "rmse_niveles"]
        _escribir_csv(base + "_parametros.csv", campos, [[_fmt(resultado[c]) for c in campos]])
        print(f"\nCSV guardados en '{carpeta_salida}': {os.path.basename(base)}_curva.csv y _parametros.csv")
    mostrar_figuras()
    return resultado

def procesar_registros(registros, carpeta_salida='resultados', area_min_px=250, area_max_px=2000):
    """Procesa varios registros sin abrir ventanas, guarda sus CSV y 'resumen_registros.csv', e imprime la variabilidad
    entre registros del mismo tipo. registros: [{'video', 'etiqueta', 't_inicio', 't_fin', 't_estimulo'}, ...]"""
    filas = []
    for reg in registros:
        r = procesar_modulo_3_2(reg['video'], t_inicio=reg.get('t_inicio', 0.0), t_fin=reg.get('t_fin'),
                                t_estimulo_manual=reg.get('t_estimulo'), area_min_px=area_min_px, area_max_px=area_max_px,
                                etiqueta=reg['etiqueta'], mostrar=False, carpeta_salida=carpeta_salida)
        if r is not None:
            filas.append(r)
    if not filas:
        print("Ningún registro pudo procesarse.")
        return None
    os.makedirs(carpeta_salida, exist_ok=True)
    campos = ["etiqueta", "tipo", "t_estimulo", "area_contraida", "area_dilatada", "amplitud", "latencia",
              "t_hasta_extremo", "vel_media", "vel_pico", "frames_artefacto", "frames_total"]
    _escribir_csv(os.path.join(carpeta_salida, 'resumen_registros.csv'), campos, [[_fmt(r[c]) for c in campos] for r in filas])

    _titulo("RESUMEN DE REGISTROS")
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

# ----------------------------------------------------------------------
# Módulo 4: compresión sin pérdida (Huffman, RLE) y con pérdida (JPEG)
# ----------------------------------------------------------------------
def _huffman_longitudes(frecuencias):
    """Longitud del código de Huffman de cada símbolo (heap de grupos)."""
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
    """Códigos canónicos: para decodificar alcanza con guardar las 256 longitudes."""
    orden = sorted(largo.items(), key=lambda kv: (kv[1], kv[0]))
    codigo, previa, tabla = 0, orden[0][1], {}
    for s, l in orden:
        codigo <<= (l - previa)
        previa = l
        tabla[s] = (l, codigo)
        codigo += 1
    return tabla

def _huffman_tamano(img):
    """Bytes del archivo Huffman: bits de datos / 8 más 256 bytes de tabla."""
    frec = np.bincount(img.ravel(), minlength=256)
    return int(np.ceil(sum(int(frec[s]) * l for s, l in _huffman_longitudes(frec).items()) / 8.0)) + 256

def _huffman_verificar(img):
    """Codifica y decodifica para comprobar que no hay pérdida (solo imágenes de hasta 400 000 px)."""
    if img.size > 400_000:
        return None
    flat = img.ravel()
    tabla = _huffman_codigos(_huffman_longitudes(np.bincount(flat, minlength=256)))
    lon, cod = np.zeros(256, np.int64), np.zeros(256, np.uint64)
    for s, (l, c) in tabla.items():
        lon[s], cod[s] = l, c
    lens, vals = lon[flat], cod[flat]
    inicio = np.cumsum(lens) - lens
    sim = np.repeat(np.arange(flat.size), lens)
    pos = np.arange(int(lens.sum())) - inicio[sim]
    bits = ((vals[sim] >> (lens[sim] - 1 - pos).astype(np.uint64)) & np.uint64(1)).astype(np.uint8)
    inversa = {(l, c): s for s, (l, c) in tabla.items()}
    salida = np.empty(flat.size, np.uint8)
    k = val = ln = 0
    for b in bits.tolist():
        val, ln = (val << 1) | b, ln + 1
        s = inversa.get((ln, val))
        if s is not None:
            salida[k] = s
            k, val, ln = k + 1, 0, 0
    return bool(np.array_equal(salida, flat))

def _rle(flat):
    """RLE de pares (valor, contador) de 1 byte; las corridas de más de 255 se parten. Devuelve (bytes, reconstrucción exacta)."""
    inicios = np.concatenate(([0], np.flatnonzero(flat[1:] != flat[:-1]) + 1))
    largos = np.diff(np.concatenate((inicios, [flat.size])))
    trozos = (largos + 254) // 255
    valores = np.repeat(flat[inicios], trozos)
    cuentas = np.full(valores.size, 255, dtype=np.int64)
    cuentas[np.cumsum(trozos) - 1] = largos - 255 * (trozos - 1)
    return 2 * int(valores.size), bool(np.array_equal(np.repeat(valores, cuentas), flat))

def _rle_dos_direcciones(img):
    """RLE recorriendo por filas y por columnas. Devuelve (bytes_h, bytes_v, ok)."""
    h, ok_h = _rle(np.ascontiguousarray(img).ravel())
    v, ok_v = _rle(np.ascontiguousarray(img.T).ravel())
    return h, v, ok_h and ok_v

def _salto_bloque(dec, ref, region=None, bloque=8):
    """
    Discontinuidad que JPEG agrega en las fronteras de los bloques de 8x8 (niveles de gris), medida sobre el error
    e = JPEG - original: salto medio de e entre bloques vecinos menos salto medio dentro de los bloques.
    Con 'region' (máscara booleana) solo cuentan los pares de vecinos que están ambos dentro.
    """
    f = dec.astype(np.float32) - ref.astype(np.float32)
    dh, dv = np.abs(f[:, 1:] - f[:, :-1]), np.abs(f[1:, :] - f[:-1, :])
    jh, jv = (np.arange(dh.shape[1]) % bloque) == bloque - 1, (np.arange(dv.shape[0]) % bloque) == bloque - 1
    rh, rv = (np.ones(dh.shape, bool), np.ones(dv.shape, bool)) if region is None else \
             (region[:, 1:] & region[:, :-1], region[1:, :] & region[:-1, :])
    borde = np.concatenate([dh[jh[None, :] & rh], dv[jv[:, None] & rv]])
    interior = np.concatenate([dh[(~jh)[None, :] & rh], dv[(~jv)[:, None] & rv]])
    return np.nan if borde.size < 50 or interior.size < 50 else float(borde.mean() - interior.mean())

def _actividad(ref, region):
    """Salto medio entre vecinos de la imagen original dentro de una región (textura que enmascara los artefactos)."""
    f = ref.astype(np.float32)
    v = np.concatenate([np.abs(f[:, 1:] - f[:, :-1])[region[:, 1:] & region[:, :-1]],
                        np.abs(f[1:, :] - f[:-1, :])[region[1:, :] & region[:-1, :]]])
    return float(v.mean()) if v.size else np.nan

def _regiones(mascara, roi_gris):
    """Banda de ±3 px sobre el contorno de la pupila y zona uniforme (a más de 10 px de la pupila, con textura local baja)."""
    borde = cv2.morphologyEx(mascara, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    lejos = cv2.dilate(mascara, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) == 0
    f = roi_gris.astype(np.float32)
    media = cv2.blur(f, (5, 5))
    desvio = np.sqrt(np.maximum(cv2.blur(f * f, (5, 5)) - media ** 2, 0))
    return borde, lejos & (desvio <= (np.median(desvio[lejos]) if lejos.any() else np.median(desvio)))

def _jpeg(img, q, color=False):
    """Ida y vuelta por JPEG. Devuelve (imagen decodificada, bytes del archivo)."""
    _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE), len(enc)

def procesar_modulo_4(frame, x, y, w, h, area_min_px=250, area_max_px=2000, canal=2):
    """
    Compara Huffman, RLE y JPEG (Q=90, 50, 10, 5) sobre el frame, la ROI y la máscara de la pupila; mide desde qué Q
    aparecen los bloques de 8x8, dónde se notan más (borde de la pupila o zonas uniformes) y si JPEG cambia el diámetro
    medido con el método del Módulo 3.2. area_min_px/area_max_px: mismo rango que el Módulo 3.2; canal: 2 = rojo.
    """
    _titulo("--- MÓDULO 4: Compresión de la información de imagen ---")
    FACTORES_Q = [90, 50, 10, 5]
    Q_BARRIDO = [95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5]
    UMBRAL_BLOQUES = 0.5                                   # salto extra (niveles de gris) desde el cual los bloques se consideran visibles
    TOL_DIAMETRO = 0.03

    roi = np.ascontiguousarray(frame[y:y + h, x:x + w])
    roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    c_roi = np.ascontiguousarray(roi[:, :, canal])
    res0, (a_min, a_max) = _medir_robusto(c_roi, area_min_px, area_max_px)
    mascara, d_orig = None, np.nan
    if res0 is not None:
        mascara = np.zeros(roi_gris.shape, np.uint8)
        cv2.ellipse(mascara, res0['elipse'], 255, -1)
        d_orig = 2 * np.sqrt(res0['area'] / np.pi)
        print(f"Pupila segmentada en el frame original: área = {res0['area']:.1f} px², diámetro equivalente = {d_orig:.2f} px")
    else:
        print("ADVERTENCIA: no se pudo segmentar la pupila en este frame ni con el rango ampliado ni con la detección "
              "independiente (¿ojo cerrado, ROI fuera del ojo?). Se omiten la ROI segmentada, el análisis por regiones y el diámetro.")

    imagenes = {'Frame original (gris)': frame_gris, 'ROI (gris)': roi_gris}
    if mascara is not None:
        imagenes['ROI segmentada (máscara 0/255)'] = mascara

    # ---- 1) Compresión con y sin pérdida ----
    resultados = {}
    for nombre, img in imagenes.items():
        img = np.ascontiguousarray(img)
        n0 = img.size
        t_huff, ok_huff = _huffman_tamano(img), _huffman_verificar(img)
        t_rh, t_rv, ok_rle = _rle_dos_direcciones(img)
        filas = [dict(metodo=m, bytes=b, ratio=n0 / b, psnr=np.inf, ssim=1.0, ok=ok)
                 for m, b, ok in (('Huffman', t_huff, ok_huff), ('RLE horizontal', t_rh, ok_rle), ('RLE vertical', t_rv, ok_rle))]
        for q in FACTORES_Q:
            dec, nb = _jpeg(img, q)
            filas.append(dict(metodo=f'JPEG Q={q}', bytes=nb, ratio=n0 / nb,
                              psnr=psnr(img, dec, data_range=255), ssim=ssim(img, dec, data_range=255), ok=None))
        resultados[nombre] = filas
        print(f"\n>>> {nombre}  [{img.shape[1]}x{img.shape[0]} px | {n0} bytes sin comprimir]")
        print(f"    {'Método':<16}{'Bytes':>10}{'Ratio':>10}{'PSNR (dB)':>12}{'SSIM':>9}")
        for f in filas:
            p = "   inf (=)" if np.isinf(f['psnr']) else f"{f['psnr']:11.2f}"
            print(f"    {f['metodo']:<16}{f['bytes']:>10d}{f['ratio']:>9.2f}:1{p:>12}{f['ssim']:>9.4f}")
        veri = ([f"Huffman reconstrucción exacta: {'SÍ' if ok_huff else 'NO'}"] if ok_huff is not None else []) + \
               [f"RLE reconstrucción exacta: {'SÍ' if ok_rle else 'NO'}"]
        print("    " + " | ".join(veri))

    # ---- 2) Barrido de Q sobre la ROI ----
    barrido, jpeg_gris = {}, {}
    if mascara is not None:
        borde, uniforme = _regiones(mascara, roi_gris)
        act_b, act_u = _actividad(roi_gris, borde), _actividad(roi_gris, uniforme)
        par_n = _parametros_segmentacion(a_min, a_max)
        par_l = _parametros_segmentacion(a_min, a_max, True)
    for q in Q_BARRIDO:
        dec = jpeg_gris[q] = _jpeg(roi_gris, q)[0]
        fila = {'sb': _salto_bloque(dec, roi_gris)}
        if mascara is not None:
            err = np.abs(roi_gris.astype(np.float32) - dec.astype(np.float32))
            fila.update(mae_b=float(err[borde].mean()), mae_u=float(err[uniforme].mean()),
                        sb_b=_salto_bloque(dec, roi_gris, borde), sb_u=_salto_bloque(dec, roi_gris, uniforme))
            c_q = np.ascontiguousarray(_jpeg(roi, q, True)[0][:, :, canal])
            rq = _medir_pupila(c_q, np.array(res0['centro']), par_n) or _medir_pupila(c_q, np.array(res0['centro']), par_l)
            fila['area'] = rq['area'] if rq is not None else np.nan
            fila['diam'] = 2 * np.sqrt(rq['area'] / np.pi) if rq is not None else np.nan
        barrido[q] = fila

    # ---- Figura 1: PSNR y SSIM contra ratio ----
    fig1, (axp, axs) = plt.subplots(1, 2, figsize=(14, 5.5))
    marcas = {'Huffman': 's', 'RLE horizontal': '^', 'RLE vertical': 'v'}
    tope = max(f['psnr'] for fs in resultados.values() for f in fs if np.isfinite(f['psnr'])) + 8
    for (nombre, filas), col in zip(resultados.items(), ['tab:blue', 'tab:orange', 'tab:green']):
        jp = [f for f in filas if f['metodo'].startswith('JPEG')]
        for ax, k in ((axp, 'psnr'), (axs, 'ssim')):
            ax.plot([f['ratio'] for f in jp], [f[k] for f in jp], '-o', color=col, label=f'JPEG - {nombre}')
            for f in jp:
                ax.annotate(f"Q{f['metodo'].split('=')[1]}", (f['ratio'], f[k]), textcoords='offset points', xytext=(4, 5), fontsize=7)
        for f in filas:
            if f['metodo'] in marcas:
                axp.plot(f['ratio'], tope, marker=marcas[f['metodo']], color=col, mfc='none', ls='')
                axs.plot(f['ratio'], 1.0, marker=marcas[f['metodo']], color=col, mfc='none', ls='')
    for m, mk in marcas.items():
        axp.plot([], [], marker=mk, color='k', mfc='none', ls='', label=f'{m} (sin pérdida)')
    axp.axhline(tope, color='gray', ls=':')
    axp.set_ylim(top=tope + 4)
    axp.text(0.02, 0.97, 'línea punteada: métodos sin pérdida (PSNR = ∞)', transform=axp.transAxes, va='top', fontsize=8, color='gray')
    for ax, t_, yl in ((axp, 'PSNR vs. ratio', 'PSNR (dB)'), (axs, 'SSIM vs. ratio', 'SSIM')):
        ax.set_xscale('log'); ax.set_xlabel('Ratio de compresión (log)'); ax.set_ylabel(yl); ax.set_title(t_)
        ax.grid(True, which='both', alpha=0.4)
    axp.legend(fontsize=7, loc='lower left')
    plt.tight_layout()

    # ---- Figura 2: artefactos de bloque ----
    sl = (slice(None), slice(None))
    if mascara is not None:
        cy, cx = int(res0['centro'][1]), int(res0['centro'][0])
        hz = int(min(max(24, 2.4 * np.sqrt(res0['area'] / np.pi)), min(roi_gris.shape) // 2 - 1))
        sl = (slice(max(0, cy - hz), max(0, cy - hz) + 2 * hz), slice(max(0, cx - hz), max(0, cx - hz) + 2 * hz))
    fig2, axs2 = plt.subplots(3, 5, figsize=(17, 10))
    for j, (titulo, im) in enumerate([('Original', roi_gris)] + [(f'JPEG Q={q}', jpeg_gris[q]) for q in FACTORES_Q]):
        axs2[0, j].imshow(im, cmap='gray', vmin=0, vmax=255); axs2[0, j].set_title(titulo)
        axs2[1, j].imshow(im[sl], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        axs2[1, j].set_title('Zoom borde pupila + iris' if j == 0 else titulo + ' (zoom)')
        if j:
            err = np.clip(np.abs(roi_gris.astype(np.float32) - im.astype(np.float32)) * 8, 0, 255)
            axs2[2, j].imshow(err, cmap='magma', vmin=0, vmax=255); axs2[2, j].set_title(f'|Error| x8  (Q={FACTORES_Q[j - 1]})')
        elif mascara is not None:
            vis = np.zeros(roi_gris.shape + (3,), np.uint8) + (roi_gris[..., None] // 2).astype(np.uint8)
            vis[borde], vis[uniforme] = (255, 0, 0), (0, 120, 255)
            axs2[2, 0].imshow(vis); axs2[2, 0].set_title('Regiones (rojo: borde, azul: uniforme)', fontsize=9)
    for a in axs2.ravel():
        a.axis('off')
    plt.suptitle('Artefactos de bloque JPEG: ROI completa (arriba), zoom (centro) y mapa de error (abajo)', fontsize=13)
    plt.tight_layout()

    # ---- Figura 3: análisis cuantitativo ----
    qs = sorted(barrido, reverse=True)
    n_graf = 3 if mascara is not None else 1
    ax3 = np.atleast_1d(plt.subplots(1, n_graf, figsize=(5.2 * n_graf, 4.6))[1])
    serie = lambda k: [barrido[q][k] for q in qs]
    ax3[0].plot(qs, serie('sb'), '-o', label='ROI completa', color='k')
    if mascara is not None:
        ax3[0].plot(qs, serie('sb_u'), '-o', label='Zona uniforme', color='tab:blue')
        ax3[0].plot(qs, serie('sb_b'), '-o', label='Borde pupila', color='tab:red')
    ax3[0].axhline(UMBRAL_BLOQUES, color='gray', ls='--', label=f'Umbral visible ({UMBRAL_BLOQUES:g})')
    ax3[0].set_title('Salto extra entre bloques 8x8 vs. Q'); ax3[0].set_ylabel('Salto extra (niveles de gris)')
    if mascara is not None:
        ax3[1].plot(qs, serie('mae_b'), '-o', color='tab:red', label='Borde pupila')
        ax3[1].plot(qs, serie('mae_u'), '-o', color='tab:blue', label='Zona uniforme')
        ax3[1].set_title('Error absoluto medio vs. Q'); ax3[1].set_ylabel('MAE (niveles de gris)')
        ax3[2].plot(qs, serie('diam'), '-o', color='tab:green', label='Diámetro con JPEG')
        ax3[2].axhline(d_orig, color='k', ls='--', label='Original')
        ax3[2].axhspan(d_orig * (1 - TOL_DIAMETRO), d_orig * (1 + TOL_DIAMETRO), color='gray', alpha=0.2, label=f'±{100 * TOL_DIAMETRO:.0f} %')
        ax3[2].set_title('Diámetro pupilar (Módulo 3) vs. Q'); ax3[2].set_ylabel('Diámetro equivalente (px)')
    for a in ax3:
        a.set_xlabel('Q'); a.invert_xaxis(); a.legend(fontsize=8); a.grid(True)
    plt.tight_layout()
    plt.show()

    # ---- 3) Respuestas a la consigna ----
    _titulo("RESPUESTAS A LAS PREGUNTAS DE LA CONSIGNA CON UN VIDEO GENÉRICO")
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

    print("\n[2] ¿Desde qué factor de calidad aparecen los artefactos de bloque?")
    ref = 'sb_u' if mascara is not None else 'sb'
    donde = 'zonas uniformes' if mascara is not None else 'toda la ROI'
    q_bloque = next((q for q in qs if not np.isnan(barrido[q][ref]) and barrido[q][ref] >= UMBRAL_BLOQUES), None)
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
                print(f"   Q={q:>2}: área = {a:7.1f} px² | diámetro = {d:6.2f} px | Δ = {d - d_orig:+.2f} px ({100 * rel:+.1f} %) -> "
                      f"{'SÍ cambia' if abs(rel) > TOL_DIAMETRO else 'no cambia significativamente'} (umbral ±{100 * TOL_DIAMETRO:.0f} %)")
        cambian = [q for q in (50, 10) if np.isnan(barrido[q]['diam']) or abs(barrido[q]['diam'] - d_orig) / d_orig > TOL_DIAMETRO]
        if cambian:
            print(f"   -> Respuesta: SÍ, a Q={' y Q='.join(map(str, cambian))} el diámetro estimado se aparta más de "
                  f"{100 * TOL_DIAMETRO:.0f} % del original.")
        else:
            print(f"   -> Respuesta: NO. A Q=50 y Q=10 el diámetro estimado queda dentro de ±{100 * TOL_DIAMETRO:.0f} % "
                  f"del original: el método basado en gradiente del borde es robusto a la compresión moderada.")
        print("   Nota: la comparación usa un único frame; para generalizar, repetilo en varios frames del video.")

    _respuesta_video("--- RESPUESTAS APLICADAS AL MÓDULO 4: Compresión de Nuestro Video ---", 'compresion')
    return {'resultados': resultados, 'barrido': barrido, 'q_bloque': q_bloque, 'diametro_original': d_orig}

def exportar_frames_para_medir(video_path, tiempos=(0.5, 1.5, 2.5, 3.5, 4.4, 5.2, 5.5, 5.8, 6.2, 6.8, 7.2, 7.4), carpeta='frames_manual'):
    """Guarda en 'carpeta' los frames de los instantes dados (mismo índice que usa el 3.2) para medirlos en ImageJ."""
    os.makedirs(carpeta, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    objetivo = {int(round(t * fps)): t for t in tiempos}
    idx = -1
    while idx < max(objetivo):
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

def obtener_o_anotar_manual_3_2(video_path, x, y, w, h, ruta_csv, tiempos=(1.0, 3.0, 4.4, 5.2, 6.0, 7.4), forzar_anotacion=False):
    """
    Anotaciones manuales del 3.2 como {frame: área px²}. Sin forzar_anotacion se leen de ruta_csv (None si no existe);
    con forzar_anotacion se abre la ventana de clics en los frames de 'tiempos' y se guarda el resultado en ruta_csv.
    """
    if not forzar_anotacion:
        if not os.path.exists(ruta_csv):
            return None
        with open(ruta_csv, encoding='utf-8') as f:
            anotaciones = {int(r['frame']): float(r['area_px2']) for r in csv.DictReader(f)}
        print(f"Anotaciones manuales 3.2 cargadas desde '{ruta_csv}' ({len(anotaciones)} frames).")
        return anotaciones

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    objetivos = sorted({int(round(t * fps)) for t in tiempos})
    anotaciones = {}
    print("\n--- ANOTACIÓN INTERACTIVA PARA EL MÓDULO 3.2 ---")
    print("Instrucciones: Clic izquierdo = marcar borde (mín. 8 puntos) | z = deshacer | Enter = confirmar | Esc = saltear")
    temp_png = os.path.join(tempfile.gettempdir(), "_temp_mask_3_2.png")
    idx = -1
    while idx < objetivos[-1]:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx in objetivos:
            print(f"\n-> Anotando Frame {idx} (t = {idx / fps:.2f} s)...")
            mascara = anotar_contorno_manual(frame[y:y + h, x:x + w], ruta_salida=temp_png, forma='elipse', escala=2)
            if mascara is not None:
                anotaciones[idx] = float((mascara > 0).sum())
                print(f"   Frame {idx} medido: Área = {anotaciones[idx]:.1f} px²")
    cap.release()
    if os.path.exists(temp_png):
        os.remove(temp_png)
    if anotaciones:
        _escribir_csv(ruta_csv, ['frame', 'tiempo_s', 'area_px2'], [[fr, f"{fr / fps:.2f}", f"{ar:.1f}"] for fr, ar in sorted(anotaciones.items())])
        print(f"\n¡Listo! Anotaciones guardadas en '{ruta_csv}'.")
    return anotaciones

def tiempos_automaticos(video_path, margen=0.05):
    """Instantes (s) a medir a mano: entre 4 y 10, repartidos parejo según la duración del video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    cap.release()
    n = int(np.clip(round(dur * 0.8), 4, 10))
    return tuple(float(t) for t in np.round(np.linspace(margen * dur, (1 - margen) * dur, n), 2))

def estimar_rango_area(video_path, n_muestras=15, fallback=(250, 2000)):
    """
    Estima (area_min, area_max) en px² a partir del área de la pupila en frames repartidos por el video. Se usa la elipse
    medida y, si no hay suficientes, el área del círculo del detector. El máximo no pasa del que admite la ROI
    (radio <= lado/6). Si no se ve la pupila en el video se devuelve 'fallback'.
    """
    areas, lado = [], None
    for laxo in (False, True):
        ms = _muestrear_pupila(video_path, n_muestras, laxo)
        lado = lado or (ms[-1]['lado'] if ms else None)
        for clave in ('area', 'area_det'):
            areas = [m[clave] for m in ms if m[clave] is not None and (clave == 'area_det' or m['valida'])]
            if len(areas) >= 3:
                break
        if len(areas) >= 3:
            break
    if len(areas) < 3:
        print(f"No se pudo estimar el rango de área; se usa {fallback}.")
        return fallback
    tope = np.pi * (lado / 6) ** 2
    a_min, a_max = int(max(0.5 * np.percentile(areas, 5), 30)), int(min(2.0 * np.percentile(areas, 95), tope))
    if a_max <= a_min:
        a_max = int(min(max(2 * a_min, fallback[1]), tope))
    print(f"Rango de área estimado automáticamente: {a_min}-{a_max} px² ({len(areas)} frames muestreados)")
    return a_min, a_max

def main():
    mostrar_instrucciones_interactivas()
    # ======================= CONFIGURACIÓN (lo único que se edita) =======================
    video_original = 'prueba2_flashon_off.mp4'
    t_inicio, t_fin = 0.0, None                  # ventana del 3.2 en segundos (t_fin = None -> hasta el final)
    area_min, area_max = estimar_rango_area(video_original)
    anotar_referencia = True                     # True = abrir ventana para dibujar la pupila del 2.3 (una vez por video)
    anotar_frames_3_2 = True                     # True = abrir ventana para medir frames del 3.2 (una vez por video)
    tiempos_a_medir_3_2 = tiempos_automaticos(video_original)
    roi_forzada = None                           # opcional: (x, y, w, h) para forzar la ROI si la automática no cae sobre el ojo
    exportar_roi_para_imagej = False             # True = guarda la ROI para medir en ImageJ
    carpeta_salida = 'resultados'                # CSV y figuras del 3.2
    registros = []                               # opcional (variabilidad): [{'video': 'A1.mp4', 'etiqueta': 'A1'}, ...]
    # =====================================================================================

    nombre_video, extension = os.path.splitext(os.path.basename(video_original))
    video_trabajo = f"{nombre_video}_copia{extension}"
    ref_pupila = f"referencia_{nombre_video}.png"
    csv_manual_3_2 = f"anotaciones_3_2_{nombre_video}.csv"
    roi_imagej = f"roi_para_anotar_{nombre_video}.png"
    nombre_pdf = f"Respuestas_{nombre_video}.pdf"

    if not os.path.exists(video_original):
        print(f"No se encontró el video '{video_original}'.")
        return

    capturador = CapturadorPDF()
    capturador.activar()
    print(f"VIDEO ANALIZADO: {video_original}")
    if not os.path.exists(video_trabajo):
        shutil.copy(video_original, video_trabajo)

    frame_representativo = obtener_mejor_frame(video_trabajo)
    if frame_representativo is None:
        print("No se pudo obtener un frame representativo del video.")
        capturador.desactivar()
        capturador.borrar_imagenes()
        return
    x, y, w, h = roi_forzada or obtener_roi_automatica(frame_representativo, video_path=video_original)
    roi = frame_representativo[y:y + h, x:x + w]
    area_min, area_max = _validar_rango(roi, area_min, area_max)
    if exportar_roi_para_imagej:
        guardar_roi_para_anotar(roi, roi_imagej, ref_pupila)

    procesar_modulo_1_1(frame_representativo, x, y, w, h)
    procesar_modulo_1_2(frame_representativo, x, y, w, h, area_min, area_max)
    procesar_modulo_1_2_4(roi, area_min, area_max)
    procesar_modulo_1_3(frame_representativo, x, y, w, h, area_min_px=area_min, area_max_px=area_max)

    mascara_limpia, mascaras_m2 = procesar_modulo_2(roi)
    if anotar_referencia:
        with capturador.ventanas_reales():
            anotar_contorno_manual(roi, ref_pupila)
    procesar_modulo_2_3(roi, mascaras_m2, ref_pupila, area_min, area_max)

    procesar_modulo_3_1(roi, mascara_limpia, area_min_px=area_min, area_max_px=area_max)

    with capturador.ventanas_reales():
        anotaciones = obtener_o_anotar_manual_3_2(video_trabajo, x, y, w, h, ruta_csv=csv_manual_3_2,
                                                  tiempos=tiempos_a_medir_3_2, forzar_anotacion=anotar_frames_3_2)
    if not anotaciones:
        print(f"(No existe '{csv_manual_3_2}' y anotar_frames_3_2=False: el 3.2 se hace sin validación manual.)")
    procesar_modulo_3_2(video_trabajo, t_inicio=t_inicio, t_fin=t_fin, anotaciones_manuales=anotaciones,
                        area_min_px=area_min, area_max_px=area_max, roi_manual=(x, y, w, h),
                        etiqueta=nombre_video, carpeta_salida=carpeta_salida)
    if registros:
        procesar_registros(registros, carpeta_salida, area_min, area_max)

    procesar_modulo_4(frame_representativo, x, y, w, h, area_min_px=area_min, area_max_px=area_max)

    capturador.desactivar()
    capturador.generar_pdf(nombre_pdf, video=video_original)
    capturador.borrar_imagenes()
    print(f"\n¡Archivo '{nombre_pdf}' generado con éxito con todos los resultados!")

if __name__ == '__main__':
    main()
