import cv2 #procesar imágenes y video
import numpy as np #números y matrices

class MouseDetector:
    """
    Archivo detector.py 
    Esta clase usa visión por computadora para encontrar ratones en vídeos.
    Combina sustracción de fondo inteligente con flujo óptico para detectar dónde está el ratón y hacia dónde se mueve. 
    """

    
    def __init__(self, metodo='MOG2', tasa_aprendizaje=0.005, historial=500, umbral_varianza=25, 
                 area_minima=None, area_maxima=None, tam_frame=None):
        """
        Después de probar muchos parámetros, estos valores me dieron los mejores resultados.

        Args:
            metodo: uso de MOG2 porque funcionó mejor que KNN en las pruebas
            tasa_aprendizaje: controla cuánto "aprende" el algoritmo del fondo
            historial: cuántos frames usa para recordar el fondo
            umbral_varianza: para distinguir qué es movimiento real
            area_minima: para filtrar objetos muy pequeños (como ruido)
            area_maxima: para ignorar objetos demasiado grandes
            tam_frame: si lo paso, calculo automáticamente las áreas óptimas
        """

        # Elijo el método de sustracción de fondo
        # Después de probar ambos, MOG2 me dio mejores resultados con los vídeos de ratones
        if metodo == 'MOG2':
            self.sustractor_fondo = cv2.createBackgroundSubtractorMOG2(
                history=historial,
                varThreshold=umbral_varianza,
                detectShadows=True
            )
        else:
            self.sustractor_fondo = cv2.createBackgroundSubtractorKNN(
                history=historial,
                dist2Threshold=umbral_varianza,
                detectShadows=True
            )
        
        # Guardo estos valores para usarlos después en el procesamiento
        self.tasa_aprendizaje = tasa_aprendizaje
        self.contador_frames = 0  # para saber cuántos frames he procesado
        self.metodo = metodo
        
        # Aquí calculo automáticamente las áreas si me dan el tamaño del frame
        # Los porcentajes (0.16% y 3.9%) los saqué de probar con los vídeos
        if tam_frame is not None and (area_minima is None or area_maxima is None):
            area_frame = tam_frame[0] * tam_frame[1]
            self.area_minima = int(area_frame * 0.0016) if area_minima is None else area_minima #0.16% del frame
            self.area_maxima = int(area_frame * 0.039) if area_maxima is None else area_maxima
        else:
            self.area_minima = area_minima if area_minima is not None else 500 #valores por defecto
            self.area_maxima = area_maxima if area_maxima is not None else 12000
        
        # Configuración de los kernels para limpiar la imagen después de la sustracción
        # Estos valores los ajusté probando hasta que quedó bien limpio
        self.tam_kernel_apertura = (2, 2)  # pequeño para quitar ruido fino
        self.tam_kernel_cierre = (7, 7)    # más grande para tapar huecos
        self.iteraciones_cierre = 3        # repetir 3 veces para mejores resultados
        
        # Variables para el flujo óptico 
        self.gris_previo = None        # guardo el frame anterior para comparar
        self.flujo_optico = None        # info de movimiento
        self.magnitud_flujo = 0        # qué tan rápido se mueve en promedio
        self.angulo_flujo = 0          # hacia dónde se mueve en promedio


    def preprocesar_frame(self, frame):
        """
        Preparación de la imagen antes de procesarla.
        Primero la convierto a gris porque el color no me importa para detectar movimiento,y luego aplico un filtro para reducir el ruido.
        
        Args:
            frame: frame de video en formato BGR (blue-green-red)
            
        Returns:
            np.ndarray: frame procesado en escala de grises con suavizado Gaussiano
        """
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convertir a escala de grises
        # El filtro Gaussiano 5x5 = mejores resultados
        difuminado = cv2.GaussianBlur(gris, (5, 5), 0) #suavizado gaussiano
        return difuminado


    def detectar_raton(self, frame):
        """
        Combino sustracción de fondo + flujo óptico.
        
        Args:
            frame: frame de vídeo preprocesado (escala de grises)
            
        Returns:
            tuple: (centroide, contorno, orientacion, mascara_binaria)
                - centroide: tupla (x, y) del centroide del ratón, o None
                - contorno: contorno del ratón detectado, o None
                - orientacion: ángulo de orientación en grados, o None
                - mascara_binaria: máscara binaria (ratón=255=blanco, fondo=0=negro)
        """
        self.contador_frames += 1  # cuenta de frames procesados
        
        # Si tengo un frame anterior, calculo el flujo óptico
        # Esto me ayuda a saber hacia dónde se mueve el ratón
        if self.gris_previo is not None:
            self.calcular_flujo_optico(self.gris_previo, frame)
        
        self.gris_previo = frame.copy()
        
        # Aplicación de sustracción de fondo
        # La tasa de aprendizaje está ajustada a los vídeos (0.005)
        mascara_fg = self.sustractor_fondo.apply(frame, learningRate=self.tasa_aprendizaje)
        mascara_fg_binaria = cv2.threshold(mascara_fg, 200, 255, cv2.THRESH_BINARY)[1]
        
        # Operaciones morfológicas 
        # El kernel de apertura elimina ruidos pequeños, el de cierre rellena huecos
        # Los tamaños los probé uno por uno hasta que funcionó bien con los vídeos
        kernel_apertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.tam_kernel_apertura) #elimina ruidos pequeños
        kernel_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.tam_kernel_cierre) #rellena huecos
        
        mascara_fg_binaria = cv2.morphologyEx(mascara_fg_binaria, cv2.MORPH_OPEN, kernel_apertura)
        mascara_fg_binaria = cv2.morphologyEx(mascara_fg_binaria, cv2.MORPH_CLOSE, kernel_cierre, 
                                         iterations=self.iteraciones_cierre)
        
        mascara_binaria = mascara_fg_binaria.copy()

        # Ahora busco los contornos del ratón en la máscara binaria
        contornos, _ = cv2.findContours(mascara_fg_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtro los contornos por área para eliminar ruidos y objetos que no son ratones
        contornos_validos = [cnt for cnt in contornos if self.area_minima < cv2.contourArea(cnt) < self.area_maxima]
        
        if not contornos_validos:
            return None, None, None, mascara_binaria

        # Selecciono el contorno más grande == ratón
        contorno_raton = max(contornos_validos, key=cv2.contourArea)
        
        # Calculo el centroide usando momentos del contorno - ¡matemáticas aplicadas!
        M = cv2.moments(contorno_raton)
        if M["m00"] == 0:
            return None, None, None, mascara_binaria
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Calculo la orientación. Primero intento usar el flujo óptico, si no tengo uso PCA (Análisis de Componentes Principales)
        # El flujo óptico me da orientaciones más suaves, pero PCA es más preciso cuando el ratón está quieto
        if self.flujo_optico is not None:
            orientacion = self.angulo_flujo
        else:
            orientacion = self._calcular_orientacion(contorno_raton)
        
        return (cx, cy), contorno_raton, orientacion, mascara_binaria

    def _calcular_orientacion(self, contorno):
        """
        Uso PCA (Análisis de Componentes Principales). Es como encontrar la "dirección principal" del contorno del ratón.
        
        Args:
            contorno: contorno del ratón detectado
            
        Returns:
            float: ángulo de orientación en grados (-180 a 180), o None si el cálculo falla
        """
        try:
            # Convierto el contorno a puntos y hago SVD para encontrar la dirección principal
            # ¡Esto es álgebra lineal aplicada a visión por computadora!
            pts = np.float32(contorno).reshape(-1, 2)
            _, _, vh = np.linalg.svd(pts - np.mean(pts, axis=0))
            direccion = vh[0]
            angulo = np.degrees(np.arctan2(direccion[1], direccion[0]))
            return angulo
        except:
            return None
    
    def calcular_flujo_optico(self, gris_previo, gris_actual):
        """
        Calculo el movimiento entre frames consecutivos usando el algoritmo Farneback. 
        Los parámetros fueron ajustados tras la lectura de papers y pruebas 
        
        Args:
            gris_previo: frame anterior en escala de grises
            gris_actual: frame actual en escala de grises
        """
        try:
            # Algoritmo Farneback. Parámetros optimizados para los vídeos
            # pyr_scale=0.5 y levels=3 me dieron el mejor balance entre velocidad y precisión
            flujo = cv2.calcOpticalFlowFarneback(
                gris_previo, gris_actual, 
                None,
                pyr_scale=0.5,  # escala de pirámide - probé 0.3, 0.5 y 0.7
                levels=3,       # niveles de pirámide - 3 buena resolución
                winsize=15,     # tamaño de ventana - 15 funciona bien 
                iterations=3,   # iteraciones - más de 3 = más lento
                poly_n=5,       # tamaño del polígono para expansión
                poly_sigma=1.2, # desviación estándar del suavizado
                flags=0
            )
            
            self.flujo_optico = flujo
            # Convierto a coordenadas polares para obtener magnitud y ángulo promedio
            mag, ang = cv2.cartToPolar(flujo[..., 0], flujo[..., 1])
            self.magnitud_flujo = np.mean(mag)
            self.angulo_flujo = np.degrees(np.mean(ang))
            
        except Exception as e:
            self.flujo_optico = None
            self.magnitud_flujo = 0
            self.angulo_flujo = 0
    
    def obtener_visualizacion_flujo_optico(self, forma_frame):
        """
        Esta función crea visualizaciones del flujo óptico. Convierto el flujoen una imagen HSV donde el color representa la dirección y el brillo la velocidad.

        
        Args:
            forma_frame: tupla (alto, ancho) con las dimensiones del frame
            
        Returns:
            np.ndarray: imagen BGR con la visualización del flujo óptico
                       (Hue=dirección, S=saturación, Value=magnitud)
        """
        if self.flujo_optico is None:
            return np.zeros((forma_frame[0], forma_frame[1], 3), dtype=np.uint8)
        
        mag, ang = cv2.cartToPolar(self.flujo_optico[..., 0], self.flujo_optico[..., 1])
        
        # Los colores me ayudan a entender hacia dónde se mueve el ratón
        hsv = np.zeros((forma_frame[0], forma_frame[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2  # Hue: dirección (0-180 grados)
        hsv[..., 1] = 255  # Saturación máxima para colores vivos
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitud
        
        vis_flujo = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return vis_flujo


