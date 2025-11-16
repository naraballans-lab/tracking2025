import cv2
import numpy as np

class MouseDetector:
    def __init__(self, method='MOG2', learning_rate=0.005, history=500, var_threshold=25):
        """
        Inicializa el detector de ratones
        Args:
            method: Método de sustracción de fondo ('MOG2' o 'KNN')
            learning_rate: Tasa de aprendizaje para el sustractor de fondo
            history: Número de frames usados para modelar el fondo
            var_threshold: Umbral de varianza para la detección de primer plano
        """
        # Configurar el sustractor de fondo según el método elegido
        if method == 'MOG2':
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=True  # Activamos detección de sombras para mejor segmentación
            )
        else:
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=history,
                dist2Threshold=var_threshold,
                detectShadows=True
            )
        
        self.learning_rate = learning_rate
        self.frame_count = 0
        self.method = method
        
        # Parámetros optimizados para ratón blanco en contenedor con fondo oscuro
        self.min_area = 500  # Área mínima ajustada para el tamaño del ratón en el contenedor
        self.max_area = 12000  # Área máxima ajustada para evitar detecciones de todo el contenedor
        
        # Parámetros optimizados para operaciones morfológicas (evitar fragmentación)
        self.morph_kernel_open_size = (2, 2)  # Kernel pequeño - eliminar ruido mínimo
        self.morph_kernel_close_size = (7, 7)  # Kernel GRANDE - unir fragmentos del ratón
        self.morph_close_iterations = 3  # MÁS iteraciones para cerrar huecos internos
        
        # Para flujo óptico
        self.prev_gray = None
        self.optical_flow = None
        self.flow_magnitude = 0
        self.flow_angle = 0
        
        # Para suavizado temporal de la máscara
        self.mask_history = []
        self.max_mask_history = 3

    def preprocess_frame(self, frame):
        """
        Preprocesa el frame para detección
        Args:
            frame: Frame de video en BGR
        Returns:
            Frame procesado en escala de grises
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Aplicar suavizado Gaussiano
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred

    def detect_mouse(self, frame, debug=False):
        """
        Detecta el ratón en el frame usando sustracción de fondo adaptativa
        Args:
            frame: Frame de video preprocesado (escala de grises)
            debug: Si es True, retorna frame con visualizaciones de debug
        Returns:
            centroid: Tuple (x, y) del centroide del ratón
            contour: Contorno del ratón
            orientation: Ángulo de orientación en grados
            binary_mask: Máscara binaria (ratón blanco, fondo negro)
            debug_frame: Frame con visualizaciones si debug=True
        """
        self.frame_count += 1
        debug_frame = None
        
        # Calcular flujo óptico si hay frame anterior
        if self.prev_gray is not None:
            self.calculate_optical_flow(self.prev_gray, frame)
        
        # Guardar frame actual para el próximo cálculo de flujo óptico
        self.prev_gray = frame.copy()
        
        # Aplicar sustracción de fondo con learning rate adaptativo
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
        
        # Separar la máscara de primer plano y sombras (valores 255 y 127 respectivamente)
        fg_mask_binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
        
        # Operaciones morfológicas optimizadas para limpiar la máscara
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.morph_kernel_open_size)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.morph_kernel_close_size)
        
        # Limpieza de ruido y mejora de la máscara con parámetros optimizados
        fg_mask_binary = cv2.morphologyEx(fg_mask_binary, cv2.MORPH_OPEN, kernel_open)  # Eliminar ruido pequeño
        fg_mask_binary = cv2.morphologyEx(fg_mask_binary, cv2.MORPH_CLOSE, kernel_close, iterations=self.morph_close_iterations)  # Cerrar huecos y mejorar continuidad
        
        # Esta es la máscara binaria final: ratón blanco (255), fondo negro (0)
        binary_mask = fg_mask_binary.copy()

        # Encontrar contornos
        contours, _ = cv2.findContours(fg_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por área
        valid_contours = [cnt for cnt in contours if self.min_area < cv2.contourArea(cnt) < self.max_area]
        
        if not valid_contours:
            if debug:
                debug_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            return None, None, None, binary_mask, debug_frame

        # Seleccionar el contorno más grande (asumimos que es el ratón)
        mouse_contour = max(valid_contours, key=cv2.contourArea)
        
        # Calcular centroide usando momentos
        M = cv2.moments(mouse_contour)
        if M["m00"] == 0:
            if debug:
                debug_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            return None, None, None, binary_mask, debug_frame
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Calcular orientación usando flujo óptico si está disponible
        if self.optical_flow is not None:
            orientation = self.flow_angle
        else:
            orientation = self._calculate_orientation(mouse_contour)
        
        if debug:
            # Dibujar contorno del ratón y su centroide
            debug_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(debug_frame, [mouse_contour], -1, (0, 255, 0), 2)
            cv2.circle(debug_frame, (cx, cy), 5, (255, 0, 0), -1)
            
            # Dibujar flecha de orientación si está disponible
            if orientation is not None:
                length = 50
                end_x = int(cx + length * np.cos(np.radians(orientation)))
                end_y = int(cy + length * np.sin(np.radians(orientation)))
                cv2.arrowedLine(debug_frame, (cx, cy), (end_x, end_y), 
                              (0, 0, 255), 2, tipLength=0.3)
            
            # Añadir información de texto
            cv2.putText(debug_frame, f"Frame: {self.frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Area: {cv2.contourArea(mouse_contour):.0f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if orientation is not None:
                cv2.putText(debug_frame, f"Angle: {orientation:.1f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Flow Mag: {self.flow_magnitude:.1f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return (cx, cy), mouse_contour, orientation, binary_mask, debug_frame

    def _calculate_orientation(self, contour):
        """
        Calcula la orientación del ratón usando PCA
        Args:
            contour: Contorno del ratón
        Returns:
            angle: Ángulo en grados (-180 a 180)
        """
        try:
            # Convertir contorno a array de puntos
            pts = np.float32(contour).reshape(-1, 2)
            
            # Calcular PCA
            _, _, vh = np.linalg.svd(pts - np.mean(pts, axis=0))
            
            # El primer componente principal es la dirección del eje mayor
            direction = vh[0]
            angle = np.degrees(np.arctan2(direction[1], direction[0]))
            
            return angle
        except:
            return None
    
    def calculate_optical_flow(self, prev_gray, curr_gray):
        """
        Calcula el flujo óptico entre dos frames consecutivos
        Args:
            prev_gray: Frame anterior en escala de grises
            curr_gray: Frame actual en escala de grises
        """
        try:
            # Calcular flujo óptico denso usando Farneback
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, 
                None,
                pyr_scale=0.5,      # Factor de escala de pirámide
                levels=3,            # Número de niveles de pirámide
                winsize=15,          # Tamaño de ventana para promediado
                iterations=3,        # Número de iteraciones
                poly_n=5,            # Tamaño del píxel vecindario
                poly_sigma=1.2,      # Desviación estándar gaussiana
                flags=0
            )
            
            self.optical_flow = flow
            
            # Calcular magnitud y ángulo del flujo
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Promediar la magnitud y ángulo en toda la imagen
            self.flow_magnitude = np.mean(mag)
            self.flow_angle = np.degrees(np.mean(ang))
            
        except Exception as e:
            self.optical_flow = None
            self.flow_magnitude = 0
            self.flow_angle = 0
    
    def get_optical_flow_visualization(self, frame_shape):
        """
        Crea una visualización del flujo óptico
        Args:
            frame_shape: Forma del frame (height, width)
        Returns:
            flow_vis: Imagen de visualización del flujo óptico en formato HSV
        """
        if self.optical_flow is None:
            return np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
        
        # Crear visualización HSV del flujo óptico
        mag, ang = cv2.cartToPolar(self.optical_flow[..., 0], self.optical_flow[..., 1])
        
        hsv = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2  # Hue: dirección del flujo
        hsv[..., 1] = 255                     # Saturación máxima
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitud
        
        # Convertir a BGR para mostrar
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return flow_vis