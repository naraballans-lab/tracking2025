import numpy as np
from scipy.signal import savgol_filter
import cv2  # Asegúrate de tener importado OpenCV

class RotationAnalyzer:
    def __init__(self, frame_width=640, frame_height=480):
        """
        Inicializa el analizador de rotaciones
        Args:
            frame_width: Ancho del frame del video
            frame_height: Alto del frame del video
        """
        # Historiales básicos
        self.angle_history = []
        self.all_angles = []  # Historial completo de ángulos para análisis global
        self.position_history = []  # Historial de posiciones sin filtrar
        self.time_history = []  
        self.last_angle = None
        self.start_time = None
        
        # Contadores de vueltas usando el método robusto
        self.total_clockwise_turns = 0
        self.total_counterclockwise_turns = 0
        
        # Contadores acumulativos reales (solo se incrementan cuando se detecta una nueva vuelta)
        self.previous_total_horarias = 0
        self.previous_total_antihorarias = 0
        self.contador_real_horarias = 0
        self.contador_real_antihorarias = 0
        
        # Parámetros de análisis
        self.fps = 30.0
        self.frame_count = 0
        
        # Ángulo acumulado total para determinar dirección predominante
        self.total_angle_accumulated_rad = 0.0
        
        # Sistema de cuadrantes (4x4 = 16 cuadrantes)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.quadrant_width = frame_width // 4
        self.quadrant_height = frame_height // 4
        self.current_quadrant = None
        self.quadrant_history = []
        self.quadrant_sequence = []
        
        # Detección de vueltas basada en cuadrantes
        self.last_flow_direction = None  # 1 = horario, -1 = antihorario, 0 = sin dirección
        self.flow_direction_history = []
        self.quadrants_crossed_in_turn = []
        self.turn_in_progress = False
        self.min_quadrants_for_turn = 4  # Mínimo 4 cuadrantes para contar como vuelta
        
        # Control de tiempo para detección de vueltas
        self.quadrant_timestamps = []  # Timestamps de cuando se visitó cada cuadrante
        self.max_time_for_turn = 15.0  # Máximo 15 segundos para completar 4 cuadrantes
        self.turn_start_frame = None
        
        # Registro de detecciones de vueltas
        self.turn_detection_log = []  # Lista de mensajes de detección de vueltas
    
    def get_quadrant(self, position):
        """
        Determina en qué cuadrante (0-15) se encuentra una posición
        Cuadrantes numerados de izquierda a derecha, arriba a abajo:
        0  1  2  3
        4  5  6  7
        8  9  10 11
        12 13 14 15
        Args:
            position: Tuple (x, y) de la posición
        Returns:
            int: Número de cuadrante (0-15) o None si la posición es inválida
        """
        if position is None:
            return None
        
        x, y = position
        
        # Verificar que la posición está dentro de los límites
        if x < 0 or x >= self.frame_width or y < 0 or y >= self.frame_height:
            return None
        
        # Calcular fila y columna del cuadrante
        col = min(x // self.quadrant_width, 3)
        row = min(y // self.quadrant_height, 3)
        
        quadrant = row * 4 + col
        return int(quadrant)
    
    def detect_flow_direction_change(self, current_flow_direction):
        """
        Detecta si ha habido un cambio en la dirección del flujo óptico
        Args:
            current_flow_direction: Dirección actual del flujo (1=horario, -1=antihorario, 0=sin dirección)
        Returns:
            bool: True si hubo cambio de dirección
        """
        if self.last_flow_direction is None:
            self.last_flow_direction = current_flow_direction
            return False
        
        # Cambio de dirección si la dirección actual es diferente y no es 0
        if current_flow_direction != 0 and current_flow_direction != self.last_flow_direction:
            self.last_flow_direction = current_flow_direction
            return True
        
        return False
    
    def update_quadrant_based_turns(self, position, flow_direction):
        """
        Actualiza el sistema de detección de vueltas basado en cuadrantes y flujo óptico
        Cuenta una vuelta cada vez que se cruzan 3 cuadrantes distintos en movimiento lineal
        dentro de un límite de tiempo
        Args:
            position: Tuple (x, y) de la posición actual
            flow_direction: Dirección del flujo óptico (1=horario, -1=antihorario, 0=sin dirección)
        """
        if position is None:
            return
        
        # Obtener cuadrante actual
        quadrant = self.get_quadrant(position)
        if quadrant is None:
            return
        
        # Calcular tiempo actual en segundos
        current_time = self.frame_count / self.fps
        
        # Actualizar historial de cuadrantes
        self.quadrant_history.append(quadrant)
        if len(self.quadrant_history) > 100:
            self.quadrant_history = self.quadrant_history[-100:]
        
        # Añadir cuadrante a la secuencia actual si es diferente al último
        if len(self.quadrants_crossed_in_turn) == 0 or quadrant != self.quadrants_crossed_in_turn[-1]:
            # Si es el primer cuadrante de la secuencia, iniciar el timer
            if len(self.quadrants_crossed_in_turn) == 0:
                self.turn_start_frame = self.frame_count
            
            self.quadrants_crossed_in_turn.append(quadrant)
            self.quadrant_timestamps.append(current_time)
            
            # Limpiar cuadrantes antiguos que excedan el límite de tiempo
            if len(self.quadrant_timestamps) > 0:
                time_elapsed = current_time - self.quadrant_timestamps[0]
                
                # Si ha pasado demasiado tiempo, reiniciar la secuencia
                if time_elapsed > self.max_time_for_turn:
                    # Mantener solo los cuadrantes dentro del límite de tiempo
                    valid_indices = [i for i, t in enumerate(self.quadrant_timestamps) 
                                   if current_time - t <= self.max_time_for_turn]
                    
                    if valid_indices:
                        self.quadrants_crossed_in_turn = [self.quadrants_crossed_in_turn[i] for i in valid_indices]
                        self.quadrant_timestamps = [self.quadrant_timestamps[i] for i in valid_indices]
                        self.turn_start_frame = self.frame_count - int((current_time - self.quadrant_timestamps[0]) * self.fps)
                    else:
                        self.quadrants_crossed_in_turn = [quadrant]
                        self.quadrant_timestamps = [current_time]
                        self.turn_start_frame = self.frame_count
            
            # Contar cuadrantes únicos en la secuencia actual
            unique_quadrants = len(set(self.quadrants_crossed_in_turn))
            
            # Si se han cruzado 3 o más cuadrantes diferentes
            if unique_quadrants >= self.min_quadrants_for_turn:
                # Verificar que la secuencia es continua (movimiento lineal)
                if self._is_continuous_quadrant_sequence(self.quadrants_crossed_in_turn):
                    # Verificar que se completó dentro del límite de tiempo
                    time_for_turn = current_time - self.quadrant_timestamps[0]
                    
                    if time_for_turn <= self.max_time_for_turn:
                        # Determinar dirección basándose en la trayectoria de cuadrantes
                        turn_direction = self._determine_turn_direction_from_quadrants(
                            self.quadrants_crossed_in_turn
                        )
                        
                        # Contar la vuelta según la dirección
                        if turn_direction == 1:
                            self.contador_real_horarias += 1
                            mensaje = f"Frame {self.frame_count}: ¡Vuelta horaria detectada! ({unique_quadrants} cuadrantes en {time_for_turn:.2f}s: {self.quadrants_crossed_in_turn}) Total: {self.contador_real_horarias}"
                            print(mensaje)
                            self.turn_detection_log.append(mensaje)
                        elif turn_direction == -1:
                            self.contador_real_antihorarias += 1
                            mensaje = f"Frame {self.frame_count}: ¡Vuelta antihoraria detectada! ({unique_quadrants} cuadrantes en {time_for_turn:.2f}s: {self.quadrants_crossed_in_turn}) Total: {self.contador_real_antihorarias}"
                            print(mensaje)
                            self.turn_detection_log.append(mensaje)
                        else:
                            # Si no se puede determinar dirección, usar flow_direction
                            if flow_direction == 1:
                                self.contador_real_horarias += 1
                                mensaje = f"Frame {self.frame_count}: ¡Vuelta horaria detectada! ({unique_quadrants} cuadrantes en {time_for_turn:.2f}s: {self.quadrants_crossed_in_turn}) Total: {self.contador_real_horarias}"
                                print(mensaje)
                                self.turn_detection_log.append(mensaje)
                            elif flow_direction == -1:
                                self.contador_real_antihorarias += 1
                                mensaje = f"Frame {self.frame_count}: ¡Vuelta antihoraria detectada! ({unique_quadrants} cuadrantes en {time_for_turn:.2f}s: {self.quadrants_crossed_in_turn}) Total: {self.contador_real_antihorarias}"
                                print(mensaje)
                                self.turn_detection_log.append(mensaje)
                        
                        # Reiniciar la secuencia, manteniendo el último cuadrante como inicio de la próxima
                        self.quadrants_crossed_in_turn = [quadrant]
                        self.quadrant_timestamps = [current_time]
                        self.turn_start_frame = self.frame_count
        
        self.current_quadrant = quadrant
    
    def _determine_turn_direction_from_quadrants(self, quadrant_sequence):
        """
        Determina la dirección de giro basándose en la secuencia de cuadrantes
        Args:
            quadrant_sequence: Lista de cuadrantes visitados
        Returns:
            int: 1 para horario, -1 para antihorario, 0 si no se puede determinar
        """
        if len(quadrant_sequence) < 3:
            return 0
        
        # Usar los últimos 3 cuadrantes únicos
        unique_quads = []
        for q in quadrant_sequence:
            if not unique_quads or q != unique_quads[-1]:
                unique_quads.append(q)
        
        if len(unique_quads) < 3:
            return 0
        
        # Tomar los últimos 3
        q1, q2, q3 = unique_quads[-3:]
        
        # Convertir a coordenadas de grilla
        def to_coords(q):
            return (q // 4, q % 4)  # (fila, columna)
        
        r1, c1 = to_coords(q1)
        r2, c2 = to_coords(q2)
        r3, c3 = to_coords(q3)
        
        # Calcular vectores
        v1 = (c2 - c1, r2 - r1)  # Vector del primer al segundo
        v2 = (c3 - c2, r3 - r2)  # Vector del segundo al tercero
        
        # Producto cruzado para determinar dirección de giro
        # En un sistema de coordenadas de imagen (Y crece hacia abajo):
        # cross_product > 0 = giro antihorario (hacia la izquierda)
        # cross_product < 0 = giro horario (hacia la derecha)
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        
        if cross_product < 0:
            return 1   # Horario
        elif cross_product > 0:
            return -1  # Antihorario
        else:
            return 0   # Movimiento lineal (sin giro)
    
    def _is_continuous_quadrant_sequence(self, quadrant_sequence):
        """
        Verifica si una secuencia de cuadrantes representa un movimiento continuo/lineal
        Args:
            quadrant_sequence: Lista de cuadrantes visitados
        Returns:
            bool: True si la secuencia es continua
        """
        if len(quadrant_sequence) < 2:
            return False
        
        # Obtener solo los cuadrantes únicos en orden
        unique_sequence = []
        for q in quadrant_sequence:
            if not unique_sequence or q != unique_sequence[-1]:
                unique_sequence.append(q)
        
        if len(unique_sequence) < 2:
            return False
        
        # Verificar que no hay saltos demasiado grandes entre cuadrantes consecutivos
        for i in range(1, len(unique_sequence)):
            prev = unique_sequence[i-1]
            curr = unique_sequence[i]
            
            # Calcular posición del cuadrante en la grilla
            prev_row, prev_col = prev // 4, prev % 4
            curr_row, curr_col = curr // 4, curr % 4
            
            # Distancia Manhattan entre cuadrantes
            distance = abs(curr_row - prev_row) + abs(curr_col - prev_col)
            
            # Permitir movimientos a cuadrantes adyacentes (distancia <= 2, incluyendo diagonales)
            # Para movimiento lineal, permitir distancia máxima de 2
            if distance > 2:
                return False
        
        return True
    
    def draw_quadrants(self, frame):
        """
        Dibuja las líneas de los cuadrantes en el frame
        Args:
            frame: Frame de OpenCV
        Returns:
            Frame con cuadrantes dibujados
        """
        frame_copy = frame.copy()
        
        # Dibujar líneas verticales
        for i in range(1, 4):
            x = i * self.quadrant_width
            cv2.line(frame_copy, (x, 0), (x, self.frame_height), (0, 255, 0), 1)
        
        # Dibujar líneas horizontales
        for i in range(1, 4):
            y = i * self.quadrant_height
            cv2.line(frame_copy, (0, y), (self.frame_width, y), (0, 255, 0), 1)
        
        # Resaltar el cuadrante actual
        if self.current_quadrant is not None:
            row = self.current_quadrant // 4
            col = self.current_quadrant % 4
            x1 = col * self.quadrant_width
            y1 = row * self.quadrant_height
            x2 = x1 + self.quadrant_width
            y2 = y1 + self.quadrant_height
            
            # Dibujar rectángulo semitransparente
            overlay = frame_copy.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
            cv2.addWeighted(overlay, 0.3, frame_copy, 0.7, 0, frame_copy)
            
            # Número del cuadrante
            center_x = x1 + self.quadrant_width // 2
            center_y = y1 + self.quadrant_height // 2
            cv2.putText(frame_copy, str(self.current_quadrant), (center_x - 10, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
        return frame_copy

    def is_continuous_rotation(self, angles):
        """
        Verifica si una secuencia de ángulos representa una vuelta completa (360°)
        en una dirección específica.
        
        Args:
            angles: Lista de ángulos en grados (-180 a 180)
        Returns:
            (bool, int): (es_vuelta_completa, dirección [1=horario, -1=antihorario, 0=no_vuelta])
        """
        if len(angles) < 4:  # Reducimos el mínimo de puntos necesarios
            return False, 0
            
        # Convertir a radianes y desenvolver para manejar transiciones 180° ↔ -180°
        angles_rad = np.radians(angles)
        unwrapped = np.unwrap(angles_rad)
        total_rotation = np.degrees(unwrapped[-1] - unwrapped[0])
        
        # Calcular cambios entre frames consecutivos
        changes = np.diff(np.degrees(unwrapped))
        
        # Filtrar pequeñas fluctuaciones (ruido)
        changes[np.abs(changes) < 2.0] = 0
        
        # Verificar dirección consistente
        clockwise = np.sum(changes > 0)
        counterclockwise = np.sum(changes < 0)
        total_nonzero_changes = np.sum(changes != 0)
        
        if total_nonzero_changes == 0:
            return False, 0
            
        # Determinar dirección predominante (ahora solo necesitamos 60% en una dirección)
        if clockwise / total_nonzero_changes > 0.6:
            direction = 1  # horario
        elif counterclockwise / total_nonzero_changes > 0.6:
            direction = -1  # antihorario
        else:
            return False, 0
            
        # Permitir saltos más grandes entre frames
        max_allowed_gap = 60  # grados máximos entre frames
        if np.any(np.abs(changes) > max_allowed_gap):
            return False, 0
            
        # Ser más permisivo con lo que consideramos una vuelta completa (360° ± 20%)
        if direction == 1 and 288 <= abs(total_rotation) <= 432:  # 360° ± 20%
            return True, 1
        elif direction == -1 and -432 <= total_rotation <= -288:  # -360° ± 20%
            return True, -1
        
        return False, 0

    def add_position_and_calculate_turns(self, position, flow_magnitude=0, flow_angle=0):
        """
        Añade una nueva posición sin filtrar y calcula las vueltas basándose en la trayectoria y flujo óptico
        Args:
            position: Tupla (x, y) de la posición actual sin filtrar
            flow_magnitude: Magnitud del flujo óptico
            flow_angle: Ángulo del flujo óptico en grados
        """
        if position is not None:
            self.frame_count += 1
            self.position_history.append(position)
            
            # Mantener solo los últimos 500 frames para evitar acumulación excesiva
            if len(self.position_history) > 500:
                self.position_history = self.position_history[-500:]
            
            # Determinar dirección del flujo basándose en el ángulo
            # Considerar flujo significativo solo si la magnitud es suficiente
            flow_direction = 0
            if flow_magnitude > 0.5:  # Umbral mínimo de magnitud
                # Normalizar ángulo a rango -180 a 180
                normalized_angle = ((flow_angle + 180) % 360) - 180
                
                # Determinar dirección basándose en cambios de ángulo
                if len(self.position_history) > 1:
                    angle = self.calculate_orientation_from_trajectory(position, self.position_history[:-1])
                    if angle is not None and self.last_angle is not None:
                        angle_diff = angle - self.last_angle
                        # Normalizar diferencia de ángulo
                        while angle_diff > 180:
                            angle_diff -= 360
                        while angle_diff < -180:
                            angle_diff += 360
                        
                        # Dirección: positivo = horario, negativo = antihorario
                        if angle_diff > 5:  # Umbral de 5 grados
                            flow_direction = 1
                        elif angle_diff < -5:
                            flow_direction = -1
            
            # Actualizar sistema de cuadrantes y detección de vueltas
            self.update_quadrant_based_turns(position, flow_direction)
            
            # Calcular orientación basada en la trayectoria cada 3 frames (más frecuente)
            if self.frame_count % 3 == 0:
                angle = self.calculate_orientation_from_trajectory(position, self.position_history[:-1])
                
                if angle is not None:
                    self.all_angles.append(angle)
                    self.last_angle = angle
            
    def get_net_rotations(self):
        """
        Calcula el número neto de rotaciones
        Returns:
            rotations: Número de vueltas completas (positivo = sentido horario)
        """
        return self.cumulative_angle / 360.0
        
    def get_rotation_metrics(self):
        """
        Retorna métricas detalladas de rotación basadas en contadores acumulativos reales
        Returns:
            dict: Diccionario con métricas de rotación usando contadores que solo se incrementan
        """
        # Usar los contadores reales acumulativos
        if hasattr(self, 'contador_real_horarias'):
            total_angle_deg = np.degrees(self.total_angle_accumulated_rad)
            predominant_direction = "horario" if total_angle_deg > 0 else "antihorario" if total_angle_deg < 0 else "ninguna"
            
            return {
                'net_rotations': self.contador_real_horarias - self.contador_real_antihorarias,
                'clockwise_turns': self.contador_real_horarias,
                'counterclockwise_turns': self.contador_real_antihorarias,
                'clockwise_degrees': self.contador_real_horarias * 360.0,
                'counterclockwise_degrees': self.contador_real_antihorarias * 360.0,
                'total_degrees': (self.contador_real_horarias + self.contador_real_antihorarias) * 360.0,
                'total_time': self.frame_count / self.fps if self.frame_count > 0 else 0,
                'angulo_acumulado_rad': self.total_angle_accumulated_rad,
                'angulo_acumulado_grados': total_angle_deg,
                'direccion_predominante': predominant_direction,
                'movimiento_total_horario_grados': max(0, total_angle_deg),
                'movimiento_total_antihorario_grados': max(0, -total_angle_deg),
                'turn_detection_log': self.turn_detection_log if hasattr(self, 'turn_detection_log') else []
            }
        else:
            # Fallback para compatibilidad (caso de inicialización)
            return {
                'net_rotations': 0,
                'clockwise_turns': 0,
                'counterclockwise_turns': 0,
                'clockwise_degrees': 0,
                'counterclockwise_degrees': 0,
                'total_degrees': 0,
                'total_time': 0,
                'angulo_acumulado_rad': 0,
                'angulo_acumulado_grados': 0,
                'direccion_predominante': 'ninguna',
                'movimiento_total_horario_grados': 0,
                'movimiento_total_antihorario_grados': 0,
                'turn_detection_log': []
            }
        
    def calculate_orientation_from_trajectory(self, current_pos, previous_positions, window_size=10):
        """
        Calcula la orientación basándose en la trayectoria de posiciones sin filtrar
        Args:
            current_pos: Posición actual (x, y)
            previous_positions: Lista de posiciones anteriores
            window_size: Número de posiciones anteriores a considerar (reducido para más sensibilidad)
        Returns:
            angle: Ángulo en grados (-180 a 180) o None si no hay suficientes datos
        """
        if len(previous_positions) < window_size:
            return None
            
        # Usar las últimas posiciones para calcular la dirección del movimiento
        recent_positions = previous_positions[-window_size:]
        recent_positions.append(current_pos)
        
        if len(recent_positions) < 2:
            return None
            
        # Solo usar el vector de la primera a la última posición para mayor estabilidad
        start_pos = recent_positions[0]
        end_pos = recent_positions[-1]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        # Verificar que hay movimiento significativo (reducido a 5 píxeles)
        distance = np.sqrt(dx*dx + dy*dy)
        if distance < 5:  # Mínimo 5 píxeles de movimiento
            return self.last_angle  # Mantener el último ángulo válido
            
        # Calcular el ángulo en grados
        angle = np.degrees(np.arctan2(dy, dx))
        return angle
        
    def analizar_vueltas_pipeline(self, lista_de_angulos, fps=None):
        """
        Analiza el número y dirección de vueltas usando la función robusta de conteo.
        Args:
            lista_de_angulos: array/list de ángulos en grados o radianes
            fps: frames por segundo (opcional, usa self.fps si no se especifica)
        Returns:
            dict con vueltas horarias, antihorarias, netas, ángulo acumulado y duración
        """
        if fps is None:
            fps = self.fps
        return contar_vueltas_angulo(lista_de_angulos, fps=fps)

    def mostrar_contador_vueltas_en_frame(self, frame, position=(10, 100)):
        """
        Dibuja el contador de vueltas horarias y antihorarias en el frame
        Muestra las vueltas detectadas por el sistema de cuadrantes
        Args:
            frame: imagen/frame de OpenCV
            position: posición inicial del texto (x, y)
        Returns:
            frame modificado
        """
        x, y = position
        
        # Obtener contadores (priorizando el sistema de cuadrantes)
        vueltas_horarias = self.contador_real_horarias if hasattr(self, 'contador_real_horarias') else 0
        vueltas_antihorarias = self.contador_real_antihorarias if hasattr(self, 'contador_real_antihorarias') else 0
        vueltas_totales = vueltas_horarias + vueltas_antihorarias
        vueltas_netas = vueltas_horarias - vueltas_antihorarias
        
        # Mostrar vueltas horarias (en verde)
        cv2.putText(frame, f"Vueltas horarias: {vueltas_horarias}", 
                   (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar vueltas antihorarias (en rojo)
        cv2.putText(frame, f"Vueltas antihorarias: {vueltas_antihorarias}", 
                   (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Mostrar vueltas totales (en cian)
        cv2.putText(frame, f"Vueltas totales: {vueltas_totales}", 
                   (x, y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Mostrar vueltas netas (en amarillo)
        cv2.putText(frame, f"Vueltas netas: {vueltas_netas:+d}", 
                   (x, y+90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Mostrar método de detección
        cv2.putText(frame, "(Sistema de cuadrantes)", 
                   (x, y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        return frame

def contar_vueltas_angulo(angulos, fps=30, suavizar=True, window_length=31, polyorder=3):
    """
    Calcula el número y dirección de vueltas completas (360°) a partir de una secuencia de ángulos.
    Versión balanceada para detectar vueltas reales sin falsos positivos.
    Args:
        angulos: array/list de ángulos en radianes (preferido) o grados
        fps: frames por segundo
        suavizar: si True, aplica filtro Savitzky-Golay
        window_length: ventana del filtro (impar) 
        polyorder: orden del polinomio para el filtro
    Returns:
        dict con vueltas horarias, antihorarias, netas y ángulo acumulado
    """
    angulos = np.asarray(angulos)
    if len(angulos) < 30:  # Reducido de 100 a 30 puntos mínimos
        return {
            'vueltas_horarias': 0,
            'vueltas_antihorarias': 0,
            'vueltas_netas': 0,
            'angulo_acumulado_rad': 0,
            'duracion_frames': len(angulos)
        }
    
    # Si los ángulos están en grados, convertir a radianes
    if np.max(np.abs(angulos)) > 2 * np.pi:
        angulos = np.radians(angulos)
    
    # Suavizado moderado
    if suavizar and len(angulos) >= window_length:
        angulos = savgol_filter(angulos, window_length=window_length, polyorder=polyorder)
    
    # Unwrap para evitar saltos de -pi a pi
    angulos_unwrap = np.unwrap(angulos)
    
    # Calcular el ángulo total recorrido
    angulo_total = angulos_unwrap[-1] - angulos_unwrap[0]
    
    # Contar vueltas completas con umbral del 80% (288° en lugar de 360°)
    DOS_PI = 2 * np.pi
    umbral_vuelta = DOS_PI * 0.8  # 80% de una vuelta completa
    
    # Contar vueltas horarias y antihorarias
    if angulo_total > 0:
        vueltas_horarias = int(angulo_total // umbral_vuelta)
        vueltas_antihorarias = 0
    else:
        vueltas_horarias = 0
        vueltas_antihorarias = int((-angulo_total) // umbral_vuelta)
    
    netas = vueltas_horarias - vueltas_antihorarias
    
    return {
        'vueltas_horarias': vueltas_horarias,
        'vueltas_antihorarias': vueltas_antihorarias,
        'vueltas_netas': netas,
        'angulo_acumulado_rad': float(angulo_total),
        'duracion_frames': len(angulos)
    }

# Ejemplo de uso:
#analyzer = RotationAnalyzer()
#resultados = analyzer.analizar_vueltas_pipeline(lista_de_angulos)
#print(resultados)

    # En tu bucle principal de análisis de video, después de calcular el ángulo instantáneo:
    # frame = ... (frame actual de OpenCV)
    # lista_de_angulos = ... (lista acumulada de ángulos hasta el frame actual)
    # frame = analyzer.mostrar_contador_vueltas_en_frame(frame, lista_de_angulos)