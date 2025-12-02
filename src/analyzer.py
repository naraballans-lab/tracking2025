import numpy as np
import cv2


class RotationAnalyzer:
    """
    RotationAnalyzer = detectar cuando un ratón da una vuelta completa. Uso cuadrantes espaciales, ángulos, tiempo...
    """
    
    def __init__(self, ancho_frame=640, alto_frame=480, id_video=None):
        """
        Aquí configuro todo lo necesario para analizar las rotaciones del ratón.
        
        Args:
            ancho_frame: ancho del vídeo en píxeles (640)
            alto_frame: alto del vídeo en píxeles (480)
            id_video: para cuando proceso múltiples vídeos en collage
        """
        # Guardo el ID del vídeo para mensajes en modo collage
        self.id_video = id_video
        
        # Guardo todo para poder analizar patrones
        self.historial_angulos = []  
        self.todos_angulos = []      
        self.historial_posiciones = []  # posiciones del ratón en cada frame
        self.historial_tiempos = []     # tiempos correspondientes
        self.ultimo_angulo = None      # el último ángulo que registré
        self.tiempo_inicio = None      
        
        # Contadores de vueltas
        self.total_vueltas_horarias = 0      
        self.total_vueltas_antihorarias = 0  
        self.total_previo_horarias = 0      
        self.total_previo_antihorarias = 0
        self.contador_real_horarias = 0      
        self.contador_real_antihorarias = 0
        
        # Parámetros básicos de análisis
        self.fps = 30.0  # vídeos de prueba = 30 frames por segundo
        self.contador_frames = 0 
        self.angulo_total_acumulado_rad = 0.0  # ángulo total acumulado en radianes
        
        # Divido el espacio en una grilla y veo cómo el ratón se mueve entre cuadrantes
        # Se configura automaticamente en base al tamaño del ratón detectado (Tamaño ratón: 39x61px-->Grilla:17x17-->Mín cuadrantes vuelta: 11)
        self.ancho_frame = ancho_frame
        self.alto_frame = alto_frame
        self.ancho_cuadrante = None             
        self.alto_cuadrante = None              
        self.cuadrante_actual = None             # en qué cuadrante está ahora
        self.historial_cuadrantes = []           # historial de cuadrantes visitados
        self.secuencia_cuadrantes = []          # secuencia actual de cuadrantes en una vuelta
        
        #Variables para detectar vueltas usando cuadrantes (criterio espacial)
        self.ultima_direccion_flujo = None      # última dirección que detecté
        self.historial_direccion_flujo = []     
        self.cuadrantes_cruzados_en_vuelta = [] # cuadrantes que ha cruzado en la vuelta actual
        self.vuelta_en_progreso = False        
        self.min_cuadrantes_para_vuelta = None  # Se calcula basado en tamaño del ratón
        
        #Control de tiempo (criterio temporal)
        self.marcas_tiempo_cuadrantes = []      
        self.tiempo_max_para_vuelta = 15.0      # máximo 15 segundos para completar una vuelta/puede modificarse
        self.frame_inicio_vuelta = None         # frame donde empezó la vuelta actual
        
        #Control de ángulos acumulados (criterio angular)
        self.cambios_angulo_en_vuelta = []      # cambios de ángulo en la vuelta actual
        self.suma_min_angulo_para_vuelta = 270.0  # mínimo 270° para vuelta (un poco menos de 360°)
        self.ultimo_angulo_en_vuelta = None     
        
        # Registro de todas las detecciones para análisis posterior
        self.registro_deteccion_vueltas = []
        
        # Necesito saber si el ratón está quieto o moviéndose (Movimiento/Parado)
        self.esta_moviendo = False              
        self.umbral_movimiento = 5.0            # píxeles mínimos para considerar movimiento
        self.ultima_posicion = None             
        self.frames_moviendo = 0                # contador de frames en movimiento
        self.frames_detenido = 0                # contador de frames detenido
        
        self.ancho_raton = None                 # ancho del ratón en píxeles
        self.alto_raton = None                  # alto del ratón en píxeles
        self.num_cuadrantes_x = None            
        self.num_cuadrantes_y = None            
    
    def configurar_cuadrantes_desde_tam_raton(self, ancho_raton, alto_raton):
        """
        Configuro los cuadrantes automáticamente según el tamaño del ratón.
        Si el ratón es grande, hago cuadrantes más grandes; si es pequeño, más pequeños.
        Esto hace que el sistema sea adaptable a diferentes tamaños de ratón.
        
        Args:
            ancho_raton: ancho del ratón detectado en píxeles
            alto_raton: alto del ratón detectado en píxeles
        """
        # Guardo las dimensiones del ratón
        self.ancho_raton = ancho_raton
        self.alto_raton = alto_raton
        
        # Calculo el tamaño objetivo del cuadrante: el ratón debe ocupar 3/4 del área
        # (4/3 del tamaño del ratón = 3/4 del área del cuadrante)
        ancho_cuadrante_objetivo = int(ancho_raton * 4 / 3)
        alto_cuadrante_objetivo = int(alto_raton * 4 / 3)
        
        # Calculo cuántos cuadrantes necesito en cada dirección
        # Aseguro mínimo 2 cuadrantes para que tenga sentido
        self.num_cuadrantes_x = max(2, self.ancho_frame // ancho_cuadrante_objetivo)
        self.num_cuadrantes_y = max(2, self.alto_frame // alto_cuadrante_objetivo)
        
        # Recalculo el tamaño real de cada cuadrante
        self.ancho_cuadrante = self.ancho_frame // self.num_cuadrantes_x
        self.alto_cuadrante = self.alto_frame // self.num_cuadrantes_y
        
        # Calculo el perímetro externo para determinar mínimo cuadrantes para vuelta (quito 4 esquinas porque se cuentan dos veces)
        cuadrantes_perimetro = 2 * (self.num_cuadrantes_x + self.num_cuadrantes_y) - 4
        self.min_cuadrantes_para_vuelta = max(4, cuadrantes_perimetro // 4)
        
        #Datos en terminal
        prefijo_video = f"[Video {self.id_video}] " if self.id_video else ""
        print(f"{prefijo_video}Cuadrantes configurados automáticamente:")
        print(f"{prefijo_video}  - Tamaño ratón: {ancho_raton}x{alto_raton} px")
        print(f"{prefijo_video}  - Grilla: {self.num_cuadrantes_x}x{self.num_cuadrantes_y} cuadrantes")
        print(f"{prefijo_video}  - Tamaño cuadrante: {self.ancho_cuadrante}x{self.alto_cuadrante} px")
        print(f"{prefijo_video}  - Mín. cuadrantes para vuelta: {self.min_cuadrantes_para_vuelta}")
        
        # Reinicio todos los contadores cuando cambio la configuración
        self.cuadrantes_cruzados_en_vuelta = []
        self.marcas_tiempo_cuadrantes = []
        self.cambios_angulo_en_vuelta = []
        self.ultimo_angulo_en_vuelta = None
        self.cuadrante_actual = None
    
    def obtener_cuadrante(self, posicion):
        """
        Convierto coordenadas (x,y) en número de cuadrante.
        
        Args:
            posicion: (x, y) con las coordenadas de la posición
            
        Returns:
            int: Número de cuadrante (basado en grilla), o None si está fuera de límites
        """
        # Si no se ha configurado el tamaño del ratón, no puedo calcular cuadrantes
        if self.ancho_cuadrante is None or self.num_cuadrantes_x is None:
            return None
            
        # Si no tengo posición, no puedo determinar cuadrante
        if posicion is None:
            return None
        
        # Separo las coordenadas x,y
        x, y = posicion
        
        # Verifico que esté dentro de los límites del frame
        if x < 0 or x >= self.ancho_frame or y < 0 or y >= self.alto_frame:
            return None
        
        # Calculo en qué columna y fila está
        # Uso min() para evitar índices fuera de rango
        col = min(x // self.ancho_cuadrante, self.num_cuadrantes_x - 1)
        fila = min(y // self.alto_cuadrante, self.num_cuadrantes_y - 1)
        
        # Convierto fila,columna en un número único de cuadrante
        # Numerar: fila0: 0,1,2,3; fila1: 4,5,6,7; etc.
        cuadrante = fila * self.num_cuadrantes_x + col
        return int(cuadrante)

    def actualizar_vueltas_basadas_en_cuadrantes(self, posicion, direccion_flujo, angulo_actual=None):
        """
        Aquí combino cuadrantes,tiempo y ángulos para decidir si el ratón completó una vuelta.
        Valida vueltas completas cuando:
        1. Se cruzan suficientes cuadrantes distintos (>mín de cuadrantes)
        2. En un tiempo límite (15 segundos máximo)
        3. Con una suma de cambios angulares significativa (>270°)
        
        Args:
            posicion: (x, y) de la posición actual del ratón
            direccion_flujo: Dirección del flujo óptico (1=horario, -1=antihorario, 0=sin dirección)
            angulo_actual: Ángulo actual de orientación en grados 
        """
        
        if posicion is None:
            return
        
        # Determino en qué cuadrante está el ratón ahora
        cuadrante = self.obtener_cuadrante(posicion)
        if cuadrante is None:
            return
        
        # Calculo el tiempo actual en segundos desde el inicio
        tiempo_actual = self.contador_frames / self.fps
        
        # Mantengo un historial de cuadrantes (últimos 100 para no gastar memoria)
        self.historial_cuadrantes.append(cuadrante)
        if len(self.historial_cuadrantes) > 100:
            self.historial_cuadrantes = self.historial_cuadrantes[-100:]
        
        # Solo agrego cuadrante si es diferente al último (para no contar el mismo cuadrante muchas veces)
        if len(self.cuadrantes_cruzados_en_vuelta) == 0 or cuadrante != self.cuadrantes_cruzados_en_vuelta[-1]:
            if len(self.cuadrantes_cruzados_en_vuelta) == 0:
                self.frame_inicio_vuelta = self.contador_frames
                # Reinicio la acumulación de ángulos para esta nueva vuelta
                self.cambios_angulo_en_vuelta = []
                self.ultimo_angulo_en_vuelta = angulo_actual
            
            self.cuadrantes_cruzados_en_vuelta.append(cuadrante)
            self.marcas_tiempo_cuadrantes.append(tiempo_actual)
            
            # Si pasa demasiado tiempo, reinicio todo
            if len(self.marcas_tiempo_cuadrantes) > 0:
                tiempo_transcurrido = tiempo_actual - self.marcas_tiempo_cuadrantes[0]
                
                # Si han pasado más de 15 segundos, reinicio completamente la secuencia (no valen giros lentos)
                if tiempo_transcurrido > self.tiempo_max_para_vuelta:
                    self.cuadrantes_cruzados_en_vuelta = [cuadrante]
                    self.marcas_tiempo_cuadrantes = [tiempo_actual]
                    self.frame_inicio_vuelta = self.contador_frames
                    # Reinicio también los ángulos acumulados
                    self.cambios_angulo_en_vuelta = []
                    self.ultimo_angulo_en_vuelta = angulo_actual
        
        # Acumulación de ángulos
        # Solo acumulo cambios si tengo ángulo actual + hay una vuelta en progreso + el ratón se está moviendo
        if angulo_actual is not None and len(self.cuadrantes_cruzados_en_vuelta) > 0 and self.esta_moviendo:
            if self.ultimo_angulo_en_vuelta is not None:
                # Calculo cuánto cambió el ángulo desde el último frame
                diferencia_angulo = angulo_actual - self.ultimo_angulo_en_vuelta
                # Normalizo al rango [-180, 180] para evitar problemas con el cambio de 360° a 0°
                while diferencia_angulo > 180:
                    diferencia_angulo -= 360
                while diferencia_angulo < -180:
                    diferencia_angulo += 360
                
                # Solo acumulo cambios significativos (>1°) para evitar ruido
                if abs(diferencia_angulo) > 1.0:
                    # Si la suma acumulada es significativa (>120°) y el nuevo cambio tiene signo opuesto, significa que cambió de dirección completamente = reinicio la acumulación
                    if len(self.cambios_angulo_en_vuelta) > 0:
                        suma_actual = sum(self.cambios_angulo_en_vuelta)

                        if abs(suma_actual) > 120:
                            if (suma_actual > 0 and diferencia_angulo < -20) or (suma_actual < 0 and diferencia_angulo > 20):
                                self.cambios_angulo_en_vuelta = [diferencia_angulo]
                            else:
                                self.cambios_angulo_en_vuelta.append(diferencia_angulo)
                        else:
                            self.cambios_angulo_en_vuelta.append(diferencia_angulo)
                    else:
                        self.cambios_angulo_en_vuelta.append(diferencia_angulo)
            
            self.ultimo_angulo_en_vuelta = angulo_actual
            
            cuadrantes_unicos = len(set(self.cuadrantes_cruzados_en_vuelta))
            
            # Si cruzó suficientes cuadrantes únicos
            if cuadrantes_unicos >= self.min_cuadrantes_para_vuelta:
                #sin saltos
                if self._es_secuencia_cuadrantes_continua(self.cuadrantes_cruzados_en_vuelta):
                    tiempo_para_vuelta = tiempo_actual - self.marcas_tiempo_cuadrantes[0]
                    
                    # Calculo el cambio angular total (en valor absoluto)
                    cambio_angulo_total = sum(abs(a) for a in self.cambios_angulo_en_vuelta)
                    
                    # 1. Tiempo entre 1-15 segundos (no demasiado rápido, no demasiado lento)
                    # 2. Cambio angular total >= 270° (casi una vuelta completa)
                    if tiempo_para_vuelta >= 1.0 and tiempo_para_vuelta <= self.tiempo_max_para_vuelta:
                        if cambio_angulo_total >= self.suma_min_angulo_para_vuelta:
                            # Vuelta detectada, ahora determino la dirección
                            # Dirección según cuadrantes
                            direccion_vuelta_cuadrantes = self._determinar_direccion_vuelta_desde_cuadrantes(
                                self.cuadrantes_cruzados_en_vuelta
                            )
                            
                            # Dirección según ángulos (más precisa)
                            suma_firmada = sum(self.cambios_angulo_en_vuelta)
                            direccion_vuelta_angulos = 1 if suma_firmada > 0 else -1 if suma_firmada < 0 else 0
                            
                            # Priorizo ángulos sobre cuadrantes en caso de conflicto
                            if direccion_vuelta_cuadrantes != 0 and direccion_vuelta_angulos != 0:
                                # Si coinciden, perfecto, si no me quedo con ángulos (más precisos)
                                direccion_vuelta = direccion_vuelta_cuadrantes if direccion_vuelta_cuadrantes == direccion_vuelta_angulos else direccion_vuelta_angulos
                            elif direccion_vuelta_angulos != 0:
                                direccion_vuelta = direccion_vuelta_angulos
                            elif direccion_vuelta_cuadrantes != 0:
                                direccion_vuelta = direccion_vuelta_cuadrantes
                            else:
                                # Como útimo recurso: uso el flujo óptico actual
                                direccion_vuelta = direccion_flujo
                            
                            # Contabilizo vuelta
                            lista_cuadrantes_unicos = []
                            for q in self.cuadrantes_cruzados_en_vuelta:
                                if not lista_cuadrantes_unicos or q != lista_cuadrantes_unicos[-1]:
                                    lista_cuadrantes_unicos.append(q)
                            cadena_cuadrantes = "→".join(map(str, lista_cuadrantes_unicos))
                            simbolo_direccion = "+" if suma_firmada > 0 else "-" if suma_firmada < 0 else "="
                            
                            # Aumento el contador 
                            if direccion_vuelta == 1:
                                self.contador_real_horarias += 1
                                prefijo_video = f"[Video {self.id_video}] " if self.id_video else ""
                                mensaje = f"{prefijo_video}Frame {self.contador_frames}: ¡Vuelta horaria detectada! ({cuadrantes_unicos} cuadrantes: {cadena_cuadrantes}, {cambio_angulo_total:.1f}° [{simbolo_direccion}] en {tiempo_para_vuelta:.2f}s) Total: {self.contador_real_horarias}"
                                print(mensaje)
                                self.registro_deteccion_vueltas.append(mensaje)
                            elif direccion_vuelta == -1:
                                self.contador_real_antihorarias += 1
                                prefijo_video = f"[Video {self.id_video}] " if self.id_video else ""
                                mensaje = f"{prefijo_video}Frame {self.contador_frames}: ¡Vuelta antihoraria detectada! ({cuadrantes_unicos} cuadrantes: {cadena_cuadrantes}, {cambio_angulo_total:.1f}° [{simbolo_direccion}] en {tiempo_para_vuelta:.2f}s) Total: {self.contador_real_antihorarias}"
                                print(mensaje)
                                self.registro_deteccion_vueltas.append(mensaje)
                            
                            # Reinicio todo para la próxima vuelta
                            self.cuadrantes_cruzados_en_vuelta = [cuadrante]
                            self.marcas_tiempo_cuadrantes = [tiempo_actual]
                            self.frame_inicio_vuelta = self.contador_frames
                            self.cambios_angulo_en_vuelta = []
                            self.ultimo_angulo_en_vuelta = angulo_actual
        
        # Actualizo el cuadrante actual para el siguiente frame
        self.cuadrante_actual = cuadrante
    
    def _determinar_direccion_vuelta_desde_cuadrantes(self, secuencia_cuadrantes):
        """
        Uso producto vectorial para determinar si la secuencia de cuadrantes representa un giro horario o antihorario. 
        Args:
            secuencia_cuadrantes: lista de cuadrantes visitados
            
        Returns:
            int: 1 para horario, -1 para antihorario, 0 si indeterminado
        """
        # Si no se ha configurado el tamaño del ratón no puedo calcular direcciones
        if self.num_cuadrantes_x is None:
            return 0
            
        # Necesito al menos 3 puntos para determinar si es un giro
        if len(secuencia_cuadrantes) < 3:
            return 0
        
        # Elimino cuadrantes consecutivos repetidos
        cuadrantes_unicos = []
        for q in secuencia_cuadrantes:
            if not cuadrantes_unicos or q != cuadrantes_unicos[-1]:
                cuadrantes_unicos.append(q)
        
        if len(cuadrantes_unicos) < 3:
            return 0
        
        # Tomo los últimos 3 puntos más recientes
        q1, q2, q3 = cuadrantes_unicos[-3:]
        
        # Convierto número de cuadrante a coordenadas (fila, columna)
        def a_coordenadas(q):
            return (q // self.num_cuadrantes_x, q % self.num_cuadrantes_x)
        
        fila1, col1 = a_coordenadas(q1)
        fila2, col2 = a_coordenadas(q2)
        fila3, col3 = a_coordenadas(q3)
        
        # Calculo los vectores entre puntos consecutivos
        # v1 va de punto1 a punto2, v2 va de punto2 a punto3
        v1 = (col2 - col1, fila2 - fila1)
        v2 = (col3 - col2, fila3 - fila2)
        
        # Si es positivo: giro horario, negativo: antihorario, cero: movimiento recto
        producto_cruz = v1[0] * v2[1] - v1[1] * v2[0]
        
        if producto_cruz > 0:
            return 1  # ¡Giro horario detectado!
        elif producto_cruz < 0:
            return -1  # ¡Giro antihorario detectado!
        else:
            return 0  # Movimiento recto o indeterminado
    
    def _es_secuencia_cuadrantes_continua(self, secuencia_cuadrantes):
        """
        Este método verifica si la secuencia de cuadrantes es continua
        Evita que el sistema cuente como vuelta cuando el ratón hace saltos grandes o movimientos erráticos
        
        Args:
            secuencia_cuadrantes: Lista de cuadrantes visitados
            
        Returns:
            bool: True si la secuencia es continua (sin saltos grandes)
        """
        if len(secuencia_cuadrantes) < 2:
            return False

        secuencia_unica = []
        for q in secuencia_cuadrantes:
            if not secuencia_unica or q != secuencia_unica[-1]:
                secuencia_unica.append(q)
        
        if len(secuencia_unica) < 2:
            return False

        for i in range(1, len(secuencia_unica)):
            previo = secuencia_unica[i-1]
            actual = secuencia_unica[i]
            
            # Conversión a coordenadas para calcular distancias
            fila_previa, col_previa = previo // self.num_cuadrantes_x, previo % self.num_cuadrantes_x
            fila_actual, col_actual = actual // self.num_cuadrantes_x, actual % self.num_cuadrantes_x
            
            # diferencia filas + diferencia columnas 
            distancia = abs(fila_actual - fila_previa) + abs(col_actual - col_previa)
            
            # Si salta más de 2 unidades=no es continuo (para evitar contar movimientos largos)
            if distancia > 2:
                return False

        return True
    
    def dibujar_cuadrantes(self, frame):
        """
        Este método dibuja la grilla de cuadrantes y resalta el cuadrante actual en amarillo
        
        Args:
            frame: Frame de OpenCV donde dibujar
            
        Returns:
            np.ndarray: Frame con cuadrantes dibujados
        """
        # Si no se ha configurado el tamaño del ratón= no puedo dibujar cuadrantes
        if self.ancho_cuadrante is None or self.num_cuadrantes_x is None:
            return frame.copy()  # Devuelvo copia sin modificaciones

        copia_frame = frame.copy()
        
        # Dibujo líneas verticales para crear las columnas de la grilla
        for i in range(1, self.num_cuadrantes_x):
            x = i * self.ancho_cuadrante
            cv2.line(copia_frame, (x, 0), (x, self.alto_frame), (0, 255, 0), 1)
        
        # Dibujo líneas horizontales para crear las filas de la grilla
        for i in range(1, self.num_cuadrantes_y):
            y = i * self.alto_cuadrante
            cv2.line(copia_frame, (0, y), (self.ancho_frame, y), (0, 255, 0), 1)
        
        # Resaltar cuadrante actual  en amarillo
        if self.cuadrante_actual is not None:
            # Cálculo de coordenadas: convierto número de cuadrante a píxeles
            row = self.cuadrante_actual // self.num_cuadrantes_x
            col = self.cuadrante_actual % self.num_cuadrantes_x
            x1 = col * self.ancho_cuadrante
            y1 = row * self.alto_cuadrante
            x2 = x1 + self.ancho_cuadrante
            y2 = y1 + self.alto_cuadrante
            
            superposicion = copia_frame.copy()
            cv2.rectangle(superposicion, (x1, y1), (x2, y2), (0, 255, 255), -1)
            cv2.addWeighted(superposicion, 0.3, copia_frame, 0.7, 0, copia_frame)

            centro_x = x1 + self.ancho_cuadrante // 2
            centro_y = y1 + self.alto_cuadrante // 2
            cv2.putText(copia_frame, str(self.cuadrante_actual), (centro_x - 10, centro_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
        return copia_frame
    
    def agregar_posicion_y_calcular_vueltas(self, posicion, magnitud_flujo=0, angulo_flujo=0):
        """
        Aquí llega cada nueva posición del ratón y decido si contar una vuelta o no
        
        Args:
            posicion: Tupla (x, y) de la posición actual sin filtrar
            magnitud_flujo: Magnitud del flujo óptico (qué tan rápido se mueve)
            angulo_flujo: Ángulo del flujo óptico en grados (hacia dónde se mueve)
        """
        if posicion is not None:
            self.contador_frames += 1
            
            #Guardo las últimas 500 posiciones
            self.historial_posiciones.append(posicion)

            if len(self.historial_posiciones) > 500:
                self.historial_posiciones = self.historial_posiciones[-500:]
            
            #Preparo variables para el análisis de dirección
            direccion_flujo = 0  
            angulo_actual = None 
  
            if len(self.historial_posiciones) > 1:
                angulo_actual = self.calcular_orientacion_desde_trayectoria(posicion, self.historial_posiciones[:-1])

                if angulo_actual is not None and self.ultimo_angulo is not None:
                    diferencia_angulo = angulo_actual - self.ultimo_angulo
                    
                    # Evito saltos de 360° convirtiendo a rango [-180, 180]
                    while diferencia_angulo > 180:
                        diferencia_angulo -= 360
                    while diferencia_angulo < -180:
                        diferencia_angulo += 360
                    
                    # Solo si el cambio es significativo (>5°)
                    if diferencia_angulo > 5:
                        direccion_flujo = 1  # ¡Girando horario!
                    elif diferencia_angulo < -5:
                        direccion_flujo = -1  # ¡Girando antihorario!
            
            self.actualizar_vueltas_basadas_en_cuadrantes(posicion, direccion_flujo, angulo_actual)
            
            #Cada 3 frames guardo el ángulo para análisis posterior
            if self.contador_frames % 3 == 0 and angulo_actual is not None:
                self.todos_angulos.append(angulo_actual)
                self.ultimo_angulo = angulo_actual
    
    def obtener_rotaciones_netas(self):
        """
        Número neto de rotaciones que hizo el ratón (horario - antihorario)
        
        Returns:
            float: Número de vueltas completas (positivo=horario, negativo=antihorario)
        """
        # Vueltas horarias - antihorarias = resultado neto
        return self.contador_real_horarias - self.contador_real_antihorarias
        
    def obtener_metricas_rotacion(self):
        """
        Metricas posibles sbre el comportamiento del ratón (entender el cmportamiento completo del ratón)
        
        Returns:
            dict: Diccionario con todas las métricas calculadas
        """
        # de radianes a grados
        angulo_total_grados = np.degrees(self.total_angle_accumulated_rad) if hasattr(self, 'total_angle_accumulated_rad') else 0
        
        direccion_predominante = "horario" if angulo_total_grados > 0 else "antihorario" if angulo_total_grados < 0 else "ninguna"
        
        segundos_moviendose = self.frames_moviendo / self.fps if self.fps > 0 else 0
        segundos_detenido = self.frames_detenido / self.fps if self.fps > 0 else 0
        
        return {
            'rotaciones_netas': self.contador_real_horarias - self.contador_real_antihorarias,  
            'vueltas_horarias': self.contador_real_horarias,  # Vueltas en sentido horario
            'vueltas_antihorarias': self.contador_real_antihorarias,  # Vueltas en sentido antihorario
            'grados_horarios': self.contador_real_horarias * 360.0, 
            'grados_antihorarios': self.contador_real_antihorarias * 360.0,  
            'grados_totales': (self.contador_real_horarias + self.contador_real_antihorarias) * 360.0,  # Total absoluto
            'tiempo_total': self.contador_frames / self.fps if self.contador_frames > 0 else 0,  # Duración total del vídeo
            'segundos_moviendose': segundos_moviendose,  # Tiempo que el ratón estuvo activo
            'segundos_detenido': segundos_detenido,  # Tiempo que el ratón estuvo quieto
            'frames_moviendose': self.frames_moviendo,  # Frames de movimiento
            'frames_detenido': self.frames_detenido,  # Frames de quietud
            'angulo_acumulado_rad': angulo_total_grados,  
            'angulo_acumulado_grados': angulo_total_grados,  
            'direccion_predominante': direccion_predominante,  # ¿Horario, antihorario o ninguno?
            'movimiento_total_horario_grados': max(0, angulo_total_grados),  
            'movimiento_total_antihorario_grados': max(0, -angulo_total_grados),  
            'registro_deteccion_vueltas': self.registro_deteccion_vueltas if hasattr(self, 'registro_deteccion_vueltas') else [],  
            'min_cuadrantes_para_vuelta': self.min_cuadrantes_para_vuelta,  
            'tamanio_cuadricula_cuadrantes': f"{self.num_cuadrantes_x}x{self.num_cuadrantes_y}",  # Tamaño de la grilla
            'suma_angulo_minima_para_vuelta': self.suma_min_angulo_para_vuelta  # Umbral angular usado
        }
    
    def contar_episodios(self, mascara_movimiento, duracion_minima_seg=1.0):
        """
        analiza el movimiento del ratón
        
        Args:
            mascara_movimiento: Array booleano indicando movimiento (True) o parada (False) por frame
            duracion_minima_seg: Duración mínima en segundos para contar un episodio
            
        Returns:
            tuple: (episodios_movimiento, episodios_parada) - número de episodios de cada tipo
        """
        mascara_movimiento = np.asarray(mascara_movimiento, dtype=bool)
        n = len(mascara_movimiento)

        frames_minimos = int(round(self.fps * duracion_minima_seg))

        # Contadores para cada tipo de episodio
        episodios_movimiento = 0
        episodios_parada = 0

        i = 0
        while i < n:
            # ¿Está moviéndose o quieto?
            estado = mascara_movimiento[i]
            
            # Busco hasta que cambie el estado
            j = i + 1
            while j < n and mascara_movimiento[j] == estado:
                j += 1
            
            # Cuánto duró este episodio?
            longitud = j - i

            if longitud >= frames_minimos:
                if estado:
                    episodios_movimiento += 1  
                else:
                    episodios_parada += 1 

            i = j

        # conteo de ambos tipos de episodios
        return episodios_movimiento, episodios_parada
    
    def actualizar_estado_movimiento(self, posicion_actual):
        """
        Este método decide si el ratón está moviéndose o está quieto en este momento
        
        Args:
            posicion_actual: Tupla (x, y) con la posición actual del ratón
        """
        # Si no tengo posición, está quieto
        if posicion_actual is None:
            self.esta_moviendo = False
            return

        if self.ultima_posicion is None:
            self.ultima_posicion = posicion_actual
            self.esta_moviendo = False
            return

        distancia = np.sqrt(
            (posicion_actual[0] - self.ultima_posicion[0])**2 + 
            (posicion_actual[1] - self.ultima_posicion[1])**2
        )
        
        # ¿Se movió lo suficiente como para considerarlo "movimiento"?
        self.esta_moviendo = distancia >= self.umbral_movimiento

        if self.esta_moviendo:
            self.frames_moviendo += 1  
        else:
            self.frames_detenido += 1  

        self.ultima_posicion = posicion_actual
    
    def dibujar_estado_movimiento(self, frame, posicion=(10, 30)):
        """
        Este método dibuja en el vídeo si el ratón está moviéndose o parado
        
        Args:
            frame: Frame de OpenCV donde dibujar
            posicion:(x, y) con la posición donde dibujar el texto
            
        Returns:
            np.ndarray: Frame con el estado dibujado
        """
        if self.esta_moviendo:
            texto = "EN MOVIMIENTO"
            color = (0, 255, 0)  # verde
        else:
            texto = "PARADO"
            color = (0, 0, 255)  # rojo

        tamanio_texto = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

        superposicion = frame.copy()
        cv2.rectangle(superposicion, 
                     (posicion[0] - 5, posicion[1] - tamanio_texto[1] - 5),
                     (posicion[0] + tamanio_texto[0] + 5, posicion[1] + 5),
                     (0, 0, 0), -1)
        cv2.addWeighted(superposicion, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, texto, posicion, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame
    
    def calcular_orientacion_desde_trayectoria(self, posicion_actual, posiciones_anteriores, tamanio_ventana=10):
        """
        Este método calcula hacia dónde apunta el ratón basándose en su trayectoria reciente
        
        Args:
            posicion_actual: Posición actual (x, y)
            posiciones_anteriores: Lista de posiciones anteriores
            tamanio_ventana: Número de posiciones anteriores a considerar
            
        Returns:
            float: Ángulo en grados (-180 a 180), o None si no hay suficientes datos
        """
        if len(posiciones_anteriores) < tamanio_ventana:
            return None

        posiciones_recientes = posiciones_anteriores[-tamanio_ventana:]
        posiciones_recientes.append(posicion_actual)

        if len(posiciones_recientes) < 2:
            return None

        posicion_inicio = posiciones_recientes[0]
        posicion_fin = posiciones_recientes[-1]
        
        # Calculo cuánto se movió en x e y
        dx = posicion_fin[0] - posicion_inicio[0]
        dy = posicion_fin[1] - posicion_inicio[1]
        
        # Si se movió muy poco=ángulo anterior
        distancia = np.sqrt(dx*dx + dy*dy)
        if distancia < 5:
            return self.ultimo_angulo

        angulo = np.degrees(np.arctan2(dy, dx))
        return angulo

    def mostrar_contador_vueltas_en_frame(self, frame, posicion=(10, 100)):
        """
        Este método dibuja en el vídeo un resumen completo de todas las vueltas detectadas
        
        Args:
            frame: Imagen/frame de OpenCV
            posicion: Posición inicial del texto (x, y)
            
        Returns:
            np.ndarray: Frame modificado con el contador dibujado
        """
        #Coordenadas donde empezar a dibujar
        x, y = posicion

        vueltas_horarias = self.contador_real_horarias
        vueltas_antihorarias = self.contador_real_antihorarias
        vueltas_totales = vueltas_horarias + vueltas_antihorarias
        vueltas_netas = vueltas_horarias - vueltas_antihorarias

        suma_angulos = sum(abs(a) for a in self.cambios_angulo_en_vuelta) if hasattr(self, 'cambios_angulo_en_vuelta') else 0
        suma_angulos_con_signo = sum(self.cambios_angulo_en_vuelta) if hasattr(self, 'cambios_angulo_en_vuelta') else 0
        direccion_suma = "+" if suma_angulos_con_signo > 0 else "-" if suma_angulos_con_signo < 0 else "="

        cv2.putText(frame, f"Vueltas horarias: {vueltas_horarias}", 
                   (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Vueltas antihorarias: {vueltas_antihorarias}", 
                   (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Vueltas totales: {vueltas_totales}", 
                   (x, y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Vueltas netas: {vueltas_netas:+d}", 
                   (x, y+90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Suma angulos: {suma_angulos:.1f}/270.0 {direccion_suma}", 
                   (x, y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cuadrantes_unicos = len(set(self.cuadrantes_cruzados_en_vuelta)) if hasattr(self, 'cuadrantes_cruzados_en_vuelta') else 0
        cv2.putText(frame, f"Cuadrantes: {cuadrantes_unicos}/{self.min_cuadrantes_para_vuelta}", 
                   (x, y+145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, "(Cuadrantes + Angulos + Tiempo)", 
                   (x, y+170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        return frame

