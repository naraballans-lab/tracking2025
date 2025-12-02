"""
Pipeline principal de procesamiento de vídeos de ratones
Aquí coordino todo el sistema: detector, tracker y analyzer
Tuve que crear dos pipelines: uno simple para vídeos individuales y otro complejo para collages
"""

import cv2  # Para procesamiento de video
import numpy as np  # Para operaciones matemáticas
import os  
from src.detector import MouseDetector  
from src.tracker import MouseTracker 
from src.analyzer import RotationAnalyzer  
from datetime import datetime  


def resize_window(image, window_name, scale_percent=50, window_index=0):
    """
    Redimensiona ventanas para que quepan en la pantalla
    Tuve que añadir esto porque con ventanas de 640x480 no cabían todas en mi pantalla
    
    Args:
        image: Imagen a mostrar
        window_name: Nombre de la ventana
        scale_percent: Porcentaje de escala (50 = mitad del tamaño)
        window_index: Índice de la ventana para posicionamiento en cuadrícula (basado en 3 columnas)
    """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow(window_name, resized)
    
    x_offset = 100 + (width * (window_index % 3))
    y_offset = 50 + (height * (window_index // 3))
    cv2.moveWindow(window_name, x_offset, y_offset)

def process_video(video_path, output_path=None, show_visualization=True, window_scale=50):
    """
    Este es el método que procesa vídeos individuales
    Coordina detector -tracker -analyzer en cada frame
    
    Args:
        video_path: Ruta al archivo de video a procesar
        output_path: Ruta para guardar el video procesado (opcional) - genera vídeos MUY pesados
        show_visualization: Si True, muestra la visualización en tiempo real
        window_scale: Escala de las ventanas en porcentaje (default 50%)
        
    Returns:
        dict: Diccionario con métricas de rotación y trayectoria suavizada
    """
    # abro el vídeo
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # leo el primer frame para saber dimensiones
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("No se pudo leer el primer frame del video")
    frame_height, frame_width = first_frame.shape[:2]
    
    print(f"\n=== INFORMACIÓN DEL VIDEO ===")
    print(f"Resolución: {frame_width}x{frame_height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total de frames: {total_frames}")
    print(f"Duración estimada: {total_frames/fps:.1f} segundos")

    detector = MouseDetector(tasa_aprendizaje=0.001)  
    tracker = MouseTracker(dt=1/fps)  # Kalman sincronizado con FPS del vídeo
    analyzer = RotationAnalyzer(ancho_frame=frame_width, alto_frame=frame_height)
    analyzer.fps = fps  
    
    # Creo imágenes negras donde dibujaré el camino del ratón
    trajectory_map = np.zeros_like(first_frame)  # Trayectoria suavizada (Kalman)
    raw_trajectory_map = np.zeros_like(first_frame)  # Trayectoria cruda (sin filtrar)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  

    writer = None
    if output_path:
        output_width = frame_width * 4  # Grid 4x2 de visualizaciones
        output_height = frame_height * 2
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec XVID (compresión decente)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    # Flags para saber el estado del procesamiento
    first_detection = True  
    quadrants_configured = False 
    frame_count = 0  
    detections_count = 0  

    print("\n=== PROCESANDO VIDEO ===")
    print("Presiona 'q' en cualquier momento para detener el análisis\n")
    
    # Loop principal: proceso frame por frame hasta el final del vídeo
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Preprocesamiento: aplico blur y conversión a escala de grises
        processed_frame = detector.preprocesar_frame(frame)
        
        # Control de teclado: permito salir con 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nAnálisis detenido por el usuario")
            break
        
        # Detección del ratón: aquí llamo al detector para encontrar al ratón en este frame
        centroid, contour, orientation, binary_mask = detector.detectar_raton(processed_frame)
        
        # Si hay detección: solo proceso si encontroal ratón
        if centroid is not None:
            detections_count += 1  
            
            # Primera detección: inicializo el tracker con la primera posición
            if first_detection:
                tracker.inicializar(centroid)
                first_detection = False
                print(f"[Frame {frame_count}] Primera detección del ratón en posición {centroid}")
            
            # Configuración automática de cuadrantes: espero 7 segundos para que el ratón esté bien visible
            current_time = analyzer.contador_frames / fps
            if not quadrants_configured and current_time >= 7.0:
                if contour is not None:
                    x, y, w, h = cv2.boundingRect(contour)  # Obtengo dimensiones del ratón
                    analyzer.configurar_cuadrantes_desde_tam_raton(w, h) 
                    quadrants_configured = True
            
            # Actualizo el tracker: Kalman filtra la posición para suavizar la trayectoria
            state = tracker.actualizar(centroid)
            filtered_position = (
                int(float(state[0].item())),  # Convierto de matriz NumPy a int
                int(float(state[1].item()))
            )
            
            # analisis de rotaciones=Envío la posición al analyzer
            analyzer.agregar_posicion_y_calcular_vueltas(
                centroid,  # Posición cruda (sin filtrar)
                magnitud_flujo=detector.magnitud_flujo,  # Velocidad del movimiento
                angulo_flujo=detector.angulo_flujo  # Dirección del movimiento
            )
            
            # Actualizo estado de movimiento: ¿Está quieto o moviéndose?
            analyzer.actualizar_estado_movimiento(centroid)
            
            # Visualización: solo si el usuario quiere ver o grabar el vídeo
            # Esto es totalmente opcional - solo sirve para debugging y presentaciones
            if show_visualization or writer:
                # === FRAME 1: ORIGINAL ===
                original_display = frame.copy()
                cv2.putText(original_display, "1. Frame Original", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # === FRAME 2: MÁSCARA BINARIA ===
                # Aquí muestro la segmentación del ratón: Blanco=ratón, Negro=fondo
                mask_display = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
                cv2.putText(mask_display, "2. Mascara Binaria", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # === FRAME 3: FLUJO ÓPTICO ===
                # Visualización del movimiento
                flow_display = detector.obtener_visualizacion_flujo_optico(processed_frame.shape)
                cv2.putText(flow_display, "3. Flujo Optico", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(flow_display, f"Magnitud: {detector.magnitud_flujo:.1f}", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # === FRAME 4: DETECCIÓN Y SEGUIMIENTO ===
                # Contorno verde + posición roja cruda + posición azul filtrada
                tracking_display = frame.copy()
                cv2.drawContours(tracking_display, [contour], -1, (0, 255, 0), 2)  # Contorno verde
                cv2.circle(tracking_display, centroid, 5, (0, 0, 255), -1)  # Rojo = posición detectada
                cv2.circle(tracking_display, filtered_position, 5, (255, 0, 0), -1)  # Azul = Kalman
                cv2.putText(tracking_display, "4. Deteccion y Seguimiento", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                tracking_display = analyzer.dibujar_estado_movimiento(tracking_display, posicion=(10, 60))
                
                # === FRAME 5: CUADRANTES ===
                # Cómo divido el espacio para detectar rotaciones
                quadrants_display = analyzer.dibujar_cuadrantes(frame.copy())
                cv2.circle(quadrants_display, centroid, 8, (255, 0, 255), -1) 
                cv2.putText(quadrants_display, "5. Cuadrantes", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if analyzer.cuadrante_actual is not None:
                    cv2.putText(quadrants_display, f"Cuadrante actual: {analyzer.cuadrante_actual}", (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    # Contador de cuadrantes únicos visitados: necesita visitar suficientes para contar vuelta
                    unique_quads = len(set(analyzer.cuadrantes_cruzados_en_vuelta))
                    cv2.putText(quadrants_display, f"Cuadrantes: {unique_quads}/{analyzer.min_cuadrantes_para_vuelta}", (10, 90), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Temporizador: muestro cuánto tiempo lleva en la vuelta actual
                    if len(analyzer.marcas_tiempo_cuadrantes) > 0:
                        current_time = analyzer.contador_frames / analyzer.fps
                        elapsed_time = current_time - analyzer.marcas_tiempo_cuadrantes[0]
                        remaining_time = max(0, analyzer.tiempo_max_para_vuelta - elapsed_time)
                        # Verde=mucho tiempo, Naranja=poco tiempo, Rojo=sin tiempo
                        color = (0, 255, 0) if remaining_time > 1.0 else (0, 165, 255) if remaining_time > 0.5 else (0, 0, 255)
                        cv2.putText(quadrants_display, f"Tiempo: {elapsed_time:.1f}s / {analyzer.tiempo_max_para_vuelta:.1f}s", (10, 120), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Secuencia de cuadrantes: muestro los últimos 5 cuadrantes visitados
                    if len(analyzer.cuadrantes_cruzados_en_vuelta) > 0:
                        seq_str = str(analyzer.cuadrantes_cruzados_en_vuelta[-5:])
                        cv2.putText(quadrants_display, f"Secuencia: {seq_str}", (10, 150), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # === FRAME 6: TRAYECTORIA FILTRADA ===
                # Mapa de calor: voy dibujando la trayectoria suavizada por Kalman
                cv2.circle(trajectory_map, filtered_position, 1, (255, 0, 0), -1)
                trajectory_display = trajectory_map.copy()
                cv2.putText(trajectory_display, "6. Trayectoria Filtrada", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # === FRAME 7: TRAYECTORIA SIN FILTRAR ===
                # Cruda=trayectoria es más ruidosa pero muestra la detección real
                cv2.circle(raw_trajectory_map, centroid, 1, (0, 0, 255), -1)
                raw_trajectory_display = raw_trajectory_map.copy()
                cv2.putText(raw_trajectory_display, "7. Trayectoria Sin Filtrar", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # === FRAME 8: ANÁLISIS DE ROTACIÓN ===
                # Orientación y el análisis angular
                rotation_display = frame.copy()
                
                # Uso la trayectoria reciente para saber hacia dónde va
                trajectory_orientation = None
                if len(analyzer.historial_posiciones) > 1:
                    trajectory_orientation = analyzer.calcular_orientacion_desde_trayectoria(
                        centroid, analyzer.historial_posiciones[:-1], tamanio_ventana=5)
                
                # Dibujo flecha de dirección si tengo orientación calculada
                if trajectory_orientation is not None:
                    length = 50
                    angle_rad = np.radians(trajectory_orientation)
                    end_point = (
                        int(centroid[0] + length * np.cos(angle_rad)),
                        int(centroid[1] + length * np.sin(angle_rad))
                    )
                    cv2.line(rotation_display, centroid, end_point, (255, 255, 0), 2)  # Flecha cyan

                if len(analyzer.historial_posiciones) > 10:
                    recent_positions = analyzer.historial_posiciones[-10:]
                    for i in range(1, len(recent_positions)):
                        cv2.line(rotation_display, recent_positions[i-1], recent_positions[i], 
                               (0, 255, 255), 2)  # Líneas amarillas
                
                cv2.putText(rotation_display, "8. Analisis de Rotacion", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # Contador de vueltas
                rotation_display = analyzer.mostrar_contador_vueltas_en_frame(rotation_display, posicion=(10, 60))

                # Si el usuario quiere ver el análisis en tiempo real
                if show_visualization:
                    # Redimensiono cada ventana y la coloco en su posición (0-7)
                    resize_window(original_display, '1. Frame Original', window_scale, 0)
                    resize_window(mask_display, '2. Mascara Binaria', window_scale, 1)
                    resize_window(flow_display, '3. Flujo Optico', window_scale, 2)
                    resize_window(tracking_display, '4. Deteccion y Seguimiento', window_scale, 3)
                    resize_window(quadrants_display, '5. Cuadrantes', window_scale, 4)
                    resize_window(trajectory_display, '6. Trayectoria Filtrada', window_scale, 5)
                    resize_window(raw_trajectory_display, '7. Trayectoria Sin Filtrar', window_scale, 6)
                    resize_window(rotation_display, '8. Analisis de Rotacion', window_scale, 7)
                
                # Guardado de video: si el usuario quiere guardar el vídeo procesado
                if writer:
                    try:
                        # Función auxiliar: asegura que todo esté en formato BGR de 3 canales
                        def ensure_bgr(img):
                            if len(img.shape) == 2:  # Si es escala de grises
                                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                            return img

                        original_display = ensure_bgr(cv2.resize(original_display, (frame_width, frame_height)))
                        mask_display = ensure_bgr(cv2.resize(mask_display, (frame_width, frame_height)))
                        flow_display = ensure_bgr(cv2.resize(flow_display, (frame_width, frame_height)))
                        tracking_display = ensure_bgr(cv2.resize(tracking_display, (frame_width, frame_height)))
                        quadrants_display = ensure_bgr(cv2.resize(quadrants_display, (frame_width, frame_height)))
                        trajectory_display = ensure_bgr(cv2.resize(trajectory_display, (frame_width, frame_height)))
                        raw_trajectory_display = ensure_bgr(cv2.resize(raw_trajectory_display, (frame_width, frame_height)))
                        rotation_display = ensure_bgr(cv2.resize(rotation_display, (frame_width, frame_height)))
                        
                        top_row = np.hstack((original_display, mask_display, flow_display, tracking_display))
                        bottom_row = np.hstack((quadrants_display, trajectory_display, raw_trajectory_display, rotation_display))
                        combined_display = np.vstack((top_row, bottom_row))
                        writer.write(combined_display)  
                    except Exception as e:
                        print(f"Error al escribir frame: {e}")
        
        # Indicador de progreso: muestro cada 100 frames para no saturar la consola
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"Progreso: {frame_count}/{total_frames} frames ({progress:.1f}%)")
    

    # Ahora calculo las estadísticas finales
    
    print(f"\n=== PROCESAMIENTO COMPLETADO ===")
    print(f"Frames procesados: {frame_count}/{total_frames}")
    print(f"Detecciones exitosas: {detections_count} ({(detections_count/frame_count*100):.1f}%)")
    print(f"Tiempo total de video: {frame_count/fps:.1f} segundos")
    
    # Suavizado final: aplico Kalman a toda la trayectoria completa
    smoothed_states = tracker.suavizar_trayectoria()
    # Métricas: obtengo todas las estadísticas calculadas por el analyzer
    metrics = analyzer.obtener_metricas_rotacion()
    
    # === REPORTE DE RESULTADOS ===
    # Aquí muestro todos los datos que recopilé
    print("\n=== RESULTADOS DEL ANÁLISIS DE ROTACIONES ===")
    print(f"Duración total del video: {metrics['total_time']:.1f} segundos")
    print(f"\nConfiguración de detección:")
    print(f"- Grilla de cuadrantes: {metrics.get('quadrant_grid_size', '4x4')}")
    print(f"- Cuadrantes mínimos para vuelta: {metrics.get('min_quadrants_for_turn', 4)}")
    print(f"- Suma de ángulos mínima: {metrics.get('min_angle_sum_for_turn', 270.0)}°")
    print("\nRotaciones:")
    print(f"- Rotaciones netas: {metrics['net_rotations']:.2f} vueltas")
    clockwise_turns = metrics['clockwise_degrees'] / 360.0
    counterclockwise_turns = metrics['counterclockwise_degrees'] / 360.0
    print(f"- Vueltas en sentido horario: {clockwise_turns:.2f}")
    print(f"- Vueltas en sentido antihorario: {counterclockwise_turns:.2f}")
    print(f"- Vueltas totales (suma absoluta): {(clockwise_turns + counterclockwise_turns):.2f}")
    
    print("\nEstadísticas de movimiento:")
    print(f"- Tiempo en movimiento: {metrics.get('seconds_moving', 0):.1f} segundos")
    print(f"- Tiempo parado: {metrics.get('seconds_stopped', 0):.1f} segundos")
    print(f"- Frames en movimiento: {metrics.get('frames_moving', 0)}")
    print(f"- Frames parado: {metrics.get('frames_stopped', 0)}")
    
    if output_path:
        results_path = output_path.replace('.avi', '_resultados.txt')
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write("=== RESULTADOS DEL ANÁLISIS DE ROTACIONES ===\n")
            f.write(f"Duración total del video: {metrics['total_time']:.1f} segundos\n\n")
            
            f.write("Configuración de detección:\n")
            f.write(f"- Grilla de cuadrantes: {metrics.get('quadrant_grid_size', '4x4')}\n")
            f.write(f"- Cuadrantes mínimos para vuelta: {metrics.get('min_quadrants_for_turn', 4)}\n")
            f.write(f"- Suma de ángulos mínima: {metrics.get('min_angle_sum_for_turn', 270.0)}°\n\n")

            f.write("Rotaciones:\n")
            f.write(f"- Rotaciones netas: {metrics['net_rotations']:.2f} vueltas\n")
            f.write(f"- Vueltas en sentido horario: {clockwise_turns:.2f}\n")
            f.write(f"- Vueltas en sentido antihorario: {counterclockwise_turns:.2f}\n")
            f.write(f"- Vueltas totales (suma absoluta): {(clockwise_turns + counterclockwise_turns):.2f}\n")
            f.write(f"- Grados totales recorridos: {metrics['total_degrees']:.2f}\n\n")
            
            f.write("Estadísticas de movimiento:\n")
            f.write(f"- Tiempo en movimiento: {metrics.get('seconds_moving', 0):.1f} segundos\n")
            f.write(f"- Tiempo parado: {metrics.get('seconds_stopped', 0):.1f} segundos\n")
            f.write(f"- Frames en movimiento: {metrics.get('frames_moving', 0)}\n")
            f.write(f"- Frames parado: {metrics.get('frames_stopped', 0)}\n\n")

            f.write("Detalle de vueltas detectadas:\n")
            f.write("-" * 80 + "\n")
            
            if 'turn_detection_log' in metrics and len(metrics['turn_detection_log']) > 0:
                for log_entry in metrics['turn_detection_log']:
                    f.write(log_entry + "\n")
            else:
                f.write("No se detectaron vueltas o el log no está disponible.\n")
            
            f.write("-" * 80 + "\n\n")
            
    cap.release()
    if writer:
        writer.release()
    
    cv2.destroyAllWindows()
    
    return {
        'rotation_metrics': analyzer.obtener_metricas_rotacion(),
        'trajectory': smoothed_states
    }

def process_collage_video(video_path, output_path=None, window_scale=40, separate_windows=False):
    """
    Modo collage 2x2 = 4 videos juntos 
    Esta función divide el frame en 4 cuadrantes y los analiza por separado
    
    Args:
        video_path: Ruta al video collage con 4 videos en disposición 2x2
        output_path: Ruta para guardar el video procesado (opcional)
        window_scale: Escala de visualización en porcentaje
        separate_windows: Si True, muestra cada cuadrante en ventana separada
        
    Returns:
        dict: Diccionario con métricas de cada cuadrante
    """
    print("\n=== PROCESAMIENTO DE VIDEO COLLAGE 2x2 ===")
    print(f"Video: {video_path}\n")
    
    # Abro el video collage: un solo archivo con 4 videos integrados
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")
    
    # Obtengo propiedades: FPS y total de frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Leo el primer frame para saber las dimensiones
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("No se pudo leer el primer frame")
    
    # Calculo dimensiones de cada cuadrante: divido entre 2 en ambas direcciones
    full_height, full_width = first_frame.shape[:2]
    quadrant_height = full_height // 2
    quadrant_width = full_width // 2
    
    print(f"Resolución completa: {full_width}x{full_height}")
    print(f"Resolución por cuadrante: {quadrant_width}x{quadrant_height}")
    print(f"FPS: {fps}")
    
    # Área de detección: calculo límites basados en el tamaño del cuadrante
    quadrant_area = quadrant_width * quadrant_height
    min_area_per_quad = int(quadrant_area * 0.0016)
    max_area_per_quad = int(quadrant_area * 0.039)
    print(f"Área de detección por cuadrante: {min_area_per_quad} - {max_area_per_quad} px")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Vuelvo al inicio
    
    # Creo 4 pipelines independientes: uno para cada cuadrante del collage
    processors = []
    for i in range(4):
        # ¡Instancio detector, tracker y analyzer para este cuadrante!
        detector = MouseDetector(
            tasa_aprendizaje=0.001,
            tamanio_frame=(quadrant_width, quadrant_height)
        )
        tracker = MouseTracker(dt=1/fps)
        analyzer = RotationAnalyzer(ancho_frame=quadrant_width, alto_frame=quadrant_height, id_video=i+1)
        analyzer.fps = fps
        
        # Todo en un diccionario
        processors.append({
            'id': i + 1,
            'detector': detector,
            'tracker': tracker,
            'analyzer': analyzer,
            'trajectory_map': np.zeros((quadrant_height, quadrant_width, 3), dtype=np.uint8),
            'raw_trajectory_map': np.zeros((quadrant_height, quadrant_width, 3), dtype=np.uint8),
            'first_detection': True,
            'quadrants_configured': False,
            'frame_count': 0,
            'detections_count': 0
        })
    
    writer = None
    if output_path:
        output_width = quadrant_width * 4 * 2
        output_height = quadrant_height * 2 * 2
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    print("\nProcesando... Presiona 'q' para detener\n")
    
    frame_count = 0
    shared_mouse_size = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        #  Superior-izq, superior-der, inferior-izq, inferior-der == 4 cuadrantes
        mid_h = full_height // 2
        mid_w = full_width // 2
        quadrants = [
            frame[0:mid_h, 0:mid_w],              # Cuadrante 1: Superior izquierdo
            frame[0:mid_h, mid_w:full_width],     # Cuadrante 2: Superior derecho
            frame[mid_h:full_height, 0:mid_w],    # Cuadrante 3: Inferior izquierdo
            frame[mid_h:full_height, mid_w:full_width]  # Cuadrante 4: Inferior derecho
        ]
        
        processed_views = []
        
        # Proceso cada cuadrante: itero sobre los 4 procesadores y sus frames
        for proc, quad_frame in zip(processors, quadrants):
            proc['frame_count'] += 1
            
            # Detección en este cuadrante: igual que en el modo individual
            processed_frame = proc['detector'].preprocesar_frame(quad_frame)
            centroid, contour, orientation, binary_mask = proc['detector'].detectar_raton(processed_frame)
            
            if centroid is not None:
                proc['detections_count'] += 1
                
                # Primera detección: inicializo el tracker de este cuadrante
                if proc['first_detection']:
                    proc['tracker'].inicializar(centroid)
                    proc['first_detection'] = False
                
                # Detección compartida del tamaño: uso el mismo tamaño para los 4 ratones
                current_time = proc['frame_count'] / fps
                if shared_mouse_size is None and current_time >= 7.0:
                    if contour is not None:
                        x, y, w, h = cv2.boundingRect(contour)
                        shared_mouse_size = (w, h)
                        print(f"[Collage] Tamaño de ratón compartido detectado: {w}x{h} px")

                if not proc['quadrants_configured'] and shared_mouse_size is not None:
                    w, h = shared_mouse_size
                    proc['analyzer'].configurar_cuadrantes_desde_tam_raton(w, h)
                    proc['quadrants_configured'] = True
                
                # Actualizo tracker: filtrado Kalman
                state = proc['tracker'].actualizar(centroid)
                filtered_pos = (
                    int(float(state[0].item())),
                    int(float(state[1].item()))
                )
                
                # Análisis de rotaciones: para este cuadrante específico
                proc['analyzer'].agregar_posicion_y_calcular_vueltas(
                    posicion=centroid,
                    magnitud_flujo=proc['detector'].magnitud_flujo,
                    angulo_flujo=proc['detector'].angulo_flujo
                )
                
                proc['analyzer'].actualizar_estado_movimiento(centroid)
                
                # Dibujo trayectorias
                cv2.circle(proc['raw_trajectory_map'], centroid, 1, (0, 0, 255), -1)
                cv2.circle(proc['trajectory_map'], filtered_pos, 1, (255, 0, 0), -1)
                
                # Visualización
                original_display = quad_frame.copy()
                
                tracking_display = quad_frame.copy()
                if contour is not None:
                    cv2.drawContours(tracking_display, [contour], -1, (0, 255, 0), 2)
                cv2.circle(tracking_display, centroid, 5, (0, 0, 255), -1)  # Rojo = detección
                cv2.circle(tracking_display, filtered_pos, 5, (255, 0, 0), -1)  # Azul = Kalman
                
                tracking_display = proc['analyzer'].dibujar_estado_movimiento(tracking_display, position=(5, 20))

                if orientation is not None:
                    length = 30
                    angle_rad = np.radians(orientation)
                    end_point = (
                        int(centroid[0] + length * np.cos(angle_rad)),
                        int(centroid[1] + length * np.sin(angle_rad))
                    )
                    cv2.arrowedLine(tracking_display, centroid, end_point, (255, 255, 0), 2, tipLength=0.3)
                
                mask_display = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
                flow_display = proc['detector'].obtener_visualizacion_flujo_optico(processed_frame.shape)
                
                # Grilla de cuadrantes
                quadrants_display = proc['analyzer'].dibujar_cuadrantes(quad_frame.copy())
                cv2.circle(quadrants_display, centroid, 8, (255, 0, 255), -1)
                if proc['analyzer'].cuadrante_actual is not None:
                    unique_quads = len(set(proc['analyzer'].cuadrantes_cruzados_en_vuelta))
                    cv2.putText(quadrants_display, f"Q:{proc['analyzer'].cuadrante_actual} U:{unique_quads}/{proc['analyzer'].min_cuadrantes_para_vuelta}", 
                               (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Análisis de rotación: similar al modo individual
                rotation_display = quad_frame.copy()
                if len(proc['analyzer'].historial_posiciones) > 10:
                    recent_positions = proc['analyzer'].historial_posiciones[-10:]
                    for i in range(1, len(recent_positions)):
                        cv2.line(rotation_display, recent_positions[i-1], recent_positions[i], 
                               (0, 255, 255), 2)
                cv2.circle(rotation_display, centroid, 5, (0, 0, 255), -1)
                
                # Orientación desde trayectoria: calculo hacia dónde va el ratón
                trajectory_orientation = None
                if len(proc['analyzer'].historial_posiciones) > 1:
                    trajectory_orientation = proc['analyzer'].calcular_orientacion_desde_trayectoria(
                        centroid, proc['analyzer'].historial_posiciones[:-1], tamanio_ventana=5)
                
                if trajectory_orientation is not None:
                    length = 40
                    angle_rad = np.radians(trajectory_orientation)
                    end_point = (
                        int(centroid[0] + length * np.cos(angle_rad)),
                        int(centroid[1] + length * np.sin(angle_rad))
                    )
                    cv2.line(rotation_display, centroid, end_point, (255, 255, 0), 2)
                
                rotation_display = proc['analyzer'].mostrar_contador_vueltas_en_frame(rotation_display, posicion=(5, 15))
                original_display = quad_frame.copy()
                
            else:
                original_display = quad_frame.copy()
                tracking_display = quad_frame.copy()
                mask_display = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
                flow_display = proc['detector'].obtener_visualizacion_flujo_optico(processed_frame.shape)
                quadrants_display = proc['analyzer'].dibujar_cuadrantes(quad_frame.copy())
                rotation_display = quad_frame.copy()
            
            # Métricas actuales
            metrics = proc['analyzer'].obtener_metricas_rotacion()
            cw_turns = metrics['clockwise_degrees'] / 360.0
            ccw_turns = metrics['counterclockwise_degrees'] / 360.0
            
            # Etiquetas de identificación: marco cada frame con el número de cuadrante
            cv2.putText(tracking_display, f"Q{proc['id']}: Deteccion", (5, quadrant_height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(mask_display, f"Q{proc['id']}: Mascara", (5, quadrant_height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(flow_display, f"Q{proc['id']}: Flujo", (5, quadrant_height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(quadrants_display, f"Q{proc['id']}: Cuadrantes", (5, quadrant_height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(rotation_display, f"Q{proc['id']}: Rotacion", (5, quadrant_height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(original_display, f"Q{proc['id']}: Original", (5, quadrant_height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            traj_display = proc['trajectory_map'].copy()
            cv2.putText(traj_display, f"Q{proc['id']}: Tray. Filtrada", (5, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(traj_display, f"H:{cw_turns:.1f}", (5, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            cv2.putText(traj_display, f"A:{ccw_turns:.1f}", (5, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 255), 1)
            
            raw_traj_display = proc['raw_trajectory_map'].copy()
            cv2.putText(raw_traj_display, f"Q{proc['id']}: Tray. Cruda", (5, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Combino las 8 vistas: creo una grilla 2x4 para cada cuadrante
            row1 = np.hstack([original_display, tracking_display, mask_display, flow_display])
            row2 = np.hstack([quadrants_display, rotation_display, traj_display, raw_traj_display])
            
            combined = np.vstack([row1, row2])
            processed_views.append(combined)
        
        # combino las 4 vistas de cuadrantes en una imagen 2x2
        top_row = np.hstack([processed_views[0], processed_views[1]])
        bottom_row = np.hstack([processed_views[2], processed_views[3]])
        output_display = np.vstack([top_row, bottom_row])
        
        # guardo el frame: si el usuario quiere grabar el resultado
        if writer:
            writer.write(output_display)
        
        # visualización: dos modos, ventanas separadas o todo junto
        if separate_windows:
            # modo separado: cada cuadrante en su propia ventana
            for i, combined in enumerate(processed_views):
                display_h = int(combined.shape[0] * window_scale / 100)
                display_w = int(combined.shape[1] * window_scale / 100)
                display_frame = cv2.resize(combined, (display_w, display_h))
                
                window_name = f'Video {i+1}'
                cv2.imshow(window_name, display_frame)

                x_offset = 10 + (display_w + 20) * (i % 2)
                y_offset = 50 + (display_h + 50) * (i // 2)
                cv2.moveWindow(window_name, x_offset, y_offset)
        else:
            # modo unificado: todo en una sola ventana gigante
            display_h = int(output_display.shape[0] * window_scale / 100)
            display_w = int(output_display.shape[1] * window_scale / 100)
            display_frame = cv2.resize(output_display, (display_w, display_h))
            cv2.imshow('Análisis Collage 2x2', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nDetenido por el usuario")
            break
        
        # reporte de progreso: cada 100 frames
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"Progreso: {frame_count}/{total_frames} frames ({progress:.1f}%)")

    print(f"\nProcesamiento completado: {frame_count} frames")
    
    # recopilo resultados: junto las métricas de los 4 cuadrantes
    results = {}
    print("\n=== RESULTADOS FINALES ===\n")
    
    quadrant_names = ["Superior Izq.", "Superior Der.", "Inferior Izq.", "Inferior Der."]

    for proc, name in zip(processors, quadrant_names):
        metrics = proc['analyzer'].obtener_metricas_rotacion()
        results[f'quadrant_{proc["id"]}'] = metrics
        
        cw_turns = metrics['clockwise_degrees'] / 360.0
        ccw_turns = metrics['counterclockwise_degrees'] / 360.0
        total_turns = cw_turns + ccw_turns
        
        print(f"Cuadrante {proc['id']} ({name}):")
        print(f"  - Vueltas horarias: {cw_turns:.2f}")
        print(f"  - Vueltas antihorarias: {ccw_turns:.2f}")
        print(f"  - Vueltas totales: {total_turns:.2f}")
        print(f"  - Tiempo en movimiento: {metrics.get('seconds_moving', 0):.1f}s")
        print(f"  - Tiempo parado: {metrics.get('seconds_stopped', 0):.1f}s")
    
    # guardo reporte en archivo 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"C:\\tfg\\Bibliografía\\pruebass\\pruebasnuevas\\reporte_collage_{timestamp}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== REPORTE DE ANÁLISIS DE VIDEO COLLAGE 2x2 ===\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Frames procesados: {frame_count}\n\n")
        
        # escribo métricas: para cada cuadrante en el reporte
        for proc, name in zip(processors, quadrant_names):
            metrics = results[f'quadrant_{proc["id"]}']
            cw = metrics['clockwise_degrees'] / 360.0
            ccw = metrics['counterclockwise_degrees'] / 360.0
            
            f.write(f"Cuadrante {proc['id']} ({name}):\n")
            
            # Configuración de detección
            f.write(f"Configuración de detección:\n")
            f.write(f"  - Grilla de cuadrantes: {metrics.get('quadrant_grid_size', 'N/A')}\n")
            f.write(f"  - Cuadrantes mínimos para vuelta: {metrics.get('min_quadrants_for_turn', 'N/A')}\n")
            f.write(f"  - Suma de ángulos mínima: {metrics.get('min_angle_sum_for_turn', 270.0)}°\n\n")
            
            f.write(f"  - Vueltas horarias: {cw:.2f}\n")
            f.write(f"  - Vueltas antihorarias: {ccw:.2f}\n")
            f.write(f"  - Vueltas totales: {cw + ccw:.2f}\n")
            f.write(f"  - Grados horarios: {metrics['clockwise_degrees']:.2f}°\n")
            f.write(f"  - Grados antihorarios: {metrics['counterclockwise_degrees']:.2f}°\n")
            f.write(f"  - Tiempo en movimiento: {metrics.get('seconds_moving', 0):.1f} segundos\n")
            f.write(f"  - Tiempo parado: {metrics.get('seconds_stopped', 0):.1f} segundos\n")
            f.write(f"  - Frames en movimiento: {metrics.get('frames_moving', 0)}\n")
            f.write(f"  - Frames parado: {metrics.get('frames_stopped', 0)}\n")
            
            # Agregar log de detecciones de vueltas
            turn_log = metrics.get('turn_detection_log', [])
            if turn_log:
                f.write(f"\n  Detecciones de vueltas ({len(turn_log)} eventos):\n")
                for log_entry in turn_log:
                    f.write(f"    {log_entry}\n")
            else:
                f.write(f"\n  No se detectaron vueltas completas.\n")
            
            f.write("\n")
    
    print(f"\nReporte guardado: {report_path}")
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    return results

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Análisis de rotación de ratones')
    parser.add_argument('video_path', help='Ruta al archivo de video')
    parser.add_argument('--output', help='Ruta para guardar el video procesado')
    parser.add_argument('--no-viz', action='store_true', 
                      help='Desactivar visualización')
    parser.add_argument('--window-scale', type=int, default=25,
                      help='Tamaño de las ventanas en porcentaje (default: 25)')
    parser.add_argument('--collage', action='store_true',
                      help='Procesar como video collage 2x2')
    parser.add_argument('--separate', action='store_true',
                      help='Mostrar cada video del collage en ventanas separadas (solo con --collage)')
    
    args = parser.parse_args()
    
    # detección automática de collage: si el nombre tiene "collage" lo detecto
    video_filename = os.path.basename(args.video_path).lower()
    is_collage = args.collage or 'collage' in video_filename
    
    # selecciono el modo: collage o video individual
    if is_collage:
        print(f"\n📹 Detectado video collage")
        print(f"   Procesando como 4 videos en disposición 2x2...\n")
        results = process_collage_video(
            args.video_path,
            output_path=args.output,
            window_scale=args.window_scale if args.window_scale != 25 else 40,
            separate_windows=args.separate
        )
    else:
        # modo normal: proceso un solo video
        results = process_video(
            args.video_path,
            output_path=args.output,
            show_visualization=not args.no_viz,
            window_scale=args.window_scale
        )
        
        # Resumen final en consola
        print("\nResultados del análisis:")
        clockwise_turns = results['rotation_metrics']['clockwise_degrees'] / 360.0
        counterclockwise_turns = results['rotation_metrics']['counterclockwise_degrees'] / 360.0
        
        print(f"Número de vueltas en sentido horario: {clockwise_turns:.2f}")
        print(f"Número de vueltas en sentido antihorario: {counterclockwise_turns:.2f}")
        print(f"Rotaciones netas (horario - antihorario): {results['rotation_metrics']['net_rotations']:.2f}")
        print(f"\nMétricas adicionales:")
        print(f"Total de vueltas (suma de ambos sentidos): {(clockwise_turns + counterclockwise_turns):.2f}")
        print(f"Grados totales recorridos: {results['rotation_metrics']['total_degrees']:.2f}")
