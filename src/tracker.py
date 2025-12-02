import numpy as np #matrices y cálculos numéricos
from filterpy.kalman import KalmanFilter #filtro de Kalman para suavizar seguimiento
from filterpy.common import Q_discrete_white_noise 


class MouseTracker:
    """
    Archivo encargado de seguir al ratón una vez que lo he detectado..
    Usando un Filtro de Kalman que es como un "adivinador inteligente" que predice dónde estará el ratón en el siguiente frame basándose en dónde estaba antes.
    """
    
    def __init__(self, dt=1/30.0):
        """
        Configuración para el seguimiento. 
        -Cómo se mueve el ratón (con velocidad constante) -Cuánto confío en las mediciones.
        
        Args:
            dt: tiempo entre frames - 1/30 porque los vídeos van a 30 FPS
        """
        # Creo el filtro de Kalman
        # dim_x=4 significa que rastreo [posición_x, posición_y, velocidad_x, velocidad_y]
        # dim_z=2 porque solo mido [posición_x, posición_y] (no veo la velocidad directamente)
        self.fk = KalmanFilter(dim_x=4, dim_z=2)
        self.dt = dt  # guardo el tiempo entre frames para usarlo después
        
        # ¡La matriz F describe cómo se mueve el ratón donde "la nueva posición = posición anterior + velocidad * tiempo"
        # Esto asume que el ratón se mueve con velocidad constante (lo cual no es del todo cierto, pero funciona bien)
        self.fk.F = np.array([
            [1, 0, dt, 0],   # x_nueva = x_anterior + vx * dt
            [0, 1, 0, dt],   # y_nueva = y_anterior + vy * dt
            [0, 0, 1, 0],    # vx se mantiene igual (velocidad constante)
            [0, 0, 0, 1]     # vy se mantiene igual (velocidad constante)
        ])
        
        # La matriz H dice qué partes del estado puedo "ver" directamente
        # Solo veo la posición [x,y], no la velocidad [vx,vy] 
        self.fk.H = np.array([
            [1, 0, 0, 0],  # veo x
            [0, 1, 0, 0]   # veo y
        ])
        
        # La matriz R mide cuánto confío en mis mediciones de posición
        # Valores altos (10) significan que mis mediciones tienen bastante ruido
        self.fk.R = np.array([[10, 0],
                             [0, 10]])
        
        # La matriz Q mide la incertidumbre en mi modelo de movimiento
        # Usé var=0.1 después de probar diferentes valores con mis vídeos
        q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)
        self.fk.Q = np.block([[q, np.zeros_like(q)],
                             [np.zeros_like(q), q]])
        
        # Guardo el historial para poder suavizar la trayectoria después
        self.historial_estados = []
        self.historial_covarianzas = []
        
        
    def inicializar(self, primera_medicion):
        """
        Este es el inicio del seguimiento. Cuando detecto al ratón por primera vez,le digo al filtro dónde está y que empiece a seguirlo. 
        
        Args:
            primera_medicion: Tupla (x, y) con la primera posición detectada del ratón
        """
        # Configuro el estado inicial del filtro con la primera posición que detecté
        # [x, y, vx, vy] - pongo vx=0, vy=0 porque no sé hacia dónde va todavía
        self.fk.x = np.array([[primera_medicion[0]],  # posición x inicial
                             [primera_medicion[1]],   # posición y inicial
                             [0],                      # velocidad x inicial = 0 
                             [0]])                     # velocidad y inicial = 0 
        
        # La covarianza P mide qué tan seguro estoy de mi estimación inicial
        # La multiplico por 100 porque existe una gran inseguridad al inicio 
        
    def actualizar(self, medicion):
        """
        Cada frame nuevo, el filtro predice dónde debería estar el ratón y luego ajusta su predicción con lo que realmente ve.
        
        Args:
            medicion: Tupla (x, y) de la posición medida, o None si no hay detección
            
        Returns:
            np.ndarray: Vector de estado estimado [x, y, vx, vy]
        """
        if medicion is None:
            # Solo es predicción basada en dónde creo que debería estar
            # Esto es útil cuando el ratón se esconde momentáneamente detrás de algo
            self.fk.predict()
        else:
            # Primero predigo dónde debería estar, luego ajusto con la medición real
            # El orden es importante: predict() primero, luego update()
            self.fk.predict()
            self.fk.update(np.array([medicion[0], medicion[1]]))
            
        # Guardo el estado y la covarianza en el historial para suavizado posterior
        # Esto me permite analizar la trayectoria completa después de procesar todo el vídeo
        self.historial_estados.append(self.fk.x.copy())
        self.historial_covarianzas.append(self.fk.P.copy())
        
        # Devuelvo la mejor estimación actual de dónde está el ratón y hacia dónde va
        return self.fk.x
        
    def suavizar_trayectoria(self):
        """
        Después de procesar todo el vídeo, aplico un suavizado para quitar las sacudidas y hacer que la trayectoria se vea más natural. 
        Uso convolución.
        
        Returns:
            np.ndarray: Array de estados suavizados, o array vacío si no hay suficientes datos
        """
        # Si no tengo ningún estado guardado, no puedo suavizar nada
        if not self.historial_estados:
            return np.array([])
            
        # Convierto la lista de matrices en un array numpy para trabajar más fácilmente
        # Cada fila será un estado [x, y, vx, vy] en un momento del tiempo
        estados = np.array([s.flatten() for s in self.historial_estados])
        
        # Si tengo menos de 3 puntos, no vale la pena suavizar = devolver los datos crudos
        if len(estados) < 3:
            return estados
            
        try:
            # Filtro de media móvil = promediar cada punto con sus vecinos
            # Una ventana de 5 puntos me dio el mejor balance entre suavizado y detalle
            tamanio_ventana = 5
            estados_suavizados = np.zeros_like(estados)
            
            # Aplico la convolución a cada columna (x, y, vx, vy) por separado
            for i in range(estados.shape[1]):  # para cada dimensión (x, y, vx, vy)
                estados_suavizados[:, i] = np.convolve(
                    estados[:, i],  # la columna actual
                    np.ones(tamanio_ventana) / tamanio_ventana,  # filtro de media móvil
                    mode='same'  # mantiene el mismo tamaño que la entrada
                )
            
            return estados_suavizados
            
        except Exception as e:
            # Si algo sale mal, devuelvo los datos originales
            print(f"Error en suavizado: {e}")
            return estados


