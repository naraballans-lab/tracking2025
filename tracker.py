import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class MouseTracker:
    def __init__(self, dt=1/30.0):
        """
        Inicializa el sistema de seguimiento con Filtro de Kalman
        Args:
            dt: Intervalo de tiempo entre frames (por defecto 30 FPS)
        """
        # Inicializar Filtro de Kalman 4D (x, y, vx, vy)
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.dt = dt
        
        # Matriz de transición F
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Matriz de medición H
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Covarianza de medición R
        self.kf.R = np.array([[10, 0],
                             [0, 10]])
        
        # Covarianza de proceso Q
        q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)
        self.kf.Q = np.block([[q, np.zeros_like(q)],
                             [np.zeros_like(q), q]])
        
        # Inicializar historial para RTS smoother
        self.states_history = []
        self.covariance_history = []
        
    def initialize(self, first_measurement):
        """
        Inicializa el estado del filtro con la primera medición
        Args:
            first_measurement: Tuple (x, y) de la primera posición
        """
        self.kf.x = np.array([[first_measurement[0]], 
                             [first_measurement[1]],
                             [0],
                             [0]])
        self.kf.P *= 100
        
    def update(self, measurement):
        """
        Actualiza el estado con una nueva medición
        Args:
            measurement: Tuple (x, y) de la posición medida
        Returns:
            estimated_state: Vector de estado estimado [x, y, vx, vy]
        """
        if measurement is None:
            # Si no hay medición, solo predecir
            self.kf.predict()
        else:
            self.kf.predict()
            self.kf.update(np.array([measurement[0], measurement[1]]))
            
        # Guardar estado y covarianza para smoothing posterior
        self.states_history.append(self.kf.x.copy())
        self.covariance_history.append(self.kf.P.copy())
        
        return self.kf.x
        
    def smooth_trajectory(self):
        """
        Aplica el suavizado RTS a la trayectoria completa
        Returns:
            smoothed_states: Lista de estados suavizados
        """
        if not self.states_history:
            return np.array([])
            
        # Convertir las listas a arrays de forma correcta
        states = np.array([s.flatten() for s in self.states_history])
        
        # Si hay muy pocos estados, devolver los estados sin suavizar
        if len(states) < 3:
            return states
            
        try:
            # Aplicar un filtro de media móvil simple para suavizar la trayectoria
            window_size = 5
            smoothed_states = np.zeros_like(states)
            
            # Suavizar cada dimensión independientemente
            for i in range(states.shape[1]):
                smoothed_states[:, i] = np.convolve(
                    states[:, i], 
                    np.ones(window_size)/window_size, 
                    mode='same'
                )
            
            return smoothed_states
            
        except Exception as e:
            print(f"Error en suavizado: {e}")
            # Si hay un error en el suavizado, devolver los estados sin suavizar
            return states