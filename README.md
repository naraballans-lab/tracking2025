# tracking2025
En este repositorio se muestran los códigos utilizados durante el desarrollo del TFG 'Diseño y construcción de un entorno controlado con sistema de detección y análisis de rotaciones en ratones de experimentación'.

# Análisis de Rotación de Ratones

Sistema automático para detectar y contar rotaciones en ratones, diseñado para estudios comportamentales y modelos de Parkinson.

## Características

- **Detección automática** con sustracción de fondo (MOG2) y filtro de Kalman
- **Análisis inteligente** mediante triple validación: cuadrantes + ángulos + tiempo
- **Dos modos**: vídeo simple o collage 2×2 con 4 ratones simultáneos
- **Visualización en tiempo real** con 8 vistas y métricas instantáneas
- **Reportes automáticos** en formato `.txt` con estadísticas completas

## Instalación

```bash
# Crear entorno virtual
python -m venv .venv

# Activar (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Modo Simple
```bash
# Análisis con visualización
python main.py Videos/video1.mp4

# Guardar vídeo procesado
python main.py Videos/video1.mp4 --output salida.avi

# Sin visualización (más rápido)
python main.py Videos/video1.mp4 --no-viz
```

### Modo Collage (4 ratones)
```bash
python main.py Videos/collage_2x2.mp4 --collage --output analisis.avi
```

## Resultados

El sistema genera automáticamente un archivo `<video>_analisis.txt` con:

- **Configuración**: tamaño del ratón, grilla de cuadrantes, umbrales
- **Resultados**: vueltas horarias/antihorarias, vueltas netas, grados totales
- **Estadísticas**: tiempo en movimiento/parado
- **Log detallado**: cada vuelta con frame, cuadrantes y ángulos

**Ejemplo de salida:**
```
Vueltas horarias: 5
Vueltas antihorarias: 8
Vueltas totales: 13
Vueltas netas: -3 (3 antihorarias netas)
Tiempo en movimiento: 85.3 segundos
```

## Validación de Rotaciones

Una vuelta completa requiere cumplir **TODOS** los criterios:

1. **Cuadrantes**: visitar mínimo calculado dinámicamente (≈perímetro/4)
2. **Tiempo**: entre 1 y 15 segundos
3. **Ángulos**: suma de cambios ≥270° con dirección consistente

## Estructura

```
src/
├── detector.py    # Detección del ratón (MOG2)
├── tracker.py     # Seguimiento (Kalman)
└── analyzer.py    # Análisis de rotaciones

main.py           # Pipeline principal
```

## Requisitos

- Python 3.8+
- OpenCV, NumPy, FilterPy, SciPy (ver `requirements.txt`)
