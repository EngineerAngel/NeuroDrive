# Cómo Contribuir

Guía para los integrantes del equipo NeuroDrive.

## Convención de Ramas

| Rama | Propósito |
|------|-----------|
| `main` | Versión estable del proyecto |
| `dev` | Rama de desarrollo (integración) |
| `feature/nombre` | Nueva funcionalidad |
| `fix/nombre` | Corrección de errores |
| `experiment/nombre` | Experimentos de entrenamiento |

### Flujo de Trabajo

```
main ← dev ← feature/mi-feature
```

1. Crear rama desde `dev`: `git checkout -b feature/mi-feature dev`
2. Hacer commits con cambios pequeños y frecuentes
3. Hacer push: `git push origin feature/mi-feature`
4. Crear Pull Request hacia `dev`
5. Revisar y aprobar en equipo
6. Merge a `dev`

## Convención de Commits

Escribimos los commits en **español** con el siguiente formato:

```
<tipo>: <descripción breve>

<descripción detallada opcional>
```

### Tipos de commit

| Tipo | Uso |
|------|-----|
| `feat` | Nueva funcionalidad |
| `fix` | Corrección de errores |
| `docs` | Cambios en documentación |
| `data` | Cambios en datos o configuración de datasets |
| `train` | Nuevo experimento de entrenamiento |
| `refactor` | Reestructuración de código sin cambiar funcionalidad |
| `test` | Agregar o modificar tests |
| `config` | Cambios en configuración |

### Ejemplos

```
feat: agregar endpoint de estadísticas por clase
fix: corregir conversión de coordenadas COCO a YOLO
docs: actualizar guía de instalación para Windows
data: agregar configuración para dataset BDD100K
train: experimento baseline con yolo11n, 50 épocas
```

## Cómo Agregar un Nuevo Dataset

1. **Crear configuración YAML** en `configs/datasets/`:
   ```bash
   # Copiar template
   cp configs/datasets/tier1_road_vehicles.yaml configs/datasets/mi_dataset.yaml
   ```

2. **Editar** el archivo con los datos del nuevo dataset

3. **Descargar** si es de Kaggle:
   ```bash
   python scripts/download_datasets.py --dataset mi_dataset
   ```

4. **Convertir** si es necesario:
   ```bash
   python scripts/convert_to_yolo.py --input data/raw/mi_dataset --format coco --output data/processed/mi_dataset
   ```

5. **Validar** con el notebook `01_exploracion_datos.ipynb`

6. **Hacer commit**:
   ```bash
   git add configs/datasets/mi_dataset.yaml
   git commit -m "data: agregar configuración para dataset Mi Dataset"
   ```

## Cómo Registrar un Nuevo Experimento

1. **Crear configuración** en `configs/training/`:
   ```bash
   cp configs/training/baseline.yaml configs/training/mi_experimento.yaml
   ```

2. **Editar** hiperparámetros según el experimento

3. **Entrenar**:
   ```bash
   python -c "from src.models.detector import VisionVialDetector; from src.config import load_config; \
     config = load_config('configs/training/mi_experimento.yaml'); \
     VisionVialDetector(config).train()"
   ```

4. **Evaluar** con el notebook `03_evaluacion_metricas.ipynb`

5. **Documentar** resultados en el commit:
   ```bash
   git add configs/training/mi_experimento.yaml
   git commit -m "train: experimento con yolo11s, mAP50=0.85"
   ```

## Estructura de Archivos

- **No subir** a Git: datasets, modelos entrenados, archivos `.env`
- **Sí subir**: configuraciones, código, notebooks, documentación
- Respetar el `.gitignore` existente

## Estilo de Código

- **Docstrings y comentarios** en español
- **Type hints** en todas las funciones
- **Nombres de variables** en inglés (convención Python)
- **Configuración** siempre por YAML, nunca hardcodeada

## Contacto del Equipo

| Rol | Contacto |
|-----|----------|
| Líder / ML | (agregar) |
| Frontend | (agregar) |
| Backend | (agregar) |
| Industrial | (agregar) |
| Gestión | (agregar) |
