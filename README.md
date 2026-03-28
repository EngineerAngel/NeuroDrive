# NeuroDrive — Detección Inteligente de Vehículos en Carreteras

> Sistema de computer vision para detección y clasificación de vehículos en carreteras usando YOLO (Ultralytics).

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![YOLO](https://img.shields.io/badge/YOLO-11-green?logo=yolo)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-En%20desarrollo-orange)

---

## Inicio Rápido

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/neurodrive.git
cd neurodrive

# 2. Crear entorno virtual e instalar dependencias
make install
# O manualmente:
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Descargar datasets de Kaggle
python scripts/download_datasets.py --dataset tier1_road_vehicles

# 4. Entrenar modelo baseline
make train

# 5. Detectar vehículos en una imagen
python scripts/quick_detect.py --image data/samples/ejemplo.jpg
```

---

## Estructura del Proyecto

```
neurodrive/
├── configs/              # Configuraciones YAML (datasets, entrenamiento)
├── data/                 # Datasets (no se sube a git)
│   ├── raw/              # Datasets descargados de Kaggle
│   ├── processed/        # Datos listos para YOLO
│   ├── semic/            # Imágenes de SEMIC
│   └── samples/          # Imágenes de ejemplo
├── notebooks/            # Jupyter notebooks de exploración
├── src/                  # Código fuente principal
│   ├── config.py         # Carga de configuración
│   ├── data/             # Preparación y augmentation de datos
│   ├── models/           # Detector YOLO y evaluación
│   ├── api/              # API REST con FastAPI
│   └── utils/            # Visualización y exportación
├── models/               # Modelos entrenados (no se sube a git)
├── results/              # Resultados de entrenamientos
├── tests/                # Tests unitarios
├── scripts/              # Scripts utilitarios
└── docs/                 # Documentación del proyecto
```

---

## Datasets

| Tier | Dataset | Fuente | Formato | Tamaño | Estado |
|------|---------|--------|---------|--------|--------|
| 1 | [Road Vehicle Images](https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset) | Kaggle | YOLO | ~120 MB | Pendiente |
| 1 | [Cars Detection](https://www.kaggle.com/datasets/sshikamaru/car-object-detection) | Kaggle | Custom | ~200 MB | Pendiente |
| 2 | [BDD100K](https://www.kaggle.com/datasets/marquis03/bdd100k) | Kaggle | COCO | ~7 GB | Placeholder |
| 2 | [Vehicles OpenImages](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) | Kaggle | Custom | ~500 MB | Placeholder |
| 2 | [DAWN](https://www.kaggle.com/datasets/datasets) | Kaggle | Custom | ~300 MB | Placeholder |
| 3 | SEMIC (privado) | Empresa | Por definir | Por definir | Pendiente |

> **Tier 1**: Datasets iniciales para entrenamiento baseline.
> **Tier 2**: Datasets complementarios para mejorar generalización.
> **Tier 3**: Datos reales de SEMIC para fine-tuning final.

---

## Equipo

| Rol | Integrante | Responsabilidades |
|-----|-----------|-------------------|
| Líder / ML Engineer | | Arquitectura del modelo, entrenamiento, evaluación |
| Frontend Developer | | Interfaz web, visualizaciones, dashboard |
| Backend Developer | | API, integración de datos, infraestructura |
| Ing. Industrial | | Análisis de requerimientos, métricas de negocio |
| Gestión de Proyecto | | Documentación, timeline, coordinación con SEMIC |

---

## Desarrollo

Para instrucciones detalladas de instalación y configuración, consulta [docs/SETUP.md](docs/SETUP.md).

Para contribuir al proyecto, consulta [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

### Comandos Útiles

```bash
make install    # Instalar dependencias
make train      # Entrenar modelo
make detect     # Detectar en imágenes de ejemplo
make api        # Levantar API REST
make test       # Correr tests
make clean      # Limpiar archivos temporales
```

---

## Tecnologías

- **Modelo**: YOLO 11 (Ultralytics)
- **Framework ML**: PyTorch
- **Computer Vision**: OpenCV, Albumentations
- **API**: FastAPI
- **Lenguaje**: Python 3.11+

---

> **Proyecto para InnovaTecNM 2026 — Certamen Nacional de Proyectos de Investigación y Desarrollo Tecnológico**
