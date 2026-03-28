# Guía de Instalación y Configuración

## Requisitos

- **Python**: 3.11 o superior
- **Git**: Para control de versiones
- **VS Code** (recomendado) o cualquier editor de código
- **GPU** (opcional): NVIDIA con CUDA para entrenamiento más rápido

## Instalación Paso a Paso

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/neurodrive.git
cd neurodrive
```

### 2. Crear Entorno Virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar (Linux/Mac)
source venv/bin/activate

# Activar (Windows)
venv\Scripts\activate
```

### 3. Instalar Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

O con Make:
```bash
make install
```

### 4. Configurar Variables de Entorno

```bash
# Copiar plantilla
cp .env.example .env

# Editar con tus valores (opcional)
# KAGGLE_USERNAME y KAGGLE_KEY son necesarios para descargar datasets
```

### 5. Verificar Instalación

```bash
# Verificar que YOLO funciona
python -c "from ultralytics import YOLO; m = YOLO('yolo11n.pt'); print('OK')"

# Verificar configuración
python -c "from src.config import load_config, print_config; print_config(load_config())"

# Correr tests
make test
```

## Configuración de VS Code

### Extensiones Recomendadas

1. **Python** (ms-python.python) — Soporte para Python
2. **Jupyter** (ms-toolsai.jupyter) — Ejecutar notebooks
3. **YAML** (redhat.vscode-yaml) — Soporte para archivos YAML
4. **GitLens** (eamodio.gitlens) — Historial de Git mejorado

### Configuración del Workspace

Crear `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "editor.formatOnSave": true,
    "python.formatting.provider": "black"
}
```

## Setup en Google Colab

### Opción 1: Desde Google Drive

1. Sube la carpeta del proyecto a Google Drive
2. Abre `notebooks/utils_colab.ipynb` en Colab
3. Ejecuta todas las celdas para configurar el entorno

### Opción 2: Desde GitHub

En Colab, ejecuta:
```python
!git clone https://github.com/tu-usuario/neurodrive.git
%cd neurodrive
!pip install -r requirements.txt
```

## Descargar Datasets de Kaggle

### Configurar API de Kaggle

1. Ir a [kaggle.com/settings](https://www.kaggle.com/settings)
2. En la sección "API", hacer click en "Create New Token"
3. Se descarga `kaggle.json`
4. Colocar en `~/.kaggle/kaggle.json` (Linux/Mac) o `%USERPROFILE%\.kaggle\kaggle.json` (Windows)
5. Asegurar permisos: `chmod 600 ~/.kaggle/kaggle.json`

### Descargar

```bash
# Ver datasets disponibles
python scripts/download_datasets.py --list

# Descargar dataset específico
python scripts/download_datasets.py --dataset tier1_road_vehicles

# Descargar todos los Tier 1
python scripts/download_datasets.py --tier 1
```

## Verificar Todo

```bash
# Detección rápida con modelo pretrained (sin datos de entrenamiento)
python scripts/quick_detect.py --image data/samples/ejemplo.jpg
```

> Si no hay imágenes de ejemplo, descarga una imagen de prueba y colócala en `data/samples/`.

## Problemas Comunes

| Problema | Solución |
|----------|----------|
| `torch.cuda.is_available()` retorna `False` | Instalar CUDA toolkit compatible con tu GPU |
| Error al importar `ultralytics` | Verificar que el venv está activado |
| Kaggle da error 403 | Verificar que `kaggle.json` tiene los permisos correctos |
| Memoria GPU insuficiente | Reducir `batch_size` en la configuración de entrenamiento |
