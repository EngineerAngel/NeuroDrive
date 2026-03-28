# Arquitectura del Proyecto

## Diagrama General

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Datasets   │────▶│  Preparación │────▶│ Entrenamiento│
│  (Kaggle)   │     │  de Datos    │     │   YOLO      │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐           │
│   SEMIC     │────▶│  Fine-tuning │◀──────────┘
│  (privado)  │     │              │
└─────────────┘     └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Modelo     │
                    │  Entrenado   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──┐  ┌──────▼──┐  ┌─────▼─────┐
       │  Script │  │   API   │  │ Notebooks │
       │  CLI    │  │  REST   │  │  Jupyter  │
       └─────────┘  └─────────┘  └───────────┘
```

## Flujo de Datos

### 1. Adquisición de Datos
```
Kaggle API → data/raw/ → scripts/convert_to_yolo.py → data/processed/
```

### 2. Entrenamiento
```
configs/training/*.yaml → src/models/detector.py → models/*.pt → results/
```

### 3. Inferencia (CLI)
```
imagen → scripts/quick_detect.py → src/models/detector.py → resultado (consola + imagen anotada)
```

### 4. Inferencia (API)
```
POST /api/v1/detect → src/api/routes.py → src/models/detector.py → JSON response
```

## Módulos Principales

### `src/config.py`
- Carga `configs/default.yaml` como configuración base
- Permite overrides con archivos YAML adicionales
- Aplica variables de entorno desde `.env`
- Resuelve rutas relativas al directorio del proyecto

### `src/data/`
- **prepare.py**: Conversión de formatos (COCO, VOC, CSV → YOLO)
- **augment.py**: Data augmentation con Albumentations
- **validate.py**: Verificación de integridad de datasets

### `src/models/`
- **detector.py**: Clase `VisionVialDetector` — wrapper de YOLO para detección
- **evaluate.py**: Métricas y comparativa entre modelos

### `src/api/`
- **main.py**: Aplicación FastAPI
- **routes.py**: Endpoints de detección
- **schemas.py**: Modelos Pydantic para request/response

### `src/utils/`
- **visualization.py**: Dibujar bounding boxes, grids de resultados
- **export.py**: Exportar a CSV/JSON

## Configuración

Toda la configuración es por archivos YAML:

```
configs/
├── default.yaml              ← Configuración base global
├── datasets/*.yaml           ← Un archivo por dataset
└── training/*.yaml           ← Un archivo por experimento
```

El código nunca tiene valores hardcodeados — todo viene de los YAML o variables de entorno.

## Stack Tecnológico

| Capa | Tecnología |
|------|-----------|
| Modelo | YOLO 11 (Ultralytics) |
| Framework ML | PyTorch |
| Visión | OpenCV, Albumentations |
| API | FastAPI + Uvicorn |
| Configuración | PyYAML + python-dotenv |
| Notebooks | Jupyter |
| Testing | pytest + httpx |
