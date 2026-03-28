# Modelos Entrenados

Este directorio almacena los modelos entrenados. **No se suben a Git** (ver `.gitignore`).

## Convención de Nombres

```
{modelo}_{dataset}_{fecha}_{metrica}.pt
```

### Ejemplos

| Archivo | Descripción |
|---------|-------------|
| `yolo11n_tier1_20260327_map50-0.72.pt` | YOLO11 nano, Tier 1, mAP50=0.72 |
| `yolo11s_tier1tier2_20260415_map50-0.85.pt` | YOLO11 small, Tier 1+2 combinados |
| `yolo11s_semic_20260501_map50-0.91.pt` | Fine-tuned con datos SEMIC |

## Campos

- **modelo**: Arquitectura (yolo11n, yolo11s, yolo11m)
- **dataset**: Dataset(s) de entrenamiento
- **fecha**: Fecha del entrenamiento (YYYYMMDD)
- **metrica**: Métrica principal (mAP50)

## Cómo Usar un Modelo

```python
from src.config import load_config
from src.models.detector import VisionVialDetector

config = load_config()
config["model"]["pretrained"] = False
detector = VisionVialDetector(config)

# El detector carga automáticamente el modelo más reciente de este directorio
detections = detector.detect("imagen.jpg")
```

## Formatos Exportados

| Formato | Extensión | Uso |
|---------|-----------|-----|
| PyTorch | `.pt` | Entrenamiento y evaluación |
| ONNX | `.onnx` | Inferencia optimizada |
| TorchScript | `.torchscript` | Despliegue en producción |
