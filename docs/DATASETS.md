# Datasets

Descripción de los datasets utilizados en el proyecto, organizados por tiers.

## Sistema de Tiers

| Tier | Propósito | Cuándo usar |
|------|-----------|-------------|
| **Tier 1** | Entrenamiento baseline | Primera fase: establecer métricas base |
| **Tier 2** | Mejorar generalización | Segunda fase: ampliar variedad de datos |
| **Tier 3** | Fine-tuning real | Fase final: adaptar a condiciones reales de SEMIC |

---

## Tier 1 — Datasets Iniciales

### Road Vehicle Images Dataset

- **Fuente**: [Kaggle - ashfakyeafi](https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset)
- **Tamaño**: ~120 MB
- **Formato**: YOLO nativo
- **Clases**: vehicle
- **Split**: Train/Val
- **Conversión necesaria**: No
- **Notas**: Imágenes de carreteras de Bangladesh. Ideal para primer entrenamiento por estar en formato YOLO.

```bash
python scripts/download_datasets.py --dataset tier1_road_vehicles
```

### Cars Detection

- **Fuente**: [Kaggle - sshikamaru](https://www.kaggle.com/datasets/sshikamaru/car-object-detection)
- **Tamaño**: ~200 MB
- **Formato**: Custom (CSV)
- **Clases**: car
- **Split**: Train/Val
- **Conversión necesaria**: Sí (CSV → YOLO)

```bash
python scripts/download_datasets.py --dataset tier1_cars_detection
python scripts/convert_to_yolo.py --input data/raw/tier1_cars_detection --format csv --output data/processed/tier1_cars_detection
```

---

## Tier 2 — Datasets Complementarios

### BDD100K

- **Fuente**: [Kaggle - marquis03](https://www.kaggle.com/datasets/marquis03/bdd100k)
- **Tamaño**: ~7 GB
- **Formato**: COCO
- **Clases**: car, truck, bus, motorcycle, bicycle, pedestrian, traffic light, traffic sign
- **Notas**: Dataset muy grande. Usar subconjunto filtrado por clases de vehículos.
- **Conversión**: COCO → YOLO

### Vehicles OpenImages

- **Fuente**: Kaggle (slug pendiente de confirmar)
- **Tamaño**: ~500 MB
- **Formato**: Custom
- **Clases**: car, truck, bus, motorcycle
- **Conversión**: Sí

### DAWN (Detection in Adverse Weather)

- **Fuente**: Kaggle (slug pendiente de confirmar)
- **Tamaño**: ~300 MB
- **Formato**: Custom
- **Clases**: vehicle
- **Notas**: Imágenes en condiciones adversas (lluvia, niebla, nieve). Útil para robustez del modelo.

---

## Tier 3 — Dataset SEMIC (Privado)

- **Fuente**: Empresa SEMIC (infraestructura carretera)
- **Tamaño**: Por definir
- **Formato**: Por definir (probablemente imágenes sin anotar)
- **Clases**: Por definir según las imágenes
- **Estado**: Pendiente de recibir

### Proceso de Preparación (cuando lleguen las imágenes)

1. Recibir imágenes de SEMIC
2. Revisar calidad y variedad
3. Anotar con herramienta (LabelImg, Roboflow, o CVAT)
4. Convertir anotaciones a formato YOLO
5. Dividir en train/val/test
6. Actualizar `configs/datasets/semic.yaml`
7. Ejecutar fine-tuning

---

## Cómo Agregar un Nuevo Dataset

1. Crear archivo de configuración en `configs/datasets/`:
   ```yaml
   dataset:
     name: "Nombre del Dataset"
     source: "kaggle"
     kaggle_slug: "usuario/dataset"
     size: "X MB"
     format: "yolo"
     needs_conversion: false
     split:
       train: true
       val: true
       test: false
     classes: ["car", "truck"]
     notes: "Descripción breve"
     tier: 1
     status: "pendiente"
   ```

2. Si necesita conversión, agregar lógica en `scripts/convert_to_yolo.py`

3. Descargar con: `python scripts/download_datasets.py --dataset nombre_config`

4. Validar con el notebook `01_exploracion_datos.ipynb`

## Formato YOLO

Cada imagen tiene un archivo `.txt` correspondiente con una línea por objeto:

```
<class_id> <x_center> <y_center> <width> <height>
```

- Todos los valores están normalizados entre 0 y 1
- `x_center`, `y_center`: centro del bounding box
- `width`, `height`: dimensiones del bounding box
