"""Funciones para preparar y convertir datasets al formato YOLO.

Este módulo se encarga de:
- Leer datasets en diferentes formatos (COCO, VOC, CSV)
- Convertirlos al formato YOLO (txt con clase x_center y_center width height)
- Generar los archivos de configuración necesarios para Ultralytics
"""

import json
import shutil
from pathlib import Path

import yaml


def create_yolo_dataset_yaml(
    name: str,
    train_path: str,
    val_path: str,
    classes: list[str],
    output_path: str,
    test_path: str | None = None,
) -> str:
    """Genera el archivo YAML de configuración de dataset para Ultralytics.

    Args:
        name: Nombre del dataset.
        train_path: Ruta al directorio de imágenes de entrenamiento.
        val_path: Ruta al directorio de imágenes de validación.
        classes: Lista de nombres de clases.
        output_path: Ruta donde guardar el archivo YAML.
        test_path: Ruta opcional al directorio de imágenes de test.

    Returns:
        Ruta al archivo YAML generado.
    """
    dataset_config = {
        "path": str(Path(output_path).parent),
        "train": train_path,
        "val": val_path,
        "names": {i: cls for i, cls in enumerate(classes)},
    }
    if test_path:
        dataset_config["test"] = test_path

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)

    print(f"✅ Dataset YAML creado: {output_file}")
    return str(output_file)


def coco_to_yolo(
    annotations_path: str,
    images_dir: str,
    output_dir: str,
    target_classes: list[str] | None = None,
) -> int:
    """Convierte anotaciones COCO JSON a formato YOLO txt.

    Args:
        annotations_path: Ruta al archivo JSON de anotaciones COCO.
        images_dir: Directorio con las imágenes originales.
        output_dir: Directorio de salida (se crean subcarpetas images/ y labels/).
        target_classes: Lista opcional de clases a incluir (filtra las demás).

    Returns:
        Número de anotaciones convertidas.
    """
    with open(annotations_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Mapeo de categorías COCO
    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # Filtrar clases si se especifican
    if target_classes:
        valid_cat_ids = {
            cat_id for cat_id, name in categories.items() if name in target_classes
        }
        class_map = {cat_id: target_classes.index(categories[cat_id]) for cat_id in valid_cat_ids}
    else:
        class_map = {cat_id: idx for idx, cat_id in enumerate(sorted(categories.keys()))}
        valid_cat_ids = set(categories.keys())

    # Mapeo de imágenes
    images = {img["id"]: img for img in coco_data["images"]}

    # Crear directorios de salida
    output_path = Path(output_dir)
    labels_dir = output_path / "labels"
    imgs_out_dir = output_path / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    imgs_out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    annotations_by_image: dict[int, list[str]] = {}

    for ann in coco_data["annotations"]:
        if ann["category_id"] not in valid_cat_ids:
            continue

        img_info = images[ann["image_id"]]
        img_w, img_h = img_info["width"], img_info["height"]

        # Convertir bbox COCO [x, y, w, h] a YOLO [x_center, y_center, w, h] normalizado
        x, y, w, h = ann["bbox"]
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        class_id = class_map[ann["category_id"]]
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

        if ann["image_id"] not in annotations_by_image:
            annotations_by_image[ann["image_id"]] = []
        annotations_by_image[ann["image_id"]].append(line)
        count += 1

    # Escribir archivos de labels y copiar imágenes
    for img_id, labels in annotations_by_image.items():
        img_info = images[img_id]
        img_name = Path(img_info["file_name"]).stem

        # Guardar labels
        label_file = labels_dir / f"{img_name}.txt"
        with open(label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(labels))

        # Copiar imagen
        src_img = Path(images_dir) / img_info["file_name"]
        if src_img.exists():
            shutil.copy2(src_img, imgs_out_dir / img_info["file_name"])

    print(f"✅ Convertidas {count} anotaciones de {len(annotations_by_image)} imágenes")
    return count


def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, int]:
    """Divide un dataset en train/val/test.

    Args:
        images_dir: Directorio con imágenes.
        labels_dir: Directorio con labels YOLO.
        output_dir: Directorio de salida.
        train_ratio: Proporción para entrenamiento.
        val_ratio: Proporción para validación.
        seed: Semilla para reproducibilidad.

    Returns:
        Diccionario con conteos por split.
    """
    import random

    random.seed(seed)

    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)

    # Obtener imágenes con labels correspondientes
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [
        f for f in images_path.iterdir()
        if f.suffix.lower() in image_extensions
        and (labels_path / f"{f.stem}.txt").exists()
    ]
    random.shuffle(images)

    # Calcular splits
    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    counts = {}
    for split_name, split_images in splits.items():
        if not split_images:
            continue

        split_imgs_dir = output_path / split_name / "images"
        split_lbls_dir = output_path / split_name / "labels"
        split_imgs_dir.mkdir(parents=True, exist_ok=True)
        split_lbls_dir.mkdir(parents=True, exist_ok=True)

        for img in split_images:
            shutil.copy2(img, split_imgs_dir / img.name)
            label = labels_path / f"{img.stem}.txt"
            shutil.copy2(label, split_lbls_dir / label.name)

        counts[split_name] = len(split_images)

    print(f"✅ Dataset dividido: {counts}")
    return counts
