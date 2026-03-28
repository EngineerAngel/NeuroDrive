"""Pipeline de data augmentation con Albumentations.

Transformaciones diseñadas para mejorar la robustez del modelo
en diferentes condiciones de iluminación, clima y ángulos de cámara.
"""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np


def get_train_transforms(img_size: int = 640) -> A.Compose:
    """Retorna el pipeline de augmentation para entrenamiento.

    Args:
        img_size: Tamaño de imagen objetivo.

    Returns:
        Pipeline de Albumentations con bounding box support.
    """
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.3
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.RandomRain(p=0.1),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
            A.CLAHE(clip_limit=2.0, p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.3
        ),
    )


def get_val_transforms(img_size: int = 640) -> A.Compose:
    """Retorna las transformaciones para validación (solo resize).

    Args:
        img_size: Tamaño de imagen objetivo.

    Returns:
        Pipeline de Albumentations solo con resize.
    """
    return A.Compose(
        [A.Resize(img_size, img_size)],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"]
        ),
    )


def augment_image(
    image_path: str,
    labels: list[list[float]],
    transform: A.Compose,
) -> tuple[np.ndarray, list[list[float]]]:
    """Aplica augmentation a una imagen con sus bounding boxes.

    Args:
        image_path: Ruta a la imagen.
        labels: Lista de labels YOLO [class_id, x_center, y_center, w, h].
        transform: Pipeline de Albumentations.

    Returns:
        Tupla de (imagen transformada, labels transformados).
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bboxes = [label[1:] for label in labels]
    class_labels = [int(label[0]) for label in labels]

    transformed = transform(
        image=image, bboxes=bboxes, class_labels=class_labels
    )

    new_labels = [
        [cls, *bbox]
        for cls, bbox in zip(transformed["class_labels"], transformed["bboxes"])
    ]

    return transformed["image"], new_labels


def augment_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    num_augmentations: int = 3,
    img_size: int = 640,
) -> int:
    """Genera versiones augmentadas de un dataset completo.

    Args:
        images_dir: Directorio con imágenes originales.
        labels_dir: Directorio con labels YOLO.
        output_dir: Directorio de salida.
        num_augmentations: Número de versiones por imagen.
        img_size: Tamaño de imagen objetivo.

    Returns:
        Número total de imágenes generadas.
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)

    out_imgs = output_path / "images"
    out_lbls = output_path / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    transform = get_train_transforms(img_size)
    count = 0

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    for img_file in images_path.iterdir():
        if img_file.suffix.lower() not in image_extensions:
            continue

        label_file = labels_path / f"{img_file.stem}.txt"
        if not label_file.exists():
            continue

        # Leer labels
        labels = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([float(p) for p in parts])

        if not labels:
            continue

        # Generar augmentaciones
        for i in range(num_augmentations):
            aug_image, aug_labels = augment_image(str(img_file), labels, transform)

            # Guardar imagen augmentada
            aug_name = f"{img_file.stem}_aug{i}{img_file.suffix}"
            aug_img_path = out_imgs / aug_name
            aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(aug_img_path), aug_image_bgr)

            # Guardar labels
            aug_label_path = out_lbls / f"{img_file.stem}_aug{i}.txt"
            with open(aug_label_path, "w") as f:
                for label in aug_labels:
                    line = " ".join(f"{v:.6f}" if j > 0 else str(int(v)) for j, v in enumerate(label))
                    f.write(line + "\n")

            count += 1

    print(f"✅ Generadas {count} imágenes augmentadas")
    return count
