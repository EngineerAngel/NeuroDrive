"""Validación de integridad de datasets.

Verifica que las imágenes y labels sean consistentes,
que las clases sean válidas y que no haya datos corruptos.
"""

from pathlib import Path

import cv2


def validate_dataset(
    images_dir: str,
    labels_dir: str,
    expected_classes: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """Valida la integridad de un dataset en formato YOLO.

    Args:
        images_dir: Directorio con imágenes.
        labels_dir: Directorio con archivos de labels.
        expected_classes: Lista de clases esperadas (para validar IDs).
        verbose: Imprimir detalles de errores encontrados.

    Returns:
        Diccionario con estadísticas de validación.
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    stats = {
        "total_images": 0,
        "total_labels": 0,
        "images_sin_label": [],
        "labels_sin_imagen": [],
        "imagenes_corruptas": [],
        "labels_invalidos": [],
        "clases_encontradas": set(),
        "distribucion_clases": {},
        "valido": True,
    }

    # Obtener listas de archivos
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = {
        f.stem: f for f in images_path.iterdir()
        if f.suffix.lower() in image_extensions
    }
    labels = {
        f.stem: f for f in labels_path.iterdir()
        if f.suffix == ".txt"
    }

    stats["total_images"] = len(images)
    stats["total_labels"] = len(labels)

    # Verificar correspondencia imagen <-> label
    stats["images_sin_label"] = [
        str(images[name]) for name in images if name not in labels
    ]
    stats["labels_sin_imagen"] = [
        str(labels[name]) for name in labels if name not in images
    ]

    # Validar cada par imagen-label
    for name in images:
        img_path = images[name]

        # Verificar que la imagen se puede leer
        img = cv2.imread(str(img_path))
        if img is None:
            stats["imagenes_corruptas"].append(str(img_path))
            continue

        # Verificar label si existe
        if name in labels:
            label_path = labels[name]
            img_h, img_w = img.shape[:2]

            with open(label_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()

                    if len(parts) != 5:
                        stats["labels_invalidos"].append(
                            f"{label_path}:{line_num} — se esperaban 5 valores, hay {len(parts)}"
                        )
                        continue

                    try:
                        class_id = int(parts[0])
                        x_center, y_center, w, h = (float(p) for p in parts[1:])
                    except ValueError:
                        stats["labels_invalidos"].append(
                            f"{label_path}:{line_num} — valores no numéricos"
                        )
                        continue

                    # Verificar rangos
                    if not all(0 <= v <= 1 for v in [x_center, y_center, w, h]):
                        stats["labels_invalidos"].append(
                            f"{label_path}:{line_num} — coordenadas fuera de rango [0, 1]"
                        )

                    # Verificar clase válida
                    if expected_classes and class_id >= len(expected_classes):
                        stats["labels_invalidos"].append(
                            f"{label_path}:{line_num} — clase {class_id} no válida "
                            f"(máximo: {len(expected_classes) - 1})"
                        )

                    stats["clases_encontradas"].add(class_id)
                    stats["distribucion_clases"][class_id] = (
                        stats["distribucion_clases"].get(class_id, 0) + 1
                    )

    # Determinar validez general
    has_errors = (
        stats["images_sin_label"]
        or stats["labels_sin_imagen"]
        or stats["imagenes_corruptas"]
        or stats["labels_invalidos"]
    )
    stats["valido"] = not has_errors

    # Convertir set a lista para serialización
    stats["clases_encontradas"] = sorted(stats["clases_encontradas"])

    if verbose:
        print_validation_report(stats)

    return stats


def print_validation_report(stats: dict) -> None:
    """Imprime un reporte legible de la validación.

    Args:
        stats: Diccionario de estadísticas de validación.
    """
    print("=" * 50)
    print("REPORTE DE VALIDACIÓN DE DATASET")
    print("=" * 50)
    print(f"Total de imágenes: {stats['total_images']}")
    print(f"Total de labels:   {stats['total_labels']}")
    print(f"Clases encontradas: {stats['clases_encontradas']}")
    print()

    if stats["distribucion_clases"]:
        print("Distribución de clases:")
        for cls_id, count in sorted(stats["distribucion_clases"].items()):
            print(f"  Clase {cls_id}: {count} instancias")
        print()

    if stats["images_sin_label"]:
        print(f"⚠️  Imágenes sin label: {len(stats['images_sin_label'])}")
        for path in stats["images_sin_label"][:5]:
            print(f"   - {path}")

    if stats["labels_sin_imagen"]:
        print(f"⚠️  Labels sin imagen: {len(stats['labels_sin_imagen'])}")
        for path in stats["labels_sin_imagen"][:5]:
            print(f"   - {path}")

    if stats["imagenes_corruptas"]:
        print(f"❌ Imágenes corruptas: {len(stats['imagenes_corruptas'])}")
        for path in stats["imagenes_corruptas"][:5]:
            print(f"   - {path}")

    if stats["labels_invalidos"]:
        print(f"❌ Labels inválidos: {len(stats['labels_invalidos'])}")
        for msg in stats["labels_invalidos"][:5]:
            print(f"   - {msg}")

    print()
    status = "✅ VÁLIDO" if stats["valido"] else "❌ ERRORES ENCONTRADOS"
    print(f"Estado: {status}")
