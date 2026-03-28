"""Script para convertir anotaciones de diferentes formatos a YOLO txt.

Uso:
    python scripts/convert_to_yolo.py --input data/raw/dataset --format coco --output data/processed/dataset
    python scripts/convert_to_yolo.py --input data/raw/dataset --format voc --output data/processed/dataset
    python scripts/convert_to_yolo.py --input data/raw/dataset --format csv --output data/processed/dataset
"""

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_coco(input_path: str, output_dir: str, classes: list[str] | None = None) -> int:
    """Convierte anotaciones COCO JSON a formato YOLO.

    Args:
        input_path: Ruta al archivo JSON de anotaciones.
        output_dir: Directorio de salida para archivos .txt.
        classes: Lista de clases a incluir (None = todas).

    Returns:
        Número de anotaciones convertidas.
    """
    from src.data.prepare import coco_to_yolo

    images_dir = str(Path(input_path).parent / "images")
    return coco_to_yolo(input_path, images_dir, output_dir, classes)


def convert_voc(input_dir: str, output_dir: str, classes: list[str] | None = None) -> int:
    """Convierte anotaciones Pascal VOC XML a formato YOLO.

    Args:
        input_dir: Directorio con archivos XML.
        output_dir: Directorio de salida.
        classes: Lista de clases (determina mapeo de IDs).

    Returns:
        Número de anotaciones convertidas.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) / "labels"
    output_path.mkdir(parents=True, exist_ok=True)

    # Detectar clases si no se proporcionan
    if classes is None:
        classes_set: set[str] = set()
        for xml_file in input_path.glob("*.xml"):
            tree = ET.parse(xml_file)
            for obj in tree.findall(".//object"):
                name_elem = obj.find("name")
                if name_elem is not None and name_elem.text:
                    classes_set.add(name_elem.text)
        classes = sorted(classes_set)
        print(f"Clases detectadas: {classes}")

    class_map = {name: idx for idx, name in enumerate(classes)}
    count = 0

    for xml_file in input_path.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find("size")
        if size is None:
            continue
        img_w = int(size.findtext("width", "0"))
        img_h = int(size.findtext("height", "0"))

        if img_w == 0 or img_h == 0:
            continue

        labels = []
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is None or name_elem.text not in class_map:
                continue

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))

            # Convertir a formato YOLO normalizado
            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            class_id = class_map[name_elem.text]
            labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            count += 1

        # Guardar archivo de labels
        label_file = output_path / f"{xml_file.stem}.txt"
        with open(label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(labels))

    print(f"✅ Convertidas {count} anotaciones VOC a YOLO")
    return count


def convert_csv_annotations(
    csv_path: str,
    output_dir: str,
    img_col: str = "filename",
    class_col: str = "class",
    x1_col: str = "xmin",
    y1_col: str = "ymin",
    x2_col: str = "xmax",
    y2_col: str = "ymax",
    w_col: str = "width",
    h_col: str = "height",
) -> int:
    """Convierte anotaciones de CSV a formato YOLO.

    Args:
        csv_path: Ruta al archivo CSV.
        output_dir: Directorio de salida.
        img_col: Nombre de la columna de imagen.
        class_col: Nombre de la columna de clase.
        x1_col, y1_col, x2_col, y2_col: Columnas de bbox.
        w_col, h_col: Columnas de dimensiones de imagen.

    Returns:
        Número de anotaciones convertidas.
    """
    output_path = Path(output_dir) / "labels"
    output_path.mkdir(parents=True, exist_ok=True)

    # Leer CSV y detectar clases
    annotations: dict[str, list[dict]] = {}
    classes_set: set[str] = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            classes_set.add(row[class_col])
            img_name = row[img_col]
            if img_name not in annotations:
                annotations[img_name] = []
            annotations[img_name].append(row)

    classes = sorted(classes_set)
    class_map = {name: idx for idx, name in enumerate(classes)}
    print(f"Clases detectadas: {classes}")

    count = 0
    for img_name, rows in annotations.items():
        labels = []
        for row in rows:
            img_w = float(row[w_col])
            img_h = float(row[h_col])

            xmin = float(row[x1_col])
            ymin = float(row[y1_col])
            xmax = float(row[x2_col])
            ymax = float(row[y2_col])

            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            class_id = class_map[row[class_col]]
            labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            count += 1

        stem = Path(img_name).stem
        label_file = output_path / f"{stem}.txt"
        with open(label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(labels))

    print(f"✅ Convertidas {count} anotaciones CSV a YOLO")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convertir anotaciones a formato YOLO"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Ruta al archivo o directorio de anotaciones",
    )
    parser.add_argument(
        "--format", "-f",
        required=True,
        choices=["coco", "voc", "csv"],
        help="Formato de las anotaciones de entrada",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Directorio de salida",
    )
    parser.add_argument(
        "--classes", "-c",
        nargs="+",
        default=None,
        help="Clases a incluir (por defecto: todas)",
    )

    args = parser.parse_args()

    if args.format == "coco":
        convert_coco(args.input, args.output, args.classes)
    elif args.format == "voc":
        convert_voc(args.input, args.output, args.classes)
    elif args.format == "csv":
        convert_csv_annotations(args.input, args.output)


if __name__ == "__main__":
    main()
