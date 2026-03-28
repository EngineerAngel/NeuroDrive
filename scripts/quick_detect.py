"""Script rápido de detección de vehículos.

Uso:
    python scripts/quick_detect.py --image data/samples/ejemplo.jpg
    python scripts/quick_detect.py --image data/samples/ --conf 0.3 --output results/
    python scripts/quick_detect.py --image foto.jpg --model models/mejor_modelo.pt
"""

import argparse
import sys
from pathlib import Path

# Agregar raíz del proyecto al path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.models.detector import VisionVialDetector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detección rápida de vehículos con NeuroDrive"
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Ruta a imagen o directorio de imágenes",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Ruta al modelo .pt (default: YOLO11n pretrained)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Umbral de confianza (default: 0.25)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Directorio para guardar imágenes anotadas",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No mostrar imagen (solo guardar/imprimir)",
    )

    args = parser.parse_args()

    # Cargar configuración
    config = load_config()

    # Override de modelo si se especifica
    if args.model:
        config["model"]["pretrained"] = False
        config["paths"]["models"] = str(Path(args.model).parent)

    # Override de confianza
    if args.conf:
        config["model"]["conf_threshold"] = args.conf

    # Crear detector
    print("🚗 NeuroDrive — Detección de Vehículos")
    print("=" * 40)
    detector = VisionVialDetector(config)

    image_path = Path(args.image)

    if image_path.is_dir():
        # Procesar directorio
        print(f"\n📁 Procesando directorio: {image_path}\n")
        results = detector.detect_batch(str(image_path))

        # Mostrar resumen
        stats = detector.get_stats(results)
        print("\n📊 Resumen:")
        for clase, data in stats.items():
            if clase.startswith("_"):
                continue
            print(f"  {clase}: {data['conteo']} ({data['confianza_promedio']:.2f} avg)")

        # Guardar imágenes anotadas si se pide
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            for img_name, dets in results.items():
                img_full_path = image_path / img_name
                out_path = str(output_dir / f"det_{img_name}")
                detector.visualize(str(img_full_path), dets, output_path=out_path)

    elif image_path.is_file():
        # Procesar imagen individual
        print(f"\n🖼️  Imagen: {image_path}\n")
        detections = detector.detect(str(image_path))

        # Mostrar resultados
        if not detections:
            print("No se detectaron vehículos.")
        else:
            print(f"Detecciones: {len(detections)}")
            for i, det in enumerate(detections, 1):
                bbox = det["bbox"]
                print(
                    f"  {i}. {det['clase']} "
                    f"(confianza: {det['confianza']:.2%}) "
                    f"[{bbox['x1']:.0f}, {bbox['y1']:.0f}, "
                    f"{bbox['x2']:.0f}, {bbox['y2']:.0f}]"
                )

        # Guardar imagen anotada
        if args.output:
            output_path = Path(args.output)
            if output_path.is_dir():
                output_path = output_path / f"det_{image_path.name}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            detector.visualize(str(image_path), detections, output_path=str(output_path))

    else:
        print(f"❌ No se encontró: {image_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
