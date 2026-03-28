"""Script para descargar datasets de Kaggle.

Uso:
    python scripts/download_datasets.py --dataset tier1_road_vehicles
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --list
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


# Directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIGS_DIR = PROJECT_ROOT / "configs" / "datasets"
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_dataset_configs() -> dict[str, dict]:
    """Carga todas las configuraciones de datasets disponibles.

    Returns:
        Diccionario {nombre: configuración}.
    """
    datasets = {}
    for yaml_file in sorted(CONFIGS_DIR.glob("*.yaml")):
        with open(yaml_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        name = yaml_file.stem
        datasets[name] = config.get("dataset", config)
    return datasets


def list_datasets() -> None:
    """Muestra los datasets disponibles y su estado."""
    datasets = load_dataset_configs()

    print("\n📋 Datasets disponibles:")
    print("-" * 70)
    print(f"{'Nombre':<30} {'Tier':<6} {'Formato':<8} {'Estado':<12} {'Tamaño'}")
    print("-" * 70)

    for name, config in datasets.items():
        tier = config.get("tier", "?")
        fmt = config.get("format", "?")
        status = config.get("status", "?")
        size = config.get("size", "?")
        print(f"{name:<30} {tier:<6} {fmt:<8} {status:<12} {size}")

    print()


def download_dataset(name: str, force: bool = False) -> bool:
    """Descarga un dataset de Kaggle.

    Args:
        name: Nombre del dataset (stem del archivo YAML).
        force: Forzar descarga aunque ya exista.

    Returns:
        True si la descarga fue exitosa.
    """
    config_path = CONFIGS_DIR / f"{name}.yaml"
    if not config_path.exists():
        print(f"❌ No se encontró configuración: {config_path}")
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset = config.get("dataset", config)
    source = dataset.get("source", "")
    kaggle_slug = dataset.get("kaggle_slug", "")

    if source != "kaggle" or not kaggle_slug:
        print(f"⚠️  {name}: No es un dataset de Kaggle o no tiene slug configurado")
        return False

    output_dir = DATA_DIR / name
    if output_dir.exists() and not force:
        print(f"⚠️  {name}: Ya existe en {output_dir}. Usa --force para re-descargar.")
        return True

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Descargando: {dataset.get('name', name)}")
    print(f"   Slug: {kaggle_slug}")
    print(f"   Destino: {output_dir}")

    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", kaggle_slug,
                "-p", str(output_dir),
                "--unzip",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"✅ {name} descargado exitosamente")
        return True
    except FileNotFoundError:
        print(
            "❌ Kaggle CLI no encontrado. Instálalo con:\n"
            "   pip install kaggle\n"
            "   Configura tu API key en ~/.kaggle/kaggle.json"
        )
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al descargar {name}: {e.stderr}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga datasets de Kaggle para NeuroDrive"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Nombre del dataset a descargar (ej: tier1_road_vehicles)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Descargar todos los datasets de Kaggle disponibles",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="Listar datasets disponibles",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Forzar re-descarga aunque ya exista",
    )
    parser.add_argument(
        "--tier", "-t",
        type=int,
        help="Descargar solo datasets de un tier específico (1, 2, 3)",
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if args.dataset:
        download_dataset(args.dataset, force=args.force)
        return

    if args.all or args.tier:
        datasets = load_dataset_configs()
        for name, config in datasets.items():
            if args.tier and config.get("tier") != args.tier:
                continue
            if config.get("source") == "kaggle" and config.get("kaggle_slug"):
                download_dataset(name, force=args.force)
                print()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
