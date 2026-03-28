"""Módulo de configuración centralizada.

Carga la configuración desde archivos YAML y variables de entorno.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# Directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def _deep_merge(base: dict, override: dict) -> dict:
    """Combina dos diccionarios recursivamente. Override tiene prioridad."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_paths(config: dict, root: Path) -> dict:
    """Resuelve rutas relativas en la configuración al directorio del proyecto."""
    if "paths" in config:
        for key, value in config["paths"].items():
            if isinstance(value, str) and not os.path.isabs(value):
                config["paths"][key] = str(root / value)
    return config


def _apply_env_overrides(config: dict) -> dict:
    """Aplica variables de entorno como overrides de configuración."""
    env_mapping = {
        "DATASET_PATH": ("paths", "data_raw"),
        "MODEL_PATH": ("paths", "models"),
        "RESULTS_PATH": ("paths", "results"),
        "DEVICE": ("device",),
    }

    for env_var, config_path in env_mapping.items():
        value = os.getenv(env_var)
        if value is not None:
            current = config
            for key in config_path[:-1]:
                current = current.setdefault(key, {})
            current[config_path[-1]] = value

    return config


def load_config(override_path: str | None = None) -> dict[str, Any]:
    """Carga la configuración del proyecto.

    Args:
        override_path: Ruta opcional a un archivo YAML que sobreescribe la config base.

    Returns:
        Diccionario con la configuración completa.
    """
    # Cargar variables de entorno
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Cargar configuración base
    default_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(default_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Aplicar override si se proporciona
    if override_path is not None:
        override_full_path = Path(override_path)
        if not override_full_path.is_absolute():
            override_full_path = PROJECT_ROOT / override_full_path
        with open(override_full_path, "r", encoding="utf-8") as f:
            override_config = yaml.safe_load(f)
        config = _deep_merge(config, override_config)

    # Resolver rutas y aplicar variables de entorno
    config = _resolve_paths(config, PROJECT_ROOT)
    config = _apply_env_overrides(config)

    return config


def print_config(config: dict, indent: int = 0) -> None:
    """Imprime la configuración activa de forma legible.

    Args:
        config: Diccionario de configuración.
        indent: Nivel de indentación actual.
    """
    for key, value in config.items():
        prefix = "  " * indent
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_config(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")


def get_device(config: dict) -> str:
    """Determina el dispositivo a usar (CPU o CUDA).

    Args:
        config: Diccionario de configuración.

    Returns:
        String con el dispositivo: 'cpu', 'cuda:0', etc.
    """
    import torch

    device = config.get("device", "auto")
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device
