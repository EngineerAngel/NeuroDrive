"""Tests para el módulo de configuración."""

from pathlib import Path

import pytest

from src.config import load_config, PROJECT_ROOT


def test_load_default_config():
    """Verificar que la configuración por defecto carga correctamente."""
    config = load_config()

    assert config is not None
    assert "project" in config
    assert config["project"]["name"] == "visionvial"
    assert config["project"]["version"] == "0.1.0"


def test_config_has_model_section():
    """Verificar que la sección de modelo existe y tiene valores válidos."""
    config = load_config()

    assert "model" in config
    assert "architecture" in config["model"]
    assert "img_size" in config["model"]
    assert config["model"]["img_size"] == 640


def test_config_has_paths():
    """Verificar que las rutas están configuradas."""
    config = load_config()

    assert "paths" in config
    assert "data_raw" in config["paths"]
    assert "models" in config["paths"]
    assert "results" in config["paths"]


def test_config_paths_are_resolved():
    """Verificar que las rutas se resuelven al directorio del proyecto."""
    config = load_config()

    for key, path in config["paths"].items():
        assert Path(path).is_absolute(), f"Ruta '{key}' no es absoluta: {path}"


def test_config_has_classes():
    """Verificar que las clases de detección están definidas."""
    config = load_config()

    assert "classes" in config
    assert isinstance(config["classes"], list)
    assert len(config["classes"]) > 0
    assert "car" in config["classes"]


def test_load_config_with_override():
    """Verificar que se puede cargar un override de configuración."""
    config = load_config("configs/training/baseline.yaml")

    assert "training" in config
    assert config["training"]["name"] == "baseline_tier1"
    assert config["training"]["epochs"] == 50


def test_default_yaml_exists():
    """Verificar que el archivo de configuración por defecto existe."""
    default_path = PROJECT_ROOT / "configs" / "default.yaml"
    assert default_path.exists(), f"No se encontró: {default_path}"


def test_dataset_configs_exist():
    """Verificar que existen archivos de configuración de datasets."""
    configs_dir = PROJECT_ROOT / "configs" / "datasets"
    yaml_files = list(configs_dir.glob("*.yaml"))
    assert len(yaml_files) > 0, "No se encontraron configuraciones de datasets"
