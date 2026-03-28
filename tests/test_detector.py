"""Tests para el módulo del detector de vehículos."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.config import load_config


def test_detector_import():
    """Verificar que el módulo del detector se puede importar."""
    from src.models.detector import VisionVialDetector
    assert VisionVialDetector is not None


def test_detector_init_with_default_config():
    """Verificar que el detector se inicializa con la configuración por defecto."""
    from src.models.detector import VisionVialDetector

    config = load_config()
    detector = VisionVialDetector(config)

    assert detector is not None
    assert detector.model is not None
    assert detector.conf_threshold == 0.25
    assert detector.iou_threshold == 0.45
    assert detector.img_size == 640


def test_detector_has_required_methods():
    """Verificar que el detector tiene todos los métodos requeridos."""
    from src.models.detector import VisionVialDetector

    config = load_config()
    detector = VisionVialDetector(config)

    assert hasattr(detector, "detect")
    assert hasattr(detector, "detect_batch")
    assert hasattr(detector, "export_results")
    assert hasattr(detector, "visualize")
    assert hasattr(detector, "get_stats")
    assert hasattr(detector, "train")


def test_get_stats_with_sample_data():
    """Verificar que get_stats genera estadísticas correctas."""
    from src.models.detector import VisionVialDetector

    config = load_config()
    detector = VisionVialDetector(config)

    # Datos de ejemplo
    sample_detections = [
        {"clase": "car", "confianza": 0.95, "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}},
        {"clase": "car", "confianza": 0.85, "bbox": {"x1": 200, "y1": 200, "x2": 300, "y2": 300}},
        {"clase": "truck", "confianza": 0.78, "bbox": {"x1": 400, "y1": 400, "x2": 500, "y2": 500}},
    ]

    stats = detector.get_stats(sample_detections)

    assert "car" in stats
    assert "truck" in stats
    assert "_total" in stats
    assert stats["car"]["conteo"] == 2
    assert stats["truck"]["conteo"] == 1
    assert stats["_total"]["detecciones"] == 3
    assert 0.89 < stats["car"]["confianza_promedio"] < 0.91


def test_get_stats_with_batch_data():
    """Verificar que get_stats funciona con formato de batch."""
    from src.models.detector import VisionVialDetector

    config = load_config()
    detector = VisionVialDetector(config)

    batch_detections = {
        "img1.jpg": [
            {"clase": "car", "confianza": 0.9, "bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}},
        ],
        "img2.jpg": [
            {"clase": "bus", "confianza": 0.7, "bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}},
        ],
    }

    stats = detector.get_stats(batch_detections)

    assert stats["_total"]["detecciones"] == 2
    assert stats["_total"]["clases_detectadas"] == 2
