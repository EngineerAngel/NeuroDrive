"""Rutas de la API de detección de vehículos."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException

from src.api.schemas import DetectionResponse, StatsResponse, Detection, BBox, StatsClase
from src.config import load_config
from src.models.detector import VisionVialDetector

router = APIRouter()

# Cargar detector una sola vez
_config = load_config()
_detector = VisionVialDetector(_config)


@router.post("/detect", response_model=DetectionResponse)
async def detect_vehicles(file: UploadFile = File(...)) -> DetectionResponse:
    """Detecta vehículos en una imagen subida.

    Args:
        file: Archivo de imagen (JPG, PNG).

    Returns:
        Detecciones encontradas en la imagen.
    """
    # Validar tipo de archivo
    valid_types = {"image/jpeg", "image/png", "image/bmp"}
    if file.content_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado: {file.content_type}. "
                   f"Usa: {', '.join(valid_types)}",
        )

    # Guardar imagen temporal
    suffix = Path(file.filename or "image.jpg").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        raw_detections = _detector.detect(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    detections = [
        Detection(
            clase=d["clase"],
            confianza=d["confianza"],
            bbox=BBox(**d["bbox"]),
        )
        for d in raw_detections
    ]

    return DetectionResponse(
        imagen=file.filename or "imagen",
        detecciones=detections,
        total=len(detections),
    )


@router.post("/detect/stats", response_model=StatsResponse)
async def detect_with_stats(file: UploadFile = File(...)) -> StatsResponse:
    """Detecta vehículos y retorna estadísticas por clase.

    Args:
        file: Archivo de imagen (JPG, PNG).

    Returns:
        Estadísticas de detección agrupadas por clase.
    """
    suffix = Path(file.filename or "image.jpg").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        raw_detections = _detector.detect(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    stats = _detector.get_stats(raw_detections)
    total_info = stats.pop("_total", {})

    clases = {
        clase: StatsClase(**data)
        for clase, data in stats.items()
    }

    return StatsResponse(
        clases=clases,
        total_detecciones=total_info.get("detecciones", 0),
    )
