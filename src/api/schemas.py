"""Esquemas Pydantic para request/response de la API."""

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Bounding box de una detección."""
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    """Una detección individual."""
    clase: str
    confianza: float = Field(ge=0, le=1)
    bbox: BBox


class DetectionResponse(BaseModel):
    """Respuesta del endpoint de detección."""
    imagen: str
    detecciones: list[Detection]
    total: int


class StatsClase(BaseModel):
    """Estadísticas de una clase."""
    conteo: int
    confianza_promedio: float
    confianza_min: float
    confianza_max: float


class StatsResponse(BaseModel):
    """Respuesta del endpoint de estadísticas."""
    clases: dict[str, StatsClase]
    total_detecciones: int


class HealthResponse(BaseModel):
    """Respuesta del endpoint de salud."""
    status: str
    modelo: str
    device: str
