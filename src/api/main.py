"""Aplicación FastAPI para detección de vehículos.

Uso:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.api.schemas import HealthResponse
from src.config import load_config, get_device

app = FastAPI(
    title="NeuroDrive API",
    description="API de detección y clasificación de vehículos en carreteras",
    version="0.1.0",
)

# CORS para permitir requests del frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, restringir a dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar rutas
app.include_router(router, prefix="/api/v1", tags=["detección"])


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Verifica el estado de la API y el modelo."""
    config = load_config()
    device = get_device(config)
    modelo = config.get("model", {}).get("architecture", "desconocido")

    return HealthResponse(
        status="ok",
        modelo=modelo,
        device=device,
    )


@app.get("/")
async def root() -> dict:
    """Endpoint raíz con información básica."""
    return {
        "proyecto": "NeuroDrive",
        "descripcion": "API de detección de vehículos en carreteras",
        "version": "0.1.0",
        "docs": "/docs",
    }
