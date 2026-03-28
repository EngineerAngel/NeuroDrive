"""Tests para la API REST."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Cliente de prueba para la API."""
    return TestClient(app)


def test_root_endpoint(client):
    """Verificar que el endpoint raíz responde correctamente."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["proyecto"] == "NeuroDrive"
    assert "version" in data
    assert "docs" in data


def test_health_endpoint(client):
    """Verificar que el endpoint de salud responde correctamente."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "modelo" in data
    assert "device" in data


def test_detect_endpoint_requires_file(client):
    """Verificar que el endpoint de detección requiere un archivo."""
    response = client.post("/api/v1/detect")
    assert response.status_code == 422  # Validation error


def test_detect_endpoint_rejects_invalid_type(client):
    """Verificar que el endpoint rechaza archivos no válidos."""
    response = client.post(
        "/api/v1/detect",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_detect_stats_endpoint_requires_file(client):
    """Verificar que el endpoint de stats requiere un archivo."""
    response = client.post("/api/v1/detect/stats")
    assert response.status_code == 422
