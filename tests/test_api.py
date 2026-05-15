"""Testy endpointów FastAPI w `api.main`.

Testy używają `TestClient` z FastAPI - nie wymagają uruchomionego serwera.
Sprawdzamy poprawność kontraktu HTTP, niezależnie od konkretnych wartości predykcji.
"""
import io

from fastapi.testclient import TestClient
from PIL import Image

from api.main import app

client = TestClient(app)


def _dummy_jpeg_bytes() -> bytes:
    """Zwraca bajty syntetycznego obrazu JPEG do testów uploadu."""
    image = Image.new("RGB", (300, 400), color=(120, 150, 80))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_root_returns_service_metadata() -> None:
    """GET / zwraca informację o usłudze i liście endpointów."""
    response = client.get("/")
    assert response.status_code == 200

    body = response.json()
    assert body["service"] == "DogID API"
    assert "endpoints" in body


def test_health_endpoint_returns_200_when_model_loaded() -> None:
    """GET /health zwraca 200 i `model_loaded=true` po pomyślnym załadowaniu wag."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_top_three_predictions() -> None:
    """POST /predict zwraca dokładnie 3 predykcje, każda z polami breed/prob/desc."""
    files = {"image": ("test.jpg", _dummy_jpeg_bytes(), "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200

    predictions = response.json()["predictions"]
    assert len(predictions) == 3
    for pred in predictions:
        assert "breed" in pred
        assert "probability" in pred
        assert "description" in pred
        assert 0.0 <= pred["probability"] <= 1.0


def test_predict_rejects_invalid_image() -> None:
    """POST /predict zwraca 400 dla pliku, który nie jest obrazem."""
    files = {"image": ("garbage.txt", b"this is not an image", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Nieprawidłowy obraz" in response.json()["detail"]
