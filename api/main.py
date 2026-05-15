"""FastAPI - alternatywny do Streamlit sposób serwowania modelu DogID.

Uruchomienie:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpointy:
    GET  /         - informacja o API i dostępnych endpointach
    GET  /health   - health check (200 jeśli model załadowany, 503 inaczej)
    POST /predict  - predykcja rasy (multipart/form-data z polem `image`)

Model jest ładowany raz przy starcie aplikacji i przechowywany w stanie modułu.
Pozwala to obsłużyć wiele żądań bez ponownego wczytywania wag z dysku.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from data.breeds import num_classes
from model.architecture import load_trained_model
from model.predict import predict_top_k

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = PROJECT_ROOT / "models" / "dogid.pt"

TOP_K = 3

app = FastAPI(
    title="DogID API",
    description="Klasyfikator rasy psa - zwraca top-3 predykcje na podstawie obrazu.",
    version="1.0.0",
)


def _try_load_model() -> tuple[Any | None, str | None]:
    """Ładuje model przy starcie aplikacji.

    Zwraca krotkę `(model, error_message)`. Dokładnie jedno pole jest None.
    Model jest opcjonalny - aplikacja startuje nawet bez wag, ale endpoint
    `/predict` będzie wtedy zwracał 503.
    """
    if not WEIGHTS_PATH.exists():
        return None, (
            f"Brak wag modelu w {WEIGHTS_PATH}. "
            "Uruchom najpierw: python -m model.train"
        )
    try:
        return load_trained_model(str(WEIGHTS_PATH), num_classes=num_classes()), None
    except (RuntimeError, OSError) as exc:
        return None, f"Nie udało się załadować modelu: {exc}"


_MODEL, _MODEL_ERROR = _try_load_model()


@app.get("/")
def root() -> dict[str, Any]:
    """Zwraca informację o usłudze i liście dostępnych endpointów."""
    return {
        "service": "DogID API",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Health check (200 jeśli model załadowany)",
            "POST /predict": "Predykcja rasy psa (multipart/form-data, pole 'image')",
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Health check - zwraca 503, jeśli model nie został załadowany."""
    if _MODEL is None:
        raise HTTPException(status_code=503, detail=_MODEL_ERROR)
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, Any]:
    """Klasyfikuje rasę psa na podstawie wgranego obrazu.

    Args:
        image: plik obrazu (JPEG, PNG) przesłany jako multipart/form-data.

    Returns:
        JSON z listą top-3 predykcji - każda zawiera polską nazwę rasy,
        prawdopodobieństwo (0-1) i krótki opis rasy.
    """
    if _MODEL is None:
        raise HTTPException(status_code=503, detail=_MODEL_ERROR)

    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
        pil_image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Nieprawidłowy obraz: {exc}",
        ) from exc

    predictions = predict_top_k(_MODEL, pil_image, k=TOP_K)
    return {
        "predictions": [
            {
                "breed": pred.breed.name_pl,
                "probability": pred.probability,
                "description": pred.breed.description,
            }
            for pred in predictions
        ]
    }
