"""Konfiguracja sesji pytest.

Top-level code w `conftest.py` wykonuje się **przed** importem modułów testowych
- to wykorzystujemy do upewnienia się, że `models/dogid.pt` istnieje, zanim
`api.main` go załaduje przy imporcie.

W CI (GitHub Actions) wagi modelu nie są commitowane (są w `.gitignore`),
więc tworzymy tu "stub" - model o losowych wagach z poprawną architekturą.
Testy API sprawdzają kontrakt HTTP (status code, kształt JSON), nie wartości
predykcji, więc stub jest wystarczający.

Lokalnie, gdzie użytkownik ma już wytrenowany model, ten kod nic nie robi.
"""
from pathlib import Path

import torch

from data.breeds import num_classes
from model.architecture import build_model

_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models" / "dogid.pt"


def _ensure_stub_model_exists() -> None:
    """Tworzy losowy `dogid.pt` jeśli nie istnieje - dla testów CI bez treningu."""
    if _WEIGHTS_PATH.exists():
        return
    _WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stub = build_model(num_classes=num_classes(), freeze_backbone=False)
    torch.save(stub.state_dict(), _WEIGHTS_PATH)


_ensure_stub_model_exists()
