"""Inference: predykcja rasy psa dla pojedynczego obrazu (top-3).

Funkcja `predict_top_k` przyjmuje obraz PIL, transformuje go do tensora,
przepuszcza przez model i zwraca k najbardziej prawdopodobnych klas wraz
z prawdopodobieństwami (po softmaxie).

Używana zarówno przez warstwę aplikacji (`app/streamlit_app.py`), jak i przez
ewentualne testy oraz skrypty CLI.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image

from data.breeds import Breed, breed_by_index
from data.preprocess import eval_transforms


@dataclass(frozen=True)
class Prediction:
    """Pojedyncza predykcja: rasa i jej prawdopodobieństwo (0.0-1.0)."""
    breed: Breed
    probability: float


def predict_top_k(model: torch.nn.Module, image: Image.Image, k: int = 3,
                  device: torch.device | str = "cpu") -> list[Prediction]:
    """Zwraca top-k predykcji rasy dla podanego obrazu PIL.

    Args:
        model: wytrenowany model w trybie ewaluacji.
        image: obraz PIL (RGB) z wgranego przez użytkownika pliku.
        k: liczba zwracanych predykcji (domyślnie 3).
        device: urządzenie inferencji ('cpu' lub 'cuda').

    Returns:
        Lista `Prediction` długości k, posortowana od najbardziej do najmniej
        prawdopodobnej rasy.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = eval_transforms()
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probabilities, k=k)

    return [
        Prediction(breed=breed_by_index(idx.item()), probability=prob.item())
        for prob, idx in zip(top_probs, top_indices)
    ]
