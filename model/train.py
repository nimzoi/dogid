"""Pętla treningowa i ewaluacja modelu DogID.

Użycie:
    python -m model.train

Pipeline:
    1. Buduje DataLoadery dla zbiorów train/val.
    2. Inicjalizuje model MobileNetV2 z zamrożonym backbonem.
    3. Trenuje przez `NUM_EPOCHS` epok, monitorując loss i accuracy.
    4. Zapisuje najlepszy model (po val accuracy) do `models/dogid.pt`.

Hyperparametry można zmienić u góry pliku - to celowo proste,
żeby trening był łatwy do uruchomienia bez argumentów CLI.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.breeds import num_classes
from data.loader import build_loader
from model.architecture import build_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_PATH = MODELS_DIR / "dogid.pt"

NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: optim.Optimizer, device: str) -> tuple[float, float]:
    """Wykonuje jedną epokę treningu.

    Args:
        model: model w trybie treningowym (ustawiany automatycznie).
        loader: DataLoader zbioru treningowego.
        criterion: funkcja straty (typowo CrossEntropyLoss).
        optimizer: optymalizator z parametrami modelu.
        device: 'cpu' lub 'cuda'.

    Returns:
        Krotka `(średnia_strata, dokładność)` dla całej epoki.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, desc="Trening", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        predicted = outputs.argmax(dim=1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: str) -> tuple[float, float]:
    """Oblicza loss i accuracy na zbiorze walidacyjnym/testowym.

    Wyłącza gradienty (`torch.no_grad`) - inferencja jest szybsza i nie zużywa
    pamięci na bufory backward.

    Args:
        model: model do ewaluacji.
        loader: DataLoader zbioru val lub test.
        criterion: ta sama funkcja straty co przy treningu.
        device: 'cpu' lub 'cuda'.

    Returns:
        Krotka `(średnia_strata, dokładność)` na całym zbiorze.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Walidacja", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            predicted = outputs.argmax(dim=1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    return running_loss / total, correct / total


def main() -> None:
    """Pełen pipeline treningowy z zapisem najlepszego modelu po val accuracy."""
    MODELS_DIR.mkdir(exist_ok=True)
    print(f"Urządzenie obliczeniowe: {DEVICE}")

    train_loader = build_loader("train")
    val_loader = build_loader("val")
    print(f"Train: {len(train_loader.dataset)} obrazów  |  "
          f"Val: {len(val_loader.dataset)} obrazów")

    model = build_model(num_classes=num_classes(), freeze_backbone=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # Optymalizujemy tylko parametry o requires_grad=True (nowa głowa klasyfikacyjna).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)

    best_val_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        print(f"Epoka {epoch:2d}/{NUM_EPOCHS}  "
              f"train: loss={train_loss:.4f} acc={train_acc:.3f}  |  "
              f"val: loss={val_loss:.4f} acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"  > Zapisano najlepszy model (val acc = {val_acc:.3f})")

    print(f"\nTrening zakończony. Najlepszy model: {WEIGHTS_PATH}")
    print(f"Najlepsza dokładność walidacji: {best_val_acc:.3f}")
    print("Następny krok: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
