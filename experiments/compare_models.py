"""Porównanie trzech backbone'ów ImageNet dla zadania DogID.

Każdy model trenujemy przez `EPOCHS_PER_MODEL` epok z zamrożonym backbonem
(tylko dotrenowanie głowy klasyfikacyjnej) na tym samym podziale danych.

Mierzymy:
    - najlepsze val accuracy
    - czas treningu (wall clock)
    - liczba parametrów trenowalnych (rozmiar głowy)
    - łączna liczba parametrów modelu

Wynik trafia do `experiments/comparison.png` (wykres) oraz tabelki w stdout.

Skrypt służy uzasadnieniu wyboru MobileNetV2 jako modelu produkcyjnego.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import models

from data.breeds import num_classes
from data.loader import build_loader
from model.train import evaluate, train_one_epoch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
COMPARISON_PLOT = EXPERIMENTS_DIR / "comparison.png"

EPOCHS_PER_MODEL = 2
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ExperimentResult:
    """Wynik treningu pojedynczego modelu w porównaniu."""
    name: str
    best_val_acc: float
    training_time_sec: float
    trainable_params: int
    total_params: int


def build_mobilenet_v2(out_classes: int) -> nn.Module:
    """MobileNetV2 - model produkcyjny (referencja)."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(in_features, out_classes),
    )
    return model


def build_resnet18(out_classes: int) -> nn.Module:
    """ResNet18 - klasyczna głęboka sieć rezydualna."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(in_features, out_classes),
    )
    return model


def build_efficientnet_b0(out_classes: int) -> nn.Module:
    """EfficientNet-B0 - nowoczesny model o korzystnym stosunku jakość/rozmiar."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(in_features, out_classes),
    )
    return model


def train_and_eval(name: str, model_factory: Callable[[int], nn.Module]) -> ExperimentResult:
    """Trenuje model przez `EPOCHS_PER_MODEL` epok i zwraca metryki."""
    print(f"\n=== {name} ===")

    train_loader = build_loader("train")
    val_loader = build_loader("val")

    model = model_factory(num_classes()).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=LEARNING_RATE)

    best_val_acc = 0.0
    start = time.perf_counter()

    for epoch in range(1, EPOCHS_PER_MODEL + 1):
        _, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        _, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        print(f"  Epoka {epoch}/{EPOCHS_PER_MODEL}  train={train_acc:.3f}  val={val_acc:.3f}")
        best_val_acc = max(best_val_acc, val_acc)

    elapsed = time.perf_counter() - start
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())

    return ExperimentResult(
        name=name,
        best_val_acc=best_val_acc,
        training_time_sec=elapsed,
        trainable_params=trainable_count,
        total_params=total_count,
    )


def render_comparison_plot(results: list[ExperimentResult]) -> None:
    """Generuje wykres 1x3 porównujący accuracy, czas i rozmiar modeli."""
    _, axes = plt.subplots(1, 3, figsize=(15, 4))
    names = [r.name for r in results]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    axes[0].bar(names, [r.best_val_acc for r in results], color=colors)
    axes[0].set_title("Najlepsza val accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("accuracy")

    axes[1].bar(names, [r.training_time_sec for r in results], color=colors)
    axes[1].set_title(f"Czas treningu ({EPOCHS_PER_MODEL} epoki, CPU)")
    axes[1].set_ylabel("sekundy")

    axes[2].bar(names, [r.total_params / 1e6 for r in results], color=colors)
    axes[2].set_title("Łączna liczba parametrów")
    axes[2].set_ylabel("miliony")

    plt.tight_layout()
    plt.savefig(COMPARISON_PLOT, dpi=100, bbox_inches="tight")
    print(f"\nWykres porównawczy: {COMPARISON_PLOT}")


def print_summary_table(results: list[ExperimentResult]) -> None:
    """Wypisuje podsumowanie wyników w formie tabeli."""
    print("\n=== PODSUMOWANIE ===")
    print(f"{'Model':<18} {'Val acc':>10} {'Czas (s)':>10} "
          f"{'Trenowane':>12} {'Łącznie':>12}")
    for res in results:
        print(f"{res.name:<18} {res.best_val_acc:>10.3f} {res.training_time_sec:>10.1f} "
              f"{res.trainable_params:>12,} {res.total_params:>12,}")


def main() -> None:
    """Trenuje 3 modele kolejno, generuje wykres i tabelę podsumowującą."""
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    print(f"Urządzenie: {DEVICE}")
    print(f"Epok na model: {EPOCHS_PER_MODEL}")

    experiments = [
        ("MobileNetV2", build_mobilenet_v2),
        ("ResNet18", build_resnet18),
        ("EfficientNet-B0", build_efficientnet_b0),
    ]
    results = [train_and_eval(name, factory) for name, factory in experiments]

    print_summary_table(results)
    render_comparison_plot(results)


if __name__ == "__main__":
    main()
