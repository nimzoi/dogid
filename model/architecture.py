"""Architektura modelu DogID: transfer learning na MobileNetV2.

Pomysł:
    1. Backbone MobileNetV2 ze wstępnie wyuczonymi wagami ImageNet (zamrożony).
    2. Wymieniamy oryginalną głowę klasyfikacyjną (1000 klas ImageNet) na własną:
       GlobalAveragePooling -> Dropout(0.3) -> Linear(num_classes).
    3. Trenujemy wyłącznie wagi nowej głowy - to drastycznie skraca czas treningu
       i pozwala uzyskać sensowne wyniki nawet na małym zbiorze danych.

Wagi backbone-u są zamrażane przez ustawienie `requires_grad=False`, dzięki czemu
optimizer pomija je przy aktualizacji.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models

DROPOUT_RATE = 0.3


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """Tworzy model MobileNetV2 z głową dostosowaną do `num_classes` ras psów.

    Args:
        num_classes: liczba klas wyjściowych (= liczba wspieranych ras).
        freeze_backbone: jeśli True, zamraża wagi backbone-u.
            Trening dotyczy wtedy wyłącznie nowej głowy klasyfikacyjnej.

    Returns:
        Skonfigurowany model `nn.Module` gotowy do treningu/inferencji.
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # MobileNetV2 ma klasyfikator postaci: Dropout -> Linear(1280, 1000)
    # Wymieniamy go na: Dropout -> Linear(1280, num_classes)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(in_features, num_classes),
    )

    return model


def load_trained_model(weights_path: str, num_classes: int,
                       device: torch.device | str = "cpu") -> nn.Module:
    """Ładuje model z zapisanymi wagami z pliku `.pt`.

    Args:
        weights_path: ścieżka do pliku `.pt` z `state_dict`.
        num_classes: liczba klas (musi się zgadzać z zapisanym modelem).
        device: urządzenie, na które ładujemy wagi ('cpu' lub 'cuda').

    Returns:
        Model w trybie ewaluacji (`model.eval()`), gotowy do inferencji.
    """
    model = build_model(num_classes=num_classes, freeze_backbone=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
