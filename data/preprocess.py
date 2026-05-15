"""Transformacje obrazów (torchvision) używane przy treningu i inferencji.

Trening korzysta z augmentacji (random flip, rotate, crop) zwiększającej
generalizację modelu. Walidacja i inferencja używają wyłącznie deterministycznych
operacji (resize + normalizacja), żeby wyniki były powtarzalne.

Normalizacja wykorzystuje statystyki ImageNet, ponieważ bazowy model MobileNetV2
był na ImageNet wstępnie trenowany.
"""
from __future__ import annotations

from torchvision import transforms

IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transforms() -> transforms.Compose:
    """Pipeline transformacji dla zbioru treningowego (z augmentacją)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def eval_transforms() -> transforms.Compose:
    """Pipeline transformacji dla walidacji/testu/inferencji (bez augmentacji)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
