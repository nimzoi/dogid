"""Fabryka PyTorch DataLoader-ów dla zbiorów train/val/test.

Korzysta z `torchvision.datasets.ImageFolder`, który automatycznie czyta strukturę
katalogów `data/processed/<split>/<breed_name>/<image>.jpg`.

TODO: zaimplementować po przygotowaniu danych w data/download.py.
"""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data.preprocess import eval_transforms, train_transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

BATCH_SIZE = 32
NUM_WORKERS = 0  # Windows: 0 unika problemów z multiprocessing w Streamlit


def build_loader(split: str, batch_size: int = BATCH_SIZE,
                 shuffle: bool | None = None) -> DataLoader:
    """Buduje DataLoader dla podanego splitu ('train', 'val' lub 'test').

    Trening używa augmentacji i tasowania, walidacja/test - transformacji
    deterministycznych bez tasowania.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Nieznany split: {split!r}. Oczekiwano 'train' | 'val' | 'test'.")

    split_dir = PROCESSED_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Brak katalogu {split_dir}. Uruchom najpierw: python -m data.download"
        )

    is_train = split == "train"
    transform = train_transforms() if is_train else eval_transforms()
    dataset = ImageFolder(root=str(split_dir), transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if shuffle is not None else is_train,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
