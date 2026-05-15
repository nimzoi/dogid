"""Pobiera Stanford Dogs Dataset i przygotowuje go do treningu DogID.

Użycie:
    python -m data.download

Pipeline:
    1. Pobranie pełnego datasetu z Kaggle (przez `kagglehub`).
       Pakiet cachuje pobrane archiwum lokalnie - kolejne wywołania są szybkie.
    2. Filtracja: kopiowane są tylko katalogi 15 ras zdefiniowanych
       w `data.breeds.BREEDS`.
    3. Deterministyczny split train/val/test (80/10/10) ustalany przez
       `random.seed(RANDOM_SEED)` - dzięki czemu trening jest powtarzalny.

Struktura wyjściowa:
    data/processed/
        train/<folder_name>/*.jpg
        val/<folder_name>/*.jpg
        test/<folder_name>/*.jpg
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import kagglehub

from data.breeds import BREEDS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

KAGGLE_DATASET = "jessicali9530/stanford-dogs-dataset"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 1.0 - TRAIN_RATIO - VAL_RATIO

RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def download_stanford_dogs() -> Path:
    """Pobiera Stanford Dogs przez kagglehub i zwraca katalog z folderami ras.

    Returns:
        Ścieżka do katalogu zawierającego podkatalogi `n02...-<breed>` z obrazami.

    Raises:
        FileNotFoundError: gdy w pobranym archiwum nie udało się odnaleźć katalogu
            z folderami ras.
    """
    print(f"Pobieranie datasetu Kaggle: {KAGGLE_DATASET}")
    cache_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"Dataset gotowy w: {cache_path}")

    # Dataset bywa rozpakowywany do różnych struktur (zależnie od wersji kagglehub).
    # Sprawdzamy typowe lokalizacje, w razie czego szukamy rekurencyjnie.
    candidates = [
        cache_path,
        cache_path / "Images",
        cache_path / "images",
        cache_path / "images" / "Images",
    ]
    for candidate in candidates:
        if _contains_breed_folders(candidate):
            return candidate

    for path in cache_path.rglob("n02*"):
        if path.is_dir() and _contains_breed_folders(path.parent):
            return path.parent

    raise FileNotFoundError(
        f"Nie znaleziono katalogu z folderami ras (n02*-...) w {cache_path}."
    )


def _contains_breed_folders(path: Path) -> bool:
    """Zwraca True, jeśli katalog `path` zawiera podkatalogi rasowe `n02...`."""
    if not path.is_dir():
        return False
    return any(child.is_dir() and child.name.startswith("n02") for child in path.iterdir())


def prepare_dataset(images_dir: Path) -> dict[str, int]:
    """Filtruje 15 ras i wykonuje split train/val/test.

    Czyści istniejący katalog `data/processed/`, kopiuje obrazy z `images_dir`
    do struktury `train/val/test/<folder_name>/`.

    Args:
        images_dir: katalog źródłowy zawierający podkatalogi ras Stanford Dogs.

    Returns:
        Słownik `{split: liczba_obrazów}` przydatny do diagnostyki.

    Raises:
        FileNotFoundError: gdy któraś z 15 wspieranych ras nie istnieje w datasecie.
    """
    if PROCESSED_DIR.exists():
        print(f"Czyszczę istniejący katalog wyjściowy: {PROCESSED_DIR}")
        shutil.rmtree(PROCESSED_DIR)

    random.seed(RANDOM_SEED)
    counts = {"train": 0, "val": 0, "test": 0}

    for breed in BREEDS:
        source = images_dir / breed.folder_name
        if not source.is_dir():
            raise FileNotFoundError(
                f"Brak folderu rasy '{breed.folder_name}' w datasecie ({source}). "
                "Zweryfikuj nazwę katalogu w data/breeds.py."
            )

        images = sorted(p for p in source.iterdir()
                        if p.suffix.lower() in IMAGE_EXTENSIONS)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, files in splits.items():
            target = PROCESSED_DIR / split_name / breed.folder_name
            target.mkdir(parents=True, exist_ok=True)
            for src in files:
                shutil.copy2(src, target / src.name)
            counts[split_name] += len(files)

        print(f"  {breed.name_pl:<22} {n_total:4d} obrazów  "
              f"(train={len(splits['train'])}, "
              f"val={len(splits['val'])}, "
              f"test={len(splits['test'])})")

    return counts


def main() -> None:
    """Pełny pipeline: pobranie -> filtracja -> split."""
    images_dir = download_stanford_dogs()
    print(f"\nPrzygotowanie zbiorów train/val/test z: {images_dir}\n")
    counts = prepare_dataset(images_dir)

    total = sum(counts.values())
    print(f"\nGotowe. Łącznie {total} obrazów:")
    for split, count in counts.items():
        print(f"  {split}: {count}")
    print(f"\nDane gotowe w: {PROCESSED_DIR}")
    print("Następny krok: python -m model.train")


if __name__ == "__main__":
    main()
