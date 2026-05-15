"""Testy modułu `data.breeds` - lista wspieranych ras psów."""
from data.breeds import BREEDS, breed_by_index, breed_names_pl, num_classes


def test_supports_fifteen_breeds() -> None:
    """Aplikacja wspiera dokładnie 15 ras (zgodnie z propozycją projektu)."""
    assert num_classes() == 15
    assert len(BREEDS) == 15


def test_breeds_sorted_alphabetically_by_folder_name() -> None:
    """Kolejność `BREEDS` musi odpowiadać sortowaniu `ImageFolder`.

    Inaczej indeksy klas zwracane przez model nie zmapowałyby się poprawnie
    na rasy (bug ciężki do wykrycia w praniu).
    """
    folder_names = [breed.folder_name for breed in BREEDS]
    assert folder_names == sorted(folder_names)


def test_breed_names_pl_are_unique() -> None:
    """Polskie nazwy ras nie powtarzają się - inaczej wyniki byłyby mylące."""
    names = breed_names_pl()
    assert len(names) == len(set(names))


def test_breed_by_index_returns_correct_breed() -> None:
    """Indeks 0 zwraca pierwszą rasę (alfabetycznie najwcześniejszą)."""
    first = breed_by_index(0)
    assert first.name_pl == "Chihuahua"
    assert first.folder_name == "n02085620-Chihuahua"


def test_all_breeds_have_non_empty_fields() -> None:
    """Każda rasa ma wypełnione folder_name, name_pl i description."""
    for breed in BREEDS:
        assert breed.folder_name.startswith("n02"), f"Niepoprawny folder_name: {breed.folder_name}"
        assert breed.name_pl, f"Pusta nazwa pl dla {breed.folder_name}"
        assert len(breed.description) > 20, f"Za krótki opis dla {breed.name_pl}"
