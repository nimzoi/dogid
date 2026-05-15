"""Lista 15 ras psów wspieranych przez aplikację DogID.

Dla każdej rasy przechowujemy:
    folder_name - nazwa katalogu w Stanford Dogs Dataset
    name_pl     - polska nazwa wyświetlana w interfejsie aplikacji
    description - krótki opis rasy pokazywany użytkownikowi
                  (charakter, rozmiar, potrzeby ruchowe)

WAŻNE: kolejność elementów w `BREEDS` musi odpowiadać kolejności katalogów
zwracanej przez `torchvision.datasets.ImageFolder` (sortowanie alfabetyczne
po nazwie folderu). Dzięki temu indeksy klas zwracane przez model można
bezpiecznie mapować na rasy przez `BREEDS[index]`.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Breed:
    """Reprezentuje pojedynczą rasę psa wspieraną przez model."""
    folder_name: str
    name_pl: str
    description: str


# UWAGA: zachować kolejność alfabetyczną po `folder_name` - patrz docstring modułu.
BREEDS: tuple[Breed, ...] = (
    Breed(
        folder_name="n02085620-Chihuahua",
        name_pl="Chihuahua",
        description=(
            "Najmniejsza rasa psa na świecie. Mimo rozmiaru - odważna i energiczna. "
            "Silnie przywiązuje się do jednej osoby. Wrażliwa na zimno, źle znosi "
            "obecność małych dzieci."
        ),
    ),
    Breed(
        folder_name="n02085936-Maltese_dog",
        name_pl="Maltańczyk",
        description=(
            "Mały pies towarzyszący o śnieżnobiałej, jedwabistej sierści. Łagodny, "
            "wesoły, idealny do mieszkania. Wymaga regularnego czesania "
            "i strzyżenia."
        ),
    ),
    Breed(
        folder_name="n02086240-Shih-Tzu",
        name_pl="Shih Tzu",
        description=(
            "Mały pies towarzyszący o długiej, jedwabistej sierści. Łagodny "
            "i przyjazny. Wymaga regularnej pielęgnacji - codzienne szczotkowanie "
            "i okresowe strzyżenie."
        ),
    ),
    Breed(
        folder_name="n02088364-beagle",
        name_pl="Beagle",
        description=(
            "Średni pies myśliwski o doskonałym węchu. Towarzyski, lubi towarzystwo "
            "innych psów. Niezależny - może być wyzwaniem w szkoleniu. Potrzebuje "
            "dużo ruchu."
        ),
    ),
    Breed(
        folder_name="n02094433-Yorkshire_terrier",
        name_pl="Yorkshire Terrier",
        description=(
            "Mały, odważny pies o długiej, jedwabistej sierści. Mimo wielkości - "
            "energiczny i charakterny. Wymaga codziennego czesania i regularnego "
            "strzyżenia."
        ),
    ),
    Breed(
        folder_name="n02099601-golden_retriever",
        name_pl="Golden Retriever",
        description=(
            "Średnio-duży, łagodny i cierpliwy pies. Doskonały kompan dla dzieci. "
            "Wymaga regularnego ruchu i pielęgnacji długiej, złotej sierści."
        ),
    ),
    Breed(
        folder_name="n02099712-Labrador_retriever",
        name_pl="Labrador Retriever",
        description=(
            "Duży, przyjazny pies rodzinny. Bardzo energiczny, potrzebuje codziennej "
            "dawki ruchu. Łatwy do szkolenia, świetnie sprawdza się jako pies "
            "przewodnik i ratowniczy."
        ),
    ),
    Breed(
        folder_name="n02106166-Border_collie",
        name_pl="Border Collie",
        description=(
            "Uznawany za najinteligentniejszą rasę psa. Pies pasterski o niezwykłej "
            "energii. Potrzebuje zarówno fizycznej, jak i mentalnej stymulacji. "
            "Nie nadaje się do mieszkania bez intensywnej aktywności."
        ),
    ),
    Breed(
        folder_name="n02106550-Rottweiler",
        name_pl="Rottweiler",
        description=(
            "Duży, silny pies stróżujący. Spokojny, ale czujny i obronny wobec "
            "rodziny. Wymaga konsekwentnego szkolenia i wczesnej socjalizacji. "
            "Nie dla początkujących właścicieli."
        ),
    ),
    Breed(
        folder_name="n02106662-German_shepherd",
        name_pl="Owczarek Niemiecki",
        description=(
            "Duży, inteligentny pies pracujący. Wykorzystywany w policji i wojsku. "
            "Lojalny i odważny, wymaga konsekwentnego szkolenia oraz dużej "
            "aktywności fizycznej."
        ),
    ),
    Breed(
        folder_name="n02108089-boxer",
        name_pl="Bokser",
        description=(
            "Duży pies o atletycznej sylwetce. Energiczny, lojalny, świetny do "
            "rodziny. Wymaga sporej dawki ruchu i wczesnej socjalizacji. "
            "Długo zachowuje szczenięce zachowanie."
        ),
    ),
    Breed(
        folder_name="n02108915-French_bulldog",
        name_pl="Buldog Francuski",
        description=(
            "Mały, kompaktowy pies o spokojnym usposobieniu. Idealny do mieszkania. "
            "Wrażliwy na upały (krótki pysk - oddychanie). Niski poziom aktywności."
        ),
    ),
    Breed(
        folder_name="n02110185-Siberian_husky",
        name_pl="Husky Syberyjski",
        description=(
            "Duży pies zaprzęgowy o ogromnej energii. Niezbędne wielogodzinne "
            "aktywności na świeżym powietrzu. Niezależny, słabo znosi samotność. "
            "Lubi chłodny klimat."
        ),
    ),
    Breed(
        folder_name="n02110958-pug",
        name_pl="Mops",
        description=(
            "Mały, krępy pies o charakterystycznym pomarszczonym pysku. Towarzyski "
            "i pogodny. Wrażliwy na wysokie temperatury. Niski poziom aktywności, "
            "idealny do mieszkania."
        ),
    ),
    Breed(
        folder_name="n02113799-standard_poodle",
        name_pl="Pudel",
        description=(
            "Inteligentny, hipoalergiczny pies. Występuje w trzech rozmiarach "
            "(toy, miniatura, standard). Łatwy do szkolenia, wymaga regularnej "
            "pielęgnacji kędzierzawej sierści."
        ),
    ),
)


# Sanity check - jeśli ktoś zmieni kolejność, dostaniemy błąd przy imporcie
# zamiast dziwnych predykcji przy inferencji.
assert list(BREEDS) == sorted(BREEDS, key=lambda b: b.folder_name), (
    "BREEDS musi być posortowane alfabetycznie po folder_name "
    "- zob. docstring modułu."
)


def breed_names_pl() -> list[str]:
    """Zwraca listę polskich nazw ras w kolejności użytej do treningu modelu."""
    return [breed.name_pl for breed in BREEDS]


def breed_by_index(index: int) -> Breed:
    """Pobiera rasę po indeksie klasy zwróconym przez model."""
    return BREEDS[index]


def num_classes() -> int:
    """Zwraca liczbę wspieranych klas (= rozmiar warstwy wyjściowej modelu)."""
    return len(BREEDS)
