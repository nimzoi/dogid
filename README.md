# DogID — "Co to za pies?"

Aplikacja webowa klasyfikująca rasę psa ze zdjęcia. Projekt zaliczeniowy
z przedmiotu **SUML** (Środowiska uruchomieniowe Machine Learning), PJATK.

Model wykorzystuje **transfer learning** na sieci **MobileNetV2** (wagi ImageNet),
dostrojony do rozpoznawania 15 popularnych ras psów z podzbioru
[Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).

Po wgraniu zdjęcia aplikacja zwraca trzy najbardziej prawdopodobne rasy
wraz z procentowym poziomem pewności oraz krótkim opisem rasy o najwyższym
prawdopodobieństwie.

## Architektura

Kod podzielony na trzy warstwy zgodnie z wymaganiami projektu:

```
dogid/
├── data/                # Warstwa danych
│   ├── breeds.py        # Lista wspieranych ras psów
│   ├── download.py      # Pobranie i filtrowanie Stanford Dogs
│   ├── preprocess.py    # Transformacje obrazów (resize, normalizacja, augmentacja)
│   └── loader.py        # PyTorch DataLoader dla zbiorów train/val/test
├── model/               # Warstwa modelu
│   ├── architecture.py  # MobileNetV2 z dostosowaną głową klasyfikacyjną
│   ├── train.py         # Pętla treningowa i ewaluacja
│   └── predict.py       # Inference (predykcja top-3 dla pojedynczego obrazu)
├── app/                 # Warstwa aplikacji
│   └── streamlit_app.py # Interfejs Streamlit
├── models/              # Wytrenowane wagi (.pt)
├── docs/                # Dokumenty zaliczeniowe
├── requirements.txt
├── .pylintrc
└── README.md
```

## Wymagania

- Python ≥ 3.10
- System: Windows / Linux / macOS (model trenowany i uruchamiany na CPU)
- Pakiety: zob. `requirements.txt`

## Instalacja

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

## Pobranie danych i trening modelu

```bash
# Pobiera Stanford Dogs i filtruje do 15 wspieranych ras
python -m data.download

# Trenuje model (zapis do models/dogid.pt)
python -m model.train
```

Trening zajmuje ok. 20–40 minut na CPU. Na maszynie z GPU (CUDA) wystarczy
ustawić `DEVICE=cuda` w `model/train.py`.

## Uruchomienie aplikacji

```bash
streamlit run app/streamlit_app.py
```

Aplikacja otworzy się pod adresem `http://localhost:8501`. Wgraj zdjęcie psa
(JPG/PNG) – w odpowiedzi otrzymasz trzy najbardziej prawdopodobne rasy oraz
informację o rasie z najwyższym prawdopodobieństwem.

## Wspierane rasy

Aplikacja rozpoznaje 15 ras: Labrador Retriever, Golden Retriever,
Owczarek Niemiecki, Buldog Francuski, Beagle, Pudel, Yorkshire Terrier,
Husky Syberyjski, Border Collie, Mops, Jamnik, Chihuahua, Bokser,
Rottweiler, Maltańczyk.

Pełna lista z opisami w pliku `data/breeds.py`.

## Autorzy

Projekt zaliczeniowy SUML, semestr letni 2025/2026.

Grupa: *do uzupełnienia (imię, nazwisko, numer indeksu)*.

## Licencja

MIT
