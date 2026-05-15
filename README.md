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
├── api/                 # Warstwa API
│   └── main.py          # REST API w FastAPI (/health, /metrics, /predict)
├── tests/               # Testy jednostkowe (pytest)
├── notebooks/           # Eksploracja danych (Jupyter, EDA + confusion matrix)
├── experiments/         # Eksperymenty porównawcze (compare_models.py)
├── models/              # Wytrenowane wagi (.pt)
├── docs/                # Dokumenty zaliczeniowe
├── .github/workflows/   # CI (GitHub Actions: pylint + pytest)
├── Dockerfile           # Multi-stage build obrazu produkcyjnego
├── docker-compose.yml   # Orkiestracja serwisów Streamlit + API
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

Aplikacja udostępnia dwa sposoby serwowania modelu — interfejs webowy
(Streamlit) i REST API (FastAPI). Można uruchomić je natywnie albo
w kontenerach Docker.

### Natywnie (Streamlit, port 8501)

```bash
streamlit run app/streamlit_app.py
```

Otworzy się pod `http://localhost:8501`. Wgraj zdjęcie psa (JPG/PNG)
– otrzymasz trzy najbardziej prawdopodobne rasy z opisem.

### Natywnie (REST API, port 8000)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Endpointy:
- `GET /` — informacja o usłudze
- `GET /health` — health check (200 jeśli model załadowany)
- `POST /predict` — predykcja (multipart/form-data, pole `image`)

Przykładowe wywołanie:
```bash
curl -X POST -F "image=@pies.jpg" http://localhost:8000/predict
```

Dokumentacja interaktywna (Swagger UI) pod `http://localhost:8000/docs`.

### W kontenerach (Docker Compose)

Najbardziej przenośny sposób — wystarczy mieć zainstalowany Docker.

```bash
docker compose up --build
```

Po zbudowaniu obrazu (~5 min) ruszą oba serwisy:
- Streamlit: `http://localhost:8501`
- FastAPI: `http://localhost:8000`

Model (`models/dogid.pt`) jest podpinany jako wolumen — można go podmienić
bez rebuildowania obrazu.

Zatrzymanie: `docker compose down`.

## Wspierane rasy

Aplikacja rozpoznaje 15 ras: Labrador Retriever, Golden Retriever,
Owczarek Niemiecki, Buldog Francuski, Beagle, Pudel, Yorkshire Terrier,
Husky Syberyjski, Border Collie, Mops, Jamnik, Chihuahua, Bokser,
Rottweiler, Maltańczyk.

Pełna lista z opisami w pliku `data/breeds.py`.

## Development

Praca nad kodem (poza samym uruchomieniem aplikacji):

### Zależności dewelopеrskie

Oprócz `requirements.txt` (potrzebnego do uruchomienia aplikacji) dostępne są
narzędzia developerskie (pylint, pytest) w `requirements-dev.txt`:

```bash
pip install -r requirements-dev.txt
```

### Analiza statyczna kodu (pylint)

Projekt utrzymuje ocenę **10.00/10** w pylint na całym kodzie aplikacji.
Konfiguracja znajduje się w `.pylintrc`.

```bash
pylint data model app
```

### Struktura katalogów generowanych przy uruchomieniu

Następujące katalogi są tworzone automatycznie po pierwszym uruchomieniu pipeline'u
i są ignorowane przez Git (zob. `.gitignore`):

- `data/processed/` — przefiltrowane i podzielone dane (`train/`, `val/`, `test/`)
- `models/dogid.pt` — wytrenowany model (artefakt binarny, ~10 MB)
- `venv/` — wirtualne środowisko Python

### Eksploracja danych (notebook)

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

Notebook generuje wykresy:
- rozkład klas w zbiorach train/val/test,
- przykładowe zdjęcia z każdej rasy,
- preview augmentacji (oryginał vs po transformacji),
- **confusion matrix** na zbiorze testowym (wymaga wytrenowanego modelu).

### Eksperymenty komparatywne

Porównanie trzech backbone'ów ImageNet (MobileNetV2 / ResNet18 / EfficientNet-B0):

```bash
python -m experiments.compare_models
```

Skrypt trenuje każdy model przez 2 epoki na tym samym podziale danych,
mierzy val accuracy, czas treningu i liczbę parametrów. Wynik trafia do
`experiments/comparison.png`.

### Monitoring API (Prometheus)

REST API udostępnia metryki pod `GET /metrics` w formacie Prometheus:

- `dogid_predict_requests_total` — counter zapytań
- `dogid_predict_errors_total{error_type=...}` — counter błędów
- `dogid_predict_duration_seconds` — histogram czasu predykcji
- `dogid_model_loaded` — gauge (1 jeśli model załadowany, 0 inaczej)

Można podpiąć do Prometheusa albo zaciągnąć ad-hoc:

```bash
curl http://localhost:8000/metrics
```

## Autorzy

Projekt zaliczeniowy SUML, semestr letni 2025/2026.

Grupa: *do uzupełnienia (imię, nazwisko, numer indeksu)*.

## Licencja

[MIT](LICENSE)
