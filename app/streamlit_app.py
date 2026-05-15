"""Aplikacja Streamlit DogID - upload zdjęcia psa i otrzymaj top-3 rasy.

Uruchomienie:
    streamlit run app/streamlit_app.py

Aplikacja:
    1. Wczytuje wytrenowany model z pliku `models/dogid.pt`.
    2. Pozwala użytkownikowi wgrać zdjęcie psa (JPG/PNG).
    3. Wyświetla zdjęcie i listę top-3 najbardziej prawdopodobnych ras.
    4. Pokazuje opis rasy o najwyższym prawdopodobieństwie.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from data.breeds import num_classes
from model.architecture import load_trained_model
from model.predict import predict_top_k

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = PROJECT_ROOT / "models" / "dogid.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 3


@st.cache_resource
def get_model() -> torch.nn.Module:
    """Ładuje model raz i cachuje w pamięci między reloadami strony."""
    if not WEIGHTS_PATH.exists():
        st.error(
            f"Nie znaleziono wag modelu: {WEIGHTS_PATH}. "
            "Uruchom najpierw trening: `python -m model.train`."
        )
        st.stop()
    return load_trained_model(
        weights_path=str(WEIGHTS_PATH),
        num_classes=num_classes(),
        device=DEVICE,
    )


def render_header() -> None:
    """Wyświetla nagłówek strony i krótki opis aplikacji."""
    st.set_page_config(page_title="DogID — Co to za pies?", page_icon=None)
    st.title("DogID — Co to za pies?")
    st.write(
        "Wgraj zdjęcie psa, a aplikacja zwróci trzy najbardziej prawdopodobne rasy. "
        "Model wspiera 15 popularnych ras (lista w `data/breeds.py`)."
    )


def render_predictions(predictions: list) -> None:
    """Renderuje wyniki predykcji: paski prawdopodobieństwa + opis topowej rasy."""
    st.subheader("Wyniki predykcji")
    for i, pred in enumerate(predictions, start=1):
        st.write(f"**{i}. {pred.breed.name_pl}** — {pred.probability * 100:.1f}%")
        st.progress(pred.probability)

    top = predictions[0]
    st.subheader(f"O rasie: {top.breed.name_pl}")
    st.write(top.breed.description)


def main() -> None:
    """Główna pętla aplikacji Streamlit."""
    render_header()

    uploaded_file = st.file_uploader(
        "Wybierz zdjęcie psa (JPG lub PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Wgraj zdjęcie, żeby zobaczyć predykcję.")
        return

    image = Image.open(uploaded_file)
    st.image(image, caption="Wgrane zdjęcie", use_column_width=True)

    with st.spinner("Klasyfikuję..."):
        model = get_model()
        predictions = predict_top_k(model=model, image=image, k=TOP_K, device=DEVICE)

    render_predictions(predictions)


if __name__ == "__main__":
    main()
