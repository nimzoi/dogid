"""Testy modułu `model.predict` - inference dla pojedynczego obrazu."""
import math

from PIL import Image

from data.breeds import num_classes
from model.architecture import build_model
from model.predict import Prediction, predict_top_k


def _dummy_rgb_image() -> Image.Image:
    """Tworzy syntetyczny obraz RGB do testów (bez potrzeby plików na dysku)."""
    return Image.new("RGB", (300, 400), color=(120, 150, 80))


def test_predict_top_k_returns_requested_count() -> None:
    """Funkcja zwraca dokładnie `k` predykcji (domyślnie 3)."""
    model = build_model(num_classes=num_classes())
    model.eval()
    predictions = predict_top_k(model, _dummy_rgb_image(), k=3)
    assert len(predictions) == 3


def test_predictions_sorted_by_probability_descending() -> None:
    """Pierwsza predykcja ma najwyższe prawdopodobieństwo, ostatnia najniższe."""
    model = build_model(num_classes=num_classes())
    model.eval()
    predictions = predict_top_k(model, _dummy_rgb_image(), k=5)

    probabilities = [p.probability for p in predictions]
    assert probabilities == sorted(probabilities, reverse=True)


def test_each_probability_in_valid_range() -> None:
    """Każde prawdopodobieństwo jest w przedziale [0, 1]."""
    model = build_model(num_classes=num_classes())
    model.eval()
    predictions = predict_top_k(model, _dummy_rgb_image(), k=3)

    for pred in predictions:
        assert 0.0 <= pred.probability <= 1.0


def test_handles_non_rgb_image_via_conversion() -> None:
    """Predykcja nie wywala się na obrazach w trybach innych niż RGB (np. grayscale)."""
    model = build_model(num_classes=num_classes())
    model.eval()
    grayscale = Image.new("L", (300, 400), color=128)

    predictions = predict_top_k(model, grayscale, k=3)
    assert len(predictions) == 3
    assert all(isinstance(p, Prediction) for p in predictions)


def test_softmax_probabilities_sum_to_one_when_k_equals_num_classes() -> None:
    """Suma wszystkich prawdopodobieństw (k = num_classes) wynosi 1.0 (softmax)."""
    n_classes = num_classes()
    model = build_model(num_classes=n_classes)
    model.eval()
    predictions = predict_top_k(model, _dummy_rgb_image(), k=n_classes)

    total = sum(p.probability for p in predictions)
    assert math.isclose(total, 1.0, abs_tol=1e-4)
