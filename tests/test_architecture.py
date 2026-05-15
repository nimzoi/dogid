"""Testy modułu `model.architecture` - konstrukcja modelu MobileNetV2."""
import torch
from torch import nn

from data.breeds import num_classes
from model.architecture import build_model


def test_build_model_returns_module() -> None:
    """Funkcja zwraca obiekt typu nn.Module gotowy do treningu/inferencji."""
    model = build_model(num_classes=num_classes())
    assert isinstance(model, nn.Module)


def test_output_size_matches_number_of_breeds() -> None:
    """Ostatnia warstwa Linear ma `num_classes` wyjść."""
    model = build_model(num_classes=15)
    final_linear = model.classifier[-1]
    assert isinstance(final_linear, nn.Linear)
    assert final_linear.out_features == 15


def test_forward_pass_returns_logits_with_correct_shape() -> None:
    """Forward pass na pojedynczym obrazie 224x224 zwraca tensor shape (1, 15)."""
    model = build_model(num_classes=15)
    model.eval()
    dummy_image = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_image)

    assert output.shape == (1, 15)


def test_freeze_backbone_disables_grad_on_features() -> None:
    """Zamrożenie backbone (`freeze_backbone=True`) wyłącza gradienty w `model.features`."""
    model = build_model(num_classes=15, freeze_backbone=True)

    for param in model.features.parameters():
        assert param.requires_grad is False

    # Głowa klasyfikacyjna pozostaje trenowalna
    for param in model.classifier.parameters():
        assert param.requires_grad is True
