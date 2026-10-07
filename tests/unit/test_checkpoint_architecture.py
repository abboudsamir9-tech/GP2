"""Restored kinematic checkpoints retain the external 138-feature contract."""

import pytest
import torch

from asl_stereo.models import (
    CompactPreLNTransformerClassifier, SignSequenceClassifier,
    hand_aperture_features, hand_closure_features, position_velocity_features,
    save_training_checkpoint, temporal_hand_features,
)
from scripts.optimize_model import load_fp32_classifier


def test_default_projection_uses_exactly_286_internal_features() -> None:
    inputs = torch.randn(2, 45, 138, dtype=torch.float32)
    original = inputs.clone()
    encoded = temporal_hand_features(inputs)

    assert encoded.shape == (2, 45, 286)
    torch.testing.assert_close(encoded[..., :276], position_velocity_features(inputs))
    torch.testing.assert_close(encoded[..., 276:282], hand_aperture_features(inputs))
    torch.testing.assert_close(encoded[..., 282:286], hand_closure_features(inputs))
    torch.testing.assert_close(inputs, original)
    for model in (
        SignSequenceClassifier(5, projection_features=16, hidden_size=16),
        CompactPreLNTransformerClassifier(5, model_dim=16),
    ):
        assert model.input_features == 138
        assert model.feature_projection[0].in_features == 286
        logits, probabilities = model.eval()(inputs)
        assert logits.shape == probabilities.shape == (2, 5)
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(2))


def test_optimizer_reconstructs_checkpoint_dimensions_and_predictions(tmp_path) -> None:
    source = SignSequenceClassifier(5, projection_features=16, hidden_size=16).eval()
    path = save_training_checkpoint(
        source, tmp_path / "best_model.pth", num_classes=5,
        window_size=45, feature_dim=138, epoch=33, best_val_acc=0.78,
    )

    restored, metadata = load_fp32_classifier(path)

    assert restored.feature_projection[0].weight.shape == (16, 286)
    assert restored.temporal_backbone.hidden_size == 16
    assert metadata["internal_feature_dim"] == 286
    assert metadata["feature_dim"] == 138
    inputs = torch.randn(1, 45, 138)
    with torch.no_grad():
        expected = source(inputs)
        actual = restored(inputs)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("internal_features", [138, 288])
def test_optimizer_rejects_incompatible_feature_schema(tmp_path, internal_features) -> None:
    source = SignSequenceClassifier(5, projection_features=16, hidden_size=16)
    path = save_training_checkpoint(
        source, tmp_path / "incompatible.pth", num_classes=5,
        window_size=45, feature_dim=138, epoch=1, best_val_acc=0.5,
    )
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["model_state_dict"]["feature_projection.0.weight"] = torch.zeros(16, internal_features)
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="restored architecture requires 286"):
        load_fp32_classifier(path)
