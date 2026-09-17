import numpy as np
import pytest
import torch

from asl_stereo.models import (
    CompactPreLNTransformerClassifier,
    InferenceEngine,
    SignSequenceClassifier,
)
from asl_stereo.models.inference import InferenceResult


def test_bilstm_classifier_returns_logits_and_probability_distribution() -> None:
    torch.manual_seed(7)
    model = SignSequenceClassifier(num_classes=5).eval()
    inputs = torch.randn(1, 45, 138, dtype=torch.float32)

    with torch.no_grad():
        logits, probabilities = model(inputs)

    assert logits.shape == (1, 5)
    assert probabilities.shape == (1, 5)
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones(1), atol=1e-6, rtol=1e-6
    )
    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)


def test_transformer_variant_returns_expected_shapes() -> None:
    model = CompactPreLNTransformerClassifier(num_classes=4).eval()
    inputs = torch.randn(1, 45, 138)
    with torch.no_grad():
        logits, probabilities = model(inputs)
    assert logits.shape == probabilities.shape == (1, 4)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(1))


def test_inference_engine_records_measurable_cpu_latency() -> None:
    model = SignSequenceClassifier(num_classes=3).eval()
    engine = InferenceEngine(model, ["hello", "help", "thanks"])

    result = engine.predict(torch.randn(1, 45, 138, dtype=torch.float32))

    assert result.predicted_gloss in {"hello", "help", "thanks"}
    assert 0.0 <= result.confidence_score <= 1.0
    assert len(result.probabilities) == 3
    assert sum(result.probabilities) == pytest.approx(1.0, abs=1e-6)
    assert result.latency_ms > 0.0
    assert engine.last_latency_ms == result.latency_ms
    assert next(engine.model.parameters()).device.type == "cpu"


def test_inference_engine_loads_weights_only_checkpoint(tmp_path) -> None:
    source = SignSequenceClassifier(
        num_classes=3, projection_features=16, hidden_size=16
    )
    checkpoint = tmp_path / "weights.pt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    target = SignSequenceClassifier(
        num_classes=3, projection_features=16, hidden_size=16
    )

    engine = InferenceEngine(
        target,
        {0: "hello", 1: "help", 2: "thanks"},
        checkpoint_path=checkpoint,
    )

    for expected, actual in zip(source.parameters(), engine.model.parameters()):
        torch.testing.assert_close(expected, actual)


def test_inference_rejects_non_finite_window() -> None:
    engine = InferenceEngine(SignSequenceClassifier(num_classes=2), ["a", "b"])
    window = np.zeros((1, 45, 138), dtype=np.float32)
    window[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        engine.predict(window)


def test_inference_confirmation_enforces_system_threshold_floors() -> None:
    engine = InferenceEngine(SignSequenceClassifier(num_classes=3), ["a", "b", "c"])
    result = InferenceResult(
        predicted_gloss="a",
        class_index=0,
        confidence_score=0.64,
        probabilities=(0.64, 0.20, 0.16),
        latency_ms=1.0,
    )

    decision = engine.gate_prediction(
        result, confidence_threshold=0.10, margin_threshold=0.0
    )

    assert not decision.accepted
    assert "confidence_below_threshold" in decision.rejection_reason
