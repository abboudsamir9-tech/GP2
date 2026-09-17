import numpy as np

from asl_stereo.translation import ConfidenceFilter, filter_logits

LABELS = ("hello", "thanks", "help")


def test_accepts_prediction_that_meets_both_thresholds() -> None:
    result = filter_logits(
        np.array([4.0, 0.0, -1.0]),
        LABELS,
        inference_duration_ms=12.5,
        window_end_timestamp_ns=100,
    )
    assert result.accepted
    assert result.label == "hello"
    assert result.rejection_reason is None
    assert result.inference_duration_ms == 12.5


def test_rejects_low_confidence_prediction() -> None:
    result = filter_logits(np.array([0.4, 0.2, 0.0]), LABELS)
    assert not result.accepted
    assert "confidence_below_threshold" in result.rejection_reason


def test_rejects_small_top_two_margin() -> None:
    result = filter_logits(
        np.array([5.0, 4.9, -10.0]), LABELS, confidence_threshold=0.50
    )
    assert not result.accepted
    assert "margin_below_threshold" in result.rejection_reason


def test_accepts_single_batch_dimension() -> None:
    assert filter_logits(np.array([[4.0, 0.0, -1.0]]), LABELS).accepted


def test_default_filter_rejects_probability_below_point_65() -> None:
    result = ConfidenceFilter().apply(np.array([0.64, 0.20, 0.16]), LABELS)
    assert not result.accepted
    assert result.rejection_reason == "confidence_below_threshold"


def test_default_filter_rejects_margin_below_point_15() -> None:
    result = ConfidenceFilter().apply(np.array([0.56, 0.43, 0.01]), LABELS)
    assert not result.accepted
    assert "margin_below_threshold" in result.rejection_reason


def test_default_filter_accepts_high_confidence_and_margin() -> None:
    result = ConfidenceFilter().apply(np.array([0.80, 0.12, 0.08]), LABELS)
    assert result.accepted
    assert result.rejection_reason is None
