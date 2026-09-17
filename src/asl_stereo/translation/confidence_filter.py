"""Pure dual-threshold filtering for classifier logits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from asl_stereo.contracts.predictions import Prediction


def filter_logits(
    logits: npt.ArrayLike,
    labels: Sequence[str],
    *,
    confidence_threshold: float = 0.65,
    margin_threshold: float = 0.15,
    inference_duration_ms: float = 0.0,
    window_end_timestamp_ns: int = 0,
) -> Prediction:
    scores = _as_numpy(logits)
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    if scores.ndim != 1 or scores.size < 2:
        raise ValueError("logits must contain at least two class scores")
    if scores.size != len(labels):
        raise ValueError("labels length must match logits")
    if not np.isfinite(scores).all():
        raise ValueError("logits must all be finite")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if not 0.0 <= margin_threshold <= 1.0:
        raise ValueError("margin_threshold must be in [0, 1]")

    shifted = scores - np.max(scores)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    return filter_probabilities(
        probabilities,
        labels,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        inference_duration_ms=inference_duration_ms,
        window_end_timestamp_ns=window_end_timestamp_ns,
    )


def filter_probabilities(
    probabilities: npt.ArrayLike,
    labels: Sequence[str],
    *,
    confidence_threshold: float = 0.65,
    margin_threshold: float = 0.15,
    inference_duration_ms: float = 0.0,
    window_end_timestamp_ns: int = 0,
) -> Prediction:
    values = _as_numpy(probabilities)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 1 or values.size < 2:
        raise ValueError("probabilities must contain at least two class scores")
    if values.size != len(labels):
        raise ValueError("labels length must match probabilities")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.isclose(values.sum(), 1.0, atol=1e-5):
        raise ValueError("probabilities must sum to one")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if not 0.0 <= margin_threshold <= 1.0:
        raise ValueError("margin_threshold must be in [0, 1]")

    order = np.argsort(values)[::-1]
    top1_index, top2_index = int(order[0]), int(order[1])
    top1 = float(values[top1_index])
    top2 = float(values[top2_index])
    margin = top1 - top2

    reasons: list[str] = []
    if top1 < confidence_threshold:
        reasons.append("confidence_below_threshold")
    if margin < margin_threshold:
        reasons.append("margin_below_threshold")
    accepted = not reasons

    return Prediction(
        class_index=top1_index,
        label=labels[top1_index],
        top1_confidence=top1,
        top2_confidence=top2,
        margin=margin,
        accepted=accepted,
        rejection_reason=None if accepted else ",".join(reasons),
        inference_duration_ms=inference_duration_ms,
        window_end_timestamp_ns=window_end_timestamp_ns,
    )


@dataclass(frozen=True, slots=True)
class ConfidenceFilter:
    confidence_threshold: float = 0.65
    margin_threshold: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not 0.0 <= self.margin_threshold <= 1.0:
            raise ValueError("margin_threshold must be in [0, 1]")

    def apply(
        self,
        probabilities: npt.ArrayLike,
        labels: Sequence[str],
        *,
        inference_duration_ms: float = 0.0,
        window_end_timestamp_ns: int = 0,
    ) -> Prediction:
        return filter_probabilities(
            probabilities,
            labels,
            confidence_threshold=self.confidence_threshold,
            margin_threshold=self.margin_threshold,
            inference_duration_ms=inference_duration_ms,
            window_end_timestamp_ns=window_end_timestamp_ns,
        )


def _as_numpy(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    detach = getattr(values, "detach", None)
    if detach is not None:
        values = detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


__all__ = ["ConfidenceFilter", "filter_logits", "filter_probabilities"]
