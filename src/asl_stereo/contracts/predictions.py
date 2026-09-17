"""Classifier output contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Prediction:
    class_index: int
    label: str
    top1_confidence: float
    top2_confidence: float
    margin: float
    accepted: bool
    rejection_reason: str | None
    inference_duration_ms: float
    window_end_timestamp_ns: int

    def __post_init__(self) -> None:
        if self.class_index < 0:
            raise ValueError("class_index must be non-negative")
        if not self.label:
            raise ValueError("label must be non-empty")
        for name, value in (
            ("top1_confidence", self.top1_confidence),
            ("top2_confidence", self.top2_confidence),
            ("margin", self.margin),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        expected_margin = self.top1_confidence - self.top2_confidence
        if abs(self.margin - expected_margin) > 1e-6:
            raise ValueError("margin must equal top1_confidence - top2_confidence")
        if self.accepted and self.rejection_reason is not None:
            raise ValueError("accepted predictions cannot have a rejection reason")
        if not self.accepted and not self.rejection_reason:
            raise ValueError("rejected predictions require a rejection reason")
        if self.inference_duration_ms < 0 or self.window_end_timestamp_ns < 0:
            raise ValueError("duration and timestamp must be non-negative")


__all__ = ["Prediction"]

