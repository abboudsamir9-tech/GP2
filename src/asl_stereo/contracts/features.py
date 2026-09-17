"""Preprocessed feature and temporal-window contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .validation import (
    FEATURE_COUNT,
    readonly_view,
    require_array_shape,
    require_c_contiguous,
    require_float32,
    require_positive_int,
)


@dataclass(frozen=True, slots=True)
class FeatureVector:
    values: npt.NDArray[np.float32]
    timestamp_ns: int

    def __post_init__(self) -> None:
        require_array_shape("values", self.values, (FEATURE_COUNT,))
        require_float32("values", self.values)
        require_c_contiguous("values", self.values)
        if not np.isfinite(self.values).all():
            raise ValueError("feature values must all be finite")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        object.__setattr__(self, "values", readonly_view(self.values))


@dataclass(frozen=True, slots=True)
class TemporalWindow:
    values: npt.NDArray[np.float32]
    start_timestamp_ns: int
    end_timestamp_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray) or self.values.ndim != 3:
            raise ValueError("values must be a 3D numpy array")
        if self.values.shape[0] != 1 or self.values.shape[2] != FEATURE_COUNT:
            raise ValueError("values must have shape (1, window_size, 138)")
        require_positive_int("window_size", self.values.shape[1])
        require_float32("values", self.values)
        require_c_contiguous("values", self.values)
        if not np.isfinite(self.values).all():
            raise ValueError("window values must all be finite")
        if self.start_timestamp_ns < 0 or self.end_timestamp_ns < 0:
            raise ValueError("timestamps must be non-negative")
        if self.start_timestamp_ns > self.end_timestamp_ns:
            raise ValueError("start timestamp must not exceed end timestamp")
        object.__setattr__(self, "values", readonly_view(self.values))


__all__ = ["FeatureVector", "TemporalWindow"]

