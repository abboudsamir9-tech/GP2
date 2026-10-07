"""NaN-gap interpolation along the temporal axis."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def interpolate_short_nan_gaps(
    sequence: npt.ArrayLike, *, max_gap: int = 4
) -> npt.NDArray[np.float32]:
    """Linearly fill interior NaN runs no longer than ``max_gap``.

    Time is axis 0 and every remaining dimension is treated as an independent
    channel. Infinite values are rejected rather than treated as observations.
    Boundary gaps and runs longer than ``max_gap`` remain NaN.
    """
    if isinstance(max_gap, bool) or not isinstance(max_gap, int) or max_gap < 0:
        raise ValueError("max_gap must be a non-negative integer")

    values = np.asarray(sequence)
    if values.ndim < 1:
        raise ValueError("sequence must have at least one dimension")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("sequence must have a floating-point dtype")
    if np.isinf(values).any():
        raise ValueError("sequence must not contain infinite values")
    if values.shape[0] == 0:
        return np.ascontiguousarray(values, dtype=np.float32)

    original_shape = values.shape
    result = values.astype(np.float64, copy=True).reshape(values.shape[0], -1)
    missing = np.isnan(result)
    if max_gap and missing.any():
        frame_count = len(result)
        timeline = np.arange(frame_count)[:, None]
        left = np.maximum.accumulate(np.where(missing, -1, timeline), axis=0)
        right = np.minimum.accumulate(
            np.where(missing, frame_count, timeline)[::-1], axis=0,
        )[::-1]
        lengths = right - left - 1
        fill = missing & (left >= 0) & (right < frame_count) & (lengths <= max_gap)
        frames, channels = np.nonzero(fill)
        left_indices = left[frames, channels]
        right_indices = right[frames, channels]
        left_values = result[left_indices, channels]
        right_values = result[right_indices, channels]
        # Match np.linspace's float64 step-then-multiply arithmetic.
        step = (right_values - left_values) / (right_indices - left_indices)
        result[frames, channels] = left_values + step * (frames - left_indices)

    return np.ascontiguousarray(result.reshape(original_shape), dtype=np.float32)


__all__ = ["interpolate_short_nan_gaps"]
