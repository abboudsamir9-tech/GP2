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
    frame_count = result.shape[0]

    for channel_index in range(result.shape[1]):
        channel = result[:, channel_index]
        cursor = 0
        while cursor < frame_count:
            if not np.isnan(channel[cursor]):
                cursor += 1
                continue

            gap_start = cursor
            while cursor < frame_count and np.isnan(channel[cursor]):
                cursor += 1
            gap_end = cursor
            gap_length = gap_end - gap_start

            is_interior = gap_start > 0 and gap_end < frame_count
            if is_interior and gap_length <= max_gap:
                left = channel[gap_start - 1]
                right = channel[gap_end]
                channel[gap_start:gap_end] = np.linspace(
                    left, right, gap_length + 2, dtype=np.float64
                )[1:-1]

    return np.ascontiguousarray(result.reshape(original_shape), dtype=np.float32)


__all__ = ["interpolate_short_nan_gaps"]
