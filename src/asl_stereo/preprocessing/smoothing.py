"""Savitzky-Golay trajectory smoothing."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter


def smooth_trajectories(
    sequence: npt.ArrayLike,
    *,
    window_length: int = 7,
    polyorder: int = 2,
) -> npt.NDArray[np.float32]:
    """Smooth each complete temporal channel and skip unsafe channels.

    Computation is performed in float64. A channel is returned unchanged when
    it contains NaNs or is shorter than ``window_length``.
    """
    values = np.asarray(sequence)
    if values.ndim < 1:
        raise ValueError("sequence must have at least one dimension")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("sequence must have a floating-point dtype")
    if window_length <= 0 or window_length % 2 == 0:
        raise ValueError("window_length must be a positive odd integer")
    if polyorder < 0 or polyorder >= window_length:
        raise ValueError("polyorder must satisfy 0 <= polyorder < window_length")
    if np.isinf(values).any():
        raise ValueError("sequence must not contain infinite values")
    if values.shape[0] == 0:
        return np.ascontiguousarray(values, dtype=np.float32)

    original_shape = values.shape
    result = values.astype(np.float64, copy=True).reshape(values.shape[0], -1)
    if result.shape[0] < window_length:
        return np.ascontiguousarray(result.reshape(original_shape), dtype=np.float32)

    for channel_index in range(result.shape[1]):
        channel = result[:, channel_index]
        if np.isnan(channel).any():
            continue
        result[:, channel_index] = savgol_filter(
            channel,
            window_length=window_length,
            polyorder=polyorder,
            axis=0,
            mode="interp",
        )

    return np.ascontiguousarray(result.reshape(original_shape), dtype=np.float32)


__all__ = ["smooth_trajectories"]
