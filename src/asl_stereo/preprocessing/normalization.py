"""Shoulder-centred spatial normalization."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from asl_stereo.contracts.landmarks import (
    LEFT_SHOULDER_INDEX,
    RIGHT_SHOULDER_INDEX,
)
from asl_stereo.contracts.validation import JOINT_COUNT


def normalize_landmarks(
    landmarks: npt.ArrayLike, *, epsilon: float = 1e-6
) -> npt.NDArray[np.float32]:
    """Translate to the mid-shoulder and scale by shoulder distance.

    If either shoulder is unavailable or the scale is below ``epsilon``, a
    fully-NaN frame is returned. Existing NaNs elsewhere propagate naturally.
    """
    values = np.asarray(landmarks)
    if values.shape != (JOINT_COUNT, 3):
        raise ValueError(f"landmarks must have shape ({JOINT_COUNT}, 3)")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("landmarks must have a floating-point dtype")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite value")
    if np.isinf(values).any():
        raise ValueError("landmarks must not contain infinite values")

    work = values.astype(np.float64, copy=True)
    left = work[LEFT_SHOULDER_INDEX]
    right = work[RIGHT_SHOULDER_INDEX]
    invalid = np.full((JOINT_COUNT, 3), np.nan, dtype=np.float32)

    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return invalid

    scale = float(np.linalg.norm(left - right))
    if not np.isfinite(scale) or scale < epsilon:
        return invalid

    midpoint = (left + right) / 2.0
    normalized = (work - midpoint) / scale
    return np.ascontiguousarray(normalized, dtype=np.float32)


__all__ = ["normalize_landmarks"]

