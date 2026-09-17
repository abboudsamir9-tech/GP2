"""Encode a normalized landmark frame into the strict model layout."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from asl_stereo.contracts.validation import FEATURE_COUNT, JOINT_COUNT


def encode_feature_vector(landmarks: npt.ArrayLike) -> npt.NDArray[np.float32]:
    values = np.asarray(landmarks)
    if values.shape != (JOINT_COUNT, 3):
        raise ValueError(f"landmarks must have shape ({JOINT_COUNT}, 3)")
    encoded = np.ascontiguousarray(values, dtype=np.float32).reshape(FEATURE_COUNT)
    return encoded


__all__ = ["encode_feature_vector"]

