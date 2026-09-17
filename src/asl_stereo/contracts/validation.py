"""Reusable validation helpers for stage-boundary contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

JOINT_COUNT = 46
COORDINATE_COUNT = 3
FEATURE_COUNT = JOINT_COUNT * COORDINATE_COUNT


def require_non_negative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def require_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def require_array_shape(
    name: str, array: npt.NDArray[np.generic], shape: tuple[int, ...]
) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")


def require_float32(name: str, array: npt.NDArray[np.generic]) -> None:
    if array.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32, got {array.dtype}")


def require_c_contiguous(name: str, array: npt.NDArray[np.generic]) -> None:
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")


def readonly_view(array: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    """Return a view whose write flag is disabled without copying its data."""
    view = array.view()
    view.setflags(write=False)
    return view


__all__ = [
    "COORDINATE_COUNT",
    "FEATURE_COUNT",
    "JOINT_COUNT",
    "readonly_view",
    "require_array_shape",
    "require_c_contiguous",
    "require_float32",
    "require_non_negative_int",
    "require_positive_int",
]

