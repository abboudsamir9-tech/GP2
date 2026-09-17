"""Deterministic 46-joint fixtures with known shoulder geometry."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from asl_stereo.contracts import JOINT_COUNT
from asl_stereo.contracts.landmarks import (
    LEFT_SHOULDER_INDEX,
    RIGHT_SHOULDER_INDEX,
)


def make_landmark_frame() -> npt.NDArray[np.float32]:
    coordinates = np.arange(JOINT_COUNT * 3, dtype=np.float32).reshape(JOINT_COUNT, 3)
    coordinates /= 20.0
    coordinates[LEFT_SHOULDER_INDEX] = (-1.0, 0.0, 2.0)
    coordinates[RIGHT_SHOULDER_INDEX] = (1.0, 0.0, 2.0)
    return coordinates


def make_normalized_motion(frame_count: int) -> npt.NDArray[np.float32]:
    base = make_landmark_frame()
    motion = np.repeat(base[np.newaxis, :, :], frame_count, axis=0)
    motion[:, :42, 1] += np.linspace(0.0, 1.0, frame_count, dtype=np.float32)[:, None]
    return motion


__all__ = ["make_landmark_frame", "make_normalized_motion"]

