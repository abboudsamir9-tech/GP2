"""Strict 46-joint landmark contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .validation import JOINT_COUNT, readonly_view, require_array_shape

LEFT_HAND_SLICE = slice(0, 21)
RIGHT_HAND_SLICE = slice(21, 42)
LEFT_SHOULDER_INDEX = 42
RIGHT_SHOULDER_INDEX = 43
LEFT_ELBOW_INDEX = 44
RIGHT_ELBOW_INDEX = 45


@dataclass(frozen=True, slots=True)
class LandmarkFrame:
    coordinates: npt.NDArray[np.float32]
    hand_presence: npt.NDArray[np.float32]
    joint_mask: npt.NDArray[np.float32]
    timestamp_ns: int
    frame_index: int

    def __post_init__(self) -> None:
        require_array_shape("coordinates", self.coordinates, (JOINT_COUNT, 3))
        require_array_shape("hand_presence", self.hand_presence, (2,))
        require_array_shape("joint_mask", self.joint_mask, (JOINT_COUNT,))
        if self.coordinates.dtype != np.float32:
            raise TypeError("coordinates must have dtype float32")
        if self.hand_presence.dtype != np.float32:
            raise TypeError("hand_presence must have dtype float32")
        if self.joint_mask.dtype != np.float32:
            raise TypeError("joint_mask must have dtype float32")
        if np.isinf(self.coordinates).any():
            raise ValueError("coordinates may contain NaN but not infinity")
        if not np.isin(self.hand_presence, (0.0, 1.0)).all():
            raise ValueError("hand_presence values must be binary")
        if not np.isin(self.joint_mask, (0.0, 1.0)).all():
            raise ValueError("joint_mask values must be binary")
        observed = np.isfinite(self.coordinates).all(axis=1).astype(np.float32)
        if not np.array_equal(self.joint_mask, observed):
            raise ValueError("joint_mask must exactly describe finite coordinates")
        if self.timestamp_ns < 0 or self.frame_index < 0:
            raise ValueError("timestamp_ns and frame_index must be non-negative")
        object.__setattr__(self, "coordinates", readonly_view(self.coordinates))
        object.__setattr__(self, "hand_presence", readonly_view(self.hand_presence))
        object.__setattr__(self, "joint_mask", readonly_view(self.joint_mask))

    @property
    def hand_present(self) -> npt.NDArray[np.float32]:
        """Specification-facing alias for the binary hand presence mask."""
        return self.hand_presence

    @classmethod
    def missing(cls, *, timestamp_ns: int, frame_index: int) -> "LandmarkFrame":
        return cls(
            coordinates=np.full((JOINT_COUNT, 3), np.nan, dtype=np.float32),
            hand_presence=np.zeros(2, dtype=np.float32),
            joint_mask=np.zeros(JOINT_COUNT, dtype=np.float32),
            timestamp_ns=timestamp_ns,
            frame_index=frame_index,
        )


__all__ = [
    "LEFT_ELBOW_INDEX",
    "LEFT_HAND_SLICE",
    "LEFT_SHOULDER_INDEX",
    "LandmarkFrame",
    "RIGHT_ELBOW_INDEX",
    "RIGHT_HAND_SLICE",
    "RIGHT_SHOULDER_INDEX",
]
