"""Capture and synchronization contracts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from .validation import require_non_negative_int


@dataclass(frozen=True, slots=True)
class TimestampedFrame:
    camera_id: str
    frame_index: int
    timestamp_ns: int
    frame_buffer: npt.NDArray[np.uint8]
    health_meta: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("camera_id must be a non-empty string")
        require_non_negative_int("frame_index", self.frame_index)
        require_non_negative_int("timestamp_ns", self.timestamp_ns)
        if not isinstance(self.frame_buffer, np.ndarray):
            raise TypeError("frame_buffer must be a numpy.ndarray")
        if self.frame_buffer.dtype != np.uint8:
            raise TypeError("frame_buffer must have dtype uint8")
        if self.frame_buffer.ndim not in (2, 3):
            raise ValueError("frame_buffer must be a 2D or 3D image array")
        if not isinstance(self.health_meta, Mapping):
            raise TypeError("health_meta must be a mapping")
        object.__setattr__(self, "health_meta", MappingProxyType(dict(self.health_meta)))


@dataclass(frozen=True, slots=True)
class SynchronizedFramePair:
    front: TimestampedFrame
    side: TimestampedFrame
    delta_t_ns: int
    pair_index: int

    def __post_init__(self) -> None:
        require_non_negative_int("pair_index", self.pair_index)
        if isinstance(self.delta_t_ns, bool) or not isinstance(self.delta_t_ns, int):
            raise TypeError("delta_t_ns must be an integer")
        expected = self.side.timestamp_ns - self.front.timestamp_ns
        if self.delta_t_ns != expected:
            raise ValueError(
                "delta_t_ns must equal side.timestamp_ns - front.timestamp_ns"
            )


__all__ = ["SynchronizedFramePair", "TimestampedFrame"]

