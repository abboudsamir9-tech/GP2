"""Valid-frame temporal buffering with stride-controlled emission."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import numpy.typing as npt

from asl_stereo.contracts.features import FeatureVector, TemporalWindow
from asl_stereo.contracts.validation import FEATURE_COUNT


class TemporalBuffer:
    """Maintain the latest valid frames and emit owned contiguous windows."""

    def __init__(self, window_size: int = 45, stride: int = 8) -> None:
        if isinstance(window_size, bool) or not isinstance(window_size, int):
            raise TypeError("window_size must be an integer")
        if not 30 <= window_size <= 60:
            raise ValueError("window_size must be in the range [30, 60]")
        if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
            raise ValueError("stride must be a positive integer")

        self.window_size = window_size
        self.stride = stride
        self._frames: deque[npt.NDArray[np.float32]] = deque(maxlen=window_size)
        self._timestamps: deque[int] = deque(maxlen=window_size)
        self._valid_since_emission = 0
        self._has_emitted = False

    def append(
        self, values: FeatureVector | npt.ArrayLike, timestamp_ns: int | None = None
    ) -> TemporalWindow | None:
        """Append one frame, returning a window only when cadence is reached.

        Non-finite or malformed arrays are invalid and do not alter buffer or
        stride state. A ``FeatureVector`` supplies its own timestamp.
        """
        if isinstance(values, FeatureVector):
            if timestamp_ns is not None and timestamp_ns != values.timestamp_ns:
                raise ValueError("timestamp_ns conflicts with FeatureVector timestamp")
            timestamp = values.timestamp_ns
            array = values.values
        else:
            if timestamp_ns is None:
                raise ValueError("timestamp_ns is required for array inputs")
            timestamp = timestamp_ns
            array = np.asarray(values)

        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
            raise ValueError("timestamp_ns must be a non-negative integer")
        if array.shape != (FEATURE_COUNT,) or array.dtype != np.float32:
            return None
        if not np.isfinite(array).all():
            return None
        if self._timestamps and timestamp <= self._timestamps[-1]:
            raise ValueError("valid-frame timestamps must be strictly increasing")

        self._frames.append(np.ascontiguousarray(array, dtype=np.float32))
        self._timestamps.append(timestamp)
        self._valid_since_emission += 1

        full = len(self._frames) == self.window_size
        cadence_reached = not self._has_emitted or self._valid_since_emission >= self.stride
        if not full or not cadence_reached:
            return None

        # The emitted array owns its memory, so producer writes cannot race inference.
        matrix = np.stack(tuple(self._frames), axis=0).astype(np.float32, copy=False)
        batch = np.ascontiguousarray(matrix[np.newaxis, :, :])
        window = TemporalWindow(
            values=batch,
            start_timestamp_ns=self._timestamps[0],
            end_timestamp_ns=self._timestamps[-1],
        )
        self._has_emitted = True
        self._valid_since_emission = 0
        return window

    def clear(self) -> None:
        self._frames.clear()
        self._timestamps.clear()
        self._valid_since_emission = 0
        self._has_emitted = False

    def __len__(self) -> int:
        return len(self._frames)


class SlidingWindowBuffer(TemporalBuffer):
    """Model-facing buffer emitting zero-copy ``torch.float32`` tensors.

    The NumPy window produced by :class:`TemporalBuffer` owns a stable snapshot;
    ``torch.from_numpy`` then creates a tensor sharing that snapshot's storage.
    """

    def __init__(self, window_size: int = 45, stride: int = 8) -> None:
        super().__init__(window_size=window_size, stride=stride)
        self.last_window_start_timestamp_ns: int | None = None
        self.last_window_end_timestamp_ns: int | None = None
        self._raw_landmarks: deque[npt.NDArray[np.float32]] = deque(
            maxlen=window_size
        )
        self._raw_timestamps: deque[int] = deque(maxlen=window_size)
        self._input_mode: str | None = None

    def append(self, values, timestamp_ns: int | None = None):
        if self._input_mode not in (None, "features"):
            raise RuntimeError("reset buffer before changing input mode")
        self._input_mode = "features"
        window = super().append(values, timestamp_ns)
        if window is None:
            return None
        import torch

        self.last_window_start_timestamp_ns = window.start_timestamp_ns
        self.last_window_end_timestamp_ns = window.end_timestamp_ns
        # TemporalWindow exposes a read-only view for contract consumers. This
        # window is local and its backing allocation is exclusively owned, so
        # make that view writable before sharing it with PyTorch.
        window.values.setflags(write=True)
        tensor = torch.from_numpy(window.values)
        assert tensor.shape == (1, self.window_size, FEATURE_COUNT)
        assert tensor.dtype == torch.float32
        return tensor

    def append_landmarks(
        self,
        landmarks: npt.ArrayLike,
        *,
        timestamp_ns: int,
        preprocessor: Any,
    ):
        """Clean a rolling raw-landmark window before stride-controlled emission.

        Raw NaN frames are retained so short interior gaps can be interpolated.
        A candidate advances the valid-frame cadence only when the complete
        cleaned window is finite.
        """
        if self._input_mode not in (None, "landmarks"):
            raise RuntimeError("reset buffer before changing input mode")
        self._input_mode = "landmarks"
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
            raise ValueError("timestamp_ns must be a non-negative integer")
        if timestamp_ns < 0:
            raise ValueError("timestamp_ns must be a non-negative integer")
        values = np.asarray(landmarks)
        if values.shape != (46, 3) or not np.issubdtype(values.dtype, np.floating):
            return None
        if np.isinf(values).any():
            return None
        if self._raw_timestamps and timestamp_ns <= self._raw_timestamps[-1]:
            raise ValueError("frame timestamps must be strictly increasing")
        self._raw_landmarks.append(
            np.ascontiguousarray(values, dtype=np.float32).copy()
        )
        self._raw_timestamps.append(timestamp_ns)
        if len(self._raw_landmarks) < self.window_size:
            return None

        raw_window = np.stack(tuple(self._raw_landmarks), axis=0)
        cleaned = preprocessor.process_live_window(raw_window)
        if cleaned.shape != (self.window_size, FEATURE_COUNT):
            raise ValueError("cleaned live window has an invalid shape")
        if cleaned.dtype != np.float32 or not np.isfinite(cleaned).all():
            return None

        self._valid_since_emission += 1
        cadence_reached = (
            not self._has_emitted or self._valid_since_emission >= self.stride
        )
        if not cadence_reached:
            return None

        import torch

        batch = np.ascontiguousarray(cleaned[np.newaxis, :, :])
        self.last_window_start_timestamp_ns = self._raw_timestamps[0]
        self.last_window_end_timestamp_ns = self._raw_timestamps[-1]
        self._has_emitted = True
        self._valid_since_emission = 0
        tensor = torch.from_numpy(batch)
        assert tensor.shape == (1, self.window_size, FEATURE_COUNT)
        assert tensor.dtype == torch.float32
        return tensor

    def reset(self) -> None:
        self.clear()
        self._raw_landmarks.clear()
        self._raw_timestamps.clear()
        self._input_mode = None
        self.last_window_start_timestamp_ns = None
        self.last_window_end_timestamp_ns = None


__all__ = ["SlidingWindowBuffer", "TemporalBuffer"]
