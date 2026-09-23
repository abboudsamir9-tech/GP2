"""Small contract boundary joining the pure preprocessing operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from asl_stereo.contracts import FEATURE_COUNT, FeatureVector, JOINT_COUNT

from .feature_encoder import encode_feature_vector
from .interpolation import interpolate_short_nan_gaps
from .normalization import normalize_landmarks
from .smoothing import smooth_trajectories


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    max_interpolation_gap: int = 4
    savgol_window_length: int = 7
    savgol_polyorder: int = 2
    normalization_epsilon: float = 1e-6


class PreprocessingPipeline:
    """Apply the shared live/offline preprocessing contract.

    Single frames use :meth:`process_frame` for validity checks. Inference uses
    :meth:`process_live_window` so interpolation and smoothing see the same
    temporal context as dataset preparation. Both paths share normalization
    and feature encoding.
    """

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    def process_frame(
        self, landmarks: npt.ArrayLike, *, timestamp_ns: int
    ) -> FeatureVector | None:
        normalized = normalize_landmarks(
            landmarks, epsilon=self.config.normalization_epsilon
        )
        encoded = encode_feature_vector(normalized)
        if not np.isfinite(encoded).all():
            return None
        return FeatureVector(values=encoded, timestamp_ns=timestamp_ns)

    def process_sequence(
        self, sequence: npt.ArrayLike
    ) -> npt.NDArray[np.float32]:
        """Apply the canonical offline/live temporal cleaning sequence."""
        values = np.asarray(sequence)
        if values.ndim != 3 or values.shape[1:] != (JOINT_COUNT, 3):
            raise ValueError("sequence must have shape (frames, 46, 3)")

        interpolated = interpolate_short_nan_gaps(
            values, max_gap=self.config.max_interpolation_gap
        )
        smoothed = smooth_trajectories(
            interpolated,
            window_length=self.config.savgol_window_length,
            polyorder=self.config.savgol_polyorder,
        )
        output = np.empty((smoothed.shape[0], FEATURE_COUNT), dtype=np.float32)
        for index, frame in enumerate(smoothed):
            output[index] = encode_feature_vector(
                normalize_landmarks(
                    frame, epsilon=self.config.normalization_epsilon
                )
            )
        return np.ascontiguousarray(output)

    def process_live_window(
        self, sequence: npt.ArrayLike
    ) -> npt.NDArray[np.float32]:
        """Clean a live window, including the offline neutral-hand convention."""
        raw = np.asarray(sequence)
        cleaned = self.process_sequence(raw)
        for joints, features in (
            (slice(0, 21), slice(0, 63)),
            (slice(21, 42), slice(63, 126)),
        ):
            if np.isnan(raw[:, joints, :]).all():
                cleaned[:, features] = np.float32(0.0)
        return cleaned


__all__ = ["PreprocessingConfig", "PreprocessingPipeline"]
