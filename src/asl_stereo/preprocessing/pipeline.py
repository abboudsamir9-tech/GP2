"""Small contract boundary joining the pure preprocessing operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from asl_stereo.contracts import FEATURE_COUNT, FeatureVector, JOINT_COUNT

from .feature_encoder import encode_feature_vector
from .interpolation import interpolate_short_nan_gaps
from .normalization import normalize_landmarks, normalize_landmark_sequence
from .smoothing import smooth_trajectories


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    max_interpolation_gap: int = 4
    savgol_window_length: int = 7
    savgol_polyorder: int = 2
    normalization_epsilon: float = 1e-6


def canonicalize_hand_presence(
    sequence: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Distinguish neutral hand absence from gaps inside an active hand span.

    A never-seen hand and frames outside its first/last observation are neutral.
    Zero or NaN frames between those observations are tracking gaps. The
    returned mask identifies neutral rows that must remain exactly zero after
    shoulder-centred normalization.
    """
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (JOINT_COUNT, 3):
        raise ValueError("sequence must have shape (frames, 46, 3)")
    if np.isinf(values).any():
        raise ValueError("sequence must not contain infinity")
    result = np.ascontiguousarray(values.copy())
    neutral = np.zeros((len(result), 2), dtype=bool)
    hand_slices = (slice(0, 21), slice(21, 42))
    detections = [
        np.isfinite(result[:, hand_slice, :]).all(axis=(1, 2))
        & np.any(result[:, hand_slice, :] != 0.0, axis=(1, 2))
        for hand_slice in hand_slices
    ]
    # A few isolated false handedness classifications must not convert an
    # otherwise absent non-dominant hand into a clip-long NaN dropout.
    dominant_count = max(int(detected.sum()) for detected in detections)
    # A second hand needs a substantial track relative to the dominant hand.
    # Short handedness flips (even several consecutive frames) are otherwise
    # misread as a long dropout across the rest of a one-handed sign.
    active_threshold = max(3, int(np.ceil(0.50 * dominant_count)))
    for hand_index, (hand_slice, detected) in enumerate(zip(hand_slices, detections)):
        hand = result[:, hand_slice, :]
        if int(detected.sum()) < active_threshold:
            neutral[:, hand_index] = True
            hand[:] = np.float32(0.0)
            continue
        if detected.any():
            indices = np.flatnonzero(detected)
            neutral[: indices[0], hand_index] = True
            neutral[indices[-1] + 1 :, hand_index] = True
            hand[~detected & ~neutral[:, hand_index]] = np.nan
        else:
            neutral[:, hand_index] = True
        hand[neutral[:, hand_index]] = np.float32(0.0)
    return result, neutral


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
        normalized = normalize_landmark_sequence(
            smoothed, epsilon=self.config.normalization_epsilon,
        )
        return normalized.reshape(smoothed.shape[0], FEATURE_COUNT)

    def process_live_window(
        self, sequence: npt.ArrayLike
    ) -> npt.NDArray[np.float32]:
        """Clean a live window, including the offline neutral-hand convention."""
        raw, neutral = canonicalize_hand_presence(sequence)
        if neutral.all():
            return np.full((len(raw), FEATURE_COUNT), np.nan, dtype=np.float32)
        cleaned = self.process_sequence(raw)
        for hand_index, features in enumerate((slice(0, 63), slice(63, 126))):
            cleaned[neutral[:, hand_index], features] = np.float32(0.0)
        return cleaned


__all__ = ["PreprocessingConfig", "PreprocessingPipeline", "canonicalize_hand_presence"]
