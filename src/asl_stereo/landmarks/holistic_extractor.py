"""CPU-conscious MediaPipe Holistic extraction into the 46-joint contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from asl_stereo.contracts import JOINT_COUNT, LandmarkFrame

from .landmark_mapping import (
    HAND_LANDMARK_COUNT,
    LEFT_HAND_OFFSET,
    POSE_SOURCE_TO_OUTPUT,
    RIGHT_HAND_OFFSET,
)


@dataclass(frozen=True, slots=True)
class OverlayPoint:
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class RestrictedHolisticResults:
    """Only data permitted to survive a Holistic inference call."""

    left_hand_landmarks: tuple[OverlayPoint, ...] | None
    right_hand_landmarks: tuple[OverlayPoint, ...] | None
    upper_body_landmarks: tuple[tuple[int, OverlayPoint], ...]


_ExtractorT = TypeVar("_ExtractorT", bound="HolisticExtractor")


class HolisticExtractor:
    """Run one Holistic pass and retain only contract-approved landmarks."""

    def __init__(
        self,
        *,
        holistic_factory: Callable[..., Any] | None = None,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if holistic_factory is None:
            import mediapipe as mp

            holistic_factory = mp.solutions.holistic.Holistic
        self._model = holistic_factory(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._last_results: RestrictedHolisticResults | None = None
        self._closed = False

    @property
    def last_results(self) -> RestrictedHolisticResults | None:
        return self._last_results

    def process(
        self,
        frame: npt.NDArray[np.uint8],
        *,
        timestamp_ns: int = 0,
        frame_index: int = 0,
    ) -> LandmarkFrame:
        if self._closed:
            raise RuntimeError("extractor is closed")
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise TypeError("frame must be a uint8 numpy array")
        if frame.ndim == 2:
            rgb = np.repeat(frame[:, :, None], 3, axis=2)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            rgb = np.ascontiguousarray(frame[:, :, ::-1])
        else:
            raise ValueError("frame must have shape (height, width) or (height, width, 3)")

        raw_results = self._model.process(rgb)
        coordinates = np.full((JOINT_COUNT, 3), np.nan, dtype=np.float32)
        hand_present = np.zeros(2, dtype=np.float32)

        left = self._extract_hand(
            getattr(raw_results, "left_hand_landmarks", None),
            coordinates,
            LEFT_HAND_OFFSET,
        )
        if left is not None:
            hand_present[0] = 1.0

        right = self._extract_hand(
            getattr(raw_results, "right_hand_landmarks", None),
            coordinates,
            RIGHT_HAND_OFFSET,
        )
        if right is not None:
            hand_present[1] = 1.0

        upper_body = self._extract_upper_body(
            getattr(raw_results, "pose_landmarks", None), coordinates
        )
        self._last_results = RestrictedHolisticResults(left, right, upper_body)

        assert coordinates.shape == (JOINT_COUNT, 3)
        joint_mask = np.isfinite(coordinates).all(axis=1).astype(np.float32)
        return LandmarkFrame(
            coordinates=coordinates,
            hand_presence=hand_present,
            joint_mask=joint_mask,
            timestamp_ns=timestamp_ns,
            frame_index=frame_index,
        )

    @staticmethod
    def _extract_hand(
        container: Any,
        output: npt.NDArray[np.float32],
        offset: int,
    ) -> tuple[OverlayPoint, ...] | None:
        if container is None:
            return None
        source = getattr(container, "landmark", None)
        if source is None or len(source) != HAND_LANDMARK_COUNT:
            return None

        retained: list[OverlayPoint] = []
        for local_index, landmark in enumerate(source):
            point = _finite_point_or_nan(landmark)
            retained.append(point)
            if np.isfinite((point.x, point.y, point.z)).all():
                output[offset + local_index] = (point.x, point.y, point.z)
        return tuple(retained)

    @staticmethod
    def _extract_upper_body(
        container: Any, output: npt.NDArray[np.float32]
    ) -> tuple[tuple[int, OverlayPoint], ...]:
        if container is None:
            return ()
        source = getattr(container, "landmark", None)
        if source is None:
            return ()

        retained: list[tuple[int, OverlayPoint]] = []
        for source_index, output_index in POSE_SOURCE_TO_OUTPUT.items():
            if source_index >= len(source):
                continue
            point = _finite_point_or_nan(source[source_index])
            retained.append((source_index, point))
            if np.isfinite((point.x, point.y, point.z)).all():
                output[output_index] = (point.x, point.y, point.z)
        return tuple(retained)

    def close(self) -> None:
        if self._closed:
            return
        close = getattr(self._model, "close", None)
        if close is not None:
            close()
        self._closed = True
        self._last_results = None

    def __enter__(self: _ExtractorT) -> _ExtractorT:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _finite_point_or_nan(landmark: Any) -> OverlayPoint:
    values = np.asarray(
        [getattr(landmark, "x", np.nan), getattr(landmark, "y", np.nan), getattr(landmark, "z", np.nan)],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        values[:] = np.nan
    return OverlayPoint(float(values[0]), float(values[1]), float(values[2]))


__all__ = ["HolisticExtractor", "OverlayPoint", "RestrictedHolisticResults"]
