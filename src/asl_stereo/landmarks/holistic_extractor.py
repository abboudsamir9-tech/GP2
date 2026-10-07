"""Pose Lite + Hands Lite extraction into the strict 46-joint contract.

The historical module and class name remain as compatibility aliases; this
module never instantiates a Holistic, face, or segmentation graph.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import TracebackType
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from asl_stereo.contracts import JOINT_COUNT, LandmarkFrame
from .landmark_mapping import HAND_LANDMARK_COUNT, LEFT_HAND_OFFSET, POSE_SOURCE_TO_OUTPUT, RIGHT_HAND_OFFSET

LOGGER = logging.getLogger(__name__)


def _load_solution_modules() -> tuple[Any, Any]:
    """Import legacy Solutions modules without relying on the mp.solutions attribute."""
    try:
        import mediapipe.solutions.pose as mp_pose
        import mediapipe.solutions.hands as mp_hands
    except (AttributeError, ImportError):
        try:
            from mediapipe.python.solutions import pose as mp_pose
            from mediapipe.python.solutions import hands as mp_hands
        except ImportError as error:
            raise ImportError(
                "MediaPipe Pose/Hands Solutions API is unavailable. "
                "Install the project's pinned mediapipe==0.10.14."
            ) from error
    return mp_pose, mp_hands


@dataclass(frozen=True, slots=True)
class OverlayPoint:
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class ExtractionTimings:
    resize_ms: float
    pose_ms: float
    hands_ms: float
    total_ms: float


def _timed_tracker_process(tracker: Any, rgb: np.ndarray) -> tuple[Any, float]:
    started = time.perf_counter()
    results = tracker.process(rgb)
    return results, (time.perf_counter() - started) * 1000.0


@dataclass(frozen=True, slots=True)
class RestrictedPoseHandsResults:
    """Only the 46 permitted landmarks retained for overlay rendering."""

    left_hand_landmarks: tuple[OverlayPoint, ...] | None
    right_hand_landmarks: tuple[OverlayPoint, ...] | None
    upper_body_landmarks: tuple[tuple[int, OverlayPoint], ...]


RestrictedHolisticResults = RestrictedPoseHandsResults
_ExtractorT = TypeVar("_ExtractorT", bound="PoseHandsExtractor")


class PoseHandsExtractor:
    """Process one downscaled RGB image with independent pose and hands graphs."""

    def __init__(
        self,
        *,
        pose_factory: Callable[..., Any] | None = None,
        hands_factory: Callable[..., Any] | None = None,
        model_complexity: int = 0,
        processing_resolution: tuple[int, int] = (640, 360),
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        input_is_mirrored: bool = False,
    ) -> None:
        if len(processing_resolution) != 2 or any(value <= 0 for value in processing_resolution):
            raise ValueError("processing_resolution must contain positive width and height")
        if model_complexity not in (0, 1):
            raise ValueError("model_complexity must be 0 or 1")
        self.processing_resolution = processing_resolution
        self.input_is_mirrored = bool(input_is_mirrored)
        if pose_factory is None or hands_factory is None:
            mp_pose, mp_hands = _load_solution_modules()
            if pose_factory is None:
                pose_factory = mp_pose.Pose
            if hands_factory is None:
                hands_factory = mp_hands.Hands
        self.model_complexity = model_complexity
        self.pose_tracker = pose_factory(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        try:
            self.hand_tracker = hands_factory(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except BaseException:
            self.pose_tracker.close()
            raise
        # The independent MediaPipe graphs release the GIL while running.
        # Parallel calls reduce per-frame latency from the sum to roughly the
        # slower tracker, without sharing tracker state across camera views.
        self._tracker_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="asl-track")
        self._last_results: RestrictedPoseHandsResults | None = None
        self._hand_seen = np.zeros(2, dtype=bool)
        self._closed = False
        self._last_timings: ExtractionTimings | None = None

    @property
    def last_results(self) -> RestrictedPoseHandsResults | None:
        return self._last_results

    @property
    def last_timings(self) -> ExtractionTimings | None:
        return self._last_timings

    def process(
        self,
        frame: npt.NDArray[np.uint8],
        *,
        timestamp_ns: int = 0,
        frame_index: int = 0,
    ) -> LandmarkFrame:
        process_started = time.perf_counter()
        if self._closed:
            raise RuntimeError("extractor is closed")
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise TypeError("frame must be a uint8 numpy array")
        if frame.ndim not in (2, 3) or 0 in frame.shape[:2]:
            raise ValueError("frame must be a non-empty image")
        import cv2

        height, width = frame.shape[:2]
        max_width, max_height = self.processing_resolution
        scale = min(1.0, max_width / width, max_height / height)
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_NEAREST,
            )
        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("frame must be grayscale or BGR with three channels")
        rgb.flags.writeable = False
        resize_ms = (time.perf_counter() - process_started) * 1000.0

        pose_future = self._tracker_pool.submit(_timed_tracker_process, self.pose_tracker, rgb)
        hands_future = self._tracker_pool.submit(_timed_tracker_process, self.hand_tracker, rgb)
        pose_results, pose_ms = pose_future.result()
        hands_results, hands_ms = hands_future.result()
        coordinates = np.full((JOINT_COUNT, 3), np.nan, dtype=np.float32)
        hand_present = np.zeros(2, dtype=np.float32)
        upper_body = self._extract_upper_body(getattr(pose_results, "pose_landmarks", None), coordinates)
        assigned = self._assign_hands(hands_results, coordinates)
        left, right = assigned.get("Left"), assigned.get("Right")
        for hand_index, (points, hand_slice) in enumerate(
            ((left, slice(0, 21)), (right, slice(21, 42)))
        ):
            detected = points is not None and bool(
                np.isfinite(coordinates[hand_slice]).all()
            )
            if detected:
                hand_present[hand_index] = 1.0
                self._hand_seen[hand_index] = True
            else:
                # A hand never observed in this take is structural absence.
                # Once seen, an undetected hand remains NaN so short tracking
                # gaps can be interpolated and long gaps can be rejected.
                coordinates[hand_slice] = (
                    np.nan if self._hand_seen[hand_index] else np.float32(0.0)
                )
                if hand_index == 0:
                    left = None
                else:
                    right = None
        self._last_results = RestrictedPoseHandsResults(left, right, upper_body)
        assert coordinates.shape == (JOINT_COUNT, 3)
        joint_mask = np.isfinite(coordinates).all(axis=1).astype(np.float32)
        self._last_timings = ExtractionTimings(
            resize_ms, pose_ms, hands_ms, (time.perf_counter() - process_started) * 1000.0,
        )
        return LandmarkFrame(
            coordinates=coordinates,
            hand_presence=hand_present,
            joint_mask=joint_mask,
            timestamp_ns=timestamp_ns,
            frame_index=frame_index,
        )

    def reset_tracking(self) -> None:
        """Start a new independent clip without carrying hand state across it."""
        if self._closed:
            raise RuntimeError("extractor is closed")
        self._hand_seen[:] = False
        self._last_results = None
        self._last_timings = None

    def _assign_hands(
        self, results: Any, output: npt.NDArray[np.float32]
    ) -> dict[str, tuple[OverlayPoint, ...]]:
        containers = getattr(results, "multi_hand_landmarks", None) or ()
        handedness = getattr(results, "multi_handedness", None) or ()
        candidates: list[tuple[tuple[OverlayPoint, ...], str | None]] = []
        for index, container in enumerate(containers[:2]):
            source = getattr(container, "landmark", None)
            if source is None or len(source) != HAND_LANDMARK_COUNT:
                continue
            points = tuple(_finite_point_or_nan(point) for point in source)
            classifications = getattr(handedness[index], "classification", ()) if index < len(handedness) else ()
            top = classifications[0] if classifications else None
            label = getattr(top, "label", None)
            score = float(getattr(top, "score", 0.0)) if top is not None else 0.0
            # MediaPipe Hands labels assume a selfie-mirrored image. OpenCV feeds
            # here are unmirrored unless input_is_mirrored is explicitly set.
            if label in ("Left", "Right") and not self.input_is_mirrored:
                label = "Right" if label == "Left" else "Left"
            if label not in ("Left", "Right") or not np.isfinite(score) or score < 0.5:
                label = None
            candidates.append((points, label))

        assigned: dict[str, tuple[OverlayPoint, ...]] = {}
        # Trusted handedness wins when the arms cross; pose is the fallback.
        for points, label in candidates:
            if label is not None and label not in assigned:
                assigned[label] = points
        for points, _ in candidates:
            if any(points is selected for selected in assigned.values()):
                continue
            available = [side for side in ("Left", "Right") if side not in assigned]
            if not available:
                break
            side = min(available, key=lambda name: self._hand_distance(points[0], name, output))
            assigned[side] = points
        for side, points in assigned.items():
            offset = LEFT_HAND_OFFSET if side == "Left" else RIGHT_HAND_OFFSET
            for local_index, point in enumerate(points):
                if np.isfinite((point.x, point.y, point.z)).all():
                    output[offset + local_index] = (point.x, point.y, point.z)
        return assigned

    def _hand_distance(
        self, wrist: OverlayPoint, side: str, output: npt.NDArray[np.float32]
    ) -> float:
        shoulder_index, elbow_index = (42, 44) if side == "Left" else (43, 45)
        for index in (elbow_index, shoulder_index):
            reference = output[index, :2]
            if np.isfinite(reference).all() and np.isfinite((wrist.x, wrist.y)).all():
                return float(np.hypot(wrist.x - reference[0], wrist.y - reference[1]))
        if not np.isfinite(wrist.x):
            return float("inf")
        # Anatomical left appears on image right in an unmirrored frame.
        left_x = wrist.x if self.input_is_mirrored else 1.0 - wrist.x
        return left_x if side == "Left" else 1.0 - left_x

    @staticmethod
    def _extract_upper_body(
        container: Any, output: npt.NDArray[np.float32]
    ) -> tuple[tuple[int, OverlayPoint], ...]:
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
        self._closed = True
        try:
            self._tracker_pool.shutdown(wait=True, cancel_futures=True)
        finally:
            try:
                self.pose_tracker.close()
            finally:
                self.hand_tracker.close()
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


HolisticExtractor = PoseHandsExtractor

__all__ = [
    "PoseHandsExtractor", "HolisticExtractor", "OverlayPoint",
    "ExtractionTimings",
    "RestrictedPoseHandsResults", "RestrictedHolisticResults",
]
