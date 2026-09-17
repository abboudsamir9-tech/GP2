"""Sequential video landmark extraction, audit export, and windowing."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, TypeVar

import cv2
import numpy as np
import numpy.typing as npt

from asl_stereo.contracts import FEATURE_COUNT, JOINT_COUNT
from asl_stereo.landmarks import HolisticExtractor
from asl_stereo.preprocessing import PreprocessingPipeline


@dataclass(frozen=True, slots=True)
class ExtractedVideo:
    video_id: str
    frame_count: int
    windows: npt.NDArray[np.float32]
    audit_csv_path: Path
    discarded_reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.discarded_reason is None


_ExtractorT = TypeVar("_ExtractorT", bound="BatchFeatureExtractor")


class BatchFeatureExtractor:
    def __init__(
        self,
        audit_csv_dir: str | Path,
        *,
        window_size: int = 45,
        stride: int = 8,
        holistic_factory: Any = HolisticExtractor,
        preprocessing_pipeline: PreprocessingPipeline | None = None,
    ) -> None:
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive")
        self.audit_csv_dir = Path(audit_csv_dir)
        self.window_size = window_size
        self.stride = stride
        self._holistic_factory = holistic_factory
        self._preprocessor = preprocessing_pipeline or PreprocessingPipeline()
        self._extractor: Any | None = None

    def extract_video(self, video_id: str, video_path: str | Path) -> ExtractedVideo:
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"video not found: {path}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"OpenCV could not open video: {path}")

        raw_frames: list[npt.NDArray[np.float32]] = []
        extractor = self._get_extractor()
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                landmark_frame = extractor.process(
                    frame,
                    timestamp_ns=frame_index,
                    frame_index=frame_index,
                )
                raw_frames.append(np.array(landmark_frame.coordinates, copy=True))
                frame_index += 1
        finally:
            capture.release()

        if raw_frames:
            raw_sequence = np.ascontiguousarray(np.stack(raw_frames), dtype=np.float32)
        else:
            raw_sequence = np.empty((0, JOINT_COUNT, 3), dtype=np.float32)
        audit_path = self.export_audit_csv(video_id, raw_sequence)

        if frame_index == 0:
            return self._discarded(video_id, frame_index, audit_path, "video_has_no_frames")
        processed = self._preprocessor.process_sequence(raw_sequence)
        if not np.isfinite(processed).all():
            return self._discarded(
                video_id,
                frame_index,
                audit_path,
                "unresolved_nan_after_preprocessing",
            )
        windows = self.segment_sequence(
            processed, window_size=self.window_size, stride=self.stride
        )
        if windows.shape[0] == 0:
            return self._discarded(
                video_id, frame_index, audit_path, "sequence_shorter_than_window"
            )
        return ExtractedVideo(video_id, frame_index, windows, audit_path)

    def export_audit_csv(
        self, video_id: str, coordinates: npt.ArrayLike
    ) -> Path:
        safe_video_id = str(video_id)
        if Path(safe_video_id).name != safe_video_id:
            raise ValueError("video_id must not contain path components")
        sequence = np.asarray(coordinates)
        if sequence.ndim != 3 or sequence.shape[1:] != (JOINT_COUNT, 3):
            raise ValueError("coordinates must have shape (frames, 46, 3)")
        if not np.issubdtype(sequence.dtype, np.floating):
            raise TypeError("coordinates must use a floating-point dtype")

        self.audit_csv_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.audit_csv_dir / f"{safe_video_id}.csv"
        header = ["frame_idx"] + [
            f"joint_{joint}_{axis}"
            for joint in range(JOINT_COUNT)
            for axis in ("x", "y", "z")
        ]
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header)
            for frame_index, frame in enumerate(sequence):
                writer.writerow([frame_index, *frame.reshape(FEATURE_COUNT).tolist()])
        return output_path

    @staticmethod
    def segment_sequence(
        features: npt.ArrayLike,
        *,
        window_size: int = 45,
        stride: int = 8,
    ) -> npt.NDArray[np.float32]:
        values = np.asarray(features)
        if values.ndim != 2 or values.shape[1] != FEATURE_COUNT:
            raise ValueError("features must have shape (frames, 138)")
        if values.dtype != np.float32:
            raise TypeError("features must have dtype float32")
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive")
        if not np.isfinite(values).all():
            raise ValueError("features must contain only finite values")
        starts = range(0, max(0, values.shape[0] - window_size + 1), stride)
        windows = [values[start : start + window_size] for start in starts]
        if not windows:
            return np.empty((0, window_size, FEATURE_COUNT), dtype=np.float32)
        return np.ascontiguousarray(np.stack(windows), dtype=np.float32)

    def close(self) -> None:
        if self._extractor is not None:
            self._extractor.close()
            self._extractor = None

    def _get_extractor(self) -> Any:
        if self._extractor is None:
            self._extractor = self._holistic_factory()
        return self._extractor

    def _discarded(
        self, video_id: str, frame_count: int, audit_path: Path, reason: str
    ) -> ExtractedVideo:
        return ExtractedVideo(
            video_id,
            frame_count,
            np.empty((0, self.window_size, FEATURE_COUNT), dtype=np.float32),
            audit_path,
            reason,
        )

    def __enter__(self: _ExtractorT) -> _ExtractorT:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["BatchFeatureExtractor", "ExtractedVideo"]
