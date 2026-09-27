"""Build a compact ASL feature dataset from filename-labelled videos.

Raw video frames are decoded and processed in memory only.  The generated HDF5
artifact contains normalized landmark features, never image or video data.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import numpy.typing as npt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from asl_stereo.contracts import FEATURE_COUNT, JOINT_COUNT
from asl_stereo.landmarks import PoseHandsExtractor
from asl_stereo.models.checkpoint import save_class_map
from asl_stereo.preprocessing import (
    PreprocessingPipeline,
    canonicalize_hand_presence,
    interpolate_short_nan_gaps,
)


WINDOW_SIZE = 45
STRIDE = 8
LEFT_HAND_JOINT_SLICE = slice(0, 21)
RIGHT_HAND_JOINT_SLICE = slice(21, 42)
POSE_JOINT_SLICE = slice(42, 46)
LEFT_HAND_FEATURE_SLICE = slice(0, 63)
RIGHT_HAND_FEATURE_SLICE = slice(63, 126)
POSE_FEATURE_SLICE = slice(126, 138)


@dataclass(frozen=True, slots=True)
class VideoRecord:
    video_id: str
    gloss: str
    path: Path


def parse_video_filename(video_path: str | Path) -> VideoRecord:
    """Parse ``{video_id}-{GLOSS_LABEL}.mp4`` at the first hyphen."""
    path = Path(video_path)
    video_id, separator, gloss = path.stem.partition("-")
    video_id = video_id.strip()
    gloss = gloss.strip()
    if path.suffix.lower() != ".mp4" or not separator or not video_id or not gloss:
        raise ValueError(
            f"invalid video filename {path.name!r}; expected "
            "{video_id}-{GLOSS_LABEL}.mp4"
        )
    return VideoRecord(video_id=video_id, gloss=gloss, path=path)


def discover_videos(videos_dir: str | Path) -> list[VideoRecord]:
    root = Path(videos_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"video directory does not exist: {root}")
    paths = sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"),
        key=lambda path: path.name.casefold(),
    )
    if not paths:
        raise FileNotFoundError(f"no .mp4 videos found in {root}")
    return [parse_video_filename(path) for path in paths]


def extract_landmark_trajectory(video_path: str | Path) -> npt.NDArray[np.float32]:
    """Decode and trim idle edges into the strict (T, 46, 3) contract."""
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"could not open video: {path}")

    frames: list[npt.NDArray[np.float32]] = []
    active_frames: list[bool] = []
    frame_index = 0
    try:
        with PoseHandsExtractor() as extractor:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = extractor.process(frame, frame_index=frame_index)
                coordinates = np.asarray(result.coordinates, dtype=np.float32)
                if coordinates.shape != (JOINT_COUNT, 3):
                    raise RuntimeError(
                        f"extractor returned {coordinates.shape}; expected ({JOINT_COUNT}, 3)"
                    )
                frames.append(coordinates.copy())
                active_frames.append(
                    bool(result.hand_present.any())
                    and bool(np.isfinite(coordinates[POSE_JOINT_SLICE]).all())
                )
                frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"video contains no decodable frames: {path}")
    if not any(active_frames):
        raise ValueError(f"video has no active frames with a hand and upper-body anchors: {path}")
    active_indices = np.flatnonzero(active_frames)
    return np.ascontiguousarray(
        np.stack(frames[active_indices[0] : active_indices[-1] + 1]),
        dtype=np.float32,
    )


def _observed_hand_frames(
    trajectory: npt.NDArray[np.float32], hand_slice: slice
) -> npt.NDArray[np.bool_]:
    hand = trajectory[:, hand_slice, :]
    return np.isfinite(hand).all(axis=(1, 2)) & np.any(hand != 0.0, axis=(1, 2))


def _trim_active_span(
    trajectory: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    active = (
        _observed_hand_frames(trajectory, LEFT_HAND_JOINT_SLICE)
        | _observed_hand_frames(trajectory, RIGHT_HAND_JOINT_SLICE)
    ) & np.isfinite(trajectory[:, POSE_JOINT_SLICE]).all(axis=(1, 2))
    if not active.any():
        return np.empty((0, JOINT_COUNT, 3), dtype=np.float32)
    indices = np.flatnonzero(active)
    return np.ascontiguousarray(trajectory[indices[0] : indices[-1] + 1])


def slice_feature_windows(
    features: npt.ArrayLike,
    *,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> npt.NDArray[np.float32]:
    """Create rolling windows, repeat-padding clips shorter than one window."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != FEATURE_COUNT:
        raise ValueError(f"features must have shape (T, {FEATURE_COUNT})")
    if values.shape[0] == 0:
        raise ValueError("features must contain at least one frame")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    if values.shape[0] < window_size:
        padding = np.repeat(values[-1:], window_size - values.shape[0], axis=0)
        padded = np.concatenate((values, padding), axis=0)
        return np.ascontiguousarray(padded[None, ...], dtype=np.float32)

    starts = range(0, values.shape[0] - window_size + 1, stride)
    return np.ascontiguousarray(
        np.stack([values[start : start + window_size] for start in starts]),
        dtype=np.float32,
    )


def prepare_feature_windows(
    trajectory: npt.ArrayLike,
    *,
    pipeline: PreprocessingPipeline | None = None,
) -> tuple[npt.NDArray[np.float32], int]:
    """Trim idle edges; interpolate short gaps; keep neutral hands exactly zero."""
    values = np.asarray(trajectory, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (JOINT_COUNT, 3):
        raise ValueError(f"trajectory must have shape (T, {JOINT_COUNT}, 3)")
    if values.shape[0] == 0:
        raise ValueError("trajectory must contain at least one frame")

    values = _trim_active_span(values)
    if len(values) == 0:
        return np.empty((0, WINDOW_SIZE, FEATURE_COUNT), dtype=np.float32), 0

    processor = pipeline or PreprocessingPipeline()
    canonical, neutral = canonicalize_hand_presence(values)
    # Interpolate on the full active take first, so a short gap straddling a
    # window edge does not become a false boundary gap in that window.
    interpolated = interpolate_short_nan_gaps(
        canonical, max_gap=processor.config.max_interpolation_gap
    )
    raw_windows = slice_feature_windows(
        interpolated.reshape(interpolated.shape[0], FEATURE_COUNT)
    ).reshape(-1, WINDOW_SIZE, JOINT_COUNT, 3)
    if len(neutral) < WINDOW_SIZE:
        neutral_windows = np.concatenate(
            (neutral, np.repeat(neutral[-1:], WINDOW_SIZE - len(neutral), axis=0))
        )[None, ...]
    else:
        neutral_windows = np.stack(
            [neutral[start : start + WINDOW_SIZE]
             for start in range(0, len(neutral) - WINDOW_SIZE + 1, STRIDE)]
        )

    accepted: list[npt.NDArray[np.float32]] = []
    rejected = 0
    for raw_window, neutral_window in zip(raw_windows, neutral_windows, strict=True):
        candidate = processor.process_sequence(raw_window)
        for hand_index, feature_slice in enumerate(
            (LEFT_HAND_FEATURE_SLICE, RIGHT_HAND_FEATURE_SLICE)
        ):
            candidate[neutral_window[:, hand_index], feature_slice] = np.float32(0.0)

        # Unresolved >=5-frame active-hand or upper-body gaps remain NaN.
        if not np.isfinite(candidate).all():
            rejected += 1
            continue
        accepted.append(np.ascontiguousarray(candidate, dtype=np.float32))

    if not accepted:
        return np.empty((0, WINDOW_SIZE, FEATURE_COUNT), dtype=np.float32), rejected
    return np.ascontiguousarray(np.stack(accepted), dtype=np.float32), rejected


def recover_tracked_segment_windows(
    trajectory: npt.ArrayLike,
    *,
    pipeline: PreprocessingPipeline | None = None,
    minimum_frames: int = 8,
) -> npt.NDArray[np.float32]:
    """Recover a clean active subclip when every full window crosses a long gap.

    This does not impute a >=5-frame loss or accept a window containing one.
    Only the longest contiguous fully tracked segment is considered, and it
    must contain enough real frames before repeat-padding to 45.
    """
    values = np.asarray(trajectory, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (JOINT_COUNT, 3):
        raise ValueError(f"trajectory must have shape (T, {JOINT_COUNT}, 3)")
    if minimum_frames < 1:
        raise ValueError("minimum_frames must be positive")
    values = _trim_active_span(values)
    if len(values) == 0:
        return np.empty((0, WINDOW_SIZE, FEATURE_COUNT), dtype=np.float32)
    processor = pipeline or PreprocessingPipeline()
    canonical, _ = canonicalize_hand_presence(values)
    interpolated = interpolate_short_nan_gaps(
        canonical, max_gap=processor.config.max_interpolation_gap
    )
    finite = np.isfinite(interpolated).all(axis=(1, 2))
    best_start = best_stop = 0
    start = 0
    for stop in range(len(finite) + 1):
        if stop < len(finite) and finite[stop]:
            continue
        if stop - start > best_stop - best_start:
            best_start, best_stop = start, stop
        start = stop + 1
    if best_stop - best_start < minimum_frames:
        return np.empty((0, WINDOW_SIZE, FEATURE_COUNT), dtype=np.float32)
    segment = interpolated[best_start:best_stop]
    recovered, _ = prepare_feature_windows(segment, pipeline=processor)
    return recovered


def write_dataset(
    output_path: str | Path,
    features: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    class_map: dict[int, str],
    source_video_count: int,
) -> Path:
    feature_array = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    if feature_array.ndim != 3 or feature_array.shape[1:] != (
        WINDOW_SIZE,
        FEATURE_COUNT,
    ):
        raise ValueError(f"features must have shape (N, {WINDOW_SIZE}, {FEATURE_COUNT})")
    if label_array.shape != (feature_array.shape[0],):
        raise ValueError("labels must have shape (N,) matching features")
    if not np.isfinite(feature_array).all():
        raise ValueError("features must contain only finite values")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as h5_file:
        h5_file.create_dataset(
            "features",
            data=feature_array,
            dtype=np.float32,
            compression="gzip",
            shuffle=True,
        )
        h5_file.create_dataset(
            "labels",
            data=label_array,
            dtype=np.int64,
            compression="gzip",
        )
        h5_file.attrs["window_size"] = WINDOW_SIZE
        h5_file.attrs["feature_dim"] = FEATURE_COUNT
        h5_file.attrs["stride"] = STRIDE
        h5_file.attrs["source_video_count"] = int(source_video_count)
        h5_file.attrs["class_map"] = json.dumps(
            {str(index): gloss for index, gloss in class_map.items()}, sort_keys=True
        )
    return output


def preprocess_dataset(
    videos_dir: str | Path,
    output_h5: str | Path,
    class_map_out: str | Path,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64], dict[int, str]]:
    records = discover_videos(videos_dir)
    glosses = sorted({record.gloss for record in records})
    class_map = {index: gloss for index, gloss in enumerate(glosses)}
    gloss_to_id = {gloss: index for index, gloss in class_map.items()}

    feature_batches: list[npt.NDArray[np.float32]] = []
    label_batches: list[npt.NDArray[np.int64]] = []
    skipped: list[str] = []
    pipeline = PreprocessingPipeline()

    for position, record in enumerate(records, start=1):
        print(f"[{position:>3}/{len(records)}] {record.path.name}", flush=True)
        try:
            trajectory = extract_landmark_trajectory(record.path)
            windows, rejected = prepare_feature_windows(
                trajectory, pipeline=pipeline
            )
            if rejected:
                print(
                    f"  warning: rejected {rejected} window(s) with unresolved landmark gaps",
                    file=sys.stderr,
                )
            if windows.shape[0] == 0:
                windows = recover_tracked_segment_windows(
                    trajectory, pipeline=pipeline
                )
                if windows.shape[0]:
                    print(
                        "  recovered longest fully tracked active segment",
                        flush=True,
                    )
            if windows.shape[0] == 0:
                skipped.append(record.path.name)
                print("  warning: no finite windows produced", file=sys.stderr)
                continue
            feature_batches.append(windows)
            label_batches.append(
                np.full(windows.shape[0], gloss_to_id[record.gloss], dtype=np.int64)
            )
        except (OSError, RuntimeError, ValueError) as error:
            skipped.append(record.path.name)
            print(f"  warning: skipped ({error})", file=sys.stderr)

    if not feature_batches:
        raise RuntimeError("no valid feature windows were extracted")

    features = np.ascontiguousarray(np.concatenate(feature_batches), dtype=np.float32)
    labels = np.ascontiguousarray(np.concatenate(label_batches), dtype=np.int64)
    write_dataset(
        output_h5,
        features,
        labels,
        class_map=class_map,
        source_video_count=len(records) - len(skipped),
    )
    # The serving contract is integer class ID -> gloss token.
    save_class_map(class_map, class_map_out)
    print_dataset_summary(features, labels, class_map, skipped=skipped)
    return features, labels, class_map


def print_dataset_summary(
    features: npt.NDArray[np.float32],
    labels: npt.NDArray[np.int64],
    class_map: dict[int, str],
    *,
    skipped: list[str],
) -> None:
    counts = Counter(int(label) for label in labels)
    print("\nDataset summary")
    print("+----------+------------------------------+----------+")
    print("| Class ID | Gloss                        | Samples  |")
    print("+----------+------------------------------+----------+")
    for class_id, gloss in class_map.items():
        print(f"| {class_id:>8} | {gloss[:28]:<28} | {counts[class_id]:>8} |")
    print("+----------+------------------------------+----------+")
    print(f"features: {features.shape} {features.dtype}")
    print(f"labels:   {labels.shape} {labels.dtype}")
    print(f"skipped videos: {len(skipped)}")
    for video_name in skipped:
        print(f"  skipped: {video_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract strict 46-joint ASL windows from filename-labelled videos."
    )
    parser.add_argument(
        "--videos-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "videos"
    )
    parser.add_argument(
        "--output-h5",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "asl_dataset.h5",
    )
    parser.add_argument(
        "--class-map-out",
        type=Path,
        default=PROJECT_ROOT / "configs" / "class_map.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    preprocess_dataset(args.videos_dir, args.output_h5, args.class_map_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
