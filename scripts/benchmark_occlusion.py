"""Compare front-only tracking with calibrated dual-camera stereo fusion."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from asl_stereo.evaluation import (  # noqa: E402
    TrackingMetrics,
    joint_recovery_rate,
    reduction_percent,
    tracking_metrics,
)
from asl_stereo.landmarks import HolisticExtractor  # noqa: E402
from asl_stereo.models import (  # noqa: E402
    InferenceEngine,
    SignSequenceClassifier,
    load_class_map,
)
from asl_stereo.preprocessing import PreprocessingPipeline  # noqa: E402
from asl_stereo.stereo import (  # noqa: E402
    JointStatus,
    StereoCalibration,
    StereoMatcher,
)
from asl_stereo.translation import ConfidenceFilter  # noqa: E402

WINDOW_SIZE = 45
STRIDE = 8


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    top1_accuracy: float
    mean_confidence: float
    false_rejection_rate: float
    evaluated_windows: int
    valid_windows: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark hand-crossing occlusion recovery with stereo fusion."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--sequence-npz",
        type=Path,
        help="Synthetic/pre-extracted NPZ containing front and side arrays.",
    )
    source.add_argument("--front-video", type=Path, help="Front-camera test take.")
    parser.add_argument("--side-video", type=Path, help="Synchronized side-camera take.")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_ROOT / "configs" / "calibration_params.json",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "weights" / "optimized_model.pt",
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        default=PROJECT_ROOT / "configs" / "class_map.json",
    )
    parser.add_argument(
        "--gloss",
        help="Ground-truth gloss for the take; may instead be embedded in NPZ.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "occlusion_benchmark.json",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=PROJECT_ROOT / "reports"
    )
    return parser


def fuse_landmark_sequences(
    front: np.ndarray,
    side: np.ndarray,
    matcher: StereoMatcher,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse aligned observations and retain every per-joint decision status."""
    front_values = _validate_sequence("front", front)
    side_values = _validate_sequence("side", side)
    if front_values.shape != side_values.shape:
        raise ValueError("front and side sequences must have identical shapes")
    fused = np.empty_like(front_values, dtype=np.float32)
    statuses = np.empty(front_values.shape[:2], dtype="<U16")
    for frame_index, (front_frame, side_frame) in enumerate(
        zip(front_values, side_values, strict=True)
    ):
        result = matcher.match(
            front_frame,
            side_frame,
            front_fallback=front_frame,
        )
        fused[frame_index] = result.coordinates
        statuses[frame_index] = [status.value for status in result.statuses]
    return fused, statuses


def classification_metrics(
    sequence: np.ndarray,
    engine: InferenceEngine,
    ground_truth_gloss: str,
) -> ClassificationMetrics:
    """Evaluate every stride-aligned candidate window, including invalid gaps."""
    raw = _validate_sequence("sequence", sequence)
    if not ground_truth_gloss:
        raise ValueError("ground_truth_gloss must be non-empty")
    features = PreprocessingPipeline().process_sequence(raw)
    labels = [engine.class_map[index] for index in range(len(engine.class_map))]
    confidence_filter = ConfidenceFilter()
    starts = range(0, max(0, features.shape[0] - WINDOW_SIZE + 1), STRIDE)
    total = 0
    valid = 0
    correct = 0
    rejected = 0
    confidences: list[float] = []
    for start in starts:
        total += 1
        window = features[start : start + WINDOW_SIZE]
        if window.shape != (WINDOW_SIZE, 138) or not np.isfinite(window).all():
            rejected += 1
            continue
        tensor = torch.from_numpy(np.ascontiguousarray(window[np.newaxis]))
        prediction = engine.predict(tensor)
        valid += 1
        correct += int(prediction.predicted_gloss == ground_truth_gloss)
        confidences.append(prediction.confidence_score)
        filtered = confidence_filter.apply(
            prediction.probabilities,
            labels,
            inference_duration_ms=prediction.latency_ms,
        )
        rejected += int(not filtered.accepted)
    return ClassificationMetrics(
        top1_accuracy=correct / total if total else 0.0,
        mean_confidence=float(np.mean(confidences)) if confidences else 0.0,
        false_rejection_rate=rejected / total if total else 0.0,
        evaluated_windows=total,
        valid_windows=valid,
    )


def extract_video_tracks(
    front_path: Path,
    side_path: Path,
    matcher: StereoMatcher,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run independent baseline and proposed extractors on aligned video frames."""
    import cv2

    front_capture = cv2.VideoCapture(str(front_path))
    side_capture = cv2.VideoCapture(str(side_path))
    if not front_capture.isOpened() or not side_capture.isOpened():
        front_capture.release()
        side_capture.release()
        raise RuntimeError("unable to open both test videos")

    baseline_frames: list[np.ndarray] = []
    fused_frames: list[np.ndarray] = []
    status_frames: list[list[str]] = []
    try:
        with HolisticExtractor() as baseline_extractor, HolisticExtractor() as front_extractor, HolisticExtractor() as side_extractor:
            frame_index = 0
            while True:
                front_ok, front_frame = front_capture.read()
                side_ok, side_frame = side_capture.read()
                if not front_ok or not side_ok:
                    break
                baseline = baseline_extractor.process(front_frame, frame_index=frame_index)
                proposed_front = front_extractor.process(front_frame, frame_index=frame_index)
                side = side_extractor.process(side_frame, frame_index=frame_index)
                front_pixels = _pixel_observations(
                    proposed_front.coordinates, front_frame.shape[1], front_frame.shape[0]
                )
                side_pixels = _pixel_observations(
                    side.coordinates, side_frame.shape[1], side_frame.shape[0]
                )
                match = matcher.match(
                    front_pixels,
                    side_pixels,
                    front_fallback=proposed_front.coordinates,
                )
                baseline_frames.append(np.array(baseline.coordinates, copy=True))
                fused_frames.append(np.array(match.coordinates, copy=True))
                status_frames.append([status.value for status in match.statuses])
                frame_index += 1
    finally:
        front_capture.release()
        side_capture.release()
    if not baseline_frames:
        raise ValueError("test videos contain no aligned frames")
    return (
        np.stack(baseline_frames).astype(np.float32),
        np.stack(fused_frames).astype(np.float32),
        np.asarray(status_frames, dtype="<U16"),
    )


def load_npz_tracks(
    path: Path, matcher: StereoMatcher
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    with np.load(path, allow_pickle=False) as artifact:
        if "front" not in artifact or "side" not in artifact:
            raise ValueError("NPZ must contain front and side arrays")
        front = np.asarray(artifact["front"], dtype=np.float32)
        side = np.asarray(artifact["side"], dtype=np.float32)
        embedded_gloss = _decode_gloss(artifact["gloss"]) if "gloss" in artifact else None
    fused, statuses = fuse_landmark_sequences(front, side, matcher)
    return front, fused, statuses, embedded_gloss


def create_figures(
    baseline: np.ndarray,
    proposed: np.ndarray,
    recovery_rate: float,
    output_dir: Path,
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "joint_recovery_comparison.png"
    trajectory_path = output_dir / "trajectory_occlusion_plot.png"
    baseline_available = np.isfinite(baseline).all(axis=2).mean() * 100.0
    proposed_available = np.isfinite(proposed).all(axis=2).mean() * 100.0

    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(
        ["Single camera", "Stereo fusion"],
        [baseline_available, proposed_available],
        color=["#d95f5f", "#4caf70"],
    )
    axis.bar_label(bars, fmt="%.1f%%")
    axis.set_ylim(0, 105)
    axis.set_ylabel("Available frame-joint observations (%)")
    axis.set_title(f"Occlusion Tracking Availability (recovery {recovery_rate * 100:.1f}%)")
    figure.tight_layout()
    figure.savefig(comparison_path, dpi=160)
    plt.close(figure)

    missing_by_joint = (~np.isfinite(baseline).all(axis=2)).sum(axis=0)
    joint_index = int(np.argmax(missing_by_joint))
    frame_axis = np.arange(baseline.shape[0])
    figure, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for coordinate, name in enumerate(("X", "Y", "Z")):
        axes[coordinate].plot(
            frame_axis, baseline[:, joint_index, coordinate], label="Single camera", color="#d95f5f"
        )
        axes[coordinate].plot(
            frame_axis, proposed[:, joint_index, coordinate], label="Stereo fusion", color="#2878b5"
        )
        axes[coordinate].set_ylabel(name)
        axes[coordinate].grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("Frame")
    figure.suptitle(f"Joint {joint_index} trajectory through occlusion")
    figure.tight_layout()
    figure.savefig(trajectory_path, dpi=160)
    plt.close(figure)
    return comparison_path, trajectory_path


def format_summary_table(
    baseline: TrackingMetrics,
    proposed: TrackingMetrics,
    baseline_classification: ClassificationMetrics,
    proposed_classification: ClassificationMetrics,
    recovery_rate: float,
) -> str:
    rows = [
        ("Available joints (%)", baseline.available_joint_rate * 100, proposed.available_joint_rate * 100),
        ("Missing joints (%)", baseline.missing_joint_rate * 100, proposed.missing_joint_rate * 100),
        ("Mean trajectory jitter", baseline.mean_jitter, proposed.mean_jitter),
        ("Loss episodes (>=5)", baseline.tracking_loss_episodes, proposed.tracking_loss_episodes),
        ("Top-1 accuracy (%)", baseline_classification.top1_accuracy * 100, proposed_classification.top1_accuracy * 100),
        ("Mean confidence (%)", baseline_classification.mean_confidence * 100, proposed_classification.mean_confidence * 100),
        ("False rejection (%)", baseline_classification.false_rejection_rate * 100, proposed_classification.false_rejection_rate * 100),
    ]
    header = ("Metric", "Single camera", "Stereo fusion")
    formatted = [(name, f"{left:.3f}", f"{right:.3f}") for name, left, right in rows]
    widths = [max(len(header[i]), *(len(row[i]) for row in formatted)) for i in range(3)]
    divider = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    line = lambda row: "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |"
    return "\n".join(
        [
            f"Joint recovery rate: {recovery_rate * 100:.3f}%",
            divider,
            line(header),
            divider,
            *(line(row) for row in formatted),
            divider,
        ]
    )


def _build_engine(model_path: Path, class_map_path: Path) -> InferenceEngine:
    class_map = load_class_map(class_map_path)
    placeholder = SignSequenceClassifier(num_classes=len(class_map))
    return InferenceEngine(placeholder, class_map, checkpoint_path=model_path)


def _pixel_observations(coordinates: np.ndarray, width: int, height: int) -> np.ndarray:
    output = np.array(coordinates, dtype=np.float32, copy=True)
    output[:, 0] *= width
    output[:, 1] *= height
    return output


def _validate_sequence(name: str, values: Any) -> np.ndarray:
    output = np.asarray(values)
    if output.ndim != 3 or output.shape[1:] != (46, 3):
        raise ValueError(f"{name} must have shape (frames, 46, 3)")
    if output.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one frame")
    if not np.issubdtype(output.dtype, np.floating):
        raise TypeError(f"{name} must use a floating-point dtype")
    if np.isinf(output).any():
        raise ValueError(f"{name} may contain NaN but not infinity")
    return np.ascontiguousarray(output, dtype=np.float32)


def _decode_gloss(value: np.ndarray) -> str:
    scalar = np.asarray(value).reshape(-1)
    if scalar.size != 1:
        raise ValueError("embedded gloss must be a scalar")
    item = scalar[0]
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def main() -> int:
    args = build_parser().parse_args()
    if args.front_video is not None and args.side_video is None:
        raise ValueError("--side-video is required with --front-video")
    calibration = StereoCalibration.from_file(args.calibration)
    matcher = StereoMatcher(calibration, fallback_scope="frame")
    embedded_gloss = None
    if args.sequence_npz is not None:
        baseline, proposed, statuses, embedded_gloss = load_npz_tracks(
            args.sequence_npz, matcher
        )
    else:
        baseline, proposed, statuses = extract_video_tracks(
            args.front_video, args.side_video, matcher
        )
    gloss = args.gloss or embedded_gloss
    if not gloss:
        raise ValueError("provide --gloss or embed a scalar gloss in the NPZ")

    engine = _build_engine(args.model, args.class_map)
    baseline_tracking = tracking_metrics(baseline)
    proposed_tracking = tracking_metrics(proposed)
    recovery = joint_recovery_rate(baseline, proposed, statuses)
    baseline_classification = classification_metrics(baseline, engine, gloss)
    proposed_classification = classification_metrics(proposed, engine, gloss)
    comparison_figure, trajectory_figure = create_figures(
        baseline, proposed, recovery, args.figures_dir
    )
    report = {
        "schema_version": 1,
        "ground_truth_gloss": gloss,
        "frames": int(baseline.shape[0]),
        "calibration_available": calibration.available,
        "joint_recovery_rate": recovery,
        "single_camera": {
            "tracking": asdict(baseline_tracking),
            "classification": asdict(baseline_classification),
        },
        "stereo_fusion": {
            "tracking": asdict(proposed_tracking),
            "classification": asdict(proposed_classification),
        },
        "improvement": {
            "jitter_reduction_percent": reduction_percent(
                baseline_tracking.mean_jitter, proposed_tracking.mean_jitter
            ),
            "tracking_gap_reduction_percent": reduction_percent(
                baseline_tracking.missing_joint_rate,
                proposed_tracking.missing_joint_rate,
            ),
            "loss_episode_reduction_percent": reduction_percent(
                float(baseline_tracking.tracking_loss_episodes),
                float(proposed_tracking.tracking_loss_episodes),
            ),
        },
        "artifacts": {
            "joint_recovery_comparison": str(comparison_figure),
            "trajectory_occlusion_plot": str(trajectory_figure),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        format_summary_table(
            baseline_tracking,
            proposed_tracking,
            baseline_classification,
            proposed_classification,
            recovery,
        )
    )
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
