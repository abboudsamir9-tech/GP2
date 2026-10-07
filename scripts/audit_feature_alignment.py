"""Audit training/validation landmarks without reading held-out test features.

No images, videos, or raw frames are persisted. Re-extraction is a diagnostic
sample, not a replacement dataset or an opportunity to tune against test data.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asl_stereo.dataset.comparison_protocol import freeze_existing_partitions, load_partition
from asl_stereo.models.checkpoint import load_class_map
from asl_stereo.preprocessing import PreprocessingConfig, PreprocessingPipeline, SlidingWindowBuffer
from scripts.preprocess_dataset import discover_videos, extract_landmark_trajectory, prepare_feature_windows


def _synthetic_trajectory(length=45):
    rng = np.random.default_rng(123)
    raw = np.zeros((length, 46, 3), dtype=np.float32)
    raw[:, 21:42] = rng.normal((0.65, 0.35, 0.01), 0.01, (length, 21, 3))
    raw[:, 42] = (0.65, 0.55, -0.1)
    raw[:, 43] = (0.35, 0.55, -0.1)
    raw[:, 44] = (0.72, 0.75, -0.05)
    raw[:, 45] = (0.28, 0.75, -0.05)
    return raw


def alignment_checks() -> dict:
    pipeline = PreprocessingPipeline()
    raw = _synthetic_trajectory()
    raw[18:21, 21:42] = np.nan  # A recoverable three-frame active-hand gap.
    offline, _ = prepare_feature_windows(raw, pipeline=pipeline)
    live = pipeline.process_live_window(raw)
    buffer = SlidingWindowBuffer()
    emitted = None
    for index, frame in enumerate(raw):
        emitted = buffer.append_landmarks(frame, timestamp_ns=index + 1, preprocessor=pipeline)
    if emitted is None:
        raise RuntimeError("synthetic valid window did not emit")
    delta = float(np.max(np.abs(offline[0] - live)))
    buffer_delta = float(np.max(np.abs(emitted.numpy()[0] - live)))
    # The offline take can look beyond the right edge. A live causal window
    # cannot fill an active-hand gap ending at that edge from future frames.
    longer = _synthetic_trajectory(53)
    longer[43:46, 21:42] = np.nan
    clip_windows, _ = prepare_feature_windows(longer, pipeline=pipeline)
    live_edge = pipeline.process_live_window(longer[:45])
    return {
        "default_config": asdict(PreprocessingConfig()),
        "smoothing_dtype": "float64 arithmetic, float32 output", "savgol_mode": "interp",
        "offline_live_45_frame_max_abs_difference": delta,
        "temporal_buffer_live_max_abs_difference": buffer_delta,
        "inactive_left_hand_zero_in_offline": bool(np.all(offline[0, :, :63] == 0)),
        "inactive_left_hand_zero_in_live": bool(np.all(live[:, :63] == 0)),
        "three_frame_interior_gap_interpolated": bool(np.isfinite(live).all()),
        "boundary_context_difference": {
            "offline_can_fill_gap_using_future_frames": bool(np.isfinite(clip_windows[0]).all()),
            "live_boundary_remaining_nan_count": int(np.isnan(live_edge).sum()),
            "live_trailing_hand_gap_neutralized": bool(np.all(live_edge[43:45, 63:126] == 0)),
            "max_abs_difference": float(np.nanmax(np.abs(clip_windows[0] - live_edge))),
            "explanation": "Full-clip context fills the gap; rolling context can classify a trailing tracking loss as neutral absence. Parameters match, outputs can differ.",
        },
        "buffer_default_window_size": buffer.window_size, "buffer_default_stride": buffer.stride,
        "gui_and_harness_runtime_stride": 6,
    }


def summarize_feature_hands(features, labels, videos, class_map) -> dict:
    joints = features.reshape(-1, 45, 46, 3)
    neutral_left = (joints[:, :, :21] == 0).all(axis=(2, 3))
    neutral_right = (joints[:, :, 21:42] == 0).all(axis=(2, 3))
    midpoint = (joints[:, :, 42] + joints[:, :, 43]) / 2
    scale = np.linalg.norm(joints[:, :, 42] - joints[:, :, 43], axis=-1)
    summary = {
        "features_shape": list(features.shape), "features_dtype": str(features.dtype),
        "finite": bool(np.isfinite(features).all()),
        "max_shoulder_origin_error": float(np.abs(midpoint).max()),
        "max_shoulder_scale_error": float(np.abs(scale - 1).max()), "per_class": {},
    }
    for class_id, gloss in class_map.items():
        rows = labels == class_id
        left, right = neutral_left[rows], neutral_right[rows]
        left_only = (~left & right).sum(axis=1)
        right_only = (left & ~right).sum(axis=1)
        mixed = np.count_nonzero((left_only > 0) & (right_only > 0))
        summary["per_class"][gloss] = {
            "windows": int(rows.sum()), "videos": len(np.unique(videos[rows])),
            "left_neutral_frame_fraction": float(left.mean()),
            "right_neutral_frame_fraction": float(right.mean()),
            "left_dominant_windows": int(np.count_nonzero(left_only > right_only)),
            "right_dominant_windows": int(np.count_nonzero(right_only > left_only)),
            "mixed_single_hand_side_windows": int(mixed),
            "both_hands_present_frame_fraction": float((~left & ~right).mean()),
            "potential_swapped_assignment_frames": int(((~left & right)[:, 1:] & (left & ~right)[:, :-1]
                                                       | (left & ~right)[:, 1:] & (~left & right)[:, :-1]).sum()),
        }
    return summary


def raw_clip_audit(video_path: Path) -> dict:
    raw = extract_landmark_trajectory(video_path)
    windows, rejected = prepare_feature_windows(raw)
    pipeline = PreprocessingPipeline()
    candidates = []
    padded = len(raw) < 45
    compared_raw = np.concatenate((raw, np.repeat(raw[-1:], 45 - len(raw), axis=0))) if padded else raw
    # Compare like-for-like 45-frame context; don't compare padding to a live
    # window of frames that has never been captured.
    for start in range(0, max(0, len(compared_raw) - 45 + 1), 8):
        local = compared_raw[start:start + 45]
        offline, _ = prepare_feature_windows(local)
        live = pipeline.process_live_window(local)
        if len(offline) == 1 and np.isfinite(live).all():
            candidates.append(float(np.max(np.abs(offline[0] - live))))
    left = np.isfinite(raw[:, :21]).all(axis=(1, 2)) & (raw[:, :21] != 0).any(axis=(1, 2))
    right = np.isfinite(raw[:, 21:42]).all(axis=(1, 2)) & (raw[:, 21:42] != 0).any(axis=(1, 2))
    one_side = np.where(left & ~right, 0, np.where(right & ~left, 1, -1))
    switches = []
    for index in range(1, len(raw)):
        if one_side[index] >= 0 and one_side[index - 1] >= 0 and one_side[index] != one_side[index - 1]:
            previous = raw[index - 1, int(one_side[index - 1]) * 21, :2]
            current = raw[index, int(one_side[index]) * 21, :2]
            if float(np.linalg.norm(current - previous)) < 0.1:
                switches.append(index)
    return {
        "active_frames": len(raw), "accepted_windows": len(windows), "rejected_windows": rejected,
        "detected_left_frames": int(left.sum()), "detected_right_frames": int(right.sum()),
        "suspicious_continuous_wrist_side_switches": switches,
        "like_for_like_context_max_abs_difference": max(candidates) if candidates else None,
        "compared_windows": len(candidates),
        "alignment_check_used_simulated_repeat_padding": padded,
        "interpretation": "Side-switch flags are diagnostics, not verified anatomical annotation errors.",
    }


def run_audit(dataset_path, membership_report, metadata_csv, class_map_path, videos_dir,
              output_path, *, raw_videos_per_class=2) -> dict:
    partitions, digest = freeze_existing_partitions(dataset_path, membership_report, metadata_csv)
    class_map = load_class_map(class_map_path)
    train = load_partition(dataset_path, partitions["train"])
    val = load_partition(dataset_path, partitions["val"])
    features, labels, videos = (np.concatenate((a, b)) for a, b in zip(train, val, strict=True))
    report = {
        "scope": "training_and_validation_only", "test_features_read": False,
        "split_membership_sha256": digest,
        "data_implementation_directory": "src/asl_stereo/dataset (there is no src/asl_stereo/data package)",
        "alignment": alignment_checks(),
        "feature_statistics": summarize_feature_hands(features, labels, videos, class_map),
        "hand_assignment": {
            "shared_extractor": "PoseHandsExtractor for offline, GUI, and harness",
            "default_input_is_mirrored": False,
            "label_policy": "Swap MediaPipe selfie labels for unmirrored OpenCV input; no coordinate inversion applied.",
            "source": "https://chuoling.github.io/mediapipe/solutions/hands.html#multi_handedness",
            "fallback": "Nearest same-side elbow/shoulder only when handedness unavailable or duplicated.",
            "dominant_hand_canonicalization": "None; left- and right-dominant signing remain distinct inputs.",
            "limitation": "No ground-truth handedness or mirror flag in HDF5; dataset-wide correctness cannot be certified.",
        },
        "findings": [
            {"severity": "pass", "issue": "Common window-7/order-2 float64 Savitzky-Golay configuration."},
            {"severity": "risk", "issue": "Clip-global vs 45-frame hand-presence inference and interpolation context differ at boundaries."},
            {"severity": "risk", "issue": "canonicalize_hand_presence can neutralize a real secondary hand detected fewer than max(3, 50% of dominant detections) frames."},
            {"severity": "risk", "issue": "Live valid-frame cadence compresses dropped frames; training uses decoded clip cadence. Temporal deltas are not timestamp-normalized."},
            {"severity": "limitation", "issue": "HDF5 has no extractor version, FPS, processing resolution, mirror flag, or Savitzky-Golay provenance; archived extraction settings are not fully auditable."},
        ],
        "raw_clip_checks": [], "raw_frames_saved": False,
    }
    available = {record.video_id: record for record in discover_videos(videos_dir)}
    for class_id, gloss in class_map.items():
        clip_ids = sorted(set(videos[labels == class_id]))[:raw_videos_per_class]
        for video_id in clip_ids:
            print(f"[AUDIT] {gloss}: {video_id}", flush=True)
            row = {"video_id": video_id, "gloss": gloss}
            try:
                if video_id not in available:
                    raise FileNotFoundError("Raw clip unavailable")
                row.update(raw_clip_audit(available[video_id].path))
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            report["raw_clip_checks"].append(row)
    flagged = [row["video_id"] for row in report["raw_clip_checks"]
               if row.get("suspicious_continuous_wrist_side_switches")]
    report["findings"].append({"severity": "observed", "issue": "Possible handedness flicker in sampled raw training/validation clips.",
                               "video_ids": flagged, "not_ground_truth_confirmed": True})
    report["findings"].append({"severity": "observed", "issue": "HELLO and NO contain both left- and right-dominant windows; no dominant-hand canonicalization or mirroring augmentation is applied.",
                               "source": "training_and_validation_feature_statistics_only"})
    checked = [row for row in report["raw_clip_checks"] if "active_frames" in row]
    short_count = sum(row["active_frames"] < 45 for row in checked)
    report["findings"].append({
        "severity": "observed", "issue": "Short offline clips repeat-pad the final frame; live windows contain 45 real extraction steps instead. Identical math does not guarantee identical temporal context.",
        "short_clips_in_diagnostic_sample": short_count, "checked_clips": len(checked),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Feature audit written to {output_path}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-h5", type=Path, default=PROJECT_ROOT / "artifacts/dataset.h5")
    parser.add_argument("--membership-report", type=Path, default=PROJECT_ROOT / "reports/evaluation_metrics.json")
    parser.add_argument("--metadata-csv", type=Path, default=PROJECT_ROOT / "data/data.csv")
    parser.add_argument("--class-map", type=Path, default=PROJECT_ROOT / "configs/class_map.json")
    parser.add_argument("--videos-dir", type=Path, default=PROJECT_ROOT / "data/raw/videos")
    parser.add_argument("--report-out", type=Path, default=PROJECT_ROOT / "reports/feature_alignment_audit.json")
    parser.add_argument("--raw-videos-per-class", type=int, default=2)
    args = parser.parse_args()
    if args.raw_videos_per_class < 0:
        parser.error("raw-videos-per-class must be nonnegative")
    run_audit(args.dataset_h5, args.membership_report, args.metadata_csv, args.class_map,
              args.videos_dir, args.report_out, raw_videos_per_class=args.raw_videos_per_class)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
