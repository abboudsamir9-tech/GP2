import numpy as np
import pytest
from types import SimpleNamespace

from asl_stereo.evaluation import (
    count_tracking_loss_episodes,
    joint_recovery_rate,
    mean_trajectory_jitter,
    reduction_percent,
    tracking_metrics,
)
from asl_stereo.stereo import JointStatus
from scripts.benchmark_occlusion import classification_metrics


def _smooth_sequence(frame_count: int = 12) -> np.ndarray:
    times = np.arange(frame_count, dtype=np.float32)[:, None, None]
    joints = np.arange(46, dtype=np.float32)[None, :, None]
    coordinates = np.concatenate(
        (
            np.broadcast_to(times, (frame_count, 46, 1)),
            np.broadcast_to(joints, (frame_count, 46, 1)),
            np.broadcast_to(times * 0.5 + joints * 0.01, (frame_count, 46, 1)),
        ),
        axis=2,
    )
    return coordinates.astype(np.float32)


def test_joint_recovery_rate_counts_only_triangulated_missing_front_joints() -> None:
    front = _smooth_sequence(10)
    fused = front.copy()
    statuses = np.full((10, 46), JointStatus.FRONT_FALLBACK.value, dtype=object)
    front[2:7, 8, 2] = np.nan
    front[2:7, 9] = np.nan
    fused[2:7, 8] = (1.0, 2.0, 3.0)
    fused[2:7, 9] = (1.0, 2.0, 3.0)
    statuses[2:7, 8] = JointStatus.TRIANGULATED.value
    statuses[2:7, 9] = JointStatus.MISSING.value

    assert joint_recovery_rate(front, fused, statuses) == pytest.approx(0.5)


def test_tracking_loss_episodes_count_runs_of_at_least_five_per_joint() -> None:
    sequence = _smooth_sequence(14)
    sequence[1:6, 2] = np.nan
    sequence[7:13, 2] = np.nan
    sequence[3:7, 3] = np.nan

    assert count_tracking_loss_episodes(sequence) == 2


def test_stereo_track_improves_gaps_and_jitter() -> None:
    baseline = _smooth_sequence(12)
    proposed = _smooth_sequence(12)
    baseline[3:9, 5] = np.nan
    baseline[4, 6, 0] += 20.0
    proposed[4, 6] = (4.0, 6.0, 2.06)

    baseline_metrics = tracking_metrics(baseline)
    proposed_metrics = tracking_metrics(proposed)

    assert proposed_metrics.missing_joint_rate < baseline_metrics.missing_joint_rate
    assert mean_trajectory_jitter(proposed) < mean_trajectory_jitter(baseline)
    assert proposed_metrics.tracking_loss_episodes < baseline_metrics.tracking_loss_episodes
    assert reduction_percent(
        baseline_metrics.mean_jitter, proposed_metrics.mean_jitter
    ) > 0.0


def test_classification_metrics_include_confidence_and_rejections() -> None:
    class SyntheticEngine:
        class_map = {0: "HELLO", 1: "NO"}

        @staticmethod
        def predict(_window):
            return SimpleNamespace(
                predicted_gloss="HELLO",
                confidence_score=0.8,
                probabilities=(0.8, 0.2),
                latency_ms=1.0,
            )

    metrics = classification_metrics(_smooth_sequence(53), SyntheticEngine(), "HELLO")

    assert metrics.evaluated_windows == 2
    assert metrics.valid_windows == 2
    assert metrics.top1_accuracy == 1.0
    assert metrics.mean_confidence == pytest.approx(0.8)
    assert metrics.false_rejection_rate == 0.0
