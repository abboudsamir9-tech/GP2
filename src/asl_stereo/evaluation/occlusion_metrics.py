"""Pure metrics for front-only versus stereo occlusion evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from asl_stereo.contracts import JOINT_COUNT
from asl_stereo.stereo import JointStatus


@dataclass(frozen=True, slots=True)
class TrackingMetrics:
    available_joint_rate: float
    missing_joint_rate: float
    mean_jitter: float
    tracking_loss_episodes: int


def joint_recovery_rate(
    front: npt.ArrayLike,
    fused: npt.ArrayLike,
    statuses: npt.ArrayLike | None = None,
) -> float:
    """Return the fraction of missing front joints recovered by triangulation."""
    front_values = _sequence("front", front)
    fused_values = _sequence("fused", fused)
    if front_values.shape != fused_values.shape:
        raise ValueError("front and fused sequences must have identical shapes")
    front_missing = ~np.isfinite(front_values).all(axis=2)
    missing_count = int(front_missing.sum())
    if missing_count == 0:
        return 0.0
    recovered = front_missing & np.isfinite(fused_values).all(axis=2)
    if statuses is not None:
        status_values = np.asarray(statuses)
        if status_values.shape != front_missing.shape:
            raise ValueError("statuses must have shape (frames, 46)")
        recovered &= status_values == JointStatus.TRIANGULATED.value
    return float(recovered.sum() / missing_count)


def mean_trajectory_jitter(sequence: npt.ArrayLike) -> float:
    """Measure mean Euclidean second-difference over contiguous valid triplets."""
    values = _sequence("sequence", sequence).astype(np.float64, copy=False)
    if values.shape[0] < 3:
        return 0.0
    valid = np.isfinite(values).all(axis=2)
    triplets = valid[:-2] & valid[1:-1] & valid[2:]
    acceleration = values[2:] - 2.0 * values[1:-1] + values[:-2]
    magnitudes = np.linalg.norm(acceleration, axis=2)
    selected = magnitudes[triplets]
    return float(selected.mean()) if selected.size else 0.0


def count_tracking_loss_episodes(
    sequence: npt.ArrayLike, *, min_run: int = 5
) -> int:
    """Count per-joint missing-data runs lasting at least ``min_run`` frames."""
    if isinstance(min_run, bool) or not isinstance(min_run, int) or min_run <= 0:
        raise ValueError("min_run must be a positive integer")
    missing = ~np.isfinite(_sequence("sequence", sequence)).all(axis=2)
    episodes = 0
    for joint_missing in missing.T:
        run_length = 0
        for is_missing in joint_missing:
            if is_missing:
                run_length += 1
            else:
                if run_length >= min_run:
                    episodes += 1
                run_length = 0
        if run_length >= min_run:
            episodes += 1
    return episodes


def tracking_metrics(sequence: npt.ArrayLike) -> TrackingMetrics:
    values = _sequence("sequence", sequence)
    available = np.isfinite(values).all(axis=2)
    available_rate = float(available.mean()) if available.size else 0.0
    return TrackingMetrics(
        available_joint_rate=available_rate,
        missing_joint_rate=1.0 - available_rate,
        mean_jitter=mean_trajectory_jitter(values),
        tracking_loss_episodes=count_tracking_loss_episodes(values),
    )


def reduction_percent(baseline: float, proposed: float) -> float:
    """Return percentage reduction, using zero for an undefined zero baseline."""
    if not np.isfinite((baseline, proposed)).all() or baseline < 0 or proposed < 0:
        raise ValueError("metric values must be finite and non-negative")
    return 0.0 if baseline == 0.0 else float((baseline - proposed) / baseline * 100.0)


def _sequence(name: str, values: Any) -> npt.NDArray[np.floating[Any]]:
    output = np.asarray(values)
    if output.ndim != 3 or output.shape[1:] != (JOINT_COUNT, 3):
        raise ValueError(f"{name} must have shape (frames, 46, 3)")
    if not np.issubdtype(output.dtype, np.floating):
        raise TypeError(f"{name} must use a floating-point dtype")
    if np.isinf(output).any():
        raise ValueError(f"{name} may contain NaN but not infinity")
    return output


__all__ = [
    "TrackingMetrics",
    "count_tracking_loss_episodes",
    "joint_recovery_rate",
    "mean_trajectory_jitter",
    "reduction_percent",
    "tracking_metrics",
]
