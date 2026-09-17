"""Per-joint stereo reconstruction with front-view fallback."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import numpy.typing as npt

from asl_stereo.contracts import JOINT_COUNT
from asl_stereo.contracts.validation import readonly_view

from .calibration import StereoCalibration
from .degeneracy import is_triangulation_degenerate
from .triangulation import triangulate_dlt


class JointStatus(str, Enum):
    TRIANGULATED = "TRIANGULATED"
    FRONT_FALLBACK = "FRONT_FALLBACK"
    MISSING = "MISSING"


@dataclass(frozen=True, slots=True)
class StereoMatchResult:
    coordinates: npt.NDArray[np.float32]
    statuses: tuple[JointStatus, ...]
    reprojection_errors: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.coordinates.shape != (JOINT_COUNT, 3):
            raise ValueError("coordinates must have shape (46, 3)")
        if self.coordinates.dtype != np.float32:
            raise TypeError("coordinates must have dtype float32")
        if len(self.statuses) != JOINT_COUNT:
            raise ValueError("statuses must contain exactly 46 entries")
        if self.reprojection_errors.shape != (JOINT_COUNT,):
            raise ValueError("reprojection_errors must have shape (46,)")
        if self.reprojection_errors.dtype != np.float32:
            raise TypeError("reprojection_errors must have dtype float32")
        object.__setattr__(self, "coordinates", readonly_view(self.coordinates))
        object.__setattr__(
            self, "reprojection_errors", readonly_view(self.reprojection_errors)
        )

    def __iter__(self):
        """Allow ``coordinates, statuses = matcher.match(...)`` unpacking."""
        yield self.coordinates
        yield self.statuses


class StereoMatcher:
    def __init__(
        self,
        calibration: StereoCalibration | None = None,
        *,
        min_ray_angle_degrees: float = 1.0,
        max_reprojection_error: float = 10.0,
        fallback_scope: Literal["joint", "frame"] = "joint",
    ) -> None:
        self.calibration = calibration or StereoCalibration()
        if min_ray_angle_degrees < 0.0 or min_ray_angle_degrees >= 180.0:
            raise ValueError("min_ray_angle_degrees must be in [0, 180)")
        if max_reprojection_error < 0.0:
            raise ValueError("max_reprojection_error must be non-negative")
        if fallback_scope not in ("joint", "frame"):
            raise ValueError("fallback_scope must be 'joint' or 'frame'")
        self.min_ray_angle_degrees = min_ray_angle_degrees
        self.max_reprojection_error = max_reprojection_error
        self.fallback_scope = fallback_scope

    def match(
        self,
        front_landmarks: npt.ArrayLike,
        side_landmarks: npt.ArrayLike,
        *,
        front_fallback: npt.ArrayLike | None = None,
    ) -> StereoMatchResult:
        front = _landmark_array("front_landmarks", front_landmarks)
        side = _landmark_array("side_landmarks", side_landmarks)
        fallback = (
            front
            if front_fallback is None
            else _landmark_array("front_fallback", front_fallback)
        )
        fused = np.full((JOINT_COUNT, 3), np.nan, dtype=np.float32)
        errors = np.full(JOINT_COUNT, np.nan, dtype=np.float32)
        statuses: list[JointStatus] = []

        for joint_index in range(JOINT_COUNT):
            front_full_valid = bool(np.isfinite(fallback[joint_index]).all())
            front_2d_valid = bool(np.isfinite(front[joint_index, :2]).all())
            side_2d_valid = bool(np.isfinite(side[joint_index, :2]).all())

            if self.calibration.available and front_2d_valid and side_2d_valid:
                assert self.calibration.P_front is not None
                assert self.calibration.P_side is not None
                point, error = triangulate_dlt(
                    front[joint_index, :2],
                    side[joint_index, :2],
                    self.calibration.P_front,
                    self.calibration.P_side,
                )
                errors[joint_index] = np.float32(error)
                degenerate = is_triangulation_degenerate(
                    point,
                    self.calibration.P_front,
                    self.calibration.P_side,
                    error,
                    min_ray_angle_degrees=self.min_ray_angle_degrees,
                    max_reprojection_error=self.max_reprojection_error,
                )
                if not degenerate:
                    fused[joint_index] = point.astype(np.float32)
                    statuses.append(JointStatus.TRIANGULATED)
                    continue

            if front_full_valid:
                fused[joint_index] = fallback[joint_index]
                statuses.append(JointStatus.FRONT_FALLBACK)
            else:
                statuses.append(JointStatus.MISSING)

        if self.fallback_scope == "frame" and any(
            status is not JointStatus.TRIANGULATED for status in statuses
        ):
            fallback_valid = np.isfinite(fallback).all(axis=1)
            fused[:] = np.nan
            fused[fallback_valid] = fallback[fallback_valid].astype(np.float32)
            statuses = [
                JointStatus.FRONT_FALLBACK if valid else JointStatus.MISSING
                for valid in fallback_valid
            ]

        return StereoMatchResult(fused, tuple(statuses), errors)


def _landmark_array(name: str, values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    output = np.asarray(values)
    if output.shape != (JOINT_COUNT, 3):
        raise ValueError(f"{name} must have shape (46, 3)")
    if not np.issubdtype(output.dtype, np.floating):
        raise TypeError(f"{name} must use a floating-point dtype")
    if np.isinf(output).any():
        raise ValueError(f"{name} may contain NaN but not infinity")
    return output.astype(np.float64, copy=False)


__all__ = ["JointStatus", "StereoMatchResult", "StereoMatcher"]
