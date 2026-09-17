"""Synthetic pinhole cameras for future stereo geometry tests."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def make_projection_matrices(
    baseline: float = 0.25,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    intrinsic = np.array(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    front = intrinsic @ np.column_stack((np.eye(3), np.zeros(3)))
    side_translation = np.array([-baseline, 0.0, 0.0], dtype=np.float64)
    side = intrinsic @ np.column_stack((np.eye(3), side_translation))
    return front, side


def project_points(
    projection: npt.NDArray[np.float64], points: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    image = (projection @ homogeneous.T).T
    return image[:, :2] / image[:, 2:3]


__all__ = ["make_projection_matrices", "project_points"]

