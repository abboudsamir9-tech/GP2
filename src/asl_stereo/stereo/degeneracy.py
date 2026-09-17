"""Geometric rejection policy for candidate triangulations."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def is_triangulation_degenerate(
    point3d: npt.ArrayLike,
    P1: npt.ArrayLike,
    P2: npt.ArrayLike,
    reprojection_error: float,
    *,
    min_ray_angle_degrees: float = 1.0,
    max_reprojection_error: float = 10.0,
) -> bool:
    """Return True for non-finite, behind-camera, parallel, or inaccurate points."""
    point = np.asarray(point3d, dtype=np.float64)
    projection1 = np.asarray(P1, dtype=np.float64)
    projection2 = np.asarray(P2, dtype=np.float64)
    if point.shape != (3,) or projection1.shape != (3, 4) or projection2.shape != (3, 4):
        raise ValueError("expected point shape (3,) and projection shapes (3, 4)")
    if not np.isfinite(point).all() or not np.isfinite(reprojection_error):
        return True
    if reprojection_error > max_reprojection_error:
        return True

    homogeneous = np.append(point, 1.0)
    depth1 = float((projection1 @ homogeneous)[2])
    depth2 = float((projection2 @ homogeneous)[2])
    if not np.isfinite((depth1, depth2)).all() or depth1 <= 0.0 or depth2 <= 0.0:
        return True

    center1 = camera_center(projection1)
    center2 = camera_center(projection2)
    if not np.isfinite(center1).all() or not np.isfinite(center2).all():
        return True
    ray1 = point - center1
    ray2 = point - center2
    norm1 = float(np.linalg.norm(ray1))
    norm2 = float(np.linalg.norm(ray2))
    if norm1 <= np.finfo(np.float64).eps or norm2 <= np.finfo(np.float64).eps:
        return True
    cosine = float(np.clip(np.dot(ray1, ray2) / (norm1 * norm2), -1.0, 1.0))
    angle_degrees = float(np.degrees(np.arccos(cosine)))
    return angle_degrees < min_ray_angle_degrees


def camera_center(projection: npt.ArrayLike) -> npt.NDArray[np.float64]:
    matrix = np.asarray(projection, dtype=np.float64)
    if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
        raise ValueError("projection must be a finite matrix with shape (3, 4)")
    try:
        _, _, right_singular_vectors = np.linalg.svd(matrix, full_matrices=True)
    except np.linalg.LinAlgError:
        return np.full(3, np.nan, dtype=np.float64)
    homogeneous = right_singular_vectors[-1]
    if abs(homogeneous[3]) < np.finfo(np.float64).eps:
        return np.full(3, np.nan, dtype=np.float64)
    return homogeneous[:3] / homogeneous[3]


__all__ = ["camera_center", "is_triangulation_degenerate"]
