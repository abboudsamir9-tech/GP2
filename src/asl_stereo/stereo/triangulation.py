"""Direct Linear Transform stereo triangulation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def triangulate_dlt(
    p1: npt.ArrayLike,
    p2: npt.ArrayLike,
    P1: npt.ArrayLike,
    P2: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], float]:
    """Reconstruct one 3D point and return mean two-view pixel error."""
    point1 = _image_point("p1", p1)
    point2 = _image_point("p2", p2)
    projection1 = _projection("P1", P1)
    projection2 = _projection("P2", P2)

    u1, v1 = point1
    u2, v2 = point2
    system = np.stack(
        (
            u1 * projection1[2] - projection1[0],
            v1 * projection1[2] - projection1[1],
            u2 * projection2[2] - projection2[0],
            v2 * projection2[2] - projection2[1],
        ),
        axis=0,
    )
    try:
        _, _, right_singular_vectors = np.linalg.svd(system, full_matrices=True)
    except np.linalg.LinAlgError:
        return np.full(3, np.nan, dtype=np.float64), float("inf")

    homogeneous = right_singular_vectors[-1]
    weight = homogeneous[3]
    if not np.isfinite(homogeneous).all() or abs(weight) < np.finfo(np.float64).eps:
        return np.full(3, np.nan, dtype=np.float64), float("inf")

    point3d = homogeneous[:3] / weight
    if not np.isfinite(point3d).all():
        return np.full(3, np.nan, dtype=np.float64), float("inf")

    error1 = _reprojection_error(point3d, point1, projection1)
    error2 = _reprojection_error(point3d, point2, projection2)
    error = float((error1 + error2) / 2.0)
    return point3d.astype(np.float64, copy=False), error


def _image_point(name: str, point: npt.ArrayLike) -> npt.NDArray[np.float64]:
    output = np.asarray(point, dtype=np.float64)
    if output.shape != (2,) or not np.isfinite(output).all():
        raise ValueError(f"{name} must be a finite point with shape (2,)")
    return output


def _projection(name: str, matrix: npt.ArrayLike) -> npt.NDArray[np.float64]:
    output = np.asarray(matrix, dtype=np.float64)
    if output.shape != (3, 4) or not np.isfinite(output).all():
        raise ValueError(f"{name} must be a finite matrix with shape (3, 4)")
    return output


def _reprojection_error(
    point3d: npt.NDArray[np.float64],
    observed: npt.NDArray[np.float64],
    projection: npt.NDArray[np.float64],
) -> float:
    homogeneous = np.append(point3d, 1.0)
    projected = projection @ homogeneous
    if not np.isfinite(projected).all() or abs(projected[2]) < np.finfo(np.float64).eps:
        return float("inf")
    image = projected[:2] / projected[2]
    return float(np.linalg.norm(image - observed))


__all__ = ["triangulate_dlt"]
