"""Synthetic test-data builders."""

from .synthetic_cameras import make_projection_matrices, project_points
from .synthetic_landmarks import make_landmark_frame, make_normalized_motion

__all__ = [
    "make_landmark_frame",
    "make_normalized_motion",
    "make_projection_matrices",
    "project_points",
]

