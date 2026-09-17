"""ASL-specific hand skeleton and discrete upper-body overlay."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .landmark_mapping import HAND_CONNECTIONS, UPPER_BODY_SOURCE_INDICES


def draw_asl_overlay(
    frame: npt.NDArray[np.uint8], results: Any
) -> npt.NDArray[np.uint8]:
    """Draw hands and four pose markers in place, returning ``frame``.

    No face data, pose connections, or torso connections are accessed.
    """
    import cv2

    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise TypeError("frame must be a uint8 numpy array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape (height, width, 3)")
    if results is None:
        return frame

    height, width = frame.shape[:2]
    for attribute, color in (
        ("left_hand_landmarks", (64, 220, 64)),
        ("right_hand_landmarks", (255, 160, 32)),
    ):
        points = _landmark_sequence(getattr(results, attribute, None))
        if points is None:
            continue
        pixels = [_to_pixel(point, width, height) for point in points]
        for start, end in HAND_CONNECTIONS:
            if start < len(pixels) and end < len(pixels):
                p1, p2 = pixels[start], pixels[end]
                if p1 is not None and p2 is not None:
                    cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
        for pixel in pixels:
            if pixel is not None:
                cv2.circle(frame, pixel, 3, color, -1, cv2.LINE_AA)

    for point in _upper_body_points(results):
        pixel = _to_pixel(point, width, height)
        if pixel is not None:
            cv2.circle(frame, pixel, 6, (0, 215, 255), -1, cv2.LINE_AA)
    return frame


def _landmark_sequence(container: Any) -> Any:
    if container is None:
        return None
    return getattr(container, "landmark", container)


def _upper_body_points(results: Any) -> list[Any]:
    restricted = getattr(results, "upper_body_landmarks", None)
    if restricted is not None:
        return [point for _, point in restricted]

    pose = getattr(results, "pose_landmarks", None)
    sequence = _landmark_sequence(pose)
    if sequence is None:
        return []
    return [sequence[index] for index in UPPER_BODY_SOURCE_INDICES if index < len(sequence)]


def _to_pixel(point: Any, width: int, height: int) -> tuple[int, int] | None:
    x = float(getattr(point, "x", np.nan))
    y = float(getattr(point, "y", np.nan))
    if not np.isfinite((x, y)).all():
        return None
    return int(round(x * (width - 1))), int(round(y * (height - 1)))


__all__ = ["draw_asl_overlay"]
