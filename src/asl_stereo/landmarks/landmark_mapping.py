"""Canonical MediaPipe-to-model joint mapping and hand topology."""

from asl_stereo.contracts import JOINT_COUNT
from asl_stereo.contracts.landmarks import (
    LEFT_ELBOW_INDEX,
    LEFT_HAND_SLICE,
    LEFT_SHOULDER_INDEX,
    RIGHT_ELBOW_INDEX,
    RIGHT_HAND_SLICE,
    RIGHT_SHOULDER_INDEX,
)

HAND_LANDMARK_COUNT = 21
LEFT_HAND_OFFSET = 0
RIGHT_HAND_OFFSET = 21
UPPER_BODY_OFFSET = 42

POSE_SOURCE_TO_OUTPUT = {
    11: LEFT_SHOULDER_INDEX,
    12: RIGHT_SHOULDER_INDEX,
    13: LEFT_ELBOW_INDEX,
    14: RIGHT_ELBOW_INDEX,
}
UPPER_BODY_SOURCE_INDICES = tuple(POSE_SOURCE_TO_OUTPUT)

HAND_LANDMARK_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

LANDMARK_NAMES = tuple(f"left_{name}" for name in HAND_LANDMARK_NAMES) + tuple(
    f"right_{name}" for name in HAND_LANDMARK_NAMES
) + ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow")

# Equivalent to MediaPipe Hands HAND_CONNECTIONS, defined locally so geometry
# and tests do not require MediaPipe to be imported.
HAND_CONNECTIONS = frozenset(
    {
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
    }
)

assert len(HAND_LANDMARK_NAMES) == HAND_LANDMARK_COUNT
assert len(LANDMARK_NAMES) == JOINT_COUNT

__all__ = [
    "HAND_CONNECTIONS",
    "HAND_LANDMARK_COUNT",
    "HAND_LANDMARK_NAMES",
    "JOINT_COUNT",
    "LANDMARK_NAMES",
    "LEFT_ELBOW_INDEX",
    "LEFT_HAND_OFFSET",
    "LEFT_HAND_SLICE",
    "LEFT_SHOULDER_INDEX",
    "POSE_SOURCE_TO_OUTPUT",
    "RIGHT_ELBOW_INDEX",
    "RIGHT_HAND_OFFSET",
    "RIGHT_HAND_SLICE",
    "RIGHT_SHOULDER_INDEX",
    "UPPER_BODY_OFFSET",
    "UPPER_BODY_SOURCE_INDICES",
]
