"""Public stage-boundary data contracts."""

from .features import FeatureVector, TemporalWindow
from .frames import SynchronizedFramePair, TimestampedFrame
from .landmarks import (
    LEFT_ELBOW_INDEX,
    LEFT_HAND_SLICE,
    LEFT_SHOULDER_INDEX,
    RIGHT_ELBOW_INDEX,
    RIGHT_HAND_SLICE,
    RIGHT_SHOULDER_INDEX,
    LandmarkFrame,
)
from .predictions import Prediction
from .validation import COORDINATE_COUNT, FEATURE_COUNT, JOINT_COUNT

__all__ = [
    "COORDINATE_COUNT",
    "FEATURE_COUNT",
    "JOINT_COUNT",
    "LEFT_ELBOW_INDEX",
    "LEFT_HAND_SLICE",
    "LEFT_SHOULDER_INDEX",
    "RIGHT_ELBOW_INDEX",
    "RIGHT_HAND_SLICE",
    "RIGHT_SHOULDER_INDEX",
    "FeatureVector",
    "LandmarkFrame",
    "Prediction",
    "SynchronizedFramePair",
    "TemporalWindow",
    "TimestampedFrame",
]
