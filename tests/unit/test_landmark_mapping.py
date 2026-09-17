import numpy as np
from types import SimpleNamespace

from asl_stereo.contracts import LandmarkFrame
from asl_stereo.landmarks import HolisticExtractor
from asl_stereo.landmarks.landmark_mapping import (
    HAND_LANDMARK_COUNT,
    LANDMARK_NAMES,
    LEFT_HAND_OFFSET,
    LEFT_HAND_SLICE,
    LEFT_SHOULDER_INDEX,
    POSE_SOURCE_TO_OUTPUT,
    RIGHT_ELBOW_INDEX,
    RIGHT_HAND_OFFSET,
    RIGHT_HAND_SLICE,
    UPPER_BODY_OFFSET,
)


def test_missing_landmark_frame_uses_nan_not_zero() -> None:
    frame = LandmarkFrame.missing(timestamp_ns=1, frame_index=0)
    assert frame.coordinates.shape == (46, 3)
    assert np.isnan(frame.coordinates).all()
    np.testing.assert_array_equal(frame.hand_presence, np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(frame.joint_mask, np.zeros(46, dtype=np.float32))


def test_joint_layout_has_exact_offsets_and_46_names() -> None:
    assert HAND_LANDMARK_COUNT == 21
    assert LEFT_HAND_OFFSET == 0
    assert LEFT_HAND_SLICE == slice(0, 21)
    assert RIGHT_HAND_OFFSET == 21
    assert RIGHT_HAND_SLICE == slice(21, 42)
    assert UPPER_BODY_OFFSET == 42
    assert POSE_SOURCE_TO_OUTPUT == {11: 42, 12: 43, 13: 44, 14: 45}
    assert LEFT_SHOULDER_INDEX == 42
    assert RIGHT_ELBOW_INDEX == 45
    assert len(LANDMARK_NAMES) == 46
    assert len(set(LANDMARK_NAMES)) == 46


class _FakeHolistic:
    def __init__(self, result, options) -> None:
        self.result = result
        self.options = options
        self.closed = False

    def process(self, _frame):
        return self.result

    def close(self) -> None:
        self.closed = True


def _point(value: float):
    return SimpleNamespace(x=value, y=value + 0.1, z=value + 0.2)


def test_extractor_outputs_exact_shape_masks_and_nan_for_absent_hand() -> None:
    pose = SimpleNamespace(landmark=[_point(index / 100.0) for index in range(33)])
    right_hand = SimpleNamespace(landmark=[_point(index / 20.0) for index in range(21)])
    raw_result = SimpleNamespace(
        left_hand_landmarks=None,
        right_hand_landmarks=right_hand,
        pose_landmarks=pose,
        face_landmarks=object(),
    )
    created = {}

    def factory(**options):
        created.update(options)
        return _FakeHolistic(raw_result, options)

    extractor = HolisticExtractor(holistic_factory=factory)
    output = extractor.process(
        np.zeros((8, 8, 3), dtype=np.uint8), timestamp_ns=10, frame_index=2
    )

    assert output.coordinates.shape == (46, 3)
    assert output.coordinates.dtype == np.float32
    assert np.isnan(output.coordinates[LEFT_HAND_SLICE]).all()
    assert not np.equal(output.coordinates[LEFT_HAND_SLICE], 0.0).any()
    np.testing.assert_array_equal(output.hand_present, np.array([0.0, 1.0], np.float32))
    np.testing.assert_array_equal(output.joint_mask[:21], np.zeros(21, np.float32))
    np.testing.assert_array_equal(output.joint_mask[21:], np.ones(25, np.float32))
    assert created["enable_segmentation"] is False
    assert created["refine_face_landmarks"] is False
    assert extractor.last_results is not None
    assert not hasattr(extractor.last_results, "face_landmarks")
    assert not hasattr(extractor.last_results, "pose_landmarks")
