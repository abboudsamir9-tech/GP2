import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from asl_stereo.contracts import LandmarkFrame
from asl_stereo.landmarks import PoseHandsExtractor
from asl_stereo.landmarks.holistic_extractor import _load_solution_modules
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


class _FakeTracker:
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


def test_extractor_zero_pads_never_seen_hand() -> None:
    pose = SimpleNamespace(landmark=[_point(index / 100.0) for index in range(33)])
    right_hand = SimpleNamespace(landmark=[_point(index / 20.0) for index in range(21)])
    pose_result = SimpleNamespace(pose_landmarks=pose, face_landmarks=object())
    hands_result = SimpleNamespace(
        multi_hand_landmarks=[right_hand],
        multi_handedness=[SimpleNamespace(classification=[
            SimpleNamespace(label="Left", score=0.99)
        ])],
    )
    pose_options, hands_options = {}, {}

    def pose_factory(**options):
        pose_options.update(options)
        return _FakeTracker(pose_result, options)

    def hands_factory(**options):
        hands_options.update(options)
        return _FakeTracker(hands_result, options)

    extractor = PoseHandsExtractor(pose_factory=pose_factory, hands_factory=hands_factory)
    output = extractor.process(
        np.zeros((8, 8, 3), dtype=np.uint8), timestamp_ns=10, frame_index=2
    )

    assert output.coordinates.shape == (46, 3)
    assert output.coordinates.dtype == np.float32
    np.testing.assert_array_equal(output.coordinates[LEFT_HAND_SLICE], 0.0)
    np.testing.assert_array_equal(output.hand_present, np.array([0.0, 1.0], np.float32))
    np.testing.assert_array_equal(output.joint_mask[:21], np.ones(21, np.float32))
    np.testing.assert_array_equal(output.joint_mask[21:], np.ones(25, np.float32))
    assert pose_options["enable_segmentation"] is False
    assert pose_options["model_complexity"] == 0
    assert hands_options["model_complexity"] == 0
    assert hands_options["max_num_hands"] == 2
    assert extractor.last_results is not None
    assert not hasattr(extractor.last_results, "face_landmarks")
    assert not hasattr(extractor.last_results, "pose_landmarks")
    assert output.coordinates.reshape(-1).shape == (138,)


def test_extractor_downscales_only_inference_image() -> None:
    pose_result = SimpleNamespace(pose_landmarks=None)
    hands_result = SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)
    received = []

    class RecordingTracker(_FakeTracker):
        def process(self, frame):
            received.append((frame.shape, frame.flags.writeable))
            return super().process(frame)

    pose_model = RecordingTracker(pose_result, {})
    hand_model = RecordingTracker(hands_result, {})
    extractor = PoseHandsExtractor(
        pose_factory=lambda **options: pose_model,
        hands_factory=lambda **options: hand_model,
    )
    full_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    output = extractor.process(full_frame, timestamp_ns=1, frame_index=0)

    assert pose_model.result is pose_result
    assert hand_model.result is hands_result
    assert received == [((360, 640, 3), False), ((360, 640, 3), False)]
    assert output.coordinates.shape == (46, 3)
    assert full_frame.shape == (720, 1280, 3)


def test_handedness_stays_stable_during_crossing() -> None:
    pose = SimpleNamespace(landmark=[_point(index / 100.0) for index in range(33)])
    pose_result = SimpleNamespace(pose_landmarks=pose)
    # Wrist locations deliberately cross the opposite elbow. Unmirrored
    # handedness labels, not nearest-elbow geometry, determine anatomical side.
    left_hand = SimpleNamespace(landmark=[_point(0.1) for _ in range(21)])
    right_hand = SimpleNamespace(landmark=[_point(0.9) for _ in range(21)])
    hands_result = SimpleNamespace(
        multi_hand_landmarks=[right_hand, left_hand],
        multi_handedness=[
            SimpleNamespace(classification=[SimpleNamespace(label="Left", score=0.9)]),
            SimpleNamespace(classification=[SimpleNamespace(label="Right", score=0.9)]),
        ],
    )
    extractor = PoseHandsExtractor(
        pose_factory=lambda **_: _FakeTracker(pose_result, {}),
        hands_factory=lambda **_: _FakeTracker(hands_result, {}),
    )
    output = extractor.process(np.zeros((8, 8, 3), np.uint8))
    np.testing.assert_allclose(output.coordinates[0], (0.1, 0.2, 0.3))
    np.testing.assert_allclose(output.coordinates[21], (0.9, 1.0, 1.1))
    np.testing.assert_array_equal(output.hand_presence, (1.0, 1.0))
    extractor.close()
    assert extractor.pose_tracker.closed and extractor.hand_tracker.closed


def test_tracked_hand_loss_is_nan_until_tracking_reset() -> None:
    pose_result = SimpleNamespace(pose_landmarks=None)
    hand = SimpleNamespace(landmark=[_point(0.3) for _ in range(21)])
    hand_tracker = _FakeTracker(
        SimpleNamespace(
            multi_hand_landmarks=[hand],
            multi_handedness=[
                SimpleNamespace(classification=[SimpleNamespace(label="Left", score=0.9)])
            ],
        ),
        {},
    )
    with PoseHandsExtractor(
        pose_factory=lambda **_: _FakeTracker(pose_result, {}),
        hands_factory=lambda **_: hand_tracker,
    ) as extractor:
        observed = extractor.process(np.zeros((8, 8, 3), np.uint8))
        np.testing.assert_array_equal(observed.hand_presence, (0.0, 1.0))
        np.testing.assert_array_equal(observed.coordinates[:21], 0.0)

        hand_tracker.result = SimpleNamespace(
            multi_hand_landmarks=None, multi_handedness=None
        )
        missed = extractor.process(np.zeros((8, 8, 3), np.uint8))
        assert np.isnan(missed.coordinates[21:42]).all()
        np.testing.assert_array_equal(missed.coordinates[:21], 0.0)

        extractor.reset_tracking()
        reset = extractor.process(np.zeros((8, 8, 3), np.uint8))
        np.testing.assert_array_equal(reset.coordinates[:42], 0.0)


def test_explicit_solutions_import_uses_internal_fallback(monkeypatch) -> None:
    import mediapipe  # Fully initialize its real package before mocking imports.

    python_package = ModuleType("mediapipe.python")
    python_package.__path__ = []
    solutions_package = ModuleType("mediapipe.python.solutions")
    solutions_package.__path__ = []
    pose_module = ModuleType("mediapipe.python.solutions.pose")
    hands_module = ModuleType("mediapipe.python.solutions.hands")
    pose_module.Pose = lambda **options: _FakeTracker(
        SimpleNamespace(pose_landmarks=None), options
    )
    hands_module.Hands = lambda **options: _FakeTracker(
        SimpleNamespace(multi_hand_landmarks=None), options
    )
    solutions_package.pose = pose_module
    solutions_package.hands = hands_module
    monkeypatch.setitem(sys.modules, "mediapipe.python", python_package)
    monkeypatch.setitem(sys.modules, "mediapipe.python.solutions", solutions_package)
    monkeypatch.setitem(sys.modules, "mediapipe.python.solutions.pose", pose_module)
    monkeypatch.setitem(sys.modules, "mediapipe.python.solutions.hands", hands_module)

    assert _load_solution_modules() == (pose_module, hands_module)
    with PoseHandsExtractor(model_complexity=1) as extractor:
        frame = extractor.process(np.zeros((8, 8, 3), np.uint8))
        assert frame.coordinates.shape == (46, 3)
        np.testing.assert_array_equal(frame.coordinates[:42], 0.0)
        assert np.isnan(frame.coordinates[42:]).all()
