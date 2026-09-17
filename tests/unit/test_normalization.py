import numpy as np

from asl_stereo.contracts.landmarks import LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX
from asl_stereo.preprocessing import encode_feature_vector, normalize_landmarks
from tests.fixtures.synthetic_landmarks import make_landmark_frame


def test_mid_shoulders_become_origin_and_scale_becomes_one() -> None:
    result = normalize_landmarks(make_landmark_frame())
    left = result[LEFT_SHOULDER_INDEX]
    right = result[RIGHT_SHOULDER_INDEX]
    np.testing.assert_allclose((left + right) / 2.0, np.zeros(3), atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(left - right), 1.0, atol=1e-7)
    assert result.dtype == np.float32


def test_zero_shoulder_scale_invalidates_entire_frame() -> None:
    landmarks = make_landmark_frame()
    landmarks[RIGHT_SHOULDER_INDEX] = landmarks[LEFT_SHOULDER_INDEX]
    assert np.isnan(normalize_landmarks(landmarks)).all()


def test_missing_shoulder_invalidates_entire_frame() -> None:
    landmarks = make_landmark_frame()
    landmarks[LEFT_SHOULDER_INDEX] = np.nan
    assert np.isnan(normalize_landmarks(landmarks)).all()


def test_feature_layout_is_joint_major_xyz() -> None:
    normalized = normalize_landmarks(make_landmark_frame())
    encoded = encode_feature_vector(normalized)
    assert encoded.shape == (138,)
    assert encoded.dtype == np.float32
    np.testing.assert_array_equal(encoded[9:12], normalized[3])

