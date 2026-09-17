import numpy as np
import pytest

from asl_stereo.preprocessing import interpolate_short_nan_gaps


def test_interpolates_interior_gaps_shorter_than_five() -> None:
    sequence = np.array([0.0, np.nan, np.nan, np.nan, np.nan, 5.0], dtype=np.float32)
    result = interpolate_short_nan_gaps(sequence)
    np.testing.assert_allclose(result, np.arange(6, dtype=np.float32))
    assert result.dtype == np.float32


def test_preserves_gap_of_five_frames() -> None:
    sequence = np.array([0.0, np.nan, np.nan, np.nan, np.nan, np.nan, 6.0])
    result = interpolate_short_nan_gaps(sequence)
    assert np.isnan(result[1:6]).all()


@pytest.mark.parametrize(
    "sequence",
    [np.array([np.nan, 1.0, 2.0]), np.array([1.0, 2.0, np.nan])],
)
def test_preserves_boundary_gaps(sequence: np.ndarray) -> None:
    result = interpolate_short_nan_gaps(sequence)
    assert np.array_equal(np.isnan(result), np.isnan(sequence))


def test_interpolates_channels_without_mutating_input() -> None:
    sequence = np.array([[0.0, np.nan], [np.nan, 1.0], [2.0, 2.0]], dtype=np.float32)
    original = sequence.copy()
    result = interpolate_short_nan_gaps(sequence)
    assert result[1, 0] == pytest.approx(1.0)
    assert np.isnan(result[0, 1])
    np.testing.assert_array_equal(sequence, original)

