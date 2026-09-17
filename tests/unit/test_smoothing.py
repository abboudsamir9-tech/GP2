import numpy as np

from asl_stereo.preprocessing import smooth_trajectories


def test_smoothing_returns_float32_and_preserves_linear_signal() -> None:
    sequence = np.arange(9, dtype=np.float32)[:, None]
    result = smooth_trajectories(sequence, window_length=5, polyorder=2)
    np.testing.assert_allclose(result, sequence, atol=1e-5)
    assert result.dtype == np.float32


def test_channel_with_nan_is_skipped() -> None:
    sequence = np.column_stack(
        (np.arange(9, dtype=np.float32), np.arange(9, dtype=np.float32) ** 2)
    )
    sequence[3, 1] = np.nan
    result = smooth_trajectories(sequence, window_length=5, polyorder=2)
    np.testing.assert_array_equal(result[:, 1], sequence[:, 1])


def test_short_sequence_is_unchanged() -> None:
    sequence = np.array([1.0, 4.0, 2.0], dtype=np.float32)
    np.testing.assert_array_equal(smooth_trajectories(sequence, window_length=5), sequence)

