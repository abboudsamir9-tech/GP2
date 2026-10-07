"""Numerical and ownership gates for the unchanged 46-joint live contract."""

import time

import numpy as np
import pytest
from scipy.signal import savgol_filter

from asl_stereo.contracts import FeatureVector
from asl_stereo.preprocessing import (
    PreprocessingPipeline,
    SlidingWindowBuffer,
    TemporalBuffer,
    interpolate_short_nan_gaps,
    smooth_trajectories,
)
from asl_stereo.preprocessing.normalization import normalize_landmarks
from tests.fixtures.synthetic_landmarks import make_landmark_frame


def _legacy_interpolation(sequence, max_gap=4):
    result = np.array(sequence, dtype=np.float64, copy=True).reshape(len(sequence), -1)
    for channel in result.T:
        cursor = 0
        while cursor < len(channel):
            if not np.isnan(channel[cursor]):
                cursor += 1
                continue
            start = cursor
            while cursor < len(channel) and np.isnan(channel[cursor]):
                cursor += 1
            if start > 0 and cursor < len(channel) and cursor - start <= max_gap:
                channel[start:cursor] = np.linspace(
                    channel[start - 1], channel[cursor], cursor - start + 2,
                )[1:-1]
    return result.reshape(sequence.shape).astype(np.float32)


def _legacy_smoothing(sequence, window_length=7, polyorder=2):
    result = np.array(sequence, dtype=np.float64, copy=True).reshape(len(sequence), -1)
    if len(sequence) >= window_length:
        for channel in result.T:
            if not np.isnan(channel).any():
                channel[:] = savgol_filter(
                    channel, window_length, polyorder, axis=0, mode="interp",
                )
    return result.reshape(sequence.shape).astype(np.float32)


@pytest.mark.parametrize("max_gap", [0, 1, 4, 7])
def test_batched_interpolation_matches_legacy_channels(max_gap):
    data = np.random.default_rng(12).normal(size=(45, 46, 3)).astype(np.float32)
    flat = data.reshape(45, 138)
    for length in range(1, 8):
        flat[10 : 10 + length, length] = np.nan
    flat[:3, 0] = np.nan
    flat[-3:, 20] = np.nan
    flat[:, 30] = np.nan
    original = data.copy()
    actual = interpolate_short_nan_gaps(data, max_gap=max_gap)
    expected = _legacy_interpolation(data, max_gap=max_gap)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6, equal_nan=True)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("shape", [(45, 138), (45, 46, 3), (45,)])
def test_batched_smoothing_matches_legacy_channels(shape):
    data = np.random.default_rng(23).normal(size=shape).astype(np.float32)
    if data.ndim > 1:
        flat = data.reshape(45, -1)
        flat[12, 2] = np.nan
        flat[:, 7] = np.nan
    original = data.copy()
    actual = smooth_trajectories(data)
    np.testing.assert_allclose(actual, _legacy_smoothing(data), rtol=0, atol=1e-6, equal_nan=True)
    np.testing.assert_array_equal(data, original)
    assert actual.dtype == np.float32 and actual.flags.c_contiguous


def test_smoothing_preserves_empty_channel_dimension():
    data = np.empty((45, 0), dtype=np.float32)
    result = smooth_trajectories(data)
    assert result.shape == data.shape and result.dtype == np.float32


def test_live_pipeline_matches_legacy_math_and_neutral_hand_convention():
    from asl_stereo.preprocessing.pipeline import canonicalize_hand_presence

    raw = np.repeat(make_landmark_frame()[None], 45, axis=0)
    raw[:, :21, 1] += np.linspace(0, 0.2, 45, dtype=np.float32)[:, None]
    raw[:, 21:42] = 0.0
    raw[10:14, 0] = np.nan
    raw[20:25, 42] = np.nan
    canonical, neutral = canonicalize_hand_presence(raw)
    smoothed = _legacy_smoothing(_legacy_interpolation(canonical))
    expected = np.stack([normalize_landmarks(frame).reshape(138) for frame in smoothed])
    for hand, feature_slice in enumerate((slice(0, 63), slice(63, 126))):
        expected[neutral[:, hand], feature_slice] = 0.0
    actual = PreprocessingPipeline().process_live_window(raw)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("frames", [0, 3, 45])
def test_batched_normalization_matches_single_frame_epsilon_and_nan_guards(frames):
    from asl_stereo.preprocessing.normalization import normalize_landmark_sequence

    data = np.random.default_rng(41).uniform(-1, 1, (frames, 46, 3)).astype(np.float32)
    if frames:
        data[0, 43] = data[0, 42]  # Zero shoulder distance.
        data[1, 42] = np.nan
        data[2, 4] = np.nan
    expected = np.stack([normalize_landmarks(frame) for frame in data]) if frames else data
    np.testing.assert_allclose(
        normalize_landmark_sequence(data), expected, rtol=0, atol=1e-6, equal_nan=True,
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_fifo_owns_each_input_even_when_producer_reuses_storage(wrapped):
    buffer = TemporalBuffer()
    shared = np.zeros(138, dtype=np.float32)
    window = None
    for index in range(45):
        shared[:] = index
        value = FeatureVector(shared, index + 1) if wrapped else shared
        window = buffer.append(value, timestamp_ns=None if wrapped else index + 1)
    shared[:] = -1
    assert window is not None
    np.testing.assert_array_equal(window.values[0, :, 0], np.arange(45, dtype=np.float32))


def test_live_preprocessing_and_raw_fifo_p95_is_below_15_ms():
    pipeline = PreprocessingPipeline()
    buffer = SlidingWindowBuffer(window_size=45, stride=8, align_features=True)
    frames = np.repeat(make_landmark_frame()[None], 45, axis=0)
    frames[:, 21:42] = 0.0
    frames[:, :21, 1] += np.linspace(0, 0.2, 45, dtype=np.float32)[:, None]
    # Exercise interpolation, not just the finite-data fast path.
    frames[12:15, 0] = np.nan
    for index in range(90):
        buffer.append_landmarks(frames[index % 45], timestamp_ns=index + 1, preprocessor=pipeline)
    durations, emissions = [], 0
    for index in range(200):
        start = time.perf_counter()
        tensor = buffer.append_landmarks(
            frames[index % 45], timestamp_ns=index + 91, preprocessor=pipeline,
        )
        durations.append((time.perf_counter() - start) * 1000)
        if tensor is not None:
            emissions += 1
            assert tuple(tensor.shape) == (1, 45, 138)
    p95 = float(np.percentile(durations, 95))
    assert emissions > 0
    assert p95 < 15.0, f"live preprocessing/FIFO p95 was {p95:.3f} ms"
