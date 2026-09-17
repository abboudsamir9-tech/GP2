import numpy as np
import pytest
import torch

from asl_stereo.preprocessing import (
    PreprocessingPipeline,
    SlidingWindowBuffer,
    TemporalBuffer,
)
from tests.fixtures.synthetic_landmarks import make_landmark_frame


def feature(value: float) -> np.ndarray:
    return np.full(138, value, dtype=np.float32)


def test_first_window_emits_at_window_size_with_expected_contract() -> None:
    buffer = TemporalBuffer(window_size=30, stride=8)
    for index in range(29):
        assert buffer.append(feature(index), timestamp_ns=index + 1) is None
    window = buffer.append(feature(29), timestamp_ns=30)
    assert window is not None
    assert window.values.shape == (1, 30, 138)
    assert window.values.dtype == np.float32
    assert window.values.flags.c_contiguous
    assert window.start_timestamp_ns == 1
    assert window.end_timestamp_ns == 30


def test_next_window_emits_after_eight_additional_valid_frames() -> None:
    buffer = TemporalBuffer(window_size=30, stride=8)
    first = None
    for index in range(30):
        first = buffer.append(feature(index), timestamp_ns=index + 1)
    assert first is not None
    for index in range(30, 37):
        assert buffer.append(feature(index), timestamp_ns=index + 1) is None
    second = buffer.append(feature(37), timestamp_ns=38)
    assert second is not None
    assert second.start_timestamp_ns == 9
    assert second.end_timestamp_ns == 38


def test_invalid_frames_do_not_advance_buffer_or_stride() -> None:
    buffer = TemporalBuffer(window_size=30, stride=8)
    for index in range(29):
        buffer.append(feature(index), timestamp_ns=index + 1)
    invalid = feature(99)
    invalid[10] = np.nan
    assert buffer.append(invalid, timestamp_ns=30) is None
    assert len(buffer) == 29
    window = buffer.append(feature(29), timestamp_ns=31)
    assert window is not None
    assert window.end_timestamp_ns == 31


def test_emitted_window_is_stable_after_future_appends() -> None:
    buffer = TemporalBuffer(window_size=30, stride=8)
    window = None
    for index in range(30):
        window = buffer.append(feature(index), timestamp_ns=index + 1)
    assert window is not None
    snapshot = window.values.copy()
    for index in range(30, 38):
        buffer.append(feature(index), timestamp_ns=index + 1)
    np.testing.assert_array_equal(window.values, snapshot)


@pytest.mark.parametrize("window_size", [29, 61])
def test_window_size_range_is_enforced(window_size: int) -> None:
    with pytest.raises(ValueError):
        TemporalBuffer(window_size=window_size)


def test_sliding_buffer_emits_strict_default_tensor_shape() -> None:
    buffer = SlidingWindowBuffer()
    tensor = None
    for valid_index in range(1, 46):
        tensor = buffer.append(feature(valid_index), timestamp_ns=valid_index)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 45, 138)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"


def test_sliding_buffer_cadence_uses_only_valid_frame_indices() -> None:
    buffer = SlidingWindowBuffer(window_size=45, stride=8)
    emitted_at: list[int] = []
    timestamp = 0
    for valid_index in range(1, 62):
        timestamp += 1
        if valid_index in (46, 50, 58):
            invalid = feature(-1)
            invalid[0] = np.nan
            assert buffer.append(invalid, timestamp_ns=timestamp) is None
            timestamp += 1
        if buffer.append(feature(valid_index), timestamp_ns=timestamp) is not None:
            emitted_at.append(valid_index)
    assert emitted_at == [45, 53, 61]


def test_sliding_buffer_reset_flushes_frames_and_cadence() -> None:
    buffer = SlidingWindowBuffer()
    for index in range(45):
        output = buffer.append(feature(index), timestamp_ns=index + 1)
    assert output is not None
    buffer.reset()
    assert len(buffer) == 0
    assert buffer.last_window_start_timestamp_ns is None
    for index in range(44):
        assert buffer.append(feature(index), timestamp_ns=100 + index) is None


def test_tensor_and_numpy_window_share_storage() -> None:
    buffer = SlidingWindowBuffer()
    tensor = None
    for index in range(45):
        tensor = buffer.append(feature(index), timestamp_ns=index + 1)
    assert tensor is not None
    numpy_view = tensor.numpy()
    numpy_view[0, 0, 0] = 123.0
    assert tensor[0, 0, 0].item() == 123.0


def test_live_landmark_window_uses_offline_temporal_cleaning() -> None:
    buffer = SlidingWindowBuffer(window_size=45, stride=8)
    pipeline = PreprocessingPipeline()
    landmarks = make_landmark_frame()
    emitted = None
    for index in range(45):
        frame = landmarks.copy()
        if index == 20:
            frame[0] = np.nan
        emitted = buffer.append_landmarks(
            frame, timestamp_ns=index + 1, preprocessor=pipeline
        )

    assert emitted is not None
    assert emitted.shape == (1, 45, 138)
    assert torch.isfinite(emitted).all()
