import numpy as np

from asl_stereo.capture import Synchronizer
from asl_stereo.contracts import TimestampedFrame


def frame(camera_id: str, timestamp_ns: int, frame_index: int = 0) -> TimestampedFrame:
    return TimestampedFrame(
        camera_id=camera_id,
        frame_index=frame_index,
        timestamp_ns=timestamp_ns,
        frame_buffer=np.zeros((2, 2, 3), dtype=np.uint8),
        health_meta={},
    )


def test_matches_nearest_frames_within_40_ms() -> None:
    synchronizer = Synchronizer(tolerance_ns=40_000_000)
    assert synchronizer.add_front(frame("front", 1_000_000_000)) is None

    pair = synchronizer.add_side(frame("side", 1_039_000_000))

    assert pair is not None
    assert pair.front.timestamp_ns == 1_000_000_000
    assert pair.side.timestamp_ns == 1_039_000_000
    assert pair.delta_t_ns == 39_000_000
    assert pair.pair_index == 0
    assert synchronizer.queue_sizes == (0, 0)


def test_boundary_at_exactly_40_ms_is_accepted() -> None:
    synchronizer = Synchronizer(tolerance_ns=40_000_000)
    synchronizer.add_side(frame("side", 960_000_000))
    pair = synchronizer.add_front(frame("front", 1_000_000_000))
    assert pair is not None
    assert pair.delta_t_ns == -40_000_000


def test_rejects_frames_outside_40_ms_tolerance() -> None:
    synchronizer = Synchronizer(tolerance_ns=40_000_000)
    synchronizer.add_front(frame("front", 1_000_000_000))

    pair = synchronizer.add_side(frame("side", 1_040_000_001))

    assert pair is None
    assert synchronizer.sync_rejects == 1
    assert synchronizer.queued_timestamps == ((), (1_040_000_001,))


def test_successful_match_prunes_frames_older_than_matched_timestamp() -> None:
    synchronizer = Synchronizer(tolerance_ns=40_000_000)
    synchronizer.add_front(frame("front", 100_000_000, 0))
    synchronizer.add_front(frame("front", 110_000_000, 1))

    pair = synchronizer.add_side(frame("side", 109_000_000, 0))

    assert pair is not None
    assert pair.front.frame_index == 1
    assert pair.delta_t_ns == -1_000_000
    assert synchronizer.queue_sizes == (0, 0)
    assert synchronizer.sync_rejects == 1


def test_pair_index_progresses_monotonically_and_survives_purge() -> None:
    synchronizer = Synchronizer(tolerance_ns=40_000_000)
    observed_indices: list[int] = []

    for index in range(3):
        timestamp = 1_000_000_000 + index * 50_000_000
        synchronizer.add_front(frame("front", timestamp, index))
        pair = synchronizer.add_side(frame("side", timestamp + 1_000_000, index))
        assert pair is not None
        observed_indices.append(pair.pair_index)

    synchronizer.add_front(frame("front", 2_000_000_000, 3))
    synchronizer.purge()
    assert synchronizer.pair_index == 2

    synchronizer.add_front(frame("front", 2_100_000_000, 4))
    pair = synchronizer.add_side(frame("side", 2_101_000_000, 4))
    assert pair is not None
    observed_indices.append(pair.pair_index)
    assert observed_indices == [0, 1, 2, 3]
