"""Fast non-functional gates for the pure CPU preprocessing boundary."""

from __future__ import annotations

import gc
import time
import tracemalloc
from pathlib import Path

import numpy as np

from asl_stereo.capture import DropReason, SyncWatchdog
from asl_stereo.preprocessing import PreprocessingPipeline, SlidingWindowBuffer
from tests.fixtures.synthetic_landmarks import make_landmark_frame


def _push_frame(
    pipeline: PreprocessingPipeline,
    buffer: SlidingWindowBuffer,
    landmarks: np.ndarray,
    timestamp_ns: int,
):
    feature = pipeline.process_frame(landmarks, timestamp_ns=timestamp_ns)
    assert feature is not None
    return buffer.append(feature)


def test_preprocessing_window_p95_latency_is_below_30_ms() -> None:
    pipeline = PreprocessingPipeline()
    buffer = SlidingWindowBuffer(window_size=45, stride=8)
    landmarks = make_landmark_frame()

    # Warm the numerical path and leave the buffer one frame short of emission.
    for index in range(44):
        _push_frame(pipeline, buffer, landmarks, index + 1)

    emitted_latencies_ms: list[float] = []
    for iteration in range(100):
        sample = landmarks.copy()
        sample[:42, 1] += np.float32(iteration * 1e-4)
        started = time.perf_counter()
        window = _push_frame(pipeline, buffer, sample, 45 + iteration)
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        if window is not None:
            emitted_latencies_ms.append(elapsed_ms)

    assert len(emitted_latencies_ms) == 13
    p95_ms = float(np.percentile(emitted_latencies_ms, 95))
    assert p95_ms < 30.0, f"preprocessing/window p95 was {p95_ms:.3f} ms"


def test_memory_stabilizes_and_pipeline_creates_no_files(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    files_before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}

    pipeline = PreprocessingPipeline()
    buffer = SlidingWindowBuffer(window_size=45, stride=8)
    landmarks = make_landmark_frame()

    # Fill fixed-capacity structures before establishing the memory baseline.
    for index in range(60):
        _push_frame(pipeline, buffer, landmarks, index + 1)

    gc.collect()
    tracemalloc.start()
    try:
        baseline_bytes, _ = tracemalloc.get_traced_memory()

        for index in range(500):
            _push_frame(pipeline, buffer, landmarks, 61 + index)
        gc.collect()
        midpoint_bytes, _ = tracemalloc.get_traced_memory()

        for index in range(500, 1_000):
            _push_frame(pipeline, buffer, landmarks, 61 + index)
        gc.collect()
        final_bytes, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    tolerance_bytes = 5 * 1024 * 1024
    assert final_bytes - baseline_bytes < tolerance_bytes
    assert final_bytes - midpoint_bytes < tolerance_bytes

    files_after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}
    assert files_after == files_before == set()


def test_watchdog_purges_above_five_percent_without_resetting_pair_index() -> None:
    purge_calls: list[bool] = []
    watchdog = SyncWatchdog(
        window_size=150,
        drop_rate_threshold=0.05,
        purge_callback=lambda: purge_calls.append(True),
    )

    # Twelve drops out of 150 observations is exactly eight percent.
    for index in range(150):
        if index < 12:
            watchdog.record_frame(DropReason.HARDWARE_CAPTURE)
        else:
            watchdog.record_frame()
            watchdog.next_pair_index()

    assert watchdog.drop_rate == 0.08
    assert watchdog.should_purge() is True
    pair_index_before_purge = watchdog.pair_index
    assert pair_index_before_purge == 137

    preserved_index = watchdog.purge()
    assert purge_calls == [True]
    assert preserved_index == pair_index_before_purge
    assert watchdog.pair_index == pair_index_before_purge
    assert watchdog.next_pair_index() == pair_index_before_purge + 1
    assert watchdog.pair_index != 0

