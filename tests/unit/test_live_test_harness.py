from __future__ import annotations

import time
import threading
from contextlib import ExitStack
from types import SimpleNamespace

import cv2
import numpy as np

from asl_stereo.capture import CameraWorker
from asl_stereo.contracts import LandmarkFrame, TimestampedFrame
from asl_stereo.models import InferenceResult
from asl_stereo.preprocessing import SlidingWindowBuffer
from scripts import live_test_harness as harness


def _record(index: int) -> TimestampedFrame:
    return TimestampedFrame(
        camera_id="front",
        frame_index=index,
        timestamp_ns=1 + index * 16_000_000,
        frame_buffer=np.zeros((48, 64, 3), dtype=np.uint8),
        health_meta={},
    )


def test_camera_worker_requests_fps_and_retries_empty_warmup() -> None:
    class Capture:
        def __init__(self) -> None:
            self.properties: dict[int, float] = {}
            self.set_order: list[int] = []
            self.reads = 0
            self.closed = False

        def isOpened(self) -> bool:
            return True

        def set(self, property_id: int, value: float) -> bool:
            self.set_order.append(property_id)
            self.properties[property_id] = value
            return True

        def get(self, property_id: int) -> float:
            return 30.0 if property_id == cv2.CAP_PROP_FPS else self.properties.get(property_id, 0.0)

        def read(self):
            self.reads += 1
            time.sleep(0.001)
            if self.reads == 1:
                return True, None
            return True, np.zeros((48, 64, 3), dtype=np.uint8)

        def release(self) -> None:
            self.closed = True

    capture = Capture()
    worker = CameraWorker(
        0, camera_id="front", warmup_frames=1,
        resolution=(1280, 720), target_fps=60.0,
        capture_factory=lambda _: capture,
    )
    try:
        worker.start(timeout=1.0)
        frame = worker.get_frame(timeout=1.0)
        assert frame is not None and frame.frame_buffer.size > 0
        assert capture.reads >= 3
        assert capture.properties[cv2.CAP_PROP_FPS] == 30.0
        assert capture.set_order.index(cv2.CAP_PROP_FOURCC) < capture.set_order.index(cv2.CAP_PROP_FPS)
        assert capture.set_order.index(cv2.CAP_PROP_FPS) < capture.set_order.index(cv2.CAP_PROP_FRAME_WIDTH)
        assert capture.set_order.index(cv2.CAP_PROP_FRAME_HEIGHT) < capture.set_order.index(cv2.CAP_PROP_BUFFERSIZE)
        assert int(capture.properties[cv2.CAP_PROP_FOURCC]) == cv2.VideoWriter_fourcc(*"MJPG")
        assert capture.properties[cv2.CAP_PROP_FRAME_WIDTH] == 1280.0
        assert capture.properties[cv2.CAP_PROP_FRAME_HEIGHT] == 720.0
        assert worker.actual_fps == 30.0
        assert worker.peek_latest_frame() is not None
        deadline = time.perf_counter() + 1.0
        while worker.capture_fps == 0.0 and time.perf_counter() < deadline:
            time.sleep(0.005)
        assert worker.capture_fps > 0.0
        assert worker.capture_read_ms > 0.0
    finally:
        worker.stop()
    assert capture.closed


def test_camera_capture_keeps_latest_frame_without_consumer() -> None:
    class FastCapture:
        def __init__(self) -> None:
            self.closed = False

        def isOpened(self) -> bool:
            return True

        def set(self, property_id: int, value: float) -> bool:
            return True

        def get(self, property_id: int) -> float:
            return 30.0 if property_id == cv2.CAP_PROP_FPS else 0.0

        def read(self):
            time.sleep(0.001)
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self) -> None:
            self.closed = True

    capture = FastCapture()
    worker = CameraWorker(
        0, camera_id="front", warmup_frames=1, target_fps=60.0,
        capture_factory=lambda _: capture,
    )
    try:
        worker.start(timeout=1.0)
        time.sleep(0.04)
        assert worker.frame_index > 5
        assert worker.backpressure_drops > 0
        assert worker._frames.maxsize == 1
        latest = worker.peek_latest_frame()
        queued = worker.get_frame()
        assert latest is not None and queued is not None
        assert queued.frame_index == latest.frame_index
    finally:
        worker.stop()
    assert capture.closed


def test_first_discarded_frame_unblocks_start_before_full_warmup() -> None:
    class SlowCapture:
        def __init__(self) -> None:
            self.released = False

        def isOpened(self) -> bool:
            return not self.released

        def getBackendName(self) -> str:
            return "DSHOW"

        def set(self, property_id: int, value: float) -> bool:
            return True

        def get(self, property_id: int) -> float:
            return 30.0 if property_id == cv2.CAP_PROP_FPS else 0.0

        def read(self):
            time.sleep(0.07)
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    capture = SlowCapture()
    worker = CameraWorker(
        0, camera_id="front", warmup_frames=10, target_fps=30.0,
        capture_factory=lambda _: capture,
    )
    try:
        started = time.perf_counter()
        worker.start(timeout=0.25)
        assert time.perf_counter() - started < 0.25
        assert worker.get_frame(timeout=2.0) is not None
    finally:
        worker.stop()
    assert capture.released


def test_slow_but_open_directshow_never_probes_msmf() -> None:
    class FakeCapture:
        def __init__(self, backend: str, delay: float, opened: bool = True) -> None:
            self.backend = backend
            self.delay = delay
            self.opened = opened
            self.released = False
            self.read_threads: list[str] = []
            self.properties: dict[int, float] = {}

        def isOpened(self) -> bool:
            return self.opened and not self.released

        def getBackendName(self) -> str:
            return self.backend

        def set(self, property_id: int, value: float) -> bool:
            self.properties[property_id] = value
            return True

        def get(self, property_id: int) -> float:
            return self.properties.get(property_id, 0.0)

        def read(self):
            self.read_threads.append(threading.current_thread().name)
            time.sleep(self.delay)
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    dshow = FakeCapture("DSHOW", 0.05)
    msmf = FakeCapture("MSMF", 0.005)

    def open_msmf(_):
        raise AssertionError("MSMF must not be probed after DSHOW opens")

    worker = CameraWorker(
        0, camera_id="front", warmup_frames=5, target_fps=30.0,
        capture_factory=lambda _: dshow,
        msmf_capture_factory=open_msmf,
    )
    try:
        worker.start(timeout=2.0)
        first = worker.get_frame(timeout=1.0)
        assert first is not None
        deadline = time.perf_counter() + 2.0
        while worker.frame_index <= first.frame_index and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert worker.backend_name == "DSHOW"
        assert worker.frame_index > first.frame_index
        assert not dshow.released
        assert all(name == "camera-worker-front" for name in dshow.read_threads)
    finally:
        worker.stop()
    assert dshow.released
    assert not msmf.released


def test_unopened_directshow_releases_handle_before_msmf_fallback() -> None:
    class FakeCapture:
        def __init__(self, opened: bool, backend: str) -> None:
            self.opened = opened
            self.backend = backend
            self.released = False
            self.properties: dict[int, float] = {}

        def isOpened(self) -> bool:
            return self.opened and not self.released

        def getBackendName(self) -> str:
            return self.backend

        def set(self, property_id: int, value: float) -> bool:
            self.properties[property_id] = value
            return True

        def get(self, property_id: int) -> float:
            return self.properties.get(property_id, 0.0)

        def read(self):
            time.sleep(0.04)
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    dshow = FakeCapture(False, "DSHOW")
    msmf = FakeCapture(True, "MSMF")

    def open_msmf(_):
        assert dshow.released
        return msmf

    worker = CameraWorker(
        0, camera_id="front", warmup_frames=5, target_fps=30.0,
        capture_factory=lambda _: dshow,
        msmf_capture_factory=open_msmf,
    )
    try:
        worker.start(timeout=2.0)
        assert worker.get_frame(timeout=1.0) is not None
        time.sleep(0.15)
        assert worker.is_running
        assert worker.backend_name == "MSMF"
        assert dshow.released
        assert not msmf.released
    finally:
        worker.stop()
    assert msmf.released


def test_camera_discovery_falls_back_to_index_zero(capsys) -> None:
    attempted: list[int] = []

    class FakeWorker:
        actual_fps = 15.0
        is_running = True
        last_error = None

        def __init__(self, index: int, **kwargs) -> None:
            self.index = index
            attempted.append(index)
            assert kwargs["target_fps"] == 30.0

        def start(self, *, timeout: float) -> None:
            if self.index == 1:
                raise RuntimeError("inactive device")

        def get_frame(self, *, timeout: float):
            return _record(0)

        def stop(self) -> None:
            pass

    with ExitStack() as stack:
        _, index, first = harness._open_camera(
            stack, 1, camera_id="front", camera_factory=FakeWorker
        )
    assert attempted == [1, 0]
    assert index == 0 and first.frame_buffer.size > 0
    output = capsys.readouterr().out
    assert "Camera index 1 unavailable" in output
    assert "Hardware capped at 15.0 FPS" in output


def test_missing_calibration_runs_harness_without_side_camera(
    monkeypatch, tmp_path, capsys
) -> None:
    opened: list[str] = []

    def fake_open(stack, preferred, *, camera_id, **kwargs):
        opened.append(camera_id)
        return object(), preferred, _record(0)

    class FakeVision:
        def __init__(self, front, side, *args) -> None:
            assert side is None

        def start(self) -> None:
            pass

        def join(self, *, timeout: float) -> None:
            pass

    monkeypatch.setattr(harness, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(harness, "_load_engine", lambda *args: object())
    monkeypatch.setattr(harness, "_open_camera", fake_open)
    monkeypatch.setattr(harness, "ExtractionWorker", FakeVision)
    monkeypatch.setattr(harness, "_display_loop", lambda *args: None)

    harness.run(False, 0, 1)

    assert opened == ["front"]
    assert "Stereo calibration missing" in capsys.readouterr().out


def test_single_camera_loop_stays_active_until_q_and_runs_inference(
    monkeypatch, capsys
) -> None:
    extractor_threads: list[str] = []

    class FakeCamera:
        last_error = None
        capture_fps = 60.0

        def __init__(self) -> None:
            self.next_index = 1
            self.stopped = False
            self.is_running = True
            self._lock = threading.Lock()
            self._latest = _record(0)

        def get_frame(self, *, timeout: float):
            time.sleep(0.001)
            with self._lock:
                record = _record(self.next_index)
                self.next_index += 1
                self._latest = record
                return record

        def peek_latest_frame(self):
            with self._lock:
                return self._latest

        def stop(self) -> None:
            self.stopped = True
            self.is_running = False

    class FakeExtractor:
        last_results = None

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            pass

        def process(self, frame, *, timestamp_ns: int, frame_index: int):
            extractor_threads.append(threading.current_thread().name)
            coordinates = np.full((46, 3), np.nan, dtype=np.float32)
            coordinates[:21] = (0.1, 0.2, 0.0)
            coordinates[42:46] = np.array(
                [(-0.5, 0, 0), (0.5, 0, 0), (-0.5, 0.5, 0), (0.5, 0.5, 0)],
                dtype=np.float32,
            )
            return LandmarkFrame(
                coordinates=coordinates,
                hand_presence=np.array([1.0, 0.0], dtype=np.float32),
                joint_mask=np.isfinite(coordinates).all(axis=1).astype(np.float32),
                timestamp_ns=timestamp_ns,
                frame_index=frame_index,
            )

    class FakeEngine:
        predictions = 0

        def predict(self, window):
            assert tuple(window.shape) == (1, 45, 138)
            self.predictions += 1
            return InferenceResult("HELLO", 0, 0.8, (0.8, 0.2), 1.0)

        def gate_prediction(self, *args, **kwargs):
            return SimpleNamespace(accepted=True)

    camera = FakeCamera()
    engine = FakeEngine()
    displayed: list[np.ndarray] = []
    key_calls = 0

    def fake_open(stack, preferred, **kwargs):
        stack.callback(camera.stop)
        return camera, 0, _record(0)

    def fake_wait_key(delay):
        nonlocal key_calls
        key_calls += 1
        return ord("q") if key_calls >= 45 and engine.predictions >= 1 else -1

    monkeypatch.setattr(harness, "_load_engine", lambda *args: engine)
    monkeypatch.setattr(harness, "_open_camera", fake_open)
    monkeypatch.setattr(harness, "PoseHandsExtractor", FakeExtractor)
    monkeypatch.setattr(cv2, "imshow", lambda title, frame: displayed.append(frame))
    monkeypatch.setattr(cv2, "waitKey", fake_wait_key)

    harness.run(True, 0, 1)

    assert camera.stopped
    assert len(displayed) > 0
    assert engine.predictions >= 1
    assert set(extractor_threads) == {"asl-live-extraction"}
    assert harness.INFERENCE_STRIDE == 6
    assert "Prediction: HELLO" in capsys.readouterr().out


def test_profile_reports_slow_stages_every_thirty_frames(capsys) -> None:
    profiler = harness.ProfileAccumulator()
    timings = SimpleNamespace(resize_ms=1.0, pose_ms=21.0, hands_ms=10.0)
    for index in range(30):
        profiler.add(
            capture_ms=20.0,
            extractor_timings=(timings,),
            norm_buffer_ms=0.5,
            total_ms=24.0,
            measured_fps=41.0,
        )
        if index < 29:
            assert capsys.readouterr().out == ""
    output = capsys.readouterr().out
    assert "[PROFILE] Capture: 20.0 ms | Resize: 1.0 ms" in output
    assert "Pose: 21.0 ms | Hands: 10.0 ms" in output
    assert "Measured FPS: 41.0" in output
    assert "Operations over 15 ms: capture, pose" in output


def test_live_inference_uses_six_valid_frame_stride(capsys) -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.predictions = 0

        def predict(self, window):
            self.predictions += 1
            assert tuple(window.shape) == (1, 45, 138)
            return InferenceResult("HELLO", 0, 0.8, (0.8, 0.2), 1.0)

        def gate_prediction(self, *args, **kwargs):
            return SimpleNamespace(accepted=True)

    coordinates = np.zeros((46, 3), dtype=np.float32)
    coordinates[:21] = (0.1, 0.2, 0.0)
    coordinates[42:46] = np.array(
        [(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0),
         (-0.5, 0.5, 0.0), (0.5, 0.5, 0.0)], dtype=np.float32,
    )
    buffer = SlidingWindowBuffer(window_size=45, stride=harness.INFERENCE_STRIDE)
    preprocessor = harness.LiveWindowPreprocessor()
    engine = FakeEngine()
    state = harness.LiveState()
    for frame_index in range(60):
        landmarks = LandmarkFrame(
            coordinates=coordinates,
            hand_presence=np.array([1.0, 0.0], dtype=np.float32),
            joint_mask=np.ones(46, dtype=np.float32),
            timestamp_ns=frame_index + 1,
            frame_index=frame_index,
        )
        harness._report_prediction(
            landmarks, coordinates, fps=30.0, frame_count=frame_index,
            buffer=buffer, preprocessor=preprocessor, engine=engine,
            confidence_threshold=0.4, state=state,
        )
    assert engine.predictions == 3  # first full window, then +6 and +12 frames
    capsys.readouterr()
