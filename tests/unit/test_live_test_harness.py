from __future__ import annotations

import time
from contextlib import ExitStack
from types import SimpleNamespace

import cv2
import numpy as np

from asl_stereo.capture import CameraWorker
from asl_stereo.contracts import LandmarkFrame, TimestampedFrame
from asl_stereo.models import InferenceResult
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
            self.reads = 0
            self.closed = False

        def isOpened(self) -> bool:
            return True

        def set(self, property_id: int, value: float) -> bool:
            self.properties[property_id] = value
            return True

        def get(self, property_id: int) -> float:
            return 30.0 if property_id == cv2.CAP_PROP_FPS else 0.0

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
        assert capture.properties[cv2.CAP_PROP_FPS] == 60.0
        assert capture.properties[cv2.CAP_PROP_FRAME_WIDTH] == 1280.0
        assert capture.properties[cv2.CAP_PROP_FRAME_HEIGHT] == 720.0
        assert worker.actual_fps == 30.0
    finally:
        worker.stop()
    assert capture.closed


def test_camera_discovery_falls_back_to_index_zero(capsys) -> None:
    attempted: list[int] = []

    class FakeWorker:
        actual_fps = 30.0
        is_running = True
        last_error = None

        def __init__(self, index: int, **kwargs) -> None:
            self.index = index
            attempted.append(index)
            assert kwargs["target_fps"] == 60.0

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
    assert "Hardware capped at 30.0 FPS" in output


def test_single_camera_loop_stays_active_until_q_and_runs_inference(
    monkeypatch, capsys
) -> None:
    class FakeCamera:
        is_running = True
        last_error = None

        def __init__(self) -> None:
            self.next_index = 1
            self.stopped = False

        def get_frame(self, *, timeout: float):
            record = _record(self.next_index)
            self.next_index += 1
            return record

        def stop(self) -> None:
            self.stopped = True

    class FakeExtractor:
        last_results = None

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            pass

        def process(self, frame, *, timestamp_ns: int, frame_index: int):
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
        return ord("q") if key_calls == 45 else -1

    monkeypatch.setattr(harness, "_load_engine", lambda *args: engine)
    monkeypatch.setattr(harness, "_open_camera", fake_open)
    monkeypatch.setattr(harness, "HolisticExtractor", FakeExtractor)
    monkeypatch.setattr(cv2, "imshow", lambda title, frame: displayed.append(frame))
    monkeypatch.setattr(cv2, "waitKey", fake_wait_key)

    harness.run(True, 0, 1)

    assert camera.stopped
    assert len(displayed) == 45
    assert engine.predictions == 1
    assert "Prediction: HELLO" in capsys.readouterr().out
