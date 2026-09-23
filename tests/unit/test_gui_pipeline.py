"""Regression checks for the live GUI inference-to-ticker path."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from asl_stereo.contracts import LandmarkFrame, TimestampedFrame
from asl_stereo.models.inference import InferenceResult
from asl_stereo.preprocessing import PreprocessingPipeline, SlidingWindowBuffer
from tests.fixtures.synthetic_landmarks import make_landmark_frame
from ui import MainWindow, PipelineWorker, RuntimeSettings, SettingsDialog


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_gui_defaults_to_diagnostic_confidence_floor(app) -> None:
    settings = RuntimeSettings()
    dialog = SettingsDialog(settings)
    worker = PipelineWorker(settings)

    assert settings.confidence_threshold == pytest.approx(0.40)
    assert settings.front_camera_index == 1
    assert dialog.confidence_slider.minimum() <= 35
    worker.update_confidence_threshold(0.40)
    assert worker._confidence_threshold == pytest.approx(0.40)


def test_front_camera_discovery_accepts_camera_one_after_zero_fails(app) -> None:
    attempted: list[int] = []

    class FakeCamera:
        actual_fps = 60.0
        last_error = None

        def __init__(self, index: int, **kwargs) -> None:
            attempted.append(index)
            self.index = index
            self.is_running = False

        def start(self, *, timeout: float) -> None:
            if self.index == 0:
                raise RuntimeError("camera unavailable")
            self.is_running = True

        def get_frame(self, *, timeout: float = 0.0) -> TimestampedFrame | None:
            return TimestampedFrame(
                camera_id="front",
                frame_index=0,
                timestamp_ns=1,
                frame_buffer=np.zeros((2, 2, 3), dtype=np.uint8),
                health_meta={},
            )

        def stop(self) -> None:
            self.is_running = False

    worker = PipelineWorker(
        RuntimeSettings(front_camera_index=0), camera_factory=FakeCamera
    )
    chosen = None
    for index in dict.fromkeys((worker.settings.front_camera_index, 0, 1, 2)):
        try:
            chosen, frame = worker._open_camera(index, "front")
            break
        except RuntimeError:
            continue

    assert attempted == [0, 1]
    assert chosen is not None and chosen.index == 1
    assert frame.frame_buffer.shape == (2, 2, 3)
    chosen.stop()


def test_one_handed_live_window_emits_finite_tensor(app) -> None:
    preprocessor = PreprocessingPipeline()
    buffer = SlidingWindowBuffer()
    frame = make_landmark_frame()
    frame[21:42] = np.nan
    window = None
    for index in range(45):
        window = buffer.append_landmarks(
            frame, timestamp_ns=index + 1, preprocessor=preprocessor
        )

    assert window is not None
    assert tuple(window.shape) == (1, 45, 138)
    assert bool(window.isfinite().all())
    assert bool((window[:, :, 63:126] == 0).all())


def test_active_hand_with_long_gap_is_not_imputed(app) -> None:
    raw = np.repeat(make_landmark_frame()[None, :, :], 45, axis=0)
    raw[10:15, 0:21, :] = np.nan

    cleaned = PreprocessingPipeline().process_live_window(raw)

    assert np.isnan(cleaned[10:15, 0:63]).all()


def test_gui_threshold_accepts_and_emits_confident_prediction(app) -> None:
    class FakeEngine:
        class_map = {0: "HELLO", 1: "NO", 2: "YES"}

        def predict(self, tensor) -> InferenceResult:
            return InferenceResult(
                predicted_gloss="HELLO",
                class_index=0,
                confidence_score=0.52,
                probabilities=(0.52, 0.30, 0.18),
                latency_ms=3.0,
            )

    worker = PipelineWorker(RuntimeSettings(), inference_engine=FakeEngine())
    received: list[tuple[str, float, float]] = []
    worker.prediction_ready.connect(lambda *args: received.append(args))

    confidence = worker._run_inference(object(), SlidingWindowBuffer())
    app.processEvents()

    assert confidence == pytest.approx(0.52)
    assert received == [("HELLO", 0.52, 3.0)]


def test_prediction_signal_reaches_visible_ticker(app, monkeypatch) -> None:
    worker = PipelineWorker(RuntimeSettings())
    monkeypatch.setattr(worker, "start", lambda: None)
    window = MainWindow(worker_factory=lambda settings: worker)
    window.start_pipeline()

    worker.prediction_ready.emit("HELLO", 0.52, 3.0)
    app.processEvents()

    assert window.ticker.text == "HELLO"
    assert "Last detected: HELLO" in window.prediction_detail.text()
    assert window.ticker.text_display.styleSheet().find("#FFFFFF") >= 0
    window.close()


def test_single_camera_frames_reach_ticker_through_buffer_and_inference(
    app, monkeypatch
) -> None:
    class FakeEngine:
        class_map = {0: "HELLO", 1: "NO", 2: "YES"}

        def predict(self, tensor) -> InferenceResult:
            assert tuple(tensor.shape) == (1, 45, 138)
            return InferenceResult(
                predicted_gloss="HELLO",
                class_index=0,
                confidence_score=0.52,
                probabilities=(0.52, 0.30, 0.18),
                latency_ms=3.0,
            )

    class FakeExtractor:
        last_results = None

        def process(self, frame, *, timestamp_ns, frame_index) -> LandmarkFrame:
            coordinates = make_landmark_frame()
            coordinates[21:42] = np.nan
            return LandmarkFrame(
                coordinates=coordinates,
                hand_presence=np.array([1.0, 0.0], dtype=np.float32),
                joint_mask=np.isfinite(coordinates).all(axis=1).astype(np.float32),
                timestamp_ns=timestamp_ns,
                frame_index=frame_index,
            )

    worker = PipelineWorker(RuntimeSettings(), inference_engine=FakeEngine())
    monkeypatch.setattr(worker, "start", lambda: None)
    window = MainWindow(worker_factory=lambda settings: worker)
    window.start_pipeline()
    extractor = FakeExtractor()
    preprocessor = PreprocessingPipeline()
    buffer = SlidingWindowBuffer()

    for index in range(45):
        frame = TimestampedFrame(
            camera_id="front",
            frame_index=index,
            timestamp_ns=index + 1,
            frame_buffer=np.zeros((100, 100, 3), dtype=np.uint8),
            health_meta={},
        )
        worker._process_single_frame(
            frame, extractor, preprocessor, buffer, index, 1.0
        )
    app.processEvents()

    assert window.ticker.text == "HELLO"
    assert "Last detected: HELLO" in window.prediction_detail.text()
    window.close()
