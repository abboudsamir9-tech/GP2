"""PyQt5 dashboard and background end-to-end pipeline worker."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QCloseEvent, QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from asl_stereo.capture import CameraWorker, DropReason, Synchronizer, SyncWatchdog
from asl_stereo.contracts import TimestampedFrame
from asl_stereo.landmarks import HolisticExtractor, draw_asl_overlay
from asl_stereo.models import InferenceEngine, SignSequenceClassifier
from asl_stereo.models.checkpoint import load_class_map
from asl_stereo.preprocessing import PreprocessingPipeline, SlidingWindowBuffer
from asl_stereo.stereo import JointStatus, StereoCalibration, StereoMatcher
from asl_stereo.translation.confidence_filter import ConfidenceFilter

from .qt_messages import map_telemetry, prediction_text
from .text_ticker import TranslationTicker

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMERA_INDICES = (0, 1, 2)
CAMERA_RESOLUTION = (1280, 720)
CAMERA_FPS = 60.0
CAMERA_TIMEOUT_S = 5.0
GUI_MIN_CONFIDENCE = 0.35


@dataclass(slots=True)
class RuntimeSettings:
    front_camera_index: int = 1
    side_camera_index: int = 0
    confidence_threshold: float = 0.40


class PipelineWorker(QThread):
    """Own and run every blocking or compute-heavy pipeline component."""

    frames_ready = pyqtSignal(np.ndarray, np.ndarray)
    prediction_ready = pyqtSignal(str, float, float)
    telemetry_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        inference_engine: InferenceEngine | None = None,
        calibration: StereoCalibration | None = None,
        camera_factory: Callable[..., CameraWorker] = CameraWorker,
        extractor_factory: Callable[..., HolisticExtractor] = HolisticExtractor,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = replace(settings)
        self.inference_engine = inference_engine
        self.calibration = calibration or StereoCalibration.from_default()
        self._camera_factory = camera_factory
        self._extractor_factory = extractor_factory
        self._stop_event = threading.Event()
        self._resource_lock = threading.RLock()
        self._front_camera: CameraWorker | None = None
        self._side_camera: CameraWorker | None = None
        self._threshold_lock = threading.Lock()
        self._confidence_threshold = max(settings.confidence_threshold, GUI_MIN_CONFIDENCE)
        self._buffer_full_logged = False
        self._invalid_window_count = 0
        self._frame_mailbox_lock = threading.Lock()
        self._latest_frames: tuple[np.ndarray, np.ndarray] | None = None
        self._frame_signal_pending = False

    def update_confidence_threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence threshold must be in [0, 1]")
        with self._threshold_lock:
            self._confidence_threshold = max(value, GUI_MIN_CONFIDENCE)

    def consume_latest_frames(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Consume the coalesced newest UI frame pair and acknowledge delivery."""
        with self._frame_mailbox_lock:
            frames = self._latest_frames
            self._latest_frames = None
            self._frame_signal_pending = False
            return frames

    def _publish_frames(self, front: np.ndarray, side: np.ndarray) -> None:
        """Keep at most one pending Qt frame event and one latest frame pair."""
        should_emit = False
        with self._frame_mailbox_lock:
            self._latest_frames = (front, side)
            if not self._frame_signal_pending:
                self._frame_signal_pending = True
                should_emit = True
        if should_emit:
            self.frames_ready.emit(front, side)

    @pyqtSlot()
    def stop(self) -> None:
        """Request non-blocking shutdown from any thread."""
        self._stop_event.set()
        self.requestInterruption()
        with self._resource_lock:
            cameras = (self._front_camera, self._side_camera)
        for camera in cameras:
            if camera is not None:
                camera.request_stop()

    def _open_camera(
        self, index: int, camera_id: str
    ) -> tuple[CameraWorker, TimestampedFrame]:
        camera = self._camera_factory(
            index,
            camera_id=camera_id,
            resolution=CAMERA_RESOLUTION,
            target_fps=CAMERA_FPS,
        )
        try:
            camera.start(timeout=CAMERA_TIMEOUT_S)
            first_frame = camera.get_frame(timeout=1.5)
            if (
                first_frame is None
                or first_frame.frame_buffer.size == 0
                or first_frame.frame_buffer.ndim != 3
                or first_frame.frame_buffer.shape[2] != 3
            ):
                raise RuntimeError("no non-empty BGR frame after warmup")
            if not camera.is_running:
                raise RuntimeError(f"capture stopped: {camera.last_error!r}")
            achieved_fps = camera.actual_fps
            height, width = first_frame.frame_buffer.shape[:2]
            print(
                f"[GUI INFO] Opened {camera_id} camera at index {index} "
                f"({width}x{height}, reported {achieved_fps or 0.0:.1f} FPS)",
                flush=True,
            )
            return camera, first_frame
        except Exception:
            try:
                camera.stop()
            except Exception:
                LOGGER.debug("Camera cleanup failed", exc_info=True)
            raise

    def _load_default_engine(self) -> None:
        if self.inference_engine is not None:
            return
        class_map = load_class_map(PROJECT_ROOT / "configs" / "class_map.json")
        failures: list[str] = []
        for artifact in (
            PROJECT_ROOT / "weights" / "optimized_model.pt",
            PROJECT_ROOT / "weights" / "best_model.pth",
        ):
            if not artifact.is_file():
                failures.append(f"{artifact}: missing")
                continue
            try:
                self.inference_engine = InferenceEngine(
                    SignSequenceClassifier(num_classes=len(class_map)),
                    class_map,
                    checkpoint_path=artifact,
                )
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                failures.append(f"{artifact}: {error}")
                LOGGER.warning("Model artifact unavailable: %s (%s)", artifact, error)
                continue
            print(f"[GUI INFO] Loaded model: {artifact}", flush=True)
            return
        raise RuntimeError("No usable model artifact. " + "; ".join(failures))

    def run(self) -> None:
        front_camera: CameraWorker | None = None
        side_camera: CameraWorker | None = None
        front_extractor: HolisticExtractor | None = None
        side_extractor: HolisticExtractor | None = None
        try:
            self._load_default_engine()
            synchronizer = Synchronizer(front_camera_id="front", side_camera_id="side")
            watchdog = SyncWatchdog(purge_callback=synchronizer.purge)
            matcher = StereoMatcher(self.calibration, fallback_scope="frame")
            preprocessor = PreprocessingPipeline()
            window_buffer = SlidingWindowBuffer()

            candidates = list(dict.fromkeys((self.settings.front_camera_index, *CAMERA_INDICES)))
            camera_failures: list[str] = []
            first_front = None
            selected_front = None
            for index in candidates:
                if self._stop_event.is_set():
                    return
                try:
                    front_camera, first_front = self._open_camera(index, "front")
                except Exception as error:
                    reason = error.__cause__ or error
                    camera_failures.append(f"index {index}: {reason}")
                    print(f"[GUI WARNING] Front camera index {index}: {reason}", flush=True)
                    continue
                selected_front = index
                break
            if front_camera is None:
                raise RuntimeError("No working front camera. " + "; ".join(camera_failures))
            with self._resource_lock:
                self._front_camera = front_camera
            front_extractor = self._extractor_factory()

            first_side = None
            if self.settings.side_camera_index != selected_front:
                try:
                    side_camera, first_side = self._open_camera(
                        self.settings.side_camera_index, "side"
                    )
                    with self._resource_lock:
                        self._side_camera = side_camera
                    side_extractor = self._extractor_factory()
                except Exception as error:
                    LOGGER.warning("Side camera unavailable: %s", error)
                    print(f"[GUI WARNING] Side camera unavailable: {error}", flush=True)
                    if side_camera is not None:
                        try:
                            side_camera.stop()
                        except Exception:
                            LOGGER.debug("Side-camera cleanup failed", exc_info=True)
                        side_camera = None
            if side_camera is None:
                print("[GUI INFO] [SINGLE-CAMERA FALLBACK]", flush=True)
                with self._resource_lock:
                    self._side_camera = None

            frame_count = 0
            fps_started = time.perf_counter()
            last_front_time = time.monotonic()
            last_pair_time = last_front_time
            previous_sync_rejects = 0
            while not self._stop_event.is_set() and not self.isInterruptionRequested():
                if not front_camera.is_running:
                    raise RuntimeError(f"front camera stopped: {front_camera.last_error!r}")
                if side_camera is not None and not side_camera.is_running:
                    LOGGER.warning(
                        "Side camera stopped; entering single-camera fallback"
                    )
                    try:
                        side_camera.stop()
                    except Exception:
                        LOGGER.debug("Side-camera cleanup failed", exc_info=True)
                    side_camera = None
                    if side_extractor is not None:
                        side_extractor.close()
                        side_extractor = None
                    synchronizer.purge()
                    window_buffer.reset()
                    self._buffer_full_logged = False
                    self._invalid_window_count = 0
                    with self._resource_lock:
                        self._side_camera = None

                if side_camera is None:
                    front_frame = first_front or front_camera.get_frame()
                    first_front = None
                    if front_frame is None:
                        if time.monotonic() - last_front_time > CAMERA_TIMEOUT_S:
                            raise RuntimeError("front camera stopped delivering frames")
                        self.msleep(1)
                        continue
                    last_front_time = time.monotonic()
                    self._process_single_frame(
                        front_frame,
                        front_extractor,
                        preprocessor,
                        window_buffer,
                        frame_count,
                        fps_started,
                    )
                    frame_count += 1
                    continue

                pairs = []
                front_frame = first_front or front_camera.get_frame()
                first_front = None
                if front_frame is not None:
                    last_front_time = time.monotonic()
                    pair = synchronizer.add_front(front_frame)
                    if pair is not None:
                        pairs.append(pair)
                side_frame = first_side or side_camera.get_frame()
                first_side = None
                if side_frame is not None:
                    pair = synchronizer.add_side(side_frame)
                    if pair is not None:
                        pairs.append(pair)

                new_rejects = synchronizer.sync_rejects - previous_sync_rejects
                for _ in range(max(0, new_rejects)):
                    watchdog.record_frame(DropReason.SYNC_REJECT)
                previous_sync_rejects = synchronizer.sync_rejects

                for pair in pairs:
                    last_pair_time = time.monotonic()
                    watchdog.record_frame()
                    watchdog.next_pair_index()
                    self._process_pair(
                        pair,
                        front_extractor,
                        side_extractor,
                        matcher,
                        preprocessor,
                        window_buffer,
                        watchdog,
                        frame_count,
                        fps_started,
                    )
                    frame_count += 1

                if watchdog.should_purge():
                    watchdog.purge()
                    window_buffer.reset()
                    self._buffer_full_logged = False
                    self._invalid_window_count = 0
                    previous_sync_rejects = synchronizer.sync_rejects
                if not pairs:
                    if time.monotonic() - last_front_time > CAMERA_TIMEOUT_S:
                        raise RuntimeError("front camera stopped delivering frames")
                    if time.monotonic() - last_pair_time > 3.0:
                        print(
                            "[GUI WARNING] No synchronized pairs; entering "
                            "[SINGLE-CAMERA FALLBACK]",
                            flush=True,
                        )
                        try:
                            side_camera.stop()
                        except Exception:
                            LOGGER.debug("Side-camera cleanup failed", exc_info=True)
                        side_camera = None
                        if side_extractor is not None:
                            side_extractor.close()
                            side_extractor = None
                        synchronizer.purge()
                        window_buffer.reset()
                        self._buffer_full_logged = False
                        self._invalid_window_count = 0
                        with self._resource_lock:
                            self._side_camera = None
                        continue
                    self.msleep(1)
        except Exception as error:
            if not self._stop_event.is_set():
                LOGGER.exception("GUI pipeline failed")
                print(f"[GUI ERROR] {type(error).__name__}: {error}", flush=True)
                self.error_occurred.emit(f"{type(error).__name__}: {error}")
        finally:
            for extractor in (front_extractor, side_extractor):
                if extractor is not None:
                    extractor.close()
            for camera in (front_camera, side_camera):
                if camera is not None:
                    try:
                        camera.stop()
                    except Exception as error:
                        self.error_occurred.emit(f"Camera shutdown: {error}")
            with self._resource_lock:
                self._front_camera = None
                self._side_camera = None

    def _process_pair(
        self,
        pair,
        front_extractor: HolisticExtractor,
        side_extractor: HolisticExtractor,
        matcher: StereoMatcher,
        preprocessor: PreprocessingPipeline,
        window_buffer: SlidingWindowBuffer,
        watchdog: SyncWatchdog,
        frame_count: int,
        fps_started: float,
    ) -> None:
        front_landmarks = front_extractor.process(
            pair.front.frame_buffer,
            timestamp_ns=pair.front.timestamp_ns,
            frame_index=pair.front.frame_index,
        )
        side_landmarks = side_extractor.process(
            pair.side.frame_buffer,
            timestamp_ns=pair.side.timestamp_ns,
            frame_index=pair.side.frame_index,
        )

        front_overlay = draw_asl_overlay(
            pair.front.frame_buffer.copy(), front_extractor.last_results
        )
        side_overlay = draw_asl_overlay(
            pair.side.frame_buffer.copy(), side_extractor.last_results
        )
        self._publish_frames(front_overlay, side_overlay)

        front_pixels = _normalized_to_pixels(
            front_landmarks.coordinates, pair.front.frame_buffer.shape
        )
        side_pixels = _normalized_to_pixels(
            side_landmarks.coordinates, pair.side.frame_buffer.shape
        )
        fused = matcher.match(
            front_pixels,
            side_pixels,
            front_fallback=front_landmarks.coordinates,
        )
        tensor = window_buffer.append_landmarks(
            fused.coordinates,
            timestamp_ns=max(pair.front.timestamp_ns, pair.side.timestamp_ns),
            preprocessor=preprocessor,
        )
        confidence = self._run_inference(tensor, window_buffer)

        elapsed = max(time.perf_counter() - fps_started, np.finfo(float).eps)
        triangulated = any(status is JointStatus.TRIANGULATED for status in fused.statuses)
        tracking = bool(front_landmarks.hand_presence.any())
        self.telemetry_ready.emit(
            {
                "fps": (frame_count + 1) / elapsed,
                "sync_delta_ms": pair.delta_t_ns / 1_000_000.0,
                "watchdog_drop_rate": watchdog.drop_rate,
                "fusion_status": "Triangulated" if triangulated else "Fallback",
                "mediapipe_tracking": tracking,
                "confidence": confidence,
            }
        )

    def _process_single_frame(
        self,
        frame,
        front_extractor: HolisticExtractor,
        preprocessor: PreprocessingPipeline,
        window_buffer: SlidingWindowBuffer,
        frame_count: int,
        fps_started: float,
    ) -> None:
        import cv2

        landmarks = front_extractor.process(
            frame.frame_buffer,
            timestamp_ns=frame.timestamp_ns,
            frame_index=frame.frame_index,
        )
        front_overlay = draw_asl_overlay(
            frame.frame_buffer.copy(), front_extractor.last_results
        )
        side_placeholder = np.zeros_like(front_overlay)
        cv2.putText(
            side_placeholder,
            "[SINGLE-CAMERA FALLBACK]",
            (20, max(40, side_placeholder.shape[0] // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )
        self._publish_frames(front_overlay, side_placeholder)

        tensor = window_buffer.append_landmarks(
            landmarks.coordinates,
            timestamp_ns=frame.timestamp_ns,
            preprocessor=preprocessor,
        )
        confidence = self._run_inference(tensor, window_buffer)
        elapsed = max(time.perf_counter() - fps_started, np.finfo(float).eps)
        self.telemetry_ready.emit(
            {
                "fps": (frame_count + 1) / elapsed,
                "sync_delta_ms": 0.0,
                "watchdog_drop_rate": 0.0,
                "fusion_status": "[SINGLE-CAMERA FALLBACK]",
                "single_camera": True,
                "mediapipe_tracking": bool(landmarks.hand_presence.any()),
                "confidence": confidence,
            }
        )

    def _run_inference(
        self, tensor, window_buffer: SlidingWindowBuffer
    ) -> float:
        if (
            window_buffer.raw_frame_count == window_buffer.window_size
            and not self._buffer_full_logged
        ):
            print(
                f"[GUI DEBUG] Temporal buffer reached "
                f"{window_buffer.window_size} frames",
                flush=True,
            )
            self._buffer_full_logged = True
        if tensor is None:
            if window_buffer.raw_frame_count == window_buffer.window_size:
                self._invalid_window_count += 1
                if self._invalid_window_count == 1 or self._invalid_window_count % 30 == 0:
                    print(
                        "[GUI DEBUG] Full window is not yet finite; waiting for "
                        "visible shoulders, elbows, and an active hand",
                        flush=True,
                    )
            return 0.0
        if self.inference_engine is None:
            raise RuntimeError("Inference engine is not initialized")
        self._invalid_window_count = 0
        result = self.inference_engine.predict(tensor)
        with self._threshold_lock:
            threshold = self._confidence_threshold
        labels = tuple(
            self.inference_engine.class_map[index]
            for index in range(len(self.inference_engine.class_map))
        )
        decision = ConfidenceFilter(
            confidence_threshold=threshold,
            margin_threshold=InferenceEngine.MIN_MARGIN,
        ).apply(
            result.probabilities,
            labels,
            inference_duration_ms=result.latency_ms,
            window_end_timestamp_ns=window_buffer.last_window_end_timestamp_ns or 0,
        )
        print(
            f"[GUI DEBUG] Inference: {result.predicted_gloss} "
            f"conf={result.confidence_score:.2f} margin={decision.margin:.2f} "
            f"latency={result.latency_ms:.1f}ms "
            f"status={'ACCEPTED' if decision.accepted else decision.rejection_reason}",
            flush=True,
        )
        if decision.accepted:
            print(
                f"[GUI DEBUG] Emitting gloss signal: '{result.predicted_gloss}'",
                flush=True,
            )
            self.prediction_ready.emit(
                result.predicted_gloss, result.confidence_score, result.latency_ms
            )
        return result.confidence_score


class SettingsDialog(QDialog):
    def __init__(
        self,
        configuration: RuntimeSettings | MutableMapping[str, Any],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Runtime Settings")
        self.configuration = configuration
        self._original_values = {
            "front_camera_index": int(_config_get(configuration, "front_camera_index")),
            "side_camera_index": int(_config_get(configuration, "side_camera_index")),
            "confidence_threshold": float(
                _config_get(configuration, "confidence_threshold")
            ),
        }

        self.front_camera_spin = QSpinBox(self)
        self.front_camera_spin.setRange(0, 32)
        self.front_camera_spin.setValue(int(_config_get(configuration, "front_camera_index")))
        self.side_camera_spin = QSpinBox(self)
        self.side_camera_spin.setRange(0, 32)
        self.side_camera_spin.setValue(int(_config_get(configuration, "side_camera_index")))
        self.confidence_slider = QSlider(Qt.Horizontal, self)
        self.confidence_slider.setRange(35, 100)
        self.confidence_slider.setValue(
            round(float(_config_get(configuration, "confidence_threshold")) * 100)
        )
        self.confidence_value = QLabel(self)

        form = QFormLayout(self)
        form.addRow("Front camera", self.front_camera_spin)
        form.addRow("45° side camera", self.side_camera_spin)
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(self.confidence_slider)
        threshold_row.addWidget(self.confidence_value)
        form.addRow("Confidence threshold", threshold_row)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self
        )
        form.addRow(buttons)

        self.front_camera_spin.valueChanged.connect(self._update_configuration)
        self.side_camera_spin.valueChanged.connect(self._update_configuration)
        self.confidence_slider.valueChanged.connect(self._update_configuration)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._update_configuration()

    @pyqtSlot()
    def _update_configuration(self) -> None:
        threshold = self.confidence_slider.value() / 100.0
        self.confidence_value.setText(f"{threshold:.2f}")
        _config_set(self.configuration, "front_camera_index", self.front_camera_spin.value())
        _config_set(self.configuration, "side_camera_index", self.side_camera_spin.value())
        _config_set(self.configuration, "confidence_threshold", threshold)

    def reject(self) -> None:
        for key, value in self._original_values.items():
            _config_set(self.configuration, key, value)
        super().reject()


class MainWindow(QMainWindow):
    def __init__(
        self,
        settings: RuntimeSettings | None = None,
        *,
        worker_factory: Callable[[RuntimeSettings], PipelineWorker] | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings or RuntimeSettings()
        self._worker_factory = worker_factory or (lambda value: PipelineWorker(value))
        self.worker: PipelineWorker | None = None
        self._last_confidence = 0.0
        self.setWindowTitle("Live ASL Stereo Translator")
        self.resize(1500, 850)
        self._build_ui()
        self._apply_theme()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        top = QHBoxLayout()
        self.front_view = _video_panel("Front Camera")
        self.side_view = _video_panel("45° Side Camera")
        top.addWidget(self.front_view, 4)
        top.addWidget(self.side_view, 4)

        status_panel = QFrame(self)
        status_panel.setObjectName("statusPanel")
        status_layout = QVBoxLayout(status_panel)
        status_layout.addWidget(QLabel("SYSTEM STATUS"))
        self.sync_badge = _status_badge("Camera Sync", "Idle")
        self.tracking_badge = _status_badge("MediaPipe", "Idle")
        self.fusion_badge = _status_badge("Stereo Fusion", "Idle")
        self.fps_label = QLabel("0.0 FPS")
        for widget in (
            self.sync_badge,
            self.tracking_badge,
            self.fusion_badge,
            self.fps_label,
        ):
            status_layout.addWidget(widget)
        status_layout.addWidget(QLabel("Prediction confidence"))
        self.confidence_bar = QProgressBar(self)
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        status_layout.addWidget(self.confidence_bar)
        self.prediction_detail = QLabel("Waiting for inference")
        self.prediction_detail.setWordWrap(True)
        status_layout.addWidget(self.prediction_detail)
        status_layout.addStretch(1)
        top.addWidget(status_panel, 2)
        root.addLayout(top, 7)

        bottom = QFrame(self)
        bottom.setObjectName("bottomPanel")
        bottom_layout = QVBoxLayout(bottom)
        self.ticker = TranslationTicker(bottom)
        bottom_layout.addWidget(self.ticker)
        controls = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.clear_button = QPushButton("Clear Text")
        self.settings_button = QPushButton("Settings")
        self.stop_button.setEnabled(False)
        for button in (
            self.start_button,
            self.stop_button,
            self.clear_button,
            self.settings_button,
        ):
            controls.addWidget(button)
        bottom_layout.addLayout(controls)
        root.addWidget(bottom, 3)
        self.setCentralWidget(central)

        self.start_button.clicked.connect(self.start_pipeline)
        self.stop_button.clicked.connect(self.stop_pipeline)
        self.clear_button.clicked.connect(self.ticker.clear)
        self.settings_button.clicked.connect(self.open_settings)

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #121212; color: #F5F5F5; }
            QFrame#statusPanel, QFrame#bottomPanel {
                background: #1B1B1B; border: 1px solid #454545; border-radius: 8px;
            }
            QLabel#videoViewport {
                background: #080808; border: 2px solid #5B5B5B; border-radius: 6px;
            }
            QLabel#statusBadge {
                background: #2A2A2A; border: 1px solid #666; border-radius: 5px;
                padding: 9px;
            }
            QTextEdit#translationText {
                background: #080808; color: #FFFFFF; border: 2px solid #5B5B5B;
                border-radius: 6px; padding: 12px;
            }
            QPushButton {
                background: #292929; border: 1px solid #777; border-radius: 5px;
                padding: 9px 18px; font-weight: 600;
            }
            QPushButton:hover { background: #383838; }
            QPushButton:disabled { color: #777; border-color: #444; }
            QProgressBar { border: 1px solid #777; border-radius: 5px; text-align: center; }
            QProgressBar::chunk { background: #35C76F; }
            """
        )

    @pyqtSlot()
    def start_pipeline(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.worker = self._worker_factory(replace(self.settings))
        self.worker.frames_ready.connect(self._update_frames)
        self.worker.prediction_ready.connect(self._handle_prediction)
        self.worker.telemetry_ready.connect(self._update_telemetry)
        self.worker.error_occurred.connect(self._show_error)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    @pyqtSlot()
    def stop_pipeline(self) -> None:
        worker = self.worker
        if worker is None:
            return
        worker.stop()
        if not worker.wait(6_000):
            self._show_error("Pipeline did not stop within six seconds")
        self._worker_finished()

    @pyqtSlot()
    def open_settings(self) -> None:
        old_front = self.settings.front_camera_index
        old_side = self.settings.side_camera_index
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        if self.worker is not None and self.worker.isRunning():
            cameras_changed = (
                old_front != self.settings.front_camera_index
                or old_side != self.settings.side_camera_index
            )
            if cameras_changed:
                self.stop_pipeline()
                self.start_pipeline()
            else:
                self.worker.update_confidence_threshold(
                    self.settings.confidence_threshold
                )

    @pyqtSlot(np.ndarray, np.ndarray)
    def _update_frames(self, front: np.ndarray, side: np.ndarray) -> None:
        if self.worker is not None:
            latest = self.worker.consume_latest_frames()
            if latest is not None:
                front, side = latest
        _set_viewport_frame(self.front_view, front)
        _set_viewport_frame(self.side_view, side)

    @pyqtSlot(str, float, float)
    def _handle_prediction(self, gloss: str, confidence: float, latency_ms: float) -> None:
        print(
            f"[GUI DEBUG] Received gloss signal: '{gloss}' "
            f"(conf: {confidence:.2f})",
            flush=True,
        )
        self._last_confidence = confidence
        self.confidence_bar.setValue(round(confidence * 100))
        self.prediction_detail.setText(
            "Last detected: " + prediction_text(gloss, confidence, latency_ms)
        )
        appended = self.ticker.add_gloss(gloss)
        print(
            f"[GUI DEBUG] Translation ticker "
            f"{'updated' if appended else 'debounced'}: {self.ticker.text!r}",
            flush=True,
        )

    @pyqtSlot(dict)
    def _update_telemetry(self, telemetry: dict[str, Any]) -> None:
        confidence = float(telemetry.get("confidence", self._last_confidence))
        display = map_telemetry(telemetry, confidence=confidence)
        _set_badge(self.sync_badge, "Camera Sync", display.sync_text, display.sync_ok)
        _set_badge(
            self.tracking_badge,
            "MediaPipe Tracking",
            display.tracking_text,
            display.tracking_ok,
        )
        _set_badge(
            self.fusion_badge,
            "Stereo Fusion",
            display.fusion_text,
            display.fusion_ok,
        )
        self.fps_label.setText(display.fps_text)
        self.confidence_bar.setValue(display.confidence_percent)

    @pyqtSlot(str)
    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Pipeline Error", message)

    @pyqtSlot()
    def _worker_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.stop_pipeline()
        event.accept()


def _normalized_to_pixels(coordinates: np.ndarray, frame_shape: tuple[int, ...]) -> np.ndarray:
    output = np.array(coordinates, dtype=np.float32, copy=True)
    height, width = frame_shape[:2]
    output[:, 0] *= width
    output[:, 1] *= height
    return output


def _video_panel(title: str) -> QLabel:
    label = QLabel(title)
    label.setObjectName("videoViewport")
    label.setAlignment(Qt.AlignCenter)
    label.setMinimumSize(420, 320)
    label.setScaledContents(False)
    return label


def _status_badge(title: str, value: str) -> QLabel:
    label = QLabel()
    label.setObjectName("statusBadge")
    _set_badge(label, title, value, False)
    return label


def _set_badge(label: QLabel, title: str, value: str, healthy: bool) -> None:
    color = "#35C76F" if healthy else "#F05B5B"
    label.setText(f"{title}: <span style='color:{color}; font-weight:700'>{value}</span>")


def _set_viewport_frame(label: QLabel, frame: np.ndarray) -> None:
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        return
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    height, width, channels = rgb.shape
    image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(image).scaled(
        label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
    )
    label.setPixmap(pixmap)


def _config_get(configuration: RuntimeSettings | MutableMapping[str, Any], key: str) -> Any:
    if isinstance(configuration, MutableMapping):
        defaults = RuntimeSettings()
        return configuration.get(key, getattr(defaults, key))
    return getattr(configuration, key)


def _config_set(
    configuration: RuntimeSettings | MutableMapping[str, Any], key: str, value: Any
) -> None:
    if isinstance(configuration, MutableMapping):
        configuration[key] = value
    else:
        setattr(configuration, key, value)


__all__ = ["MainWindow", "PipelineWorker", "RuntimeSettings", "SettingsDialog"]
