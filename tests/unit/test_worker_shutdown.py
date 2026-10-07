"""Bounded cleanup and responsive Qt shutdown without real camera devices."""

import os
import importlib.util
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from asl_stereo.capture import CameraWorker
from ui import MainWindow, PipelineWorker, RuntimeSettings


class BlockingCapture:
    def __init__(self, *, delay_release=False):
        self.read_entered = threading.Event()
        self.read_unblocked = threading.Event()
        self.release_entered = threading.Event()
        self.release_unblocked = threading.Event()
        self.delay_release = delay_release
        self.releases = 0

    def isOpened(self):
        return not self.read_unblocked.is_set()

    def set(self, *args):
        return True

    def read(self):
        self.read_entered.set()
        self.read_unblocked.wait(2.0)
        return False, None

    def release(self):
        self.releases += 1
        self.release_entered.set()
        if self.delay_release:
            self.release_unblocked.wait(2.0)
        self.read_unblocked.set()


def test_camera_stop_unblocks_read_and_releases_handle_exactly_once():
    capture = BlockingCapture()
    worker = CameraWorker(0, camera_id="front", warmup_frames=0,
                          capture_factory=lambda _: capture)
    try:
        worker.start(timeout=1.0)
        assert capture.read_entered.wait(1.0)
        started = time.perf_counter()
        worker.stop(timeout=0.3)
        assert time.perf_counter() - started < 0.4
        assert capture.releases == 1
        assert worker.peek_latest_frame() is None
        assert worker._thread is None
    finally:
        capture.release_unblocked.set()
        capture.read_unblocked.set()
        worker.stop(timeout=1.0)


def test_camera_stop_deadline_includes_blocked_release_and_disallows_restart():
    capture = BlockingCapture(delay_release=True)
    worker = CameraWorker(0, camera_id="front", warmup_frames=0,
                          capture_factory=lambda _: capture)
    try:
        worker.start(timeout=1.0)
        assert capture.read_entered.wait(1.0)
        started = time.perf_counter()
        with pytest.raises(TimeoutError):
            worker.stop(timeout=0.08)
        assert time.perf_counter() - started < 0.25
        assert capture.release_entered.wait(0.2)
        with pytest.raises(RuntimeError, match="stopping"):
            worker.start(timeout=0.1)
    finally:
        capture.release_unblocked.set()
        capture.read_unblocked.set()
        worker.stop(timeout=1.0)
    assert capture.releases == 1


def test_stop_during_device_open_cancels_startup_and_releases_late_handle():
    entered, allow_open = threading.Event(), threading.Event()
    capture = BlockingCapture()
    errors = []

    def factory(_):
        entered.set()
        allow_open.wait(1.0)
        return capture

    worker = CameraWorker(0, camera_id="front", warmup_frames=0, capture_factory=factory)

    def start():
        try:
            worker.start(timeout=1.0)
        except RuntimeError as error:
            errors.append(error)

    starter = threading.Thread(target=start)
    try:
        starter.start()
        assert entered.wait(1.0)
        worker.request_stop()
        allow_open.set()
        starter.join(1.0)
        assert not starter.is_alive()
        assert len(errors) == 1 and "cancelled" in str(errors[0])
        assert capture.releases == 1
        assert not capture.read_entered.is_set()
    finally:
        allow_open.set()
        capture.read_unblocked.set()
        starter.join(1.0)
        worker.stop(timeout=1.0)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _process_until(app, condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert condition()


class SlowShutdownWorker(PipelineWorker):
    def __init__(self, settings):
        super().__init__(settings)
        self.entered = threading.Event()
        self.allow_finish = threading.Event()

    def run(self):
        self.entered.set()
        self._stop_event.wait(2.0)
        self.allow_finish.wait(2.0)


def test_window_close_is_deferred_and_gui_events_continue_until_worker_finishes(app):
    worker = SlowShutdownWorker(RuntimeSettings())
    window = MainWindow(worker_factory=lambda _: worker)
    pulses = []
    timer = QTimer(window)
    timer.setInterval(5)
    timer.timeout.connect(lambda: pulses.append(1))
    try:
        window.show()
        window.start_pipeline()
        assert worker.entered.wait(1.0)
        window._restart_after_stop = True
        timer.start()
        started = time.perf_counter()
        assert window.close() is False
        assert time.perf_counter() - started < 0.1
        assert worker._stop_event.is_set()
        assert not window._restart_after_stop
        assert window.isVisible()
        assert not window.start_button.isEnabled()
        _process_until(app, lambda: len(pulses) >= 3)
        worker.allow_finish.set()
        _process_until(app, lambda: not window.isVisible())
        assert worker.wait(0)
    finally:
        worker.stop()
        worker.allow_finish.set()
        assert worker.wait(1_000)
        timer.stop()
        window.close()
        app.processEvents()


def test_shutdown_timeout_does_not_claim_worker_finished_or_show_modal_error(app, monkeypatch):
    worker = SlowShutdownWorker(RuntimeSettings())
    window = MainWindow(worker_factory=lambda _: worker)
    critical = Mock()
    monkeypatch.setattr("ui.dashboard.QMessageBox.critical", critical)
    monkeypatch.setattr("ui.dashboard.SHUTDOWN_TIMEOUT_S", 0.03)
    try:
        window.show()
        window.start_pipeline()
        assert worker.entered.wait(1.0)
        window.stop_pipeline()
        _process_until(app, lambda: "Shutdown" in window.prediction_detail.text())
        window._worker_finished()
        assert worker.isRunning()
        assert not window.start_button.isEnabled()
        assert window.isVisible()
        critical.assert_not_called()
        worker.allow_finish.set()
        _process_until(app, lambda: window.start_button.isEnabled())
    finally:
        worker.stop()
        worker.allow_finish.set()
        assert worker.wait(1_000)
        window.close()
        app.processEvents()


def test_pipeline_parent_remains_alive_until_vision_cleanup_finishes(app, monkeypatch):
    allow_finish, entered = threading.Event(), threading.Event()
    children = []
    errors = []

    class CleanupWorker(PipelineWorker):
        def run(self):
            vision = threading.Thread(target=lambda: allow_finish.wait(2.0))
            children.append(vision)
            vision.start()
            entered.set()
            self._shutdown_resources(vision)

    monkeypatch.setattr("ui.dashboard.SHUTDOWN_TIMEOUT_S", 0.03)
    worker = CleanupWorker(RuntimeSettings())
    worker.error_occurred.connect(errors.append)
    try:
        worker.start()
        assert entered.wait(1.0)
        _process_until(app, lambda: bool(errors))
        assert worker.isRunning() and children[0].is_alive()
        allow_finish.set()
        _process_until(app, lambda: worker.wait(0))
        assert not children[0].is_alive()
    finally:
        allow_finish.set()
        assert worker.wait(1_000)


def test_launcher_ctrl_c_requests_window_close_and_restores_signal_handler(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "shutdown_test_launcher", Path(__file__).resolve().parents[2] / "main.py",
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    window, application, timer = Mock(), Mock(), Mock()
    previous = object()
    handlers = []
    fake_signal = SimpleNamespace(
        SIGINT=2, getsignal=lambda _: previous,
        signal=lambda _, handler: handlers.append(handler),
    )
    monkeypatch.setattr(launcher, "signal", fake_signal)
    monkeypatch.setattr(launcher, "MainWindow", Mock(return_value=window))
    monkeypatch.setattr(launcher, "QApplication", Mock(return_value=application))
    monkeypatch.setattr(launcher, "QTimer", Mock(return_value=timer))

    def execute():
        handlers[0](2, None)
        window.close.assert_called_once_with()
        return 0

    application.exec_.side_effect = execute
    assert launcher.main() == 0
    assert window.close.call_count == 2  # SIGINT request and final cleanup.
    timer.stop.assert_called_once_with()
    assert handlers[-1] is previous


def test_launcher_explicit_app_quit_drains_worker_before_returning(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "quit_test_launcher", Path(__file__).resolve().parents[2] / "main.py",
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    window, application, timer = Mock(), Mock(), Mock()
    window.worker.wait.side_effect = [False, True]
    application.exec_.return_value = 0
    monkeypatch.setattr(launcher, "MainWindow", Mock(return_value=window))
    monkeypatch.setattr(launcher, "QApplication", Mock(return_value=application))
    monkeypatch.setattr(launcher, "QTimer", Mock(return_value=timer))
    monkeypatch.setattr(launcher, "signal", SimpleNamespace(
        SIGINT=2, getsignal=lambda _: None, signal=Mock(),
    ))
    assert launcher.main() == 0
    window.close.assert_called_once_with()
    window.stop_pipeline.assert_called_once_with()
    assert window.worker.wait.call_count == 2
    window.worker.wait.assert_called_with(50)
    application.processEvents.assert_called_once_with()
