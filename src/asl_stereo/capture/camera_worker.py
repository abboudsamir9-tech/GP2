"""Low-latency, RAM-only OpenCV camera ingestion."""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections.abc import Callable
from types import TracebackType
from typing import Any, Protocol, TypeVar

import numpy as np

from asl_stereo.contracts import TimestampedFrame


class VideoCaptureLike(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, Any]: ...

    def set(self, property_id: int, value: float) -> bool: ...

    def release(self) -> None: ...


CaptureSource = int | str
CaptureFactory = Callable[[CaptureSource], VideoCaptureLike]
_CameraWorkerT = TypeVar("_CameraWorkerT", bound="CameraWorker")


def _open_cv_capture(source: CaptureSource) -> VideoCaptureLike:
    import cv2

    if sys.platform.startswith("win") and isinstance(source, int):
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if capture.isOpened():
            return capture
        capture.release()
    return cv2.VideoCapture(source)


def _buffer_size_property() -> int:
    import cv2

    return int(cv2.CAP_PROP_BUFFERSIZE)


def _resolution_properties() -> tuple[int, int]:
    import cv2

    return int(cv2.CAP_PROP_FRAME_WIDTH), int(cv2.CAP_PROP_FRAME_HEIGHT)


class CameraWorker:
    """Capture frames on a daemon thread and expose only the newest frame.

    The one-slot output queue makes consumer reads non-blocking and prevents
    backlog latency. Frames never leave process memory; this class contains no
    persistence or networking path.
    """

    def __init__(
        self,
        source: CaptureSource,
        *,
        camera_id: str,
        warmup_frames: int = 10,
        resolution: tuple[int, int] | None = None,
        capture_factory: CaptureFactory | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        failure_backoff_s: float = 0.005,
    ) -> None:
        if not isinstance(camera_id, str) or not camera_id:
            raise ValueError("camera_id must be a non-empty string")
        if isinstance(warmup_frames, bool) or not isinstance(warmup_frames, int):
            raise TypeError("warmup_frames must be an integer")
        if warmup_frames < 0:
            raise ValueError("warmup_frames must be non-negative")
        if failure_backoff_s < 0:
            raise ValueError("failure_backoff_s must be non-negative")
        if resolution is not None:
            if (
                len(resolution) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in resolution)
                or any(value <= 0 for value in resolution)
            ):
                raise ValueError("resolution must contain positive integer width and height")

        self.source = source
        self.camera_id = camera_id
        self.warmup_frames = warmup_frames
        self.resolution = resolution
        self._capture_factory = capture_factory or _open_cv_capture
        self._monotonic_ns = monotonic_ns
        self._failure_backoff_s = failure_backoff_s

        self._frames: queue.Queue[TimestampedFrame] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._state_lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._capture: VideoCaptureLike | None = None
        self._startup_error: BaseException | None = None
        self._startup_deadline = 0.0
        self._frame_index = 0
        self._capture_failures = 0
        self._backpressure_drops = 0

    def start(self, *, timeout: float = 5.0) -> None:
        """Start ingestion and wait until camera setup/warmup completes."""
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        with self._state_lock:
            if self.is_running:
                return
            self._stop_event.clear()
            self._ready_event.clear()
            self._startup_error = None
            self._startup_deadline = time.monotonic() + timeout
            self._thread = threading.Thread(
                target=self._run,
                name=f"camera-worker-{self.camera_id}",
                daemon=True,
            )
            self._thread.start()

        if not self._ready_event.wait(timeout):
            self.stop(timeout=timeout)
            raise TimeoutError(f"camera {self.camera_id!r} did not become ready")
        if self._startup_error is not None:
            error = self._startup_error
            self.stop(timeout=timeout)
            raise RuntimeError(f"camera {self.camera_id!r} failed to start") from error

    def _run(self) -> None:
        capture: VideoCaptureLike | None = None
        try:
            capture = self._capture_factory(self.source)
            with self._state_lock:
                self._capture = capture
            if not capture.isOpened():
                raise RuntimeError(f"unable to open camera source {self.source!r}")

            # Backends may report this property as unsupported. The software
            # queue remains one slot, preserving bounded consumer latency.
            capture.set(_buffer_size_property(), 1.0)
            if self.resolution is not None:
                width_property, height_property = _resolution_properties()
                width, height = self.resolution
                capture.set(width_property, float(width))
                capture.set(height_property, float(height))

            warmed_frames = 0
            while warmed_frames < self.warmup_frames:
                if self._stop_event.is_set():
                    return
                ok, _ = capture.read()
                if ok:
                    warmed_frames += 1
                    continue

                self._capture_failures += 1
                if time.monotonic() >= self._startup_deadline:
                    raise TimeoutError(
                        f"camera {self.camera_id!r} warmup timed out after "
                        f"{warmed_frames}/{self.warmup_frames} frames"
                    )
                time.sleep(0.02)

            self._ready_event.set()

            while not self._stop_event.is_set():
                ok, raw_frame = capture.read()
                timestamp_ns = self._monotonic_ns()
                if not ok or not isinstance(raw_frame, np.ndarray):
                    self._capture_failures += 1
                    if self._failure_backoff_s:
                        self._stop_event.wait(self._failure_backoff_s)
                    continue
                if raw_frame.dtype != np.uint8 or raw_frame.ndim not in (2, 3):
                    self._capture_failures += 1
                    continue

                record = TimestampedFrame(
                    camera_id=self.camera_id,
                    frame_index=self._frame_index,
                    timestamp_ns=timestamp_ns,
                    frame_buffer=raw_frame,
                    health_meta={
                        "capture_failures": self._capture_failures,
                        "backpressure_drops": self._backpressure_drops,
                    },
                )
                self._frame_index += 1
                self._publish_latest(record)
        except BaseException as error:
            self._startup_error = error
            self._ready_event.set()
        finally:
            if capture is not None:
                capture.release()
            with self._state_lock:
                self._capture = None
            self._ready_event.set()

    def _publish_latest(self, frame: TimestampedFrame) -> None:
        try:
            self._frames.put_nowait(frame)
            return
        except queue.Full:
            pass

        try:
            self._frames.get_nowait()
        except queue.Empty:
            pass
        else:
            self._backpressure_drops += 1
        self._frames.put_nowait(frame)

    def get_frame(self, *, timeout: float = 0.0) -> TimestampedFrame | None:
        """Return one available frame; the default call never blocks."""
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        try:
            if timeout == 0:
                return self._frames.get_nowait()
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self, *, timeout: float = 2.0) -> None:
        """Request shutdown, unblock a stuck backend if needed, and join."""
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        self._stop_event.set()
        with self._state_lock:
            thread = self._thread
            capture = self._capture
        if thread is None:
            return

        thread.join(timeout)
        if thread.is_alive() and capture is not None:
            capture.release()
            thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError(f"camera {self.camera_id!r} did not stop cleanly")
        with self._state_lock:
            self._thread = None

    def request_stop(self) -> None:
        """Signal shutdown without blocking the calling thread."""
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive() and not self._stop_event.is_set()

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def capture_failures(self) -> int:
        return self._capture_failures

    @property
    def backpressure_drops(self) -> int:
        return self._backpressure_drops

    def __enter__(self: _CameraWorkerT) -> _CameraWorkerT:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()


__all__ = ["CameraWorker", "CaptureFactory", "CaptureSource", "VideoCaptureLike"]
