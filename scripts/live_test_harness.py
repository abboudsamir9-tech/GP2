"""RAM-only live ASL camera and inference test; press q or Ctrl+C to exit."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from asl_stereo.capture import CameraWorker, Synchronizer  # noqa: E402
from asl_stereo.contracts import LandmarkFrame, TimestampedFrame  # noqa: E402
from asl_stereo.landmarks import PoseHandsExtractor, draw_asl_overlay  # noqa: E402
from asl_stereo.models import InferenceEngine, SignSequenceClassifier  # noqa: E402
from asl_stereo.models.checkpoint import load_class_map  # noqa: E402
from asl_stereo.preprocessing import PreprocessingPipeline, SlidingWindowBuffer  # noqa: E402
from asl_stereo.stereo import StereoCalibration, StereoMatcher  # noqa: E402

WINDOW_TITLE = "ASL Live Test"
CAPTURE_RESOLUTION = (1280, 720)
DISPLAY_RESOLUTION = (640, 360)
REQUESTED_FPS = 60.0
INFERENCE_STRIDE = 6
CAMERA_INDICES = (0, 1, 2)
NO_FRAME_TIMEOUT_S = 5.0
NO_PAIR_TIMEOUT_S = 3.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test live ASL landmark and model inference.")
    parser.add_argument("--single-camera", action="store_true")
    parser.add_argument(
        "--front", "--camera-index", dest="front", type=int, default=0,
        help="Preferred front camera index (default: 0).",
    )
    parser.add_argument("--side", type=int, default=1, help="Preferred side index.")
    parser.add_argument(
        "--weights-path", type=Path, default=None,
        help="Default: optimized_model.pt, then best_model.pth.",
    )
    parser.add_argument(
        "--class-map", type=Path,
        default=PROJECT_ROOT / "configs" / "class_map.json",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.40,
        help="Diagnostic gloss display threshold; acceptance retains 0.65 confidence "
        "and 0.15 margin floors.",
    )
    return parser


class FpsMeter:
    def __init__(self) -> None:
        self._last_time: float | None = None
        self.fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last_time is not None:
            elapsed = now - self._last_time
            if elapsed > 0:
                instantaneous = 1.0 / elapsed
                self.fps = instantaneous if self.fps == 0 else 0.9 * self.fps + 0.1 * instantaneous
        self._last_time = now
        return self.fps


@dataclass(slots=True)
class LiveState:
    """Small, lock-protected handoff from vision to the display loop."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    side_camera: CameraWorker | None = None
    front_landmarks: LandmarkFrame | None = None
    front_results: object = None
    side_landmarks: LandmarkFrame | None = None
    side_results: object = None
    extraction_fps: float = 0.0
    last_gloss: str = "--"
    last_confidence: float = 0.0
    last_latency_ms: float = 0.0
    last_status: str = "BUFFERING"


class LiveWindowPreprocessor:
    """Match training's neutral pose for a hand absent for a whole window."""

    def __init__(self) -> None:
        self.pipeline = PreprocessingPipeline()

    def process_live_window(self, raw_window: np.ndarray) -> np.ndarray:
        return self.pipeline.process_live_window(raw_window)


def _load_engine(weights_path: Path | None, class_map_path: Path) -> InferenceEngine:
    import torch

    torch.set_num_threads(1)
    class_map = load_class_map(class_map_path)
    candidates = [weights_path] if weights_path is not None else [
        PROJECT_ROOT / "weights" / "optimized_model.pt",
        PROJECT_ROOT / "weights" / "best_model.pth",
    ]
    failures: list[str] = []
    for candidate in candidates:
        if not candidate.is_file():
            failures.append(f"{candidate}: missing")
            continue
        try:
            engine = InferenceEngine(
                SignSequenceClassifier(num_classes=len(class_map)),
                class_map,
                checkpoint_path=candidate,
            )
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            failures.append(f"{candidate}: {error}")
            print(f"[WARNING] Model artifact unavailable: {candidate} ({error})", flush=True)
            continue
        print(f"[INFO] Loaded model: {candidate}", flush=True)
        return engine
    raise RuntimeError("No usable model artifact. " + "; ".join(failures))


def _camera_candidates(preferred: int, excluded: set[int]) -> list[int]:
    if preferred < 0:
        raise ValueError("camera indices must be non-negative")
    result: list[int] = []
    for index in (preferred, *CAMERA_INDICES):
        if index not in excluded and index not in result:
            result.append(index)
    return result


def _stop_camera(camera: CameraWorker) -> None:
    try:
        camera.stop()
    except Exception as error:
        print(f"[WARNING] Camera shutdown: {error}", file=sys.stderr, flush=True)


def _open_camera(
    stack: ExitStack,
    preferred: int,
    *,
    camera_id: str,
    excluded: set[int] | None = None,
    camera_factory: Callable[..., CameraWorker] = CameraWorker,
) -> tuple[CameraWorker, int, TimestampedFrame]:
    failures: list[str] = []
    for index in _camera_candidates(preferred, excluded or set()):
        for attempt in (1, 2):
            camera: CameraWorker | None = None
            try:
                camera = camera_factory(
                    index, camera_id=camera_id,
                    resolution=CAPTURE_RESOLUTION, target_fps=REQUESTED_FPS,
                )
                camera.start(timeout=5.0)
                first = camera.get_frame(timeout=1.5)
                if (
                    first is None
                    or first.frame_buffer.size == 0
                    or first.frame_buffer.ndim != 3
                    or first.frame_buffer.shape[2] != 3
                ):
                    raise RuntimeError("no non-empty frame after warmup")
                if not camera.is_running:
                    raise RuntimeError(f"capture stopped: {camera.last_error!r}")
            except Exception as error:
                reason = error.__cause__ or error
                if camera is not None:
                    _stop_camera(camera)
                if isinstance(reason, TimeoutError) and attempt == 1:
                    print(f"[WARNING] Camera index {index} timed out; retrying", flush=True)
                    continue
                failures.append(f"index {index}: {reason}")
                print(f"[WARNING] Camera index {index} unavailable: {reason}", flush=True)
                break

            stack.callback(_stop_camera, camera)
            height, width = first.frame_buffer.shape[:2]
            actual_fps = camera.actual_fps
            fps_text = f"{actual_fps:.1f}" if actual_fps is not None else "unknown"
            print(
                f"[INFO] Successfully opened camera at index {index} "
                f"at {width}x{height} @ {fps_text} FPS",
                flush=True,
            )
            if actual_fps is None:
                print("[WARNING] Driver did not report FPS; requested 60 FPS.", flush=True)
            elif actual_fps < REQUESTED_FPS - 0.5:
                print(
                    f"[WARNING] Hardware capped at {actual_fps:.1f} FPS. "
                    "Continuing with available rate.",
                    flush=True,
                )
            return camera, index, first
    raise RuntimeError(f"No usable {camera_id} camera: {'; '.join(failures)}")


def annotate_frame(
    frame: np.ndarray,
    landmarks: LandmarkFrame,
    results: object,
    *,
    camera_name: str,
    fps: float,
    sync_delta_ms: float | None = None,
) -> np.ndarray:
    import cv2

    annotated = draw_asl_overlay(frame.copy(), results)
    present = landmarks.hand_present.astype(bool)
    hands = f"L:{'ON' if present[0] else 'LOST'} R:{'ON' if present[1] else 'LOST'}"
    lines = [
        f"{camera_name} | FPS {fps:5.1f}",
        f"Hands {hands} | Joints {int(landmarks.joint_mask.sum())}/46",
    ]
    if sync_delta_ms is not None:
        lines.append(f"Sync delta {sync_delta_ms:+.1f} ms")
    for line_index, label in enumerate(lines):
        origin = (12, 28 + line_index * 26)
        cv2.putText(annotated, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(annotated, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (80, 255, 120), 1, cv2.LINE_AA)
    return annotated


def _normalized_to_pixels(coordinates: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    pixels = np.array(coordinates, dtype=np.float32, copy=True)
    pixels[:, 0] *= shape[1]
    pixels[:, 1] *= shape[0]
    return pixels


def _report_prediction(
    landmarks: LandmarkFrame,
    coordinates: np.ndarray,
    *,
    fps: float,
    frame_count: int,
    buffer: SlidingWindowBuffer,
    preprocessor: LiveWindowPreprocessor,
    engine: InferenceEngine,
    confidence_threshold: float,
    state: LiveState,
) -> None:
    pose_visible = bool(np.isfinite(coordinates[42:46]).all())
    hand_visible = bool(landmarks.hand_present.any())
    window = buffer.append_landmarks(
        coordinates,
        timestamp_ns=landmarks.timestamp_ns,
        preprocessor=preprocessor,
    )
    if not pose_visible or not hand_visible:
        if frame_count % 30 == 0:
            print(
                f"[Extraction: {fps:.1f} FPS] Waiting for signer framing "
                "(shoulders, elbows, and at least one hand required)...",
                flush=True,
            )
        return
    if window is None:
        return

    result = engine.predict(window)
    decision = engine.gate_prediction(
        result,
        confidence_threshold=confidence_threshold,
        window_end_timestamp_ns=buffer.last_window_end_timestamp_ns or 0,
    )
    gloss = result.predicted_gloss if result.confidence_score >= confidence_threshold else "--"
    status = "ACCEPTED" if decision.accepted else "LOW_CONF"
    with state.lock:
        state.last_latency_ms = result.latency_ms
        state.last_status = status
        if decision.accepted:
            state.last_gloss = result.predicted_gloss
            state.last_confidence = result.confidence_score
    print(
        f"[Extraction: {fps:.1f} FPS] Landmark Lock: YES | Prediction: {gloss} | "
        f"Conf: {result.confidence_score:.2f} | Status: {status} "
        f"| Inference: {result.latency_ms:.1f} ms",
        flush=True,
    )


def run(
    single_camera: bool,
    front_index: int,
    side_index: int,
    *,
    weights_path: Path | None = None,
    class_map_path: Path = PROJECT_ROOT / "configs" / "class_map.json",
    confidence_threshold: float = 0.40,
) -> None:
    import cv2

    if front_index < 0 or side_index < 0:
        raise ValueError("camera indices must be non-negative")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be in [0, 1]")

    engine = _load_engine(weights_path, class_map_path)
    with ExitStack() as stack:
        front, selected_front, first_front = _open_camera(
            stack, front_index, camera_id="front"
        )
        side: CameraWorker | None = None
        first_side: TimestampedFrame | None = None
        if not single_camera:
            try:
                side, _, first_side = _open_camera(
                    stack, side_index, camera_id="side", excluded={selected_front}
                )
            except RuntimeError as error:
                print(f"[WARNING] {error}. Entering single-camera mode.", flush=True)
        calibration = StereoCalibration()
        if side is not None:
            try:
                calibration = StereoCalibration.from_default(
                    PROJECT_ROOT / "configs" / "calibration_params.json"
                )
            except (OSError, ValueError) as error:
                print(
                    f"[WARNING] Stereo calibration unavailable: {error}. "
                    "Using front-camera coordinates.",
                    flush=True,
                )
        state = LiveState(side_camera=side)
        stop_event = threading.Event()
        errors: list[Exception] = []
        vision = threading.Thread(
            target=_vision_loop,
            args=(front, side, first_front, first_side, calibration, engine,
                  confidence_threshold, state, stop_event, errors),
            name="asl-live-vision",
            daemon=True,
        )
        vision.start()
        try:
            _display_loop(front, state, stop_event)
        finally:
            stop_event.set()
            vision.join(timeout=5.0)
        if errors:
            raise errors[0]


def _vision_loop(
    front: CameraWorker,
    side: CameraWorker | None,
    first_front: TimestampedFrame,
    first_side: TimestampedFrame | None,
    calibration: StereoCalibration,
    engine: InferenceEngine,
    confidence_threshold: float,
    state: LiveState,
    stop_event: threading.Event,
    errors: list[Exception],
) -> None:
    try:
        with ExitStack() as extractors:
            front_extractor = extractors.enter_context(PoseHandsExtractor())
            side_extractor = extractors.enter_context(PoseHandsExtractor()) if side else None
            meter = FpsMeter()
            preprocessor = LiveWindowPreprocessor()
            buffer = SlidingWindowBuffer(window_size=45, stride=INFERENCE_STRIDE)
            matcher = StereoMatcher(calibration, fallback_scope="frame")
            synchronizer = Synchronizer()
            last_frame_time = time.monotonic()
            last_pair_time = last_frame_time
            frame_count = 0

            while front.is_running and not stop_event.is_set():
                if side is not None and not side.is_running:
                    print("[WARNING] Side camera stopped. Entering single-camera mode.", flush=True)
                    side = None
                    with state.lock:
                        state.side_camera = None
                        state.side_landmarks = None
                        state.side_results = None
                    buffer.reset()

                if side is None:
                    record = first_front or front.get_frame(timeout=0.01)
                    first_front = None
                    if record is None:
                        if time.monotonic() - last_frame_time > NO_FRAME_TIMEOUT_S:
                            raise RuntimeError("front camera stopped delivering frames")
                        continue
                    last_frame_time = time.monotonic()
                    landmarks = front_extractor.process(
                        record.frame_buffer,
                        timestamp_ns=record.timestamp_ns,
                        frame_index=record.frame_index,
                    )
                    fps = meter.tick()
                    with state.lock:
                        state.front_landmarks = landmarks
                        state.front_results = front_extractor.last_results
                        state.extraction_fps = fps
                    _report_prediction(
                        landmarks, landmarks.coordinates, fps=fps,
                        frame_count=frame_count, buffer=buffer,
                        preprocessor=preprocessor, engine=engine,
                        confidence_threshold=confidence_threshold, state=state,
                    )
                    frame_count += 1
                    continue

                front_record = first_front or front.get_frame(timeout=0.01)
                side_record = first_side or side.get_frame(timeout=0.01)
                first_front = None
                first_side = None
                pair = None
                if front_record is not None:
                    last_frame_time = time.monotonic()
                    pair = synchronizer.add_front(front_record)
                if side_record is not None:
                    side_pair = synchronizer.add_side(side_record)
                    if side_pair is not None:
                        pair = side_pair
                if pair is not None:
                    last_pair_time = time.monotonic()
                    front_landmarks = front_extractor.process(
                        pair.front.frame_buffer,
                        timestamp_ns=pair.front.timestamp_ns,
                        frame_index=pair.front.frame_index,
                    )
                    assert side_extractor is not None
                    side_landmarks = side_extractor.process(
                        pair.side.frame_buffer,
                        timestamp_ns=pair.side.timestamp_ns,
                        frame_index=pair.side.frame_index,
                    )
                    fps = meter.tick()
                    with state.lock:
                        state.front_landmarks = front_landmarks
                        state.front_results = front_extractor.last_results
                        state.side_landmarks = side_landmarks
                        state.side_results = side_extractor.last_results
                        state.extraction_fps = fps
                    coordinates = front_landmarks.coordinates
                    if calibration.available:
                        coordinates = matcher.match(
                            _normalized_to_pixels(coordinates, pair.front.frame_buffer.shape),
                            _normalized_to_pixels(
                                side_landmarks.coordinates, pair.side.frame_buffer.shape
                            ),
                            front_fallback=coordinates,
                        ).coordinates
                    _report_prediction(
                        front_landmarks, coordinates, fps=fps,
                        frame_count=frame_count, buffer=buffer,
                        preprocessor=preprocessor, engine=engine,
                        confidence_threshold=confidence_threshold, state=state,
                    )
                    frame_count += 1
                elif time.monotonic() - last_pair_time > NO_PAIR_TIMEOUT_S:
                    print("[WARNING] No synchronized pairs. Entering single-camera mode.",
                          flush=True)
                    _stop_camera(side)
                    side = None
                    with state.lock:
                        state.side_camera = None
                        state.side_landmarks = None
                        state.side_results = None
                    buffer.reset()
                if time.monotonic() - last_frame_time > NO_FRAME_TIMEOUT_S:
                    raise RuntimeError("front camera stopped delivering frames")
            if not stop_event.is_set():
                raise RuntimeError(f"front camera stopped: {front.last_error!r}")
    except Exception as error:
        errors.append(error)
        stop_event.set()


def _display_loop(front: CameraWorker, state: LiveState, stop_event: threading.Event) -> None:
    import cv2

    next_display_at = time.perf_counter()
    last_front_index = -1
    last_side_index = -1
    last_report_at = 0.0
    while not stop_event.is_set():
        now = time.perf_counter()
        if now < next_display_at and stop_event.wait(next_display_at - now):
            break
        next_display_at = max(next_display_at + 1.0 / 60.0, time.perf_counter())
        front_record = front.peek_latest_frame()
        with state.lock:
            side_camera = state.side_camera
            front_landmarks = state.front_landmarks
            front_results = state.front_results
            side_landmarks = state.side_landmarks
            side_results = state.side_results
            extraction_fps = state.extraction_fps
            latency_ms = state.last_latency_ms
            last_gloss = state.last_gloss
            last_confidence = state.last_confidence
            last_status = state.last_status
        side_record = side_camera.peek_latest_frame() if side_camera is not None else None
        capture_fps = front.capture_fps
        if front_record is not None:
            side_index = side_record.frame_index if side_record is not None else -1
            if (front_record.frame_index, side_index) != (last_front_index, last_side_index):
                last_front_index, last_side_index = front_record.frame_index, side_index
                if front_landmarks is None:
                    front_landmarks = LandmarkFrame.missing(
                        timestamp_ns=front_record.timestamp_ns,
                        frame_index=front_record.frame_index,
                    )
                front_display = annotate_frame(
                    cv2.resize(front_record.frame_buffer, DISPLAY_RESOLUTION),
                    front_landmarks, front_results,
                    camera_name="FRONT", fps=capture_fps,
                )
                cv2.putText(
                    front_display,
                    f"Last: {last_gloss} ({last_confidence:.2f}) | {last_status}",
                    (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.54,
                    (0, 215, 255), 2, cv2.LINE_AA,
                )
                if side_record is None:
                    cv2.imshow(WINDOW_TITLE, front_display)
                else:
                    if side_landmarks is None:
                        side_landmarks = LandmarkFrame.missing(
                            timestamp_ns=side_record.timestamp_ns,
                            frame_index=side_record.frame_index,
                        )
                    side_display = annotate_frame(
                        cv2.resize(side_record.frame_buffer, DISPLAY_RESOLUTION),
                        side_landmarks, side_results,
                        camera_name="SIDE", fps=side_camera.capture_fps,
                    )
                    cv2.imshow(WINDOW_TITLE, np.hstack((front_display, side_display)))
        if now - last_report_at >= 1.0:
            print(
                f"[Capture: {capture_fps:.1f} FPS | "
                f"Extraction: {extraction_fps:.1f} FPS | "
                f"Inference Latency: {latency_ms:.1f} ms]",
                flush=True,
            )
            last_report_at = now
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit requested.", flush=True)
            break


def main() -> int:
    args = build_parser().parse_args()
    import cv2

    try:
        run(
            args.single_camera, args.front, args.side,
            weights_path=args.weights_path,
            class_map_path=args.class_map,
            confidence_threshold=args.confidence_threshold,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted; releasing cameras.", flush=True)
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
