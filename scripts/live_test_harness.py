"""Interactive live verification for the ASL capture and landmark pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from asl_stereo.capture import CameraWorker, Synchronizer  # noqa: E402
from asl_stereo.contracts import LandmarkFrame, SynchronizedFramePair  # noqa: E402
from asl_stereo.landmarks import HolisticExtractor, draw_asl_overlay  # noqa: E402

WINDOW_TITLE = "ASL Live Test"
DISPLAY_RESOLUTION = (640, 480)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify live ASL landmark extraction using one or two webcams."
    )
    parser.add_argument(
        "--single-camera",
        action="store_true",
        help="Run only the front camera (dual-camera mode is the default).",
    )
    parser.add_argument("--front", type=int, default=0, help="Front camera index.")
    parser.add_argument("--side", type=int, default=1, help="Side camera index.")
    return parser


class FpsMeter:
    """Low-noise exponential moving-average FPS meter."""

    def __init__(self) -> None:
        self._last_time: float | None = None
        self.fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last_time is not None:
            elapsed = now - self._last_time
            if elapsed > 0.0:
                instantaneous = 1.0 / elapsed
                self.fps = instantaneous if self.fps == 0.0 else 0.9 * self.fps + 0.1 * instantaneous
        self._last_time = now
        return self.fps


def annotate_frame(
    frame: np.ndarray,
    landmarks: LandmarkFrame,
    results: object,
    *,
    camera_name: str,
    fps: float,
    sync_delta_ms: float | None = None,
) -> np.ndarray:
    """Draw contract-approved geometry and lightweight live telemetry."""
    import cv2

    annotated = draw_asl_overlay(frame.copy(), results)
    present = landmarks.hand_present.astype(bool)
    hands = f"L:{'ON' if present[0] else 'LOST'} R:{'ON' if present[1] else 'LOST'}"
    joint_count = int(landmarks.joint_mask.sum())
    lines = [
        f"{camera_name} | FPS {fps:5.1f}",
        f"Hands {hands} | Joints {joint_count}/46",
    ]
    if sync_delta_ms is not None:
        lines.append(f"Sync delta {sync_delta_ms:+.1f} ms")
    for line_index, text in enumerate(lines):
        origin = (12, 28 + line_index * 26)
        cv2.putText(
            annotated,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (80, 255, 120),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _process_pair(
    pair: SynchronizedFramePair,
    front_extractor: HolisticExtractor,
    side_extractor: HolisticExtractor,
    fps: float,
) -> np.ndarray:
    import cv2

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
    delta_ms = pair.delta_t_ns / 1_000_000.0
    front = annotate_frame(
        pair.front.frame_buffer,
        front_landmarks,
        front_extractor.last_results,
        camera_name="FRONT",
        fps=fps,
        sync_delta_ms=delta_ms,
    )
    side = annotate_frame(
        pair.side.frame_buffer,
        side_landmarks,
        side_extractor.last_results,
        camera_name="SIDE",
        fps=fps,
        sync_delta_ms=delta_ms,
    )
    front = cv2.resize(front, DISPLAY_RESOLUTION)
    side = cv2.resize(side, DISPLAY_RESOLUTION)
    return np.hstack((front, side))


def run(single_camera: bool, front_index: int, side_index: int) -> None:
    import cv2

    if front_index < 0 or side_index < 0:
        raise ValueError("camera indices must be non-negative")
    if not single_camera and front_index == side_index:
        raise ValueError("front and side camera indices must differ")

    meter = FpsMeter()
    with ExitStack() as stack:
        front_worker = stack.enter_context(
            CameraWorker(
                front_index,
                camera_id="front",
                resolution=DISPLAY_RESOLUTION,
            )
        )
        front_extractor = stack.enter_context(HolisticExtractor())

        if single_camera:
            while True:
                frame = front_worker.get_frame(timeout=0.05)
                if frame is not None:
                    landmarks = front_extractor.process(
                        frame.frame_buffer,
                        timestamp_ns=frame.timestamp_ns,
                        frame_index=frame.frame_index,
                    )
                    display = annotate_frame(
                        frame.frame_buffer,
                        landmarks,
                        front_extractor.last_results,
                        camera_name="FRONT",
                        fps=meter.tick(),
                    )
                    cv2.imshow(WINDOW_TITLE, cv2.resize(display, DISPLAY_RESOLUTION))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            return

        side_worker = stack.enter_context(
            CameraWorker(
                side_index,
                camera_id="side",
                resolution=DISPLAY_RESOLUTION,
            )
        )
        side_extractor = stack.enter_context(HolisticExtractor())
        synchronizer = Synchronizer()
        while True:
            pair = None
            front_frame = front_worker.get_frame()
            if front_frame is not None:
                pair = synchronizer.add_front(front_frame)
            side_frame = side_worker.get_frame()
            if side_frame is not None:
                side_pair = synchronizer.add_side(side_frame)
                if side_pair is not None:
                    pair = side_pair
            if pair is not None:
                cv2.imshow(
                    WINDOW_TITLE,
                    _process_pair(pair, front_extractor, side_extractor, meter.tick()),
                )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


def main() -> int:
    args = build_parser().parse_args()
    import cv2

    try:
        run(args.single_camera, args.front, args.side)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
