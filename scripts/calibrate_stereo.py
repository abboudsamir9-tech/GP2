"""Interactive, synchronized dual-camera checkerboard calibration.

Only detected corner coordinates are retained during capture. Live frames are
displayed from RAM and are never written to disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from asl_stereo.capture import CameraWorker, Synchronizer

LOGGER = logging.getLogger("asl_stereo.calibration")


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    image_size: tuple[int, int]
    K_front: npt.NDArray[np.float64]
    distortion_front: npt.NDArray[np.float64]
    K_side: npt.NDArray[np.float64]
    distortion_side: npt.NDArray[np.float64]
    R: npt.NDArray[np.float64]
    t: npt.NDArray[np.float64]
    E: npt.NDArray[np.float64]
    F: npt.NDArray[np.float64]
    P_front: npt.NDArray[np.float64]
    P_side: npt.NDArray[np.float64]
    rms_front_px: float
    rms_side_px: float
    rms_stereo_px: float
    valid_pair_count: int


@dataclass(slots=True)
class CalibrationSamples:
    board_size: tuple[int, int]
    square_size_mm: float
    object_points: list[npt.NDArray[np.float32]] = field(default_factory=list)
    front_points: list[npt.NDArray[np.float32]] = field(default_factory=list)
    side_points: list[npt.NDArray[np.float32]] = field(default_factory=list)
    image_size: tuple[int, int] | None = None

    def add(
        self,
        front_corners: npt.NDArray[np.float32],
        side_corners: npt.NDArray[np.float32],
        image_size: tuple[int, int],
    ) -> None:
        expected_count = self.board_size[0] * self.board_size[1]
        if front_corners.shape != (expected_count, 1, 2):
            raise ValueError("front checkerboard corners have an invalid shape")
        if side_corners.shape != (expected_count, 1, 2):
            raise ValueError("side checkerboard corners have an invalid shape")
        if self.image_size is not None and image_size != self.image_size:
            raise ValueError("camera resolution changed during calibration")
        self.image_size = image_size
        self.object_points.append(make_object_points(self.board_size, self.square_size_mm))
        self.front_points.append(np.array(front_corners, dtype=np.float32, copy=True))
        self.side_points.append(np.array(side_corners, dtype=np.float32, copy=True))

    def __len__(self) -> int:
        return len(self.object_points)


def make_object_points(
    board_size: tuple[int, int], square_size_mm: float
) -> npt.NDArray[np.float32]:
    columns, rows = board_size
    if columns < 2 or rows < 2:
        raise ValueError("checkerboard dimensions must each be at least two")
    if not np.isfinite(square_size_mm) or square_size_mm <= 0:
        raise ValueError("square size must be a positive finite value")
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    points[:, :2] *= np.float32(square_size_mm)
    return points


def detect_checkerboard(
    frame: npt.NDArray[np.uint8], board_size: tuple[int, int]
) -> tuple[bool, npt.NDArray[np.float32] | None]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    detected, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if not detected or corners is None:
        return False, None
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-3,
    )
    refined = cv2.cornerSubPix(
        gray,
        corners.astype(np.float32, copy=False),
        (11, 11),
        (-1, -1),
        criteria,
    )
    return True, refined


def solve_stereo_calibration(samples: CalibrationSamples) -> CalibrationResult:
    if samples.image_size is None or len(samples) < 3:
        raise ValueError("at least three valid checkerboard pairs are required")

    rms_front, K_front, distortion_front, _, _ = cv2.calibrateCamera(
        samples.object_points,
        samples.front_points,
        samples.image_size,
        None,
        None,
    )
    rms_side, K_side, distortion_side, _, _ = cv2.calibrateCamera(
        samples.object_points,
        samples.side_points,
        samples.image_size,
        None,
        None,
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-6,
    )
    (
        rms_stereo,
        K_front,
        distortion_front,
        K_side,
        distortion_side,
        rotation,
        translation,
        essential,
        fundamental,
    ) = cv2.stereoCalibrate(
        samples.object_points,
        samples.front_points,
        samples.side_points,
        K_front,
        distortion_front,
        K_side,
        distortion_side,
        samples.image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    identity_extrinsic = np.column_stack(
        (np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    )
    side_extrinsic = np.column_stack((rotation, translation.reshape(3)))
    P_front = K_front @ identity_extrinsic
    P_side = K_side @ side_extrinsic
    return CalibrationResult(
        image_size=samples.image_size,
        K_front=np.asarray(K_front, dtype=np.float64),
        distortion_front=np.asarray(distortion_front, dtype=np.float64),
        K_side=np.asarray(K_side, dtype=np.float64),
        distortion_side=np.asarray(distortion_side, dtype=np.float64),
        R=np.asarray(rotation, dtype=np.float64),
        t=np.asarray(translation, dtype=np.float64).reshape(3, 1),
        E=np.asarray(essential, dtype=np.float64),
        F=np.asarray(fundamental, dtype=np.float64),
        P_front=np.asarray(P_front, dtype=np.float64),
        P_side=np.asarray(P_side, dtype=np.float64),
        rms_front_px=float(rms_front),
        rms_side_px=float(rms_side),
        rms_stereo_px=float(rms_stereo),
        valid_pair_count=len(samples),
    )


def export_calibration(
    result: CalibrationResult,
    json_path: str | Path,
    npz_path: str | Path,
    *,
    board_size: tuple[int, int],
    square_size_mm: float,
) -> None:
    json_output = Path(json_path)
    npz_output = Path(npz_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    npz_output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "board_size_internal_corners": list(board_size),
        "square_size_mm": square_size_mm,
        "image_size": list(result.image_size),
        "valid_pair_count": result.valid_pair_count,
        "rms_front_px": result.rms_front_px,
        "rms_side_px": result.rms_side_px,
        "rms_stereo_px": result.rms_stereo_px,
        "K_front": result.K_front.tolist(),
        "distortion_front": result.distortion_front.tolist(),
        "K_side": result.K_side.tolist(),
        "distortion_side": result.distortion_side.tolist(),
        "R": result.R.tolist(),
        "t": result.t.tolist(),
        "E": result.E.tolist(),
        "F": result.F.tolist(),
        "P_front": result.P_front.tolist(),
        "P_side": result.P_side.tolist(),
    }
    json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        npz_output,
        P_front=result.P_front,
        P_side=result.P_side,
        K_front=result.K_front,
        distortion_front=result.distortion_front,
        K_side=result.K_side,
        distortion_side=result.distortion_side,
        R=result.R,
        t=result.t,
        E=result.E,
        F=result.F,
    )


def run_interactive(args: argparse.Namespace) -> int:
    board_size = (args.grid_columns, args.grid_rows)
    samples = CalibrationSamples(board_size, args.square_size_mm)
    synchronizer = Synchronizer(
        front_camera_id="front",
        side_camera_id="side",
        tolerance_ns=round(args.sync_tolerance_ms * 1_000_000),
    )
    latest_display: npt.NDArray[np.uint8] | None = None
    latest_corners: tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        tuple[int, int],
    ] | None = None
    latest_pair_index: int | None = None
    last_captured_pair_index: int | None = None
    window_name = "ASL Stereo Calibration"

    LOGGER.info(
        "Starting cameras %d and %d; board=%dx%d, square=%.2f mm",
        args.front_camera,
        args.side_camera,
        board_size[0],
        board_size[1],
        args.square_size_mm,
    )
    try:
        with CameraWorker(args.front_camera, camera_id="front") as front_camera, CameraWorker(
            args.side_camera, camera_id="side"
        ) as side_camera:
            while True:
                pair = None
                front = front_camera.get_frame()
                if front is not None:
                    pair = synchronizer.add_front(front)
                side = side_camera.get_frame()
                if side is not None:
                    side_pair = synchronizer.add_side(side)
                    pair = side_pair or pair

                if pair is not None:
                    front_view = pair.front.frame_buffer.copy()
                    side_view = pair.side.frame_buffer.copy()
                    if front_view.shape[:2] != side_view.shape[:2]:
                        LOGGER.error("Both cameras must use the same resolution")
                        return 2
                    image_size = (front_view.shape[1], front_view.shape[0])
                    front_found, front_corners = detect_checkerboard(front_view, board_size)
                    side_found, side_corners = detect_checkerboard(side_view, board_size)
                    if front_found and front_corners is not None:
                        cv2.drawChessboardCorners(
                            front_view, board_size, front_corners, front_found
                        )
                    if side_found and side_corners is not None:
                        cv2.drawChessboardCorners(
                            side_view, board_size, side_corners, side_found
                        )
                    latest_corners = (
                        front_corners,
                        side_corners,
                        image_size,
                    ) if front_found and side_found else None
                    latest_pair_index = pair.pair_index
                    latest_display = _compose_display(
                        front_view,
                        side_view,
                        len(samples),
                        args.min_pairs,
                        pair.delta_t_ns / 1_000_000.0,
                        latest_corners is not None,
                    )

                if latest_display is not None:
                    cv2.imshow(window_name, latest_display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    LOGGER.info("Calibration aborted; no artifacts were written")
                    return 1
                if key == ord(" "):
                    if latest_corners is None:
                        LOGGER.warning("Snapshot rejected: checkerboard not found in both views")
                    elif latest_pair_index == last_captured_pair_index:
                        LOGGER.warning("Snapshot rejected: this synchronized pair is already saved")
                    else:
                        front_corners, side_corners, image_size = latest_corners
                        samples.add(front_corners, side_corners, image_size)
                        last_captured_pair_index = latest_pair_index
                        LOGGER.info("Captured valid pair %d/%d", len(samples), args.min_pairs)
                if key == ord("c"):
                    if len(samples) < args.min_pairs:
                        LOGGER.warning(
                            "Need at least %d valid pairs; currently have %d",
                            args.min_pairs,
                            len(samples),
                        )
                        continue
                    result = solve_stereo_calibration(samples)
                    _log_result(result)
                    export_calibration(
                        result,
                        args.json_output,
                        args.npz_output,
                        board_size=board_size,
                        square_size_mm=args.square_size_mm,
                    )
                    LOGGER.info("Wrote %s and %s", args.json_output, args.npz_output)
                    return 0
    finally:
        cv2.destroyAllWindows()


def _compose_display(
    front: npt.NDArray[np.uint8],
    side: npt.NDArray[np.uint8],
    pair_count: int,
    minimum_pairs: int,
    delta_ms: float,
    board_ready: bool,
) -> npt.NDArray[np.uint8]:
    display = np.concatenate((front, side), axis=1)
    status = "READY - SPACE TO CAPTURE" if board_ready else "SHOW BOARD IN BOTH CAMERAS"
    lines = (
        f"Valid pairs: {pair_count}/{minimum_pairs}",
        f"Sync delta: {delta_ms:+.1f} ms",
        status,
        "SPACE capture   C calibrate   Q quit",
    )
    for index, text in enumerate(lines):
        color = (80, 230, 80) if board_ready and index == 2 else (255, 255, 255)
        cv2.putText(
            display,
            text,
            (18, 32 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return display


def _log_result(result: CalibrationResult) -> None:
    LOGGER.info("Front camera RMS: %.4f px", result.rms_front_px)
    LOGGER.info("Side camera RMS: %.4f px", result.rms_side_px)
    LOGGER.info("Stereo RMS: %.4f px", result.rms_stereo_px)
    if result.rms_stereo_px > 1.0:
        LOGGER.warning(
            "Stereo RMS exceeds 1.0 px; capture more diverse checkerboard poses"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate synchronized front and side webcams with a checkerboard."
    )
    parser.add_argument("--front-camera", type=int, default=0)
    parser.add_argument("--side-camera", type=int, default=1)
    parser.add_argument("--grid-columns", type=int, default=9)
    parser.add_argument("--grid-rows", type=int, default=6)
    parser.add_argument("--square-size-mm", type=float, default=25.0)
    parser.add_argument("--min-pairs", type=int, default=15)
    parser.add_argument("--sync-tolerance-ms", type=float, default=40.0)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "configs" / "calibration_params.json",
    )
    parser.add_argument(
        "--npz-output",
        type=Path,
        default=PROJECT_ROOT / "calibration" / "projection_matrices.npz",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = build_parser().parse_args(argv)
    if args.min_pairs < 3:
        raise SystemExit("--min-pairs must be at least 3")
    if args.front_camera == args.side_camera:
        raise SystemExit("front and side camera indices must differ")
    if args.sync_tolerance_ms < 0:
        raise SystemExit("--sync-tolerance-ms must be non-negative")
    make_object_points((args.grid_columns, args.grid_rows), args.square_size_mm)
    return run_interactive(args)


if __name__ == "__main__":
    raise SystemExit(main())
