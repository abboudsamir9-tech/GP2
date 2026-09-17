from pathlib import Path

import numpy as np

from asl_stereo.contracts import JOINT_COUNT
from asl_stereo.contracts.landmarks import LEFT_HAND_SLICE
from asl_stereo.stereo import (
    JointStatus,
    StereoCalibration,
    StereoMatcher,
    is_triangulation_degenerate,
    triangulate_dlt,
)
from tests.fixtures.synthetic_cameras import make_projection_matrices, project_points


def test_dlt_reconstructs_known_3d_point() -> None:
    front_projection, side_projection = make_projection_matrices()
    ground_truth = np.array([[0.15, -0.08, 2.5]], dtype=np.float64)
    front_pixel = project_points(front_projection, ground_truth)[0]
    side_pixel = project_points(side_projection, ground_truth)[0]

    reconstructed, reprojection_error = triangulate_dlt(
        front_pixel, side_pixel, front_projection, side_projection
    )

    np.testing.assert_allclose(reconstructed, ground_truth[0], atol=1e-9)
    assert reprojection_error < 1e-9


def test_near_parallel_rays_are_degenerate() -> None:
    front_projection, side_projection = make_projection_matrices(baseline=1e-6)
    point = np.array([0.1, 0.1, 3.0])
    assert is_triangulation_degenerate(
        point,
        front_projection,
        side_projection,
        reprojection_error=0.0,
        min_ray_angle_degrees=1.0,
    )


def test_point_behind_either_camera_is_degenerate() -> None:
    front_projection, side_projection = make_projection_matrices()
    point = np.array([0.1, 0.1, -3.0])
    assert is_triangulation_degenerate(
        point, front_projection, side_projection, reprojection_error=0.0
    )


def test_side_hand_occlusion_falls_back_per_joint() -> None:
    front_projection, side_projection = make_projection_matrices()
    calibration = StereoCalibration(front_projection, side_projection)
    matcher = StereoMatcher(calibration)

    ground_truth = np.empty((JOINT_COUNT, 3), dtype=np.float64)
    ground_truth[:, 0] = np.linspace(-0.3, 0.3, JOINT_COUNT)
    ground_truth[:, 1] = np.linspace(-0.2, 0.2, JOINT_COUNT)
    ground_truth[:, 2] = 3.0
    front = np.column_stack(
        (project_points(front_projection, ground_truth), ground_truth[:, 2])
    ).astype(np.float32)
    side = np.column_stack(
        (project_points(side_projection, ground_truth), ground_truth[:, 2])
    ).astype(np.float32)
    side[LEFT_HAND_SLICE] = np.nan

    result = matcher.match(front, side)

    assert result.coordinates.shape == (46, 3)
    assert result.statuses[:21] == (JointStatus.FRONT_FALLBACK,) * 21
    assert result.statuses[21:] == (JointStatus.TRIANGULATED,) * 25
    np.testing.assert_array_equal(result.coordinates[:21], front[:21])
    np.testing.assert_allclose(result.coordinates[21:], ground_truth[21:], atol=1e-5)


def test_missing_calibration_enables_graceful_front_fallback(tmp_path: Path) -> None:
    calibration = StereoCalibration.from_file(tmp_path / "missing.json")
    assert not calibration.available
    front = np.ones((JOINT_COUNT, 3), dtype=np.float32)
    side = np.ones((JOINT_COUNT, 3), dtype=np.float32)

    result = StereoMatcher(calibration).match(front, side)

    assert result.statuses == (JointStatus.FRONT_FALLBACK,) * JOINT_COUNT
    np.testing.assert_array_equal(result.coordinates, front)


def test_frame_fallback_never_mixes_triangulated_and_front_coordinates() -> None:
    front_projection, side_projection = make_projection_matrices()
    calibration = StereoCalibration(front_projection, side_projection)
    matcher = StereoMatcher(calibration, fallback_scope="frame")
    ground_truth = np.column_stack(
        (
            np.linspace(-0.3, 0.3, JOINT_COUNT),
            np.linspace(-0.2, 0.2, JOINT_COUNT),
            np.full(JOINT_COUNT, 3.0),
        )
    )
    front = np.column_stack(
        (project_points(front_projection, ground_truth), ground_truth[:, 2])
    ).astype(np.float32)
    side = np.column_stack(
        (project_points(side_projection, ground_truth), ground_truth[:, 2])
    ).astype(np.float32)
    normalized_fallback = np.full((JOINT_COUNT, 3), 0.25, dtype=np.float32)
    side[0] = np.nan

    result = matcher.match(front, side, front_fallback=normalized_fallback)

    assert result.statuses == (JointStatus.FRONT_FALLBACK,) * JOINT_COUNT
    np.testing.assert_array_equal(result.coordinates, normalized_fallback)
