from types import SimpleNamespace

import numpy as np

from sledge.semantic_control.occluded_pedestrian_pipeline.generation.lane_polyline_regularizer import (
    LanePolylineRegularizerConfig,
    regularize_polyline,
    regularize_sledge_lines,
)


def test_wiggly_straight_line_is_regularized_without_moving_endpoints():
    x = np.linspace(0.0, 20.0, 20)
    y = 0.35 * np.sin(np.linspace(0.0, 8.0 * np.pi, 20))
    points = np.stack([x, y], axis=1)

    corrected, report = regularize_polyline(
        points,
        LanePolylineRegularizerConfig(),
    )

    assert report["applied"] is True
    np.testing.assert_allclose(corrected[0], points[0])
    np.testing.assert_allclose(corrected[-1], points[-1])
    assert (
        report["after"]["excess_turn_rad"]
        < report["before"]["excess_turn_rad"]
    )
    assert report["max_point_displacement_m"] <= 0.80 + 1e-6


def test_smooth_real_curve_is_not_straightened():
    theta = np.linspace(0.0, np.deg2rad(60.0), 20)
    radius = 30.0
    points = np.stack(
        [
            radius * np.sin(theta),
            radius * (1.0 - np.cos(theta)),
        ],
        axis=1,
    )

    corrected, report = regularize_polyline(
        points,
        LanePolylineRegularizerConfig(),
    )

    assert report["applied"] is False
    assert report["reason"] == "not_irregular"
    np.testing.assert_allclose(corrected, points)


def test_only_valid_generated_lines_are_modified():
    x = np.linspace(0.0, 20.0, 20)
    wiggly = np.stack(
        [x, 0.40 * np.sin(np.linspace(0.0, 8.0 * np.pi, 20))],
        axis=1,
    )
    smooth = np.stack([x, np.zeros_like(x)], axis=1)
    inactive = wiggly.copy()

    states = np.stack([wiggly, smooth, inactive], axis=0).astype(np.float32)
    original = states.copy()
    vector = SimpleNamespace(
        lines=SimpleNamespace(
            states=states,
            mask=np.asarray([0.95, 0.95, 0.10], dtype=np.float32),
        )
    )

    report = regularize_sledge_lines(vector)

    assert report["valid_line_count"] == 2
    assert report["regularized_line_count"] == 1
    assert not np.allclose(vector.lines.states[0], original[0])
    np.testing.assert_allclose(vector.lines.states[1], original[1])
    np.testing.assert_allclose(vector.lines.states[2], original[2])
