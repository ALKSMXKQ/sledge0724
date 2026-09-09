from __future__ import annotations

import math

import numpy as np

from sledge.semantic_control.occluded_pedestrian_pipeline.generation.road_constraint_context import (
    RoadConstraintContext,
)


def _line(x0: float, x1: float, y: float, samples: int = 61) -> np.ndarray:
    x = np.linspace(x0, x1, samples)
    return np.column_stack([x, np.full_like(x, y)])


def _left_turn_boundary(
    radius: float,
    *,
    center_x: float = 0.0,
    center_y: float = 20.0,
    samples: int = 121,
) -> np.ndarray:
    theta = np.linspace(-math.pi / 2.0, 0.0, samples)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    return np.column_stack([x, y])


def test_straight_corridor_advances_by_arc_length() -> None:
    context = RoadConstraintContext.from_components(
        {
            "right": _line(-10.0, 30.0, -1.75),
            "left": _line(-10.0, 30.0, 1.75),
        },
        edges=[],
    )

    corridor = context.infer_ego_corridor(
        ego_heading=0.0,
        lane_width_hint_m=3.5,
    )
    frame = context.frame_at_route_distance(
        corridor,
        route_distance_m=12.0,
        lane_width_hint_m=3.5,
    )

    np.testing.assert_allclose(frame.center_xy, [12.0, 0.0], atol=0.08)
    assert abs(frame.heading) < 0.02
    assert abs(frame.lane_width_m - 3.5) < 0.05
    np.testing.assert_allclose(frame.normal, [0.0, 1.0], atol=0.02)


def test_curved_corridor_uses_local_tangent_and_normal() -> None:
    # Centerline radius=20 m.  For a left turn the left boundary is the inner
    # radius and the right boundary is the outer radius.
    context = RoadConstraintContext.from_components(
        {
            "right": _left_turn_boundary(21.75),
            "left": _left_turn_boundary(18.25),
        },
        edges=[],
    )

    corridor = context.infer_ego_corridor(
        ego_heading=0.0,
        lane_width_hint_m=3.5,
    )
    frame = context.frame_at_route_distance(
        corridor,
        route_distance_m=15.0,
        lane_width_hint_m=3.5,
    )

    # The old x=v*TTC approximation would keep heading near zero.  Arc-length
    # advancement must rotate the local frame substantially into the bend.
    assert 0.45 < frame.heading < 1.05
    assert 2.8 < frame.lane_width_m < 4.2
    assert frame.center_xy[0] > 10.0
    assert frame.center_xy[1] > 3.0
    assert abs(float(np.dot(frame.tangent, frame.normal))) < 1e-6


def test_intersection_connector_continues_across_graph_edge() -> None:
    # Straight approach ending at x=5, followed by a 90-degree left connector.
    paths = {
        "right_approach": _line(-10.0, 5.0, -1.75, samples=41),
        "left_approach": _line(-10.0, 5.0, 1.75, samples=41),
        "right_connector": _left_turn_boundary(
            21.75,
            center_x=5.0,
            center_y=20.0,
        ),
        "left_connector": _left_turn_boundary(
            18.25,
            center_x=5.0,
            center_y=20.0,
        ),
    }
    context = RoadConstraintContext.from_components(
        paths,
        edges=[
            ("right_approach", "right_connector"),
            ("left_approach", "left_connector"),
        ],
    )

    corridor = context.infer_ego_corridor(
        ego_heading=0.0,
        lane_width_hint_m=3.5,
    )
    frame = context.frame_at_route_distance(
        corridor,
        route_distance_m=13.0,
        lane_width_hint_m=3.5,
    )

    assert frame.right_route_path_ids == (
        "right_approach",
        "right_connector",
    )
    assert frame.left_route_path_ids == (
        "left_approach",
        "left_connector",
    )
    assert frame.heading > 0.20
    assert frame.center_xy[0] > 7.0


def test_successor_selection_prefers_heading_continuity() -> None:
    # One boundary path has two graph successors.  The straight continuation
    # should beat a sharp turn when no explicit route semantic is available.
    context = RoadConstraintContext.from_components(
        {
            "approach": _line(-5.0, 5.0, 0.0, samples=21),
            "straight": _line(5.0, 20.0, 0.0, samples=31),
            "turn": np.column_stack(
                [
                    np.full(31, 5.0),
                    np.linspace(0.0, 15.0, 31),
                ]
            ),
        },
        edges=[("approach", "straight"), ("approach", "turn")],
    )
    start = context.project_to_path("approach", (0.0, 0.0)).pose
    pose, route = context.advance(start, 12.0)

    assert route == ["approach", "straight"]
    assert pose.path_id == "straight"
    assert abs(pose.heading) < 0.02
