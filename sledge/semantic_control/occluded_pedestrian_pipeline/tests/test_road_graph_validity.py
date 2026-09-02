from __future__ import annotations

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeVector,
    SledgeVectorElement,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.road_graph_validity import (
    evaluate_global_road_graph_validity,
)


def _resample_line(points, num_points: int = 20):
    points = np.asarray(points, dtype=np.float32)
    src = np.linspace(0.0, 1.0, len(points))
    dst = np.linspace(0.0, 1.0, num_points)
    x = np.interp(dst, src, points[:, 0])
    y = np.interp(dst, src, points[:, 1])
    return np.stack([x, y], axis=-1).astype(np.float32)


def _empty_element(num_rows: int, state_dim: int):
    return SledgeVectorElement(
        states=np.zeros((num_rows, state_dim), dtype=np.float32),
        mask=np.zeros((num_rows,), dtype=np.float32),
    )


def _make_scene(lines):
    line_states = np.zeros((50, 20, 2), dtype=np.float32)
    line_mask = np.zeros((50,), dtype=np.float32)
    for index, points in enumerate(lines):
        line_states[index] = _resample_line(points)
        line_mask[index] = 1.0

    return SledgeVector(
        lines=SledgeVectorElement(states=line_states, mask=line_mask),
        vehicles=_empty_element(50, 6),
        pedestrians=_empty_element(20, 6),
        static_objects=_empty_element(30, 5),
        green_lights=SledgeVectorElement(
            states=np.zeros((20, 20, 2), dtype=np.float32),
            mask=np.zeros((20,), dtype=np.float32),
        ),
        red_lights=SledgeVectorElement(
            states=np.zeros((20, 20, 2), dtype=np.float32),
            mask=np.zeros((20,), dtype=np.float32),
        ),
        ego=SledgeVectorElement(
            states=np.zeros((4,), dtype=np.float32),
            mask=np.asarray(1.0, dtype=np.float32),
        ),
    )


def test_connected_generated_road_graph_passes():
    scene = _make_scene(
        [
            [[-12.0, 0.0], [0.0, 0.0], [12.0, 0.0]],
            [[12.5, 0.0], [24.0, 0.0], [36.0, 0.0]],
            [[36.5, 0.0], [48.0, 0.0], [60.0, 0.0]],
        ]
    )
    result = evaluate_global_road_graph_validity(scene)
    assert result["passed"], result
    assert result["checks"]["ego_route_length_ok"]
    assert result["checks"]["ego_reachable_lane_count_ok"]
    assert result["orphan_lane_length_ratio"] == 0.0


def test_fragmented_generated_road_graph_fails():
    scene = _make_scene(
        [
            [[-12.0, 0.0], [0.0, 0.0], [10.0, 0.0]],
            [[18.0, 10.0], [28.0, 10.0], [38.0, 10.0]],
            [[-30.0, -18.0], [-20.0, -18.0], [-10.0, -18.0]],
            [[22.0, -25.0], [22.0, -15.0], [22.0, -5.0]],
        ]
    )
    result = evaluate_global_road_graph_validity(scene)
    assert not result["passed"]
    assert result["num_weak_components"] >= 3
    assert (
        not result["checks"]["largest_component_ratio_ok"]
        or not result["checks"]["orphan_fragment_ratio_ok"]
        or not result["checks"]["ego_reachable_lane_count_ok"]
    )


def test_large_floating_road_is_rejected_even_with_valid_ego_route():
    scene = _make_scene(
        [
            [[-12.0, 0.0], [0.0, 0.0], [12.0, 0.0]],
            [[12.5, 0.0], [24.0, 0.0], [36.0, 0.0]],
            [[36.5, 0.0], [48.0, 0.0], [60.0, 0.0]],
            [[-30.0, 20.0], [0.0, 20.0], [30.0, 20.0]],
            [[30.5, 20.0], [50.0, 20.0], [70.0, 20.0]],
        ]
    )
    result = evaluate_global_road_graph_validity(scene)
    assert not result["passed"]
    assert not result["checks"]["orphan_fragment_ratio_ok"]


def test_single_lane_scene_is_not_globally_valid():
    scene = _make_scene(
        [
            [[-10.0, 0.0], [0.0, 0.0], [10.0, 0.0]],
        ]
    )
    result = evaluate_global_road_graph_validity(scene)
    assert not result["passed"]
    assert not result["checks"]["ego_component_size_ok"]
    assert not result["checks"]["ego_reachable_lane_count_ok"]
