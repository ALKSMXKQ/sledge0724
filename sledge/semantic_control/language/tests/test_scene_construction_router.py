from __future__ import annotations

from sledge.semantic_control.language.scene_construction_router import (
    SceneConstructionMode,
    SceneConstructionRouter,
)


def _node(node_type: str, value: str, source: str):
    return {"node_type": node_type, "value": value, "source": source}


def _spec(nodes, completed=None):
    return {
        "hierarchy_layer": {"selected_path": list(nodes)},
        "parameter_layer": {"completed": dict(completed or {})},
    }


def _parameter(value, source="hierarchical_prior"):
    return {"value": value, "source": source}


def test_ego_lane_is_local_and_does_not_trigger_synthesis():
    spec = _spec(
        [
            _node("road_topology", "straight_segment", "hierarchical_default"),
            _node("ego_traffic_space", "single_lane", "hierarchical_default"),
            _node("primary_actor_type", "pedestrian", "normalized_to_nuplan"),
            _node("hazard_interaction", "enter_ego_lane", "inferred"),
            _node("target_region", "ego_lane", "explicit"),
        ],
        completed={
            "lane_width_m": _parameter([3.2, 3.8]),
            "lane_count": _parameter([1, 2]),
            "road_curvature": _parameter([-0.005, 0.005]),
            "actor_speed_mps": _parameter(1.2, "user_input"),
        },
    )

    routed = SceneConstructionRouter().attach(spec)

    assert routed["scene_construction"]["mode"] == SceneConstructionMode.EDIT_EXISTING.value
    assert routed["scene_construction"]["explicit_global_constraints"] == []
    assert "target_region=ego_lane" in routed["scene_construction"]["local_hazard_constraints"]
    policy = routed["parameter_layer"]["execution_policy"]
    assert policy["road_geometry_source"] == "inherit_b0"
    assert set(policy["inactive_parameter_names"]) == {
        "lane_width_m",
        "lane_count",
        "road_curvature",
    }
    assert "actor_speed_mps" in policy["active_parameter_names"]


def test_explicit_intersection_triggers_synthesis():
    spec = _spec(
        [
            _node("road_topology", "intersection", "explicit"),
            _node("ego_traffic_space", "straight_through", "hierarchical_default"),
            _node("hazard_interaction", "occluded_emergence", "explicit"),
            _node("target_region", "ego_path", "inferred"),
        ],
        completed={"lane_width_m": _parameter([3.2, 3.8])},
    )

    routed = SceneConstructionRouter().attach(spec)

    construction = routed["scene_construction"]
    assert construction["mode"] == SceneConstructionMode.SYNTHESIZE_NEW.value
    assert construction["reason"] == "explicit_global_road_structure"
    assert "road_topology=intersection" in construction["explicit_global_constraints"]
    assert routed["parameter_layer"]["execution_policy"]["road_geometry_source"] == "parameter_template"
    assert routed["parameter_layer"]["execution_policy"]["inactive_parameter_names"] == []


def test_explicit_lane_count_parameter_triggers_synthesis():
    spec = _spec(
        [
            _node("road_topology", "straight_segment", "hierarchical_default"),
            _node("ego_traffic_space", "single_lane", "hierarchical_default"),
            _node("target_region", "ego_lane", "explicit"),
        ],
        completed={
            "lane_count": _parameter(2, "user_input"),
            "lane_width_m": _parameter([3.2, 3.8]),
        },
    )

    decision = SceneConstructionRouter().route(spec)

    assert decision.mode == SceneConstructionMode.SYNTHESIZE_NEW
    assert "lane_count=2" in decision.explicit_global_constraints


def test_inferred_bidirectional_road_does_not_trigger_synthesis():
    spec = _spec(
        [
            _node("road_topology", "straight_segment", "hierarchical_default"),
            _node("ego_traffic_space", "bidirectional_road", "inferred"),
            _node("hazard_interaction", "path_crossing", "inferred"),
            _node("target_region", "ego_path", "inferred"),
        ]
    )

    decision = SceneConstructionRouter().route(spec)

    assert decision.mode == SceneConstructionMode.EDIT_EXISTING


def test_explicit_bidirectional_layout_triggers_synthesis():
    spec = _spec(
        [
            _node("road_topology", "straight_segment", "hierarchical_default"),
            _node("ego_traffic_space", "bidirectional_road", "explicit"),
            _node("hazard_interaction", "path_crossing", "inferred"),
        ]
    )

    decision = SceneConstructionRouter().route(spec)

    assert decision.mode == SceneConstructionMode.SYNTHESIZE_NEW
    assert "ego_traffic_space=bidirectional_road" in decision.explicit_global_constraints


def test_explicit_curbs_and_ego_path_remain_edit_existing():
    spec = _spec(
        [
            _node("road_topology", "straight_segment", "hierarchical_default"),
            _node("ego_traffic_space", "curbside_zone", "inferred"),
            _node("source_region", "right_side", "explicit"),
            _node("target_region", "ego_path", "explicit"),
            _node("hazard_interaction", "occluded_emergence", "explicit"),
            _node("auxiliary_entity", "parked_truck_occluder", "explicit"),
        ]
    )

    decision = SceneConstructionRouter().route(spec)

    assert decision.mode == SceneConstructionMode.EDIT_EXISTING
    assert "source_region=right_side" in decision.local_hazard_constraints
    assert "target_region=ego_path" in decision.local_hazard_constraints
