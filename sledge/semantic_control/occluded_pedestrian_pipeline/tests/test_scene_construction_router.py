import pytest

from sledge.semantic_control.occluded_pedestrian_pipeline.language.scene_construction_router import (
    EDIT_EXISTING,
    SYNTHESIZE_NEW,
    SceneConstructionRouter,
)


def _spec():
    return {
        "hierarchy_layer": {
            "path_values": {
                "road_topology": "straight_segment",
                "ego_traffic_space": "curbside_zone",
                "primary_actor_type": "pedestrian",
                "hazard_interaction": "occluded_emergence",
                "auxiliary_entity": "parked_car_occluder",
                "source_region": "curbside",
                "target_region": "ego_path",
                "visibility": "fully_occluded",
                "motion_direction": "occluder_to_ego_path",
                "risk_level": "moderate",
            }
        },
        "parameter_layer": {
            "completed": {}
        },
    }


@pytest.mark.parametrize(
    "prompt",
    [
        "A pedestrian enters the ego lane from behind a parked car.",
        "A child emerges from behind a parked truck into the ego lane.",
        "A jogger hidden behind a bus on the right side rushes toward the ego path.",
        "A pedestrian appears from behind a roadside barrier.",
        "A girl emerges from a parked car at the curb and moves toward the ego path.",
    ],
)
def test_local_hazard_language_routes_to_edit_existing(prompt: str) -> None:
    plan = SceneConstructionRouter().route(
        prompt=prompt,
        hierarchical_spec=_spec(),
    )
    assert plan.mode == EDIT_EXISTING
    assert plan.explicit_global_constraints == {}
    assert plan.source_scene_usage == "road_fixed_elastic_semantic_base_scene"
    assert plan.hard_preserve_from_b0 == [
        "road_geometry",
        "lane_geometry",
        "road_topology",
    ]
    assert "ego_state" in plan.semantic_edit_controls
    assert "ego_speed_mps" in plan.sample_from_template
    assert "ego_acceleration_mps2" in plan.sample_from_template
    assert "background_vehicles" in plan.elastic_context
    assert plan.context_edit_policy["allow_removal"] is True
    assert plan.context_edit_policy["allow_reposition"] is True
    assert plan.context_edit_policy["removal_scope"] == "local_occluder_reserved_region_only"
    assert plan.context_edit_policy["traffic_lights_removable"] is False


@pytest.mark.parametrize(
    ("prompt", "key", "expected"),
    [
        (
            "At an intersection, a pedestrian emerges from behind a bus.",
            "road_topology",
            "intersection",
        ),
        (
            "At a roundabout, a pedestrian is hidden behind a van.",
            "road_topology",
            "roundabout",
        ),
        (
            "On a two-lane road, a pedestrian emerges behind a car.",
            "lane_count",
            2,
        ),
        (
            "On a bidirectional road, a pedestrian crosses from behind a truck.",
            "directionality",
            "bidirectional",
        ),
        (
            "In a work zone with one closed lane, a pedestrian emerges.",
            "road_topology",
            "work_zone",
        ),
        (
            "On a straight road, a pedestrian emerges behind a car.",
            "road_topology",
            "straight_segment",
        ),
    ],
)
def test_explicit_global_structure_routes_to_synthesis(
    prompt: str,
    key: str,
    expected,
) -> None:
    plan = SceneConstructionRouter().route(
        prompt=prompt,
        hierarchical_spec=_spec(),
    )
    assert plan.mode == SYNTHESIZE_NEW
    assert plan.explicit_global_constraints[key] == expected
    assert plan.source_scene_usage == "blank_capacity_scaffold_only"


def test_explicit_lane_width_routes_to_synthesis() -> None:
    plan = SceneConstructionRouter().route(
        prompt=(
            "On a road with lane width 3.7 m, a pedestrian emerges "
            "from behind a parked car."
        ),
        hierarchical_spec=_spec(),
    )
    assert plan.mode == SYNTHESIZE_NEW
    assert plan.explicit_global_constraints["lane_width_m"] == pytest.approx(3.7)


def test_hierarchy_default_road_value_does_not_trigger_synthesis() -> None:
    # The spec already contains straight_segment, but the prompt says nothing
    # about global road structure. The original prompt remains authoritative.
    plan = SceneConstructionRouter().route(
        prompt="A pedestrian emerges from behind a parked car.",
        hierarchical_spec=_spec(),
    )
    assert plan.mode == EDIT_EXISTING


def test_current_twelve_diagnostic_prompts_all_route_to_edit_existing() -> None:
    prompts = [
        "A child suddenly emerges from behind a parked truck into the ego lane.",
        "An adult pedestrian steps out from behind a parked car on the left roadside and enters the ego lane at 1.2 m/s.",
        "A jogger hidden behind a bus on the right side rushes toward the ego path.",
        "A person becomes visible from behind a roadside van and crosses into the ego lane at 1.6 m/s.",
        "A wheelchair user emerges from behind a road barrier on the left and moves into the ego path.",
        "A pedestrian is fully hidden by a parked truck at the curb, then abruptly enters the ego lane.",
        "A slowly moving pedestrian appears from behind a parked car on the right roadside and enters the ego lane at 0.8 m/s.",
        "A pedestrian suddenly comes out from behind a bus on the left side and heads for the ego path at 1.9 m/s.",
        "A schoolboy is obscured by a parked van and then steps into the ego lane.",
        "A runner hidden behind a roadside barrier becomes visible and moves across the ego path.",
        "A pedestrian partially concealed behind a parked truck on the right begins entering the ego lane.",
        "A girl emerges from behind a parked car at the curb and moves toward the ego path; the side is not specified.",
    ]
    router = SceneConstructionRouter()
    assert all(
        router.route(prompt=prompt, hierarchical_spec=_spec()).mode
        == EDIT_EXISTING
        for prompt in prompts
    )