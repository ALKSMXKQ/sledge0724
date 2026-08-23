import pytest

from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hierarchical_template_sampler import (
    EDIT_EXISTING,
    SYNTHESIZE_NEW,
    HierarchicalTemplateSampler,
    SamplingOverrides,
)


def _entry(value, source="hierarchical_prior"):
    return {
        "value": value,
        "unit": "",
        "source": source,
        "reason": "",
        "confidence": 1.0,
        "evidence": [],
        "conditioned_on": {},
        "condition_path": "",
        "is_assumption": source != "user_input",
        "alternatives": [],
    }


def _template():
    return {
        "hierarchy_layer": {
            "path_values": {
                "road_topology": "straight_segment",
                "primary_actor_type": "pedestrian",
                "hazard_interaction": "occluded_emergence",
                "auxiliary_entity": "parked_car_occluder",
                "source_region": "left_side",
                "target_region": "ego_path",
                "visibility": "fully_occluded",
                "motion_direction": "occluder_to_ego_path",
                "risk_level": "moderate",
            },
            "attributes": {
                "language_actor_detail": "adult_or_unspecified",
            },
        },
        "parameter_layer": {
            "completed": {
                "lane_width_m": _entry([3.2, 3.8]),
                "lane_count": _entry([1, 2]),
                "road_curvature": _entry([-0.005, 0.005]),
                "ego_speed_mps": _entry([5.0, 10.0]),
                "ego_acceleration_mps2": _entry([0.0, 0.0]),
                "ego_distance_to_conflict_m": _entry([10.0, 10.0]),
                "actor_speed_mps": _entry(1.2, "user_input"),
                "actor_acceleration_mps2": _entry([0.0, 0.0]),
                "actor_start_time_s": _entry([0.8, 0.8]),
                "occluder_position": _entry({"x_m": [7.0, 7.0]}),
                "occluder_lateral_offset_m": _entry([2.0, 2.0]),
                "occluder_length_m": _entry([4.5, 4.5]),
                "occluder_width_m": _entry([2.0, 2.0]),
                "reveal_distance_m": _entry([5.0, 5.0]),
                "minimum_clearance_m": _entry([1.0, 1.0]),
                "braking_deceleration_mps2": _entry([5.0, 5.0]),
            }
        },
    }


def test_edit_existing_preserves_b0_road_but_samples_ego_semantic_control() -> None:
    sample = HierarchicalTemplateSampler().sample(
        _template(),
        prompt="A pedestrian emerges from behind a car at 1.2 m/s.",
        case_id="edit",
        overrides=SamplingOverrides(seed=11),
        construction_plan={"mode": EDIT_EXISTING},
        scene_context={
            "lane_width_m": 3.91,
            "lane_count": 3,
            "road_curvature": 0.012,
            "ego_speed_mps": 9.4,
            "ego_acceleration_mps2": 0.2,
        },
    )
    assert sample.valid, sample.issues
    assert sample.construction_mode == EDIT_EXISTING
    assert sample.lane_width_m == pytest.approx(3.91)
    assert sample.lane_count == 3
    assert sample.road_curvature == pytest.approx(0.012)
    assert 5.0 <= sample.ego_speed_mps <= 10.0
    assert sample.ego_acceleration_mps2 == pytest.approx(0.0)
    assert sample.actor_speed_mps == pytest.approx(1.2)
    assert sample.road_parameter_source == "b0_scene_context"
    assert sample.ego_state_source == "hierarchical_template_semantic_control"


def test_synthesis_obeys_explicit_global_constraints() -> None:
    sample = HierarchicalTemplateSampler().sample(
        _template(),
        prompt=(
            "On a three-lane road with lane width 3.7 m, "
            "a pedestrian emerges behind a car."
        ),
        case_id="synth",
        overrides=SamplingOverrides(seed=3),
        construction_plan={
            "mode": SYNTHESIZE_NEW,
            "explicit_global_constraints": {
                "road_topology": "straight_segment",
                "lane_count": 3,
                "lane_width_m": 3.7,
                "road_curvature": 0.01,
            },
        },
        scene_context={
            # Must be ignored in synthesis mode.
            "lane_width_m": 4.2,
            "lane_count": 1,
            "road_curvature": 0.0,
            "ego_speed_mps": 2.0,
            "ego_acceleration_mps2": 0.0,
        },
    )
    assert sample.valid, sample.issues
    assert sample.construction_mode == SYNTHESIZE_NEW
    assert sample.lane_count == 3
    assert sample.lane_width_m == pytest.approx(3.7)
    assert sample.road_curvature == pytest.approx(0.01)
    assert sample.road_parameter_source == "explicit_language_plus_hierarchical_template"
    assert sample.ego_state_source == "hierarchical_template"


def test_bidirectional_synthesis_has_at_least_two_lanes_without_count() -> None:
    sample = HierarchicalTemplateSampler().sample(
        _template(),
        case_id="two-way",
        overrides=SamplingOverrides(seed=9),
        construction_plan={
            "mode": SYNTHESIZE_NEW,
            "explicit_global_constraints": {
                "directionality": "bidirectional",
            },
        },
    )
    assert sample.valid, sample.issues
    assert sample.lane_count >= 2