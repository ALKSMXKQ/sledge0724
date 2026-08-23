import pytest

from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hierarchical_spec_adapter import (
    HierarchicalHazardSpecAdapter,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hierarchical_template_sampler import (
    ConcreteOccludedPedestrianParameters,
)


def _sample(*, risk_level: str = "moderate", pre_ttc: float = 6.5):
    return ConcreteOccludedPedestrianParameters(
        seed=1,
        semantic_direction="occluder_to_ego_path",
        occluder_side="right",
        concrete_direction="right_to_left",
        language_actor_detail="adult_or_unspecified",
        semantic_occluder_type="parked_car_occluder",
        executable_occluder_type="vehicle",
        risk_level=risk_level,
        lane_width_m=3.5,
        lane_count=1,
        road_curvature=0.0,
        ego_speed_mps=4.0,
        ego_acceleration_mps2=0.0,
        ego_distance_to_conflict_m=26.0,
        actor_speed_mps=1.6,
        actor_acceleration_mps2=0.0,
        actor_start_time_s=0.0,
        occluder_longitudinal_m=10.0,
        occluder_lateral_offset_m=2.0,
        occluder_length_m=5.0,
        occluder_width_m=2.0,
        reveal_distance_m=6.0,
        minimum_clearance_m=1.0,
        braking_deceleration_mps2=4.5,
        time_to_collision_s=pre_ttc,
        construction_mode="edit_existing",
        road_topology="straight_segment",
        road_parameter_source="b0_scene_context",
        ego_state_source="hierarchical_template_semantic_control",
    )


def _hierarchical_spec(ttc_entry=None):
    completed = {}
    if ttc_entry is not None:
        completed["time_to_collision_s"] = ttc_entry
    return {
        "hierarchy_layer": {
            "path_values": {
                "road_topology": "straight_segment",
                "visibility": "fully_occluded",
            }
        },
        "parameter_layer": {"completed": completed},
        "scene_construction": {"mode": "edit_existing"},
    }


def test_derived_preconstruction_ttc_does_not_become_hard_target() -> None:
    sample = _sample(risk_level="moderate", pre_ttc=6.5)
    spec = HierarchicalHazardSpecAdapter().adapt(
        prompt="A pedestrian emerges from behind a parked car.",
        hierarchical_spec=_hierarchical_spec(
            {
                "value": {
                    "definition": "ego_distance_to_conflict_m / ego_speed_mps"
                },
                "source": "derived_constraint",
            }
        ),
        sample=sample,
        spec_id="case",
        construction_plan={"mode": "edit_existing"},
    )

    assert spec.risk_layer.ttc_range_s == pytest.approx((2.0, 3.0))
    assert spec.debug["preconstruction_ttc_estimate_s"] == pytest.approx(6.5)
    assert spec.debug["ttc_control_source"] == "risk_level:moderate"


def test_explicit_user_ttc_remains_authoritative() -> None:
    sample = _sample(risk_level="moderate", pre_ttc=6.5)
    spec = HierarchicalHazardSpecAdapter().adapt(
        prompt="A pedestrian emerges with TTC about 1.5 seconds.",
        hierarchical_spec=_hierarchical_spec(
            {
                "value": 1.5,
                "source": "user_input",
            }
        ),
        sample=sample,
        spec_id="case",
        construction_plan={"mode": "edit_existing"},
    )

    assert spec.risk_layer.ttc_range_s == pytest.approx((1.25, 1.75))
    assert spec.debug["ttc_control_source"] == "explicit_user_ttc"