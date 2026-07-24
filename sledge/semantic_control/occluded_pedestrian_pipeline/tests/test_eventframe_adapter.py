from __future__ import annotations

import pytest

from sledge.semantic_control.occluded_pedestrian_pipeline.generation.primitive_compiler import (
    compile_spec_to_ops,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.language.eventframe_adapter import (
    ControlOverrides,
    OccludedPedestrianEventFrameAdapter,
)


def test_explicit_controls_override_eventframe_defaults() -> None:
    adapter = OccludedPedestrianEventFrameAdapter(llm_provider="none")
    result = adapter.adapt(
        "A pedestrian hidden behind a bus suddenly crosses from left to right at 1.9 m/s.",
        ControlOverrides(
            occluder_type="truck",
            direction="right_to_left",
            pedestrian_speed_mps=1.2,
            risk_level="aggressive",
        ),
    )
    spec = result.hazard_spec
    assert spec.actor_layer.primary_actor == "pedestrian"
    assert spec.object_layer.occlusion.enabled is True
    assert spec.object_layer.occlusion.occluder_type == "vehicle"
    assert spec.interaction_layer.conflict_direction == "right_to_left"
    assert spec.risk_layer.target_actor_speed_mps == pytest.approx(1.2)
    assert spec.risk_layer.risk_level == "aggressive"
    assert all(item["source"] == "control_override" for item in result.provenance.values())

    ops = compile_spec_to_ops(spec)
    names = [op.name for op in ops]
    assert "place_actor_laterally" in names
    assert "set_lateral_or_crossing_motion" in names
    assert "add_or_select_occluder" in names
    occ_op = next(op for op in ops if op.name == "add_or_select_occluder")
    assert occ_op.params["occluder_type"] == "vehicle"
    assert occ_op.params["compensate_frame0_offset"] is True


def test_prompt_evidence_is_preserved_with_provenance() -> None:
    adapter = OccludedPedestrianEventFrameAdapter(llm_provider="none")
    result = adapter.adapt(
        "A pedestrian hidden behind a delivery van crosses from right to left at pedestrian speed 1.6 m/s."
    )
    assert result.hazard_spec.object_layer.occlusion.occluder_type == "vehicle"
    assert result.hazard_spec.interaction_layer.conflict_direction == "right_to_left"
    assert result.hazard_spec.risk_layer.target_actor_speed_mps == pytest.approx(1.6)
    assert result.provenance["occluder_type"]["source"] == "prompt_evidence"
    assert result.provenance["direction"]["source"] == "prompt_evidence"
    assert result.provenance["pedestrian_speed_mps"]["source"] == "prompt_evidence"


def test_nuplan_visible_occluder_types_are_extracted() -> None:
    adapter = OccludedPedestrianEventFrameAdapter(llm_provider="none")
    prompts = {
        "bicycle": "A pedestrian hidden behind a bicycle crosses from right to left at 1.6 m/s.",
        "generic_object": "A pedestrian hidden behind a generic object crosses from right to left at 1.6 m/s.",
        "traffic_cone": "A pedestrian hidden behind a traffic cone crosses from right to left at 1.6 m/s.",
        "barrier": "A pedestrian hidden behind a road barrier crosses from right to left at 1.6 m/s.",
        "czone_sign": "A pedestrian hidden behind a construction-zone sign crosses from right to left at 1.6 m/s.",
    }
    for expected, prompt in prompts.items():
        assert adapter.adapt(prompt).hazard_spec.object_layer.occlusion.occluder_type == expected


def test_speed_outside_rvae_range_is_rejected() -> None:
    adapter = OccludedPedestrianEventFrameAdapter(llm_provider="none")
    with pytest.raises(ValueError, match="outside"):
        adapter.adapt(
            "A pedestrian hidden behind a parked car crosses right to left.",
            ControlOverrides(pedestrian_speed_mps=2.5),
        )
