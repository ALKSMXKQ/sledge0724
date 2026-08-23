from __future__ import annotations

import pytest

from sledge.semantic_control.occluded_pedestrian_pipeline.generation.primitive_compiler import (
    compile_spec_to_ops,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.language.eventframe_adapter import (
    ControlOverrides,
    OccludedPedestrianEventFrameAdapter,
)


def test_prompt_only_values_drive_hazard_spec() -> None:
    adapter = (
        OccludedPedestrianEventFrameAdapter(
            llm_provider="none"
        )
    )

    result = adapter.adapt(
        (
            "A pedestrian hidden behind a road "
            "barrier crosses from left to right "
            "at 1.9 m/s in a high-risk event."
        )
    )

    spec = result.hazard_spec

    assert (
        spec.object_layer
        .occlusion
        .occluder_type
        == "barrier"
    )

    assert (
        spec.interaction_layer
        .conflict_direction
        == "left_to_right"
    )

    assert (
        spec.risk_layer
        .target_actor_speed_mps
        == pytest.approx(1.9)
    )

    assert (
        spec.risk_layer.risk_level
        == "aggressive"
    )

    assert (
        result.provenance[
            "occluder_type"
        ]["source"]
        == "prompt_evidence"
    )

    assert (
        result.provenance[
            "direction"
        ]["source"]
        == "prompt_evidence"
    )

    assert (
        result.provenance[
            "pedestrian_speed_mps"
        ]["source"]
        == "prompt_evidence"
    )


def test_partial_explicit_control_does_not_mask_other_prompt_fields() -> None:
    adapter = (
        OccludedPedestrianEventFrameAdapter(
            llm_provider="none"
        )
    )

    result = adapter.adapt(
        (
            "A pedestrian hidden behind a bus "
            "suddenly crosses from left to right "
            "at 1.9 m/s."
        ),
        ControlOverrides(
            pedestrian_speed_mps=1.2
        ),
    )

    spec = result.hazard_spec

    assert (
        spec.actor_layer.primary_actor
        == "pedestrian"
    )

    assert (
        spec.object_layer
        .occlusion
        .enabled
        is True
    )

    assert (
        spec.object_layer
        .occlusion
        .occluder_type
        == "vehicle"
    )

    assert (
        spec.interaction_layer
        .conflict_direction
        == "left_to_right"
    )

    assert (
        spec.risk_layer
        .target_actor_speed_mps
        == pytest.approx(1.2)
    )

    assert (
        result.provenance[
            "occluder_type"
        ]["source"]
        == "prompt_evidence"
    )

    assert (
        result.provenance[
            "direction"
        ]["source"]
        == "prompt_evidence"
    )

    assert (
        result.provenance[
            "pedestrian_speed_mps"
        ]["source"]
        == "explicit_override"
    )

    operations = compile_spec_to_ops(
        spec
    )

    names = [
        operation.name
        for operation in operations
    ]

    assert (
        "place_actor_laterally"
        in names
    )

    assert (
        "set_lateral_or_crossing_motion"
        in names
    )

    assert (
        "add_or_select_occluder"
        in names
    )

    occluder_operation = next(
        operation
        for operation in operations
        if (
            operation.name
            == "add_or_select_occluder"
        )
    )

    assert (
        occluder_operation.params[
            "occluder_type"
        ]
        == "vehicle"
    )

    assert (
        occluder_operation.params[
            "compensate_frame0_offset"
        ]
        is True
    )


def test_explicit_controls_are_locked_and_traceable() -> None:
    adapter = (
        OccludedPedestrianEventFrameAdapter(
            llm_provider="none"
        )
    )

    result = adapter.adapt(
        (
            "A pedestrian hidden behind a bus "
            "crosses from left to right "
            "at 1.9 m/s."
        ),
        ControlOverrides(
            occluder_type="truck",
            direction="right_to_left",
            pedestrian_speed_mps=1.2,
            risk_level="aggressive",
        ),
    )

    assert set(
        result.resolution.locked_fields
    ) == {
        "occluder_type",
        "direction",
        "pedestrian_speed_mps",
        "risk_level",
    }

    assert (
        result.hazard_spec
        .object_layer
        .occlusion
        .occluder_type
        == "vehicle"
    )

    assert (
        result.hazard_spec
        .interaction_layer
        .conflict_direction
        == "right_to_left"
    )


def test_prompt_evidence_is_preserved_with_provenance() -> None:
    adapter = (
        OccludedPedestrianEventFrameAdapter(
            llm_provider="none"
        )
    )

    result = adapter.adapt(
        (
            "A pedestrian hidden behind a "
            "delivery van crosses from right "
            "to left at pedestrian speed "
            "1.6 m/s."
        )
    )

    assert (
        result.hazard_spec
        .object_layer
        .occlusion
        .occluder_type
        == "vehicle"
    )

    assert (
        result.hazard_spec
        .interaction_layer
        .conflict_direction
        == "right_to_left"
    )

    assert (
        result.hazard_spec
        .risk_layer
        .target_actor_speed_mps
        == pytest.approx(1.6)
    )

    assert (
        result.provenance[
            "occluder_type"
        ]["source"]
        == "prompt_evidence"
    )

    assert (
        result.provenance[
            "direction"
        ]["source"]
        == "prompt_evidence"
    )

    assert (
        result.provenance[
            "pedestrian_speed_mps"
        ]["source"]
        == "prompt_evidence"
    )


def test_nuplan_visible_occluder_types_are_extracted() -> None:
    adapter = (
        OccludedPedestrianEventFrameAdapter(
            llm_provider="none"
        )
    )

    prompts = {
        "bicycle": (
            "A pedestrian hidden behind a "
            "bicycle crosses from right to "
            "left at 1.6 m/s."
        ),
        "generic_object": (
            "A pedestrian hidden behind a "
            "generic object crosses from "
            "right to left at 1.6 m/s."
        ),
        "traffic_cone": (
            "A pedestrian hidden behind a "
            "traffic cone crosses from "
            "right to left at 1.6 m/s."
        ),
        "barrier": (
            "A pedestrian hidden behind a "
            "road barrier crosses from "
            "right to left at 1.6 m/s."
        ),
        "czone_sign": (
            "A pedestrian hidden behind a "
            "construction-zone sign crosses "
            "from right to left at 1.6 m/s."
        ),
    }

    for (
        expected,
        prompt,
    ) in prompts.items():
        assert (
            adapter.adapt(
                prompt
            ).hazard_spec
            .object_layer
            .occlusion
            .occluder_type
            == expected
        )


def test_kmh_speed_is_supported() -> None:
    adapter = (
        OccludedPedestrianEventFrameAdapter(
            llm_provider="none"
        )
    )

    result = adapter.adapt(
        (
            "A pedestrian hidden behind a "
            "parked car crosses from right "
            "to left at 5.76 km/h."
        )
    )

    assert (
        result.hazard_spec
        .risk_layer
        .target_actor_speed_mps
        == pytest.approx(1.6)
    )


def test_speed_outside_rvae_range_is_rejected() -> None:
    adapter = (
        OccludedPedestrianEventFrameAdapter(
            llm_provider="none"
        )
    )

    with pytest.raises(
        ValueError,
        match="outside",
    ):
        adapter.adapt(
            (
                "A pedestrian hidden behind "
                "a parked car crosses right "
                "to left."
            ),
            ControlOverrides(
                pedestrian_speed_mps=2.5
            ),
        )


def test_left_emergence_with_right_to_left_direction_is_rejected() -> None:
    adapter = (
        OccludedPedestrianEventFrameAdapter(
            llm_provider="none"
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "emergence side contradicts"
        ),
    ):
        adapter.adapt(
            (
                "行人从车辆左侧冲出，"
                "由右向左横穿，"
                "速度为1.6米每秒。"
            )
        )


def test_consistent_left_emergence_is_accepted() -> None:
    adapter = (
        OccludedPedestrianEventFrameAdapter(
            llm_provider="none"
        )
    )

    result = adapter.adapt(
        (
            "行人从车辆左侧冲出，"
            "由左向右横穿，"
            "速度为1.6米每秒。"
        )
    )

    assert (
        result.hazard_spec
        .interaction_layer
        .conflict_direction
        == "left_to_right"
    )


def test_negated_occlusion_is_rejected() -> None:
    adapter = (
        OccludedPedestrianEventFrameAdapter(
            llm_provider="none"
        )
    )

    with pytest.raises(
        ValueError,
        match="not occluded",
    ):
        adapter.adapt(
            (
                "The pedestrian is not "
                "hidden by the vehicle and "
                "crosses from right to left."
            )
        )


def test_non_occluded_crossing_does_not_silently_enter_pipeline() -> None:
    adapter = (
        OccludedPedestrianEventFrameAdapter(
            llm_provider="none"
        )
    )

    with pytest.raises(
        ValueError
    ):
        adapter.adapt(
            (
                "A pedestrian walks across "
                "an open road with no visual "
                "occlusion."
            )
        )


def test_control_override_can_preserve_resolution_provenance() -> None:
    adapter = (
        OccludedPedestrianEventFrameAdapter(
            llm_provider="none"
        )
    )

    result = adapter.adapt(
        (
            "A pedestrian hidden behind a "
            "vehicle crosses from right "
            "to left."
        ),
        ControlOverrides(
            occluder_type="barrier",
            direction="left_to_right",
            pedestrian_speed_mps=1.9,
            risk_level="moderate",
            locked_fields=(
                "direction",
                "pedestrian_speed_mps",
            ),
            source_by_field={
                "occluder_type": (
                    "edited_resolution"
                ),
                "direction": (
                    "prompt_evidence"
                ),
                "pedestrian_speed_mps": (
                    "edited_resolution"
                ),
                "risk_level": (
                    "template_default"
                ),
            },
        ),
    )

    assert (
        result.provenance[
            "occluder_type"
        ]["source"]
        == "edited_resolution"
    )

    assert (
        result.provenance[
            "direction"
        ]["source"]
        == "prompt_evidence"
    )

    assert (
        result.provenance[
            "pedestrian_speed_mps"
        ]["source"]
        == "edited_resolution"
    )