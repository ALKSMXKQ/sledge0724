from __future__ import annotations

import pytest

from sledge.semantic_control.occluded_pedestrian_pipeline.language.eventframe_adapter import (
    ControlOverrides,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.language.resolution import (
    RuleBasedPromptParser,
    merge_resolution_with_explicit_values,
    resolve_prompt_controls,
    resolution_from_payload,
)


def _mapped(
    *,
    occluder: str = "vehicle",
    direction: str = "right_to_left",
    speed_range=(1.2, 1.8),
    risk: str = "moderate",
):
    return {
        "actor_layer": {
            "primary_actor": "pedestrian"
        },
        "object_layer": {
            "occlusion": {
                "enabled": True,
                "occluder_type": occluder,
            }
        },
        "interaction_layer": {
            "conflict_type": (
                "lateral_conflict"
            ),
            "conflict_direction": direction,
        },
        "parameter_layer": {
            "completed": {
                "actor_speed_mps": {
                    "value": list(
                        speed_range
                    )
                },
            }
        },
        "risk_layer": {
            "risk_level": risk
        },
    }


def test_prompt_values_take_priority_over_eventframe_completion() -> None:
    result = resolve_prompt_controls(
        (
            "A pedestrian hidden behind a "
            "barrier crosses from left to "
            "right at 1.9 m/s in a "
            "high-risk event."
        ),
        _mapped(
            occluder="vehicle",
            direction="right_to_left",
            speed_range=(
                1.0,
                1.4,
            ),
        ),
    )

    values = result.values()

    assert values == {
        "occluder_type": "barrier",
        "direction": "left_to_right",
        "pedestrian_speed_mps": (
            pytest.approx(1.9)
        ),
        "risk_level": "aggressive",
    }

    assert (
        result.fields[
            "occluder_type"
        ].source
        == "prompt_evidence"
    )

    assert (
        result.fields[
            "direction"
        ].source
        == "prompt_evidence"
    )

    assert (
        result.fields[
            "pedestrian_speed_mps"
        ].source
        == "prompt_evidence"
    )

    assert any(
        issue.code
        == (
            "dual_parser_"
            "disagreement_direction"
        )
        for issue in result.issues
    )


def test_missing_values_are_visible_template_defaults() -> None:
    mapped = _mapped()

    mapped[
        "parameter_layer"
    ] = {
        "completed": {}
    }

    mapped[
        "risk_layer"
    ] = {}

    mapped[
        "interaction_layer"
    ].pop(
        "conflict_direction"
    )

    result = resolve_prompt_controls(
        (
            "A pedestrian hidden behind a "
            "parked vehicle suddenly crosses "
            "the ego path."
        ),
        mapped,
    )

    assert (
        result.fields[
            "direction"
        ].source
        == "template_default"
    )

    assert (
        result.fields[
            "pedestrian_speed_mps"
        ].source
        == "template_default"
    )

    assert (
        result.fields[
            "risk_level"
        ].source
        == "template_default"
    )

    assert (
        result.requires_confirmation
        is True
    )

    assert set(
        result.defaults_used
    ) == {
        "direction",
        "pedestrian_speed_mps",
        "risk_level",
    }


def test_kmh_is_normalized_to_mps() -> None:
    result = resolve_prompt_controls(
        (
            "A pedestrian hidden behind a "
            "truck crosses from right to "
            "left at 5.76 km/h."
        ),
        _mapped(),
    )

    assert (
        result.values()[
            "pedestrian_speed_mps"
        ]
        == pytest.approx(1.6)
    )

    assert (
        "km/h"
        in result.fields[
            "pedestrian_speed_mps"
        ].normalized_from
    )


def test_qualitative_fast_speed_is_completed_with_range() -> None:
    result = resolve_prompt_controls(
        (
            "A pedestrian hidden behind a "
            "bus quickly rushes from right "
            "to left."
        ),
        _mapped(),
    )

    speed = result.fields[
        "pedestrian_speed_mps"
    ]

    assert (
        speed.value
        == pytest.approx(1.9)
    )

    assert (
        speed.source
        == "semantic_range_completion"
    )

    assert (
        speed.value_range
        == pytest.approx(
            (
                1.6,
                2.0,
            )
        )
    )


def test_explicit_override_wins_and_is_locked() -> None:
    result = resolve_prompt_controls(
        (
            "A pedestrian hidden behind a "
            "barrier crosses from left to "
            "right at 1.9 m/s."
        ),
        _mapped(),
        overrides=ControlOverrides(
            pedestrian_speed_mps=1.2
        ),
    )

    assert (
        result.values()[
            "pedestrian_speed_mps"
        ]
        == pytest.approx(1.2)
    )

    assert (
        result.fields[
            "pedestrian_speed_mps"
        ].source
        == "explicit_override"
    )

    assert (
        result.fields[
            "pedestrian_speed_mps"
        ].locked
        is True
    )

    assert any(
        issue.code
        == (
            "override_prompt_conflict_"
            "pedestrian_speed_mps"
        )
        for issue in result.issues
    )


def test_vehicle_left_side_requires_left_to_right() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid",
    ):
        result = resolve_prompt_controls(
            (
                "行人从车辆左侧冲出，"
                "由右向左横穿，"
                "速度为1.6米每秒。"
            ),
            _mapped(),
        )

        if not result.semantic_valid:
            raise ValueError(
                "Invalid"
            )


def test_vehicle_left_side_consistent_direction_passes() -> None:
    result = resolve_prompt_controls(
        (
            "行人从车辆左侧冲出，"
            "由左向右横穿，"
            "速度为1.6米每秒。"
        ),
        _mapped(
            direction=(
                "left_to_right"
            )
        ),
    )

    assert (
        result.semantic_valid
        is True
    )

    assert (
        result.rule_parse
        .emergence_side
        == "left"
    )

    assert (
        result.values()[
            "direction"
        ]
        == "left_to_right"
    )


def test_negated_occlusion_is_rejected() -> None:
    result = resolve_prompt_controls(
        (
            "The pedestrian is not hidden "
            "by the vehicle and crosses "
            "from right to left."
        ),
        _mapped(),
    )

    assert (
        result.semantic_valid
        is False
    )

    assert any(
        issue.code
        == "negated_occlusion"
        for issue in result.issues
    )


def test_exclusion_selects_named_replacement() -> None:
    result = resolve_prompt_controls(
        (
            "Use a road barrier, not a "
            "vehicle, to hide a pedestrian "
            "crossing from right to left "
            "at 1.6 m/s."
        ),
        _mapped(),
    )

    assert (
        result.values()[
            "occluder_type"
        ]
        == "barrier"
    )

    assert (
        "vehicle"
        in result.rule_parse
        .excluded_occluders
    )


def test_non_target_cut_in_prompt_is_rejected() -> None:
    mapped = {
        "actor_layer": {
            "primary_actor": "vehicle"
        },
        "object_layer": {
            "occlusion": {
                "enabled": False
            }
        },
        "interaction_layer": {
            "conflict_type": (
                "merging_conflict"
            )
        },
    }

    result = resolve_prompt_controls(
        (
            "A vehicle cuts in from the "
            "adjacent lane."
        ),
        mapped,
    )

    assert (
        result.semantic_valid
        is False
    )

    assert any(
        issue.code
        == "unsupported_prompt_family"
        for issue in result.issues
    )


def test_resolution_payload_can_be_edited_and_reloaded() -> None:
    original = resolve_prompt_controls(
        (
            "A pedestrian hidden behind a "
            "vehicle crosses from right "
            "to left at 1.6 m/s."
        ),
        _mapped(),
    )

    payload = original.to_dict()

    payload[
        "fields"
    ][
        "pedestrian_speed_mps"
    ][
        "value"
    ] = 1.9

    payload[
        "fields"
    ][
        "pedestrian_speed_mps"
    ][
        "source"
    ] = "edited_resolution"

    payload[
        "fields"
    ][
        "pedestrian_speed_mps"
    ][
        "locked"
    ] = True

    payload[
        "fields"
    ][
        "pedestrian_speed_mps"
    ][
        "adjustable"
    ] = False

    loaded = resolution_from_payload(
        payload
    )

    assert (
        loaded.values()[
            "pedestrian_speed_mps"
        ]
        == pytest.approx(1.9)
    )

    assert (
        loaded.fields[
            "pedestrian_speed_mps"
        ].locked
        is True
    )


def test_direct_value_can_replace_loaded_resolution() -> None:
    original = resolve_prompt_controls(
        (
            "A pedestrian hidden behind a "
            "vehicle crosses from right "
            "to left at 1.6 m/s."
        ),
        _mapped(),
    )

    merged = (
        merge_resolution_with_explicit_values(
            original,
            {
                "occluder_type": (
                    "barrier"
                )
            },
            {
                "direction"
            },
        )
    )

    assert (
        merged.values()[
            "occluder_type"
        ]
        == "barrier"
    )

    assert (
        merged.fields[
            "occluder_type"
        ].locked
        is True
    )

    assert (
        merged.fields[
            "direction"
        ].locked
        is True
    )


def test_rule_parser_records_reference_frame() -> None:
    parsed = (
        RuleBasedPromptParser().parse(
            (
                "行人从车辆右侧冲出，"
                "由右向左横穿。"
            )
        )
    )

    assert (
        parsed.reference_frame
        == "ego"
    )

    assert (
        parsed.emergence_side
        == "right"
    )