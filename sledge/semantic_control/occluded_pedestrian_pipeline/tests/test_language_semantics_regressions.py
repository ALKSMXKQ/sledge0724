"""Regression tests for the occluded-pedestrian language pipeline."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from sledge.semantic_control.language.hierarchical_pipeline import (
    HierarchicalEventFramePipeline,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.language.hierarchical_template_validator import (
    HierarchicalOccludedTemplateValidator,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.language.occluded_prompt_matrix import (
    default_prompt_cases,
)


@pytest.fixture(scope="module")
def results() -> Dict[str, Dict[str, Any]]:
    pipeline = HierarchicalEventFramePipeline(
        llm_provider="none",
        allow_fallback=True,
    )

    validator = (
        HierarchicalOccludedTemplateValidator()
    )

    output: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for case in default_prompt_cases():
        result = pipeline.parse_to_result(
            case.prompt
        )

        validation = validator.validate(
            result.spec,
            expected=case.expected,
        )

        output[case.case_id] = {
            "case":
                case,
            "result":
                result,
            "validation":
                validation,
        }

    return output


def test_all_twelve_prompt_templates_pass(
    results: Dict[
        str,
        Dict[str, Any],
    ],
) -> None:
    failures = {}

    for (
        case_id,
        payload,
    ) in results.items():
        validation = payload[
            "validation"
        ]

        result = payload[
            "result"
        ]

        if not validation.passed:
            failures[case_id] = {
                "prompt":
                    payload[
                        "case"
                    ].prompt,
                "validation_issues":
                    validation.issues,
                "frame_issues":
                    result.frame_issues,
                "spec_issues":
                    result.spec_issues,
                "hierarchy_issues":
                    result.hierarchy_issues,
            }

    assert not failures, failures


def test_all_twelve_keep_occluded_pedestrian_contract(
    results: Dict[
        str,
        Dict[str, Any],
    ],
) -> None:
    for (
        case_id,
        payload,
    ) in results.items():
        result = payload[
            "result"
        ]

        spec = result.spec

        hierarchy = spec[
            "hierarchy_layer"
        ]

        values = hierarchy[
            "path_values"
        ]

        projection = hierarchy[
            "nuplan_projection"
        ]

        assert (
            values[
                "primary_actor_type"
            ]
            == "pedestrian"
        ), case_id

        assert (
            values[
                "hazard_interaction"
            ]
            == "occluded_emergence"
        ), case_id

        assert (
            values[
                "motion_direction"
            ]
            == "occluder_to_ego_path"
        ), case_id

        assert (
            str(
                values[
                    "auxiliary_entity"
                ]
            ).endswith(
                "_occluder"
            )
        ), case_id

        assert (
            projection[
                "tracked_object_type"
            ]
            == (
                "TrackedObjectType."
                "PEDESTRIAN"
            )
        ), case_id

        assert (
            projection[
                "sledge_collection"
            ]
            == "pedestrians"
        ), case_id

        crossing_entry = (
            spec[
                "parameter_layer"
            ][
                "completed"
            ][
                "crossing_direction"
            ]
        )

        assert (
            crossing_entry["value"]
            == "occluder_to_ego_path"
        ), case_id


@pytest.mark.parametrize(
    "case_id",
    [
        "occ_ped_003",
        "occ_ped_005",
        "occ_ped_008",
        "occ_ped_010",
        "occ_ped_012",
    ],
)
def test_ego_path_is_not_collapsed_to_ego_lane(
    results: Dict[
        str,
        Dict[str, Any],
    ],
    case_id: str,
) -> None:
    spec = results[
        case_id
    ][
        "result"
    ].spec

    assert (
        spec[
            "semantic_slots"
        ][
            "target_path"
        ]
        == "ego_path"
    )

    assert (
        spec[
            "hierarchy_layer"
        ][
            "path_values"
        ][
            "target_region"
        ]
        == "ego_path"
    )

    assert (
        spec[
            "parameter_layer"
        ][
            "completed"
        ][
            "target_path"
        ][
            "value"
        ]
        == "ego_path"
    )


@pytest.mark.parametrize(
    (
        "case_id",
        "speed_mps",
    ),
    [
        (
            "occ_ped_002",
            1.2,
        ),
        (
            "occ_ped_004",
            1.6,
        ),
        (
            "occ_ped_007",
            0.8,
        ),
        (
            "occ_ped_008",
            1.9,
        ),
    ],
)
def test_explicit_pedestrian_speed_binds_to_actor(
    results: Dict[
        str,
        Dict[str, Any],
    ],
    case_id: str,
    speed_mps: float,
) -> None:
    completed = (
        results[
            case_id
        ][
            "result"
        ].spec[
            "parameter_layer"
        ][
            "completed"
        ]
    )

    actor_speed = completed[
        "actor_speed_mps"
    ]

    assert (
        actor_speed["value"]
        == pytest.approx(
            speed_mps
        )
    )

    assert (
        actor_speed["source"]
        == "user_input"
    )

    # The pedestrian's explicit speed must not become the ego speed.
    ego_speed = completed[
        "ego_speed_mps"
    ]

    assert (
        ego_speed["value"]
        != pytest.approx(
            speed_mps
        )
    )


@pytest.mark.parametrize(
    (
        "case_id",
        "expected_detail",
        "expected_occluder",
    ),
    [
        (
            "occ_ped_009",
            "child",
            "van_occluder",
        ),
        (
            "occ_ped_012",
            "child",
            "parked_car_occluder",
        ),
    ],
)
def test_child_subject_wins_over_occluder_vehicle(
    results: Dict[
        str,
        Dict[str, Any],
    ],
    case_id: str,
    expected_detail: str,
    expected_occluder: str,
) -> None:
    result = results[
        case_id
    ][
        "result"
    ]

    hierarchy = result.spec[
        "hierarchy_layer"
    ]

    values = hierarchy[
        "path_values"
    ]

    attributes = hierarchy[
        "attributes"
    ]

    assert (
        values[
            "primary_actor_type"
        ]
        == "pedestrian"
    )

    assert (
        attributes[
            "language_actor_detail"
        ]
        == expected_detail
    )

    assert (
        values[
            "auxiliary_entity"
        ]
        == expected_occluder
    )

    assert (
        result.frame.main_actor.actor_class
        == "human_on_foot"
    )


def test_jogger_rushes_toward_is_a_lateral_hazard(
    results: Dict[
        str,
        Dict[str, Any],
    ],
) -> None:
    result = results[
        "occ_ped_003"
    ][
        "result"
    ]

    assert (
        "missing_main_event_type"
        not in result.frame_issues
    )

    assert (
        "missing_motion_axis"
        not in result.frame_issues
    )

    assert (
        result.frame.main_actor.actor_class
        == "human_on_foot"
    )

    assert (
        result.frame.main_event.motion_axis
        == "lateral"
    )

    assert (
        result.spec[
            "hierarchy_layer"
        ][
            "path_values"
        ][
            "hazard_interaction"
        ]
        == "occluded_emergence"
    )


def test_explicit_side_is_preserved(
    results: Dict[
        str,
        Dict[str, Any],
    ],
) -> None:
    for (
        case_id,
        payload,
    ) in results.items():
        expected = (
            payload[
                "case"
            ].expected
        )

        expected_side = (
            expected.get(
                "occluder_side"
            )
        )

        if expected_side not in {
            "left",
            "right",
        }:
            continue

        spec = payload[
            "result"
        ].spec

        completed = (
            spec[
                "parameter_layer"
            ][
                "completed"
            ]
        )

        hierarchy_values = (
            spec[
                "hierarchy_layer"
            ][
                "path_values"
            ]
        )

        side_entry = completed[
            "occluder_side"
        ]

        assert (
            side_entry["value"]
            == expected_side
        ), case_id

        expected_region = (
            "left_side"
            if expected_side
            == "left"
            else "right_side"
        )

        assert (
            hierarchy_values[
                "source_region"
            ]
            == expected_region
        ), case_id


def test_unspecified_side_remains_sample_once(
    results: Dict[
        str,
        Dict[str, Any],
    ],
) -> None:
    spec = results[
        "occ_ped_012"
    ][
        "result"
    ].spec

    values = (
        spec[
            "hierarchy_layer"
        ][
            "path_values"
        ]
    )

    assert (
        values[
            "source_region"
        ]
        == "curbside"
    )

    side_entry = (
        spec[
            "parameter_layer"
        ][
            "completed"
        ][
            "occluder_side"
        ]
    )

    side_value = (
        side_entry["value"]
    )

    assert isinstance(
        side_value,
        dict,
    )

    assert (
        side_value[
            "distribution"
        ]
        == "categorical"
    )

    assert set(
        side_value["values"]
    ) == {
        "left",
        "right",
    }

    assert (
        side_value[
            "sample_once"
        ]
        is True
    )


def test_unspecified_side_does_not_reintroduce_two_semantic_directions(
    results: Dict[
        str,
        Dict[str, Any],
    ],
) -> None:
    spec = results[
        "occ_ped_012"
    ][
        "result"
    ].spec

    crossing_entry = (
        spec[
            "parameter_layer"
        ][
            "completed"
        ][
            "crossing_direction"
        ]
    )

    assert (
        crossing_entry["value"]
        == "occluder_to_ego_path"
    )

    assert (
        "left_to_right"
        not in crossing_entry.get(
            "alternatives",
            [],
        )
    )

    assert (
        "right_to_left"
        not in crossing_entry.get(
            "alternatives",
            [],
        )
    )