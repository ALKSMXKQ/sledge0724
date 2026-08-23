from __future__ import annotations

import json
from pathlib import Path

import pytest

from sledge.semantic_control.occluded_pedestrian_pipeline.language.resolution import (
    resolve_prompt_controls,
    values_equivalent,
)


PACKAGE_ROOT = Path(
    __file__
).resolve().parents[1]


def _supported_mapped(
    case,
):
    """Build EventFrame candidates without leaking expected defaults.

    Fields listed in ``expected_defaults`` are intentionally omitted from the
    mocked EventFrame output. This forces the resolver to exercise the real
    ``template_default`` branch for those fields.
    """

    expected = case.get(
        "expected",
        {},
    )

    expected_defaults = set(
        case.get(
            "expected_defaults",
            [],
        )
    )

    occlusion = {
        "enabled": True,
    }

    if "occluder_type" not in expected_defaults:
        occlusion["occluder_type"] = (
            expected.get(
                "occluder_type",
                "vehicle",
            )
        )

    interaction_layer = {
        "conflict_type": (
            "lateral_conflict"
        ),
    }

    if "direction" not in expected_defaults:
        interaction_layer[
            "conflict_direction"
        ] = expected.get(
            "direction",
            "right_to_left",
        )

    completed = {}

    if (
        "pedestrian_speed_mps"
        not in expected_defaults
    ):
        speed = float(
            expected.get(
                "pedestrian_speed_mps",
                1.6,
            )
        )

        completed[
            "actor_speed_mps"
        ] = {
            "value": [
                speed,
                speed,
            ]
        }

    risk_layer = {}

    if "risk_level" not in expected_defaults:
        risk_layer[
            "risk_level"
        ] = expected.get(
            "risk_level",
            "moderate",
        )

    return {
        "actor_layer": {
            "primary_actor": (
                "pedestrian"
            )
        },
        "object_layer": {
            "occlusion": occlusion,
        },
        "interaction_layer": (
            interaction_layer
        ),
        "parameter_layer": {
            "completed": completed,
        },
        "risk_layer": risk_layer,
    }


def _unsupported_mapped():
    return {
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
                "longitudinal_conflict"
            )
        },
    }


def _load_cases():
    path = (
        PACKAGE_ROOT
        / (
            "configs/"
            "language_benchmark.jsonl"
        )
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as fp:
        return [
            json.loads(line)
            for line in fp
            if line.strip()
        ]


@pytest.mark.parametrize(
    "case",
    _load_cases(),
    ids=lambda item: item["id"],
)
def test_language_benchmark(
    case,
) -> None:
    mapped = (
        _supported_mapped(
            case
        )
        if case["valid"]
        else _unsupported_mapped()
    )

    if case.get(
        "expected_error"
    ) in {
        "emergence_direction_conflict",
        "negated_occlusion",
    }:
        mapped = _supported_mapped(
            case
        )

    result = resolve_prompt_controls(
        case["prompt"],
        mapped,
    )

    assert (
        result.semantic_valid
        is case["valid"]
    )

    if case["valid"]:
        expected = case[
            "expected"
        ]

        actual = result.values()

        for (
            name,
            value,
        ) in expected.items():
            assert values_equivalent(
                name,
                actual[name],
                value,
            ), (
                case["id"],
                name,
                actual[name],
                value,
            )

        assert set(
            result.defaults_used
        ) == set(
            case.get(
                "expected_defaults",
                [],
            )
        )

    else:
        assert any(
            issue.code
            == case[
                "expected_error"
            ]
            for issue in result.issues
        )