from __future__ import annotations

from pathlib import Path

from sledge.semantic_control.occluded_pedestrian_pipeline.experiment_matrix import (
    build_experiment_cases,
    load_matrix_config,
    summarize_case_matrix,
)


PACKAGE_ROOT = Path(
    __file__
).resolve().parents[1]


def _fake_cache(
    root: Path,
    num_scenes_per_type: int = 12,
) -> None:
    for source_type in [
        "freeway",
        "intersection",
        "urban",
    ]:
        for index in range(
            num_scenes_per_type
        ):
            path = (
                root
                / source_type
                / f"token_{index:02d}"
                / "sledge_raw.gz"
            )

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            path.touch()


def test_debug20_is_reproducible_and_source_stratified(
    tmp_path: Path,
) -> None:
    _fake_cache(
        tmp_path
    )

    config = load_matrix_config(
        PACKAGE_ROOT
        / (
            "configs/"
            "experiment_matrix.json"
        )
    )

    first = build_experiment_cases(
        input_root=tmp_path,
        profile="debug20",
        matrix_config=config,
    )

    second = build_experiment_cases(
        input_root=tmp_path,
        profile="debug20",
        matrix_config=config,
    )

    assert first == second
    assert len(first) == 20

    summary = summarize_case_matrix(
        first
    )

    assert (
        summary[
            "num_conditions"
        ]
        == 1
    )

    assert set(
        summary[
            "source_scenario_types"
        ]
    ) == {
        "freeway",
        "intersection",
        "urban",
    }

    assert max(
        summary[
            "axis_counts"
        ][
            "source_scenario_type"
        ].values()
    ) <= 7


def test_prompt_only_matrix_values_are_gold_not_overrides(
    tmp_path: Path,
) -> None:
    _fake_cache(
        tmp_path
    )

    config = load_matrix_config(
        PACKAGE_ROOT
        / (
            "configs/"
            "experiment_matrix.json"
        )
    )

    case = build_experiment_cases(
        input_root=tmp_path,
        profile="debug20",
        matrix_config=config,
        max_cases=1,
        control_mode="prompt_only",
    )[0]

    assert (
        case.expected_controls[
            "occluder_type"
        ]
        == "vehicle"
    )

    assert (
        case.overrides
        .provided_fields()
        == ()
    )


def test_controlled_mode_injects_and_locks_matrix_values(
    tmp_path: Path,
) -> None:
    _fake_cache(
        tmp_path
    )

    config = load_matrix_config(
        PACKAGE_ROOT
        / (
            "configs/"
            "experiment_matrix.json"
        )
    )

    case = build_experiment_cases(
        input_root=tmp_path,
        profile="debug20",
        matrix_config=config,
        max_cases=1,
        control_mode="controlled",
    )[0]

    assert set(
        case.overrides
        .provided_fields()
    ) == {
        "occluder_type",
        "direction",
        "pedestrian_speed_mps",
        "risk_level",
    }

    assert set(
        case.overrides.locked_fields
    ) == {
        "occluder_type",
        "direction",
        "pedestrian_speed_mps",
        "risk_level",
    }


def test_paper18_has_18_conditions_and_180_cases(
    tmp_path: Path,
) -> None:
    _fake_cache(
        tmp_path
    )

    config = load_matrix_config(
        PACKAGE_ROOT
        / (
            "configs/"
            "experiment_matrix.json"
        )
    )

    cases = build_experiment_cases(
        input_root=tmp_path,
        profile="paper18",
        matrix_config=config,
    )

    summary = summarize_case_matrix(
        cases
    )

    assert len(cases) == 180

    assert (
        summary[
            "num_conditions"
        ]
        == 18
    )

    assert set(
        summary[
            "axis_counts"
        ][
            "occluder_type"
        ]
    ) == {
        "vehicle",
        "barrier",
        "generic_object",
    }


def test_pilot100_is_balanced_and_uses_unique_sources(
    tmp_path: Path,
) -> None:
    _fake_cache(
        tmp_path,
        num_scenes_per_type=50,
    )

    config = load_matrix_config(
        PACKAGE_ROOT
        / (
            "configs/"
            "experiment_matrix.json"
        )
    )

    cases = build_experiment_cases(
        input_root=tmp_path,
        profile="pilot100",
        matrix_config=config,
    )

    summary = summarize_case_matrix(
        cases
    )

    assert len(cases) == 100

    assert len(
        {
            case.input_raw
            for case in cases
        }
    ) == 100

    assert set(
        summary[
            "axis_counts"
        ][
            "occluder_type"
        ]
    ) == {
        "vehicle",
        "bicycle",
        "generic_object",
        "traffic_cone",
        "barrier",
        "czone_sign",
    }

    assert (
        max(
            summary[
                "axis_counts"
            ][
                "occluder_type"
            ].values()
        )
        - min(
            summary[
                "axis_counts"
            ][
                "occluder_type"
            ].values()
        )
        <= 1
    )

    assert set(
        summary[
            "axis_counts"
        ][
            "direction"
        ]
    ) == {
        "left_to_right",
        "right_to_left",
    }

    assert set(
        summary[
            "axis_counts"
        ][
            "pedestrian_speed_mps"
        ]
    ) == {
        "1.6",
        "1.9",
        "2.0",
    }


def test_pilot_candidate_pool_can_exceed_final_target(
    tmp_path: Path,
) -> None:
    _fake_cache(
        tmp_path,
        num_scenes_per_type=50,
    )
    config = load_matrix_config(
        PACKAGE_ROOT
        / "configs/experiment_matrix.json"
    )

    cases = build_experiment_cases(
        input_root=tmp_path,
        profile="pilot100",
        matrix_config=config,
        candidate_pool_size=120,
    )

    assert len(cases) == 120
    assert len(
        {
            case.input_raw
            for case in cases
        }
    ) == 120
