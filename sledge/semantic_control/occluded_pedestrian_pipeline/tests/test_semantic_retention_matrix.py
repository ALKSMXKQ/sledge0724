"""Fast tests for the semantic-retention experiment matrix."""

from pathlib import Path

from sledge.semantic_control.occluded_pedestrian_pipeline.semantic_retention_experiment import (
    build_semantic_retention_cases,
    load_semantic_retention_matrix,
    summarize_semantic_retention_cases,
)


def test_matrix_balanced_axes(tmp_path: Path):
    matrix = load_semantic_retention_matrix(
        Path(__file__).parents[1] / "configs/semantic_retention_matrix.json"
    )
    raw = tmp_path / "high_magnitude_speed" / "log" / "token" / "sledge_raw.gz"
    raw.parent.mkdir(parents=True)
    raw.touch()

    cases = build_semantic_retention_cases(
        input_root=tmp_path,
        profile="debug12",
        matrix_config=matrix,
    )

    summary = summarize_semantic_retention_cases(cases)
    assert summary["num_cases"] == 12
    assert set(summary["axis_counts"]["occluder_position"]) == {
        "near_ego",
        "midway",
        "near_pedestrian",
    }
    assert set(summary["axis_counts"]["risk_level"]) == {"moderate"}


def test_position_names_are_preserved():
    matrix = load_semantic_retention_matrix(
        Path(__file__).parents[1] / "configs/semantic_retention_matrix.json"
    )
    cases = build_semantic_retention_cases(
        input_root=Path(__file__).parents[1],
        profile="debug12",
        matrix_config=matrix,
        glob_pattern="configs/semantic_retention_matrix.json",
        total_samples_override=1,
    )
    assert cases[0].occluder_position in {
        "near_ego",
        "midway",
        "near_pedestrian",
    }
