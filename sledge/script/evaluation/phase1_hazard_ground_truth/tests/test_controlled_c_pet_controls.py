from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path

import numpy as np

from sledge.script.evaluation.phase1_hazard_ground_truth.audit import controlled_c_pet_controls as cc
from sledge.script.evaluation.phase1_hazard_ground_truth.conflict import compute_spatiotemporal_conflict
from sledge.script.evaluation.phase1_hazard_ground_truth.types import TimedTrajectory2D


def _straight_crossing() -> tuple[TimedTrajectory2D, TimedTrajectory2D]:
    ego = TimedTrajectory2D(
        np.asarray([0.0, 4.0]),
        np.asarray([[-2.0, 0.0], [2.0, 0.0]]),
    )
    ped = TimedTrajectory2D(
        np.asarray([0.0, 4.0]),
        np.asarray([[0.0, -2.0], [0.0, 2.0]]),
    )
    return ego, ped


def test_independent_circle_occupancy_is_analytic() -> None:
    ego, _ = _straight_crossing()
    intervals = cc._circle_intervals(
        ego.timestamps_s, ego.positions_xy, np.asarray([0.0, 0.0]), 1.0
    )
    assert len(intervals) == 1
    assert np.allclose(intervals[0], (1.0, 3.0), atol=1e-12)


def test_target_pet_shift_is_exact() -> None:
    source = cc.SourceGeometry(
        scenario_id="s",
        pedestrian_id="p",
        scenario_type="test",
        crossing_angle_deg=90.0,
        intersection_xy=(0.0, 0.0),
        ego_arrival_s=2.0,
        pedestrian_arrival_s=2.0,
        ego_interval_s=(1.0, 3.0),
        pedestrian_interval_s=(1.0, 3.0),
    )
    control = cc._make_control(source, "main_positive", 1, 1.25)
    assert math.isclose(control.pedestrian_time_shift_s, 3.25, abs_tol=1e-12)
    assert np.allclose(control.pedestrian_interval_s, (4.25, 6.25))
    assert math.isclose(control.independent_pet_s, 1.25, abs_tol=1e-12)
    assert control.expected_C is True


def test_main_positive_and_negative_match_public_evaluator() -> None:
    ego, ped = _straight_crossing()
    center = np.asarray([0.0, 0.0])
    ego_occ = cc._circle_intervals(ego.timestamps_s, ego.positions_xy, center, 1.0)[0]
    ped_occ = cc._circle_intervals(ped.timestamps_s, ped.positions_xy, center, 1.0)[0]
    source = cc.SourceGeometry(
        scenario_id="s",
        pedestrian_id="p",
        scenario_type="test",
        crossing_angle_deg=90.0,
        intersection_xy=(0.0, 0.0),
        ego_arrival_s=2.0,
        pedestrian_arrival_s=2.0,
        ego_interval_s=ego_occ,
        pedestrian_interval_s=ped_occ,
    )

    for target, expected in [(1.50, True), (2.50, False)]:
        control = cc._make_control(source, "unit", 1, target)
        shifted = TimedTrajectory2D(
            ped.timestamps_s + control.pedestrian_time_shift_s,
            ped.positions_xy,
        )
        result = compute_spatiotemporal_conflict(ego, shifted)
        assert result.conflict_exists is expected
        assert result.PET is not None
        assert math.isclose(result.PET, target, rel_tol=0.0, abs_tol=1e-12)


def test_boundary_semantics_are_explicit() -> None:
    ego, ped = _straight_crossing()
    source = cc.SourceGeometry(
        scenario_id="s",
        pedestrian_id="p",
        scenario_type="test",
        crossing_angle_deg=90.0,
        intersection_xy=(0.0, 0.0),
        ego_arrival_s=2.0,
        pedestrian_arrival_s=2.0,
        ego_interval_s=(1.0, 3.0),
        pedestrian_interval_s=(1.0, 3.0),
    )
    low = cc._make_control(source, "boundary", 1, 1.99)
    high = cc._make_control(source, "boundary", 2, 2.01)
    assert low.expected_C is True
    assert high.expected_C is False
    assert compute_spatiotemporal_conflict(
        ego,
        TimedTrajectory2D(ped.timestamps_s + low.pedestrian_time_shift_s, ped.positions_xy),
    ).conflict_exists is True
    assert compute_spatiotemporal_conflict(
        ego,
        TimedTrajectory2D(ped.timestamps_s + high.pedestrian_time_shift_s, ped.positions_xy),
    ).conflict_exists is False


def test_diverse_source_selection_is_deterministic() -> None:
    rows = [
        {
            "scenario_id": f"s{i:02d}",
            "pedestrian_id": f"p{i:02d}",
            "raw_path_intersection_exists": True,
            "raw_path_intersection_geometry_type": "Point",
            "raw_path_intersection_points": [[float(i), 0.0]],
            "crossing_angle_deg": float(50 + i),
            "possible_right_censoring": False,
        }
        for i in range(12)
    ]
    first = cc.select_diverse_source_records(rows, count=6)
    second = cc.select_diverse_source_records(list(reversed(rows)), count=6)
    assert [(x["scenario_id"], x["pedestrian_id"]) for x in first] == [
        (x["scenario_id"], x["pedestrian_id"]) for x in second
    ]
    assert len({x["scenario_id"] for x in first}) == 6


def test_oracle_hash_detects_mutation(tmp_path: Path) -> None:
    path = tmp_path / "oracle.json"
    path.write_text('{"a": 1}\n')
    before = cc.sha256_file(path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == expected
    path.write_text('{"a": 2}\n')
    assert cc.sha256_file(path) != before


def test_independent_oracle_does_not_import_conflict_private_helpers() -> None:
    source = inspect.getsource(cc)
    forbidden = [
        "_arrival_time_at_point as",
        "_circle_occupancy_intervals as",
        "_pet as",
        "_candidate_points as",
        "_synchronized_minimum_distance as",
    ]
    for token in forbidden:
        assert token not in source
