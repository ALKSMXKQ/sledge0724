import numpy as np
import pytest
from sledge.hazard_equivalence.ground_truth import (
    FailureReason, OccludedPedestrianContract, RevealConfig, TimedTrajectory2D, TimingClass
)


def traj(times, xy):
    return TimedTrajectory2D(np.asarray(times, float), np.asarray(xy, float))


def crossing_at(t_conflict):
    ego = traj([0.0, t_conflict], [[0, 0], [10, 0]])
    ped = traj([0.0, t_conflict], [[10, -10], [10, 0]])
    return ego, ped


def contract(k=2):
    return OccludedPedestrianContract(reveal_config=RevealConfig(0.5, k, 1))


def test_gate_case_1_real_occlusion_reveal_conflict_is_hazard():
    ego, ped = crossing_at(4.0)
    r = contract().verify([0.0, 0.1, 0.9, 1.0, 1.0], [0, 1, 2, 3, 4], ego, ped)
    assert (r.O, r.E, r.C, r.T, r.H) == (True, True, True, True, True)
    assert r.timing.timing_class == TimingClass.MODERATE


def test_gate_case_2_unoccluded_crossing_is_not_hazard():
    ego, ped = crossing_at(4.0)
    r = contract().verify([1.0, 1.0, 1.0, 1.0, 1.0], [0, 1, 2, 3, 4], ego, ped)
    assert not r.O and not r.H
    assert r.failure_reason == FailureReason.NO_PRE_REVEAL_OCCLUSION


def test_gate_case_3_occluded_but_never_reveals_is_not_hazard():
    ego, ped = crossing_at(4.0)
    r = contract().verify([0.0, 0.1, 0.2, 0.1, 0.2], [0, 1, 2, 3, 4], ego, ped)
    assert r.O and not r.E and not r.H
    assert r.failure_reason == FailureReason.NO_STABLE_REVEAL


def test_gate_case_4_reveal_without_path_conflict_is_not_hazard():
    ego = traj([0, 4], [[0, 0], [10, 0]])
    ped = traj([0, 4], [[20, -5], [20, 5]])
    r = contract().verify([0.0, 0.1, 0.9, 1.0, 1.0], [0, 1, 2, 3, 4], ego, ped)
    assert r.O and r.E and not r.C and not r.H
    assert r.failure_reason == FailureReason.NO_SPATIOTEMPORAL_CONFLICT


def test_gate_case_5_conflict_with_five_second_reaction_time_is_safe():
    ego, ped = crossing_at(6.0)
    r = contract().verify([0.0, 0.9, 1.0, 1.0], [0, 1, 2, 3], ego, ped)
    assert (r.O, r.E, r.C, r.T, r.H) == (True, True, True, False, False)
    assert r.timing.rttc_s == pytest.approx(5.0)
    assert r.timing.timing_class == TimingClass.SAFE
    assert r.failure_reason == FailureReason.NON_CRITICAL_TIMING


def test_gate_case_6_reveal_to_conflict_point_eight_seconds_is_aggressive():
    ego, ped = crossing_at(1.8)
    r = contract().verify([0.0, 0.9, 1.0, 1.0], [0, 1, 2, 3], ego, ped)
    assert (r.O, r.E, r.C, r.T, r.H) == (True, True, True, True, True)
    assert r.timing.rttc_s == pytest.approx(0.8)
    assert r.timing.timing_class == TimingClass.AGGRESSIVE
