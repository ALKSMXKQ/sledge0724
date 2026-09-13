import numpy as np
import pytest
from sledge.hazard_equivalence.ground_truth import TimedTrajectory2D, compute_spatiotemporal_conflict


def traj(times, xy):
    return TimedTrajectory2D(np.asarray(times, float), np.asarray(xy, float))


def test_crossing_same_time_is_conflict():
    ego = traj([0, 5], [[0, 0], [10, 0]])
    ped = traj([0, 5], [[10, -10], [10, 0]])
    r = compute_spatiotemporal_conflict(ego, ped)
    assert r.conflict_exists
    assert r.ego_arrival_time == pytest.approx(5.0)
    assert r.ped_arrival_time == pytest.approx(5.0)
    assert r.arrival_time_gap == pytest.approx(0.0)
    assert r.PET == pytest.approx(0.0)
    assert r.minimum_distance == pytest.approx(0.0)


def test_spatial_crossing_far_apart_in_time_is_not_spatiotemporal_conflict():
    ego = traj([0, 5], [[0, 0], [10, 0]])
    ped = traj([20, 25], [[10, -10], [10, 0]])
    r = compute_spatiotemporal_conflict(ego, ped)
    assert not r.conflict_exists
    assert r.arrival_time_gap == pytest.approx(20.0)
