import numpy as np
from sledge.script.evaluation.phase1_hazard_ground_truth.scene_types import ActorState, ActorType
from sledge.script.evaluation.phase1_hazard_ground_truth import OcclusionType, compute_visibility


def actor(track_id, kind, x, y, length, width):
    return ActorState(track_id, kind, np.array([x, y]), 0.0, length_m=length, width_m=width)


def test_truck_between_ego_and_pedestrian_is_fully_occluded():
    ego = actor("ego", ActorType.EGO, 0, 0, 4.5, 2.0)
    truck = actor("truck", ActorType.VEHICLE, 5, 0, 4.0, 3.0)
    ped = actor("ped", ActorType.PEDESTRIAN, 10, 0, 0.6, 0.6)
    result = compute_visibility(ego, ped, [truck])
    assert result.visibility_fraction == 0.0
    assert result.occluding_object == "truck"
    assert result.occlusion_type == OcclusionType.FULLY_OCCLUDED


def test_truck_behind_pedestrian_does_not_occlude():
    ego = actor("ego", ActorType.EGO, 0, 0, 4.5, 2.0)
    ped = actor("ped", ActorType.PEDESTRIAN, 5, 0, 0.6, 0.6)
    truck = actor("truck", ActorType.VEHICLE, 10, 0, 4.0, 3.0)
    result = compute_visibility(ego, ped, [truck])
    assert result.visibility_fraction == 1.0
    assert result.occluding_object is None
    assert result.occlusion_type == OcclusionType.VISIBLE
