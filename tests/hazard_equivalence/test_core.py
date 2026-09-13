import numpy as np

from sledge.hazard_equivalence.core import (
    ActorState,
    ActorType,
    LaneState,
    SceneState,
    TrajectoryState,
    validate_scene,
    validate_trajectory,
)


def test_scene_validation_passes():
    scene = SceneState(
        scene_id="s",
        actors=[
            ActorState(
                track_id="ego",
                actor_type=ActorType.EGO,
                position_xy=[0, 0],
                heading_rad=0,
                velocity_xy=[5, 0],
                length_m=4.5,
                width_m=1.9,
            ),
            ActorState(
                track_id="ped",
                actor_type=ActorType.PEDESTRIAN,
                position_xy=[10, 2],
                heading_rad=-np.pi / 2,
                velocity_xy=[0, -1],
                length_m=0.6,
                width_m=0.5,
            ),
        ],
        lanes=[
            LaneState("l0", [[0, 0], [10, 0]], successor_ids=("l1",)),
            LaneState("l1", [[10, 0], [20, 0]], predecessor_ids=("l0",)),
        ],
    )
    assert validate_scene(scene) == []


def test_trajectory_validation_passes():
    traj = TrajectoryState(
        timestamps_s=[0.0, 0.1],
        actor_ids=("ego", "ped"),
        positions_xy=np.zeros((2, 2, 2)),
        headings_rad=np.zeros((2, 2)),
        valid=np.ones((2, 2), dtype=bool),
    )
    assert validate_trajectory(traj) == []
