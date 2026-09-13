import numpy as np

from phase0_audit import (
    ActorState,
    ActorType,
    ExplicitActorMatrixSchema,
    ExplicitMatrixSceneAdapter,
    SceneState,
    roundtrip_check,
    structure_report,
)


def make_adapter():
    schema = ExplicitActorMatrixSchema(
        x=0, y=1, heading=2, length=3, width=4,
        vx=5, vy=6, actor_type=7, track_id=8, valid=9,
    )
    adapter = ExplicitMatrixSceneAdapter(
        schema,
        {0: ActorType.VEHICLE, 1: ActorType.PEDESTRIAN, 2: ActorType.EGO},
        frame="ego_local",
    )
    matrix = np.array([
        [0.0, 0.0, 0.0, 4.5, 1.9, 5.0, 0.0, 2, 100, 1, 999],
        [8.0, 2.0, -1.57, 0.6, 0.5, 0.0, -1.2, 1, 101, 1, 123],
    ], dtype=float)
    return schema, adapter, matrix


def test_structure_report_numpy():
    report = structure_report({"actors": np.zeros((3, 11), dtype=np.float32)})
    assert report["items"]["actors"]["shape"] == [3, 11]


def test_canonical_actor_fields():
    actor = ActorState("ped", ActorType.PEDESTRIAN, [1, 2], 0.0, [0, 1], 0.6, 0.5)
    scene = SceneState("s", [actor])
    assert scene.actor_by_id("ped").speed_mps == 1.0


def test_adapter_fields():
    _, adapter, matrix = make_adapter()
    scene = adapter.to_canonical(matrix, scene_id="s")
    ped = scene.actor_by_id("101")
    assert ped.actor_type == ActorType.PEDESTRIAN
    assert np.allclose(ped.position_xy, [8.0, 2.0])
    assert np.allclose(ped.velocity_xy, [0.0, -1.2])


def test_roundtrip_preserves_unknown_columns():
    schema, adapter, matrix = make_adapter()
    _, restored, _, report = roundtrip_check(adapter, matrix, scene_id="s")
    assert report.ok
    assert np.allclose(matrix[:, schema.mapped_columns], restored[:, schema.mapped_columns])
    assert np.allclose(matrix[:, 10], restored[:, 10])
