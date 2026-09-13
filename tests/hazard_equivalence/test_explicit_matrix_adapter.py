import numpy as np

from sledge.hazard_equivalence.adapters import (
    ExplicitActorMatrixSchema,
    ExplicitMatrixSceneAdapter,
)
from sledge.hazard_equivalence.audit import roundtrip_check
from sledge.hazard_equivalence.core import ActorType


def _make():
    # Test-only schema; not a claim about SLEDGE.
    schema = ExplicitActorMatrixSchema(
        x=0,
        y=1,
        heading=2,
        length=3,
        width=4,
        vx=5,
        vy=6,
        actor_type=7,
        track_id=8,
        valid=9,
    )
    adapter = ExplicitMatrixSceneAdapter(
        schema,
        {0: ActorType.VEHICLE, 1: ActorType.PEDESTRIAN, 2: ActorType.EGO},
        frame="ego_local",
    )
    matrix = np.array(
        [
            [0.0, 0.0, 0.0, 4.5, 1.9, 5.0, 0.0, 2, 100, 1, 999],
            [8.0, 2.0, -1.57, 0.6, 0.5, 0.0, -1.2, 1, 101, 1, 123],
        ],
        dtype=float,
    )
    return schema, adapter, matrix


def test_to_canonical_fields():
    _, adapter, matrix = _make()
    scene = adapter.to_canonical(matrix, scene_id="s")
    ped = scene.actor_by_id("101")
    assert ped.actor_type == ActorType.PEDESTRIAN
    assert np.allclose(ped.position_xy, [8.0, 2.0])
    assert np.allclose(ped.velocity_xy, [0.0, -1.2])
    assert ped.length_m == 0.6
    assert ped.width_m == 0.5


def test_roundtrip_preserves_mapped_and_unmapped_columns():
    schema, adapter, matrix = _make()
    _, legacy_after, _, report = roundtrip_check(adapter, matrix, scene_id="s")
    assert report.ok, report.messages
    assert np.allclose(matrix[:, schema.mapped_columns], legacy_after[:, schema.mapped_columns])
    assert np.allclose(matrix[:, 10], legacy_after[:, 10])  # unmapped template data preserved
