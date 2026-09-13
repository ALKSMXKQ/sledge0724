from types import SimpleNamespace

import numpy as np

from phase0_audit import ActorType
from phase0_audit.audit import roundtrip_check
from phase0_audit.sledge_adapter import SledgeProcessedVectorAdapter
from phase0_audit.sledge_contract import AGENT_INDEX, STATIC_OBJECT_INDEX, verified_contract


def element(states, mask):
    return SimpleNamespace(states=np.asarray(states, dtype=np.float32), mask=np.asarray(mask, dtype=bool))


def make_vector():
    vehicles = np.zeros((3, 6), dtype=np.float32)
    vehicles[0] = [10.0, 2.0, 0.0, 2.0, 4.5, 8.0]
    vehicles[2] = [-3.0, 1.0, np.pi / 2, 1.8, 4.2, 3.0]

    pedestrians = np.zeros((2, 6), dtype=np.float32)
    pedestrians[1] = [7.0, -2.0, -np.pi / 2, 0.5, 0.6, 1.2]

    static = np.zeros((2, 5), dtype=np.float32)
    static[0] = [6.0, 3.0, 0.2, 2.2, 5.0]

    lines = np.zeros((2, 4, 2), dtype=np.float32)
    lines[0] = [[0, 0], [1, 0], [2, 0], [3, 0]]

    ego = np.array([5.0, 0.0, 0.2, 0.0], dtype=np.float32)

    return SimpleNamespace(
        vehicles=element(vehicles, [True, False, True]),
        pedestrians=element(pedestrians, [False, True]),
        static_objects=element(static, [True, False]),
        lines=element(lines, [True, False]),
        green_lights=element(np.zeros((1, 4, 2)), [False]),
        red_lights=element(np.zeros((1, 4, 2)), [False]),
        ego=element(ego, np.array(True)),
    )


def test_verified_layout_values():
    contract = verified_contract()
    assert contract["agent_index"] == {
        "x": 0, "y": 1, "heading": 2, "width": 3, "length": 4, "speed": 5
    }
    assert contract["rvae"]["latent_frame"] == [8, 8]
    assert contract["rvae"]["latent_channels"] == 64


def test_sledge_processed_vector_to_canonical():
    adapter = SledgeProcessedVectorAdapter()
    scene = adapter.to_canonical(make_vector(), scene_id="toy")

    assert scene.frame == "ego_local"
    assert len(scene.actors) == 4
    assert len(scene.lanes) == 1
    assert scene.metadata["identity_preserved"] is False
    assert scene.metadata["lane_topology_preserved"] is False

    ped = scene.actor_by_id("pedestrian:1")
    assert ped.actor_type == ActorType.PEDESTRIAN
    assert np.allclose(ped.position_xy, [7.0, -2.0])
    assert np.isclose(ped.width_m, 0.5)
    assert np.isclose(ped.length_m, 0.6)
    assert np.isclose(ped.speed_mps, 1.2)
    assert np.allclose(ped.velocity_xy, [0.0, -1.2], atol=1e-6)


def test_sledge_roundtrip_preserves_valid_actor_fields_and_template():
    vector = make_vector()
    original_vehicle_padding = vector.vehicles.states[1].copy()
    adapter = SledgeProcessedVectorAdapter()

    before, restored, after, report = roundtrip_check(adapter, vector, scene_id="toy")
    assert report.ok
    assert len(before.actors) == len(after.actors)
    assert np.array_equal(restored.vehicles.states[1], original_vehicle_padding)
    assert np.array_equal(restored.vehicles.mask, vector.vehicles.mask)
    assert np.array_equal(restored.lines.states, vector.lines.states)

    vehicle = restored.vehicles.states[0]
    assert np.isclose(vehicle[AGENT_INDEX["width"]], 2.0)
    assert np.isclose(vehicle[AGENT_INDEX["length"]], 4.5)
    static = restored.static_objects.states[0]
    assert np.isclose(static[STATIC_OBJECT_INDEX["width"]], 2.2)
    assert np.isclose(static[STATIC_OBJECT_INDEX["length"]], 5.0)
