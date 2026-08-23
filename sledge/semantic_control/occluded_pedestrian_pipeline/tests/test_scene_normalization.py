from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVectorElement
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.scene_normalization import (
    compact_edited_scene,
    normalize_editable_scene,
)


def _element(states, width: int, *, mask=None):
    array = np.asarray(states, dtype=np.float32)
    if not len(array):
        array = np.asarray([], dtype=np.float32)
    if mask is None:
        mask = np.ones(len(array), dtype=bool)
    return SledgeVectorElement(states=array, mask=np.asarray(mask, dtype=bool))


def test_empty_elements_can_be_inserted_then_compacted() -> None:
    scene = SimpleNamespace(
        vehicles=_element([], 6),
        pedestrians=_element([], 6),
        static_objects=_element([[5.0, 3.0, 0.0, 1.0, 1.0]], 5),
    )
    editable, _ = normalize_editable_scene(scene)
    assert editable.vehicles.states.shape == (1, 6)
    assert editable.pedestrians.states.shape == (1, 6)
    editable.vehicles.states[0] = [7.0, -2.0, 1.57, 2.1, 5.0, 0.0]
    editable.vehicles.mask[0] = True
    editable.pedestrians.states[0] = [12.0, -7.0, 1.57, 0.75, 0.75, 1.6]
    editable.pedestrians.mask[0] = True
    result = SimpleNamespace(
        pedestrian_index=0,
        occluder_index=0,
        occluder_elem_name="vehicles",
        static_obstacle_index=-1,
        primary_actor_type="pedestrian",
        primary_actor_index=0,
    )
    compacted, report = compact_edited_scene(editable, result)
    assert compacted.vehicles.states.shape == (1, 6)
    assert compacted.pedestrians.states.shape == (1, 6)
    assert compacted.vehicles.mask.tolist() == [True]
    assert compacted.pedestrians.mask.tolist() == [True]
    assert result.pedestrian_index == 0
    assert result.occluder_index == 0
    assert report["final_indices"]["pedestrian_index"] == 0


def test_raw_false_masks_do_not_allow_source_rows_to_be_overwritten() -> None:
    source_vehicle = [20.0, 3.5, 0.0, 2.0, 4.8, 5.0]
    source_pedestrian = [30.0, -2.0, 1.57, 0.75, 0.75, 1.2]
    source_static = [12.0, 4.0, 0.0, 1.0, 2.0]
    scene = SimpleNamespace(
        vehicles=_element([source_vehicle], 6, mask=[False]),
        pedestrians=_element([source_pedestrian], 6, mask=[False]),
        static_objects=_element([source_static], 5, mask=[False]),
    )

    editable, normalization = normalize_editable_scene(scene)

    # Raw SLEDGE rows are entities even when the serialized raw mask is false.
    assert editable.vehicles.mask.tolist() == [True, False]
    assert editable.pedestrians.mask.tolist() == [True, False]
    assert editable.static_objects.mask.tolist() == [True, False]
    assert normalization["elements"]["vehicles"]["insertion_slot_index"] == 1

    editable.vehicles.states[1] = [8.0, -3.0, 1.57, 2.1, 5.0, 0.0]
    editable.vehicles.mask[1] = True
    editable.pedestrians.states[1] = [12.0, -7.0, 1.57, 0.75, 0.75, 1.6]
    editable.pedestrians.mask[1] = True
    result = SimpleNamespace(
        pedestrian_index=1,
        occluder_index=1,
        occluder_elem_name="vehicles",
        static_obstacle_index=-1,
        primary_actor_type="pedestrian",
        primary_actor_index=1,
    )

    compacted, _ = compact_edited_scene(editable, result)

    assert compacted.vehicles.states.shape == (2, 6)
    assert compacted.pedestrians.states.shape == (2, 6)
    assert compacted.static_objects.states.shape == (1, 5)
    np.testing.assert_allclose(compacted.vehicles.states[0], source_vehicle)
    np.testing.assert_allclose(compacted.pedestrians.states[0], source_pedestrian)
    np.testing.assert_allclose(compacted.static_objects.states[0], source_static)
    assert result.pedestrian_index == 1
    assert result.occluder_index == 1
