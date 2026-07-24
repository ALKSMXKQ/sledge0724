from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVectorElement
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.scene_normalization import (
    compact_edited_scene,
    normalize_editable_scene,
)


def _element(states, width: int):
    array = np.asarray(states, dtype=np.float32)
    if not len(array):
        array = np.asarray([], dtype=np.float32)
    return SledgeVectorElement(states=array, mask=np.ones(len(array), dtype=bool))


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
