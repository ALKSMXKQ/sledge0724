from __future__ import annotations

from pathlib import Path

import numpy as np

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    AgentIndex,
    SledgeVector,
    SledgeVectorElement,
    StaticObjectIndex,
)
from sledge.semantic_control.io import save_gz_pickle
from sledge.semantic_control.occluded_pedestrian_pipeline.object_types import (
    embed_type_overrides,
    normalize_occluder_type,
    read_type_overrides,
)
from sledge.simulation.scenarios.sledge_scenario.sledge_scenario_utils import (
    sledge_vector_to_detection_tracks,
)


def _element(states: np.ndarray) -> SledgeVectorElement:
    return SledgeVectorElement(states=states, mask=np.ones(len(states), dtype=bool))


def test_vehicle_subtypes_canonicalize_to_visible_vehicle() -> None:
    for alias in ("car", "van", "truck", "bus", "parked_vehicle"):
        assert normalize_occluder_type(alias) == "vehicle"


def test_vector_type_overrides_decode_to_nuplan_categories() -> None:
    vehicle = np.zeros((1, AgentIndex.size()), dtype=np.float32)
    vehicle[0, AgentIndex.WIDTH] = 0.7
    vehicle[0, AgentIndex.LENGTH] = 1.8
    static = np.zeros((1, StaticObjectIndex.size()), dtype=np.float32)
    static[0, StaticObjectIndex.WIDTH] = 0.6
    static[0, StaticObjectIndex.LENGTH] = 2.0
    empty_agent = np.zeros((0, AgentIndex.size()), dtype=np.float32)
    empty_line = np.zeros((0, 20, 2), dtype=np.float32)
    vector = SledgeVector(
        lines=_element(empty_line),
        vehicles=_element(vehicle),
        pedestrians=_element(empty_agent),
        static_objects=_element(static),
        green_lights=_element(empty_line),
        red_lights=_element(empty_line),
        ego=SledgeVectorElement(np.zeros(1, dtype=np.float32), np.ones(1, dtype=bool)),
    )
    detections = sledge_vector_to_detection_tracks(
        vector,
        0,
        {"vehicles": {"0": "BICYCLE"}, "static_objects": {"0": "BARRIER"}},
    )
    decoded = {obj.track_token: obj.tracked_object_type for obj in detections.tracked_objects}
    assert decoded == {"2_0": TrackedObjectType.BICYCLE, "4_0": TrackedObjectType.BARRIER}


def test_type_metadata_is_embedded_in_gzip(tmp_path: Path) -> None:
    path = save_gz_pickle(tmp_path / "sledge_vector", {"lines": {}})
    overrides = {"static_objects": {"0": "CZONE_SIGN"}}
    embed_type_overrides(path, overrides)
    assert read_type_overrides(path) == overrides
