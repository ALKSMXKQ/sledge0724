import numpy as np
import pytest

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeVectorElement,
    SledgeVectorRaw,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.elastic_context_editor import (
    ElasticContextHazardConstructor,
    ElasticContextPolicy,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_spec import (
    HazardSemanticSpec,
)


def _scene() -> SledgeVectorRaw:
    lines = SledgeVectorElement(
        states=np.arange(4 * 20 * 2, dtype=np.float32).reshape(4, 20, 2),
        mask=np.ones((4, 20), dtype=bool),
    )
    vehicles = SledgeVectorElement(
        states=np.asarray(
            [
                [8.0, 4.5, 0.0, 2.0, 4.8, 4.0],
                [18.0, 0.0, 0.0, 2.0, 4.8, 6.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        mask=np.asarray([True, True, False]),
    )
    pedestrians = SledgeVectorElement(
        states=np.zeros((2, 6), dtype=np.float32),
        mask=np.zeros((2,), dtype=bool),
    )
    static_objects = SledgeVectorElement(
        states=np.zeros((2, 5), dtype=np.float32),
        mask=np.zeros((2,), dtype=bool),
    )
    lights = SledgeVectorElement(
        states=np.zeros((1, 5), dtype=np.float32),
        mask=np.zeros((1,), dtype=bool),
    )
    ego = SledgeVectorElement(
        states=np.asarray([0.1, 0.0, 0.0, 0.0], dtype=np.float32),
        mask=np.asarray([True]),
    )
    return SledgeVectorRaw(
        lines=lines,
        vehicles=vehicles,
        pedestrians=pedestrians,
        static_objects=static_objects,
        green_lights=lights,
        red_lights=SledgeVectorElement(
            states=np.zeros((1, 5), dtype=np.float32),
            mask=np.zeros((1,), dtype=bool),
        ),
        ego=ego,
    )


def _spec() -> HazardSemanticSpec:
    spec = HazardSemanticSpec(spec_id="elastic-test")
    spec.road_layer.lane_width_m = 3.5
    spec.risk_layer.lateral_gap_range_m = (1.0, 2.0)
    spec.risk_layer.longitudinal_distance_range_m = (10.0, 18.0)
    return spec


def test_prepare_edits_ego_but_does_not_guess_background_blockers() -> None:
    scene = _scene()
    road_before = np.asarray(scene.lines.states).copy()
    vehicle_before = np.asarray(scene.vehicles.states).copy()
    vehicle_mask_before = np.asarray(scene.vehicles.mask).copy()

    report = ElasticContextHazardConstructor().prepare_scene_for_attempt(
        scene,
        _spec(),
        {
            "ego_speed_mps": 8.0,
            "ego_acceleration_mps2": 0.0,
            "occluder_side": "left",
            "occluder_length_m": 7.5,
            "occluder_width_m": 2.4,
        },
        attempt_index=0,
    )

    assert np.array_equal(scene.lines.states, road_before)
    assert np.array_equal(scene.vehicles.states, vehicle_before)
    assert np.array_equal(scene.vehicles.mask, vehicle_mask_before)
    assert float(np.asarray(scene.ego.states).reshape(-1)[0]) == pytest.approx(8.0)
    assert report["background_actor_edit_count"] == 0
    assert report["geometry_edit"]["lateral_gap_range_after_m"] == [1.0, 2.0]
    assert report["execution_hints"]["semantic_lane_center_y"] == pytest.approx(0.0)


def test_later_attempt_still_leaves_background_to_candidate_aware_solver() -> None:
    scene = _scene()
    vehicle_before = np.asarray(scene.vehicles.states).copy()
    mask_before = np.asarray(scene.vehicles.mask).copy()
    spec = _spec()

    report = ElasticContextHazardConstructor().prepare_scene_for_attempt(
        scene,
        spec,
        {
            "ego_speed_mps": 8.0,
            "ego_acceleration_mps2": 0.0,
            "occluder_length_m": 10.0,
            "occluder_width_m": 2.6,
        },
        attempt_index=2,
    )

    assert np.array_equal(scene.vehicles.states, vehicle_before)
    assert np.array_equal(scene.vehicles.mask, mask_before)
    assert report["background_actor_edits"] == []
    assert report["background_removal_count"] == 0
    assert report["execution_hints"]["occluder_length_m"] == pytest.approx(10.0)
    assert report["execution_hints"]["occluder_width_m"] == pytest.approx(2.6)
    assert spec.debug["occluder_length_m"] == pytest.approx(10.0)
    assert spec.debug["occluder_width_m"] == pytest.approx(2.6)


def test_policy_explicitly_allows_move_then_delete() -> None:
    policy = ElasticContextPolicy()
    assert policy.allow_background_reposition is True
    assert policy.allow_background_removal is True
    assert policy.background_policy == "move_then_delete_local_blockers"