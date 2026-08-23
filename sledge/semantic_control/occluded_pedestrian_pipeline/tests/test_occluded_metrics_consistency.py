import math

import numpy as np
import pytest

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeVectorElement,
    SledgeVectorRaw,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.metrics import (
    evaluate_occluded_pedestrian_scene,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_spec import (
    HazardSemanticSpec,
)


def _scene(*, lane_center_y: float = 0.0) -> SledgeVectorRaw:
    lane_half = 1.75
    ped_speed = 1.6
    target_t = 2.5
    ped_x = 18.0
    ped_y = lane_center_y + lane_half + ped_speed * target_t
    ego_speed = ped_x / target_t

    # Place a stationary vehicle exactly on the ego->pedestrian sight line.
    ratio = 0.80
    occ_x = ratio * ped_x
    occ_y = ratio * ped_y

    lines = SledgeVectorElement(
        states=np.zeros((2, 20, 2), dtype=np.float32),
        mask=np.zeros((2, 20), dtype=bool),
    )
    vehicles = SledgeVectorElement(
        states=np.asarray(
            [[occ_x, occ_y, 0.0, 2.0, 5.0, 0.0]],
            dtype=np.float32,
        ),
        mask=np.asarray([True]),
    )
    pedestrians = SledgeVectorElement(
        states=np.asarray(
            [[ped_x, ped_y, -math.pi / 2.0, 0.75, 0.75, ped_speed]],
            dtype=np.float32,
        ),
        mask=np.asarray([True]),
    )
    static_objects = SledgeVectorElement(
        states=np.zeros((1, 5), dtype=np.float32),
        mask=np.zeros((1,), dtype=bool),
    )
    lights = SledgeVectorElement(
        states=np.zeros((1, 5), dtype=np.float32),
        mask=np.zeros((1,), dtype=bool),
    )
    ego = SledgeVectorElement(
        states=np.asarray([ego_speed, 0.0, 0.0, 0.0], dtype=np.float32),
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
    spec = HazardSemanticSpec(spec_id="metrics")
    spec.road_layer.lane_width_m = 3.5
    spec.actor_layer.primary_actor = "pedestrian"
    spec.object_layer.occlusion.enabled = True
    spec.object_layer.occlusion.occluder_type = "vehicle"
    spec.interaction_layer.conflict_type = "lateral_conflict"
    spec.interaction_layer.conflict_direction = "left_to_right"
    spec.risk_layer.target_actor_speed_mps = 1.6
    spec.risk_layer.ttc_range_s = (2.0, 3.0)
    return spec


def test_metrics_use_lane_boundary_timing_and_report_true_pedestrian_heading() -> None:
    metrics = evaluate_occluded_pedestrian_scene(
        _scene(lane_center_y=0.0),
        _spec(),
        preferred_pedestrian_index=0,
        preferred_occluder_index=0,
        preferred_occluder_elem_name="vehicles",
        lane_center_y=0.0,
    )

    assert metrics["overall_pass"] is True
    assert metrics["interaction"]["pedestrian_lane_entry_time_s"] == pytest.approx(2.5, abs=1e-5)
    assert metrics["interaction"]["ego_arrival_time_s"] == pytest.approx(2.5, abs=1e-5)
    assert metrics["interaction"]["arrival_time_error_s"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["pedestrian"]["heading"] == pytest.approx(-math.pi / 2.0, abs=1e-5)
    assert metrics["checks"]["occluder_stationary"] is True


def test_metrics_respect_explicit_nonzero_lane_center_reference() -> None:
    lane_center_y = 1.0
    metrics = evaluate_occluded_pedestrian_scene(
        _scene(lane_center_y=lane_center_y),
        _spec(),
        preferred_pedestrian_index=0,
        preferred_occluder_index=0,
        preferred_occluder_elem_name="vehicles",
        lane_center_y=lane_center_y,
    )

    assert metrics["interaction"]["lane_center_y"] == pytest.approx(1.0)
    assert metrics["interaction"]["pedestrian_lane_entry_time_s"] == pytest.approx(2.5, abs=1e-5)
    assert metrics["checks"]["interaction_timing_match"] is True
    assert metrics["checks"]["occluder_clear_of_ego_corridor"] is True