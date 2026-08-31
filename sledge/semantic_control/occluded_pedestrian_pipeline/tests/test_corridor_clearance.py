from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVector, SledgeVectorElement
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.metrics import (
    evaluate_occluded_pedestrian_scene,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.refinement_alignment import (
    evaluate_road_topology_preservation,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.stage_comparison import (
    _make_simulation_compatible_vector,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_spec import (
    HazardSemanticSpec,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.primitive_ops import (
    OCCLUDER_SPECS,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.refinement_runner import (
    OccludedPedestrianHalfDenoiseRunner,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.language.eventframe_adapter import (
    OccludedPedestrianEventFrameAdapter,
)


def _elem(states):
    states = np.asarray(states, dtype=np.float32)
    if states.size == 0:
        states = np.zeros((0, 6), dtype=np.float32)
    return SledgeVectorElement(states=states, mask=np.ones(len(states), dtype=bool))


def _scene(occluder_y: float, occluder_heading: float):
    return SimpleNamespace(
        lines=_elem([]),
        vehicles=_elem([[7.15, occluder_y, occluder_heading, 2.1, 5.0, 0.0]]),
        pedestrians=_elem([[11.0, -4.0, np.pi / 2.0, 0.75, 0.75, 1.6]]),
        static_objects=_elem([]),
        green_lights=_elem([]),
        red_lights=_elem([]),
        ego=_elem([[4.4, 0.0, 0.0, 0.0]]),
    )


def test_vehicle_occluders_are_parallel_to_road() -> None:
    for name in ("vehicle", "bicycle"):
        assert OCCLUDER_SPECS[name].default_heading == 0.0


def test_occluder_must_occlude_without_entering_ego_corridor() -> None:
    spec = OccludedPedestrianEventFrameAdapter(llm_provider="none").adapt(
        "A pedestrian hidden behind a parked vehicle crosses right to left at 1.6 m/s."
    ).hazard_spec
    safe = evaluate_occluded_pedestrian_scene(
        _scene(-4.50, 0.0),
        spec,
        preferred_pedestrian_index=0,
        preferred_occluder_index=0,
    )
    assert safe["checks"]["line_of_sight_occlusion"] is True
    assert safe["checks"]["occluder_clear_of_ego_corridor"] is True

    unsafe = evaluate_occluded_pedestrian_scene(
        _scene(-2.20, np.pi / 2.0),
        spec,
        preferred_pedestrian_index=0,
        preferred_occluder_index=0,
    )
    assert unsafe["checks"]["line_of_sight_occlusion"] is True
    assert unsafe["checks"]["occluder_clear_of_ego_corridor"] is False
    assert unsafe["overall_pass"] is False


def test_metrics_use_source_lane_center_and_require_stationary_parked_occluder() -> None:
    """Keep metric tests independent from natural-language parsing.

    This case specifically exercises the legacy ``occluder_stationary`` key for
    the new ``roadside_parked`` mode.  Dynamic adjacent-lane vehicle occluders
    are intentionally allowed to move and are governed by
    ``occluder_motion_plausibility`` instead.
    """

    spec = HazardSemanticSpec(spec_id="parked-occluder-metrics")
    spec.road_layer.lane_width_m = 3.5
    spec.actor_layer.primary_actor = "pedestrian"
    spec.actor_layer.actor_role = "crossing_actor"
    spec.object_layer.occlusion.enabled = True
    spec.object_layer.occlusion.occluder_type = "vehicle"
    spec.interaction_layer.conflict_type = "lateral_conflict"
    spec.interaction_layer.conflict_direction = "right_to_left"
    spec.risk_layer.target_actor_speed_mps = 1.6
    spec.risk_layer.ttc_range_s = (2.0, 3.0)
    spec.debug["occlusion_mode"] = "roadside_parked"

    scene = _scene(-3.50, np.pi / 2.0)
    scene.vehicles.states[0, 3:6] = [0.6, 1.0, 0.0]
    shifted = evaluate_occluded_pedestrian_scene(
        scene,
        spec,
        preferred_pedestrian_index=0,
        preferred_occluder_index=0,
        lane_center_y=1.0,
    )
    assert shifted["interaction"]["lane_center_y"] == 1.0
    assert shifted["checks"]["occluder_stationary"] is True

    scene.vehicles.states[0, 5] = 0.5
    moving = evaluate_occluded_pedestrian_scene(
        scene,
        spec,
        preferred_pedestrian_index=0,
        preferred_occluder_index=0,
        lane_center_y=1.0,
    )
    assert moving["checks"]["occluder_stationary"] is False
    assert moving["traffic_realism"]["checks"]["occluder_motion_plausibility"] is False
    assert moving["overall_pass"] is False


def test_tangled_diffusion_lines_fail_topology_preservation() -> None:
    reference = SimpleNamespace(
        lines=SledgeVectorElement(
            states=np.asarray([[[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]], dtype=np.float32),
            mask=np.asarray([1.0], dtype=np.float32),
        )
    )
    coherent = SimpleNamespace(
        lines=SledgeVectorElement(
            states=np.asarray([[[0.0, 0.2], [5.0, 0.2], [10.0, 0.2]]], dtype=np.float32),
            mask=np.asarray([1.0], dtype=np.float32),
        )
    )
    tangled = SimpleNamespace(
        lines=SledgeVectorElement(
            states=np.asarray([[[0.0, 8.0], [5.0, 8.0], [10.0, 8.0]]], dtype=np.float32),
            mask=np.asarray([1.0], dtype=np.float32),
        )
    )
    assert evaluate_road_topology_preservation(reference, coherent)["passed"] is True
    assert evaluate_road_topology_preservation(reference, tangled)["passed"] is False


def test_stage_cache_converts_raw_ego_vector_to_scalar_speed() -> None:
    raw = _scene(-4.50, 0.0)
    raw.ego = SledgeVectorElement(
        states=np.asarray([7.5, 0.2, -0.1, 0.0], dtype=np.float32),
        mask=np.asarray([True], dtype=bool),
    )
    processed = SledgeVector(
        lines=raw.lines,
        vehicles=raw.vehicles,
        pedestrians=raw.pedestrians,
        static_objects=raw.static_objects,
        green_lights=raw.green_lights,
        red_lights=raw.red_lights,
        ego=raw.ego,
    )
    adapted = _make_simulation_compatible_vector(processed, raw)
    assert adapted.ego.states.shape == (1,)
    assert float(adapted.ego.states[0]) == 7.5
    assert adapted.ego.mask.shape == (1,)


def test_refinement_hard_locks_b1_road_topology() -> None:
    template = _scene(-4.50, 0.0)
    template.lines = SledgeVectorElement(
        states=np.asarray([[[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]], dtype=np.float32),
        mask=np.asarray([True]),
    )
    decoded = _scene(0.0, np.pi / 2.0)
    decoded.lines = SledgeVectorElement(
        states=np.asarray([[[0.0, 8.0], [5.0, -8.0], [10.0, 8.0]]], dtype=np.float32),
        mask=np.asarray([True]),
    )
    OccludedPedestrianHalfDenoiseRunner._composite_protected_slots(
        decoded,
        template,
        {"pedestrian_index": 0, "occluder_elem_name": "vehicles", "occluder_index": 0},
    )
    np.testing.assert_array_equal(decoded.lines.states, template.lines.states)
    np.testing.assert_array_equal(decoded.lines.mask, template.lines.mask)
    np.testing.assert_array_equal(decoded.vehicles.states[0], template.vehicles.states[0])
