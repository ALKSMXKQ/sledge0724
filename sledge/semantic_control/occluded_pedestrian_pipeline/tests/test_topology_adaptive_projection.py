from types import SimpleNamespace

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeVector,
    SledgeVectorElement,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.metrics import (
    evaluate_occluded_pedestrian_scene,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.refinement_alignment import (
    OccludedPedestrianRefinementAlignmentEvaluator,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_spec import (
    ActorLayer,
    HazardSemanticSpec,
    InteractionLayer,
    ObjectLayer,
    OcclusionSpec,
    RiskLayer,
    RoadLayer,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.topology_adaptive_projection import (
    TopologyAdaptiveHazardProjector,
    heading_alignment_error,
)


def _straight_scene() -> SledgeVector:
    num_points = 20
    xs = np.linspace(-10.0, 40.0, num_points, dtype=np.float32)
    lines = np.zeros((6, num_points, 2), dtype=np.float32)
    line_mask = np.zeros(6, dtype=np.float32)
    for slot, y in enumerate((-5.25, -1.75, 1.75, 5.25)):
        lines[slot, :, 0] = xs
        lines[slot, :, 1] = y
        line_mask[slot] = 1.0

    vehicles = np.zeros((8, 6), dtype=np.float32)
    vehicle_mask = np.zeros(8, dtype=np.float32)
    vehicles[0] = np.asarray(
        [28.0, 0.0, 0.0, 2.0, 4.8, 8.0],
        dtype=np.float32,
    )
    vehicle_mask[0] = 1.0

    pedestrians = np.zeros((5, 6), dtype=np.float32)
    pedestrian_mask = np.zeros(5, dtype=np.float32)
    static_objects = np.zeros((5, 5), dtype=np.float32)
    static_mask = np.zeros(5, dtype=np.float32)

    empty_lines = SledgeVectorElement(
        states=np.zeros((2, num_points, 2), dtype=np.float32),
        mask=np.zeros(2, dtype=np.float32),
    )
    return SledgeVector(
        lines=SledgeVectorElement(lines, line_mask),
        vehicles=SledgeVectorElement(vehicles, vehicle_mask),
        pedestrians=SledgeVectorElement(
            pedestrians,
            pedestrian_mask,
        ),
        static_objects=SledgeVectorElement(
            static_objects,
            static_mask,
        ),
        green_lights=empty_lines,
        red_lights=empty_lines,
        ego=SledgeVectorElement(
            states=np.asarray(
                [8.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            mask=np.asarray([1.0], dtype=np.float32),
        ),
    )


def _angled_scene() -> SledgeVector:
    """Generated multi-lane road with a non-zero local tangent."""

    scene = _straight_scene()
    theta = 0.30
    slope = float(np.tan(theta))
    pivot_x = 20.0
    line_states = np.asarray(scene.lines.states)
    for slot, base_y in enumerate((-5.25, -1.75, 1.75, 5.25)):
        xs = line_states[slot, :, 0]
        line_states[slot, :, 1] = (
            base_y + slope * (xs - pivot_x)
        )
    scene.vehicles.states[0, 2] = theta
    return scene


def _spec(
    prompt: str,
    *,
    ttc_range_s=(2.0, 3.0),
) -> HazardSemanticSpec:
    return HazardSemanticSpec(
        spec_id="adaptive_test",
        raw_prompt=prompt,
        road_layer=RoadLayer(
            road_topology="straight",
            lane_width_m=3.5,
        ),
        actor_layer=ActorLayer(
            primary_actor="pedestrian",
            actor_role="crossing_actor",
        ),
        object_layer=ObjectLayer(
            occlusion=OcclusionSpec(
                enabled=True,
                occluder_type="vehicle",
                occlusion_position="between_ego_and_actor",
                occlusion_level="full",
            )
        ),
        interaction_layer=InteractionLayer(
            conflict_type="lateral_conflict",
            conflict_direction="left_to_right",
            interaction_goal="near_miss",
        ),
        risk_layer=RiskLayer(
            risk_level="moderate",
            ttc_range_s=ttc_range_s,
            target_actor_speed_mps=1.6,
        ),
    )


def _evaluate_projected(
    projected: SledgeVector,
    report,
    spec: HazardSemanticSpec,
):
    slots = report["projected_slots"]
    return evaluate_occluded_pedestrian_scene(
        projected,
        spec,
        preferred_pedestrian_index=slots["pedestrians"],
        preferred_occluder_index=slots["occluder_index"],
        preferred_occluder_elem_name=slots[
            "occluder_element"
        ],
        projection_time_s=2.1,
        lane_center_y=report["road_context"]["lane_center_y"],
    )


def test_topology_adaptive_uses_generated_road_and_ego() -> None:
    scene = _straight_scene()
    source_lines = np.asarray(scene.lines.states).copy()
    source_ego = np.asarray(scene.ego.states).copy()

    projector = TopologyAdaptiveHazardProjector(
        projection_time_s=2.1
    )
    spec = _spec(
        "A pedestrian suddenly emerges from behind a vehicle."
    )
    projected, report = projector.project(
        scene,
        spec,
        attempt_seed=7,
    )

    assert np.allclose(projected.lines.states, source_lines)
    assert np.allclose(projected.ego.states, source_ego)
    assert report["ego_state_source"] == "diffusion_generated_ego"
    assert report["hazard_variant"] == "adjacent_lane_dynamic"
    assert abs(report["road_context"]["lane_center_y"]) < 0.15
    assert (
        report["road_context"]["adjacent_lane_center_left_y"]
        is not None
    )

    occ = report["occluder"]["display_state"]
    assert occ[5] > 0.35
    assert (
        heading_alignment_error(
            occ[2],
            report["road_context"]["local_tangent_heading"],
        )
        < 0.10
    )

    metrics = _evaluate_projected(projected, report, spec)
    assert metrics["semantic_pass"] is True
    assert metrics["traffic_realism_pass"] is True
    assert metrics["overall_pass"] is True


def test_explicit_parked_vehicle_remains_stationary() -> None:
    projector = TopologyAdaptiveHazardProjector(
        projection_time_s=2.1
    )
    spec = _spec(
        "A pedestrian emerges from behind a parked vehicle "
        "on the left."
    )
    projected, report = projector.project(
        _straight_scene(),
        spec,
        attempt_seed=11,
    )
    assert report["hazard_variant"] == "roadside_parked"
    assert report["occluder"]["display_state"][5] == 0.0
    assert (
        report["occluder"]["placement"]["actor_occluder_overlap"]
        is False
    )
    assert (
        report["occluder"]["placement"]["actor_far_side_margin_m"]
        >= 0.10
    )

    metrics = _evaluate_projected(projected, report, spec)
    assert metrics["checks"]["no_actor_occluder_initial_overlap"] is True
    assert metrics["semantic_pass"] is True
    assert metrics["traffic_realism_pass"] is True


def test_aggressive_parked_vehicle_on_angled_road_is_feasible() -> None:
    """Regression for the real 1/8 smoke failure mode."""

    prompt = (
        "A pedestrian suddenly emerges from behind a parked vehicle "
        "on the left."
    )
    spec = _spec(prompt, ttc_range_s=(1.2, 2.0))
    projector = TopologyAdaptiveHazardProjector(
        projection_time_s=2.1
    )
    projected, report = projector.project(
        _angled_scene(),
        spec,
        attempt_seed=19,
    )

    placement = report["occluder"]["placement"]
    assert report["hazard_variant"] == "roadside_parked"
    assert placement["placement_solver"] == (
        "robust_rotated_footprint_far_side"
    )
    assert placement["actor_occluder_overlap"] is False
    assert placement["actor_far_side_margin_m"] >= 0.10
    assert placement["lane_boundary_gap_m"] >= 0.05
    assert placement["occluder_lateral_half_extent_m"] > 1.0

    metrics = _evaluate_projected(projected, report, spec)
    assert metrics["checks"]["line_of_sight_occlusion"] is True
    assert metrics["checks"]["occluder_clear_of_ego_corridor"] is True
    assert metrics["checks"]["no_actor_occluder_initial_overlap"] is True
    assert metrics["semantic_pass"] is True
    assert metrics["traffic_realism_pass"] is True
    assert metrics["overall_pass"] is True


def test_alignment_reuses_authoritative_b1_hazard_spec() -> None:
    """B2 alignment must not re-sample an ambiguous prompt direction."""

    prompt = "A pedestrian suddenly emerges from behind a vehicle."
    authoritative_spec = _spec(prompt)
    projector = TopologyAdaptiveHazardProjector(
        projection_time_s=2.1
    )
    projected, report = projector.project(
        _straight_scene(),
        authoritative_spec,
        attempt_seed=17,
    )

    evaluator = OccludedPedestrianRefinementAlignmentEvaluator(
        projection_time_s=2.1
    )
    evaluator.set_hazard_spec(authoritative_spec)
    evaluator.set_reference_scene(None)
    slots = report["projected_slots"]
    evaluator.set_preferred_slots(
        slots["pedestrians"],
        slots["occluder_index"],
        slots["occluder_element"],
    )
    evaluator.set_lane_center_y(
        report["road_context"]["lane_center_y"]
    )

    def _unexpected_prompt_readaptation(*args, **kwargs):
        raise AssertionError(
            "authoritative B1 spec should bypass prompt re-adaptation"
        )

    evaluator.adapter.adapt = _unexpected_prompt_readaptation
    prompt_spec = SimpleNamespace(
        raw_prompt=prompt,
        normalized_prompt=prompt,
    )
    result = evaluator.evaluate(projected, prompt_spec)

    assert result.accepted is True
    assert result.details["crossing_direction_score"] == 1.0
    assert any(
        "authoritative_b1_hazard_spec" in note
        for note in result.notes
    )
