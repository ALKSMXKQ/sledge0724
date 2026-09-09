"""Topology-adaptive hazard projection in a routed road-local frame.

This is the active v2 projector for generated roads.  It keeps the existing
semantic contract and slot-writing behavior, but replaces the global
``conflict_x = ego_speed * TTC`` geometry with an arc-length route distance on
SLEDGE's own directed road graph.  A legacy projection fallback is retained so
we can measure adoption before deleting the old path.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    AgentIndex,
    EgoIndex,
    SledgeVector,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.road_graph_validity import (
    evaluate_global_road_graph_validity,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.geometry_metrics import (
    line_of_sight_intersects_box,
    wrap_angle,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.primitive_ops import (
    OCCLUDER_SPECS,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.road_constraint_context import (
    EgoCorridor,
    LocalRoadFrame,
    RoadConstraintContext,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.topology_adaptive_projection import (
    MAX_CONFLICT_X_M,
    MIN_CONFLICT_X_M,
    TopologyAdaptiveHazardProjector as BaseTopologyAdaptiveHazardProjector,
    _point_line_distance,
    _rough_overlap,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.object_types import (
    element_name_for_occluder,
    normalize_occluder_type,
)


MIN_EGO_LANE_CLEARANCE_M = 0.05
ROADSIDE_TARGET_CLEARANCE_M = 0.15
MIN_ACTOR_FAR_SIDE_MARGIN_M = 0.10


class PathRelativeTopologyAdaptiveHazardProjector(
    BaseTopologyAdaptiveHazardProjector
):
    """Road-gated semantic projector using graph arc length and local normals."""

    def __init__(
        self,
        *,
        projection_time_s: float = 2.1,
        allow_legacy_fallback: bool = True,
    ) -> None:
        super().__init__(projection_time_s=projection_time_s)
        self.allow_legacy_fallback = bool(allow_legacy_fallback)

    def project(
        self,
        vector: SledgeVector,
        spec: Any,
        *,
        attempt_seed: int = 0,
    ) -> Tuple[SledgeVector, Dict[str, Any]]:
        """Validate the generated road, then project hazard geometry in ``s/d``."""

        road_graph = evaluate_global_road_graph_validity(vector)
        if not bool(road_graph.get("passed", False)):
            failed = [
                name
                for name, passed in road_graph.get("checks", {}).items()
                if not bool(passed)
            ]
            raise RuntimeError(
                "global road graph validity failed: "
                f"checks={failed}; "
                f"nodes={road_graph.get('num_lane_nodes')}; "
                f"edges={road_graph.get('num_lane_edges')}; "
                f"ego_forward_route_length_m="
                f"{road_graph.get('ego_forward_route_length_m')}; "
                f"orphan_lane_length_ratio="
                f"{road_graph.get('orphan_lane_length_ratio')}"
            )

        context_error = None
        try:
            context = RoadConstraintContext.from_scene(vector)
            ego_heading = self._generated_ego_heading(vector)
            corridor = context.infer_ego_corridor(
                ego_xy=(0.0, 0.0),
                ego_heading=ego_heading,
                lane_width_hint_m=float(spec.road_layer.lane_width_m),
            )
        except Exception as exc:
            context = None
            corridor = None
            context_error = f"{type(exc).__name__}: {exc}"

        if context is not None and corridor is not None:
            try:
                projected, report = self._project_path_relative(
                    vector,
                    spec,
                    context=context,
                    corridor=corridor,
                    attempt_seed=attempt_seed,
                )
                report["road_graph_validity"] = road_graph
                report["road_graph_policy"] = (
                    "reject_fragmented_diffusion_road_no_geometry_repair"
                )
                return projected, report
            except Exception as exc:
                context_error = f"{type(exc).__name__}: {exc}"
                if not self.allow_legacy_fallback:
                    raise

        if not self.allow_legacy_fallback:
            raise RuntimeError(
                "path-relative road projection unavailable: "
                f"{context_error or 'unknown error'}"
            )

        projected, report = super().project(
            vector,
            spec,
            attempt_seed=attempt_seed,
        )
        report = dict(report)
        report["schema_version"] = "topology_adaptive_hazard_projection_v2"
        report["projection_coordinate_system"] = "legacy_global_x_fallback"
        report["path_relative_fallback_reason"] = context_error
        report["road_graph_validity"] = road_graph
        report["road_graph_policy"] = (
            "reject_fragmented_diffusion_road_no_geometry_repair"
        )
        return projected, report

    def _project_path_relative(
        self,
        vector: SledgeVector,
        spec: Any,
        *,
        context: RoadConstraintContext,
        corridor: EgoCorridor,
        attempt_seed: int,
    ) -> Tuple[SledgeVector, Dict[str, Any]]:
        scene = deepcopy(vector)
        rng = np.random.default_rng(int(attempt_seed))

        ego_speed, ego_source, ego_repaired = self._generated_ego_speed(scene)
        ttc_values = self._ttc_candidates(spec, rng)
        direction = str(spec.interaction_layer.conflict_direction)
        if direction not in {"left_to_right", "right_to_left"}:
            raise RuntimeError(
                "topology-adaptive projection requires lateral direction, "
                f"got {direction!r}"
            )
        side_sign = 1.0 if direction == "left_to_right" else -1.0

        prompt = str(getattr(spec, "raw_prompt", "") or "")
        canonical_occluder = normalize_occluder_type(
            spec.object_layer.occlusion.occluder_type,
            strict=False,
        )
        target_elem_name = element_name_for_occluder(canonical_occluder)
        occ_spec = OCCLUDER_SPECS.get(
            canonical_occluder,
            OCCLUDER_SPECS["vehicle"],
        )
        debug = dict(getattr(spec, "debug", {}) or {})
        occ_width = self._positive(
            debug.get("occluder_width_m"),
            default=float(occ_spec.width),
            low=0.35,
            high=3.5,
        )
        occ_length = self._positive(
            debug.get("occluder_length_m"),
            default=float(occ_spec.length),
            low=0.45,
            high=16.0,
        )
        actor_speed = float(
            np.clip(spec.risk_layer.target_actor_speed_mps, 0.5, 2.0)
        )

        failures: List[str] = []
        for target_ttc in ttc_values:
            route_distance = float(ego_speed * target_ttc)
            if not MIN_CONFLICT_X_M <= route_distance <= MAX_CONFLICT_X_M:
                failures.append(
                    f"ttc={target_ttc:.2f}: route_distance={route_distance:.2f} "
                    "outside projection frame"
                )
                continue

            try:
                road = context.frame_at_route_distance(
                    corridor,
                    route_distance_m=route_distance,
                    lane_width_hint_m=float(spec.road_layer.lane_width_m),
                )
            except Exception as exc:
                failures.append(
                    f"ttc={target_ttc:.2f}: route advance failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            if ego_repaired:
                self._write_repaired_ego_velocity(
                    scene,
                    ego_speed,
                    road.heading,
                )

            actor_heading = float(
                wrap_angle(road.heading - side_sign * math.pi / 2.0)
            )
            actor_d = float(
                side_sign
                * (road.lane_half_width_m + actor_speed * target_ttc)
            )
            actor_xy = road.point_at_lateral(actor_d)
            actor_display = np.asarray(
                [
                    float(actor_xy[0]),
                    float(actor_xy[1]),
                    actor_heading,
                    0.75,
                    0.75,
                    actor_speed,
                ],
                dtype=np.float32,
            )

            adjacent_offset = road.adjacent_lane_center_offset(side_sign)
            variant = self._select_hazard_variant(
                canonical_occluder,
                prompt,
                adjacent_offset,
            )
            placement = self._solve_occluder_frame(
                actor_display=actor_display,
                road=road,
                side_sign=side_sign,
                variant=variant,
                occluder_width=occ_width,
                occluder_length=occ_length,
                ego_speed=ego_speed,
            )
            if placement is None:
                failures.append(
                    f"ttc={target_ttc:.2f}: no LOS-valid {variant} "
                    "occluder placement in path-relative frame"
                )
                continue

            pedestrian_index, ped_replaced = self._allocate_slot(
                scene.pedestrians
            )
            occluder_elem = getattr(scene, target_elem_name)
            occluder_index, occ_replaced = self._allocate_slot(
                occluder_elem
            )

            actor_raw = self._display_to_raw_agent(actor_display)
            self._write_agent(
                scene.pedestrians,
                pedestrian_index,
                actor_raw,
            )

            occ_display = np.asarray(
                [
                    placement["x"],
                    placement["y"],
                    placement["heading"],
                    occ_width,
                    occ_length,
                    placement["speed_mps"],
                ],
                dtype=np.float32,
            )
            if target_elem_name == "vehicles":
                occ_raw = self._display_to_raw_agent(occ_display)
                self._write_agent(
                    occluder_elem,
                    occluder_index,
                    occ_raw,
                )
            else:
                self._write_static(
                    occluder_elem,
                    occluder_index,
                    occ_display[:5],
                )

            background_edits = self._clear_local_overlaps(
                scene,
                pedestrian_ref=("pedestrians", pedestrian_index),
                occluder_ref=(target_elem_name, occluder_index),
            )

            road_context = road.to_dict()
            # Transitional compatibility for metrics that still accept a
            # scalar lane-center y.  New consumers should use center_xy + d.
            road_context["lane_center_y"] = float(road.center_xy[1])
            road_context["ego_corridor"] = corridor.to_dict()

            report = {
                "schema_version": "topology_adaptive_hazard_projection_v2",
                "projection_policy": "copy_semantics_recompute_geometry",
                "projection_coordinate_system": "path_relative_s_d",
                "semantic_projection_time_s": float(self.projection_time_s),
                "generated_road_preserved": True,
                "generated_ego_preserved": not ego_repaired,
                "ego_speed_mps": float(ego_speed),
                "ego_state_source": ego_source,
                "road_context": road_context,
                "semantic_direction": direction,
                "hazard_side": "left" if side_sign > 0 else "right",
                "hazard_variant": variant,
                "target_ttc_s": float(target_ttc),
                "target_ttc_range_s": [
                    float(spec.risk_layer.ttc_range_s[0]),
                    float(spec.risk_layer.ttc_range_s[1]),
                ],
                "conflict_s_m": float(route_distance),
                "conflict_xy": [float(v) for v in road.center_xy],
                # Kept only for old report readers; it is no longer the
                # independent variable used to choose the conflict point.
                "conflict_x_m": float(road.center_xy[0]),
                "pedestrian": {
                    "index": int(pedestrian_index),
                    "replaced_generated_slot": bool(ped_replaced),
                    "display_state": actor_display.tolist(),
                    "raw_state": actor_raw.tolist(),
                    "lateral_d_m": float(actor_d),
                },
                "occluder": {
                    "element": target_elem_name,
                    "index": int(occluder_index),
                    "canonical_type": canonical_occluder,
                    "replaced_generated_slot": bool(occ_replaced),
                    "display_state": occ_display.tolist(),
                    "placement": dict(placement),
                },
                "projected_slots": {
                    "pedestrians": int(pedestrian_index),
                    "occluder_element": target_elem_name,
                    "occluder_index": int(occluder_index),
                },
                "background_local_edits": background_edits,
                "background_local_edit_count": len(background_edits),
                "candidate_failures_before_success": failures,
            }
            return scene, report

        raise RuntimeError(
            "path-relative topology-adaptive hazard projection failed for all "
            "risk-conditioned TTC candidates: "
            + "; ".join(failures[-8:])
        )

    @staticmethod
    def _generated_ego_heading(scene: SledgeVector) -> float:
        states = np.asarray(scene.ego.states, dtype=np.float64).reshape(-1)
        if states.size >= 2:
            vx = float(states[EgoIndex.VELOCITY_X])
            vy = float(states[EgoIndex.VELOCITY_Y])
            if math.isfinite(vx) and math.isfinite(vy) and math.hypot(vx, vy) > 0.5:
                return float(math.atan2(vy, vx))
        return 0.0

    def _solve_occluder_frame(
        self,
        *,
        actor_display: np.ndarray,
        road: LocalRoadFrame,
        side_sign: float,
        variant: str,
        occluder_width: float,
        occluder_length: float,
        ego_speed: float,
    ) -> Optional[Dict[str, float]]:
        actor_xy = np.asarray(
            [
                float(actor_display[AgentIndex.X]),
                float(actor_display[AgentIndex.Y]),
            ],
            dtype=np.float64,
        )
        center = np.asarray(road.center_xy, dtype=np.float64)
        tangent = road.tangent
        normal = road.normal
        actor_d = float(np.dot(actor_xy - center, normal))
        adjacent = road.adjacent_lane_center_offset(side_sign)

        if variant == "adjacent_lane_dynamic" and adjacent is not None:
            desired_d = float(adjacent)
            speed = float(np.clip(0.75 * ego_speed, 2.0, 13.0))
        else:
            desired_d = float(
                side_sign
                * (
                    road.lane_half_width_m
                    + 0.5 * float(occluder_width)
                    + ROADSIDE_TARGET_CLEARANCE_M
                )
            )
            speed = 0.0

        heading = float(road.heading)
        actor_range_sq = float(np.dot(actor_xy, actor_xy))
        if actor_range_sq <= 1.0:
            return None

        candidates: List[Tuple[float, Dict[str, float]]] = []
        for ratio in (0.42, 0.48, 0.55, 0.62, 0.70, 0.78, 0.84, 0.90):
            ray = float(ratio) * actor_xy
            longitudinal = float(np.dot(ray - center, tangent))
            ray_d = float(np.dot(ray - center, normal))
            for mix in (0.0, 0.20, 0.40, 0.60, 0.80, 1.0):
                candidate_d = float((1.0 - mix) * ray_d + mix * desired_d)
                candidate_xy = (
                    center
                    + longitudinal * tangent
                    + candidate_d * normal
                )
                los_fraction = float(
                    np.dot(candidate_xy, actor_xy) / actor_range_sq
                )
                if not 0.25 <= los_fraction <= 0.97:
                    continue

                far_side_margin = float(
                    side_sign * (actor_d - candidate_d)
                )
                if far_side_margin < MIN_ACTOR_FAR_SIDE_MARGIN_M:
                    continue

                occ = np.asarray(
                    [
                        float(candidate_xy[0]),
                        float(candidate_xy[1]),
                        heading,
                        occluder_width,
                        occluder_length,
                        speed,
                    ],
                    dtype=np.float32,
                )
                if _rough_overlap(
                    actor_display,
                    occ,
                    "pedestrians",
                    "vehicles",
                ):
                    continue
                if not line_of_sight_intersects_box(
                    (0.0, 0.0),
                    (float(actor_xy[0]), float(actor_xy[1])),
                    occ,
                    margin=0.18,
                ):
                    continue

                local_lateral_half = 0.5 * float(occluder_width)
                inner_gap = float(
                    abs(candidate_d)
                    - local_lateral_half
                    - road.lane_half_width_m
                )
                if inner_gap < MIN_EGO_LANE_CLEARANCE_M:
                    continue
                if abs(candidate_d) > max(9.0, 2.8 * road.lane_width_m):
                    continue

                perpendicular = _point_line_distance(
                    (float(candidate_xy[0]), float(candidate_xy[1])),
                    (0.0, 0.0),
                    (float(actor_xy[0]), float(actor_xy[1])),
                )
                score = float(
                    abs(candidate_d - desired_d)
                    + 0.20 * perpendicular
                    + 0.50 * abs(los_fraction - 0.68)
                    + 0.10 * max(0.0, 0.40 - far_side_margin)
                )
                candidates.append(
                    (
                        score,
                        {
                            "x": float(candidate_xy[0]),
                            "y": float(candidate_xy[1]),
                            "heading": heading,
                            "speed_mps": speed,
                            "lateral_d_m": float(candidate_d),
                            "desired_lateral_d_m": float(desired_d),
                            "lane_boundary_gap_m": inner_gap,
                            "line_perpendicular_error_m": float(perpendicular),
                            "actor_far_side_margin_m": far_side_margin,
                            "los_fraction": los_fraction,
                            "placement_solver": "path_relative_rotated_frame",
                        },
                    )
                )

        if not candidates:
            return None
        candidates.sort(key=lambda row: row[0])
        return candidates[0][1]


__all__ = ["PathRelativeTopologyAdaptiveHazardProjector"]
