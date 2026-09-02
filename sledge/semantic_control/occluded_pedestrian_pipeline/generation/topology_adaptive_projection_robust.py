"""Robust topology-adaptive projector with global road-graph gating.

This module keeps the historical base topology-adaptive projector intact for
ablation/reference while adding two active protections:

1. generated roads must pass SLEDGE's own global lane-graph validity gate;
2. roadside parked/static occluders use robust rotated-footprint geometry.

The road gate never repairs or copies road geometry.  A fragmented diffusion
road is rejected so the refinement runner can sample another repair attempt.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    AgentIndex,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.road_graph_validity import (
    evaluate_global_road_graph_validity,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.geometry_metrics import (
    line_of_sight_intersects_box,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.topology_adaptive_projection import (
    LocalRoadContext,
    TopologyAdaptiveHazardProjector as BaseTopologyAdaptiveHazardProjector,
    _lateral_half_extent,
    _point_line_distance,
    _rough_overlap,
)


MIN_EGO_LANE_CLEARANCE_M = 0.05
ROADSIDE_TARGET_CLEARANCE_M = 0.15
MIN_ACTOR_FAR_SIDE_MARGIN_M = 0.10


class RobustTopologyAdaptiveHazardProjector(
    BaseTopologyAdaptiveHazardProjector
):
    """Active topology-adaptive projector with road and hazard hard gates."""

    def project(
        self,
        vector: Any,
        spec: Any,
        *,
        attempt_seed: int = 0,
    ):
        """Reject fragmented generated roads before hazard re-projection.

        ``vector`` is the raw diffusion candidate.  Global road validity is
        evaluated before any pedestrian/occluder geometry is changed, so this
        gate measures the diffusion-generated road itself rather than a
        post-processed version.
        """

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
                f"components={road_graph.get('num_weak_components')}; "
                f"ego_route_length_m="
                f"{road_graph.get('ego_route_length_m')}; "
                f"largest_component_length_ratio="
                f"{road_graph.get('largest_component_length_ratio')}; "
                f"orphan_lane_length_ratio="
                f"{road_graph.get('orphan_lane_length_ratio')}"
            )

        projected, report = super().project(
            vector,
            spec,
            attempt_seed=attempt_seed,
        )
        report = dict(report)
        report["road_graph_validity"] = road_graph
        report["road_graph_policy"] = (
            "reject_fragmented_diffusion_road_no_geometry_repair"
        )
        return projected, report

    def _solve_occluder(
        self,
        *,
        actor_display: np.ndarray,
        road: LocalRoadContext,
        side_sign: float,
        variant: str,
        occluder_width: float,
        occluder_length: float,
        ego_speed: float,
    ) -> Optional[Dict[str, float]]:
        # Preserve the already-working adjacent-lane dynamic behavior exactly.
        if variant == "adjacent_lane_dynamic":
            return super()._solve_occluder(
                actor_display=actor_display,
                road=road,
                side_sign=side_sign,
                variant=variant,
                occluder_width=occluder_width,
                occluder_length=occluder_length,
                ego_speed=ego_speed,
            )

        ax = float(actor_display[AgentIndex.X])
        ay = float(actor_display[AgentIndex.Y])
        heading = float(road.local_tangent_heading)
        boundary = (
            float(road.upper_boundary_y)
            if side_sign > 0.0
            else float(road.lower_boundary_y)
        )

        # A parked/static occluder is aligned to the generated road.  Its
        # rotated length contributes to lateral road occupancy, so roadside
        # placement must use the full lateral half extent rather than width/2.
        lateral_half = _lateral_half_extent(
            heading,
            float(occluder_width),
            float(occluder_length),
        )
        desired_y = float(
            boundary
            + side_sign
            * (lateral_half + ROADSIDE_TARGET_CLEARANCE_M)
        )
        speed = 0.0

        candidates: List[Tuple[float, Dict[str, float]]] = []
        ratios = (0.42, 0.48, 0.55, 0.62, 0.70, 0.78, 0.84, 0.90)
        mixes = (0.0, 0.20, 0.40, 0.60, 0.80, 1.0)

        for ratio in ratios:
            x = float(ratio * ax)
            if not 3.0 <= x <= ax - 0.8:
                continue

            ray_y = float(
                road.lane_center_y
                + ratio * (ay - road.lane_center_y)
            )

            for mix in mixes:
                y = float(
                    (1.0 - mix) * ray_y
                    + mix * desired_y
                )

                # The actor must remain farther from the ego lane than the
                # occluder center; otherwise it is not an emergence-from-behind
                # configuration even if the LOS happens to intersect the box.
                far_side_margin = float(side_sign * (ay - y))
                if far_side_margin < MIN_ACTOR_FAR_SIDE_MARGIN_M:
                    continue

                occ = np.asarray(
                    [
                        x,
                        y,
                        heading,
                        occluder_width,
                        occluder_length,
                        speed,
                    ],
                    dtype=np.float32,
                )

                # Hard non-overlap before ranking.  _rough_overlap is slightly
                # more conservative than the final semantic evaluator, so a
                # candidate accepted here cannot later fail the canonical
                # no_actor_occluder_initial_overlap check merely because of the
                # projector's own placement.
                if _rough_overlap(
                    actor_display,
                    occ,
                    "pedestrians",
                    "vehicles",
                ):
                    continue

                if not line_of_sight_intersects_box(
                    (0.0, float(road.lane_center_y)),
                    (ax, ay),
                    occ,
                    margin=0.18,
                ):
                    continue

                inner_gap = float(
                    abs(y - road.lane_center_y)
                    - lateral_half
                    - road.lane_half_width_m
                )
                if inner_gap < MIN_EGO_LANE_CLEARANCE_M:
                    continue

                if abs(y - road.lane_center_y) > max(
                    9.0,
                    2.8 * road.lane_width_m,
                ):
                    continue

                perpendicular = _point_line_distance(
                    (x, y),
                    (0.0, float(road.lane_center_y)),
                    (ax, ay),
                )

                # Prefer a natural roadside placement close to the rotated
                # footprint target and LOS ray, while maintaining useful
                # longitudinal separation from the pedestrian.
                score = float(
                    abs(y - desired_y)
                    + 0.20 * perpendicular
                    + 0.025 * abs(x - 0.68 * ax)
                    + 0.10 * max(
                        0.0,
                        0.40 - far_side_margin,
                    )
                )

                candidates.append(
                    (
                        score,
                        {
                            "x": float(x),
                            "y": float(y),
                            "heading": float(heading),
                            "speed_mps": float(speed),
                            "lane_boundary_gap_m": float(inner_gap),
                            "line_perpendicular_error_m": float(
                                perpendicular
                            ),
                            "occluder_lateral_half_extent_m": float(
                                lateral_half
                            ),
                            "roadside_target_y_m": float(desired_y),
                            "actor_far_side_margin_m": float(
                                far_side_margin
                            ),
                            "actor_occluder_overlap": False,
                            "placement_solver": (
                                "robust_rotated_footprint_far_side"
                            ),
                        },
                    )
                )

        if not candidates:
            return None

        candidates.sort(key=lambda row: row[0])
        return candidates[0][1]


__all__ = [
    "RobustTopologyAdaptiveHazardProjector",
]
