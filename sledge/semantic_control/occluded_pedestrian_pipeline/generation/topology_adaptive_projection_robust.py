"""Robust roadside/static occluder solver for topology-adaptive projection.

This module keeps the existing topology-adaptive projector intact for the
historical ablation, while replacing only the roadside parked/static placement
logic used by the active generation package.  The override makes four
constraints hard before candidate ranking:

1. line-of-sight blocking,
2. ego-lane footprint clearance using the rotated occluder footprint,
3. pedestrian remains on the far side of the occluder,
4. pedestrian and occluder do not initially overlap.

The dynamic adjacent-lane solver is delegated unchanged to the base projector.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    AgentIndex,
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
    """Topology-adaptive projector with robust roadside hazard geometry."""

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
