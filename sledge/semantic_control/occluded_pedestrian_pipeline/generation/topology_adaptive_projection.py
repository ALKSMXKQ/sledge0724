"""Topology-adaptive hazard projection for diffusion-generated SLEDGE scenes.

The diffusion model is allowed to generate the road, ego state and background
traffic. This module preserves only the requested hazard relations and
re-solves pedestrian/occluder geometry against the generated scene.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    AgentIndex,
    EgoIndex,
    SledgeVector,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.geometry_metrics import (
    line_of_sight_intersects_box,
    wrap_angle,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_spec import (
    HazardSemanticSpec,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.primitive_ops import (
    OCCLUDER_SPECS,
    SLEDGEBOARD_FRAME0_OFFSET_S,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.object_types import (
    element_name_for_occluder,
    normalize_occluder_type,
)


LABEL_THRESHOLD = 0.3
DEFAULT_LANE_WIDTH_M = 3.5
MIN_EGO_SPEED_MPS = 2.0
MAX_EGO_SPEED_MPS = 18.0
MIN_CONFLICT_X_M = 5.0
MAX_CONFLICT_X_M = 36.0


@dataclass(frozen=True)
class LocalRoadContext:
    """Generated local road frame used by semantic re-projection."""

    lane_center_y: float
    lane_width_m: float
    lane_half_width_m: float
    local_tangent_heading: float
    lower_boundary_y: float
    upper_boundary_y: float
    adjacent_lane_center_left_y: Optional[float]
    adjacent_lane_center_right_y: Optional[float]
    boundary_samples_y: Tuple[float, ...]
    source: str

    def adjacent_lane_center(self, side_sign: float) -> Optional[float]:
        return (
            self.adjacent_lane_center_left_y
            if side_sign > 0.0
            else self.adjacent_lane_center_right_y
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lane_center_y": float(self.lane_center_y),
            "lane_width_m": float(self.lane_width_m),
            "lane_half_width_m": float(self.lane_half_width_m),
            "local_tangent_heading": float(self.local_tangent_heading),
            "lower_boundary_y": float(self.lower_boundary_y),
            "upper_boundary_y": float(self.upper_boundary_y),
            "adjacent_lane_center_left_y": (
                None
                if self.adjacent_lane_center_left_y is None
                else float(self.adjacent_lane_center_left_y)
            ),
            "adjacent_lane_center_right_y": (
                None
                if self.adjacent_lane_center_right_y is None
                else float(self.adjacent_lane_center_right_y)
            ),
            "boundary_samples_y": [float(v) for v in self.boundary_samples_y],
            "source": self.source,
        }


def infer_local_road_context(
    scene: Any,
    *,
    target_x: float = 12.0,
    fallback_lane_width_m: float = DEFAULT_LANE_WIDTH_M,
) -> LocalRoadContext:
    """Infer an ego-lane cross section and tangent from generated polylines."""

    fallback_lane_width_m = float(np.clip(fallback_lane_width_m, 2.5, 5.0))
    samples: List[Tuple[float, float, float]] = []
    states = np.asarray(scene.lines.states, dtype=np.float32)
    masks = np.asarray(scene.lines.mask).reshape(-1)
    if states.ndim == 3:
        for line_index in range(min(len(states), len(masks))):
            mask = masks[line_index]
            valid = (
                bool(mask)
                if isinstance(mask, (bool, np.bool_))
                else float(mask) >= LABEL_THRESHOLD
            )
            if not valid:
                continue
            points = np.asarray(states[line_index, :, :2], dtype=np.float32)
            finite = np.isfinite(points).all(axis=1)
            points = points[finite]
            if len(points) < 2:
                continue
            idx = int(np.argmin(np.abs(points[:, 0] - float(target_x))))
            dx = abs(float(points[idx, 0]) - float(target_x))
            if dx > 10.0:
                continue
            heading = _polyline_heading(points, idx)
            samples.append((float(points[idx, 1]), float(heading), float(dx)))

    if not samples:
        half = 0.5 * fallback_lane_width_m
        return LocalRoadContext(
            lane_center_y=0.0,
            lane_width_m=fallback_lane_width_m,
            lane_half_width_m=half,
            local_tangent_heading=0.0,
            lower_boundary_y=-half,
            upper_boundary_y=half,
            adjacent_lane_center_left_y=None,
            adjacent_lane_center_right_y=None,
            boundary_samples_y=(),
            source="spec_fallback_no_generated_lines",
        )

    clustered_y = _cluster_values([row[0] for row in samples], tolerance=0.65)
    negatives = sorted([y for y in clustered_y if y < -0.20], reverse=True)
    positives = sorted([y for y in clustered_y if y > 0.20])

    source = "generated_lines"
    if negatives and positives:
        lower = float(negatives[0])
        upper = float(positives[0])
        width = upper - lower
        if not 2.4 <= width <= 5.5:
            source = "generated_tangent_spec_width"
            center = (
                0.5 * (upper + lower)
                if abs(0.5 * (upper + lower)) <= 2.0
                else 0.0
            )
            width = fallback_lane_width_m
            lower = center - 0.5 * width
            upper = center + 0.5 * width
        else:
            center = 0.5 * (upper + lower)
    else:
        source = "generated_tangent_spec_width"
        center = 0.0
        width = fallback_lane_width_m
        lower = center - 0.5 * width
        upper = center + 0.5 * width

    tangent_candidates = [
        (heading, 1.0 / (1.0 + dx))
        for y, heading, dx in samples
        if abs(y - center) <= max(6.0, 1.8 * width)
    ]
    tangent = _weighted_forward_heading(tangent_candidates)

    left_outer = [y for y in clustered_y if y > upper + 1.5]
    right_outer = [y for y in clustered_y if y < lower - 1.5]
    left_adjacent = None
    right_adjacent = None
    if left_outer:
        outer = min(left_outer)
        candidate_width = float(outer - upper)
        if 2.4 <= candidate_width <= 5.5:
            left_adjacent = 0.5 * (upper + outer)
    if right_outer:
        outer = max(right_outer)
        candidate_width = float(lower - outer)
        if 2.4 <= candidate_width <= 5.5:
            right_adjacent = 0.5 * (lower + outer)

    return LocalRoadContext(
        lane_center_y=float(center),
        lane_width_m=float(width),
        lane_half_width_m=0.5 * float(width),
        local_tangent_heading=float(tangent),
        lower_boundary_y=float(lower),
        upper_boundary_y=float(upper),
        adjacent_lane_center_left_y=(
            None if left_adjacent is None else float(left_adjacent)
        ),
        adjacent_lane_center_right_y=(
            None if right_adjacent is None else float(right_adjacent)
        ),
        boundary_samples_y=tuple(float(v) for v in clustered_y),
        source=source,
    )


def heading_alignment_error(heading: float, reference: float) -> float:
    """Undirected line/vehicle alignment error in radians."""

    error = abs(wrap_angle(float(heading) - float(reference)))
    return float(min(error, abs(math.pi - error)))


class TopologyAdaptiveHazardProjector:
    """Project an occluded-pedestrian relation onto a diffusion scene."""

    def __init__(
        self,
        *,
        projection_time_s: float = SLEDGEBOARD_FRAME0_OFFSET_S,
    ) -> None:
        self.projection_time_s = float(max(0.0, projection_time_s))

    def project(
        self,
        vector: SledgeVector,
        spec: HazardSemanticSpec,
        *,
        attempt_seed: int = 0,
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
            conflict_x = float(ego_speed * target_ttc)
            if not MIN_CONFLICT_X_M <= conflict_x <= MAX_CONFLICT_X_M:
                failures.append(
                    f"ttc={target_ttc:.2f}: conflict_x={conflict_x:.2f} "
                    "outside projection frame"
                )
                continue

            road = infer_local_road_context(
                scene,
                target_x=conflict_x,
                fallback_lane_width_m=float(spec.road_layer.lane_width_m),
            )
            if ego_repaired:
                self._write_repaired_ego_velocity(
                    scene,
                    ego_speed,
                    road.local_tangent_heading,
                )

            actor_heading = float(
                wrap_angle(
                    road.local_tangent_heading
                    - side_sign * math.pi / 2.0
                )
            )
            actor_display_y = float(
                road.lane_center_y
                + side_sign
                * (road.lane_half_width_m + actor_speed * target_ttc)
            )
            actor_display = np.asarray(
                [
                    conflict_x,
                    actor_display_y,
                    actor_heading,
                    0.75,
                    0.75,
                    actor_speed,
                ],
                dtype=np.float32,
            )

            variant = self._select_hazard_variant(
                canonical_occluder,
                prompt,
                road.adjacent_lane_center(side_sign),
            )
            placement = self._solve_occluder(
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
                    "occluder placement"
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

            report = {
                "schema_version": "topology_adaptive_hazard_projection_v1",
                "projection_policy": "copy_semantics_recompute_geometry",
                "semantic_projection_time_s": float(self.projection_time_s),
                "generated_road_preserved": True,
                "generated_ego_preserved": not ego_repaired,
                "ego_speed_mps": float(ego_speed),
                "ego_state_source": ego_source,
                "road_context": road.to_dict(),
                "semantic_direction": direction,
                "hazard_side": "left" if side_sign > 0 else "right",
                "hazard_variant": variant,
                "target_ttc_s": float(target_ttc),
                "target_ttc_range_s": [
                    float(spec.risk_layer.ttc_range_s[0]),
                    float(spec.risk_layer.ttc_range_s[1]),
                ],
                "conflict_x_m": float(conflict_x),
                "pedestrian": {
                    "index": int(pedestrian_index),
                    "replaced_generated_slot": bool(ped_replaced),
                    "display_state": actor_display.tolist(),
                    "raw_state": actor_raw.tolist(),
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
            "topology-adaptive hazard projection failed for all "
            "risk-conditioned TTC candidates: "
            + "; ".join(failures[-8:])
        )

    @staticmethod
    def _positive(
        value: Any,
        *,
        default: float,
        low: float,
        high: float,
    ) -> float:
        try:
            out = float(value)
        except Exception:
            out = float(default)
        if not math.isfinite(out):
            out = float(default)
        return float(np.clip(out, low, high))

    @staticmethod
    def _ttc_candidates(
        spec: HazardSemanticSpec,
        rng: np.random.Generator,
    ) -> List[float]:
        low, high = sorted(
            [
                float(spec.risk_layer.ttc_range_s[0]),
                float(spec.risk_layer.ttc_range_s[1]),
            ]
        )
        low = max(0.35, low)
        high = max(low, high)
        mid = 0.5 * (low + high)
        values = [
            mid,
            0.75 * mid + 0.25 * low,
            0.75 * mid + 0.25 * high,
            low,
            high,
        ]
        if high - low > 1e-4:
            values.insert(1, float(rng.uniform(low, high)))
        out: List[float] = []
        for value in values:
            value = float(value)
            if not any(abs(value - prev) < 1e-5 for prev in out):
                out.append(value)
        return out

    @staticmethod
    def _select_hazard_variant(
        canonical_occluder: str,
        prompt: str,
        adjacent_lane_center: Optional[float],
    ) -> str:
        if canonical_occluder not in {"vehicle", "bicycle"}:
            return "roadside_static"
        parked = bool(
            re.search(
                r"\b(?:parked|parking|stopped|stationary)\b|"
                r"停放|停车|停着|静止",
                prompt,
                flags=re.IGNORECASE,
            )
        )
        if adjacent_lane_center is not None and not parked:
            return "adjacent_lane_dynamic"
        return "roadside_parked"

    def _generated_ego_speed(
        self,
        scene: SledgeVector,
    ) -> Tuple[float, str, bool]:
        states = np.asarray(scene.ego.states, dtype=np.float32).reshape(-1)
        speed = float("nan")
        if states.size >= 2:
            speed = float(np.linalg.norm(states[:2]))
        elif states.size == 1:
            speed = float(abs(states[0]))
        if (
            math.isfinite(speed)
            and MIN_EGO_SPEED_MPS <= speed <= MAX_EGO_SPEED_MPS
        ):
            return speed, "diffusion_generated_ego", False

        candidates: List[float] = []
        vehicle_states = np.asarray(
            scene.vehicles.states,
            dtype=np.float32,
        )
        vehicle_masks = np.asarray(scene.vehicles.mask).reshape(-1)
        if vehicle_states.ndim == 2:
            for idx in range(
                min(len(vehicle_states), len(vehicle_masks))
            ):
                if float(vehicle_masks[idx]) < LABEL_THRESHOLD:
                    continue
                row = vehicle_states[idx]
                if row.size <= AgentIndex.VELOCITY:
                    continue
                v = float(row[AgentIndex.VELOCITY])
                if 1.0 <= v <= 15.0 and math.isfinite(v):
                    candidates.append(v)
        fallback = (
            float(np.median(candidates)) if candidates else 6.0
        )
        fallback = float(np.clip(fallback, 2.5, 15.0))
        return fallback, "generated_context_fallback", True

    @staticmethod
    def _write_repaired_ego_velocity(
        scene: SledgeVector,
        speed: float,
        heading: float,
    ) -> None:
        states = np.asarray(scene.ego.states)
        flat = states.reshape(-1)
        if flat.size >= 2:
            flat[EgoIndex.VELOCITY_X] = float(
                speed * math.cos(heading)
            )
            flat[EgoIndex.VELOCITY_Y] = float(
                speed * math.sin(heading)
            )
        if flat.size >= 4:
            flat[EgoIndex.ACCELERATION_X] = 0.0
            flat[EgoIndex.ACCELERATION_Y] = 0.0
        np.asarray(scene.ego.mask).reshape(-1)[:] = 1.0

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
        ax = float(actor_display[AgentIndex.X])
        ay = float(actor_display[AgentIndex.Y])
        boundary = (
            road.upper_boundary_y
            if side_sign > 0
            else road.lower_boundary_y
        )
        adjacent = road.adjacent_lane_center(side_sign)

        if variant == "adjacent_lane_dynamic" and adjacent is not None:
            desired_y = float(adjacent)
            speed = float(np.clip(0.75 * ego_speed, 2.0, 13.0))
        else:
            lateral_half = 0.5 * float(occluder_width)
            desired_y = float(
                boundary + side_sign * (lateral_half + 0.35)
            )
            speed = 0.0

        heading = float(road.local_tangent_heading)
        candidates: List[Tuple[float, Dict[str, float]]] = []
        for ratio in (0.48, 0.55, 0.62, 0.70, 0.78, 0.86, 0.92):
            x = ratio * ax
            ray_y = (
                road.lane_center_y
                + ratio * (ay - road.lane_center_y)
            )
            for mix in (0.0, 0.25, 0.50, 0.75, 1.0):
                y = float(
                    (1.0 - mix) * ray_y
                    + mix * desired_y
                )
                if not 3.0 <= x <= ax - 0.8:
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
                if not line_of_sight_intersects_box(
                    (0.0, road.lane_center_y),
                    (ax, ay),
                    occ,
                    margin=0.18,
                ):
                    continue
                lateral_half = _lateral_half_extent(
                    heading,
                    occluder_width,
                    occluder_length,
                )
                inner_gap = (
                    abs(y - road.lane_center_y)
                    - lateral_half
                    - road.lane_half_width_m
                )
                if inner_gap < 0.05:
                    continue
                if abs(y - road.lane_center_y) > max(
                    9.0,
                    2.8 * road.lane_width_m,
                ):
                    continue
                perpendicular = _point_line_distance(
                    (x, y),
                    (0.0, road.lane_center_y),
                    (ax, ay),
                )
                score = (
                    abs(y - desired_y)
                    + 0.20 * perpendicular
                    + 0.03 * abs(x - 0.70 * ax)
                )
                candidates.append(
                    (
                        float(score),
                        {
                            "x": float(x),
                            "y": float(y),
                            "heading": float(heading),
                            "speed_mps": float(speed),
                            "lane_boundary_gap_m": float(inner_gap),
                            "line_perpendicular_error_m": float(
                                perpendicular
                            ),
                        },
                    )
                )
        if not candidates:
            return None
        candidates.sort(key=lambda row: row[0])
        return candidates[0][1]

    def _display_to_raw_agent(
        self,
        display: np.ndarray,
    ) -> np.ndarray:
        raw = np.asarray(display, dtype=np.float32).copy()
        if self.projection_time_s <= 0.0:
            return raw
        speed = float(max(raw[AgentIndex.VELOCITY], 0.0))
        heading = float(raw[AgentIndex.HEADING])
        raw[AgentIndex.X] -= (
            speed * math.cos(heading) * self.projection_time_s
        )
        raw[AgentIndex.Y] -= (
            speed * math.sin(heading) * self.projection_time_s
        )
        return raw

    @staticmethod
    def _allocate_slot(elem: Any) -> Tuple[int, bool]:
        masks = np.asarray(elem.mask).reshape(-1)
        free = np.where(masks < LABEL_THRESHOLD)[0]
        if len(free):
            return int(free[0]), False
        states = np.asarray(elem.states)
        if len(states) == 0:
            raise RuntimeError(
                "no capacity available for topology-adaptive "
                "semantic projection"
            )
        if states.ndim == 1:
            return 0, True
        distance = np.linalg.norm(states[:, :2], axis=1)
        return int(np.argmax(distance)), True

    @staticmethod
    def _write_agent(
        elem: Any,
        index: int,
        state: np.ndarray,
    ) -> None:
        target = np.asarray(elem.states)
        width = min(target.shape[-1], len(state))
        target[index, :width] = np.asarray(
            state[:width],
            dtype=target.dtype,
        )
        np.asarray(elem.mask).reshape(-1)[index] = 1.0

    @staticmethod
    def _write_static(
        elem: Any,
        index: int,
        state5: np.ndarray,
    ) -> None:
        target = np.asarray(elem.states)
        width = min(target.shape[-1], len(state5))
        target[index, :width] = np.asarray(
            state5[:width],
            dtype=target.dtype,
        )
        np.asarray(elem.mask).reshape(-1)[index] = 1.0

    def _clear_local_overlaps(
        self,
        scene: SledgeVector,
        *,
        pedestrian_ref: Tuple[str, int],
        occluder_ref: Tuple[str, int],
    ) -> List[Dict[str, Any]]:
        protected = {pedestrian_ref, occluder_ref}
        ped_state = np.asarray(
            scene.pedestrians.states[pedestrian_ref[1]],
            dtype=np.float32,
        )
        occ_elem = getattr(scene, occluder_ref[0])
        occ_state = np.asarray(
            occ_elem.states[occluder_ref[1]],
            dtype=np.float32,
        )
        edits: List[Dict[str, Any]] = []

        for elem_name in (
            "vehicles",
            "pedestrians",
            "static_objects",
        ):
            elem = getattr(scene, elem_name)
            states = np.asarray(elem.states)
            masks = np.asarray(elem.mask).reshape(-1)
            if states.ndim != 2:
                continue
            for idx in range(min(len(states), len(masks))):
                if (
                    (elem_name, idx) in protected
                    or float(masks[idx]) < LABEL_THRESHOLD
                ):
                    continue
                state = np.asarray(states[idx], dtype=np.float32)
                if _rough_overlap(
                    state,
                    ped_state,
                    elem_name,
                    "pedestrians",
                ) or _rough_overlap(
                    state,
                    occ_state,
                    elem_name,
                    occluder_ref[0],
                ):
                    masks[idx] = 0.0
                    edits.append(
                        {
                            "element": elem_name,
                            "index": int(idx),
                            "operation": "delete",
                            "reason": (
                                "overlap_with_reprojected_hazard_only"
                            ),
                        }
                    )
        return edits


def _polyline_heading(points: np.ndarray, idx: int) -> float:
    if idx <= 0:
        a, b = points[0], points[1]
    elif idx >= len(points) - 1:
        a, b = points[-2], points[-1]
    else:
        a, b = points[idx - 1], points[idx + 1]
    heading = math.atan2(
        float(b[1] - a[1]),
        float(b[0] - a[0]),
    )
    if math.cos(heading) < 0.0:
        heading = wrap_angle(heading + math.pi)
    return float(heading)


def _cluster_values(
    values: Iterable[float],
    tolerance: float,
) -> List[float]:
    ordered = sorted(
        float(v)
        for v in values
        if math.isfinite(float(v))
    )
    if not ordered:
        return []
    groups: List[List[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if abs(value - float(np.mean(groups[-1]))) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [float(np.mean(group)) for group in groups]


def _weighted_forward_heading(
    values: Sequence[Tuple[float, float]],
) -> float:
    if not values:
        return 0.0
    sx = 0.0
    sy = 0.0
    for heading, weight in values:
        h = float(heading)
        if math.cos(h) < 0.0:
            h = wrap_angle(h + math.pi)
        sx += float(weight) * math.cos(h)
        sy += float(weight) * math.sin(h)
    if abs(sx) + abs(sy) < 1e-6:
        return 0.0
    heading = math.atan2(sy, sx)
    if math.cos(heading) < 0.0:
        heading = wrap_angle(heading + math.pi)
    return float(heading)


def _lateral_half_extent(
    heading: float,
    width: float,
    length: float,
) -> float:
    return float(
        abs(math.sin(heading)) * 0.5 * length
        + abs(math.cos(heading)) * 0.5 * width
    )


def _point_line_distance(
    point: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    px, py = point
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    denom = math.hypot(dx, dy)
    if denom <= 1e-6:
        return math.hypot(px - ax, py - ay)
    return abs(dx * (ay - py) - (ax - px) * dy) / denom


def _state_size_for_overlap(
    state: np.ndarray,
    elem_name: str,
) -> Tuple[float, float]:
    state = np.asarray(state)
    if state.size >= 5:
        width = float(max(state[3], 0.5))
        length = float(max(state[4], 0.5))
    else:
        width, length = 1.0, 1.0
    if elem_name == "pedestrians":
        width = max(width, 0.65)
        length = max(length, 0.65)
    return width, length


def _rough_overlap(
    a: np.ndarray,
    b: np.ndarray,
    a_name: str,
    b_name: str,
) -> bool:
    if a.size < 2 or b.size < 2:
        return False
    aw, al = _state_size_for_overlap(a, a_name)
    bw, bl = _state_size_for_overlap(b, b_name)
    return (
        abs(float(a[0]) - float(b[0]))
        < 0.45 * (al + bl)
        and abs(float(a[1]) - float(b[1]))
        < 0.45 * (aw + bw)
    )


__all__ = [
    "LocalRoadContext",
    "TopologyAdaptiveHazardProjector",
    "heading_alignment_error",
    "infer_local_road_context",
]
