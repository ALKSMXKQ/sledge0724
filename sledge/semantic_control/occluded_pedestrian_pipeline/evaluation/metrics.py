"""Stage-independent metrics for occluded-pedestrian scenes.

The evaluator exposes two independent gates:

Level 1 - danger semantic validity
    The requested occluded-emergence relation, direction, speed and timing.

Level 2 - traffic realism
    Whether the hazard and diffusion-generated background remain plausible in
    the generated local road frame.

A scene passes only when both levels pass. The semantic satisfaction rate is
kept separate from the traffic-realism rate so experiments can diagnose the
trade-off instead of collapsing both failure modes into one number.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, Optional

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import AgentIndex
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.geometry_metrics import (
    estimate_ego_speed,
    line_of_sight_intersects_box,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_spec import (
    HazardSemanticSpec,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.primitive_ops import (
    OCCLUDER_SPECS,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.topology_adaptive_projection import (
    heading_alignment_error,
    infer_local_road_context,
)


LABEL_THRESHOLD = 0.3
DEFAULT_LANE_HALF_WIDTH_M = 1.75
OCCLUDER_CORRIDOR_CLEARANCE_M = 0.05
TIMING_RANGE_TOLERANCE_S = 0.60
ARRIVAL_TIME_TOLERANCE_S = 1.50
CROSSING_HORIZON_S = 6.0
OCCLUDER_STATIONARY_SPEED_MPS = 0.10
VEHICLE_HEADING_TOLERANCE_RAD = math.radians(40.0)
PEDESTRIAN_NORMAL_TOLERANCE_RAD = math.radians(55.0)


def evaluate_occluded_pedestrian_scene(
    scene: Any,
    spec: HazardSemanticSpec,
    *,
    preferred_pedestrian_index: Optional[int] = None,
    preferred_occluder_index: Optional[int] = None,
    preferred_occluder_elem_name: str = "vehicles",
    projection_time_s: float = 0.0,
    lane_center_y: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate danger semantics and traffic realism for one scene."""

    provisional_lane_center = float(
        0.0 if lane_center_y is None else lane_center_y
    )
    road_context = infer_local_road_context(
        scene,
        target_x=12.0,
        fallback_lane_width_m=float(
            getattr(
                spec.road_layer,
                "lane_width_m",
                2.0 * DEFAULT_LANE_HALF_WIDTH_M,
            )
        ),
    )
    if lane_center_y is None:
        lane_center_y = float(road_context.lane_center_y)
    else:
        lane_center_y = provisional_lane_center
    lane_half_width = 0.5 * float(
        road_context.lane_width_m
        if road_context.source.startswith("generated")
        else getattr(
            spec.road_layer,
            "lane_width_m",
            2.0 * DEFAULT_LANE_HALF_WIDTH_M,
        )
    )

    pedestrians = [
        (index, _project_agent_state(state, projection_time_s))
        for index, state in _valid_rows(scene.pedestrians)
    ]
    ped_choice = _select_pedestrian(
        pedestrians,
        spec,
        preferred_index=preferred_pedestrian_index,
        lane_center_y=lane_center_y,
        lane_half_width=lane_half_width,
    )
    if ped_choice is None:
        return _empty_result(spec, "no valid pedestrian candidate")
    ped_index, ped = ped_choice

    occluders = []
    for elem_name in ("vehicles", "static_objects"):
        elem = getattr(scene, elem_name)
        for index, state in _valid_rows(elem):
            occluders.append(
                (
                    elem_name,
                    index,
                    _project_agent_state(state, projection_time_s),
                )
            )
    occ_choice = _select_occluder(
        occluders,
        ped,
        spec,
        preferred_index=preferred_occluder_index,
        preferred_elem_name=preferred_occluder_elem_name,
    )

    speed = float(max(ped[AgentIndex.VELOCITY], 0.0))
    ped_heading = float(ped[AgentIndex.HEADING])
    ped_vy = speed * math.sin(ped_heading)
    direction = _direction_from_motion(
        float(ped[AgentIndex.Y]),
        ped_vy,
        lane_center_y=lane_center_y,
    )
    direction_ok = direction == spec.interaction_layer.conflict_direction
    speed_error = abs(
        speed - float(spec.risk_layer.target_actor_speed_mps)
    )
    speed_ok = speed_error <= 0.35

    t_ped = _time_to_lane_entry(
        ped,
        lane_center_y=lane_center_y,
        lane_half_width=lane_half_width,
    )
    ego_speed = float(max(estimate_ego_speed(scene), 1e-3))
    conflict_x = float(ped[AgentIndex.X])
    t_ego = conflict_x / ego_speed if conflict_x > 0.0 else float("inf")
    arrival_error = (
        abs(t_ped - t_ego)
        if math.isfinite(t_ped) and math.isfinite(t_ego)
        else float("inf")
    )
    crossing_ok = (
        math.isfinite(t_ped)
        and 0.0 <= t_ped <= CROSSING_HORIZON_S
    )
    interaction_ttc = (
        max(t_ped, t_ego)
        if math.isfinite(t_ped) and math.isfinite(t_ego)
        else float("inf")
    )
    ttc_low, ttc_high = _normalized_range(
        spec.risk_layer.ttc_range_s,
        (2.0, 3.0),
    )
    timing_ok = (
        math.isfinite(interaction_ttc)
        and float(ttc_low) - TIMING_RANGE_TOLERANCE_S
        <= interaction_ttc
        <= float(ttc_high) + TIMING_RANGE_TOLERANCE_S
        and arrival_error <= ARRIVAL_TIME_TOLERANCE_S
    )

    occ_exists = occ_choice is not None
    los_blocked = False
    between_ok = False
    corridor_clear = False
    occluder_motion_semantics = False
    occ_payload: Dict[str, Any] = {
        "element": None,
        "index": -1,
        "xy": None,
        "size": None,
        "speed_mps": None,
    }

    occ_state = None
    if occ_choice is not None:
        elem_name, occ_index, occ = occ_choice
        occ_state = occ
        ax = float(ped[AgentIndex.X])
        ay = float(ped[AgentIndex.Y])
        ox = float(occ[AgentIndex.X])
        oy = float(occ[AgentIndex.Y])

        actor_dist2 = ax * ax + ay * ay
        ratio = (
            (ox * ax + oy * ay) / actor_dist2
            if actor_dist2 > 1e-6
            else -1.0
        )
        perpendicular = (
            abs(ox * ay - oy * ax) / math.sqrt(actor_dist2)
            if actor_dist2 > 1e-6
            else float("inf")
        )
        between_ok = 0.1 <= ratio <= 0.95 and perpendicular <= 3.0
        los_blocked = bool(
            line_of_sight_intersects_box(
                (0.0, lane_center_y),
                (ax, ay),
                occ,
                margin=0.25,
            )
        )

        occ_heading = float(occ[AgentIndex.HEADING])
        occ_width = float(max(occ[AgentIndex.WIDTH], 0.5))
        occ_length = float(max(occ[AgentIndex.LENGTH], 0.5))
        lateral_half_extent = (
            abs(math.sin(occ_heading)) * occ_length / 2.0
            + abs(math.cos(occ_heading)) * occ_width / 2.0
        )
        lane_boundary_gap = (
            abs(oy - lane_center_y)
            - lateral_half_extent
            - lane_half_width
        )
        corridor_clear = (
            lane_boundary_gap >= OCCLUDER_CORRIDOR_CLEARANCE_M
        )

        occ_speed = _state_speed_if_available(occ)
        occluder_motion_semantics = _occluder_motion_matches_prompt(
            elem_name,
            occ_speed,
            str(getattr(spec, "raw_prompt", "") or ""),
        )
        occ_payload = {
            "element": elem_name,
            "index": int(occ_index),
            "xy": [ox, oy],
            "size": [occ_width, occ_length],
            "heading": occ_heading,
            "projection_ratio": float(ratio),
            "perpendicular_distance_m": float(perpendicular),
            "lane_boundary_gap_m": float(lane_boundary_gap),
            "speed_mps": float(occ_speed),
        }

    no_initial_actor_occluder_overlap = True
    if occ_choice is not None:
        no_initial_actor_occluder_overlap = not _axis_aligned_overlap(
            ped,
            occ_choice[2],
            scale=0.35,
        )

    semantic_checks = {
        "pedestrian_exists": True,
        "occluder_exists": occ_exists,
        "occluder_between_ego_and_actor": between_ok,
        "line_of_sight_occlusion": los_blocked,
        "occluder_clear_of_ego_corridor": corridor_clear,
        "direction_match": direction_ok,
        "speed_match": speed_ok,
        "crossing_reaches_ego_lane": crossing_ok,
        "interaction_timing_match": timing_ok,
        "no_actor_occluder_initial_overlap": (
            no_initial_actor_occluder_overlap
        ),
        "occluder_motion_semantics": occluder_motion_semantics,
    }
    semantic_pass = all(bool(value) for value in semantic_checks.values())

    traffic_checks, traffic_payload = _traffic_realism_checks(
        scene=scene,
        spec=spec,
        ped_index=ped_index,
        ped=ped,
        occ_choice=occ_choice,
        occ_state=occ_state,
        lane_center_y=lane_center_y,
        lane_half_width=lane_half_width,
        road_context=road_context,
        t_ped=t_ped,
        ttc_high=ttc_high,
        projection_time_s=projection_time_s,
    )
    traffic_realism_pass = all(
        bool(value) for value in traffic_checks.values()
    )
    overall_pass = bool(semantic_pass and traffic_realism_pass)

    return {
        "schema_version": (
            "occluded_pedestrian_metrics_v3_semantic_plus_traffic"
        ),
        "overall_pass": overall_pass,
        "semantic_pass": bool(semantic_pass),
        "traffic_realism_pass": bool(traffic_realism_pass),
        "semantic_satisfaction_rate": float(
            sum(bool(v) for v in semantic_checks.values())
            / len(semantic_checks)
        ),
        "traffic_realism_rate": float(
            sum(bool(v) for v in traffic_checks.values())
            / len(traffic_checks)
        ),
        "checks": semantic_checks,
        "semantic_checks": semantic_checks,
        "traffic_checks": traffic_checks,
        "traffic_realism": traffic_payload,
        "road_context": road_context.to_dict(),
        "pedestrian": {
            "index": int(ped_index),
            "xy": [
                float(ped[AgentIndex.X]),
                float(ped[AgentIndex.Y]),
            ],
            "heading": float(ped_heading),
            "speed_mps": speed,
            "expected_speed_mps": float(
                spec.risk_layer.target_actor_speed_mps
            ),
            "speed_error_mps": float(speed_error),
            "inferred_direction": direction,
            "expected_direction": (
                spec.interaction_layer.conflict_direction
            ),
        },
        "occluder": occ_payload,
        "interaction": {
            "ego_speed_mps": ego_speed,
            "projection_time_s": float(projection_time_s),
            "lane_center_y": float(lane_center_y),
            "lane_half_width_m": float(lane_half_width),
            "conflict_x_m": float(conflict_x),
            "pedestrian_lane_entry_time_s": _finite_or_none(t_ped),
            "ego_arrival_time_s": _finite_or_none(t_ego),
            "arrival_time_error_s": _finite_or_none(arrival_error),
            "interaction_ttc_s": _finite_or_none(interaction_ttc),
            "target_ttc_range_s": [float(ttc_low), float(ttc_high)],
        },
    }


def aggregate_stage_metrics(
    rows: Iterable[Dict[str, Any]],
    stage: str,
) -> Dict[str, Any]:
    items = list(rows)
    evaluated = [row for row in items if not row.get("error")]
    checks = sorted(
        {key for row in items for key in row.get("checks", {})}
    )
    traffic_checks = sorted(
        {
            key
            for row in items
            for key in row.get("traffic_checks", {})
        }
    )
    denominator = len(items)
    return {
        "schema_version": (
            "occluded_pedestrian_stage_summary_v2_semantic_plus_traffic"
        ),
        "stage": stage,
        "num_rows": len(items),
        "num_evaluated": len(evaluated),
        "num_errors": len(items) - len(evaluated),
        "overall_pass_count": sum(
            bool(row.get("overall_pass")) for row in items
        ),
        "overall_pass_rate": (
            sum(bool(row.get("overall_pass")) for row in items)
            / denominator
            if denominator
            else 0.0
        ),
        "semantic_pass_rate": (
            sum(
                bool(
                    row.get(
                        "semantic_pass",
                        row.get("overall_pass"),
                    )
                )
                for row in items
            )
            / denominator
            if denominator
            else 0.0
        ),
        "traffic_realism_pass_rate": (
            sum(
                bool(row.get("traffic_realism_pass", False))
                for row in items
            )
            / denominator
            if denominator
            else 0.0
        ),
        "mean_semantic_satisfaction_rate": (
            float(
                np.mean(
                    [
                        row.get("semantic_satisfaction_rate", 0.0)
                        for row in items
                    ]
                )
            )
            if items
            else 0.0
        ),
        "mean_traffic_realism_rate": (
            float(
                np.mean(
                    [
                        row.get("traffic_realism_rate", 0.0)
                        for row in items
                    ]
                )
            )
            if items
            else 0.0
        ),
        "check_pass_rates": (
            {
                key: sum(
                    bool(row.get("checks", {}).get(key, False))
                    for row in items
                )
                / denominator
                for key in checks
            }
            if items
            else {}
        ),
        "traffic_check_pass_rates": (
            {
                key: sum(
                    bool(
                        row.get("traffic_checks", {}).get(key, False)
                    )
                    for row in items
                )
                / denominator
                for key in traffic_checks
            }
            if items
            else {}
        ),
    }


def _traffic_realism_checks(
    *,
    scene: Any,
    spec: HazardSemanticSpec,
    ped_index: int,
    ped: np.ndarray,
    occ_choice: Any,
    occ_state: Optional[np.ndarray],
    lane_center_y: float,
    lane_half_width: float,
    road_context: Any,
    t_ped: float,
    ttc_high: float,
    projection_time_s: float,
):
    road_heading = float(road_context.local_tangent_heading)
    road_relaxed = str(spec.road_layer.road_topology).lower() in {
        "intersection",
        "merge",
        "roundabout",
    }

    occ_lane_relation = False
    occ_heading_alignment = False
    occ_motion_plausibility = False
    ped_emergence = False
    occ_side = 0.0
    if occ_choice is not None and occ_state is not None:
        elem_name, _, occ = occ_choice
        oy = float(occ[AgentIndex.Y])
        occ_side = (
            math.copysign(1.0, oy - lane_center_y)
            if abs(oy - lane_center_y) > 1e-4
            else 0.0
        )
        occ_width = float(max(occ[AgentIndex.WIDTH], 0.5))
        occ_length = float(max(occ[AgentIndex.LENGTH], 0.5))
        occ_heading = float(occ[AgentIndex.HEADING])
        half = (
            abs(math.sin(occ_heading)) * occ_length / 2.0
            + abs(math.cos(occ_heading)) * occ_width / 2.0
        )
        gap = (
            abs(oy - lane_center_y)
            - half
            - lane_half_width
        )
        occ_lane_relation = (
            gap >= -0.05
            and abs(oy - lane_center_y)
            <= max(10.0, 3.0 * road_context.lane_width_m)
        )
        if elem_name == "vehicles":
            occ_heading_alignment = (
                road_relaxed
                or heading_alignment_error(
                    occ_heading,
                    road_heading,
                )
                <= VEHICLE_HEADING_TOLERANCE_RAD
            )
            occ_speed = _state_speed_if_available(occ)
            parked_prompt = _prompt_requires_stationary_occluder(
                str(getattr(spec, "raw_prompt", "") or "")
            )
            if parked_prompt:
                occ_motion_plausibility = occ_speed <= 0.35
            else:
                occ_motion_plausibility = (
                    occ_speed <= 0.35
                    or (
                        0.35 < occ_speed <= 15.0
                        and occ_heading_alignment
                    )
                )
        else:
            occ_heading_alignment = True
            occ_motion_plausibility = (
                _state_speed_if_available(occ) <= 0.10
            )

        ped_side = (
            math.copysign(
                1.0,
                float(ped[AgentIndex.Y]) - lane_center_y,
            )
            if abs(float(ped[AgentIndex.Y]) - lane_center_y) > 1e-4
            else 0.0
        )
        target_normal = (
            road_heading - ped_side * math.pi / 2.0
            if ped_side
            else road_heading
        )
        heading_ok = (
            _directed_angle_error(
                float(ped[AgentIndex.HEADING]),
                target_normal,
            )
            <= PEDESTRIAN_NORMAL_TOLERANCE_RAD
        )
        farther_out = (
            ped_side != 0.0
            and ped_side == occ_side
            and ped_side * (float(ped[AgentIndex.Y]) - oy) >= -0.35
        )
        ped_emergence = bool(heading_ok and farther_out)

    reveal_time_plausible = bool(
        math.isfinite(t_ped)
        and 0.15
        <= t_ped
        <= min(CROSSING_HORIZON_S, float(ttc_high) + 1.5)
    )

    (
        background_pedestrian_density,
        pedestrian_payload,
    ) = _background_pedestrian_realism(
        scene,
        target_index=ped_index,
        target_state=ped,
        projection_time_s=projection_time_s,
    )
    (
        background_vehicle_motion,
        no_cross_lane_vehicle_pose,
        vehicle_payload,
    ) = _background_vehicle_realism(
        scene,
        road_heading=road_heading,
        road_relaxed=road_relaxed,
        projection_time_s=projection_time_s,
        protected_occ_choice=occ_choice,
    )

    checks = {
        "occluder_lane_relation": bool(occ_lane_relation),
        "occluder_heading_alignment": bool(occ_heading_alignment),
        "occluder_motion_plausibility": bool(
            occ_motion_plausibility
        ),
        "pedestrian_emergence_geometry": bool(ped_emergence),
        "reveal_time_plausible": bool(reveal_time_plausible),
        "background_pedestrian_density": bool(
            background_pedestrian_density
        ),
        "background_vehicle_motion": bool(background_vehicle_motion),
        "no_cross_lane_vehicle_pose": bool(
            no_cross_lane_vehicle_pose
        ),
    }
    payload = {
        "road_heading_rad": road_heading,
        "road_heading_relaxed_for_topology": bool(road_relaxed),
        "pedestrian_background": pedestrian_payload,
        "vehicle_background": vehicle_payload,
    }
    return checks, payload


def _background_pedestrian_realism(
    scene: Any,
    *,
    target_index: int,
    target_state: np.ndarray,
    projection_time_s: float,
):
    nearby = []
    overlaps = 0
    tx = float(target_state[AgentIndex.X])
    ty = float(target_state[AgentIndex.Y])
    for idx, state in _valid_rows(scene.pedestrians):
        if idx == target_index:
            continue
        projected = _project_agent_state(
            state,
            projection_time_s,
        )
        dist = math.hypot(
            float(projected[AgentIndex.X]) - tx,
            float(projected[AgentIndex.Y]) - ty,
        )
        if dist <= 10.0:
            nearby.append(idx)
        if dist <= 0.85:
            overlaps += 1
    passed = len(nearby) <= 6 and overlaps == 0
    return passed, {
        "nearby_non_target_pedestrians_10m": len(nearby),
        "near_duplicate_pedestrians": overlaps,
    }


def _background_vehicle_realism(
    scene: Any,
    *,
    road_heading: float,
    road_relaxed: bool,
    projection_time_s: float,
    protected_occ_choice: Any,
):
    protected = None
    if (
        protected_occ_choice is not None
        and protected_occ_choice[0] == "vehicles"
    ):
        protected = int(protected_occ_choice[1])

    moving_count = 0
    moving_misaligned = 0
    severe_cross_lane = 0
    for idx, state in _valid_rows(scene.vehicles):
        if protected is not None and idx == protected:
            continue
        projected = _project_agent_state(
            state,
            projection_time_s,
        )
        x = float(projected[AgentIndex.X])
        y = float(projected[AgentIndex.Y])
        if not (-8.0 <= x <= 35.0 and abs(y) <= 12.0):
            continue
        speed = _state_speed_if_available(projected)
        if speed <= 0.5:
            continue
        moving_count += 1
        error = heading_alignment_error(
            float(projected[AgentIndex.HEADING]),
            road_heading,
        )
        if error > math.radians(55.0):
            moving_misaligned += 1
        if error > math.radians(75.0):
            severe_cross_lane += 1

    if road_relaxed:
        motion_pass = True
        pose_pass = True
    elif moving_count == 0:
        motion_pass = True
        pose_pass = True
    else:
        motion_pass = moving_misaligned <= max(
            1,
            int(math.ceil(0.25 * moving_count)),
        )
        pose_pass = severe_cross_lane == 0
    return motion_pass, pose_pass, {
        "moving_background_vehicle_count": moving_count,
        "moving_heading_outlier_count": moving_misaligned,
        "severe_cross_lane_pose_count": severe_cross_lane,
    }


def _occluder_motion_matches_prompt(
    elem_name: str,
    speed: float,
    prompt: str,
) -> bool:
    if elem_name != "vehicles":
        return speed <= OCCLUDER_STATIONARY_SPEED_MPS
    if _prompt_requires_stationary_occluder(prompt):
        return speed <= 0.35
    return 0.0 <= speed <= 15.0


def _prompt_requires_stationary_occluder(prompt: str) -> bool:
    return bool(
        re.search(
            r"\b(?:parked|parking|stopped|stationary)\b|"
            r"停放|停车|停着|静止",
            str(prompt or ""),
            flags=re.IGNORECASE,
        )
    )


def _directed_angle_error(
    heading: float,
    reference: float,
) -> float:
    return abs(
        math.atan2(
            math.sin(heading - reference),
            math.cos(heading - reference),
        )
    )


def _valid_rows(elem: Any):
    states = np.asarray(elem.states)
    masks = np.asarray(elem.mask)
    if states.ndim == 1:
        states = states[None, :]
    if masks.ndim == 0:
        masks = masks[None]
    masks = masks.reshape(-1)
    for index, (state, mask) in enumerate(zip(states, masks)):
        valid = (
            bool(mask)
            if isinstance(mask, (bool, np.bool_))
            else float(mask) >= LABEL_THRESHOLD
        )
        if valid:
            yield index, np.asarray(state, dtype=np.float32)


def _select_pedestrian(
    candidates,
    spec,
    *,
    preferred_index=None,
    lane_center_y: float,
    lane_half_width: float,
):
    for index, state in candidates:
        if preferred_index is not None and index == preferred_index:
            return index, state

    best = None
    for index, state in candidates:
        speed = _state_speed_if_available(state)
        heading = float(state[AgentIndex.HEADING])
        vy = speed * math.sin(heading)
        inferred = _direction_from_motion(
            float(state[AgentIndex.Y]),
            vy,
            lane_center_y=lane_center_y,
        )
        score = 0.0
        score += (
            4.0
            if inferred == spec.interaction_layer.conflict_direction
            else 0.0
        )
        score += max(
            0.0,
            2.0
            - abs(speed - spec.risk_layer.target_actor_speed_mps),
        )
        score += (
            1.0
            if 3.0 <= float(state[AgentIndex.X]) <= 35.0
            else 0.0
        )
        score += (
            1.0
            if abs(float(state[AgentIndex.Y]) - lane_center_y)
            >= lane_half_width
            else 0.0
        )
        t_entry = _time_to_lane_entry(
            state,
            lane_center_y=lane_center_y,
            lane_half_width=lane_half_width,
        )
        score += (
            1.0
            if math.isfinite(t_entry)
            and t_entry <= CROSSING_HORIZON_S
            else 0.0
        )
        if best is None or score > best[0]:
            best = (score, index, state)
    return (best[1], best[2]) if best is not None else None


def _select_occluder(
    candidates,
    ped,
    spec,
    *,
    preferred_index=None,
    preferred_elem_name="vehicles",
):
    for elem_name, index, state in candidates:
        if (
            preferred_index is not None
            and index == preferred_index
            and elem_name == preferred_elem_name
        ):
            return elem_name, index, state

    target = OCCLUDER_SPECS.get(
        spec.object_layer.occlusion.occluder_type,
        OCCLUDER_SPECS["vehicle"],
    )
    best = None
    actor_xy = (
        float(ped[AgentIndex.X]),
        float(ped[AgentIndex.Y]),
    )
    actor_dist2 = actor_xy[0] ** 2 + actor_xy[1] ** 2

    for elem_name, index, state in candidates:
        if elem_name != target.elem_name:
            continue
        los = line_of_sight_intersects_box(
            (0.0, 0.0),
            actor_xy,
            state,
            margin=0.25,
        )
        ratio = (
            (
                float(state[AgentIndex.X]) * actor_xy[0]
                + float(state[AgentIndex.Y]) * actor_xy[1]
            )
            / actor_dist2
            if actor_dist2 > 1e-6
            else -1.0
        )
        perpendicular = (
            abs(
                float(state[AgentIndex.X]) * actor_xy[1]
                - float(state[AgentIndex.Y]) * actor_xy[0]
            )
            / math.sqrt(actor_dist2)
            if actor_dist2 > 1e-6
            else float("inf")
        )
        between = (
            0.1 <= ratio <= 0.95
            and perpendicular <= 3.0
        )
        size_error = (
            abs(float(state[AgentIndex.WIDTH]) - target.width)
            + abs(float(state[AgentIndex.LENGTH]) - target.length)
        )
        score = (
            8.0 * float(los)
            + 4.0 * float(between)
            - 0.10 * size_error
            - 0.05 * min(perpendicular, 20.0)
        )
        if best is None or score > best[0]:
            best = (score, elem_name, index, state)

    return (
        (best[1], best[2], best[3])
        if best is not None
        else None
    )


def _direction_from_motion(
    y: float,
    vy: float,
    *,
    lane_center_y: float,
) -> str:
    relative_y = y - lane_center_y
    if relative_y < 0.0 and vy > 0.0:
        return "right_to_left"
    if relative_y > 0.0 and vy < 0.0:
        return "left_to_right"
    return "unknown"


def _time_to_lane_entry(
    state: np.ndarray,
    *,
    lane_center_y: float,
    lane_half_width: float,
) -> float:
    y = float(state[AgentIndex.Y])
    speed = _state_speed_if_available(state)
    heading = float(state[AgentIndex.HEADING])
    vy = speed * math.sin(heading)
    upper = lane_center_y + lane_half_width
    lower = lane_center_y - lane_half_width

    if y > upper and vy < -1e-5:
        return (y - upper) / (-vy)
    if y < lower and vy > 1e-5:
        return (lower - y) / vy
    if lower <= y <= upper:
        return 0.0
    return float("inf")


def _axis_aligned_overlap(
    a: np.ndarray,
    b: np.ndarray,
    scale: float,
) -> bool:
    return (
        abs(
            float(a[AgentIndex.X])
            - float(b[AgentIndex.X])
        )
        < scale
        * (
            float(max(a[AgentIndex.LENGTH], 0.5))
            + float(max(b[AgentIndex.LENGTH], 0.5))
        )
        and abs(
            float(a[AgentIndex.Y])
            - float(b[AgentIndex.Y])
        )
        < scale
        * (
            float(max(a[AgentIndex.WIDTH], 0.5))
            + float(max(b[AgentIndex.WIDTH], 0.5))
        )
    )


def _finite_or_none(value: float):
    return float(value) if math.isfinite(value) else None


def _project_agent_state(
    state: np.ndarray,
    time_s: float,
) -> np.ndarray:
    projected = np.asarray(state, dtype=np.float32).copy()
    if (
        time_s <= 0.0
        or projected.size <= AgentIndex.VELOCITY
    ):
        return projected
    speed = float(max(projected[AgentIndex.VELOCITY], 0.0))
    heading = float(projected[AgentIndex.HEADING])
    projected[AgentIndex.X] += (
        speed * math.cos(heading) * time_s
    )
    projected[AgentIndex.Y] += (
        speed * math.sin(heading) * time_s
    )
    return projected


def _state_speed_if_available(state: np.ndarray) -> float:
    state = np.asarray(state)
    if state.size <= AgentIndex.VELOCITY:
        return 0.0
    return float(max(state[AgentIndex.VELOCITY], 0.0))


def _normalized_range(value: Any, default) -> tuple[float, float]:
    try:
        vals = list(value)
        if len(vals) < 2:
            raise ValueError
        low, high = sorted([float(vals[0]), float(vals[1])])
        return float(low), float(high)
    except Exception:
        return float(default[0]), float(default[1])


def _empty_result(
    spec: HazardSemanticSpec,
    reason: str,
) -> Dict[str, Any]:
    return {
        "schema_version": (
            "occluded_pedestrian_metrics_v3_semantic_plus_traffic"
        ),
        "overall_pass": False,
        "semantic_pass": False,
        "traffic_realism_pass": False,
        "semantic_satisfaction_rate": 0.0,
        "traffic_realism_rate": 0.0,
        "checks": {"pedestrian_exists": False},
        "semantic_checks": {"pedestrian_exists": False},
        "traffic_checks": {},
        "reason": reason,
        "expected": {
            "occluder_type": (
                spec.object_layer.occlusion.occluder_type
            ),
            "direction": (
                spec.interaction_layer.conflict_direction
            ),
            "pedestrian_speed_mps": (
                spec.risk_layer.target_actor_speed_mps
            ),
        },
    }


__all__ = [
    "aggregate_stage_metrics",
    "evaluate_occluded_pedestrian_scene",
]
