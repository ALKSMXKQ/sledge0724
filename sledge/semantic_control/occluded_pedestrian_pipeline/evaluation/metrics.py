"""Stage-independent metrics for occluded-pedestrian scenes.

This module is the canonical B1/B2 semantic contract for the
occluded-pedestrian experiment.  Generation and evaluation use the same
lane-entry timing definition and the same ego-path reference.
"""

from __future__ import annotations

import math
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


LABEL_THRESHOLD = 0.3
DEFAULT_LANE_HALF_WIDTH_M = 1.75
OCCLUDER_CORRIDOR_CLEARANCE_M = 1.50
TIMING_RANGE_TOLERANCE_S = 0.60
ARRIVAL_TIME_TOLERANCE_S = 1.50
CROSSING_HORIZON_S = 6.0
OCCLUDER_STATIONARY_SPEED_MPS = 0.10


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
    """Evaluate one B0/B1/B2 scene without assuming decoder slot identity.

    ``lane_center_y`` is the authoritative semantic lane/path center used by the
    generator.  For the current ego-centric occluded-pedestrian baseline it is
    normally 0.0, but keeping it explicit prevents future generation/evaluation
    drift.

    Timing definition:
      * pedestrian time = time to first enter the ego-lane boundary;
      * ego time = time for ego to reach pedestrian conflict-x;
      * interaction TTC = max(pedestrian time, ego time);
      * dangerous timing requires TTC inside the risk window and sufficiently
        synchronized arrival times.
    """

    lane_center_y = float(0.0 if lane_center_y is None else lane_center_y)
    lane_half_width = 0.5 * float(
        getattr(spec.road_layer, "lane_width_m", 2.0 * DEFAULT_LANE_HALF_WIDTH_M)
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
                (elem_name, index, _project_agent_state(state, projection_time_s))
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
    speed_error = abs(speed - float(spec.risk_layer.target_actor_speed_mps))
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
    crossing_ok = math.isfinite(t_ped) and 0.0 <= t_ped <= CROSSING_HORIZON_S
    interaction_ttc = (
        max(t_ped, t_ego)
        if math.isfinite(t_ped) and math.isfinite(t_ego)
        else float("inf")
    )
    ttc_low, ttc_high = _normalized_range(spec.risk_layer.ttc_range_s, (2.0, 3.0))
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
    occluder_stationary = False
    occ_payload: Dict[str, Any] = {
        "element": None,
        "index": -1,
        "xy": None,
        "size": None,
        "speed_mps": None,
    }

    if occ_choice is not None:
        elem_name, occ_index, occ = occ_choice
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
                (0.0, 0.0),
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
        corridor_clear = lane_boundary_gap >= OCCLUDER_CORRIDOR_CLEARANCE_M

        occ_speed = _state_speed_if_available(occ)
        occluder_stationary = occ_speed <= OCCLUDER_STATIONARY_SPEED_MPS
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

    checks = {
        "pedestrian_exists": True,
        "occluder_exists": occ_exists,
        "occluder_between_ego_and_actor": between_ok,
        "line_of_sight_occlusion": los_blocked,
        "occluder_clear_of_ego_corridor": corridor_clear,
        "direction_match": direction_ok,
        "speed_match": speed_ok,
        "crossing_reaches_ego_lane": crossing_ok,
        "interaction_timing_match": timing_ok,
        "no_actor_occluder_initial_overlap": no_initial_actor_occluder_overlap,
        "occluder_stationary": occluder_stationary,
    }
    overall_pass = all(bool(value) for value in checks.values())

    return {
        "schema_version": "occluded_pedestrian_metrics_v2_timing_consistent",
        "overall_pass": bool(overall_pass),
        "semantic_satisfaction_rate": float(
            sum(bool(v) for v in checks.values()) / len(checks)
        ),
        "checks": checks,
        "pedestrian": {
            "index": int(ped_index),
            "xy": [
                float(ped[AgentIndex.X]),
                float(ped[AgentIndex.Y]),
            ],
            "heading": float(ped_heading),
            "speed_mps": speed,
            "expected_speed_mps": float(spec.risk_layer.target_actor_speed_mps),
            "speed_error_mps": float(speed_error),
            "inferred_direction": direction,
            "expected_direction": spec.interaction_layer.conflict_direction,
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


def aggregate_stage_metrics(rows: Iterable[Dict[str, Any]], stage: str) -> Dict[str, Any]:
    items = list(rows)
    evaluated = [row for row in items if not row.get("error")]
    checks = sorted({key for row in items for key in row.get("checks", {})})
    denominator = len(items)
    return {
        "schema_version": "occluded_pedestrian_stage_summary_v1",
        "stage": stage,
        "num_rows": len(items),
        "num_evaluated": len(evaluated),
        "num_errors": len(items) - len(evaluated),
        "overall_pass_count": sum(bool(row.get("overall_pass")) for row in items),
        "overall_pass_rate": (
            sum(bool(row.get("overall_pass")) for row in items) / denominator
            if denominator
            else 0.0
        ),
        "mean_semantic_satisfaction_rate": (
            float(np.mean([row.get("semantic_satisfaction_rate", 0.0) for row in items]))
            if items
            else 0.0
        ),
        "check_pass_rates": (
            {
                key: sum(
                    bool(row.get("checks", {}).get(key, False)) for row in items
                )
                / denominator
                for key in checks
            }
            if items
            else {}
        ),
    }


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
        score += 4.0 if inferred == spec.interaction_layer.conflict_direction else 0.0
        score += max(0.0, 2.0 - abs(speed - spec.risk_layer.target_actor_speed_mps))
        score += 1.0 if 3.0 <= float(state[AgentIndex.X]) <= 35.0 else 0.0
        score += 1.0 if abs(float(state[AgentIndex.Y]) - lane_center_y) >= lane_half_width else 0.0
        t_entry = _time_to_lane_entry(
            state,
            lane_center_y=lane_center_y,
            lane_half_width=lane_half_width,
        )
        score += 1.0 if math.isfinite(t_entry) and t_entry <= CROSSING_HORIZON_S else 0.0
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
        between = 0.1 <= ratio <= 0.95 and perpendicular <= 3.0
        size_error = (
            abs(float(state[AgentIndex.WIDTH]) - target.width)
            + abs(float(state[AgentIndex.LENGTH]) - target.length)
        )
        # LOS and geometry dominate; physical size is only a weak tie-breaker
        # because car/truck/bus all project to VEHICLE but legitimately use
        # different sampled dimensions.
        score = (
            8.0 * float(los)
            + 4.0 * float(between)
            - 0.10 * size_error
            - 0.05 * min(perpendicular, 20.0)
        )
        if best is None or score > best[0]:
            best = (score, elem_name, index, state)

    return (best[1], best[2], best[3]) if best is not None else None


def _direction_from_motion(y: float, vy: float, *, lane_center_y: float) -> str:
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


def _axis_aligned_overlap(a: np.ndarray, b: np.ndarray, scale: float) -> bool:
    return (
        abs(float(a[AgentIndex.X]) - float(b[AgentIndex.X]))
        < scale
        * (
            float(max(a[AgentIndex.LENGTH], 0.5))
            + float(max(b[AgentIndex.LENGTH], 0.5))
        )
        and abs(float(a[AgentIndex.Y]) - float(b[AgentIndex.Y]))
        < scale
        * (
            float(max(a[AgentIndex.WIDTH], 0.5))
            + float(max(b[AgentIndex.WIDTH], 0.5))
        )
    )


def _finite_or_none(value: float):
    return float(value) if math.isfinite(value) else None


def _project_agent_state(state: np.ndarray, time_s: float) -> np.ndarray:
    projected = np.asarray(state, dtype=np.float32).copy()
    if time_s <= 0.0 or projected.size <= AgentIndex.VELOCITY:
        return projected
    speed = float(max(projected[AgentIndex.VELOCITY], 0.0))
    heading = float(projected[AgentIndex.HEADING])
    projected[AgentIndex.X] += speed * math.cos(heading) * time_s
    projected[AgentIndex.Y] += speed * math.sin(heading) * time_s
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


def _empty_result(spec: HazardSemanticSpec, reason: str) -> Dict[str, Any]:
    return {
        "schema_version": "occluded_pedestrian_metrics_v2_timing_consistent",
        "overall_pass": False,
        "semantic_satisfaction_rate": 0.0,
        "checks": {"pedestrian_exists": False},
        "reason": reason,
        "expected": {
            "occluder_type": spec.object_layer.occlusion.occluder_type,
            "direction": spec.interaction_layer.conflict_direction,
            "pedestrian_speed_mps": spec.risk_layer.target_actor_speed_mps,
        },
    }


__all__ = [
    "aggregate_stage_metrics",
    "evaluate_occluded_pedestrian_scene",
]