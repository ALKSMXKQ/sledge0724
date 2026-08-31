"""Canonical danger-semantic + stage-aware traffic-realism metrics."""
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
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.traffic_realism import (
    evaluate_traffic_realism,
    infer_occlusion_mode,
)


LABEL_THRESHOLD = 0.3
DEFAULT_LANE_HALF_WIDTH_M = 1.75
ADJACENT_LANE_MIN_EDGE_CLEARANCE_M = 0.10
ROADSIDE_MIN_EDGE_CLEARANCE_M = 0.10
TIMING_RANGE_TOLERANCE_S = 0.60
ARRIVAL_TIME_TOLERANCE_S = 1.50
CROSSING_HORIZON_S = 6.0
OCCLUDER_STATIONARY_SPEED_MPS = 0.10

# Traffic realism is intentionally split into two contracts.
#
# Controlled hazard realism applies to every accepted B1/protected scene.  It
# answers whether the *injected hazard itself* is physically plausible.
# Background realism applies as a hard gate only to generated protected outputs
# (RVAE/diffusion), because B1 keeps a genuine nuPlan background where queues,
# crowds and stopped traffic may be legitimate source-scene content.
CONTROLLED_TRAFFIC_REALISM_KEYS = (
    "occluder_lane_relation",
    "occluder_heading_alignment",
    "occluder_motion_plausibility",
    "pedestrian_emergence_geometry",
    "reveal_time",
)
BACKGROUND_TRAFFIC_REALISM_KEYS = (
    "background_pedestrian_density",
    "background_vehicle_motion",
    "no_cross_lane_vehicle_pose",
    "background_static_lane_safety",
)


def evaluate_occluded_pedestrian_scene(
    scene: Any,
    spec: HazardSemanticSpec,
    *,
    preferred_pedestrian_index: Optional[int] = None,
    preferred_occluder_index: Optional[int] = None,
    preferred_occluder_elem_name: str = "vehicles",
    projection_time_s: float = 0.0,
    lane_center_y: Optional[float] = None,
    require_background_realism: bool = False,
) -> Dict[str, Any]:
    """Evaluate danger semantics plus stage-aware traffic realism.

    Acceptance always requires:
      1. all historical dangerous-semantic checks; and
      2. controlled-hazard traffic realism (lane relation, heading, motion,
         far-side emergence and short reveal time).

    ``require_background_realism`` additionally requires generated-background
    realism.  It should be ``False`` for B1/source-background diagnostics and
    raw baselines, and ``True`` for simulator-ready semantic-protected RVAE and
    diffusion products.

    The controlled pedestrian/occluder are projected by ``projection_time_s``
    because the B1 constructor compensates the SledgeBoard frame-0 offset.
    Background realism is deliberately evaluated at the vector initialization
    frame.  Straight-line projecting every background vehicle for 2.1 seconds
    and comparing it against a curved static road creates false cross-lane
    failures and does not model the reactive simulation dynamics.
    """

    lane_center_y = float(0.0 if lane_center_y is None else lane_center_y)
    lane_half_width = 0.5 * float(
        getattr(
            spec.road_layer,
            "lane_width_m",
            2.0 * DEFAULT_LANE_HALF_WIDTH_M,
        )
    )

    pedestrians = [
        (index, _project_agent_state(state, projection_time_s))
        for index, state in _valid_rows(scene.pedestrians)
    ]
    pedestrian_choice = _select_pedestrian(
        pedestrians,
        spec,
        preferred_index=preferred_pedestrian_index,
        lane_center_y=lane_center_y,
        lane_half_width=lane_half_width,
    )
    if pedestrian_choice is None:
        return _empty_result(
            spec,
            "no valid pedestrian candidate",
            require_background_realism=require_background_realism,
        )
    pedestrian_index, pedestrian = pedestrian_choice

    occluders = []
    for element_name in ("vehicles", "static_objects"):
        for index, state in _valid_rows(getattr(scene, element_name)):
            occluders.append(
                (
                    element_name,
                    index,
                    _project_agent_state(state, projection_time_s),
                )
            )
    occluder_choice = _select_occluder(
        occluders,
        pedestrian,
        spec,
        preferred_index=preferred_occluder_index,
        preferred_elem_name=preferred_occluder_elem_name,
    )

    speed = float(max(pedestrian[AgentIndex.VELOCITY], 0.0))
    pedestrian_heading = float(pedestrian[AgentIndex.HEADING])
    pedestrian_vy = speed * math.sin(pedestrian_heading)
    direction = _direction_from_motion(
        float(pedestrian[AgentIndex.Y]),
        pedestrian_vy,
        lane_center_y=lane_center_y,
    )
    direction_ok = direction == spec.interaction_layer.conflict_direction
    speed_error = abs(
        speed - float(spec.risk_layer.target_actor_speed_mps)
    )
    speed_ok = speed_error <= 0.35

    pedestrian_lane_entry_time = _time_to_lane_entry(
        pedestrian,
        lane_center_y=lane_center_y,
        lane_half_width=lane_half_width,
    )
    ego_speed = float(max(estimate_ego_speed(scene), 1e-3))
    conflict_x = float(pedestrian[AgentIndex.X])
    ego_arrival_time = (
        conflict_x / ego_speed
        if conflict_x > 0.0
        else float("inf")
    )
    arrival_error = (
        abs(pedestrian_lane_entry_time - ego_arrival_time)
        if math.isfinite(pedestrian_lane_entry_time)
        and math.isfinite(ego_arrival_time)
        else float("inf")
    )
    crossing_ok = bool(
        math.isfinite(pedestrian_lane_entry_time)
        and 0.0 <= pedestrian_lane_entry_time <= CROSSING_HORIZON_S
    )
    interaction_ttc = (
        max(pedestrian_lane_entry_time, ego_arrival_time)
        if math.isfinite(pedestrian_lane_entry_time)
        and math.isfinite(ego_arrival_time)
        else float("inf")
    )
    ttc_low, ttc_high = _normalized_range(
        spec.risk_layer.ttc_range_s,
        (2.0, 3.0),
    )
    timing_ok = bool(
        math.isfinite(interaction_ttc)
        and ttc_low - TIMING_RANGE_TOLERANCE_S
        <= interaction_ttc
        <= ttc_high + TIMING_RANGE_TOLERANCE_S
        and arrival_error <= ARRIVAL_TIME_TOLERANCE_S
    )

    occluder_exists = occluder_choice is not None
    los_blocked = False
    between_ok = False
    corridor_clear = False
    legacy_stationary_ok = False
    occluder_payload: Dict[str, Any] = {
        "element": None,
        "index": -1,
        "xy": None,
        "size": None,
        "speed_mps": None,
    }
    occluder_state = None
    occluder_element = None
    occluder_index = -1

    debug = dict(getattr(spec, "debug", {}) or {})
    explicit_mode = str(debug.get("occlusion_mode", "") or "").strip()
    occlusion_mode = infer_occlusion_mode(
        spec,
        explicit=explicit_mode or None,
    )

    if occluder_choice is not None:
        occluder_element, occluder_index, occluder_state = occluder_choice
        actor_x = float(pedestrian[0])
        actor_y = float(pedestrian[1])
        occ_x = float(occluder_state[0])
        occ_y = float(occluder_state[1])
        actor_distance_sq = actor_x * actor_x + actor_y * actor_y
        ratio = (
            (occ_x * actor_x + occ_y * actor_y) / actor_distance_sq
            if actor_distance_sq > 1e-6
            else -1.0
        )
        perpendicular = (
            abs(occ_x * actor_y - occ_y * actor_x)
            / math.sqrt(actor_distance_sq)
            if actor_distance_sq > 1e-6
            else float("inf")
        )
        between_ok = bool(
            0.1 <= ratio <= 0.95 and perpendicular <= 3.0
        )
        los_blocked = bool(
            line_of_sight_intersects_box(
                (0.0, 0.0),
                (actor_x, actor_y),
                occluder_state,
                margin=0.25,
            )
        )

        heading = float(occluder_state[2])
        width = max(float(occluder_state[3]), 0.5)
        length = max(float(occluder_state[4]), 0.5)
        lateral_half_extent = (
            abs(math.sin(heading)) * length / 2.0
            + abs(math.cos(heading)) * width / 2.0
        )
        lane_boundary_gap = (
            abs(occ_y - lane_center_y)
            - lateral_half_extent
            - lane_half_width
        )

        occ_speed = _state_speed_if_available(occluder_state)
        requested_type = str(
            spec.object_layer.occlusion.occluder_type or ""
        ).lower()
        if (
            explicit_mode
            not in {
                "adjacent_lane_dynamic",
                "roadside_parked",
                "roadside_static",
            }
            and requested_type in {"vehicle", "bicycle"}
            and occluder_element == "vehicles"
            and occ_speed > OCCLUDER_STATIONARY_SPEED_MPS
        ):
            # Executable construction prefers a moving adjacent-lane occluder.
            # Trust the actual generated state instead of the retention prompt's
            # language carrier (which may still contain "parked vehicle").
            occlusion_mode = "adjacent_lane_dynamic"

        # Static-object slots are an authoritative physical cue.  This protects
        # barrier/cone/sign cases from the canonical prompt carrier, whose text
        # may still mention a parked vehicle while the executable override is a
        # static occluder.
        if occluder_element == "static_objects":
            occlusion_mode = "roadside_static"

        required_edge = (
            ADJACENT_LANE_MIN_EDGE_CLEARANCE_M
            if occlusion_mode == "adjacent_lane_dynamic"
            else ROADSIDE_MIN_EDGE_CLEARANCE_M
        )
        corridor_clear = bool(lane_boundary_gap >= required_edge)

        # Keep the historical key for downstream compatibility.  For the new
        # dynamic-adjacent occluder, stationarity is not required and the new
        # occluder_motion_plausibility check is authoritative.
        legacy_stationary_ok = bool(
            True
            if occlusion_mode == "adjacent_lane_dynamic"
            else occ_speed <= OCCLUDER_STATIONARY_SPEED_MPS
        )
        occluder_payload = {
            "element": occluder_element,
            "index": int(occluder_index),
            "xy": [occ_x, occ_y],
            "size": [width, length],
            "heading": heading,
            "projection_ratio": float(ratio),
            "perpendicular_distance_m": float(perpendicular),
            "lane_boundary_gap_m": float(lane_boundary_gap),
            "speed_mps": float(occ_speed),
            "stationary_required": (
                occlusion_mode != "adjacent_lane_dynamic"
            ),
        }

    no_overlap = bool(
        True
        if occluder_state is None
        else not _axis_aligned_overlap(
            pedestrian,
            occluder_state,
            scale=0.35,
        )
    )

    danger_checks = {
        "pedestrian_exists": True,
        "occluder_exists": occluder_exists,
        "occluder_between_ego_and_actor": between_ok,
        "line_of_sight_occlusion": los_blocked,
        "occluder_clear_of_ego_corridor": corridor_clear,
        "direction_match": direction_ok,
        "speed_match": speed_ok,
        "crossing_reaches_ego_lane": crossing_ok,
        "interaction_timing_match": timing_ok,
        "no_actor_occluder_initial_overlap": no_overlap,
        "occluder_stationary": legacy_stationary_ok,
    }
    danger_pass = bool(all(danger_checks.values()))

    traffic_realism = evaluate_traffic_realism(
        scene,
        pedestrian_index=int(pedestrian_index),
        pedestrian_state=pedestrian,
        occluder_element=occluder_element,
        occluder_index=int(occluder_index),
        occluder_state=occluder_state,
        spec=spec,
        lane_center_y=lane_center_y,
        lane_half_width=lane_half_width,
        ego_speed_mps=ego_speed,
        # Background quality is assessed at vector initialization.  The
        # controlled hazard states above are already projected to the semantic
        # display frame, so this does not weaken reveal/occlusion evaluation.
        projection_time_s=0.0,
        occlusion_mode=occlusion_mode,
    )
    all_traffic_checks = dict(traffic_realism.get("checks", {}))
    controlled_traffic_checks = {
        key: bool(all_traffic_checks.get(key, False))
        for key in CONTROLLED_TRAFFIC_REALISM_KEYS
    }
    background_traffic_checks = {
        key: bool(all_traffic_checks.get(key, False))
        for key in BACKGROUND_TRAFFIC_REALISM_KEYS
    }
    controlled_traffic_pass = bool(all(controlled_traffic_checks.values()))
    background_traffic_pass = bool(all(background_traffic_checks.values()))
    required_traffic_checks = dict(controlled_traffic_checks)
    if require_background_realism:
        required_traffic_checks.update(background_traffic_checks)

    traffic_realism_pass = bool(
        controlled_traffic_pass
        and (background_traffic_pass if require_background_realism else True)
    )
    required_checks = {**danger_checks, **required_traffic_checks}
    all_checks = {**danger_checks, **all_traffic_checks}
    overall_pass = bool(danger_pass and traffic_realism_pass)

    traffic_realism = dict(traffic_realism)
    traffic_realism.update(
        {
            "controlled_checks": controlled_traffic_checks,
            "background_checks": background_traffic_checks,
            "controlled_pass": controlled_traffic_pass,
            "background_pass": background_traffic_pass,
            "background_required": bool(require_background_realism),
            "required_overall_pass": traffic_realism_pass,
            "background_evaluation_projection_time_s": 0.0,
        }
    )

    return {
        "schema_version": "occluded_pedestrian_metrics_v4_stage_aware_realism",
        "overall_pass": overall_pass,
        "danger_semantic_pass": danger_pass,
        "traffic_realism_pass": traffic_realism_pass,
        "controlled_traffic_realism_pass": controlled_traffic_pass,
        "background_realism_pass": background_traffic_pass,
        "background_realism_required": bool(require_background_realism),
        # Preserve the original interpretation: semantic SSR covers only the
        # historical danger checks.  Realism has its own rates below.
        "semantic_satisfaction_rate": float(
            sum(bool(value) for value in danger_checks.values())
            / len(danger_checks)
        ),
        "traffic_realism_satisfaction_rate": float(
            traffic_realism.get("satisfaction_rate", 0.0)
        ),
        "controlled_traffic_realism_satisfaction_rate": float(
            sum(controlled_traffic_checks.values())
            / len(controlled_traffic_checks)
        ),
        "background_realism_satisfaction_rate": float(
            sum(background_traffic_checks.values())
            / len(background_traffic_checks)
        ),
        "combined_satisfaction_rate": float(
            sum(bool(value) for value in required_checks.values())
            / len(required_checks)
        ),
        # ``checks`` contains only the checks that are gates for this stage.
        # Full diagnostics remain available in ``all_checks`` and in the two
        # traffic-realism dictionaries below.
        "checks": required_checks,
        "all_checks": all_checks,
        "danger_checks": danger_checks,
        "traffic_realism_checks": all_traffic_checks,
        "controlled_traffic_realism_checks": controlled_traffic_checks,
        "background_realism_checks": background_traffic_checks,
        "traffic_realism": traffic_realism,
        "pedestrian": {
            "index": int(pedestrian_index),
            "xy": [
                float(pedestrian[0]),
                float(pedestrian[1]),
            ],
            "heading": pedestrian_heading,
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
        "occluder": occluder_payload,
        "interaction": {
            "ego_speed_mps": ego_speed,
            "projection_time_s": float(projection_time_s),
            "lane_center_y": lane_center_y,
            "lane_half_width_m": lane_half_width,
            "conflict_x_m": conflict_x,
            "pedestrian_lane_entry_time_s": _finite_or_none(
                pedestrian_lane_entry_time
            ),
            "ego_arrival_time_s": _finite_or_none(ego_arrival_time),
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
    check_keys = sorted(
        {
            key
            for row in items
            for key in row.get("checks", {})
        }
    )
    count = len(items)
    overall_pass_count = sum(
        bool(row.get("overall_pass")) for row in items
    )
    return {
        "schema_version": "occluded_pedestrian_stage_summary_v3_stage_aware_realism",
        "stage": stage,
        "num_rows": count,
        "num_evaluated": len(evaluated),
        "num_errors": count - len(evaluated),
        "overall_pass_count": overall_pass_count,
        "overall_pass_rate": (
            overall_pass_count / count if count else 0.0
        ),
        "danger_semantic_pass_count": sum(
            bool(row.get("danger_semantic_pass")) for row in items
        ),
        "traffic_realism_pass_count": sum(
            bool(row.get("traffic_realism_pass")) for row in items
        ),
        "controlled_traffic_realism_pass_count": sum(
            bool(row.get("controlled_traffic_realism_pass")) for row in items
        ),
        "background_realism_pass_count": sum(
            bool(row.get("background_realism_pass")) for row in items
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
        "mean_traffic_realism_satisfaction_rate": (
            float(
                np.mean(
                    [
                        row.get(
                            "traffic_realism_satisfaction_rate",
                            0.0,
                        )
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
                / count
                for key in check_keys
            }
            if count
            else {}
        ),
    }


def _valid_rows(elem: Any):
    states = np.asarray(elem.states)
    masks = np.asarray(elem.mask).reshape(-1)
    if states.ndim == 1:
        states = states[None, :]
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
        heading = float(state[2])
        velocity_y = speed * math.sin(heading)
        inferred = _direction_from_motion(
            float(state[1]),
            velocity_y,
            lane_center_y=lane_center_y,
        )
        score = 4.0 * (
            inferred == spec.interaction_layer.conflict_direction
        )
        score += max(
            0.0,
            2.0
            - abs(
                speed - spec.risk_layer.target_actor_speed_mps
            ),
        )
        score += 1.0 if 3.0 <= float(state[0]) <= 35.0 else 0.0
        score += (
            1.0
            if abs(float(state[1]) - lane_center_y) >= lane_half_width
            else 0.0
        )
        lane_entry = _time_to_lane_entry(
            state,
            lane_center_y=lane_center_y,
            lane_half_width=lane_half_width,
        )
        score += (
            1.0
            if math.isfinite(lane_entry)
            and lane_entry <= CROSSING_HORIZON_S
            else 0.0
        )
        if best is None or score > best[0]:
            best = (score, index, state)
    return (best[1], best[2]) if best else None


def _select_occluder(
    candidates,
    pedestrian,
    spec,
    *,
    preferred_index=None,
    preferred_elem_name="vehicles",
):
    for element_name, index, state in candidates:
        if (
            preferred_index is not None
            and index == preferred_index
            and element_name == preferred_elem_name
        ):
            return element_name, index, state

    target = OCCLUDER_SPECS.get(
        spec.object_layer.occlusion.occluder_type,
        OCCLUDER_SPECS["vehicle"],
    )
    actor_x = float(pedestrian[0])
    actor_y = float(pedestrian[1])
    distance_sq = actor_x * actor_x + actor_y * actor_y
    best = None

    for element_name, index, state in candidates:
        if element_name != target.elem_name:
            continue
        los = line_of_sight_intersects_box(
            (0.0, 0.0),
            (actor_x, actor_y),
            state,
            margin=0.25,
        )
        ratio = (
            (float(state[0]) * actor_x + float(state[1]) * actor_y)
            / distance_sq
            if distance_sq > 1e-6
            else -1.0
        )
        perpendicular = (
            abs(float(state[0]) * actor_y - float(state[1]) * actor_x)
            / math.sqrt(distance_sq)
            if distance_sq > 1e-6
            else float("inf")
        )
        between = bool(
            0.1 <= ratio <= 0.95 and perpendicular <= 3.0
        )
        size_error = (
            abs(float(state[3]) - target.width)
            + abs(float(state[4]) - target.length)
        )
        score = (
            8.0 * float(los)
            + 4.0 * float(between)
            - 0.10 * size_error
            - 0.05 * min(perpendicular, 20.0)
        )
        if best is None or score > best[0]:
            best = (score, element_name, index, state)
    return (best[1], best[2], best[3]) if best else None


def _direction_from_motion(
    y: float,
    velocity_y: float,
    *,
    lane_center_y: float,
) -> str:
    relative_y = y - lane_center_y
    if relative_y < 0.0 and velocity_y > 0.0:
        return "right_to_left"
    if relative_y > 0.0 and velocity_y < 0.0:
        return "left_to_right"
    return "unknown"


def _time_to_lane_entry(
    state: np.ndarray,
    *,
    lane_center_y: float,
    lane_half_width: float,
) -> float:
    y = float(state[1])
    speed = _state_speed_if_available(state)
    velocity_y = speed * math.sin(float(state[2]))
    upper = lane_center_y + lane_half_width
    lower = lane_center_y - lane_half_width
    if y > upper and velocity_y < -1e-5:
        return (y - upper) / (-velocity_y)
    if y < lower and velocity_y > 1e-5:
        return (lower - y) / velocity_y
    if lower <= y <= upper:
        return 0.0
    return float("inf")


def _axis_aligned_overlap(
    first: np.ndarray,
    second: np.ndarray,
    scale: float,
) -> bool:
    return bool(
        abs(float(first[0]) - float(second[0]))
        < scale
        * (
            max(float(first[4]), 0.5)
            + max(float(second[4]), 0.5)
        )
        and abs(float(first[1]) - float(second[1]))
        < scale
        * (
            max(float(first[3]), 0.5)
            + max(float(second[3]), 0.5)
        )
    )


def _finite_or_none(value: float):
    return float(value) if math.isfinite(value) else None


def _project_agent_state(
    state: np.ndarray,
    time_s: float,
) -> np.ndarray:
    out = np.asarray(state, dtype=np.float32).copy()
    if time_s <= 0.0 or out.size <= AgentIndex.VELOCITY:
        return out
    speed = max(float(out[AgentIndex.VELOCITY]), 0.0)
    heading = float(out[AgentIndex.HEADING])
    out[AgentIndex.X] += speed * math.cos(heading) * time_s
    out[AgentIndex.Y] += speed * math.sin(heading) * time_s
    return out


def _state_speed_if_available(state: np.ndarray) -> float:
    state = np.asarray(state)
    if state.size <= AgentIndex.VELOCITY:
        return 0.0
    return float(max(state[AgentIndex.VELOCITY], 0.0))


def _normalized_range(
    value: Any,
    default,
) -> tuple[float, float]:
    try:
        values = list(value)
        low, high = sorted([float(values[0]), float(values[1])])
        return low, high
    except Exception:
        return float(default[0]), float(default[1])


def _empty_result(
    spec: HazardSemanticSpec,
    reason: str,
    *,
    require_background_realism: bool,
) -> Dict[str, Any]:
    return {
        "schema_version": "occluded_pedestrian_metrics_v4_stage_aware_realism",
        "overall_pass": False,
        "danger_semantic_pass": False,
        "traffic_realism_pass": False,
        "controlled_traffic_realism_pass": False,
        "background_realism_pass": False,
        "background_realism_required": bool(require_background_realism),
        "semantic_satisfaction_rate": 0.0,
        "traffic_realism_satisfaction_rate": 0.0,
        "checks": {"pedestrian_exists": False},
        "all_checks": {"pedestrian_exists": False},
        "danger_checks": {"pedestrian_exists": False},
        "traffic_realism_checks": {},
        "controlled_traffic_realism_checks": {},
        "background_realism_checks": {},
        "reason": reason,
        "expected": {
            "occluder_type": spec.object_layer.occlusion.occluder_type,
            "direction": spec.interaction_layer.conflict_direction,
            "pedestrian_speed_mps": spec.risk_layer.target_actor_speed_mps,
        },
    }


__all__ = [
    "BACKGROUND_TRAFFIC_REALISM_KEYS",
    "CONTROLLED_TRAFFIC_REALISM_KEYS",
    "aggregate_stage_metrics",
    "evaluate_occluded_pedestrian_scene",
]
