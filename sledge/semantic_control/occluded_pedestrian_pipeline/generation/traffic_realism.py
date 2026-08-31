"""Traffic-realism utilities for protected occluded-pedestrian scenes.

This module adds a second, explicit contract on top of the dangerous-semantic
contract:

* vehicle/bicycle occluders use a nearby adjacent lane and move with traffic;
* parked/static occluders stay roadside and align with local road direction;
* the pedestrian emerges from the far side of the occluder with a short reveal
  time;
* unrelated generated pedestrians/vehicles/static objects are cleaned only in
  semantic-protected products;
* raw RVAE/diffusion outputs are never modified by these helpers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    AgentIndex,
    EgoIndex,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.geometry_metrics import (
    line_of_sight_intersects_box,
)


@dataclass(frozen=True)
class TrafficRealismPolicy:
    """Numerical traffic-realism contract shared by generation and evaluation."""

    label_threshold: float = 0.3

    # Occluder placement.
    adjacent_lane_center_tolerance_m: float = 0.85
    min_lane_edge_clearance_m: float = 0.10
    roadside_min_edge_gap_m: float = 0.10
    roadside_max_edge_gap_m: float = 2.50
    heading_tolerance_deg: float = 35.0
    dynamic_occluder_min_speed_mps: float = 1.0
    dynamic_occluder_max_speed_mps: float = 12.0
    parked_speed_max_mps: float = 0.30

    # Sudden reveal geometry.
    reveal_min_time_s: float = 0.10
    reveal_max_time_s: float = 1.50
    reveal_step_s: float = 0.10
    emergence_min_far_side_gap_m: float = 0.20
    emergence_max_far_side_gap_m: float = 5.50
    emergence_max_longitudinal_gap_m: float = 8.00

    # Protected-background cleanup.
    max_background_pedestrians: int = 8
    pedestrian_min_spacing_m: float = 1.40
    pedestrian_stationary_speed_mps: float = 0.20
    pedestrian_road_proximity_m: float = 2.20
    vehicle_lane_proximity_m: float = 2.80
    vehicle_min_moving_speed_mps: float = 0.75
    vehicle_max_speed_mps: float = 12.0
    vehicle_heading_snap_deg: float = 35.0
    static_lane_center_exclusion_m: float = 0.80
    max_background_static_objects: int = 12

    # Validation tolerances. These are deliberately looser than protected
    # cleanup because genuine source scenes may contain queues/crowds.
    max_stationary_road_pedestrians: int = 3
    max_background_pedestrians_for_gate: int = 12
    max_close_pedestrian_pairs: int = 2
    max_stationary_road_vehicles_absolute: int = 2
    max_stationary_road_vehicle_fraction: float = 0.40
    max_misaligned_road_vehicle_fraction: float = 0.15
    evaluation_frame_half_extent_m: float = 32.0


DEFAULT_TRAFFIC_REALISM_POLICY = TrafficRealismPolicy()


def infer_occlusion_mode(
    spec: Any,
    *,
    explicit: Optional[str] = None,
    occluder_type: Optional[str] = None,
) -> str:
    """Resolve the physical occlusion mode.

    An executable/debug override is authoritative.  Prompt text is only a
    backwards-compatible fallback for old saved specs.  New B1 construction
    deliberately defaults vehicle/bicycle occluders to dynamic adjacent-lane
    traffic even if the retention prompt carrier contains the word ``parked``.
    """

    debug = dict(getattr(spec, "debug", {}) or {})
    mode = str(explicit or debug.get("occlusion_mode", "") or "")
    if mode in {
        "adjacent_lane_dynamic",
        "roadside_parked",
        "roadside_static",
    }:
        return mode

    if occluder_type is None:
        occluder_type = str(
            getattr(spec.object_layer.occlusion, "occluder_type", "vehicle")
        )
    occluder_type = str(occluder_type)
    prompt = str(
        getattr(spec, "raw_prompt", "")
        or getattr(spec, "description", "")
        or ""
    ).lower()

    if occluder_type == "vehicle" and any(
        token in prompt for token in ("parked", "parking", "stopped roadside")
    ):
        return "roadside_parked"
    if occluder_type in {"vehicle", "bicycle"}:
        return "adjacent_lane_dynamic"
    return "roadside_static"


def nearest_lane_tangent(
    scene_or_lines: Any,
    x: float,
    y: float,
) -> Optional[Tuple[float, float]]:
    """Return ``(distance, heading)`` for the nearest line segment.

    Works with both raw lines (per-point masks) and processed lines (one mask
    value per polyline).
    """

    lines = getattr(scene_or_lines, "lines", scene_or_lines)
    states = np.asarray(getattr(lines, "states", []))
    masks = np.asarray(getattr(lines, "mask", []))
    if states.size == 0:
        return None
    if states.ndim == 2:
        states = states[None, ...]
    if states.ndim < 3:
        return None

    target = np.asarray([float(x), float(y)], dtype=np.float64)
    best: Optional[Tuple[float, float]] = None

    for line_index in range(states.shape[0]):
        points = np.asarray(states[line_index], dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
            continue

        if masks.size:
            if masks.ndim == 1:
                if line_index >= len(masks) or not _active(masks[line_index]):
                    continue
                valid = points
            else:
                if line_index >= masks.shape[0]:
                    continue
                point_mask = np.asarray(masks[line_index]).reshape(-1)
                usable = min(len(points), len(point_mask))
                active = np.asarray(
                    [_active(value) for value in point_mask[:usable]],
                    dtype=bool,
                )
                valid = points[:usable][active]
        else:
            valid = points

        if len(valid) < 2:
            continue

        xy = valid[:, :2]
        for p0, p1 in zip(xy[:-1], xy[1:]):
            delta = p1 - p0
            denom = float(np.dot(delta, delta))
            if denom <= 1e-8:
                continue
            ratio = float(
                np.clip(np.dot(target - p0, delta) / denom, 0.0, 1.0)
            )
            closest = p0 + ratio * delta
            distance = float(np.linalg.norm(target - closest))
            heading = float(math.atan2(float(delta[1]), float(delta[0])))
            if best is None or distance < best[0]:
                best = (distance, heading)

    return best


def heading_error_mod_pi(heading: float, lane_heading: float) -> float:
    """Unsigned heading error where either lane-line direction is equivalent."""

    delta = _wrap(float(heading) - float(lane_heading))
    return float(min(abs(delta), abs(_wrap(delta + math.pi))))


def align_heading_to_lane(heading: float, lane_heading: float) -> float:
    """Choose the lane-line direction closest to the current heading."""

    first = _wrap(lane_heading)
    second = _wrap(lane_heading + math.pi)
    return float(
        first
        if abs(_wrap(heading - first)) <= abs(_wrap(heading - second))
        else second
    )


def project_agent_state(state: np.ndarray, time_s: float) -> np.ndarray:
    out = np.asarray(state, dtype=np.float32).copy().reshape(-1)
    if time_s <= 0.0 or out.size <= AgentIndex.VELOCITY:
        return out
    speed = max(float(out[AgentIndex.VELOCITY]), 0.0)
    heading = float(out[AgentIndex.HEADING])
    out[AgentIndex.X] += speed * math.cos(heading) * time_s
    out[AgentIndex.Y] += speed * math.sin(heading) * time_s
    return out


def state_speed(state: np.ndarray) -> float:
    state = np.asarray(state).reshape(-1)
    if state.size <= AgentIndex.VELOCITY:
        return 0.0
    return float(max(state[AgentIndex.VELOCITY], 0.0))


def estimate_reveal_time(
    pedestrian_state: np.ndarray,
    occluder_state: np.ndarray,
    *,
    ego_speed_mps: float,
    policy: TrafficRealismPolicy = DEFAULT_TRAFFIC_REALISM_POLICY,
) -> Optional[float]:
    """First time at which the occluder no longer blocks ego->pedestrian LOS."""

    pedestrian = np.asarray(pedestrian_state, dtype=np.float32).reshape(-1)
    occluder = np.asarray(occluder_state, dtype=np.float32).reshape(-1)
    if pedestrian.size < 5 or occluder.size < 5:
        return None

    if not line_of_sight_intersects_box(
        (0.0, 0.0),
        (float(pedestrian[0]), float(pedestrian[1])),
        occluder,
        margin=0.25,
    ):
        return None

    times = np.arange(
        policy.reveal_step_s,
        policy.reveal_max_time_s + policy.reveal_step_s * 0.5,
        policy.reveal_step_s,
    )
    for time_s in times:
        projected_ped = project_agent_state(pedestrian, float(time_s))
        projected_occ = project_agent_state(occluder, float(time_s))
        ego_xy = (float(max(ego_speed_mps, 0.0) * time_s), 0.0)
        blocked = line_of_sight_intersects_box(
            ego_xy,
            (float(projected_ped[0]), float(projected_ped[1])),
            projected_occ,
            margin=0.25,
        )
        if not blocked:
            return float(time_s)
    return None


def sanitize_generated_background(
    vector: Any,
    *,
    protected_pedestrian_index: int,
    protected_occluder_element: str,
    protected_occluder_index: int,
    policy: TrafficRealismPolicy = DEFAULT_TRAFFIC_REALISM_POLICY,
) -> Dict[str, Any]:
    """Clean only unrelated generated actors in a protected output.

    The exact B1 road, ego, controlled pedestrian and controlled occluder must
    already have been copied into ``vector``.  Those controlled slots are never
    altered here.
    """

    report: Dict[str, Any] = {
        "schema_version": "protected_background_traffic_realism_v1",
        "policy": asdict(policy),
        "pedestrians_removed": [],
        "vehicles_removed": [],
        "vehicles_heading_aligned": [],
        "vehicles_speed_repaired": [],
        "static_objects_removed": [],
    }

    _sanitize_pedestrians(
        vector,
        protected_pedestrian_index,
        report,
        policy,
    )
    _sanitize_vehicles(
        vector,
        protected_occluder_element,
        protected_occluder_index,
        report,
        policy,
    )
    _sanitize_static(
        vector,
        protected_occluder_element,
        protected_occluder_index,
        report,
        policy,
    )

    for key in (
        "pedestrians_removed",
        "vehicles_removed",
        "vehicles_heading_aligned",
        "vehicles_speed_repaired",
        "static_objects_removed",
    ):
        report[f"{key}_count"] = len(report[key])
    return report


def _sanitize_pedestrians(
    vector: Any,
    protected_index: int,
    report: Dict[str, Any],
    policy: TrafficRealismPolicy,
) -> None:
    elem = vector.pedestrians
    states = np.asarray(elem.states)
    masks = np.asarray(elem.mask).reshape(-1)
    usable = min(len(states), len(masks))

    kept_xy = []
    if (
        0 <= protected_index < usable
        and _active(masks[protected_index], policy.label_threshold)
    ):
        kept_xy.append(
            tuple(map(float, states[protected_index, AgentIndex.POINT]))
        )

    candidates = []
    for index in range(usable):
        if index == protected_index or not _active(
            masks[index], policy.label_threshold
        ):
            continue
        state = states[index]
        # Moving, high-confidence, near actors are considered before weak
        # stationary duplicates.
        key = (
            0
            if state_speed(state) >= policy.pedestrian_stationary_speed_mps
            else 1,
            -_score(masks[index]),
            float(np.linalg.norm(state[:2])),
        )
        candidates.append((key, index))
    candidates.sort()

    background_kept = 0
    for _, index in candidates:
        state = states[index]
        x = float(state[AgentIndex.X])
        y = float(state[AgentIndex.Y])
        speed = state_speed(state)
        tangent = nearest_lane_tangent(vector, x, y)
        road_distance = tangent[0] if tangent is not None else float("inf")

        reason = None
        if (
            speed < policy.pedestrian_stationary_speed_mps
            and road_distance <= policy.pedestrian_road_proximity_m
        ):
            reason = "stationary_pedestrian_on_drivable_lane"
        elif any(
            math.hypot(x - kept_x, y - kept_y)
            < policy.pedestrian_min_spacing_m
            for kept_x, kept_y in kept_xy
        ):
            reason = "duplicate_or_overdense_pedestrian"
        elif background_kept >= policy.max_background_pedestrians:
            reason = "background_pedestrian_cap"

        if reason is not None:
            _deactivate(elem, index)
            report["pedestrians_removed"].append(
                {
                    "index": int(index),
                    "reason": reason,
                    "x": x,
                    "y": y,
                    "speed_mps": speed,
                }
            )
        else:
            kept_xy.append((x, y))
            background_kept += 1


def _sanitize_vehicles(
    vector: Any,
    occluder_element: str,
    occluder_index: int,
    report: Dict[str, Any],
    policy: TrafficRealismPolicy,
) -> None:
    elem = vector.vehicles
    states = np.asarray(elem.states)
    masks = np.asarray(elem.mask).reshape(-1)
    usable = min(len(states), len(masks))

    ego_speed = _ego_speed(vector)
    target_speed = float(
        np.clip(0.65 * max(ego_speed, 3.0), 2.5, 8.0)
    )

    occupied = [_ego_aabb()]
    if (
        occluder_element == "vehicles"
        and 0 <= occluder_index < usable
        and _active(masks[occluder_index], policy.label_threshold)
    ):
        occupied.append(_state_aabb(states[occluder_index], margin=0.15))

    for index in range(usable):
        if not _active(masks[index], policy.label_threshold):
            continue
        if occluder_element == "vehicles" and index == occluder_index:
            continue

        state = states[index]
        x = float(state[AgentIndex.X])
        y = float(state[AgentIndex.Y])
        tangent = nearest_lane_tangent(vector, x, y)

        if tangent is not None and tangent[0] <= policy.vehicle_lane_proximity_m:
            old_heading = float(state[AgentIndex.HEADING])
            heading_error_deg = math.degrees(
                heading_error_mod_pi(old_heading, tangent[1])
            )
            if heading_error_deg > policy.vehicle_heading_snap_deg:
                new_heading = align_heading_to_lane(old_heading, tangent[1])
                state[AgentIndex.HEADING] = new_heading
                report["vehicles_heading_aligned"].append(
                    {
                        "index": int(index),
                        "old_heading": old_heading,
                        "new_heading": float(new_heading),
                        "old_error_deg": float(heading_error_deg),
                    }
                )

            old_speed = state_speed(state)
            if old_speed < policy.vehicle_min_moving_speed_mps:
                new_speed = float(
                    np.clip(
                        target_speed * (0.90 + 0.05 * (index % 3)),
                        policy.vehicle_min_moving_speed_mps,
                        policy.vehicle_max_speed_mps,
                    )
                )
                state[AgentIndex.VELOCITY] = new_speed
                report["vehicles_speed_repaired"].append(
                    {
                        "index": int(index),
                        "old_speed_mps": old_speed,
                        "new_speed_mps": new_speed,
                    }
                )
            elif old_speed > policy.vehicle_max_speed_mps:
                state[AgentIndex.VELOCITY] = policy.vehicle_max_speed_mps
                report["vehicles_speed_repaired"].append(
                    {
                        "index": int(index),
                        "old_speed_mps": old_speed,
                        "new_speed_mps": policy.vehicle_max_speed_mps,
                    }
                )

        box = _state_aabb(state, margin=0.10)
        if any(_overlap(box, other) for other in occupied):
            _deactivate(elem, index)
            report["vehicles_removed"].append(
                {
                    "index": int(index),
                    "reason": "initial_vehicle_overlap_after_realism_repair",
                    "x": x,
                    "y": y,
                }
            )
        else:
            occupied.append(box)


def _sanitize_static(
    vector: Any,
    occluder_element: str,
    occluder_index: int,
    report: Dict[str, Any],
    policy: TrafficRealismPolicy,
) -> None:
    elem = vector.static_objects
    states = np.asarray(elem.states)
    masks = np.asarray(elem.mask).reshape(-1)
    usable = min(len(states), len(masks))

    kept_xy = []
    background_kept = 0
    for index in range(usable):
        if not _active(masks[index], policy.label_threshold):
            continue
        if occluder_element == "static_objects" and index == occluder_index:
            continue

        state = states[index]
        x = float(state[0])
        y = float(state[1])
        tangent = nearest_lane_tangent(vector, x, y)
        reason = None

        if (
            tangent is not None
            and tangent[0] < policy.static_lane_center_exclusion_m
        ):
            reason = "background_static_object_inside_lane_center"
        elif any(math.hypot(x - kx, y - ky) < 1.0 for kx, ky in kept_xy):
            reason = "duplicate_background_static_object"
        elif background_kept >= policy.max_background_static_objects:
            reason = "background_static_object_cap"

        if reason is not None:
            _deactivate(elem, index)
            report["static_objects_removed"].append(
                {
                    "index": int(index),
                    "reason": reason,
                    "x": x,
                    "y": y,
                }
            )
        else:
            kept_xy.append((x, y))
            background_kept += 1


def evaluate_traffic_realism(
    scene: Any,
    *,
    pedestrian_index: int,
    pedestrian_state: np.ndarray,
    occluder_element: Optional[str],
    occluder_index: int,
    occluder_state: Optional[np.ndarray],
    spec: Any,
    lane_center_y: float,
    lane_half_width: float,
    ego_speed_mps: float,
    projection_time_s: float,
    occlusion_mode: Optional[str] = None,
    policy: TrafficRealismPolicy = DEFAULT_TRAFFIC_REALISM_POLICY,
) -> Dict[str, Any]:
    """Evaluate Level-2 traffic realism independently of danger semantics."""

    mode = infer_occlusion_mode(spec, explicit=occlusion_mode)
    lane_width = float(
        getattr(spec.road_layer, "lane_width_m", 2.0 * lane_half_width)
    )
    pedestrian = np.asarray(pedestrian_state, dtype=np.float32).reshape(-1)
    occluder = (
        None
        if occluder_state is None
        else np.asarray(occluder_state, dtype=np.float32).reshape(-1)
    )

    lane_ok = False
    heading_ok = False
    motion_ok = False
    emergence_ok = False
    reveal_ok = False
    reveal_time = None
    heading_error_deg = None
    lane_distance = None
    lane_gap = None

    if occluder is not None and occluder.size >= 5:
        ox = float(occluder[0])
        oy = float(occluder[1])
        px = float(pedestrian[0])
        py = float(pedestrian[1])
        heading = float(occluder[2])
        width = max(float(occluder[3]), 0.25)
        length = max(float(occluder[4]), 0.25)

        tangent = nearest_lane_tangent(scene, ox, oy)
        lane_heading = tangent[1] if tangent is not None else 0.0
        lane_distance = tangent[0] if tangent is not None else None
        heading_error_deg = math.degrees(
            heading_error_mod_pi(heading, lane_heading)
        )
        # Very small static objects have no meaningful longitudinal axis.
        heading_ok = (
            heading_error_deg <= policy.heading_tolerance_deg
            or max(width, length) <= 1.0
        )

        lateral_half_extent = (
            abs(math.sin(heading)) * length / 2.0
            + abs(math.cos(heading)) * width / 2.0
        )
        lateral_center = abs(oy - lane_center_y)
        lane_gap = lateral_center - lateral_half_extent - lane_half_width
        same_side = (py - lane_center_y) * (oy - lane_center_y) > 0.0
        speed = state_speed(occluder)

        if mode == "adjacent_lane_dynamic":
            lane_ok = bool(
                same_side
                and abs(lateral_center - lane_width)
                <= policy.adjacent_lane_center_tolerance_m
                and lane_gap >= policy.min_lane_edge_clearance_m
            )
            motion_ok = bool(
                policy.dynamic_occluder_min_speed_mps
                <= speed
                <= policy.dynamic_occluder_max_speed_mps
            )
        else:
            lane_ok = bool(
                same_side
                and policy.roadside_min_edge_gap_m
                <= lane_gap
                <= policy.roadside_max_edge_gap_m
            )
            motion_ok = bool(speed <= policy.parked_speed_max_mps)

        far_side_gap = (
            abs(py - lane_center_y) - abs(oy - lane_center_y)
        )
        emergence_ok = bool(
            same_side
            and policy.emergence_min_far_side_gap_m
            <= far_side_gap
            <= policy.emergence_max_far_side_gap_m
            and abs(px - ox) <= policy.emergence_max_longitudinal_gap_m
        )

        reveal_time = estimate_reveal_time(
            pedestrian,
            occluder,
            ego_speed_mps=ego_speed_mps,
            policy=policy,
        )
        reveal_ok = bool(
            reveal_time is not None
            and policy.reveal_min_time_s
            <= reveal_time
            <= policy.reveal_max_time_s
        )

    background = _evaluate_background(
        scene,
        pedestrian_index,
        str(occluder_element or ""),
        occluder_index,
        projection_time_s,
        policy,
    )

    checks = {
        "occluder_lane_relation": bool(lane_ok),
        "occluder_heading_alignment": bool(heading_ok),
        "occluder_motion_plausibility": bool(motion_ok),
        "pedestrian_emergence_geometry": bool(emergence_ok),
        "reveal_time": bool(reveal_ok),
        "background_pedestrian_density": bool(
            background["background_pedestrian_density_pass"]
        ),
        "background_vehicle_motion": bool(
            background["background_vehicle_motion_pass"]
        ),
        "no_cross_lane_vehicle_pose": bool(
            background["no_cross_lane_vehicle_pose_pass"]
        ),
        "background_static_lane_safety": bool(
            background["background_static_lane_safety_pass"]
        ),
    }

    return {
        "schema_version": "occluded_pedestrian_traffic_realism_v1",
        "occlusion_mode": mode,
        "overall_pass": bool(all(checks.values())),
        "satisfaction_rate": float(sum(checks.values()) / len(checks)),
        "checks": checks,
        "occluder": {
            "nearest_lane_distance_m": lane_distance,
            "heading_error_deg": heading_error_deg,
            "lane_boundary_gap_m": lane_gap,
            "reveal_time_s": reveal_time,
        },
        "background": background,
    }


def _evaluate_background(
    scene: Any,
    pedestrian_index: int,
    occluder_element: str,
    occluder_index: int,
    projection_time_s: float,
    policy: TrafficRealismPolicy,
) -> Dict[str, Any]:
    half_extent = policy.evaluation_frame_half_extent_m

    pedestrians = []
    for index, state in _valid_rows(scene.pedestrians, policy.label_threshold):
        if index == pedestrian_index:
            continue
        projected = project_agent_state(state, projection_time_s)
        if (
            abs(float(projected[0])) <= half_extent
            and abs(float(projected[1])) <= half_extent
        ):
            pedestrians.append((index, projected))

    stationary_road_pedestrians = 0
    close_pedestrian_pairs = 0
    for position, (_, state) in enumerate(pedestrians):
        tangent = nearest_lane_tangent(
            scene,
            float(state[0]),
            float(state[1]),
        )
        if (
            state_speed(state) < policy.pedestrian_stationary_speed_mps
            and tangent is not None
            and tangent[0] <= policy.pedestrian_road_proximity_m
        ):
            stationary_road_pedestrians += 1
        for _, other in pedestrians[position + 1 :]:
            if (
                np.linalg.norm(state[:2] - other[:2])
                < policy.pedestrian_min_spacing_m
            ):
                close_pedestrian_pairs += 1

    pedestrian_pass = bool(
        len(pedestrians) <= policy.max_background_pedestrians_for_gate
        and stationary_road_pedestrians
        <= policy.max_stationary_road_pedestrians
        and close_pedestrian_pairs <= policy.max_close_pedestrian_pairs
    )

    onroad_vehicles = 0
    stationary_onroad_vehicles = 0
    misaligned_onroad_vehicles = 0
    for index, state in _valid_rows(scene.vehicles, policy.label_threshold):
        if occluder_element == "vehicles" and index == occluder_index:
            continue
        projected = project_agent_state(state, projection_time_s)
        if (
            abs(float(projected[0])) > half_extent
            or abs(float(projected[1])) > half_extent
        ):
            continue
        tangent = nearest_lane_tangent(
            scene,
            float(projected[0]),
            float(projected[1]),
        )
        if tangent is None or tangent[0] > policy.vehicle_lane_proximity_m:
            continue
        onroad_vehicles += 1
        if state_speed(projected) < policy.vehicle_min_moving_speed_mps:
            stationary_onroad_vehicles += 1
        if (
            math.degrees(
                heading_error_mod_pi(float(projected[2]), tangent[1])
            )
            > policy.heading_tolerance_deg
        ):
            misaligned_onroad_vehicles += 1

    allowed_stationary = max(
        policy.max_stationary_road_vehicles_absolute,
        int(
            math.floor(
                policy.max_stationary_road_vehicle_fraction
                * onroad_vehicles
            )
        ),
    )
    vehicle_motion_pass = bool(
        stationary_onroad_vehicles <= allowed_stationary
    )
    allowed_misaligned = int(
        math.floor(
            policy.max_misaligned_road_vehicle_fraction * onroad_vehicles
        )
    )
    vehicle_pose_pass = bool(
        misaligned_onroad_vehicles <= allowed_misaligned
    )

    static_inside_lane_center = 0
    for index, state in _valid_rows(
        scene.static_objects,
        policy.label_threshold,
    ):
        if occluder_element == "static_objects" and index == occluder_index:
            continue
        if (
            abs(float(state[0])) > half_extent
            or abs(float(state[1])) > half_extent
        ):
            continue
        tangent = nearest_lane_tangent(scene, float(state[0]), float(state[1]))
        if (
            tangent is not None
            and tangent[0] < policy.static_lane_center_exclusion_m
        ):
            static_inside_lane_center += 1

    # Allow one source-scene exception (e.g. an actual construction object),
    # but generated protected outputs are cleaned more aggressively above.
    static_pass = bool(static_inside_lane_center <= 1)

    return {
        "num_background_pedestrians": len(pedestrians),
        "stationary_road_pedestrians": stationary_road_pedestrians,
        "close_pedestrian_pairs": close_pedestrian_pairs,
        "background_pedestrian_density_pass": pedestrian_pass,
        "onroad_background_vehicles": onroad_vehicles,
        "stationary_onroad_background_vehicles": stationary_onroad_vehicles,
        "misaligned_onroad_background_vehicles": misaligned_onroad_vehicles,
        "background_vehicle_motion_pass": vehicle_motion_pass,
        "no_cross_lane_vehicle_pose_pass": vehicle_pose_pass,
        "background_static_objects_inside_lane_center": static_inside_lane_center,
        "background_static_lane_safety_pass": static_pass,
    }


def _valid_rows(
    elem: Any,
    threshold: float,
) -> Iterable[Tuple[int, np.ndarray]]:
    states = np.asarray(elem.states)
    masks = np.asarray(elem.mask).reshape(-1)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    for index in range(min(len(states), len(masks))):
        if _active(masks[index], threshold):
            yield index, np.asarray(states[index], dtype=np.float32)


def _active(value: Any, threshold: float = 0.3) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        return float(value) >= threshold
    except Exception:
        return False


def _score(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _deactivate(elem: Any, index: int) -> None:
    mask = np.asarray(elem.mask).reshape(-1)
    if 0 <= index < len(mask):
        mask[index] = False if mask.dtype == np.bool_ else 0.0


def _ego_speed(vector: Any) -> float:
    state = np.asarray(vector.ego.states, dtype=np.float32).reshape(-1)
    if state.size > EgoIndex.VELOCITY_Y:
        return float(
            math.hypot(
                float(state[EgoIndex.VELOCITY_X]),
                float(state[EgoIndex.VELOCITY_Y]),
            )
        )
    return float(state[0]) if state.size else 0.0


def _ego_aabb() -> Tuple[float, float, float, float]:
    return (0.0, 0.0, 2.7, 1.35)


def _state_aabb(
    state: np.ndarray,
    margin: float = 0.0,
) -> Tuple[float, float, float, float]:
    state = np.asarray(state).reshape(-1)
    heading = float(state[2])
    width = max(float(state[3]), 0.5)
    length = max(float(state[4]), 0.5)
    half_x = (
        abs(math.cos(heading)) * length / 2.0
        + abs(math.sin(heading)) * width / 2.0
        + margin
    )
    half_y = (
        abs(math.sin(heading)) * length / 2.0
        + abs(math.cos(heading)) * width / 2.0
        + margin
    )
    return (float(state[0]), float(state[1]), half_x, half_y)


def _overlap(
    first: Tuple[float, float, float, float],
    second: Tuple[float, float, float, float],
) -> bool:
    return bool(
        abs(first[0] - second[0]) < first[2] + second[2]
        and abs(first[1] - second[1]) < first[3] + second[3]
    )


def _wrap(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


__all__ = [
    "DEFAULT_TRAFFIC_REALISM_POLICY",
    "TrafficRealismPolicy",
    "align_heading_to_lane",
    "estimate_reveal_time",
    "evaluate_traffic_realism",
    "heading_error_mod_pi",
    "infer_occlusion_mode",
    "nearest_lane_tangent",
    "project_agent_state",
    "sanitize_generated_background",
    "state_speed",
]

# Import the mature move-then-delete constructor only after the standalone
# utilities above are defined.  This keeps the existing hazard-clearance module
# intact and avoids a generation-module import cycle.
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_clearance_ops import (
    HazardClearancePrimitiveOps,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.primitive_ops import (
    DEFAULT_EGO_LENGTH_M,
    DEFAULT_EGO_WIDTH_M,
    SLEDGEBOARD_FRAME0_OFFSET_S,
)


class TrafficRealismHazardClearancePrimitiveOps(HazardClearancePrimitiveOps):
    """Traffic-realistic occlusion planner plus inherited blocker clearance."""

    def _plan_occluded_actor_layout(
        self,
        scene: Any,
        ctx: Any,
        occluder_spec: Any,
        occluder_index: int,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        actor_elem = self._get_ctx_elem(scene, ctx)
        actor_state = np.asarray(
            actor_elem.states[ctx.actor_index],
            dtype=np.float32,
        )

        lane_y = float(params.get("lane_center_y_m", 0.0))
        lane_width = float(ctx.spec.road_layer.lane_width_m)
        lane_half_width = 0.5 * lane_width
        ctx.anchor["lane_y"] = lane_y
        ctx.anchor["y"] = lane_y
        ctx.extra["conflict_lane_y"] = lane_y

        direction = params.get(
            "direction",
            ctx.spec.interaction_layer.conflict_direction,
        )
        side_sign = self._crossing_direction_to_side_sign(direction)
        risk = ctx.spec.risk_layer
        debug = dict(getattr(ctx.spec, "debug", {}) or {})

        requested_mode = params.get(
            "occlusion_mode",
            debug.get("occlusion_mode"),
        )
        if requested_mode in {
            "adjacent_lane_dynamic",
            "roadside_parked",
            "roadside_static",
        }:
            mode = str(requested_mode)
        elif occluder_spec.elem_name == "static_objects":
            mode = "roadside_static"
        else:
            # Executable default for vehicle/bicycle occlusion: nearby traffic
            # in the immediately adjacent lane.  This intentionally overrides
            # the natural-language carrier's word "parked" when the matrix
            # explicitly requests a vehicle/bicycle occluder.
            mode = "adjacent_lane_dynamic"

        debug["occlusion_mode"] = mode
        debug["traffic_realism_contract"] = (
            "adjacent_dynamic_or_roadside_parallel_v1"
        )
        ctx.spec.debug = debug

        frame0_offset_s = float(
            params.get(
                "frame0_time_offset_s",
                SLEDGEBOARD_FRAME0_OFFSET_S,
            )
        )
        compensate_frame0 = bool(
            params.get("compensate_frame0_offset", True)
        )

        occ_width = self._positive_float(
            params.get("occluder_width_m", debug.get("occluder_width_m")),
            default=float(occluder_spec.width),
            floor=0.35,
        )
        occ_length = self._positive_float(
            params.get("occluder_length_m", debug.get("occluder_length_m")),
            default=float(occluder_spec.length),
            floor=0.50,
        )

        actor_speed = max(
            0.1,
            float(
                params.get(
                    "target_actor_speed_mps",
                    risk.target_actor_speed_mps,
                )
            ),
        )
        actor_heading = float(
            params.get("actor_heading", -side_sign * math.pi / 2.0)
        )
        actor_width = float(max(actor_state[AgentIndex.WIDTH], 0.75))
        actor_length = float(max(actor_state[AgentIndex.LENGTH], 0.75))

        ttc_low, ttc_high = self._normalized_positive_range(
            getattr(risk, "ttc_range_s", (2.0, 3.0)),
            default=(2.0, 3.0),
            floor=0.2,
        )
        ttc_mid = 0.5 * (ttc_low + ttc_high)
        ttc_candidates = self._unique_floats(
            [
                ttc_mid,
                0.75 * ttc_mid + 0.25 * ttc_high,
                ttc_high,
                0.75 * ttc_mid + 0.25 * ttc_low,
                ttc_low,
            ]
        )

        input_ego_speed = float(
            ctx.anchor.get("ego_speed", self._estimate_ego_speed(scene))
        )
        min_ego_speed = float(params.get("min_ego_speed_mps", 2.5))
        max_ego_speed = float(params.get("max_ego_speed_mps", 15.0))
        risk_min_x, risk_max_x = self._normalized_positive_range(
            getattr(
                risk,
                "longitudinal_distance_range_m",
                (10.0, 18.0),
            ),
            default=(10.0, 18.0),
            floor=1.0,
        )

        ego_front_x = DEFAULT_EGO_LENGTH_M / 2.0
        min_actor_x = (
            ego_front_x
            + 1.0
            + max(occ_length, occ_width)
            + 0.8
        )
        protected_refs = {
            (str(ctx.actor_elem_name), int(ctx.actor_index)),
            (str(occluder_spec.elem_name), int(occluder_index)),
        }
        ego_box = self._make_aabb_from_values(
            0.0,
            0.0,
            0.0,
            DEFAULT_EGO_WIDTH_M,
            DEFAULT_EGO_LENGTH_M,
            margin=0.30,
        )

        x_offsets = [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5]
        y_offsets = [0.0, 0.15, -0.15, 0.30, -0.30, 0.50, -0.50]
        timing_candidate_count = 0
        hard_candidate_count = 0

        for target_ttc in ttc_candidates:
            # Pedestrian reaches the near boundary of the ego lane at the
            # requested interaction time.
            actor_display_y = float(
                lane_y
                + side_sign
                * (lane_half_width + actor_speed * target_ttc)
            )
            actor_arrival_time = (
                max(
                    abs(actor_display_y - lane_y) - lane_half_width,
                    0.0,
                )
                / actor_speed
            )
            if actor_arrival_time <= 1e-4:
                continue

            preferred_x = float(input_ego_speed * actor_arrival_time)
            lower_x = max(float(min_actor_x), 4.0)
            raw_x_candidates = [
                preferred_x,
                0.5 * (risk_min_x + risk_max_x),
                risk_min_x,
                risk_max_x,
                lower_x,
                lower_x + 4.0,
                lower_x + 8.0,
                18.0,
                24.0,
                30.0,
            ]
            actor_x_candidates = []
            for value in raw_x_candidates:
                x = float(np.clip(value, lower_x, 31.0))
                ego_speed = x / actor_arrival_time
                if not (
                    min_ego_speed <= ego_speed <= max_ego_speed
                ):
                    continue
                if not any(
                    abs(x - seen) < 1e-5
                    for seen in actor_x_candidates
                ):
                    actor_x_candidates.append(x)
            actor_x_candidates.sort(
                key=lambda value: abs(value - preferred_x)
            )

            for actor_display_x in actor_x_candidates:
                timing_candidate_count += 1
                synchronized_ego_speed = float(
                    actor_display_x / actor_arrival_time
                )

                actor_raw_x, actor_raw_y = self._maybe_display_to_raw(
                    actor_display_x,
                    actor_display_y,
                    actor_heading,
                    actor_speed,
                    frame0_offset_s,
                    compensate_frame0,
                )
                actor_raw = {
                    "x": float(actor_raw_x),
                    "y": float(actor_raw_y),
                    "heading": float(actor_heading),
                    "width": float(actor_width),
                    "length": float(actor_length),
                    "velocity": float(actor_speed),
                }
                actor_display = {
                    "x": float(actor_display_x),
                    "y": float(actor_display_y),
                    "heading": float(actor_heading),
                    "width": float(actor_width),
                    "length": float(actor_length),
                    "velocity": float(actor_speed),
                }

                raw_actor_box = self._make_aabb_from_values(
                    actor_raw_x,
                    actor_raw_y,
                    actor_heading,
                    actor_width,
                    actor_length,
                    margin=0.20,
                )
                display_actor_box = self._make_aabb_from_values(
                    actor_display_x,
                    actor_display_y,
                    actor_heading,
                    actor_width,
                    actor_length,
                    margin=0.20,
                )
                if self._aabb_overlap(display_actor_box, ego_box):
                    continue

                if mode == "adjacent_lane_dynamic":
                    target_abs_y = lane_width
                else:
                    # Roadside objects sit just outside the ego-lane edge.
                    target_abs_y = (
                        lane_half_width + 0.5 * occ_width + 0.35
                    )

                for y_offset in y_offsets:
                    occ_display_y = float(
                        lane_y
                        + side_sign
                        * max(
                            target_abs_y + y_offset,
                            lane_half_width + 0.2,
                        )
                    )
                    denominator = actor_display_y - lane_y
                    if abs(denominator) < 1e-4:
                        continue
                    los_ratio = (
                        (occ_display_y - lane_y) / denominator
                    )
                    if not 0.20 <= los_ratio <= 0.90:
                        continue
                    base_x = los_ratio * actor_display_x

                    for x_offset in x_offsets:
                        occ_display_x = float(base_x + x_offset)
                        if not (
                            ego_front_x + 0.5
                            < occ_display_x
                            < actor_display_x - 0.5
                        ):
                            continue

                        tangent = nearest_lane_tangent(
                            scene,
                            occ_display_x,
                            occ_display_y,
                        )
                        lane_heading = (
                            tangent[1] if tangent is not None else 0.0
                        )
                        # The occluder is always parallel to local road/lane
                        # direction, never a global fixed heading.
                        occ_heading = align_heading_to_lane(
                            0.0,
                            lane_heading,
                        )

                        if mode == "adjacent_lane_dynamic":
                            if occluder_spec.name == "bicycle":
                                occ_speed = float(
                                    np.clip(
                                        0.45 * synchronized_ego_speed,
                                        1.5,
                                        5.0,
                                    )
                                )
                            else:
                                occ_speed = float(
                                    np.clip(
                                        0.72 * synchronized_ego_speed,
                                        2.5,
                                        10.0,
                                    )
                                )
                        else:
                            occ_speed = 0.0

                        occ_raw_x, occ_raw_y = self._maybe_display_to_raw(
                            occ_display_x,
                            occ_display_y,
                            occ_heading,
                            occ_speed,
                            frame0_offset_s,
                            compensate_frame0,
                        )
                        occ_raw = {
                            "x": float(occ_raw_x),
                            "y": float(occ_raw_y),
                            "heading": float(occ_heading),
                            "width": float(occ_width),
                            "length": float(occ_length),
                            "velocity": float(occ_speed),
                        }
                        occ_display = {
                            "x": float(occ_display_x),
                            "y": float(occ_display_y),
                            "heading": float(occ_heading),
                            "width": float(occ_width),
                            "length": float(occ_length),
                            "velocity": float(occ_speed),
                        }

                        raw_occ_box = self._make_aabb_from_values(
                            occ_raw_x,
                            occ_raw_y,
                            occ_heading,
                            occ_width,
                            occ_length,
                            margin=0.30,
                        )
                        display_occ_box = self._make_aabb_from_values(
                            occ_display_x,
                            occ_display_y,
                            occ_heading,
                            occ_width,
                            occ_length,
                            margin=0.30,
                        )
                        if (
                            self._aabb_overlap(raw_occ_box, raw_actor_box)
                            or self._aabb_overlap(
                                display_occ_box,
                                display_actor_box,
                            )
                            or self._aabb_overlap(display_occ_box, ego_box)
                        ):
                            continue

                        lateral_half_extent = self._half_extent_y(
                            occ_length,
                            occ_width,
                            occ_heading,
                        )
                        edge_gap = (
                            abs(occ_display_y - lane_y)
                            - lateral_half_extent
                            - lane_half_width
                        )
                        if mode == "adjacent_lane_dynamic":
                            if (
                                edge_gap
                                < DEFAULT_TRAFFIC_REALISM_POLICY.min_lane_edge_clearance_m
                            ):
                                continue
                            if (
                                abs(
                                    abs(occ_display_y - lane_y)
                                    - lane_width
                                )
                                > DEFAULT_TRAFFIC_REALISM_POLICY.adjacent_lane_center_tolerance_m
                            ):
                                continue
                        else:
                            if not (
                                DEFAULT_TRAFFIC_REALISM_POLICY.roadside_min_edge_gap_m
                                <= edge_gap
                                <= DEFAULT_TRAFFIC_REALISM_POLICY.roadside_max_edge_gap_m
                            ):
                                continue

                        # The pedestrian must be physically on the far side of
                        # the occluder, close enough to emerge from its edge.
                        far_side_gap = (
                            abs(actor_display_y - lane_y)
                            - abs(occ_display_y - lane_y)
                        )
                        if not (
                            DEFAULT_TRAFFIC_REALISM_POLICY.emergence_min_far_side_gap_m
                            <= far_side_gap
                            <= DEFAULT_TRAFFIC_REALISM_POLICY.emergence_max_far_side_gap_m
                        ):
                            continue
                        if (
                            abs(actor_display_x - occ_display_x)
                            > DEFAULT_TRAFFIC_REALISM_POLICY.emergence_max_longitudinal_gap_m
                        ):
                            continue

                        occ_los_state = np.asarray(
                            [
                                occ_display_x,
                                occ_display_y,
                                occ_heading,
                                occ_width,
                                occ_length,
                                occ_speed,
                            ],
                            dtype=np.float32,
                        )
                        pedestrian_los_state = np.asarray(
                            [
                                actor_display_x,
                                actor_display_y,
                                actor_heading,
                                actor_width,
                                actor_length,
                                actor_speed,
                            ],
                            dtype=np.float32,
                        )
                        if not line_of_sight_intersects_box(
                            (0.0, 0.0),
                            (actor_display_x, actor_display_y),
                            occ_los_state,
                            margin=0.25,
                        ):
                            continue

                        reveal_time = estimate_reveal_time(
                            pedestrian_los_state,
                            occ_los_state,
                            ego_speed_mps=synchronized_ego_speed,
                        )
                        if reveal_time is None:
                            continue

                        hard_candidate_count += 1
                        reserved_region = self._union_boxes(
                            [raw_actor_box, raw_occ_box],
                            margin=self.HAZARD_REGION_MARGIN_M,
                        )
                        clearance_edits = self._make_background_room(
                            scene=scene,
                            ctx=ctx,
                            reserved_region=reserved_region,
                            protected_refs=protected_refs,
                            side_sign=side_sign,
                            lane_y=lane_y,
                        )
                        residual = self._background_blockers(
                            scene,
                            reserved_region,
                            protected_refs=protected_refs,
                        )
                        if residual:
                            raise RuntimeError(
                                "background clearance policy left residual "
                                f"blockers: {residual}"
                            )

                        self._set_ego_longitudinal_speed(
                            scene,
                            synchronized_ego_speed,
                        )
                        ctx.anchor.update(
                            {
                                "x": float(actor_display_x),
                                "y": float(lane_y),
                                "lane_y": float(lane_y),
                                "ego_speed": float(synchronized_ego_speed),
                            }
                        )
                        ctx.extra["conflict_lane_y"] = float(lane_y)
                        ctx.extra["timing_solver"] = {
                            "definition": (
                                "pedestrian_lane_entry_equals_ego_arrival"
                            ),
                            "target_ttc_range_s": [
                                float(ttc_low),
                                float(ttc_high),
                            ],
                            "selected_target_ttc_s": float(target_ttc),
                            "pedestrian_lane_entry_time_s": float(
                                actor_arrival_time
                            ),
                            "ego_arrival_time_s": float(
                                actor_display_x / synchronized_ego_speed
                            ),
                            "arrival_time_error_s": 0.0,
                        }
                        ctx.extra["traffic_realism_target"] = {
                            "occlusion_mode": mode,
                            "occluder_heading_rad": float(occ_heading),
                            "occluder_speed_mps": float(occ_speed),
                            "reveal_time_s": float(reveal_time),
                        }
                        ctx.notes.append(
                            "traffic_realism_occlusion_v1: "
                            f"mode={mode}, reveal={reveal_time:.2f}s, "
                            f"occ_heading={occ_heading:.2f}, "
                            f"occ_speed={occ_speed:.2f}m/s"
                        )

                        return {
                            "actor_raw": actor_raw,
                            "actor_display": actor_display,
                            "occluder_raw": occ_raw,
                            "occluder_display": occ_display,
                            "occlusion_mode": mode,
                            "frame0_time_offset_s": float(frame0_offset_s),
                            "compensate_frame0_offset": bool(
                                compensate_frame0
                            ),
                            "lane_center_y": float(lane_y),
                            "input_ego_speed_mps": float(input_ego_speed),
                            "synchronized_ego_speed_mps": float(
                                synchronized_ego_speed
                            ),
                            "pedestrian_lane_entry_time_s": float(
                                actor_arrival_time
                            ),
                            "ego_arrival_time_s": float(
                                actor_display_x / synchronized_ego_speed
                            ),
                            "arrival_time_error_s": 0.0,
                            "target_interaction_ttc_s": float(target_ttc),
                            "target_ttc_range_s": [
                                float(ttc_low),
                                float(ttc_high),
                            ],
                            "occluder_lane_boundary_gap_m": float(edge_gap),
                            "occluder_width_m": float(occ_width),
                            "occluder_length_m": float(occ_length),
                            "occluder_heading_rad": float(occ_heading),
                            "occluder_speed_mps": float(occ_speed),
                            "reveal_time_s": float(reveal_time),
                            "background_clearance_edits": clearance_edits,
                            "background_clearance_edit_count": len(
                                clearance_edits
                            ),
                            "background_removal_count": sum(
                                1
                                for row in clearance_edits
                                if row.get("operation") == "delete"
                            ),
                            "hazard_reserved_region": self._box_payload(
                                reserved_region
                            ),
                        }

        raise RuntimeError(
            "no traffic-realistic timing-aware occluder layout found: "
            f"type={occluder_spec.name}, mode={mode}, "
            f"ttc=({ttc_low:.2f},{ttc_high:.2f}), "
            f"timing_candidates={timing_candidate_count}, "
            f"hard_candidates={hard_candidate_count}"
        )


__all__.append("TrafficRealismHazardClearancePrimitiveOps")
