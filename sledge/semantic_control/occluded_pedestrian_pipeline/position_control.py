"""Position-aware occluder placement for the complete experiment runner.

The base editor already solves actor timing, line-of-sight occlusion, overlap,
and ego-corridor clearance. This module wraps only the final occluder layout
selection so a batch can request several relative occluder positions while all
of the original safety checks remain active.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

from sledge.semantic_control.occluded_pedestrian_pipeline.generation.geometry_metrics import (
    line_of_sight_intersects_box,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.primitive_ops import (
    DEFAULT_EGO_LENGTH_M,
    DEFAULT_EGO_WIDTH_M,
    PrimitiveOps,
)


POSITION_BANDS: Dict[str, Tuple[float, float]] = {
    "near_ego": (0.45, 0.60),
    "midway": (0.60, 0.72),
    "near_pedestrian": (0.72, 0.88),
}
POSITION_TARGETS: Dict[str, float] = {
    "near_ego": 0.54,
    "midway": 0.66,
    "near_pedestrian": 0.79,
}
POSITION_ALIASES = {
    "ego_side": "near_ego",
    "middle": "midway",
    "center": "midway",
    "actor_side": "near_pedestrian",
    "pedestrian_side": "near_pedestrian",
    "between_ego_and_actor": "midway",
}

_REQUESTED_POSITION: ContextVar[Optional[str]] = ContextVar(
    "occluded_pedestrian_requested_position",
    default=None,
)
_PATCH_INSTALLED = False
_ORIGINAL_PLAN = None


def normalize_occluder_position(value: Optional[str]) -> str:
    normalized = str(value or "midway").strip().lower().replace("-", "_").replace(" ", "_")
    normalized = POSITION_ALIASES.get(normalized, normalized)
    if normalized not in POSITION_BANDS:
        raise ValueError(
            f"Unsupported occluder_position={value!r}; expected one of {sorted(POSITION_BANDS)}"
        )
    return normalized


def position_band(position: str) -> Tuple[float, float]:
    return POSITION_BANDS[normalize_occluder_position(position)]


def projection_ratio(actor_xy: Tuple[float, float], occluder_xy: Tuple[float, float]) -> float:
    ax, ay = float(actor_xy[0]), float(actor_xy[1])
    ox, oy = float(occluder_xy[0]), float(occluder_xy[1])
    denominator = ax * ax + ay * ay
    if denominator <= 1e-8:
        return -1.0
    return float((ox * ax + oy * ay) / denominator)


def position_matches(position: str, ratio: float, tolerance: float = 1e-4) -> bool:
    low, high = position_band(position)
    return bool(low - tolerance <= float(ratio) <= high + tolerance)


@contextmanager
def requested_occluder_position(position: str) -> Iterator[None]:
    token = _REQUESTED_POSITION.set(normalize_occluder_position(position))
    try:
        yield
    finally:
        _REQUESTED_POSITION.reset(token)


def install_position_aware_layout_patch() -> None:
    """Install the wrapper once for the current Python process."""

    global _PATCH_INSTALLED, _ORIGINAL_PLAN
    if _PATCH_INSTALLED:
        return
    _ORIGINAL_PLAN = PrimitiveOps._plan_occluded_actor_layout

    def _patched_plan(
        self: PrimitiveOps,
        scene: Any,
        ctx: Any,
        occluder_spec: Any,
        occluder_index: int,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        layout = _ORIGINAL_PLAN(
            self,
            scene=scene,
            ctx=ctx,
            occluder_spec=occluder_spec,
            occluder_index=occluder_index,
            params=params,
        )
        requested = _REQUESTED_POSITION.get()
        if requested is None:
            requested = normalize_occluder_position(params.get("occlusion_position", "midway"))
        return _relocate_occluder(
            self,
            scene=scene,
            ctx=ctx,
            occluder_spec=occluder_spec,
            occluder_index=occluder_index,
            params=params,
            baseline_layout=layout,
            position=requested,
        )

    PrimitiveOps._plan_occluded_actor_layout = _patched_plan
    _PATCH_INSTALLED = True


def _candidate_ratios(position: str) -> list[float]:
    low, high = position_band(position)
    target = POSITION_TARGETS[normalize_occluder_position(position)]
    raw = [target, target - 0.03, target + 0.03, target - 0.06, target + 0.06, low, high]
    values = []
    for value in raw:
        clipped = float(np.clip(value, low, high))
        if all(abs(clipped - existing) > 1e-6 for existing in values):
            values.append(clipped)
    return values


def _relocate_occluder(
    ops: PrimitiveOps,
    *,
    scene: Any,
    ctx: Any,
    occluder_spec: Any,
    occluder_index: int,
    params: Dict[str, Any],
    baseline_layout: Dict[str, Any],
    position: str,
) -> Dict[str, Any]:
    position = normalize_occluder_position(position)
    actor_display = dict(baseline_layout["actor_display"])
    actor_raw = dict(baseline_layout["actor_raw"])
    actor_xy = (float(actor_display["x"]), float(actor_display["y"]))
    lane_y = float(ctx.anchor.get("lane_y", 0.0))
    lane_half_width = 0.5 * float(ctx.spec.road_layer.lane_width_m)
    corridor_clearance = float(params.get("ego_corridor_clearance_m", 1.50))
    frame0_offset_s = float(baseline_layout.get("frame0_time_offset_s", 2.1))
    compensate_frame0 = bool(baseline_layout.get("compensate_frame0_offset", True))
    occ_heading = float(baseline_layout["occluder_display"]["heading"])
    occ_speed = float(baseline_layout["occluder_display"].get("velocity", 0.0))
    ego_front_x = DEFAULT_EGO_LENGTH_M / 2.0

    occupied = ops._collect_occupied_boxes(
        scene,
        ignore=[(ctx.actor_elem_name, ctx.actor_index), (occluder_spec.elem_name, occluder_index)],
        include_ego=True,
        use_display_time=False,
    )
    raw_actor_box = ops._make_aabb_from_values(
        actor_raw["x"],
        actor_raw["y"],
        actor_raw["heading"],
        actor_raw["width"],
        actor_raw["length"],
        margin=0.20,
    )
    display_actor_box = ops._make_aabb_from_values(
        actor_display["x"],
        actor_display["y"],
        actor_display["heading"],
        actor_display["width"],
        actor_display["length"],
        margin=0.20,
    )
    display_ego_box = ops._make_aabb_from_values(
        0.0,
        0.0,
        0.0,
        DEFAULT_EGO_WIDTH_M,
        DEFAULT_EGO_LENGTH_M,
        margin=0.30,
    )

    x_offsets = [0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.6, -1.6, 2.0, -2.0]
    y_offsets = [
        0.0, -0.25, 0.25, -0.50, 0.50, -0.80, 0.80,
        -1.10, 1.10, -1.40, 1.40, -1.70, 1.70, -2.00, 2.00,
    ]

    for ratio in _candidate_ratios(position):
        for dx in x_offsets:
            for dy in y_offsets:
                occ_display_x = ratio * actor_xy[0] + dx
                occ_display_y = ratio * actor_xy[1] + dy
                actual_ratio = projection_ratio(actor_xy, (occ_display_x, occ_display_y))
                if not position_matches(position, actual_ratio):
                    continue
                if not (ego_front_x + 0.5 < occ_display_x < actor_xy[0] - 0.5):
                    continue

                occ_display = {
                    "x": float(occ_display_x),
                    "y": float(occ_display_y),
                    "heading": occ_heading,
                    "width": float(occluder_spec.width),
                    "length": float(occluder_spec.length),
                    "velocity": occ_speed,
                }
                if compensate_frame0:
                    raw_x, raw_y = ops._display_to_raw_position(
                        occ_display_x,
                        occ_display_y,
                        occ_heading,
                        occ_speed,
                        frame0_offset_s,
                    )
                else:
                    raw_x, raw_y = occ_display_x, occ_display_y
                occ_raw = dict(occ_display)
                occ_raw["x"], occ_raw["y"] = float(raw_x), float(raw_y)

                raw_occ_box = ops._make_aabb_from_values(
                    occ_raw["x"],
                    occ_raw["y"],
                    occ_heading,
                    occluder_spec.width,
                    occluder_spec.length,
                    margin=0.30,
                )
                display_occ_box = ops._make_aabb_from_values(
                    occ_display_x,
                    occ_display_y,
                    occ_heading,
                    occluder_spec.width,
                    occluder_spec.length,
                    margin=0.30,
                )
                if ops._aabb_overlap(raw_occ_box, raw_actor_box):
                    continue
                if ops._aabb_overlap(display_occ_box, display_actor_box):
                    continue
                if ops._aabb_overlap(display_occ_box, display_ego_box):
                    continue

                lateral_half_extent = ops._half_extent_y(
                    occluder_spec.length,
                    occluder_spec.width,
                    occ_heading,
                )
                lane_boundary_gap = abs(occ_display_y - lane_y) - lateral_half_extent - lane_half_width
                if lane_boundary_gap < corridor_clearance:
                    continue

                occ_state = np.asarray(
                    [
                        occ_display_x,
                        occ_display_y,
                        occ_heading,
                        occluder_spec.width,
                        occluder_spec.length,
                    ],
                    dtype=np.float32,
                )
                if not line_of_sight_intersects_box((0.0, 0.0), actor_xy, occ_state, margin=0.25):
                    continue
                if ops._box_overlaps_any(raw_occ_box, occupied):
                    continue

                result = deepcopy(baseline_layout)
                result["occluder_raw"] = occ_raw
                result["occluder_display"] = occ_display
                result["requested_occluder_position"] = position
                result["occluder_projection_ratio"] = float(actual_ratio)
                result["occluder_position_band"] = list(position_band(position))
                result["occluder_lane_boundary_gap_m"] = float(lane_boundary_gap)
                return result

    baseline_ratio = projection_ratio(
        actor_xy,
        (
            float(baseline_layout["occluder_display"]["x"]),
            float(baseline_layout["occluder_display"]["y"]),
        ),
    )
    raise RuntimeError(
        "Unable to place a collision-free, LOS-valid occluder in requested "
        f"position={position}; baseline_ratio={baseline_ratio:.3f}, actor={actor_xy}"
    )
