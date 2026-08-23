"""Occluder placement with local move-then-delete background clearance.

This module intentionally leaves the existing :mod:`primitive_ops` implementation
untouched.  It subclasses ``PrimitiveOps`` and overrides only the occluded-
pedestrian layout planner.

EDIT_EXISTING contract used here:

* road/lane geometry stays untouched;
* ego, target pedestrian and occluder are semantic controls;
* unrelated vehicles/pedestrians/static objects do not have veto power over a
  semantically valid occluder placement;
* if an unrelated background entity overlaps the reserved hazard region, try a
  small relocation first; if no collision-free relocation exists, remove that
  entity by clearing its mask;
* hard hazard checks (ego-corridor clearance, LOS occlusion, ego/target overlap)
  are never weakened.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    AgentIndex,
    SledgeVectorRaw,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.geometry_metrics import (
    line_of_sight_intersects_box,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.primitive_ops import (
    DEFAULT_EGO_LENGTH_M,
    DEFAULT_EGO_WIDTH_M,
    SLEDGEBOARD_FRAME0_OFFSET_S,
    OccluderSpec,
    PrimitiveOps,
)


AABB = Tuple[float, float, float, float]
EntityRef = Tuple[str, int]


class HazardClearancePrimitiveOps(PrimitiveOps):
    """PrimitiveOps variant that clears nuisance background around a hazard.

    The important difference from the base implementation is the treatment of
    background collision.  The base implementation rejects an otherwise valid
    occluder candidate when ``raw_occ_box`` overlaps any existing background
    object.  Here we first validate the *hard* hazard geometry, then make local
    room for that candidate by moving or deleting only the overlapping nuisance
    entities.
    """

    RELOCATION_X_OFFSETS_M: Sequence[float] = (
        4.0,
        -4.0,
        8.0,
        -8.0,
        12.0,
        -12.0,
        18.0,
        -18.0,
    )
    RELOCATION_Y_OFFSETS_M: Sequence[float] = (3.0, 5.0, 7.0)
    HAZARD_REGION_MARGIN_M: float = 0.80
    RELOCATION_MARGIN_M: float = 0.25
    WORLD_X_LIMIT_M: float = 55.0
    WORLD_Y_LIMIT_M: float = 25.0

    def _plan_occluded_actor_layout(
        self,
        scene: SledgeVectorRaw,
        ctx,
        occluder_spec: OccluderSpec,
        occluder_index: int,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Jointly solve pedestrian, occluder and ego timing.

        The authoritative semantic frame for this occluded-pedestrian baseline
        is the ego-local path frame: ego is at ``(0, 0)`` and the current ego
        path/lane center is ``y = 0``.  Earlier code estimated lane center from
        nearby vehicles; that could drift by +/-1 m and made generation and
        evaluation disagree about corridor clearance.

        Construction order:
          1. choose a target interaction time inside the risk-conditioned TTC
             window;
          2. place the pedestrian so it reaches the ego-lane boundary at that
             time;
          3. search a hard-valid occluder between ego and pedestrian;
          4. move local nuisance background actors, deleting only if relocation
             fails;
          5. set ego speed so ego reaches pedestrian conflict-x at the same time.

        This keeps timing a construction constraint instead of a post-hoc test.
        """

        actor_elem = self._get_ctx_elem(scene, ctx)
        actor_state = np.asarray(actor_elem.states[ctx.actor_index], dtype=np.float32)

        # SLEDGE is ego-centric.  Use the ego path x-axis as the one canonical
        # lane reference for generation, strict validation and B1/B2 metrics.
        lane_y = float(params.get("lane_center_y_m", 0.0))
        ctx.anchor["lane_y"] = lane_y
        ctx.anchor["y"] = lane_y
        ctx.extra["conflict_lane_y"] = lane_y

        risk = ctx.spec.risk_layer
        direction = params.get("direction", ctx.spec.interaction_layer.conflict_direction)
        side_sign = self._crossing_direction_to_side_sign(direction)

        frame0_offset_s = float(
            params.get("frame0_time_offset_s", SLEDGEBOARD_FRAME0_OFFSET_S)
        )
        compensate_frame0 = bool(params.get("compensate_frame0_offset", True))

        # Preserve the nuPlan/SLEDGE category while using sampled physical size
        # for car / truck / bus / barrier geometry.
        debug = dict(getattr(ctx.spec, "debug", {}) or {})
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
        occ_heading = float(
            params.get(
                "occluder_heading",
                debug.get("occluder_heading", occluder_spec.default_heading),
            )
        )
        occ_velocity = float(occluder_spec.velocity)

        lane_half_width = 0.5 * float(ctx.spec.road_layer.lane_width_m)
        corridor_clearance = float(params.get("ego_corridor_clearance_m", 1.50))
        ego_front_x = DEFAULT_EGO_LENGTH_M / 2.0
        ego_margin = float(params.get("ego_clearance_m", 1.0))
        actor_margin = float(params.get("actor_clearance_m", 0.8))
        occ_hx = self._half_extent_x(occ_length, occ_width, occ_heading)
        min_actor_x = ego_front_x + ego_margin + 2.0 * occ_hx + actor_margin

        actor_heading = float(
            params.get("actor_heading", -side_sign * math.pi / 2.0)
        )
        actor_speed = max(
            0.1,
            float(params.get("target_actor_speed_mps", risk.target_actor_speed_mps)),
        )
        actor_width = float(max(actor_state[AgentIndex.WIDTH], 0.75))
        actor_length = float(max(actor_state[AgentIndex.LENGTH], 0.75))

        ttc_low, ttc_high = self._normalized_positive_range(
            getattr(risk, "ttc_range_s", (2.0, 3.0)),
            default=(2.0, 3.0),
            floor=0.2,
        )
        ttc_mid = 0.5 * (ttc_low + ttc_high)
        # Start near the middle, then favor the upper half because larger TTC
        # creates more lateral room for a legal roadside occluder while still
        # remaining inside the declared danger window.
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
            getattr(risk, "longitudinal_distance_range_m", (10.0, 18.0)),
            default=(10.0, 18.0),
            floor=1.0,
        )

        # Favor occluders near the hidden pedestrian.  This is important for
        # corridor-clear layouts: the occluder can stay roadside without pushing
        # the pedestrian arbitrarily far from the lane.
        ratio_candidates = [0.92, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
        x_offsets = [0.0, 0.6, -0.6, 1.2, -1.2, 2.0, -2.0]
        # Positive values move farther away from the ego lane on the chosen side.
        y_offsets = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, -0.25, -0.50]

        protected_refs = {
            (str(ctx.actor_elem_name), int(ctx.actor_index)),
            (str(occluder_spec.elem_name), int(occluder_index)),
        }
        display_ego_box = self._make_aabb_from_values(
            0.0,
            0.0,
            0.0,
            DEFAULT_EGO_WIDTH_M,
            DEFAULT_EGO_LENGTH_M,
            margin=0.30,
        )

        hard_candidate_count = 0
        timing_candidate_count = 0

        for target_ttc in ttc_candidates:
            if target_ttc <= 0.0:
                continue

            # Pedestrian is positioned so that, at the displayed validation
            # frame, it reaches the ego-lane boundary exactly at target_ttc.
            actor_display_y = float(
                lane_y
                + side_sign * (lane_half_width + actor_speed * target_ttc)
            )
            actor_arrival_time = (
                max(abs(actor_display_y - lane_y) - lane_half_width, 0.0)
                / actor_speed
            )
            if actor_arrival_time <= 1e-4:
                continue

            preferred_x = float(input_ego_speed * actor_arrival_time)
            lower_x = max(float(min_actor_x), 4.0)

            # Risk longitudinal range is a preference, not allowed to override
            # hard geometry/timing.  Long buses/trucks sometimes need the actor
            # farther ahead so the whole occluder fits between ego and actor.
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
                35.0,
            ]
            actor_x_candidates = []
            for value in raw_x_candidates:
                x = float(np.clip(value, lower_x, 35.0))
                ego_speed = x / actor_arrival_time
                if ego_speed < min_ego_speed - 1e-6 or ego_speed > max_ego_speed + 1e-6:
                    continue
                if not any(abs(x - existing) < 1e-5 for existing in actor_x_candidates):
                    actor_x_candidates.append(x)

            actor_x_candidates.sort(key=lambda x: abs(x - preferred_x))
            if not actor_x_candidates:
                continue

            for cur_actor_display_x in actor_x_candidates:
                timing_candidate_count += 1
                synchronized_ego_speed = float(
                    cur_actor_display_x / actor_arrival_time
                )

                arx, ary = self._maybe_display_to_raw(
                    cur_actor_display_x,
                    actor_display_y,
                    actor_heading,
                    actor_speed,
                    frame0_offset_s,
                    compensate_frame0,
                )
                cur_actor_raw: Dict[str, float] = {
                    "x": float(arx),
                    "y": float(ary),
                    "heading": float(actor_heading),
                    "width": float(actor_width),
                    "length": float(actor_length),
                    "velocity": float(actor_speed),
                }
                cur_actor_display: Dict[str, float] = {
                    "x": float(cur_actor_display_x),
                    "y": float(actor_display_y),
                    "heading": float(actor_heading),
                    "width": float(actor_width),
                    "length": float(actor_length),
                    "velocity": float(actor_speed),
                }

                raw_actor_box = self._make_aabb_from_values(
                    cur_actor_raw["x"],
                    cur_actor_raw["y"],
                    actor_heading,
                    actor_width,
                    actor_length,
                    margin=0.20,
                )
                display_actor_box = self._make_aabb_from_values(
                    cur_actor_display_x,
                    actor_display_y,
                    actor_heading,
                    actor_width,
                    actor_length,
                    margin=0.20,
                )
                if self._aabb_overlap(display_actor_box, display_ego_box):
                    continue

                for ratio in ratio_candidates:
                    base_x = ratio * cur_actor_display_x
                    base_y = lane_y + ratio * (actor_display_y - lane_y)
                    for dx in x_offsets:
                        for dy in y_offsets:
                            occ_display_x = float(base_x + dx)
                            occ_display_y = float(base_y + side_sign * dy)
                            if not (
                                ego_front_x + 0.5
                                < occ_display_x
                                < cur_actor_display_x - 0.5
                            ):
                                continue

                            orx, ory = self._maybe_display_to_raw(
                                occ_display_x,
                                occ_display_y,
                                occ_heading,
                                occ_velocity,
                                frame0_offset_s,
                                compensate_frame0,
                            )
                            occ_raw: Dict[str, float] = {
                                "x": float(orx),
                                "y": float(ory),
                                "heading": float(occ_heading),
                                "width": float(occ_width),
                                "length": float(occ_length),
                                "velocity": float(occ_velocity),
                            }
                            occ_display: Dict[str, float] = {
                                "x": float(occ_display_x),
                                "y": float(occ_display_y),
                                "heading": float(occ_heading),
                                "width": float(occ_width),
                                "length": float(occ_length),
                                "velocity": float(occ_velocity),
                            }

                            raw_occ_box = self._make_aabb_from_values(
                                occ_raw["x"],
                                occ_raw["y"],
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

                            # Hard semantic geometry. Background occupancy is
                            # handled only after these checks pass.
                            if self._aabb_overlap(raw_occ_box, raw_actor_box):
                                continue
                            if self._aabb_overlap(display_occ_box, display_actor_box):
                                continue
                            if self._aabb_overlap(display_occ_box, display_ego_box):
                                continue

                            occ_inner_edge_gap = (
                                abs(occ_display_y - lane_y)
                                - self._half_extent_y(
                                    occ_length,
                                    occ_width,
                                    occ_heading,
                                )
                                - lane_half_width
                            )
                            if occ_inner_edge_gap < corridor_clearance:
                                continue

                            occ_state_for_los = np.asarray(
                                [
                                    occ_display_x,
                                    occ_display_y,
                                    occ_heading,
                                    occ_width,
                                    occ_length,
                                ],
                                dtype=np.float32,
                            )
                            if not line_of_sight_intersects_box(
                                (0.0, 0.0),
                                (cur_actor_display_x, actor_display_y),
                                occ_state_for_los,
                                margin=0.25,
                            ):
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
                            ctx.anchor["x"] = float(cur_actor_display_x)
                            ctx.anchor["y"] = float(lane_y)
                            ctx.anchor["lane_y"] = float(lane_y)
                            ctx.anchor["ego_speed"] = float(synchronized_ego_speed)
                            ctx.extra["conflict_lane_y"] = float(lane_y)
                            ctx.extra["timing_solver"] = {
                                "definition": "pedestrian_lane_entry_equals_ego_arrival",
                                "target_ttc_range_s": [float(ttc_low), float(ttc_high)],
                                "selected_target_ttc_s": float(target_ttc),
                                "pedestrian_lane_entry_time_s": float(actor_arrival_time),
                                "ego_arrival_time_s": float(
                                    cur_actor_display_x / synchronized_ego_speed
                                ),
                                "arrival_time_error_s": 0.0,
                            }

                            ctx.notes.append(
                                "synchronize_occluded_interaction_v2: "
                                f"target_ttc={target_ttc:.2f}s, "
                                f"ped_lane_entry={actor_arrival_time:.2f}s, "
                                f"ego_speed={input_ego_speed:.2f}->"
                                f"{synchronized_ego_speed:.2f}m/s, "
                                f"lane_y={lane_y:.2f}"
                            )
                            if clearance_edits:
                                ctx.notes.append(
                                    "clear_occluder_reserved_region: "
                                    f"moved_or_deleted={len(clearance_edits)}"
                                )

                            return {
                                "actor_raw": cur_actor_raw,
                                "actor_display": cur_actor_display,
                                "occluder_raw": occ_raw,
                                "occluder_display": occ_display,
                                "frame0_time_offset_s": float(frame0_offset_s),
                                "compensate_frame0_offset": bool(compensate_frame0),
                                "lane_center_y": float(lane_y),
                                "input_ego_speed_mps": float(input_ego_speed),
                                "synchronized_ego_speed_mps": float(
                                    synchronized_ego_speed
                                ),
                                "pedestrian_lane_entry_time_s": float(
                                    actor_arrival_time
                                ),
                                "ego_arrival_time_s": float(
                                    cur_actor_display_x / synchronized_ego_speed
                                ),
                                "arrival_time_error_s": 0.0,
                                "target_interaction_ttc_s": float(target_ttc),
                                "target_ttc_range_s": [
                                    float(ttc_low),
                                    float(ttc_high),
                                ],
                                "ego_corridor_clearance_m": float(
                                    corridor_clearance
                                ),
                                "occluder_lane_boundary_gap_m": float(
                                    occ_inner_edge_gap
                                ),
                                "occluder_width_m": float(occ_width),
                                "occluder_length_m": float(occ_length),
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
            "no hard-valid timing-aware occluder layout found before background handling "
            f"for type={occluder_spec.name}, "
            f"ttc_range=({ttc_low:.2f},{ttc_high:.2f}), "
            f"timing_candidate_count={timing_candidate_count}, "
            f"hard_candidate_count={hard_candidate_count}"
        )

    @staticmethod
    def _normalized_positive_range(
        value: Any,
        *,
        default: Tuple[float, float],
        floor: float,
    ) -> Tuple[float, float]:
        try:
            vals = list(value)
            if len(vals) < 2:
                raise ValueError
            low, high = sorted([float(vals[0]), float(vals[1])])
        except Exception:
            low, high = float(default[0]), float(default[1])
        low = max(float(floor), low)
        high = max(low, high)
        return float(low), float(high)

    @staticmethod
    def _unique_floats(values: Sequence[float]) -> List[float]:
        out: List[float] = []
        for value in values:
            value = float(value)
            if not any(abs(value - seen) < 1e-6 for seen in out):
                out.append(value)
        return out

    def _make_background_room(
        self,
        *,
        scene: SledgeVectorRaw,
        ctx,
        reserved_region: AABB,
        protected_refs: set,
        side_sign: float,
        lane_y: float,
    ) -> List[Dict[str, Any]]:
        """Move local blockers first; delete only when relocation fails."""

        blockers = self._background_blockers(
            scene,
            reserved_region,
            protected_refs=protected_refs,
        )
        edits: List[Dict[str, Any]] = []

        # Sort by distance to the reserved-region center so the most relevant
        # blocker is handled first. Each edit updates the scene immediately.
        rx, ry, _, _ = reserved_region
        blockers.sort(
            key=lambda ref: self._entity_center_distance(scene, ref, rx, ry)
        )

        for elem_name, idx in blockers:
            elem = self._elem_by_name(scene, elem_name)
            if not bool(np.asarray(elem.mask).reshape(-1)[idx]):
                continue
            before = np.asarray(elem.states[idx], dtype=np.float32).copy()
            destination = self._find_background_relocation(
                scene=scene,
                elem_name=elem_name,
                idx=idx,
                source_state=before,
                reserved_region=reserved_region,
                protected_refs=protected_refs,
                side_sign=side_sign,
                lane_y=lane_y,
            )
            if destination is not None:
                elem.states[idx, AgentIndex.X] = float(destination[0])
                elem.states[idx, AgentIndex.Y] = float(destination[1])
                after = np.asarray(elem.states[idx], dtype=np.float32).copy()
                edits.append(
                    {
                        "element": elem_name,
                        "index": int(idx),
                        "operation": "reposition",
                        "before": self._state_payload(before),
                        "after": self._state_payload(after),
                        "displacement_m": float(
                            math.hypot(
                                float(after[AgentIndex.X] - before[AgentIndex.X]),
                                float(after[AgentIndex.Y] - before[AgentIndex.Y]),
                            )
                        ),
                        "reason": "clear_occluder_reserved_region",
                    }
                )
                continue

            # Relocation failed. This is an unrelated nuisance entity, so it is
            # removed rather than being allowed to veto the target hazard.
            elem.mask[idx] = False
            if elem_name == "vehicles" and hasattr(ctx, "removed_vehicle_indices"):
                ctx.removed_vehicle_indices.append(int(idx))
            edits.append(
                {
                    "element": elem_name,
                    "index": int(idx),
                    "operation": "delete",
                    "before": self._state_payload(before),
                    "after": None,
                    "reason": (
                        "no_collision_free_relocation_for_occluder_reserved_region"
                    ),
                }
            )

        if edits:
            existing = list(
                ctx.extra.get("background_clearance_edits", []) or []
            )
            existing.extend(edits)
            ctx.extra["background_clearance_edits"] = existing
            ctx.extra["background_clearance_edit_count"] = len(existing)
            ctx.extra["background_removal_count"] = sum(
                1 for row in existing if row.get("operation") == "delete"
            )
        return edits

    def _background_blockers(
        self,
        scene: SledgeVectorRaw,
        reserved_region: AABB,
        *,
        protected_refs: set,
    ) -> List[EntityRef]:
        blockers: List[EntityRef] = []
        for elem_name in ("vehicles", "pedestrians", "static_objects"):
            elem = self._elem_by_name(scene, elem_name)
            states = np.asarray(elem.states)
            mask = np.asarray(elem.mask).reshape(-1).astype(bool)
            for idx in np.where(mask)[0]:
                ref = (elem_name, int(idx))
                if ref in protected_refs:
                    continue
                state = np.asarray(states[idx], dtype=np.float32)
                box = self._state_box(state, margin=0.20)
                if self._aabb_overlap(reserved_region, box):
                    blockers.append(ref)
        return blockers

    def _find_background_relocation(
        self,
        *,
        scene: SledgeVectorRaw,
        elem_name: str,
        idx: int,
        source_state: np.ndarray,
        reserved_region: AABB,
        protected_refs: set,
        side_sign: float,
        lane_y: float,
    ) -> Optional[Tuple[float, float]]:
        x0 = float(source_state[AgentIndex.X])
        y0 = float(source_state[AgentIndex.Y])
        _, _, _, rhy = reserved_region

        candidates: List[Tuple[float, float]] = []
        # 1) minimal longitudinal motion first.
        for dx in self.RELOCATION_X_OFFSETS_M:
            candidates.append((x0 + float(dx), y0))
        # 2) if needed, move farther away from the ego lane while keeping x.
        outward_sign = side_sign
        if abs(y0 - lane_y) > 0.25:
            outward_sign = 1.0 if y0 >= lane_y else -1.0
        for dy in self.RELOCATION_Y_OFFSETS_M:
            candidates.append((x0, y0 + outward_sign * float(dy)))
        # 3) small combined moves as a final relocation attempt.
        for dx in (4.0, -4.0, 8.0, -8.0):
            for dy in (3.0, 5.0):
                candidates.append(
                    (x0 + dx, y0 + outward_sign * dy)
                )

        for x, y in candidates:
            if abs(x) > self.WORLD_X_LIMIT_M or abs(y) > self.WORLD_Y_LIMIT_M:
                continue
            test_state = np.asarray(source_state, dtype=np.float32).copy()
            test_state[AgentIndex.X] = float(x)
            test_state[AgentIndex.Y] = float(y)
            test_box = self._state_box(
                test_state,
                margin=self.RELOCATION_MARGIN_M,
            )
            if self._aabb_overlap(test_box, reserved_region):
                continue

            # Do not move a nuisance actor into ego or another active entity.
            ego_box = self._make_aabb_from_values(
                0.0,
                0.0,
                0.0,
                DEFAULT_EGO_WIDTH_M,
                DEFAULT_EGO_LENGTH_M,
                margin=0.30,
            )
            if self._aabb_overlap(test_box, ego_box):
                continue

            occupied = self._collect_occupied_boxes(
                scene,
                ignore=[
                    (elem_name, int(idx)),
                    *list(protected_refs),
                ],
                include_ego=False,
                use_display_time=False,
            )
            if self._box_overlaps_any(test_box, occupied):
                continue
            return float(x), float(y)

        return None

    @staticmethod
    def _union_boxes(boxes: Sequence[AABB], *, margin: float = 0.0) -> AABB:
        min_x = min(x - hx for x, _, hx, _ in boxes) - margin
        max_x = max(x + hx for x, _, hx, _ in boxes) + margin
        min_y = min(y - hy for _, y, _, hy in boxes) - margin
        max_y = max(y + hy for _, y, _, hy in boxes) + margin
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        return (
            float(cx),
            float(cy),
            float(0.5 * (max_x - min_x)),
            float(0.5 * (max_y - min_y)),
        )

    @staticmethod
    def _box_payload(box: AABB) -> Dict[str, float]:
        x, y, hx, hy = box
        return {
            "center_x": float(x),
            "center_y": float(y),
            "half_extent_x": float(hx),
            "half_extent_y": float(hy),
            "x_min": float(x - hx),
            "x_max": float(x + hx),
            "y_min": float(y - hy),
            "y_max": float(y + hy),
        }

    @staticmethod
    def _state_payload(state: np.ndarray) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "x": float(state[AgentIndex.X]),
            "y": float(state[AgentIndex.Y]),
        }
        if state.size > AgentIndex.HEADING:
            payload["heading"] = float(state[AgentIndex.HEADING])
        if state.size > AgentIndex.WIDTH:
            payload["width"] = float(state[AgentIndex.WIDTH])
        if state.size > AgentIndex.LENGTH:
            payload["length"] = float(state[AgentIndex.LENGTH])
        if state.size > AgentIndex.VELOCITY:
            payload["velocity"] = float(state[AgentIndex.VELOCITY])
        return payload

    def _state_box(self, state: np.ndarray, *, margin: float = 0.0) -> AABB:
        return self._make_aabb_from_values(
            float(state[AgentIndex.X]),
            float(state[AgentIndex.Y]),
            float(state[AgentIndex.HEADING])
            if state.size > AgentIndex.HEADING
            else 0.0,
            max(
                float(state[AgentIndex.WIDTH])
                if state.size > AgentIndex.WIDTH
                else 0.8,
                0.5,
            ),
            max(
                float(state[AgentIndex.LENGTH])
                if state.size > AgentIndex.LENGTH
                else 0.8,
                0.5,
            ),
            margin=margin,
        )

    def _entity_center_distance(
        self,
        scene: SledgeVectorRaw,
        ref: EntityRef,
        x: float,
        y: float,
    ) -> float:
        elem = self._elem_by_name(scene, ref[0])
        state = np.asarray(elem.states[ref[1]], dtype=np.float32)
        return float(
            math.hypot(
                float(state[AgentIndex.X]) - x,
                float(state[AgentIndex.Y]) - y,
            )
        )

    @staticmethod
    def _positive_float(value: Any, *, default: float, floor: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = float(default)
        if not math.isfinite(parsed):
            parsed = float(default)
        return float(max(parsed, floor))

    def _maybe_display_to_raw(
        self,
        x: float,
        y: float,
        heading: float,
        speed: float,
        time_offset_s: float,
        compensate: bool,
    ) -> Tuple[float, float]:
        if not compensate:
            return float(x), float(y)
        return self._display_to_raw_position(
            float(x),
            float(y),
            float(heading),
            float(speed),
            float(time_offset_s),
        )


__all__ = ["HazardClearancePrimitiveOps"]