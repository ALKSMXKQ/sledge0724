"""Semantic-slot based missing-parameter completion.

This module intentionally does not select a full predefined scenario template and
then fill that template. Instead, it uses the compositional ``semantic_slots``
emitted by ``event_frame_mapper``. Each slot contributes a small set of
parameters or constraints:

- actor_type contributes actor speed priors,
- motion_geometry contributes interaction parameters,
- anchor_region contributes distance priors,
- visibility / occlusion contributes reveal / occluder parameters,
- road_topology and ego_maneuver contribute road- or maneuver-specific priors.

The final parameter layer is the union of these slot-conditioned completions.
This preserves the engineering stability of defaults while avoiding a rigid
``one scenario family -> one template`` design.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional
import re

from sledge.semantic_control.language.event_frame import EventFrame


class MissingInfoFiller:
    """Fill missing fine-grained parameters from semantic slot composition."""

    UNCONTROLLABLE_ENVIRONMENT_PARAMS = {
        "weather",
        "time_of_day",
        "road_friction",
        "lighting",
        "visibility_condition",
    }

    def fill(self, spec: Dict[str, Any], frame: EventFrame) -> Dict[str, Any]:
        out = deepcopy(spec)
        params = out.setdefault("parameter_layer", {})
        completed: Dict[str, Dict[str, Any]] = {
            k: v
            for k, v in dict(params.get("completed", {})).items()
            if k not in self.UNCONTROLLABLE_ENVIRONMENT_PARAMS
        }
        slots = out.get("semantic_slots", {}) or {}
        explicit_speed_mps = self._extract_speed_mps(frame.sentence)

        def add(
            name: str,
            value: Any,
            unit: str = "",
            source: str = "inferred_default",
            reason: str = "",
            *,
            overwrite: bool = False,
        ) -> None:
            if name in self.UNCONTROLLABLE_ENVIRONMENT_PARAMS:
                return
            if overwrite or name not in completed:
                completed[name] = self.slot(value, unit, source, reason)

        # Explicit ego speed overrides all priors.
        if explicit_speed_mps is not None:
            add(
                "ego_speed_mps",
                explicit_speed_mps,
                "m/s",
                "user_input",
                "explicit speed detected in prompt",
                overwrite=True,
            )

        # Actor-type priors.
        actor_type = slots.get("actor_type", "unknown")

        if actor_type == "pedestrian":
            add(
                "actor_speed_mps",
                [1.0, 2.2],
                "m/s",
                reason="human-on-foot speed prior",
            )

        elif actor_type == "cyclist":
            add(
                "actor_speed_mps",
                [3.0, 7.0],
                "m/s",
                reason="cyclist speed prior",
            )

        elif actor_type == "vehicle":
            add(
                "actor_speed_mps",
                [7.0, 22.0],
                "m/s",
                reason="vehicle speed prior",
            )

        elif actor_type == "traffic_object":
            add(
                "obstacle_speed_mps",
                0.0,
                "m/s",
                reason="static traffic object prior",
            )

        # Motion-geometry priors.
        motion = slots.get("motion_geometry", "unknown")
        anchor = slots.get("anchor_region", "unknown")
        visibility = slots.get("visibility", "visible")

        if motion == "lateral":
            if "ego_speed_mps" not in completed:
                add(
                    "ego_speed_mps",
                    [5.0, 12.0],
                    "m/s",
                    reason="lateral crossing usually evaluated in urban/lane context",
                )

            distance = [8.0, 25.0] if anchor == "front" else [10.0, 30.0]

            if visibility == "occluded":
                distance = [5.0, 18.0]

            add(
                "initial_distance_m",
                distance,
                "m",
                reason="distance from ego to lateral conflict region",
            )

            add(
                "crossing_direction",
                ["left_to_right", "right_to_left"],
                source="sampled_default",
                reason="side not specified; sample both crossing directions",
            )

            add(
                "target_path",
                slots.get("target_path", "ego_lane"),
                reason="target path inferred from EventFrame slot",
            )

            add(
                "trigger_condition",
                "actor_enters_ego_lane_ahead",
                reason="lateral conflict starts when actor enters/crosses ego path",
            )

        elif motion == "merging":
            if "ego_speed_mps" not in completed:
                add(
                    "ego_speed_mps",
                    [8.0, 20.0],
                    "m/s",
                    reason="merge/cut-in moving traffic speed prior",
                )

            add(
                "actor_speed_mps",
                [8.0, 22.0],
                "m/s",
                reason="neighboring/circulating vehicle speed prior",
                overwrite=False,
            )

            add(
                "initial_lateral_offset_m",
                [3.0, 4.0],
                "m",
                reason="one-lane lateral offset for merge-like conflicts",
            )

            # Roundabout entry uses gap rather than lateral offset as its main
            # interaction parameter.
            if slots.get("road_topology") == "roundabout":
                add(
                    "entry_gap_m",
                    [4.0, 15.0],
                    "m",
                    reason="available gap at roundabout entry",
                )

                add(
                    "circulating_vehicle_speed_mps",
                    [4.0, 12.0],
                    "m/s",
                    reason="vehicle already circulating in roundabout",
                )

                add(
                    "entry_relation",
                    "circulating_vehicle_crosses_entry_path",
                    reason="roundabout entry conflict relation",
                )

                add(
                    "trigger_condition",
                    "ego_attempts_entry_while_circulating_vehicle_closes_gap",
                    reason="roundabout entry trigger",
                    overwrite=True,
                )

            else:
                add(
                    "initial_longitudinal_gap_m",
                    [5.0, 20.0],
                    "m",
                    reason="longitudinal gap for merge/cut-in conflict",
                )

                add(
                    "source_side",
                    slots.get("source_side", ["left", "right"]),
                    reason="side inferred from source_relation or prompt",
                )

                add(
                    "target_lane",
                    "ego_lane",
                    reason="merge target is ego lane",
                )

                add(
                    "trigger_condition",
                    "actor_crosses_lane_boundary_into_ego_lane",
                    reason="cut-in trigger",
                )

        elif motion == "longitudinal":
            if "ego_speed_mps" not in completed:
                add(
                    "ego_speed_mps",
                    [8.0, 22.0],
                    "m/s",
                    reason="following scenario speed prior",
                )

            add(
                "lead_speed_mps",
                [5.0, 20.0],
                "m/s",
                reason="lead vehicle initial speed",
            )

            hard = out.get("motion_layer", {}).get("hazard_event_type") == "hard_stop_ahead"

            add(
                "lead_deceleration_mps2",
                [4.0, 9.0] if hard else [3.0, 7.0],
                "m/s^2",
                reason="braking strength prior",
            )

            add(
                "initial_headway_m",
                [8.0, 30.0],
                "m",
                reason="short to moderate following gap",
            )

            add(
                "trigger_condition",
                "lead_vehicle_decelerates",
                reason="braking trigger",
            )

        elif motion == "oncoming":
            if "ego_speed_mps" not in completed:
                add(
                    "ego_speed_mps",
                    [3.0, 10.0],
                    "m/s",
                    reason="turning ego vehicle speed range",
                )

            add(
                "oncoming_speed_mps",
                [8.0, 22.0],
                "m/s",
                reason="opposing through vehicle speed range",
            )

            add(
                "initial_oncoming_distance_m",
                [15.0, 45.0],
                "m",
                reason="gap range for oncoming conflict",
            )

            add(
                "trigger_condition",
                "ego_turn_path_intersects_oncoming_path",
                reason="oncoming/left-turn conflict trigger",
            )

        elif motion == "static":
            if "ego_speed_mps" not in completed:
                add(
                    "ego_speed_mps",
                    [5.0, 15.0],
                    "m/s",
                    reason="approach speed for static obstacle",
                )

            add(
                "obstacle_distance_m",
                [10.0, 35.0],
                "m",
                reason="distance to static obstacle",
            )

            add(
                "obstacle_lateral_offset_m",
                [0.0, 1.0],
                "m",
                reason="object occupies ego lane center or near-center",
            )

            add(
                "trigger_condition",
                "ego_approaches_static_obstacle_in_lane",
                reason="static obstacle trigger",
            )

        else:
            if "ego_speed_mps" not in completed:
                add(
                    "ego_speed_mps",
                    [5.0, 15.0],
                    "m/s",
                    reason="generic driving speed range",
                )

            add(
                "initial_distance_m",
                [10.0, 30.0],
                "m",
                reason="generic interaction distance",
            )

            add(
                "trigger_condition",
                "hazard_event_becomes_relevant_to_ego_path",
                reason="generic trigger",
            )

        # Road and maneuver slots add additional requirements without selecting a
        # full scene template.
        road = slots.get("road_topology", "unknown")

        if road == "intersection":
            add(
                "intersection_type",
                "four_way_or_unprotected",
                reason="intersection context inferred from semantic slots",
            )

        elif road == "roundabout":
            add(
                "roundabout_entry_layout",
                "single_entry_with_circulating_lane",
                reason="roundabout topology inferred from semantic slots",
            )

        if slots.get("ego_maneuver") == "left_turn":
            add(
                "ego_maneuver",
                "left_turn",
                reason="ego maneuver inferred or explicitly stated",
            )

            add(
                "ego_turn_radius_m",
                [6.0, 14.0],
                "m",
                reason="left-turn geometry prior",
            )

        # Occlusion slots add reveal parameters.
        if visibility == "occluded" or slots.get("occlusion_enabled"):
            add(
                "occlusion_enabled",
                True,
                source="inferred_default",
                reason="occlusion evidence in prompt",
            )

            add(
                "occluder_type",
                slots.get("occluder_type", "vehicle"),
                source="inferred_default",
                reason="occluder type from prompt or EventFrame",
            )

            add(
                "reveal_distance_m",
                [3.0, 12.0],
                "m",
                reason="distance after occluded actor becomes visible",
            )

            add(
                "occluder_lateral_offset_m",
                [1.0, 4.0],
                "m",
                reason="occluder placed near roadside or adjacent lane",
            )

        # Respect any explicit parameters returned by the parser/LLM, but keep
        # deterministic slot completion as the default source of truth.
        for name, pv in frame.completed_parameters.items():
            if name in self.UNCONTROLLABLE_ENVIRONMENT_PARAMS:
                continue
            if name not in completed:
                completed[name] = {
                    "value": pv.value,
                    "unit": pv.unit,
                    "source": pv.source,
                    "reason": pv.reason or "provided by EventFrame parser",
                }

        params["completed"] = completed
        params["required_missing"] = sorted(
            {
                p
                for p in params.get("required_missing", []) + frame.missing_information.required
                if p not in self.UNCONTROLLABLE_ENVIRONMENT_PARAMS
            }
        )
        params["defaultable_missing"] = sorted(
            {
                p
                for p in params.get("defaultable_missing", []) + frame.missing_information.defaultable
                if p not in self.UNCONTROLLABLE_ENVIRONMENT_PARAMS
            }
        )
        params["distributional_defaults"] = {
            k: v
            for k, v in {
                **params.get("distributional_defaults", {}),
                **frame.missing_information.distributional,
            }.items()
            if k not in self.UNCONTROLLABLE_ENVIRONMENT_PARAMS
        }
        params["completion_policy"] = "semantic_slot_conditioned_completion"

        out["parameter_layer"] = params
        return out

    @staticmethod
    def slot(
        value: Any,
        unit: str = "",
        source: str = "inferred_default",
        reason: str = "",
    ) -> Dict[str, Any]:
        return {
            "value": value,
            "unit": unit,
            "source": source,
            "reason": reason,
        }

    @staticmethod
    def _extract_speed_mps(sentence: str) -> Optional[float]:
        """Best-effort extraction of a single explicit speed from short prompts."""

        s = sentence.lower()

        m = re.search(
            r"(\d+(?:\.\d+)?)\s*(m/s|meter per second|meters per second)",
            s,
        )
        if m:
            return float(m.group(1))

        m = re.search(r"(\d+(?:\.\d+)?)\s*(mph)", s)
        if m:
            return round(float(m.group(1)) * 0.44704, 3)

        m = re.search(r"(\d+(?:\.\d+)?)\s*(km/h|kph|kmph)", s)
        if m:
            return round(float(m.group(1)) / 3.6, 3)

        return None
