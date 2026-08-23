"""Semantic-slot based missing-parameter completion.

The filler completes controllable traffic-scene parameters from independent
semantic slots. Explicit user values always have priority over inferred priors.

Important explicit-speed rule
-----------------------------
A bare numeric speed must not automatically be assigned to ego.

For a vulnerable-road-user crossing prompt such as:

    "A pedestrian enters the ego lane at 1.2 m/s."

the speed describes the hazard actor and is stored as:

    actor_speed_mps = 1.2
    source = user_input

Only an explicitly ego-scoped speed such as:

    "Ego travels at 8 m/s."

is stored as ``ego_speed_mps``.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Dict, Optional

from sledge.semantic_control.language.event_frame import (
    EventFrame,
)


class MissingInfoFiller:
    """Fill fine-grained parameters from semantic-slot composition."""

    UNCONTROLLABLE_ENVIRONMENT_PARAMS = {
        "weather",
        "time_of_day",
        "road_friction",
        "lighting",
        "visibility_condition",
    }

    EXPLICIT_SOURCES = {
        "user_input",
        "explicit",
        "llm_explicit",
    }

    def fill(
        self,
        spec: Dict[str, Any],
        frame: EventFrame,
    ) -> Dict[str, Any]:
        out = deepcopy(spec)

        params = out.setdefault(
            "parameter_layer",
            {},
        )

        completed: Dict[
            str,
            Dict[str, Any],
        ] = {
            key: value
            for key, value
            in dict(
                params.get(
                    "completed",
                    {},
                )
            ).items()
            if key
            not in (
                self
                .UNCONTROLLABLE_ENVIRONMENT_PARAMS
            )
        }

        slots = (
            out.get(
                "semantic_slots",
                {},
            )
            or {}
        )

        actor_type = str(
            slots.get(
                "actor_type",
                "unknown",
            )
        )

        explicit_speed_mps = (
            self._extract_speed_mps(
                frame.sentence
            )
        )

        explicit_speed_target = (
            self._infer_explicit_speed_target(
                frame,
                actor_type=actor_type,
            )
            if explicit_speed_mps
            is not None
            else None
        )

        def add(
            name: str,
            value: Any,
            unit: str = "",
            source: str = "inferred_default",
            reason: str = "",
            *,
            overwrite: bool = False,
        ) -> None:
            if (
                name
                in self
                .UNCONTROLLABLE_ENVIRONMENT_PARAMS
            ):
                return

            if (
                overwrite
                or name not in completed
            ):
                completed[name] = (
                    self.slot(
                        value,
                        unit,
                        source,
                        reason,
                    )
                )

        # --------------------------------------------------------------
        # Explicit speed binding
        # --------------------------------------------------------------
        if explicit_speed_mps is not None:
            if (
                explicit_speed_target
                == "ego"
            ):
                add(
                    "ego_speed_mps",
                    explicit_speed_mps,
                    "m/s",
                    "user_input",
                    (
                        "explicit speed is "
                        "linguistically scoped to ego"
                    ),
                    overwrite=True,
                )

            else:
                # Actor is the safe default for an explicitly stated speed in
                # a hazard-actor sentence, especially pedestrian/cyclist
                # crossing descriptions.
                add(
                    "actor_speed_mps",
                    explicit_speed_mps,
                    "m/s",
                    "user_input",
                    (
                        "explicit speed is "
                        "linguistically scoped to "
                        "the hazard actor"
                    ),
                    overwrite=True,
                )

        # --------------------------------------------------------------
        # Actor speed priors
        #
        # These are used only when no explicit actor speed already exists.
        # --------------------------------------------------------------
        if actor_type == "pedestrian":
            add(
                "actor_speed_mps",
                [1.0, 2.2],
                "m/s",
                reason=(
                    "human-on-foot speed prior"
                ),
            )

        elif actor_type == "cyclist":
            add(
                "actor_speed_mps",
                [3.0, 7.0],
                "m/s",
                reason=(
                    "cyclist speed prior"
                ),
            )

        elif actor_type == "vehicle":
            add(
                "actor_speed_mps",
                [7.0, 22.0],
                "m/s",
                reason=(
                    "vehicle speed prior"
                ),
            )

        elif actor_type == "traffic_object":
            add(
                "obstacle_speed_mps",
                0.0,
                "m/s",
                reason=(
                    "static traffic object prior"
                ),
            )

        motion = str(
            slots.get(
                "motion_geometry",
                "unknown",
            )
        )

        anchor = str(
            slots.get(
                "anchor_region",
                "unknown",
            )
        )

        visibility = str(
            slots.get(
                "visibility",
                "visible",
            )
        )

        # --------------------------------------------------------------
        # Lateral crossing
        # --------------------------------------------------------------
        if motion == "lateral":
            if (
                "ego_speed_mps"
                not in completed
            ):
                add(
                    "ego_speed_mps",
                    [5.0, 12.0],
                    "m/s",
                    reason=(
                        "lateral crossing usually "
                        "evaluated in urban/lane "
                        "context"
                    ),
                )

            distance = (
                [8.0, 25.0]
                if anchor == "front"
                else [10.0, 30.0]
            )

            if (
                visibility
                == "occluded"
            ):
                distance = [
                    5.0,
                    18.0,
                ]

            add(
                "initial_distance_m",
                distance,
                "m",
                reason=(
                    "distance from ego to "
                    "lateral conflict region"
                ),
            )

            # For an occluded-emergence scene there is only one semantic
            # direction. Absolute left/right direction is derived later after
            # occluder side has been selected.
            if (
                visibility
                == "occluded"
                or bool(
                    slots.get(
                        "occlusion_enabled",
                        False,
                    )
                )
            ):
                add(
                    "crossing_direction",
                    "occluder_to_ego_path",
                    source=(
                        "derived_constraint"
                    ),
                    reason=(
                        "occluded actor moves "
                        "from its occluder toward "
                        "the ego path"
                    ),
                    overwrite=True,
                )

            else:
                add(
                    "crossing_direction",
                    [
                        "left_to_right",
                        "right_to_left",
                    ],
                    source=(
                        "sampled_default"
                    ),
                    reason=(
                        "visible crossing side "
                        "is unspecified"
                    ),
                )

            add(
                "target_path",
                slots.get(
                    "target_path",
                    "ego_lane",
                ),
                reason=(
                    "target path inferred "
                    "from EventFrame slot"
                ),
            )

            add(
                "trigger_condition",
                (
                    "actor_enters_or_crosses_"
                    "ego_path"
                ),
                reason=(
                    "lateral conflict starts "
                    "when the actor enters or "
                    "crosses the ego conflict "
                    "region"
                ),
            )

        # --------------------------------------------------------------
        # Merge / cut-in
        # --------------------------------------------------------------
        elif motion == "merging":
            if (
                "ego_speed_mps"
                not in completed
            ):
                add(
                    "ego_speed_mps",
                    [8.0, 20.0],
                    "m/s",
                    reason=(
                        "merge/cut-in moving "
                        "traffic speed prior"
                    ),
                )

            add(
                "actor_speed_mps",
                [8.0, 22.0],
                "m/s",
                reason=(
                    "neighboring/circulating "
                    "vehicle speed prior"
                ),
                overwrite=False,
            )

            add(
                "initial_lateral_offset_m",
                [3.0, 4.0],
                "m",
                reason=(
                    "one-lane lateral offset "
                    "for merge-like conflicts"
                ),
            )

            if (
                slots.get(
                    "road_topology"
                )
                == "roundabout"
            ):
                add(
                    "entry_gap_m",
                    [4.0, 15.0],
                    "m",
                    reason=(
                        "available gap at "
                        "roundabout entry"
                    ),
                )

                add(
                    "circulating_vehicle_speed_mps",
                    [4.0, 12.0],
                    "m/s",
                    reason=(
                        "vehicle already "
                        "circulating in "
                        "roundabout"
                    ),
                )

                add(
                    "entry_relation",
                    (
                        "circulating_vehicle_"
                        "crosses_entry_path"
                    ),
                    reason=(
                        "roundabout entry "
                        "conflict relation"
                    ),
                )

                add(
                    "trigger_condition",
                    (
                        "ego_attempts_entry_while_"
                        "circulating_vehicle_"
                        "closes_gap"
                    ),
                    reason=(
                        "roundabout entry "
                        "trigger"
                    ),
                    overwrite=True,
                )

            else:
                add(
                    "initial_longitudinal_gap_m",
                    [5.0, 20.0],
                    "m",
                    reason=(
                        "longitudinal gap for "
                        "merge/cut-in conflict"
                    ),
                )

                add(
                    "source_side",
                    slots.get(
                        "source_side",
                        [
                            "left",
                            "right",
                        ],
                    ),
                    reason=(
                        "side inferred from "
                        "source relation"
                    ),
                )

                add(
                    "target_lane",
                    "ego_lane",
                    reason=(
                        "merge target is "
                        "ego lane"
                    ),
                )

                add(
                    "trigger_condition",
                    (
                        "actor_crosses_lane_"
                        "boundary_into_ego_lane"
                    ),
                    reason=(
                        "cut-in trigger"
                    ),
                )

        # --------------------------------------------------------------
        # Longitudinal
        # --------------------------------------------------------------
        elif motion == "longitudinal":
            if (
                "ego_speed_mps"
                not in completed
            ):
                add(
                    "ego_speed_mps",
                    [8.0, 22.0],
                    "m/s",
                    reason=(
                        "following-scenario "
                        "speed prior"
                    ),
                )

            add(
                "lead_speed_mps",
                [5.0, 20.0],
                "m/s",
                reason=(
                    "lead vehicle initial "
                    "speed"
                ),
            )

            hard = (
                out.get(
                    "motion_layer",
                    {},
                )
                .get(
                    "hazard_event_type"
                )
                == "hard_stop_ahead"
            )

            add(
                "lead_deceleration_mps2",
                (
                    [4.0, 9.0]
                    if hard
                    else [3.0, 7.0]
                ),
                "m/s^2",
                reason=(
                    "braking-strength prior"
                ),
            )

            add(
                "initial_headway_m",
                [8.0, 30.0],
                "m",
                reason=(
                    "short to moderate "
                    "following gap"
                ),
            )

            add(
                "trigger_condition",
                "lead_vehicle_decelerates",
                reason=(
                    "braking trigger"
                ),
            )

        # --------------------------------------------------------------
        # Oncoming
        # --------------------------------------------------------------
        elif motion == "oncoming":
            if (
                "ego_speed_mps"
                not in completed
            ):
                add(
                    "ego_speed_mps",
                    [3.0, 10.0],
                    "m/s",
                    reason=(
                        "turning ego vehicle "
                        "speed range"
                    ),
                )

            add(
                "oncoming_speed_mps",
                [8.0, 22.0],
                "m/s",
                reason=(
                    "opposing through "
                    "vehicle speed range"
                ),
            )

            add(
                "initial_oncoming_distance_m",
                [15.0, 45.0],
                "m",
                reason=(
                    "gap range for oncoming "
                    "conflict"
                ),
            )

            add(
                "trigger_condition",
                (
                    "ego_turn_path_intersects_"
                    "oncoming_path"
                ),
                reason=(
                    "oncoming/left-turn "
                    "conflict trigger"
                ),
            )

        # --------------------------------------------------------------
        # Static obstacle
        # --------------------------------------------------------------
        elif motion == "static":
            if (
                "ego_speed_mps"
                not in completed
            ):
                add(
                    "ego_speed_mps",
                    [5.0, 15.0],
                    "m/s",
                    reason=(
                        "approach speed for "
                        "static obstacle"
                    ),
                )

            add(
                "obstacle_distance_m",
                [10.0, 35.0],
                "m",
                reason=(
                    "distance to static "
                    "obstacle"
                ),
            )

            add(
                "obstacle_lateral_offset_m",
                [0.0, 1.0],
                "m",
                reason=(
                    "object occupies ego "
                    "lane center or near-center"
                ),
            )

            add(
                "trigger_condition",
                (
                    "ego_approaches_static_"
                    "obstacle_in_lane"
                ),
                reason=(
                    "static obstacle trigger"
                ),
            )

        else:
            if (
                "ego_speed_mps"
                not in completed
            ):
                add(
                    "ego_speed_mps",
                    [5.0, 15.0],
                    "m/s",
                    reason=(
                        "generic driving "
                        "speed range"
                    ),
                )

            add(
                "initial_distance_m",
                [10.0, 30.0],
                "m",
                reason=(
                    "generic interaction "
                    "distance"
                ),
            )

            add(
                "trigger_condition",
                (
                    "hazard_event_becomes_"
                    "relevant_to_ego_path"
                ),
                reason=(
                    "generic trigger"
                ),
            )

        # --------------------------------------------------------------
        # Road / maneuver
        # --------------------------------------------------------------
        road = str(
            slots.get(
                "road_topology",
                "unknown",
            )
        )

        if road == "intersection":
            add(
                "intersection_type",
                "four_way_or_unprotected",
                reason=(
                    "intersection context "
                    "inferred from semantic "
                    "slots"
                ),
            )

        elif road == "roundabout":
            add(
                "roundabout_entry_layout",
                (
                    "single_entry_with_"
                    "circulating_lane"
                ),
                reason=(
                    "roundabout topology "
                    "inferred from semantic "
                    "slots"
                ),
            )

        if (
            slots.get(
                "ego_maneuver"
            )
            == "left_turn"
        ):
            add(
                "ego_maneuver",
                "left_turn",
                reason=(
                    "ego maneuver inferred "
                    "or explicitly stated"
                ),
            )

            add(
                "ego_turn_radius_m",
                [6.0, 14.0],
                "m",
                reason=(
                    "left-turn geometry prior"
                ),
            )

        # --------------------------------------------------------------
        # Occlusion
        # --------------------------------------------------------------
        if (
            visibility == "occluded"
            or bool(
                slots.get(
                    "occlusion_enabled",
                    False,
                )
            )
        ):
            add(
                "occlusion_enabled",
                True,
                source=(
                    "inferred_default"
                ),
                reason=(
                    "occlusion evidence "
                    "in prompt"
                ),
            )

            add(
                "occluder_type",
                slots.get(
                    "occluder_type",
                    "vehicle",
                ),
                source=(
                    "inferred_default"
                ),
                reason=(
                    "occluder type from "
                    "prompt/EventFrame"
                ),
            )

            add(
                "reveal_distance_m",
                [3.0, 12.0],
                "m",
                reason=(
                    "distance after "
                    "occluded actor becomes "
                    "visible"
                ),
            )

            add(
                "occluder_lateral_offset_m",
                [1.0, 4.0],
                "m",
                reason=(
                    "occluder roadside/"
                    "adjacent-lane offset"
                ),
            )

        # --------------------------------------------------------------
        # Explicit EventFrame parameters must overwrite inferred defaults.
        # --------------------------------------------------------------
        for (
            name,
            parameter,
        ) in frame.completed_parameters.items():
            if (
                name
                in self
                .UNCONTROLLABLE_ENVIRONMENT_PARAMS
            ):
                continue

            source = str(
                parameter.source
                or "unknown"
            )

            if (
                name not in completed
                or source
                in self.EXPLICIT_SOURCES
            ):
                completed[name] = {
                    "value":
                        parameter.value,
                    "unit":
                        parameter.unit,
                    "source": (
                        "user_input"
                        if source
                        in self.EXPLICIT_SOURCES
                        else source
                    ),
                    "reason": (
                        parameter.reason
                        or (
                            "provided by "
                            "EventFrame parser"
                        )
                    ),
                }

        params[
            "completed"
        ] = completed

        params[
            "required_missing"
        ] = sorted(
            {
                parameter
                for parameter
                in (
                    params.get(
                        "required_missing",
                        [],
                    )
                    + frame
                    .missing_information
                    .required
                )
                if parameter
                not in self
                .UNCONTROLLABLE_ENVIRONMENT_PARAMS
            }
        )

        params[
            "defaultable_missing"
        ] = sorted(
            {
                parameter
                for parameter
                in (
                    params.get(
                        "defaultable_missing",
                        [],
                    )
                    + frame
                    .missing_information
                    .defaultable
                )
                if parameter
                not in self
                .UNCONTROLLABLE_ENVIRONMENT_PARAMS
            }
        )

        params[
            "distributional_defaults"
        ] = {
            key: value
            for key, value
            in {
                **params.get(
                    "distributional_defaults",
                    {},
                ),
                **frame
                .missing_information
                .distributional,
            }.items()
            if key
            not in self
            .UNCONTROLLABLE_ENVIRONMENT_PARAMS
        }

        params[
            "completion_policy"
        ] = (
            "semantic_slot_"
            "conditioned_completion"
        )

        out[
            "parameter_layer"
        ] = params

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

    @classmethod
    def _infer_explicit_speed_target(
        cls,
        frame: EventFrame,
        *,
        actor_type: str,
    ) -> str:
        """Infer whether a single explicit speed belongs to ego or actor."""

        sentence = frame.sentence.lower()

        # Explicit ego phrases have the highest priority.
        ego_patterns = [
            r"\bego\b.{0,80}\bat\s+\d",
            r"\bego\b.{0,80}\btravels?\b.{0,40}\d",
            r"\bego\b.{0,80}\bmoves?\b.{0,40}\d",
            r"\bego vehicle\b.{0,80}\d",
            r"\bego car\b.{0,80}\d",
            r"\bvehicle speed of ego\b",
        ]

        if any(
            re.search(
                pattern,
                sentence,
            )
            for pattern in ego_patterns
        ):
            return "ego"

        # For pedestrian/cyclist scenes, a lone explicit speed is normally the
        # hazard actor speed unless ego is explicitly named.
        if actor_type in {
            "pedestrian",
            "cyclist",
        }:
            return "actor"

        if (
            frame.main_actor.actor_class
            in {
                "human_on_foot",
                "cyclist",
            }
        ):
            return "actor"

        actor_terms = [
            "pedestrian",
            "walker",
            "person",
            "child",
            "girl",
            "boy",
            "schoolboy",
            "schoolgirl",
            "jogger",
            "runner",
            "wheelchair user",
            "cyclist",
            "vehicle ahead",
            "lead vehicle",
            "oncoming vehicle",
        ]

        if any(
            term in sentence
            for term in actor_terms
        ):
            return "actor"

        # For a generic hazard-actor description, actor is still safer than
        # silently applying the speed to ego.
        return "actor"

    @staticmethod
    def _extract_speed_mps(
        sentence: str,
    ) -> Optional[float]:
        """Extract one explicit speed and normalize it to m/s."""

        normalized = sentence.lower()

        match = re.search(
            (
                r"(\d+(?:\.\d+)?)"
                r"\s*"
                r"(m/s|meter per second|"
                r"meters per second)"
            ),
            normalized,
        )

        if match:
            return float(
                match.group(1)
            )

        match = re.search(
            r"(\d+(?:\.\d+)?)\s*mph",
            normalized,
        )

        if match:
            return round(
                float(
                    match.group(1)
                )
                * 0.44704,
                3,
            )

        match = re.search(
            (
                r"(\d+(?:\.\d+)?)"
                r"\s*"
                r"(km/h|kph|kmph)"
            ),
            normalized,
        )

        if match:
            return round(
                float(
                    match.group(1)
                )
                / 3.6,
                3,
            )

        return None