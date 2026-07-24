"""Verification and deterministic repair for EventFrame.

The verifier checks the intermediate semantic representation before mapping it
to scene-control slots.  It also contains small deterministic repair rules for
common LLM/fallback parser mistakes.

The repair rules operate on EventFrame fields rather than raw final templates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from sledge.semantic_control.language.event_frame import EventFrame
from sledge.semantic_control.language.narrative_semantics import (
    HazardFocusResolver,
    NarrativeDecomposer,
    candidate_to_event_frame_dict,
)


@dataclass
class VerificationResult:
    passed: bool
    issues: List[str]


def _contains(text: str, terms: List[str]) -> bool:
    text = (text or "").lower()
    return any(t in text for t in terms)


class EventFrameVerifier:
    """Verify and repair EventFrame and mapped spec consistency."""

    def verify_frame(self, frame: EventFrame) -> VerificationResult:
        issues: List[str] = []

        if not frame.sentence:
            issues.append("missing_sentence")

        if frame.main_actor.actor_class in {"", "unknown"}:
            issues.append("missing_main_actor_class")

        if frame.main_event.event_type in {"", "unknown"}:
            issues.append("missing_main_event_type")

        if frame.main_event.motion_axis in {"", "unknown"}:
            issues.append("missing_motion_axis")

        if (
            frame.main_event.event_location_relation in {"ahead_of", "in_front_of"}
            and frame.main_event.motion_axis == "longitudinal"
            and frame.main_event.event_type in {"path_crossing", "enter_ego_lane"}
        ):
            issues.append("lateral_crossing_misread_as_longitudinal")

        if frame.occlusion.enabled and frame.occlusion.occluder_type in {"", "unknown"}:
            issues.append("occlusion_enabled_but_unknown_occluder")

        if not frame.event_sequence:
            issues.append("missing_event_sequence")

        return VerificationResult(passed=len(issues) == 0, issues=issues)

    def repair_frame(self, frame: EventFrame) -> EventFrame:
        """Apply deterministic repairs to common semantic inconsistencies."""

        s = frame.sentence.lower()
        ev = frame.main_event

        if (
            frame.main_actor.actor_class in {"", "unknown"}
            or ev.event_type in {"", "unknown"}
            or ev.motion_axis in {"", "unknown"}
            or _contains(s, ["actual hazard", "critical event", "instead", "do not create", "not crossing", "only an occluder"])
        ):
            analysis = HazardFocusResolver().resolve(NarrativeDecomposer().analyze(frame.sentence))
            if analysis.selected_event is not None:
                repaired = EventFrame.from_dict(
                    candidate_to_event_frame_dict(frame.sentence, analysis, analysis.selected_event)
                )
                repaired.parser_notes = (
                    frame.parser_notes + " | repaired_by=narrative_focus_resolver"
                ).strip(" |")
                return repaired

        # Actor normalization.
        actor_text = (frame.main_actor.text or "").lower()
        if frame.main_actor.actor_class in {"", "unknown"}:
            if _contains(
                s,
                [
                    "pedestrian",
                    "walker",
                    "person",
                    "child",
                    "kid",
                    "schoolkid",
                    "jogger",
                    "runner",
                    "jaywalker",
                    "figure",
                    "shopper",
                    "commuter",
                    "passerby",
                    "wheelchair user",
                    "on foot",
                ],
            ):
                frame.main_actor.actor_class = "human_on_foot"
                frame.main_actor.actor_role = "hazard_actor"
            elif _contains(s, ["cyclist", "bicyclist", "bicycle", "bike", "e-bike", "ebike", "scooter rider", "bicycle rider"]):
                frame.main_actor.actor_class = "cyclist"
                frame.main_actor.actor_role = "hazard_actor"
            elif _contains(s, ["vehicle", "car", "sedan", "suv", "truck", "bus", "van", "taxi", "pickup", "traffic ahead", "traffic"]):
                frame.main_actor.actor_class = "vehicle"
                frame.main_actor.actor_role = "hazard_actor"
            elif _contains(s, ["object", "debris", "obstacle", "cone", "barrier"]):
                frame.main_actor.actor_class = "traffic_object"
                frame.main_actor.actor_role = "hazard_actor"

        if not frame.main_actor.text or frame.main_actor.text == "unknown actor":
            if frame.main_actor.actor_class == "human_on_foot":
                frame.main_actor.text = "pedestrian"
            elif frame.main_actor.actor_class == "cyclist":
                frame.main_actor.text = "cyclist"
            elif frame.main_actor.actor_class == "vehicle":
                frame.main_actor.text = "vehicle"
            elif frame.main_actor.actor_class == "traffic_object":
                frame.main_actor.text = "traffic object"

        # Occlusion repair.
        if _contains(
            s,
            [
                "hidden",
                "concealed",
                "screened",
                "masked",
                "blind side",
                "behind a",
                "behind an",
                "blocked by",
                "blocked from view",
                "line of sight",
                "line-of-sight",
                "obstructed",
                "obstructed by",
                "occluded by",
                "partially occluded",
                "between parked cars",
                "between parked vehicles",
                "from behind",
                "shadow of",
                "pops out",
                "emerges from",
                "out of sight",
                "occluding vehicle",
            ],
        ):
            frame.occlusion.enabled = True
            frame.diagnostics.is_occluded = True

            if _contains(s, ["truck"]):
                frame.occlusion.occluder_type = "truck"
            elif _contains(s, ["bus"]):
                frame.occlusion.occluder_type = "bus"
            elif _contains(s, ["van"]):
                frame.occlusion.occluder_type = "van"
            elif _contains(s, ["parked car", "parked cars", "parked vehicle", "parked vehicles"]):
                frame.occlusion.occluder_type = "parked_vehicle"
            elif frame.occlusion.occluder_type in {"", "unknown"}:
                frame.occlusion.occluder_type = "vehicle"

            if frame.occlusion.relation_to_actor in {"", "unknown"}:
                frame.occlusion.relation_to_actor = "behind"

        # Roundabout / rotary repair has priority over generic cut-in.
        if _contains(s, ["roundabout", "rotary", "traffic circle", "gyratory", "circular junction", "circular intersection"]) or (
            _contains(
                s,
                [
                    "circulating lane",
                    "circulating vehicle",
                    "circulating stream",
                    "circulatory lane",
                    "inside-lane",
                    "in the roundabout",
                    "in the circle",
                    "vehicle inside the circle",
                    "vehicle already inside",
                    "already in the circle",
                    "already in the roundabout",
                    "already inside",
                ],
            )
            and _contains(s, ["entry", "entrance", "approach", "merge", "round entry", "joins", "opening"])
        ):
            frame.road_context.road_type = "roundabout"
            frame.road_context.lane_context = "roundabout_entry"
            frame.ego_event.ego_maneuver = "enter_roundabout"
            frame.diagnostics.is_roundabout_entry = True
            frame.main_actor.actor_class = "vehicle"
            if not frame.main_actor.text or frame.main_actor.text in {"unknown actor", "vehicle"}:
                frame.main_actor.text = "circulating vehicle"

            ev.event_type = "roundabout_entry_conflict"
            ev.motion_axis = "merging"
            ev.motion_direction = "circulating_across_entry"
            ev.event_location_relation = "at_roundabout_entry"
            ev.location_relation_function = "event_anchor"
            ev.source_relation = "from_circulating_lane"
            ev.target_relation = "roundabout_entry"
            if ev.path_or_object in {"", "unknown"}:
                ev.path_or_object = "roundabout entry"

            return frame

        # Left-turn / oncoming repair.
        if _contains(
            s,
            [
                "left turn",
                "left-turn",
                "turns left",
                "turning left",
                "unprotected left",
                "oncoming",
                "opposing traffic",
                "opposite direction",
                "opposite lane",
                "far side",
                "other direction",
                "coming the other way",
                "straight-through traffic",
                "inbound",
            ],
        ):
            if _contains(s, ["left turn", "left-turn", "turns left", "turning left", "unprotected left"]):
                frame.ego_event.ego_maneuver = "left_turn"
                frame.diagnostics.is_ego_left_turn = True

            if _contains(
                s,
                [
                    "oncoming",
                    "opposing",
                    "opposite direction",
                    "opposite-direction",
                    "opposite lane",
                    "far side",
                    "other direction",
                    "coming the other way",
                    "opposite traffic",
                    "approaching vehicle",
                    "approaching car",
                    "fast approaching vehicle",
                    "through traffic",
                    "straight-through traffic",
                    "opposite approach",
                    "keeps coming",
                    "through vehicle",
                    "inbound",
                ],
            ):
                frame.main_actor.actor_class = "vehicle"
                if not frame.main_actor.text or frame.main_actor.text == "unknown actor":
                    frame.main_actor.text = "oncoming vehicle"

                ev.event_type = (
                    "ego_left_turn_across_oncoming"
                    if frame.ego_event.ego_maneuver == "left_turn"
                    else "oncoming_through_conflict"
                )
                ev.motion_axis = "oncoming"
                ev.motion_direction = "opposing_through"
                ev.event_location_relation = "at_intersection"
                ev.location_relation_function = "event_anchor"
                ev.source_relation = "from_opposite_direction"
                ev.target_relation = "intersection"
                frame.road_context.road_type = "intersection"
                frame.road_context.lane_context = "opposing_lane"

        # Vehicle cut-in / merge repair.
        if frame.main_actor.actor_class == "vehicle" and _contains(
            s,
            [
                "cuts in",
                "cut in",
                "swerves into",
                "encroaches into",
                "drifts",
                "slides over",
                "slide over",
                "nudges over",
                "edges into",
                "weaves into",
                "veers into",
                "slots into",
                "slot into",
                "crosses the lane boundary",
                "moving over",
                "moves laterally",
                "from adjacent",
                "adjacent",
                "from the right lane",
                "right-hand lane",
                "from the left lane",
                "left-hand lane",
                "neighboring right",
                "neighboring left",
                "takes the ego lane",
                "across the lane marker",
                "lane marker",
                "squeezes into",
                "slips into",
                "changes lanes",
                "cuts over",
                "into ego's lane",
                "into the ego lane",
                "occupies part of ego's lane",
                "lane boundary",
                "adjacent lane",
            ],
        ):
            ev.event_type = "lane_change_into_ego_lane"
            ev.motion_axis = "merging"
            ev.motion_direction = "into_ego_lane"
            ev.target_relation = "ego_lane"
            ev.location_relation_function = "target_lane"
            frame.diagnostics.is_lane_change_into_ego_lane = True
            frame.road_context.road_type = (
                frame.road_context.road_type
                if frame.road_context.road_type != "unknown"
                else "multi_lane_road"
            )
            frame.road_context.lane_context = "adjacent_lane"

            if _contains(s, ["from the left", "from the left lane", "left lane", "left side", "adjacent left", "neighboring left", "left-hand lane", "left-to-right"]):
                ev.source_relation = "from_left"
                ev.event_location_relation = "left_of"
            elif _contains(s, ["from the right", "from the right lane", "right lane", "right-hand lane", "right side", "adjacent right", "neighboring right"]):
                ev.source_relation = "from_right"
                ev.event_location_relation = "right_of"
            elif ev.source_relation in {"", "unknown"}:
                ev.source_relation = "from_adjacent_lane"

        # Lead braking / longitudinal repair.
        if frame.main_actor.actor_class == "vehicle" and _contains(
            s,
            [
                "brakes",
                "brake",
                "slams on the brakes",
                "slows",
                "sheds speed",
                "drops speed",
                "loses speed",
                "panic stop",
                "hard stop",
                "sudden stop",
                "comes to a halt",
                "reduces speed",
                "decelerates",
                "deceleration",
                "rapid deceleration",
                "checks up",
                "taps the brakes",
                "headway shrinks",
                "stops short",
                "stops suddenly",
                "nearly stops",
                "unexpectedly stops",
                "stops in the travel lane",
                "traffic ahead stops",
                "front vehicle stops",
            ],
        ):
            ev.event_type = (
                "hard_stop_ahead"
                if _contains(s, ["panic stop", "hard stop", "sudden stop", "comes to a halt", "without warning"])
                else "lead_vehicle_braking"
            )
            ev.motion_axis = "longitudinal"
            ev.motion_direction = "decelerating_ahead"
            ev.event_location_relation = "ahead_of"
            ev.location_relation_function = "actor_position"
            ev.target_relation = "ego_lane"
            frame.diagnostics.is_longitudinal_following = True
            frame.road_context.road_type = (
                frame.road_context.road_type
                if frame.road_context.road_type != "unknown"
                else "straight_lane"
            )
            frame.road_context.lane_context = "ego_lane"

        # Pedestrian/cyclist crossing repair.
        if frame.main_actor.actor_class in {"human_on_foot", "cyclist"} and _contains(
            s,
            [
                "cross",
                "crosses",
                "across",
                "traverses",
                "perpendicular",
                "steps out",
                "steps into",
                "stepping into",
                "step into",
                "steps off",
                "bolts",
                "darts",
                "hurries across",
                "shoots across",
                "rolls from",
                "moves laterally across",
                "angles across",
                "walks across",
                "runs across",
                "leaves the sidewalk",
                "leaves sidewalk",
                "leaves the median",
                "leaves the roadside",
                "slips between",
                "emerges",
                "appears",
                "enters the travel lane",
                "enters the roadway",
                "enters the road",
                "enters the travel path",
                "enters the lane",
                "enters the ego lane",
                "enters ego's path",
                "entering ego's path",
                "enters the ego vehicle's path",
                "entering the ego vehicle's path",
                "enters ego path",
                "entering ego path",
                "moving into",
                "moves into",
                "walks into",
                "walking into",
                "cuts across",
            ],
        ):
            ev.event_type = (
                "path_crossing"
                if _contains(s, ["cross", "crosses", "across", "traverses", "perpendicular", "runs across", "shoots across", "hurries across", "walks across", "cuts across"])
                else "enter_ego_lane"
            )
            ev.motion_axis = "lateral"
            ev.motion_direction = "across_ego_path"
            ev.target_relation = "ego_lane" if _contains(s, ["lane", "travel lane", "ego lane"]) else "ego_path"
            ev.location_relation_function = "event_anchor" if _contains(s, ["ahead", "in front", "just ahead"]) else "target_lane"
            frame.diagnostics.is_path_crossing = True

            if _contains(s, ["ahead", "just ahead"]):
                ev.event_location_relation = "ahead_of"
            elif _contains(s, ["in front", "forward path"]):
                ev.event_location_relation = "in_front_of"

            if ev.path_or_object in {"", "unknown"}:
                ev.path_or_object = "ego lane"

            if frame.road_context.road_type == "unknown":
                frame.road_context.road_type = "straight_lane"
            if frame.road_context.lane_context == "unknown":
                frame.road_context.lane_context = "ego_lane"

        # Static obstacle repair.
        if frame.main_actor.actor_class == "traffic_object" or _contains(
            s,
            ["object in the lane", "debris", "obstacle", "blocked lane", "cone", "cones", "traffic cones", "barrier", "construction", "work zone", "work-zone"],
        ):
            frame.main_actor.actor_class = "traffic_object"
            if not frame.main_actor.text or frame.main_actor.text == "unknown actor":
                frame.main_actor.text = "traffic object"
            ev.event_type = "object_in_lane"
            ev.motion_axis = "static"
            ev.motion_direction = "stationary"
            ev.target_relation = "ego_lane"
            ev.location_relation_function = "actor_position"
            if ev.event_location_relation == "unknown":
                ev.event_location_relation = "ahead_of"
            frame.road_context.road_type = (
                frame.road_context.road_type
                if frame.road_context.road_type != "unknown"
                else "straight_lane"
            )
            frame.road_context.lane_context = "ego_lane"

        # Relation-function sanity.
        if ev.motion_axis == "lateral" and ev.event_location_relation in {"ahead_of", "in_front_of"}:
            ev.location_relation_function = "event_anchor"

        if ev.motion_axis == "longitudinal" and ev.event_location_relation in {"ahead_of", "in_front_of"}:
            ev.location_relation_function = "actor_position"

        if ev.evidence_text == "":
            ev.evidence_text = frame.sentence

        if frame.main_actor.evidence_text == "":
            frame.main_actor.evidence_text = frame.sentence

        return frame

    def verify_spec(self, spec: Dict[str, Any]) -> VerificationResult:
        """Check mapped HazardSemanticSpec-like consistency."""

        issues: List[str] = []

        actor = spec.get("actor_layer", {}).get("primary_actor", "unknown")
        role = spec.get("actor_layer", {}).get("actor_role", "unknown")
        conflict = spec.get("interaction_layer", {}).get("conflict_type", "unknown")
        motion = spec.get("motion_layer", {}).get("motion_axis", "unknown")
        event_type = spec.get("motion_layer", {}).get("hazard_event_type", "unknown")

        if actor in {"unknown", ""}:
            issues.append("spec_missing_primary_actor")

        if role in {"unknown", ""}:
            issues.append("spec_missing_actor_role")

        if conflict in {"unknown", ""}:
            issues.append("spec_missing_conflict_type")

        if motion in {"unknown", ""}:
            issues.append("spec_missing_motion_axis")

        if event_type in {"unknown", ""}:
            issues.append("spec_missing_hazard_event_type")

        if role == "crossing_actor" and conflict != "lateral_conflict":
            issues.append("spec_crossing_actor_not_lateral_conflict")

        if role == "braking_actor" and conflict != "longitudinal_conflict":
            issues.append("spec_braking_actor_not_longitudinal_conflict")

        if role == "merging_actor" and conflict != "merging_conflict":
            issues.append("spec_merging_actor_not_merging_conflict")

        if conflict == "lateral_conflict" and motion != "lateral":
            issues.append("spec_lateral_conflict_not_lateral_motion")

        if conflict == "longitudinal_conflict" and motion != "longitudinal":
            issues.append("spec_longitudinal_conflict_not_longitudinal_motion")

        if conflict == "oncoming_conflict" and motion != "oncoming":
            issues.append("spec_oncoming_conflict_not_oncoming_motion")

        completed = spec.get("parameter_layer", {}).get("completed", {})
        if not completed:
            issues.append("spec_missing_completed_parameters")

        return VerificationResult(passed=len(issues) == 0, issues=issues)
