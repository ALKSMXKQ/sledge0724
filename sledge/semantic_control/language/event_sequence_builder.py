"""Deterministic event-sequence reconstruction for EventFrame.

Small LLMs often omit ego events or produce incomplete event order.  This module
rebuilds a minimal, simulation-oriented temporal chain from the main event
semantics.

The reconstructed sequence is intentionally simple:

1. ego baseline motion / maneuver,
2. hazard actor event,
3. conflict approach,
4. ego response.

This keeps event ordering stable and makes evaluation independent of whether a
7B model generated a perfect event_sequence JSON list.
"""

from __future__ import annotations

from typing import List

from sledge.semantic_control.language.event_frame import EventFrame, EventSequenceStep


class EventSequenceBuilder:
    """Build or repair EventFrame.event_sequence."""

    def build(self, frame: EventFrame, *, overwrite: bool = True) -> EventFrame:
        if frame.event_sequence and not overwrite:
            return frame

        frame.event_sequence = self._build_sequence(frame)
        return frame

    def _build_sequence(self, frame: EventFrame) -> List[EventSequenceStep]:
        ev = frame.main_event
        slots_event = ev.event_type
        motion = ev.motion_axis
        ego_maneuver = frame.ego_event.ego_maneuver

        if ego_maneuver in {"unknown", ""}:
            ego_maneuver = self._infer_ego_maneuver(frame)

        hazard_actor = frame.main_actor.text or "hazard_actor"

        sequence: List[EventSequenceStep] = []

        # E1: ego baseline.
        sequence.append(
            EventSequenceStep(
                order=1,
                actor="ego",
                event_type="ego_driving" if ego_maneuver == "drive_forward" else f"ego_{ego_maneuver}",
                action=ego_maneuver,
                relation_to_previous="start",
                evidence_text=frame.ego_event.evidence_text,
            )
        )

        # E2: hazard event.
        sequence.append(
            EventSequenceStep(
                order=2,
                actor=hazard_actor,
                event_type=slots_event if slots_event != "unknown" else self._infer_event_type(frame),
                action=ev.predicate_text or self._action_from_event(frame),
                relation_to_previous="after_ego_baseline",
                evidence_text=ev.evidence_text,
            )
        )

        # E3: conflict approach.
        sequence.append(
            EventSequenceStep(
                order=3,
                actor="ego_and_hazard_actor",
                event_type="conflict_point_approach",
                action=self._conflict_action(frame, motion),
                relation_to_previous="after_hazard_event",
                evidence_text="inferred: ego approaches the conflict region",
            )
        )

        # E4: ego response.
        sequence.append(
            EventSequenceStep(
                order=4,
                actor="ego",
                event_type="ego_response",
                action=self._ego_response_action(frame),
                relation_to_previous="after_conflict_approach",
                evidence_text="inferred: ego response required by hazard",
            )
        )

        return sequence

    def _infer_ego_maneuver(self, frame: EventFrame) -> str:
        if frame.diagnostics.is_ego_left_turn or frame.main_event.motion_axis == "oncoming":
            return "left_turn"

        if frame.diagnostics.is_roundabout_entry or frame.road_context.road_type == "roundabout":
            return "enter_roundabout"

        return "drive_forward"

    def _infer_event_type(self, frame: EventFrame) -> str:
        ev = frame.main_event

        if frame.diagnostics.is_roundabout_entry:
            return "roundabout_entry_conflict"

        if frame.diagnostics.is_ego_left_turn:
            return "ego_left_turn_across_oncoming"

        if frame.diagnostics.is_lane_change_into_ego_lane:
            return "lane_change_into_ego_lane"

        if frame.diagnostics.is_longitudinal_following:
            return "lead_vehicle_braking"

        if frame.diagnostics.is_path_crossing:
            return "path_crossing"

        if ev.motion_axis == "lateral":
            return "path_crossing"

        if ev.motion_axis == "merging":
            return "lane_change_into_ego_lane"

        if ev.motion_axis == "longitudinal":
            return "lead_vehicle_braking"

        if ev.motion_axis == "oncoming":
            return "oncoming_through_conflict"

        if ev.motion_axis == "static":
            return "object_in_lane"

        return "unknown"

    def _action_from_event(self, frame: EventFrame) -> str:
        event_type = self._infer_event_type(frame)

        mapping = {
            "path_crossing": "cross_ego_path",
            "enter_ego_lane": "enter_ego_lane",
            "lane_change_into_ego_lane": "merge_into_ego_lane",
            "lead_vehicle_braking": "decelerate_ahead",
            "hard_stop_ahead": "hard_stop_ahead",
            "ego_left_turn_across_oncoming": "ego_left_turn_across_oncoming",
            "oncoming_through_conflict": "oncoming_vehicle_approaches",
            "roundabout_entry_conflict": "circulating_vehicle_blocks_entry",
            "object_in_lane": "static_object_in_ego_lane",
        }

        return mapping.get(event_type, "hazard_event")

    def _conflict_action(self, frame: EventFrame, motion: str) -> str:
        event_type = frame.main_event.event_type

        if frame.diagnostics.is_roundabout_entry or event_type == "roundabout_entry_conflict":
            return "ego_entry_gap_closes"

        if frame.diagnostics.is_ego_left_turn or motion == "oncoming":
            return "ego_turn_path_intersects_oncoming_path"

        if motion == "lateral" or event_type in {"path_crossing", "enter_ego_lane"}:
            return "ego_approaches_lateral_crossing_point"

        if motion == "merging" or event_type == "lane_change_into_ego_lane":
            return "ego_gap_compresses_due_to_merging_actor"

        if motion == "longitudinal" or event_type in {"lead_vehicle_braking", "hard_stop_ahead"}:
            return "ego_headway_closes"

        if motion == "static" or event_type == "object_in_lane":
            return "ego_approaches_static_obstacle"

        return "ego_approaches_conflict_region"

    def _ego_response_action(self, frame: EventFrame) -> str:
        event_type = frame.main_event.event_type
        motion = frame.main_event.motion_axis

        if frame.diagnostics.is_roundabout_entry or event_type == "roundabout_entry_conflict":
            return "yield_or_abort_roundabout_entry"

        if frame.diagnostics.is_ego_left_turn or motion == "oncoming":
            return "yield_or_brake_for_oncoming_vehicle"

        if motion == "lateral" or event_type in {"path_crossing", "enter_ego_lane"}:
            return "brake_or_yield_to_crossing_actor"

        if motion == "merging" or event_type == "lane_change_into_ego_lane":
            return "brake_or_adjust_gap_for_cut_in"

        if motion == "longitudinal" or event_type in {"lead_vehicle_braking", "hard_stop_ahead"}:
            return "brake_to_avoid_rear_end_risk"

        if motion == "static" or event_type == "object_in_lane":
            return "brake_or_steer_around_static_obstacle"

        return "brake_or_yield"