"""Compositional mapping from EventFrame to HazardSemanticSpec-like slots.

EventFrame v4 changes the generation logic from *scenario-family driven* to
*semantic-slot driven*.

Earlier versions used a high-level ``scenario_family`` (pedestrian_crossing,
vehicle_cut_in, lead_brake, ... ) as the main switch.  That is useful for
logging/evaluation, but it still looks like template matching.  This mapper
instead first composes a set of reusable semantic slots from EventFrame:

    actor_type, actor_role_hint, motion_geometry, path_role, anchor_region,
    visibility, occlusion_type, road_topology, ego_maneuver, source_side,
    target_path, distance_relation, interaction_goal, ...

The final spec is then projected from those slots.  ``scenario_family`` is kept
only as a derived compatibility label for the existing evaluator and old gold
ontology; it is not the primary generation driver.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple

from sledge.semantic_control.language.event_frame import EventFrame


def flatten_dict(obj: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def _contains(text: str, terms: List[str]) -> bool:
    text = (text or "").lower()
    return any(t in text for t in terms)


def _clean(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    s = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return s or default


def _actor_to_base(actor_class: str) -> str:
    if actor_class == "human_on_foot":
        return "pedestrian"
    if actor_class == "cyclist":
        return "cyclist"
    if actor_class == "traffic_object":
        return "traffic_object"
    if actor_class == "vehicle":
        return "vehicle"
    return "unknown"


def _anchor_region(frame: EventFrame) -> str:
    ev = frame.main_event
    rel = ev.event_location_relation
    road = frame.road_context.road_type
    fields = " ".join([rel, ev.evidence_text, frame.sentence]).lower()

    if road == "roundabout" or rel == "at_roundabout_entry" or _contains(
        frame.sentence.lower(), ["roundabout", "rotary", "traffic circle", "gyratory"]
    ):
        return "roundabout_entry"

    if rel in {"ahead_of", "in_front_of"} or _contains(
        fields,
        [
            "ahead of ego",
            "in front of ego",
            "just ahead",
            "ego car's trajectory",
            "vehicle's path",
            "ego path",
            "forward path",
            "front bumper",
            "ego lane",
            "ego's lane",
            "intended path",
            "near the ego",
            "just before",
        ],
    ):
        return "front"

    if rel == "left_of" or "left of ego" in fields:
        return "left"

    if rel == "right_of" or "right of ego" in fields:
        return "right"

    if rel == "at_intersection" or road == "intersection" or "intersection" in frame.sentence.lower():
        return "intersection"

    return "unknown"


def _sequence_labels(frame: EventFrame) -> List[str]:
    return [
        f"{s.order}:{s.actor}:{s.event_type}:{s.action}"
        for s in frame.event_sequence
    ]


def _specific_occluder_type(frame: EventFrame) -> str:
    s = frame.sentence.lower()
    raw = _clean(frame.occlusion.occluder_type)

    if "truck" in s:
        return "truck"

    if "bus" in s:
        return "bus"

    if "van" in s:
        return "van"

    if "parked car" in s or "parked cars" in s or "roadside parked" in s:
        return "parked_vehicle"

    if "parked vehicle" in s or "parked vehicles" in s:
        return "parked_vehicle"

    if "suv" in s:
        return "vehicle"

    if raw in {"truck", "bus", "van", "parked_vehicle", "vehicle"}:
        return raw

    return "vehicle" if frame.occlusion.enabled else "unknown"


def _is_construction_zone(frame: EventFrame) -> bool:
    s = frame.sentence.lower()
    return _contains(
        s,
        [
            "construction",
            "work zone",
            "work-zone",
            "road work",
            "roadwork",
            "traffic cones",
            "cone",
            "cones",
            "barrier",
            "construction barrier",
        ],
    )


def _obstacle_type(frame: EventFrame) -> str:
    s = frame.sentence.lower()
    if _contains(s, ["cone", "cones", "traffic cones"]):
        return "cone"
    if _contains(s, ["barrier", "construction barrier"]):
        return "barrier"
    if _is_construction_zone(frame):
        return "construction_zone"
    if _contains(s, ["debris"]):
        return "debris"
    if _contains(s, ["obstacle", "object"]):
        return "object"
    return "unknown"


def _merge_direction(frame: EventFrame) -> str:
    ev = frame.main_event
    s = frame.sentence.lower()
    fields = " ".join(
        [
            ev.source_relation,
            ev.motion_direction,
            ev.event_location_relation,
            ev.evidence_text,
            s,
        ]
    ).lower()

    if _contains(
        fields,
        [
            "from_left",
            "left side",
            "from the left",
            "adjacent left",
            "neighboring left",
            "left-hand lane",
            "left-to-right",
        ],
    ):
        return "left_merge"

    if _contains(
        fields,
        [
            "from_right",
            "right side",
            "from the right",
            "adjacent right",
            "neighboring right",
            "right-hand lane",
        ],
    ):
        return "right_merge"

    return "unknown"


def _crossing_direction(frame: EventFrame) -> str:
    ev = frame.main_event
    fields = " ".join(
        [
            ev.source_relation,
            ev.motion_direction,
            ev.evidence_text,
            frame.sentence,
        ]
    ).lower()
    if _contains(fields, ["from_right", "from the right", "right side", "right curb", "right sidewalk"]):
        return "right_to_left"
    if _contains(fields, ["from_left", "from the left", "left side", "left curb", "left sidewalk"]):
        return "left_to_right"
    return "crossing"


def _speed_relation(frame: EventFrame) -> str:
    s = frame.sentence.lower()
    if _contains(s, ["fast", "quickly", "rapidly", "high speed", "approaches quickly"]):
        return "fast_approach"
    if _contains(s, ["normal speed", "normally"]):
        return "normal"
    if _contains(s, ["slowly", "mildly"]):
        return "slow"
    return "unknown"


def _risk_level_from_text(frame: EventFrame, default: str = "moderate") -> str:
    s = frame.sentence.lower()
    if _contains(
        s,
        [
            "dangerous",
            "high collision risk",
            "near-miss",
            "near miss",
            "suddenly",
            "very close",
            "aggressive",
            "abrupt",
            "abruptly",
            "hard",
            "panic",
            "almost no",
            "no buffer",
            "no room",
            "small gap",
            "tight gap",
            "short headway",
            "fast",
            "quickly",
        ],
    ):
        return "aggressive"
    if _contains(
        s,
        [
            "mild",
            "mildly",
            "slowly",
            "comfortable",
            "enough reaction distance",
            "enough distance",
            "medium range",
        ],
    ):
        return "mild"
    return default


def _interaction_goal_from_text(frame: EventFrame, default: str) -> str:
    s = frame.sentence.lower()
    if _contains(s, ["high collision risk", "collision risk", "dangerous", "risky interaction", "risky"]):
        return "collision_risk"
    if _contains(s, ["near-miss", "near miss"]):
        return "near_miss"
    if _contains(s, ["yield", "yielding"]):
        return "yielding_conflict"
    if _contains(s, ["slows down for", "slow down for", "brake for"]):
        return "braking_pressure"
    if _contains(s, ["block", "blocks", "occupy", "occupies"]):
        return "trajectory_blocking"
    return default


def _side_from_merge_direction(direction: str) -> Any:
    if direction == "left_merge":
        return "left"
    if direction == "right_merge":
        return "right"
    return ["left", "right"]


def _is_vehicle_cutin(frame: EventFrame, actor_base: str) -> bool:
    ev = frame.main_event
    s = frame.sentence.lower()
    if ev.event_type == "path_crossing" or ev.motion_axis == "lateral":
        return False
    fields = " ".join(
        [
            ev.predicate_text,
            ev.path_or_object,
            ev.motion_direction,
            ev.source_relation,
            ev.target_relation,
            ev.evidence_text,
            s,
        ]
    ).lower()
    has_merge_predicate = _contains(
        fields,
        [
            "encroach",
            "swerve",
            "intrude",
            "edge",
            "weave",
            "move over",
            "moving over",
            "changes lanes",
            "cuts over",
            "cuts into",
            "merges",
            "merge",
            "slips into",
            "squeezes into",
            "nudges over",
            "nudge over",
            "crosses the lane boundary",
            "lane boundary",
            "occupies part",
            "neighboring lane",
            "adjacent lane",
            "cuts in",
            "cut in",
            "merge",
            "merging",
        ],
    )
    has_source_lane = ev.source_relation in {
        "from_left",
        "from_right",
        "from_adjacent_lane",
    } or _contains(fields, ["from adjacent", "neighboring left", "neighboring right"])

    return actor_base == "vehicle" and (
        ev.event_type == "lane_change_into_ego_lane"
        or frame.diagnostics.is_lane_change_into_ego_lane
        or ev.motion_axis == "merging"
        or has_merge_predicate
        or (has_source_lane and _contains(fields, ["ego_lane", "ego lane", "ego's lane", "into_ego_lane"]))
    )


def _is_pedestrian_or_cyclist_crossing(frame: EventFrame, actor_base: str) -> bool:
    ev = frame.main_event
    s = frame.sentence.lower()
    fields = " ".join(
        [
            ev.predicate_text,
            ev.path_or_object,
            ev.target_relation,
            ev.motion_direction,
            ev.evidence_text,
            s,
        ]
    ).lower()

    return actor_base in {"pedestrian", "cyclist"} and (
        ev.event_type in {"path_crossing", "enter_ego_lane"}
        or ev.motion_axis == "lateral"
        or _contains(
            fields,
            [
                "cross",
                "across",
                "traverse",
                "perpendicular",
                "step",
                "enter",
                "comes out",
                "come out",
                "appears",
                "pops out",
                "emerges",
                "ego path",
                "ego lane",
                "travel lane",
            ],
        )
    )


def _is_lead_braking(frame: EventFrame) -> bool:
    ev = frame.main_event
    s = frame.sentence.lower()
    fields = " ".join([ev.predicate_text, ev.event_type, ev.evidence_text, s]).lower()

    return (
        ev.event_type in {"lead_vehicle_braking", "hard_stop_ahead"}
        or frame.diagnostics.is_longitudinal_following
        or _contains(
            fields,
            [
                "brakes",
                "braking",
                "panic stop",
                "hard stop",
                "sheds speed",
                "slows sharply",
                "loses speed",
                "comes to a halt",
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
                "lead car",
                "lead vehicle",
            ],
        )
    )


def _is_left_turn_oncoming(frame: EventFrame) -> bool:
    ev = frame.main_event
    s = frame.sentence.lower()
    fields = " ".join(
        [ev.event_type, ev.evidence_text, frame.ego_event.ego_maneuver, s]
    ).lower()
    has_left_turn = (
        frame.ego_event.ego_maneuver == "left_turn"
        or frame.diagnostics.is_ego_left_turn
        or _contains(fields, ["left turn", "turns left", "turning left", "unprotected left"])
    )
    has_oncoming_actor = (
        ev.event_type in {"ego_left_turn_across_oncoming", "oncoming_through_conflict"}
        or ev.motion_axis == "oncoming"
        or ev.source_relation == "from_opposite_direction"
        or _contains(
            fields,
            [
                "opposing traffic",
                "opposing",
                "oncoming",
                "opposite direction",
                "opposite-direction",
                "opposite lane",
                "far side",
                "opposing vehicle",
                "other direction",
                "coming the other way",
                "opposite traffic",
                "opposite approach",
                "approaching vehicle",
                "approaching car",
                "fast approaching vehicle",
                "straight-through traffic",
                "opposite approach",
                "keeps coming",
                "through vehicle",
                "inbound",
            ],
        )
    )

    return has_oncoming_actor and (has_left_turn or ev.event_type == "oncoming_through_conflict")


def _source_side_from_relation(relation: str) -> str:
    if relation == "from_left":
        return "left"
    if relation == "from_right":
        return "right"
    if relation == "from_curb":
        return "curbside"
    if relation == "from_opposite_direction":
        return "opposite"
    return "unknown_side"


def _is_roundabout_entry(frame: EventFrame) -> bool:
    ev = frame.main_event
    s = frame.sentence.lower()
    fields = " ".join(
        [
            ev.event_type,
            frame.road_context.road_type,
            frame.road_context.lane_context,
            ev.evidence_text,
            s,
        ]
    ).lower()

    return (
        frame.road_context.road_type == "roundabout"
        or ev.event_type == "roundabout_entry_conflict"
        or frame.diagnostics.is_roundabout_entry
        or _contains(
            fields,
            [
                "roundabout",
                "rotary",
                "traffic circle",
                "gyratory",
                "circular junction",
                "circulating lane",
                "circulatory lane",
                "circulating vehicle",
                "inside-lane",
                "vehicle inside the circle",
                "inside the circle",
                "in the circle",
                "in the roundabout",
                "already in the circle",
                "already in the roundabout",
            ],
        )
    )


def _is_static_object(frame: EventFrame, actor_base: str) -> bool:
    ev = frame.main_event
    s = frame.sentence.lower()
    fields = " ".join([ev.event_type, ev.predicate_text, ev.path_or_object, s]).lower()

    return (
        actor_base == "traffic_object"
        or ev.event_type == "object_in_lane"
        or _contains(fields, ["debris", "obstacle", "object in the lane", "blocked lane", "barrier", "cone", "cones"])
    )


def _is_vehicle_crossing_path(frame: EventFrame, actor_base: str) -> bool:
    ev = frame.main_event
    s = frame.sentence.lower()
    return actor_base == "vehicle" and (
        ev.motion_axis == "lateral"
        or ev.event_type == "path_crossing"
        or (
            _contains(s, ["crosses from", "crosses the path", "cross traffic", "cuts across"])
            and not _contains(s, ["cuts in", "cut in", "lane boundary", "into ego's lane"])
        )
    )


def _distance_from_slots(frame: EventFrame, slots: Dict[str, Any]) -> str:
    """Derive distance relation from lexical evidence and semantic context.

    Important: ``close range`` means close, not necessarily ``small_gap``. Small
    gap is reserved for explicit gap/clearance language.
    """

    s = frame.sentence.lower()
    motion = slots.get("motion_geometry", "unknown")
    road = slots.get("road_topology", "unknown")

    if road == "roundabout":
        if _contains(
            s,
            [
                "tight gap",
                "tight entry gap",
                "little room",
                "leaves little room",
                "closes the entry gap",
                "close the entry gap",
                "blocks the gap",
                "closes in",
                "narrow opening",
                "opening",
                "small gap",
                "small-gap",
                "round entry",
            ],
        ):
            return "small_gap"

        if _contains(
            s,
            [
                "passes across",
                "sweeps past",
                "in front of ego",
                "already in the roundabout",
                "already circulating",
                "cross traffic",
                "close ahead",
                "passes close",
                "just before ego enters",
            ],
        ):
            return "close"

        return "medium"

    if motion == "merging":
        if _contains(
            s,
            [
                "little clearance",
                "almost no buffer",
                "almost no room",
                "no buffer",
                "no room",
                "little room",
                "little space",
                "tight gap",
                "small gap",
                "small-gap",
                "very tight",
                "squeezes",
                "squeezes into",
            ],
        ):
            return "small_gap"

        if _contains(s, ["comfortable gap", "enough gap", "medium gap", "medium range"]):
            return "medium"

        if _contains(s, ["close range", "close", "nearby"]):
            return "close"

        return "close"

    if motion == "longitudinal":
        if _contains(
            s,
            [
                "short headway",
                "very short headway",
                "shrinking headway",
            ],
        ):
            return "short_headway"

        if _contains(s, ["medium range"]):
            return "medium"

        if _contains(s, ["nearly stops", "close following conflict"]):
            return "close"

        if _contains(
            s,
            [
                "compresses",
                "closes the gap",
                "little gap",
                "small gap",
                "following distance",
                "short following distance",
                "short following gap",
            ],
        ):
            return "small_gap"

        if _contains(
            s,
            ["panic", "sharply", "abruptly", "heavily", "without warning", "rapidly", "slams on the brakes", "decelerates hard"],
        ):
            return "close"

        return "medium"

    if _contains(s, ["close to", "closer to", "closer", "very close", "close range", "near-miss", "near miss", "near the ego"]):
        return "close"

    if slots.get("visibility") == "occluded" or slots.get("anchor_region") == "front":
        return "close"

    return "medium"


def compose_semantic_slots(frame: EventFrame) -> Dict[str, Any]:
    """Compose reusable semantic slots from EventFrame.

    This is the core v4 abstraction. It does not pick a full scenario template.
    Instead, it derives independent slot values which can be combined by the
    mapper and parameter filler.
    """

    actor_base = _actor_to_base(frame.main_actor.actor_class)
    ev = frame.main_event

    slots: Dict[str, Any] = {
        "actor_type": actor_base,
        "actor_source_text": frame.main_actor.text,
        "predicate_text": ev.predicate_text,
        "event_type": ev.event_type,
        "motion_axis": ev.motion_axis,
        "path_or_object": ev.path_or_object,
        "source_relation": ev.source_relation,
        "target_relation": ev.target_relation,
        "relation_function": ev.location_relation_function,
        "anchor_region": _anchor_region(frame),
        "visibility": "occluded" if frame.occlusion.enabled else "visible",
        "occlusion_enabled": bool(frame.occlusion.enabled),
        "occluder_type": _specific_occluder_type(frame),
        "ego_maneuver": frame.ego_event.ego_maneuver,
        "road_topology": (
            frame.road_context.road_type
            if frame.road_context.road_type != "unknown"
            else "unknown"
        ),
        "road_layout": (
            frame.road_context.lane_context
            if frame.road_context.lane_context != "unknown"
            else "unknown"
        ),
        "target_path": (
            "ego_lane"
            if _contains(
                " ".join([ev.path_or_object, ev.target_relation, frame.sentence]).lower(),
                ["ego lane", "ego_lane", "ego path", "travel lane"],
            )
            else ev.path_or_object
        ),
        "source_side": "unknown",
        "motion_geometry": "unknown",
        "actor_role": "unknown",
        "conflict_geometry": "unknown",
        "conflict_direction": "unknown",
        "lane_context": (
            frame.road_context.lane_context
            if frame.road_context.lane_context != "unknown"
            else "unknown"
        ),
        "speed_relation": _speed_relation(frame),
        "interaction_goal": "unknown",
        "risk_level": "moderate",
        "collision_allowed": False,
        "obstacle_type": _obstacle_type(frame),
        "fine_grained_conflict_type": "unknown",
        "fine_grained_actor_role": "unknown",
    }

    if _is_roundabout_entry(frame):
        slots.update(
            {
                "road_topology": "roundabout",
                "road_layout": "roundabout_entry",
                "motion_geometry": "merging",
                "actor_role": "merging_actor",
                "conflict_geometry": "merging",
                "conflict_direction": "roundabout_entry",
                "source_side": "roundabout_inside",
                "target_path": "roundabout_entry",
                "interaction_goal": (
                    "yielding_conflict"
                    if _contains(frame.sentence.lower(), ["yield", "forces ego"])
                    else "yield_or_merge"
                ),
                "fine_grained_conflict_type": "roundabout_entry_conflict",
                "fine_grained_actor_role": "merging_actor",
            }
        )

    elif _is_left_turn_oncoming(frame):
        slots.update(
            {
                "road_topology": "intersection",
                "road_layout": "unprotected_left_turn",
                "ego_maneuver": "left_turn",
                "motion_geometry": "oncoming",
                "actor_role": "approaching_actor",
                "conflict_geometry": "crossing_path",
                "conflict_direction": "opposite",
                "lane_context": "opposite_lane",
                "source_side": "opposite",
                "target_path": "ego_turn_path",
                "interaction_goal": "yield_to_oncoming",
                "fine_grained_conflict_type": "left_turn_across_oncoming",
                "fine_grained_actor_role": "approaching_actor",
            }
        )

        if _contains(
            frame.sentence.lower(),
            ["quickly", "fast", "approaches quickly", "high speed", "keeps moving"],
        ):
            slots["risk_level"] = "aggressive"
            slots["interaction_goal"] = "collision_risk"

    elif _is_lead_braking(frame):
        slots.update(
            {
                "road_topology": (
                    "straight_lane"
                    if slots["road_topology"] == "unknown"
                    else slots["road_topology"]
                ),
                "road_layout": (
                    "same_lane_following"
                    if slots["road_layout"] == "unknown"
                    else slots["road_layout"]
                ),
                "motion_geometry": "longitudinal",
                "actor_role": "braking_actor",
                "conflict_geometry": "longitudinal",
                "conflict_direction": "front",
                "source_side": "front",
                "target_path": "same_lane",
                "interaction_goal": "braking_pressure",
                "fine_grained_conflict_type": (
                    "hard_stop_ahead"
                    if ev.event_type == "hard_stop_ahead"
                    or _contains(frame.sentence.lower(), ["panic", "halt", "without warning"])
                    else "lead_vehicle_braking"
                ),
                "fine_grained_actor_role": "braking_actor",
            }
        )

        if slots["fine_grained_conflict_type"] == "hard_stop_ahead" or _contains(
            frame.sentence.lower(),
            ["panic", "abrupt", "sharply", "heavily", "without warning", "slams on the brakes", "sudden stop", "rapid"],
        ):
            slots["risk_level"] = "aggressive"

    elif _is_vehicle_cutin(frame, actor_base):
        md = _merge_direction(frame)
        conflict_direction = md if md != "unknown" else "unknown"
        slots.update(
            {
                "road_topology": (
                    "multi_lane_road"
                    if slots["road_topology"] == "unknown"
                    else slots["road_topology"]
                ),
                "road_layout": (
                    "adjacent_lane_cut_in"
                    if slots["road_layout"] == "unknown"
                    else slots["road_layout"]
                ),
                "motion_geometry": "merging",
                "actor_role": "merging_actor",
                "conflict_geometry": "merging",
                "conflict_direction": conflict_direction,
                "source_side": _side_from_merge_direction(md),
                "target_path": "ego_lane",
                "interaction_goal": "merge_into_ego_lane",
                "fine_grained_conflict_type": "lane_change_into_ego_lane",
                "fine_grained_actor_role": "merging_actor",
            }
        )

        if _contains(
            frame.sentence.lower(),
            [
                "aggressive",
                "almost no room",
                "almost no buffer",
                "no room",
                "no buffer",
                "little clearance",
                "little space",
                "small gap",
                "small-gap",
                "tight gap",
            ],
        ):
            slots["risk_level"] = "aggressive"

    elif _is_vehicle_crossing_path(frame, actor_base):
        slots.update(
            {
                "road_topology": (
                    "intersection"
                    if _contains(frame.sentence.lower(), ["intersection", "junction"])
                    else slots["road_topology"]
                ),
                "road_layout": (
                    "crossing_path"
                    if slots["road_layout"] == "unknown"
                    else slots["road_layout"]
                ),
                "motion_geometry": "lateral",
                "actor_role": "crossing_actor",
                "conflict_geometry": "lateral",
                "conflict_direction": _crossing_direction(frame),
                "source_side": _source_side_from_relation(ev.source_relation),
                "target_path": "ego_lane" if slots["target_path"] == "unknown" else slots["target_path"],
                "interaction_goal": "avoid_crossing_actor",
                "fine_grained_conflict_type": "crossing_path_conflict",
                "fine_grained_actor_role": "crossing_actor",
            }
        )

    elif _is_pedestrian_or_cyclist_crossing(frame, actor_base):
        slots.update(
            {
                "road_topology": (
                    "intersection"
                    if _contains(frame.sentence.lower(), ["intersection", "junction"])
                    else "crosswalk_area"
                    if _contains(frame.sentence.lower(), ["crosswalk"])
                    else
                    "straight_lane"
                    if slots["road_topology"] == "unknown"
                    else slots["road_topology"]
                ),
                "road_layout": (
                    "crossing_ego_lane"
                    if slots["road_layout"] == "unknown"
                    else slots["road_layout"]
                ),
                "motion_geometry": "lateral",
                "actor_role": "crossing_actor",
                "conflict_geometry": "lateral",
                "conflict_direction": _crossing_direction(frame),
                "source_side": _source_side_from_relation(ev.source_relation),
                "target_path": "ego_lane" if slots["target_path"] == "unknown" else slots["target_path"],
                "interaction_goal": "avoid_crossing_actor",
                "fine_grained_conflict_type": (
                    ev.event_type if ev.event_type != "unknown" else "path_crossing"
                ),
                "fine_grained_actor_role": "crossing_actor",
            }
        )

    elif _is_static_object(frame, actor_base):
        slots.update(
            {
                "road_topology": (
                    "construction_zone"
                    if _is_construction_zone(frame)
                    else "straight_lane"
                    if slots["road_topology"] == "unknown"
                    else slots["road_topology"]
                ),
                "road_layout": (
                    "lane_blocking"
                    if _is_construction_zone(frame)
                    else "object_in_ego_lane"
                    if slots["road_layout"] == "unknown"
                    else slots["road_layout"]
                ),
                "motion_geometry": "static",
                "actor_role": "blocking_actor" if _is_construction_zone(frame) else "static_obstacle",
                "conflict_geometry": "static",
                "conflict_direction": "front",
                "source_side": "front",
                "target_path": "ego_lane",
                "interaction_goal": (
                    "trajectory_blocking"
                    if _is_construction_zone(frame)
                    else "avoid_static_obstacle"
                ),
                "fine_grained_conflict_type": (
                    "lane_blocking_conflict"
                    if _is_construction_zone(frame)
                    else "object_in_lane"
                ),
                "fine_grained_actor_role": "blocking_actor" if _is_construction_zone(frame) else "static_obstacle",
            }
        )

    slots["distance_relation"] = _distance_from_slots(frame, slots)
    slots["risk_level"] = _risk_level_from_text(frame, str(slots.get("risk_level", "moderate")))
    slots["interaction_goal"] = _interaction_goal_from_text(
        frame, str(slots.get("interaction_goal", "unknown"))
    )
    if slots.get("actor_role") == "unknown" and slots.get("fine_grained_actor_role") != "unknown":
        slots["actor_role"] = slots["fine_grained_actor_role"]
    if slots.get("conflict_direction") == "unknown":
        motion = slots.get("motion_geometry")
        if motion == "longitudinal":
            slots["conflict_direction"] = "front"
        elif motion == "oncoming":
            slots["conflict_direction"] = "opposite"
        elif motion == "lateral":
            slots["conflict_direction"] = _crossing_direction(frame)
        elif motion == "merging" and slots.get("source_side") == "roundabout_inside":
            slots["conflict_direction"] = "roundabout_entry"
    return slots


def _legacy_projection(slots: Dict[str, Any]) -> Dict[str, str]:
    """Project compositional slots to legacy HazardSemanticSpec slot names.

    This keeps compatibility with the existing evaluator and primitive compiler.
    The high-level scenario_family is derived from slots, not used as input.
    """

    actor = slots.get("actor_type", "unknown")
    motion = slots.get("motion_geometry", "unknown")
    road = slots.get("road_topology", "unknown")
    fine = slots.get("fine_grained_conflict_type", "unknown")

    primary_actor = actor
    actor_role = "unknown"
    conflict_type = "unknown"
    conflict_direction = "unknown"
    scenario_family = "unknown"

    if road == "roundabout" and slots.get("road_layout") == "roundabout_entry":
        primary_actor = "cutin_vehicle"
        actor_role = "merging_actor"
        conflict_type = "merging_conflict"
        conflict_direction = "roundabout_entry"
        scenario_family = "roundabout_entry"

    elif (
        slots.get("ego_maneuver") == "left_turn"
        or motion == "oncoming"
        or fine in {"left_turn_across_oncoming", "oncoming_through_conflict"}
    ):
        primary_actor = "vehicle"
        actor_role = "approaching_actor"
        conflict_type = "oncoming_conflict"
        conflict_direction = "opposite"
        scenario_family = "left_turn_oncoming"

    elif motion == "longitudinal" or fine in {"lead_vehicle_braking", "hard_stop_ahead"}:
        primary_actor = "lead_vehicle"
        actor_role = "braking_actor"
        conflict_type = "longitudinal_conflict"
        conflict_direction = "front"
        scenario_family = "lead_vehicle_braking"

    elif motion == "merging":
        primary_actor = "cutin_vehicle"
        actor_role = "merging_actor"
        conflict_type = "merging_conflict"

        src = slots.get("source_side")
        if src == "left":
            conflict_direction = "left_merge"
        elif src == "right":
            conflict_direction = "right_merge"
        elif src == "roundabout_inside":
            conflict_direction = "roundabout_entry"
        else:
            conflict_direction = "unknown"

        scenario_family = "vehicle_cut_in"

    elif motion == "lateral":
        actor_role = "crossing_actor"
        conflict_type = "lateral_conflict"
        conflict_direction = slots.get("conflict_direction", "crossing")
        scenario_family = (
            "pedestrian_or_cyclist_crossing"
            if actor in {"pedestrian", "cyclist"}
            else "lateral_crossing"
        )

    elif motion == "static" or actor == "traffic_object":
        primary_actor = "static_obstacle" if road == "construction_zone" else "traffic_object"
        actor_role = "blocking_actor" if road == "construction_zone" else "static_obstacle"
        conflict_type = "lane_blocking_conflict" if road == "construction_zone" else "static_obstacle_conflict"
        conflict_direction = "front"
        scenario_family = "object_in_lane"

    return {
        "primary_actor": primary_actor,
        "actor_role": actor_role,
        "conflict_type": conflict_type,
        "conflict_direction": conflict_direction,
        "scenario_family": scenario_family,
    }


class EventFrameToHazardSpecMapper:
    """Map EventFrame semantics to a compositional scenario-control slot space."""

    def map(self, frame: EventFrame) -> Dict[str, Any]:
        slots = compose_semantic_slots(frame)
        legacy = _legacy_projection(slots)
        ev = frame.main_event

        hazard_event_type = slots.get("fine_grained_conflict_type", "unknown")

        if hazard_event_type == "unknown":
            hazard_event_type = ev.event_type

        if hazard_event_type == "unknown" and slots.get("motion_geometry") == "lateral":
            hazard_event_type = "path_crossing"

        if hazard_event_type == "unknown" and slots.get("motion_geometry") == "merging":
            hazard_event_type = "lane_change_into_ego_lane"

        if hazard_event_type == "unknown" and slots.get("motion_geometry") == "longitudinal":
            hazard_event_type = "lead_vehicle_braking"

        if hazard_event_type == "unknown" and slots.get("motion_geometry") == "oncoming":
            hazard_event_type = "oncoming_through_conflict"

        motion_axis = (
            ev.motion_axis
            if ev.motion_axis != "unknown"
            else slots.get("motion_geometry", "unknown")
        )

        if slots.get("motion_geometry") == "merging":
            motion_axis = "merging"
        elif slots.get("motion_geometry") == "lateral":
            motion_axis = "lateral"
        elif slots.get("motion_geometry") == "longitudinal":
            motion_axis = "longitudinal"
        elif slots.get("motion_geometry") == "oncoming":
            motion_axis = "oncoming"

        spec: Dict[str, Any] = {
            "schema_version": "eventframe_v4_slot_composition",
            "scenario_family": legacy["scenario_family"],
            "semantic_slots": deepcopy(slots),
            "actor_layer": {
                "primary_actor": legacy["primary_actor"],
                "actor_role": legacy["actor_role"],
                "actor_source_text": frame.main_actor.text,
                "base_actor_type": slots.get("actor_type", "unknown"),
            },
            "interaction_layer": {
                "conflict_type": legacy["conflict_type"],
                "anchor_region": slots.get("anchor_region", "unknown"),
                "relative_relation": ev.event_location_relation,
                "relation_function": ev.location_relation_function,
                "source_relation": ev.source_relation,
                "target_relation": ev.target_relation,
                "conflict_direction": legacy["conflict_direction"],
                "distance_relation": slots.get("distance_relation", "unknown"),
                "interaction_goal": slots.get("interaction_goal", "unknown"),
                "speed_relation": slots.get("speed_relation", "unknown"),
            },
            "motion_layer": {
                "hazard_event_type": hazard_event_type,
                "predicate_text": ev.predicate_text,
                "motion_axis": motion_axis,
                "motion_direction": ev.motion_direction,
                "path_or_object": slots.get("target_path", ev.path_or_object),
                "ego_maneuver": slots.get("ego_maneuver", frame.ego_event.ego_maneuver),
            },
            "event_layer": {
                "event_sequence": [step.__dict__ for step in frame.event_sequence],
                "event_sequence_labels": _sequence_labels(frame),
                "num_events": len(frame.event_sequence),
                "fine_grained_conflict_type": slots.get("fine_grained_conflict_type", "unknown"),
                "fine_grained_actor_role": slots.get("fine_grained_actor_role", "unknown"),
            },
            "object_layer": {
                "occlusion": {
                    "enabled": bool(slots.get("occlusion_enabled", False)),
                    "occluder_type": slots.get("occluder_type", "unknown"),
                    "occlusion_level": (
                        "partial"
                        if bool(slots.get("occlusion_enabled", False))
                        and _contains(frame.sentence.lower(), ["partial", "partially"])
                        else "full"
                        if bool(slots.get("occlusion_enabled", False))
                        else "none"
                    ),
                    "relation_to_actor": frame.occlusion.relation_to_actor,
                },
                "static_obstacle": {
                    "enabled": slots.get("motion_geometry") == "static",
                    "obstacle_type": slots.get("obstacle_type", "unknown"),
                },
            },
            "road_layer": {
                "road_type": slots.get("road_topology", "unknown"),
                "lane_context": (
                    slots.get("lane_context", "unknown")
                    if slots.get("lane_context", "unknown") != "unknown"
                    else slots.get("road_layout", "unknown")
                ),
                "road_topology": slots.get("road_topology", "unknown"),
                "generated_road_layout": slots.get("road_layout", "unknown"),
                "anchor_type": (
                    "intersection_center"
                    if slots.get("road_topology") == "intersection"
                    and slots.get("road_layout") == "unprotected_left_turn"
                    else slots.get("anchor_region", "unknown")
                ),
                "has_crosswalk": _contains(frame.sentence.lower(), ["crosswalk"]),
            },
            "risk_layer": {
                "risk_level": slots.get("risk_level", "moderate"),
                "collision_allowed": bool(slots.get("collision_allowed", False)),
            },
            "validation_layer": {
                "require_visibility_match": bool(slots.get("occlusion_enabled", False)),
                "composition_policy": "semantic_slots_not_full_scene_template",
            },
            "parameter_layer": {
                "required_missing": frame.missing_information.required,
                "defaultable_missing": frame.missing_information.defaultable,
                "distributional_defaults": frame.missing_information.distributional,
                "completed": {},
                "completion_policy": "pending",
            },
            "evidence": {
                "sentence": frame.sentence,
                "actor": frame.main_actor.evidence_text,
                "event": ev.evidence_text,
                "ego_event": frame.ego_event.evidence_text,
                "road_context": frame.road_context.evidence_text,
                "occlusion": frame.occlusion.evidence_text,
                "event_sequence": [
                    step.evidence_text
                    for step in frame.event_sequence
                    if step.evidence_text
                ],
            },
            "confidence": frame.confidence,
        }

        return spec


def validate_spec(spec: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Lightweight structural validation before primitive compilation."""

    errors: List[str] = []
    flat = flatten_dict(spec)

    required = [
        "actor_layer.primary_actor",
        "actor_layer.actor_role",
        "interaction_layer.conflict_type",
        "motion_layer.hazard_event_type",
        "semantic_slots.motion_geometry",
    ]

    for key in required:
        if key not in flat or flat[key] in {None, "", "unknown"}:
            errors.append(f"missing_or_unknown:{key}")

    actor = flat.get("actor_layer.primary_actor")
    role = flat.get("actor_layer.actor_role")
    conflict = flat.get("interaction_layer.conflict_type")
    motion_axis = flat.get("motion_layer.motion_axis")
    event_type = flat.get("motion_layer.hazard_event_type")
    num_events = flat.get("event_layer.num_events", 0)

    if role == "crossing_actor" and conflict != "lateral_conflict":
        errors.append("crossing_actor_requires_lateral_conflict")

    if conflict == "lateral_conflict" and motion_axis not in {"lateral", "unknown"}:
        errors.append("lateral_conflict_requires_lateral_or_unknown_motion_axis")

    if role == "braking_actor" and event_type not in {"lead_vehicle_braking", "hard_stop_ahead"}:
        errors.append("braking_actor_requires_braking_event")

    if role == "merging_actor" and conflict != "merging_conflict":
        errors.append("merging_actor_requires_merging_conflict")

    if actor == "pedestrian" and conflict == "longitudinal_conflict":
        errors.append("pedestrian_should_not_default_to_longitudinal_conflict")

    if not isinstance(num_events, int) or num_events <= 0:
        errors.append("event_sequence_required")

    completed = spec.get("parameter_layer", {}).get("completed", {})
    if not completed:
        errors.append("completed_parameters_required")

    return len(errors) == 0, errors
