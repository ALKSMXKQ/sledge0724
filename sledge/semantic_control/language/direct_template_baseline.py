"""Direct-template baseline natural-language semantic-control pipeline.

This module intentionally does not call the EventFrame v4 parser, mapper, or
missing-info filler. It parses prompts directly into semantic slots and then
projects those slots into the final parameter-template shape.

The design goal is to keep the baseline independent from the main EventFrame
approach:

- main: EventFrame roles/events first, ordered event sequence, then mapping.
- baseline: direct slot/template parsing, without an explicit event frame.
"""

from __future__ import annotations

import json
import urllib.request
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


DIRECT_TEMPLATE_PROMPT = r"""
You are a direct scene-parameter template generator for autonomous-driving scene descriptions.
Do NOT create an intermediate EventFrame. Read the sentence and directly output the final
parameter-template JSON object.

Return ONLY one JSON object with this shape:
{
  "schema_version": "direct_template_baseline_spec",
  "canonical_type": "short scenario type",
  "semantic_slots": {
    "actor_type": "pedestrian | cyclist | vehicle | traffic_object | unknown",
    "actor_role": "crossing_actor | merging_actor | braking_actor | approaching_actor | blocking_actor | static_obstacle | unknown",
    "motion_geometry": "lateral_crossing | merging | longitudinal | crossing_path | static | unknown",
    "conflict_geometry": "lateral | merging | longitudinal | crossing_path | static | unknown",
    "conflict_direction": "right_to_left | left_to_right | left_merge | right_merge | front | opposite | roundabout_entry | crossing | unknown",
    "source_side": "left | right | front | opposite | roundabout_inside | unknown",
    "target_path": "ego_lane | ego_path | ego_turn_path | roundabout_entry | same_lane | unknown",
    "anchor_region": "front | intersection | roundabout_entry | unknown",
    "visibility": "visible | occluded",
    "occlusion_enabled": false,
    "occluder_type": "vehicle | parked_vehicle | bus | truck | van | unknown",
    "road_topology": "straight_lane | intersection | roundabout | construction_zone | crosswalk_area | unknown",
    "road_layout": "crossing_ego_lane | adjacent_lane_cut_in | same_lane_following | unprotected_left_turn | roundabout_entry | lane_blocking | unknown",
    "lane_context": "ego_lane | adjacent_lane | same_lane_following | opposite_lane | roundabout_entry | unknown",
    "ego_maneuver": "drive_forward | left_turn | enter_roundabout",
    "event_type": "path_crossing | enter_ego_lane | lane_change_into_ego_lane | lead_vehicle_braking | left_turn_across_oncoming | roundabout_entry_conflict | object_in_lane | lane_blocking_conflict | unknown",
    "hazard_event_type": "same as event_type or finer hazard type",
    "interaction_goal": "avoid_crossing_actor | merge_into_ego_lane | braking_pressure | yield_to_oncoming | yielding_conflict | trajectory_blocking | collision_risk | unknown",
    "distance_relation": "close | medium | small_gap | short_headway | unknown",
    "speed_relation": "fast_approach | normal | slow | unknown",
    "risk_level": "mild | moderate | aggressive",
    "collision_allowed": false,
    "obstacle_type": "cone | barrier | construction_zone | debris | object | unknown",
    "has_crosswalk": false
  },
  "actor_layer": {"primary_actor": "string", "actor_role": "string", "base_actor_type": "string"},
  "interaction_layer": {"conflict_type": "lateral_conflict | merging_conflict | longitudinal_conflict | oncoming_conflict | lane_blocking_conflict | static_obstacle_conflict | unknown"},
  "motion_layer": {"hazard_event_type": "string", "motion_axis": "string", "motion_direction": "string", "path_or_object": "string", "ego_maneuver": "string"},
  "event_layer": {
    "event_sequence": [
      {"order": 1, "actor": "ego", "event_type": "ego_baseline", "action": "drive_forward", "relation_to_previous": "start"},
      {"order": 2, "actor": "hazard_actor", "event_type": "hazard_event", "action": "main hazard action", "relation_to_previous": "after_ego_baseline"},
      {"order": 3, "actor": "ego_and_hazard_actor", "event_type": "conflict_point_approach", "action": "approach conflict", "relation_to_previous": "after_hazard_event"}
    ],
    "event_sequence_labels": ["1:ego:ego_baseline:drive_forward"],
    "num_events": 3
  },
  "object_layer": {},
  "road_layer": {},
  "risk_layer": {"risk_level": "string", "collision_allowed": false},
  "validation_layer": {"pipeline_design": "direct_template_baseline_llm"},
  "parameter_layer": {},
  "detailed_spec": {}
}

Sentence:
<<SENTENCE>>
""".strip()


def flatten_dict(obj: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in obj.items():
        flat_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_dict(value, flat_key))
        else:
            out[flat_key] = value
    return out


def _contains(text: str, terms: List[str]) -> bool:
    text = (text or "").lower()
    return any(term in text for term in terms)


def _slot(value: Any, unit: str = "", source: str = "inferred_default", reason: str = "") -> Dict[str, Any]:
    return {"value": value, "unit": unit, "source": source, "reason": reason}


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise TypeError(f"expected string, got {type(text).__name__}")

    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object start found")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                obj = json.loads(text[start : i + 1])
                if not isinstance(obj, dict):
                    raise ValueError("extracted JSON is not an object")
                return obj
    raise ValueError("no complete JSON object found")


class DirectTemplateOllamaClient:
    def __init__(self, model: str, url: str = "http://127.0.0.1:11434", timeout: int = 60) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        return obj.get("response", "")


@dataclass
class DirectTemplateFrame:
    sentence: str
    semantic_slots: Dict[str, Any]
    event_sequence: List[Dict[str, Any]] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8
    parser_notes: str = "parsed_by=direct_template_baseline_rules"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DirectTemplateParser:
    """Parse a prompt directly into direct-template semantic slots."""

    def __init__(
        self,
        *,
        llm_provider: str = "none",
        llm_model: str = "qwen2.5:7b",
        ollama_url: str = "http://127.0.0.1:11434",
        allow_fallback: bool = True,
    ) -> None:
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.ollama_url = ollama_url
        self.allow_fallback = allow_fallback

    def parse(self, sentence: str) -> DirectTemplateFrame:
        # The direct-template baseline implementation is deterministic for now.  Keeping
        # the LLM-shaped constructor lets the evaluator compare providers later.
        return self._fallback_parse(sentence)

    def _fallback_parse(self, sentence: str) -> DirectTemplateFrame:
        s = sentence.lower()
        actor_type = self._actor_type(s)
        occlusion_enabled = self._is_occluded(s)
        occluder_type = self._occluder_type(s) if occlusion_enabled else "unknown"

        slots: Dict[str, Any] = {
            "actor_type": actor_type,
            "actor_role": "unknown",
            "motion_geometry": "unknown",
            "conflict_geometry": "unknown",
            "conflict_direction": "unknown",
            "source_side": "unknown",
            "target_path": self._target_path(s),
            "anchor_region": self._anchor_region(s),
            "visibility": "occluded" if occlusion_enabled else "visible",
            "occlusion_enabled": occlusion_enabled,
            "occluder_type": occluder_type,
            "road_topology": "unknown",
            "road_layout": "unknown",
            "lane_context": "unknown",
            "ego_maneuver": "drive_forward",
            "event_type": "unknown",
            "hazard_event_type": "unknown",
            "interaction_goal": "unknown",
            "distance_relation": "medium",
            "speed_relation": self._speed_relation(s),
            "risk_level": self._risk_level(s),
            "collision_allowed": False,
            "obstacle_type": self._obstacle_type(s),
            "has_crosswalk": "crosswalk" in s,
        }

        if self._is_roundabout(s):
            slots.update(
                {
                    "actor_type": "vehicle",
                    "actor_role": "merging_actor",
                    "motion_geometry": "merging",
                    "conflict_geometry": "merging",
                    "conflict_direction": "roundabout_entry",
                    "source_side": "roundabout_inside",
                    "target_path": "roundabout_entry",
                    "anchor_region": "roundabout_entry",
                    "road_topology": "roundabout",
                    "road_layout": "roundabout_entry",
                    "lane_context": "roundabout_entry",
                    "ego_maneuver": "enter_roundabout",
                    "event_type": "roundabout_entry_conflict",
                    "hazard_event_type": "roundabout_entry_conflict",
                    "interaction_goal": "yielding_conflict" if _contains(s, ["yield", "forces ego"]) else "yield_or_merge",
                    "distance_relation": self._roundabout_distance(s),
                }
            )
        elif self._is_left_turn_oncoming(s):
            slots.update(
                {
                    "actor_type": "vehicle",
                    "actor_role": "approaching_actor",
                    "motion_geometry": "oncoming",
                    "conflict_geometry": "crossing_path",
                    "conflict_direction": "opposite",
                    "source_side": "opposite",
                    "target_path": "ego_turn_path",
                    "anchor_region": "intersection",
                    "road_topology": "intersection",
                    "road_layout": "unprotected_left_turn",
                    "lane_context": "opposite_lane",
                    "ego_maneuver": "left_turn",
                    "event_type": "left_turn_across_oncoming",
                    "hazard_event_type": "left_turn_across_oncoming",
                    "interaction_goal": "collision_risk" if self._speed_relation(s) == "fast_approach" else "yield_to_oncoming",
                    "distance_relation": "medium",
                }
            )
        elif self._is_lead_braking(s, actor_type):
            slots.update(
                {
                    "actor_type": "vehicle",
                    "actor_role": "braking_actor",
                    "motion_geometry": "longitudinal",
                    "conflict_geometry": "longitudinal",
                    "conflict_direction": "front",
                    "source_side": "front",
                    "target_path": "same_lane",
                    "anchor_region": "front",
                    "road_topology": "straight_lane",
                    "road_layout": "same_lane_following",
                    "lane_context": "same_lane_following",
                    "event_type": "lead_vehicle_braking",
                    "hazard_event_type": "lead_vehicle_braking",
                    "interaction_goal": "braking_pressure",
                    "distance_relation": self._longitudinal_distance(s),
                }
            )
        elif self._is_cut_in(s, actor_type):
            source_side = self._merge_source_side(s)
            slots.update(
                {
                    "actor_type": "vehicle",
                    "actor_role": "merging_actor",
                    "motion_geometry": "merging",
                    "conflict_geometry": "merging",
                    "conflict_direction": f"{source_side}_merge" if source_side in {"left", "right"} else "unknown",
                    "source_side": source_side,
                    "target_path": "ego_lane",
                    "anchor_region": "front" if _contains(s, ["ahead", "front", "directly in front"]) else "unknown",
                    "road_topology": "multi_lane_road",
                    "road_layout": "adjacent_lane_cut_in",
                    "lane_context": "adjacent_lane",
                    "event_type": "lane_change_into_ego_lane",
                    "hazard_event_type": "lane_change_into_ego_lane",
                    "interaction_goal": "merge_into_ego_lane",
                    "distance_relation": self._merge_distance(s),
                }
            )
        elif self._is_vehicle_crossing(s, actor_type):
            direction = self._crossing_direction(s)
            slots.update(
                {
                    "actor_type": "vehicle",
                    "actor_role": "crossing_actor",
                    "motion_geometry": "lateral_crossing",
                    "conflict_geometry": "lateral",
                    "conflict_direction": direction,
                    "source_side": self._source_side_from_direction(direction),
                    "target_path": "ego_lane",
                    "anchor_region": "front",
                    "road_topology": "intersection" if _contains(s, ["intersection", "junction"]) else "straight_lane",
                    "road_layout": "crossing_path",
                    "lane_context": "ego_lane",
                    "event_type": "crossing_path_conflict",
                    "hazard_event_type": "path_crossing",
                    "interaction_goal": "avoid_crossing_actor",
                    "distance_relation": self._lateral_distance(s),
                }
            )
        elif self._is_lateral_crossing(s, actor_type):
            direction = self._crossing_direction(s)
            slots.update(
                {
                    "actor_role": "crossing_actor",
                    "motion_geometry": "lateral_crossing",
                    "conflict_geometry": "lateral",
                    "conflict_direction": direction,
                    "source_side": self._source_side_from_direction(direction),
                    "target_path": self._target_path(s),
                    "anchor_region": "front",
                    "road_topology": self._crossing_road_topology(s),
                    "road_layout": "crossing_ego_lane",
                    "lane_context": "ego_lane",
                    "event_type": "path_crossing" if _contains(s, ["cross", "across", "traverse", "perpendicular"]) else "enter_ego_lane",
                    "hazard_event_type": "path_crossing" if _contains(s, ["cross", "across", "traverse", "perpendicular"]) else "enter_ego_lane",
                    "interaction_goal": self._interaction_goal(s, "avoid_crossing_actor"),
                    "distance_relation": self._lateral_distance(s),
                }
            )
        elif self._is_static_obstacle(s, actor_type):
            construction = self._is_construction(s)
            slots.update(
                {
                    "actor_type": "traffic_object",
                    "actor_role": "blocking_actor" if construction else "static_obstacle",
                    "motion_geometry": "static",
                    "conflict_geometry": "static",
                    "conflict_direction": "front",
                    "source_side": "front",
                    "target_path": "ego_lane",
                    "anchor_region": "front",
                    "road_topology": "construction_zone" if construction else "straight_lane",
                    "road_layout": "lane_blocking" if construction else "object_in_ego_lane",
                    "lane_context": "ego_lane",
                    "event_type": "lane_blocking_conflict" if construction else "object_in_lane",
                    "hazard_event_type": "lane_blocking_conflict" if construction else "object_in_lane",
                    "interaction_goal": "trajectory_blocking" if construction else "avoid_static_obstacle",
                    "distance_relation": "close" if _contains(s, ["ahead", "near"]) else "medium",
                }
            )

        slots["risk_level"] = self._risk_level(s, slots.get("risk_level", "moderate"))
        slots["interaction_goal"] = self._interaction_goal(s, slots.get("interaction_goal", "unknown"))

        return DirectTemplateFrame(
            sentence=sentence,
            semantic_slots=slots,
            event_sequence=self._event_sequence(slots),
            evidence={"sentence": sentence},
            confidence=0.8 if slots["event_type"] != "unknown" else 0.35,
        )

    def _actor_type(self, s: str) -> str:
        if _contains(
            s,
            [
                "pedestrian",
                "walker",
                "person",
                "child",
                "kid",
                "schoolkid",
                "someone",
                "jogger",
                "runner",
                "jaywalker",
                "figure",
                "shopper",
                "commuter",
                "passerby",
                "wheelchair user",
                "road user on foot",
                "on foot",
            ],
        ):
            return "pedestrian"
        if _contains(s, ["cyclist", "bicyclist", "bicycle", "bike", "e-bike", "ebike", "scooter rider", "bicycle rider"]):
            return "cyclist"
        if _contains(
            s,
            [
                "vehicle",
                "car",
                "sedan",
                "suv",
                "hatchback",
                "truck",
                "bus",
                "van",
                "taxi",
                "pickup",
                "traffic ahead",
                "traffic",
            ],
        ):
            return "vehicle"
        if self._is_static_obstacle(s, "unknown"):
            return "traffic_object"
        return "unknown"

    def _is_occluded(self, s: str) -> bool:
        return _contains(
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
        )

    def _occluder_type(self, s: str) -> str:
        if "truck" in s:
            return "truck"
        if "bus" in s:
            return "bus"
        if "van" in s:
            return "van"
        if _contains(s, ["parked car", "parked cars", "parked vehicle", "parked vehicles", "row of parked"]):
            return "parked_vehicle"
        return "vehicle"

    def _is_roundabout(self, s: str) -> bool:
        return _contains(s, ["roundabout", "rotary", "traffic circle", "gyratory", "circular junction", "circular intersection"]) or (
            _contains(
                s,
                [
                    "circulating lane",
                    "circulating vehicle",
                    "circulating stream",
                    "circulatory lane",
                    "inside-lane",
                    "inside the circle",
                    "in the circle",
                    "in the roundabout",
                    "already in the circle",
                    "already in the roundabout",
                    "already inside",
                ],
            )
            and _contains(s, ["entry", "entrance", "approach", "merge", "round entry", "joins", "opening"])
        )

    def _is_left_turn_oncoming(self, s: str) -> bool:
        has_left_turn = _contains(s, ["left turn", "left-turn", "turns left", "turning left", "unprotected left", "left-turning"])
        has_oncoming = _contains(
            s,
            [
                "oncoming",
                "opposing",
                "opposite direction",
                "opposite-direction",
                "opposite lane",
                "opposite traffic",
                "opposite approach",
                "far side",
                "far-side",
                "other direction",
                "coming the other way",
                "through traffic",
                "straight-through traffic",
                "inbound",
                "approaching vehicle",
                "approaching car",
                "fast approaching vehicle",
            ],
        )
        return has_left_turn and has_oncoming

    def _is_lead_braking(self, s: str, actor_type: str) -> bool:
        return actor_type in {"vehicle", "unknown"} and _contains(
            s,
            [
                "brakes",
                "brake",
                "slams on the brakes",
                "slows",
                "slow-down",
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
        )

    def _is_cut_in(self, s: str, actor_type: str) -> bool:
        if actor_type != "vehicle":
            return False
        if self._is_vehicle_crossing(s, actor_type):
            return False
        return _contains(
            s,
            [
                "cuts in",
                "cut in",
                "cuts into",
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
                "takes the ego lane",
                "crosses the lane boundary",
                "crosses the lane marking",
                "across the lane marker",
                "lane marker",
                "lane boundary",
                "moving over",
                "moves laterally",
                "changes lanes",
                "cuts over",
                "merges",
                "merge",
                "slips into",
                "squeezes into",
                "from adjacent",
                "adjacent lane",
                "neighboring left",
                "neighboring right",
                "right-hand lane",
                "left-hand lane",
                "into ego's lane",
                "into the ego lane",
                "occupies part of ego's lane",
            ],
        )

    def _is_vehicle_crossing(self, s: str, actor_type: str) -> bool:
        return actor_type == "vehicle" and _contains(s, ["crosses from", "crosses the path", "cross traffic", "cuts across"]) and not _contains(
            s, ["cuts in", "cut in", "lane boundary", "into ego's lane", "into the ego lane"]
        )

    def _is_lateral_crossing(self, s: str, actor_type: str) -> bool:
        return actor_type in {"pedestrian", "cyclist"} and _contains(
            s,
            [
                "cross",
                "crosses",
                "across",
                "traverses",
                "perpendicular",
                "steps out",
                "steps into",
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
                "enters the travel lane",
                "entering the travel lane",
                "enters the roadway",
                "entering the roadway",
                "enters the road",
                "entering the road",
                "enters the travel path",
                "entering the travel path",
                "enters the lane",
                "entering the lane",
                "enters the ego lane",
                "entering the ego lane",
                "enters ego's path",
                "entering ego's path",
                "enters the ego vehicle's path",
                "entering the ego vehicle's path",
                "enters ego's intended path",
                "entering ego's intended path",
                "enters ego path",
                "entering ego path",
                "stepping into",
                "moving into",
                "moves into",
                "walks into",
                "walking into",
                "appears",
            ],
        )

    def _is_static_obstacle(self, s: str, actor_type: str) -> bool:
        return actor_type == "traffic_object" or _contains(
            s,
            [
                "object in the lane",
                "debris",
                "obstacle",
                "blocked lane",
                "cone",
                "cones",
                "traffic cones",
                "barrier",
                "construction",
                "work zone",
                "work-zone",
            ],
        )

    def _is_construction(self, s: str) -> bool:
        return _contains(s, ["construction", "work zone", "work-zone", "traffic cones", "cone", "cones", "barrier"])

    def _target_path(self, s: str) -> str:
        if _contains(s, ["ego path", "ego's path", "vehicle's path", "trajectory", "intended path", "entry path"]):
            return "ego_path"
        if _contains(s, ["ego lane", "ego's lane", "travel lane", "lane"]):
            return "ego_lane"
        return "ego_lane"

    def _anchor_region(self, s: str) -> str:
        if _contains(s, ["roundabout", "rotary", "traffic circle", "gyratory"]):
            return "roundabout_entry"
        if _contains(s, ["intersection", "junction"]):
            return "intersection"
        if _contains(
            s,
            [
                "ahead",
                "in front",
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
        return "unknown"

    def _crossing_road_topology(self, s: str) -> str:
        if _contains(s, ["intersection", "junction"]):
            return "intersection"
        if _contains(s, ["crosswalk"]):
            return "crosswalk_area"
        return "straight_lane"

    def _crossing_direction(self, s: str) -> str:
        if _contains(s, ["from the right", "right side", "right curb", "right sidewalk"]):
            return "right_to_left"
        if _contains(s, ["from the left", "left side", "left curb", "left sidewalk"]):
            return "left_to_right"
        return "crossing"

    def _source_side_from_direction(self, direction: str) -> str:
        if direction == "right_to_left":
            return "right"
        if direction == "left_to_right":
            return "left"
        return "unknown_side"

    def _merge_source_side(self, s: str) -> str:
        if _contains(s, ["from the left", "left lane", "adjacent left", "neighboring left", "left-hand lane", "left-to-right"]):
            return "left"
        if _contains(s, ["from the right", "right lane", "right-hand lane", "adjacent right", "neighboring right"]):
            return "right"
        return "unknown"

    def _speed_relation(self, s: str) -> str:
        if _contains(s, ["fast", "quickly", "rapidly", "high speed", "approaches quickly"]):
            return "fast_approach"
        if _contains(s, ["normal speed", "normally"]):
            return "normal"
        if _contains(s, ["slowly", "mildly"]):
            return "slow"
        return "unknown"

    def _risk_level(self, s: str, default: str = "moderate") -> str:
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
                "slams on the brakes",
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
        if _contains(s, ["mild", "mildly", "slowly", "comfortable", "enough reaction distance", "enough distance", "medium range"]):
            return "mild"
        return default

    def _interaction_goal(self, s: str, default: str) -> str:
        if _contains(s, ["high collision risk", "collision risk", "dangerous"]):
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

    def _merge_distance(self, s: str) -> str:
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
            ],
        ):
            return "small_gap"
        if _contains(s, ["comfortable gap", "enough gap", "medium gap", "medium range"]):
            return "medium"
        if _contains(s, ["close", "nearby"]):
            return "close"
        return "close"

    def _longitudinal_distance(self, s: str) -> str:
        if _contains(s, ["short headway", "very short headway", "shrinking headway"]):
            return "short_headway"
        if _contains(s, ["medium range"]):
            return "medium"
        if _contains(s, ["nearly stops", "close following conflict"]):
            return "close"
        if _contains(s, ["compresses", "closes the gap", "little gap", "small gap", "following distance", "short following distance", "short following gap"]):
            return "small_gap"
        if _contains(s, ["panic", "sharply", "abruptly", "heavily", "without warning", "rapidly", "slams on the brakes", "decelerates hard"]):
            return "close"
        return "medium"

    def _lateral_distance(self, s: str) -> str:
        if _contains(s, ["close to", "very close", "near-miss", "near miss", "near the ego"]):
            return "close"
        return "close" if _contains(s, ["ahead", "front", "ego path", "ego lane"]) else "medium"

    def _roundabout_distance(self, s: str) -> str:
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
        if _contains(s, ["passes across", "sweeps past", "in front of ego", "already in the roundabout", "already circulating", "cross traffic", "close ahead", "passes close", "just before ego enters"]):
            return "close"
        return "medium"

    def _obstacle_type(self, s: str) -> str:
        if _contains(s, ["cone", "cones", "traffic cones"]):
            return "cone"
        if _contains(s, ["barrier"]):
            return "barrier"
        if self._is_construction(s):
            return "construction_zone"
        if _contains(s, ["debris"]):
            return "debris"
        if _contains(s, ["obstacle", "object"]):
            return "object"
        return "unknown"

    def _event_sequence(self, slots: Dict[str, Any]) -> List[Dict[str, Any]]:
        ego_action = slots.get("ego_maneuver", "drive_forward")
        hazard_event = slots.get("hazard_event_type", slots.get("event_type", "unknown"))
        return [
            {"order": 1, "actor": "ego", "event_type": f"ego_{ego_action}", "action": ego_action, "relation_to_previous": "start"},
            {"order": 2, "actor": slots.get("actor_type", "hazard_actor"), "event_type": hazard_event, "action": hazard_event, "relation_to_previous": "after_ego_baseline"},
            {"order": 3, "actor": "ego_and_hazard_actor", "event_type": "conflict_point_approach", "action": slots.get("conflict_geometry", "approach_conflict"), "relation_to_previous": "after_hazard_event"},
            {"order": 4, "actor": "ego", "event_type": "ego_response", "action": "brake_or_yield", "relation_to_previous": "after_conflict_approach"},
        ]


class DirectTemplateMapper:
    """Map direct-template semantic slots to both top-level and detailed-spec layers."""

    def map(self, frame: DirectTemplateFrame) -> Dict[str, Any]:
        slots = deepcopy(frame.semantic_slots)
        actor_layer = self._actor_layer(slots)
        interaction_layer = self._interaction_layer(slots)
        motion_layer = self._motion_layer(slots)
        object_layer = self._object_layer(slots)
        road_layer = self._road_layer(slots)
        risk_layer = {
            "risk_level": slots.get("risk_level", "moderate"),
            "collision_allowed": bool(slots.get("collision_allowed", False)),
        }
        validation_layer = {
            "require_visibility_match": bool(slots.get("occlusion_enabled", False)),
            "composition_policy": "semantic_slots_not_full_scene_template",
            "pipeline_design": "direct_template_baseline_independent",
        }
        parameter_layer = self._parameter_layer(slots)

        detailed_spec = {
            "actor_layer": deepcopy(actor_layer),
            "interaction_layer": deepcopy(interaction_layer),
            "motion_layer": deepcopy(motion_layer),
            "object_layer": deepcopy(object_layer),
            "road_layer": deepcopy(road_layer),
            "risk_layer": deepcopy(risk_layer),
            "validation_layer": deepcopy(validation_layer),
            "parameter_layer": deepcopy(parameter_layer),
        }

        return {
            "schema_version": "direct_template_baseline_spec",
            "canonical_type": self._canonical_type(slots, actor_layer, interaction_layer, road_layer),
            "semantic_slots": slots,
            "actor_layer": actor_layer,
            "interaction_layer": interaction_layer,
            "motion_layer": motion_layer,
            "event_layer": {
                "event_sequence": deepcopy(frame.event_sequence),
                "event_sequence_labels": [
                    f"{step['order']}:{step['actor']}:{step['event_type']}:{step['action']}"
                    for step in frame.event_sequence
                ],
                "num_events": len(frame.event_sequence),
            },
            "object_layer": object_layer,
            "road_layer": road_layer,
            "risk_layer": risk_layer,
            "validation_layer": validation_layer,
            "parameter_layer": parameter_layer,
            "detailed_spec": detailed_spec,
            "evidence": deepcopy(frame.evidence),
            "confidence": frame.confidence,
        }

    def _actor_layer(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        actor_type = slots.get("actor_type", "unknown")
        role = slots.get("actor_role", "unknown")
        primary_actor = actor_type
        if role == "merging_actor":
            primary_actor = "cutin_vehicle"
        elif role == "braking_actor":
            primary_actor = "lead_vehicle"
        elif role == "blocking_actor":
            primary_actor = "static_obstacle"
        return {"primary_actor": primary_actor, "actor_role": role, "base_actor_type": actor_type}

    def _interaction_layer(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        geometry = slots.get("conflict_geometry", "unknown")
        conflict_type = {
            "lateral": "crossing_path_conflict" if slots.get("actor_type") == "vehicle" else "lateral_conflict",
            "merging": "merging_conflict",
            "longitudinal": "longitudinal_conflict",
            "crossing_path": "oncoming_conflict",
            "static": "lane_blocking_conflict" if slots.get("actor_role") == "blocking_actor" else "static_obstacle_conflict",
        }.get(geometry, "unknown")
        return {
            "conflict_type": conflict_type,
            "anchor_region": slots.get("anchor_region", "unknown"),
            "conflict_direction": slots.get("conflict_direction", "unknown"),
            "distance_relation": slots.get("distance_relation", "unknown"),
            "interaction_goal": slots.get("interaction_goal", "unknown"),
            "speed_relation": slots.get("speed_relation", "unknown"),
            "source_relation": self._source_relation(slots.get("source_side", "unknown")),
            "target_relation": slots.get("target_path", "unknown"),
        }

    def _source_relation(self, source_side: Any) -> Any:
        if source_side == "left":
            return "from_left"
        if source_side == "right":
            return "from_right"
        if source_side == "opposite":
            return "from_opposite_direction"
        if source_side == "roundabout_inside":
            return "from_circulating_lane"
        return source_side

    def _motion_layer(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        motion_geometry = slots.get("motion_geometry", "unknown")
        motion_axis = {
            "lateral_crossing": "lateral",
            "lane_change": "merging",
            "longitudinal_braking": "longitudinal",
        }.get(motion_geometry, motion_geometry)
        return {
            "hazard_event_type": slots.get("hazard_event_type", slots.get("event_type", "unknown")),
            "motion_axis": motion_axis,
            "motion_direction": slots.get("conflict_direction", "unknown"),
            "path_or_object": slots.get("target_path", "unknown"),
            "ego_maneuver": slots.get("ego_maneuver", "drive_forward"),
        }

    def _object_layer(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        occluded = bool(slots.get("occlusion_enabled", False))
        return {
            "occlusion": {
                "enabled": occluded,
                "occluder_type": slots.get("occluder_type", "unknown"),
                "occlusion_level": "partial" if occluded else "none",
            },
            "static_obstacle": {
                "enabled": slots.get("motion_geometry") == "static",
                "obstacle_type": slots.get("obstacle_type", "unknown"),
            },
        }

    def _road_layer(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        topology = slots.get("road_topology", "unknown")
        layout = slots.get("road_layout", "unknown")
        return {
            "road_type": topology,
            "road_topology": topology,
            "generated_road_layout": layout,
            "lane_context": slots.get("lane_context", layout),
            "anchor_type": "intersection_center" if topology == "intersection" and layout == "unprotected_left_turn" else slots.get("anchor_region", "unknown"),
            "has_crosswalk": bool(slots.get("has_crosswalk", False)) or layout == "crosswalk_area",
        }

    def _parameter_layer(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        completed: Dict[str, Dict[str, Any]] = {
            "trigger_condition": _slot(self._trigger_condition(slots), reason="direct-template baseline trigger"),
        }
        if slots.get("source_side") in {"left", "right", "roundabout_inside", "front", "opposite"}:
            completed["source_side"] = _slot(slots["source_side"], reason="source side inferred from direct-template slots")
        if slots.get("ego_maneuver") == "left_turn":
            completed["ego_maneuver"] = _slot("left_turn", reason="ego maneuver inferred from direct-template slots")
        return {
            "required_missing": ["ego_speed", "actor_speed", "initial_distance"],
            "defaultable_missing": [],
            "distributional_defaults": {},
            "completed": completed,
            "completion_policy": "direct_template_baseline_completion",
        }

    def _trigger_condition(self, slots: Dict[str, Any]) -> str:
        event_type = slots.get("hazard_event_type", "unknown")
        if event_type in {"path_crossing", "enter_ego_lane"}:
            return "actor_enters_ego_lane_ahead"
        if event_type == "lane_change_into_ego_lane":
            return "actor_crosses_lane_boundary_into_ego_lane"
        if event_type == "lead_vehicle_braking":
            return "lead_vehicle_decelerates"
        if event_type == "left_turn_across_oncoming":
            return "ego_turn_path_intersects_oncoming_path"
        if event_type == "roundabout_entry_conflict":
            return "ego_attempts_entry_while_circulating_vehicle_closes_gap"
        if event_type == "lane_blocking_conflict":
            return "ego_approaches_blocked_lane"
        return "hazard_event_becomes_relevant_to_ego_path"

    def _canonical_type(self, slots: Dict[str, Any], actor_layer: Dict[str, Any], interaction_layer: Dict[str, Any], road_layer: Dict[str, Any]) -> str:
        if road_layer.get("road_topology") == "intersection" and road_layer.get("generated_road_layout") == "unprotected_left_turn":
            return "Unprotected-Left-Turn-Oncoming"
        if road_layer.get("road_topology") == "roundabout":
            return "Roundabout-Entry-Merge"
        if interaction_layer.get("conflict_type") == "merging_conflict":
            return "Vehicle-Cut-In"
        if interaction_layer.get("conflict_type") == "longitudinal_conflict":
            return "Lead-Vehicle-Braking"
        if actor_layer.get("primary_actor") in {"pedestrian", "cyclist"} and interaction_layer.get("conflict_type") == "lateral_conflict":
            return "Pedestrian-Cyclist-Crossing"
        if interaction_layer.get("conflict_type") == "lane_blocking_conflict":
            return "Construction-Lane-Blocking"
        return "Unknown"


class DirectTemplateBaseline:
    """End-to-end direct-template baseline pipeline."""

    def __init__(
        self,
        *,
        llm_provider: str = "none",
        llm_model: str = "qwen2.5:7b",
        ollama_url: str = "http://127.0.0.1:11434",
        allow_fallback: bool = True,
    ) -> None:
        self.allow_fallback = allow_fallback
        self.client: Optional[DirectTemplateOllamaClient] = None
        if llm_provider == "ollama":
            self.client = DirectTemplateOllamaClient(model=llm_model, url=ollama_url)
        elif llm_provider not in {"none", "fallback"}:
            raise ValueError(f"Unsupported llm_provider: {llm_provider}")
        self.parser = DirectTemplateParser(
            llm_provider=llm_provider,
            llm_model=llm_model,
            ollama_url=ollama_url,
            allow_fallback=allow_fallback,
        )
        self.mapper = DirectTemplateMapper()

    def parse(self, sentence: str) -> DirectTemplateFrame:
        return self.parser.parse(sentence)

    def parse_to_spec(self, sentence: str) -> Tuple[DirectTemplateFrame, Dict[str, Any]]:
        if self.client is not None:
            try:
                spec = self._llm_generate_spec(sentence)
                ok, errors = validate_direct_template_spec(spec)
                if not ok:
                    raise ValueError(f"invalid_direct_template_spec:{errors}")
                frame = self._frame_from_spec(sentence, spec, "parsed_by=direct_template_llm")
                return frame, spec
            except Exception as exc:
                if not self.allow_fallback:
                    raise
                frame = self.parse(sentence)
                frame.parser_notes = f"direct_template_llm_failed={type(exc).__name__}: {exc}; parsed_by=direct_template_baseline_rules"
                return frame, self.mapper.map(frame)

        frame = self.parse(sentence)
        return frame, self.mapper.map(frame)

    def _llm_generate_spec(self, sentence: str) -> Dict[str, Any]:
        if self.client is None:
            raise RuntimeError("LLM client is not configured")
        prompt = DIRECT_TEMPLATE_PROMPT.replace("<<SENTENCE>>", sentence)
        response = self.client.generate(prompt, temperature=0.0)
        spec = _extract_json_object(response)
        spec.setdefault("schema_version", "direct_template_baseline_spec")
        spec.setdefault("evidence", {"sentence": sentence})
        spec.setdefault("confidence", 0.65)
        return spec

    def _frame_from_spec(self, sentence: str, spec: Dict[str, Any], parser_notes: str) -> DirectTemplateFrame:
        slots = dict(spec.get("semantic_slots") or {})
        event_layer = spec.get("event_layer") or {}
        sequence = event_layer.get("event_sequence") or []
        return DirectTemplateFrame(
            sentence=sentence,
            semantic_slots=slots,
            event_sequence=sequence if isinstance(sequence, list) else [],
            evidence=dict(spec.get("evidence") or {"sentence": sentence}),
            confidence=float(spec.get("confidence", 0.65) or 0.65),
            parser_notes=parser_notes,
        )


def validate_direct_template_spec(spec: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    flat = flatten_dict(spec)
    required = [
        "semantic_slots.actor_type",
        "semantic_slots.actor_role",
        "semantic_slots.conflict_geometry",
        "detailed_spec.actor_layer.primary_actor",
        "detailed_spec.actor_layer.actor_role",
        "detailed_spec.interaction_layer.conflict_type",
        "detailed_spec.risk_layer.collision_allowed",
        "event_layer.num_events",
    ]
    for key in required:
        value = flat.get(key, "__MISSING__")
        if value is None or value == "" or value == "unknown" or value == "__MISSING__":
            errors.append(f"missing_or_unknown:{key}")
    if flat.get("event_layer.num_events", 0) <= 0:
        errors.append("event_sequence_required")
    return len(errors) == 0, errors
