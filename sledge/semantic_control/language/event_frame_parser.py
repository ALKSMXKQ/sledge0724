"""LLM and fallback parser from natural language to EventFrame v2."""

from __future__ import annotations

from typing import Any, Dict
import json
import re
import urllib.request

from sledge.semantic_control.language.event_frame import EventFrame, extract_json_object
from sledge.semantic_control.language.narrative_semantics import (
    HazardFocusResolver,
    NarrativeDecomposer,
    candidate_to_event_frame_dict,
    extract_raw_json_object,
    narrative_analysis_from_dict,
)


NARRATIVE_CANDIDATE_PROMPT = r"""
You are the first-stage semantic parser for autonomous-driving scene generation.
Do NOT output the final simulator template. Do NOT directly choose an EventFrame.

Your job is to decompose the user's sentence into candidate actors and candidate
events, mark background/negated/occluder roles, and identify the event that
should drive scene generation.

Return ONLY one JSON object:
{
  "sentence": "original sentence",
  "clauses": [
    {
      "text": "clause text",
      "cue_before": "but | however | instead | while | then | after that | seconds later | empty"
    }
  ],
  "candidate_events": [
    {
      "event_id": "event_1",
      "clause_index": 0,
      "clause_text": "exact evidence clause",
      "actor": {
        "text": "surface actor phrase",
        "actor_class": "human_on_foot | cyclist | vehicle | traffic_object | unknown",
        "actor_role": "crossing_actor | merging_actor | braking_actor | approaching_actor | blocking_actor | static_obstacle | occluder | background_actor | unknown"
      },
      "event_type": "path_crossing | enter_ego_lane | lane_change_into_ego_lane | lead_vehicle_braking | ego_left_turn_across_oncoming | roundabout_entry_conflict | lane_blocking_conflict | object_in_lane | background_context | unknown",
      "motion_axis": "lateral | longitudinal | merging | oncoming | static | unknown",
      "source_relation": "from_left | from_right | from_curb | from_adjacent_lane | from_opposite_direction | from_circulating_lane | unknown",
      "target_relation": "ego_lane | ego_path | intersection | roundabout_entry | unknown",
      "road_type": "straight_lane | intersection | roundabout | construction_zone | curbside | unknown",
      "ego_maneuver": "drive_forward | left_turn | enter_roundabout",
      "occlusion_enabled": false,
      "occluder_type": "vehicle | parked_vehicle | bus | truck | van | unknown",
      "obstacle_type": "cone | barrier | construction_zone | debris | object | unknown",
      "negated": false,
      "background": false,
      "hazard_likelihood": "high | medium | low",
      "evidence_text": "short evidence quote"
    }
  ],
  "selected_hazard_event_id": "event id for the main hazard",
  "selection_reason": "brief explanation; no hidden chain-of-thought"
}

Ontology decision rules:
- A waiting, standing, parked, strapped, safe, or explicitly negated actor is background.
- "no pedestrian crossing", "not crossing", "do not create", "ignore", and
  "stays in its lane" negate that candidate event.
- "actual hazard", "critical event", "instead", "but", "however", and
  "obstacle to generate" often identify the clause that should drive generation.
- If a truck/bus/van/parked vehicle only hides another actor, mark it as occluder,
  not the hazard actor.
- "ego slows/brakes for a pedestrian" is NOT lead_vehicle_braking. It is usually
  a pedestrian/cyclist crossing or yielding conflict.
- lead_vehicle_braking requires a lead/front/ahead vehicle or traffic ahead.
- lane_change_into_ego_lane requires a vehicle changing/merging from an adjacent
  lane; a pedestrian stepping into the ego lane is enter_ego_lane, not cut-in.
- If multiple candidates exist, select the non-negated non-background candidate
  with the strongest hazard evidence.

Hard negative examples:
1. "A pedestrian waits safely, but the lead car brakes hard."
   pedestrian waiting = background; lead car braking = selected hazard.
2. "No pedestrian crossing; instead traffic cones occupy the ego lane."
   pedestrian crossing = negated; traffic cones blocking lane = selected hazard.
3. "The truck is only an occluder and a child comes out from behind it."
   truck = occluder; child entering ego path = selected hazard.
4. "Ego slows for a pedestrian in the crosswalk."
   ego slowing is not lead vehicle braking; pedestrian/crosswalk event is selected.

Sentence:
<<SENTENCE>>
""".strip()


EVENT_FRAME_PROMPT = r"""
You are a semantic role parser for autonomous-driving scene descriptions.
Do NOT choose the final simulator scenario template.  Decompose the sentence into
entities, events, relations, event order, and missing information.

Return ONLY one JSON object matching this schema:
{
  "sentence": "original input",
  "main_actor": {
    "text": "surface phrase for the moving/hazard actor",
    "actor_class": "human_on_foot | cyclist | vehicle | traffic_object | unknown",
    "actor_role": "hazard_actor | ego_vehicle | occluder | unknown",
    "evidence_text": "short quote supporting actor"
  },
  "main_event": {
    "predicate_text": "surface predicate, e.g. cuts across",
    "event_type": "path_crossing | enter_ego_lane | lane_change_into_ego_lane | lead_vehicle_braking | hard_stop_ahead | ego_left_turn_across_oncoming | oncoming_through_conflict | roundabout_entry_conflict | object_in_lane | unknown",
    "path_or_object": "lane/path/object involved, e.g. ego lane",
    "motion_axis": "lateral | longitudinal | oncoming | merging | turning | static | unknown",
    "motion_direction": "natural label, e.g. across_lane/from_left_to_right/into_ego_lane",
    "event_location_relation": "ahead_of | in_front_of | behind | left_of | right_of | at_intersection | at_roundabout_entry | unknown",
    "location_relation_function": "event_anchor | actor_position | motion_direction | source_lane | target_lane | unknown",
    "source_relation": "from_left | from_right | from_curb | from_adjacent_lane | from_opposite_direction | unknown",
    "target_relation": "ego_lane | ego_path | intersection | roundabout_entry | unknown",
    "evidence_text": "short quote supporting event"
  },
  "ego_event": {
    "ego_maneuver": "drive_forward | left_turn | enter_roundabout | yield | brake | unknown",
    "evidence_text": "short quote supporting ego maneuver"
  },
  "road_context": {
    "road_type": "straight_lane | intersection | roundabout | curbside | unknown",
    "lane_context": "ego_lane | adjacent_lane | opposing_lane | roundabout_entry | unknown",
    "evidence_text": "short quote supporting road context"
  },
  "occlusion": {
    "enabled": false,
    "occluder_type": "vehicle | parked_vehicle | bus | truck | van | roadside_object | unknown",
    "relation_to_actor": "behind | screened_by | hidden_by | masked_by | blind_side | unknown",
    "evidence_text": "short quote supporting occlusion"
  },
  "event_sequence": [
    {
      "order": 1,
      "actor": "ego or hazard actor",
      "event_type": "ego_driving | path_crossing | lane_change_into_ego_lane | lead_vehicle_braking | ego_left_turn_across_oncoming | roundabout_entry_conflict | conflict_point_approach | ego_response | unknown",
      "action": "short action label",
      "relation_to_previous": "start | after_ego_baseline | after_hazard_event | after_conflict_approach | unknown",
      "evidence_text": "quote if explicit, empty if inferred"
    }
  ],
  "diagnostics": {
    "is_path_crossing": false,
    "is_longitudinal_following": false,
    "is_lane_change_into_ego_lane": false,
    "is_ego_left_turn": false,
    "is_roundabout_entry": false,
    "is_occluded": false
  },
  "missing_information": {
    "required": ["ego_speed", "actor_speed", "initial_distance"],
    "defaultable": [],
    "distributional": {
      "ego_speed_mps": [5, 15],
      "actor_speed_mps": [1, 2]
    }
  },
  "completed_parameters": {},
  "confidence": 0.0,
  "parser_notes": "brief explanation only; do not include hidden chain-of-thought"
}

Important semantic distinctions:
- actor_class is the physical actor type; actor_role is its role in the hazard.
- predicate_text is the verb phrase only, such as "cuts across", "brakes", or "swerves into".
- path_or_object is the lane/path/object affected by the predicate, not a spatial modifier.
- event_location_relation is where the event/actor is relative to ego or road context.
- location_relation_function explains how to use that relation:
  event_anchor = where the event happens; actor_position = where a lead/static actor is;
  source_lane = where a merge starts; target_lane = where the actor moves into;
  motion_direction = direction of travel.
- motion_axis is the conflict geometry: lateral crossing, longitudinal following,
  merging/cut-in, oncoming, turning, or static.
- Phrases like "ahead of ego", "in front of ego", and "just ahead" often describe
  where the whole lateral event occurs. In that case set location_relation_function = "event_anchor".
- Those phrases do NOT by themselves imply longitudinal following conflict.
- Conflict geometry is determined mainly by event_type + motion_axis + event_sequence.
- "cuts across", "crosses", "traverses", "moves perpendicular to" indicate lateral path crossing.
- "vehicle ahead slows/brakes/stops" indicates longitudinal following conflict.
- "from adjacent lane into ego lane", "cuts in", "swerves into ego lane" indicates merging/cut-in.
- "ego turns left" + "oncoming/opposing vehicle goes straight" indicates left-turn oncoming conflict.
- If a sentence contains multiple actors/events, first identify all candidate events.
- Do not make a safe, negated, waiting, standing, parked, or background actor the main_actor.
- Cues such as "actual hazard", "critical event", "instead", "but", "however",
  and "obstacle to generate" usually indicate the clause that should drive the
  generated scene.
- Phrases such as "no pedestrian crossing", "do not create a cut-in", "ignore
  the pedestrian", or "stays in its lane" negate that candidate event.
- Objects described as "only an occluder" should fill occlusion fields, not
  become the hazard actor.
- For short inputs, event_sequence may be empty; a deterministic rule layer will rebuild it.
- Missing information should list controllable lane, actor, motion, and interaction
  parameters needed by simulation but absent from the sentence.
- Do not output weather, time_of_day, road_friction, lighting, or other environment
  parameters; this evaluator only controls lanes and traffic participants.

Contrastive examples:
Sentence: A walker cuts across the lane just ahead of ego.
Interpretation:
- actor = walker, class = human_on_foot
- predicate = cuts across
- path_or_object = lane / ego lane
- event_type = path_crossing, motion_axis = lateral
- just ahead of ego = event_anchor, not longitudinal following
- event_sequence: ego drives forward -> walker crosses ego lane ahead -> ego approaches conflict point -> ego may brake/yield.

Sentence: A vehicle ahead slows down in the ego lane.
Interpretation:
- actor = vehicle, class = vehicle
- predicate = slows down
- ahead of ego = actor_position in same lane
- event_type = lead_vehicle_braking, motion_axis = longitudinal
- event_sequence: ego follows lead vehicle -> lead vehicle decelerates -> headway closes -> ego braking required.

Now parse this sentence:
<<SENTENCE>>
""".strip()


class OllamaClient:
    def __init__(self, model: str, url: str = "http://127.0.0.1:11434", timeout: int = 60):
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


class EventFrameParser:
    def __init__(
        self,
        *,
        llm_provider: str = "none",
        llm_model: str = "qwen2.5:7b",
        ollama_url: str = "http://127.0.0.1:11434",
        allow_fallback: bool = True,
        timeout: int = 60,
    ):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.allow_fallback = allow_fallback
        self.client = None
        if llm_provider == "ollama":
            self.client = OllamaClient(model=llm_model, url=ollama_url, timeout=timeout)
        elif llm_provider not in {"none", "fallback"}:
            raise ValueError(f"Unsupported llm_provider: {llm_provider}")

    def parse(self, sentence: str) -> EventFrame:
        if self.client is not None:
            try:
                return self._parse_with_llm_candidates(sentence)
            except Exception as exc:
                candidate_error = exc
                try:
                    return self._parse_with_legacy_eventframe_prompt(sentence, candidate_error=candidate_error)
                except Exception as legacy_exc:
                    if not self.allow_fallback:
                        raise legacy_exc
                    frame = self.fallback_parse(sentence)
                    frame.parser_notes = (
                        f"llm_candidate_failed={type(candidate_error).__name__}: {candidate_error}; "
                        f"llm_eventframe_failed={type(legacy_exc).__name__}: {legacy_exc}; "
                        "parsed_by=fallback"
                    )
                    return frame

        frame = self.fallback_parse(sentence)
        frame.parser_notes = "parsed_by=fallback"
        return frame

    def _parse_with_llm_candidates(self, sentence: str) -> EventFrame:
        if self.client is None:
            raise RuntimeError("LLM client is not configured")
        prompt = NARRATIVE_CANDIDATE_PROMPT.replace("<<SENTENCE>>", sentence)
        response = self.client.generate(prompt, temperature=0.0)
        data = extract_raw_json_object(response)
        analysis = narrative_analysis_from_dict(sentence, data)
        analysis = HazardFocusResolver().resolve(analysis)
        if analysis.selected_event is None:
            raise ValueError("LLM candidate parse produced no selected hazard event")
        frame = EventFrame.from_dict(
            candidate_to_event_frame_dict(sentence, analysis, analysis.selected_event)
        )
        frame.parser_notes = (
            frame.parser_notes + " | parsed_by=llm_candidate_events"
        ).strip(" |")
        return frame

    def _parse_with_legacy_eventframe_prompt(self, sentence: str, *, candidate_error: Exception) -> EventFrame:
        if self.client is None:
            raise RuntimeError("LLM client is not configured")
        prompt = EVENT_FRAME_PROMPT.replace("<<SENTENCE>>", sentence)
        response = self.client.generate(prompt, temperature=0.0)
        data = extract_json_object(response)
        data.setdefault("sentence", sentence)
        frame = EventFrame.from_dict(data)
        frame.parser_notes = (
            frame.parser_notes
            + f" | parsed_by=legacy_eventframe_llm_after_candidate_failure={type(candidate_error).__name__}"
        ).strip(" |")
        return frame

    @staticmethod
    def fallback_parse(sentence: str) -> EventFrame:
        """Small deterministic bootstrap parser.

        This is not the main semantic method. It exists to keep experiments runnable
        when the LLM is unavailable and to repair obvious cases. It still writes an
        EventFrame first; the mapper never maps directly from raw words to final slots.
        """
        analysis = HazardFocusResolver().resolve(NarrativeDecomposer().analyze(sentence))
        if analysis.selected_event is not None:
            return EventFrame.from_dict(
                candidate_to_event_frame_dict(sentence, analysis, analysis.selected_event)
            )
        partial = EventFrameParser._partial_control_frame(sentence)
        if partial is not None:
            return partial

        s = sentence.lower()

        def has(*terms: str) -> bool:
            return any(t in s for t in terms)

        actor_text = "unknown actor"
        actor_class = "unknown"

        if has(
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
        ):
            actor_text = "pedestrian" if "pedestrian" in s else "walker/person"
            actor_class = "human_on_foot"
        elif has("cyclist", "bicyclist", "bike", "bicycle", "e-bike", "ebike", "scooter rider", "bicycle rider"):
            actor_text = "cyclist"
            actor_class = "cyclist"
        elif has("sedan", "car", "vehicle", "suv", "hatchback", "truck", "bus", "van", "taxi", "pickup", "traffic ahead", "traffic"):
            actor_text = "vehicle"
            actor_class = "vehicle"
        elif has("object", "debris", "barrier", "cone", "cones", "construction", "work zone", "work-zone"):
            actor_text = "traffic object"
            actor_class = "traffic_object"

        occluded = has(
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
        )

        occluder_type = "unknown"
        for t in ["box truck", "delivery truck", "truck", "bus", "van", "suv", "parked cars", "parked car", "parked vehicle", "vehicle"]:
            if t in s:
                occluder_type = "parked_vehicle" if "parked" in t else t.replace(" ", "_")
                if occluder_type in {"box_truck", "delivery_truck"}:
                    occluder_type = "truck"
                break

        event_type = "unknown"
        motion_axis = "unknown"
        motion_direction = "unknown"
        relation_function = "unknown"
        loc_relation = "unknown"
        loc_ref = "ego"
        source_relation = "unknown"
        target_relation = "unknown"
        ego_maneuver = "unknown"
        road_type = "unknown"
        lane_context = "unknown"

        if has("roundabout", "rotary", "traffic circle", "gyratory", "circular junction", "circular intersection") or (
            has(
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
            )
            and has("entry", "entrance", "approach", "merge", "round entry", "joins", "opening")
        ):
            event_type = "roundabout_entry_conflict"
            motion_axis = "merging"
            motion_direction = "circulating_across_entry"
            loc_relation = "at_roundabout_entry"
            relation_function = "event_anchor"
            target_relation = "roundabout_entry"
            ego_maneuver = "enter_roundabout"
            road_type = "roundabout"
            lane_context = "roundabout_entry"
            actor_class = "vehicle"
            actor_text = "circulating vehicle"

        elif has("left turn", "left-turn", "turns left", "turning left", "unprotected left"):
            loc_relation = "at_intersection"
            loc_ref = "intersection"
            relation_function = "event_anchor"
            target_relation = "intersection"
            ego_maneuver = "left_turn"
            road_type = "intersection"

            if has(
                "oncoming",
                "opposing",
                "opposite direction",
                "opposite-direction",
                "opposite lane",
                "far side",
                "other direction",
                "coming the other way",
                "opposite traffic",
                "toward ego",
                "approaching vehicle",
                "approaching car",
                "fast approaching vehicle",
                "through traffic",
                "straight-through traffic",
                "inbound",
                "opposite approach",
                "keeps coming",
                "through vehicle",
            ):
                event_type = "ego_left_turn_across_oncoming"
                motion_axis = "oncoming"
                motion_direction = "opposing_through"
                source_relation = "from_opposite_direction"
                lane_context = "opposing_lane"
                actor_class = "vehicle"
                actor_text = "oncoming vehicle"

        elif (
            actor_class == "vehicle"
            and has("cross", "crosses", "cross traffic", "cuts across")
            and has("intersection", "junction", "from the right side", "from the left side")
            and not has("cuts in", "cut in", "lane boundary", "into ego's lane", "into the ego lane")
        ):
            event_type = "path_crossing"
            motion_axis = "lateral"
            motion_direction = "across_ego_path"
            relation_function = "event_anchor" if has("ahead", "in front") else "target_lane"
            target_relation = "ego_lane"
            road_type = "intersection" if has("intersection", "junction") else "straight_lane"
            lane_context = "ego_lane"
            loc_relation = "at_intersection" if road_type == "intersection" else "unknown"
            if has("from the right", "right side"):
                source_relation = "from_right"
            elif has("from the left", "left side"):
                source_relation = "from_left"

        elif has(
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
        ) and actor_class == "vehicle":
            event_type = "lane_change_into_ego_lane"
            motion_axis = "merging"
            motion_direction = "into_ego_lane"
            target_relation = "ego_lane"
            relation_function = "target_lane"
            lane_context = "adjacent_lane"

            if has("from the left", "from the left lane", "left lane", "left side", "adjacent left", "neighboring left", "left-hand lane", "left-to-right"):
                source_relation = "from_left"
                loc_relation = "left_of"
            elif has("from the right", "from the right lane", "right lane", "right-hand lane", "right side", "adjacent right", "neighboring right"):
                source_relation = "from_right"
                loc_relation = "right_of"
            else:
                source_relation = "from_adjacent_lane"

        elif has(
            "brakes",
            "brake",
            "slams on the brakes",
            "slows",
            "slow-down",
            "sheds speed",
            "drops speed",
            "loses speed",
            "panic stop",
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
        ) and actor_class == "vehicle":
            event_type = "lead_vehicle_braking" if not has("halt", "panic stop", "hard stop", "sudden stop") else "hard_stop_ahead"
            motion_axis = "longitudinal"
            motion_direction = "decelerating_ahead"
            loc_relation = "ahead_of"
            relation_function = "actor_position"
            target_relation = "ego_lane"
            road_type = "straight_lane"
            lane_context = "ego_lane"

        elif has(
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
            "entering the ego vehicle's path",
            "moving into",
            "moves into",
            "walks into",
            "walking into",
        ):
            event_type = (
                "path_crossing"
                if has(
                    "cross",
                    "crosses",
                    "across",
                    "traverses",
                    "perpendicular",
                    "runs across",
                    "shoots across",
                    "hurries across",
                    "walks across",
                )
                else "enter_ego_lane"
            )
            motion_axis = "lateral"
            motion_direction = "across_ego_path"
            target_relation = "ego_path" if has("path", "trajectory") else "ego_lane"
            relation_function = "event_anchor" if has("ahead", "in front", "forward path", "just ahead") else "target_lane"

            if has("ahead", "just ahead"):
                loc_relation = "ahead_of"
            elif has("in front", "forward path"):
                loc_relation = "in_front_of"
            else:
                loc_relation = "unknown"

            if has("curb", "roadside", "sidewalk"):
                source_relation = "from_curb"
                road_type = "curbside"
            elif has("from the left", "left side", "left curb", "left sidewalk"):
                source_relation = "from_left"
            elif has("from the right", "right side", "right curb", "right sidewalk"):
                source_relation = "from_right"

            lane_context = "ego_lane"

        elif actor_class == "traffic_object":
            event_type = "object_in_lane"
            motion_axis = "static"
            loc_relation = "ahead_of" if has("ahead", "front") else "unknown"
            relation_function = "actor_position"
            target_relation = "ego_lane"
            lane_context = "ego_lane"

        data: Dict[str, Any] = {
            "sentence": sentence,
            "main_actor": {
                "text": actor_text,
                "actor_class": actor_class,
                "actor_role": "hazard_actor",
                "evidence_text": sentence,
            },
            "main_event": {
                "predicate_text": EventFrameParser._guess_predicate(sentence),
                "event_type": event_type,
                "path_or_object": (
                    "ego lane"
                    if "lane" in s
                    else "ego path"
                    if "path" in s or "trajectory" in s
                    else "unknown"
                ),
                "motion_axis": motion_axis,
                "motion_direction": motion_direction,
                "event_location_relation": loc_relation,
                "location_relation_function": relation_function,
                "source_relation": source_relation,
                "target_relation": target_relation,
                "evidence_text": sentence,
            },
            "ego_event": {
                "ego_maneuver": ego_maneuver,
                "evidence_text": sentence if ego_maneuver != "unknown" else "",
            },
            "road_context": {
                "road_type": road_type,
                "lane_context": lane_context,
                "evidence_text": sentence,
            },
            "occlusion": {
                "enabled": occluded,
                "occluder_text": occluder_type if occluded else "",
                "occluder_type": occluder_type if occluded else "unknown",
                "relation_to_actor": "behind" if occluded else "unknown",
                "evidence_text": sentence if occluded else "",
            },
            "event_sequence": [],
            "diagnostics": {},
            "missing_information": {
                "required": ["ego_speed", "actor_speed", "initial_distance"],
                "defaultable": [],
                "distributional": {},
            },
            "completed_parameters": {},
            "confidence": 0.55,
        }

        return EventFrame.from_dict(data)

    @staticmethod
    def _guess_predicate(sentence: str) -> str:
        s = sentence.lower()
        patterns = [
            r"cuts across",
            r"crosses",
            r"traverses",
            r"moves perpendicular",
            r"darts across",
            r"bolts",
            r"leaves the sidewalk",
            r"steps out",
            r"steps into",
            r"stepping into",
            r"step into",
            r"swerves into",
            r"encroaches into",
            r"drifts",
            r"slides over",
            r"nudges over",
            r"edges into",
            r"weaves into",
            r"veers into",
            r"slots into",
            r"sheds speed",
            r"drops speed",
            r"slams on the brakes",
            r"loses speed",
            r"panic stop",
            r"sudden stop",
            r"slows",
            r"brakes",
            r"decelerates",
            r"deceleration",
            r"turns left",
            r"left turn",
            r"entering a rotary",
            r"joins a traffic circle",
            r"enters the lane",
            r"enters the ego lane",
            r"walks into",
            r"walking into",
            r"comes to a halt",
        ]

        for p in patterns:
            m = re.search(p, s)
            if m:
                return m.group(0)

        return "unknown"

    @staticmethod
    def _partial_control_frame(sentence: str) -> EventFrame | None:
        """Represent scene-modifier-only requests with a minimal default event.

        These prompts intentionally omit the concrete actor/event, e.g. "make the
        scene more dangerous" or "add occlusion".  The frame keeps the pipeline
        valid while the mapper extracts the requested control dimension from text.
        """

        s = sentence.lower()
        has_control_intent = any(
            term in s
            for term in [
                "more dangerous",
                "risky interaction",
                "near-miss",
                "near miss",
                "safer",
                "mild",
                "closer",
                "close gap",
                "enough reaction distance",
                "add an occlusion",
                "add occlusion",
                "occlusion before",
            ]
        )
        if not has_control_intent:
            return None

        occluded = "occlusion" in s or "occluded" in s
        data: Dict[str, Any] = {
            "sentence": sentence,
            "main_actor": {
                "text": "hazard actor",
                "actor_class": "human_on_foot",
                "actor_role": "hazard_actor",
                "evidence_text": sentence,
            },
            "main_event": {
                "event_type": "enter_ego_lane",
                "predicate_text": "enters ego lane",
                "path_or_object": "ego lane",
                "event_location_relation": "ahead_of",
                "location_relation_function": "event_anchor",
                "source_relation": "unknown",
                "target_relation": "ego_lane",
                "motion_axis": "lateral",
                "motion_direction": "across_ego_path",
                "evidence_text": sentence,
            },
            "ego_event": {"ego_maneuver": "drive_forward", "evidence_text": ""},
            "road_context": {
                "road_type": "straight_lane",
                "lane_context": "ego_lane",
                "evidence_text": sentence,
            },
            "occlusion": {
                "enabled": occluded,
                "occluder_type": "vehicle" if occluded else "unknown",
                "relation_to_actor": "behind" if occluded else "unknown",
                "evidence_text": sentence if occluded else "",
            },
            "event_sequence": [],
            "diagnostics": {"is_path_crossing": True, "is_occluded": occluded},
            "missing_information": {
                "required": ["ego_speed", "actor_speed", "initial_distance"],
                "defaultable": [],
                "distributional": {},
            },
            "completed_parameters": {},
            "confidence": 0.45,
            "parser_notes": "parsed_by=partial_control_intent_default_frame",
        }
        return EventFrame.from_dict(data)
