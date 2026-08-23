"""Narrative decomposition and hazard-focus resolution.

This module sits before EventFrame construction. It extracts candidate actors
and events from clauses, marks background/negated/occluder roles, and selects
the event that should drive scene generation.

The deterministic fallback is intentionally conservative. It does not build the
final SLEDGE scene directly. It only provides a robust semantic EventFrame
candidate layer.

Important occluded-pedestrian rules
-----------------------------------
1. Human-on-foot terms always win over a parked vehicle that merely acts as an
   occluder.
2. A clause such as "and then steps into the ego lane" may inherit the human
   actor from an earlier clause.
3. left/right roadside information has priority over generic curbside.
4. ego_path and ego_lane are kept semantically distinct.
5. "rushes toward", "heads for" and "moves toward" are valid lateral hazard
   predicates for vulnerable road users.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Dict, Iterable, List, Optional


HUMAN_TERMS = [
    "adult pedestrian",
    "schoolboy",
    "schoolgirl",
    "wheelchair user",
    "road user on foot",
    "pedestrian",
    "walker",
    "person",
    "child",
    "girl",
    "boy",
    "kid",
    "schoolkid",
    "adult",
    "woman",
    "man",
    "someone",
    "jogger",
    "runner",
    "jaywalker",
    "figure",
    "shopper",
    "commuter",
    "passerby",
    "on foot",
]

CYCLIST_TERMS = [
    "cyclist",
    "bicyclist",
    "bicycle rider",
    "bike rider",
    "bicycle",
    "bike",
    "e-bike",
    "ebike",
    "scooter rider",
]

VEHICLE_TERMS = [
    "lead vehicle",
    "leading vehicle",
    "front vehicle",
    "vehicle ahead",
    "car ahead",
    "lead car",
    "front car",
    "oncoming vehicle",
    "oncoming car",
    "approaching vehicle",
    "opposing vehicle",
    "opposite traffic",
    "parked vehicle",
    "parked car",
    "parked truck",
    "parked van",
    "vehicle",
    "car",
    "sedan",
    "suv",
    "truck",
    "bus",
    "van",
    "taxi",
    "pickup",
    "traffic",
]

STATIC_TERMS = [
    "traffic cone",
    "traffic cones",
    "construction barrier",
    "temporary barrier",
    "barrier",
    "cone",
    "cones",
    "construction",
    "work zone",
    "work-zone",
    "roadwork",
    "object",
    "obstacle",
    "debris",
    "blocked lane",
]

NEGATION_TERMS = [
    "no ",
    "not ",
    "do not",
    "don't",
    "without",
    "instead of",
]

BACKGROUND_TERMS = [
    "waiting safely",
    "waiting",
    "standing",
    "strapped",
    "stays in its lane",
    "stay in its lane",
    "not crossing",
    "is not crossing",
    "only an occluder",
    "only a background",
    "safe",
]

FOCUS_TERMS = [
    "actual hazard",
    "critical event",
    "hazard is",
    "hazard:",
    "the hazard",
    "obstacle to generate",
    "generate is",
    "instead",
    "rather",
]

CONTRAST_TERMS = [
    "but",
    "however",
    "instead",
    "while",
    "whereas",
    "after that",
    "then",
    "seconds later",
]

OCCLUSION_TERMS = [
    "hidden",
    "concealed",
    "screened",
    "masked",
    "blind side",
    "behind",
    "behind a",
    "behind an",
    "blocked by",
    "blocked from view",
    "line of sight",
    "line-of-sight",
    "obstructed",
    "obstructed by",
    "occluded",
    "occluded by",
    "partially occluded",
    "obscured",
    "obscured by",
    "between parked",
    "from behind",
    "emerges from",
    "comes out from",
    "out of sight",
    "pops out",
]

LEFT_TERMS = [
    "from the left",
    "on the left roadside",
    "on the left side",
    "on the left",
    "left roadside",
    "left-hand roadside",
    "left side",
    "left curb",
    "left sidewalk",
    "left lane",
    "adjacent left",
    "neighboring left",
    "left-hand lane",
]

RIGHT_TERMS = [
    "from the right",
    "on the right roadside",
    "on the right side",
    "on the right",
    "right roadside",
    "right-hand roadside",
    "right side",
    "right curb",
    "right sidewalk",
    "right lane",
    "adjacent right",
    "neighboring right",
    "right-hand lane",
]

BRAKE_TERMS = [
    "brakes",
    "brake",
    "braking",
    "slams on the brakes",
    "slows",
    "slows down",
    "stops",
    "stopping",
    "stopping short",
    "stops short",
    "hard stop",
    "sudden stop",
    "comes to a halt",
    "decelerates",
    "deceleration",
    "drops speed",
    "reduces speed",
    "checks up",
    "taps the brakes",
]

CUTIN_TERMS = [
    "cuts in",
    "cut in",
    "cuts into",
    "swerves into",
    "encroaches into",
    "drifts",
    "slides over",
    "moves laterally",
    "veers into",
    "slots into",
    "squeezes into",
    "slips into",
    "changes lanes",
    "takes the ego lane",
    "crosses the lane boundary",
    "across the lane marker",
    "into ego's lane",
    "into the ego lane",
    "occupies part of ego's lane",
]

CROSSING_TERMS = [
    "cross",
    "crosses",
    "crossing",
    "across",
    "traverses",
    "perpendicular",
    "cuts across",
    "darts",
    "bolts",
    "hurries across",
    "shoots across",
    "walks across",
    "runs across",
    "moves across",
    "rolls from",
]

TOWARD_PATH_TERMS = [
    "rushes toward",
    "rush toward",
    "rushing toward",
    "runs toward",
    "run toward",
    "moves toward",
    "move toward",
    "moving toward",
    "heads for",
    "head for",
    "heads toward",
    "head toward",
    "approaches the ego path",
    "approaches ego path",
]

ENTER_TERMS = [
    "steps out",
    "steps into",
    "stepping into",
    "step into",
    "steps off",
    "leaves the sidewalk",
    "leaves sidewalk",
    "leaves the median",
    "leaves the roadside",
    "slips between",
    "emerges",
    "appears",
    "comes out",
    "enters",
    "entering",
    "moves into",
    "moving into",
    "walks into",
    "walking into",
    *TOWARD_PATH_TERMS,
]

LEFT_TURN_TERMS = [
    "left turn",
    "left-turn",
    "turns left",
    "turning left",
    "unprotected left",
    "left-turning",
]

ONCOMING_TERMS = [
    "oncoming",
    "opposing",
    "opposite direction",
    "opposite lane",
    "opposite traffic",
    "opposite approach",
    "far side",
    "other direction",
    "coming the other way",
    "through traffic",
    "straight-through traffic",
    "inbound",
    "continues straight",
    "keeps coming",
]

ROUNDABOUT_TERMS = [
    "roundabout",
    "rotary",
    "traffic circle",
    "gyratory",
    "circular junction",
    "circular intersection",
]

ROUNDABOUT_ENTRY_TERMS = [
    "entry",
    "entrance",
    "approach",
    "merge",
    "joins",
    "opening",
    "round entry",
]


def contains_any(text: str, terms: Iterable[str]) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def first_term(
    text: str,
    terms: Iterable[str],
    default: str = "",
) -> str:
    lower = text.lower()
    for term in terms:
        if term in lower:
            return term
    return default


def extract_raw_json_object(text: str) -> Dict[str, Any]:
    """Extract the first raw JSON object from an LLM response."""

    if not isinstance(text, str):
        raise TypeError(
            f"expected string, got {type(text).__name__}"
        )

    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object start found")

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        ch = text[index]

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
                obj = json.loads(
                    text[start : index + 1]
                )
                if not isinstance(obj, dict):
                    raise ValueError(
                        "extracted JSON is not an object"
                    )
                return obj

    raise ValueError("no complete JSON object found")


@dataclass
class Clause:
    text: str
    index: int
    cue_before: str = ""
    has_focus_cue: bool = False
    has_contrast_cue: bool = False


@dataclass
class EventCandidate:
    event_id: str
    clause_index: int
    clause_text: str
    actor_text: str
    actor_class: str
    actor_role: str
    event_type: str
    motion_axis: str

    source_relation: str = "unknown"
    target_relation: str = "unknown"
    motion_direction: str = "unknown"

    location_relation: str = "unknown"
    location_relation_function: str = "unknown"

    road_type: str = "unknown"
    lane_context: str = "unknown"
    ego_maneuver: str = "drive_forward"

    path_or_object: str = "unknown"

    occlusion_enabled: bool = False
    occluder_type: str = "unknown"
    obstacle_type: str = "unknown"

    negated: bool = False
    background: bool = False
    focus_score: float = 0.0

    evidence: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NarrativeAnalysis:
    sentence: str
    clauses: List[Clause]
    candidates: List[EventCandidate]
    selected_event: Optional[EventCandidate] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence": self.sentence,
            "clauses": [
                asdict(c)
                for c in self.clauses
            ],
            "candidates": [
                c.to_dict()
                for c in self.candidates
            ],
            "selected_event": (
                self.selected_event.to_dict()
                if self.selected_event
                else None
            ),
        }


class NarrativeDecomposer:
    """Extract candidate events without selecting a final template."""

    def analyze(
        self,
        sentence: str,
    ) -> NarrativeAnalysis:
        clauses = self._split_clauses(sentence)

        candidates: List[EventCandidate] = []

        for clause in clauses:
            candidates.extend(
                self._candidates_from_clause(
                    sentence,
                    clause,
                )
            )

        return NarrativeAnalysis(
            sentence=sentence,
            clauses=clauses,
            candidates=candidates,
        )

    def _split_clauses(
        self,
        sentence: str,
    ) -> List[Clause]:
        protected = re.sub(
            r"\s+",
            " ",
            sentence.strip(),
        )

        pattern = (
            r"\s*("
            r";"
            r"|,?\s+\bbut\b"
            r"|,?\s+\bhowever\b"
            r"|,?\s+\binstead\b"
            r"|,?\s+\bwhile\b"
            r"|,?\s+\bwhereas\b"
            r"|,?\s+\bthen\b"
            r"|,?\s+\bafter that\b"
            r"|,?\s+\bseconds later\b"
            r")\s*"
        )

        parts = re.split(
            pattern,
            protected,
            flags=re.IGNORECASE,
        )

        clauses: List[Clause] = []
        cue = ""

        for part in parts:
            if not part or not part.strip():
                continue

            stripped = part.strip(" ,;")

            if not stripped:
                continue

            is_connector = (
                stripped.lower()
                in {
                    ";",
                    "but",
                    "however",
                    "instead",
                    "while",
                    "whereas",
                    "then",
                    "after that",
                    "seconds later",
                }
            )

            if is_connector:
                cue = stripped.lower()
                continue

            text = stripped
            lower = text.lower()

            clauses.append(
                Clause(
                    text=text,
                    index=len(clauses),
                    cue_before=cue,
                    has_focus_cue=contains_any(
                        lower,
                        FOCUS_TERMS,
                    ),
                    has_contrast_cue=(
                        bool(cue)
                        or contains_any(
                            lower,
                            CONTRAST_TERMS,
                        )
                    ),
                )
            )

            cue = ""

        if not clauses:
            clauses.append(
                Clause(
                    text=sentence,
                    index=0,
                )
            )

        return clauses

    def _candidates_from_clause(
        self,
        sentence: str,
        clause: Clause,
    ) -> List[EventCandidate]:
        full = sentence.lower()
        lower = clause.text.lower()

        candidates: List[EventCandidate] = []

        for event_type in self._event_types(
            lower,
            full,
        ):
            candidate = self._build_candidate(
                sentence,
                clause,
                event_type,
            )

            if candidate is not None:
                candidates.append(candidate)

        if (
            not candidates
            and self._mentions_actor(lower)
        ):
            candidate = self._background_candidate(
                sentence,
                clause,
            )

            if candidate is not None:
                candidates.append(candidate)

        return candidates

    def _event_types(
        self,
        lower: str,
        full: str,
    ) -> List[str]:
        out: List[str] = []

        if (
            contains_any(
                lower,
                ROUNDABOUT_TERMS,
            )
            or (
                contains_any(
                    full,
                    ROUNDABOUT_TERMS,
                )
                and contains_any(
                    lower,
                    ROUNDABOUT_ENTRY_TERMS
                    + [
                        "circulating",
                        "inside",
                    ],
                )
            )
        ):
            out.append(
                "roundabout_entry_conflict"
            )

        if (
            contains_any(
                full,
                LEFT_TURN_TERMS,
            )
            and (
                contains_any(
                    lower,
                    ONCOMING_TERMS,
                )
                or contains_any(
                    full,
                    ONCOMING_TERMS,
                )
            )
        ):
            out.append(
                "ego_left_turn_across_oncoming"
            )

        if (
            contains_any(
                lower,
                STATIC_TERMS,
            )
            and contains_any(
                lower,
                [
                    "block",
                    "blocking",
                    "occupy",
                    "occupies",
                    "in the lane",
                    "ego lane",
                    "work zone",
                    "construction",
                ],
            )
        ):
            out.append(
                "lane_blocking_conflict"
            )

        if (
            contains_any(
                lower,
                BRAKE_TERMS,
            )
            and contains_any(
                lower,
                [
                    "lead",
                    "leading",
                    "front",
                    "ahead",
                    "vehicle ahead",
                    "car ahead",
                    "front vehicle",
                    "front car",
                    "traffic ahead",
                    "headway",
                ],
            )
        ):
            out.append(
                "lead_vehicle_braking"
            )

        # Important:
        # The current clause may omit its subject:
        #
        # "A schoolboy is obscured by a parked van
        #  and then steps into the ego lane."
        #
        # The second clause contains no "schoolboy", but the full sentence does.
        vulnerable_actor = (
            contains_any(
                lower,
                HUMAN_TERMS + CYCLIST_TERMS,
            )
            or contains_any(
                full,
                HUMAN_TERMS + CYCLIST_TERMS,
            )
        )

        explicit_lane_change = contains_any(
            lower,
            [
                "cuts in",
                "cut in",
                "cuts into",
                "swerves into",
                "encroaches into",
                "drifts",
                "slides over",
                "moves laterally",
                "veers into",
                "slots into",
                "squeezes into",
                "slips into",
                "changes lanes",
                "takes the ego lane",
                "crosses the lane boundary",
                "across the lane marker",
            ],
        )

        # Do not classify "steps into the ego lane" as a vehicle cut-in merely
        # because the same sentence contains a parked van/car occluder.
        if (
            contains_any(
                lower,
                CUTIN_TERMS,
            )
            and contains_any(
                lower,
                [
                    "vehicle",
                    "car",
                    "sedan",
                    "van",
                    "suv",
                    "lane",
                ],
            )
            and (
                explicit_lane_change
                or not vulnerable_actor
            )
        ):
            out.append(
                "lane_change_into_ego_lane"
            )

        moving_actor_available = (
            self._mentions_moving_actor(lower)
            or (
                contains_any(
                    lower,
                    CROSSING_TERMS
                    + ENTER_TERMS,
                )
                and self._mentions_moving_actor(
                    full
                )
            )
        )

        if (
            contains_any(
                lower,
                CROSSING_TERMS,
            )
            and moving_actor_available
        ):
            out.append(
                "path_crossing"
            )

        if (
            contains_any(
                lower,
                ENTER_TERMS,
            )
            and moving_actor_available
        ):
            out.append(
                "enter_ego_lane"
            )

        return _dedupe(out)

    def _build_candidate(
        self,
        sentence: str,
        clause: Clause,
        event_type: str,
    ) -> Optional[EventCandidate]:
        lower = clause.text.lower()
        full = sentence.lower()

        actor_class = self._actor_class_for_event(
            lower,
            full,
            event_type,
        )

        actor_text = self._actor_text(
            lower,
            actor_class,
            event_type,
        )

        actor_role = self._actor_role(
            event_type
        )

        motion_axis = self._motion_axis(
            event_type
        )

        source = self._source_relation(
            lower,
            full=full,
        )

        target = self._target_relation(
            lower,
            full,
            event_type,
        )

        road_type = self._road_type(
            lower,
            full,
            event_type,
        )

        lane_context = self._lane_context(
            event_type,
            road_type,
        )

        if event_type == (
            "ego_left_turn_across_oncoming"
        ):
            ego_maneuver = "left_turn"

        elif event_type == (
            "roundabout_entry_conflict"
        ):
            ego_maneuver = "enter_roundabout"

        else:
            ego_maneuver = "drive_forward"

        occlusion = self._is_occluded(
            lower,
            full,
            event_type,
        )

        score = self._score_candidate(
            clause,
            lower,
            event_type,
        )

        negated = self._is_negated(
            lower,
            event_type,
        )

        background = self._is_background(
            lower,
            event_type,
        )

        return EventCandidate(
            event_id=(
                f"event_{clause.index}_"
                f"{event_type}"
            ),
            clause_index=clause.index,
            clause_text=clause.text,
            actor_text=actor_text,
            actor_class=actor_class,
            actor_role=actor_role,
            event_type=event_type,
            motion_axis=motion_axis,
            source_relation=source,
            target_relation=target,
            motion_direction=(
                self._motion_direction(
                    event_type,
                    source,
                )
            ),
            location_relation=(
                self._location_relation(
                    lower,
                    full,
                    event_type,
                )
            ),
            location_relation_function=(
                self._location_relation_function(
                    event_type
                )
            ),
            road_type=road_type,
            lane_context=lane_context,
            ego_maneuver=ego_maneuver,
            path_or_object=(
                self._path_or_object(
                    lower,
                    full,
                    event_type,
                )
            ),
            occlusion_enabled=occlusion,
            occluder_type=(
                self._occluder_type(
                    lower,
                    full,
                )
                if occlusion
                else "unknown"
            ),
            obstacle_type=(
                self._obstacle_type(
                    lower,
                    event_type,
                )
            ),
            negated=negated,
            background=background,
            focus_score=score,
            evidence={
                "focus_cue": (
                    clause.has_focus_cue
                ),
                "contrast_cue": (
                    clause.has_contrast_cue
                ),
                "cue_before": (
                    clause.cue_before
                ),
                "negation_terms": [
                    t
                    for t in NEGATION_TERMS
                    if t in lower
                ],
                "background_terms": [
                    t
                    for t in BACKGROUND_TERMS
                    if t in lower
                ],
            },
        )

    def _background_candidate(
        self,
        sentence: str,
        clause: Clause,
    ) -> Optional[EventCandidate]:
        lower = clause.text.lower()

        actor_class = (
            self._actor_class_for_event(
                lower,
                sentence.lower(),
                "background",
            )
        )

        if actor_class == "unknown":
            return None

        return EventCandidate(
            event_id=(
                f"event_{clause.index}_"
                "background"
            ),
            clause_index=clause.index,
            clause_text=clause.text,
            actor_text=self._actor_text(
                lower,
                actor_class,
                "background",
            ),
            actor_class=actor_class,
            actor_role="background_actor",
            event_type="background_context",
            motion_axis="unknown",
            negated=self._is_negated(
                lower,
                "background",
            ),
            background=True,
            focus_score=-3.0,
        )

    def _mentions_actor(
        self,
        lower: str,
    ) -> bool:
        return contains_any(
            lower,
            HUMAN_TERMS
            + CYCLIST_TERMS
            + VEHICLE_TERMS
            + STATIC_TERMS,
        )

    def _mentions_moving_actor(
        self,
        lower: str,
    ) -> bool:
        return contains_any(
            lower,
            HUMAN_TERMS
            + CYCLIST_TERMS
            + VEHICLE_TERMS,
        )

    def _actor_class_for_event(
        self,
        lower: str,
        full: str,
        event_type: str,
    ) -> str:
        if event_type in {
            "lane_blocking_conflict",
            "object_in_lane",
        }:
            return "traffic_object"

        if event_type in {
            "lead_vehicle_braking",
            "lane_change_into_ego_lane",
            "ego_left_turn_across_oncoming",
            "roundabout_entry_conflict",
        }:
            return "vehicle"

        # The actor explicitly named in the clause has highest priority.
        if contains_any(
            lower,
            CYCLIST_TERMS,
        ):
            return "cyclist"

        if contains_any(
            lower,
            HUMAN_TERMS,
        ):
            return "human_on_foot"

        # A parked vehicle used only as an occluder must not steal the actor.
        lower_without_parked_occluder = re.sub(
            r"\bparked\s+"
            r"(car|vehicle|truck|van|bus|suv)\b",
            "",
            lower,
        )

        if contains_any(
            lower_without_parked_occluder,
            VEHICLE_TERMS,
        ):
            return "vehicle"

        if contains_any(
            lower,
            STATIC_TERMS,
        ):
            return "traffic_object"

        # Subject inheritance across clauses.
        if event_type in {
            "path_crossing",
            "enter_ego_lane",
        }:
            if contains_any(
                full,
                CYCLIST_TERMS,
            ):
                return "cyclist"

            if contains_any(
                full,
                HUMAN_TERMS,
            ):
                return "human_on_foot"

        if contains_any(
            full,
            VEHICLE_TERMS,
        ):
            return "vehicle"

        return "unknown"

    def _actor_text(
        self,
        lower: str,
        actor_class: str,
        event_type: str,
    ) -> str:
        if (
            event_type
            == "lead_vehicle_braking"
        ):
            return "lead vehicle"

        if (
            event_type
            == "ego_left_turn_across_oncoming"
        ):
            return "oncoming vehicle"

        if (
            event_type
            == "roundabout_entry_conflict"
        ):
            return "circulating vehicle"

        if (
            event_type
            == "lane_change_into_ego_lane"
        ):
            return first_term(
                lower,
                [
                    "sedan",
                    "van",
                    "suv",
                    "car",
                    "vehicle",
                ],
                "vehicle",
            )

        if actor_class == "traffic_object":
            return first_term(
                lower,
                STATIC_TERMS,
                "traffic object",
            )

        if actor_class == "cyclist":
            return first_term(
                lower,
                CYCLIST_TERMS,
                "cyclist",
            )

        if actor_class == "human_on_foot":
            return first_term(
                lower,
                HUMAN_TERMS,
                "pedestrian",
            )

        if actor_class == "vehicle":
            return first_term(
                lower,
                VEHICLE_TERMS,
                "vehicle",
            )

        return "unknown actor"

    @staticmethod
    def _actor_role(
        event_type: str,
    ) -> str:
        return {
            "lead_vehicle_braking":
                "braking_actor",
            "lane_change_into_ego_lane":
                "merging_actor",
            "ego_left_turn_across_oncoming":
                "approaching_actor",
            "roundabout_entry_conflict":
                "merging_actor",
            "lane_blocking_conflict":
                "blocking_actor",
            "object_in_lane":
                "static_obstacle",
            "path_crossing":
                "crossing_actor",
            "enter_ego_lane":
                "crossing_actor",
        }.get(
            event_type,
            "unknown",
        )

    @staticmethod
    def _motion_axis(
        event_type: str,
    ) -> str:
        return {
            "lead_vehicle_braking":
                "longitudinal",
            "lane_change_into_ego_lane":
                "merging",
            "ego_left_turn_across_oncoming":
                "oncoming",
            "roundabout_entry_conflict":
                "merging",
            "lane_blocking_conflict":
                "static",
            "object_in_lane":
                "static",
            "path_crossing":
                "lateral",
            "enter_ego_lane":
                "lateral",
        }.get(
            event_type,
            "unknown",
        )

    def _source_relation(
        self,
        lower: str,
        *,
        full: str = "",
    ) -> str:
        # Explicit left/right has priority over generic curb/roadside.
        combined = " ".join(
            [lower, full]
        )

        if contains_any(
            lower,
            LEFT_TERMS,
        ):
            return "from_left"

        if contains_any(
            lower,
            RIGHT_TERMS,
        ):
            return "from_right"

        # Subject/predicate may be split across clauses, so use the whole
        # sentence for explicit side if the current clause omits it.
        if contains_any(
            combined,
            LEFT_TERMS,
        ):
            return "from_left"

        if contains_any(
            combined,
            RIGHT_TERMS,
        ):
            return "from_right"

        if contains_any(
            lower,
            [
                "opposite",
                "oncoming",
                "far side",
                "other direction",
            ],
        ):
            return "from_opposite_direction"

        if contains_any(
            lower,
            [
                "curb",
                "sidewalk",
                "roadside",
            ],
        ):
            return "from_curb"

        if contains_any(
            full,
            [
                "curb",
                "sidewalk",
                "roadside",
            ],
        ):
            return "from_curb"

        if contains_any(
            lower,
            [
                "circulating",
                "inside",
                "roundabout",
            ],
        ):
            return "from_circulating_lane"

        if contains_any(
            lower,
            [
                "adjacent",
                "neighboring",
                "lane",
            ],
        ):
            return "from_adjacent_lane"

        return "unknown"

    @staticmethod
    def _target_relation(
        lower: str,
        full: str,
        event_type: str,
    ) -> str:
        if (
            event_type
            == "ego_left_turn_across_oncoming"
        ):
            return "intersection"

        if (
            event_type
            == "roundabout_entry_conflict"
        ):
            return "roundabout_entry"

        if (
            event_type
            == "lead_vehicle_braking"
        ):
            return "ego_lane"

        # Preserve ego_path before testing generic "lane".
        if contains_any(
            lower,
            [
                "ego path",
                "ego's path",
                "ego vehicle's path",
                "vehicle's path",
                "ego trajectory",
                "trajectory",
                "entry path",
            ],
        ):
            return "ego_path"

        if contains_any(
            lower,
            [
                "ego lane",
                "ego's lane",
                "travel lane",
            ],
        ):
            return "ego_lane"

        if contains_any(
            full,
            [
                "ego path",
                "ego's path",
                "ego vehicle's path",
                "vehicle's path",
                "ego trajectory",
                "trajectory",
            ],
        ):
            return "ego_path"

        if contains_any(
            full,
            [
                "ego lane",
                "ego's lane",
                "travel lane",
            ],
        ):
            return "ego_lane"

        return "ego_lane"

    @staticmethod
    def _motion_direction(
        event_type: str,
        source: str,
    ) -> str:
        if (
            event_type
            == "lane_change_into_ego_lane"
        ):
            return "into_ego_lane"

        if (
            event_type
            == "lead_vehicle_braking"
        ):
            return "decelerating_ahead"

        if (
            event_type
            == "ego_left_turn_across_oncoming"
        ):
            return "opposing_through"

        if (
            event_type
            == "roundabout_entry_conflict"
        ):
            return "circulating_across_entry"

        if event_type in {
            "path_crossing",
            "enter_ego_lane",
        }:
            if source == "from_left":
                return "left_to_right"

            if source == "from_right":
                return "right_to_left"

            return "across_ego_path"

        if event_type in {
            "lane_blocking_conflict",
            "object_in_lane",
        }:
            return "stationary"

        return "unknown"

    @staticmethod
    def _location_relation(
        lower: str,
        full: str,
        event_type: str,
    ) -> str:
        if (
            event_type
            == "roundabout_entry_conflict"
        ):
            return "at_roundabout_entry"

        if (
            event_type
            == "ego_left_turn_across_oncoming"
            or contains_any(
                lower,
                [
                    "intersection",
                    "junction",
                ],
            )
        ):
            return "at_intersection"

        if event_type in {
            "lead_vehicle_braking",
            "lane_blocking_conflict",
            "object_in_lane",
        }:
            return "ahead_of"

        if contains_any(
            lower,
            [
                "ahead",
                "just ahead",
            ],
        ):
            return "ahead_of"

        if contains_any(
            lower,
            [
                "in front",
                "forward path",
            ],
        ):
            return "in_front_of"

        if contains_any(
            lower,
            LEFT_TERMS,
        ):
            return "left_of"

        if contains_any(
            lower,
            RIGHT_TERMS,
        ):
            return "right_of"

        if contains_any(
            full,
            [
                "ahead",
                "in front",
            ],
        ):
            return "ahead_of"

        return "unknown"

    @staticmethod
    def _location_relation_function(
        event_type: str,
    ) -> str:
        if event_type in {
            "path_crossing",
            "enter_ego_lane",
            "ego_left_turn_across_oncoming",
            "roundabout_entry_conflict",
        }:
            return "event_anchor"

        if (
            event_type
            == "lane_change_into_ego_lane"
        ):
            return "target_lane"

        return "actor_position"

    @staticmethod
    def _road_type(
        lower: str,
        full: str,
        event_type: str,
    ) -> str:
        if (
            event_type
            == "roundabout_entry_conflict"
        ):
            return "roundabout"

        if (
            event_type
            == "ego_left_turn_across_oncoming"
            or contains_any(
                lower,
                [
                    "intersection",
                    "junction",
                ],
            )
        ):
            return "intersection"

        if (
            event_type
            == "lane_blocking_conflict"
            or contains_any(
                lower,
                [
                    "construction",
                    "work zone",
                    "work-zone",
                    "traffic cone",
                    "traffic cones",
                    "barrier",
                ],
            )
        ):
            return "construction_zone"

        if contains_any(
            lower,
            [
                "curb",
                "sidewalk",
                "roadside",
            ],
        ):
            return "curbside"

        if (
            contains_any(
                full,
                [
                    "intersection",
                    "junction",
                ],
            )
            and event_type
            in {
                "path_crossing",
                "enter_ego_lane",
            }
        ):
            return "intersection"

        return "straight_lane"

    @staticmethod
    def _lane_context(
        event_type: str,
        road_type: str,
    ) -> str:
        if (
            event_type
            == "roundabout_entry_conflict"
        ):
            return "roundabout_entry"

        if (
            event_type
            == "ego_left_turn_across_oncoming"
        ):
            return "opposing_lane"

        if (
            event_type
            == "lane_change_into_ego_lane"
        ):
            return "adjacent_lane"

        if (
            event_type
            == "lead_vehicle_braking"
        ):
            return "ego_lane"

        if road_type == "construction_zone":
            return "ego_lane"

        return "ego_lane"

    def _path_or_object(
        self,
        lower: str,
        full: str,
        event_type: str,
    ) -> str:
        if (
            event_type
            == "lane_blocking_conflict"
        ):
            return self._obstacle_type(
                lower,
                event_type,
            )

        if (
            event_type
            == "roundabout_entry_conflict"
        ):
            return "roundabout entry"

        if (
            event_type
            == "ego_left_turn_across_oncoming"
        ):
            return "ego turn path"

        if contains_any(
            lower,
            [
                "ego path",
                "ego's path",
                "trajectory",
                "path",
            ],
        ):
            return "ego path"

        if contains_any(
            lower,
            [
                "ego lane",
                "ego's lane",
                "travel lane",
                "lane",
            ],
        ):
            return "ego lane"

        if contains_any(
            full,
            [
                "ego path",
                "ego's path",
                "trajectory",
            ],
        ):
            return "ego path"

        return "ego lane"

    @staticmethod
    def _is_occluded(
        lower: str,
        full: str,
        event_type: str,
    ) -> bool:
        if event_type in {
            "path_crossing",
            "enter_ego_lane",
        }:
            return (
                contains_any(
                    lower,
                    OCCLUSION_TERMS,
                )
                or contains_any(
                    full,
                    OCCLUSION_TERMS,
                )
            )

        return False

    @staticmethod
    def _occluder_type(
        lower: str,
        full: str,
    ) -> str:
        text = " ".join(
            [lower, full]
        )

        if (
            "parked truck" in text
            or "truck" in text
        ):
            return "truck"

        if "bus" in text:
            return "bus"

        if (
            "parked van" in text
            or "van" in text
        ):
            return "van"

        if contains_any(
            text,
            [
                "parked car",
                "parked cars",
                "parked vehicle",
                "parked vehicles",
                "parked suv",
                "that car",
            ],
        ):
            return "parked_vehicle"

        if "suv" in text:
            return "vehicle"

        if "barrier" in text:
            return "roadside_object"

        return "vehicle"

    @staticmethod
    def _obstacle_type(
        lower: str,
        event_type: str,
    ) -> str:
        if event_type not in {
            "lane_blocking_conflict",
            "object_in_lane",
        }:
            return "unknown"

        if contains_any(
            lower,
            [
                "traffic cone",
                "traffic cones",
                "cone",
                "cones",
            ],
        ):
            return "cone"

        if "barrier" in lower:
            return "barrier"

        if contains_any(
            lower,
            [
                "construction",
                "work zone",
                "work-zone",
            ],
        ):
            return "construction_zone"

        if "debris" in lower:
            return "debris"

        return "object"

    @staticmethod
    def _is_negated(
        lower: str,
        event_type: str,
    ) -> bool:
        if contains_any(
            lower,
            [
                "do not create",
                "don't create",
                "ignore",
            ],
        ):
            return True

        if (
            event_type
            in {
                "path_crossing",
                "enter_ego_lane",
            }
            and contains_any(
                lower,
                [
                    "no pedestrian crossing",
                    "no crossing",
                    "not crossing",
                    "is not crossing",
                ],
            )
        ):
            return True

        if (
            event_type
            == "lane_change_into_ego_lane"
            and contains_any(
                lower,
                [
                    "do not create a cut-in",
                    "no cut-in",
                    "not a cut-in",
                    "stays in its lane",
                ],
            )
        ):
            return True

        return False

    @staticmethod
    def _is_background(
        lower: str,
        event_type: str,
    ) -> bool:
        if event_type in {
            "background_context",
            "unknown",
        }:
            return True

        if contains_any(
            lower,
            [
                "only a background",
            ],
        ):
            return True

        if (
            "only an occluder"
            in lower
            and event_type
            not in {
                "path_crossing",
                "enter_ego_lane",
            }
        ):
            return True

        if (
            event_type
            in {
                "path_crossing",
                "enter_ego_lane",
            }
            and contains_any(
                lower,
                [
                    "waiting safely",
                    "standing",
                    "not crossing",
                ],
            )
        ):
            return True

        if (
            event_type
            == "lane_change_into_ego_lane"
            and "stays in its lane"
            in lower
        ):
            return True

        return False

    def _score_candidate(
        self,
        clause: Clause,
        lower: str,
        event_type: str,
    ) -> float:
        score = {
            "ego_left_turn_across_oncoming":
                7.0,
            "roundabout_entry_conflict":
                6.5,
            "lane_change_into_ego_lane":
                6.0,
            "lead_vehicle_braking":
                6.0,
            "lane_blocking_conflict":
                5.5,
            "path_crossing":
                5.0,
            "enter_ego_lane":
                5.0,
        }.get(
            event_type,
            0.0,
        )

        if clause.has_focus_cue:
            score += 4.0

        if clause.has_contrast_cue:
            score += 2.0

        if contains_any(
            lower,
            [
                "actual hazard",
                "critical event",
                "hazard is",
                "obstacle to generate",
            ],
        ):
            score += 4.0

        if contains_any(
            lower,
            [
                "suddenly",
                "abrupt",
                "hard",
                "no room",
                "small gap",
                "near-miss",
                "dangerous",
                "tight",
                "rushes toward",
                "darts",
                "bolts",
            ],
        ):
            score += 1.5

        if self._is_negated(
            lower,
            event_type,
        ):
            score -= 8.0

        if self._is_background(
            lower,
            event_type,
        ):
            score -= 5.0

        return score


class HazardFocusResolver:
    """Select the most likely hazard event from narrative candidates."""

    def resolve(
        self,
        analysis: NarrativeAnalysis,
    ) -> NarrativeAnalysis:
        viable = [
            candidate
            for candidate
            in analysis.candidates
            if (
                candidate.event_type
                not in {
                    "background_context",
                    "unknown",
                }
                and not candidate.negated
                and not candidate.background
            )
        ]

        if viable:
            analysis.selected_event = max(
                viable,
                key=lambda candidate: (
                    candidate.focus_score,
                    candidate.clause_index,
                ),
            )

        return analysis


def narrative_analysis_from_dict(
    sentence: str,
    data: Dict[str, Any],
) -> NarrativeAnalysis:
    """Normalize an LLM-produced candidate-event object."""

    decomposer = NarrativeDecomposer()

    clauses = _clauses_from_dict(
        sentence,
        data,
    )

    if not clauses:
        clauses = decomposer._split_clauses(
            sentence
        )

    clause_by_index = {
        clause.index: clause
        for clause in clauses
    }

    raw_candidates = data.get(
        "candidate_events"
    )

    if raw_candidates is None:
        raw_candidates = data.get(
            "candidates"
        )

    if raw_candidates is None:
        raw_candidates = data.get(
            "events"
        )

    if not isinstance(
        raw_candidates,
        list,
    ):
        raw_candidates = []

    selected_id = str(
        data.get(
            "selected_hazard_event_id"
        )
        or data.get(
            "selected_event_id"
        )
        or data.get(
            "selected_hazard_event"
        )
        or ""
    )

    candidates: List[EventCandidate] = []

    for index, raw in enumerate(
        raw_candidates
    ):
        if not isinstance(raw, dict):
            continue

        candidate = _candidate_from_llm_dict(
            sentence=sentence,
            raw=raw,
            index=index,
            clauses=clauses,
            clause_by_index=(
                clause_by_index
            ),
            decomposer=decomposer,
            selected_id=selected_id,
        )

        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return decomposer.analyze(
            sentence
        )

    return NarrativeAnalysis(
        sentence=sentence,
        clauses=clauses,
        candidates=candidates,
    )


def _clauses_from_dict(
    sentence: str,
    data: Dict[str, Any],
) -> List[Clause]:
    raw_clauses = data.get(
        "clauses"
    )

    if not isinstance(
        raw_clauses,
        list,
    ):
        return []

    clauses: List[Clause] = []

    for index, raw in enumerate(
        raw_clauses
    ):
        if isinstance(raw, str):
            text = raw
            cue_before = ""

        elif isinstance(raw, dict):
            text = str(
                raw.get("text")
                or raw.get("clause")
                or ""
            )

            cue_before = str(
                raw.get("cue_before")
                or raw.get("cue")
                or ""
            )

        else:
            continue

        text = text.strip()

        if not text:
            continue

        lower = text.lower()

        clauses.append(
            Clause(
                text=text,
                index=len(clauses),
                cue_before=(
                    cue_before.lower()
                ),
                has_focus_cue=(
                    contains_any(
                        lower,
                        FOCUS_TERMS,
                    )
                ),
                has_contrast_cue=(
                    bool(cue_before)
                    or contains_any(
                        lower,
                        CONTRAST_TERMS,
                    )
                ),
            )
        )

    if not clauses and sentence:
        clauses.append(
            Clause(
                text=sentence,
                index=0,
            )
        )

    return clauses


def _candidate_from_llm_dict(
    *,
    sentence: str,
    raw: Dict[str, Any],
    index: int,
    clauses: List[Clause],
    clause_by_index: Dict[int, Clause],
    decomposer: NarrativeDecomposer,
    selected_id: str,
) -> Optional[EventCandidate]:
    raw_event_id = str(
        raw.get("event_id")
        or raw.get("id")
        or f"llm_event_{index}"
    )

    event_type = _canonical_event_type(
        str(
            raw.get("event_type")
            or raw.get("event")
            or raw.get(
                "candidate_event"
            )
            or "unknown"
        )
    )

    if event_type == "unknown":
        return None

    clause_index = _coerce_int(
        raw.get("clause_index"),
        default=index,
    )

    clause = clause_by_index.get(
        clause_index
    )

    if clause is None:
        text = str(
            raw.get("clause_text")
            or raw.get("text")
            or raw.get("evidence_text")
            or sentence
        )

        lower = text.lower()

        clause = Clause(
            text=text,
            index=clause_index,
            has_focus_cue=(
                contains_any(
                    lower,
                    FOCUS_TERMS,
                )
            ),
            has_contrast_cue=(
                contains_any(
                    lower,
                    CONTRAST_TERMS,
                )
            ),
        )

    seed = decomposer._build_candidate(
        sentence,
        clause,
        event_type,
    )

    if seed is None:
        return None

    actor_obj = raw.get("actor")

    if isinstance(actor_obj, dict):
        actor_text = str(
            actor_obj.get("text")
            or actor_obj.get("name")
            or raw.get("actor_text")
            or seed.actor_text
        )

        actor_class = (
            _canonical_actor_class(
                str(
                    actor_obj.get(
                        "actor_class"
                    )
                    or actor_obj.get("type")
                    or raw.get(
                        "actor_class"
                    )
                    or seed.actor_class
                )
            )
        )

        actor_role = (
            _canonical_actor_role(
                str(
                    actor_obj.get(
                        "actor_role"
                    )
                    or actor_obj.get("role")
                    or raw.get(
                        "actor_role"
                    )
                    or seed.actor_role
                )
            )
        )

    else:
        actor_text = str(
            raw.get("actor_text")
            or raw.get("actor")
            or seed.actor_text
        )

        actor_class = (
            _canonical_actor_class(
                str(
                    raw.get("actor_class")
                    or raw.get(
                        "actor_type"
                    )
                    or seed.actor_class
                )
            )
        )

        actor_role = (
            _canonical_actor_role(
                str(
                    raw.get("actor_role")
                    or raw.get("role")
                    or seed.actor_role
                )
            )
        )

    # Guard the LLM output against the same occluder-as-actor failure.
    if (
        event_type
        in {
            "path_crossing",
            "enter_ego_lane",
        }
        and contains_any(
            sentence,
            HUMAN_TERMS,
        )
        and contains_any(
            sentence,
            OCCLUSION_TERMS,
        )
    ):
        actor_class = "human_on_foot"

        if (
            actor_text.lower()
            in VEHICLE_TERMS
            or actor_text.lower()
            in {
                "vehicle",
                "car",
                "truck",
                "bus",
                "van",
            }
        ):
            actor_text = first_term(
                sentence,
                HUMAN_TERMS,
                "pedestrian",
            )

    negated = _coerce_bool(
        raw.get("negated"),
        default=seed.negated,
    )

    background = _coerce_bool(
        raw.get("background"),
        default=seed.background,
    )

    semantic_role = str(
        raw.get("semantic_role")
        or raw.get("role")
        or ""
    ).lower()

    if semantic_role in {
        "background",
        "background_actor",
        "context",
    }:
        background = True

    if semantic_role in {
        "negated",
        "negated_actor",
        "negated_event",
    }:
        negated = True

    likelihood = str(
        raw.get("hazard_likelihood")
        or raw.get("likelihood")
        or ""
    ).lower()

    score = (
        seed.focus_score
        + _likelihood_score(
            likelihood
        )
    )

    if (
        raw_event_id
        and raw_event_id == selected_id
    ):
        score += 3.5

    if _coerce_bool(
        raw.get("selected"),
        default=False,
    ):
        score += 3.5

    source_relation = (
        _canonical_source_relation(
            str(
                raw.get("source_relation")
                or raw.get("source_side")
                or seed.source_relation
            )
        )
    )

    target_relation = (
        _canonical_target_relation(
            str(
                raw.get("target_relation")
                or raw.get("target_path")
                or seed.target_relation
            )
        )
    )

    # Deterministic text evidence has priority if the model collapsed path/lane.
    text_lower = sentence.lower()

    if "ego path" in text_lower:
        target_relation = "ego_path"

    elif (
        "ego lane" in text_lower
        or "ego's lane" in text_lower
    ):
        target_relation = "ego_lane"

    return EventCandidate(
        event_id=raw_event_id,
        clause_index=clause.index,
        clause_text=clause.text,
        actor_text=actor_text,
        actor_class=actor_class,
        actor_role=actor_role,
        event_type=event_type,
        motion_axis=(
            _canonical_motion_axis(
                str(
                    raw.get(
                        "motion_axis"
                    )
                    or raw.get(
                        "motion_geometry"
                    )
                    or seed.motion_axis
                )
            )
        ),
        source_relation=(
            source_relation
        ),
        target_relation=(
            target_relation
        ),
        motion_direction=str(
            raw.get("motion_direction")
            or seed.motion_direction
        ),
        location_relation=str(
            raw.get("location_relation")
            or raw.get(
                "event_location_relation"
            )
            or seed.location_relation
        ),
        location_relation_function=str(
            raw.get(
                "location_relation_function"
            )
            or seed.location_relation_function
        ),
        road_type=(
            _canonical_road_type(
                str(
                    raw.get("road_type")
                    or raw.get(
                        "road_topology"
                    )
                    or seed.road_type
                )
            )
        ),
        lane_context=str(
            raw.get("lane_context")
            or seed.lane_context
        ),
        ego_maneuver=str(
            raw.get("ego_maneuver")
            or seed.ego_maneuver
        ),
        path_or_object=(
            "ego path"
            if target_relation
            == "ego_path"
            else "ego lane"
            if target_relation
            == "ego_lane"
            else str(
                raw.get(
                    "path_or_object"
                )
                or raw.get(
                    "target_path"
                )
                or seed.path_or_object
            )
        ),
        occlusion_enabled=(
            _coerce_bool(
                raw.get(
                    "occlusion_enabled"
                ),
                default=(
                    seed.occlusion_enabled
                ),
            )
            or (
                event_type
                in {
                    "path_crossing",
                    "enter_ego_lane",
                }
                and contains_any(
                    sentence,
                    OCCLUSION_TERMS,
                )
            )
        ),
        occluder_type=str(
            raw.get("occluder_type")
            or seed.occluder_type
        ),
        obstacle_type=str(
            raw.get("obstacle_type")
            or seed.obstacle_type
        ),
        negated=negated,
        background=background,
        focus_score=score,
        evidence={
            **seed.evidence,
            "llm_raw": raw,
            "hazard_likelihood": (
                likelihood
            ),
        },
    )


def _canonical_event_type(
    value: str,
) -> str:
    normalized = _norm(value)

    aliases = {
        "left_turn_across_oncoming":
            "ego_left_turn_across_oncoming",
        "oncoming_through_conflict":
            "ego_left_turn_across_oncoming",
        "oncoming_conflict":
            "ego_left_turn_across_oncoming",
        "lead_braking":
            "lead_vehicle_braking",
        "lead_vehicle_deceleration":
            "lead_vehicle_braking",
        "hard_stop_ahead":
            "lead_vehicle_braking",
        "cut_in":
            "lane_change_into_ego_lane",
        "vehicle_cut_in":
            "lane_change_into_ego_lane",
        "lane_change":
            "lane_change_into_ego_lane",
        "merge_into_ego_lane":
            "lane_change_into_ego_lane",
        "crossing":
            "path_crossing",
        "lateral_crossing":
            "path_crossing",
        "enter_lane":
            "enter_ego_lane",
        "object_blocking_lane":
            "lane_blocking_conflict",
        "static_obstacle":
            "lane_blocking_conflict",
        "construction_blocking":
            "lane_blocking_conflict",
        "roundabout":
            "roundabout_entry_conflict",
        "roundabout_merge":
            "roundabout_entry_conflict",
    }

    return aliases.get(
        normalized,
        normalized,
    )


def _canonical_actor_class(
    value: str,
) -> str:
    normalized = _norm(value)

    aliases = {
        "pedestrian":
            "human_on_foot",
        "person":
            "human_on_foot",
        "walker":
            "human_on_foot",
        "child":
            "human_on_foot",
        "girl":
            "human_on_foot",
        "boy":
            "human_on_foot",
        "schoolboy":
            "human_on_foot",
        "schoolgirl":
            "human_on_foot",
        "jogger":
            "human_on_foot",
        "runner":
            "human_on_foot",
        "human":
            "human_on_foot",
        "human_on_foot":
            "human_on_foot",
        "bicycle":
            "cyclist",
        "bicyclist":
            "cyclist",
        "bike":
            "cyclist",
        "scooter_rider":
            "cyclist",
        "traffic_object":
            "traffic_object",
        "static_obstacle":
            "traffic_object",
        "object":
            "traffic_object",
        "construction":
            "traffic_object",
        "vehicle":
            "vehicle",
        "car":
            "vehicle",
        "truck":
            "vehicle",
        "bus":
            "vehicle",
        "van":
            "vehicle",
    }

    return aliases.get(
        normalized,
        (
            normalized
            if normalized
            in {
                "cyclist",
                "vehicle",
                "traffic_object",
                "human_on_foot",
            }
            else "unknown"
        ),
    )


def _canonical_actor_role(
    value: str,
) -> str:
    normalized = _norm(value)

    aliases = {
        "hazard":
            "hazard_actor",
        "hazard_actor":
            "hazard_actor",
        "crossing":
            "crossing_actor",
        "crossing_actor":
            "crossing_actor",
        "merging":
            "merging_actor",
        "merging_actor":
            "merging_actor",
        "braking":
            "braking_actor",
        "braking_actor":
            "braking_actor",
        "approaching":
            "approaching_actor",
        "approaching_actor":
            "approaching_actor",
        "blocking":
            "blocking_actor",
        "blocking_actor":
            "blocking_actor",
        "static_obstacle":
            "static_obstacle",
        "background":
            "background_actor",
        "background_actor":
            "background_actor",
        "occluder":
            "occluder",
    }

    return aliases.get(
        normalized,
        normalized or "unknown",
    )


def _canonical_motion_axis(
    value: str,
) -> str:
    normalized = _norm(value)

    aliases = {
        "lateral_crossing":
            "lateral",
        "crossing_path":
            "lateral",
        "lane_change":
            "merging",
        "longitudinal_braking":
            "longitudinal",
        "static_obstacle":
            "static",
    }

    return aliases.get(
        normalized,
        normalized or "unknown",
    )


def _canonical_source_relation(
    value: str,
) -> str:
    normalized = _norm(value)

    aliases = {
        "left":
            "from_left",
        "left_side":
            "from_left",
        "right":
            "from_right",
        "right_side":
            "from_right",
        "front":
            "front",
        "opposite":
            "from_opposite_direction",
        "roundabout_inside":
            "from_circulating_lane",
        "adjacent_lane":
            "from_adjacent_lane",
        "curb":
            "from_curb",
        "curbside":
            "from_curb",
    }

    return aliases.get(
        normalized,
        normalized or "unknown",
    )


def _canonical_target_relation(
    value: str,
) -> str:
    normalized = _norm(value)

    aliases = {
        "lane":
            "ego_lane",
        "ego_lane":
            "ego_lane",
        "ego's_lane":
            "ego_lane",
        "path":
            "ego_path",
        "ego_path":
            "ego_path",
        "ego's_path":
            "ego_path",
        "turn_path":
            "intersection",
    }

    return aliases.get(
        normalized,
        normalized or "unknown",
    )


def _canonical_road_type(
    value: str,
) -> str:
    normalized = _norm(value)

    aliases = {
        "road":
            "straight_lane",
        "lane":
            "straight_lane",
        "crosswalk":
            "crosswalk_area",
        "work_zone":
            "construction_zone",
    }

    return aliases.get(
        normalized,
        normalized or "unknown",
    )


def _likelihood_score(
    value: str,
) -> float:
    normalized = _norm(value)

    if normalized in {
        "high",
        "hazard",
        "critical",
        "main",
        "true",
    }:
        return 4.0

    if normalized in {
        "medium",
        "possible",
    }:
        return 1.0

    if normalized in {
        "low",
        "background",
        "safe",
        "false",
    }:
        return -4.0

    return 0.0


def _coerce_bool(
    value: Any,
    *,
    default: bool = False,
) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        if value.strip().lower() in {
            "true",
            "yes",
            "1",
        }:
            return True

        if value.strip().lower() in {
            "false",
            "no",
            "0",
        }:
            return False

    return default


def _coerce_int(
    value: Any,
    *,
    default: int = 0,
) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _norm(value: str) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def candidate_to_event_frame_dict(
    sentence: str,
    analysis: NarrativeAnalysis,
    candidate: EventCandidate,
) -> Dict[str, Any]:
    diagnostics = {
        "is_path_crossing": (
            candidate.event_type
            in {
                "path_crossing",
                "enter_ego_lane",
            }
        ),
        "is_lane_change_into_ego_lane": (
            candidate.event_type
            == "lane_change_into_ego_lane"
        ),
        "is_longitudinal_following": (
            candidate.event_type
            == "lead_vehicle_braking"
        ),
        "is_ego_left_turn": (
            candidate.event_type
            == "ego_left_turn_across_oncoming"
        ),
        "is_roundabout_entry": (
            candidate.event_type
            == "roundabout_entry_conflict"
        ),
        "is_occluded": (
            candidate.occlusion_enabled
        ),
    }

    completed_parameters: Dict[
        str,
        Dict[str, Any],
    ] = {}

    if (
        candidate.obstacle_type
        != "unknown"
    ):
        completed_parameters[
            "obstacle_type"
        ] = {
            "value":
                candidate.obstacle_type,
            "source":
                "narrative_focus",
            "reason":
                (
                    "static obstacle identified "
                    "before EventFrame mapping"
                ),
        }

    return {
        "sentence": sentence,
        "main_actor": {
            "text":
                candidate.actor_text,
            "actor_class":
                candidate.actor_class,
            "actor_role":
                "hazard_actor",
            "evidence_text":
                candidate.clause_text,
        },
        "main_event": {
            "event_type":
                candidate.event_type,
            "predicate_text":
                _predicate_from_event(
                    candidate.event_type
                ),
            "path_or_object":
                candidate.path_or_object,
            "event_location_relation":
                candidate.location_relation,
            "location_relation_function":
                candidate.location_relation_function,
            "source_relation":
                candidate.source_relation,
            "target_relation":
                candidate.target_relation,
            "motion_axis":
                candidate.motion_axis,
            "motion_direction":
                candidate.motion_direction,
            "evidence_text":
                candidate.clause_text,
        },
        "ego_event": {
            "ego_maneuver":
                candidate.ego_maneuver,
            "evidence_text": (
                candidate.clause_text
                if candidate.ego_maneuver
                != "drive_forward"
                else ""
            ),
        },
        "road_context": {
            "road_type":
                candidate.road_type,
            "lane_context":
                candidate.lane_context,
            "evidence_text":
                candidate.clause_text,
        },
        "occlusion": {
            "enabled":
                candidate.occlusion_enabled,
            "occluder_type": (
                candidate.occluder_type
                if candidate.occlusion_enabled
                else "unknown"
            ),
            "relation_to_actor": (
                "behind"
                if candidate.occlusion_enabled
                else "unknown"
            ),
            "evidence_text": (
                candidate.clause_text
                if candidate.occlusion_enabled
                else ""
            ),
        },
        "event_sequence": [],
        "diagnostics": diagnostics,
        "missing_information": {
            "required": [
                "ego_speed",
                "actor_speed",
                "initial_distance",
            ],
            "defaultable": [],
            "distributional": {},
        },
        "completed_parameters":
            completed_parameters,
        "confidence": (
            0.72
            if candidate.focus_score >= 5.0
            else 0.55
        ),
        "parser_notes": (
            "parsed_by=narrative_focus_fallback; "
            f"selected={candidate.event_id}; "
            f"score={candidate.focus_score:.2f}; "
            "num_candidates="
            f"{len(analysis.candidates)}"
        ),
    }


def _predicate_from_event(
    event_type: str,
) -> str:
    return {
        "lead_vehicle_braking":
            "brakes",
        "lane_change_into_ego_lane":
            "merges into ego lane",
        "ego_left_turn_across_oncoming":
            "continues through ego left-turn path",
        "roundabout_entry_conflict":
            "conflicts at roundabout entry",
        "lane_blocking_conflict":
            "blocks ego lane",
        "object_in_lane":
            "blocks ego lane",
        "path_crossing":
            "crosses ego path",
        "enter_ego_lane":
            "enters ego lane",
    }.get(
        event_type,
        "unknown",
    )


def _dedupe(
    items: Iterable[str],
) -> List[str]:
    out: List[str] = []
    seen = set()

    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)

    return out