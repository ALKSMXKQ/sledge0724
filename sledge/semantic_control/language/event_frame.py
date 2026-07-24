"""EventFrame data model for natural-language traffic scene understanding.

EventFrame is an intermediate semantic representation between free-form
natural language and HazardSemanticSpec-like scene-control slots.

The goal is to prevent direct keyword-to-template matching.  Instead, the
pipeline first extracts event-level roles:

- who acts,
- what action is performed,
- what path/object the action applies to,
- where the event is anchored relative to ego,
- what function the spatial relation plays,
- what motion geometry the event implies.

This file intentionally keeps the data model lightweight and JSON-serializable
so it can be produced by an LLM, repaired by rules, and consumed by downstream
mappers/evaluators.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Type, TypeVar
import json


@dataclass
class ActorFrame:
    """Main non-ego actor mentioned in the prompt."""

    text: str = ""
    actor_class: str = "unknown"
    actor_role: str = "unknown"
    evidence_text: str = ""


@dataclass
class EgoEventFrame:
    """Ego vehicle event or maneuver context."""

    ego_maneuver: str = "drive_forward"
    ego_state: str = "normal_driving"
    evidence_text: str = ""


@dataclass
class MainEventFrame:
    """Main hazardous event extracted from the sentence."""

    event_type: str = "unknown"
    predicate_text: str = ""
    path_or_object: str = "unknown"

    # Spatial relation anchoring the event relative to ego or road context.
    event_location_relation: str = "unknown"
    location_relation_function: str = "unknown"

    # Source/target relations are especially useful for cut-in, merge,
    # left-turn and roundabout cases.
    source_relation: str = "unknown"
    target_relation: str = "unknown"

    # Motion semantics.
    motion_axis: str = "unknown"
    motion_direction: str = "unknown"

    evidence_text: str = ""


@dataclass
class RoadContextFrame:
    """Road topology and lane context."""

    road_type: str = "unknown"
    lane_context: str = "unknown"
    evidence_text: str = ""


@dataclass
class OcclusionFrame:
    """Visibility and occlusion context."""

    enabled: bool = False
    occluder_type: str = "unknown"
    relation_to_actor: str = "unknown"
    evidence_text: str = ""


@dataclass
class MissingInformationFrame:
    """Parameters that are not explicitly stated in natural language."""

    required: List[str] = field(default_factory=list)
    defaultable: List[str] = field(default_factory=list)
    distributional: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletedParameter:
    """A parameter explicitly stated or inferred by the parser."""

    value: Any = None
    unit: str = ""
    source: str = "unknown"
    reason: str = ""


@dataclass
class EventSequenceStep:
    """One event in the reconstructed temporal chain."""

    order: int = 0
    actor: str = "unknown"
    event_type: str = "unknown"
    action: str = "unknown"
    relation_to_previous: str = "unknown"
    evidence_text: str = ""


@dataclass
class DiagnosticsFrame:
    """Boolean diagnostics used by deterministic repair/mapping logic."""

    is_path_crossing: bool = False
    is_lane_change_into_ego_lane: bool = False
    is_longitudinal_following: bool = False
    is_ego_left_turn: bool = False
    is_roundabout_entry: bool = False
    is_occluded: bool = False


@dataclass
class EventFrame:
    """Top-level EventFrame object."""

    sentence: str = ""
    main_actor: ActorFrame = field(default_factory=ActorFrame)
    ego_event: EgoEventFrame = field(default_factory=EgoEventFrame)
    main_event: MainEventFrame = field(default_factory=MainEventFrame)
    road_context: RoadContextFrame = field(default_factory=RoadContextFrame)
    occlusion: OcclusionFrame = field(default_factory=OcclusionFrame)
    missing_information: MissingInformationFrame = field(default_factory=MissingInformationFrame)
    completed_parameters: Dict[str, CompletedParameter] = field(default_factory=dict)
    event_sequence: List[EventSequenceStep] = field(default_factory=list)
    diagnostics: DiagnosticsFrame = field(default_factory=DiagnosticsFrame)
    confidence: float = 0.0
    parser_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventFrame":
        data = normalize_event_frame_dict(data)

        completed_parameters = {}
        for k, v in dict(data.get("completed_parameters", {}) or {}).items():
            if isinstance(v, CompletedParameter):
                completed_parameters[k] = v
            elif isinstance(v, dict):
                completed_parameters[k] = _dataclass_from_dict(CompletedParameter, v)
            else:
                completed_parameters[k] = CompletedParameter(value=v)

        event_sequence = []
        for item in list(data.get("event_sequence", []) or []):
            if isinstance(item, EventSequenceStep):
                event_sequence.append(item)
            elif isinstance(item, dict):
                event_sequence.append(_dataclass_from_dict(EventSequenceStep, item))

        return cls(
            sentence=str(data.get("sentence", "")),
            main_actor=(
                data.get("main_actor")
                if isinstance(data.get("main_actor"), ActorFrame)
                else _dataclass_from_dict(ActorFrame, data.get("main_actor", {}))
            ),
            ego_event=(
                data.get("ego_event")
                if isinstance(data.get("ego_event"), EgoEventFrame)
                else _dataclass_from_dict(EgoEventFrame, data.get("ego_event", {}))
            ),
            main_event=(
                data.get("main_event")
                if isinstance(data.get("main_event"), MainEventFrame)
                else _dataclass_from_dict(MainEventFrame, data.get("main_event", {}))
            ),
            road_context=(
                data.get("road_context")
                if isinstance(data.get("road_context"), RoadContextFrame)
                else _dataclass_from_dict(RoadContextFrame, data.get("road_context", {}))
            ),
            occlusion=(
                data.get("occlusion")
                if isinstance(data.get("occlusion"), OcclusionFrame)
                else _dataclass_from_dict(OcclusionFrame, data.get("occlusion", {}))
            ),
            missing_information=(
                data.get("missing_information")
                if isinstance(data.get("missing_information"), MissingInformationFrame)
                else _dataclass_from_dict(MissingInformationFrame, data.get("missing_information", {}))
            ),
            completed_parameters=completed_parameters,
            event_sequence=event_sequence,
            diagnostics=(
                data.get("diagnostics")
                if isinstance(data.get("diagnostics"), DiagnosticsFrame)
                else _dataclass_from_dict(DiagnosticsFrame, data.get("diagnostics", {}))
            ),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            parser_notes=str(data.get("parser_notes", "")),
        )


T = TypeVar("T")


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _dataclass_from_dict(cls: Type[T], data: Any) -> T:
    """Construct a dataclass while ignoring extra LLM/debug fields."""

    data = _as_dict(data)
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in allowed})


def _with_aliases(data: Dict[str, Any], aliases: Dict[str, str]) -> Dict[str, Any]:
    out = dict(data)
    for alias, canonical in aliases.items():
        if alias in out and canonical not in out:
            out[canonical] = out[alias]
    return out


def _filter_keys(data: Dict[str, Any], cls: Type[Any]) -> Dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in allowed}


def normalize_event_frame_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize partially filled LLM JSON before EventFrame.from_dict.

    This helper is intentionally permissive because small LLMs often omit
    fields or use slightly different nested structures.
    """

    data = dict(data) if isinstance(data, dict) else {}

    data.setdefault("sentence", "")
    data.setdefault("main_actor", {})
    data.setdefault("ego_event", {})
    data.setdefault("main_event", {})
    data.setdefault("road_context", {})
    data.setdefault("occlusion", {})
    data.setdefault("missing_information", {})
    data.setdefault("completed_parameters", {})
    data.setdefault("event_sequence", [])
    data.setdefault("diagnostics", {})
    data.setdefault("confidence", 0.0)

    main_actor = _with_aliases(
        _as_dict(data.get("main_actor", {})),
        {
            "class": "actor_class",
            "actor_type": "actor_class",
            "type": "actor_class",
            "role": "actor_role",
        },
    )
    data["main_actor"] = _filter_keys({
        "text": "",
        "actor_class": "unknown",
        "actor_role": "unknown",
        "evidence_text": "",
        **main_actor,
    }, ActorFrame)

    data["ego_event"] = _filter_keys({
        "ego_maneuver": "drive_forward",
        "ego_state": "normal_driving",
        "evidence_text": "",
        **_as_dict(data.get("ego_event", {})),
    }, EgoEventFrame)

    main_event = _with_aliases(
        _as_dict(data.get("main_event", {})),
        {
            "type": "event_type",
            "predicate": "predicate_text",
            "action": "predicate_text",
            "object": "path_or_object",
            "path": "path_or_object",
            "location_relation": "event_location_relation",
            "relation_to_ego": "event_location_relation",
            "axis": "motion_axis",
            "direction": "motion_direction",
            "source": "source_relation",
            "target": "target_relation",
        },
    )
    data["main_event"] = _filter_keys({
        "event_type": "unknown",
        "predicate_text": "",
        "path_or_object": "unknown",
        "event_location_relation": "unknown",
        "location_relation_function": "unknown",
        "source_relation": "unknown",
        "target_relation": "unknown",
        "motion_axis": "unknown",
        "motion_direction": "unknown",
        "evidence_text": "",
        **main_event,
    }, MainEventFrame)

    data["road_context"] = _filter_keys({
        "road_type": "unknown",
        "lane_context": "unknown",
        "evidence_text": "",
        **_as_dict(data.get("road_context", {})),
    }, RoadContextFrame)

    occlusion = _with_aliases(
        _as_dict(data.get("occlusion", {})),
        {
            "occluded": "enabled",
            "type": "occluder_type",
            "class": "occluder_type",
        },
    )
    data["occlusion"] = _filter_keys({
        "enabled": False,
        "occluder_type": "unknown",
        "relation_to_actor": "unknown",
        "evidence_text": "",
        **occlusion,
    }, OcclusionFrame)

    data["missing_information"] = _filter_keys({
        "required": [],
        "defaultable": [],
        "distributional": {},
        **_as_dict(data.get("missing_information", {})),
    }, MissingInformationFrame)

    diagnostics = _as_dict(data.get("diagnostics", {}))
    if diagnostics.get("is_lateral_motion") and "is_path_crossing" not in diagnostics:
        diagnostics["is_path_crossing"] = True
    data["diagnostics"] = _filter_keys({
        "is_path_crossing": False,
        "is_lane_change_into_ego_lane": False,
        "is_longitudinal_following": False,
        "is_ego_left_turn": False,
        "is_roundabout_entry": False,
        "is_occluded": False,
        **diagnostics,
    }, DiagnosticsFrame)

    normalized_sequence = []
    for item in list(data.get("event_sequence", []) or []):
        if isinstance(item, EventSequenceStep):
            normalized_sequence.append(item)
        elif isinstance(item, dict):
            normalized_sequence.append(_filter_keys(item, EventSequenceStep))
    data["event_sequence"] = normalized_sequence

    normalized_params = {}
    for key, value in dict(data.get("completed_parameters", {}) or {}).items():
        if isinstance(value, CompletedParameter):
            normalized_params[key] = value
        elif isinstance(value, dict):
            normalized_params[key] = _filter_keys(value, CompletedParameter)
        else:
            normalized_params[key] = value
    data["completed_parameters"] = normalized_params

    data = _filter_keys(data, EventFrame)

    return data

def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from an LLM response string."""

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
                raw = text[start : i + 1]
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("extracted JSON is not an object")
                return normalize_event_frame_dict(data)

    raise ValueError("no complete JSON object found")
