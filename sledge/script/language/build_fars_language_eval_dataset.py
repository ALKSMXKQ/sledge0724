#!/usr/bin/env python3
"""Build language-evaluation cases from official NHTSA FARS crash data.

The script converts structured FARS crash records into:

1. a short natural-language hazardous-scene description, and
2. an `expected` slot template compatible with
   `sledge/script/language/compare_language_control_experiments.py`.

FARS source:
https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars
https://static.nhtsa.gov/nhtsa/downloads/FARS/2024/National/FARS2024NationalCSV.zip

The conversion is deliberately conservative. It only emits cases whose FARS
coded fields support one of the semantic-control scenario families already
represented by the language module: pedestrian/cyclist crossing, lead-vehicle
braking, adjacent-lane cut-in, left-turn across oncoming traffic, and in-lane
static obstacle.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_YEAR = 2024
DEFAULT_SOURCE_URL = (
    "https://static.nhtsa.gov/nhtsa/downloads/FARS/2024/National/"
    "FARS2024NationalCSV.zip"
)
DEFAULT_ZIP_PATH = Path(
    "sledge/semantic_control/configs/language_eval/raw/fars_2024/"
    "FARS2024NationalCSV.zip"
)
DEFAULT_OUTPUT_DIR = Path("sledge/semantic_control/configs/language_eval")


UNKNOWN_TERMS = {
    "",
    "unknown",
    "not reported",
    "reported as unknown",
    "not applicable",
    "not applicable (n/a)",
    "not an intersection",
    "not a cyclist",
    "not a pedestrian",
    "no",
    "none",
    "none noted",
}

FIXED_OBJECT_TERMS = [
    "concrete traffic barrier",
    "guardrail",
    "traffic barrier",
    "traffic sign",
    "utility pole",
    "tree",
    "ditch",
    "embankment",
    "culvert",
    "fence",
    "wall",
    "parked motor vehicle",
    "other object",
]


def stable_variant_index(key: str, count: int) -> int:
    if count <= 0:
        raise ValueError("count must be positive")
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(key)) % count


def render_variant(key: str, templates: Sequence[str], **values: str) -> str:
    return templates[stable_variant_index(key, len(templates))].format(**values)


def sentence_start(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def norm(text: Any) -> str:
    return str(text or "").strip()


def low(text: Any) -> str:
    return norm(text).lower()


def known_label(text: Any) -> str:
    value = norm(text)
    return "" if low(value) in UNKNOWN_TERMS else value


def has_any(text: Any, terms: Sequence[str]) -> bool:
    haystack = low(text)
    return any(term in haystack for term in terms)


def first_known(*values: Any) -> str:
    for value in values:
        label = known_label(value)
        if label:
            return label
    return ""


def open_csv_from_zip(zf: zipfile.ZipFile, basename: str) -> Iterable[Dict[str, str]]:
    matches = [name for name in zf.namelist() if name.lower().endswith("/" + basename.lower())]
    if not matches:
        raise FileNotFoundError(f"{basename} not found in {zf.filename}")
    raw = zf.open(matches[0])
    text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
    return csv.DictReader(text)


def download_if_needed(zip_path: Path, url: str, force: bool = False) -> None:
    if zip_path.exists() and not force:
        return
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    zip_path.write_bytes(data)
    print(f"[DOWNLOAD] wrote {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")


def read_fars_tables(zip_path: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[Dict[str, str]]], Dict[str, List[Dict[str, str]]]]:
    accidents: Dict[str, Dict[str, str]] = {}
    vehicles: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    pbtypes: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    with zipfile.ZipFile(zip_path) as zf:
        for row in open_csv_from_zip(zf, "accident.csv"):
            accidents[norm(row["ST_CASE"])] = row
        for row in open_csv_from_zip(zf, "vehicle.csv"):
            vehicles[norm(row["ST_CASE"])].append(row)
        for row in open_csv_from_zip(zf, "pbtype.csv"):
            pbtypes[norm(row["ST_CASE"])].append(row)

    return accidents, vehicles, pbtypes


def road_topology(accident: Dict[str, str], pb: Optional[Dict[str, str]] = None) -> str:
    fields = " ".join(
        [
            low(accident.get("RELJCT1NAME")),
            low(accident.get("RELJCT2NAME")),
            low(accident.get("TYP_INTNAME")),
            low(accident.get("FUNC_SYSNAME")),
            low(pb.get("PEDLOCNAME") if pb else ""),
            low(pb.get("BIKELOCNAME") if pb else ""),
        ]
    )
    if "intersection" in fields or "junction" in fields:
        return "intersection"
    return "straight_lane"


def environment_phrase(accident: Dict[str, str], vehicle: Optional[Dict[str, str]] = None) -> str:
    parts: List[str] = []
    light = known_label(accident.get("LGT_CONDNAME"))
    weather = known_label(accident.get("WEATHERNAME"))
    surface = known_label(vehicle.get("VSURCONDNAME") if vehicle else "")
    area = known_label(accident.get("RUR_URBNAME"))
    road = known_label(accident.get("FUNC_SYSNAME"))

    if light:
        parts.append(light.lower())
    if weather:
        parts.append(weather.lower())
    if surface and surface.lower() != "dry":
        parts.append(surface.lower() + " road surface")
    if area:
        parts.append(area.lower() + " area")
    if road:
        parts.append(road.lower())

    return ", ".join(parts)


def risk_level(accident: Dict[str, str], vehicle: Optional[Dict[str, str]] = None) -> str:
    fatals = int(norm(accident.get("FATALS")) or "0")
    speed = low(vehicle.get("SPEEDRELNAME") if vehicle else "")
    if fatals > 1 or "exceeded speed" in speed or "too fast" in speed:
        return "aggressive"
    return "moderate"


def expected_risk() -> Dict[str, List[str]]:
    # FARS contains fatal crashes only. The language module usually infers
    # either "moderate" or "aggressive" from short prompts, so the gold label
    # should not over-penalize risk wording when the core hazard semantics match.
    return {"any_of": ["moderate", "aggressive"]}


def source_metadata(
    accident: Dict[str, str],
    vehicle: Optional[Dict[str, str]] = None,
    pb: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "source_dataset": "NHTSA FARS",
        "source_year": int(norm(accident.get("YEAR")) or DEFAULT_YEAR),
        "st_case": norm(accident.get("ST_CASE")),
        "state": norm(accident.get("STATENAME")),
        "harmful_event": norm(accident.get("HARM_EVNAME")),
        "manner_of_collision": norm(accident.get("MAN_COLLNAME")),
        "light": norm(accident.get("LGT_CONDNAME")),
        "weather": norm(accident.get("WEATHERNAME")),
        "relation_to_junction": first_known(accident.get("RELJCT1NAME"), accident.get("RELJCT2NAME")),
        "road_relation": norm(accident.get("REL_ROADNAME")),
        "fatalities": int(norm(accident.get("FATALS")) or "0"),
    }
    if vehicle:
        metadata["vehicle_no"] = norm(vehicle.get("VEH_NO"))
        metadata["vehicle_body"] = norm(vehicle.get("BODY_TYPNAME"))
        metadata["precrash_movement"] = norm(vehicle.get("P_CRASH1NAME"))
        metadata["critical_precrash_event"] = norm(vehicle.get("P_CRASH2NAME"))
        metadata["accident_type"] = norm(vehicle.get("ACC_TYPENAME"))
        metadata["speed_related"] = norm(vehicle.get("SPEEDRELNAME"))
        metadata["road_surface"] = norm(vehicle.get("VSURCONDNAME"))
    if pb:
        metadata["nonmotorist_type"] = norm(pb.get("PBPTYPENAME"))
        metadata["pedestrian_crash_type"] = norm(pb.get("PEDCTYPENAME"))
        metadata["bicycle_crash_type"] = norm(pb.get("BIKECTYPENAME"))
        metadata["pedestrian_position"] = norm(pb.get("PEDPOSNAME"))
        metadata["bicycle_position"] = norm(pb.get("BIKEPOSNAME"))
        metadata["pedestrian_group"] = norm(pb.get("PEDCGPNAME"))
        metadata["bicycle_group"] = norm(pb.get("BIKECGPNAME"))
    return metadata


def case_record(
    *,
    case_id: str,
    group: str,
    prompt: str,
    expected: Dict[str, Any],
    accident: Dict[str, str],
    vehicle: Optional[Dict[str, str]] = None,
    pb: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    return {
        "id": case_id,
        "category": "fars_real_hazard_semantics",
        "group": group,
        "prompt": prompt,
        "expected": expected,
        "metadata": source_metadata(accident, vehicle, pb),
    }


def clean_fragment(text: Any) -> str:
    label = known_label(text)
    return label if label else "not specified"


def report_context(metadata: Dict[str, Any]) -> str:
    context = []
    if metadata.get("state"):
        context.append(f"state {metadata['state']}")
    if metadata.get("relation_to_junction"):
        context.append(f"junction relation {metadata['relation_to_junction']}")
    if metadata.get("road_relation"):
        context.append(f"road relation {metadata['road_relation']}")
    env = []
    if metadata.get("light"):
        env.append(str(metadata["light"]).lower())
    if metadata.get("weather"):
        env.append(str(metadata["weather"]).lower())
    if metadata.get("road_surface"):
        env.append(str(metadata["road_surface"]).lower() + " surface")
    if env:
        context.append("environment " + ", ".join(env))
    return "; ".join(context)


def render_report_prompt(record: Dict[str, Any]) -> str:
    """Render a FARS case as a report-like natural-language input.

    The text is intentionally based on original FARS named fields instead of the
    normalized hazard template. The expected JSON remains the gold label, but the
    model must infer that label from report semantics.
    """

    m = record["metadata"]
    group = record["group"]
    context = report_context(m)
    context_text = f" Context: {context}." if context else ""
    fatals = m.get("fatalities", 1)
    first_event = clean_fragment(m.get("harmful_event"))
    collision = clean_fragment(m.get("manner_of_collision"))
    movement = clean_fragment(m.get("precrash_movement"))
    critical = clean_fragment(m.get("critical_precrash_event"))
    acc_type = clean_fragment(m.get("accident_type"))

    if group in {"fars_pedestrian_crossing", "fars_cyclist_crossing"}:
        actor = clean_fragment(m.get("nonmotorist_type")).lower()
        ped_type = clean_fragment(m.get("pedestrian_crash_type"))
        bike_type = clean_fragment(m.get("bicycle_crash_type"))
        nm_type = bike_type if group == "fars_cyclist_crossing" else ped_type
        position = first_known(m.get("pedestrian_position"), m.get("bicycle_position"))
        cg = first_known(m.get("pedestrian_group"), m.get("bicycle_group"))
        details = [
            f"the first harmful event was recorded as {first_event}",
            f"the motor vehicle pre-crash movement was {movement}",
            f"the critical pre-crash event was {critical}",
            f"the non-motorist was listed as {actor}",
            f"crash type detail: {nm_type}",
        ]
        if position:
            details.append(f"non-motorist position: {position}")
        if cg:
            details.append(f"non-motorist circumstance group: {cg}")
        return (
            "Crash report excerpt for semantic scene reconstruction. "
            f"Treat the involved motor vehicle as ego. {'; '.join(details)}. "
            f"The report lists {fatals} fatality/fatalities.{context_text}"
        )

    if group == "fars_lead_vehicle_braking":
        return (
            "Crash report excerpt for semantic scene reconstruction. Treat the following vehicle as ego. "
            f"Manner of collision: {collision}; crash type: {acc_type}; "
            f"vehicle pre-crash movement: {movement}; critical pre-crash event: {critical}. "
            f"The report lists {fatals} fatality/fatalities.{context_text}"
        )

    if group == "fars_adjacent_lane_cutin":
        return (
            "Crash report excerpt for semantic scene reconstruction. Treat one vehicle as ego and infer the hazardous "
            "other vehicle from the pre-crash fields. "
            f"Manner of collision: {collision}; crash type: {acc_type}; "
            f"vehicle pre-crash movement: {movement}; critical pre-crash event: {critical}. "
            f"The report lists {fatals} fatality/fatalities.{context_text}"
        )

    if group == "fars_left_turn_oncoming":
        return (
            "Crash report excerpt for semantic scene reconstruction. Treat the turning vehicle as ego. "
            f"Manner of collision: {collision}; crash type: {acc_type}; "
            f"vehicle pre-crash movement: {movement}; critical pre-crash event: {critical}. "
            "The coded crash type indicates a vehicle turn across the path of traffic from the opposite direction. "
            f"The report lists {fatals} fatality/fatalities.{context_text}"
        )

    if group == "fars_static_obstacle":
        return (
            "Crash report excerpt for semantic scene reconstruction. Treat the moving vehicle as ego. "
            f"First harmful event: {first_event}; manner of collision: {collision}; "
            f"vehicle pre-crash movement: {movement}; critical pre-crash event: {critical}. "
            f"The report lists {fatals} fatality/fatalities.{context_text}"
        )

    return record["prompt"]


def apply_narrative_style(records: Sequence[Dict[str, Any]], style: str) -> List[Dict[str, Any]]:
    styled: List[Dict[str, Any]] = []
    for record in records:
        item = json.loads(json.dumps(record, ensure_ascii=False))
        item["metadata"]["canonical_prompt"] = item["prompt"]
        item["metadata"]["narrative_style"] = style
        if style == "report":
            item["prompt"] = render_report_prompt(item)
            item["category"] = "fars_report_narrative_semantics"
        elif style != "canonical":
            raise ValueError(f"Unsupported narrative style: {style}")
        styled.append(item)
    return styled


def classify_nonmotorist(
    accident: Dict[str, str],
    vehicles: List[Dict[str, str]],
    pb: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    actor = low(pb.get("PBPTYPENAME"))
    is_cyclist = "bicyclist" in actor or "pedalcyclist" in actor
    is_pedestrian = "pedestrian" in actor or "personal conveyance" in actor
    if not (is_cyclist or is_pedestrian):
        return None

    vehicle = vehicles[0] if vehicles else None
    actor_label = "cyclist" if is_cyclist else "pedestrian"
    actor_phrase = "a cyclist" if is_cyclist else "a pedestrian"
    actor_alt = "a bicyclist" if is_cyclist else "a person on foot"
    actor_start = sentence_start(actor_phrase)
    actor_alt_start = sentence_start(actor_alt)
    group = "fars_cyclist_crossing" if is_cyclist else "fars_pedestrian_crossing"

    crash_type = first_known(pb.get("BIKECTYPENAME") if is_cyclist else pb.get("PEDCTYPENAME"))
    group_name = first_known(pb.get("BIKECGPNAME") if is_cyclist else pb.get("PEDCGPNAME"))
    topo = road_topology(accident, pb)
    where = "at an intersection" if topo == "intersection" else "on a straight road segment"
    case_id = f"fars{accident.get('YEAR', DEFAULT_YEAR)}_{accident['ST_CASE']}_{actor_label}"
    occluded = has_any(crash_type + " " + group_name, ["visual obstruction", "dart-out"])

    if occluded:
        prompt = render_variant(
            case_id,
            [
                "{actor_start} emerges from behind an obstruction and crosses into ego's path {where}.",
                "A hidden {actor_plain} appears from the roadside and enters the lane ahead of ego {where}.",
                "{actor_start} is screened from view, then moves laterally into ego's path {where}.",
                "Ego approaches {where} as {actor} pops out from occlusion and cuts across the lane.",
                "A previously occluded {actor_plain} steps into the ego lane just ahead {where}.",
                "{actor_start} comes out from a blind spot and traverses the ego vehicle's forward path {where}.",
            ],
            actor=actor_phrase,
            actor_start=actor_start,
            actor_plain=actor_label if is_cyclist else "pedestrian",
            where=where,
        )
    else:
        prompt = render_variant(
            case_id,
            [
                "{actor_start} crosses into the ego lane just ahead {where}.",
                "Ego continues forward {where} while {actor} moves laterally across its path.",
                "{actor_alt_start} enters the lane in front of ego, creating a crossing conflict {where}.",
                "A crossing {actor_plain} cuts across ego's forward path {where}.",
                "{actor_start} traverses the ego lane with little room for ego to react {where}.",
                "Ego faces a lateral conflict as {actor} crosses directly ahead {where}.",
                "{actor_alt_start} moves from the roadside into ego's travel path {where}.",
                "The ego vehicle is confronted by {actor} crossing the lane ahead {where}.",
            ],
            actor=actor_phrase,
            actor_start=actor_start,
            actor_alt_start=actor_alt_start,
            actor_plain=actor_label,
            where=where,
        )

    expected: Dict[str, Any] = {
        "actor_layer.primary_actor": actor_label,
        "actor_layer.actor_role": "crossing_actor",
        "interaction_layer.conflict_type": {"any_of": ["lateral_conflict", "crossing_path_conflict"]},
        "motion_layer.hazard_event_type": {"any_of": ["path_crossing", "enter_ego_lane"]},
        "motion_layer.motion_axis": "lateral",
        "interaction_layer.anchor_region": "front",
        "road_layer.road_topology": topo,
        "risk_layer.risk_level": expected_risk(),
    }
    if occluded:
        expected["object_layer.occlusion.enabled"] = True

    return case_record(
        case_id=case_id,
        group=group,
        prompt=prompt,
        expected=expected,
        accident=accident,
        vehicle=vehicle,
        pb=pb,
    )


def classify_left_turn_oncoming(accident: Dict[str, str], vehicles: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    joined = " ".join(low(v.get("ACC_TYPENAME")) + " " + low(v.get("P_CRASH1NAME")) + " " + low(v.get("P_CRASH2NAME")) for v in vehicles)
    if not (
        "turn across path" in joined
        or ("turning left" in joined and ("opposite direction" in joined or "oncoming" in joined))
    ):
        return None
    vehicle = next((v for v in vehicles if has_any(v.get("P_CRASH1NAME"), ["turning left"])), vehicles[0] if vehicles else None)
    case_id = f"fars{accident.get('YEAR', DEFAULT_YEAR)}_{accident['ST_CASE']}_left_turn_oncoming"
    prompt = render_variant(
        case_id,
        [
            "Ego turns left across an intersection while an oncoming vehicle continues straight toward the conflict point.",
            "An unprotected left turn puts ego across the path of a straight-moving oncoming vehicle.",
            "Ego begins a left turn and conflicts with traffic approaching from the opposite direction.",
            "A vehicle coming from the opposite direction goes straight as ego cuts across for a left turn.",
            "Ego's left-turn path crosses in front of an oncoming through vehicle at the intersection.",
            "At an intersection, ego turns left into the path of an approaching vehicle.",
            "Ego attempts a left turn while opposing traffic proceeds through the same conflict area.",
        ],
    )
    expected = {
        "road_layer.road_topology": "intersection",
        "road_layer.generated_road_layout": "unprotected_left_turn",
        "actor_layer.primary_actor": "vehicle",
        "actor_layer.actor_role": "approaching_actor",
        "interaction_layer.conflict_type": "oncoming_conflict",
        "interaction_layer.conflict_direction": "opposite",
        "motion_layer.ego_maneuver": "left_turn",
        "motion_layer.hazard_event_type": {"any_of": ["left_turn_across_oncoming", "ego_left_turn_across_oncoming", "oncoming_through_conflict"]},
        "risk_layer.risk_level": expected_risk(),
    }
    return case_record(
        case_id=case_id,
        group="fars_left_turn_oncoming",
        prompt=prompt,
        expected=expected,
        accident=accident,
        vehicle=vehicle,
    )


def classify_lead_braking(accident: Dict[str, str], vehicles: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    vehicle = next(
        (
            v
            for v in vehicles
            if has_any(
                low(v.get("ACC_TYPENAME")) + " " + low(v.get("P_CRASH2NAME")),
                ["rear end-stopped", "rear end-slower", "rear end-decelerating", "other vehicle stopped", "decelerating", "lower steady speed"],
            )
        ),
        None,
    )
    if vehicle is None:
        return None
    event = low(vehicle.get("P_CRASH2NAME"))
    action = "stops short" if "stopped" in event else "decelerates"
    if "lower steady speed" in event:
        action = "slows"
    case_id = f"fars{accident.get('YEAR', DEFAULT_YEAR)}_{accident['ST_CASE']}_lead_braking"
    prompt = render_variant(
        case_id,
            [
                "The lead vehicle in ego's lane {action}, leaving ego with very little reaction distance.",
                "A vehicle directly ahead of ego {action}, creating a close longitudinal conflict.",
                "Ego follows a lead vehicle that {action}, forcing an urgent braking response.",
                "The front vehicle {action} in the same lane, compressing the gap to ego.",
                "Traffic ahead of ego {action}, producing a short-headway braking hazard.",
                "Ego closes on a lead vehicle that {action} with almost no buffer.",
                "A same-lane vehicle ahead {action}, putting braking pressure on ego.",
            ],
            action=action,
        )
    expected = {
        "actor_layer.primary_actor": "lead_vehicle",
        "actor_layer.actor_role": "braking_actor",
        "interaction_layer.conflict_type": "longitudinal_conflict",
        "interaction_layer.conflict_direction": "front",
        "interaction_layer.interaction_goal": "braking_pressure",
        "interaction_layer.distance_relation": {"any_of": ["small_gap", "close", "short_headway", "medium"]},
        "motion_layer.hazard_event_type": {"any_of": ["lead_vehicle_braking", "hard_stop_ahead"]},
        "risk_layer.risk_level": expected_risk(),
    }
    return case_record(
        case_id=case_id,
        group="fars_lead_vehicle_braking",
        prompt=prompt,
        expected=expected,
        accident=accident,
        vehicle=vehicle,
    )


def classify_cutin(accident: Dict[str, str], vehicles: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    vehicle = next(
        (
            v
            for v in vehicles
            if has_any(
                low(v.get("P_CRASH1NAME")) + " " + low(v.get("P_CRASH2NAME")) + " " + low(v.get("ACC_TYPENAME")),
                ["changing lanes", "merging", "from adjacent lane", "over left lane line", "over right lane line"],
            )
        ),
        None,
    )
    if vehicle is None:
        return None
    vehicle_fields = low(vehicle.get("P_CRASH1NAME")) + " " + low(vehicle.get("P_CRASH2NAME")) + " " + low(vehicle.get("ACC_TYPENAME"))
    if has_any(vehicle_fields, ["opposite direction", "pedestrian in road", "pedalcyclist", "non-motorist"]):
        return None
    has_lane_evidence = has_any(vehicle_fields, ["from adjacent lane", "over left lane line", "over right lane line", "changing lanes to"])
    is_motor_vehicle_event = "motor vehicle in-transport" in low(accident.get("HARM_EVNAME"))
    if not (has_lane_evidence or (has_any(vehicle.get("P_CRASH1NAME"), ["changing lanes", "merging"]) and is_motor_vehicle_event)):
        return None
    fields = low(vehicle.get("P_CRASH2NAME")) + " " + low(vehicle.get("ACC_TYPENAME"))
    direction = "left_merge" if "over left" in fields or "left/right" in fields else "right_merge" if "over right" in fields else {"any_of": ["left_merge", "right_merge"]}
    side_from = "from ego's left" if direction == "left_merge" else "from ego's right" if direction == "right_merge" else "from a neighboring lane"
    side_lane = "the left adjacent lane" if direction == "left_merge" else "the right adjacent lane" if direction == "right_merge" else "an adjacent lane"
    case_id = f"fars{accident.get('YEAR', DEFAULT_YEAR)}_{accident['ST_CASE']}_cutin"
    prompt = render_variant(
        case_id,
        [
            "Another vehicle moves {side_from} into ego's lane, cutting in with a very small gap.",
            "A same-direction vehicle merges out of {side_lane} and squeezes into the ego lane.",
            "Ego's lane is cut off by a vehicle entering {side_from} at close range.",
            "A neighboring vehicle makes a tight lane change {side_from}, leaving ego little room.",
            "Traffic beside ego shifts into the ego lane {side_from}, creating a close cut-in.",
            "A vehicle from {side_lane} crosses into ego's path with minimal clearance.",
            "Ego is forced to react as an adjacent vehicle abruptly occupies its lane.",
            "A nearby vehicle changes lanes into ego's lane before a safe gap opens.",
        ],
        side_from=side_from,
        side_lane=side_lane,
    )
    expected = {
        "actor_layer.primary_actor": "cutin_vehicle",
        "actor_layer.actor_role": "merging_actor",
        "interaction_layer.conflict_type": "merging_conflict",
        "interaction_layer.distance_relation": {"any_of": ["small_gap", "close"]},
        "motion_layer.hazard_event_type": "lane_change_into_ego_lane",
        "risk_layer.risk_level": expected_risk(),
    }
    if isinstance(direction, str):
        expected["interaction_layer.conflict_direction"] = direction
    return case_record(
        case_id=case_id,
        group="fars_adjacent_lane_cutin",
        prompt=prompt,
        expected=expected,
        accident=accident,
        vehicle=vehicle,
    )


def classify_static_obstacle(accident: Dict[str, str], vehicles: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    harm = low(accident.get("HARM_EVNAME"))
    vehicle = next((v for v in vehicles if has_any(v.get("P_CRASH2NAME"), ["object in road", "animal in road"])), vehicles[0] if vehicles else None)
    pcrash = low(vehicle.get("P_CRASH2NAME") if vehicle else "")
    if not (has_any(harm, FIXED_OBJECT_TERMS) or "object in road" in pcrash):
        return None
    road_relation = low(accident.get("REL_ROADNAME"))
    if "object in road" not in pcrash and "animal in road" not in pcrash and "on roadway" not in road_relation:
        return None
    if has_any(harm + " " + pcrash, ["barrier", "guardrail"]):
        obstacle_type = "barrier"
        object_phrase = "a traffic barrier"
    elif "parked motor vehicle" in harm:
        obstacle_type = "object"
        object_phrase = "a parked vehicle"
    elif "animal in road" in pcrash:
        obstacle_type = "object"
        object_phrase = "an object or animal"
    else:
        obstacle_type = "object"
        object_phrase = "a fixed object"
    object_start = sentence_start(object_phrase)
    case_id = f"fars{accident.get('YEAR', DEFAULT_YEAR)}_{accident['ST_CASE']}_static_obstacle"
    prompt = render_variant(
        case_id,
        [
            "{object_start} occupies the ego lane ahead and blocks the vehicle's path.",
            "Ego approaches {object_phrase} sitting in the travel lane ahead.",
            "The lane in front of ego is obstructed by {object_phrase}.",
            "{object_start} lies ahead in ego's lane, creating a front blocking hazard.",
            "Ego's forward path is blocked by {object_phrase} in the lane.",
            "{object_start} is directly ahead in the ego lane as a static obstacle.",
            "Ego encounters {object_phrase} occupying the same lane ahead.",
        ],
        object_phrase=object_phrase,
        object_start=object_start,
    )
    expected = {
        "actor_layer.primary_actor": "static_obstacle",
        "actor_layer.actor_role": "blocking_actor",
        "interaction_layer.conflict_type": "lane_blocking_conflict",
        "interaction_layer.conflict_direction": "front",
        "motion_layer.hazard_event_type": {"any_of": ["object_in_lane", "lane_blocking_conflict"]},
        "object_layer.static_obstacle.obstacle_type": obstacle_type,
        "risk_layer.risk_level": expected_risk(),
    }
    return case_record(
        case_id=case_id,
        group="fars_static_obstacle",
        prompt=prompt,
        expected=expected,
        accident=accident,
        vehicle=vehicle,
    )


def classify_case(
    accident: Dict[str, str],
    vehicles: List[Dict[str, str]],
    pbtypes: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for pb in pbtypes:
        record = classify_nonmotorist(accident, vehicles, pb)
        if record:
            records.append(record)
    if records:
        return records

    for classifier in [
        classify_left_turn_oncoming,
        classify_lead_braking,
        classify_cutin,
        classify_static_obstacle,
    ]:
        record = classifier(accident, vehicles)
        if record:
            records.append(record)
            break

    return records


def build_records(
    accidents: Dict[str, Dict[str, str]],
    vehicles: Dict[str, List[Dict[str, str]]],
    pbtypes: Dict[str, List[Dict[str, str]]],
    *,
    max_per_group: int,
    seed: int,
    narrative_style: str,
) -> List[Dict[str, Any]]:
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for st_case in sorted(accidents):
        for record in classify_case(accidents[st_case], vehicles.get(st_case, []), pbtypes.get(st_case, [])):
            by_group[record["group"]].append(record)

    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    for group in sorted(by_group):
        group_records = by_group[group]
        rng.shuffle(group_records)
        selected.extend(group_records[:max_per_group])

    selected.sort(key=lambda item: (item["group"], item["id"]))
    return apply_narrative_style(selected, narrative_style)


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"[WRITE] {path} ({len(records)} cases)")


def split_records(records: Sequence[Dict[str, Any]], train_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[record["group"]].append(record)

    rng = random.Random(seed)
    train: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for group_records in by_group.values():
        shuffled = list(group_records)
        rng.shuffle(shuffled)
        n_train = int(round(len(shuffled) * train_ratio))
        if len(shuffled) > 1:
            n_train = min(max(1, n_train), len(shuffled) - 1)
        train.extend(shuffled[:n_train])
        test.extend(shuffled[n_train:])

    train.sort(key=lambda item: (item["group"], item["id"]))
    test.sort(key=lambda item: (item["group"], item["id"]))
    return train, test


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for record in records:
        counts[record["group"]] += 1
    return dict(sorted(counts.items()))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip_path", type=Path, default=DEFAULT_ZIP_PATH)
    ap.add_argument("--download", action="store_true", help="Download FARS zip if it is missing.")
    ap.add_argument("--force_download", action="store_true", help="Re-download even when zip_path exists.")
    ap.add_argument("--source_url", default=DEFAULT_SOURCE_URL)
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--output_name", default=None)
    ap.add_argument(
        "--narrative_style",
        choices=["report", "canonical"],
        default="report",
        help=(
            "report uses FARS report-field excerpts as the natural-language input; "
            "canonical uses normalized scene descriptions."
        ),
    )
    ap.add_argument("--max_per_group", type=int, default=80)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=17)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.download or not args.zip_path.exists():
        download_if_needed(args.zip_path, args.source_url, force=args.force_download)
    if not args.zip_path.exists():
        raise FileNotFoundError(f"FARS zip not found: {args.zip_path}")

    accidents, vehicles, pbtypes = read_fars_tables(args.zip_path)
    output_name = args.output_name
    if output_name is None:
        output_name = (
            "fars2024_report_narrative_cases"
            if args.narrative_style == "report"
            else "fars2024_real_hazard_cases"
        )

    records = build_records(
        accidents,
        vehicles,
        pbtypes,
        max_per_group=args.max_per_group,
        seed=args.seed,
        narrative_style=args.narrative_style,
    )
    if not records:
        raise RuntimeError("No FARS records matched the supported semantic scenario families.")

    all_path = args.output_dir / f"{output_name}.jsonl"
    train_path = args.output_dir / f"{output_name}_train.jsonl"
    test_path = args.output_dir / f"{output_name}_test.jsonl"

    train, test = split_records(records, args.train_ratio, args.seed)
    write_jsonl(all_path, records)
    write_jsonl(train_path, train)
    write_jsonl(test_path, test)

    summary = {
        "source_url": args.source_url,
        "zip_path": str(args.zip_path),
        "narrative_style": args.narrative_style,
        "all": len(records),
        "train": len(train),
        "test": len(test),
        "groups": summarize(records),
    }
    summary_path = args.output_dir / f"{output_name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
