"""nuPlan-visible object categories for occluded-pedestrian exports."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping


SUPPORTED_OCCLUDER_TYPES = (
    "vehicle",
    "bicycle",
    "generic_object",
    "traffic_cone",
    "barrier",
    "czone_sign",
)

OCCLUDER_TRACKED_OBJECT_TYPES: Dict[str, str] = {
    "vehicle": "VEHICLE",
    "bicycle": "BICYCLE",
    "generic_object": "GENERIC_OBJECT",
    "traffic_cone": "TRAFFIC_CONE",
    "barrier": "BARRIER",
    "czone_sign": "CZONE_SIGN",
}

OCCLUDER_ELEMENT_NAMES: Dict[str, str] = {
    "vehicle": "vehicles",
    "bicycle": "vehicles",
    "generic_object": "static_objects",
    "traffic_cone": "static_objects",
    "barrier": "static_objects",
    "czone_sign": "static_objects",
}

OCCLUDER_ALIASES: Dict[str, str] = {
    "car": "vehicle",
    "parked_car": "vehicle",
    "parked_vehicle": "vehicle",
    "occluding_vehicle": "vehicle",
    # nuPlan does not expose these vehicle subtypes in this SLEDGE cache.
    "van": "vehicle",
    "delivery_van": "vehicle",
    "minivan": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "bike": "bicycle",
    "cyclist": "bicycle",
    "generic": "generic_object",
    "static_object": "generic_object",
    "cone": "traffic_cone",
    "road_cone": "traffic_cone",
    "road_barrier": "barrier",
    "construction_sign": "czone_sign",
    "construction_zone_sign": "czone_sign",
    "c_zone_sign": "czone_sign",
}

TYPE_METADATA_KEY = "__sledge_object_type_overrides__"


def normalize_occluder_type(value: Any, *, strict: bool = False) -> str:
    raw = str(value or "vehicle").lower().strip().replace("-", "_").replace(" ", "_")
    normalized = OCCLUDER_ALIASES.get(raw, raw)
    if normalized in SUPPORTED_OCCLUDER_TYPES:
        return normalized
    if strict:
        raise ValueError(
            f"Unsupported occluder_type={value!r}; expected one of {list(SUPPORTED_OCCLUDER_TYPES)}"
        )
    return "vehicle"


def tracked_object_type_name(occluder_type: Any) -> str:
    return OCCLUDER_TRACKED_OBJECT_TYPES[normalize_occluder_type(occluder_type, strict=True)]


def element_name_for_occluder(occluder_type: Any) -> str:
    return OCCLUDER_ELEMENT_NAMES[normalize_occluder_type(occluder_type, strict=True)]


def make_type_override(element_name: str, index: int, tracked_object_type: str) -> Dict[str, Dict[str, str]]:
    if element_name not in {"vehicles", "pedestrians", "static_objects"}:
        raise ValueError(f"Unsupported element_name={element_name!r}")
    return {element_name: {str(int(index)): str(tracked_object_type)}}


def embed_type_overrides(gz_path: Path, overrides: Mapping[str, Mapping[str, str]]) -> Path:
    """Embed type metadata in a normal SLEDGE pickle without breaking old readers.

    ``SledgeVector.deserialize`` ignores unknown dictionary keys, while the
    updated scenario loader consumes this reserved key. Thus the cache remains
    readable by existing feature tooling and carries its visible nuPlan type in
    the same ``.gz`` file.
    """

    path = Path(gz_path)
    with gzip.open(path, "rb") as fp:
        payload = pickle.load(fp)
    if not isinstance(payload, dict):
        raise TypeError(f"Typed SLEDGE cache must contain a dictionary, got {type(payload)}")
    payload[TYPE_METADATA_KEY] = {
        str(element): {str(index): str(type_name) for index, type_name in entries.items()}
        for element, entries in overrides.items()
    }
    with gzip.open(path, "wb", compresslevel=1) as fp:
        pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def read_type_overrides(gz_path: Path) -> Dict[str, Dict[str, str]]:
    path = Path(gz_path)
    with gzip.open(path, "rb") as fp:
        payload = pickle.load(fp)
    if not isinstance(payload, dict):
        return {}
    raw = payload.get(TYPE_METADATA_KEY, {})
    if not isinstance(raw, dict):
        return {}
    return {
        str(element): {str(index): str(type_name) for index, type_name in entries.items()}
        for element, entries in raw.items()
        if isinstance(entries, dict)
    }


def audit_simulator_roundtrip(
    gz_path: Path,
    overrides: Mapping[str, Mapping[str, str]],
) -> Dict[str, Any]:
    """Open a gzip through ``SledgeScenario`` and verify visible types."""

    from nuplan.common.actor_state.tracked_objects_types import (
        TrackedObjectType,
    )
    from sledge.simulation.scenarios.sledge_scenario.sledge_scenario import (
        SledgeScenario,
    )

    path = Path(gz_path)
    scenario = SledgeScenario(path.with_suffix(""))
    detections = scenario.initial_tracked_objects.tracked_objects
    observed = {
        str(obj.track_token): str(obj.tracked_object_type.name)
        for obj in detections
    }
    checks: Dict[str, bool] = {}
    for element_name, entries in overrides.items():
        for index, type_name in entries.items():
            expected_type = TrackedObjectType[str(type_name)]
            token = f"{expected_type.value}_{int(index)}"
            checks[f"{element_name}:{index}:{type_name}"] = bool(
                observed.get(token) == str(type_name)
            )
    return {
        "overall_pass": bool(checks and all(checks.values())),
        "checks": checks,
        "observed_types": observed,
        "gzip_path": str(path),
    }
