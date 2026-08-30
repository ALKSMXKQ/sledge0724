"""Hard semantic protection shared by RVAE reconstruction and diffusion.

The learned decoder is allowed to change background traffic, but the scene
elements that define an occluded-pedestrian event are copied from the verified
B1 template and checked byte-for-byte before a cache is accepted.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeVector,
    SledgeVectorElement,
)


PROTECTED_ELEMENT_NAMES = ("lines", "ego")


def make_simulation_compatible_vector(
    processed: SledgeVector,
    raw_scene: Any,
) -> SledgeVector:
    """Return the scalar-ego representation consumed by ``SledgeScenario``."""

    raw_states = np.asarray(raw_scene.ego.states, dtype=np.float32).reshape(-1)
    raw_mask = np.asarray(raw_scene.ego.mask).reshape(-1)
    speed = float(raw_states[0]) if raw_states.size else 0.0
    valid = bool(raw_mask[0]) if raw_mask.size else True
    return SledgeVector(
        lines=processed.lines,
        vehicles=processed.vehicles,
        pedestrians=processed.pedestrians,
        static_objects=processed.static_objects,
        green_lights=processed.green_lights,
        red_lights=processed.red_lights,
        ego=SledgeVectorElement(
            states=np.asarray([speed], dtype=np.float32),
            mask=np.asarray([valid], dtype=bool),
        ),
    )


def match_processed_slot(
    raw_elem: Any,
    raw_index: int,
    vector_elem: Any,
) -> int:
    """Find the processed fixed-capacity slot corresponding to one raw slot."""

    raw_states = np.asarray(raw_elem.states)
    if raw_index < 0 or raw_index >= len(raw_states):
        return -1
    target = raw_states[raw_index]
    states = np.asarray(vector_elem.states)
    masks = np.asarray(vector_elem.mask).reshape(-1) >= 0.3
    valid = np.where(masks)[0]
    if not len(valid):
        return -1
    width = min(5, states.shape[-1], target.shape[-1])
    scales = np.asarray(
        [1.0, 1.0, 0.5, 0.25, 0.25],
        dtype=np.float32,
    )[:width]
    errors = np.linalg.norm(
        (states[valid, :width] - target[:width]) * scales,
        axis=1,
    )
    return int(valid[int(np.argmin(errors))])


def resolve_processed_slots(
    raw_scene: Any,
    processed_vector: Any,
    edit_report: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve controlled raw indices after fixed-capacity preprocessing."""

    resolved = dict(edit_report)
    resolved["pedestrian_index"] = match_processed_slot(
        raw_scene.pedestrians,
        int(edit_report.get("pedestrian_index", -1)),
        processed_vector.pedestrians,
    )
    occluder_name = str(edit_report.get("occluder_elem_name", "vehicles"))
    if occluder_name not in {"vehicles", "static_objects"}:
        raise ValueError(f"Unsupported occluder element: {occluder_name!r}")
    resolved["occluder_elem_name"] = occluder_name
    resolved["occluder_index"] = match_processed_slot(
        getattr(raw_scene, occluder_name),
        int(edit_report.get("occluder_index", -1)),
        getattr(processed_vector, occluder_name),
    )
    if int(resolved["pedestrian_index"]) < 0:
        raise RuntimeError("Controlled pedestrian was lost during preprocessing")
    if int(resolved["occluder_index"]) < 0:
        raise RuntimeError("Controlled occluder was lost during preprocessing")
    return resolved


def protected_slots(edit_report: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the serialized semantic-protection contract."""

    return {
        "road_topology": "all_lines",
        "ego": "complete_state",
        "pedestrians": int(edit_report.get("pedestrian_index", -1)),
        "occluder_element": str(
            edit_report.get("occluder_elem_name", "vehicles")
        ),
        "occluder_index": int(edit_report.get("occluder_index", -1)),
    }


def composite_protected_semantics(
    target: Any,
    template: Any,
    edit_report: Mapping[str, Any],
) -> None:
    """Copy the complete occlusion-defining state from B1 into ``target``."""

    for name in PROTECTED_ELEMENT_NAMES:
        target_elem = getattr(target, name)
        source_elem = getattr(template, name)
        target_elem.states = np.asarray(source_elem.states).copy()
        target_elem.mask = np.asarray(source_elem.mask).copy()

    copy_element_slot(
        target.pedestrians,
        template.pedestrians,
        int(edit_report.get("pedestrian_index", -1)),
    )
    occluder_name = str(edit_report.get("occluder_elem_name", "vehicles"))
    if occluder_name not in {"vehicles", "static_objects"}:
        raise ValueError(f"Unsupported occluder element: {occluder_name!r}")
    copy_element_slot(
        getattr(target, occluder_name),
        getattr(template, occluder_name),
        int(edit_report.get("occluder_index", -1)),
    )


def copy_element_slot(
    target_elem: Any,
    source_elem: Any,
    index: int,
) -> None:
    """Copy one complete entity state and mask without changing slot identity."""

    target_states = np.asarray(target_elem.states)
    source_states = np.asarray(source_elem.states)
    target_mask = np.asarray(target_elem.mask)
    source_mask = np.asarray(source_elem.mask)
    if index < 0:
        raise IndexError("Protected slot must be non-negative")
    if index >= len(target_states) or index >= len(source_states):
        raise IndexError(
            f"Protected slot {index} is outside decoded/template capacity"
        )
    if target_states[index].shape != source_states[index].shape:
        raise ValueError(
            "Protected target/template state shapes differ: "
            f"{target_states[index].shape} vs {source_states[index].shape}"
        )
    if target_mask.reshape(-1).size <= index or source_mask.reshape(-1).size <= index:
        raise IndexError(f"Protected mask slot {index} is outside capacity")
    target_states[index] = source_states[index]
    target_mask.reshape(-1)[index] = source_mask.reshape(-1)[index]
    target_elem.states = target_states
    target_elem.mask = target_mask


def audit_protected_semantics(
    candidate: Any,
    template: Any,
    edit_report: Mapping[str, Any],
) -> Dict[str, Any]:
    """Prove that every field defining the hazard survived exactly."""

    pedestrian_index = int(edit_report.get("pedestrian_index", -1))
    occluder_name = str(edit_report.get("occluder_elem_name", "vehicles"))
    occluder_index = int(edit_report.get("occluder_index", -1))
    checks = {
        "road_states_exact": _element_states_exact(
            candidate.lines,
            template.lines,
        ),
        "road_mask_exact": _element_mask_exact(
            candidate.lines,
            template.lines,
        ),
        "ego_states_exact": _element_states_exact(
            candidate.ego,
            template.ego,
        ),
        "ego_mask_exact": _element_mask_exact(
            candidate.ego,
            template.ego,
        ),
        "pedestrian_state_exact": _slot_states_exact(
            candidate.pedestrians,
            template.pedestrians,
            pedestrian_index,
        ),
        "pedestrian_mask_exact": _slot_mask_exact(
            candidate.pedestrians,
            template.pedestrians,
            pedestrian_index,
        ),
        "occluder_state_exact": _slot_states_exact(
            getattr(candidate, occluder_name),
            getattr(template, occluder_name),
            occluder_index,
        ),
        "occluder_mask_exact": _slot_mask_exact(
            getattr(candidate, occluder_name),
            getattr(template, occluder_name),
            occluder_index,
        ),
    }
    return {
        "schema_version": "occluded_pedestrian_semantic_protection_v1",
        "overall_pass": bool(all(checks.values())),
        "checks": checks,
        "protected_slots": protected_slots(edit_report),
    }


def _element_states_exact(left: Any, right: Any) -> bool:
    return bool(np.array_equal(np.asarray(left.states), np.asarray(right.states)))


def _element_mask_exact(left: Any, right: Any) -> bool:
    return bool(np.array_equal(np.asarray(left.mask), np.asarray(right.mask)))


def _slot_states_exact(left: Any, right: Any, index: int) -> bool:
    left_states = np.asarray(left.states)
    right_states = np.asarray(right.states)
    return bool(
        index >= 0
        and index < len(left_states)
        and index < len(right_states)
        and np.array_equal(left_states[index], right_states[index])
    )


def _slot_mask_exact(left: Any, right: Any, index: int) -> bool:
    left_mask = np.asarray(left.mask).reshape(-1)
    right_mask = np.asarray(right.mask).reshape(-1)
    return bool(
        index >= 0
        and index < len(left_mask)
        and index < len(right_mask)
        and left_mask[index] == right_mask[index]
    )


__all__ = [
    "audit_protected_semantics",
    "composite_protected_semantics",
    "copy_element_slot",
    "make_simulation_compatible_vector",
    "match_processed_slot",
    "protected_slots",
    "resolve_processed_slots",
]
