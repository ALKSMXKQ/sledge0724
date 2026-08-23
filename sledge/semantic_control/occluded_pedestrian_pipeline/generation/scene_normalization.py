"""Normalize variable-length raw cache elements into safe editable arrays."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Tuple

import numpy as np


DEFAULT_CAPACITIES = {
    "vehicles": (1, 6),
    "pedestrians": (1, 6),
    "static_objects": (1, 5),
}


def normalize_editable_scene(scene: Any) -> Tuple[Any, Dict[str, Any]]:
    """Return a copy with insertion capacity, preserving every existing row."""

    normalized = deepcopy(scene)
    report: Dict[str, Any] = {"schema_version": "editable_scene_normalization_v1", "elements": {}}
    for name, (minimum_capacity, default_width) in DEFAULT_CAPACITIES.items():
        elem = getattr(normalized, name)
        states = np.asarray(elem.states)
        mask = np.asarray(elem.mask).reshape(-1)
        before_shape = list(states.shape)

        if states.ndim >= 2:
            row_width = int(states.shape[-1])
            rows = states.reshape((-1, row_width))
        elif states.size == 0:
            row_width = default_width
            rows = np.zeros((0, row_width), dtype=np.float32)
        else:
            row_width = default_width
            if states.size % row_width != 0:
                raise ValueError(f"Cannot normalize {name} states with shape={states.shape}")
            rows = states.reshape((-1, row_width))

        existing = int(rows.shape[0])
        capacity = max(minimum_capacity, existing + 1)
        out_states = np.zeros((capacity, row_width), dtype=rows.dtype if rows.size else np.float32)
        out_mask = np.zeros((capacity,), dtype=mask.dtype if mask.size else np.float32)
        if existing:
            out_states[:existing] = rows

            # Variable-length SLEDGE raw caches contain semantic entity rows
            # even when their stored raw mask is all False. Downstream raw
            # preprocessing consumes the rows themselves and does not use that
            # mask to discard entities. Treat every pre-existing row as
            # occupied so insertion can only use the newly appended slot and
            # compaction can never erase the source scene.
            out_mask[:existing] = True
        elem.states = out_states
        elem.mask = out_mask
        report["elements"][name] = {
            "before_states_shape": before_shape,
            "before_mask_shape": list(mask.shape),
            "after_states_shape": list(out_states.shape),
            "after_mask_shape": list(out_mask.shape),
            "source_rows_forced_active": int(
                existing
            ),
            "insertion_slot_index": int(
                existing
            ),
        }
    return normalized, report


def compact_edited_scene(scene: Any, edit_result: Any) -> Tuple[Any, Dict[str, Any]]:
    """Remove inactive insertion slots and remap final report indices.

    SLEDGE raw preprocessing currently ignores agent masks when selecting the
    nearest agents. Leaving padded zero rows would therefore create fake agents
    at the ego origin and hide the controlled pedestrian/occluder.
    """

    mappings: Dict[str, Dict[int, int]] = {}
    report: Dict[str, Any] = {"schema_version": "edited_scene_compaction_v1", "elements": {}}
    for name in DEFAULT_CAPACITIES:
        elem = getattr(scene, name)
        states = np.asarray(elem.states)
        mask = np.asarray(elem.mask).reshape(-1).astype(bool)
        valid_indices = np.where(mask)[0]
        mappings[name] = {int(old): int(new) for new, old in enumerate(valid_indices)}
        elem.states = states[valid_indices].copy()
        elem.mask = np.ones((len(valid_indices),), dtype=bool)
        report["elements"][name] = {
            "before_rows": int(len(states)),
            "after_rows": int(len(valid_indices)),
            "old_to_new": {str(k): v for k, v in mappings[name].items()},
        }

    edit_result.pedestrian_index = mappings["pedestrians"].get(int(edit_result.pedestrian_index), -1)
    occ_elem = str(getattr(edit_result, "occluder_elem_name", "vehicles"))
    edit_result.occluder_index = mappings.get(occ_elem, {}).get(int(edit_result.occluder_index), -1)
    edit_result.static_obstacle_index = mappings["static_objects"].get(
        int(edit_result.static_obstacle_index), -1
    )
    actor_elem = {
        "pedestrian": "pedestrians",
        "static_obstacle": "static_objects",
    }.get(str(edit_result.primary_actor_type), "vehicles")
    edit_result.primary_actor_index = mappings[actor_elem].get(int(edit_result.primary_actor_index), -1)
    report["final_indices"] = {
        "pedestrian_index": int(edit_result.pedestrian_index),
        "occluder_elem_name": occ_elem,
        "occluder_index": int(edit_result.occluder_index),
        "primary_actor_index": int(edit_result.primary_actor_index),
    }
    return scene, report
