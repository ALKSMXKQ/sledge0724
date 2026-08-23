"""Hard integrity checks for conservative occluded-pedestrian editing.

The focused pipeline is additive by design:

* B0 is immutable.
* B1 may add exactly one pedestrian and exactly one occluder.
* Diffusion input must preserve every processed B0 element.
* Accepted B2 output must exactly preserve the diffusion-input vector.

These checks deliberately reject a source scene instead of silently deleting,
replacing, slowing, or moving an existing entity.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


AGENT_ELEMENTS: Tuple[str, ...] = (
    "vehicles",
    "pedestrians",
    "static_objects",
)

IMMUTABLE_ELEMENTS: Tuple[str, ...] = (
    "lines",
    "green_lights",
    "red_lights",
    "ego",
)

ALL_ELEMENTS: Tuple[str, ...] = (
    *IMMUTABLE_ELEMENTS,
    *AGENT_ELEMENTS,
)

FIXED_VECTOR_CAPACITIES = {
    "vehicles": 50,
    "pedestrians": 20,
    "static_objects": 30,
}


def evaluate_fixed_vector_capacity(
    original: Any,
    *,
    occluder_elem_name: str,
) -> Dict[str, Any]:
    """Reject raw sources whose additions cannot fit the RVAE vector."""

    expected_additions = {
        "vehicles": int(
            occluder_elem_name
            == "vehicles"
        ),
        "pedestrians": 1,
        "static_objects": int(
            occluder_elem_name
            == "static_objects"
        ),
    }
    elements = {}
    for name, capacity in (
        FIXED_VECTOR_CAPACITIES.items()
    ):
        states = np.asarray(
            getattr(
                original,
                name,
            ).states
        )
        source_count = (
            int(
                states.shape[0]
            )
            if states.ndim >= 2
            else int(
                states.size > 0
            )
        )
        required = (
            source_count
            + expected_additions[name]
        )
        elements[name] = {
            "source_count": source_count,
            "expected_additions": (
                expected_additions[name]
            ),
            "required_capacity": required,
            "fixed_capacity": int(
                capacity
            ),
            "passed": bool(
                required
                <= capacity
            ),
        }

    return {
        "schema_version": (
            "fixed_vector_capacity_v1"
        ),
        "overall_pass": all(
            row["passed"]
            for row in elements.values()
        ),
        "policy": (
            "No B0 entity may be "
            "truncated when adding the "
            "controlled pedestrian and "
            "occluder"
        ),
        "occluder_element": (
            occluder_elem_name
        ),
        "elements": elements,
    }


def evaluate_strict_additive_edit(
    original: Any,
    candidate: Any,
    *,
    pedestrian_index: int,
    occluder_index: int,
    occluder_elem_name: str,
    atol: float = 1e-6,
    source_rows_are_entities: bool = False,
) -> Dict[str, Any]:
    """Verify that candidate is B0 plus one pedestrian and one occluder."""

    if occluder_elem_name not in {"vehicles", "static_objects"}:
        return {
            "schema_version": "strict_additive_integrity_v1",
            "overall_pass": False,
            "policy": "B0 + exactly one pedestrian + exactly one occluder",
            "checks": {
                "supported_occluder_element": False,
            },
            "elements": {},
            "error": f"unsupported occluder element: {occluder_elem_name}",
        }

    immutable_reports = {
        name: _evaluate_exact_element(
            getattr(original, name),
            getattr(candidate, name),
            atol=atol,
        )
        for name in IMMUTABLE_ELEMENTS
    }

    expected_additions = {
        "vehicles": int(occluder_elem_name == "vehicles"),
        "pedestrians": 1,
        "static_objects": int(occluder_elem_name == "static_objects"),
    }

    agent_reports: Dict[str, Dict[str, Any]] = {}
    unmatched_by_element: Dict[str, Sequence[int]] = {}

    for name in AGENT_ELEMENTS:
        report = _evaluate_original_rows_preserved(
            getattr(original, name),
            getattr(candidate, name),
            expected_additions=expected_additions[name],
            atol=atol,
            source_rows_are_entities=(
                source_rows_are_entities
            ),
        )
        agent_reports[name] = report
        unmatched_by_element[name] = report["unmatched_candidate_indices"]

    pedestrian_is_new = (
        int(pedestrian_index)
        in unmatched_by_element["pedestrians"]
    )
    occluder_is_new = (
        int(occluder_index)
        in unmatched_by_element[occluder_elem_name]
    )

    checks = {
        "immutable_scene_layers_exact": all(
            report["passed"]
            for report in immutable_reports.values()
        ),
        "all_original_entities_preserved": all(
            report["all_original_rows_preserved"]
            for report in agent_reports.values()
        ),
        "exact_added_entity_counts": all(
            report["count_matches"]
            for report in agent_reports.values()
        ),
        "pedestrian_is_new": bool(pedestrian_is_new),
        "occluder_is_new": bool(occluder_is_new),
        "no_extra_entities": all(
            len(report["unmatched_candidate_indices"])
            == expected_additions[name]
            for name, report in agent_reports.items()
        ),
    }

    return {
        "schema_version": "strict_additive_integrity_v1",
        "overall_pass": bool(all(checks.values())),
        "policy": "B0 + exactly one pedestrian + exactly one occluder",
        "atol": float(atol),
        "source_rows_are_entities": bool(
            source_rows_are_entities
        ),
        "checks": checks,
        "controlled_indices": {
            "pedestrian_index": int(pedestrian_index),
            "occluder_element": occluder_elem_name,
            "occluder_index": int(occluder_index),
        },
        "expected_additions": expected_additions,
        "elements": {
            **immutable_reports,
            **agent_reports,
        },
    }


def evaluate_exact_scene_preservation(
    reference: Any,
    candidate: Any,
    *,
    atol: float = 1e-6,
) -> Dict[str, Any]:
    """Verify exact semantic-vector preservation across all scene elements."""

    elements = {
        name: _evaluate_exact_element(
            getattr(reference, name),
            getattr(candidate, name),
            atol=atol,
        )
        for name in ALL_ELEMENTS
    }

    checks = {
        "all_element_shapes_match": all(
            report["states_shape_match"]
            and report["mask_shape_match"]
            for report in elements.values()
        ),
        "all_element_states_match": all(
            report["states_match"]
            for report in elements.values()
        ),
        "all_element_masks_match": all(
            report["mask_match"]
            for report in elements.values()
        ),
    }

    return {
        "schema_version": "exact_scene_preservation_v1",
        "overall_pass": bool(all(checks.values())),
        "atol": float(atol),
        "checks": checks,
        "elements": elements,
    }


def _evaluate_exact_element(
    reference_elem: Any,
    candidate_elem: Any,
    *,
    atol: float,
) -> Dict[str, Any]:
    reference_states = np.asarray(reference_elem.states)
    candidate_states = np.asarray(candidate_elem.states)
    reference_mask = np.asarray(reference_elem.mask)
    candidate_mask = np.asarray(candidate_elem.mask)

    states_shape_match = (
        reference_states.shape
        == candidate_states.shape
    )
    mask_shape_match = (
        reference_mask.shape
        == candidate_mask.shape
    )
    states_match = bool(
        states_shape_match
        and np.allclose(
            reference_states,
            candidate_states,
            rtol=0.0,
            atol=atol,
            equal_nan=True,
        )
    )
    mask_match = bool(
        mask_shape_match
        and np.array_equal(
            reference_mask,
            candidate_mask,
        )
    )

    max_abs_state_error = None
    if states_shape_match and reference_states.size:
        delta = np.abs(
            reference_states.astype(
                np.float64,
            )
            - candidate_states.astype(
                np.float64,
            )
        )
        finite = delta[
            np.isfinite(delta)
        ]
        max_abs_state_error = (
            float(finite.max())
            if finite.size
            else 0.0
        )

    return {
        "passed": bool(
            states_match
            and mask_match
        ),
        "states_shape_match": bool(
            states_shape_match
        ),
        "mask_shape_match": bool(
            mask_shape_match
        ),
        "states_match": states_match,
        "mask_match": mask_match,
        "reference_states_shape": list(
            reference_states.shape
        ),
        "candidate_states_shape": list(
            candidate_states.shape
        ),
        "max_abs_state_error": (
            max_abs_state_error
        ),
    }


def _evaluate_original_rows_preserved(
    original_elem: Any,
    candidate_elem: Any,
    *,
    expected_additions: int,
    atol: float,
    source_rows_are_entities: bool,
) -> Dict[str, Any]:
    original_rows, original_indices = (
        _active_rows(
            original_elem,
            force_all=(
                source_rows_are_entities
            ),
        )
    )
    candidate_rows, candidate_indices = (
        _active_rows(
            candidate_elem,
        )
    )

    matched_candidate_positions: List[
        int
    ] = []
    missing_original_indices: List[
        int
    ] = []

    available = list(
        range(
            len(candidate_rows)
        )
    )

    for original_position, row in enumerate(
        original_rows
    ):
        match = next(
            (
                candidate_position
                for candidate_position
                in available
                if _rows_equal(
                    row,
                    candidate_rows[
                        candidate_position
                    ],
                    atol=atol,
                )
            ),
            None,
        )

        if match is None:
            missing_original_indices.append(
                int(
                    original_indices[
                        original_position
                    ]
                )
            )
            continue

        matched_candidate_positions.append(
            int(match)
        )
        available.remove(
            match
        )

    unmatched_candidate_indices = [
        int(
            candidate_indices[
                position
            ]
        )
        for position in available
    ]
    expected_count = (
        len(original_rows)
        + int(expected_additions)
    )

    return {
        "passed": bool(
            not missing_original_indices
            and len(candidate_rows)
            == expected_count
            and len(
                unmatched_candidate_indices
            )
            == int(expected_additions)
        ),
        "all_original_rows_preserved": (
            not missing_original_indices
        ),
        "count_matches": (
            len(candidate_rows)
            == expected_count
        ),
        "original_active_count": int(
            len(original_rows)
        ),
        "candidate_active_count": int(
            len(candidate_rows)
        ),
        "expected_additions": int(
            expected_additions
        ),
        "missing_original_indices": (
            missing_original_indices
        ),
        "unmatched_candidate_indices": (
            unmatched_candidate_indices
        ),
    }


def _active_rows(
    elem: Any,
    *,
    force_all: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    states = np.asarray(
        elem.states
    )
    masks = np.asarray(
        elem.mask
    ).reshape(-1)

    if states.ndim == 1:
        states = states.reshape(
            (1, -1)
        )
    elif states.ndim > 2:
        states = states.reshape(
            (
                states.shape[0],
                -1,
            )
        )

    usable = min(
        len(states),
        len(masks),
    )
    states = states[:usable]
    masks = masks[:usable]
    active = (
        np.ones(
            usable,
            dtype=bool,
        )
        if force_all
        else (
            masks.astype(
                np.float64
            )
            >= 0.3
        )
    )
    indices = np.where(
        active
    )[0]
    return (
        states[indices],
        indices,
    )


def _rows_equal(
    left: np.ndarray,
    right: np.ndarray,
    *,
    atol: float,
) -> bool:
    return bool(
        left.shape == right.shape
        and np.allclose(
            left,
            right,
            rtol=0.0,
            atol=atol,
            equal_nan=True,
        )
    )
