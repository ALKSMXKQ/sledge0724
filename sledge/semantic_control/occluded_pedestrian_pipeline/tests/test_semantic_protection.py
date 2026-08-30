from __future__ import annotations

from copy import deepcopy

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeVector,
    SledgeVectorElement,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.semantic_protection import (
    audit_protected_semantics,
    composite_protected_semantics,
    protected_slots,
)


def _element(rows, width: int) -> SledgeVectorElement:
    states = np.asarray(rows, dtype=np.float32).reshape(-1, width)
    return SledgeVectorElement(
        states=states,
        mask=np.ones(len(states), dtype=bool),
    )


def _scene() -> SledgeVector:
    return SledgeVector(
        lines=SledgeVectorElement(
            states=np.asarray(
                [[[0.0, 0.0], [10.0, 0.0]]],
                dtype=np.float32,
            ),
            mask=np.asarray([True]),
        ),
        vehicles=_element(
            [
                [8.0, -3.0, 0.0, 2.0, 4.8, 0.0],
                [30.0, 2.0, 0.0, 2.0, 4.8, 4.0],
            ],
            6,
        ),
        pedestrians=_element(
            [[9.0, -4.0, 1.57, 0.7, 0.7, 1.6]],
            6,
        ),
        static_objects=_element([], 5),
        green_lights=SledgeVectorElement(
            states=np.zeros((0, 2, 2), dtype=np.float32),
            mask=np.zeros(0, dtype=bool),
        ),
        red_lights=SledgeVectorElement(
            states=np.zeros((0, 2, 2), dtype=np.float32),
            mask=np.zeros(0, dtype=bool),
        ),
        ego=SledgeVectorElement(
            states=np.asarray([6.0], dtype=np.float32),
            mask=np.asarray([True]),
        ),
    )


def test_hard_protection_restores_hazard_and_keeps_background_generation() -> None:
    template = _scene()
    candidate = deepcopy(template)
    candidate.lines.states += 5.0
    candidate.ego.states[:] = 12.0
    candidate.pedestrians.states[0, 0] += 3.0
    candidate.vehicles.states[0, 1] += 2.0
    candidate.vehicles.states[1, 0] += 7.0
    report = {
        "pedestrian_index": 0,
        "occluder_elem_name": "vehicles",
        "occluder_index": 0,
    }

    assert not audit_protected_semantics(
        candidate,
        template,
        report,
    )["overall_pass"]
    composite_protected_semantics(candidate, template, report)
    audit = audit_protected_semantics(candidate, template, report)

    assert audit["overall_pass"] is True
    np.testing.assert_array_equal(
        candidate.vehicles.states[0],
        template.vehicles.states[0],
    )
    assert candidate.vehicles.states[1, 0] == 37.0
    assert protected_slots(report)["ego"] == "complete_state"
