from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeVectorElement,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.additive_integrity import (
    evaluate_exact_scene_preservation,
    evaluate_fixed_vector_capacity,
    evaluate_strict_additive_edit,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.refinement_runner import (
    OccludedPedestrianHalfDenoiseRunner,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.spec_presets import (
    normalize_spec,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.language.eventframe_adapter import (
    OccludedPedestrianEventFrameAdapter,
)


def _element(
    rows,
    *,
    width: int,
) -> SledgeVectorElement:
    states = np.asarray(
        rows,
        dtype=np.float32,
    )
    if states.size == 0:
        states = np.zeros(
            (0, width),
            dtype=np.float32,
        )
    return SledgeVectorElement(
        states=states,
        mask=np.ones(
            len(states),
            dtype=bool,
        ),
    )


def _scene():
    line_states = np.asarray(
        [
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    return SimpleNamespace(
        lines=SledgeVectorElement(
            states=line_states,
            mask=np.ones(
                (1, 2),
                dtype=bool,
            ),
        ),
        vehicles=_element(
            [
                [
                    20.0,
                    3.5,
                    0.0,
                    2.0,
                    4.8,
                    5.0,
                ]
            ],
            width=6,
        ),
        pedestrians=_element(
            [],
            width=6,
        ),
        static_objects=_element(
            [],
            width=5,
        ),
        green_lights=_element(
            [],
            width=3,
        ),
        red_lights=_element(
            [],
            width=3,
        ),
        ego=SledgeVectorElement(
            states=np.asarray(
                [6.0, 0.0],
                dtype=np.float32,
            ),
            mask=np.asarray(
                [True],
                dtype=bool,
            ),
        ),
    )


def _strict_additive_candidate():
    candidate = deepcopy(
        _scene()
    )
    candidate.pedestrians = _element(
        [
            [
                12.0,
                -4.0,
                np.pi / 2.0,
                0.75,
                0.75,
                1.6,
            ]
        ],
        width=6,
    )
    candidate.static_objects = (
        _element(
            [
                [
                    7.0,
                    -3.6,
                    np.pi / 2.0,
                    0.6,
                    2.0,
                ]
            ],
            width=5,
        )
    )
    return candidate


def test_strict_additive_edit_accepts_only_two_new_entities():
    report = evaluate_strict_additive_edit(
        _scene(),
        _strict_additive_candidate(),
        pedestrian_index=0,
        occluder_index=0,
        occluder_elem_name=(
            "static_objects"
        ),
    )
    assert report["overall_pass"] is True
    assert report["checks"][
        "immutable_scene_layers_exact"
    ] is True
    assert report["checks"][
        "all_original_entities_preserved"
    ] is True


def test_raw_source_rows_are_preserved_even_when_source_masks_are_false():
    original = _scene()
    original.vehicles.mask[:] = False
    candidate = _strict_additive_candidate()
    candidate.vehicles.mask[:] = True

    report = evaluate_strict_additive_edit(
        original,
        candidate,
        pedestrian_index=0,
        occluder_index=0,
        occluder_elem_name="static_objects",
        source_rows_are_entities=True,
    )

    assert report["overall_pass"] is True
    assert report["elements"]["vehicles"]["original_active_count"] == 1
    assert report["elements"]["vehicles"]["candidate_active_count"] == 1


def test_strict_additive_edit_rejects_ego_or_source_vehicle_changes():
    candidate = (
        _strict_additive_candidate()
    )
    candidate.ego.states[0] = 8.0
    candidate.vehicles.states[
        0,
        5,
    ] = 9.0
    report = evaluate_strict_additive_edit(
        _scene(),
        candidate,
        pedestrian_index=0,
        occluder_index=0,
        occluder_elem_name=(
            "static_objects"
        ),
    )
    assert report["overall_pass"] is False
    assert report["checks"][
        "immutable_scene_layers_exact"
    ] is False
    assert report["checks"][
        "all_original_entities_preserved"
    ] is False


def test_strict_additive_edit_rejects_unrequested_extra_entity():
    candidate = (
        _strict_additive_candidate()
    )
    candidate.vehicles = _element(
        [
            candidate
            .vehicles
            .states[0],
            [
                25.0,
                -4.0,
                0.0,
                2.0,
                4.8,
                0.0,
            ],
        ],
        width=6,
    )
    report = evaluate_strict_additive_edit(
        _scene(),
        candidate,
        pedestrian_index=0,
        occluder_index=0,
        occluder_elem_name=(
            "static_objects"
        ),
    )
    assert report["overall_pass"] is False
    assert report["checks"][
        "no_extra_entities"
    ] is False


def test_strict_b2_compositing_preserves_every_element():
    template = (
        _strict_additive_candidate()
    )
    decoded = deepcopy(
        template
    )
    decoded.lines.states += 5.0
    decoded.vehicles.states[
        0,
        0,
    ] += 10.0
    decoded.pedestrians.mask[
        0
    ] = False
    decoded.static_objects.states[
        0,
        0,
    ] += 2.0
    decoded.ego.states[
        0
    ] = 12.0

    (
        OccludedPedestrianHalfDenoiseRunner
        ._composite_protected_slots(
            decoded,
            template,
            {
                "pedestrian_index": 0,
                (
                    "occluder_"
                    "elem_name"
                ): "static_objects",
                "occluder_index": 0,
            },
        )
    )

    report = (
        evaluate_exact_scene_preservation(
            template,
            decoded,
        )
    )
    assert report["overall_pass"] is True


def test_fixed_vector_capacity_rejects_source_that_would_be_truncated():
    original = _scene()
    original.pedestrians = _element(
        [
            [
                float(index),
                -4.0,
                np.pi / 2.0,
                0.75,
                0.75,
                1.0,
            ]
            for index in range(
                20
            )
        ],
        width=6,
    )

    report = evaluate_fixed_vector_capacity(
        original,
        occluder_elem_name="vehicles",
    )

    assert report["overall_pass"] is False
    assert report["elements"]["pedestrians"]["required_capacity"] == 21
    assert report["elements"]["pedestrians"]["fixed_capacity"] == 20


def test_occlusion_spec_forces_conservative_additive_policy():
    spec = (
        OccludedPedestrianEventFrameAdapter(
            llm_provider="none"
        )
        .adapt(
            "A pedestrian hidden behind "
            "a barrier crosses right to "
            "left at 1.6 m/s."
        )
        .hazard_spec
    )
    spec.actor_layer.prefer_existing_actor = True
    spec.actor_layer.allow_actor_replacement = True
    spec.road_layer.allow_lane_generation = True
    spec.road_layer.generated_road_layout = (
        "roundabout_entry"
    )

    normalized = normalize_spec(
        spec
    )

    assert (
        normalized.actor_layer
        .prefer_existing_actor
        is False
    )
    assert (
        normalized.actor_layer
        .allow_actor_replacement
        is False
    )
    assert (
        normalized.road_layer
        .allow_lane_generation
        is False
    )
    assert (
        normalized.road_layer
        .generated_road_layout
        == "none"
    )
