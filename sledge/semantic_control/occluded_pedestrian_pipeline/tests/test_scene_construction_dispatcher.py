from __future__ import annotations

import pytest

from sledge.semantic_control.occluded_pedestrian_pipeline.generation.scene_construction_dispatcher import (
    SceneConstructionRoutingError,
    dispatch_scene_construction,
)


def test_edit_existing_uses_b0_only():
    calls = []

    def edit_existing(b0, spec):
        calls.append(("edit", b0, spec["scene_construction"]["mode"]))
        return {"scene": "edited"}

    def synthesize_new(spec):
        calls.append(("synthesize", spec))
        return {"scene": "new"}

    spec = {
        "scene_construction": {
            "mode": "edit_existing",
            "reason": "no_explicit_global_road_structure",
        }
    }
    result = dispatch_scene_construction(
        spec,
        b0_scene={"scene": "b0"},
        edit_existing=edit_existing,
        synthesize_new=synthesize_new,
    )

    assert result.scene == {"scene": "edited"}
    assert result.used_b0 is True
    assert [call[0] for call in calls] == ["edit"]


def test_synthesize_new_never_passes_b0_to_backend():
    calls = []

    def edit_existing(b0, spec):
        calls.append(("edit", b0, spec))
        return {"scene": "edited"}

    def synthesize_new(spec):
        calls.append(("synthesize", spec["scene_construction"]["mode"]))
        return {"scene": "new"}

    spec = {
        "scene_construction": {
            "mode": "synthesize_new",
            "reason": "explicit_global_road_structure",
        }
    }
    result = dispatch_scene_construction(
        spec,
        b0_scene={"scene": "must_not_be_used"},
        edit_existing=edit_existing,
        synthesize_new=synthesize_new,
    )

    assert result.scene == {"scene": "new"}
    assert result.used_b0 is False
    assert [call[0] for call in calls] == ["synthesize"]


def test_edit_existing_requires_b0():
    spec = {"scene_construction": {"mode": "edit_existing"}}

    with pytest.raises(SceneConstructionRoutingError, match="requires b0_scene"):
        dispatch_scene_construction(
            spec,
            edit_existing=lambda b0, routed_spec: b0,
            synthesize_new=lambda routed_spec: {},
        )


def test_unrouted_spec_is_rejected_instead_of_guessed():
    with pytest.raises(SceneConstructionRoutingError, match="routed hierarchical"):
        dispatch_scene_construction(
            {},
            b0_scene={},
            edit_existing=lambda b0, routed_spec: b0,
            synthesize_new=lambda routed_spec: {},
        )
