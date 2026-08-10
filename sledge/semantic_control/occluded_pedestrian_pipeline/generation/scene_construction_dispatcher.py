"""Downstream dispatcher for routed B1 scene construction.

This module deliberately contains no language heuristics.  The language stage
must already have attached ``spec['scene_construction']['mode']``.  That keeps
batch generation from re-interpreting prompts or accidentally treating
``ego_lane`` as a global road request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from sledge.semantic_control.language.scene_construction_router import (
    SceneConstructionMode,
)


class SceneConstructionRoutingError(ValueError):
    """Raised when a downstream caller receives an unrouted/invalid spec."""


@dataclass(frozen=True)
class SceneConstructionResult:
    """Result of dispatching one routed B1 construction request."""

    scene: Any
    mode: str
    used_b0: bool
    reason: str


EditExistingFn = Callable[[Any, Mapping[str, Any]], Any]
SynthesizeNewFn = Callable[[Mapping[str, Any]], Any]


def scene_construction_mode(spec: Mapping[str, Any]) -> SceneConstructionMode:
    """Read the already-decided mode without falling back to prompt heuristics."""

    construction = spec.get("scene_construction", {})
    if not isinstance(construction, Mapping):
        raise SceneConstructionRoutingError("scene_construction must be a mapping")
    raw_mode = str(construction.get("mode", "")).strip().lower()
    try:
        return SceneConstructionMode(raw_mode)
    except ValueError as exc:
        raise SceneConstructionRoutingError(
            "spec is missing a valid scene_construction.mode; run the routed "
            "hierarchical language pipeline before B1 construction"
        ) from exc


def dispatch_scene_construction(
    spec: Mapping[str, Any],
    *,
    edit_existing: EditExistingFn,
    synthesize_new: SynthesizeNewFn,
    b0_scene: Optional[Any] = None,
) -> SceneConstructionResult:
    """Dispatch exactly once according to ``scene_construction.mode``.

    ``EDIT_EXISTING`` requires a B0 scene and receives both B0 and the complete
    routed spec. ``SYNTHESIZE_NEW`` receives only the spec so a synthesis backend
    cannot accidentally inherit B0 road geometry.
    """

    mode = scene_construction_mode(spec)
    construction = spec["scene_construction"]
    reason = str(construction.get("reason", ""))

    if mode == SceneConstructionMode.EDIT_EXISTING:
        if b0_scene is None:
            raise SceneConstructionRoutingError(
                "edit_existing requires b0_scene because road/map geometry must be inherited"
            )
        scene = edit_existing(b0_scene, spec)
        return SceneConstructionResult(
            scene=scene,
            mode=mode.value,
            used_b0=True,
            reason=reason,
        )

    scene = synthesize_new(spec)
    return SceneConstructionResult(
        scene=scene,
        mode=mode.value,
        used_b0=False,
        reason=reason,
    )
