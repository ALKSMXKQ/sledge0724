from __future__ import annotations

from typing import Any, Protocol

from ..core.types import SceneState


class SceneAdapter(Protocol):
    """Required interface for every legacy SLEDGE representation adapter."""

    def to_canonical(self, legacy_scene: Any, *, scene_id: str) -> SceneState:
        ...

    def from_canonical(self, scene: SceneState, *, template: Any) -> Any:
        ...
