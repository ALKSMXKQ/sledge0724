"""Backward-compatible editor imports.

The implementations now live under
``sledge.semantic_control.generation.legacy.editors``.
"""

from sledge.semantic_control.generation.legacy.editors import (
    CutInEditor,
    HardBrakeEditor,
    PedestrianCrossingEditor,
)

__all__ = ["PedestrianCrossingEditor", "CutInEditor", "HardBrakeEditor"]
