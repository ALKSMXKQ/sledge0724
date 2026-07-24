"""Compatibility imports for the legacy scene-generation specification.

New code should import from ``sledge.semantic_control.generation.legacy``.
"""

from sledge.semantic_control.generation.legacy.prompt_spec import (
    PromptSpec,
    SceneEditResult,
    SceneEditROI,
)

__all__ = ["PromptSpec", "SceneEditResult", "SceneEditROI"]
