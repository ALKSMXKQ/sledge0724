"""Natural-language understanding and scene-generation controls.

The package has two explicit boundaries:

- :mod:`sledge.semantic_control.language` contains the EventFrame research path.
- :mod:`sledge.semantic_control.generation` contains executable scene editing.

Legacy public names are resolved lazily so existing scripts keep working
without importing optional simulation dependencies during package discovery.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "NaturalLanguagePromptParser",
    "PromptAlignmentEvaluator",
    "SemanticSceneEditor",
]

_LEGACY_EXPORTS = {
    "NaturalLanguagePromptParser": ("sledge.semantic_control.prompt_parser", "NaturalLanguagePromptParser"),
    "PromptAlignmentEvaluator": ("sledge.semantic_control.prompt_alignment", "PromptAlignmentEvaluator"),
    "SemanticSceneEditor": ("sledge.semantic_control.vector_editor", "SemanticSceneEditor"),
}


def __getattr__(name: str) -> Any:
    if name not in _LEGACY_EXPORTS:
        raise AttributeError(name)
    module_name, attribute_name = _LEGACY_EXPORTS[name]
    return getattr(import_module(module_name), attribute_name)
