"""Hierarchical language pipeline with explicit scene-construction routing."""

from __future__ import annotations

from typing import Optional

from sledge.semantic_control.language.hierarchical_pipeline import (
    HierarchicalEventFramePipeline,
    HierarchicalPipelineResult,
)
from sledge.semantic_control.language.scene_construction_router import (
    SceneConstructionRouter,
)


class RoutedHierarchicalEventFramePipeline(HierarchicalEventFramePipeline):
    """Add a provenance-aware B1 construction decision to the base pipeline.

    The parser, verifier, hierarchy resolver and parameter filler remain exactly
    the same as the existing hierarchical implementation. Routing is a final
    post-processing layer, which keeps language-understanding regressions
    isolated from construction policy changes.
    """

    def __init__(
        self,
        *args,
        scene_router: Optional[SceneConstructionRouter] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.scene_router = scene_router or SceneConstructionRouter()

    def parse_to_result(self, sentence: str) -> HierarchicalPipelineResult:
        result = super().parse_to_result(sentence)
        result.spec = self.scene_router.attach(result.spec, prompt=sentence)

        validation = result.spec.setdefault("validation_layer", {})
        construction = result.spec["scene_construction"]
        validation["scene_construction_mode"] = construction["mode"]
        validation["scene_construction_reason"] = construction["reason"]
        validation["scene_construction_routed"] = True
        validation["pipeline_design"] = (
            "eventframe_v7_dual_mode_provenance_routed_hierarchy"
        )
        return result


DefaultRoutedLanguageUnderstandingPipeline = RoutedHierarchicalEventFramePipeline
