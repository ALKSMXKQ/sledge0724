"""Language semantic-control modules.

This package contains the EventFrame-based natural-language scene understanding
pipeline used by the language evaluation scripts.

The imports are intentionally lightweight so importing this package does not
eagerly load old evaluator modules or optional experiment dependencies.
"""

from sledge.semantic_control.language.event_frame import (
    ActorFrame,
    CompletedParameter,
    DiagnosticsFrame,
    EgoEventFrame,
    EventFrame,
    EventSequenceStep,
    MainEventFrame,
    MissingInformationFrame,
    OcclusionFrame,
    RoadContextFrame,
    normalize_event_frame_dict,
)

from sledge.semantic_control.language.event_frame_parser import EventFrameParser
from sledge.semantic_control.language.event_sequence_builder import EventSequenceBuilder
from sledge.semantic_control.language.event_frame_mapper import (
    EventFrameToHazardSpecMapper,
    flatten_dict,
    validate_spec,
)
from sledge.semantic_control.language.event_frame_verifier import (
    EventFrameVerifier,
    VerificationResult,
)
from sledge.semantic_control.language.missing_info_filler import MissingInfoFiller
from sledge.semantic_control.language.narrative_semantics import (
    EventCandidate,
    HazardFocusResolver,
    NarrativeAnalysis,
    NarrativeDecomposer,
)
from sledge.semantic_control.language.direct_template_baseline import (
    DirectTemplateBaseline,
    DirectTemplateFrame,
    DirectTemplateMapper,
    DirectTemplateParser,
    validate_direct_template_spec,
)
from sledge.semantic_control.language.hierarchical_ontology import (
    HierarchicalScenePath,
    HierarchicalSceneResolver,
    HierarchyNode,
)
from sledge.semantic_control.language.hierarchical_pipeline import (
    HierarchicalEventFramePipeline,
    HierarchicalParameterFiller,
    HierarchicalPipelineResult,
    attach_hierarchy,
    validate_hierarchical_spec,
)
from sledge.semantic_control.language.scene_construction_router import (
    SceneConstructionDecision,
    SceneConstructionMode,
    SceneConstructionRouter,
    attach_scene_construction,
)
from sledge.semantic_control.language.routed_hierarchical_pipeline import (
    DefaultRoutedLanguageUnderstandingPipeline,
    RoutedHierarchicalEventFramePipeline,
)

# Package-level callers now receive a routed spec by default. The original
# HierarchicalEventFramePipeline remains available explicitly for low-level
# hierarchy-only tests and backwards-compatible debugging.
DefaultLanguageUnderstandingPipeline = RoutedHierarchicalEventFramePipeline


__all__ = [
    "ActorFrame",
    "CompletedParameter",
    "DiagnosticsFrame",
    "EgoEventFrame",
    "EventFrame",
    "EventSequenceStep",
    "MainEventFrame",
    "MissingInformationFrame",
    "OcclusionFrame",
    "RoadContextFrame",
    "normalize_event_frame_dict",
    "EventFrameParser",
    "EventSequenceBuilder",
    "EventFrameToHazardSpecMapper",
    "flatten_dict",
    "validate_spec",
    "EventFrameVerifier",
    "VerificationResult",
    "MissingInfoFiller",
    "NarrativeDecomposer",
    "HazardFocusResolver",
    "NarrativeAnalysis",
    "EventCandidate",
    "DirectTemplateFrame",
    "DirectTemplateParser",
    "DirectTemplateMapper",
    "DirectTemplateBaseline",
    "validate_direct_template_spec",
    "HierarchyNode",
    "HierarchicalScenePath",
    "HierarchicalSceneResolver",
    "HierarchicalParameterFiller",
    "HierarchicalPipelineResult",
    "HierarchicalEventFramePipeline",
    "DefaultLanguageUnderstandingPipeline",
    "attach_hierarchy",
    "validate_hierarchical_spec",
    "SceneConstructionMode",
    "SceneConstructionDecision",
    "SceneConstructionRouter",
    "attach_scene_construction",
    "RoutedHierarchicalEventFramePipeline",
    "DefaultRoutedLanguageUnderstandingPipeline",
]
