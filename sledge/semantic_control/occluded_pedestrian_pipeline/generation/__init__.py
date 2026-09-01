"""Deterministic semantic scene construction and hierarchical adapters.

Keep this package initializer deliberately lightweight.  In particular, do
not eagerly import ``elastic_context_editor`` here: the evaluation package
imports generation submodules (for example ``generation.geometry_metrics``),
and an eager elastic-context import would in turn import evaluation metrics and
create a package-initialization cycle.

Modules that need elastic editing should import it directly from
``generation.elastic_context_editor``.  This is already how the main pipeline
imports it.
"""

from .b0_scene_context import B0SceneContext, B0SceneContextExtractor
from .compositional_editor import CompositionalSemanticSceneEditor
from .diffusion_modes import (
    RAW_DIFFUSION_BASELINE,
    SEMANTIC_PROTECTED,
    SUPPORTED_DIFFUSION_MODES,
    TOPOLOGY_ADAPTIVE,
)
from .hazard_spec import HazardSemanticSpec
from .hierarchical_spec_adapter import HierarchicalHazardSpecAdapter
from .hierarchical_template_sampler import (
    ConcreteOccludedPedestrianParameters,
    HierarchicalTemplateSampler,
    SamplingOverrides,
)
from .template_scene_synthesizer import TemplateSceneSynthesizer

__all__ = [
    "B0SceneContext",
    "B0SceneContextExtractor",
    "CompositionalSemanticSceneEditor",
    "RAW_DIFFUSION_BASELINE",
    "SEMANTIC_PROTECTED",
    "TOPOLOGY_ADAPTIVE",
    "SUPPORTED_DIFFUSION_MODES",
    "HazardSemanticSpec",
    "HierarchicalHazardSpecAdapter",
    "ConcreteOccludedPedestrianParameters",
    "HierarchicalTemplateSampler",
    "SamplingOverrides",
    "TemplateSceneSynthesizer",
]
