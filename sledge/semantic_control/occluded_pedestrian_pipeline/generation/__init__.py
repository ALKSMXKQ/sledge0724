"""Deterministic semantic scene construction and hierarchical adapters.

Keep this package initializer deliberately lightweight.  In particular, do
not eagerly import ``elastic_context_editor`` here: the evaluation package
imports generation submodules (for example ``generation.geometry_metrics``),
and an eager elastic-context import would in turn import evaluation metrics and
create a package-initialization cycle.

Topology-adaptive projection now uses the path-relative v2 implementation by
default.  The original global-x projector and the previous robust wrapper stay
available explicitly for ablation/reproducibility, but package initialization
no longer monkey-patches the public class after import.
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
from .topology_adaptive_projection import (
    LegacyTopologyAdaptiveHazardProjector,
    PathRelativeTopologyAdaptiveHazardProjector,
    TopologyAdaptiveHazardProjector,
)
from .topology_adaptive_projection_robust import (
    RobustTopologyAdaptiveHazardProjector,
)


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
    "TopologyAdaptiveHazardProjector",
    "PathRelativeTopologyAdaptiveHazardProjector",
    "LegacyTopologyAdaptiveHazardProjector",
    "RobustTopologyAdaptiveHazardProjector",
]
