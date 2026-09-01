"""Deterministic semantic scene construction and hierarchical adapters.

Keep this package initializer deliberately lightweight.  In particular, do
not eagerly import ``elastic_context_editor`` here: the evaluation package
imports generation submodules (for example ``generation.geometry_metrics``),
and an eager elastic-context import would in turn import evaluation metrics and
create a package-initialization cycle.

Modules that need elastic editing should import it directly from
``generation.elastic_context_editor``.  This is already how the main pipeline
imports it.

The topology-adaptive projector has a small roadside/static robustness override
installed here.  The historical base implementation remains in
``topology_adaptive_projection.py`` for ablation/reference, while all normal
imports of ``TopologyAdaptiveHazardProjector`` receive the robust subclass.
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

# Install the robust roadside/static solver transparently for direct imports
# from generation.topology_adaptive_projection.  Importing the submodule here
# is safe: it depends only on geometry/spec/object-type helpers, not evaluation
# metrics, so it does not recreate the elastic-context cycle documented above.
from . import topology_adaptive_projection as _topology_adaptive_projection
from .topology_adaptive_projection_robust import (
    RobustTopologyAdaptiveHazardProjector,
)

_topology_adaptive_projection.TopologyAdaptiveHazardProjector = (
    RobustTopologyAdaptiveHazardProjector
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
    "RobustTopologyAdaptiveHazardProjector",
]
