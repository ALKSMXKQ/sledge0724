"""Compatibility entry point for topology-adaptive hazard projection.

The historical global-x implementation is preserved verbatim in
``topology_adaptive_projection_legacy``.  The public class exported from this
module is now the path-relative v2 projector, so existing runner imports keep
working without a large wiring-only rewrite.

Import order is intentional: the legacy symbols are populated first because
the v2 module reuses constants and helper methods from this compatibility
module while it is being imported.
"""

from __future__ import annotations

from sledge.semantic_control.occluded_pedestrian_pipeline.generation.topology_adaptive_projection_legacy import *  # noqa: F401,F403
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.topology_adaptive_projection_legacy import (
    MAX_CONFLICT_X_M,
    MIN_CONFLICT_X_M,
    LocalRoadContext,
    TopologyAdaptiveHazardProjector as LegacyTopologyAdaptiveHazardProjector,
    _lateral_half_extent,
    _point_line_distance,
    _rough_overlap,
)

# Keep the legacy class bound while importing v2 so its circular compatibility
# import sees a complete base class rather than a partially initialized module.
TopologyAdaptiveHazardProjector = LegacyTopologyAdaptiveHazardProjector

from sledge.semantic_control.occluded_pedestrian_pipeline.generation.topology_adaptive_projection_path import (  # noqa: E402
    PathRelativeTopologyAdaptiveHazardProjector,
)

TopologyAdaptiveHazardProjector = PathRelativeTopologyAdaptiveHazardProjector

__all__ = [
    "LegacyTopologyAdaptiveHazardProjector",
    "LocalRoadContext",
    "PathRelativeTopologyAdaptiveHazardProjector",
    "TopologyAdaptiveHazardProjector",
]
