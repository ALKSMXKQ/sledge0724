"""Topology-adaptive projection with lightweight generated-line cleanup.

The diffusion model itself is intentionally untouched.  This wrapper only
regularizes decoded road polylines before the existing road-validity gate and
hazard projection run.
"""

from __future__ import annotations

from typing import Any

from sledge.semantic_control.occluded_pedestrian_pipeline.generation.lane_polyline_regularizer import (
    LanePolylineRegularizerConfig,
    regularize_sledge_lines,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.topology_adaptive_projection_robust import (
    RobustTopologyAdaptiveHazardProjector,
)


class LaneRegularizedTopologyAdaptiveHazardProjector(
    RobustTopologyAdaptiveHazardProjector
):
    """Existing robust projector preceded by decoded-line regularization."""

    def __init__(
        self,
        *,
        projection_time_s: float = 2.1,
        lane_regularizer_config: LanePolylineRegularizerConfig | None = None,
    ) -> None:
        super().__init__(projection_time_s=projection_time_s)
        self.lane_regularizer_config = (
            lane_regularizer_config or LanePolylineRegularizerConfig()
        )

    def project(
        self,
        vector: Any,
        spec: Any,
        *,
        attempt_seed: int = 0,
    ):
        lane_report = regularize_sledge_lines(
            vector,
            self.lane_regularizer_config,
        )
        projected, report = super().project(
            vector,
            spec,
            attempt_seed=attempt_seed,
        )
        report = dict(report)
        report["lane_polyline_regularization"] = lane_report
        return projected, report


__all__ = ["LaneRegularizedTopologyAdaptiveHazardProjector"]
