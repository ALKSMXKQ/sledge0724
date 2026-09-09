"""Compatibility entry point for occluded-pedestrian refinement.

The full historical implementation is preserved in ``refinement_runner_legacy``.
This wrapper keeps its behavior and adds diffusion lane-guidance diagnostics to
the already persisted topology-adaptive projection report.
"""

from __future__ import annotations

from typing import Any, Dict

from sledge.semantic_control.occluded_pedestrian_pipeline.generation.refinement_runner_legacy import *  # noqa: F401,F403
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.refinement_runner_legacy import (
    OccludedPedestrianHalfDenoiseRunner as LegacyOccludedPedestrianHalfDenoiseRunner,
)


class OccludedPedestrianHalfDenoiseRunner(
    LegacyOccludedPedestrianHalfDenoiseRunner
):
    """Historical semantic runner plus persisted lane-guidance diagnostics."""

    def _attempt_repair(self, *args, **kwargs):
        vector, final_latents, start_idx = super()._attempt_repair(
            *args,
            **kwargs,
        )

        if not self.topology_adaptive_enabled:
            return vector, final_latents, start_idx

        attempt_idx = kwargs.get("attempt_idx")
        if attempt_idx is None and len(args) >= 4:
            attempt_idx = args[3]
        attempt_idx = int(attempt_idx or 0)

        diagnostics: Dict[str, Any] = {}
        if hasattr(self, "lane_guidance_diagnostics"):
            diagnostics = self.lane_guidance_diagnostics(attempt_idx)

        projection = dict(
            self._adaptive_projection_reports.get(attempt_idx, {}) or {}
        )
        if projection:
            projection["lane_guidance"] = diagnostics
            projection["lane_guidance_policy"] = (
                "local_corridor_confidence_plus_endpoint_connectivity"
            )
            projection["lane_guidance_enabled"] = bool(
                diagnostics.get("enabled", False)
            )
            projection["lane_guidance_applied_step_count"] = int(
                diagnostics.get("applied_step_count", 0) or 0
            )
            projection["lane_guidance_mean_grad_rms"] = float(
                diagnostics.get("mean_grad_rms", 0.0) or 0.0
            )
            projection["lane_guidance_max_grad_abs"] = float(
                diagnostics.get("max_grad_abs", 0.0) or 0.0
            )
            projection["lane_guidance_mean_loss"] = float(
                diagnostics.get("mean_loss", 0.0) or 0.0
            )
            self._adaptive_projection_reports[attempt_idx] = projection

        return vector, final_latents, start_idx


__all__ = ["OccludedPedestrianHalfDenoiseRunner"]
