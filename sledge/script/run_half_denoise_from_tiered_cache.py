"""Compatibility entry point for half-denoise generation.

The historical runner is preserved in ``run_half_denoise_from_tiered_cache_legacy``.
This module keeps its public API but adds one narrowly scoped behavior: when a
semantic runner identifies itself as ``topology_adaptive`` and exposes its B1
road template, denoising receives topology-local latent lane guidance.

Raw diffusion and semantic-protected modes intentionally keep the legacy path
unchanged for clean ablations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from sledge.diffusion.modelling.unified_lane_guidance import (
    LaneConstraintReference,
    LaneGuidanceConfig,
    LatentLaneConstraintGuidance,
    infer_lane_constraint_reference,
)
from sledge.script.run_half_denoise_from_tiered_cache_legacy import *  # noqa: F401,F403
from sledge.script.run_half_denoise_from_tiered_cache_legacy import (
    MultiScenarioHalfDenoiseRunner as LegacyMultiScenarioHalfDenoiseRunner,
    build_argparser,
)


class MultiScenarioHalfDenoiseRunner(LegacyMultiScenarioHalfDenoiseRunner):
    """Legacy half-denoise runner plus topology-local latent road guidance."""

    def _lane_guidance_for_active_scene(
        self,
    ) -> Tuple[
        Optional[LatentLaneConstraintGuidance],
        Optional[LaneConstraintReference],
    ]:
        # Preserve clean baseline behavior.  Only the generated-road mode gets
        # road guidance; semantic_protected still hard-copies B1 road later.
        if str(getattr(self, "diffusion_mode", "")) != "topology_adaptive":
            return None, None

        template = getattr(self, "_active_template", None)
        if template is None or not hasattr(template, "lines"):
            return None, None

        template_id = id(template)
        cached_id = getattr(
            self,
            "_lane_guidance_reference_template_id",
            None,
        )
        reference = getattr(self, "_lane_guidance_reference", None)
        if cached_id != template_id or reference is None:
            reference = infer_lane_constraint_reference(
                template.lines.states,
                template.lines.mask,
            )
            self._lane_guidance_reference = reference
            self._lane_guidance_reference_template_id = template_id

        config_value = getattr(self.args, "lane_guidance", None)
        if isinstance(config_value, dict):
            config = LaneGuidanceConfig.from_mapping(config_value)
        else:
            config = LaneGuidanceConfig(
                enabled=bool(
                    getattr(
                        self.args,
                        "lane_guidance_enabled",
                        True,
                    )
                )
            )

        return (
            LatentLaneConstraintGuidance(
                reference=reference,
                config=config,
            ),
            reference,
        )

    def _attempt_repair(
        self,
        init_latents: torch.Tensor,
        preserve_mask: torch.Tensor,
        map_id: int,
        attempt_idx: int,
        scene_index: int,
    ):
        start_idx = self.start_step_candidates[
            attempt_idx % len(self.start_step_candidates)
        ]
        gen = torch.Generator(device=self.args.device)
        gen.manual_seed(
            int(self.args.seed) + scene_index * 1000 + attempt_idx
        )

        guidance_callback, reference = self._lane_guidance_for_active_scene()
        guidance_trace = []
        with torch.no_grad():
            denoised_vectors, final_latents = self.pipeline(
                class_labels=[map_id],
                num_inference_timesteps=self.args.num_inference_timesteps,
                guidance_scale=self.args.guidance_scale,
                num_classes=self.num_classes,
                init_latents=init_latents,
                start_timestep_index=start_idx,
                preserve_mask=preserve_mask,
                generator=gen,
                return_latents=True,
                guidance_callback=guidance_callback,
                guidance_trace=(
                    guidance_trace
                    if guidance_callback is not None
                    else None
                ),
            )

        vector = denoised_vectors[0].torch_to_numpy(apply_sigmoid=True)

        if not hasattr(self, "_lane_guidance_traces"):
            self._lane_guidance_traces: Dict[int, Dict[str, Any]] = {}
        applied = [
            row
            for row in guidance_trace
            if bool(row.get("lane_guidance_applied", False))
        ]
        self._lane_guidance_traces[int(attempt_idx)] = {
            "enabled": guidance_callback is not None,
            "reference": (
                reference.to_dict() if reference is not None else None
            ),
            "step_count": len(guidance_trace),
            "applied_step_count": len(applied),
            "mean_grad_rms": (
                float(
                    sum(
                        float(row.get("lane_guidance_grad_rms", 0.0))
                        for row in applied
                    )
                    / len(applied)
                )
                if applied
                else 0.0
            ),
            "max_grad_abs": (
                max(
                    float(row.get("lane_guidance_grad_abs_max", 0.0))
                    for row in applied
                )
                if applied
                else 0.0
            ),
            "mean_loss": (
                float(
                    sum(
                        float(row.get("lane_guidance_loss", 0.0))
                        for row in applied
                    )
                    / len(applied)
                )
                if applied
                else 0.0
            ),
            "trace": guidance_trace,
        }
        return vector, final_latents, start_idx

    def lane_guidance_diagnostics(
        self,
        attempt_idx: int,
    ) -> Dict[str, Any]:
        """Return diagnostics for one repair attempt without mutating state."""

        return dict(
            getattr(self, "_lane_guidance_traces", {}).get(
                int(attempt_idx),
                {},
            )
        )


def main() -> None:
    args = build_argparser().parse_args()
    runner = MultiScenarioHalfDenoiseRunner(args)
    runner.run_batch()


if __name__ == "__main__":
    main()
