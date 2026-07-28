"""Natural and protected diffusion channels for semantic-retention experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from sledge.script.run_half_denoise_from_tiered_cache import MultiScenarioHalfDenoiseRunner
from sledge.semantic_control.generation.legacy.evaluators.crossing_evaluator import (
    PromptAlignmentResult,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.refinement_alignment import (
    OccludedPedestrianRefinementAlignmentEvaluator,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.refinement_runner import (
    OccludedPedestrianHalfDenoiseRunner,
)


class SaveEveryGeneratedCandidateEvaluator:
    """Selection-only evaluator used to persist a natural diffusion sample.

    The output is intentionally *not* used as the semantic result. Natural
    samples are strictly re-evaluated after cache writing by the same
    stage-independent occluded-pedestrian evaluator used for B1 and protected
    B2. Returning a valid score here prevents the legacy runner from dropping
    semantically failed generations before they can be inspected.
    """

    def evaluate(self, sledge_vector: Any, prompt_spec: Any = None) -> PromptAlignmentResult:
        return PromptAlignmentResult(
            total=1.0,
            details={
                "pedestrian_presence_score": 1.0,
                "roadside_emergence_score": 1.0,
                "crossing_direction_score": 1.0,
                "ego_lane_conflict_score": 1.0,
                "immediacy_score": 1.0,
            },
            notes=["selection-only evaluator; use post-hoc strict semantic report"],
            accepted=True,
        )


def _common_args(
    *,
    run_root: Path,
    output_reports: Path,
    output_cache: Path,
    config: Path,
    autoencoder_checkpoint: Path,
    diffusion_checkpoint: Path,
    device: str,
    max_scenes: Optional[int],
    repair_attempts: int,
    seed: int,
    save_latents: bool,
    save_visuals: bool,
    overwrite: bool,
) -> Dict[str, Any]:
    return {
        "original_dir": str(Path(run_root) / "b0_original_cache"),
        "edited_dir": str(Path(run_root) / "b1_edited_cache"),
        "output": str(output_reports),
        "config": str(config),
        "autoencoder_checkpoint": str(autoencoder_checkpoint),
        "diffusion_checkpoint": str(diffusion_checkpoint),
        "scenario_cache_root": str(output_cache),
        "map_id": None,
        "glob_pattern": "**/sledge_raw.gz",
        "max_scenes": max_scenes,
        "skip_existing": not overwrite,
        "output_layout": "mirror",
        "num_inference_timesteps": 24,
        "guidance_scale": 4.0,
        "low_noise_start_step_seq": "20,21,22,23",
        "repair_attempts": repair_attempts,
        "seed": seed,
        "alignment_threshold": 0.70,
        "min_preservation_ratio": 0.95,
        "strict_save_only_passing": True,
        "diff_threshold": 1e-4,
        "diff_mask_dilation": 3,
        "roi_mask_dilation": 2,
        "pedestrian_roi_strength": 1.0,
        "vehicle_roi_strength": 1.0,
        "roadside_anchor_strength": 1.0,
        "lane_anchor_strength": 1.0,
        "crossing_corridor_strength": 0.95,
        "generic_roi_strength": 0.95,
        "device": device,
        "save_latents": save_latents,
        "save_visuals": save_visuals,
    }


def run_natural_diffusion(
    *,
    run_root: Path,
    config: Path,
    autoencoder_checkpoint: Path,
    diffusion_checkpoint: Path,
    device: str,
    max_scenes: Optional[int] = None,
    repair_attempts: int = 1,
    seed: int = 0,
    save_latents: bool = False,
    save_visuals: bool = False,
    overwrite: bool = False,
) -> Dict[str, str]:
    """Run low-noise img2img without semantic ROI or hard slot compositing."""

    run_root = Path(run_root).resolve()
    reports = run_root / "b2_natural_reports"
    cache = run_root / "b2_natural_cache"
    kwargs = _common_args(
        run_root=run_root,
        output_reports=reports,
        output_cache=cache,
        config=config,
        autoencoder_checkpoint=autoencoder_checkpoint,
        diffusion_checkpoint=diffusion_checkpoint,
        device=device,
        max_scenes=max_scenes,
        repair_attempts=max(1, int(repair_attempts)),
        seed=seed,
        save_latents=save_latents,
        save_visuals=save_visuals,
        overwrite=overwrite,
    )
    # Disable every preservation mechanism. A very high raster-difference
    # threshold produces an all-zero edit mask; zero ROI strengths produce an
    # all-zero ROI mask. The base runner therefore denoises the complete latent.
    kwargs.update(
        {
            "alignment_threshold": 0.0,
            "min_preservation_ratio": 0.0,
            "diff_threshold": 1e12,
            "diff_mask_dilation": 0,
            "roi_mask_dilation": 0,
            "pedestrian_roi_strength": 0.0,
            "vehicle_roi_strength": 0.0,
            "roadside_anchor_strength": 0.0,
            "lane_anchor_strength": 0.0,
            "crossing_corridor_strength": 0.0,
            "generic_roi_strength": 0.0,
        }
    )
    runner = MultiScenarioHalfDenoiseRunner(argparse.Namespace(**kwargs))
    runner.alignment_evaluator = SaveEveryGeneratedCandidateEvaluator()
    runner.run_batch()
    return {"reports": str(reports), "cache": str(cache)}


def run_protected_diffusion(
    *,
    run_root: Path,
    config: Path,
    autoencoder_checkpoint: Path,
    diffusion_checkpoint: Path,
    device: str,
    max_scenes: Optional[int] = None,
    repair_attempts: int = 6,
    seed: int = 0,
    save_latents: bool = False,
    save_visuals: bool = False,
    overwrite: bool = False,
) -> Dict[str, str]:
    """Run the existing ROI-preserved and hard-composited production channel."""

    run_root = Path(run_root).resolve()
    reports = run_root / "b2_protected_reports"
    cache = run_root / "b2_protected_cache"
    kwargs = _common_args(
        run_root=run_root,
        output_reports=reports,
        output_cache=cache,
        config=config,
        autoencoder_checkpoint=autoencoder_checkpoint,
        diffusion_checkpoint=diffusion_checkpoint,
        device=device,
        max_scenes=max_scenes,
        repair_attempts=max(1, int(repair_attempts)),
        seed=seed,
        save_latents=save_latents,
        save_visuals=save_visuals,
        overwrite=overwrite,
    )
    runner = OccludedPedestrianHalfDenoiseRunner(argparse.Namespace(**kwargs))
    runner.alignment_evaluator = OccludedPedestrianRefinementAlignmentEvaluator(projection_time_s=2.1)
    runner.run_batch()
    return {"reports": str(reports), "cache": str(cache)}
