"""Run topology-adaptive occluded-pedestrian refinement from an existing B1 run.

This small entry point keeps the historical pipeline CLI backward-compatible
while exposing the new third experimental mode immediately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import torch

from sledge.semantic_control.occluded_pedestrian_pipeline.cli import (
    DEFAULT_AE_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_DIFFUSION_CHECKPOINT,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.diffusion_modes import (
    TOPOLOGY_ADAPTIVE,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.pipeline import (
    run_half_denoise,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run topology-adaptive hazard semantic projection on an existing "
            "occluded-pedestrian B1 run."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        default=DEFAULT_AE_CHECKPOINT,
    )
    parser.add_argument(
        "--diffusion-checkpoint",
        type=Path,
        default=DEFAULT_DIFFUSION_CHECKPOINT,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--max-refine-scenes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=6,
    )
    parser.add_argument("--save-visuals", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_half_denoise(
        run_root=args.run_root,
        config=args.config,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        diffusion_checkpoint=args.diffusion_checkpoint,
        device=args.device,
        diffusion_mode=TOPOLOGY_ADAPTIVE,
        max_scenes=args.max_refine_scenes,
        repair_attempts=args.repair_attempts,
        save_visuals=args.save_visuals,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
