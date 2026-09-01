"""Run topology-adaptive occluded-pedestrian refinement from an existing B1 run.

This small entry point keeps the historical pipeline CLI backward-compatible
while exposing the new third experimental mode immediately.  In addition to
the historical full-matrix B2 evaluation, it reports run-scoped smoke metrics
so ``--max-refine-scenes`` uses the number of actually attempted scenes as its
denominator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

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


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _safe_rate(numerator: int, denominator: int) -> float:
    return (
        float(numerator / denominator)
        if denominator > 0
        else 0.0
    )


def _build_run_scoped_summary(
    run_root: Path,
    full_matrix_evaluation: Dict[str, Any],
) -> Dict[str, Any]:
    """Separate smoke-run acceptance from full experiment coverage.

    ``evaluate_b2_cache`` intentionally evaluates against the complete
    ``cases.jsonl`` matrix for historical experiments.  During a smoke run,
    however, ``--max-refine-scenes`` may attempt only a subset.  This helper
    reads the runner's batch summary and reports both meanings explicitly.
    """

    run_root = Path(run_root).resolve()
    report_root = (
        run_root
        / "b2_diffusion"
        / TOPOLOGY_ADAPTIVE
        / "reports"
    )
    batch_path = report_root / "batch_summary.json"

    batch: Dict[str, Any] = {}
    if batch_path.exists():
        batch = _load_json(batch_path)

    num_language_cases = int(
        full_matrix_evaluation.get("num_expected", 0)
    )
    num_b1_available = sum(
        1
        for _ in (
            run_root
            / "b1_edited_cache"
        ).glob("**/sledge_raw.gz")
    )
    num_attempted = int(batch.get("total_seen", 0))
    num_finished = int(batch.get("finished", 0))
    num_accepted = int(batch.get("repair_success", 0))
    num_hard_errors = max(0, num_attempted - num_finished)
    num_rejected = max(0, num_finished - num_accepted)

    run_scope = {
        "schema_version": (
            "topology_adaptive_run_scope_v1"
        ),
        "num_language_cases": num_language_cases,
        "num_b1_available": num_b1_available,
        "num_attempted": num_attempted,
        "num_finished": num_finished,
        "num_accepted": num_accepted,
        "num_rejected": num_rejected,
        "num_hard_errors": num_hard_errors,
        "b1_availability_rate": _safe_rate(
            num_b1_available,
            num_language_cases,
        ),
        "generation_completion_rate_among_attempted": (
            _safe_rate(num_finished, num_attempted)
        ),
        "acceptance_rate_among_attempted": _safe_rate(
            num_accepted,
            num_attempted,
        ),
        "acceptance_rate_among_finished": _safe_rate(
            num_accepted,
            num_finished,
        ),
        "end_to_end_coverage_vs_language_matrix": _safe_rate(
            num_accepted,
            num_language_cases,
        ),
        "batch_summary_path": str(batch_path),
    }

    payload = {
        "schema_version": (
            "occluded_pedestrian_topology_adaptive_summary_v3"
        ),
        "diffusion_mode": TOPOLOGY_ADAPTIVE,
        "run_scope": run_scope,
        "full_matrix_evaluation": full_matrix_evaluation,
    }

    manifest_path = (
        run_root
        / "manifests"
        / "b2_topology_adaptive_run_scoped_summary.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
        )
    return payload


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
    payload = _build_run_scoped_summary(
        args.run_root,
        result,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
