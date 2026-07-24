"""Single command-line entry point for the complete pipeline.

Run with:

    python -m sledge.semantic_control.occluded_pedestrian_pipeline.cli --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch

from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.simulation import (
    run_simulation,
    summarize_simulation_metrics,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.stage_comparison import (
    run_and_visualize_stage_simulations,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.pilot_export import (
    export_b1_simulation_cache,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.experiment_matrix import (
    ExperimentCase,
    build_experiment_cases,
    load_matrix_config,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.pipeline import (
    OccludedPedestrianPipeline,
    RunLayout,
    evaluate_b2_cache,
    run_half_denoise,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_MATRIX = PACKAGE_ROOT / "configs/experiment_matrix.json"
DEFAULT_INPUT_ROOT = WORKSPACE_ROOT / "exp/caches/autoencoder_cache"
DEFAULT_CONFIG = WORKSPACE_ROOT / "semantic_img2img_cfg.yaml"
DEFAULT_AE_CHECKPOINT = (
    WORKSPACE_ROOT
    / "exp/exp/training_rvae_model/training_rvae_model/2025.10.17.06.17.03/best_model/epoch45.ckpt"
)
DEFAULT_DIFFUSION_CHECKPOINT = (
    WORKSPACE_ROOT
    / "exp/exp/training_dit_model/training_dit_diffusion/2025.10.17.18.36.55/checkpoint"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Occluded-pedestrian EventFrame-to-simulation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    single = sub.add_parser("single", help="Run EventFrame and B1 editing for one source scene")
    single.add_argument("--input-raw", type=Path, required=True)
    single.add_argument("--output-root", type=Path, required=True)
    single.add_argument("--prompt", required=True)
    single.add_argument(
        "--occluder",
        choices=["vehicle", "bicycle", "generic_object", "traffic_cone", "barrier", "czone_sign"],
        default="vehicle",
    )
    single.add_argument("--direction", choices=["left_to_right", "right_to_left"], default="right_to_left")
    single.add_argument("--speed", type=float, default=1.6)
    single.add_argument("--risk", choices=["mild", "moderate", "aggressive"], default="moderate")
    single.add_argument("--no-b1-visuals", action="store_true")
    _add_language_args(single)

    batch = sub.add_parser("batch", help="Run B0/B1 debug20 or diversity matrix")
    _add_batch_args(batch)

    refine = sub.add_parser("refine", help="Run B2 half-denoise and strict post-validation")
    _add_refine_args(refine)

    evaluate = sub.add_parser("evaluate-b2", help="Re-evaluate an existing B2 cache")
    evaluate.add_argument("--run-root", type=Path, required=True)

    export_b1 = sub.add_parser("export-b1", help="Export accepted B1 scenes as typed simulation gzip caches")
    export_b1.add_argument("--run-root", type=Path, required=True)
    export_b1.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    export_b1.add_argument("--limit", type=int, default=None)
    export_b1.add_argument("--no-previews", action="store_true")

    simulate = sub.add_parser("simulate", help="Run closed-loop simulation on the accepted B2 cache")
    simulate.add_argument("--run-root", type=Path, required=True)
    simulate.add_argument("--planner", default="pdm_closed_planner")
    simulate.add_argument("--limit", type=int, default=100)
    simulate.add_argument("--dry-run", action="store_true")

    compare = sub.add_parser(
        "compare-stages",
        help="Convert, simulate and visualize B0/B1/B2 independently",
    )
    compare.add_argument("--run-root", type=Path, required=True)
    compare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    compare.add_argument("--planner", default="pdm_closed_planner")
    compare.add_argument("--limit", type=int, default=20)

    sim_summary = sub.add_parser("summarize-simulation", help="Summarize a nuPlan metric parquet")
    sim_summary.add_argument("--metrics-parquet", type=Path, required=True)
    sim_summary.add_argument("--output-json", type=Path, required=True)

    all_cmd = sub.add_parser("all", help="Run B0/B1, B2, and closed-loop simulation")
    _add_batch_args(all_cmd)
    _add_refine_args(all_cmd, include_run_root=False)
    all_cmd.add_argument("--planner", default="pdm_closed_planner")
    all_cmd.add_argument("--simulation-limit", type=int, default=100)
    all_cmd.add_argument("--skip-refine", action="store_true")
    all_cmd.add_argument("--skip-simulation", action="store_true")
    all_cmd.add_argument("--simulation-dry-run", action="store_true")
    return parser


def _add_language_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-provider", choices=["none", "ollama"], default="none")
    parser.add_argument("--llm-model", default="qwen2.5:7b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")


def _add_batch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", choices=["debug20", "paper18", "extended24", "pilot100"], default="debug20")
    parser.add_argument("--matrix-config", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--glob-pattern", default="**/sledge_raw.gz")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--no-b1-visuals", action="store_true")
    _add_language_args(parser)


def _add_refine_args(parser: argparse.ArgumentParser, include_run_root: bool = True) -> None:
    if include_run_root:
        parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--autoencoder-checkpoint", type=Path, default=DEFAULT_AE_CHECKPOINT)
    parser.add_argument("--diffusion-checkpoint", type=Path, default=DEFAULT_DIFFUSION_CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-refine-scenes", type=int, default=None)
    parser.add_argument("--repair-attempts", type=int, default=6)
    parser.add_argument("--save-visuals", action="store_true")


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "single":
        case = ExperimentCase(
            sample_id="single",
            condition_id=f"occ-{args.occluder}__dir-{args.direction}__speed-{args.speed:.1f}",
            input_raw=str(args.input_raw),
            source_relative_path=args.input_raw.name,
            source_scenario_type=args.input_raw.parent.parent.name,
            prompt=args.prompt,
            occluder_type=args.occluder,
            direction=args.direction,
            pedestrian_speed_mps=args.speed,
            risk_level=args.risk,
            replicate=0,
        )
        pipeline = _pipeline_from_args(args)
        print(json.dumps(pipeline.run_case(case), ensure_ascii=False, indent=2))
        return

    if args.command in {"batch", "all"}:
        config = load_matrix_config(args.matrix_config)
        cases = build_experiment_cases(
            input_root=args.input_root,
            profile=args.profile,
            matrix_config=config,
            glob_pattern=args.glob_pattern,
            max_cases=args.max_cases,
        )
        pipeline = _pipeline_from_args(args)
        print(json.dumps(pipeline.run_batch(cases), ensure_ascii=False, indent=2))
        if args.command == "batch":
            return
        if not args.skip_refine:
            print(json.dumps(_run_refine(args, args.output_root), ensure_ascii=False, indent=2))
        if not args.skip_simulation:
            result = run_and_visualize_stage_simulations(
                repo_root=REPO_ROOT,
                run_root=args.output_root,
                config=args.config,
                planner=args.planner,
                limit=args.simulation_limit,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "refine":
        print(json.dumps(_run_refine(args, args.run_root), ensure_ascii=False, indent=2))
        return

    if args.command == "evaluate-b2":
        print(json.dumps(evaluate_b2_cache(RunLayout(args.run_root.resolve())), ensure_ascii=False, indent=2))
        return

    if args.command == "export-b1":
        result = export_b1_simulation_cache(
            args.run_root,
            args.config,
            limit=args.limit,
            save_previews=not args.no_previews,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "simulate":
        layout = RunLayout(args.run_root.resolve())
        result = run_simulation(
            repo_root=REPO_ROOT,
            scenario_cache=layout.b2_cache,
            output_manifest=layout.manifests / "simulation_launch.json",
            planner=args.planner,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "compare-stages":
        result = run_and_visualize_stage_simulations(
            repo_root=REPO_ROOT,
            run_root=args.run_root,
            config=args.config,
            planner=args.planner,
            limit=args.limit,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "summarize-simulation":
        result = summarize_simulation_metrics(args.metrics_parquet, args.output_json)
        print(json.dumps(result, ensure_ascii=False, indent=2))


def _pipeline_from_args(args) -> OccludedPedestrianPipeline:
    return OccludedPedestrianPipeline(
        args.output_root,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        ollama_url=args.ollama_url,
        strict_check=True,
        save_visuals=not args.no_b1_visuals,
    )


def _run_refine(args, run_root: Path):
    return run_half_denoise(
        run_root=run_root,
        config=args.config,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        diffusion_checkpoint=args.diffusion_checkpoint,
        device=args.device,
        max_scenes=args.max_refine_scenes,
        repair_attempts=args.repair_attempts,
        save_visuals=args.save_visuals,
    )


if __name__ == "__main__":
    main()
