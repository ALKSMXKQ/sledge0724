
"""Command-line entry for the hierarchical occluded-pedestrian chain."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from sledge.semantic_control.language.hierarchical_pipeline import (
    HierarchicalEventFramePipeline,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.pilot_export import (
    export_b1_simulation_cache,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.gzip_manifest import (
    build_generation_gzip_manifest,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.simulation import (
    run_simulation,
    summarize_simulation_metrics,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.stage_comparison import (
    run_and_visualize_stage_simulations,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.experiment_matrix import (
    ExperimentCase,
    build_experiment_cases,
    build_prompt_matrix_cases,
    load_matrix_config,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.diffusion_modes import (
    RAW_DIFFUSION_BASELINE,
    SEMANTIC_PROTECTED,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.rvae_reconstruction import (
    run_rvae_reconstruction,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.language.hierarchical_template_validator import (
    HierarchicalOccludedTemplateValidator,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.language.occluded_prompt_matrix import (
    default_prompt_cases,
    read_prompt_cases,
    write_prompt_cases,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.language.scene_construction_router import (
    EDIT_EXISTING,
    SYNTHESIZE_NEW,
    SceneConstructionRouter,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.pipeline import (
    OccludedPedestrianPipeline,
    RunLayout,
    evaluate_b2_cache,
    run_diffusion_comparison,
    run_half_denoise,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_MATRIX = PACKAGE_ROOT / "configs/experiment_matrix.json"
DEFAULT_PROMPT_MATRIX = (
    PACKAGE_ROOT / "configs/occluded_language_cases.jsonl"
)
DEFAULT_INPUT_ROOT = (
    WORKSPACE_ROOT / "exp/caches/autoencoder_cache"
)
DEFAULT_CONFIG = WORKSPACE_ROOT / "semantic_img2img_cfg.yaml"
DEFAULT_AE_CHECKPOINT = (
    WORKSPACE_ROOT
    / "exp/exp/training_rvae_model/training_rvae_model/"
    "2025.10.17.06.17.03/best_model/epoch45.ckpt"
)
DEFAULT_DIFFUSION_CHECKPOINT = (
    WORKSPACE_ROOT
    / "exp/exp/training_dit_model/training_dit_diffusion/"
    "2025.10.17.18.36.55/checkpoint"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hierarchical occluded-pedestrian language-to-diffusion pipeline"
        )
    )
    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    generate_prompts = sub.add_parser(
        "generate-prompts",
        help="Write the 12-case diagnostic prompt matrix",
    )
    generate_prompts.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PROMPT_MATRIX,
    )

    validate_prompts = sub.add_parser(
        "validate-prompts",
        help="Generate and validate hierarchical templates for JSONL prompts",
    )
    validate_prompts.add_argument(
        "--input-jsonl",
        type=Path,
        default=DEFAULT_PROMPT_MATRIX,
    )
    validate_prompts.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
    )
    _add_language_args(validate_prompts)

    single = sub.add_parser(
        "single",
        help="Run hierarchical language and B1 editing for one source scene",
    )
    single.add_argument(
        "--input-raw",
        type=Path,
        required=True,
    )
    single.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    single.add_argument(
        "--prompt",
        required=True,
    )
    single.add_argument(
        "--occluder",
        choices=[
            "vehicle",
            "bicycle",
            "generic_object",
            "traffic_cone",
            "barrier",
            "czone_sign",
        ],
        default=None,
        help="Optional execution override; omit to test language-derived type",
    )
    single.add_argument(
        "--side",
        choices=["auto", "left", "right"],
        default="auto",
        help="Occluder side. auto samples once when language omits the side.",
    )
    single.add_argument(
        "--direction",
        choices=["left_to_right", "right_to_left"],
        default=None,
        help="Deprecated compatibility override; converted to occluder side.",
    )
    single.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Optional pedestrian-speed override",
    )
    single.add_argument(
        "--risk",
        choices=["mild", "moderate", "aggressive"],
        default=None,
    )
    single.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    single.add_argument(
        "--no-b1-visuals",
        action="store_true",
    )
    _add_language_args(single)

    batch = sub.add_parser(
        "batch",
        help="Run a configured side-based B0/B1 matrix",
    )
    _add_batch_args(batch)

    language_batch = sub.add_parser(
        "batch-language",
        help="Pair each diagnostic language prompt with source scenes",
    )
    language_batch.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
    )
    language_batch.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    language_batch.add_argument(
        "--prompt-jsonl",
        type=Path,
        default=DEFAULT_PROMPT_MATRIX,
    )
    language_batch.add_argument(
        "--scenes-per-prompt",
        type=int,
        default=2,
    )
    language_batch.add_argument(
        "--glob-pattern",
        default="**/sledge_raw.gz",
    )
    language_batch.add_argument(
        "--max-cases",
        type=int,
        default=None,
    )
    language_batch.add_argument(
        "--no-b1-visuals",
        action="store_true",
    )
    _add_language_args(language_batch)

    refine = sub.add_parser(
        "refine",
        help="Run raw diffusion, protected diffusion, or both",
    )
    _add_refine_args(refine)
    refine.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    reconstruct = sub.add_parser(
        "reconstruct",
        help=(
            "RVAE encode/decode accepted B1 scenes and write raw-candidate "
            "plus semantic-protected simulation gzip caches"
        ),
    )
    reconstruct.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )
    reconstruct.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    reconstruct.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        default=DEFAULT_AE_CHECKPOINT,
    )
    reconstruct.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    reconstruct.add_argument(
        "--max-scenes",
        type=int,
        default=None,
    )
    reconstruct.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write the summary without failing if some scenes are rejected",
    )

    evaluate = sub.add_parser(
        "evaluate-b2",
        help="Re-evaluate an existing B2 cache",
    )
    evaluate.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )
    evaluate.add_argument(
        "--mode",
        choices=[
            RAW_DIFFUSION_BASELINE,
            SEMANTIC_PROTECTED,
        ],
        required=True,
    )

    export_b1 = sub.add_parser(
        "export-b1",
        help="Export accepted B1 scenes as typed simulation caches",
    )
    export_b1.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    audit_gz = sub.add_parser(
        "audit-gz",
        help="Re-open and index all simulator-facing generated gzip stages",
    )
    audit_gz.add_argument("--run-root", type=Path, required=True)
    audit_gz.add_argument("--no-rvae", action="store_true")
    audit_gz.add_argument("--no-b2", action="store_true")
    audit_gz.add_argument("--allow-partial", action="store_true")
    export_b1.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    export_b1.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    export_b1.add_argument(
        "--no-previews",
        action="store_true",
    )

    simulate = sub.add_parser(
        "simulate",
        help="Run simulation on one B2 mode",
    )
    simulate.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )
    simulate.add_argument(
        "--stage",
        choices=["b1", "rvae", "b2"],
        default="b2",
        help="Simulator-facing gzip stage to run",
    )
    simulate.add_argument(
        "--mode",
        choices=[
            RAW_DIFFUSION_BASELINE,
            SEMANTIC_PROTECTED,
        ],
        default=SEMANTIC_PROTECTED,
    )
    simulate.add_argument(
        "--planner",
        default="pdm_closed_planner",
    )
    simulate.add_argument(
        "--limit",
        type=int,
        default=100,
    )
    simulate.add_argument(
        "--dry-run",
        action="store_true",
    )

    compare = sub.add_parser(
        "compare-stages",
        help="Use the historical B0/B1/protected-B2 simulation comparison",
    )
    compare.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )
    compare.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    compare.add_argument(
        "--planner",
        default="pdm_closed_planner",
    )
    compare.add_argument(
        "--limit",
        type=int,
        default=20,
    )

    simulation_summary = sub.add_parser(
        "summarize-simulation",
        help="Summarize a nuPlan metric parquet",
    )
    simulation_summary.add_argument(
        "--metrics-parquet",
        type=Path,
        required=True,
    )
    simulation_summary.add_argument(
        "--output-json",
        type=Path,
        required=True,
    )

    all_command = sub.add_parser(
        "all",
        help="Run configured B1, both B2 modes, and protected-B2 simulation",
    )
    _add_batch_args(all_command)
    _add_refine_args(
        all_command,
        include_mode=False,
    )
    all_command.add_argument(
        "--planner",
        default="pdm_closed_planner",
    )
    all_command.add_argument(
        "--simulation-limit",
        type=int,
        default=100,
    )
    all_command.add_argument(
        "--skip-refine",
        action="store_true",
    )
    all_command.add_argument(
        "--skip-b1-export",
        action="store_true",
    )
    all_command.add_argument(
        "--skip-rvae-reconstruction",
        action="store_true",
    )
    all_command.add_argument(
        "--skip-simulation",
        action="store_true",
    )
    return parser


def _add_language_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llm-provider",
        choices=["none", "ollama"],
        default="none",
    )
    parser.add_argument(
        "--llm-model",
        default="qwen2.5:7b",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
    )


def _add_batch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--profile",
        default="debug20",
    )
    parser.add_argument(
        "--matrix-config",
        type=Path,
        default=DEFAULT_MATRIX,
    )
    parser.add_argument(
        "--glob-pattern",
        default="**/sledge_raw.gz",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--target-accepted",
        type=int,
        default=None,
        help="Stop only after this many B1 scenes pass the strict gate",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=None,
        help="Build this many deterministic candidates before acceptance filtering",
    )
    parser.add_argument(
        "--control-mode",
        choices=["controlled", "prompt_only"],
        default="controlled",
        help="Use explicit overrides or derive all controls from each prompt",
    )
    parser.add_argument(
        "--accept-defaults",
        action="store_true",
        help="Compatibility flag; deterministic parser defaults remain enabled",
    )
    parser.add_argument(
        "--no-b1-visuals",
        action="store_true",
    )
    _add_language_args(parser)


def _add_refine_args(
    parser: argparse.ArgumentParser,
    *,
    include_mode: bool = True,
) -> None:
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
    parser.add_argument(
        "--save-visuals",
        action="store_true",
    )
    if include_mode:
        parser.add_argument(
            "--mode",
            choices=[
                RAW_DIFFUSION_BASELINE,
                SEMANTIC_PROTECTED,
                "both",
            ],
            default="both",
        )


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "generate-prompts":
        path = write_prompt_cases(
            args.output,
            default_prompt_cases(),
        )
        print(json.dumps({"output": str(path)}, indent=2))
        return

    if args.command == "validate-prompts":
        print(
            json.dumps(
                _validate_prompt_matrix(args),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "single":
        side = None if args.side == "auto" else args.side
        case = ExperimentCase(
            sample_id="single",
            condition_id="single_hierarchical_occlusion",
            input_raw=str(args.input_raw),
            source_relative_path=args.input_raw.name,
            source_scenario_type=args.input_raw.parent.parent.name,
            prompt=args.prompt,
            occluder_type=args.occluder,
            occluder_side=side or "sample_once",
            pedestrian_speed_mps=args.speed,
            risk_level=args.risk,
            replicate=0,
            template_seed=args.seed,
        )
        # Deprecated direction is injected through a one-off wrapper only.
        if args.direction:
            derived_side = (
                "left"
                if args.direction == "left_to_right"
                else "right"
            )
            case = replace(
                case,
                occluder_side=derived_side,
            )
        pipeline = _pipeline_from_args(args)
        print(
            json.dumps(
                pipeline.run_case(case),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command in {"batch", "all"}:
        config = load_matrix_config(args.matrix_config)
        if args.candidate_pool_size is not None:
            config = json.loads(json.dumps(config))
            profile_cfg = config.get("profiles", {}).get(args.profile)
            if profile_cfg is None:
                raise KeyError(f"Unknown profile={args.profile!r}")
            profile_cfg["total_samples"] = int(args.candidate_pool_size)
            profile_cfg["samples_per_condition"] = 0
        cases = build_experiment_cases(
            input_root=args.input_root,
            profile=args.profile,
            matrix_config=config,
            glob_pattern=args.glob_pattern,
            max_cases=args.max_cases,
        )
        if args.control_mode == "prompt_only":
            cases = [
                replace(
                    case,
                    occluder_type=None,
                    occluder_side="sample_once",
                    pedestrian_speed_mps=None,
                    risk_level=None,
                )
                for case in cases
            ]
        pipeline = _pipeline_from_args(args)
        batch_summary = pipeline.run_batch(
            cases,
            target_accepted=args.target_accepted,
        )
        print(
            json.dumps(
                batch_summary,
                ensure_ascii=False,
                indent=2,
            )
        )
        if not batch_summary["target_reached"]:
            raise RuntimeError(
                "Candidate pool exhausted before target_accepted was reached: "
                f"accepted={batch_summary['accepted_count']}, "
                f"target={batch_summary['target_accepted']}"
            )
        if args.command == "batch":
            return
        if not args.skip_b1_export:
            print(
                json.dumps(
                    export_b1_simulation_cache(
                        args.output_root,
                        args.config,
                        limit=args.target_accepted,
                        save_previews=not args.no_b1_visuals,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        if not args.skip_rvae_reconstruction:
            print(
                json.dumps(
                    run_rvae_reconstruction(
                        run_root=args.output_root,
                        config=args.config,
                        autoencoder_checkpoint=args.autoencoder_checkpoint,
                        device=args.device,
                        max_scenes=args.max_refine_scenes,
                        strict=True,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        if not args.skip_refine:
            print(
                json.dumps(
                    _run_refine(
                        args,
                        args.output_root,
                        mode="both",
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        if not args.skip_b1_export:
            print(
                json.dumps(
                    build_generation_gzip_manifest(
                        args.output_root,
                        require_rvae=not args.skip_rvae_reconstruction,
                        require_b2=not args.skip_refine,
                        strict=True,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        if not args.skip_simulation:
            result = run_and_visualize_stage_simulations(
                repo_root=REPO_ROOT,
                run_root=args.output_root,
                config=args.config,
                planner=args.planner,
                limit=args.simulation_limit,
            )
            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return

    if args.command == "batch-language":
        prompt_cases = read_prompt_cases(args.prompt_jsonl)
        cases = build_prompt_matrix_cases(
            input_root=args.input_root,
            prompt_cases=prompt_cases,
            scenes_per_prompt=args.scenes_per_prompt,
            glob_pattern=args.glob_pattern,
            max_cases=args.max_cases,
        )
        pipeline = _pipeline_from_args(args)
        print(
            json.dumps(
                pipeline.run_batch(cases),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "refine":
        print(
            json.dumps(
                _run_refine(
                    args,
                    args.run_root,
                    mode=args.mode,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "reconstruct":
        print(
            json.dumps(
                run_rvae_reconstruction(
                    run_root=args.run_root,
                    config=args.config,
                    autoencoder_checkpoint=args.autoencoder_checkpoint,
                    device=args.device,
                    max_scenes=args.max_scenes,
                    strict=not args.allow_partial,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "evaluate-b2":
        print(
            json.dumps(
                evaluate_b2_cache(
                    RunLayout(args.run_root.resolve()),
                    diffusion_mode=args.mode,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "export-b1":
        result = export_b1_simulation_cache(
            args.run_root,
            args.config,
            limit=args.limit,
            save_previews=not args.no_previews,
        )
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "audit-gz":
        print(
            json.dumps(
                build_generation_gzip_manifest(
                    args.run_root,
                    require_rvae=not args.no_rvae,
                    require_b2=not args.no_b2,
                    strict=not args.allow_partial,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "simulate":
        layout = RunLayout(args.run_root.resolve())
        scenario_cache = {
            "b1": args.run_root.resolve() / "b1_simulation_cache",
            "rvae": layout.rvae_cache,
            "b2": layout.b2_cache_for(args.mode),
        }[args.stage]
        result = run_simulation(
            repo_root=REPO_ROOT,
            scenario_cache=scenario_cache,
            output_manifest=(
                layout.manifests
                / f"simulation_launch_{args.stage}_{args.mode}.json"
            ),
            planner=args.planner,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "compare-stages":
        result = run_and_visualize_stage_simulations(
            repo_root=REPO_ROOT,
            run_root=args.run_root,
            config=args.config,
            planner=args.planner,
            limit=args.limit,
        )
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "summarize-simulation":
        result = summarize_simulation_metrics(
            args.metrics_parquet,
            args.output_json,
        )
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )


def _validate_prompt_matrix(args) -> Dict[str, object]:
    cases = read_prompt_cases(args.input_jsonl)
    pipeline = HierarchicalEventFramePipeline(
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        ollama_url=args.ollama_url,
    )
    validator = HierarchicalOccludedTemplateValidator()
    router = SceneConstructionRouter()
    rows = []
    args.output_jsonl.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with args.output_jsonl.open(
        "w",
        encoding="utf-8",
    ) as stream:
        for case in cases:
            result = pipeline.parse_to_result(case.prompt)
            construction = router.route(
                prompt=case.prompt,
                hierarchical_spec=result.spec,
            )
            # Keep the language template itself as the primary object, but make
            # the downstream execution decision explicit in the same spec.
            result.spec["scene_construction"] = construction.to_dict()
            validation = validator.validate(
                result.spec,
                case.expected,
            )
            row = {
                "case_id": case.case_id,
                "prompt": case.prompt,
                "passed": validation.passed,
                "construction_mode": construction.mode,
                "scene_construction": construction.to_dict(),
                "frame_issues": result.frame_issues,
                "hierarchy_issues": result.hierarchy_issues,
                "validation": validation.to_dict(),
                "spec": result.spec,
            }
            rows.append(row)
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )
    return {
        "num_cases": len(rows),
        "num_passed": sum(
            bool(row["passed"])
            for row in rows
        ),
        "construction_modes": {
            EDIT_EXISTING: sum(
                row["construction_mode"] == EDIT_EXISTING
                for row in rows
            ),
            SYNTHESIZE_NEW: sum(
                row["construction_mode"] == SYNTHESIZE_NEW
                for row in rows
            ),
        },
        "output_jsonl": str(args.output_jsonl),
        "failures": [
            {
                "case_id": row["case_id"],
                "issues": row["validation"]["issues"],
            }
            for row in rows
            if not row["passed"]
        ],
    }


def _pipeline_from_args(args) -> OccludedPedestrianPipeline:
    return OccludedPedestrianPipeline(
        args.output_root,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        ollama_url=args.ollama_url,
        strict_check=True,
        save_visuals=not args.no_b1_visuals,
    )


def _run_refine(
    args,
    run_root: Path,
    *,
    mode: str,
):
    common = {
        "run_root": run_root,
        "config": args.config,
        "autoencoder_checkpoint": args.autoencoder_checkpoint,
        "diffusion_checkpoint": args.diffusion_checkpoint,
        "device": args.device,
        "max_scenes": args.max_refine_scenes,
        "repair_attempts": args.repair_attempts,
        "save_visuals": args.save_visuals,
    }
    if mode == "both":
        return run_diffusion_comparison(**common)
    return run_half_denoise(
        **common,
        diffusion_mode=mode,
    )


if __name__ == "__main__":
    main()
