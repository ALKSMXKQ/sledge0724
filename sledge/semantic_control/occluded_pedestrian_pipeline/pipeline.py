
"""Structured B0/B1/B2 pipeline using hierarchical language templates."""

from __future__ import annotations

from copy import deepcopy
import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sledge.semantic_control.io import load_raw_scene, save_json, save_raw_scene
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.diffusion_semantic_retention import (
    aggregate_retention,
    compare_modes,
    evaluate_retention,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.metrics import (
    aggregate_stage_metrics,
    evaluate_occluded_pedestrian_scene,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.visualization import (
    save_scene_comparison,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.experiment_matrix import (
    ExperimentCase,
    summarize_case_matrix,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.b0_scene_context import (
    B0SceneContextExtractor,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.compositional_editor import (
    CompositionalSemanticSceneEditor,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.elastic_context_editor import (
    ElasticContextHazardConstructor,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_spec import (
    HazardSemanticSpec,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.primitive_compiler import (
    compile_spec_to_ops,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.diffusion_modes import (
    RAW_DIFFUSION_BASELINE,
    SEMANTIC_PROTECTED,
    SUPPORTED_DIFFUSION_MODES,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.scene_normalization import (
    compact_edited_scene,
    normalize_editable_scene,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.template_scene_synthesizer import (
    TemplateSceneSynthesizer,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.language.eventframe_adapter import (
    OccludedPedestrianEventFrameAdapter,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.language.scene_construction_router import (
    EDIT_EXISTING,
    SYNTHESIZE_NEW,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.object_types import (
    embed_type_overrides,
    make_type_override,
    normalize_occluder_type,
    tracked_object_type_name,
)


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @property
    def b0_cache(self) -> Path:
        return self.root / "b0_original_cache"

    @property
    def b1_cache(self) -> Path:
        return self.root / "b1_edited_cache"

    @property
    def b2_root(self) -> Path:
        return self.root / "b2_diffusion"

    def b2_reports_for(self, mode: str) -> Path:
        return self.b2_root / mode / "reports"

    def b2_cache_for(self, mode: str) -> Path:
        return self.b2_root / mode / "generated_cache"

    # Backward-compatible aliases point to the protected comparison.
    @property
    def b2_reports(self) -> Path:
        return self.b2_reports_for(SEMANTIC_PROTECTED)

    @property
    def b2_cache(self) -> Path:
        return self.b2_cache_for(SEMANTIC_PROTECTED)

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    def ensure(self) -> None:
        paths = [
            self.b0_cache,
            self.b1_cache,
            self.artifacts,
            self.manifests,
        ]
        for mode in SUPPORTED_DIFFUSION_MODES:
            paths.extend(
                [
                    self.b2_reports_for(mode),
                    self.b2_cache_for(mode),
                ]
            )
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)


class OccludedPedestrianPipeline:
    """Natural language -> hierarchical template -> B1 edited scene."""

    def __init__(
        self,
        output_root: Path,
        *,
        llm_provider: str = "none",
        llm_model: str = "qwen2.5:7b",
        ollama_url: str = "http://127.0.0.1:11434",
        strict_check: bool = True,
        save_visuals: bool = True,
    ) -> None:
        self.layout = RunLayout(Path(output_root).resolve())
        self.layout.ensure()
        self.adapter = OccludedPedestrianEventFrameAdapter(
            llm_provider=llm_provider,
            llm_model=llm_model,
            ollama_url=ollama_url,
        )
        self.editor = CompositionalSemanticSceneEditor(
            strict_check=strict_check
        )
        self.b0_context_extractor = B0SceneContextExtractor()
        self.synthesizer = TemplateSceneSynthesizer()
        self.elastic_constructor = ElasticContextHazardConstructor()
        self.save_visuals = save_visuals

    def run_case(self, case: ExperimentCase) -> Dict[str, Any]:
        sample_id = case.sample_id
        artifact_root = self.layout.artifacts / sample_id
        language_dir = artifact_root / "01_language"
        specification_dir = artifact_root / "02_specification"
        editing_dir = artifact_root / "03_editing"
        evaluation_dir = artifact_root / "04_evaluation"
        for path in [
            language_dir,
            specification_dir,
            editing_dir,
            evaluation_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        # A source raw file is always loaded. In edit mode it is the semantic
        # B0. In synthesis mode it is used only as a fixed-capacity SLEDGE
        # scaffold; TemplateSceneSynthesizer clears all semantic content before
        # creating the new road/ego base.
        source_scene, source_format = load_raw_scene(case.input_raw)
        editable_source, normalization_report = normalize_editable_scene(
            source_scene
        )
        b0_context = self.b0_context_extractor.extract(editable_source)

        adaptation = self.adapter.adapt(
            case.prompt,
            case.overrides,
            case_id=case.sample_id,
            b0_scene_context=b0_context.to_dict(),
        )
        construction = dict(adaptation.scene_construction or {})
        construction_mode = str(
            construction.get("mode", EDIT_EXISTING)
        )
        if construction_mode not in {EDIT_EXISTING, SYNTHESIZE_NEW}:
            raise ValueError(
                f"Unsupported scene construction mode: {construction_mode!r}"
            )

        synthesis_report: Dict[str, Any] = {
            "construction_mode": construction_mode,
            "source_scene_usage": construction.get(
                "source_scene_usage",
                "semantic_base_scene",
            ),
            "source_semantic_content_preserved": (
                construction_mode == EDIT_EXISTING
            ),
        }
        if construction_mode == EDIT_EXISTING:
            construction_base_scene = editable_source
            baseline_scene = source_scene
        else:
            construction_base_scene, synthesis_report = self.synthesizer.synthesize(
                scaffold_scene=editable_source,
                hierarchical_spec=adaptation.hierarchical_spec,
                sampled_parameters=adaptation.sampled_parameters,
                construction_plan=construction,
            )
            # For synthesized cases, B0 means the newly synthesized clean base
            # (road + ego, before the hazard is inserted), not the source
            # scaffold file.
            baseline_scene = construction_base_scene

        spec = adaptation.hazard_spec
        sampled = adaptation.sampled_parameters
        context_edit_report: Dict[str, Any] = {
            "schema_version": "elastic_context_edit_v2_move_then_delete",
            "construction_mode": construction_mode,
            "hard_road_lane_preserved": True,
            "background_actor_edit_count": 0,
            "background_reposition_count": 0,
            "background_removal_count": 0,
            "removal_used": False,
            "background_actor_edits": [],
        }

        if construction_mode == EDIT_EXISTING:
            (
                edited_scene,
                edit_result,
                editor_report,
                context_edit_report,
                spec,
            ) = self.elastic_constructor.construct(
                construction_base_scene,
                spec,
                sampled,
                self.editor,
            )
            # Ego state is intentionally NOT restored. In the current contract,
            # ego motion is a semantic control and may be synchronized to the
            # requested occluded-pedestrian interaction.
            editor_report["ego_state_restored_from_b0"] = False
            editor_report["ego_state_edit_policy"] = "semantic_control"
        else:
            edited_scene, edit_result, editor_report = self.editor.edit(
                construction_base_scene,
                spec,
            )
            editor_report["ego_state_restored_from_b0"] = False
            editor_report["ego_state_edit_policy"] = "template_synthesis"

        # Save the primitive program for the *effective* spec. In elastic edit
        # mode the risk geometry may have been widened between attempts while
        # the road/lane geometry stayed immutable.
        primitive_ops = compile_spec_to_ops(spec)

        edited_scene, compaction_report = compact_edited_scene(
            edited_scene,
            edit_result,
        )
        editor_report["slot_compaction"] = compaction_report
        editor_report["scene_construction"] = construction
        editor_report["synthesis_report"] = synthesis_report
        editor_report["context_edit_report"] = context_edit_report

        semantic_payload = {
            "schema_version": (
                "occluded_pedestrian_semantic_report_v6_timing_consistent"
            ),
            "sample_id": sample_id,
            "input_raw": case.input_raw,
            "source_format": source_format,
            "construction_mode": construction_mode,
            "scene_construction": construction,
            "b0_scene_context": b0_context.to_dict(),
            "synthesis_report": synthesis_report,
            "context_edit_report": context_edit_report,
            "semantic_direction": sampled["semantic_direction"],
            "occluder_side": sampled["occluder_side"],
            "concrete_direction": sampled["concrete_direction"],
            "sampled_parameters": sampled,
            "edit_result": edit_result.to_dict(),
            "report": editor_report,
        }
        semantic_projection_time_s = float(
            editor_report.get("extra", {}).get(
                "semantic_validation_time_offset_s",
                0.0,
            )
        )
        semantic_lane_center_y = float(
            editor_report.get("extra", {}).get(
                "conflict_lane_y",
                0.0,
            )
        )

        b0_metrics = evaluate_occluded_pedestrian_scene(
            baseline_scene,
            spec,
            lane_center_y=semantic_lane_center_y,
        )
        b1_metrics = evaluate_occluded_pedestrian_scene(
            edited_scene,
            spec,
            preferred_pedestrian_index=edit_result.pedestrian_index,
            preferred_occluder_index=edit_result.occluder_index,
            preferred_occluder_elem_name=edit_result.occluder_elem_name,
            projection_time_s=semantic_projection_time_s,
            lane_center_y=semantic_lane_center_y,
        )

        # One canonical semantic contract now drives both the detailed stage
        # metrics and the strict B1 acceptance gate.  The previous standalone
        # validator used a different lateral-TTC definition, which caused the
        # same scene to pass one checker and fail the other.  No hard semantic
        # check is removed; the strict report simply exposes the same 11 checks
        # in required/passed form and retains the legacy editor-side validation
        # for diagnostics only.
        strict_validation = _strict_validation_from_metrics(
            b1_metrics,
            legacy_editor_validation=editor_report.get("validation"),
        )
        b1_pass = bool(
            strict_validation.get("overall_pass", False)
            and b1_metrics.get("overall_pass", False)
        )

        b0_dir = self.layout.b0_cache / sample_id
        b1_dir = self.layout.b1_cache / sample_id
        baseline_raw_path = save_raw_scene(
            b0_dir / "sledge_raw",
            baseline_scene,
            source_format=source_format,
        )
        edited_raw_path = save_raw_scene(
            b1_dir / "sledge_raw",
            edited_scene,
            source_format=source_format,
        )

        canonical_occluder = normalize_occluder_type(
            sampled["executable_occluder_type"],
            strict=True,
        )
        occluder_tracked_type = tracked_object_type_name(
            canonical_occluder
        )
        raw_type_overrides = make_type_override(
            edit_result.occluder_elem_name,
            edit_result.occluder_index,
            occluder_tracked_type,
        )
        embed_type_overrides(
            edited_raw_path,
            raw_type_overrides,
        )

        scenario_label = {
            "schema_version": "occluded_pedestrian_label_v6_timing_consistent",
            "sample_id": sample_id,
            "scenario_type": "sudden_pedestrian_crossing",
            "semantic_family": "occluded_pedestrian",
            "severity_level": spec.risk_layer.risk_level,
            "prompt": case.prompt,
            "prompt_case_id": case.prompt_case_id,
            "condition_id": case.condition_id,
            "construction_mode": construction_mode,
            "scene_construction": construction,
            "source_scene_path": case.input_raw,
            "source_scene_usage": construction.get(
                "source_scene_usage",
                "semantic_base_scene",
            ),
            "source_scenario_type": case.source_scenario_type,
            "language_actor_detail": sampled["language_actor_detail"],
            "semantic_occluder_type": sampled[
                "semantic_occluder_type"
            ],
            "occluder_type": canonical_occluder,
            "occluder_tracked_object_type": occluder_tracked_type,
            "object_type_overrides": raw_type_overrides,
            "semantic_direction": sampled["semantic_direction"],
            "occluder_side": sampled["occluder_side"],
            "direction": sampled["concrete_direction"],
            "pedestrian_speed_mps": sampled["actor_speed_mps"],
            "sample_seed": sampled["seed"],
            "road_parameter_source": sampled.get("road_parameter_source"),
            "ego_state_source": sampled.get("ego_state_source"),
            "hard_road_lane_preserved": bool(
                context_edit_report.get("hard_road_lane_preserved", True)
            ),
            "background_actor_edit_count": int(
                context_edit_report.get("background_actor_edit_count", 0)
            ),
            "background_reposition_count": int(
                context_edit_report.get("background_reposition_count", 0)
            ),
            "background_removal_count": int(
                context_edit_report.get("background_removal_count", 0)
            ),
            "background_removal_used": bool(
                context_edit_report.get("removal_used", False)
            ),
            "ego_edit": context_edit_report.get("ego_edit"),
            "semantic_lane_center_y": float(semantic_lane_center_y),
            "semantic_projection_time_s": float(semantic_projection_time_s),
            "accepted": b1_pass,
            "artifact_root": str(artifact_root),
        }
        save_json(b1_dir / "scenario_label.json", scenario_label)
        save_json(b1_dir / "edit_report.json", edit_result.to_dict())
        save_json(
            b1_dir / "semantic_report.json",
            semantic_payload,
        )

        save_json(
            language_dir / "prompt.json",
            {
                "prompt": case.prompt,
                "overrides": case.overrides.__dict__,
            },
        )
        save_json(
            language_dir / "event_frame.json",
            adaptation.event_frame,
        )
        save_json(
            language_dir / "hierarchical_spec.json",
            adaptation.hierarchical_spec,
        )
        save_json(
            language_dir / "template_validation.json",
            adaptation.template_validation,
        )
        save_json(
            language_dir / "scene_construction.json",
            construction,
        )
        save_json(
            language_dir / "b0_scene_context.json",
            b0_context.to_dict(),
        )
        save_json(
            language_dir / "sampled_parameters.json",
            adaptation.sampled_parameters,
        )
        save_json(
            language_dir / "adaptation.json",
            adaptation.to_dict(),
        )
        save_json(
            specification_dir / "hazard_spec.json",
            spec.to_dict(),
        )
        save_json(
            specification_dir / "primitive_ops.json",
            [operation.to_dict() for operation in primitive_ops],
        )
        save_json(
            editing_dir / "edit_result.json",
            edit_result.to_dict(),
        )
        save_json(
            editing_dir / "scene_normalization.json",
            normalization_report,
        )
        save_json(
            editing_dir / "scene_synthesis.json",
            synthesis_report,
        )
        save_json(
            editing_dir / "context_edit_report.json",
            context_edit_report,
        )
        save_json(
            editing_dir / "scene_compaction.json",
            compaction_report,
        )
        save_json(
            editing_dir / "editor_report.json",
            editor_report,
        )
        save_json(
            editing_dir / "strict_validation.json",
            strict_validation,
        )
        save_json(
            evaluation_dir / "b0_metrics.json",
            b0_metrics,
        )
        save_json(
            evaluation_dir / "b1_metrics.json",
            b1_metrics,
        )

        visualization_path = None
        visualization_warning = None
        if self.save_visuals:
            try:
                visualization_path = save_scene_comparison(
                    baseline_scene,
                    edited_scene,
                    edit_result.to_dict(),
                    editing_dir / "b0_b1_comparison.png",
                    prompt=case.prompt,
                )
            except Exception as exc:
                visualization_warning = (
                    f"{type(exc).__name__}: {exc}"
                )
                save_json(
                    editing_dir / "visualization_warning.json",
                    {"warning": visualization_warning},
                )

        summary = {
            **case.to_dict(),
            "source_format": source_format,
            "construction_mode": construction_mode,
            "source_scene_usage": construction.get(
                "source_scene_usage",
                "semantic_base_scene",
            ),
            "b0_raw": str(baseline_raw_path),
            "b1_raw": str(edited_raw_path),
            "artifact_root": str(artifact_root),
            "template_validation_pass": bool(
                adaptation.template_validation.get("passed", False)
            ),
            # The sampled control vector becomes a concrete scene only after
            # B1 editing and strict geometric/semantic validation.
            "concrete_parameter_sample_ready": bool(
                sampled.get("concrete_parameter_sample_ready", False)
            ),
            "sampled_scene_ready": b1_pass,
            "sampled_occluder_side": sampled["occluder_side"],
            "sampled_concrete_direction": sampled[
                "concrete_direction"
            ],
            "road_parameter_source": sampled.get("road_parameter_source"),
            "ego_state_source": sampled.get("ego_state_source"),
            "hard_road_lane_preserved": bool(
                context_edit_report.get("hard_road_lane_preserved", True)
            ),
            "background_actor_edit_count": int(
                context_edit_report.get("background_actor_edit_count", 0)
            ),
            "background_reposition_count": int(
                context_edit_report.get("background_reposition_count", 0)
            ),
            "background_removal_count": int(
                context_edit_report.get("background_removal_count", 0)
            ),
            "background_removal_used": bool(
                context_edit_report.get("removal_used", False)
            ),
            "ego_state_edited": bool(
                construction_mode == EDIT_EXISTING
                and context_edit_report.get("ego_edit")
            ),
            "elastic_selected_attempt": context_edit_report.get("selected_attempt"),
            "semantic_lane_center_y": float(semantic_lane_center_y),
            "semantic_projection_time_s": float(semantic_projection_time_s),
            "frame_check_pass": bool(
                adaptation.frame_verification.get("passed", False)
            ),
            "mapped_spec_check_pass": bool(
                adaptation.spec_verification.get("passed", False)
            ),
            "strict_validation_pass": bool(
                strict_validation.get("overall_pass", False)
            ),
            "b0_pass": bool(b0_metrics.get("overall_pass", False)),
            "b1_pass": b1_pass,
            "b1_semantic_satisfaction_rate": float(
                b1_metrics.get(
                    "semantic_satisfaction_rate",
                    0.0,
                )
            ),
            "visualization": (
                str(visualization_path)
                if visualization_path
                else None
            ),
            "visualization_warning": visualization_warning,
        }
        save_json(
            artifact_root / "sample_summary.json",
            summary,
        )
        return summary

    def run_batch(
        self,
        cases: Iterable[ExperimentCase],
    ) -> Dict[str, Any]:
        cases = list(cases)
        rows: List[Dict[str, Any]] = []
        b0_metrics_rows: List[Dict[str, Any]] = []
        b1_metrics_rows: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []

        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case.sample_id}")
            try:
                row = self.run_case(case)
                rows.append(row)
                artifact = Path(row["artifact_root"])
                b0_metrics_rows.append(
                    _load_json(
                        artifact / "04_evaluation/b0_metrics.json"
                    )
                )
                b1_metrics_rows.append(
                    _load_json(
                        artifact / "04_evaluation/b1_metrics.json"
                    )
                )
                print(
                    "  "
                    f"mode={row.get('construction_mode')} "
                    f"template={row['template_validation_pass']} "
                    f"b1_pass={row['b1_pass']} "
                    f"ssr={row['b1_semantic_satisfaction_rate']:.3f}"
                )
            except Exception as exc:
                failure = {
                    **case.to_dict(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                failed_metric = {
                    "sample_id": case.sample_id,
                    "overall_pass": False,
                    "semantic_satisfaction_rate": 0.0,
                    "checks": {},
                    "error": failure["error"],
                }
                b0_metrics_rows.append(dict(failed_metric))
                b1_metrics_rows.append(dict(failed_metric))
                save_json(
                    self.layout.artifacts
                    / case.sample_id
                    / "error.json",
                    failure,
                )
                print(
                    f"  failed: {type(exc).__name__}: {exc}"
                )

        _write_jsonl(
            self.layout.manifests / "cases.jsonl",
            [case.to_dict() for case in cases],
        )
        _write_jsonl(
            self.layout.manifests / "b1_results.jsonl",
            rows,
        )
        _write_jsonl(
            self.layout.manifests / "failures.jsonl",
            failures,
        )
        _write_csv(
            self.layout.manifests / "b1_results.csv",
            rows,
        )

        summary = {
            "schema_version": "occluded_pedestrian_run_summary_v2_hierarchical",
            "matrix": summarize_case_matrix(cases),
            "num_finished": len(rows),
            "num_failed": len(failures),
            "construction_mode_counts": {
                EDIT_EXISTING: sum(
                    row.get("construction_mode") == EDIT_EXISTING
                    for row in rows
                ),
                SYNTHESIZE_NEW: sum(
                    row.get("construction_mode") == SYNTHESIZE_NEW
                    for row in rows
                ),
            },
            "template_pass_count": sum(
                bool(row.get("template_validation_pass"))
                for row in rows
            ),
            "concrete_parameter_sample_ready_count": sum(
                bool(row.get("concrete_parameter_sample_ready"))
                for row in rows
            ),
            "sampled_scene_ready_count": sum(
                bool(row.get("sampled_scene_ready"))
                for row in rows
            ),
            "b0": aggregate_stage_metrics(
                b0_metrics_rows,
                "B0_construction_base",
            ),
            "b1": aggregate_stage_metrics(
                b1_metrics_rows,
                "B1_hierarchical_edited",
            ),
            "paths": {
                "b0_cache": str(self.layout.b0_cache),
                "b1_cache": str(self.layout.b1_cache),
                "artifacts": str(self.layout.artifacts),
            },
        }
        save_json(
            self.layout.manifests / "b1_summary.json",
            summary,
        )
        return summary


def _strict_validation_from_metrics(
    metrics: Dict[str, Any],
    *,
    legacy_editor_validation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    checks = {
        name: {
            "required": True,
            "passed": bool(passed),
        }
        for name, passed in dict(metrics.get("checks", {}) or {}).items()
    }
    required = list(checks.values())
    passed_count = sum(bool(row["passed"]) for row in required)
    return {
        "schema_version": "occluded_pedestrian_strict_validation_v2_canonical_metrics",
        "source": "evaluation.metrics.evaluate_occluded_pedestrian_scene",
        "checks": checks,
        "semantic_satisfaction_rate": (
            float(passed_count / len(required)) if required else 0.0
        ),
        "overall_pass": bool(required and passed_count == len(required)),
        "interaction": dict(metrics.get("interaction", {}) or {}),
        "pedestrian": dict(metrics.get("pedestrian", {}) or {}),
        "occluder": dict(metrics.get("occluder", {}) or {}),
        "legacy_editor_validation": legacy_editor_validation,
    }


def run_half_denoise(
    *,
    run_root: Path,
    config: Path,
    autoencoder_checkpoint: Path,
    diffusion_checkpoint: Path,
    device: str,
    diffusion_mode: str,
    max_scenes: Optional[int] = None,
    repair_attempts: int = 6,
    save_visuals: bool = False,
) -> Dict[str, Any]:
    """Run one B2 mode and evaluate all expected samples."""

    if diffusion_mode not in SUPPORTED_DIFFUSION_MODES:
        raise ValueError(
            f"Unsupported diffusion_mode={diffusion_mode!r}"
        )

    from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.refinement_alignment import (
        OccludedPedestrianRefinementAlignmentEvaluator,
    )
    from sledge.semantic_control.occluded_pedestrian_pipeline.generation.refinement_runner import (
        OccludedPedestrianHalfDenoiseRunner,
    )

    layout = RunLayout(Path(run_root).resolve())
    layout.ensure()
    args = argparse.Namespace(
        original_dir=str(layout.b0_cache),
        edited_dir=str(layout.b1_cache),
        output=str(layout.b2_reports_for(diffusion_mode)),
        config=str(config),
        autoencoder_checkpoint=str(autoencoder_checkpoint),
        diffusion_checkpoint=str(diffusion_checkpoint),
        scenario_cache_root=str(layout.b2_cache_for(diffusion_mode)),
        map_id=None,
        glob_pattern="**/sledge_raw.gz",
        max_scenes=max_scenes,
        skip_existing=True,
        output_layout="mirror",
        num_inference_timesteps=24,
        guidance_scale=4.0,
        low_noise_start_step_seq=(
            "20"
            if diffusion_mode == RAW_DIFFUSION_BASELINE
            else "20,21,22,23"
        ),
        # Raw baseline is one unprotected generation. Multiple repair attempts
        # would select for semantic preservation and invalidate the baseline.
        repair_attempts=(
            1
            if diffusion_mode == RAW_DIFFUSION_BASELINE
            else repair_attempts
        ),
        seed=0,
        alignment_threshold=0.70,
        min_preservation_ratio=0.95,
        # Raw baseline must keep failing generations so their failure modes are
        # included in the denominator and can be inspected.
        strict_save_only_passing=(
            diffusion_mode == SEMANTIC_PROTECTED
        ),
        diff_threshold=1e-4,
        diff_mask_dilation=3,
        roi_mask_dilation=(
            0
            if diffusion_mode == RAW_DIFFUSION_BASELINE
            else 2
        ),
        pedestrian_roi_strength=(
            0.0
            if diffusion_mode == RAW_DIFFUSION_BASELINE
            else 1.0
        ),
        vehicle_roi_strength=(
            0.0
            if diffusion_mode == RAW_DIFFUSION_BASELINE
            else 1.0
        ),
        roadside_anchor_strength=(
            0.0
            if diffusion_mode == RAW_DIFFUSION_BASELINE
            else 1.0
        ),
        lane_anchor_strength=(
            0.0
            if diffusion_mode == RAW_DIFFUSION_BASELINE
            else 1.0
        ),
        crossing_corridor_strength=(
            0.0
            if diffusion_mode == RAW_DIFFUSION_BASELINE
            else 0.95
        ),
        generic_roi_strength=(
            0.0
            if diffusion_mode == RAW_DIFFUSION_BASELINE
            else 0.95
        ),
        device=device,
        save_latents=False,
        save_visuals=save_visuals,
        diffusion_mode=diffusion_mode,
    )
    runner = OccludedPedestrianHalfDenoiseRunner(args)
    runner.alignment_evaluator = (
        OccludedPedestrianRefinementAlignmentEvaluator(
            projection_time_s=2.1
        )
    )
    runner.run_batch()
    return evaluate_b2_cache(
        layout,
        diffusion_mode=diffusion_mode,
    )


def run_diffusion_comparison(
    *,
    run_root: Path,
    config: Path,
    autoencoder_checkpoint: Path,
    diffusion_checkpoint: Path,
    device: str,
    max_scenes: Optional[int] = None,
    repair_attempts: int = 6,
    save_visuals: bool = False,
) -> Dict[str, Any]:
    """Run raw baseline first, then the semantic-protected comparison."""

    raw = run_half_denoise(
        run_root=run_root,
        config=config,
        autoencoder_checkpoint=autoencoder_checkpoint,
        diffusion_checkpoint=diffusion_checkpoint,
        device=device,
        diffusion_mode=RAW_DIFFUSION_BASELINE,
        max_scenes=max_scenes,
        repair_attempts=repair_attempts,
        save_visuals=save_visuals,
    )
    protected = run_half_denoise(
        run_root=run_root,
        config=config,
        autoencoder_checkpoint=autoencoder_checkpoint,
        diffusion_checkpoint=diffusion_checkpoint,
        device=device,
        diffusion_mode=SEMANTIC_PROTECTED,
        max_scenes=max_scenes,
        repair_attempts=repair_attempts,
        save_visuals=save_visuals,
    )
    comparison = compare_modes(
        raw.get("retention", {}),
        protected.get("retention", {}),
    )
    layout = RunLayout(Path(run_root).resolve())
    save_json(
        layout.manifests
        / "diffusion_semantic_retention_comparison.json",
        comparison,
    )
    return comparison


def evaluate_b2_cache(
    layout: RunLayout,
    *,
    diffusion_mode: str = SEMANTIC_PROTECTED,
) -> Dict[str, Any]:
    """Evaluate B2 without assuming decoder slot identity."""

    if diffusion_mode not in SUPPORTED_DIFFUSION_MODES:
        raise ValueError(
            f"Unsupported diffusion_mode={diffusion_mode!r}"
        )
    expected_cases = _read_jsonl(
        layout.manifests / "cases.jsonl"
    )
    expected_ids = {
        str(row["sample_id"]): row
        for row in expected_cases
    }
    metrics_by_id: Dict[str, Dict[str, Any]] = {}
    retention_by_id: Dict[str, Dict[str, Any]] = {}
    cache_root = layout.b2_cache_for(diffusion_mode)

    for label_path in cache_root.glob("**/scenario_label.json"):
        label = _load_json(label_path)
        sample_id = str(label.get("sample_id", ""))
        if not sample_id:
            edited_path = Path(
                str(label.get("edited_scene_path", ""))
            )
            sample_id = edited_path.parent.name
        if sample_id not in expected_ids:
            continue

        vector_path = label_path.parent / "sledge_vector.gz"
        if not vector_path.exists():
            continue
        scene, _ = load_raw_scene(vector_path)
        spec_path = (
            layout.artifacts
            / sample_id
            / "02_specification/hazard_spec.json"
        )
        spec = HazardSemanticSpec.from_dict(
            _load_json(spec_path)
        )
        protected = dict(label.get("protected_slots", {}) or {})
        use_preferred = (
            diffusion_mode == SEMANTIC_PROTECTED
            and bool(protected)
        )
        b1_label_path = layout.b1_cache / sample_id / "scenario_label.json"
        b1_label = _load_json(b1_label_path) if b1_label_path.exists() else {}
        semantic_lane_center_y = float(
            label.get(
                "semantic_lane_center_y",
                b1_label.get("semantic_lane_center_y", 0.0),
            )
        )
        semantic_projection_time_s = float(
            label.get(
                "semantic_projection_time_s",
                b1_label.get("semantic_projection_time_s", 2.1),
            )
        )
        metrics = evaluate_occluded_pedestrian_scene(
            scene,
            spec,
            preferred_pedestrian_index=(
                protected.get("pedestrians")
                if use_preferred
                else None
            ),
            preferred_occluder_index=(
                protected.get("occluder_index")
                if use_preferred
                else None
            ),
            preferred_occluder_elem_name=str(
                protected.get(
                    "occluder_element",
                    "vehicles",
                )
            ),
            projection_time_s=semantic_projection_time_s,
            lane_center_y=semantic_lane_center_y,
        )
        metrics["sample_id"] = sample_id
        metrics["diffusion_mode"] = diffusion_mode
        metrics["vector_path"] = str(vector_path)
        metrics_by_id[sample_id] = metrics

        retention = evaluate_retention(
            metrics,
            sample_id=sample_id,
            diffusion_mode=diffusion_mode,
        ).to_dict()
        retention_by_id[sample_id] = retention
        evaluation_dir = (
            layout.artifacts
            / sample_id
            / "04_evaluation"
        )
        save_json(
            evaluation_dir
            / f"b2_{diffusion_mode}_metrics.json",
            metrics,
        )
        save_json(
            evaluation_dir
            / f"b2_{diffusion_mode}_retention.json",
            retention,
        )

    metric_rows: List[Dict[str, Any]] = []
    retention_rows: List[Dict[str, Any]] = []
    for sample_id in expected_ids:
        metric_rows.append(
            metrics_by_id.get(
                sample_id,
                {
                    "sample_id": sample_id,
                    "diffusion_mode": diffusion_mode,
                    "overall_pass": False,
                    "semantic_satisfaction_rate": 0.0,
                    "checks": {},
                    "error": "no B2 output",
                },
            )
        )
        retention_rows.append(
            retention_by_id.get(
                sample_id,
                {
                    "sample_id": sample_id,
                    "diffusion_mode": diffusion_mode,
                    "pedestrian_retained": False,
                    "occluder_retained": False,
                    "occlusion_retained": False,
                    "unique_direction_retained": False,
                    "ego_path_intersection_retained": False,
                    "reveal_event_proxy_retained": False,
                    "interaction_timing_retained": False,
                    "full_hazard_semantics_retained": False,
                    "failure_reasons": ["no_b2_output"],
                },
            )
        )

    _write_jsonl(
        layout.manifests
        / f"b2_{diffusion_mode}_results.jsonl",
        metric_rows,
    )
    _write_jsonl(
        layout.manifests
        / f"b2_{diffusion_mode}_retention.jsonl",
        retention_rows,
    )
    stage_summary = aggregate_stage_metrics(
        metric_rows,
        f"B2_{diffusion_mode}",
    )
    retention_summary = aggregate_retention(
        retention_rows,
        diffusion_mode=diffusion_mode,
    )
    summary = {
        "schema_version": "occluded_pedestrian_b2_summary_v2",
        "diffusion_mode": diffusion_mode,
        "num_expected": len(expected_ids),
        "num_generated": len(metrics_by_id),
        "stage_metrics": stage_summary,
        "retention": retention_summary,
        "cache_root": str(cache_root),
    }
    save_json(
        layout.manifests
        / f"b2_{diffusion_mode}_summary.json",
        summary,
    )
    return summary


def _load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_jsonl(
    path: Path,
    rows: Iterable[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [
            json.loads(line)
            for line in stream
            if line.strip()
        ]


def _write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    if not rows:
        return
    scalar_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(
            value,
            (str, int, float, bool),
        )
        or value is None
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=scalar_keys,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: row.get(key)
                    for key in scalar_keys
                }
            )