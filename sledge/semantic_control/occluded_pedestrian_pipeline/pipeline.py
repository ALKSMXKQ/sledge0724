"""Structured B0/B1/B2 pipeline orchestration."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sledge.semantic_control.io import load_raw_scene, save_json, save_raw_scene
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
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.compositional_editor import (
    CompositionalSemanticSceneEditor,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_spec import (
    HazardSemanticSpec,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.primitive_compiler import (
    compile_spec_to_ops,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.scene_normalization import (
    compact_edited_scene,
    normalize_editable_scene,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.semantic_validator import (
    validate_scene_against_report,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.language.eventframe_adapter import (
    OccludedPedestrianEventFrameAdapter,
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
    def b2_reports(self) -> Path:
        return self.root / "b2_diffusion_reports"

    @property
    def b2_cache(self) -> Path:
        return self.root / "b2_generated_cache"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    def ensure(self) -> None:
        for path in [
            self.b0_cache,
            self.b1_cache,
            self.b2_reports,
            self.b2_cache,
            self.artifacts,
            self.manifests,
        ]:
            path.mkdir(parents=True, exist_ok=True)


class OccludedPedestrianPipeline:
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
        self.editor = CompositionalSemanticSceneEditor(strict_check=strict_check)
        self.save_visuals = save_visuals

    def run_case(self, case: ExperimentCase) -> Dict[str, Any]:
        sample_id = case.sample_id
        artifact_root = self.layout.artifacts / sample_id
        language_dir = artifact_root / "01_language"
        specification_dir = artifact_root / "02_specification"
        editing_dir = artifact_root / "03_editing"
        evaluation_dir = artifact_root / "04_evaluation"
        for path in [language_dir, specification_dir, editing_dir, evaluation_dir]:
            path.mkdir(parents=True, exist_ok=True)

        original_scene, source_format = load_raw_scene(case.input_raw)
        editable_scene, normalization_report = normalize_editable_scene(original_scene)
        adaptation = self.adapter.adapt(case.prompt, case.overrides)
        spec = adaptation.hazard_spec
        primitive_ops = compile_spec_to_ops(spec)

        edited_scene, edit_result, editor_report = self.editor.edit(editable_scene, spec)
        edited_scene, compaction_report = compact_edited_scene(edited_scene, edit_result)
        editor_report["slot_compaction"] = compaction_report
        semantic_payload = {
            "schema_version": "occluded_pedestrian_semantic_report_v1",
            "sample_id": sample_id,
            "input_raw": case.input_raw,
            "source_format": source_format,
            "edit_result": edit_result.to_dict(),
            "report": editor_report,
        }
        strict_validation = validate_scene_against_report(edited_scene, semantic_payload)

        b0_metrics = evaluate_occluded_pedestrian_scene(original_scene, spec)
        b1_metrics = evaluate_occluded_pedestrian_scene(
            edited_scene,
            spec,
            preferred_pedestrian_index=edit_result.pedestrian_index,
            preferred_occluder_index=edit_result.occluder_index,
            preferred_occluder_elem_name=edit_result.occluder_elem_name,
            projection_time_s=float(editor_report.get("extra", {}).get("semantic_validation_time_offset_s", 0.0)),
        )
        b1_pass = bool(strict_validation.get("overall_pass", False) and b1_metrics.get("overall_pass", False))

        b0_dir = self.layout.b0_cache / sample_id
        b1_dir = self.layout.b1_cache / sample_id
        original_raw_path = save_raw_scene(b0_dir / "sledge_raw", original_scene, source_format=source_format)
        edited_raw_path = save_raw_scene(b1_dir / "sledge_raw", edited_scene, source_format=source_format)
        canonical_occluder = normalize_occluder_type(case.occluder_type, strict=True)
        occluder_tracked_type = tracked_object_type_name(canonical_occluder)
        raw_type_overrides = make_type_override(
            edit_result.occluder_elem_name,
            edit_result.occluder_index,
            occluder_tracked_type,
        )
        # Keep the visible nuPlan category inside the same gzip payload. Legacy
        # SLEDGE feature readers ignore the reserved metadata key.
        embed_type_overrides(edited_raw_path, raw_type_overrides)

        scenario_label = {
            "schema_version": "occluded_pedestrian_label_v1",
            "sample_id": sample_id,
            "scenario_type": "sudden_pedestrian_crossing",
            "semantic_family": "occluded_pedestrian",
            "severity_level": spec.risk_layer.risk_level,
            "prompt": case.prompt,
            "condition_id": case.condition_id,
            "source_scene_path": case.input_raw,
            "source_scenario_type": case.source_scenario_type,
            "occluder_type": canonical_occluder,
            "occluder_tracked_object_type": occluder_tracked_type,
            "object_type_overrides": raw_type_overrides,
            "direction": case.direction,
            "pedestrian_speed_mps": case.pedestrian_speed_mps,
            "accepted": b1_pass,
            "artifact_root": str(artifact_root),
        }
        save_json(b1_dir / "scenario_label.json", scenario_label)
        save_json(b1_dir / "edit_report.json", edit_result.to_dict())
        save_json(b1_dir / "semantic_report.json", semantic_payload)

        save_json(language_dir / "prompt.json", {"prompt": case.prompt, "overrides": case.overrides.__dict__})
        save_json(language_dir / "event_frame.json", adaptation.event_frame)
        save_json(language_dir / "mapped_eventframe_spec.json", adaptation.mapped_eventframe_spec)
        save_json(language_dir / "adaptation.json", adaptation.to_dict())
        save_json(specification_dir / "hazard_spec.json", spec.to_dict())
        save_json(specification_dir / "primitive_ops.json", [op.to_dict() for op in primitive_ops])
        save_json(editing_dir / "edit_result.json", edit_result.to_dict())
        save_json(editing_dir / "scene_normalization.json", normalization_report)
        save_json(editing_dir / "scene_compaction.json", compaction_report)
        save_json(editing_dir / "editor_report.json", editor_report)
        save_json(editing_dir / "strict_validation.json", strict_validation)
        save_json(evaluation_dir / "b0_metrics.json", b0_metrics)
        save_json(evaluation_dir / "b1_metrics.json", b1_metrics)
        visualization_path = None
        visualization_warning = None
        if self.save_visuals:
            try:
                visualization_path = save_scene_comparison(
                    original_scene,
                    edited_scene,
                    edit_result.to_dict(),
                    editing_dir / "b0_b1_comparison.png",
                    prompt=case.prompt,
                )
            except Exception as exc:
                visualization_warning = f"{type(exc).__name__}: {exc}"
                save_json(editing_dir / "visualization_warning.json", {"warning": visualization_warning})

        summary = {
            **case.to_dict(),
            "source_format": source_format,
            "b0_raw": str(original_raw_path),
            "b1_raw": str(edited_raw_path),
            "artifact_root": str(artifact_root),
            "frame_check_pass": bool(adaptation.frame_verification.get("passed", False)),
            "mapped_spec_check_pass": bool(adaptation.spec_verification.get("passed", False)),
            "strict_validation_pass": bool(strict_validation.get("overall_pass", False)),
            "b0_pass": bool(b0_metrics.get("overall_pass", False)),
            "b1_pass": b1_pass,
            "b1_semantic_satisfaction_rate": float(b1_metrics.get("semantic_satisfaction_rate", 0.0)),
            "visualization": str(visualization_path) if visualization_path else None,
            "visualization_warning": visualization_warning,
        }
        save_json(artifact_root / "sample_summary.json", summary)
        return summary

    def run_batch(self, cases: Iterable[ExperimentCase]) -> Dict[str, Any]:
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
                b0_metrics_rows.append(_load_json(artifact / "04_evaluation/b0_metrics.json"))
                b1_metrics_rows.append(_load_json(artifact / "04_evaluation/b1_metrics.json"))
                print(f"  b1_pass={row['b1_pass']} ssr={row['b1_semantic_satisfaction_rate']:.3f}")
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
                save_json(self.layout.artifacts / case.sample_id / "error.json", failure)
                print(f"  failed: {type(exc).__name__}: {exc}")

        _write_jsonl(self.layout.manifests / "cases.jsonl", [case.to_dict() for case in cases])
        _write_jsonl(self.layout.manifests / "b1_results.jsonl", rows)
        _write_jsonl(self.layout.manifests / "failures.jsonl", failures)
        _write_csv(self.layout.manifests / "b1_results.csv", rows)

        summary = {
            "schema_version": "occluded_pedestrian_run_summary_v1",
            "matrix": summarize_case_matrix(cases),
            "num_finished": len(rows),
            "num_failed": len(failures),
            "b0": aggregate_stage_metrics(b0_metrics_rows, "B0_original"),
            "b1": aggregate_stage_metrics(b1_metrics_rows, "B1_edited"),
            "paths": {
                "b0_cache": str(self.layout.b0_cache),
                "b1_cache": str(self.layout.b1_cache),
                "artifacts": str(self.layout.artifacts),
            },
        }
        save_json(self.layout.manifests / "b1_summary.json", summary)
        return summary


def run_half_denoise(
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
    """Run the existing half-denoise engine and add strict B2 evaluation."""

    from sledge.semantic_control.occluded_pedestrian_pipeline.generation.refinement_runner import (
        OccludedPedestrianHalfDenoiseRunner,
    )
    from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.refinement_alignment import (
        OccludedPedestrianRefinementAlignmentEvaluator,
    )

    layout = RunLayout(Path(run_root).resolve())
    layout.ensure()
    args = argparse.Namespace(
        original_dir=str(layout.b0_cache),
        edited_dir=str(layout.b1_cache),
        output=str(layout.b2_reports),
        config=str(config),
        autoencoder_checkpoint=str(autoencoder_checkpoint),
        diffusion_checkpoint=str(diffusion_checkpoint),
        scenario_cache_root=str(layout.b2_cache),
        map_id=None,
        glob_pattern="**/sledge_raw.gz",
        max_scenes=max_scenes,
        skip_existing=True,
        output_layout="mirror",
        num_inference_timesteps=24,
        guidance_scale=4.0,
        # With this scheduler, a larger start index means fewer denoising
        # iterations and therefore lower injected noise. Step 20 preserves the
        # source road graph while still producing a distinct decoded scene.
        low_noise_start_step_seq="20,21,22,23",
        repair_attempts=repair_attempts,
        seed=0,
        alignment_threshold=0.70,
        min_preservation_ratio=0.95,
        strict_save_only_passing=True,
        diff_threshold=1e-4,
        diff_mask_dilation=3,
        roi_mask_dilation=2,
        pedestrian_roi_strength=1.0,
        vehicle_roi_strength=1.0,
        roadside_anchor_strength=1.0,
        lane_anchor_strength=1.0,
        crossing_corridor_strength=0.95,
        generic_roi_strength=0.95,
        device=device,
        save_latents=False,
        save_visuals=save_visuals,
    )
    runner = OccludedPedestrianHalfDenoiseRunner(args)
    # The legacy runner's ordinary-crossing evaluator does not check an
    # occluder and uses a different TTC definition. Replace only the evaluator;
    # model loading, masks, denoising and cache writing remain unchanged.
    runner.alignment_evaluator = OccludedPedestrianRefinementAlignmentEvaluator(projection_time_s=2.1)
    runner.run_batch()
    return evaluate_b2_cache(layout)


def evaluate_b2_cache(layout: RunLayout) -> Dict[str, Any]:
    expected_cases = _read_jsonl(layout.manifests / "cases.jsonl")
    expected_ids = {str(row["sample_id"]): row for row in expected_cases}
    metrics_by_id: Dict[str, Dict[str, Any]] = {}

    for label_path in layout.b2_cache.glob("**/scenario_label.json"):
        label = _load_json(label_path)
        edited_path = Path(str(label.get("edited_scene_path", "")))
        sample_id = edited_path.parent.name
        if sample_id not in expected_ids:
            continue
        vector_path = label_path.parent / "sledge_vector.gz"
        if not vector_path.exists():
            continue
        scene, _ = load_raw_scene(vector_path)
        spec_path = layout.artifacts / sample_id / "02_specification/hazard_spec.json"
        spec = HazardSemanticSpec.from_dict(_load_json(spec_path))
        protected = label.get("protected_slots", {})
        metrics = evaluate_occluded_pedestrian_scene(
            scene,
            spec,
            preferred_pedestrian_index=protected.get("pedestrians"),
            preferred_occluder_index=protected.get("occluder_index"),
            preferred_occluder_elem_name=str(protected.get("occluder_element", "vehicles")),
            projection_time_s=float(label.get("semantic_projection_time_s", 2.1)),
        )
        metrics["sample_id"] = sample_id
        metrics["vector_path"] = str(vector_path)
        metrics_by_id[sample_id] = metrics
        save_json(layout.artifacts / sample_id / "04_evaluation/b2_metrics.json", metrics)

    rows: List[Dict[str, Any]] = []
    for sample_id in expected_ids:
        rows.append(
            metrics_by_id.get(
                sample_id,
                {
                    "sample_id": sample_id,
                    "overall_pass": False,
                    "semantic_satisfaction_rate": 0.0,
                    "checks": {},
                    "error": "no accepted B2 output",
                },
            )
        )
    _write_jsonl(layout.manifests / "b2_results.jsonl", rows)
    summary = aggregate_stage_metrics(rows, "B2_half_denoise")
    summary.update(
        {
            "num_expected": len(expected_ids),
            "num_generated": len(metrics_by_id),
            "generated_pass_count": sum(
                bool(row.get("overall_pass")) for row in metrics_by_id.values()
            ),
            "generated_pass_rate": (
                sum(bool(row.get("overall_pass")) for row in metrics_by_id.values())
                / len(metrics_by_id)
                if metrics_by_id
                else 0.0
            ),
        }
    )
    save_json(layout.manifests / "b2_summary.json", summary)
    return summary


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    scalar_keys = [
        key for key, value in rows[0].items()
        if isinstance(value, (str, int, float, bool)) or value is None
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in scalar_keys})
