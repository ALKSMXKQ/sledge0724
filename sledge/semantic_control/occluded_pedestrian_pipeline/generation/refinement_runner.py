"""Occlusion-aware half-denoise runner with raw and protected modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import (
    sledge_raw_feature_processing,
)
from sledge.script.run_half_denoise_from_tiered_cache import (
    MultiScenarioHalfDenoiseRunner,
)
from sledge.semantic_control.io import load_raw_scene, save_json
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.diffusion_modes import (
    RAW_DIFFUSION_BASELINE,
    SEMANTIC_PROTECTED,
    SUPPORTED_DIFFUSION_MODES,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.metrics import (
    evaluate_occluded_pedestrian_scene,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_spec import (
    HazardSemanticSpec,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.semantic_protection import (
    audit_protected_semantics,
    composite_protected_semantics,
    copy_element_slot,
    make_simulation_compatible_vector,
    match_processed_slot,
    protected_slots,
    resolve_processed_slots,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.object_types import (
    audit_simulator_roundtrip,
    embed_type_overrides,
    make_type_override,
    read_type_overrides,
    tracked_object_type_name,
)


class OccludedPedestrianHalfDenoiseRunner(MultiScenarioHalfDenoiseRunner):
    """Run diffusion either without object restoration or with hard protection.

    ``raw_diffusion_baseline`` never copies the B1 pedestrian, occluder, road or
    ego state back into the decoded vector. It is the valid mode for measuring
    the diffusion model's own dangerous-semantic retention.

    ``semantic_protected`` preserves the historical protected-compositing
    behavior and is used only as a controlled comparison.
    """

    def __init__(self, args) -> None:
        super().__init__(args)
        self.diffusion_mode = str(
            getattr(args, "diffusion_mode", SEMANTIC_PROTECTED)
        )
        if self.diffusion_mode not in SUPPORTED_DIFFUSION_MODES:
            raise ValueError(
                f"Unsupported diffusion_mode={self.diffusion_mode!r}; "
                f"expected {sorted(SUPPORTED_DIFFUSION_MODES)}"
            )
        self._active_template = None
        self._active_edit_report: Dict[str, Any] = {}

    @property
    def semantic_compositing_enabled(self) -> bool:
        return self.diffusion_mode == SEMANTIC_PROTECTED

    def run_one(
        self,
        edited_scene_path: Path,
        out_dir: Path,
        index: int,
    ) -> Dict[str, object]:
        edited_raw, _ = load_raw_scene(edited_scene_path)
        template_vector, _ = sledge_raw_feature_processing(
            edited_raw,
            self.ae_config,
        )
        report_path = edited_scene_path.parent / "edit_report.json"
        with report_path.open("r", encoding="utf-8") as stream:
            edit_report = json.load(stream)
        label_path = edited_scene_path.parent / "scenario_label.json"
        with label_path.open("r", encoding="utf-8") as stream:
            source_label = json.load(stream)
        processed_report = self._resolve_processed_slots(
            edited_raw,
            template_vector,
            edit_report,
        )
        self._active_template = template_vector
        self._active_edit_report = processed_report

        if hasattr(self.alignment_evaluator, "set_reference_scene"):
            self.alignment_evaluator.set_reference_scene(template_vector)
        if hasattr(self.alignment_evaluator, "set_lane_center_y"):
            self.alignment_evaluator.set_lane_center_y(
                float(source_label.get("semantic_lane_center_y", 0.0))
            )
        if hasattr(self.alignment_evaluator, "projection_time_s"):
            self.alignment_evaluator.projection_time_s = float(
                source_label.get("semantic_projection_time_s", 2.1)
            )
        # Preferred slot indices are allowed only in the protected comparison.
        # Raw diffusion evaluation must rediscover objects after decoding.
        if (
            self.semantic_compositing_enabled
            and hasattr(self.alignment_evaluator, "set_preferred_slots")
        ):
            self.alignment_evaluator.set_preferred_slots(
                int(processed_report.get("pedestrian_index", -1)),
                int(processed_report.get("occluder_index", -1)),
                str(
                    processed_report.get(
                        "occluder_elem_name",
                        "vehicles",
                    )
                ),
            )

        try:
            summary = super().run_one(edited_scene_path, out_dir, index)
        finally:
            self._active_template = None
            self._active_edit_report = {}

        summary["diffusion_mode"] = self.diffusion_mode
        summary["semantic_vector_compositing"] = (
            self.semantic_compositing_enabled
        )
        summary["protected_slots"] = (
            self._protected_slots(processed_report)
            if self.semantic_compositing_enabled
            else {}
        )
        save_json(out_dir / "summary.json", summary)

        vector_path = summary.get("scenario_cache_vector_path")
        if vector_path:
            generated_path = Path(str(vector_path))
            generated_label_path = generated_path.parent / "scenario_label.json"
            semantic_audit: Dict[str, Any] = {}
            simulator_audit: Dict[str, Any] = {}
            canonical_metrics: Dict[str, Any] = {}
            type_overrides: Dict[str, Any] = {}
            gzip_roundtrip_pass = True
            if self.semantic_compositing_enabled:
                tracked_type = source_label.get(
                    "occluder_tracked_object_type"
                )
                if not tracked_type:
                    tracked_type = tracked_object_type_name(
                        source_label.get("occluder_type", "vehicle")
                    )
                type_overrides = make_type_override(
                    str(processed_report["occluder_elem_name"]),
                    int(processed_report["occluder_index"]),
                    str(tracked_type),
                )
                embed_type_overrides(generated_path, type_overrides)
                roundtrip_scene, _ = load_raw_scene(generated_path)
                template_sim = make_simulation_compatible_vector(
                    template_vector,
                    edited_raw,
                )
                semantic_audit = audit_protected_semantics(
                    roundtrip_scene,
                    template_sim,
                    processed_report,
                )
                spec_path = (
                    Path(self.edited_dir).parent
                    / "artifacts"
                    / edited_scene_path.parent.name
                    / "02_specification/hazard_spec.json"
                )
                with spec_path.open("r", encoding="utf-8") as stream:
                    spec = HazardSemanticSpec.from_dict(json.load(stream))
                canonical_metrics = evaluate_occluded_pedestrian_scene(
                    roundtrip_scene,
                    spec,
                    preferred_pedestrian_index=int(
                        processed_report["pedestrian_index"]
                    ),
                    preferred_occluder_index=int(
                        processed_report["occluder_index"]
                    ),
                    preferred_occluder_elem_name=str(
                        processed_report["occluder_elem_name"]
                    ),
                    projection_time_s=float(
                        source_label.get("semantic_projection_time_s", 2.1)
                    ),
                    lane_center_y=float(
                        source_label.get("semantic_lane_center_y", 0.0)
                    ),
                )
                simulator_audit = audit_simulator_roundtrip(
                    generated_path,
                    type_overrides,
                )
                gzip_roundtrip_pass = bool(
                    semantic_audit["overall_pass"]
                    and read_type_overrides(generated_path)
                    == type_overrides
                    and simulator_audit["overall_pass"]
                    and canonical_metrics["overall_pass"]
                )
                save_json(
                    out_dir / "semantic_protection_audit.json",
                    semantic_audit,
                )
                save_json(
                    out_dir / "simulator_roundtrip_audit.json",
                    simulator_audit,
                )
                save_json(
                    out_dir / "semantic_contract_metrics.json",
                    canonical_metrics,
                )
                if not gzip_roundtrip_pass:
                    generated_path.unlink(missing_ok=True)
                    generated_label_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        "Protected diffusion gzip failed exact semantic/type "
                        "round-trip validation"
                    )

            if generated_label_path.exists():
                with generated_label_path.open("r", encoding="utf-8") as stream:
                    label = json.load(stream)
                label.update(
                    {
                        "sample_id": edited_scene_path.parent.name,
                        "semantic_family": "occluded_pedestrian",
                        "diffusion_mode": self.diffusion_mode,
                        "semantic_vector_compositing": (
                            self.semantic_compositing_enabled
                        ),
                        "semantic_projection_time_s": 2.1,
                        "road_topology_lock": (
                            "exact_b1_lines"
                            if self.semantic_compositing_enabled
                            else "none"
                        ),
                        "protected_slots": (
                            self._protected_slots(processed_report)
                            if self.semantic_compositing_enabled
                            else {}
                        ),
                        "semantic_contract_pass": bool(
                            summary.get("selected_semantic_pass", False)
                            and gzip_roundtrip_pass
                        ),
                        "gzip_roundtrip_pass": gzip_roundtrip_pass,
                        "semantic_protection_audit": semantic_audit,
                        "simulator_roundtrip_audit": simulator_audit,
                        "semantic_contract_metrics": canonical_metrics,
                        "object_type_overrides": type_overrides,
                    }
                )
                save_json(generated_label_path, label)
            summary.update(
                {
                    "semantic_contract_pass": bool(
                        summary.get("selected_semantic_pass", False)
                        and gzip_roundtrip_pass
                    ),
                    "gzip_roundtrip_pass": gzip_roundtrip_pass,
                    "semantic_protection_audit": semantic_audit,
                    "simulator_roundtrip_audit": simulator_audit,
                }
            )
            save_json(out_dir / "summary.json", summary)
        return summary

    def _attempt_repair(self, *args, **kwargs):
        vector, final_latents, start_idx = super()._attempt_repair(
            *args,
            **kwargs,
        )
        if (
            self.semantic_compositing_enabled
            and self._active_template is not None
        ):
            self._composite_protected_slots(
                vector,
                self._active_template,
                self._active_edit_report,
            )
        return vector, final_latents, start_idx

    @staticmethod
    def _protected_slots(report: Dict[str, Any]) -> Dict[str, Any]:
        return protected_slots(report)

    @staticmethod
    def _composite_protected_slots(
        vector: Any,
        template: Any,
        report: Dict[str, Any],
    ) -> None:
        composite_protected_semantics(vector, template, report)

    @staticmethod
    def _copy_slot(
        target_elem: Any,
        source_elem: Any,
        index: int,
    ) -> None:
        copy_element_slot(target_elem, source_elem, index)

    @staticmethod
    def _resolve_processed_slots(
        raw: Any,
        vector: Any,
        report: Dict[str, Any],
    ) -> Dict[str, Any]:
        return resolve_processed_slots(raw, vector, report)

    @staticmethod
    def _match_slot(
        raw_elem: Any,
        raw_index: int,
        vector_elem: Any,
    ) -> int:
        return match_processed_slot(raw_elem, raw_index, vector_elem)
