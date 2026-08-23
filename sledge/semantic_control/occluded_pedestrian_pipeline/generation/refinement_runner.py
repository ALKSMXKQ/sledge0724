"""Occlusion-aware half-denoise runner with raw and protected modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

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
        processed_report = self._resolve_processed_slots(
            edited_raw,
            template_vector,
            edit_report,
        )
        self._active_template = template_vector
        self._active_edit_report = processed_report

        if hasattr(self.alignment_evaluator, "set_reference_scene"):
            self.alignment_evaluator.set_reference_scene(template_vector)
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
            label_path = Path(str(vector_path)).parent / "scenario_label.json"
            if label_path.exists():
                with label_path.open("r", encoding="utf-8") as stream:
                    label = json.load(stream)
                label.update(
                    {
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
                    }
                )
                save_json(label_path, label)
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
        return {
            "road_topology": "all_lines",
            "pedestrians": int(report.get("pedestrian_index", -1)),
            "occluder_element": str(
                report.get("occluder_elem_name", "vehicles")
            ),
            "occluder_index": int(report.get("occluder_index", -1)),
        }

    @staticmethod
    def _composite_protected_slots(
        vector: Any,
        template: Any,
        report: Dict[str, Any],
    ) -> None:
        vector.lines.states = np.asarray(template.lines.states).copy()
        vector.lines.mask = np.asarray(template.lines.mask).copy()
        vector.ego.states = np.asarray(template.ego.states).copy()
        vector.ego.mask = np.asarray(template.ego.mask).copy()

        pedestrian_index = int(report.get("pedestrian_index", -1))
        if pedestrian_index >= 0:
            OccludedPedestrianHalfDenoiseRunner._copy_slot(
                vector.pedestrians,
                template.pedestrians,
                pedestrian_index,
            )

        occluder_name = str(
            report.get("occluder_elem_name", "vehicles")
        )
        occluder_index = int(report.get("occluder_index", -1))
        if (
            occluder_index >= 0
            and occluder_name in {"vehicles", "static_objects"}
        ):
            OccludedPedestrianHalfDenoiseRunner._copy_slot(
                getattr(vector, occluder_name),
                getattr(template, occluder_name),
                occluder_index,
            )

    @staticmethod
    def _copy_slot(
        target_elem: Any,
        source_elem: Any,
        index: int,
    ) -> None:
        target_states = np.asarray(target_elem.states)
        source_states = np.asarray(source_elem.states)
        target_mask = np.asarray(target_elem.mask)
        source_mask = np.asarray(source_elem.mask)
        if index >= len(target_states) or index >= len(source_states):
            raise IndexError(
                f"Protected slot {index} is outside decoded/template capacity"
            )
        width = min(
            target_states.shape[-1],
            source_states.shape[-1],
        )
        target_states[index, :width] = source_states[index, :width]
        target_mask.reshape(-1)[index] = source_mask.reshape(-1)[index]

    @staticmethod
    def _resolve_processed_slots(
        raw: Any,
        vector: Any,
        report: Dict[str, Any],
    ) -> Dict[str, Any]:
        resolved = dict(report)
        raw_pedestrian_index = int(report.get("pedestrian_index", -1))
        resolved["pedestrian_index"] = (
            OccludedPedestrianHalfDenoiseRunner._match_slot(
                raw.pedestrians,
                raw_pedestrian_index,
                vector.pedestrians,
            )
        )
        occluder_name = str(
            report.get("occluder_elem_name", "vehicles")
        )
        raw_occluder_index = int(report.get("occluder_index", -1))
        resolved["occluder_index"] = (
            OccludedPedestrianHalfDenoiseRunner._match_slot(
                getattr(raw, occluder_name),
                raw_occluder_index,
                getattr(vector, occluder_name),
            )
        )
        return resolved

    @staticmethod
    def _match_slot(
        raw_elem: Any,
        raw_index: int,
        vector_elem: Any,
    ) -> int:
        raw_states = np.asarray(raw_elem.states)
        if raw_index < 0 or raw_index >= len(raw_states):
            return -1
        target = raw_states[raw_index]
        states = np.asarray(vector_elem.states)
        masks = np.asarray(vector_elem.mask).reshape(-1) >= 0.3
        valid = np.where(masks)[0]
        if not len(valid):
            return -1
        width = min(5, states.shape[-1], target.shape[-1])
        scales = np.asarray(
            [1.0, 1.0, 0.5, 0.25, 0.25],
            dtype=np.float32,
        )[:width]
        errors = np.linalg.norm(
            (states[valid, :width] - target[:width]) * scales,
            axis=1,
        )
        return int(valid[int(np.argmin(errors))])
