"""Occlusion-aware extension of the existing half-denoise runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import (
    sledge_raw_feature_processing,
)
from sledge.script.run_half_denoise_from_tiered_cache import MultiScenarioHalfDenoiseRunner
from sledge.semantic_control.io import load_raw_scene, save_json


class OccludedPedestrianHalfDenoiseRunner(MultiScenarioHalfDenoiseRunner):
    """Denoise the background while exactly protecting controlled entities.

    The RVAE may drop a newly inserted pedestrian or distort road connectivity
    even when its latent ROI is protected.  This runner treats the road graph,
    ego timing, controlled pedestrian and occluder as hard conditioning layers
    after each decode. Other traffic participants remain diffusion-generated.
    """

    def __init__(self, args) -> None:
        super().__init__(args)
        self._active_template = None
        self._active_edit_report: Dict[str, Any] = {}

    def run_one(self, edited_scene_path: Path, out_dir: Path, index: int) -> Dict[str, object]:
        edited_raw, _ = load_raw_scene(edited_scene_path)
        template_vector, _ = sledge_raw_feature_processing(edited_raw, self.ae_config)
        report_path = edited_scene_path.parent / "edit_report.json"
        with report_path.open("r", encoding="utf-8") as fp:
            edit_report = json.load(fp)
        processed_report = self._resolve_processed_slots(edited_raw, template_vector, edit_report)
        self._active_template = template_vector
        self._active_edit_report = processed_report
        if hasattr(self.alignment_evaluator, "set_reference_scene"):
            self.alignment_evaluator.set_reference_scene(template_vector)
        if hasattr(self.alignment_evaluator, "set_preferred_slots"):
            self.alignment_evaluator.set_preferred_slots(
                int(processed_report.get("pedestrian_index", -1)),
                int(processed_report.get("occluder_index", -1)),
                str(processed_report.get("occluder_elem_name", "vehicles")),
            )
        try:
            summary = super().run_one(edited_scene_path, out_dir, index)
        finally:
            self._active_template = None
            self._active_edit_report = {}

        summary["semantic_vector_compositing"] = True
        summary["protected_slots"] = self._protected_slots(processed_report)
        save_json(out_dir / "summary.json", summary)
        vector_path = summary.get("scenario_cache_vector_path")
        if vector_path:
            label_path = Path(str(vector_path)).parent / "scenario_label.json"
            if label_path.exists():
                with label_path.open("r", encoding="utf-8") as fp:
                    label = json.load(fp)
                label.update(
                    {
                        "semantic_family": "occluded_pedestrian",
                        "semantic_vector_compositing": True,
                        "semantic_projection_time_s": 2.1,
                        "road_topology_lock": "exact_b1_lines",
                        "protected_slots": self._protected_slots(processed_report),
                    }
                )
                save_json(label_path, label)
        return summary

    def _attempt_repair(self, *args, **kwargs):
        vector, final_latents, start_idx = super()._attempt_repair(*args, **kwargs)
        if self._active_template is not None:
            self._composite_protected_slots(vector, self._active_template, self._active_edit_report)
        return vector, final_latents, start_idx

    @staticmethod
    def _protected_slots(report: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "road_topology": "all_lines",
            "pedestrians": int(report.get("pedestrian_index", -1)),
            "occluder_element": str(report.get("occluder_elem_name", "vehicles")),
            "occluder_index": int(report.get("occluder_index", -1)),
        }

    @staticmethod
    def _composite_protected_slots(vector: Any, template: Any, report: Dict[str, Any]) -> None:
        # Road connectivity and ego timing are structural conditions, not
        # background diversity variables. Exact line compositing prevents the
        # locally tangled road graphs that nearest-point metrics can miss.
        vector.lines.states = np.asarray(template.lines.states).copy()
        vector.lines.mask = np.asarray(template.lines.mask).copy()
        vector.ego.states = np.asarray(template.ego.states).copy()
        vector.ego.mask = np.asarray(template.ego.mask).copy()
        ped_index = int(report.get("pedestrian_index", -1))
        if ped_index >= 0:
            OccludedPedestrianHalfDenoiseRunner._copy_slot(
                vector.pedestrians, template.pedestrians, ped_index
            )
        occ_name = str(report.get("occluder_elem_name", "vehicles"))
        occ_index = int(report.get("occluder_index", -1))
        if occ_index >= 0 and occ_name in {"vehicles", "static_objects"}:
            OccludedPedestrianHalfDenoiseRunner._copy_slot(
                getattr(vector, occ_name), getattr(template, occ_name), occ_index
            )

    @staticmethod
    def _copy_slot(target_elem: Any, source_elem: Any, index: int) -> None:
        target_states = np.asarray(target_elem.states)
        source_states = np.asarray(source_elem.states)
        target_mask = np.asarray(target_elem.mask)
        source_mask = np.asarray(source_elem.mask)
        if index >= len(target_states) or index >= len(source_states):
            raise IndexError(f"Protected slot {index} is outside decoded/template capacity")
        width = min(target_states.shape[-1], source_states.shape[-1])
        target_states[index, :width] = source_states[index, :width]
        target_mask.reshape(-1)[index] = source_mask.reshape(-1)[index]

    @staticmethod
    def _resolve_processed_slots(raw: Any, vector: Any, report: Dict[str, Any]) -> Dict[str, Any]:
        resolved = dict(report)
        raw_ped_index = int(report.get("pedestrian_index", -1))
        resolved["pedestrian_index"] = OccludedPedestrianHalfDenoiseRunner._match_slot(
            raw.pedestrians, raw_ped_index, vector.pedestrians
        )
        occ_name = str(report.get("occluder_elem_name", "vehicles"))
        raw_occ_index = int(report.get("occluder_index", -1))
        resolved["occluder_index"] = OccludedPedestrianHalfDenoiseRunner._match_slot(
            getattr(raw, occ_name), raw_occ_index, getattr(vector, occ_name)
        )
        return resolved

    @staticmethod
    def _match_slot(raw_elem: Any, raw_index: int, vector_elem: Any) -> int:
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
        scales = np.asarray([1.0, 1.0, 0.5, 0.25, 0.25], dtype=np.float32)[:width]
        errors = np.linalg.norm((states[valid, :width] - target[:width]) * scales, axis=1)
        return int(valid[int(np.argmin(errors))])
