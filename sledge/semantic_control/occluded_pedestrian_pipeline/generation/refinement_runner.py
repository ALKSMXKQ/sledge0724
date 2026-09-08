"""Occlusion-aware half-denoise runner with raw and protected modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import (
    sledge_raw_feature_processing,
)
from sledge.diffusion.modelling.lane_geometry_guidance import infer_adjacency_pairs
from sledge.script.run_half_denoise_from_tiered_cache import (
    MultiScenarioHalfDenoiseRunner,
)
from sledge.semantic_control.io import load_raw_scene, save_json
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.diffusion_modes import (
    RAW_DIFFUSION_BASELINE,
    SEMANTIC_PROTECTED,
    SUPPORTED_DIFFUSION_MODES,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_spec import (
    HazardSemanticSpec,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.topology_filter import (
    classify_topology,
)

COPY_B1_ROAD = "copy_b1"
GENERATED_WITH_GUIDANCE = "generated_with_guidance"
SUPPORTED_ROAD_PROTECTION_MODES = {COPY_B1_ROAD, GENERATED_WITH_GUIDANCE}


class OccludedPedestrianHalfDenoiseRunner(MultiScenarioHalfDenoiseRunner):
    """Run diffusion with optional semantic protection and lane guidance.

    ``road_protection_mode=copy_b1`` keeps the legacy exact B1 road copy.
    ``generated_with_guidance`` leaves the diffusion road generated, while
    applying geometry guidance during denoising and preserving only the hazard
    actors/ego in semantic-protected mode.
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
        self.road_protection_mode = str(
            self.cfg.get("road_protection_mode", COPY_B1_ROAD)
        )
        if self.road_protection_mode not in SUPPORTED_ROAD_PROTECTION_MODES:
            raise ValueError(
                f"Unsupported road_protection_mode={self.road_protection_mode!r}; "
                f"expected {sorted(SUPPORTED_ROAD_PROTECTION_MODES)}"
            )
        raw_guidance = self.cfg.get("lane_guidance", {})
        self.lane_guidance_config = (
            OmegaConf.to_container(raw_guidance, resolve=True)
            if raw_guidance
            else {}
        )
        if not isinstance(self.lane_guidance_config, dict):
            self.lane_guidance_config = {}
        if self.road_protection_mode == GENERATED_WITH_GUIDANCE:
            self.lane_guidance_config = dict(self.lane_guidance_config)
            self.lane_guidance_config.setdefault("enabled", True)

        self._active_template = None
        self._active_edit_report: Dict[str, Any] = {}
        self._active_topology_family = "road_segment"
        self._active_lane_pairs = []
        self._lane_guidance_trace = []
        self._active_semantic_projection_time_s = 2.1
        self._active_semantic_lane_center_y = 0.0
        self._active_semantic_contract_source = "prompt_reparsed_fallback"
        self._active_road_topology_gate_enabled = False

    @property
    def semantic_compositing_enabled(self) -> bool:
        return self.diffusion_mode == SEMANTIC_PROTECTED

    @property
    def exact_road_copy_enabled(self) -> bool:
        return (
            self.semantic_compositing_enabled
            and self.road_protection_mode == COPY_B1_ROAD
        )

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

        scenario_label = self._load_optional_json(
            edited_scene_path.parent / "scenario_label.json"
        )
        reference_spec, contract_source = self._load_reference_hazard_spec(
            edited_scene_path.parent,
            scenario_label,
        )
        semantic_projection_time_s = float(
            scenario_label.get("semantic_projection_time_s", 2.1)
        )
        semantic_lane_center_y = float(
            scenario_label.get("semantic_lane_center_y", 0.0)
        )
        road_topology_gate_enabled = bool(self.exact_road_copy_enabled)

        topology = classify_topology(edited_raw)
        self._active_template = template_vector
        self._active_edit_report = processed_report
        self._active_topology_family = topology.family
        self._active_lane_pairs = infer_adjacency_pairs(
            np.asarray(template_vector.lines.states),
            np.asarray(template_vector.lines.mask),
        )
        self._lane_guidance_trace = []
        self._active_semantic_projection_time_s = semantic_projection_time_s
        self._active_semantic_lane_center_y = semantic_lane_center_y
        self._active_semantic_contract_source = contract_source
        self._active_road_topology_gate_enabled = road_topology_gate_enabled

        if hasattr(self.alignment_evaluator, "set_reference_scene"):
            self.alignment_evaluator.set_reference_scene(template_vector)
        if hasattr(self.alignment_evaluator, "set_reference_spec"):
            self.alignment_evaluator.set_reference_spec(reference_spec)
        if hasattr(self.alignment_evaluator, "set_projection_time_s"):
            self.alignment_evaluator.set_projection_time_s(
                semantic_projection_time_s
            )
        if hasattr(self.alignment_evaluator, "set_lane_center_y"):
            self.alignment_evaluator.set_lane_center_y(
                semantic_lane_center_y
            )
        if hasattr(self.alignment_evaluator, "set_topology_gate_enabled"):
            self.alignment_evaluator.set_topology_gate_enabled(
                road_topology_gate_enabled
            )
        if (
            self.semantic_compositing_enabled
            and hasattr(self.alignment_evaluator, "set_preferred_slots")
        ):
            self.alignment_evaluator.set_preferred_slots(
                int(processed_report.get("pedestrian_index", -1)),
                int(processed_report.get("occluder_index", -1)),
                str(processed_report.get("occluder_elem_name", "vehicles")),
            )

        try:
            summary = super().run_one(edited_scene_path, out_dir, index)
        finally:
            trace = list(self._lane_guidance_trace)
            topology_family = self._active_topology_family
            pair_count = len(self._active_lane_pairs)
            projection_time_s = self._active_semantic_projection_time_s
            lane_center_y = self._active_semantic_lane_center_y
            semantic_contract_source = self._active_semantic_contract_source
            topology_gate_enabled = self._active_road_topology_gate_enabled
            self._active_template = None
            self._active_edit_report = {}
            self._active_lane_pairs = []

        summary["diffusion_mode"] = self.diffusion_mode
        summary["semantic_vector_compositing"] = self.semantic_compositing_enabled
        summary["road_protection_mode"] = self.road_protection_mode
        summary["topology_family"] = topology_family
        summary["lane_adjacency_pair_count"] = int(pair_count)
        summary["lane_guidance_enabled"] = bool(
            self.lane_guidance_config.get("enabled", False)
        )
        summary["lane_guidance_trace"] = trace
        summary["semantic_contract_source"] = semantic_contract_source
        summary["semantic_projection_time_s"] = float(projection_time_s)
        summary["semantic_lane_center_y"] = float(lane_center_y)
        summary["road_topology_gate_enabled"] = bool(topology_gate_enabled)
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
                        "semantic_vector_compositing": self.semantic_compositing_enabled,
                        "semantic_contract_source": semantic_contract_source,
                        "semantic_projection_time_s": float(projection_time_s),
                        "semantic_lane_center_y": float(lane_center_y),
                        "topology_family": topology_family,
                        "road_topology_lock": (
                            "exact_b1_lines"
                            if self.exact_road_copy_enabled
                            else "generated_with_lane_geometry_guidance"
                        ),
                        "road_topology_gate_enabled": bool(topology_gate_enabled),
                        "road_protection_mode": self.road_protection_mode,
                        "lane_guidance_enabled": bool(
                            self.lane_guidance_config.get("enabled", False)
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

    def _attempt_repair(
        self,
        init_latents,
        preserve_mask,
        map_id,
        attempt_idx,
        scene_index,
    ):
        start_idx = self.start_step_candidates[
            attempt_idx % len(self.start_step_candidates)
        ]
        gen = torch.Generator(device=self.args.device)
        gen.manual_seed(
            int(self.args.seed) + scene_index * 1000 + attempt_idx
        )
        attempt_trace = []
        with torch.no_grad():
            denoised_vectors, final_latents = self.pipeline(
                class_labels=[map_id],
                num_inference_timesteps=self.args.num_inference_timesteps,
                guidance_scale=self.args.guidance_scale,
                num_classes=self.num_classes,
                init_latents=init_latents,
                start_timestep_index=start_idx,
                preserve_mask=preserve_mask,
                generator=gen,
                return_latents=True,
                lane_guidance=self.lane_guidance_config,
                lane_adjacency_pairs=self._active_lane_pairs,
                topology_family=self._active_topology_family,
                lane_guidance_trace=attempt_trace,
            )
        self._lane_guidance_trace.append(
            {"attempt": int(attempt_idx), "steps": attempt_trace}
        )
        vector = denoised_vectors[0].torch_to_numpy(apply_sigmoid=True)
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

    def _protected_slots(self, report: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "road_topology": (
                "all_lines"
                if self.exact_road_copy_enabled
                else "geometry_guided_generated"
            ),
            "pedestrians": int(report.get("pedestrian_index", -1)),
            "occluder_element": str(
                report.get("occluder_elem_name", "vehicles")
            ),
            "occluder_index": int(report.get("occluder_index", -1)),
        }

    def _composite_protected_slots(
        self,
        vector: Any,
        template: Any,
        report: Dict[str, Any],
    ) -> None:
        if self.exact_road_copy_enabled:
            vector.lines.states = np.asarray(template.lines.states).copy()
            vector.lines.mask = np.asarray(template.lines.mask).copy()
        vector.ego.states = np.asarray(template.ego.states).copy()
        vector.ego.mask = np.asarray(template.ego.mask).copy()

        pedestrian_index = int(report.get("pedestrian_index", -1))
        if pedestrian_index >= 0:
            self._copy_slot(
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
            self._copy_slot(
                getattr(vector, occluder_name),
                getattr(template, occluder_name),
                occluder_index,
            )

    @staticmethod
    def _load_optional_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _load_reference_hazard_spec(
        edited_scene_dir: Path,
        scenario_label: Dict[str, Any],
    ) -> Tuple[Optional[HazardSemanticSpec], str]:
        candidates = [edited_scene_dir / "hazard_spec.json"]
        artifact_root = scenario_label.get("artifact_root")
        if artifact_root:
            candidates.append(
                Path(str(artifact_root))
                / "02_specification"
                / "hazard_spec.json"
            )

        for path in candidates:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as stream:
                    payload = json.load(stream)
                if isinstance(payload, dict):
                    return HazardSemanticSpec.from_dict(payload), str(path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return None, "prompt_reparsed_fallback"

    @staticmethod
    def _copy_slot(target_elem: Any, source_elem: Any, index: int) -> None:
        target_states = np.asarray(target_elem.states)
        source_states = np.asarray(source_elem.states)
        target_mask = np.asarray(target_elem.mask)
        source_mask = np.asarray(source_elem.mask)
        if index >= len(target_states) or index >= len(source_states):
            raise IndexError(
                f"Protected slot {index} is outside decoded/template capacity"
            )
        width = min(target_states.shape[-1], source_states.shape[-1])
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
        scales = np.asarray(
            [1.0, 1.0, 0.5, 0.25, 0.25],
            dtype=np.float32,
        )[:width]
        errors = np.linalg.norm(
            (states[valid, :width] - target[:width]) * scales,
            axis=1,
        )
        return int(valid[int(np.argmin(errors))])
