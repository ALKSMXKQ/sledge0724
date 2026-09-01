"""Occlusion-aware half-denoise runner with three experimental modes.

Modes:

``raw_diffusion_baseline``
    Diffusion output is evaluated without restoring B1 hazard objects.

``semantic_protected``
    Historical hard-copy control: B1 road, ego, controlled pedestrian and
    occluder are copied back after diffusion.

``topology_adaptive``
    Diffusion generates road/ego/background. Only the hazard semantics are
    retained, then pedestrian and occluder geometry are re-projected against
    the generated local road context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import (
    sledge_raw_feature_processing,
)
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeVector,
    SledgeVectorElement,
)
from sledge.common.visualization.sledge_visualization_utils import (
    get_sledge_vector_as_raster,
)
from sledge.script.run_half_denoise_from_tiered_cache import (
    MultiScenarioHalfDenoiseRunner,
    basic_scene_compliance,
    build_raster_diff_mask,
    encode_raster,
    resolve_map_id,
    save_image,
    summarize_multiscenario_semantics,
)
from sledge.semantic_control.io import (
    feature_to_raw_scene_dict,
    load_raw_scene,
    save_gz_pickle,
    save_json,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.metrics import (
    evaluate_occluded_pedestrian_scene,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.diffusion_modes import (
    RAW_DIFFUSION_BASELINE,
    SEMANTIC_PROTECTED,
    SUPPORTED_DIFFUSION_MODES,
    TOPOLOGY_ADAPTIVE,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.geometry_metrics import (
    estimate_ego_speed,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_spec import (
    HazardSemanticSpec,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.topology_adaptive_projection import (
    TopologyAdaptiveHazardProjector,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.object_types import (
    embed_type_overrides,
    make_type_override,
    tracked_object_type_name,
)


class OccludedPedestrianHalfDenoiseRunner(MultiScenarioHalfDenoiseRunner):
    """Run raw, hard-protected, or topology-adaptive diffusion."""

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
        self._active_hazard_spec: Optional[HazardSemanticSpec] = None
        self._adaptive_topology_locked = False
        self._adaptive_projector = TopologyAdaptiveHazardProjector(
            projection_time_s=2.1
        )
        self._adaptive_projection_reports: Dict[int, Dict[str, Any]] = {}

    @property
    def semantic_compositing_enabled(self) -> bool:
        return self.diffusion_mode == SEMANTIC_PROTECTED

    @property
    def topology_adaptive_enabled(self) -> bool:
        return self.diffusion_mode == TOPOLOGY_ADAPTIVE

    def run_one(
        self,
        edited_scene_path: Path,
        out_dir: Path,
        index: int,
    ) -> Dict[str, object]:
        if self.topology_adaptive_enabled:
            return self._run_topology_adaptive_one(
                edited_scene_path,
                out_dir,
                index,
            )

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
            summary = super().run_one(
                edited_scene_path,
                out_dir,
                index,
            )
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
            label_path = (
                Path(str(vector_path)).parent
                / "scenario_label.json"
            )
            if label_path.exists():
                with label_path.open(
                    "r",
                    encoding="utf-8",
                ) as stream:
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

    def _run_topology_adaptive_one(
        self,
        edited_scene_path: Path,
        out_dir: Path,
        index: int,
    ) -> Dict[str, object]:
        """Run unconstrained diffusion then re-project hazard semantics."""

        out_dir.mkdir(parents=True, exist_ok=True)
        rel = edited_scene_path.relative_to(self.edited_dir)
        original_scene_path = self.original_dir / rel
        if not original_scene_path.exists():
            raise FileNotFoundError(
                f"Cannot find paired original scene: {original_scene_path}"
            )

        prompt_spec, prompt, scenario_meta = self._load_prompt_spec(
            edited_scene_path.parent
        )
        map_id = resolve_map_id(
            edited_scene_path,
            self.args.map_id,
            getattr(prompt_spec, "map_id", None),
        )
        original_raw, _ = load_raw_scene(original_scene_path)
        edited_raw, source_format = load_raw_scene(
            edited_scene_path
        )
        _, original_raster = sledge_raw_feature_processing(
            original_raw,
            self.ae_config,
        )
        template_vector, edited_raster = (
            sledge_raw_feature_processing(
                edited_raw,
                self.ae_config,
            )
        )

        self._active_template = template_vector
        self._active_hazard_spec = self._load_hazard_spec(
            edited_scene_path,
            scenario_meta,
        )
        self._adaptive_topology_locked = (
            self._prompt_locks_topology(scenario_meta)
        )
        self._adaptive_projection_reports = {}

        if hasattr(self.alignment_evaluator, "set_reference_scene"):
            self.alignment_evaluator.set_reference_scene(
                template_vector
                if self._adaptive_topology_locked
                else None
            )
        if hasattr(self.alignment_evaluator, "set_preferred_slots"):
            self.alignment_evaluator.set_preferred_slots(
                -1,
                -1,
                "vehicles",
            )
        if hasattr(self.alignment_evaluator, "set_lane_center_y"):
            self.alignment_evaluator.set_lane_center_y(
                float(
                    scenario_meta.get(
                        "semantic_lane_center_y",
                        0.0,
                    )
                )
            )

        edited_alignment = self.alignment_evaluator.evaluate(
            template_vector,
            prompt_spec,
        )
        edited_semantic = summarize_multiscenario_semantics(
            edited_alignment,
            prompt_spec,
            template_vector,
            self.args.alignment_threshold,
        )

        init_latents = encode_raster(
            self.autoencoder_model,
            edited_raster,
            self.args.device,
        )
        shape_mask = build_raster_diff_mask(
            original_raster=original_raster,
            edited_raster=edited_raster,
            latent_shape=init_latents.shape,
            device=self.args.device,
            diff_threshold=self.args.diff_threshold,
            dilation=0,
        )
        preserve_mask = torch.zeros_like(shape_mask)

        candidate_rows = []
        valid_repairs = []
        try:
            for attempt_idx in range(
                max(1, int(self.args.repair_attempts))
            ):
                try:
                    (
                        repaired_vector,
                        final_latents,
                        used_start_idx,
                    ) = self._attempt_repair(
                        init_latents=init_latents,
                        preserve_mask=preserve_mask,
                        map_id=map_id,
                        attempt_idx=attempt_idx,
                        scene_index=index,
                    )
                    projection = dict(
                        self._adaptive_projection_reports.get(
                            attempt_idx,
                            {},
                        )
                    )
                    slots = dict(
                        projection.get("projected_slots", {})
                        or {}
                    )
                    lane_center_y = float(
                        projection.get("road_context", {}).get(
                            "lane_center_y",
                            0.0,
                        )
                    )

                    repaired_alignment = (
                        self.alignment_evaluator.evaluate(
                            repaired_vector,
                            prompt_spec,
                        )
                    )
                    repaired_semantic = (
                        summarize_multiscenario_semantics(
                            repaired_alignment,
                            prompt_spec,
                            repaired_vector,
                            self.args.alignment_threshold,
                        )
                    )
                    strict_metrics = (
                        evaluate_occluded_pedestrian_scene(
                            repaired_vector,
                            self._active_hazard_spec,
                            preferred_pedestrian_index=(
                                slots.get("pedestrians")
                            ),
                            preferred_occluder_index=(
                                slots.get("occluder_index")
                            ),
                            preferred_occluder_elem_name=str(
                                slots.get(
                                    "occluder_element",
                                    "vehicles",
                                )
                            ),
                            projection_time_s=float(
                                projection.get(
                                    "semantic_projection_time_s",
                                    2.1,
                                )
                            ),
                            lane_center_y=lane_center_y,
                        )
                    )
                    sim_vector = (
                        self._make_generated_ego_sim_vector(
                            repaired_vector
                        )
                    )
                    compliance = basic_scene_compliance(
                        sim_vector
                    )
                    semantic_ok = bool(
                        repaired_semantic.get(
                            "semantic_pass",
                            False,
                        )
                        and strict_metrics.get(
                            "semantic_pass",
                            False,
                        )
                    )
                    traffic_ok = bool(
                        strict_metrics.get(
                            "traffic_realism_pass",
                            False,
                        )
                    )
                    compliance_ok = bool(
                        compliance.get("compliant", False)
                    )
                    rank_score = (
                        60.0 * float(semantic_ok)
                        + 30.0 * float(traffic_ok)
                        + 5.0 * float(compliance_ok)
                        + 3.0
                        * float(
                            strict_metrics.get(
                                "semantic_satisfaction_rate",
                                0.0,
                            )
                        )
                        + 2.0
                        * float(
                            strict_metrics.get(
                                "traffic_realism_rate",
                                0.0,
                            )
                        )
                    )
                    row = {
                        "source": (
                            f"repair_attempt_{attempt_idx:03d}"
                        ),
                        "alignment_total": float(
                            repaired_alignment.total
                        ),
                        "semantic_summary": repaired_semantic,
                        "strict_metrics": strict_metrics,
                        "traffic_realism_pass": traffic_ok,
                        "compliance": compliance,
                        "used_start_timestep_index": int(
                            used_start_idx
                        ),
                        "rank_score": float(rank_score),
                        "projection": projection,
                    }
                    candidate_rows.append(row)
                    if semantic_ok and traffic_ok and compliance_ok:
                        valid_repairs.append(
                            {
                                **row,
                                "vector": sim_vector,
                                "processed_vector": (
                                    repaired_vector
                                ),
                                "final_latents": (
                                    final_latents.detach().cpu()
                                ),
                            }
                        )
                except Exception as exc:
                    candidate_rows.append(
                        {
                            "source": (
                                f"repair_attempt_{attempt_idx:03d}"
                            ),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "rank_score": -1.0,
                        }
                    )

            best = (
                max(
                    valid_repairs,
                    key=lambda row: float(row["rank_score"]),
                )
                if valid_repairs
                else None
            )

            save_json(
                out_dir / "edited_prompt_alignment.json",
                {
                    **edited_alignment.to_dict(),
                    **edited_semantic,
                    "prompt": prompt,
                    "scenario_meta": scenario_meta,
                    "topology_policy": (
                        "explicit_prompt_lock"
                        if self._adaptive_topology_locked
                        else "diffusion_free"
                    ),
                },
            )
            save_json(
                out_dir / "candidate_scores.json",
                candidate_rows,
            )

            scenario_vector_path = None
            if best is not None:
                scenario_cache_dir = self._scenario_cache_dir(
                    edited_scene_path,
                    prompt_spec,
                    index,
                )
                scenario_cache_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                scenario_vector_path = save_gz_pickle(
                    scenario_cache_dir / "sledge_vector",
                    feature_to_raw_scene_dict(best["vector"]),
                )

                projection = dict(best["projection"])
                slots = dict(
                    projection.get("projected_slots", {})
                    or {}
                )
                occ = dict(
                    projection.get("occluder", {}) or {}
                )
                canonical_type = str(
                    occ.get("canonical_type", "vehicle")
                )
                if slots.get("occluder_index") is not None:
                    type_overrides = make_type_override(
                        str(
                            slots.get(
                                "occluder_element",
                                "vehicles",
                            )
                        ),
                        int(slots["occluder_index"]),
                        tracked_object_type_name(
                            canonical_type
                        ),
                    )
                    embed_type_overrides(
                        Path(str(scenario_vector_path)),
                        type_overrides,
                    )
                else:
                    type_overrides = {}

                label = dict(scenario_meta)
                label.update(
                    {
                        "sample_id": str(
                            scenario_meta.get(
                                "sample_id",
                                edited_scene_path.parent.name,
                            )
                        ),
                        "dangerous_scenario_type": str(
                            prompt_spec.scenario_type
                        ),
                        "scenario_type": str(
                            prompt_spec.scenario_type
                        ),
                        "semantic_family": (
                            "occluded_pedestrian"
                        ),
                        "severity_level": str(
                            getattr(
                                prompt_spec,
                                "severity_level",
                                "moderate",
                            )
                        ),
                        "prompt": prompt,
                        "original_scene_path": str(
                            original_scene_path
                        ),
                        "edited_scene_path": str(
                            edited_scene_path
                        ),
                        "diffusion_mode": TOPOLOGY_ADAPTIVE,
                        "semantic_vector_compositing": False,
                        "hazard_projection_policy": (
                            "copy_semantics_recompute_geometry"
                        ),
                        "topology_policy": (
                            "explicit_prompt_lock"
                            if self._adaptive_topology_locked
                            else "diffusion_free"
                        ),
                        "road_topology_lock": (
                            "explicit_prompt_b1_lines"
                            if self._adaptive_topology_locked
                            else "none_diffusion_generated"
                        ),
                        "ego_state_lock": (
                            "none_diffusion_generated"
                        ),
                        "generated_components": [
                            (
                                "road_locked_by_prompt"
                                if self._adaptive_topology_locked
                                else "road"
                            ),
                            "ego",
                            "background_vehicles",
                            "background_pedestrians",
                        ],
                        "semantic_projection_time_s": float(
                            projection.get(
                                "semantic_projection_time_s",
                                2.1,
                            )
                        ),
                        "semantic_lane_center_y": float(
                            projection.get(
                                "road_context",
                                {},
                            ).get(
                                "lane_center_y",
                                0.0,
                            )
                        ),
                        "projected_slots": slots,
                        "protected_slots": {},
                        "hazard_projection": projection,
                        "ego_state_source": (
                            projection.get("ego_state_source")
                        ),
                        "ego_speed_mps": projection.get(
                            "ego_speed_mps"
                        ),
                        "object_type_overrides": (
                            type_overrides
                        ),
                        "selected_source": best["source"],
                        "selected_alignment_total": float(
                            best["alignment_total"]
                        ),
                        "selected_semantic_pass": bool(
                            best["strict_metrics"].get(
                                "semantic_pass",
                                False,
                            )
                        ),
                        "selected_traffic_realism_pass": bool(
                            best["strict_metrics"].get(
                                "traffic_realism_pass",
                                False,
                            )
                        ),
                        "selected_compliant": bool(
                            best["compliance"].get(
                                "compliant",
                                False,
                            )
                        ),
                        "used_start_timestep_index": int(
                            best[
                                "used_start_timestep_index"
                            ]
                        ),
                    }
                )
                save_json(
                    scenario_cache_dir / "scenario_label.json",
                    label,
                )
                save_json(
                    out_dir / "final_prompt_alignment.json",
                    {
                        "source": best["source"],
                        "alignment_total": best[
                            "alignment_total"
                        ],
                        "semantic_summary": best[
                            "semantic_summary"
                        ],
                        "strict_metrics": best[
                            "strict_metrics"
                        ],
                        "traffic_realism_pass": best[
                            "traffic_realism_pass"
                        ],
                        "compliance": best["compliance"],
                        "projection": projection,
                        "used_start_timestep_index": best[
                            "used_start_timestep_index"
                        ],
                    },
                )
                if self.args.save_visuals:
                    save_image(
                        out_dir
                        / "best_topology_adaptive_vector.png",
                        get_sledge_vector_as_raster(
                            best["processed_vector"],
                            self.ae_config,
                        ),
                    )
            else:
                save_json(
                    out_dir / "final_prompt_alignment.json",
                    {
                        "source": None,
                        "alignment_total": None,
                        "semantic_summary": None,
                        "strict_metrics": None,
                        "traffic_realism_pass": False,
                        "compliance": None,
                        "projection": None,
                        "used_start_timestep_index": None,
                    },
                )

            if self.args.save_latents:
                torch.save(
                    init_latents.detach().cpu(),
                    out_dir / "init_latents.pt",
                )
                torch.save(
                    preserve_mask.detach().cpu(),
                    out_dir / "preserve_mask.pt",
                )
                if best is not None:
                    torch.save(
                        best["final_latents"],
                        out_dir / "best_final_latents.pt",
                    )

            summary = {
                "scene_path": str(edited_scene_path),
                "original_scene_path": str(
                    original_scene_path
                ),
                "output_dir": str(out_dir),
                "scenario_cache_vector_path": (
                    str(scenario_vector_path)
                    if scenario_vector_path
                    else None
                ),
                "prompt": prompt,
                "scenario_meta": scenario_meta,
                "source_format": source_format,
                "scenario_type": str(prompt_spec.scenario_type),
                "diffusion_mode": TOPOLOGY_ADAPTIVE,
                "semantic_vector_compositing": False,
                "topology_policy": (
                    "explicit_prompt_lock"
                    if self._adaptive_topology_locked
                    else "diffusion_free"
                ),
                "repair_success": best is not None,
                "selected_source": (
                    best["source"]
                    if best is not None
                    else None
                ),
                "selected_alignment_total": (
                    float(best["alignment_total"])
                    if best is not None
                    else None
                ),
                "selected_semantic_pass": (
                    bool(
                        best["strict_metrics"].get(
                            "semantic_pass",
                            False,
                        )
                    )
                    if best is not None
                    else False
                ),
                "selected_traffic_realism_pass": (
                    bool(
                        best["strict_metrics"].get(
                            "traffic_realism_pass",
                            False,
                        )
                    )
                    if best is not None
                    else False
                ),
                "selected_compliant": (
                    bool(
                        best["compliance"].get(
                            "compliant",
                            False,
                        )
                    )
                    if best is not None
                    else False
                ),
                "used_start_timestep_index": (
                    best["used_start_timestep_index"]
                    if best is not None
                    else None
                ),
                "hazard_projection": (
                    best["projection"]
                    if best is not None
                    else None
                ),
            }
            save_json(out_dir / "summary.json", summary)
            return summary
        finally:
            self._active_template = None
            self._active_hazard_spec = None
            self._adaptive_projection_reports = {}
            self._adaptive_topology_locked = False

    def _attempt_repair(self, *args, **kwargs):
        if self.topology_adaptive_enabled:
            preserve_mask = kwargs.get("preserve_mask")
            if preserve_mask is not None:
                kwargs["preserve_mask"] = torch.zeros_like(
                    preserve_mask
                )

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
        elif self.topology_adaptive_enabled:
            if (
                self._active_template is None
                or self._active_hazard_spec is None
            ):
                raise RuntimeError(
                    "topology-adaptive repair called without "
                    "active template/spec"
                )
            if self._adaptive_topology_locked:
                vector.lines.states = np.asarray(
                    self._active_template.lines.states
                ).copy()
                vector.lines.mask = np.asarray(
                    self._active_template.lines.mask
                ).copy()
            attempt_idx = int(kwargs.get("attempt_idx", 0))
            scene_index = int(kwargs.get("scene_index", 0))
            vector, projection = self._adaptive_projector.project(
                vector,
                self._active_hazard_spec,
                attempt_seed=(
                    int(self.args.seed)
                    + scene_index * 1000
                    + attempt_idx
                ),
            )
            projection["road_topology_locked_by_prompt"] = bool(
                self._adaptive_topology_locked
            )
            self._adaptive_projection_reports[
                attempt_idx
            ] = projection
            slots = dict(
                projection.get("projected_slots", {}) or {}
            )
            if hasattr(
                self.alignment_evaluator,
                "set_preferred_slots",
            ):
                self.alignment_evaluator.set_preferred_slots(
                    int(slots.get("pedestrians", -1)),
                    int(slots.get("occluder_index", -1)),
                    str(
                        slots.get(
                            "occluder_element",
                            "vehicles",
                        )
                    ),
                )
            if hasattr(
                self.alignment_evaluator,
                "set_lane_center_y",
            ):
                self.alignment_evaluator.set_lane_center_y(
                    float(
                        projection.get(
                            "road_context",
                            {},
                        ).get(
                            "lane_center_y",
                            0.0,
                        )
                    )
                )
        return vector, final_latents, start_idx

    @staticmethod
    def _make_generated_ego_sim_vector(
        processed_vector: SledgeVector,
    ) -> SledgeVector:
        """Convert processed vector without reusing B1 ego state."""

        ego_speed = float(estimate_ego_speed(processed_vector))
        sim_ego = SledgeVectorElement(
            states=np.asarray(
                [ego_speed],
                dtype=np.float32,
            ),
            mask=np.asarray([1.0], dtype=np.float32),
        )
        return SledgeVector(
            lines=processed_vector.lines,
            vehicles=processed_vector.vehicles,
            pedestrians=processed_vector.pedestrians,
            static_objects=processed_vector.static_objects,
            green_lights=processed_vector.green_lights,
            red_lights=processed_vector.red_lights,
            ego=sim_ego,
        )

    @staticmethod
    def _load_hazard_spec(
        edited_scene_path: Path,
        scenario_meta: Dict[str, Any],
    ) -> HazardSemanticSpec:
        candidates = []
        artifact_root = scenario_meta.get("artifact_root")
        if artifact_root:
            candidates.append(
                Path(str(artifact_root))
                / "02_specification"
                / "hazard_spec.json"
            )
        sample_id = str(
            scenario_meta.get(
                "sample_id",
                edited_scene_path.parent.name,
            )
        )
        try:
            run_root = edited_scene_path.parents[2]
            candidates.append(
                run_root
                / "artifacts"
                / sample_id
                / "02_specification"
                / "hazard_spec.json"
            )
        except IndexError:
            pass
        for path in candidates:
            if path.exists():
                with path.open(
                    "r",
                    encoding="utf-8",
                ) as stream:
                    return HazardSemanticSpec.from_dict(
                        json.load(stream)
                    )
        raise FileNotFoundError(
            "Cannot locate hazard_spec.json for topology-adaptive "
            "projection; checked: "
            f"{[str(path) for path in candidates]}"
        )

    @staticmethod
    def _prompt_locks_topology(
        scenario_meta: Dict[str, Any],
    ) -> bool:
        construction = dict(
            scenario_meta.get("scene_construction", {}) or {}
        )
        explicit = construction.get(
            "explicit_global_constraints"
        )
        if isinstance(explicit, dict):
            if any(
                value not in (None, "", [], {}, False)
                for value in explicit.values()
            ):
                return True
        elif explicit:
            return True
        road_source = str(
            scenario_meta.get("road_parameter_source", "")
            or ""
        ).lower()
        return road_source in {
            "language",
            "language_explicit",
            "prompt",
            "prompt_explicit",
        }

    @staticmethod
    def _protected_slots(
        report: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "road_topology": "all_lines",
            "pedestrians": int(
                report.get("pedestrian_index", -1)
            ),
            "occluder_element": str(
                report.get(
                    "occluder_elem_name",
                    "vehicles",
                )
            ),
            "occluder_index": int(
                report.get("occluder_index", -1)
            ),
        }

    @staticmethod
    def _composite_protected_slots(
        vector: Any,
        template: Any,
        report: Dict[str, Any],
    ) -> None:
        vector.lines.states = np.asarray(
            template.lines.states
        ).copy()
        vector.lines.mask = np.asarray(
            template.lines.mask
        ).copy()
        vector.ego.states = np.asarray(
            template.ego.states
        ).copy()
        vector.ego.mask = np.asarray(
            template.ego.mask
        ).copy()

        pedestrian_index = int(
            report.get("pedestrian_index", -1)
        )
        if pedestrian_index >= 0:
            OccludedPedestrianHalfDenoiseRunner._copy_slot(
                vector.pedestrians,
                template.pedestrians,
                pedestrian_index,
            )

        occluder_name = str(
            report.get("occluder_elem_name", "vehicles")
        )
        occluder_index = int(
            report.get("occluder_index", -1)
        )
        if (
            occluder_index >= 0
            and occluder_name
            in {"vehicles", "static_objects"}
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
        if (
            index >= len(target_states)
            or index >= len(source_states)
        ):
            raise IndexError(
                f"Protected slot {index} is outside "
                "decoded/template capacity"
            )
        width = min(
            target_states.shape[-1],
            source_states.shape[-1],
        )
        target_states[index, :width] = source_states[
            index,
            :width,
        ]
        target_mask.reshape(-1)[index] = (
            source_mask.reshape(-1)[index]
        )

    @staticmethod
    def _resolve_processed_slots(
        raw: Any,
        vector: Any,
        report: Dict[str, Any],
    ) -> Dict[str, Any]:
        resolved = dict(report)
        raw_pedestrian_index = int(
            report.get("pedestrian_index", -1)
        )
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
        raw_occluder_index = int(
            report.get("occluder_index", -1)
        )
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
        masks = np.asarray(
            vector_elem.mask
        ).reshape(-1) >= 0.3
        valid = np.where(masks)[0]
        if not len(valid):
            return -1
        width = min(
            5,
            states.shape[-1],
            target.shape[-1],
        )
        scales = np.asarray(
            [1.0, 1.0, 0.5, 0.25, 0.25],
            dtype=np.float32,
        )[:width]
        errors = np.linalg.norm(
            (
                states[valid, :width]
                - target[:width]
            )
            * scales,
            axis=1,
        )
        return int(valid[int(np.argmin(errors))])
