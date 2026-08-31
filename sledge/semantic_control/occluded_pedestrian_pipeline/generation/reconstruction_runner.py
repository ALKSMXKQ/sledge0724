"""RVAE reconstruction checkpoint for occluded-pedestrian experiments.

The diffusion path already encodes the edited B1 raster before denoising, but
historically that deterministic RVAE encode->decode result was not persisted.
This module makes the bottleneck observable and keeps two scientifically
separate products:

* ``raw_cache``: the RVAE model's own deterministic reconstruction (latent mu).
* ``semantic_protected_cache``: the same reconstruction after projecting the
  B1 road/ego and controlled pedestrian/occluder slots back into the decoded
  vector. This simulator-ready product additionally sanitizes unrelated decoded
  background actors and must pass the full generated-background realism gate.

Both products are normal SLEDGE ``sledge_vector.gz`` caches and are round-trip
opened with ``SledgeScenario`` before they are reported as valid.
"""

from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle
from omegaconf import OmegaConf

from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig
from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import (
    sledge_raw_feature_processing,
)
from sledge.script.builders.model_builder import build_autoencoder_torch_module_wrapper
from sledge.script.run_half_denoise_from_tiered_cache import (
    basic_scene_compliance,
    encode_raster,
    make_simulation_compatible_vector,
)
from sledge.semantic_control.io import load_raw_scene, save_json
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.metrics import (
    evaluate_occluded_pedestrian_scene,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_spec import (
    HazardSemanticSpec,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.refinement_runner import (
    OccludedPedestrianHalfDenoiseRunner,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.object_types import (
    embed_type_overrides,
    make_type_override,
)
from sledge.simulation.scenarios.sledge_scenario.sledge_scenario import SledgeScenario


RAW_RECONSTRUCTION = "raw"
SEMANTIC_PROTECTED_RECONSTRUCTION = "semantic_protected"


class RVAEReconstructionLayout:
    """Filesystem contract for the RVAE checkpoint."""

    def __init__(self, run_root: Path) -> None:
        self.run_root = Path(run_root).resolve()
        self.root = self.run_root / "rvae_reconstruction"
        self.raw_cache = self.root / "raw_cache"
        self.protected_cache = self.root / "semantic_protected_cache"
        self.reports = self.root / "reports"
        self.manifests = self.run_root / "manifests"

    def ensure(self) -> None:
        for path in (
            self.raw_cache,
            self.protected_cache,
            self.reports,
            self.manifests,
        ):
            path.mkdir(parents=True, exist_ok=True)


class OccludedPedestrianRVAEReconstructor:
    """Persist and evaluate the actual RVAE bottleneck reconstruction."""

    def __init__(
        self,
        *,
        run_root: Path,
        config: Path,
        autoencoder_checkpoint: Path,
        device: str,
        max_scenes: Optional[int] = None,
        strict_protected: bool = True,
    ) -> None:
        self.layout = RVAEReconstructionLayout(run_root)
        self.layout.ensure()
        self.device = str(device)
        self.max_scenes = max_scenes
        self.strict_protected = bool(strict_protected)

        self.cfg = OmegaConf.load(str(config))
        self.cfg.autoencoder_checkpoint = str(autoencoder_checkpoint)
        ae_cfg_dict = OmegaConf.to_container(
            self.cfg.autoencoder_model.config,
            resolve=True,
        )
        if not isinstance(ae_cfg_dict, dict):
            raise TypeError(
                "Expected autoencoder_model.config to resolve to a dict, "
                f"got {type(ae_cfg_dict)}"
            )
        filtered = {
            key: value
            for key, value in ae_cfg_dict.items()
            if key in RVAEConfig.__annotations__
        }
        self.ae_config = RVAEConfig(**filtered)
        self.autoencoder_model = build_autoencoder_torch_module_wrapper(self.cfg)
        if hasattr(self.autoencoder_model, "eval"):
            self.autoencoder_model.eval()
        self.decoder = self.autoencoder_model.get_decoder().to(self.device)
        self.decoder.eval()
        self.feature_store = FeatureCachePickle()

    def run(self) -> Dict[str, Any]:
        cases = self._accepted_cases()
        if self.max_scenes is not None:
            cases = cases[: int(self.max_scenes)]
        if not cases:
            raise RuntimeError("No accepted B1 scenes are available for RVAE reconstruction")

        rows: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        for index, case in enumerate(cases, start=1):
            sample_id = str(case["sample_id"])
            try:
                row = self._run_one(case)
                rows.append(row)
                print(
                    f"[RVAE {index}/{len(cases)}] {sample_id}: "
                    f"raw_pass={row['raw_semantic_pass']} "
                    f"protected_pass={row['protected_semantic_pass']}"
                )
            except Exception as exc:
                failure = {
                    "sample_id": sample_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                save_json(self.layout.reports / f"{sample_id}.error.json", failure)
                print(
                    f"[RVAE {index}/{len(cases)}] {sample_id}: "
                    f"FAILED {type(exc).__name__}: {exc}"
                )

        _write_csv(self.layout.manifests / "rvae_reconstruction.csv", rows)
        save_json(
            self.layout.manifests / "rvae_reconstruction_failures.json",
            failures,
        )
        summary = {
            "schema_version": "occluded_pedestrian_rvae_reconstruction_v2_stage_aware_realism",
            "num_accepted_b1": len(cases),
            "num_reconstructed": len(rows),
            "num_failures": len(failures),
            "raw_semantic_pass_count": sum(
                bool(row["raw_semantic_pass"]) for row in rows
            ),
            "protected_semantic_pass_count": sum(
                bool(row["protected_semantic_pass"]) for row in rows
            ),
            "protected_background_realism_pass_count": sum(
                bool(row.get("protected_background_realism_pass", False))
                for row in rows
            ),
            "raw_gzip_round_trip_pass_count": sum(
                bool(row["raw_gzip_round_trip_pass"]) for row in rows
            ),
            "protected_gzip_round_trip_pass_count": sum(
                bool(row["protected_gzip_round_trip_pass"]) for row in rows
            ),
            "raw_cache": str(self.layout.raw_cache),
            "semantic_protected_cache": str(self.layout.protected_cache),
            "reports": str(self.layout.reports),
            "manifest_csv": str(
                self.layout.manifests / "rvae_reconstruction.csv"
            ),
        }
        save_json(
            self.layout.manifests / "rvae_reconstruction_summary.json",
            summary,
        )

        if self.strict_protected:
            bad = [
                row["sample_id"]
                for row in rows
                if not (
                    row["protected_semantic_pass"]
                    and row["protected_gzip_round_trip_pass"]
                )
            ]
            if failures or bad or len(rows) != len(cases):
                raise RuntimeError(
                    "RVAE semantic-protected checkpoint is incomplete. "
                    f"failures={[row['sample_id'] for row in failures]}, "
                    f"semantic_or_gzip_failures={bad}"
                )
        return summary

    def _accepted_cases(self) -> List[Dict[str, Any]]:
        cases_path = self.layout.run_root / "manifests/cases.jsonl"
        if not cases_path.exists():
            raise FileNotFoundError(
                f"Missing B1 case manifest: {cases_path}. Run batch editing first."
            )
        cases = _read_jsonl(cases_path)
        accepted: List[Dict[str, Any]] = []
        for case in cases:
            sample_id = str(case["sample_id"])
            label_path = (
                self.layout.run_root
                / "b1_edited_cache"
                / sample_id
                / "scenario_label.json"
            )
            if not label_path.exists():
                continue
            if bool(_read_json(label_path).get("accepted", False)):
                accepted.append(case)
        return accepted

    def _run_one(self, case: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = str(case["sample_id"])
        b1_dir = self.layout.run_root / "b1_edited_cache" / sample_id
        raw_scene, _ = load_raw_scene(b1_dir / "sledge_raw.gz")
        source_label = _read_json(b1_dir / "scenario_label.json")
        edit_report = _read_json(b1_dir / "edit_report.json")
        spec = HazardSemanticSpec.from_dict(
            _read_json(
                self.layout.run_root
                / "artifacts"
                / sample_id
                / "02_specification/hazard_spec.json"
            )
        )

        b1_vector, b1_raster = sledge_raw_feature_processing(
            raw_scene,
            self.ae_config,
        )
        processed_report = (
            OccludedPedestrianHalfDenoiseRunner._resolve_processed_slots(
                raw_scene,
                b1_vector,
                edit_report,
            )
        )
        ped_index = int(processed_report.get("pedestrian_index", -1))
        occ_index = int(processed_report.get("occluder_index", -1))
        occ_element = str(
            processed_report.get("occluder_elem_name", "vehicles")
        )
        if ped_index < 0 or occ_index < 0:
            raise RuntimeError(
                "Controlled B1 object could not be resolved after feature processing: "
                f"pedestrian={ped_index}, occluder={occ_element}[{occ_index}]"
            )

        latent_mu = encode_raster(
            self.autoencoder_model,
            b1_raster,
            self.device,
        )
        with torch.no_grad():
            decoded = self.decoder(latent_mu)
        raw_reconstruction = decoded.torch_to_numpy(apply_sigmoid=True)
        protected_reconstruction = deepcopy(raw_reconstruction)
        protected_realism_sanitization = (
            OccludedPedestrianHalfDenoiseRunner._composite_protected_slots(
                protected_reconstruction,
                b1_vector,
                processed_report,
            )
        )

        raw_sim = make_simulation_compatible_vector(
            raw_reconstruction,
            raw_scene,
        )
        protected_sim = make_simulation_compatible_vector(
            protected_reconstruction,
            raw_scene,
        )
        projection_time_s = float(
            source_label.get("semantic_projection_time_s", 0.0)
        )
        lane_center_y = float(
            source_label.get("semantic_lane_center_y", 0.0)
        )

        # Raw reconstruction is diagnostic: keep generated-background realism
        # visible in the report, but do not let it redefine raw hazard retention.
        raw_metrics = evaluate_occluded_pedestrian_scene(
            raw_sim,
            spec,
            projection_time_s=projection_time_s,
            lane_center_y=lane_center_y,
            require_background_realism=False,
        )
        # Protected reconstruction is simulator-ready and therefore must pass
        # both controlled-hazard realism and generated-background realism after
        # the sanitizer has run.
        protected_metrics = evaluate_occluded_pedestrian_scene(
            protected_sim,
            spec,
            preferred_pedestrian_index=ped_index,
            preferred_occluder_index=occ_index,
            preferred_occluder_elem_name=occ_element,
            projection_time_s=projection_time_s,
            lane_center_y=lane_center_y,
            require_background_realism=True,
        )
        raw_compliance = basic_scene_compliance(raw_sim)
        protected_compliance = basic_scene_compliance(protected_sim)

        tracked_type_name = str(
            source_label.get("occluder_tracked_object_type", "VEHICLE")
        )
        raw_overrides = _overrides_from_metrics(
            raw_metrics,
            tracked_type_name,
        )
        protected_overrides = make_type_override(
            occ_element,
            occ_index,
            tracked_type_name,
        )

        raw_out = (
            self.layout.raw_cache
            / "log"
            / "sudden_pedestrian_crossing"
            / sample_id
        )
        protected_out = (
            self.layout.protected_cache
            / "log"
            / "sudden_pedestrian_crossing"
            / sample_id
        )
        raw_vector_path, raw_round_trip = self._store_vector(
            raw_out,
            raw_sim,
            raw_overrides,
        )

        protected_semantic_pass = bool(protected_metrics.get("overall_pass", False))
        protected_compliance_pass = bool(protected_compliance.get("compliant", False))
        if self.strict_protected and not (
            protected_semantic_pass and protected_compliance_pass
        ):
            save_json(
                self.layout.reports / f"{sample_id}.json",
                {
                    "sample_id": sample_id,
                    "raw_metrics": raw_metrics,
                    "protected_metrics": protected_metrics,
                    "raw_compliance": raw_compliance,
                    "protected_compliance": protected_compliance,
                    "protected_realism_sanitization": protected_realism_sanitization,
                    "error": "semantic-protected reconstruction failed acceptance gate",
                },
            )
            raise RuntimeError(
                "Semantic-protected RVAE reconstruction failed the canonical gate "
                f"for {sample_id}; failed_checks="
                f"{[name for name, passed in protected_metrics.get('checks', {}).items() if not passed]}"
            )

        protected_vector_path, protected_round_trip = self._store_vector(
            protected_out,
            protected_sim,
            protected_overrides,
        )

        raw_label = self._make_label(
            sample_id=sample_id,
            source_label=source_label,
            mode=RAW_RECONSTRUCTION,
            vector_path=raw_vector_path,
            metrics=raw_metrics,
            compliance=raw_compliance,
            overrides=raw_overrides,
            round_trip=raw_round_trip,
            processed_report=processed_report,
            realism_sanitization={},
        )
        protected_label = self._make_label(
            sample_id=sample_id,
            source_label=source_label,
            mode=SEMANTIC_PROTECTED_RECONSTRUCTION,
            vector_path=protected_vector_path,
            metrics=protected_metrics,
            compliance=protected_compliance,
            overrides=protected_overrides,
            round_trip=protected_round_trip,
            processed_report=processed_report,
            realism_sanitization=protected_realism_sanitization,
        )
        save_json(raw_out / "scenario_label.json", raw_label)
        save_json(protected_out / "scenario_label.json", protected_label)

        evaluation_dir = (
            self.layout.run_root
            / "artifacts"
            / sample_id
            / "04_evaluation"
        )
        save_json(evaluation_dir / "rvae_raw_metrics.json", raw_metrics)
        save_json(
            evaluation_dir / "rvae_semantic_protected_metrics.json",
            protected_metrics,
        )
        save_json(
            evaluation_dir / "rvae_protected_realism_sanitization.json",
            protected_realism_sanitization,
        )
        report = {
            "schema_version": "occluded_pedestrian_rvae_scene_report_v2_stage_aware_realism",
            "sample_id": sample_id,
            "latent_policy": "encoder_mu_deterministic",
            "raw": raw_label,
            "semantic_protected": protected_label,
            "raw_metrics": raw_metrics,
            "protected_metrics": protected_metrics,
            "raw_compliance": raw_compliance,
            "protected_compliance": protected_compliance,
            "protected_realism_sanitization": protected_realism_sanitization,
            "protected_slots": {
                "road_topology": "all_lines",
                "pedestrians": ped_index,
                "occluder_element": occ_element,
                "occluder_index": occ_index,
            },
        }
        save_json(self.layout.reports / f"{sample_id}.json", report)

        return {
            "sample_id": sample_id,
            "raw_vector_gz": str(raw_vector_path),
            "protected_vector_gz": str(protected_vector_path),
            "raw_semantic_pass": bool(raw_metrics.get("overall_pass", False)),
            "protected_semantic_pass": protected_semantic_pass,
            "raw_semantic_satisfaction_rate": float(
                raw_metrics.get("semantic_satisfaction_rate", 0.0)
            ),
            "protected_semantic_satisfaction_rate": float(
                protected_metrics.get("semantic_satisfaction_rate", 0.0)
            ),
            "protected_background_realism_pass": bool(
                protected_metrics.get("background_realism_pass", False)
            ),
            "raw_gzip_round_trip_pass": bool(raw_round_trip),
            "protected_gzip_round_trip_pass": bool(protected_round_trip),
            "raw_compliance_pass": bool(raw_compliance.get("compliant", False)),
            "protected_compliance_pass": protected_compliance_pass,
            "pedestrian_index": ped_index,
            "occluder_element": occ_element,
            "occluder_index": occ_index,
            "occluder_tracked_object_type": tracked_type_name,
        }

    def _store_vector(
        self,
        out_dir: Path,
        vector: Any,
        overrides: Mapping[str, Mapping[str, str]],
    ) -> Tuple[Path, bool]:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.feature_store.store_computed_feature_to_folder(
            out_dir / "sledge_vector",
            vector,
        )
        vector_path = out_dir / "sledge_vector.gz"
        if overrides:
            embed_type_overrides(vector_path, overrides)
        self._assert_sledge_scenario_round_trip(
            out_dir / "sledge_vector",
            overrides,
        )
        return vector_path, True

    @staticmethod
    def _assert_sledge_scenario_round_trip(
        cache_base: Path,
        overrides: Mapping[str, Mapping[str, str]],
    ) -> None:
        scenario = SledgeScenario(cache_base)
        detections = scenario.initial_tracked_objects.tracked_objects
        by_token = {obj.track_token: obj for obj in detections}
        for entries in overrides.values():
            for index_text, type_name in entries.items():
                expected_type = TrackedObjectType[str(type_name)]
                expected_token = f"{expected_type.value}_{int(index_text)}"
                obj = by_token.get(expected_token)
                if obj is None or obj.tracked_object_type != expected_type:
                    observed = [
                        (item.track_token, item.tracked_object_type.name)
                        for item in detections
                    ]
                    raise RuntimeError(
                        "SledgeScenario gzip round-trip/type check failed: "
                        f"expected={(expected_token, expected_type.name)}, "
                        f"observed={observed}"
                    )

    @staticmethod
    def _make_label(
        *,
        sample_id: str,
        source_label: Dict[str, Any],
        mode: str,
        vector_path: Path,
        metrics: Dict[str, Any],
        compliance: Dict[str, Any],
        overrides: Mapping[str, Mapping[str, str]],
        round_trip: bool,
        processed_report: Dict[str, Any],
        realism_sanitization: Mapping[str, Any],
    ) -> Dict[str, Any]:
        ped_index = int(processed_report.get("pedestrian_index", -1))
        occ_index = int(processed_report.get("occluder_index", -1))
        occ_element = str(
            processed_report.get("occluder_elem_name", "vehicles")
        )
        return {
            "schema_version": "occluded_pedestrian_rvae_cache_label_v2_stage_aware_realism",
            "sample_id": sample_id,
            "scenario_type": "sudden_pedestrian_crossing",
            "semantic_family": "occluded_pedestrian",
            "stage": "RVAE_RECONSTRUCTION",
            "reconstruction_mode": mode,
            "latent_policy": "encoder_mu_deterministic",
            "semantic_vector_compositing": (
                mode == SEMANTIC_PROTECTED_RECONSTRUCTION
            ),
            "prompt": str(source_label.get("prompt", "")),
            "severity_level": str(
                source_label.get("severity_level", "moderate")
            ),
            "semantic_lane_center_y": float(
                source_label.get("semantic_lane_center_y", 0.0)
            ),
            "semantic_projection_time_s": float(
                source_label.get("semantic_projection_time_s", 0.0)
            ),
            "semantic_pass": bool(metrics.get("overall_pass", False)),
            "danger_semantic_pass": bool(
                metrics.get("danger_semantic_pass", False)
            ),
            "controlled_traffic_realism_pass": bool(
                metrics.get("controlled_traffic_realism_pass", False)
            ),
            "background_realism_pass": bool(
                metrics.get("background_realism_pass", False)
            ),
            "background_realism_required": bool(
                metrics.get("background_realism_required", False)
            ),
            "semantic_satisfaction_rate": float(
                metrics.get("semantic_satisfaction_rate", 0.0)
            ),
            "compliance_pass": bool(compliance.get("compliant", False)),
            "gzip_round_trip_pass": bool(round_trip),
            "vector_gz": str(vector_path),
            "pedestrian_index": ped_index,
            "occluder_element": occ_element,
            "occluder_index": occ_index,
            "occluder_tracked_object_type": str(
                source_label.get("occluder_tracked_object_type", "VEHICLE")
            ),
            "object_type_overrides": dict(overrides),
            "traffic_realism_sanitization": dict(realism_sanitization),
            "protected_slots": (
                {
                    "road_topology": "all_lines",
                    "pedestrians": ped_index,
                    "occluder_element": occ_element,
                    "occluder_index": occ_index,
                }
                if mode == SEMANTIC_PROTECTED_RECONSTRUCTION
                else {}
            ),
        }


def run_rvae_reconstruction(
    *,
    run_root: Path,
    config: Path,
    autoencoder_checkpoint: Path,
    device: str,
    max_scenes: Optional[int] = None,
    strict_protected: bool = True,
) -> Dict[str, Any]:
    """Public entry point used by the three-stage experiment runner."""

    return OccludedPedestrianRVAEReconstructor(
        run_root=run_root,
        config=config,
        autoencoder_checkpoint=autoencoder_checkpoint,
        device=device,
        max_scenes=max_scenes,
        strict_protected=strict_protected,
    ).run()


def _overrides_from_metrics(
    metrics: Mapping[str, Any],
    tracked_type_name: str,
) -> Dict[str, Dict[str, str]]:
    occluder = dict(metrics.get("occluder", {}) or {})
    element = occluder.get("element")
    index = int(occluder.get("index", -1))
    if element not in {"vehicles", "static_objects"} or index < 0:
        return {}
    return make_type_override(
        str(element),
        index,
        tracked_type_name,
    )


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )
