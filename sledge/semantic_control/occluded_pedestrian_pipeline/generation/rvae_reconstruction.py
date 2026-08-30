"""Batch RVAE reconstruction with a fail-closed semantic contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf
import torch

from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig
from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import (
    sledge_raw_feature_processing,
)
from sledge.script.builders.model_builder import (
    build_autoencoder_torch_module_wrapper,
)
from sledge.semantic_control.io import (
    feature_to_raw_scene_dict,
    load_raw_scene,
    save_gz_pickle,
    save_json,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.metrics import (
    aggregate_stage_metrics,
    evaluate_occluded_pedestrian_scene,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_spec import (
    HazardSemanticSpec,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.semantic_protection import (
    audit_protected_semantics,
    composite_protected_semantics,
    make_simulation_compatible_vector,
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


def run_rvae_reconstruction(
    *,
    run_root: Path,
    config: Path,
    autoencoder_checkpoint: Path,
    device: str,
    max_scenes: Optional[int] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Encode/decode accepted B1 scenes and save protected simulation caches.

    Both the raw decoder candidate and the protected accepted reconstruction are
    retained as gzip files.  This keeps model behavior auditable while ensuring
    that only semantically valid scenes enter the simulator-facing cache.
    """

    run_root = Path(run_root).resolve()
    cfg = OmegaConf.load(config)
    cfg.autoencoder_checkpoint = str(Path(autoencoder_checkpoint).resolve())
    ae_cfg_dict = OmegaConf.to_container(
        cfg.autoencoder_model.config,
        resolve=True,
    )
    if not isinstance(ae_cfg_dict, dict):
        raise TypeError(
            "autoencoder_model.config must resolve to a dictionary"
        )
    ae_config = RVAEConfig(
        **{
            key: value
            for key, value in ae_cfg_dict.items()
            if key in RVAEConfig.__annotations__
        }
    )
    model = build_autoencoder_torch_module_wrapper(cfg)
    if hasattr(model, "eval"):
        model.eval()
    encoder = model.get_encoder().to(device)
    decoder = model.get_decoder().to(device)
    encoder.eval()
    decoder.eval()

    cases = _accepted_cases(run_root)
    if max_scenes is not None:
        cases = cases[: int(max_scenes)]

    candidate_root = run_root / "rvae_reconstruction/candidate_cache"
    accepted_root = run_root / "rvae_reconstruction/generated_cache"
    report_root = run_root / "rvae_reconstruction/reports"
    rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        sample_id = str(case["sample_id"])
        try:
            row, metrics = _reconstruct_one(
                run_root=run_root,
                sample_id=sample_id,
                case=case,
                ae_config=ae_config,
                encoder=encoder,
                decoder=decoder,
                device=device,
                candidate_root=candidate_root,
                accepted_root=accepted_root,
                report_root=report_root,
            )
            rows.append(row)
            metric_rows.append(metrics)
            print(
                f"[{index}/{len(cases)}] RVAE reconstructed: "
                f"{sample_id} semantic_pass={metrics['overall_pass']}"
            )
        except Exception as exc:
            failure = {
                "sample_id": sample_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            metric_rows.append(
                {
                    "sample_id": sample_id,
                    "overall_pass": False,
                    "semantic_satisfaction_rate": 0.0,
                    "checks": {},
                    "error": str(exc),
                }
            )
            save_json(report_root / sample_id / "error.json", failure)

    summary = {
        "schema_version": "occluded_pedestrian_rvae_reconstruction_v1",
        "num_expected": len(cases),
        "num_generated": len(rows),
        "num_failed": len(failures),
        "all_semantics_preserved": bool(
            len(rows) == len(cases)
            and all(row["semantic_contract_pass"] for row in rows)
        ),
        "stage_metrics": aggregate_stage_metrics(
            metric_rows,
            "R1_RVAE_reconstruction",
        ),
        "candidate_cache_root": str(candidate_root),
        "generated_cache_root": str(accepted_root),
        "reports_root": str(report_root),
        "rows": rows,
        "failures": failures,
    }
    save_json(
        run_root / "manifests/rvae_reconstruction_summary.json",
        summary,
    )
    _write_jsonl(
        run_root / "manifests/rvae_reconstruction_results.jsonl",
        rows,
    )
    if strict and not summary["all_semantics_preserved"]:
        raise RuntimeError(
            "RVAE reconstruction failed the semantic contract: "
            f"generated={len(rows)}/{len(cases)}, failures={len(failures)}"
        )
    return summary


def _reconstruct_one(
    *,
    run_root: Path,
    sample_id: str,
    case: Dict[str, Any],
    ae_config: RVAEConfig,
    encoder: Any,
    decoder: Any,
    device: str,
    candidate_root: Path,
    accepted_root: Path,
    report_root: Path,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    source_dir = run_root / "b1_edited_cache" / sample_id
    raw_scene, _ = load_raw_scene(source_dir / "sledge_raw.gz")
    source_label = _read_json(source_dir / "scenario_label.json")
    edit_report = _read_json(source_dir / "edit_report.json")
    spec = HazardSemanticSpec.from_dict(
        _read_json(
            run_root
            / "artifacts"
            / sample_id
            / "02_specification/hazard_spec.json"
        )
    )
    template_vector, raster = sledge_raw_feature_processing(
        raw_scene,
        ae_config,
    )
    processed_report = resolve_processed_slots(
        raw_scene,
        template_vector,
        edit_report,
    )
    raster_tensor = (
        raster.to_feature_tensor().data.unsqueeze(0).to(device)
    )
    with torch.no_grad():
        latent = encoder(raster_tensor).mu
        decoded = decoder.decode(latent).torch_to_numpy(
            apply_sigmoid=True
        )

    template_sim = make_simulation_compatible_vector(
        template_vector,
        raw_scene,
    )
    candidate_sim = make_simulation_compatible_vector(
        decoded,
        raw_scene,
    )
    candidate_dir = (
        candidate_root
        / "log"
        / "sudden_pedestrian_crossing"
        / sample_id
    )
    candidate_path = save_gz_pickle(
        candidate_dir / "sledge_vector",
        feature_to_raw_scene_dict(candidate_sim),
    )
    raw_candidate_metrics = evaluate_occluded_pedestrian_scene(
        candidate_sim,
        spec,
        projection_time_s=float(
            source_label.get("semantic_projection_time_s", 0.0)
        ),
        lane_center_y=float(
            source_label.get("semantic_lane_center_y", 0.0)
        ),
    )
    save_json(candidate_dir / "semantic_metrics.json", raw_candidate_metrics)
    save_json(
        candidate_dir / "scenario_label.json",
        {
            "schema_version": (
                "occluded_pedestrian_rvae_raw_candidate_label_v1"
            ),
            "sample_id": sample_id,
            "stage": "R1_raw_decoder_candidate",
            "prompt": str(case.get("prompt", "")),
            "source_b1_raw": str(source_dir / "sledge_raw.gz"),
            "sledge_vector_gz": str(candidate_path),
            "latent_policy": "posterior_mean_mu",
            "semantic_protection": "none",
            "semantic_contract_pass": bool(
                raw_candidate_metrics.get("overall_pass", False)
            ),
            "simulator_acceptance": False,
            "purpose": "decoder_audit_only",
        },
    )

    composite_protected_semantics(
        candidate_sim,
        template_sim,
        processed_report,
    )
    protection_audit = audit_protected_semantics(
        candidate_sim,
        template_sim,
        processed_report,
    )
    projection_time_s = float(
        source_label.get("semantic_projection_time_s", 0.0)
    )
    lane_center_y = float(
        source_label.get("semantic_lane_center_y", 0.0)
    )
    metrics = evaluate_occluded_pedestrian_scene(
        candidate_sim,
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
        projection_time_s=projection_time_s,
        lane_center_y=lane_center_y,
    )
    semantic_contract_pass = bool(
        protection_audit["overall_pass"]
        and metrics["overall_pass"]
    )
    if not semantic_contract_pass:
        raise RuntimeError(
            "Protected RVAE reconstruction did not retain the complete "
            "occluded-pedestrian semantic contract"
        )

    accepted_dir = (
        accepted_root
        / "log"
        / "sudden_pedestrian_crossing"
        / sample_id
    )
    accepted_path = save_gz_pickle(
        accepted_dir / "sledge_vector",
        feature_to_raw_scene_dict(candidate_sim),
    )
    tracked_type = source_label.get("occluder_tracked_object_type")
    if not tracked_type:
        tracked_type = tracked_object_type_name(
            source_label.get(
                "occluder_type",
                case.get("occluder_type", "vehicle"),
            )
        )
    tracked_type = str(tracked_type)
    overrides = make_type_override(
        str(processed_report["occluder_elem_name"]),
        int(processed_report["occluder_index"]),
        tracked_type,
    )
    embed_type_overrides(accepted_path, overrides)
    simulator_audit = audit_simulator_roundtrip(
        accepted_path,
        overrides,
    )

    roundtrip_scene, _ = load_raw_scene(accepted_path)
    roundtrip_audit = audit_protected_semantics(
        roundtrip_scene,
        template_sim,
        processed_report,
    )
    roundtrip_metrics = evaluate_occluded_pedestrian_scene(
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
        projection_time_s=projection_time_s,
        lane_center_y=lane_center_y,
    )
    gzip_roundtrip_pass = bool(
        roundtrip_audit["overall_pass"]
        and roundtrip_metrics["overall_pass"]
        and read_type_overrides(accepted_path) == overrides
        and simulator_audit["overall_pass"]
    )
    if not gzip_roundtrip_pass:
        raise RuntimeError(
            "RVAE reconstruction gzip round-trip failed semantic/type checks"
        )

    report_dir = report_root / sample_id
    save_json(report_dir / "semantic_protection_audit.json", protection_audit)
    save_json(report_dir / "semantic_metrics.json", metrics)
    save_json(report_dir / "gzip_roundtrip_audit.json", roundtrip_audit)
    save_json(report_dir / "gzip_roundtrip_metrics.json", roundtrip_metrics)
    save_json(report_dir / "simulator_roundtrip_audit.json", simulator_audit)
    label = {
        "schema_version": "occluded_pedestrian_rvae_reconstruction_label_v1",
        "sample_id": sample_id,
        "stage": "R1_RVAE_reconstruction",
        "prompt": str(case.get("prompt", "")),
        "source_b1_raw": str(source_dir / "sledge_raw.gz"),
        "raw_decoder_candidate_gz": str(candidate_path),
        "sledge_vector_gz": str(accepted_path),
        "latent_policy": "posterior_mean_mu",
        "semantic_protection": "hard_vector_compositing",
        "semantic_contract_pass": True,
        "gzip_roundtrip_pass": True,
        "simulator_roundtrip_pass": True,
        "protected_slots": protected_slots(processed_report),
        "object_type_overrides": overrides,
        "semantic_projection_time_s": projection_time_s,
        "semantic_lane_center_y": lane_center_y,
    }
    save_json(accepted_dir / "scenario_label.json", label)
    metrics["sample_id"] = sample_id
    return label, metrics


def _accepted_cases(run_root: Path) -> List[Dict[str, Any]]:
    cases = _read_jsonl(run_root / "manifests/cases.jsonl")
    return [
        case
        for case in cases
        if _is_accepted(run_root, str(case["sample_id"]))
    ]


def _is_accepted(run_root: Path, sample_id: str) -> bool:
    label_path = (
        run_root
        / "b1_edited_cache"
        / sample_id
        / "scenario_label.json"
    )
    return bool(
        label_path.exists()
        and _read_json(label_path).get("accepted", False)
    )


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


__all__ = ["run_rvae_reconstruction"]
