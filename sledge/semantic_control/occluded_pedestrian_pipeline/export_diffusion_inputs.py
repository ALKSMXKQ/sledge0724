"""Export accepted occluded-pedestrian B1 scenes as SLEDGE diffusion inputs.

The generated directory follows ``RVAELatentBuilderConfig.find_file_paths``::

    <output_root>/<log>/<source_scenario_type>/<sample_id>/
        rvae_latent.gz
        scenario_type.gz
        scenario_label.json
        diffusion_input_metadata.json

The exporter deliberately keeps the original five-class scenario-type label used
by the trained diffusion model.  The dangerous semantic family is retained in
``scenario_label.json`` and is evaluated separately after generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig
from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import (
    sledge_raw_feature_processing,
)
from sledge.script.builders.model_builder import build_autoencoder_torch_module_wrapper
from sledge.semantic_control.io import load_raw_scene, save_gz_pickle, save_json


SCENARIO_TYPE_TO_ID: Dict[str, int] = {
    "high_magnitude_speed": 0,
    "medium_magnitude_speed": 1,
    "traversing_intersection": 2,
    "traversing_traffic_light_intersection": 3,
    "unknown": 4,
}


def _load_autoencoder(
    config_path: Path,
    checkpoint_path: Path,
    device: str,
) -> Tuple[Any, RVAEConfig]:
    cfg = OmegaConf.load(str(config_path))
    cfg.autoencoder_checkpoint = str(checkpoint_path)

    ae_cfg_dict = OmegaConf.to_container(cfg.autoencoder_model.config, resolve=True)
    if not isinstance(ae_cfg_dict, dict):
        raise TypeError(f"Expected autoencoder_model.config to resolve to dict, got {type(ae_cfg_dict)}")
    filtered = {key: value for key, value in ae_cfg_dict.items() if key in RVAEConfig.__annotations__}
    ae_config = RVAEConfig(**filtered)

    model = build_autoencoder_torch_module_wrapper(cfg)
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model, ae_config


def _encode_scene(raw_path: Path, model: Any, ae_config: RVAEConfig, device: str) -> Tuple[np.ndarray, np.ndarray]:
    raw, _ = load_raw_scene(raw_path)
    _, raster = sledge_raw_feature_processing(raw, ae_config)
    raster_tensor = raster.to_feature_tensor().data.unsqueeze(0).to(device)
    encoder = model.get_encoder().to(device)
    encoder.eval()
    with torch.no_grad():
        latent_dist = encoder(raster_tensor)

    mu = latent_dist.mu.detach().float().cpu()
    log_var = getattr(latent_dist, "log_var", None)
    if log_var is None:
        log_var = getattr(latent_dist, "logvar", None)
    if log_var is None:
        variance = getattr(latent_dist, "variance", None)
        if variance is not None:
            log_var = torch.log(torch.clamp(variance, min=1e-12))
    if log_var is None:
        log_var = torch.zeros_like(mu)
    else:
        log_var = log_var.detach().float().cpu()

    return mu.squeeze(0).numpy().astype(np.float32), log_var.squeeze(0).numpy().astype(np.float32)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _iter_scene_records(input_root: Path, include_rejected: bool) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    for raw_path in sorted(input_root.glob("**/sledge_raw.gz")):
        label_path = raw_path.parent / "scenario_label.json"
        if not label_path.exists():
            continue
        label = _read_json(label_path)
        if not include_rejected and not bool(label.get("accepted", False)):
            continue
        yield raw_path, label


def _scenario_type_id(label: Dict[str, Any]) -> Tuple[str, int]:
    source_type = str(label.get("source_scenario_type", "unknown"))
    if source_type not in SCENARIO_TYPE_TO_ID:
        source_type = "unknown"
    return source_type, SCENARIO_TYPE_TO_ID[source_type]


def export_diffusion_inputs(
    *,
    input_root: Path,
    output_root: Path,
    config_path: Path,
    autoencoder_checkpoint: Path,
    device: str,
    include_rejected: bool = False,
    overwrite: bool = False,
    max_scenes: Optional[int] = None,
    log_name: str = "occluded_pedestrian",
) -> Dict[str, Any]:
    """Encode B1 raw scenes and write a standard SLEDGE latent cache."""

    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records = list(_iter_scene_records(input_root, include_rejected=include_rejected))
    if max_scenes is not None:
        records = records[: max(0, int(max_scenes))]
    if not records:
        raise FileNotFoundError(f"No eligible B1 scenes found under {input_root}")

    model, ae_config = _load_autoencoder(config_path, autoencoder_checkpoint, device)
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for index, (raw_path, label) in enumerate(records, start=1):
        sample_id = str(label.get("sample_id") or raw_path.parent.name)
        source_type, scenario_id = _scenario_type_id(label)
        out_dir = output_root / log_name / source_type / sample_id
        latent_path = out_dir / "rvae_latent.gz"
        label_cache_path = out_dir / "scenario_type.gz"

        if not overwrite and latent_path.exists() and label_cache_path.exists():
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_raw": str(raw_path),
                    "cache_dir": str(out_dir),
                    "source_scenario_type": source_type,
                    "scenario_type_id": scenario_id,
                    "status": "skipped_existing",
                }
            )
            continue

        try:
            mu, log_var = _encode_scene(raw_path, model, ae_config, device)
            expected_shape = (
                int(ae_config.latent_channel),
                int(ae_config.latent_frame[0]),
                int(ae_config.latent_frame[1]),
            )
            if tuple(mu.shape) != expected_shape:
                raise ValueError(f"Latent shape {tuple(mu.shape)} does not match configured {expected_shape}")
            if tuple(log_var.shape) != expected_shape:
                raise ValueError(f"log_var shape {tuple(log_var.shape)} does not match configured {expected_shape}")
            if not np.isfinite(mu).all() or not np.isfinite(log_var).all():
                raise ValueError("Latent contains NaN or infinite values")
            out_dir.mkdir(parents=True, exist_ok=True)
            save_gz_pickle(out_dir / "rvae_latent", {"mu": mu, "log_var": log_var})
            save_gz_pickle(out_dir / "scenario_type", {"id": int(scenario_id)})
            save_json(out_dir / "scenario_label.json", label)
            metadata = {
                "schema_version": "occluded_pedestrian_diffusion_input_v1",
                "sample_id": sample_id,
                "source_raw": str(raw_path),
                "source_scenario_type": source_type,
                "scenario_type_id": int(scenario_id),
                "semantic_family": str(label.get("semantic_family", "occluded_pedestrian")),
                "occluder_type": label.get("occluder_type"),
                "occluder_position": label.get("occluder_position"),
                "severity_level": label.get("severity_level"),
                "direction": label.get("direction"),
                "pedestrian_speed_mps": label.get("pedestrian_speed_mps"),
                "latent_shape": list(mu.shape),
                "latent_dtype": str(mu.dtype),
                "accepted_b1": bool(label.get("accepted", False)),
            }
            save_json(out_dir / "diffusion_input_metadata.json", metadata)
            rows.append({**metadata, "cache_dir": str(out_dir), "status": "encoded"})
            print(f"[{index}/{len(records)}] encoded {sample_id} -> {out_dir}")
        except Exception as exc:
            failure = {
                "sample_id": sample_id,
                "source_raw": str(raw_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(f"[{index}/{len(records)}] failed {sample_id}: {type(exc).__name__}: {exc}")

    manifest_path = output_root / "metadata" / "diffusion_inputs.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    failure_path = output_root / "metadata" / "diffusion_input_failures.jsonl"
    with failure_path.open("w", encoding="utf-8") as fp:
        for row in failures:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "schema_version": "occluded_pedestrian_diffusion_export_summary_v1",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_eligible": len(records),
        "num_exported_or_existing": len(rows),
        "num_failed": len(failures),
        "manifest": str(manifest_path),
        "failures": str(failure_path),
    }
    save_json(output_root / "metadata" / "summary.json", summary)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export accepted B1 scenes to SLEDGE diffusion latent cache")
    parser.add_argument("--input-root", type=Path, required=True, help="B1 cache root containing sample/sledge_raw.gz")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--autoencoder-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include-rejected", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--log-name", default="occluded_pedestrian")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = export_diffusion_inputs(
        input_root=args.input_root,
        output_root=args.output_root,
        config_path=args.config,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        device=args.device,
        include_rejected=args.include_rejected,
        overwrite=args.overwrite,
        max_scenes=args.max_scenes,
        log_name=args.log_name,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
