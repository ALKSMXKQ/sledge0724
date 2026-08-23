#!/usr/bin/env python3
"""Export a fixed nuPlan-derived cache sample and PyTorch reference outputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import (
    sledge_raw_feature_processing,
)
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVectorRaw

from deployment.sledge_rvae.python.contract import INPUT_SHAPE, OUTPUT_NAMES, validate_shapes
from deployment.sledge_rvae.python.model import (
    DeterministicRVAE,
    as_numpy_dict,
    deployment_inference_mode,
    load_config,
    load_rvae,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    workspace = root.parent
    default_cache = workspace / "exp/caches/autoencoder_cache/2021.09.30.06.13.47_veh-53_01477_01820/near_long_vehicle/bfe03305f807583a/sledge_raw.gz"
    default_checkpoint = workspace / "exp/exp/training_rvae_model/training_rvae_model/2025.10.17.06.17.03/best_model/epoch45.ckpt"
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-sample", type=Path, default=default_cache)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--model-config", type=Path, default=root / "deployment/sledge_rvae/configs/model_config.yaml")
    parser.add_argument("--output-dir", type=Path, default=root / "deployment/sledge_rvae/artifacts/validation/sample_000")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    for path in (args.cache_sample, args.checkpoint, args.model_config):
        if not path.is_file():
            raise FileNotFoundError(path)

    with gzip.open(args.cache_sample, "rb") as stream:
        raw = SledgeVectorRaw.deserialize(pickle.load(stream))
    config = load_config(args.model_config)
    _, raster = sledge_raw_feature_processing(raw, config)
    input_tensor = raster.to_feature_tensor().data.unsqueeze(0).contiguous().float()
    if tuple(input_tensor.shape) != INPUT_SHAPE:
        raise RuntimeError(f"Expected input {INPUT_SHAPE}, got {tuple(input_tensor.shape)}")
    if not torch.isfinite(input_tensor).all():
        raise RuntimeError("Preprocessed raster contains NaN or Inf")

    device = torch.device(args.device)
    model = DeterministicRVAE(load_rvae(args.checkpoint, args.model_config, device)).eval()
    with deployment_inference_mode():
        outputs = as_numpy_dict(model(input_tensor.to(device)))
    validate_shapes(outputs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "input_raster.npy", input_tensor.numpy())
    np.savez(args.output_dir / "pytorch_reference.npz", **outputs)
    reference_dir = args.output_dir / "pytorch"
    reference_dir.mkdir(exist_ok=True)
    for name in OUTPUT_NAMES:
        np.save(reference_dir / f"{name}.npy", outputs[name])

    metadata: Dict[str, object] = {
        "source": "nuPlan-derived SLEDGE feature cache",
        "cache_sample": str(args.cache_sample.resolve()),
        "cache_sample_sha256": sha256(args.cache_sample),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "model_config": str(args.model_config.resolve()),
        "deterministic_latent": "mu",
        "input": {
            "name": "raster",
            "shape": list(input_tensor.shape),
            "dtype": str(input_tensor.numpy().dtype),
            "layout": "NCHW",
            "min": float(input_tensor.min()),
            "max": float(input_tensor.max()),
        },
        "outputs": {
            name: {"shape": list(outputs[name].shape), "dtype": str(outputs[name].dtype)}
            for name in OUTPUT_NAMES
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
