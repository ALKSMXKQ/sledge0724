#!/usr/bin/env python3
"""One-command fixed-shape ONNX export with structural verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from deployment.sledge_rvae.python.contract import INPUT_NAME, INPUT_SHAPE, OUTPUT_NAMES
from deployment.sledge_rvae.python.model import DeterministicRVAE, deployment_inference_mode, load_rvae


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    workspace = root.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=workspace / "exp/exp/training_rvae_model/training_rvae_model/2025.10.17.06.17.03/best_model/epoch45.ckpt",
    )
    parser.add_argument("--model-config", type=Path, default=root / "deployment/sledge_rvae/configs/model_config.yaml")
    parser.add_argument("--sample", type=Path, default=root / "deployment/sledge_rvae/artifacts/validation/sample_000/input_raster.npy")
    parser.add_argument("--output", type=Path, default=root / "deployment/sledge_rvae/artifacts/sledge_rvae.onnx")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.opset != 17:
        raise ValueError("This package is qualified with opset 17 for TensorRT 8.6.1")
    sample = np.load(args.sample, allow_pickle=False).astype(np.float32, copy=False)
    if tuple(sample.shape) != INPUT_SHAPE:
        raise ValueError(f"Expected sample shape {INPUT_SHAPE}, got {sample.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(sample))
    model = DeterministicRVAE(load_rvae(args.checkpoint, args.model_config, torch.device("cpu"))).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with deployment_inference_mode():
        torch.onnx.export(
            model,
            tensor,
            str(args.output),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=[INPUT_NAME],
            output_names=list(OUTPUT_NAMES),
            dynamic_axes=None,
        )

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("onnx is required to verify the exported file") from exc
    graph = onnx.load(str(args.output), load_external_data=True)
    onnx.checker.check_model(graph)
    actual_inputs = [value.name for value in graph.graph.input]
    actual_outputs = [value.name for value in graph.graph.output]
    if actual_inputs != [INPUT_NAME] or actual_outputs != list(OUTPUT_NAMES):
        raise RuntimeError(f"ONNX I/O mismatch: inputs={actual_inputs}, outputs={actual_outputs}")
    manifest = {
        "onnx": str(args.output.resolve()),
        "sha256": sha256(args.output),
        "bytes": args.output.stat().st_size,
        "opset": args.opset,
        "input_names": actual_inputs,
        "output_names": actual_outputs,
        "fixed_input_shape": list(INPUT_SHAPE),
        "dtype": "float32",
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
