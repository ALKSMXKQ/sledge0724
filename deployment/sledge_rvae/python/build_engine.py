#!/usr/bin/env python3
"""Build a fixed-profile TensorRT 8.6.1 engine without requiring trtexec."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import tensorrt as trt

from deployment.sledge_rvae.python.contract import INPUT_NAME, INPUT_SHAPE, OUTPUT_NAMES


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    base = root / "deployment/sledge_rvae"
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=base / "artifacts/sledge_rvae.onnx")
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--workspace-mib", type=int, default=4096)
    parser.add_argument("--timing-cache", type=Path, default=base / "artifacts/timing.cache")
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def main() -> None:
    args = parse_args()
    if trt.__version__ != "8.6.1":
        raise RuntimeError(f"TensorRT 8.6.1 required, found {trt.__version__}")
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    if args.workspace_mib <= 0:
        raise ValueError("--workspace-mib must be positive")
    if args.engine is None:
        args.engine = args.onnx.parent / f"sledge_rvae_{args.precision}.engine"
    if args.manifest is None:
        args.manifest = args.engine.with_suffix(".build.json")

    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(args.onnx.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))
    if network.num_inputs != 1 or network.get_input(0).name != INPUT_NAME:
        raise RuntimeError("TensorRT network input contract mismatch")
    actual_outputs = [network.get_output(i).name for i in range(network.num_outputs)]
    if actual_outputs != list(OUTPUT_NAMES):
        raise RuntimeError(f"TensorRT network outputs mismatch: {actual_outputs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib * 1024 * 1024)
    profile = builder.create_optimization_profile()
    profile.set_shape(INPUT_NAME, INPUT_SHAPE, INPUT_SHAPE, INPUT_SHAPE)
    config.add_optimization_profile(profile)
    if args.precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("Target GPU does not report fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)

    existing_cache = args.timing_cache.read_bytes() if args.timing_cache.is_file() else b""
    timing_cache = config.create_timing_cache(existing_cache)
    config.set_timing_cache(timing_cache, ignore_mismatch=False)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(bytes(serialized))
    args.timing_cache.parent.mkdir(parents=True, exist_ok=True)
    args.timing_cache.write_bytes(bytes(config.get_timing_cache().serialize()))
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "builder": "TensorRT Python API",
        "tensorrt": trt.__version__,
        "precision": args.precision,
        "workspace_mib": args.workspace_mib,
        "fixed_profile": {INPUT_NAME: list(INPUT_SHAPE)},
        "custom_plugins": [],
        "onnx": {"path": str(args.onnx.resolve()), "sha256": sha256(args.onnx)},
        "engine": {
            "path": str(args.engine.resolve()),
            "bytes": args.engine.stat().st_size,
            "sha256": sha256(args.engine),
        },
        "host": platform.platform(),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "nvcc": command_output(["nvcc", "--version"]),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

