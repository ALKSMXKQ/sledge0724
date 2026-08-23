#!/usr/bin/env python3
"""Run/compare PyTorch, ONNX Runtime and C++ TensorRT outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch

from deployment.sledge_rvae.python.comparison import compare_outputs
from deployment.sledge_rvae.python.contract import INPUT_NAME, OUTPUT_NAMES, TOLERANCES, validate_shapes
from deployment.sledge_rvae.python.model import (
    DeterministicRVAE,
    as_numpy_dict,
    deployment_inference_mode,
    load_rvae,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    workspace = root.parent
    base = root / "deployment/sledge_rvae"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=base / "artifacts/validation/sample_000/input_raster.npy")
    parser.add_argument("--checkpoint", type=Path, default=workspace / "exp/exp/training_rvae_model/training_rvae_model/2025.10.17.06.17.03/best_model/epoch45.ckpt")
    parser.add_argument("--model-config", type=Path, default=base / "configs/model_config.yaml")
    parser.add_argument("--onnx", type=Path, default=base / "artifacts/sledge_rvae.onnx")
    parser.add_argument("--trt-runner", type=Path, help="Optional compiled sledge_rvae_trt executable")
    parser.add_argument("--runner-config", type=Path, default=base / "configs/runtime.ini")
    parser.add_argument("--trt-output-dir", type=Path, default=base / "artifacts/validation/sample_000/tensorrt")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--report", type=Path, default=base / "reports/consistency.json")
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--skip-tensorrt", action="store_true")
    return parser.parse_args()


def save_outputs(directory: Path, outputs: Mapping[str, np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        np.save(directory / f"{name}.npy", outputs[name])


def load_outputs(directory: Path) -> Dict[str, np.ndarray]:
    outputs = {name: np.load(directory / f"{name}.npy", allow_pickle=False) for name in OUTPUT_NAMES}
    validate_shapes(outputs)
    return outputs


def run_pytorch(args: argparse.Namespace, sample: np.ndarray) -> Dict[str, np.ndarray]:
    model = DeterministicRVAE(load_rvae(args.checkpoint, args.model_config, torch.device("cpu"))).eval()
    with deployment_inference_mode():
        outputs = as_numpy_dict(model(torch.from_numpy(np.ascontiguousarray(sample))))
    validate_shapes(outputs)
    return outputs  # type: ignore[return-value]


def run_onnx(args: argparse.Namespace, sample: np.ndarray) -> Dict[str, np.ndarray]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for ONNX validation") from exc
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    values = session.run(list(OUTPUT_NAMES), {INPUT_NAME: np.ascontiguousarray(sample, dtype=np.float32)})
    outputs = dict(zip(OUTPUT_NAMES, values))
    validate_shapes(outputs)
    return outputs


def write_markdown(path: Path, report: Mapping[str, object]) -> None:
    lines = [
        "# SLEDGE RVAE backend consistency",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"- Completed backends: {', '.join(report['backends'])}",
        f"- Pending backends: {', '.join(report['pending_backends']) or 'none'}",
        f"- Three-backend acceptance: {'PASS' if report['three_backend_passed'] else 'PENDING/FAIL'}",
        "",
    ]
    comparisons = report["comparisons"]
    if not comparisons:
        lines.append("No backend comparison was completed.")
    for name, value in comparisons.items():
        global_metrics = value["global"]
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Result: {'PASS' if value['passed'] else 'FAIL'}",
                f"- Max absolute error: {global_metrics['max_abs']:.8g}",
                f"- Mean absolute error: {global_metrics['mean_abs']:.8g}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    sample = np.load(args.input, allow_pickle=False).astype(np.float32, copy=False)
    results: Dict[str, Dict[str, np.ndarray]] = {}
    base = args.input.parent
    if not args.skip_pytorch:
        results["pytorch"] = run_pytorch(args, sample)
        save_outputs(base / "pytorch", results["pytorch"])
    elif (base / "pytorch").is_dir():
        results["pytorch"] = load_outputs(base / "pytorch")

    if not args.skip_onnx:
        results["onnxruntime"] = run_onnx(args, sample)
        save_outputs(base / "onnxruntime", results["onnxruntime"])

    if not args.skip_tensorrt:
        if args.trt_runner:
            subprocess.run([str(args.trt_runner), "--config", str(args.runner_config)], check=True)
        results["tensorrt"] = load_outputs(args.trt_output_dir)

    if "pytorch" not in results:
        raise RuntimeError("PyTorch reference is required; run it or retain the exported reference directory")
    comparisons: Dict[str, object] = {}
    tolerances = TOLERANCES[args.precision]
    for backend in ("onnxruntime", "tensorrt"):
        if backend in results:
            comparisons[f"pytorch_vs_{backend}"] = compare_outputs(
                results["pytorch"], results[backend], args.threshold, tolerances
            )
    if "onnxruntime" in results and "tensorrt" in results:
        comparisons["onnxruntime_vs_tensorrt"] = compare_outputs(
            results["onnxruntime"], results["tensorrt"], args.threshold, tolerances
        )
    completed_passed = bool(comparisons) and all(value["passed"] for value in comparisons.values())
    pending_backends = sorted({"pytorch", "onnxruntime", "tensorrt"}.difference(results))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "precision": args.precision,
        "postprocess_threshold": args.threshold,
        "backends": sorted(results),
        "pending_backends": pending_backends,
        "comparisons": comparisons,
        "completed_comparisons_passed": completed_passed,
        "three_backend_complete": not pending_backends,
        "three_backend_passed": not pending_backends and completed_passed,
        "passed": completed_passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.report.with_suffix(".md"), report)
    print(json.dumps(report, indent=2))
    if comparisons and not completed_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
