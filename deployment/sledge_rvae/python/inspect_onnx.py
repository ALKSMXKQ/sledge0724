#!/usr/bin/env python3
"""Static TensorRT-oriented inspection of the fixed ONNX graph."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import onnx

from deployment.sledge_rvae.python.contract import INPUT_NAME, INPUT_SHAPE, OUTPUT_NAMES, OUTPUT_SHAPES


PROHIBITED_OPS = {"ATen", "If", "Loop", "Scan", "ScatterND", "CumSum"}


def tensor_shape(value: onnx.ValueInfoProto) -> list[int | str]:
    dims = []
    for dim in value.type.tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.HasField("dim_value") else dim.dim_param)
    return dims


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=root / "deployment/sledge_rvae/artifacts/sledge_rvae.onnx")
    parser.add_argument("--output", type=Path, default=root / "deployment/sledge_rvae/reports/onnx_compatibility.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = onnx.load(str(args.onnx), load_external_data=True)
    onnx.checker.check_model(model)
    counts = Counter(node.op_type for node in model.graph.node)
    custom_domains = sorted({node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")})
    prohibited_found = sorted(PROHIBITED_OPS.intersection(counts))
    inputs = {value.name: tensor_shape(value) for value in model.graph.input}
    outputs = {value.name: tensor_shape(value) for value in model.graph.output}
    contract_ok = (
        inputs == {INPUT_NAME: list(INPUT_SHAPE)}
        and list(outputs) == list(OUTPUT_NAMES)
        and outputs == {name: list(OUTPUT_SHAPES[name]) for name in OUTPUT_NAMES}
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "onnx": str(args.onnx.resolve()),
        "onnx_opset": max(item.version for item in model.opset_import if item.domain in ("", "ai.onnx")),
        "tensorrt_python_installed": importlib.metadata.version("tensorrt"),
        "node_count": len(model.graph.node),
        "operator_counts": dict(sorted(counts.items())),
        "custom_domains": custom_domains,
        "prohibited_ops_found": prohibited_found,
        "fixed_contract_ok": contract_ok,
        "static_precheck_passed": contract_ok and not custom_domains and not prohibited_found,
        "parser_check": {
            "completed": False,
            "reason": "TensorRT Builder creation requires a working NVIDIA driver/GPU; this host returns CUDA error 100",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = [
        "# ONNX TensorRT-oriented compatibility precheck",
        "",
        f"- Static result: {'PASS' if report['static_precheck_passed'] else 'FAIL'}",
        f"- Fixed I/O contract: {'PASS' if contract_ok else 'FAIL'}",
        f"- Custom domains: {custom_domains or 'none'}",
        f"- Prohibited/control-flow ops: {prohibited_found or 'none'}",
        f"- Node count: {len(model.graph.node)}",
        f"- Installed TensorRT Python package: {report['tensorrt_python_installed']}",
        "- Native TensorRT parser: pending target GPU because Builder initialization requires a CUDA device",
    ]
    args.output.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["static_precheck_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

