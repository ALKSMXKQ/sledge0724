#!/usr/bin/env python3
"""Merge runner metrics and nvidia-smi samples into a reproducible report."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    base = root / "deployment/sledge_rvae"
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=base / "reports/performance_fp32.json")
    parser.add_argument("--gpu-samples", type=Path, default=base / "reports/gpu_samples.csv")
    parser.add_argument("--environment", type=Path, default=base / "reports/environment.txt")
    parser.add_argument("--output", type=Path, default=base / "reports/performance_report.json")
    return parser.parse_args()


def read_gpu_samples(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {"available": False, "reason": f"missing {path}"}
    utilization: List[float] = []
    memory: List[float] = []
    with path.open("r", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if not row or row[0].strip().lower().startswith("timestamp"):
                continue
            try:
                utilization.append(float(row[1].strip().split()[0]))
                memory.append(float(row[2].strip().split()[0]))
            except (ValueError, IndexError):
                continue
    if not utilization:
        return {"available": False, "reason": "no parseable samples"}
    return {
        "available": True,
        "samples": len(utilization),
        "gpu_utilization_percent": {
            "mean": statistics.fmean(utilization),
            "peak": max(utilization),
        },
        "device_memory_used_mib": {
            "mean": statistics.fmean(memory),
            "peak": max(memory),
        },
    }


def main() -> None:
    args = parse_args()
    runner = json.loads(args.metrics.read_text(encoding="utf-8"))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "runner": runner,
        "gpu_sampling": read_gpu_samples(args.gpu_samples),
        "environment_log": str(args.environment.resolve()) if args.environment.is_file() else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    latency = runner["latency_ms"]
    gpu = report["gpu_sampling"]
    markdown = [
        "# SLEDGE RVAE performance report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "| Metric | Mean | P50 | P95 | P99 |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in ("preprocess", "h2d", "engine", "d2h", "postprocess", "end_to_end"):
        item = latency[stage]
        markdown.append(
            f"| {stage} (ms) | {item['mean']:.3f} | {item['p50']:.3f} | {item['p95']:.3f} | {item['p99']:.3f} |"
        )
    markdown.extend(
        [
            "",
            f"- Throughput: {runner['throughput_fps']:.3f} samples/s",
            f"- Engine load: {runner['engine_load_ms']:.3f} ms",
            f"- Runner stable CUDA-memory delta: {runner['cuda_memory_mib']['runner_stable_delta']:.3f} MiB",
            f"- Runner peak CUDA-memory delta: {runner['cuda_memory_mib']['runner_peak_delta']:.3f} MiB",
        ]
    )
    if gpu.get("available"):
        markdown.extend(
            [
                f"- GPU utilization mean/peak: {gpu['gpu_utilization_percent']['mean']:.2f}% / {gpu['gpu_utilization_percent']['peak']:.2f}%",
                f"- Device GPU memory used mean/peak: {gpu['device_memory_used_mib']['mean']:.2f} / {gpu['device_memory_used_mib']['peak']:.2f} MiB",
            ]
        )
    else:
        markdown.append(f"- GPU utilization sampling unavailable: {gpu['reason']}")
    args.output.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
