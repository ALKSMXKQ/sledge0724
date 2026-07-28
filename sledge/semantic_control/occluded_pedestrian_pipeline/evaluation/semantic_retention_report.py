"""Aggregate strict B2 hazard-retention metrics and diversity-axis results."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from sledge.semantic_control.io import save_json
from sledge.semantic_control.occluded_pedestrian_pipeline.position_control import position_matches

AXES = (
    "occluder_type",
    "occluder_position",
    "direction",
    "risk_level",
    "pedestrian_speed_mps",
    "source_scenario_type",
)


def build_semantic_retention_report(run_root: Path) -> Dict[str, Any]:
    run_root = Path(run_root).resolve()
    cases = _read_jsonl(run_root / "manifests/cases.jsonl")
    metrics = {str(row.get("sample_id")): row for row in _read_jsonl(run_root / "manifests/b2_results.jsonl")}

    rows: List[Dict[str, Any]] = []
    failures: Counter = Counter()
    for case in cases:
        sample_id = str(case["sample_id"])
        metric = metrics.get(sample_id, {})
        row = dict(case)
        row.update({"generated": False, "hazard_semantic_pass": False, "strict_retention_pass": False})
        if metric.get("error") or not metric:
            failures["no_generated_b2_output"] += 1
            row["failed_checks"] = ["no_generated_b2_output"]
        else:
            checks = metric.get("checks", {})
            failed = [key for key, value in checks.items() if not bool(value)]
            row.update({
                "generated": True,
                "hazard_semantic_pass": bool(metric.get("overall_pass", False)),
                "semantic_satisfaction_rate": float(metric.get("semantic_satisfaction_rate", 0.0)),
                "failed_checks": failed,
                "generated_vector": metric.get("vector_path"),
            })
            if row["hazard_semantic_pass"]:
                row["strict_retention_pass"] = True
            for item in failed:
                failures[item] += 1
        rows.append(row)

    report = {
        "schema_version": "occluded_pedestrian_semantic_retention_report_v1",
        "num_requested": len(rows),
        "num_generated": sum(bool(row.get("generated")) for row in rows),
        "generation_rate": _rate(rows, "generated"),
        "hazard_semantic_retention_rate": _rate(rows, "hazard_semantic_pass"),
        "strict_retention_rate": _rate(rows, "strict_retention_pass"),
        "failure_reasons": dict(failures),
        "by_axis": {axis: _axis(rows, axis) for axis in AXES},
        "rows": rows,
    }
    output = run_root / "reports"
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "semantic_retention_report.json", report)
    _write_csv(output / "semantic_retention_rows.csv", rows)
    return report


def _axis(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(key))].append(row)
    return {name: {
        "count": len(items),
        "retention_rate": _rate(items, "hazard_semantic_pass"),
    } for name, items in sorted(groups.items())}


def _rate(rows: Iterable[Dict[str, Any]], key: str) -> float:
    rows = list(rows)
    return sum(bool(row.get(key)) for row in rows) / len(rows) if rows else 0.0


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row if isinstance(row[key], (str, int, float, bool)) or row[key] is None})
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})
