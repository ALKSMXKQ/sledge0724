"""Build a simulator-facing manifest for every generated gzip stage."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from sledge.semantic_control.io import load_raw_scene, save_json
from sledge.semantic_control.occluded_pedestrian_pipeline.object_types import (
    read_type_overrides,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.visualization import (
    save_generated_gzip_comparison,
)


def build_generation_gzip_manifest(
    run_root: Path,
    *,
    require_rvae: bool = True,
    require_b2: bool = True,
    strict: bool = True,
) -> Dict[str, Any]:
    """Index and re-open the B1, RVAE and protected-diffusion gz files."""

    root = Path(run_root).resolve()
    cases = _read_jsonl(root / "manifests/cases.jsonl")
    rows: List[Dict[str, Any]] = []
    for case in cases:
        sample_id = str(case["sample_id"])
        b1_dir = root / "b1_edited_cache" / sample_id
        b1_label_path = b1_dir / "scenario_label.json"
        if not b1_label_path.exists():
            continue
        b1_label = _read_json(b1_label_path)
        if not bool(b1_label.get("accepted", False)):
            continue

        paths = {
            "b1_edited_raw_gz": b1_dir / "sledge_raw.gz",
            "b1_simulation_gz": (
                root
                / "b1_simulation_cache/log/sudden_pedestrian_crossing"
                / sample_id
                / "sledge_vector.gz"
            ),
            "rvae_raw_candidate_gz": (
                root
                / "rvae_reconstruction/candidate_cache/log/"
                "sudden_pedestrian_crossing"
                / sample_id
                / "sledge_vector.gz"
            ),
            "rvae_protected_gz": (
                root
                / "rvae_reconstruction/generated_cache/log/"
                "sudden_pedestrian_crossing"
                / sample_id
                / "sledge_vector.gz"
            ),
            "diffusion_protected_gz": (
                root
                / "b2_diffusion/semantic_protected/generated_cache/log/"
                "sudden_pedestrian_crossing"
                / sample_id
                / "sledge_vector.gz"
            ),
        }
        required = ["b1_edited_raw_gz", "b1_simulation_gz"]
        if require_rvae:
            required.append("rvae_protected_gz")
        if require_b2:
            required.append("diffusion_protected_gz")

        readable: Dict[str, bool] = {}
        typed: Dict[str, bool] = {}
        for name, path in paths.items():
            if not path.exists():
                readable[name] = False
                typed[name] = False
                continue
            try:
                load_raw_scene(path)
                readable[name] = True
            except Exception:
                readable[name] = False
            typed[name] = bool(read_type_overrides(path))

        rvae_label = _optional_json(
            paths["rvae_protected_gz"].parent / "scenario_label.json"
        )
        b2_label = _optional_json(
            paths["diffusion_protected_gz"].parent / "scenario_label.json"
        )
        checks = {
            "required_gzip_readable": all(readable[name] for name in required),
            "b1_accepted": True,
            "b1_simulation_typed": typed["b1_simulation_gz"],
            "rvae_semantic_contract": (
                bool(rvae_label.get("semantic_contract_pass", False))
                if require_rvae
                else True
            ),
            "rvae_simulator_roundtrip": (
                bool(rvae_label.get("simulator_roundtrip_pass", False))
                if require_rvae
                else True
            ),
            "diffusion_semantic_contract": (
                bool(b2_label.get("semantic_contract_pass", False))
                if require_b2
                else True
            ),
            "diffusion_simulator_roundtrip": (
                bool(b2_label.get("gzip_roundtrip_pass", False))
                if require_b2
                else True
            ),
        }
        preview = None
        if all(
            readable[name]
            for name in (
                "b1_simulation_gz",
                "rvae_protected_gz",
                "diffusion_protected_gz",
            )
        ):
            b1_scene, _ = load_raw_scene(paths["b1_simulation_gz"])
            rvae_scene, _ = load_raw_scene(paths["rvae_protected_gz"])
            b2_scene, _ = load_raw_scene(paths["diffusion_protected_gz"])
            protected = dict(rvae_label.get("protected_slots", {}) or {})
            edit_result = {
                "pedestrian_index": int(protected.get("pedestrians", -1)),
                "occluder_elem_name": str(
                    protected.get("occluder_element", "vehicles")
                ),
                "occluder_index": int(protected.get("occluder_index", -1)),
                "occluder_source": str(
                    b1_label.get("occluder_type", "occluder")
                ),
            }
            preview = save_generated_gzip_comparison(
                b1_scene,
                rvae_scene,
                b2_scene,
                edit_result,
                root
                / "visualizations/generated_gzip_stages"
                / f"{sample_id}.png",
                prompt=str(case.get("prompt", "")),
            )
        rows.append(
            {
                "sample_id": sample_id,
                "prompt": str(case.get("prompt", "")),
                **{name: str(path) for name, path in paths.items()},
                "readable": readable,
                "typed": typed,
                "checks": checks,
                "overall_pass": bool(all(checks.values())),
                "preview": str(preview) if preview else None,
            }
        )

    summary = {
        "schema_version": "occluded_pedestrian_generation_gzip_manifest_v1",
        "num_accepted_b1": len(rows),
        "num_complete": sum(bool(row["overall_pass"]) for row in rows),
        "all_complete": bool(rows and all(row["overall_pass"] for row in rows)),
        "require_rvae": require_rvae,
        "require_b2": require_b2,
        "rows": rows,
    }
    save_json(root / "manifests/generated_gzip_stages.json", summary)
    _write_csv(root / "manifests/generated_gzip_stages.csv", rows)
    if strict and not summary["all_complete"]:
        raise RuntimeError(
            "Generated gzip stage audit failed: "
            f"complete={summary['num_complete']}/{summary['num_accepted_b1']}"
        )
    return summary


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _optional_json(path: Path) -> Dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "overall_pass",
        "b1_edited_raw_gz",
        "b1_simulation_gz",
        "rvae_raw_candidate_gz",
        "rvae_protected_gz",
        "diffusion_protected_gz",
        "preview",
        "prompt",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


__all__ = ["build_generation_gzip_manifest"]
