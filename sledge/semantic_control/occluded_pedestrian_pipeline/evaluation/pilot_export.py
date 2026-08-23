"""Export accepted B1 edits as typed, simulation-readable gzip caches."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon as MplPolygon, Rectangle
import numpy as np

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle

from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import (
    sledge_raw_feature_processing,
)
from sledge.script.build_paired_original_edited_vector_caches import build_sledge_config
from sledge.semantic_control.io import load_raw_scene, save_json
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.stage_comparison import (
    _make_simulation_compatible_vector,
    _match_processed_slot,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.additive_integrity import (
    evaluate_strict_additive_edit,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.visualization import _draw_lines
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.geometry_metrics import oriented_box_corners
from sledge.semantic_control.occluded_pedestrian_pipeline.object_types import (
    embed_type_overrides,
    make_type_override,
    read_type_overrides,
    tracked_object_type_name,
)
from sledge.simulation.scenarios.sledge_scenario.sledge_scenario import SledgeScenario


TYPE_COLORS = {
    "VEHICLE": "#4c78a8",
    "PEDESTRIAN": "#e45756",
    "BICYCLE": "#f28e2b",
    "GENERIC_OBJECT": "#b07aa1",
    "TRAFFIC_CONE": "#76b7b2",
    "BARRIER": "#59a14f",
    "CZONE_SIGN": "#edc948",
}


def export_b1_simulation_cache(
    run_root: Path,
    config: Path,
    *,
    limit: Optional[int] = None,
    save_previews: bool = True,
) -> Dict[str, Any]:
    """Convert accepted B1 raw scenes and verify their visible nuPlan types."""

    run_root = Path(run_root).resolve()
    candidate_cases = _read_jsonl(
        run_root
        / "manifests/cases.jsonl"
    )
    cases = []
    for case in candidate_cases:
        label_path = (
            run_root
            / "b1_edited_cache"
            / str(case["sample_id"])
            / "scenario_label.json"
        )
        if not label_path.exists():
            continue
        if bool(
            _read_json(
                label_path
            ).get(
                "accepted",
                False,
            )
        ):
            cases.append(case)
    accepted_candidate_count = len(
        cases
    )
    if limit is not None:
        cases = cases[: int(limit)]
    sledge_config = build_sledge_config(str(config))
    feature_store = FeatureCachePickle()
    cache_root = run_root / "b1_simulation_cache"
    preview_root = run_root / "visualizations/b1_typed_cache"
    rows: List[Dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        sample_id = str(case["sample_id"])
        source_dir = run_root / "b1_edited_cache" / sample_id
        source_label = _read_json(source_dir / "scenario_label.json")
        if not bool(source_label.get("accepted", False)):
            raise RuntimeError(f"B1 scene was not accepted: {sample_id}")
        edit_report = _read_json(source_dir / "edit_report.json")
        scene, _ = load_raw_scene(source_dir / "sledge_raw.gz")
        vector, _ = sledge_raw_feature_processing(scene, sledge_config)
        vector = _make_simulation_compatible_vector(vector, scene)
        original_dir = (
            run_root
            / "b0_original_cache"
            / sample_id
        )
        original_scene, _ = load_raw_scene(
            original_dir
            / "sledge_raw.gz"
        )
        original_vector, _ = (
            sledge_raw_feature_processing(
                original_scene,
                sledge_config,
            )
        )
        original_vector = (
            _make_simulation_compatible_vector(
                original_vector,
                original_scene,
            )
        )

        ped_index = _match_processed_slot(
            scene.pedestrians,
            int(edit_report["pedestrian_index"]),
            vector.pedestrians,
        )
        occ_element = str(edit_report["occluder_elem_name"])
        occ_index = _match_processed_slot(
            getattr(scene, occ_element),
            int(edit_report["occluder_index"]),
            getattr(vector, occ_element),
        )
        if ped_index < 0 or occ_index < 0:
            raise RuntimeError(f"Controlled object was lost during vector conversion: {sample_id}")

        processed_integrity = (
            evaluate_strict_additive_edit(
                original_vector,
                vector,
                pedestrian_index=(
                    ped_index
                ),
                occluder_index=(
                    occ_index
                ),
                occluder_elem_name=(
                    occ_element
                ),
            )
        )
        if not processed_integrity[
            "overall_pass"
        ]:
            raise RuntimeError(
                "Processed B1 cache is not "
                "strictly additive for "
                f"{sample_id}"
            )

        tracked_type_name = tracked_object_type_name(case["occluder_type"])
        overrides = make_type_override(occ_element, occ_index, tracked_type_name)
        out_dir = cache_root / "log" / "sudden_pedestrian_crossing" / sample_id
        out_dir.mkdir(parents=True, exist_ok=True)
        vector_path = out_dir / "sledge_vector.gz"
        feature_store.store_computed_feature_to_folder(out_dir / "sledge_vector", vector)
        embed_type_overrides(vector_path, overrides)
        save_json(
            out_dir
            / (
                "strict_additive_"
                "integrity.json"
            ),
            processed_integrity,
        )

        expected_type = TrackedObjectType[tracked_type_name]
        expected_token = f"{expected_type.value}_{occ_index}"
        scenario = SledgeScenario(out_dir / "sledge_vector")
        detections = scenario.initial_tracked_objects.tracked_objects
        decoded = next((obj for obj in detections if obj.track_token == expected_token), None)
        if decoded is None or decoded.tracked_object_type != expected_type:
            observed = [(obj.track_token, obj.tracked_object_type.name) for obj in detections]
            raise RuntimeError(
                f"Typed gzip round-trip failed for {sample_id}: expected {(expected_token, tracked_type_name)}, "
                f"observed={observed}"
            )
        if read_type_overrides(vector_path) != overrides:
            raise RuntimeError(f"Embedded type metadata round-trip failed: {sample_id}")

        preview_path = None
        if save_previews:
            preview_path = preview_root / f"{sample_id}.png"
            _save_typed_preview(scenario, preview_path, prompt=str(case.get("prompt", "")))

        label = {
            "schema_version": "occluded_pedestrian_typed_simulation_cache_v1",
            "sample_id": sample_id,
            "prompt": str(case.get("prompt", "")),
            "source_raw": str(source_dir / "sledge_raw.gz"),
            "source_scenario_type": str(case.get("source_scenario_type", "unknown")),
            "occluder_type": str(case["occluder_type"]),
            "occluder_tracked_object_type": tracked_type_name,
            "direction": str(case["direction"]),
            "pedestrian_speed_mps": float(case["pedestrian_speed_mps"]),
            "lane_center_y": float(
                source_label.get(
                    "lane_center_y",
                    0.0,
                )
            ),
            "pedestrian_index": int(ped_index),
            "occluder_index": int(occ_index),
            "occluder_element": occ_element,
            "controlled_tokens": [f"1_{ped_index}", expected_token],
            "object_type_overrides": overrides,
            "gzip_round_trip_pass": True,
            (
                "strict_additive_"
                "integrity_pass"
            ): True,
            "preview": str(preview_path) if preview_path else None,
        }
        save_json(out_dir / "scenario_label.json", label)
        rows.append({**label, "sledge_vector_gz": str(vector_path)})
        print(f"[{index}/{len(cases)}] {sample_id}: {tracked_type_name} -> {expected_token}")

    if len(rows) != len(cases):
        raise RuntimeError(f"Exported {len(rows)} of {len(cases)} requested scenes")
    _write_csv(run_root / "manifests/b1_simulation_cache.csv", rows)
    montage_path = None
    if save_previews and rows:
        montage_path = preview_root / "occluder_type_montage.png"
        _save_type_montage(rows, montage_path)
    type_counts = {
        name: sum(row["occluder_tracked_object_type"] == name for row in rows)
        for name in sorted({row["occluder_tracked_object_type"] for row in rows})
    }
    payload = {
        "schema_version": "occluded_pedestrian_typed_export_summary_v1",
        "num_candidates": len(
            candidate_cases
        ),
        "num_rejected_or_failed": (
            len(candidate_cases)
            - accepted_candidate_count
        ),
        "num_requested": len(cases),
        "num_exported": len(rows),
        "num_unique_source_scenes": len({str(case["input_raw"]) for case in cases}),
        "gzip_round_trip_pass_count": sum(bool(row["gzip_round_trip_pass"]) for row in rows),
        "occluder_tracked_object_type_counts": type_counts,
        "cache_root": str(cache_root),
        "preview_root": str(preview_root) if save_previews else None,
        "occluder_type_montage": str(montage_path) if montage_path else None,
        "manifest_csv": str(run_root / "manifests/b1_simulation_cache.csv"),
    }
    save_json(run_root / "manifests/b1_simulation_cache_summary.json", payload)
    return payload


def _save_typed_preview(scenario: SledgeScenario, output: Path, *, prompt: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    _draw_lines(ax, scenario._sledge_vector.lines)
    ax.add_patch(Rectangle((-2.4, -0.95), 4.8, 1.9, facecolor="#ff9da7", edgecolor="#9c3f52", alpha=0.45))
    ax.text(0.0, 0.0, "EGO", fontsize=7, ha="center", va="center")
    objects = scenario.initial_tracked_objects.tracked_objects
    for obj in objects:
        name = obj.tracked_object_type.name
        color = TYPE_COLORS.get(name, "#999999")
        corners = oriented_box_corners(
            obj.center.x,
            obj.center.y,
            obj.center.heading,
            obj.box.width,
            obj.box.length,
        )
        ax.add_patch(MplPolygon(corners, closed=True, facecolor=color, edgecolor=color, alpha=0.55))
        ax.text(obj.center.x, obj.center.y, f"{name}\n{obj.track_token}", fontsize=6, ha="center", va="center")
    ax.set_xlim(-8, 35)
    ax.set_ylim(-15, 15)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.6)
    ax.set_xlabel("longitudinal x / m")
    ax.set_ylabel("lateral y / m")
    ax.set_title(prompt, fontsize=9)
    present = sorted({obj.tracked_object_type.name for obj in objects})
    handles = [Patch(facecolor=TYPE_COLORS.get(name, "#999999"), label=name) for name in present]
    if handles:
        ax.legend(handles=handles, loc="upper left", fontsize=7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _save_type_montage(rows: List[Dict[str, Any]], output: Path) -> None:
    representatives: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        representatives.setdefault(str(row["occluder_tracked_object_type"]), row)
    ordered = [
        "VEHICLE",
        "BICYCLE",
        "GENERIC_OBJECT",
        "TRAFFIC_CONE",
        "BARRIER",
        "CZONE_SIGN",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, type_name in zip(axes.flat, ordered):
        row = representatives.get(type_name)
        if row is None or not row.get("preview"):
            ax.axis("off")
            continue
        ax.imshow(plt.imread(str(row["preview"])))
        ax.set_title(type_name)
        ax.axis("off")
    fig.suptitle("nuPlan-visible occluder types: typed B1 cache representatives", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
