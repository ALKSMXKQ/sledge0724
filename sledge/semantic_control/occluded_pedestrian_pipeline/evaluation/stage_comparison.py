"""Prepare, simulate and visualize comparable B0/B1/B2 scene stages."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd
from PIL import Image

from nuplan.common.actor_state.oriented_box import in_collision
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.simulation_log import SimulationLog
from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle

from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import (
    sledge_raw_feature_processing,
)
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeVector,
    SledgeVectorElement,
)
from sledge.script.build_paired_original_edited_vector_caches import build_sledge_config
from sledge.semantic_control.io import load_raw_scene, save_json
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.simulation import run_simulation
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.visualization import (
    _draw_lines,
    save_three_stage_comparison,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.geometry_metrics import (
    oriented_box_corners,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.object_types import (
    embed_type_overrides,
    make_type_override,
    tracked_object_type_name,
)


STAGES = ("B0", "B1", "B2")


def prepare_stage_vector_caches(
    run_root: Path,
    config: Path,
    *,
    max_scenes: Optional[int] = None,
) -> Dict[str, Path]:
    """Convert B0/B1 raw scenes into simulation-readable vectors; reuse true B2."""
    run_root = Path(run_root).resolve()
    cases = _read_jsonl(run_root / "manifests/cases.jsonl")
    if max_scenes is not None:
        cases = cases[: int(max_scenes)]
    sledge_config = build_sledge_config(str(config))
    feature_store = FeatureCachePickle()
    roots = {
        "B0": run_root / "stage_vector_caches/B0",
        "B1": run_root / "stage_vector_caches/B1",
        "B2": run_root / "b2_generated_cache",
    }
    manifest_rows: List[Dict[str, Any]] = []
    for case in cases:
        sample_id = str(case["sample_id"])
        for stage, source_name, scenario_type in (
            ("B0", "b0_original_cache", "B0_original"),
            ("B1", "b1_edited_cache", "B1_semantic_edit"),
        ):
            source = run_root / source_name / sample_id / "sledge_raw.gz"
            scene, _ = load_raw_scene(source)
            vector, _ = sledge_raw_feature_processing(scene, sledge_config)
            vector = _make_simulation_compatible_vector(vector, scene)
            out_dir = roots[stage] / "log" / scenario_type / sample_id
            out_dir.mkdir(parents=True, exist_ok=True)
            feature_store.store_computed_feature_to_folder(out_dir / "sledge_vector", vector)

            label: Dict[str, Any] = {
                "schema_version": "occluded_pedestrian_stage_cache_v1",
                "stage": stage,
                "sample_id": sample_id,
                "prompt": str(case.get("prompt", "")),
                "source_raw": str(source),
                "controlled_tokens": [],
            }
            if stage == "B1":
                report = _read_json(run_root / "b1_edited_cache" / sample_id / "edit_report.json")
                ped = _match_processed_slot(scene.pedestrians, int(report.get("pedestrian_index", -1)), vector.pedestrians)
                occ_elem = str(report.get("occluder_elem_name", "vehicles"))
                occ = _match_processed_slot(
                    getattr(scene, occ_elem),
                    int(report.get("occluder_index", -1)),
                    getattr(vector, occ_elem),
                )
                tracked_type_name = tracked_object_type_name(case.get("occluder_type", "vehicle"))
                overrides = make_type_override(occ_elem, occ, tracked_type_name)
                embed_type_overrides(out_dir / "sledge_vector.gz", overrides)
                occ_token = f"{TrackedObjectType[tracked_type_name].value}_{occ}"
                label.update(
                    {
                        "pedestrian_index": ped,
                        "occluder_index": occ,
                        "occluder_element": occ_elem,
                        "occluder_tracked_object_type": tracked_type_name,
                        "object_type_overrides": overrides,
                        "controlled_tokens": [f"1_{ped}", occ_token],
                    }
                )
            save_json(out_dir / "scenario_label.json", label)
            manifest_rows.append({"stage": stage, "sample_id": sample_id, "vector": str(out_dir / "sledge_vector.gz")})

    # B2 must be a real accepted diffusion output for every selected case.
    b2_by_sample = _b2_labels_by_sample(roots["B2"])
    missing = [str(case["sample_id"]) for case in cases if str(case["sample_id"]) not in b2_by_sample]
    if missing:
        raise RuntimeError(f"Missing accepted B2 diffusion outputs for: {missing}")
    for case in cases:
        sample_id = str(case["sample_id"])
        manifest_rows.append(
            {"stage": "B2", "sample_id": sample_id, "vector": str(b2_by_sample[sample_id].parent / "sledge_vector.gz")}
        )
    _write_csv(run_root / "manifests/stage_vector_caches.csv", manifest_rows)
    return roots


def _make_simulation_compatible_vector(processed: SledgeVector, raw_scene: Any) -> SledgeVector:
    """Adapt the four-component raw ego state to SledgeScenario's scalar-speed cache contract.

    Raw B0/B1 scenes preserve ``[vx, vy, ax, ay]`` for editing and validation, while
    diffusion-decoded B2 caches contain the scalar longitudinal speed expected by the
    legacy closed-loop scenario implementation.  Keep that representation difference
    at the simulation-cache boundary instead of mutating either source stage.
    """
    raw_states = np.asarray(raw_scene.ego.states, dtype=np.float32).reshape(-1)
    raw_mask = np.asarray(raw_scene.ego.mask).reshape(-1)
    speed = float(raw_states[0]) if raw_states.size else 0.0
    valid = bool(raw_mask[0]) if raw_mask.size else True
    sim_ego = SledgeVectorElement(
        states=np.asarray([speed], dtype=np.float32),
        mask=np.asarray([valid], dtype=bool),
    )
    return SledgeVector(
        lines=processed.lines,
        vehicles=processed.vehicles,
        pedestrians=processed.pedestrians,
        static_objects=processed.static_objects,
        green_lights=processed.green_lights,
        red_lights=processed.red_lights,
        ego=sim_ego,
    )


def save_all_stage_scene_visualizations(run_root: Path, *, max_scenes: Optional[int] = None) -> Dict[str, Any]:
    """Create direct B0/B1/B2 vector comparisons for every selected sample."""
    run_root = Path(run_root).resolve()
    cases = _read_jsonl(run_root / "manifests/cases.jsonl")
    if max_scenes is not None:
        cases = cases[: int(max_scenes)]
    b2_by_sample = _b2_labels_by_sample(run_root / "b2_generated_cache")
    output_dir = run_root / "visualizations/stage_scenes"
    rows = []
    diffusion_audit_rows = []
    for case in cases:
        sample_id = str(case["sample_id"])
        b0, _ = load_raw_scene(run_root / "b0_original_cache" / sample_id / "sledge_raw.gz")
        b1, _ = load_raw_scene(run_root / "b1_edited_cache" / sample_id / "sledge_raw.gz")
        b2_label_path = b2_by_sample[sample_id]
        b2, _ = load_raw_scene(b2_label_path.parent / "sledge_vector.gz")
        edit_result = _read_json(run_root / "b1_edited_cache" / sample_id / "edit_report.json")
        b2_label = _read_json(b2_label_path)
        protected = b2_label.get("protected_slots", {})
        b1_vector, _ = load_raw_scene(
            run_root / "stage_vector_caches/B1/log/B1_semantic_edit" / sample_id / "sledge_vector.gz"
        )
        diffusion_audit_rows.append(_audit_diffusion_effect(sample_id, b1_vector, b2, protected))
        b2_result = dict(edit_result)
        b2_result.update(
            {
                "pedestrian_index": int(protected.get("pedestrians", -1)),
                "occluder_index": int(protected.get("occluder_index", -1)),
                "occluder_elem_name": str(protected.get("occluder_element", "vehicles")),
            }
        )
        output = output_dir / f"{sample_id}.png"
        save_three_stage_comparison(
            b0,
            b1,
            b2,
            edit_result,
            b2_result,
            output,
            prompt=str(case.get("prompt", "")),
        )
        rows.append({"sample_id": sample_id, "image": str(output), "b2_vector": str(b2_label_path.parent / "sledge_vector.gz")})
    payload = {"schema_version": "occluded_pedestrian_stage_visuals_v1", "num_images": len(rows), "rows": rows}
    save_json(run_root / "manifests/stage_scene_visualizations.json", payload)
    save_json(
        run_root / "manifests/b2_diffusion_effect_audit.json",
        {
            "schema_version": "occluded_pedestrian_diffusion_effect_audit_v1",
            "num_samples": len(diffusion_audit_rows),
            "road_topology_exact_count": sum(bool(row["road_topology_exact"]) for row in diffusion_audit_rows),
            "nonprotected_changed_count": sum(bool(row["nonprotected_changed"]) for row in diffusion_audit_rows),
            "rows": diffusion_audit_rows,
        },
    )
    return payload


def _audit_diffusion_effect(
    sample_id: str,
    b1_vector: Any,
    b2_vector: Any,
    protected: Mapping[str, Any],
) -> Dict[str, Any]:
    road_exact = bool(
        np.array_equal(np.asarray(b1_vector.lines.states), np.asarray(b2_vector.lines.states))
        and np.array_equal(np.asarray(b1_vector.lines.mask), np.asarray(b2_vector.lines.mask))
    )
    ped_index = int(protected.get("pedestrians", -1))
    occ_index = int(protected.get("occluder_index", -1))
    occ_element = str(protected.get("occluder_element", "vehicles"))
    element_differences: Dict[str, float] = {}
    element_mask_flips: Dict[str, int] = {}
    for name in ("vehicles", "pedestrians", "static_objects", "green_lights", "red_lights"):
        b1_states = np.asarray(getattr(b1_vector, name).states)
        b2_states = np.asarray(getattr(b2_vector, name).states)
        b1_mask = np.asarray(getattr(b1_vector, name).mask).reshape(-1) >= 0.3
        b2_mask = np.asarray(getattr(b2_vector, name).mask).reshape(-1) >= 0.3
        keep = np.ones(len(b1_states), dtype=bool)
        if name == "pedestrians" and 0 <= ped_index < len(keep):
            keep[ped_index] = False
        if name == occ_element and 0 <= occ_index < len(keep):
            keep[occ_index] = False
        active = keep & (b1_mask | b2_mask)
        element_mask_flips[name] = int(np.sum(keep & (b1_mask != b2_mask)))
        if np.any(active):
            element_differences[name] = float(np.mean(np.abs(b1_states[active] - b2_states[active])))
    max_difference = max(element_differences.values(), default=0.0)
    mask_flip_count = sum(element_mask_flips.values())
    return {
        "sample_id": sample_id,
        "road_topology_exact": road_exact,
        "nonprotected_changed": bool(max_difference > 1e-6 or mask_flip_count > 0),
        "max_element_mean_absolute_difference": max_difference,
        "nonprotected_mask_flip_count": mask_flip_count,
        "element_mean_absolute_differences": element_differences,
        "element_mask_flip_counts": element_mask_flips,
    }


def run_and_visualize_stage_simulations(
    *,
    repo_root: Path,
    run_root: Path,
    config: Path,
    planner: str = "pdm_closed_planner",
    limit: int = 20,
) -> Dict[str, Any]:
    """Run independent closed-loop simulations for B0, B1 and B2."""
    run_root = Path(run_root).resolve()
    caches = prepare_stage_vector_caches(run_root, config, max_scenes=limit)
    save_all_stage_scene_visualizations(run_root, max_scenes=limit)
    stage_payloads: Dict[str, Any] = {}
    for stage in STAGES:
        stage_root = run_root / "stage_simulations" / stage
        result = run_simulation(
            repo_root=Path(repo_root).resolve(),
            scenario_cache=caches[stage],
            output_manifest=stage_root / "manifests/simulation_launch.json",
            planner=planner,
            limit=limit,
            simulation_output=stage_root / "simulation",
        )
        visuals = visualize_simulation_logs(
            stage=stage,
            scenario_cache=caches[stage],
            simulation_root=stage_root / "simulation",
            output_root=run_root / "visualizations/simulation_trajectories" / stage,
            make_representative_gif=True,
        )
        stage_payloads[stage] = {"simulation": result, "visualization": visuals}
    comparison = compare_stage_metrics(run_root)
    payload = {
        "schema_version": "occluded_pedestrian_stage_simulation_comparison_v1",
        "stages": stage_payloads,
        "comparison": comparison,
    }
    save_json(run_root / "manifests/stage_simulation_comparison.json", payload)
    return payload


def visualize_simulation_logs(
    *,
    stage: str,
    scenario_cache: Path,
    simulation_root: Path,
    output_root: Path,
    make_representative_gif: bool = True,
) -> Dict[str, Any]:
    labels = {path.parent.name: _read_json(path) for path in Path(scenario_cache).glob("**/scenario_label.json")}
    log_paths = sorted(Path(simulation_root).glob("simulation_log/**/*.msgpack.xz"))
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    representative: Optional[Tuple[SimulationLog, Sequence[str], Path]] = None
    for log_path in log_paths:
        log = SimulationLog.load_data(log_path)
        scenario_name = str(log.scenario.scenario_name)
        label = labels.get(scenario_name, {})
        controlled = _controlled_tokens(label)
        output = output_root / f"{scenario_name}.png"
        _save_simulation_trajectory(log, controlled, output, stage=stage)
        rows.append({"scenario_name": scenario_name, "log": str(log_path), "trajectory_image": str(output)})
        if representative is None:
            representative = (log, controlled, output_root / "representative.gif")
    gif_path = None
    if make_representative_gif and representative is not None:
        _save_simulation_gif(*representative, stage=stage)
        gif_path = str(representative[2])
    payload = {
        "schema_version": "occluded_pedestrian_simulation_visuals_v1",
        "stage": stage,
        "num_logs": len(log_paths),
        "num_trajectory_images": len(rows),
        "representative_gif": gif_path,
        "rows": rows,
    }
    save_json(output_root / "manifest.json", payload)
    return payload


def compare_stage_metrics(run_root: Path) -> Dict[str, Any]:
    run_root = Path(run_root).resolve()
    metric_files = {
        "collision_free_rate": "no_ego_at_fault_collisions.parquet",
        "ttc_safe_rate": "time_to_collision_within_bound.parquet",
        "comfort_rate": "ego_is_comfortable.parquet",
        "progress_rate": "ego_progress_along_expert_route.parquet",
        "drivable_area_rate": "drivable_area_compliance.parquet",
    }
    rows: List[Dict[str, Any]] = []
    for stage in STAGES:
        metrics_dir = run_root / "stage_simulations" / stage / "simulation/metrics"
        row: Dict[str, Any] = {"stage": stage}
        for name, filename in metric_files.items():
            frame = pd.read_parquet(metrics_dir / filename)
            row[name] = float(frame["metric_score"].mean())
            row[f"{name}_count"] = int(len(frame))
        collisions = pd.read_parquet(metrics_dir / "no_ego_at_fault_collisions.parquet")
        row["vehicle_collision_count"] = int(collisions["number_of_at_fault_collisions_with_vehicles_stat_value"].sum())
        row["pedestrian_collision_count"] = int(collisions["number_of_at_fault_collisions_with_VRUs_stat_value"].sum())
        row["all_collision_count"] = int(collisions["number_of_all_at_fault_collisions_stat_value"].sum())
        cache_root = run_root / ("b2_generated_cache" if stage == "B2" else f"stage_vector_caches/{stage}")
        contact_summary = _summarize_contact_targets(
            cache_root,
            run_root / "stage_simulations" / stage / "simulation",
        )
        row.update(contact_summary["counts"])
        save_json(run_root / "manifests" / f"{stage.lower()}_contact_targets.json", contact_summary)
        rows.append(row)
    output_dir = run_root / "visualizations/metric_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "stage_metrics.csv", rows)
    _plot_stage_metrics(rows, output_dir / "stage_metrics.png")
    _plot_contact_targets(rows, output_dir / "contact_targets.png")
    payload = {
        "schema_version": "occluded_pedestrian_stage_metrics_v1",
        "rows": rows,
        "metric_plot": str(output_dir / "stage_metrics.png"),
        "contact_target_plot": str(output_dir / "contact_targets.png"),
    }
    save_json(run_root / "manifests/stage_metrics_comparison.json", payload)
    return payload


def _summarize_contact_targets(scenario_cache: Path, simulation_root: Path) -> Dict[str, Any]:
    """Count geometric contacts by controlled identity across closed-loop logs.

    nuPlan's official at-fault metric reports broad object classes. This audit
    supplements it with track-token identity so an occluder contact cannot be
    mistaken for the intended pedestrian conflict.
    """
    labels = {path.parent.name: _read_json(path) for path in Path(scenario_cache).glob("**/scenario_label.json")}
    rows: List[Dict[str, Any]] = []
    scenario_category_sets: Dict[str, set] = defaultdict(set)
    for log_path in sorted(Path(simulation_root).glob("simulation_log/**/*.msgpack.xz")):
        log = SimulationLog.load_data(log_path)
        scenario_name = str(log.scenario.scenario_name)
        controlled = _controlled_tokens(labels.get(scenario_name, {}))
        controlled_pedestrian = next((token for token in controlled if token.startswith("1_")), None)
        controlled_occluder = next((token for token in controlled if not token.startswith("1_")), None)
        contacts: Dict[str, str] = {}
        for sample in log.simulation_history.data:
            ego_box = sample.ego_state.car_footprint.oriented_box
            for obj in sample.observation.tracked_objects.tracked_objects:
                token = str(obj.track_token)
                if token in contacts or not in_collision(ego_box, obj.box):
                    continue
                object_type = str(obj.tracked_object_type.name).lower()
                if token == controlled_pedestrian:
                    category = "controlled_pedestrian"
                elif token == controlled_occluder:
                    category = "controlled_occluder"
                elif object_type in {"pedestrian", "bicycle"}:
                    category = "other_vru"
                elif object_type == "vehicle":
                    category = "other_vehicle"
                else:
                    category = "other_object"
                contacts[token] = category
                scenario_category_sets[category].add(scenario_name)
        rows.append(
            {
                "scenario_name": scenario_name,
                "contact_tokens": sorted(contacts),
                "contact_categories": sorted(set(contacts.values())),
            }
        )
    categories = ("controlled_pedestrian", "controlled_occluder", "other_vru", "other_vehicle", "other_object")
    counts = {f"contact_scenarios_{name}": len(scenario_category_sets[name]) for name in categories}
    counts["contact_scenarios_any"] = sum(bool(row["contact_tokens"]) for row in rows)
    return {
        "schema_version": "occluded_pedestrian_contact_targets_v1",
        "num_logs": len(rows),
        "counts": counts,
        "rows": rows,
    }


def _save_simulation_trajectory(log: SimulationLog, controlled: Sequence[str], output: Path, *, stage: str) -> None:
    ego_xy, tracks, types = _history_tracks(log)
    fig, ax = plt.subplots(figsize=(10, 7))
    _draw_lines(ax, log.scenario._sledge_vector.lines)
    ax.plot(ego_xy[:, 0], ego_xy[:, 1], color="#1f77b4", linewidth=2.5, label="planned ego")
    for token, points in tracks.items():
        points_arr = np.asarray(points)
        highlight = token in controlled
        color = "#e45756" if token.startswith("1_") else "#f58518" if highlight else "#999999"
        ax.plot(points_arr[:, 0], points_arr[:, 1], color=color, linewidth=2.4 if highlight else 0.7, alpha=1.0 if highlight else 0.45)
        if highlight:
            ax.scatter(points_arr[0, 0], points_arr[0, 1], s=45, color=color, zorder=5)
            ax.text(points_arr[0, 0], points_arr[0, 1], f" {types[token]} {token}", fontsize=8)
    _set_history_limits(ax, ego_xy, tracks, controlled)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3)
    ax.set_title(f"{stage} closed-loop simulation trajectory")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _save_simulation_gif(log: SimulationLog, controlled: Sequence[str], output: Path, *, stage: str) -> None:
    ego_xy, tracks, _ = _history_tracks(log)
    frames: List[Image.Image] = []
    history = log.simulation_history.data
    indices = list(range(0, len(history), max(1, len(history) // 30)))
    if indices[-1] != len(history) - 1:
        indices.append(len(history) - 1)
    for frame_index in indices:
        fig, ax = plt.subplots(figsize=(8, 6))
        _draw_lines(ax, log.scenario._sledge_vector.lines)
        ax.plot(ego_xy[: frame_index + 1, 0], ego_xy[: frame_index + 1, 1], color="#1f77b4", linewidth=2.2)
        sample = history[frame_index]
        for obj in sample.observation.tracked_objects.tracked_objects:
            token = str(obj.track_token)
            highlight = token in controlled
            color = "#e45756" if token.startswith("1_") else "#f58518" if highlight else "#999999"
            corners = oriented_box_corners(obj.center.x, obj.center.y, obj.center.heading, obj.box.width, obj.box.length)
            ax.add_patch(MplPolygon(corners, closed=True, facecolor=color if highlight else "none", edgecolor=color, alpha=0.45 if highlight else 0.35))
        ego = sample.ego_state.center
        ax.scatter([ego.x], [ego.y], s=55, color="#1f77b4", zorder=6)
        _set_history_limits(ax, ego_xy, tracks, controlled)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3)
        ax.set_title(f"{stage} closed loop | t={frame_index * 0.1:.1f}s")
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
        frames.append(Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE))
        plt.close(fig)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=False)


def _history_tracks(log: SimulationLog):
    ego = []
    tracks: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    types: Dict[str, str] = {}
    for sample in log.simulation_history.data:
        ego.append((sample.ego_state.center.x, sample.ego_state.center.y))
        for obj in sample.observation.tracked_objects.tracked_objects:
            token = str(obj.track_token)
            tracks[token].append((obj.center.x, obj.center.y))
            types[token] = str(obj.tracked_object_type.name).lower()
    return np.asarray(ego), dict(tracks), types


def _set_history_limits(ax, ego_xy: np.ndarray, tracks: Mapping[str, Sequence[Tuple[float, float]]], controlled: Sequence[str]) -> None:
    arrays = [ego_xy]
    arrays.extend(np.asarray(tracks[token]) for token in controlled if token in tracks)
    points = np.concatenate(arrays, axis=0)
    ax.set_xlim(float(points[:, 0].min() - 8.0), float(points[:, 0].max() + 8.0))
    ax.set_ylim(float(points[:, 1].min() - 8.0), float(points[:, 1].max() + 8.0))


def _controlled_tokens(label: Mapping[str, Any]) -> List[str]:
    if label.get("controlled_tokens"):
        return [str(item) for item in label["controlled_tokens"]]
    protected = label.get("protected_slots", {})
    if not protected:
        return []
    ped = int(protected.get("pedestrians", -1))
    occ = int(protected.get("occluder_index", -1))
    occ_elem = str(protected.get("occluder_element", "vehicles"))
    return [f"1_{ped}", f"{0 if occ_elem == 'vehicles' else 2}_{occ}"]


def _match_processed_slot(raw_elem: Any, raw_index: int, vector_elem: Any) -> int:
    raw_states = np.asarray(raw_elem.states)
    if raw_index < 0 or raw_index >= len(raw_states):
        return -1
    states = np.asarray(vector_elem.states)
    masks = np.asarray(vector_elem.mask).reshape(-1) >= 0.3
    valid = np.where(masks)[0]
    width = min(5, states.shape[-1], raw_states.shape[-1])
    scales = np.asarray([1.0, 1.0, 0.5, 0.25, 0.25], dtype=np.float32)[:width]
    errors = np.linalg.norm((states[valid, :width] - raw_states[raw_index, :width]) * scales, axis=1)
    return int(valid[int(np.argmin(errors))])


def _b2_labels_by_sample(cache_root: Path) -> Dict[str, Path]:
    result = {}
    for label_path in Path(cache_root).glob("**/scenario_label.json"):
        label = _read_json(label_path)
        sample_id = Path(str(label.get("edited_scene_path", ""))).parent.name
        if sample_id:
            result[sample_id] = label_path
    return result


def _plot_stage_metrics(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    keys = ["collision_free_rate", "ttc_safe_rate", "comfort_rate", "progress_rate"]
    labels = ["collision-free", "TTC-safe", "comfortable", "progress"]
    x = np.arange(len(keys))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, row in enumerate(rows):
        ax.bar(x + (index - 1) * width, [float(row[key]) for key in keys], width=width, label=str(row["stage"]))
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("mean metric score")
    ax.set_title("B0 / B1 / B2 closed-loop comparison")
    ax.grid(axis="y", linewidth=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_contact_targets(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    keys = [
        "contact_scenarios_controlled_pedestrian",
        "contact_scenarios_controlled_occluder",
        "contact_scenarios_other_vru",
        "contact_scenarios_other_vehicle",
    ]
    labels = ["target pedestrian", "controlled occluder", "other VRU", "other vehicle"]
    x = np.arange(len(keys))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, row in enumerate(rows):
        values = [int(row[key]) for key in keys]
        bars = ax.bar(x + (index - 1) * width, values, width=width, label=str(row["stage"]))
        ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("scenarios with geometric contact")
    ax.set_title("B0 / B1 / B2 contact-target audit")
    ax.grid(axis="y", linewidth=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
