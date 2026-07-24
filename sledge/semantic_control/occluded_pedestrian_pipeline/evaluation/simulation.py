"""Closed-loop simulation launch and metric summarization."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

import pandas as pd

from sledge.semantic_control.io import save_json


def build_simulation_command(
    *,
    repo_root: Path,
    scenario_cache: Path,
    planner: str,
    limit: int,
    simulation_output: Path | None = None,
) -> List[str]:
    simulation_output = simulation_output or (scenario_cache.parent / "simulation")
    return [
        sys.executable,
        str(Path(__file__).with_name("simulation_entrypoint.py")),
        "+simulation=sledge_reactive_agents",
        f"planner={planner}",
        "observation=sledge_agents_observation",
        "+observation.stationary_vehicle_speed_threshold=0.1",
        "scenario_builder=nuplan",
        f"cache.scenario_cache_path={scenario_cache}",
        f"output_dir={simulation_output}",
        "worker=sequential",
        "number_of_cpus_allocated_per_simulation=1",
        "number_of_gpus_allocated_per_simulation=0",
        f"scenario_filter.limit_total_scenarios={limit}",
        "run_metric=true",
        # Generated SLEDGE maps can legitimately represent overlapping lane
        # candidates at intersections. nuPlan's speed-limit metric asserts a
        # unique lane assignment and aborts the whole callback otherwise.
        # It is unrelated to this hazard study, so retain the safety/dynamics
        # metrics and remove only that incompatible metric.
        "~simulation_metric.high_level.speed_limit_compliance_statistics",
        "enable_simulation_progress_bar=true",
    ]


def run_simulation(
    *,
    repo_root: Path,
    scenario_cache: Path,
    output_manifest: Path,
    planner: str = "pdm_closed_planner",
    limit: int = 100,
    dry_run: bool = False,
    simulation_output: Path | None = None,
) -> Dict[str, Any]:
    final_output = (
        Path(simulation_output).resolve()
        if simulation_output is not None
        else scenario_cache.parent / "simulation"
    )
    command = build_simulation_command(
        repo_root=repo_root,
        scenario_cache=scenario_cache,
        planner=planner,
        limit=limit,
        simulation_output=final_output,
    )
    payload: Dict[str, Any] = {
        "schema_version": "occluded_pedestrian_simulation_launch_v1",
        "scenario_cache": str(scenario_cache),
        "planner": planner,
        "limit": limit,
        "command": command,
        "dry_run": dry_run,
        "simulation_output": str(final_output),
    }
    save_json(output_manifest, payload)
    if not dry_run:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["SLEDGE_RANDOM_SAMPLE"] = str(max(0, int(limit)))
        env["SLEDGE_RANDOM_SEED"] = "0"
        # nuPlan's asynchronous NuBoard writer can block on this NFS mount.
        # Run on local /tmp and copy the complete structured result back.
        with tempfile.TemporaryDirectory(prefix="occluded_pedestrian_sim_") as temp_dir:
            execution_output = Path(temp_dir) / "simulation"
            execution_command = build_simulation_command(
                repo_root=repo_root,
                scenario_cache=scenario_cache,
                planner=planner,
                limit=limit,
                simulation_output=execution_output,
            )
            payload["execution_command"] = execution_command
            payload["temporary_execution_output"] = str(execution_output)
            completed = subprocess.run(execution_command, cwd=repo_root, check=False, env=env)
            if execution_output.exists():
                final_output.mkdir(parents=True, exist_ok=True)
                shutil.copytree(execution_output, final_output, dirs_exist_ok=True)
        payload["return_code"] = int(completed.returncode)
        payload["results_copied_to"] = str(final_output)
        if completed.returncode == 0:
            metric_summary = summarize_simulation_directory(
                final_output / "metrics",
                output_manifest.parent / "simulation_summary.json",
            )
            payload["metric_summary"] = metric_summary
            available = sum(1 for _ in Path(scenario_cache).glob("**/sledge_vector.gz"))
            expected = min(max(0, int(limit)), available)
            payload["expected_scenarios"] = expected
            payload["completed_scenarios"] = int(metric_summary["num_scenarios"])
            if int(metric_summary["num_scenarios"]) != expected:
                payload["completion_status"] = "incomplete"
                save_json(output_manifest, payload)
                raise RuntimeError(
                    "Simulation process exited successfully but produced metrics for "
                    f"{metric_summary['num_scenarios']}/{expected} scenarios; inspect "
                    f"{final_output / 'log.txt'}"
                )
            payload["completion_status"] = "complete"
        save_json(output_manifest, payload)
        if completed.returncode != 0:
            raise RuntimeError(f"Simulation failed with return code {completed.returncode}")
    return payload


def summarize_simulation_metrics(parquet_path: Path, output_json: Path) -> Dict[str, Any]:
    frame = pd.read_parquet(parquet_path)
    numeric = frame.select_dtypes(include="number")
    means = {str(key): float(value) for key, value in numeric.mean(numeric_only=True).items()}
    finite_counts = {str(key): int(value) for key, value in numeric.notna().sum().items()}
    payload = {
        "schema_version": "occluded_pedestrian_simulation_summary_v1",
        "metrics_parquet": str(parquet_path),
        "num_rows": int(len(frame)),
        "numeric_means": means,
        "finite_counts": finite_counts,
        "columns": [str(column) for column in frame.columns],
    }
    save_json(output_json, payload)
    return payload


def summarize_simulation_directory(metrics_dir: Path, output_json: Path) -> Dict[str, Any]:
    metric_files = sorted(Path(metrics_dir).glob("*.parquet"))
    metrics: Dict[str, Any] = {}
    scenario_names = set()
    for path in metric_files:
        frame = pd.read_parquet(path)
        if "scenario_name" in frame:
            scenario_names.update(str(value) for value in frame["scenario_name"].dropna())
        scores = pd.to_numeric(frame.get("metric_score"), errors="coerce")
        stat_means: Dict[str, float] = {}
        for column in frame.columns:
            if not str(column).endswith("_stat_value"):
                continue
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if len(values):
                stat_means[str(column)] = float(values.mean())
        metrics[path.stem] = {
            "num_rows": int(len(frame)),
            "mean_metric_score": float(scores.dropna().mean()) if len(scores.dropna()) else None,
            "stat_means": stat_means,
        }
    payload = {
        "schema_version": "occluded_pedestrian_simulation_directory_summary_v1",
        "metrics_dir": str(metrics_dir),
        "num_metric_files": len(metric_files),
        "num_scenarios": len(scenario_names),
        "metrics": metrics,
    }
    save_json(output_json, payload)
    return payload
