from __future__ import annotations

"""Phase 2: unbiased audit of the existing B0 -> B1 hazard-anchor builder.

The protocol deliberately separates *selection* from *generation*:

1. ``freeze`` chooses N unique real B0 scenes before the editor is run.
2. ``run`` executes the existing B0->B1 pipeline exactly once for every frozen
   case, keeps failures in the denominator, rolls each produced B1 forward with
   an explicit constant-velocity adapter, and evaluates the frozen Phase-1
   contract H = O & E & C & T.

No accepted-only resampling is allowed in this audit.
"""

import argparse
import csv
from dataclasses import asdict, fields
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    AgentIndex,
    EgoIndex,
    StaticObjectIndex,
)
from sledge.script.evaluation.phase1_hazard_ground_truth import (
    OccludedPedestrianContract,
    TimedTrajectory2D,
    compute_visibility,
)
from sledge.script.evaluation.phase1_hazard_ground_truth.scene_types import (
    ActorState,
    ActorType,
)
from sledge.semantic_control.io import load_raw_scene
from sledge.semantic_control.occluded_pedestrian_pipeline.experiment_matrix import (
    ExperimentCase,
    build_experiment_cases,
    load_matrix_config,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.pipeline import (
    OccludedPedestrianPipeline,
)


LABEL_THRESHOLD = 0.3
DEFAULT_N = 50
DEFAULT_SEED = 20260915
DEFAULT_HORIZON_S = 6.0
DEFAULT_DT_S = 0.2
TARGET_RATE = 0.90
IDEAL_RATE = 0.95


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(payload), stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _case_from_dict(payload: Dict[str, Any]) -> ExperimentCase:
    kwargs = {}
    for item in fields(ExperimentCase):
        if item.name in payload:
            kwargs[item.name] = payload[item.name]
    return ExperimentCase(**kwargs)


def freeze_b0_manifest(
    *,
    input_root: Path,
    matrix_config_path: Path,
    profile: str,
    output_manifest: Path,
    n: int = DEFAULT_N,
    glob_pattern: str = "**/sledge_raw.gz",
) -> Dict[str, Any]:
    """Freeze N unique B0s *before* any editor/evaluator acceptance is observed."""
    if not 20 <= int(n) <= 50:
        raise ValueError("Phase-2 audit requires N in [20, 50]; N=50 is recommended")

    matrix_config = load_matrix_config(matrix_config_path)
    # Ask for extra cases because legacy profiles may schedule repeated source files.
    candidates = build_experiment_cases(
        input_root=Path(input_root),
        profile=profile,
        matrix_config=matrix_config,
        glob_pattern=glob_pattern,
        max_cases=max(int(n) * 5, int(n)),
    )

    selected: List[ExperimentCase] = []
    seen_b0 = set()
    for case in candidates:
        key = str(Path(case.input_raw).resolve())
        if key in seen_b0:
            continue
        seen_b0.add(key)
        selected.append(case)
        if len(selected) == int(n):
            break

    if len(selected) != int(n):
        raise RuntimeError(
            f"Could only freeze {len(selected)} unique B0 scenes; requested {n}. "
            "Use a larger real-data root/profile rather than resampling accepted B1s."
        )

    rows = [
        {
            **case.to_dict(),
            "phase2_selection_index": index,
            "phase2_selection_locked": True,
        }
        for index, case in enumerate(selected)
    ]
    _write_jsonl(output_manifest, rows)
    meta = {
        "schema_version": "phase2_b0_frozen_manifest_v1",
        "n": len(rows),
        "profile": profile,
        "input_root": str(Path(input_root).resolve()),
        "matrix_config": str(Path(matrix_config_path).resolve()),
        "glob_pattern": glob_pattern,
        "selection_policy": "first_N_unique_cases_from_deterministic_experiment_matrix",
        "accepted_only_filtering": False,
        "editor_observed_before_selection": False,
        "recommended_n": DEFAULT_N,
    }
    _write_json(Path(output_manifest).with_suffix(".meta.json"), meta)
    return meta


def _mask_valid(mask: Any, index: int) -> bool:
    arr = np.asarray(mask).reshape(-1)
    if index < 0 or index >= len(arr):
        return False
    value = arr[index]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return bool(float(value) >= LABEL_THRESHOLD)


def _agent_actor(scene: Any, element: str, index: int, track_id: str, actor_type: ActorType) -> ActorState:
    elem = getattr(scene, element)
    states = np.asarray(elem.states)
    if states.ndim == 1:
        states = states[None, :]
    if index < 0 or index >= len(states) or not _mask_valid(elem.mask, index):
        raise IndexError(f"Invalid {element}[{index}] in B1")
    state = np.asarray(states[index], dtype=np.float64)
    speed = float(state[AgentIndex.VELOCITY])
    heading = float(state[AgentIndex.HEADING])
    velocity = np.array([speed * math.cos(heading), speed * math.sin(heading)], dtype=np.float64)
    return ActorState(
        track_id=track_id,
        actor_type=actor_type,
        position_xy=np.array([state[AgentIndex.X], state[AgentIndex.Y]], dtype=np.float64),
        heading_rad=heading,
        velocity_xy=velocity,
        length_m=float(max(state[AgentIndex.LENGTH], 0.2)),
        width_m=float(max(state[AgentIndex.WIDTH], 0.2)),
        source_index=int(index),
        metadata={"source_element": element},
    )


def _static_actor(scene: Any, index: int, track_id: str) -> ActorState:
    elem = scene.static_objects
    states = np.asarray(elem.states)
    if states.ndim == 1:
        states = states[None, :]
    if index < 0 or index >= len(states) or not _mask_valid(elem.mask, index):
        raise IndexError(f"Invalid static_objects[{index}] in B1")
    state = np.asarray(states[index], dtype=np.float64)
    return ActorState(
        track_id=track_id,
        actor_type=ActorType.STATIC,
        position_xy=np.array([state[StaticObjectIndex.X], state[StaticObjectIndex.Y]], dtype=np.float64),
        heading_rad=float(state[StaticObjectIndex.HEADING]),
        velocity_xy=np.zeros(2, dtype=np.float64),
        length_m=float(max(state[StaticObjectIndex.LENGTH], 0.2)),
        width_m=float(max(state[StaticObjectIndex.WIDTH], 0.2)),
        source_index=int(index),
        metadata={"source_element": "static_objects"},
    )


def _ego_actor(scene: Any) -> ActorState:
    state = np.asarray(scene.ego.states, dtype=np.float64).reshape(-1)
    if len(state) < 2:
        raise ValueError(f"Unexpected ego state shape: {np.asarray(scene.ego.states).shape}")
    velocity = np.array([state[EgoIndex.VELOCITY_X], state[EgoIndex.VELOCITY_Y]], dtype=np.float64)
    return ActorState(
        track_id="ego",
        actor_type=ActorType.EGO,
        position_xy=np.zeros(2, dtype=np.float64),
        heading_rad=0.0,
        velocity_xy=velocity,
        length_m=4.5,
        width_m=2.0,
        metadata={"rollout_frame": "B1 ego-local t0"},
    )


def _all_occluders(scene: Any) -> List[ActorState]:
    output: List[ActorState] = []
    for element, actor_type in (("vehicles", ActorType.VEHICLE),):
        elem = getattr(scene, element)
        states = np.asarray(elem.states)
        if states.ndim == 1:
            states = states[None, :]
        for idx in range(len(states)):
            if _mask_valid(elem.mask, idx):
                output.append(_agent_actor(scene, element, idx, f"vehicle:{idx}", actor_type))
    elem = scene.static_objects
    states = np.asarray(elem.states)
    if states.ndim == 1:
        states = states[None, :]
    for idx in range(len(states)):
        if _mask_valid(elem.mask, idx):
            output.append(_static_actor(scene, idx, f"static:{idx}"))
    return output


def _propagate(actor: ActorState, t_s: float) -> ActorState:
    return ActorState(
        track_id=actor.track_id,
        actor_type=actor.actor_type,
        position_xy=actor.position_xy + actor.velocity_xy * float(t_s),
        heading_rad=actor.heading_rad,
        velocity_xy=actor.velocity_xy,
        length_m=actor.length_m,
        width_m=actor.width_m,
        valid=actor.valid,
        timestamp_s=float(t_s),
        source_index=actor.source_index,
        metadata=dict(actor.metadata),
    )


def evaluate_b1_hazard(
    scene: Any,
    edit_result: Dict[str, Any],
    *,
    horizon_s: float = DEFAULT_HORIZON_S,
    dt_s: float = DEFAULT_DT_S,
    contract: OccludedPedestrianContract | None = None,
) -> Dict[str, Any]:
    """Evaluate one produced B1 through the frozen Phase-1 O/E/C/T contract.

    SLEDGE B1 is a single-frame semantic vector. The adapter therefore uses an
    explicit constant-velocity world rollout. This is an adapter assumption,
    not a redefinition of H, and is surfaced in every output row.
    """
    if horizon_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("horizon_s and dt_s must be positive")
    contract = contract or OccludedPedestrianContract()

    ped_index = int(edit_result.get("pedestrian_index", -1))
    pedestrian = _agent_actor(scene, "pedestrians", ped_index, f"pedestrian:{ped_index}", ActorType.PEDESTRIAN)
    ego = _ego_actor(scene)
    occluders = _all_occluders(scene)

    timestamps = np.arange(0.0, float(horizon_s) + 0.5 * float(dt_s), float(dt_s), dtype=np.float64)
    ego_positions = []
    ped_positions = []
    visibility = []
    dominant_blockers = []

    for t_s in timestamps:
        ego_t = _propagate(ego, float(t_s))
        ped_t = _propagate(pedestrian, float(t_s))
        occ_t = [_propagate(actor, float(t_s)) for actor in occluders]
        visibility_result = compute_visibility(ego_t, ped_t, occ_t)
        ego_positions.append(ego_t.position_xy)
        ped_positions.append(ped_t.position_xy)
        visibility.append(float(visibility_result.visibility_fraction))
        dominant_blockers.append(visibility_result.occluding_object)

    result = contract.verify(
        visibility_fractions=visibility,
        visibility_timestamps_s=timestamps,
        ego_trajectory=TimedTrajectory2D(timestamps, np.asarray(ego_positions)),
        pedestrian_trajectory=TimedTrajectory2D(timestamps, np.asarray(ped_positions)),
    )

    primary_occluder = None
    occ_elem = str(edit_result.get("occluder_elem_name", "vehicles"))
    occ_index = int(edit_result.get("occluder_index", -1))
    if occ_index >= 0:
        primary_occluder = f"{'static' if occ_elem == 'static_objects' else 'vehicle'}:{occ_index}"

    return {
        "O": bool(result.O),
        "E": bool(result.E),
        "C": bool(result.C),
        "T": bool(result.T),
        "H": bool(result.H),
        "failure_reason": result.failure_reason.value,
        "reveal_time_s": result.reveal.reveal_time_s,
        "conflict_time_s": result.conflict.conflict_time_s,
        "PET_s": result.conflict.PET,
        "minimum_distance_m": float(result.conflict.minimum_distance),
        "RTTC_s": result.timing.rttc_s if result.timing is not None else None,
        "timing_class": result.timing.timing_class.value if result.timing is not None else None,
        "visibility_fractions": visibility,
        "dominant_blockers": dominant_blockers,
        "primary_editor_occluder": primary_occluder,
        "adapter": {
            "name": "sledge_single_frame_constant_velocity_v1",
            "horizon_s": float(horizon_s),
            "dt_s": float(dt_s),
            "coordinate_frame": "B1 ego-local at t=0, fixed frame rollout",
            "phase1_contract_modified": False,
        },
        "phase1_diagnostics": _jsonable(result.diagnostics),
    }


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> List[float]:
    if n <= 0:
        return [0.0, 0.0]
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize_audit(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    built = [row for row in rows if row.get("builder_status") == "built"]
    n_built = len(built)
    rates_built = {
        key: (sum(bool(row.get(key)) for row in built) / n_built if n_built else 0.0)
        for key in ("O", "E", "C", "T", "H")
    }
    rates_all = {
        key: (sum(bool(row.get(key)) for row in rows) / n if n else 0.0)
        for key in ("O", "E", "C", "T", "H")
    }
    h_successes = sum(bool(row.get("H")) for row in rows)
    failures: Dict[str, int] = {}
    for row in rows:
        reason = str(row.get("failure_reason") or row.get("builder_status") or "unknown")
        failures[reason] = failures.get(reason, 0) + 1

    end_to_end = h_successes / n if n else 0.0
    conditional = sum(bool(row.get("H")) for row in built) / n_built if n_built else 0.0
    return {
        "schema_version": "phase2_b1_hazard_audit_summary_v1",
        "N_selected_B0": n,
        "N_B1_built": n_built,
        "builder_yield": n_built / n if n else 0.0,
        "stage_table_built_only": {
            "stage": "B1",
            "N": n_built,
            **rates_built,
        },
        "stage_table_all_selected": {
            "stage": "B0->B1 end-to-end",
            "N": n,
            **rates_all,
        },
        "P_H_given_B1_built": conditional,
        "P_H_end_to_end": end_to_end,
        "H_count_end_to_end": h_successes,
        "H_95pct_wilson_end_to_end": _wilson_interval(h_successes, n),
        "failure_counts": dict(sorted(failures.items(), key=lambda item: (-item[1], item[0]))),
        "gate": {
            "metric": "P_H_end_to_end",
            "target": TARGET_RATE,
            "ideal_strictly_greater_than": IDEAL_RATE,
            "pass_target": bool(end_to_end >= TARGET_RATE),
            "pass_ideal": bool(end_to_end > IDEAL_RATE),
            "decision": (
                "PROCEED_TO_DIFFUSION" if end_to_end >= TARGET_RATE else "FIX_B1_BUILDER_BEFORE_DIFFUSION"
            ),
        },
        "protocol": {
            "accepted_only_filtering": False,
            "failed_B1_removed_from_denominator": False,
            "one_pipeline_call_per_frozen_B0": True,
        },
    }


def run_frozen_audit(
    *,
    manifest: Path,
    output_root: Path,
    audit_output: Path,
    horizon_s: float = DEFAULT_HORIZON_S,
    dt_s: float = DEFAULT_DT_S,
    strict_check: bool = True,
    save_visuals: bool = False,
) -> Dict[str, Any]:
    frozen_rows = _read_jsonl(manifest)
    if not 20 <= len(frozen_rows) <= 50:
        raise ValueError(f"Frozen manifest must contain 20..50 rows, got {len(frozen_rows)}")
    if len({str(row.get('input_raw')) for row in frozen_rows}) != len(frozen_rows):
        raise ValueError("Frozen manifest contains duplicate B0 input_raw paths")

    pipeline = OccludedPedestrianPipeline(
        Path(output_root),
        strict_check=bool(strict_check),
        save_visuals=bool(save_visuals),
    )
    contract = OccludedPedestrianContract()
    rows: List[Dict[str, Any]] = []

    for index, payload in enumerate(frozen_rows, start=1):
        case = _case_from_dict(payload)
        print(f"[phase2 {index}/{len(frozen_rows)}] {case.sample_id}")
        base = {
            "selection_index": int(payload.get("phase2_selection_index", index - 1)),
            "sample_id": case.sample_id,
            "condition_id": case.condition_id,
            "input_raw": case.input_raw,
            "source_relative_path": case.source_relative_path,
        }
        try:
            summary = pipeline.run_case(case)
            b1_raw = Path(summary["b1_raw"])
            scene, _ = load_raw_scene(b1_raw)
            edit_result = _read_json(Path(summary["artifact_root"]) / "03_editing" / "edit_result.json")
            hazard = evaluate_b1_hazard(
                scene,
                edit_result,
                horizon_s=horizon_s,
                dt_s=dt_s,
                contract=contract,
            )
            row = {
                **base,
                "builder_status": "built",
                "legacy_b1_pass": bool(summary.get("b1_pass", False)),
                "legacy_b1_semantic_satisfaction_rate": float(summary.get("b1_semantic_satisfaction_rate", 0.0)),
                "b1_raw": str(b1_raw),
                **hazard,
            }
        except Exception as exc:
            row = {
                **base,
                "builder_status": "builder_or_adapter_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "O": False,
                "E": False,
                "C": False,
                "T": False,
                "H": False,
                "failure_reason": "builder_or_adapter_error",
            }
        rows.append(row)
        print(
            f"  status={row['builder_status']} "
            f"O={int(bool(row['O']))} E={int(bool(row['E']))} "
            f"C={int(bool(row['C']))} T={int(bool(row['T']))} H={int(bool(row['H']))}"
        )

    audit_output = Path(audit_output)
    _write_jsonl(audit_output, rows)
    summary = summarize_audit(rows)
    _write_json(audit_output.with_suffix(".summary.json"), summary)

    csv_path = audit_output.with_suffix(".csv")
    fieldnames = [
        "selection_index", "sample_id", "condition_id", "input_raw", "builder_status",
        "legacy_b1_pass", "O", "E", "C", "T", "H", "failure_reason",
        "reveal_time_s", "conflict_time_s", "PET_s", "RTTC_s", "timing_class",
        "minimum_distance_m", "b1_raw", "error_type", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase-2 unbiased B1 hazard-anchor audit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="Freeze 20-50 unique real B0 cases before editing")
    freeze.add_argument("--input-root", type=Path, required=True)
    freeze.add_argument("--matrix-config", type=Path, required=True)
    freeze.add_argument("--profile", required=True)
    freeze.add_argument("--output-manifest", type=Path, required=True)
    freeze.add_argument("--n", type=int, default=DEFAULT_N)
    freeze.add_argument("--glob-pattern", default="**/sledge_raw.gz")

    run = subparsers.add_parser("run", help="Run each frozen B0 exactly once and evaluate H(B1)")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--audit-output", type=Path, required=True)
    run.add_argument("--horizon-s", type=float, default=DEFAULT_HORIZON_S)
    run.add_argument("--dt-s", type=float, default=DEFAULT_DT_S)
    run.add_argument("--no-strict-check", action="store_true")
    run.add_argument("--save-visuals", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "freeze":
        meta = freeze_b0_manifest(
            input_root=args.input_root,
            matrix_config_path=args.matrix_config,
            profile=args.profile,
            output_manifest=args.output_manifest,
            n=args.n,
            glob_pattern=args.glob_pattern,
        )
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        return 0

    summary = run_frozen_audit(
        manifest=args.manifest,
        output_root=args.output_root,
        audit_output=args.audit_output,
        horizon_s=args.horizon_s,
        dt_s=args.dt_s,
        strict_check=not args.no_strict_check,
        save_visuals=args.save_visuals,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
