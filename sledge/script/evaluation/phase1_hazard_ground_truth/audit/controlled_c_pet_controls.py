"""Controlled counterfactual validation for the Phase-1 C/PET evaluator.

Implementation-validation only. It does not estimate real-world prevalence or
real-data sensitivity. Source XY trajectories are preserved exactly; only a
constant pedestrian timestamp shift is applied. The independent oracle is
frozen before the public C/PET evaluator is imported or executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from shapely.geometry import LineString

DEFAULT_INPUT_DIR = Path(
    "sledge/script/evaluation/phase1_hazard_ground_truth/outputs/"
    "c_positive_control_2000_seed0_all"
)
DEFAULT_OUTPUT_DIR = Path(
    "sledge/script/evaluation/phase1_hazard_ground_truth/outputs/controlled_c_pet"
)
DEFAULT_MAP_VERSION = "nuplan-maps-v1.0"
DEFAULT_DURATION_S = 15.0
DEFAULT_EXTRACTION_OFFSET_S = -3.0
RADIUS_M = 1.0
MAX_PET_S = 2.0
TOL = 1e-6
POSITIVE_PETS = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
NEGATIVE_PETS = (2.50, 3.00, 3.50, 4.00)
BOUNDARY_PETS = (1.99, 2.01)


@dataclass(frozen=True)
class SourceGeometry:
    scenario_id: str
    pedestrian_id: str
    scenario_type: str
    crossing_angle_deg: float
    intersection_xy: tuple[float, float]
    ego_arrival_s: float
    pedestrian_arrival_s: float
    ego_interval_s: tuple[float, float]
    pedestrian_interval_s: tuple[float, float]


@dataclass(frozen=True)
class Control:
    control_id: str
    group: str
    source_scenario_id: str
    source_pedestrian_id: str
    source_scenario_type: str
    crossing_angle_deg: float
    geometry_unchanged: bool
    pedestrian_time_shift_s: float
    target_pet_s: float
    intersection_xy: tuple[float, float]
    ego_arrival_s: float
    pedestrian_arrival_s: float
    ego_interval_s: tuple[float, float]
    pedestrian_interval_s: tuple[float, float]
    independent_pet_s: float
    expected_C: bool
    provenance: str = "CONTROLLED_COUNTERFACTUAL"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _single_intersection(ego_xy: np.ndarray, ped_xy: np.ndarray) -> np.ndarray:
    origin = np.asarray(ego_xy[0], dtype=float)
    g = LineString(np.asarray(ego_xy) - origin).intersection(
        LineString(np.asarray(ped_xy) - origin)
    )
    if g.geom_type != "Point":
        raise ValueError(f"expected one Point intersection, got {g.geom_type}")
    return np.asarray(g.coords[0], dtype=float) + origin


def _arrival_and_tangent(
    t: Sequence[float], xy: Sequence[Sequence[float]], point: Sequence[float]
) -> tuple[float, float, np.ndarray]:
    t = np.asarray(t, dtype=float)
    xy = np.asarray(xy, dtype=float)
    point = np.asarray(point, dtype=float)
    best_d = float("inf")
    best_t = float(t[0])
    best_v: np.ndarray | None = None
    for i in range(len(t) - 1):
        p0, p1 = xy[i], xy[i + 1]
        v = p1 - p0
        vv = float(v @ v)
        if vv <= 1e-15:
            u, tangent = 0.0, None
        else:
            u = float(np.clip(((point - p0) @ v) / vv, 0.0, 1.0))
            tangent = v / math.sqrt(vv)
        q = p0 + u * v
        d = float(np.linalg.norm(point - q))
        if d < best_d:
            best_d = d
            best_t = float(t[i] + u * (t[i + 1] - t[i]))
            best_v = tangent
    if best_v is None:
        raise ValueError("cannot determine non-zero path tangent")
    return best_t, best_d, best_v


def _circle_intervals(
    t: Sequence[float], xy: Sequence[Sequence[float]], center: Sequence[float], radius: float
) -> list[tuple[float, float]]:
    t = np.asarray(t, dtype=float)
    xy = np.asarray(xy, dtype=float)
    c0 = np.asarray(center, dtype=float)
    out: list[tuple[float, float]] = []
    r2 = radius * radius
    for i in range(len(t) - 1):
        p = xy[i] - c0
        v = xy[i + 1] - xy[i]
        a = float(v @ v)
        b = 2.0 * float(p @ v)
        c = float(p @ p) - r2
        if a <= 1e-15:
            if c <= 0:
                out.append((float(t[i]), float(t[i + 1])))
            continue
        disc = b * b - 4 * a * c
        if disc < 0:
            if c <= 0:
                out.append((float(t[i]), float(t[i + 1])))
            continue
        root = math.sqrt(max(0.0, disc))
        u0, u1 = sorted(((-b - root) / (2 * a), (-b + root) / (2 * a)))
        lo, hi = max(0.0, u0), min(1.0, u1)
        if lo <= hi:
            dt = float(t[i + 1] - t[i])
            out.append((float(t[i] + lo * dt), float(t[i] + hi * dt)))
    if not out:
        return []
    out.sort()
    merged = [out[0]]
    for a, b in out[1:]:
        x, y = merged[-1]
        if a <= y + 1e-9:
            merged[-1] = (x, max(y, b))
        else:
            merged.append((a, b))
    return merged


def _nearest_interval(
    intervals: Sequence[tuple[float, float]], arrival: float
) -> tuple[float, float] | None:
    if not intervals:
        return None
    def d(z: tuple[float, float]) -> float:
        a, b = z
        return 0.0 if a <= arrival <= b else min(abs(arrival - a), abs(arrival - b))
    return min(intervals, key=d)


def _pet(a: tuple[float, float], b: tuple[float, float]) -> float:
    a0, a1 = a
    b0, b1 = b
    if max(a0, b0) <= min(a1, b1):
        return 0.0
    return float(b0 - a1) if a1 < b0 else float(a0 - b1)


def select_diverse_source_records(
    rows: Sequence[Mapping[str, Any]], count: int = 6
) -> list[dict[str, Any]]:
    eligible = [
        dict(r) for r in rows
        if bool(r.get("raw_path_intersection_exists"))
        and r.get("raw_path_intersection_geometry_type") == "Point"
        and len(r.get("raw_path_intersection_points") or []) == 1
        and r.get("crossing_angle_deg") is not None
        and not bool(r.get("possible_right_censoring"))
    ]
    eligible.sort(key=lambda r: (float(r["crossing_angle_deg"]), str(r["scenario_id"]), str(r["pedestrian_id"])))
    if len(eligible) < count:
        raise ValueError(f"need {count} eligible sources, found {len(eligible)}")
    idx = [round(i * (len(eligible) - 1) / (count - 1)) for i in range(count)]
    return [eligible[i] for i in idx]


def _source_geometry(scenario: Any, pedestrian_id: str, scenario_type: str) -> SourceGeometry:
    track = scenario.pedestrian_tracks[pedestrian_id]
    if track.skip:
        raise ValueError(f"skipped track: {track.skip_reason}")
    ego, ped = scenario.ego_trajectory, track.trajectory()
    point = _single_intersection(ego.positions_xy, ped.positions_xy)
    et, ee, ev = _arrival_and_tangent(ego.timestamps_s, ego.positions_xy, point)
    pt, pe, pv = _arrival_and_tangent(ped.timestamps_s, ped.positions_xy, point)
    if max(ee, pe) > TOL:
        raise ValueError(f"intersection projection error too large: {ee}, {pe}")
    ei = _nearest_interval(_circle_intervals(ego.timestamps_s, ego.positions_xy, point, RADIUS_M), et)
    pi = _nearest_interval(_circle_intervals(ped.timestamps_s, ped.positions_xy, point, RADIUS_M), pt)
    if ei is None or pi is None:
        raise ValueError("missing independent occupancy interval")
    angle = float(np.degrees(np.arccos(np.clip(float(ev @ pv), -1.0, 1.0))))
    return SourceGeometry(
        str(scenario.scenario_id), pedestrian_id, scenario_type, angle,
        (float(point[0]), float(point[1])), float(et), float(pt),
        tuple(map(float, ei)), tuple(map(float, pi)),
    )


def _make_control(source: SourceGeometry, group: str, ordinal: int, target: float) -> Control:
    shift = float(source.ego_interval_s[1] + target - source.pedestrian_interval_s[0])
    ped_interval = (
        source.pedestrian_interval_s[0] + shift,
        source.pedestrian_interval_s[1] + shift,
    )
    pet = _pet(source.ego_interval_s, ped_interval)
    if not math.isclose(pet, target, rel_tol=0.0, abs_tol=TOL):
        raise AssertionError((target, pet))
    return Control(
        f"source{ordinal:02d}_{group}_pet_{str(target).replace('.', 'p')}", group,
        source.scenario_id, source.pedestrian_id, source.scenario_type,
        source.crossing_angle_deg, True, shift, float(target), source.intersection_xy,
        source.ego_arrival_s, source.pedestrian_arrival_s + shift,
        source.ego_interval_s, ped_interval, pet, bool(pet <= MAX_PET_S),
    )


def _infer_map_root(manifest: Mapping[str, Any]) -> Path:
    sources = list(manifest.get("scenario_sources", []))
    if not sources:
        raise ValueError("review manifest contains no scenario_sources")
    log_db = Path(str(sources[0]["log_db"]))
    return log_db.parents[3] / "maps"


def _build_controls(
    input_dir: Path, map_root: Path, map_version: str, duration_s: float, extraction_offset_s: float
) -> tuple[list[Control], dict[tuple[str, str], tuple[Any, Any]], list[SourceGeometry]]:
    from sledge.script.evaluation.phase1_hazard_ground_truth.audit.interface_audit import build_real_scenario
    from sledge.script.evaluation.phase1_hazard_ground_truth.audit.scenario_adapter import adapt_scenario

    candidates = _load_jsonl(input_dir / "candidates.jsonl")
    manifest = json.loads((input_dir / "review_manifest.json").read_text())
    chosen = select_diverse_source_records(candidates, 6)
    source_map = {str(s["scenario_id"]): dict(s) for s in manifest["scenario_sources"]}
    geoms: list[SourceGeometry] = []
    cache: dict[tuple[str, str], tuple[Any, Any]] = {}
    for row in chosen:
        sid, pid = str(row["scenario_id"]), str(row["pedestrian_id"])
        src = source_map[sid]
        scenario = adapt_scenario(build_real_scenario(
            Path(str(src["log_db"])), sid, str(src["scenario_type"]), map_root,
            map_version, duration_s, extraction_offset_s,
        ))
        geom = _source_geometry(scenario, pid, str(src["scenario_type"]))
        if not math.isclose(geom.crossing_angle_deg, float(row["crossing_angle_deg"]), abs_tol=1e-5):
            raise ValueError(f"crossing-angle drift for {(sid, pid)}")
        geoms.append(geom)
        cache[(sid, pid)] = (scenario.ego_trajectory, scenario.pedestrian_tracks[pid].trajectory())

    controls = [_make_control(g, "main_positive", i, p) for i, (g, p) in enumerate(zip(geoms, POSITIVE_PETS), 1)]
    controls += [_make_control(g, "main_negative", i, p) for i, (g, p) in enumerate(zip(geoms[:4], NEGATIVE_PETS), 1)]
    controls += [_make_control(geoms[0], "boundary", i, p) for i, p in enumerate(BOUNDARY_PETS, 1)]
    return controls, cache, geoms


def _freeze_oracle(output_dir: Path, input_dir: Path, controls: Sequence[Control], sources: Sequence[SourceGeometry]) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "phase1_controlled_c_pet_oracle_v1",
        "purpose": "implementation validation only; not prevalence or real sensitivity estimation",
        "source_input_dir": str(input_dir),
        "source_selection": "six deterministic crossing-angle quantiles; no C/PET output used",
        "counterfactual_operation": "constant pedestrian timestamp shift only; all XY unchanged",
        "oracle_contract": {"conflict_region_radius_m": RADIUS_M, "max_pet_s": MAX_PET_S, "private_conflict_helpers_used": False},
        "main_gate": {"positive_target_pets_s": list(POSITIVE_PETS), "negative_target_pets_s": list(NEGATIVE_PETS), "pet_abs_error_tolerance_s": TOL},
        "boundary_controls_excluded_from_main_gate": list(BOUNDARY_PETS),
        "source_geometries": [asdict(x) for x in sources],
        "controls": [asdict(x) for x in controls],
    }
    path = output_dir / "oracle_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path, sha256_file(path)


def _evaluate(controls: Sequence[Control], cache: Mapping[tuple[str, str], tuple[Any, Any]]) -> list[dict[str, Any]]:
    from sledge.script.evaluation.phase1_hazard_ground_truth.conflict import compute_spatiotemporal_conflict
    from sledge.script.evaluation.phase1_hazard_ground_truth.types import TimedTrajectory2D

    rows = []
    for c in controls:
        ego, ped = cache[(c.source_scenario_id, c.source_pedestrian_id)]
        shifted = TimedTrajectory2D(ped.timestamps_s + c.pedestrian_time_shift_s, ped.positions_xy.copy())
        r = compute_spatiotemporal_conflict(ego, shifted)
        pet = None if r.PET is None else float(r.PET)
        center_err = radius_err = None
        if r.conflict_region is not None:
            center_err = float(np.linalg.norm(np.asarray(r.conflict_region.center_xy) - np.asarray(c.intersection_xy)))
            radius_err = abs(float(r.conflict_region.radius_m) - RADIUS_M)
        rows.append({
            "control_id": c.control_id, "group": c.group,
            "source_scenario_id": c.source_scenario_id, "source_pedestrian_id": c.source_pedestrian_id,
            "target_pet_s": c.target_pet_s, "pedestrian_time_shift_s": c.pedestrian_time_shift_s,
            "expected_C": c.expected_C, "actual_C": bool(r.conflict_exists),
            "independent_PET_s": c.independent_pet_s, "evaluator_PET_s": pet,
            "PET_abs_error_s": None if pet is None else abs(pet - c.independent_pet_s),
            "conflict_center_abs_error_m": center_err, "conflict_radius_abs_error_m": radius_err,
            "evaluator_arrival_time_gap_s": r.arrival_time_gap,
            "evaluator_minimum_distance_m": r.minimum_distance,
        })
    return rows


def _summary(rows: Sequence[Mapping[str, Any]], sha: str, unchanged: bool) -> dict[str, Any]:
    main = [r for r in rows if r["group"] != "boundary"]
    boundary = [r for r in rows if r["group"] == "boundary"]
    cm = {"TP": 0, "FN": 0, "TN": 0, "FP": 0}
    for r in main:
        e, a = bool(r["expected_C"]), bool(r["actual_C"])
        cm["TP" if e and a else "FN" if e else "FP" if a else "TN"] += 1
    pos = [r for r in main if r["expected_C"]]
    finite = sum(r["evaluator_PET_s"] is not None and math.isfinite(float(r["evaluator_PET_s"])) for r in pos)
    errs = [float(r["PET_abs_error_s"]) for r in rows if r["PET_abs_error_s"] is not None]
    geometry_ok = all(
        r["conflict_center_abs_error_m"] is not None and float(r["conflict_center_abs_error_m"]) <= TOL
        and r["conflict_radius_abs_error_m"] is not None and float(r["conflict_radius_abs_error_m"]) <= TOL
        for r in rows
    )
    max_err = max(errs) if errs else None
    gate = cm == {"TP": 6, "FN": 0, "TN": 4, "FP": 0} and finite == 6 and max_err is not None and max_err <= TOL and geometry_ok and unchanged
    return {
        "status": "PASS" if gate else "ATTENTION_REQUIRED",
        "purpose": "controlled counterfactual C/PET implementation validation; not prevalence",
        "oracle_manifest_sha256": sha,
        "oracle_manifest_unchanged_after_evaluator": unchanged,
        "main_control_counts": {"positive": 6, "negative": 4, "boundary_excluded": len(boundary)},
        "confusion_matrix_main_gate": cm,
        "finite_pet_rate_on_expected_positives": finite / 6.0,
        "PET_mean_abs_error_s": None if not errs else float(np.mean(errs)),
        "PET_max_abs_error_s": max_err,
        "geometry_match_with_oracle": geometry_ok,
        "boundary_expected_match": all(bool(r["expected_C"]) == bool(r["actual_C"]) for r in boundary),
        "boundary_results": boundary,
        "main_gate_verdict": "PASS" if gate else "FAIL",
        "thresholds_tuned": False,
        "core_conflict_code_modified": False,
        "prevalence_claim": False,
    }


def run_validation(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    map_root: Path | None = None,
    map_version: str = DEFAULT_MAP_VERSION,
    duration_s: float = DEFAULT_DURATION_S,
    extraction_offset_s: float = DEFAULT_EXTRACTION_OFFSET_S,
) -> dict[str, Any]:
    manifest = json.loads((input_dir / "review_manifest.json").read_text())
    map_root = map_root or _infer_map_root(manifest)
    controls, cache, sources = _build_controls(input_dir, map_root, map_version, duration_s, extraction_offset_s)
    oracle_path, sha = _freeze_oracle(output_dir, input_dir, controls, sources)
    rows = _evaluate(controls, cache)  # evaluator runs only after oracle freeze
    unchanged = sha256_file(oracle_path) == sha
    results_path = output_dir / "validation_results.jsonl"
    results_path.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows))
    summary = _summary(rows, sha, unchanged)
    summary.update({"oracle_manifest": str(oracle_path), "validation_results": str(results_path), "map_root": str(map_root), "map_version": map_version})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--map-root", type=Path, default=None)
    p.add_argument("--map-version", default=DEFAULT_MAP_VERSION)
    p.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    p.add_argument("--extraction-offset-s", type=float, default=DEFAULT_EXTRACTION_OFFSET_S)
    a = p.parse_args()
    summary = run_validation(a.input_dir, a.output_dir, a.map_root, a.map_version, a.duration_s, a.extraction_offset_s)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    if summary["main_gate_verdict"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
