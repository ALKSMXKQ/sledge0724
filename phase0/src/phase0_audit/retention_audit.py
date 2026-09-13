from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


FRAME_WIDTH_M = 64.0
FRAME_HEIGHT_M = 64.0
CAPS = {
    "vehicles": 50,
    "pedestrians": 20,
    "static_objects": 30,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_cache_root(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    exp_root = os.getenv("SLEDGE_EXP_ROOT")
    if exp_root:
        candidates.append(Path(exp_root) / "caches" / "autoencoder_cache")
    candidates.append(_repo_root().parent / "exp" / "caches" / "autoencoder_cache")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not find autoencoder_cache; pass --root explicitly")


def reservoir_sample(paths: Iterable[Path], k: int, seed: int) -> tuple[list[Path], int]:
    rng = random.Random(seed)
    reservoir: list[Path] = []
    total = 0
    for total, path in enumerate(paths, start=1):
        if len(reservoir) < k:
            reservoir.append(path)
            continue
        j = rng.randrange(total)
        if j < k:
            reservoir[j] = path
    return reservoir, total


def load_gzip_pickle(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def coords_in_frame(states: np.ndarray) -> np.ndarray:
    if states.size == 0:
        return np.zeros(0, dtype=bool)
    x = states[:, 0]
    y = states[:, 1]
    return (
        (-FRAME_WIDTH_M / 2 <= x)
        & (x <= FRAME_WIDTH_M / 2)
        & (-FRAME_HEIGHT_M / 2 <= y)
        & (y <= FRAME_HEIGHT_M / 2)
    )


def analyze_actor_element(data: Mapping[str, Any], key: str, cap: int) -> dict[str, Any]:
    element = data[key]
    states = np.asarray(element["states"])
    raw_mask = np.asarray(element.get("mask", []))

    if states.size == 0:
        states = np.zeros((0, 6 if key != "static_objects" else 5), dtype=np.float32)
    if states.ndim != 2 or states.shape[1] < 2:
        raise ValueError(f"{key}.states expected [N,D], observed {states.shape}")

    raw_count = int(states.shape[0])
    frame_mask = coords_in_frame(states)
    in_frame_states = states[frame_mask]
    in_frame_count = int(len(in_frame_states))

    distances = np.linalg.norm(in_frame_states[:, :2], axis=-1) if in_frame_count else np.zeros(0)
    order = np.argsort(distances)[:cap]
    kept_count = int(len(order))
    dropped_outside = raw_count - in_frame_count
    dropped_by_cap = max(in_frame_count - cap, 0)

    nearest_dropped_distance = None
    if dropped_by_cap > 0:
        sorted_distances = np.sort(distances)
        nearest_dropped_distance = float(sorted_distances[cap])

    return {
        "raw_count": raw_count,
        "in_frame_count": in_frame_count,
        "kept_count": kept_count,
        "dropped_outside_frame": dropped_outside,
        "dropped_by_cap": dropped_by_cap,
        "raw_to_kept_fraction": float(kept_count / raw_count) if raw_count else 1.0,
        "in_frame_to_kept_fraction": float(kept_count / in_frame_count) if in_frame_count else 1.0,
        "nearest_cap_dropped_distance_m": nearest_dropped_distance,
        "raw_mask_true_count": int(raw_mask.sum()) if raw_mask.size else 0,
        "raw_mask_false_count": int(raw_mask.size - raw_mask.sum()) if raw_mask.size else 0,
    }


def _scenario_info(root: Path, path: Path) -> tuple[str, str, str]:
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) >= 4:
        return parts[-4], parts[-3], parts[-2]
    return "unknown_log", "unknown_type", path.parent.name


def summarize(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    rows = [record[key] for record in records if key in record]
    if not rows:
        return {}

    def arr(field: str) -> np.ndarray:
        return np.asarray([r[field] for r in rows], dtype=float)

    raw = arr("raw_count")
    in_frame = arr("in_frame_count")
    kept = arr("kept_count")
    outside = arr("dropped_outside_frame")
    cap_drop = arr("dropped_by_cap")

    return {
        "scene_count": len(rows),
        "nonempty_scene_count": int((raw > 0).sum()),
        "total_raw": int(raw.sum()),
        "total_in_frame": int(in_frame.sum()),
        "total_kept": int(kept.sum()),
        "total_dropped_outside_frame": int(outside.sum()),
        "total_dropped_by_cap": int(cap_drop.sum()),
        "scenes_with_outside_frame_drop": int((outside > 0).sum()),
        "scenes_with_cap_drop": int((cap_drop > 0).sum()),
        "scene_fraction_with_cap_drop": float((cap_drop > 0).mean()),
        "global_raw_to_kept_fraction": float(kept.sum() / raw.sum()) if raw.sum() else 1.0,
        "global_in_frame_to_kept_fraction": float(kept.sum() / in_frame.sum()) if in_frame.sum() else 1.0,
        "raw_count_quantiles": _quantiles(raw),
        "in_frame_count_quantiles": _quantiles(in_frame),
        "kept_count_quantiles": _quantiles(kept),
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "p00": float(np.quantile(values, 0.00)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "p100": float(np.quantile(values, 1.00)),
    }


def scenario_type_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["scenario_type"]].append(record)
    return {
        scenario_type: {
            "sample_count": len(rows),
            "pedestrians": summarize(rows, "pedestrians"),
            "vehicles": summarize(rows, "vehicles"),
            "static_objects": summarize(rows, "static_objects"),
        }
        for scenario_type, rows in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    }


def audit(root: Path, sample_size: int, seed: int, keep_examples: int) -> dict[str, Any]:
    sample_paths, total_raw_files = reservoir_sample(root.rglob("sledge_raw.gz"), sample_size, seed)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for path in sorted(sample_paths, key=str):
        log_name, scenario_type, token = _scenario_info(root, path)
        try:
            data = load_gzip_pickle(path)
            if not isinstance(data, Mapping):
                raise TypeError(f"expected dict-like sledge_raw, got {type(data)!r}")
            record: dict[str, Any] = {
                "path": str(path.relative_to(root)),
                "log_name": log_name,
                "scenario_type": scenario_type,
                "token": token,
            }
            for key, cap in CAPS.items():
                record[key] = analyze_actor_element(data, key, cap)
            records.append(record)
        except Exception as exc:
            failures.append({"path": str(path), "error": repr(exc)})

    pedestrian_cap_examples = sorted(
        [r for r in records if r["pedestrians"]["dropped_by_cap"] > 0],
        key=lambda r: r["pedestrians"]["dropped_by_cap"],
        reverse=True,
    )[:keep_examples]

    vehicle_cap_examples = sorted(
        [r for r in records if r["vehicles"]["dropped_by_cap"] > 0],
        key=lambda r: r["vehicles"]["dropped_by_cap"],
        reverse=True,
    )[:keep_examples]

    return {
        "status": "pass" if records and not failures else "attention_required",
        "cache_root": str(root),
        "sledge_raw_file_count_total": total_raw_files,
        "sample_size_requested": sample_size,
        "sample_size_loaded": len(records),
        "seed": seed,
        "processing_contract": {
            "frame_m": [FRAME_WIDTH_M, FRAME_HEIGHT_M],
            "frame_bounds_m": {"x": [-32.0, 32.0], "y": [-32.0, 32.0]},
            "selection": "filter_to_frame_then_sort_by_ego_distance_then_take_nearest_cap",
            "caps": CAPS,
            "raw_agent_mask_semantics": "dummy_all_false_not_used_for_agent_selection",
            "processed_agent_mask_semantics": "true_means_valid_kept_slot",
        },
        "aggregate": {
            "vehicles": summarize(records, "vehicles"),
            "pedestrians": summarize(records, "pedestrians"),
            "static_objects": summarize(records, "static_objects"),
        },
        "by_scenario_type": scenario_type_summary(records),
        "examples": {
            "largest_pedestrian_cap_drops": pedestrian_cap_examples,
            "largest_vehicle_cap_drops": vehicle_cap_examples,
        },
        "load_failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 0B retention audit: quantify information removed when raw B0 actors "
            "are converted to the fixed-size RVAE representation."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=9102026)
    parser.add_argument("--keep-examples", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("phase0/outputs/retention_audit.json"))
    args = parser.parse_args()

    root = resolve_cache_root(args.root)
    report = audit(
        root=root,
        sample_size=max(1, args.sample_size),
        seed=args.seed,
        keep_examples=max(1, args.keep_examples),
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")

    if report["load_failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
