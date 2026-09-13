from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXPECTED_LAST_DIMS = {
    "vehicles": 6,
    "pedestrians": 6,
    "static_objects": 5,
    "ego": 4,
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
    rendered = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find autoencoder cache. Checked:\n  - " + rendered +
        "\nPass --root explicitly if your cache is elsewhere."
    )


def find_token_dirs(root: Path) -> list[Path]:
    # A nuPlan/SLEDGE cache sample directory contains one or more *.gz feature files.
    token_dirs = {p.parent for p in root.rglob("*.gz") if p.is_file()}
    return sorted(token_dirs, key=lambda p: str(p))


def load_gzip_pickle(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def array_summary(value: Any) -> dict[str, Any]:
    arr = np.asarray(value)
    out: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
    }
    if arr.size and np.issubdtype(arr.dtype, np.number):
        finite = np.isfinite(arr)
        out["finite_fraction"] = float(finite.mean())
        if finite.any():
            vals = arr[finite]
            out.update(
                min=float(vals.min()),
                max=float(vals.max()),
                mean=float(vals.mean()),
                std=float(vals.std()),
            )
    if arr.dtype == np.bool_:
        out["true_count"] = int(arr.sum())
        out["false_count"] = int(arr.size - arr.sum())
    return out


def structure_summary(value: Any, depth: int = 0, max_depth: int = 4) -> Any:
    if depth > max_depth:
        return {"type": type(value).__name__, "truncated": True}
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", **array_summary(value)}
    if isinstance(value, Mapping):
        return {
            "type": type(value).__name__,
            "keys": sorted(map(str, value.keys())),
            "items": {str(k): structure_summary(v, depth + 1, max_depth) for k, v in value.items()},
        }
    if isinstance(value, (list, tuple)):
        preview = list(value[:5]) if isinstance(value, list) else list(value[:5])
        return {
            "type": type(value).__name__,
            "len": len(value),
            "items": [structure_summary(v, depth + 1, max_depth) for v in preview],
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"type": type(value).__name__, "value": value}
    if hasattr(value, "__dict__"):
        public = {k: v for k, v in vars(value).items() if not k.startswith("_")}
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "attrs": structure_summary(public, depth + 1, max_depth),
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)[:200]}


def _element_dict(container: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    value = container.get(key)
    return value if isinstance(value, Mapping) else None


def check_sledge_vector_dict(data: Mapping[str, Any], *, raw: bool) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for key, expected_dim in EXPECTED_LAST_DIMS.items():
        element = _element_dict(data, key)
        if element is None or "states" not in element:
            continue
        states = np.asarray(element["states"])
        actual = int(states.shape[-1]) if states.ndim else None
        checks.append({
            "name": f"{key}.states_last_dim",
            "ok": actual == expected_dim,
            "expected": expected_dim,
            "observed": actual,
        })
        if "mask" in element:
            mask = np.asarray(element["mask"])
            checks.append({
                "name": f"{key}.mask_dtype_bool",
                "ok": mask.dtype == np.bool_,
                "expected": "bool",
                "observed": str(mask.dtype),
            })

    lines = _element_dict(data, "lines")
    if lines is not None and "states" in lines:
        states = np.asarray(lines["states"])
        expected_line_dim = 3 if raw else 2
        actual = int(states.shape[-1]) if states.ndim else None
        checks.append({
            "name": "lines.states_last_dim",
            "ok": actual == expected_line_dim,
            "expected": expected_line_dim,
            "observed": actual,
        })
    return checks


def check_latent_dict(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for key in ("mu", "log_var"):
        if key not in data:
            checks.append({"name": f"latent.{key}_present", "ok": False, "expected": True, "observed": False})
            continue
        arr = np.asarray(data[key])
        shape = list(arr.shape)
        checks.append({
            "name": f"latent.{key}_shape",
            "ok": shape == [64, 8, 8],
            "expected": [64, 8, 8],
            "observed": shape,
        })
        checks.append({
            "name": f"latent.{key}_finite",
            "ok": bool(np.isfinite(arr).all()),
            "expected": True,
            "observed": bool(np.isfinite(arr).all()),
        })
    return checks


def infer_checks(filename: str, data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, Mapping):
        return []
    stem = filename[:-3] if filename.endswith(".gz") else filename
    if stem == "sledge_raw":
        return check_sledge_vector_dict(data, raw=True)
    if stem == "sledge_vector":
        return check_sledge_vector_dict(data, raw=False)
    if stem in {"rvae_latent", "latent"} or ("mu" in data and "log_var" in data):
        return check_latent_dict(data)
    return []


def audit_cache(root: Path, max_samples: int) -> dict[str, Any]:
    token_dirs = find_token_dirs(root)
    selected = token_dirs[:max_samples]
    samples: list[dict[str, Any]] = []
    all_checks: list[dict[str, Any]] = []
    feature_file_counts: dict[str, int] = {}

    for token_dir in selected:
        rel = token_dir.relative_to(root)
        sample: dict[str, Any] = {"token_dir": str(rel), "files": {}}
        for file_path in sorted(token_dir.glob("*.gz")):
            feature_file_counts[file_path.name] = feature_file_counts.get(file_path.name, 0) + 1
            try:
                data = load_gzip_pickle(file_path)
                checks = infer_checks(file_path.name, data)
                all_checks.extend({"sample": str(rel), "file": file_path.name, **c} for c in checks)
                sample["files"][file_path.name] = {
                    "load_ok": True,
                    "structure": structure_summary(data),
                    "checks": checks,
                }
            except Exception as exc:
                sample["files"][file_path.name] = {
                    "load_ok": False,
                    "error": repr(exc),
                }
        samples.append(sample)

    failed_checks = [c for c in all_checks if not c["ok"]]
    load_failures = [
        {"sample": s["token_dir"], "file": name, "error": payload.get("error")}
        for s in samples
        for name, payload in s["files"].items()
        if not payload.get("load_ok", False)
    ]

    return {
        "status": "pass" if selected and not failed_checks and not load_failures else "attention_required",
        "cache_root": str(root),
        "token_dir_count_total": len(token_dirs),
        "token_dir_count_sampled": len(selected),
        "feature_file_counts_in_sample": dict(sorted(feature_file_counts.items())),
        "contract_check_count": len(all_checks),
        "contract_check_failures": failed_checks,
        "load_failures": load_failures,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0B: inspect real SLEDGE autoencoder cache without importing nuPlan")
    parser.add_argument("--root", type=Path, default=None, help="Path to autoencoder_cache")
    parser.add_argument("--max-samples", type=int, default=12, help="Number of token directories to inspect")
    parser.add_argument("--out", type=Path, default=Path("phase0/outputs/cache_audit.json"))
    args = parser.parse_args()

    root = resolve_cache_root(args.root)
    report = audit_cache(root, max_samples=max(1, args.max_samples))
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")

    # Do not fail merely because a feature is absent from a sampled token. We fail only
    # on a concrete load/contract mismatch. This keeps Phase 0B diagnostic rather than brittle.
    if report["contract_check_failures"] or report["load_failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
