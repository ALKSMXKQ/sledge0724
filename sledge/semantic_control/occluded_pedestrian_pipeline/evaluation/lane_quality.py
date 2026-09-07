"""Automatic lane-geometry quality metrics for generated SLEDGE scenes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from sledge.diffusion.modelling.lane_geometry_guidance import (
    infer_adjacency_pairs,
    polyline_quality,
    resample_polyline,
)


@dataclass(frozen=True)
class LaneQualityThresholds:
    heading_jump_max_rad: float = 0.32
    curvature_max_per_m: float = 0.28
    curvature_change_mean: float = 0.12
    adjacent_heading_difference_rad: float = 0.25
    lane_width_min_m: float = 2.4
    lane_width_max_m: float = 5.2
    lane_width_std_max_m: float = 0.90
    max_self_intersections: int = 0
    max_boundary_crossings: int = 0


@dataclass(frozen=True)
class LaneQualityResult:
    passed: bool
    metrics: Dict[str, Any]
    checks: Dict[str, bool]
    thresholds: Dict[str, Any]
    issues: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lane_geometry_pass": self.passed,
            "metrics": self.metrics,
            "checks": self.checks,
            "thresholds": self.thresholds,
            "issues": self.issues,
        }


def evaluate_lane_quality(
    scene_or_vector: Any,
    *,
    topology_family: str = "road_segment",
    thresholds: LaneQualityThresholds | None = None,
    adjacency_pairs: Sequence[Mapping[str, Any]] | None = None,
) -> LaneQualityResult:
    thresholds = thresholds or LaneQualityThresholds()
    lines = getattr(scene_or_vector, "lines", scene_or_vector)
    states = np.asarray(getattr(lines, "states", []), dtype=np.float64)
    mask = np.asarray(getattr(lines, "mask", [])).reshape(-1)
    valid = []
    if states.ndim == 3:
        for i in range(min(len(states), len(mask))):
            if float(mask[i]) >= 0.3 and states.shape[1] >= 3:
                pts = states[i, :, :2]
                pts = pts[np.all(np.isfinite(pts), axis=1)]
                if len(pts) >= 3:
                    valid.append((i, pts))

    qualities = [polyline_quality(p) for _, p in valid]
    finite = lambda xs: [float(x) for x in xs if np.isfinite(x)]
    heading_values = finite([q["heading_jump_max"] for q in qualities])
    curvature_values = finite([q["curvature_max"] for q in qualities])
    jitter_values = finite([q["curvature_change"] for q in qualities])
    heading_max = max(heading_values, default=float("nan"))
    curvature_max = max(curvature_values, default=float("nan"))
    curvature_change = float(np.mean(jitter_values)) if jitter_values else float("nan")

    self_intersections = sum(_self_intersection_count(p) for _, p in valid)
    pair_info = list(adjacency_pairs or [])
    if not pair_info and states.ndim == 3:
        pair_info = infer_adjacency_pairs(states, mask)

    widths: List[float] = []
    width_stds: List[float] = []
    pair_heading_diffs: List[float] = []
    boundary_crossings = 0
    by_idx = {i: p for i, p in valid}
    for pair in pair_info:
        i, j = int(pair["a"]), int(pair["b"])
        if i not in by_idx or j not in by_idx:
            continue
        a, b = by_idx[i], by_idx[j]
        if bool(pair.get("reverse_b", False)):
            b = b[::-1]
        n = max(4, min(len(a), len(b)))
        aa = _resample_count(a, n)
        bb = _resample_count(b, n)
        w = np.linalg.norm(aa - bb, axis=1)
        widths.append(float(np.mean(w)))
        width_stds.append(float(np.std(w)))
        ha = np.unwrap(np.arctan2(np.diff(aa, axis=0)[:, 1], np.diff(aa, axis=0)[:, 0]))
        hb = np.unwrap(np.arctan2(np.diff(bb, axis=0)[:, 1], np.diff(bb, axis=0)[:, 0]))
        d = np.arctan2(np.sin(ha - hb), np.cos(ha - hb))
        pair_heading_diffs.append(float(np.mean(np.abs(d))))
        boundary_crossings += _between_intersection_count(aa, bb)

    metrics = {
        "active_line_count": len(valid),
        "lane_heading_jump_max": heading_max,
        "lane_curvature_max": curvature_max,
        "lane_curvature_change": curvature_change,
        "lane_width_mean": float(np.mean(widths)) if widths else None,
        "lane_width_std": float(np.mean(width_stds)) if width_stds else None,
        "adjacent_heading_difference": float(np.mean(pair_heading_diffs)) if pair_heading_diffs else None,
        "boundary_crossing_count": int(boundary_crossings),
        "polyline_self_intersection_count": int(self_intersections),
        "lane_gap_count": 0,
        "adjacency_pair_count": len(pair_info),
    }

    checks = {
        "has_road_lines": len(valid) > 0,
        "heading_continuity": bool(np.isfinite(heading_max) and heading_max <= thresholds.heading_jump_max_rad),
        "curvature_reasonable": bool(np.isfinite(curvature_max) and curvature_max <= thresholds.curvature_max_per_m),
        "curvature_continuity": bool(np.isfinite(curvature_change) and curvature_change <= thresholds.curvature_change_mean),
        "no_self_intersection": self_intersections <= thresholds.max_self_intersections,
        "no_boundary_crossing": boundary_crossings <= thresholds.max_boundary_crossings,
    }
    # Pairwise constraints are meaningful only when actual/confident pairs exist.
    if pair_heading_diffs and topology_family != "intersection":
        checks["adjacent_heading_consistency"] = metrics["adjacent_heading_difference"] <= thresholds.adjacent_heading_difference_rad
    if widths and topology_family != "intersection":
        checks["lane_width_range"] = thresholds.lane_width_min_m <= metrics["lane_width_mean"] <= thresholds.lane_width_max_m
        checks["lane_width_continuity"] = metrics["lane_width_std"] <= thresholds.lane_width_std_max_m

    issues = [name for name, ok in checks.items() if not ok]
    return LaneQualityResult(
        passed=all(checks.values()),
        metrics=metrics,
        checks=checks,
        thresholds=asdict(thresholds),
        issues=issues,
    )


def _resample_count(points: np.ndarray, n: int) -> np.ndarray:
    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 1e-6:
        return np.repeat(points[:1], n, axis=0)
    target = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(target, s, points[:, 0]), np.interp(target, s, points[:, 1])], axis=1)


def _orientation(a, b, c) -> float:
    return float(np.cross(b - a, c - a))


def _segments_intersect(a, b, c, d, eps=1e-8) -> bool:
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    return (o1 * o2 < -eps) and (o3 * o4 < -eps)


def _self_intersection_count(points: np.ndarray) -> int:
    count = 0
    for i in range(len(points) - 1):
        for j in range(i + 2, len(points) - 1):
            if j == i + 1:
                continue
            if _segments_intersect(points[i], points[i + 1], points[j], points[j + 1]):
                count += 1
    return count


def _between_intersection_count(a: np.ndarray, b: np.ndarray) -> int:
    count = 0
    for i in range(len(a) - 1):
        for j in range(len(b) - 1):
            if _segments_intersect(a[i], a[i + 1], b[j], b[j + 1]):
                count += 1
    return count


__all__ = ["LaneQualityThresholds", "LaneQualityResult", "evaluate_lane_quality"]
