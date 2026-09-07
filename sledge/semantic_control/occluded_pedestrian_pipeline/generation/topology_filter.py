"""Coarse, read-only B0 topology classification and filtering.

The classifier intentionally exposes only broad families that can be recovered
from existing SLEDGE line geometry.  It never edits, snaps, synthesizes or
otherwise mutates a source scene.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, List, Sequence

import numpy as np

TOPOLOGY_ANY = "any"
TOPOLOGY_ROAD_SEGMENT = "road_segment"
TOPOLOGY_INTERSECTION = "intersection"
TOPOLOGY_MERGE_SPLIT = "merge_split"
SUPPORTED_TOPOLOGY_FAMILIES = {
    TOPOLOGY_ANY,
    TOPOLOGY_ROAD_SEGMENT,
    TOPOLOGY_INTERSECTION,
    TOPOLOGY_MERGE_SPLIT,
}


@dataclass(frozen=True)
class TopologyClassification:
    family: str
    confidence: float
    active_line_count: int
    orientation_clusters: int
    local_crossings: int
    convergence_pairs: int

    def matches(self, requested: str) -> bool:
        requested = normalize_topology_family(requested)
        return requested == TOPOLOGY_ANY or self.family == requested


def normalize_topology_family(value: str | None) -> str:
    value = str(value or TOPOLOGY_ANY).strip().lower().replace("-", "_")
    aliases = {
        "straight": TOPOLOGY_ROAD_SEGMENT,
        "straight_segment": TOPOLOGY_ROAD_SEGMENT,
        "curve": TOPOLOGY_ROAD_SEGMENT,
        "curved": TOPOLOGY_ROAD_SEGMENT,
        "road": TOPOLOGY_ROAD_SEGMENT,
        "junction": TOPOLOGY_INTERSECTION,
        "crossing": TOPOLOGY_INTERSECTION,
        "merge": TOPOLOGY_MERGE_SPLIT,
        "split": TOPOLOGY_MERGE_SPLIT,
    }
    value = aliases.get(value, value)
    if value not in SUPPORTED_TOPOLOGY_FAMILIES:
        raise ValueError(
            f"Unsupported topology_family={value!r}; expected "
            f"{sorted(SUPPORTED_TOPOLOGY_FAMILIES)}"
        )
    return value


def valid_polylines(scene: Any) -> List[np.ndarray]:
    elem = getattr(scene, "lines", None)
    if elem is None:
        return []
    states = np.asarray(getattr(elem, "states", []), dtype=np.float64)
    mask = np.asarray(getattr(elem, "mask", []))
    if states.ndim < 3 or states.shape[-1] < 2:
        return []
    output: List[np.ndarray] = []
    for idx in range(states.shape[0]):
        if mask.size:
            if mask.ndim == 1 and (idx >= len(mask) or float(mask[idx]) < 0.3):
                continue
            if mask.ndim > 1:
                row = np.asarray(mask[idx]).reshape(-1)
                valid = row[: states.shape[1]] >= 0.3
            else:
                valid = np.ones(states.shape[1], dtype=bool)
        else:
            valid = np.ones(states.shape[1], dtype=bool)
        pts = states[idx, valid, :2]
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) >= 3:
            output.append(pts)
    return output


def classify_topology(scene: Any, *, local_radius_m: float = 35.0) -> TopologyClassification:
    """Classify B0 from line geometry only; ``scene`` is never modified."""
    lines = valid_polylines(scene)
    local = [_local_part(line, local_radius_m) for line in lines]
    local = [line for line in local if len(line) >= 3]
    if not local:
        return TopologyClassification(TOPOLOGY_ROAD_SEGMENT, 0.25, 0, 0, 0, 0)

    headings = [_dominant_heading(line) for line in local]
    clusters = _orientation_cluster_count(headings)
    crossings = 0
    convergence = 0
    for i in range(len(local)):
        for j in range(i + 1, len(local)):
            if _polyline_intersects(local[i], local[j]):
                crossings += 1
            elif _looks_convergent(local[i], local[j]):
                convergence += 1

    # Intersections are characterized by genuinely different local directions
    # and crossing line geometry. Curved parallel roads stay road_segment.
    if clusters >= 2 and crossings >= 1:
        family = TOPOLOGY_INTERSECTION
        confidence = min(1.0, 0.60 + 0.08 * crossings + 0.08 * (clusters - 2))
    elif crossings == 0 and convergence >= 2 and clusters <= 2:
        family = TOPOLOGY_MERGE_SPLIT
        confidence = min(0.90, 0.55 + 0.08 * convergence)
    else:
        family = TOPOLOGY_ROAD_SEGMENT
        confidence = 0.80 if clusters <= 1 and crossings == 0 else 0.60

    return TopologyClassification(
        family=family,
        confidence=float(confidence),
        active_line_count=len(local),
        orientation_clusters=int(clusters),
        local_crossings=int(crossings),
        convergence_pairs=int(convergence),
    )


def scene_matches_topology(scene: Any, requested: str | None) -> bool:
    requested = normalize_topology_family(requested)
    return requested == TOPOLOGY_ANY or classify_topology(scene).family == requested


def _local_part(line: np.ndarray, radius: float) -> np.ndarray:
    keep = np.linalg.norm(line[:, :2], axis=1) <= radius
    pts = line[keep]
    return pts if len(pts) >= 3 else line


def _dominant_heading(line: np.ndarray) -> float:
    delta = line[-1] - line[0]
    angle = math.atan2(float(delta[1]), float(delta[0]))
    # Undirected lane-marking orientation lives in [0, pi).
    return angle % math.pi


def _orientation_cluster_count(headings: Sequence[float], threshold_rad: float = 0.45) -> int:
    if not headings:
        return 0
    clusters: List[float] = []
    for h in sorted(headings):
        if not any(_undirected_angle_difference(h, c) <= threshold_rad for c in clusters):
            clusters.append(h)
    return len(clusters)


def _undirected_angle_difference(a: float, b: float) -> float:
    d = abs((a - b) % math.pi)
    return min(d, math.pi - d)


def _segments(polyline: np.ndarray):
    for i in range(len(polyline) - 1):
        yield polyline[i], polyline[i + 1]


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _segment_intersection(a, b, c, d, eps: float = 1e-8) -> bool:
    ab = b - a
    cd = d - c
    den = _cross(ab, cd)
    if abs(den) <= eps:
        return False
    ac = c - a
    t = _cross(ac, cd) / den
    u = _cross(ac, ab) / den
    # Ignore touching endpoints; those are common at map-fragment boundaries.
    return eps < t < 1.0 - eps and eps < u < 1.0 - eps


def _polyline_intersects(a: np.ndarray, b: np.ndarray) -> bool:
    for p0, p1 in _segments(a):
        for q0, q1 in _segments(b):
            if _segment_intersection(p0, p1, q0, q1):
                return True
    return False


def _looks_convergent(a: np.ndarray, b: np.ndarray) -> bool:
    if len(a) < 3 or len(b) < 3:
        return False
    ha, hb = _dominant_heading(a), _dominant_heading(b)
    if _undirected_angle_difference(ha, hb) > 0.35:
        return False
    start = min(np.linalg.norm(a[0] - b[0]), np.linalg.norm(a[0] - b[-1]))
    end = min(np.linalg.norm(a[-1] - b[-1]), np.linalg.norm(a[-1] - b[0]))
    far, near = max(start, end), min(start, end)
    return far >= 4.0 and near <= 1.8 and (far - near) >= 2.0


__all__ = [
    "TopologyClassification",
    "SUPPORTED_TOPOLOGY_FAMILIES",
    "TOPOLOGY_ANY",
    "TOPOLOGY_ROAD_SEGMENT",
    "TOPOLOGY_INTERSECTION",
    "TOPOLOGY_MERGE_SPLIT",
    "classify_topology",
    "normalize_topology_family",
    "scene_matches_topology",
    "valid_polylines",
]
