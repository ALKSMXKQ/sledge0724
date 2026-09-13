from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from sledge.script.evaluation.phase1_hazard_ground_truth.scene_types import ActorState
from sledge.script.evaluation.phase1_hazard_ground_truth.types import OcclusionType, VisibilityResult


@dataclass(frozen=True)
class VisibilityConfig:
    samples_per_axis: int = 5
    distance_epsilon_m: float = 1e-6

    def __post_init__(self) -> None:
        if self.samples_per_axis < 1:
            raise ValueError("samples_per_axis must be >= 1")
        if self.distance_epsilon_m < 0:
            raise ValueError("distance_epsilon_m must be >= 0")


def _actor_polygon(actor: ActorState) -> Polygon:
    if actor.length_m <= 0.0 or actor.width_m <= 0.0:
        raise ValueError(f"actor {actor.track_id!r} must have positive length_m/width_m")
    hl, hw = 0.5 * actor.length_m, 0.5 * actor.width_m
    local = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]], dtype=np.float64)
    c, s = np.cos(actor.heading_rad), np.sin(actor.heading_rad)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    world = local @ rotation.T + actor.position_xy
    return Polygon(world)


def _sample_actor_footprint(actor: ActorState, samples_per_axis: int) -> np.ndarray:
    if actor.length_m <= 0.0 or actor.width_m <= 0.0:
        return np.asarray(actor.position_xy, dtype=np.float64).reshape(1, 2)
    xs = np.linspace(-0.5 * actor.length_m, 0.5 * actor.length_m, samples_per_axis)
    ys = np.linspace(-0.5 * actor.width_m, 0.5 * actor.width_m, samples_per_axis)
    local = np.asarray([(x, y) for x in xs for y in ys], dtype=np.float64)
    c, s = np.cos(actor.heading_rad), np.sin(actor.heading_rad)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rotation.T + actor.position_xy


def compute_visibility(
    ego: ActorState,
    pedestrian: ActorState,
    occluders: Iterable[ActorState],
    config: VisibilityConfig | None = None,
) -> VisibilityResult:
    """Compute deterministic 2-D BEV line-of-sight visibility of a pedestrian footprint.

    Each target sample is visible iff no occluder intersects the finite segment from
    the ego observer point to that sample *before* the sample itself. Therefore an
    object behind the pedestrian cannot cause occlusion.
    """
    config = config or VisibilityConfig()
    if not ego.valid or not pedestrian.valid:
        raise ValueError("ego and pedestrian must be valid")

    observer = np.asarray(ego.position_xy, dtype=np.float64)
    observer_point = Point(observer)
    samples = _sample_actor_footprint(pedestrian, config.samples_per_axis)

    occluder_polygons: list[tuple[str, Polygon]] = []
    for obj in occluders:
        if not obj.valid or obj.track_id in {ego.track_id, pedestrian.track_id}:
            continue
        if obj.length_m <= 0.0 or obj.width_m <= 0.0:
            continue
        occluder_polygons.append((obj.track_id, _actor_polygon(obj)))

    blocker_counts: Counter[str] = Counter()
    blocked = 0
    for sample in samples:
        target_distance = float(np.linalg.norm(sample - observer))
        if target_distance <= config.distance_epsilon_m:
            continue
        ray = LineString([observer, sample])
        nearest_blocker: str | None = None
        nearest_distance = float("inf")
        for track_id, polygon in occluder_polygons:
            intersection = ray.intersection(polygon)
            if intersection.is_empty:
                continue
            intersection_distance = float(observer_point.distance(intersection))
            if intersection_distance < target_distance - config.distance_epsilon_m:
                if intersection_distance < nearest_distance:
                    nearest_distance = intersection_distance
                    nearest_blocker = track_id
        if nearest_blocker is not None:
            blocked += 1
            blocker_counts[nearest_blocker] += 1

    total = int(len(samples))
    visibility_fraction = float((total - blocked) / total) if total else 1.0
    if blocked == 0:
        occlusion_type = OcclusionType.VISIBLE
    elif blocked == total:
        occlusion_type = OcclusionType.FULLY_OCCLUDED
    else:
        occlusion_type = OcclusionType.PARTIALLY_OCCLUDED

    dominant = None
    if blocker_counts:
        dominant = sorted(blocker_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    return VisibilityResult(
        visibility_fraction=visibility_fraction,
        occluding_object=dominant,
        occlusion_type=occlusion_type,
        blocked_samples=blocked,
        total_samples=total,
    )
