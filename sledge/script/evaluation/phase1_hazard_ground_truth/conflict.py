from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPoint, Point
from shapely.ops import nearest_points

from sledge.script.evaluation.phase1_hazard_ground_truth.types import ConflictRegion, ConflictResult, TimedTrajectory2D


@dataclass(frozen=True)
class ConflictConfig:
    spatial_path_tolerance_m: float = 0.25
    conflict_region_radius_m: float = 1.0
    max_pet_s: float = 2.0

    def __post_init__(self) -> None:
        if self.spatial_path_tolerance_m < 0.0:
            raise ValueError("spatial_path_tolerance_m must be >= 0")
        if self.conflict_region_radius_m <= 0.0:
            raise ValueError("conflict_region_radius_m must be > 0")
        if self.max_pet_s < 0.0:
            raise ValueError("max_pet_s must be >= 0")


def _candidate_points(geometry) -> list[np.ndarray]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Point):
        return [np.asarray(geometry.coords[0], dtype=np.float64)]
    if isinstance(geometry, MultiPoint):
        return [np.asarray(g.coords[0], dtype=np.float64) for g in geometry.geoms]
    if isinstance(geometry, LineString):
        if geometry.length == 0.0:
            return [np.asarray(geometry.coords[0], dtype=np.float64)]
        p = geometry.interpolate(0.5, normalized=True)
        return [np.asarray(p.coords[0], dtype=np.float64)]
    if isinstance(geometry, MultiLineString):
        points: list[np.ndarray] = []
        for g in geometry.geoms:
            points.extend(_candidate_points(g))
        return points
    if isinstance(geometry, GeometryCollection):
        points = []
        for g in geometry.geoms:
            points.extend(_candidate_points(g))
        return points
    return []


def _arrival_time_at_point(trajectory: TimedTrajectory2D, point_xy: np.ndarray) -> tuple[float, float]:
    best_distance = float("inf")
    best_time = float(trajectory.timestamps_s[0])
    for i in range(len(trajectory.timestamps_s) - 1):
        p0, p1 = trajectory.positions_xy[i], trajectory.positions_xy[i + 1]
        segment = p1 - p0
        denom = float(segment @ segment)
        u = 0.0 if denom <= 1e-15 else float(np.clip(((point_xy - p0) @ segment) / denom, 0.0, 1.0))
        projection = p0 + u * segment
        distance = float(np.linalg.norm(point_xy - projection))
        if distance < best_distance:
            best_distance = distance
            t0, t1 = trajectory.timestamps_s[i], trajectory.timestamps_s[i + 1]
            best_time = float(t0 + u * (t1 - t0))
    return best_time, best_distance


def _circle_occupancy_intervals(
    trajectory: TimedTrajectory2D, center_xy: np.ndarray, radius_m: float
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    r2 = radius_m * radius_m
    for i in range(len(trajectory.timestamps_s) - 1):
        p0, p1 = trajectory.positions_xy[i], trajectory.positions_xy[i + 1]
        d = p1 - p0
        f = p0 - center_xy
        a = float(d @ d)
        b = 2.0 * float(f @ d)
        c = float(f @ f) - r2
        if a <= 1e-15:
            if c <= 0.0:
                intervals.append((float(trajectory.timestamps_s[i]), float(trajectory.timestamps_s[i + 1])))
            continue
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            if c <= 0.0:
                intervals.append((float(trajectory.timestamps_s[i]), float(trajectory.timestamps_s[i + 1])))
            continue
        root = sqrt(max(disc, 0.0))
        u0 = (-b - root) / (2.0 * a)
        u1 = (-b + root) / (2.0 * a)
        lo, hi = max(0.0, min(u0, u1)), min(1.0, max(u0, u1))
        if lo <= hi:
            t0, t1 = trajectory.timestamps_s[i], trajectory.timestamps_s[i + 1]
            intervals.append((float(t0 + lo * (t1 - t0)), float(t0 + hi * (t1 - t0))))

    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1e-9:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _interval_nearest_time(intervals: list[tuple[float, float]], time_s: float) -> tuple[float, float] | None:
    if not intervals:
        return None

    def distance(interval: tuple[float, float]) -> float:
        start, end = interval
        if start <= time_s <= end:
            return 0.0
        return min(abs(time_s - start), abs(time_s - end))

    return min(intervals, key=distance)


def _pet(
    ego_intervals: list[tuple[float, float]],
    ped_intervals: list[tuple[float, float]],
    ego_arrival: float,
    ped_arrival: float,
) -> float | None:
    ego_interval = _interval_nearest_time(ego_intervals, ego_arrival)
    ped_interval = _interval_nearest_time(ped_intervals, ped_arrival)
    if ego_interval is None or ped_interval is None:
        return None
    e0, e1 = ego_interval
    p0, p1 = ped_interval
    if max(e0, p0) <= min(e1, p1):
        return 0.0
    if e1 < p0:
        return float(p0 - e1)
    return float(e0 - p1)


def _synchronized_minimum_distance(a: TimedTrajectory2D, b: TimedTrajectory2D) -> float:
    start = max(float(a.timestamps_s[0]), float(b.timestamps_s[0]))
    end = min(float(a.timestamps_s[-1]), float(b.timestamps_s[-1]))
    if start > end:
        return float("inf")

    breakpoints = np.unique(np.concatenate([
        a.timestamps_s[(a.timestamps_s >= start) & (a.timestamps_s <= end)],
        b.timestamps_s[(b.timestamps_s >= start) & (b.timestamps_s <= end)],
        np.asarray([start, end]),
    ]))

    def interp(traj: TimedTrajectory2D, t: float) -> np.ndarray:
        x = np.interp(t, traj.timestamps_s, traj.positions_xy[:, 0])
        y = np.interp(t, traj.timestamps_s, traj.positions_xy[:, 1])
        return np.asarray([x, y], dtype=np.float64)

    best = float("inf")
    if len(breakpoints) == 1:
        return float(np.linalg.norm(interp(a, start) - interp(b, start)))
    for left, right in zip(breakpoints[:-1], breakpoints[1:]):
        pa0, pb0 = interp(a, float(left)), interp(b, float(left))
        pa1, pb1 = interp(a, float(right)), interp(b, float(right))
        r0 = pa0 - pb0
        dr = (pa1 - pb1) - r0
        denom = float(dr @ dr)
        u = 0.0 if denom <= 1e-15 else float(np.clip(-(r0 @ dr) / denom, 0.0, 1.0))
        best = min(best, float(np.linalg.norm(r0 + u * dr)))
    return best


def compute_spatiotemporal_conflict(
    ego_trajectory: TimedTrajectory2D,
    pedestrian_trajectory: TimedTrajectory2D,
    config: ConflictConfig | None = None,
) -> ConflictResult:
    config = config or ConflictConfig()
    ego_line = LineString(ego_trajectory.positions_xy)
    ped_line = LineString(pedestrian_trajectory.positions_xy)
    candidates = _candidate_points(ego_line.intersection(ped_line))

    if not candidates and ego_line.distance(ped_line) <= config.spatial_path_tolerance_m:
        a, b = nearest_points(ego_line, ped_line)
        candidates = [0.5 * (np.asarray(a.coords[0]) + np.asarray(b.coords[0]))]

    minimum_distance = _synchronized_minimum_distance(ego_trajectory, pedestrian_trajectory)
    if not candidates:
        return ConflictResult(False, None, None, None, None, None, minimum_distance)

    evaluated = []
    for point_xy in candidates:
        ego_arrival, _ = _arrival_time_at_point(ego_trajectory, point_xy)
        ped_arrival, _ = _arrival_time_at_point(pedestrian_trajectory, point_xy)
        ego_occ = _circle_occupancy_intervals(ego_trajectory, point_xy, config.conflict_region_radius_m)
        ped_occ = _circle_occupancy_intervals(pedestrian_trajectory, point_xy, config.conflict_region_radius_m)
        pet = _pet(ego_occ, ped_occ, ego_arrival, ped_arrival)
        gap = abs(ego_arrival - ped_arrival)
        rank_pet = float("inf") if pet is None else pet
        evaluated.append((rank_pet, gap, point_xy, ego_arrival, ped_arrival, pet))

    _, gap, point_xy, ego_arrival, ped_arrival, pet = min(evaluated, key=lambda x: (x[0], x[1]))
    exists = pet is not None and pet <= config.max_pet_s
    region = ConflictRegion((float(point_xy[0]), float(point_xy[1])), config.conflict_region_radius_m)
    return ConflictResult(
        conflict_exists=exists,
        conflict_region=region,
        ego_arrival_time=float(ego_arrival),
        ped_arrival_time=float(ped_arrival),
        arrival_time_gap=float(gap),
        PET=None if pet is None else float(pet),
        minimum_distance=minimum_distance,
    )
