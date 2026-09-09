"""Path-relative road geometry shared by generation-time constraints.

The original topology-adaptive projector inferred road geometry from a single
``x = v * TTC`` cross-section.  That works on straight roads but the global-x
assumption breaks on curves and at intersections.  This module instead reuses
the same directed lane graph SLEDGE builds for simulation and exposes a local
Frenet-like frame ``(s, d)`` around the generated road.

The implementation deliberately does not repair topology.  It only projects,
selects a local corridor around the ego, and advances along existing directed
connections.  Structural validity remains the responsibility of the global
road-graph gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from sledge.simulation.maps.sledge_map.sledge_map_graph import (
    construct_sledge_map_graph,
)


_EPS = 1e-6


@dataclass(frozen=True)
class PathPose:
    """Pose sampled on one directed generated-road polyline."""

    path_id: str
    s: float
    point: np.ndarray
    heading: float

    @property
    def tangent(self) -> np.ndarray:
        return np.asarray(
            [math.cos(self.heading), math.sin(self.heading)],
            dtype=np.float64,
        )

    @property
    def normal(self) -> np.ndarray:
        tangent = self.tangent
        return np.asarray([-tangent[1], tangent[0]], dtype=np.float64)


@dataclass(frozen=True)
class PathProjection:
    """Nearest projection of a point onto a directed road path."""

    pose: PathPose
    distance: float
    signed_lateral_m: float


@dataclass(frozen=True)
class EgoCorridor:
    """Two generated road lines that locally bound the ego corridor."""

    right: PathProjection
    left: PathProjection
    width_m: float
    center_offset_m: float
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "right_path_id": self.right.pose.path_id,
            "left_path_id": self.left.pose.path_id,
            "right_s_m": float(self.right.pose.s),
            "left_s_m": float(self.left.pose.s),
            "width_m": float(self.width_m),
            "center_offset_m": float(self.center_offset_m),
            "confidence": float(self.confidence),
        }


@dataclass(frozen=True)
class LocalRoadFrame:
    """Road-local frame at a routed conflict location.

    Lateral offsets are measured along ``normal``.  Positive ``d`` is the
    local-left side of the routed road, independent of global x/y axes.
    """

    center_xy: np.ndarray
    heading: float
    lane_width_m: float
    right_boundary_offset_m: float
    left_boundary_offset_m: float
    adjacent_left_center_offset_m: Optional[float]
    adjacent_right_center_offset_m: Optional[float]
    route_distance_m: float
    right_route_path_ids: Tuple[str, ...]
    left_route_path_ids: Tuple[str, ...]
    source: str = "sledge_map_graph_path_relative"

    @property
    def tangent(self) -> np.ndarray:
        return np.asarray(
            [math.cos(self.heading), math.sin(self.heading)],
            dtype=np.float64,
        )

    @property
    def normal(self) -> np.ndarray:
        tangent = self.tangent
        return np.asarray([-tangent[1], tangent[0]], dtype=np.float64)

    @property
    def lane_half_width_m(self) -> float:
        return 0.5 * float(self.lane_width_m)

    def point_at_lateral(self, d_m: float) -> np.ndarray:
        return np.asarray(self.center_xy, dtype=np.float64) + float(d_m) * self.normal

    def adjacent_lane_center_offset(self, side_sign: float) -> Optional[float]:
        return (
            self.adjacent_left_center_offset_m
            if side_sign > 0.0
            else self.adjacent_right_center_offset_m
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coordinate_system": "path_relative_s_d",
            "center_xy": [float(v) for v in self.center_xy],
            "local_tangent_heading": float(self.heading),
            "lane_width_m": float(self.lane_width_m),
            "lane_half_width_m": float(self.lane_half_width_m),
            "right_boundary_offset_m": float(self.right_boundary_offset_m),
            "left_boundary_offset_m": float(self.left_boundary_offset_m),
            "adjacent_lane_center_left_d_m": (
                None
                if self.adjacent_left_center_offset_m is None
                else float(self.adjacent_left_center_offset_m)
            ),
            "adjacent_lane_center_right_d_m": (
                None
                if self.adjacent_right_center_offset_m is None
                else float(self.adjacent_right_center_offset_m)
            ),
            "route_distance_m": float(self.route_distance_m),
            "right_route_path_ids": list(self.right_route_path_ids),
            "left_route_path_ids": list(self.left_route_path_ids),
            "source": self.source,
        }


class RoadConstraintContext:
    """Generated-road graph plus path-relative projection helpers."""

    def __init__(
        self,
        *,
        paths: Dict[str, np.ndarray],
        directed_graph: nx.DiGraph,
    ) -> None:
        clean: Dict[str, np.ndarray] = {}
        for path_id, poses in paths.items():
            arr = np.asarray(poses, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
                continue
            finite = np.isfinite(arr[:, :2]).all(axis=1)
            arr = arr[finite]
            if len(arr) < 2:
                continue
            clean[str(path_id)] = arr
        if not clean:
            raise ValueError("road constraint context has no usable paths")
        self.paths = clean
        self.directed_graph = directed_graph.copy()

    @classmethod
    def from_scene(cls, scene: Any) -> "RoadConstraintContext":
        map_graph = construct_sledge_map_graph(scene)
        return cls(
            paths={
                str(path_id): np.asarray(poses, dtype=np.float64)
                for path_id, poses in map_graph.baseline_paths_dict.items()
            },
            directed_graph=map_graph.directed_lane_graph,
        )

    @classmethod
    def from_components(
        cls,
        paths: Dict[str, np.ndarray],
        edges: Iterable[Tuple[str, str]],
    ) -> "RoadConstraintContext":
        """Small dependency-free constructor used by geometry tests."""

        graph = nx.DiGraph()
        graph.add_nodes_from(str(path_id) for path_id in paths)
        graph.add_edges_from((str(a), str(b)) for a, b in edges)
        return cls(paths=paths, directed_graph=graph)

    def project_to_path(
        self,
        path_id: str,
        point_xy: Sequence[float],
    ) -> PathProjection:
        poses = self.paths[str(path_id)]
        s, point, heading, distance, signed_lateral = _project_polyline(
            poses,
            np.asarray(point_xy, dtype=np.float64),
        )
        return PathProjection(
            pose=PathPose(
                path_id=str(path_id),
                s=float(s),
                point=point,
                heading=float(heading),
            ),
            distance=float(distance),
            signed_lateral_m=float(signed_lateral),
        )

    def infer_ego_corridor(
        self,
        *,
        ego_xy: Sequence[float] = (0.0, 0.0),
        ego_heading: float = 0.0,
        lane_width_hint_m: float = 3.5,
        max_distance_m: float = 7.0,
        max_heading_error_rad: float = math.radians(65.0),
    ) -> EgoCorridor:
        """Choose the most plausible pair of road lines bracketing ego."""

        ego_xy_arr = np.asarray(ego_xy, dtype=np.float64)
        ego_tangent = np.asarray(
            [math.cos(ego_heading), math.sin(ego_heading)],
            dtype=np.float64,
        )
        ego_normal = np.asarray([-ego_tangent[1], ego_tangent[0]])
        candidates: List[Tuple[PathProjection, float, float]] = []

        for path_id in self.paths:
            projection = self.project_to_path(path_id, ego_xy_arr)
            heading_error = _directed_heading_error(
                projection.pose.heading,
                ego_heading,
            )
            if (
                projection.distance > float(max_distance_m)
                or heading_error > float(max_heading_error_rad)
            ):
                continue
            offset = float(
                np.dot(projection.pose.point - ego_xy_arr, ego_normal)
            )
            candidates.append((projection, offset, heading_error))

        right = [row for row in candidates if row[1] < -0.15]
        left = [row for row in candidates if row[1] > 0.15]
        if not right or not left:
            raise ValueError("could not find generated road lines on both sides of ego")

        hint = float(np.clip(lane_width_hint_m, 2.5, 5.0))
        best: Optional[Tuple[float, PathProjection, PathProjection, float, float]] = None
        for right_row in right:
            for left_row in left:
                right_proj, right_d, right_heading_error = right_row
                left_proj, left_d, left_heading_error = left_row
                if right_proj.pose.path_id == left_proj.pose.path_id:
                    continue
                width = float(left_d - right_d)
                if not 2.2 <= width <= 6.2:
                    continue
                center_offset = 0.5 * (left_d + right_d)
                score = (
                    abs(width - hint)
                    + 2.0 * abs(center_offset)
                    + 0.20 * (
                        right_proj.distance + left_proj.distance
                    )
                    + 0.75 * (
                        right_heading_error + left_heading_error
                    )
                )
                if best is None or score < best[0]:
                    best = (
                        float(score),
                        right_proj,
                        left_proj,
                        width,
                        center_offset,
                    )

        if best is None:
            raise ValueError("no plausible ego-lane corridor pair")

        score, right_proj, left_proj, width, center_offset = best
        confidence = float(math.exp(-0.35 * max(score, 0.0)))
        return EgoCorridor(
            right=right_proj,
            left=left_proj,
            width_m=float(width),
            center_offset_m=float(center_offset),
            confidence=confidence,
        )

    def frame_at_route_distance(
        self,
        corridor: EgoCorridor,
        *,
        route_distance_m: float,
        lane_width_hint_m: float = 3.5,
    ) -> LocalRoadFrame:
        """Advance both corridor boundaries through the directed road graph."""

        distance = max(0.0, float(route_distance_m))
        right_pose, right_route = self.advance(
            corridor.right.pose,
            distance,
        )
        left_pose, left_route = self.advance(
            corridor.left.pose,
            distance,
        )

        right_xy = np.asarray(right_pose.point, dtype=np.float64)
        left_xy = np.asarray(left_pose.point, dtype=np.float64)
        center = 0.5 * (right_xy + left_xy)

        right_tangent = right_pose.tangent
        left_tangent = left_pose.tangent
        if float(np.dot(right_tangent, left_tangent)) < 0.0:
            left_tangent = -left_tangent
        tangent = right_tangent + left_tangent
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= _EPS:
            tangent = right_tangent
        else:
            tangent = tangent / tangent_norm
        heading = float(math.atan2(tangent[1], tangent[0]))
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)

        right_offset = float(np.dot(right_xy - center, normal))
        left_offset = float(np.dot(left_xy - center, normal))
        if right_offset > left_offset:
            right_offset, left_offset = left_offset, right_offset
        measured_width = float(left_offset - right_offset)
        hint = float(np.clip(lane_width_hint_m, 2.5, 5.0))
        if not 2.2 <= measured_width <= 6.2:
            right_offset = -0.5 * hint
            left_offset = 0.5 * hint
            measured_width = hint

        adjacent_left, adjacent_right = self._adjacent_lane_centers(
            center_xy=center,
            heading=heading,
            right_boundary_offset_m=right_offset,
            left_boundary_offset_m=left_offset,
        )

        return LocalRoadFrame(
            center_xy=center,
            heading=heading,
            lane_width_m=measured_width,
            right_boundary_offset_m=right_offset,
            left_boundary_offset_m=left_offset,
            adjacent_left_center_offset_m=adjacent_left,
            adjacent_right_center_offset_m=adjacent_right,
            route_distance_m=distance,
            right_route_path_ids=tuple(right_route),
            left_route_path_ids=tuple(left_route),
        )

    def advance(
        self,
        start_pose: PathPose,
        distance_m: float,
        *,
        max_hops: int = 20,
    ) -> Tuple[PathPose, List[str]]:
        """Advance arc length across graph edges using smoothest successor."""

        remaining = max(0.0, float(distance_m))
        current_id = str(start_pose.path_id)
        current_s = float(start_pose.s)
        route = [current_id]
        visited_counts: Dict[str, int] = {current_id: 1}

        for _ in range(max_hops + 1):
            poses = self.paths[current_id]
            length = _polyline_length(poses)
            available = max(0.0, length - current_s)
            if remaining <= available + 1e-5:
                return self.sample(current_id, current_s + remaining), route

            remaining -= available
            end_pose = self.sample(current_id, length)
            successors = [
                str(node)
                for node in self.directed_graph.successors(current_id)
                if str(node) in self.paths
            ]
            successors = [
                node
                for node in successors
                if not (node == current_id and len(successors) > 1)
            ]
            if not successors:
                raise ValueError(
                    f"route ended on path {current_id!r} with "
                    f"{remaining:.2f} m still requested"
                )

            next_id = min(
                successors,
                key=lambda candidate: self._successor_score(
                    end_pose,
                    candidate,
                    visited_counts,
                ),
            )
            current_id = next_id
            current_s = 0.0
            route.append(current_id)
            visited_counts[current_id] = visited_counts.get(current_id, 0) + 1

        raise ValueError(
            f"route advance exceeded {max_hops} graph hops for {distance_m:.2f} m"
        )

    def sample(self, path_id: str, s_m: float) -> PathPose:
        poses = self.paths[str(path_id)]
        point, heading = _sample_polyline(poses, float(s_m))
        return PathPose(
            path_id=str(path_id),
            s=float(np.clip(s_m, 0.0, _polyline_length(poses))),
            point=point,
            heading=float(heading),
        )

    def _successor_score(
        self,
        end_pose: PathPose,
        candidate_id: str,
        visited_counts: Dict[str, int],
    ) -> float:
        start = self.sample(candidate_id, 0.0)
        heading_error = _directed_heading_error(
            start.heading,
            end_pose.heading,
        )
        endpoint_gap = float(np.linalg.norm(start.point - end_pose.point))
        revisit_penalty = 10.0 * float(visited_counts.get(candidate_id, 0))
        return 3.0 * heading_error + endpoint_gap + revisit_penalty

    def _adjacent_lane_centers(
        self,
        *,
        center_xy: np.ndarray,
        heading: float,
        right_boundary_offset_m: float,
        left_boundary_offset_m: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        tangent = np.asarray(
            [math.cos(heading), math.sin(heading)],
            dtype=np.float64,
        )
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        offsets: List[float] = []
        for path_id in self.paths:
            projection = self.project_to_path(path_id, center_xy)
            if projection.distance > 10.0:
                continue
            if _undirected_heading_error(
                projection.pose.heading,
                heading,
            ) > math.radians(35.0):
                continue
            offsets.append(
                float(np.dot(projection.pose.point - center_xy, normal))
            )

        clustered = _cluster_values(offsets, tolerance=0.65)
        left_outer = [
            value
            for value in clustered
            if value > left_boundary_offset_m + 1.5
        ]
        right_outer = [
            value
            for value in clustered
            if value < right_boundary_offset_m - 1.5
        ]

        adjacent_left = None
        if left_outer:
            outer = min(left_outer)
            width = float(outer - left_boundary_offset_m)
            if 2.4 <= width <= 5.5:
                adjacent_left = 0.5 * (left_boundary_offset_m + outer)

        adjacent_right = None
        if right_outer:
            outer = max(right_outer)
            width = float(right_boundary_offset_m - outer)
            if 2.4 <= width <= 5.5:
                adjacent_right = 0.5 * (right_boundary_offset_m + outer)
        return adjacent_left, adjacent_right


def _polyline_progress(poses: np.ndarray) -> np.ndarray:
    points = np.asarray(poses, dtype=np.float64)[:, :2]
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def _polyline_length(poses: np.ndarray) -> float:
    return float(_polyline_progress(poses)[-1])


def _sample_polyline(
    poses: np.ndarray,
    s_m: float,
) -> Tuple[np.ndarray, float]:
    poses = np.asarray(poses, dtype=np.float64)
    points = poses[:, :2]
    progress = _polyline_progress(poses)
    s = float(np.clip(s_m, 0.0, progress[-1]))
    idx = int(np.searchsorted(progress, s, side="right") - 1)
    idx = int(np.clip(idx, 0, len(points) - 2))
    span = float(progress[idx + 1] - progress[idx])
    ratio = 0.0 if span <= _EPS else (s - progress[idx]) / span
    point = (1.0 - ratio) * points[idx] + ratio * points[idx + 1]
    delta = points[idx + 1] - points[idx]
    heading = float(math.atan2(delta[1], delta[0]))
    return np.asarray(point, dtype=np.float64), heading


def _project_polyline(
    poses: np.ndarray,
    point_xy: np.ndarray,
) -> Tuple[float, np.ndarray, float, float, float]:
    poses = np.asarray(poses, dtype=np.float64)
    points = poses[:, :2]
    progress = _polyline_progress(poses)
    point_xy = np.asarray(point_xy, dtype=np.float64)[:2]

    best = None
    for idx in range(len(points) - 1):
        a = points[idx]
        b = points[idx + 1]
        segment = b - a
        denom = float(np.dot(segment, segment))
        if denom <= _EPS:
            continue
        ratio = float(np.clip(np.dot(point_xy - a, segment) / denom, 0.0, 1.0))
        projected = a + ratio * segment
        delta = point_xy - projected
        distance = float(np.linalg.norm(delta))
        heading = float(math.atan2(segment[1], segment[0]))
        tangent = segment / math.sqrt(denom)
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        signed_lateral = float(np.dot(point_xy - projected, normal))
        s = float(progress[idx] + ratio * math.sqrt(denom))
        row = (distance, s, projected, heading, signed_lateral)
        if best is None or row[0] < best[0]:
            best = row

    if best is None:
        raise ValueError("polyline contains no non-zero-length segment")
    distance, s, projected, heading, signed_lateral = best
    return s, np.asarray(projected), heading, distance, signed_lateral


def _wrap_angle(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def _directed_heading_error(a: float, b: float) -> float:
    return abs(_wrap_angle(float(a) - float(b)))


def _undirected_heading_error(a: float, b: float) -> float:
    error = _directed_heading_error(a, b)
    return float(min(error, abs(math.pi - error)))


def _cluster_values(values: Sequence[float], tolerance: float) -> List[float]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return []
    groups: List[List[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if abs(value - float(np.mean(groups[-1]))) <= float(tolerance):
            groups[-1].append(value)
        else:
            groups.append([value])
    return [float(np.mean(group)) for group in groups]


__all__ = [
    "EgoCorridor",
    "LocalRoadFrame",
    "PathPose",
    "PathProjection",
    "RoadConstraintContext",
]
