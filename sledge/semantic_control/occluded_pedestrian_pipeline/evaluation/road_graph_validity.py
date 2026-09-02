"""Global road-network validity for topology-adaptive SLEDGE scenes.

Generation-time road validation reuses SLEDGE's own map construction.  The
validator combines two complementary views:

1. directed lane graph, used by SLEDGE route construction;
2. spatial connectivity of SLEDGE lane polygons, used to detect floating road
   islands without falsely rejecting parallel lanes or one long valid lane.

The module never repairs road geometry.  Invalid diffusion roads are rejected
so the caller can sample another repair attempt.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

import networkx as nx
import numpy as np

from sledge.simulation.maps.sledge_map.sledge_map_graph import (
    construct_sledge_map_graph,
)


MIN_EGO_FORWARD_ROUTE_LENGTH_M = 24.0
MIN_EGO_SPATIAL_COMPONENT_LENGTH_M = 30.0
MIN_LARGEST_SPATIAL_COMPONENT_LENGTH_RATIO = 0.60
MAX_ORPHAN_LENGTH_RATIO = 0.35
SPATIAL_COMPONENT_GAP_M = 0.75


def evaluate_global_road_graph_validity(scene: Any) -> Dict[str, Any]:
    """Evaluate global usability and fragmentation of a generated road."""

    try:
        map_graph = construct_sledge_map_graph(scene)
    except Exception as exc:
        return _failed_payload(
            "map_graph_construction_failed: "
            f"{type(exc).__name__}: {exc}"
        )

    graph = map_graph.directed_lane_graph
    baselines = map_graph.baseline_paths_dict
    polygons = map_graph.polygon_dict
    node_ids = [str(node) for node in graph.nodes()]

    if not node_ids or not baselines:
        return _failed_payload("no_valid_lane_graph")

    lane_lengths = {
        str(node): _polyline_length(np.asarray(baselines[str(node)]))
        for node in graph.nodes()
        if str(node) in baselines
    }
    total_lane_length = float(sum(lane_lengths.values()))

    ego_node = _find_ego_start_lane(baselines)
    if ego_node is None or ego_node not in graph:
        return _failed_payload(
            "ego_start_lane_missing",
            num_lane_nodes=len(node_ids),
            num_lane_edges=int(graph.number_of_edges()),
            total_lane_length_m=total_lane_length,
        )

    # ------------------------------------------------------------------
    # Directed route view: same connection graph used by SLEDGE.
    # A single long lane is valid, so route validity is based on available
    # forward distance rather than requiring multiple graph nodes.
    # ------------------------------------------------------------------
    sink_nodes = [
        str(node)
        for node in graph.nodes()
        if graph.out_degree(node) == 0
    ]
    available_paths: List[List[str]] = []
    for sink_node in sink_nodes:
        try:
            path = nx.shortest_path(
                graph,
                source=ego_node,
                target=sink_node,
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        available_paths.append([str(node) for node in path])

    route_fallback_used = len(available_paths) == 0
    if available_paths:
        best_route = max(
            available_paths,
            key=lambda path: _available_forward_route_length(
                path,
                baselines,
                lane_lengths,
            ),
        )
    else:
        best_route = [str(ego_node)]

    ego_forward_route_length = _available_forward_route_length(
        best_route,
        baselines,
        lane_lengths,
    )

    try:
        directed_reachable = {
            str(node) for node in nx.descendants(graph, ego_node)
        }
    except Exception:
        directed_reachable = set()
    directed_reachable.add(str(ego_node))

    # ------------------------------------------------------------------
    # Spatial road view.
    # Directed lane edges do not represent lateral adjacency, so weak graph
    # components would incorrectly label normal parallel lanes as islands.
    # Instead, connect lane polygons that overlap/touch or are separated by a
    # very small gap.  This matches what is visibly perceived as one road body
    # in SledgeBoard.
    # ------------------------------------------------------------------
    spatial_graph = nx.Graph()
    spatial_graph.add_nodes_from(node_ids)
    for i, node_i in enumerate(node_ids):
        polygon_i = polygons.get(node_i)
        if polygon_i is None:
            continue
        for node_j in node_ids[i + 1 :]:
            polygon_j = polygons.get(node_j)
            if polygon_j is None:
                continue
            try:
                distance = float(polygon_i.distance(polygon_j))
            except Exception:
                continue
            if distance <= SPATIAL_COMPONENT_GAP_M:
                spatial_graph.add_edge(node_i, node_j)

    spatial_components = [
        {str(node) for node in component}
        for component in nx.connected_components(spatial_graph)
    ]
    ego_spatial_component: Set[str] = set()
    for component in spatial_components:
        if ego_node in component:
            ego_spatial_component = component
            break

    component_lengths = [
        float(sum(lane_lengths.get(node, 0.0) for node in component))
        for component in spatial_components
    ]
    largest_component_length = max(component_lengths, default=0.0)
    largest_component_length_ratio = (
        float(largest_component_length / total_lane_length)
        if total_lane_length > 1e-6
        else 0.0
    )
    ego_component_length = float(
        sum(lane_lengths.get(node, 0.0) for node in ego_spatial_component)
    )

    all_nodes = set(node_ids)
    orphan_nodes = all_nodes - ego_spatial_component
    orphan_lane_length = float(
        sum(lane_lengths.get(node, 0.0) for node in orphan_nodes)
    )
    orphan_length_ratio = (
        float(orphan_lane_length / total_lane_length)
        if total_lane_length > 1e-6
        else 1.0
    )

    checks = {
        "ego_forward_route_length_ok": bool(
            ego_forward_route_length >= MIN_EGO_FORWARD_ROUTE_LENGTH_M
        ),
        "ego_spatial_component_length_ok": bool(
            ego_component_length >= MIN_EGO_SPATIAL_COMPONENT_LENGTH_M
        ),
        "largest_spatial_component_ratio_ok": bool(
            largest_component_length_ratio
            >= MIN_LARGEST_SPATIAL_COMPONENT_LENGTH_RATIO
        ),
        "orphan_fragment_ratio_ok": bool(
            orphan_length_ratio <= MAX_ORPHAN_LENGTH_RATIO
        ),
    }
    passed = all(checks.values())

    return {
        "schema_version": "global_road_graph_validity_v2",
        "passed": bool(passed),
        "checks": checks,
        "reason": None if passed else "road_graph_gate_failed",
        "num_lane_nodes": int(len(node_ids)),
        "num_lane_edges": int(graph.number_of_edges()),
        "num_directed_weak_components": int(
            nx.number_weakly_connected_components(graph)
        ),
        "num_spatial_components": int(len(spatial_components)),
        "largest_spatial_component_length_m": float(
            largest_component_length
        ),
        "largest_spatial_component_length_ratio": float(
            largest_component_length_ratio
        ),
        "ego_start_lane_id": str(ego_node),
        "ego_spatial_component_lane_count": int(
            len(ego_spatial_component)
        ),
        "ego_spatial_component_length_m": float(ego_component_length),
        "ego_directed_reachable_lane_count": int(len(directed_reachable)),
        "ego_route_lane_ids": list(best_route),
        "ego_forward_route_length_m": float(ego_forward_route_length),
        "route_fallback_used": bool(route_fallback_used),
        "total_lane_length_m": float(total_lane_length),
        "orphan_lane_count": int(len(orphan_nodes)),
        "orphan_lane_length_m": float(orphan_lane_length),
        "orphan_lane_length_ratio": float(orphan_length_ratio),
        "thresholds": _threshold_payload(),
    }


def _find_ego_start_lane(
    baselines: Dict[str, np.ndarray],
) -> Optional[str]:
    """Mirror SLEDGE get_route(): nearest baseline to local origin."""

    best = None
    for lane_id, poses in baselines.items():
        poses = np.asarray(poses, dtype=np.float64)
        if poses.ndim != 2 or len(poses) < 2 or poses.shape[1] < 2:
            continue
        distance = float(np.linalg.norm(poses[:, :2], axis=1).min())
        if best is None or distance < best[0]:
            best = (distance, str(lane_id))
    return None if best is None else best[1]


def _polyline_length(poses: np.ndarray) -> float:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[0] < 2 or poses.shape[1] < 2:
        return 0.0
    delta = np.diff(poses[:, :2], axis=0)
    return float(np.linalg.norm(delta, axis=1).sum())


def _available_forward_on_first_lane(poses: np.ndarray) -> float:
    """Approximate arc length from local ego origin to lane end."""

    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or len(poses) < 2 or poses.shape[1] < 2:
        return 0.0
    segment_lengths = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    nearest_index = int(np.argmin(np.linalg.norm(poses[:, :2], axis=1)))
    return float(max(0.0, cumulative[-1] - cumulative[nearest_index]))


def _available_forward_route_length(
    path: Iterable[str],
    baselines: Dict[str, np.ndarray],
    lane_lengths: Dict[str, float],
) -> float:
    path = [str(node) for node in path]
    if not path:
        return 0.0
    first = path[0]
    first_forward = _available_forward_on_first_lane(
        np.asarray(baselines[first])
    )
    remaining = sum(lane_lengths.get(node, 0.0) for node in path[1:])
    return float(first_forward + remaining)


def _threshold_payload() -> Dict[str, Any]:
    return {
        "min_ego_forward_route_length_m": float(
            MIN_EGO_FORWARD_ROUTE_LENGTH_M
        ),
        "min_ego_spatial_component_length_m": float(
            MIN_EGO_SPATIAL_COMPONENT_LENGTH_M
        ),
        "min_largest_spatial_component_length_ratio": float(
            MIN_LARGEST_SPATIAL_COMPONENT_LENGTH_RATIO
        ),
        "max_orphan_length_ratio": float(MAX_ORPHAN_LENGTH_RATIO),
        "spatial_component_gap_m": float(SPATIAL_COMPONENT_GAP_M),
    }


def _failed_payload(reason: str, **extra: Any) -> Dict[str, Any]:
    checks = {
        "ego_forward_route_length_ok": False,
        "ego_spatial_component_length_ok": False,
        "largest_spatial_component_ratio_ok": False,
        "orphan_fragment_ratio_ok": False,
    }
    payload = {
        "schema_version": "global_road_graph_validity_v2",
        "passed": False,
        "checks": checks,
        "reason": str(reason),
        "num_lane_nodes": 0,
        "num_lane_edges": 0,
        "num_directed_weak_components": 0,
        "num_spatial_components": 0,
        "largest_spatial_component_length_m": 0.0,
        "largest_spatial_component_length_ratio": 0.0,
        "ego_start_lane_id": None,
        "ego_spatial_component_lane_count": 0,
        "ego_spatial_component_length_m": 0.0,
        "ego_directed_reachable_lane_count": 0,
        "ego_route_lane_ids": [],
        "ego_forward_route_length_m": 0.0,
        "route_fallback_used": True,
        "total_lane_length_m": 0.0,
        "orphan_lane_count": 0,
        "orphan_lane_length_m": 0.0,
        "orphan_lane_length_ratio": 1.0,
        "thresholds": _threshold_payload(),
    }
    payload.update(extra)
    return payload


__all__ = ["evaluate_global_road_graph_validity"]
