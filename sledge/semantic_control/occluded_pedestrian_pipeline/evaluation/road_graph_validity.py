"""Global road-graph validity for topology-adaptive SLEDGE scenes.

Generation-time road validation must use the same lane-graph construction as
SLEDGE simulation.  This module therefore reuses ``construct_sledge_map_graph``
and never repairs road geometry.  Invalid diffusion road graphs are rejected so
the caller can sample another repair attempt.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

import networkx as nx
import numpy as np

from sledge.simulation.maps.sledge_map.sledge_map_graph import (
    construct_sledge_map_graph,
)


MIN_EGO_ROUTE_LENGTH_M = 24.0
MIN_EGO_COMPONENT_LANES = 2
MIN_EGO_REACHABLE_LANES = 2
MIN_LARGEST_COMPONENT_LENGTH_RATIO = 0.55
MAX_ORPHAN_LENGTH_RATIO = 0.35
DISALLOW_SINGLE_LANE_ROUTE_FALLBACK = True


def evaluate_global_road_graph_validity(scene: Any) -> Dict[str, Any]:
    """Evaluate whether a generated road graph is globally usable.

    The start lane heuristic, lane graph and sink-route semantics mirror the
    SLEDGE simulation stack.  The returned payload is JSON serializable and is
    intended both for candidate gating and diagnostics.
    """

    try:
        map_graph = construct_sledge_map_graph(scene)
    except Exception as exc:
        return _failed_payload(
            "map_graph_construction_failed: "
            f"{type(exc).__name__}: {exc}"
        )

    graph = map_graph.directed_lane_graph
    baselines = map_graph.baseline_paths_dict
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

    weak_components = list(nx.weakly_connected_components(graph))
    component_sets = [
        {str(node) for node in component}
        for component in weak_components
    ]
    ego_component: Set[str] = set()
    for component in component_sets:
        if ego_node in component:
            ego_component = component
            break

    component_lengths = [
        float(sum(lane_lengths.get(node, 0.0) for node in component))
        for component in component_sets
    ]
    largest_component_length = max(component_lengths, default=0.0)
    largest_component_length_ratio = (
        float(largest_component_length / total_lane_length)
        if total_lane_length > 1e-6
        else 0.0
    )
    ego_component_length = float(
        sum(lane_lengths.get(node, 0.0) for node in ego_component)
    )

    try:
        reachable_nodes = {
            str(node) for node in nx.descendants(graph, ego_node)
        }
    except Exception:
        reachable_nodes = set()
    reachable_nodes.add(str(ego_node))

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
            key=lambda path: _path_length(path, lane_lengths),
        )
    else:
        # Mirrors the current SLEDGE get_route() fallback, but the gate below
        # rejects this as evidence of an invalid generated road network.
        best_route = [str(ego_node)]

    ego_route_length = _path_length(best_route, lane_lengths)

    all_nodes = set(node_ids)
    orphan_nodes = all_nodes - ego_component
    orphan_lane_length = float(
        sum(lane_lengths.get(node, 0.0) for node in orphan_nodes)
    )
    orphan_length_ratio = (
        float(orphan_lane_length / total_lane_length)
        if total_lane_length > 1e-6
        else 1.0
    )

    checks = {
        "ego_route_exists": bool(
            not route_fallback_used
            if DISALLOW_SINGLE_LANE_ROUTE_FALLBACK
            else True
        ),
        "ego_route_length_ok": bool(
            ego_route_length >= MIN_EGO_ROUTE_LENGTH_M
        ),
        "ego_component_size_ok": bool(
            len(ego_component) >= MIN_EGO_COMPONENT_LANES
        ),
        "ego_reachable_lane_count_ok": bool(
            len(reachable_nodes) >= MIN_EGO_REACHABLE_LANES
        ),
        "largest_component_ratio_ok": bool(
            largest_component_length_ratio
            >= MIN_LARGEST_COMPONENT_LENGTH_RATIO
        ),
        "orphan_fragment_ratio_ok": bool(
            orphan_length_ratio <= MAX_ORPHAN_LENGTH_RATIO
        ),
    }
    passed = all(checks.values())

    return {
        "schema_version": "global_road_graph_validity_v1",
        "passed": bool(passed),
        "checks": checks,
        "reason": None if passed else "road_graph_gate_failed",
        "num_lane_nodes": int(len(node_ids)),
        "num_lane_edges": int(graph.number_of_edges()),
        "num_weak_components": int(len(weak_components)),
        "largest_component_length_m": float(largest_component_length),
        "largest_component_length_ratio": float(
            largest_component_length_ratio
        ),
        "ego_start_lane_id": str(ego_node),
        "ego_component_lane_count": int(len(ego_component)),
        "ego_component_length_m": float(ego_component_length),
        "ego_reachable_lane_count": int(len(reachable_nodes)),
        "ego_route_lane_ids": list(best_route),
        "ego_route_length_m": float(ego_route_length),
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


def _path_length(
    path: Iterable[str],
    lane_lengths: Dict[str, float],
) -> float:
    return float(
        sum(lane_lengths.get(str(node), 0.0) for node in path)
    )


def _threshold_payload() -> Dict[str, Any]:
    return {
        "min_ego_route_length_m": float(MIN_EGO_ROUTE_LENGTH_M),
        "min_ego_component_lanes": int(MIN_EGO_COMPONENT_LANES),
        "min_ego_reachable_lanes": int(MIN_EGO_REACHABLE_LANES),
        "min_largest_component_length_ratio": float(
            MIN_LARGEST_COMPONENT_LENGTH_RATIO
        ),
        "max_orphan_length_ratio": float(MAX_ORPHAN_LENGTH_RATIO),
        "disallow_single_lane_route_fallback": bool(
            DISALLOW_SINGLE_LANE_ROUTE_FALLBACK
        ),
    }


def _failed_payload(reason: str, **extra: Any) -> Dict[str, Any]:
    checks = {
        "ego_route_exists": False,
        "ego_route_length_ok": False,
        "ego_component_size_ok": False,
        "ego_reachable_lane_count_ok": False,
        "largest_component_ratio_ok": False,
        "orphan_fragment_ratio_ok": False,
    }
    payload = {
        "schema_version": "global_road_graph_validity_v1",
        "passed": False,
        "checks": checks,
        "reason": str(reason),
        "num_lane_nodes": 0,
        "num_lane_edges": 0,
        "num_weak_components": 0,
        "largest_component_length_m": 0.0,
        "largest_component_length_ratio": 0.0,
        "ego_start_lane_id": None,
        "ego_component_lane_count": 0,
        "ego_component_length_m": 0.0,
        "ego_reachable_lane_count": 0,
        "ego_route_lane_ids": [],
        "ego_route_length_m": 0.0,
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
