"""Build a new road/ego base scene from a hierarchical construction template.

The input ``scaffold_scene`` contributes only tensor shapes, dtypes and slot
capacities. All semantic content is cleared before road and ego states are
written. This makes the synthesis branch semantically independent of the source
B0 while remaining compatible with the exact SLEDGE cache representation used
by the current repository.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


class TemplateSceneSynthesizer:
    """Synthesize road geometry and ego state in a blank SLEDGE scaffold."""

    def synthesize(
        self,
        *,
        scaffold_scene: Any,
        hierarchical_spec: Mapping[str, Any],
        sampled_parameters: Mapping[str, Any],
        construction_plan: Mapping[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        scene = deepcopy(scaffold_scene)
        self._clear_scene(scene)

        sample = dict(sampled_parameters or {})
        plan = dict(construction_plan or {})
        explicit = dict(plan.get("explicit_global_constraints", {}) or {})
        hierarchy = dict(hierarchical_spec.get("hierarchy_layer", {}) or {})
        values = dict(hierarchy.get("path_values", {}) or {})

        topology = str(
            explicit.get(
                "road_topology",
                sample.get("road_topology", values.get("road_topology", "straight_segment")),
            )
        )
        lane_count = max(1, int(sample.get("lane_count", explicit.get("lane_count", 1))))
        lane_width = float(sample.get("lane_width_m", explicit.get("lane_width_m", 3.5)))
        curvature = float(sample.get("road_curvature", explicit.get("road_curvature", 0.0)))
        if explicit.get("road_shape") == "curved" and abs(curvature) < 1e-5:
            curvature = 0.004

        conflict_x = float(sample.get("ego_distance_to_conflict_m", 12.0))
        directionality = str(explicit.get("directionality", "unspecified"))
        lane_layout = list(explicit.get("lane_layout", []) or [])

        polylines = self._build_road_polylines(
            topology=topology,
            lane_count=lane_count,
            lane_width_m=lane_width,
            curvature=curvature,
            conflict_x=conflict_x,
            directionality=directionality,
            lane_layout=lane_layout,
        )
        num_lines = self._write_polylines(scene, polylines)
        self._write_ego_state(
            scene,
            speed_mps=float(sample.get("ego_speed_mps", 8.0)),
            acceleration_mps2=float(sample.get("ego_acceleration_mps2", 0.0)),
        )

        report = {
            "schema_version": "template_scene_synthesis_v1",
            "construction_mode": "synthesize_new",
            "source_scene_usage": "blank_capacity_scaffold_only",
            "source_semantic_content_preserved": False,
            "road_topology": topology,
            "lane_count": lane_count,
            "lane_width_m": lane_width,
            "road_curvature": curvature,
            "directionality": directionality,
            "lane_layout": lane_layout,
            "conflict_x_m": conflict_x,
            "generated_polyline_count": int(num_lines),
            "requested_polyline_count": int(len(polylines)),
            "ego_speed_mps": float(sample.get("ego_speed_mps", 8.0)),
            "ego_acceleration_mps2": float(sample.get("ego_acceleration_mps2", 0.0)),
        }
        return scene, report

    @staticmethod
    def _clear_scene(scene: Any) -> None:
        for name in (
            "lines",
            "vehicles",
            "pedestrians",
            "static_objects",
            "green_lights",
            "red_lights",
            "ego",
        ):
            elem = getattr(scene, name, None)
            if elem is None:
                continue
            states = np.asarray(getattr(elem, "states", []))
            mask = np.asarray(getattr(elem, "mask", []))
            if states.size:
                states[...] = 0
                elem.states = states
            if mask.size:
                mask[...] = False
                elem.mask = mask

    @staticmethod
    def _write_ego_state(
        scene: Any,
        *,
        speed_mps: float,
        acceleration_mps2: float,
    ) -> None:
        states = np.asarray(scene.ego.states)
        if states.size == 0:
            scene.ego.states = np.asarray(
                [speed_mps, 0.0, acceleration_mps2, 0.0],
                dtype=np.float32,
            )
        else:
            flat = states.reshape(-1)
            flat[...] = 0
            flat[0] = float(speed_mps)
            if flat.size >= 2:
                flat[1] = 0.0
            if flat.size >= 3:
                flat[2] = float(acceleration_mps2)
            if flat.size >= 4:
                flat[3] = 0.0
            scene.ego.states = states

        mask = np.asarray(scene.ego.mask)
        if mask.size:
            mask[...] = False
            mask.reshape(-1)[0] = True
            scene.ego.mask = mask

    def _build_road_polylines(
        self,
        *,
        topology: str,
        lane_count: int,
        lane_width_m: float,
        curvature: float,
        conflict_x: float,
        directionality: str,
        lane_layout: Sequence[str],
    ) -> List[np.ndarray]:
        value = topology.lower()
        if value in {"intersection", "junction"}:
            return self._intersection_lines(
                lane_count=lane_count,
                lane_width=lane_width_m,
                conflict_x=conflict_x,
            )
        if value in {"roundabout", "traffic_circle"}:
            return self._roundabout_lines(
                lane_count=lane_count,
                lane_width=lane_width_m,
                conflict_x=conflict_x,
            )
        if value in {"merge_diverge", "merge", "diverge"}:
            return self._merge_diverge_lines(
                lane_width=lane_width_m,
                conflict_x=conflict_x,
            )
        if value in {"work_zone", "construction_zone"}:
            return self._work_zone_lines(
                lane_count=lane_count,
                lane_width=lane_width_m,
                curvature=curvature,
                conflict_x=conflict_x,
                lane_layout=lane_layout,
            )
        lines = self._straight_lines(
            lane_count=lane_count,
            lane_width=lane_width_m,
            curvature=curvature,
            directionality=directionality,
        )
        return self._apply_lane_layout(
            lines,
            lane_layout=lane_layout,
            lane_width=lane_width_m,
            conflict_x=conflict_x,
        )


    @staticmethod
    def _apply_lane_layout(
        lines: List[np.ndarray],
        *,
        lane_layout: Sequence[str],
        lane_width: float,
        conflict_x: float,
    ) -> List[np.ndarray]:
        output = list(lines)
        layouts = set(str(item) for item in lane_layout)
        if "dedicated_left_turn_lane" in layouts or "turn_lane" in layouts:
            start_x = max(-10.0, conflict_x - 24.0)
            end_x = max(start_x + 12.0, conflict_x + 4.0)
            x = np.linspace(start_x, end_x, 64, dtype=np.float32)
            progress = (x - start_x) / max(end_x - start_x, 1e-3)
            center = lane_width * progress
            output.append(
                np.stack(
                    [x, (center - 0.5 * lane_width).astype(np.float32)],
                    axis=-1,
                )
            )
            output.append(
                np.stack(
                    [x, (center + 0.5 * lane_width).astype(np.float32)],
                    axis=-1,
                )
            )
        if "dedicated_right_turn_lane" in layouts:
            start_x = max(-10.0, conflict_x - 24.0)
            end_x = max(start_x + 12.0, conflict_x + 4.0)
            x = np.linspace(start_x, end_x, 64, dtype=np.float32)
            progress = (x - start_x) / max(end_x - start_x, 1e-3)
            center = -lane_width * progress
            output.append(
                np.stack(
                    [x, (center - 0.5 * lane_width).astype(np.float32)],
                    axis=-1,
                )
            )
            output.append(
                np.stack(
                    [x, (center + 0.5 * lane_width).astype(np.float32)],
                    axis=-1,
                )
            )
        return output

    @staticmethod
    def _straight_lines(
        *,
        lane_count: int,
        lane_width: float,
        curvature: float,
        directionality: str,
    ) -> List[np.ndarray]:
        x = np.linspace(-32.0, 52.0, 96, dtype=np.float32)
        # Keep the ego lane centered on y=0. Additional lanes are placed on
        # alternating sides to avoid moving the ego path away from the origin.
        centers = [0.0]
        step = 1
        while len(centers) < lane_count:
            centers.append(step * lane_width)
            if len(centers) < lane_count:
                centers.append(-step * lane_width)
            step += 1
        centers = centers[:lane_count]

        boundary_values = set()
        for center in centers:
            boundary_values.add(round(center - 0.5 * lane_width, 6))
            boundary_values.add(round(center + 0.5 * lane_width, 6))
        if directionality == "bidirectional" and lane_count == 1:
            boundary_values.update({-1.5 * lane_width, 1.5 * lane_width})

        lines: List[np.ndarray] = []
        bend = 0.5 * float(curvature) * x * x
        for y0 in sorted(boundary_values):
            y = float(y0) + bend
            lines.append(np.stack([x, y.astype(np.float32)], axis=-1))
        # Add ego lane centerline as a route anchor.
        lines.append(np.stack([x, bend.astype(np.float32)], axis=-1))
        return lines

    @staticmethod
    def _intersection_lines(
        *,
        lane_count: int,
        lane_width: float,
        conflict_x: float,
    ) -> List[np.ndarray]:
        x = np.linspace(-32.0, 52.0, 96, dtype=np.float32)
        lines: List[np.ndarray] = []
        for y in (-0.5 * lane_width, 0.5 * lane_width):
            lines.append(np.stack([x, np.full_like(x, y)], axis=-1))
        lines.append(np.stack([x, np.zeros_like(x)], axis=-1))

        y = np.linspace(-32.0, 32.0, 80, dtype=np.float32)
        cross_width = lane_width * max(1, min(lane_count, 2))
        for dx in (-0.5 * cross_width, 0.5 * cross_width):
            lines.append(
                np.stack(
                    [np.full_like(y, conflict_x + dx), y],
                    axis=-1,
                )
            )
        lines.append(
            np.stack([np.full_like(y, conflict_x), y], axis=-1)
        )
        return lines

    @staticmethod
    def _roundabout_lines(
        *,
        lane_count: int,
        lane_width: float,
        conflict_x: float,
    ) -> List[np.ndarray]:
        lane_count = max(1, int(lane_count))
        radius = max(9.0, 3.0 * lane_width)
        center_x = max(conflict_x + radius, 14.0 + radius)
        theta = np.linspace(-math.pi, math.pi, 128, dtype=np.float32)
        lines: List[np.ndarray] = []
        inner = radius - 0.5 * lane_count * lane_width
        radii = [inner + i * lane_width for i in range(lane_count + 1)]
        radii.append(radius)
        for r in sorted(set(max(3.0, float(v)) for v in radii)):
            lines.append(
                np.stack(
                    [
                        center_x + r * np.cos(theta),
                        r * np.sin(theta),
                    ],
                    axis=-1,
                ).astype(np.float32)
            )
        x = np.linspace(-32.0, center_x - radius - 0.5, 72, dtype=np.float32)
        for y0 in (-0.5 * lane_width, 0.5 * lane_width, 0.0):
            lines.append(
                np.stack([x, np.full_like(x, y0)], axis=-1)
            )
        return lines

    @staticmethod
    def _merge_diverge_lines(
        *,
        lane_width: float,
        conflict_x: float,
    ) -> List[np.ndarray]:
        x = np.linspace(-32.0, 52.0, 96, dtype=np.float32)
        lines = [
            np.stack([x, np.full_like(x, -0.5 * lane_width)], axis=-1),
            np.stack([x, np.full_like(x, 0.5 * lane_width)], axis=-1),
            np.stack([x, np.zeros_like(x)], axis=-1),
        ]
        merge_start = max(-8.0, conflict_x - 20.0)
        merge_end = max(8.0, conflict_x + 4.0)
        mx = np.linspace(merge_start, merge_end, 64, dtype=np.float32)
        progress = (mx - merge_start) / max(merge_end - merge_start, 1e-3)
        outer_y = -2.5 * lane_width + progress * 2.0 * lane_width
        inner_y = -1.5 * lane_width + progress * lane_width
        lines.append(np.stack([mx, outer_y.astype(np.float32)], axis=-1))
        lines.append(np.stack([mx, inner_y.astype(np.float32)], axis=-1))
        return lines

    @staticmethod
    def _work_zone_lines(
        *,
        lane_count: int,
        lane_width: float,
        curvature: float,
        conflict_x: float,
        lane_layout: Sequence[str],
    ) -> List[np.ndarray]:
        lines = TemplateSceneSynthesizer._straight_lines(
            lane_count=max(1, lane_count),
            lane_width=lane_width,
            curvature=curvature,
            directionality="unspecified",
        )
        if "closed_lane" in lane_layout or not lane_layout:
            x0 = max(2.0, conflict_x - 8.0)
            x1 = max(x0 + 8.0, conflict_x + 8.0)
            x = np.linspace(x0, x1, 40, dtype=np.float32)
            progress = (x - x0) / max(x1 - x0, 1e-3)
            y = 1.5 * lane_width - progress * lane_width
            lines.append(np.stack([x, y.astype(np.float32)], axis=-1))
        return lines

    def _write_polylines(
        self,
        scene: Any,
        polylines: Sequence[np.ndarray],
    ) -> int:
        elem = scene.lines
        states = np.asarray(elem.states)
        mask = np.asarray(elem.mask)
        if states.ndim < 3 or states.shape[0] == 0 or states.shape[1] == 0:
            return 0

        capacity = int(states.shape[0])
        num_points = int(states.shape[1])
        written = 0
        for slot, polyline in enumerate(polylines[:capacity]):
            sampled = self._resample_polyline(np.asarray(polyline, dtype=np.float32), num_points)
            states[slot, :, 0] = sampled[:, 0]
            if states.shape[-1] >= 2:
                states[slot, :, 1] = sampled[:, 1]
            if states.shape[-1] >= 3:
                gradients = np.gradient(sampled, axis=0)
                states[slot, :, 2] = np.arctan2(
                    gradients[:, 1], gradients[:, 0]
                )
            if mask.ndim == 1:
                mask[slot] = True
            elif mask.ndim >= 2:
                mask[slot, ...] = True
            written += 1
        elem.states = states
        elem.mask = mask
        return written

    @staticmethod
    def _resample_polyline(points: np.ndarray, num_points: int) -> np.ndarray:
        if len(points) == 0:
            return np.zeros((num_points, 2), dtype=np.float32)
        if len(points) == 1:
            return np.repeat(points[:, :2], num_points, axis=0).astype(np.float32)
        xy = np.asarray(points[:, :2], dtype=np.float32)
        delta = np.diff(xy, axis=0)
        segment = np.linalg.norm(delta, axis=1)
        distance = np.concatenate([[0.0], np.cumsum(segment)])
        if float(distance[-1]) <= 1e-6:
            return np.repeat(xy[:1], num_points, axis=0)
        sample_distance = np.linspace(0.0, float(distance[-1]), num_points)
        x = np.interp(sample_distance, distance, xy[:, 0])
        y = np.interp(sample_distance, distance, xy[:, 1])
        return np.stack([x, y], axis=-1).astype(np.float32)


__all__ = ["TemplateSceneSynthesizer"]