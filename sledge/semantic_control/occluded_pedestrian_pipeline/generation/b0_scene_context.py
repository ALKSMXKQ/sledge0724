"""Extract road/ego context that must be preserved in edit-existing mode."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
        EgoIndex,
    )
except Exception:  # pragma: no cover - only for lightweight tooling/imports
    EgoIndex = None  # type: ignore


@dataclass(frozen=True)
class B0SceneContext:
    """Small, serializable snapshot of B0 values used by the sampler."""

    lane_width_m: float
    lane_count: int
    road_curvature: float
    ego_speed_mps: float
    ego_acceleration_mps2: float
    active_line_count: int
    extraction_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class B0SceneContextExtractor:
    """Infer conservative editable context without changing ``SledgeVectorRaw``.

    SLEDGE raw vectors do not store a canonical lane count/width scalar. Those
    values are therefore estimated only for parameter compatibility and metric
    normalization. The original line geometry remains the source of truth in
    edit-existing mode and is never rewritten from these estimates.
    """

    DEFAULT_LANE_WIDTH_M = 3.5

    def extract(self, scene: Any) -> B0SceneContext:
        notes: List[str] = []
        ego_speed, ego_accel = self._ego_kinematics(scene, notes)
        polylines = self._valid_polylines(scene)
        active_line_count = len(polylines)
        lane_width = self._estimate_lane_width(polylines, notes)
        lane_count = self._estimate_lane_count(polylines, lane_width, notes)
        curvature = self._estimate_road_curvature(polylines, notes)

        return B0SceneContext(
            lane_width_m=float(lane_width),
            lane_count=int(max(1, lane_count)),
            road_curvature=float(curvature),
            ego_speed_mps=float(max(ego_speed, 0.1)),
            ego_acceleration_mps2=float(ego_accel),
            active_line_count=int(active_line_count),
            extraction_notes=notes,
        )

    @staticmethod
    def _ego_kinematics(scene: Any, notes: List[str]) -> Tuple[float, float]:
        states = np.asarray(getattr(scene.ego, "states", []), dtype=np.float32).reshape(-1)
        if states.size == 0:
            notes.append("ego state missing; used 6.0 m/s and 0.0 m/s^2 fallback")
            return 6.0, 0.0

        # Sledge EgoIndex layout: vx, vy, ax, ay.
        vx = float(states[0]) if states.size >= 1 else 0.0
        vy = float(states[1]) if states.size >= 2 else 0.0
        ax = float(states[2]) if states.size >= 3 else 0.0
        ay = float(states[3]) if states.size >= 4 else 0.0
        speed = math.hypot(vx, vy)
        acceleration = math.hypot(ax, ay)
        if speed < 0.1:
            # Keep the source meaning (possibly nearly stopped) while avoiding
            # division-by-zero later in the hazard sampler.
            notes.append("ego speed below 0.1 m/s; sampler will clamp to 0.1 m/s")
        return speed, acceleration

    @staticmethod
    def _valid_polylines(scene: Any) -> List[np.ndarray]:
        elem = getattr(scene, "lines", None)
        if elem is None:
            return []
        states = np.asarray(getattr(elem, "states", []), dtype=np.float32)
        masks = np.asarray(getattr(elem, "mask", []))
        if states.ndim < 3 or states.shape[-1] < 2:
            return []

        output: List[np.ndarray] = []
        for index in range(states.shape[0]):
            state = states[index, :, :2]
            if masks.size == 0:
                valid_points = np.ones((state.shape[0],), dtype=bool)
            elif masks.ndim == 1:
                if index >= masks.shape[0] or not bool(masks[index]):
                    continue
                valid_points = np.ones((state.shape[0],), dtype=bool)
            else:
                if index >= masks.shape[0]:
                    continue
                row = np.asarray(masks[index]).reshape(-1)
                valid_points = row[: state.shape[0]].astype(bool)
                if not np.any(valid_points):
                    continue
            points = state[valid_points]
            points = points[np.all(np.isfinite(points), axis=1)]
            if len(points) >= 2:
                output.append(points.astype(np.float32, copy=False))
        return output

    def _estimate_lane_width(
        self,
        polylines: Sequence[np.ndarray],
        notes: List[str],
    ) -> float:
        y_near_ego: List[float] = []
        for line in polylines:
            points = np.asarray(line, dtype=np.float32)
            near = points[np.abs(points[:, 0]) <= 12.0]
            if len(near) == 0:
                nearest = points[int(np.argmin(np.abs(points[:, 0])))]
                if abs(float(nearest[0])) <= 25.0:
                    y_near_ego.append(float(nearest[1]))
            else:
                y_near_ego.append(float(np.median(near[:, 1])))

        if len(y_near_ego) < 2:
            notes.append("lane width not recoverable from lines; used 3.5 m fallback")
            return self.DEFAULT_LANE_WIDTH_M

        # Merge near-duplicate markings/centerlines before taking spacings.
        ys = sorted(y_near_ego)
        merged: List[float] = []
        for value in ys:
            if not merged or abs(value - merged[-1]) >= 0.6:
                merged.append(value)
            else:
                merged[-1] = 0.5 * (merged[-1] + value)
        spacings = [
            merged[i + 1] - merged[i]
            for i in range(len(merged) - 1)
            if 2.4 <= merged[i + 1] - merged[i] <= 5.2
        ]
        if not spacings:
            notes.append("no plausible lane-boundary spacing; used 3.5 m fallback")
            return self.DEFAULT_LANE_WIDTH_M
        estimate = float(np.median(spacings))
        notes.append(f"estimated lane width from local line spacing: {estimate:.3f} m")
        return float(np.clip(estimate, 2.8, 4.5))

    @staticmethod
    def _estimate_lane_count(
        polylines: Sequence[np.ndarray],
        lane_width_m: float,
        notes: List[str],
    ) -> int:
        y_values: List[float] = []
        for line in polylines:
            points = np.asarray(line, dtype=np.float32)
            near = points[np.abs(points[:, 0]) <= 10.0]
            if len(near):
                y_values.append(float(np.median(near[:, 1])))
        if not y_values:
            notes.append("lane count not recoverable; used 1-lane compatibility value")
            return 1

        # This is deliberately only a compatibility estimate. It must never be
        # used to rewrite B0 in edit-existing mode.
        width = max(float(lane_width_m), 2.5)
        bands = sorted({int(round(value / width)) for value in y_values if abs(value) <= 12.0})
        if len(bands) <= 1:
            return 1
        estimate = int(np.clip(max(1, len(bands) - 1), 1, 6))
        notes.append(f"estimated lane-count compatibility value: {estimate}")
        return estimate

    @staticmethod
    def _estimate_road_curvature(
        polylines: Sequence[np.ndarray],
        notes: List[str],
    ) -> float:
        if not polylines:
            notes.append("road curvature not recoverable; used 0.0 fallback")
            return 0.0

        # Choose the line closest to ego and fit y = ax^2 + bx + c. For a
        # near-horizontal lane, curvature is approximately 2a around x=0.
        def score(line: np.ndarray) -> float:
            points = np.asarray(line)
            near = points[np.abs(points[:, 0]) <= 15.0]
            target = near if len(near) else points
            return float(np.median(np.abs(target[:, 1])))

        line = min(polylines, key=score)
        points = np.asarray(line, dtype=np.float64)
        points = points[np.abs(points[:, 0]) <= 35.0]
        if len(points) < 5 or float(np.ptp(points[:, 0])) < 8.0:
            notes.append("insufficient line span for curvature; used 0.0 fallback")
            return 0.0
        try:
            coeff = np.polyfit(points[:, 0], points[:, 1], 2)
            curvature = float(2.0 * coeff[0])
        except Exception:
            notes.append("curvature fit failed; used 0.0 fallback")
            return 0.0
        curvature = float(np.clip(curvature, -0.05, 0.05))
        notes.append(f"estimated local road curvature: {curvature:.6f} 1/m")
        return curvature


__all__ = ["B0SceneContext", "B0SceneContextExtractor"]