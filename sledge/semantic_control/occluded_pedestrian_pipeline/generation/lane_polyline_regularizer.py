"""Lightweight post-decoding regularization for generated road polylines.

This module intentionally does not modify the diffusion model, scheduler, or
latent denoising process. It only removes high-frequency geometric wiggle from
decoded SLEDGE line polylines while preserving endpoints and genuine low-
frequency bends.

The key distinction is between a smooth bend and an irregular wiggle:
- a smooth bend accumulates heading change mostly in one direction;
- a wiggle repeatedly changes curvature sign and therefore has substantial
  ``total_turn - abs(net_turn)``.

Only suspicious polylines are regularized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class LanePolylineRegularizerConfig:
    enabled: bool = True
    mask_threshold: float = 0.30
    min_points: int = 6
    min_excess_turn_rad: float = np.deg2rad(8.0)
    min_sign_flip_ratio: float = 0.20
    min_heading_roughness_rad: float = np.deg2rad(4.0)
    min_large_heading_jump_rad: float = np.deg2rad(16.0)
    base_smoothing_lambda: float = 2.0
    max_smoothing_lambda: float = 12.0
    max_point_displacement_m: float = 0.80

    @classmethod
    def from_mapping(cls, value: Any) -> "LanePolylineRegularizerConfig":
        if not isinstance(value, dict):
            return cls()
        return cls(
            enabled=bool(value.get("enabled", True)),
            mask_threshold=float(value.get("mask_threshold", 0.30)),
            min_points=int(value.get("min_points", 6)),
            min_excess_turn_rad=float(
                value.get("min_excess_turn_rad", np.deg2rad(8.0))
            ),
            min_sign_flip_ratio=float(value.get("min_sign_flip_ratio", 0.20)),
            min_heading_roughness_rad=float(
                value.get("min_heading_roughness_rad", np.deg2rad(4.0))
            ),
            min_large_heading_jump_rad=float(
                value.get("min_large_heading_jump_rad", np.deg2rad(16.0))
            ),
            base_smoothing_lambda=float(value.get("base_smoothing_lambda", 2.0)),
            max_smoothing_lambda=float(value.get("max_smoothing_lambda", 12.0)),
            max_point_displacement_m=float(
                value.get("max_point_displacement_m", 0.80)
            ),
        )


def _polyline_metrics(points: np.ndarray) -> Dict[str, float]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) < 3:
        return {
            "net_turn_rad": 0.0,
            "total_turn_rad": 0.0,
            "excess_turn_rad": 0.0,
            "sign_flip_ratio": 0.0,
            "heading_roughness_rad": 0.0,
            "max_heading_jump_rad": 0.0,
        }

    delta = np.diff(points[:, :2], axis=0)
    seg_len = np.linalg.norm(delta, axis=1)
    valid = seg_len > 1e-5
    if int(valid.sum()) < 2:
        return {
            "net_turn_rad": 0.0,
            "total_turn_rad": 0.0,
            "excess_turn_rad": 0.0,
            "sign_flip_ratio": 0.0,
            "heading_roughness_rad": 0.0,
            "max_heading_jump_rad": 0.0,
        }

    heading = np.unwrap(np.arctan2(delta[valid, 1], delta[valid, 0]))
    turn = np.diff(heading)
    if len(turn) == 0:
        return {
            "net_turn_rad": 0.0,
            "total_turn_rad": 0.0,
            "excess_turn_rad": 0.0,
            "sign_flip_ratio": 0.0,
            "heading_roughness_rad": 0.0,
            "max_heading_jump_rad": 0.0,
        }

    net_turn = float(abs(np.sum(turn)))
    total_turn = float(np.sum(np.abs(turn)))
    excess_turn = float(max(0.0, total_turn - net_turn))

    significant = turn[np.abs(turn) >= np.deg2rad(1.5)]
    if len(significant) >= 2:
        signs = np.sign(significant)
        sign_flip_ratio = float(
            np.mean(signs[1:] * signs[:-1] < 0.0)
        )
    else:
        sign_flip_ratio = 0.0

    heading_roughness = (
        float(np.mean(np.abs(np.diff(turn)))) if len(turn) >= 2 else 0.0
    )
    max_heading_jump = float(np.max(np.abs(turn)))
    return {
        "net_turn_rad": net_turn,
        "total_turn_rad": total_turn,
        "excess_turn_rad": excess_turn,
        "sign_flip_ratio": sign_flip_ratio,
        "heading_roughness_rad": heading_roughness,
        "max_heading_jump_rad": max_heading_jump,
    }


def _is_irregular(
    metrics: Dict[str, float],
    config: LanePolylineRegularizerConfig,
) -> bool:
    oscillatory = (
        metrics["excess_turn_rad"] >= config.min_excess_turn_rad
        and (
            metrics["sign_flip_ratio"] >= config.min_sign_flip_ratio
            or metrics["heading_roughness_rad"]
            >= config.min_heading_roughness_rad
        )
    )
    severe_local_wiggle = (
        metrics["max_heading_jump_rad"] >= config.min_large_heading_jump_rad
        and metrics["sign_flip_ratio"] >= config.min_sign_flip_ratio
    )
    return bool(oscillatory or severe_local_wiggle)


def _adaptive_lambda(
    metrics: Dict[str, float],
    config: LanePolylineRegularizerConfig,
) -> float:
    excess_scale = min(
        1.0,
        metrics["excess_turn_rad"] / max(np.deg2rad(35.0), 1e-6),
    )
    rough_scale = min(
        1.0,
        metrics["heading_roughness_rad"] / max(np.deg2rad(18.0), 1e-6),
    )
    severity = float(np.clip(0.65 * excess_scale + 0.35 * rough_scale, 0.0, 1.0))
    return float(
        config.base_smoothing_lambda
        + severity
        * (config.max_smoothing_lambda - config.base_smoothing_lambda)
    )


def _smooth_with_fixed_endpoints(points: np.ndarray, smoothing_lambda: float) -> np.ndarray:
    """Second-difference Tikhonov smoothing with exact endpoint preservation."""

    y = np.asarray(points, dtype=np.float64)
    n = len(y)
    if n < 4 or smoothing_lambda <= 0.0:
        return y.copy()

    d2 = np.zeros((n - 2, n), dtype=np.float64)
    rows = np.arange(n - 2)
    d2[rows, rows] = 1.0
    d2[rows, rows + 1] = -2.0
    d2[rows, rows + 2] = 1.0

    a = np.eye(n, dtype=np.float64) + float(smoothing_lambda) * (d2.T @ d2)
    interior = np.arange(1, n - 1)
    boundary = np.asarray([0, n - 1], dtype=np.int64)

    a_ii = a[np.ix_(interior, interior)]
    a_ib = a[np.ix_(interior, boundary)]
    rhs = y[interior] - a_ib @ y[boundary]

    out = y.copy()
    out[interior] = np.linalg.solve(a_ii, rhs)
    out[0] = y[0]
    out[-1] = y[-1]
    return out


def regularize_polyline(
    points: np.ndarray,
    config: LanePolylineRegularizerConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    points = np.asarray(points, dtype=np.float64)
    before = _polyline_metrics(points)
    if len(points) < config.min_points or not _is_irregular(before, config):
        return points.copy(), {
            "applied": False,
            "reason": "not_irregular",
            "before": before,
            "after": before,
            "smoothing_lambda": 0.0,
            "max_point_displacement_m": 0.0,
        }

    smoothing_lambda = _adaptive_lambda(before, config)
    smoothed = _smooth_with_fixed_endpoints(points, smoothing_lambda)

    delta = smoothed - points
    displacement = np.linalg.norm(delta[:, :2], axis=1)
    max_displacement = float(displacement.max(initial=0.0))
    if max_displacement > config.max_point_displacement_m > 0.0:
        scale = config.max_point_displacement_m / max(max_displacement, 1e-6)
        smoothed = points + scale * delta
        smoothed[0] = points[0]
        smoothed[-1] = points[-1]
        max_displacement = float(config.max_point_displacement_m)

    after = _polyline_metrics(smoothed)
    # Never keep a correction that makes high-frequency turn worse.
    if after["excess_turn_rad"] > before["excess_turn_rad"] + 1e-6:
        return points.copy(), {
            "applied": False,
            "reason": "rejected_no_improvement",
            "before": before,
            "after": before,
            "smoothing_lambda": smoothing_lambda,
            "max_point_displacement_m": 0.0,
        }

    return smoothed, {
        "applied": True,
        "reason": "high_frequency_wiggle",
        "before": before,
        "after": after,
        "smoothing_lambda": smoothing_lambda,
        "max_point_displacement_m": max_displacement,
    }


def regularize_sledge_lines(
    vector: Any,
    config: LanePolylineRegularizerConfig | None = None,
) -> Dict[str, Any]:
    """Regularize valid ``vector.lines`` in place and return diagnostics."""

    config = config or LanePolylineRegularizerConfig()
    report: Dict[str, Any] = {
        "schema_version": "lane_polyline_regularization_v1",
        "enabled": bool(config.enabled),
        "policy": "post_decode_high_frequency_wiggle_only",
        "valid_line_count": 0,
        "regularized_line_count": 0,
        "lines": [],
    }
    if not config.enabled:
        return report

    states = np.asarray(vector.lines.states)
    masks = np.asarray(vector.lines.mask).reshape(-1)
    if states.ndim != 3 or states.shape[-1] < 2:
        report["error"] = f"unexpected line state shape {tuple(states.shape)}"
        return report

    for index in range(min(len(states), len(masks))):
        if float(masks[index]) < config.mask_threshold:
            continue
        report["valid_line_count"] += 1
        original = np.asarray(states[index, :, :2], dtype=np.float64)
        corrected, diag = regularize_polyline(original, config)
        if bool(diag["applied"]):
            states[index, :, :2] = corrected.astype(states.dtype, copy=False)
            report["regularized_line_count"] += 1
        report["lines"].append({"index": int(index), **diag})

    return report


__all__ = [
    "LanePolylineRegularizerConfig",
    "regularize_polyline",
    "regularize_sledge_lines",
]
