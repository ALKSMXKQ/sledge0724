"""Inference-time lane geometry guidance and quality metrics.

Guidance operates on decoded line queries and differentiates the geometry loss
back to diffusion latents. It does not force the generated road to equal B1.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class LaneGuidanceConfig:
    enabled: bool = False
    start_ratio: float = 0.50
    strong_ratio: float = 0.80
    weak_strength: float = 0.010
    strong_strength: float = 0.035
    heading_weight: float = 1.0
    curvature_weight: float = 0.7
    adjacent_heading_weight: float = 0.35
    width_weight: float = 0.20
    mask_threshold: float = 0.30
    grad_clip: float = 4.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "LaneGuidanceConfig":
        if not value:
            return cls()
        schedule = dict(value.get("schedule", {}) or {})
        return cls(
            enabled=bool(value.get("enabled", False)),
            start_ratio=float(schedule.get("start_ratio", value.get("start_ratio", 0.50))),
            strong_ratio=float(schedule.get("strong_ratio", value.get("strong_ratio", 0.80))),
            weak_strength=float(value.get("weak_strength", 0.010)),
            strong_strength=float(value.get("strong_strength", 0.035)),
            heading_weight=float(value.get("heading_weight", 1.0)),
            curvature_weight=float(value.get("curvature_weight", 0.7)),
            adjacent_heading_weight=float(value.get("adjacent_heading_weight", 0.35)),
            width_weight=float(value.get("width_weight", 0.20)),
            mask_threshold=float(value.get("mask_threshold", 0.30)),
            grad_clip=float(value.get("grad_clip", 4.0)),
        )

    def strength(self, progress: float, topology_family: str = "road_segment") -> float:
        if not self.enabled or progress < self.start_ratio:
            return 0.0
        if progress >= self.strong_ratio:
            base = self.strong_strength
        else:
            span = max(self.strong_ratio - self.start_ratio, 1e-6)
            alpha = (progress - self.start_ratio) / span
            base = self.weak_strength + alpha * (self.strong_strength - self.weak_strength)
        if topology_family == "intersection":
            return 0.65 * base
        if topology_family == "merge_split":
            return 0.80 * base
        return base


def lane_geometry_loss(
    line_states: torch.Tensor,
    line_mask_logits: torch.Tensor,
    *,
    config: LaneGuidanceConfig,
    topology_family: str = "road_segment",
    adjacency_pairs: Sequence[Mapping[str, Any]] | None = None,
) -> torch.Tensor:
    """Differentiable smooth-road loss on decoded line queries."""
    xy = line_states[..., :2]
    if xy.ndim != 4 or xy.shape[-2] < 4:
        return xy.sum() * 0.0
    mask = torch.sigmoid(line_mask_logits)
    valid_weight = (mask * (mask >= config.mask_threshold).to(mask.dtype)).unsqueeze(-1)

    seg = xy[:, :, 1:] - xy[:, :, :-1]
    ds = torch.linalg.vector_norm(seg, dim=-1).clamp_min(0.20)
    unit = seg / ds.unsqueeze(-1)
    # 1-cos(theta) gives wrapped local heading continuity without atan2 jumps.
    cos_turn = (unit[:, :, 1:] * unit[:, :, :-1]).sum(-1).clamp(-1.0, 1.0)
    heading = ((1.0 - cos_turn) / ds[:, :, 1:].clamp_min(0.20)).mean(-1)

    tangent_delta = unit[:, :, 1:] - unit[:, :, :-1]
    curvature = torch.linalg.vector_norm(tangent_delta, dim=-1) / ds[:, :, 1:].clamp_min(0.20)
    curvature_change = (curvature[:, :, 1:] - curvature[:, :, :-1]).square().mean(-1)
    single_loss = (
        config.heading_weight * heading
        + config.curvature_weight * curvature_change
    )
    loss = (single_loss * valid_weight.squeeze(-1)).sum() / valid_weight.sum().clamp_min(1.0)

    pairs = list(adjacency_pairs or [])
    if pairs and topology_family != "intersection":
        pair_losses = []
        for pair in pairs:
            i, j = int(pair["a"]), int(pair["b"])
            if i >= xy.shape[1] or j >= xy.shape[1]:
                continue
            b = torch.flip(xy[:, j], dims=[1]) if bool(pair.get("reverse_b", False)) else xy[:, j]
            a = xy[:, i]
            ua = a[:, 1:] - a[:, :-1]
            ub = b[:, 1:] - b[:, :-1]
            ua = ua / torch.linalg.vector_norm(ua, dim=-1, keepdim=True).clamp_min(0.20)
            ub = ub / torch.linalg.vector_norm(ub, dim=-1, keepdim=True).clamp_min(0.20)
            parallel = (1.0 - (ua * ub).sum(-1).clamp(-1.0, 1.0)).mean()
            widths = torch.linalg.vector_norm(a - b, dim=-1)
            target = float(pair.get("target_width", 0.0))
            width_var = (widths - widths.mean(dim=-1, keepdim=True)).square().mean()
            width_target = (widths.mean() - target).square() if target > 0.0 else widths.sum() * 0.0
            pair_losses.append(
                config.adjacent_heading_weight * parallel
                + config.width_weight * (width_var + 0.25 * width_target)
            )
        if pair_losses:
            loss = loss + torch.stack(pair_losses).mean()
    return loss


def apply_latent_lane_guidance(
    decoder: Any,
    latents: torch.Tensor,
    *,
    config: LaneGuidanceConfig,
    progress: float,
    topology_family: str = "road_segment",
    adjacency_pairs: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    strength = config.strength(progress, topology_family)
    if strength <= 0.0:
        return latents, {"lane_guidance_strength": 0.0, "lane_guidance_loss": 0.0}
    with torch.enable_grad():
        x = latents.detach().requires_grad_(True)
        decoded = decoder.decode(x)
        loss = lane_geometry_loss(
            decoded.lines.states,
            decoded.lines.mask,
            config=config,
            topology_family=topology_family,
            adjacency_pairs=adjacency_pairs,
        )
        grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
        rms = grad.square().mean().sqrt().clamp_min(1e-6)
        grad = (grad / rms).clamp(-config.grad_clip, config.grad_clip)
        corrected = (x - strength * grad).detach()
    return corrected, {
        "lane_guidance_strength": float(strength),
        "lane_guidance_loss": float(loss.detach().cpu()),
    }


def infer_adjacency_pairs(states: np.ndarray, mask: np.ndarray) -> List[Dict[str, Any]]:
    """Infer conservative pairs from B1 query slots; returns no pair when uncertain."""
    states = np.asarray(states, dtype=np.float64)
    mask = np.asarray(mask).reshape(-1)
    pairs: List[Dict[str, Any]] = []
    if states.ndim != 3:
        return pairs
    valid = [i for i in range(min(len(mask), len(states))) if float(mask[i]) >= 0.3]
    for pos, i in enumerate(valid):
        a = states[i, :, :2]
        if len(a) < 4:
            continue
        for j in valid[pos + 1:]:
            b0 = states[j, :, :2]
            choices = [(False, b0), (True, b0[::-1])]
            reverse, b = min(choices, key=lambda rb: float(np.mean(np.linalg.norm(a - rb[1], axis=1))))
            widths = np.linalg.norm(a - b, axis=1)
            median_width = float(np.median(widths))
            if not 2.4 <= median_width <= 5.2:
                continue
            ua = np.diff(a, axis=0); ub = np.diff(b, axis=0)
            ua /= np.maximum(np.linalg.norm(ua, axis=1, keepdims=True), 1e-6)
            ub /= np.maximum(np.linalg.norm(ub, axis=1, keepdims=True), 1e-6)
            mean_cos = float(np.mean(np.sum(ua * ub, axis=1)))
            if mean_cos < math.cos(0.25):
                continue
            if float(np.std(widths)) > 1.2:
                continue
            pairs.append({"a": i, "b": j, "reverse_b": reverse, "target_width": median_width})
    return pairs


def resample_polyline(points: np.ndarray, spacing_m: float = 1.0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points.copy()
    seg = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], np.diff(s) > 1e-6])
    points, s = points[keep], s[keep]
    if len(points) < 2 or s[-1] < spacing_m:
        return points[:, :2].copy()
    target = np.arange(0.0, s[-1] + 1e-6, spacing_m)
    return np.stack([np.interp(target, s, points[:, 0]), np.interp(target, s, points[:, 1])], axis=1)


def polyline_quality(points: np.ndarray, spacing_m: float = 1.0) -> Dict[str, float]:
    p = resample_polyline(points, spacing_m)
    if len(p) < 4:
        return {"heading_jump_max": float("nan"), "curvature_max": float("nan"), "curvature_change": float("nan")}
    d = np.diff(p, axis=0)
    heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    jump = np.diff(heading)
    curvature = jump / max(spacing_m, 1e-6)
    return {
        "heading_jump_max": float(np.max(np.abs(jump))) if len(jump) else 0.0,
        "curvature_max": float(np.max(np.abs(curvature))) if len(curvature) else 0.0,
        "curvature_change": float(np.mean(np.abs(np.diff(curvature)))) if len(curvature) > 1 else 0.0,
    }


__all__ = [
    "LaneGuidanceConfig", "apply_latent_lane_guidance", "lane_geometry_loss",
    "infer_adjacency_pairs", "resample_polyline", "polyline_quality",
]
