"""Topology-local inference-time road guidance for latent diffusion.

The previous experimental guidance used one scene-wide topology label and
turned corridor-pair losses off for the whole scene when an intersection was
detected.  Real scenes mix straight approaches, curves, junction interiors and
exits, so this module uses local confidence instead:

* intrinsic line regularity is always available;
* corridor width/parallelism is weighted per polyline sample;
* low-confidence junction interiors are naturally down-weighted;
* directed endpoint connectors preserve approach -> connector -> exit
  continuity without snapping generated geometry back to the source scene.

The reference contains relationships only (width, local confidence and graph
connectivity).  It does not contain an absolute target polyline, so guidance
can change the generated road while retaining usable topology.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LaneGuidanceConfig:
    """Schedule and loss weights for latent road guidance."""

    enabled: bool = True
    start_ratio: float = 0.45
    strong_ratio: float = 0.80
    weak_strength: float = 0.006
    strong_strength: float = 0.025
    # Keep absolute bending pressure modest so constant-curvature roads are not
    # artificially flattened.  Curvature *change* carries most smoothness.
    heading_weight: float = 0.20
    curvature_weight: float = 0.70
    corridor_heading_weight: float = 0.35
    corridor_width_weight: float = 0.20
    corridor_width_variation_weight: float = 0.12
    connectivity_weight: float = 0.35
    connector_heading_weight: float = 0.20
    mask_threshold: float = 0.30
    min_pair_confidence: float = 0.15
    grad_clip: float = 4.0

    @classmethod
    def from_mapping(
        cls,
        value: Optional[Mapping[str, Any]],
    ) -> "LaneGuidanceConfig":
        if not value:
            return cls()
        schedule = dict(value.get("schedule", {}) or {})
        kwargs: Dict[str, Any] = {}
        for field_name in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            if field_name in value:
                kwargs[field_name] = value[field_name]
        if "start_ratio" in schedule:
            kwargs["start_ratio"] = schedule["start_ratio"]
        if "strong_ratio" in schedule:
            kwargs["strong_ratio"] = schedule["strong_ratio"]
        return cls(**kwargs)

    def strength(self, progress: float) -> float:
        """Progress-only schedule; topology does not globally scale strength."""

        progress = float(progress)
        if not self.enabled or progress < self.start_ratio:
            return 0.0
        if progress >= self.strong_ratio:
            return float(self.strong_strength)
        span = max(self.strong_ratio - self.start_ratio, 1e-6)
        alpha = (progress - self.start_ratio) / span
        return float(
            self.weak_strength
            + alpha * (self.strong_strength - self.weak_strength)
        )


@dataclass(frozen=True)
class LaneCorridorReference:
    """Two query slots that locally act as boundaries of one road corridor."""

    a: int
    b: int
    reverse_b: bool
    target_width_m: float
    point_confidence: Tuple[float, ...]
    mean_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "a": int(self.a),
            "b": int(self.b),
            "reverse_b": bool(self.reverse_b),
            "target_width_m": float(self.target_width_m),
            "mean_confidence": float(self.mean_confidence),
            "point_confidence": [float(v) for v in self.point_confidence],
        }


@dataclass(frozen=True)
class LaneConnectorReference:
    """Directed endpoint relationship between two generated road queries."""

    source: int
    target: int
    target_gap_m: float
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": int(self.source),
            "target": int(self.target),
            "target_gap_m": float(self.target_gap_m),
            "confidence": float(self.confidence),
        }


@dataclass(frozen=True)
class LaneConstraintReference:
    """Topology relationships extracted from the edited/source road queries."""

    corridors: Tuple[LaneCorridorReference, ...] = ()
    connectors: Tuple[LaneConnectorReference, ...] = ()

    @property
    def mean_pair_confidence(self) -> float:
        if not self.corridors:
            return 0.0
        return float(np.mean([pair.mean_confidence for pair in self.corridors]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corridor_count": len(self.corridors),
            "connector_count": len(self.connectors),
            "mean_pair_confidence": self.mean_pair_confidence,
            "corridors": [pair.to_dict() for pair in self.corridors],
            "connectors": [connector.to_dict() for connector in self.connectors],
        }


def infer_lane_constraint_reference(
    states: np.ndarray,
    mask: np.ndarray,
    *,
    mask_threshold: float = 0.30,
) -> LaneConstraintReference:
    """Infer local corridor confidence and directed endpoint connectivity.

    A corridor pair is retained when a meaningful fraction of its samples looks
    lane-like.  Samples that diverge/cross inside a junction receive low local
    confidence instead of causing the entire pair to be discarded.
    """

    states = np.asarray(states, dtype=np.float64)
    mask = np.asarray(mask).reshape(-1)
    if states.ndim != 3 or states.shape[-1] < 2:
        return LaneConstraintReference()

    valid = [
        index
        for index in range(min(len(states), len(mask)))
        if _mask_value(mask[index]) >= float(mask_threshold)
    ]
    corridors: List[LaneCorridorReference] = []

    for pos, i in enumerate(valid):
        a = np.asarray(states[i, :, :2], dtype=np.float64)
        if len(a) < 4 or not np.isfinite(a).all():
            continue
        for j in valid[pos + 1 :]:
            b0 = np.asarray(states[j, :, :2], dtype=np.float64)
            if b0.shape != a.shape or not np.isfinite(b0).all():
                continue

            choices = [(False, b0), (True, b0[::-1])]
            reverse_b, b = min(
                choices,
                key=lambda row: float(
                    np.mean(np.linalg.norm(a - row[1], axis=1))
                ),
            )
            widths = np.linalg.norm(a - b, axis=1)
            ua = _unit_segments(a)
            ub = _unit_segments(b)
            if len(ua) == 0 or len(ub) != len(ua):
                continue
            cos_parallel = np.sum(ua * ub, axis=1)
            point_cos = _segment_values_to_points(cos_parallel, len(a))

            stable = (
                (widths >= 2.2)
                & (widths <= 6.2)
                & (point_cos >= math.cos(math.radians(50.0)))
            )
            stable_fraction = float(np.mean(stable))
            if stable_fraction < 0.30:
                continue

            target_width = float(np.median(widths[stable]))
            width_scale = max(0.65, 0.25 * target_width)
            width_conf = np.exp(
                -0.5 * ((widths - target_width) / width_scale) ** 2
            )
            heading_floor = math.cos(math.radians(65.0))
            heading_conf = np.clip(
                (point_cos - heading_floor) / max(1.0 - heading_floor, 1e-6),
                0.0,
                1.0,
            )
            confidence = width_conf * heading_conf
            # Samples with implausibly tiny/huge width should exert almost no
            # corridor pressure even if headings happen to align.
            confidence *= np.where(
                (widths >= 1.8) & (widths <= 7.0),
                1.0,
                0.05,
            )
            mean_confidence = float(np.mean(confidence))
            if mean_confidence < 0.12:
                continue

            corridors.append(
                LaneCorridorReference(
                    a=int(i),
                    b=int(j),
                    reverse_b=bool(reverse_b),
                    target_width_m=target_width,
                    point_confidence=tuple(float(v) for v in confidence),
                    mean_confidence=mean_confidence,
                )
            )

    connectors: List[LaneConnectorReference] = []
    for i in valid:
        source = np.asarray(states[i, :, :2], dtype=np.float64)
        if len(source) < 2 or not np.isfinite(source).all():
            continue
        source_heading = _endpoint_heading(source, at_start=False)
        for j in valid:
            if i == j:
                continue
            target = np.asarray(states[j, :, :2], dtype=np.float64)
            if len(target) < 2 or not np.isfinite(target).all():
                continue
            gap = float(np.linalg.norm(source[-1] - target[0]))
            if gap > 2.5:
                continue
            target_heading = _endpoint_heading(target, at_start=True)
            heading_error = abs(_wrap_angle(source_heading - target_heading))
            if heading_error > math.radians(60.0):
                continue
            confidence = float(
                math.exp(-gap / 1.5)
                * max(0.0, math.cos(heading_error))
            )
            if confidence < 0.10:
                continue
            connectors.append(
                LaneConnectorReference(
                    source=int(i),
                    target=int(j),
                    target_gap_m=min(gap, 0.75),
                    confidence=confidence,
                )
            )

    # Keep only the strongest directed relation for each ordered query pair.
    dedup: Dict[Tuple[int, int], LaneConnectorReference] = {}
    for connector in connectors:
        key = (connector.source, connector.target)
        if key not in dedup or connector.confidence > dedup[key].confidence:
            dedup[key] = connector

    return LaneConstraintReference(
        corridors=tuple(corridors),
        connectors=tuple(dedup.values()),
    )


def lane_constraint_loss(
    line_states: torch.Tensor,
    line_mask_logits: torch.Tensor,
    *,
    reference: LaneConstraintReference,
    config: LaneGuidanceConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Differentiable topology-local road loss on decoded line queries."""

    xy = line_states[..., :2]
    if xy.ndim != 4 or xy.shape[-2] < 4:
        zero = xy.sum() * 0.0
        return zero, {
            "intrinsic": zero,
            "corridor": zero,
            "connectivity": zero,
        }

    query_weight = _query_probability(line_mask_logits, xy.shape[:2])

    seg = xy[:, :, 1:] - xy[:, :, :-1]
    ds = torch.linalg.vector_norm(seg, dim=-1).clamp_min(0.20)
    unit = seg / ds.unsqueeze(-1)
    cos_turn = (
        unit[:, :, 1:] * unit[:, :, :-1]
    ).sum(-1).clamp(-1.0, 1.0)
    local_bend = ((1.0 - cos_turn) / ds[:, :, 1:].clamp_min(0.20)).mean(-1)

    tangent_delta = unit[:, :, 1:] - unit[:, :, :-1]
    curvature = (
        torch.linalg.vector_norm(tangent_delta, dim=-1)
        / ds[:, :, 1:].clamp_min(0.20)
    )
    if curvature.shape[-1] >= 2:
        curvature_change = (
            curvature[:, :, 1:] - curvature[:, :, :-1]
        ).square().mean(-1)
    else:
        curvature_change = curvature * 0.0
        curvature_change = curvature_change.mean(-1)

    intrinsic_per_query = (
        config.heading_weight * local_bend
        + config.curvature_weight * curvature_change
    )
    intrinsic = (
        intrinsic_per_query * query_weight
    ).sum() / query_weight.sum().clamp_min(1.0)

    corridor_terms: List[torch.Tensor] = []
    for pair in reference.corridors:
        i, j = int(pair.a), int(pair.b)
        if i >= xy.shape[1] or j >= xy.shape[1]:
            continue
        a = xy[:, i]
        b = xy[:, j]
        if pair.reverse_b:
            b = torch.flip(b, dims=[1])

        point_conf = _confidence_tensor(
            pair.point_confidence,
            points=a.shape[1],
            device=a.device,
            dtype=a.dtype,
        )
        if float(point_conf.mean().detach().cpu()) < config.min_pair_confidence:
            continue
        pair_presence = query_weight[:, i] * query_weight[:, j]

        widths = torch.linalg.vector_norm(a - b, dim=-1)
        width_target = torch.full_like(widths, float(pair.target_width_m))
        width_error = F.smooth_l1_loss(
            widths,
            width_target,
            reduction="none",
            beta=0.5,
        )
        width_loss = _weighted_batch_mean(
            width_error,
            point_conf,
            pair_presence,
        )

        if widths.shape[-1] >= 2:
            width_delta = widths[:, 1:] - widths[:, :-1]
            width_conf = 0.5 * (point_conf[1:] + point_conf[:-1])
            width_variation = _weighted_batch_mean(
                width_delta.square(),
                width_conf,
                pair_presence,
            )
        else:
            width_variation = widths.sum() * 0.0

        ua = a[:, 1:] - a[:, :-1]
        ub = b[:, 1:] - b[:, :-1]
        ua = ua / torch.linalg.vector_norm(
            ua,
            dim=-1,
            keepdim=True,
        ).clamp_min(0.20)
        ub = ub / torch.linalg.vector_norm(
            ub,
            dim=-1,
            keepdim=True,
        ).clamp_min(0.20)
        parallel_error = 1.0 - (ua * ub).sum(-1).clamp(-1.0, 1.0)
        seg_conf = 0.5 * (point_conf[1:] + point_conf[:-1])
        parallel_loss = _weighted_batch_mean(
            parallel_error,
            seg_conf,
            pair_presence,
        )

        corridor_terms.append(
            config.corridor_heading_weight * parallel_loss
            + config.corridor_width_weight * width_loss
            + config.corridor_width_variation_weight * width_variation
        )

    corridor = (
        torch.stack(corridor_terms).mean()
        if corridor_terms
        else xy.sum() * 0.0
    )

    connector_terms: List[torch.Tensor] = []
    for connector in reference.connectors:
        i, j = int(connector.source), int(connector.target)
        if i >= xy.shape[1] or j >= xy.shape[1]:
            continue
        source = xy[:, i]
        target = xy[:, j]
        presence = query_weight[:, i] * query_weight[:, j]

        gap = torch.linalg.vector_norm(source[:, -1] - target[:, 0], dim=-1)
        # Do not snap endpoints together.  Only penalize gaps beyond the
        # reference-compatible tolerance, preserving diffusion freedom.
        tolerance = max(0.75, float(connector.target_gap_m) + 0.35)
        gap_error = torch.relu(gap - tolerance).square()

        source_tangent = source[:, -1] - source[:, -2]
        target_tangent = target[:, 1] - target[:, 0]
        source_tangent = source_tangent / torch.linalg.vector_norm(
            source_tangent,
            dim=-1,
            keepdim=True,
        ).clamp_min(0.20)
        target_tangent = target_tangent / torch.linalg.vector_norm(
            target_tangent,
            dim=-1,
            keepdim=True,
        ).clamp_min(0.20)
        heading_error = 1.0 - (
            source_tangent * target_tangent
        ).sum(-1).clamp(-1.0, 1.0)

        weight = float(connector.confidence)
        denominator = presence.sum().clamp_min(1.0)
        connector_terms.append(
            weight
            * (
                config.connectivity_weight
                * (gap_error * presence).sum()
                / denominator
                + config.connector_heading_weight
                * (heading_error * presence).sum()
                / denominator
            )
        )

    connectivity = (
        torch.stack(connector_terms).mean()
        if connector_terms
        else xy.sum() * 0.0
    )

    total = intrinsic + corridor + connectivity
    return total, {
        "intrinsic": intrinsic,
        "corridor": corridor,
        "connectivity": connectivity,
    }


def apply_latent_lane_guidance(
    decoder: Any,
    latents: torch.Tensor,
    *,
    reference: LaneConstraintReference,
    config: LaneGuidanceConfig,
    progress: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Decode, differentiate road loss to latent space, and apply one update."""

    strength = config.strength(progress)
    base_diagnostics: Dict[str, Any] = {
        "lane_guidance_strength": float(strength),
        "lane_guidance_progress": float(progress),
        "lane_guidance_corridor_count": len(reference.corridors),
        "lane_guidance_connector_count": len(reference.connectors),
        "lane_guidance_mean_pair_confidence": reference.mean_pair_confidence,
    }
    if strength <= 0.0:
        return latents, {
            **base_diagnostics,
            "lane_guidance_loss": 0.0,
            "lane_guidance_intrinsic_loss": 0.0,
            "lane_guidance_corridor_loss": 0.0,
            "lane_guidance_connectivity_loss": 0.0,
            "lane_guidance_grad_rms": 0.0,
            "lane_guidance_grad_abs_max": 0.0,
            "lane_guidance_applied": False,
        }

    with torch.enable_grad():
        x = latents.detach().requires_grad_(True)
        decoded = decoder.decode(x)
        loss, parts = lane_constraint_loss(
            decoded.lines.states,
            decoded.lines.mask,
            reference=reference,
            config=config,
        )
        grad = torch.autograd.grad(
            loss,
            x,
            retain_graph=False,
            create_graph=False,
        )[0]
        grad_rms = grad.square().mean().sqrt().clamp_min(1e-6)
        grad_abs_max = grad.abs().max()
        normalized = (grad / grad_rms).clamp(
            -float(config.grad_clip),
            float(config.grad_clip),
        )
        corrected = (x - float(strength) * normalized).detach()

    return corrected, {
        **base_diagnostics,
        "lane_guidance_loss": float(loss.detach().cpu()),
        "lane_guidance_intrinsic_loss": float(parts["intrinsic"].detach().cpu()),
        "lane_guidance_corridor_loss": float(parts["corridor"].detach().cpu()),
        "lane_guidance_connectivity_loss": float(parts["connectivity"].detach().cpu()),
        "lane_guidance_grad_rms": float(grad_rms.detach().cpu()),
        "lane_guidance_grad_abs_max": float(grad_abs_max.detach().cpu()),
        "lane_guidance_applied": True,
    }


class LatentLaneConstraintGuidance:
    """Callable adapter accepted by ``LDMPipeline.guidance_callback``."""

    def __init__(
        self,
        *,
        reference: LaneConstraintReference,
        config: Optional[LaneGuidanceConfig] = None,
    ) -> None:
        self.reference = reference
        self.config = config or LaneGuidanceConfig()

    def __call__(
        self,
        *,
        latents: torch.Tensor,
        decoder: Any,
        progress: float,
        step_index: Optional[int] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        corrected, diagnostics = apply_latent_lane_guidance(
            decoder,
            latents,
            reference=self.reference,
            config=self.config,
            progress=progress,
        )
        diagnostics["lane_guidance_step_index"] = (
            None if step_index is None else int(step_index)
        )
        if timestep is not None:
            try:
                diagnostics["lane_guidance_timestep"] = int(timestep.item())
            except Exception:
                diagnostics["lane_guidance_timestep"] = None
        return corrected, diagnostics


def _query_probability(
    logits: torch.Tensor,
    expected_shape: Sequence[int],
) -> torch.Tensor:
    """Return query confidence shaped [batch, queries]."""

    batch, queries = int(expected_shape[0]), int(expected_shape[1])
    value = logits
    while value.ndim > 2 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.shape[0] != batch or value.shape[1] != queries:
        value = value.reshape(batch, queries, -1).mean(-1)
    probability = torch.sigmoid(value)
    return probability


def _weighted_batch_mean(
    values: torch.Tensor,
    sample_weight: torch.Tensor,
    batch_weight: torch.Tensor,
) -> torch.Tensor:
    sample_weight = sample_weight.to(device=values.device, dtype=values.dtype)
    while sample_weight.ndim < values.ndim:
        sample_weight = sample_weight.unsqueeze(0)
    batch_weight = batch_weight.to(device=values.device, dtype=values.dtype)
    while batch_weight.ndim < values.ndim:
        batch_weight = batch_weight.unsqueeze(-1)
    weight = sample_weight * batch_weight
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def _confidence_tensor(
    confidence: Sequence[float],
    *,
    points: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    array = np.asarray(confidence, dtype=np.float32)
    if len(array) != points:
        if len(array) == 0:
            array = np.zeros(points, dtype=np.float32)
        elif len(array) == 1:
            array = np.full(points, float(array[0]), dtype=np.float32)
        else:
            old = np.linspace(0.0, 1.0, len(array))
            new = np.linspace(0.0, 1.0, points)
            array = np.interp(new, old, array).astype(np.float32)
    return torch.as_tensor(array, device=device, dtype=dtype).clamp(0.0, 1.0)


def _mask_value(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if bool(value) else 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _unit_segments(points: np.ndarray) -> np.ndarray:
    segment = np.diff(np.asarray(points, dtype=np.float64), axis=0)
    norm = np.linalg.norm(segment, axis=1, keepdims=True)
    valid = norm[:, 0] > 1e-6
    out = np.zeros_like(segment)
    out[valid] = segment[valid] / norm[valid]
    return out


def _segment_values_to_points(values: np.ndarray, point_count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if point_count <= 0:
        return np.empty(0, dtype=np.float64)
    if len(values) == 0:
        return np.zeros(point_count, dtype=np.float64)
    out = np.empty(point_count, dtype=np.float64)
    out[0] = values[0]
    out[-1] = values[-1]
    if point_count > 2:
        out[1:-1] = 0.5 * (values[:-1] + values[1:])
    return out


def _endpoint_heading(points: np.ndarray, *, at_start: bool) -> float:
    points = np.asarray(points, dtype=np.float64)
    if at_start:
        delta = points[1] - points[0]
    else:
        delta = points[-1] - points[-2]
    return float(math.atan2(delta[1], delta[0]))


def _wrap_angle(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


__all__ = [
    "LaneConstraintReference",
    "LaneConnectorReference",
    "LaneCorridorReference",
    "LaneGuidanceConfig",
    "LatentLaneConstraintGuidance",
    "apply_latent_lane_guidance",
    "infer_lane_constraint_reference",
    "lane_constraint_loss",
]
