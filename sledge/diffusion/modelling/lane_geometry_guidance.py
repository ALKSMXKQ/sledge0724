"""Differentiable local-geometry guidance for decoded SLEDGE lane polylines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LaneGeometryGuidanceConfig:
    """Inference-only lane guidance settings.

    ``scale`` is the maximum L2 norm of the normalized latent update before
    ``max_guidance_update_norm`` is applied.  Guidance starts during the latter
    part of the reverse process and follows a smoothstep ramp.
    """

    enabled: bool = False
    scale: float = 0.05
    start_fraction: float = 0.50
    heading_jump_threshold: float = math.radians(15.0)
    jump_weight: float = 1.0
    instability_weight: float = 0.5
    instability_beta: float = math.radians(3.0)
    mask_threshold: float = 0.30
    min_segment_length: float = 0.05
    max_guidance_update_norm: float = 0.10
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.scale < 0.0:
            raise ValueError("lane geometry guidance scale must be non-negative")
        if not 0.0 <= self.start_fraction < 1.0:
            raise ValueError("lane geometry guidance start_fraction must be in [0, 1)")
        if not 0.0 <= self.mask_threshold < 1.0:
            raise ValueError("lane geometry guidance mask_threshold must be in [0, 1)")
        if self.heading_jump_threshold < 0.0:
            raise ValueError("heading_jump_threshold must be non-negative")
        if self.instability_beta <= 0.0:
            raise ValueError("instability_beta must be positive")
        if self.max_guidance_update_norm < 0.0:
            raise ValueError("max_guidance_update_norm must be non-negative")


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Differentiably wrap angles to [-pi, pi]."""

    return torch.atan2(torch.sin(angle), torch.cos(angle))


def guidance_schedule(step_index: int, num_steps: int, start_fraction: float) -> float:
    """Smoothly ramp guidance from zero to one in the late denoising steps."""

    if num_steps <= 1:
        progress = 1.0
    else:
        progress = float(step_index) / float(num_steps - 1)
    ramp = max(0.0, min(1.0, (progress - start_fraction) / (1.0 - start_fraction)))
    return ramp * ramp * (3.0 - 2.0 * ramp)


def _weighted_sequence_mean(
    values: torch.Tensor, weights: torch.Tensor, eps: float
) -> torch.Tensor:
    return (values * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(eps)


def lane_geometry_loss(
    line_states: torch.Tensor,
    line_mask_logits: torch.Tensor,
    config: LaneGeometryGuidanceConfig,
) -> Dict[str, torch.Tensor]:
    """Compute per-sample jump and heading-instability losses.

    Args:
        line_states: decoded lane points with shape ``[B, Q, P, 2]``.
        line_mask_logits: decoded validity logits with shape ``[B, Q]``.
        config: loss and validity settings.

    Returns:
        Per-sample tensors named ``jump_loss``, ``instability_loss``,
        ``total_loss`` and ``effective_line_count``.
    """

    if line_states.ndim != 4 or line_states.shape[-1] != 2:
        raise ValueError(
            f"line_states must have shape [B,Q,P,2], got {tuple(line_states.shape)}"
        )
    if line_mask_logits.shape != line_states.shape[:2]:
        raise ValueError(
            "line_mask_logits must match line_states [B,Q], got "
            f"{tuple(line_mask_logits.shape)} and {tuple(line_states.shape)}"
        )

    # Geometry is evaluated in float32 for stable atan2 gradients when the
    # denoising pipeline itself runs in reduced precision.
    points = line_states.float()
    segments = points[..., 1:, :] - points[..., :-1, :]
    segment_lengths = torch.linalg.vector_norm(segments, dim=-1)
    headings = torch.atan2(segments[..., 1], segments[..., 0] + config.eps)

    # A smooth segment-validity weight suppresses gradients from collapsed
    # point pairs without introducing a hard, graph-breaking point mask.
    length_weights = segment_lengths / (segment_lengths + config.min_segment_length)
    turn = wrap_angle(headings[..., 1:] - headings[..., :-1])
    turn_weights = length_weights[..., 1:] * length_weights[..., :-1]

    jump_values = F.relu(turn.abs() - config.heading_jump_threshold).square()
    jump_per_line = _weighted_sequence_mean(jump_values, turn_weights, config.eps)

    if turn.shape[-1] >= 2:
        heading_instability = wrap_angle(turn[..., 1:] - turn[..., :-1])
        instability_weights = turn_weights[..., 1:] * turn_weights[..., :-1]
        instability_values = F.smooth_l1_loss(
            heading_instability,
            torch.zeros_like(heading_instability),
            reduction="none",
            beta=config.instability_beta,
        )
        instability_per_line = _weighted_sequence_mean(
            instability_values, instability_weights, config.eps
        )
    else:
        instability_per_line = jump_per_line * 0.0

    # Decoder masks are logits. ReLU gives an exact zero below the configured
    # confidence threshold while retaining a differentiable soft weight above it.
    # Stop-gradient prevents guidance from taking the shortcut of hiding a bad
    # line by lowering its confidence instead of repairing its XY geometry.
    line_probabilities = line_mask_logits.float().sigmoid().detach()
    line_weights = F.relu(line_probabilities - config.mask_threshold) / (
        1.0 - config.mask_threshold
    )
    line_denominator = line_weights.sum(dim=-1).clamp_min(config.eps)
    jump_loss = (jump_per_line * line_weights).sum(dim=-1) / line_denominator
    instability_loss = (instability_per_line * line_weights).sum(
        dim=-1
    ) / line_denominator
    total_loss = (
        config.jump_weight * jump_loss + config.instability_weight * instability_loss
    )

    return {
        "jump_loss": jump_loss,
        "instability_loss": instability_loss,
        "total_loss": total_loss,
        "effective_line_count": line_weights.sum(dim=-1),
    }
