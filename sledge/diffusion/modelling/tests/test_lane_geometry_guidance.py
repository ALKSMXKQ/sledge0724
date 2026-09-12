import math

import torch

from sledge.diffusion.modelling.lane_geometry_guidance import (
    LaneGeometryGuidanceConfig,
    lane_geometry_loss,
)


def _loss(points: torch.Tensor, mask_logit: float = 10.0):
    states = points.reshape(1, 1, -1, 2)
    masks = torch.tensor([[mask_logit]], dtype=states.dtype)
    return lane_geometry_loss(states, masks, LaneGeometryGuidanceConfig())


def test_snaking_straight_lane_has_large_instability_penalty() -> None:
    x = torch.linspace(0.0, 20.0, 20)
    y = 0.50 * torch.sin(torch.linspace(0.0, 8.0 * math.pi, 20))
    snaking = _loss(torch.stack([x, y], dim=-1))
    straight = _loss(torch.stack([x, torch.zeros_like(x)], dim=-1))

    assert snaking["instability_loss"].item() > 0.10
    assert snaking["total_loss"].item() > straight["total_loss"].item() + 0.10


def test_smooth_sixty_degree_arc_has_low_instability_penalty() -> None:
    angle = torch.linspace(0.0, math.radians(60.0), 20)
    radius = 30.0
    arc = torch.stack(
        [radius * torch.sin(angle), radius * (1.0 - torch.cos(angle))], dim=-1
    )
    losses = _loss(arc)

    assert losses["jump_loss"].item() < 1e-6
    assert losses["instability_loss"].item() < 1e-5


def test_local_sharp_corner_raises_heading_jump_loss() -> None:
    x = torch.arange(20, dtype=torch.float32)
    y = torch.zeros_like(x)
    y[10:] = torch.arange(10, dtype=torch.float32) * 2.0
    corner = _loss(torch.stack([x, y], dim=-1))
    straight = _loss(torch.stack([x, torch.zeros_like(x)], dim=-1))

    assert corner["jump_loss"].item() > 1e-3
    assert corner["jump_loss"].item() > straight["jump_loss"].item()


def test_invalid_lane_mask_contributes_no_constraint() -> None:
    x = torch.linspace(0.0, 20.0, 20)
    y = 0.50 * torch.sin(torch.linspace(0.0, 8.0 * math.pi, 20))
    points = torch.stack([x, y], dim=-1).requires_grad_(True)
    losses = _loss(points, mask_logit=-10.0)

    assert losses["effective_line_count"].item() == 0.0
    assert losses["total_loss"].item() == 0.0
    gradient = torch.autograd.grad(losses["total_loss"].sum(), points)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))
