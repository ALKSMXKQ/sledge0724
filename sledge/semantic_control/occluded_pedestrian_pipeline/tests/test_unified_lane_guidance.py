from __future__ import annotations

import numpy as np
import torch

from sledge.diffusion.modelling.unified_lane_guidance import (
    LaneGuidanceConfig,
    infer_lane_constraint_reference,
    lane_constraint_loss,
)


def _line(x0: float, x1: float, y: float, samples: int = 41) -> np.ndarray:
    x = np.linspace(x0, x1, samples, dtype=np.float64)
    return np.column_stack([x, np.full_like(x, y)])


def test_straight_corridor_has_high_local_confidence() -> None:
    states = np.stack(
        [
            _line(-10.0, 30.0, -1.75),
            _line(-10.0, 30.0, 1.75),
        ]
    )
    reference = infer_lane_constraint_reference(
        states,
        np.ones(2, dtype=np.float32),
    )

    assert len(reference.corridors) == 1
    pair = reference.corridors[0]
    assert abs(pair.target_width_m - 3.5) < 0.05
    assert pair.mean_confidence > 0.90
    assert min(pair.point_confidence) > 0.85


def test_junction_like_divergence_is_downweighted_locally_not_globally_disabled() -> None:
    a = _line(-10.0, 30.0, -1.75)
    b = _line(-10.0, 30.0, 1.75)
    # Mimic a junction interior where the opposite boundary diverges locally.
    bump = np.exp(-0.5 * ((np.arange(len(b)) - 20.0) / 3.0) ** 2)
    b[:, 1] += 4.5 * bump
    reference = infer_lane_constraint_reference(
        np.stack([a, b]),
        np.ones(2, dtype=np.float32),
    )

    assert len(reference.corridors) == 1
    confidence = np.asarray(reference.corridors[0].point_confidence)
    assert confidence[2] > 0.75
    assert confidence[-3] > 0.75
    assert confidence[len(confidence) // 2] < 0.20
    # This is the key behavior change from the old scene-wide intersection
    # switch: useful approach/exit constraints remain active.
    assert reference.corridors[0].mean_confidence > 0.40


def test_directed_endpoint_connector_is_inferred() -> None:
    states = np.stack(
        [
            _line(0.0, 10.0, 0.0),
            _line(10.3, 25.0, 0.0),
        ]
    )
    reference = infer_lane_constraint_reference(
        states,
        np.ones(2, dtype=np.float32),
    )

    connectors = {
        (item.source, item.target): item
        for item in reference.connectors
    }
    assert (0, 1) in connectors
    assert connectors[(0, 1)].confidence > 0.70


def test_lane_constraint_loss_is_differentiable() -> None:
    source = np.stack(
        [
            _line(-10.0, 30.0, -1.75),
            _line(-10.0, 30.0, 1.75),
        ]
    )
    reference = infer_lane_constraint_reference(
        source,
        np.ones(2, dtype=np.float32),
    )

    generated = source.copy()
    generated[1, :, 1] += 0.8
    line_states = torch.tensor(
        generated[None, ...],
        dtype=torch.float32,
        requires_grad=True,
    )
    line_mask_logits = torch.full((1, 2), 5.0, dtype=torch.float32)
    loss, parts = lane_constraint_loss(
        line_states,
        line_mask_logits,
        reference=reference,
        config=LaneGuidanceConfig(),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert float(parts["corridor"].detach()) > 0.0
    assert line_states.grad is not None
    assert float(line_states.grad.abs().sum()) > 0.0


def test_strength_schedule_is_not_scene_topology_dependent() -> None:
    config = LaneGuidanceConfig(
        start_ratio=0.5,
        strong_ratio=0.8,
        weak_strength=0.01,
        strong_strength=0.03,
    )
    assert config.strength(0.49) == 0.0
    assert 0.01 < config.strength(0.65) < 0.03
    assert config.strength(0.9) == 0.03
