from types import SimpleNamespace

import numpy as np
import torch

from sledge.diffusion.modelling.lane_geometry_guidance import (
    LaneGuidanceConfig,
    apply_latent_lane_guidance,
    lane_geometry_loss,
    polyline_quality,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.topology_filter import (
    classify_topology,
    scene_matches_topology,
)


def _scene(lines):
    states = np.asarray(lines, dtype=np.float32)
    mask = np.ones((len(states),), dtype=np.float32)
    return SimpleNamespace(lines=SimpleNamespace(states=states, mask=mask))


def test_topology_filter_is_read_only_and_recognizes_road_segment():
    x = np.linspace(-25.0, 25.0, 20)
    scene = _scene([
        np.stack([x, np.full_like(x, -1.75)], axis=1),
        np.stack([x, np.full_like(x, 1.75)], axis=1),
    ])
    before = scene.lines.states.copy()
    result = classify_topology(scene)
    assert result.family == "road_segment"
    assert scene_matches_topology(scene, "road_segment")
    np.testing.assert_array_equal(scene.lines.states, before)


def test_topology_filter_recognizes_intersection():
    t = np.linspace(-20.0, 20.0, 20)
    horizontal = np.stack([t, np.zeros_like(t)], axis=1)
    vertical = np.stack([np.zeros_like(t), t], axis=1)
    result = classify_topology(_scene([horizontal, vertical]))
    assert result.family == "intersection"
    assert result.local_crossings >= 1


def test_lane_quality_distinguishes_smooth_and_wiggly_lines():
    x = np.linspace(0.0, 30.0, 61)
    smooth = np.stack([x, 0.02 * x * x / 30.0], axis=1)
    wiggly = np.stack([x, 0.8 * np.sin(1.4 * x)], axis=1)
    q_smooth = polyline_quality(smooth, spacing_m=0.5)
    q_wiggly = polyline_quality(wiggly, spacing_m=0.5)
    assert q_smooth["heading_jump_max"] < q_wiggly["heading_jump_max"]
    assert q_smooth["curvature_change"] < q_wiggly["curvature_change"]


def test_guidance_schedule_off_then_weak_then_strong():
    cfg = LaneGuidanceConfig(enabled=True, start_ratio=0.5, strong_ratio=0.8)
    assert cfg.strength(0.49) == 0.0
    assert 0.0 < cfg.strength(0.6) < cfg.strength(0.9)
    assert cfg.strength(0.9, "intersection") < cfg.strength(0.9, "road_segment")


class _DummyDecoder:
    def decode(self, latent):
        # latent [B,1,1,P] -> one line [B,1,P,2], x fixed and y trainable
        b, _, _, p = latent.shape
        x = torch.linspace(0.0, 10.0, p, device=latent.device, dtype=latent.dtype)
        x = x.view(1, 1, p).expand(b, 1, p)
        y = latent[:, 0, 0, :].view(b, 1, p)
        states = torch.stack([x, y], dim=-1)
        mask = torch.full((b, 1), 5.0, device=latent.device, dtype=latent.dtype)
        return SimpleNamespace(lines=SimpleNamespace(states=states, mask=mask))


def test_latent_guidance_reduces_lane_geometry_loss():
    decoder = _DummyDecoder()
    latent = torch.tensor([[[[0.0, 0.8, -0.6, 0.9, -0.5, 0.7, 0.0]]]], dtype=torch.float32)
    cfg = LaneGuidanceConfig(enabled=True, start_ratio=0.0, strong_ratio=0.0, strong_strength=0.03)
    before_vec = decoder.decode(latent)
    before = lane_geometry_loss(before_vec.lines.states, before_vec.lines.mask, config=cfg)
    corrected, trace = apply_latent_lane_guidance(decoder, latent, config=cfg, progress=1.0)
    after_vec = decoder.decode(corrected)
    after = lane_geometry_loss(after_vec.lines.states, after_vec.lines.mask, config=cfg)
    assert trace["lane_guidance_strength"] > 0.0
    assert float(after) < float(before)
