from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sledge.script.evaluation.phase1_hazard_ground_truth.types import RevealResult


@dataclass(frozen=True)
class RevealConfig:
    visibility_threshold: float = 0.5
    consecutive_visible_frames: int = 3
    minimum_pre_occluded_frames: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.visibility_threshold <= 1.0:
            raise ValueError("visibility_threshold must be in [0, 1]")
        if self.consecutive_visible_frames < 1:
            raise ValueError("consecutive_visible_frames must be >= 1")
        if self.minimum_pre_occluded_frames < 1:
            raise ValueError("minimum_pre_occluded_frames must be >= 1")


def detect_reveal(
    visibility_fractions: Sequence[float],
    timestamps_s: Sequence[float],
    config: RevealConfig | None = None,
) -> RevealResult:
    config = config or RevealConfig()
    visibility = np.asarray(visibility_fractions, dtype=np.float64)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if visibility.ndim != 1 or timestamps.ndim != 1 or len(visibility) != len(timestamps):
        raise ValueError("visibility_fractions and timestamps_s must be equal-length 1-D arrays")
    if len(visibility) == 0:
        raise ValueError("visibility sequence must be non-empty")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps_s must be strictly increasing")
    if np.any(~np.isfinite(visibility)) or np.any((visibility < 0.0) | (visibility > 1.0)):
        raise ValueError("visibility fractions must be finite and in [0, 1]")

    tau = config.visibility_threshold
    k = config.consecutive_visible_frames
    pre = config.minimum_pre_occluded_frames
    for i in range(pre, len(visibility) - k + 1):
        before = visibility[i - pre : i]
        after = visibility[i : i + k]
        if np.all(before < tau) and np.all(after > tau):
            return RevealResult(True, i, float(timestamps[i]), tau, k)

    return RevealResult(False, None, None, tau, k)


def has_occluded_run(
    visibility_fractions: Sequence[float],
    threshold: float,
    minimum_frames: int = 1,
) -> bool:
    visibility = np.asarray(visibility_fractions, dtype=np.float64)
    if minimum_frames < 1:
        raise ValueError("minimum_frames must be >= 1")
    run = 0
    for value in visibility:
        run = run + 1 if value < threshold else 0
        if run >= minimum_frames:
            return True
    return False
