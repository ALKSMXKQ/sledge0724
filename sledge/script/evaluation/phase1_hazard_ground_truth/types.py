from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class OcclusionType(str, Enum):
    VISIBLE = "visible"
    PARTIALLY_OCCLUDED = "partially_occluded"
    FULLY_OCCLUDED = "fully_occluded"


class TimingClass(str, Enum):
    SAFE = "safe"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class FailureReason(str, Enum):
    NONE = "none"
    NO_PRE_REVEAL_OCCLUSION = "no_pre_reveal_occlusion"
    NO_STABLE_REVEAL = "no_stable_reveal"
    NO_SPATIOTEMPORAL_CONFLICT = "no_spatiotemporal_conflict"
    CONFLICT_NOT_AFTER_REVEAL = "conflict_not_after_reveal"
    NON_CRITICAL_TIMING = "non_critical_timing"


@dataclass(frozen=True)
class VisibilityResult:
    visibility_fraction: float
    occluding_object: Optional[str]
    occlusion_type: OcclusionType
    blocked_samples: int
    total_samples: int


@dataclass(frozen=True)
class RevealResult:
    reveal_exists: bool
    reveal_index: Optional[int]
    reveal_time_s: Optional[float]
    visibility_threshold: float
    consecutive_visible_frames: int


@dataclass(frozen=True)
class TimedTrajectory2D:
    timestamps_s: np.ndarray
    positions_xy: np.ndarray

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_s, dtype=np.float64)
        positions = np.asarray(self.positions_xy, dtype=np.float64)
        if timestamps.ndim != 1:
            raise ValueError(f"timestamps_s must be 1-D, got {timestamps.shape}")
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"positions_xy must have shape (N, 2), got {positions.shape}")
        if len(timestamps) != len(positions):
            raise ValueError("timestamps_s and positions_xy must have equal length")
        if len(timestamps) < 2:
            raise ValueError("trajectory needs at least two samples")
        if not np.all(np.isfinite(timestamps)) or not np.all(np.isfinite(positions)):
            raise ValueError("trajectory values must be finite")
        if np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("timestamps_s must be strictly increasing")
        object.__setattr__(self, "timestamps_s", timestamps)
        object.__setattr__(self, "positions_xy", positions)


@dataclass(frozen=True)
class ConflictRegion:
    center_xy: tuple[float, float]
    radius_m: float


@dataclass(frozen=True)
class ConflictResult:
    conflict_exists: bool
    conflict_region: Optional[ConflictRegion]
    ego_arrival_time: Optional[float]
    ped_arrival_time: Optional[float]
    arrival_time_gap: Optional[float]
    PET: Optional[float]
    minimum_distance: float

    @property
    def conflict_time_s(self) -> Optional[float]:
        return self.ego_arrival_time


@dataclass(frozen=True)
class TimingResult:
    rttc_s: float
    timing_class: TimingClass
    is_critical: bool


@dataclass(frozen=True)
class HazardResult:
    O: bool
    E: bool
    C: bool
    T: bool
    H: bool
    failure_reason: FailureReason
    reveal: RevealResult
    conflict: ConflictResult
    timing: Optional[TimingResult]
    diagnostics: dict[str, Any] = field(default_factory=dict)
