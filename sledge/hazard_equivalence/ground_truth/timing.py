from __future__ import annotations

from dataclasses import dataclass

from sledge.hazard_equivalence.ground_truth.types import TimingClass, TimingResult


@dataclass(frozen=True)
class CriticalTimingConfig:
    aggressive_max_s: float = 1.0
    moderate_max_s: float = 3.0

    def __post_init__(self) -> None:
        if self.aggressive_max_s <= 0.0:
            raise ValueError("aggressive_max_s must be > 0")
        if self.moderate_max_s <= self.aggressive_max_s:
            raise ValueError("moderate_max_s must be > aggressive_max_s")


def compute_critical_timing(
    reveal_time_s: float,
    conflict_time_s: float,
    config: CriticalTimingConfig | None = None,
) -> TimingResult:
    config = config or CriticalTimingConfig()
    rttc = float(conflict_time_s - reveal_time_s)
    if rttc <= 0.0:
        raise ValueError("conflict_time_s must be strictly after reveal_time_s")
    if rttc <= config.aggressive_max_s:
        timing_class = TimingClass.AGGRESSIVE
    elif rttc <= config.moderate_max_s:
        timing_class = TimingClass.MODERATE
    else:
        timing_class = TimingClass.SAFE
    return TimingResult(rttc, timing_class, timing_class != TimingClass.SAFE)
