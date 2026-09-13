from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from sledge.hazard_equivalence.ground_truth.conflict import ConflictConfig, compute_spatiotemporal_conflict
from sledge.hazard_equivalence.ground_truth.reveal import RevealConfig, detect_reveal, has_occluded_run
from sledge.hazard_equivalence.ground_truth.timing import CriticalTimingConfig, compute_critical_timing
from sledge.hazard_equivalence.ground_truth.types import (
    FailureReason,
    HazardResult,
    TimedTrajectory2D,
)


@dataclass(frozen=True)
class OccludedPedestrianContract:
    reveal_config: RevealConfig = field(default_factory=RevealConfig)
    conflict_config: ConflictConfig = field(default_factory=ConflictConfig)
    timing_config: CriticalTimingConfig = field(default_factory=CriticalTimingConfig)

    def verify(
        self,
        visibility_fractions: Sequence[float],
        visibility_timestamps_s: Sequence[float],
        ego_trajectory: TimedTrajectory2D,
        pedestrian_trajectory: TimedTrajectory2D,
    ) -> HazardResult:
        reveal = detect_reveal(visibility_fractions, visibility_timestamps_s, self.reveal_config)
        O = has_occluded_run(
            visibility_fractions,
            self.reveal_config.visibility_threshold,
            self.reveal_config.minimum_pre_occluded_frames,
        )
        E = reveal.reveal_exists
        conflict = compute_spatiotemporal_conflict(ego_trajectory, pedestrian_trajectory, self.conflict_config)
        C = conflict.conflict_exists

        timing = None
        T = False
        if E and C and reveal.reveal_time_s is not None and conflict.conflict_time_s is not None:
            if conflict.conflict_time_s > reveal.reveal_time_s:
                timing = compute_critical_timing(
                    reveal.reveal_time_s, conflict.conflict_time_s, self.timing_config
                )
                T = timing.is_critical

        H = bool(O and E and C and T)
        if not O:
            reason = FailureReason.NO_PRE_REVEAL_OCCLUSION
        elif not E:
            reason = FailureReason.NO_STABLE_REVEAL
        elif not C:
            reason = FailureReason.NO_SPATIOTEMPORAL_CONFLICT
        elif conflict.conflict_time_s is None or reveal.reveal_time_s is None or conflict.conflict_time_s <= reveal.reveal_time_s:
            reason = FailureReason.CONFLICT_NOT_AFTER_REVEAL
        elif not T:
            reason = FailureReason.NON_CRITICAL_TIMING
        else:
            reason = FailureReason.NONE

        return HazardResult(
            O=O,
            E=E,
            C=C,
            T=T,
            H=H,
            failure_reason=reason,
            reveal=reveal,
            conflict=conflict,
            timing=timing,
            diagnostics={
                "visibility_threshold": self.reveal_config.visibility_threshold,
                "max_pet_s": self.conflict_config.max_pet_s,
                "aggressive_max_s": self.timing_config.aggressive_max_s,
                "moderate_max_s": self.timing_config.moderate_max_s,
            },
        )
