"""Evaluate whether diffusion itself retains the complete dangerous semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping


@dataclass(frozen=True)
class DiffusionSemanticRetention:
    sample_id: str
    diffusion_mode: str
    pedestrian_retained: bool
    occluder_retained: bool
    occlusion_retained: bool
    unique_direction_retained: bool
    ego_path_intersection_retained: bool
    reveal_event_proxy_retained: bool
    interaction_timing_retained: bool
    full_hazard_semantics_retained: bool
    failure_reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_retention(
    metrics: Mapping[str, Any],
    *,
    sample_id: str,
    diffusion_mode: str,
) -> DiffusionSemanticRetention:
    """Convert stage metrics into explicit semantic-retention indicators."""

    checks = dict(metrics.get("checks", {}) or {})
    pedestrian = bool(checks.get("pedestrian_exists", False))
    occluder = bool(checks.get("occluder_exists", False))
    occlusion = bool(
        checks.get("occluder_between_ego_and_actor", False)
        and checks.get("line_of_sight_occlusion", False)
    )
    direction = bool(checks.get("direction_match", False))
    intersection = bool(checks.get("crossing_reaches_ego_lane", False))
    timing = bool(checks.get("interaction_timing_match", False))
    reveal_proxy = bool(pedestrian and occluder and occlusion and intersection)
    full = bool(
        pedestrian
        and occluder
        and occlusion
        and direction
        and intersection
        and timing
        and checks.get("no_actor_occluder_initial_overlap", False)
    )

    failure_reasons = []
    names = {
        "pedestrian_retained": pedestrian,
        "occluder_retained": occluder,
        "occlusion_retained": occlusion,
        "unique_direction_retained": direction,
        "ego_path_intersection_retained": intersection,
        "reveal_event_proxy_retained": reveal_proxy,
        "interaction_timing_retained": timing,
    }
    for name, passed in names.items():
        if not passed:
            failure_reasons.append(name)

    return DiffusionSemanticRetention(
        sample_id=sample_id,
        diffusion_mode=diffusion_mode,
        pedestrian_retained=pedestrian,
        occluder_retained=occluder,
        occlusion_retained=occlusion,
        unique_direction_retained=direction,
        ego_path_intersection_retained=intersection,
        reveal_event_proxy_retained=reveal_proxy,
        interaction_timing_retained=timing,
        full_hazard_semantics_retained=full,
        failure_reasons=failure_reasons,
    )


def aggregate_retention(
    rows: Iterable[Mapping[str, Any]],
    *,
    diffusion_mode: str,
) -> Dict[str, Any]:
    items = [dict(row) for row in rows]
    fields = [
        "pedestrian_retained",
        "occluder_retained",
        "occlusion_retained",
        "unique_direction_retained",
        "ego_path_intersection_retained",
        "reveal_event_proxy_retained",
        "interaction_timing_retained",
        "full_hazard_semantics_retained",
    ]
    denominator = len(items)
    return {
        "schema_version": "diffusion_semantic_retention_summary_v1",
        "diffusion_mode": diffusion_mode,
        "num_samples": denominator,
        "rates": {
            field: (
                sum(bool(row.get(field, False)) for row in items) / denominator
                if denominator
                else 0.0
            )
            for field in fields
        },
        "full_hazard_retained_count": sum(
            bool(row.get("full_hazard_semantics_retained", False)) for row in items
        ),
    }


def compare_modes(
    raw_summary: Mapping[str, Any],
    protected_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_rates = dict(raw_summary.get("rates", {}) or {})
    protected_rates = dict(protected_summary.get("rates", {}) or {})
    fields = sorted(set(raw_rates) | set(protected_rates))
    return {
        "schema_version": "diffusion_mode_comparison_v1",
        "raw_diffusion_baseline": dict(raw_summary),
        "semantic_protected": dict(protected_summary),
        "rate_delta_protected_minus_raw": {
            field: float(protected_rates.get(field, 0.0))
            - float(raw_rates.get(field, 0.0))
            for field in fields
        },
    }
