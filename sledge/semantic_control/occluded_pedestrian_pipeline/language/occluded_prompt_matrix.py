"""Deterministic natural-language test matrix for occluded-pedestrian prompts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class OccludedPromptCase:
    """One prompt plus the semantic fields that must survive parsing."""

    case_id: str
    prompt: str
    expected: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_prompt_cases() -> List[OccludedPromptCase]:
    """Return a small diagnostic matrix before large-scale scene generation."""

    rows = [
        (
            "occ_ped_001",
            "A child suddenly emerges from behind a parked truck into the ego lane.",
            {
                "language_actor_detail": "child",
                "auxiliary_entity": "parked_truck_occluder",
                "source_region": "curbside",
                "target_region": "ego_lane",
                "risk_level": "aggressive",
                "occluder_side_mode": "sample_once",
            },
        ),
        (
            "occ_ped_002",
            "An adult pedestrian steps out from behind a parked car on the left roadside and enters the ego lane at 1.2 m/s.",
            {
                "language_actor_detail": "adult_or_unspecified",
                "auxiliary_entity": "parked_car_occluder",
                "source_region": "left_side",
                "target_region": "ego_lane",
                "actor_speed_mps": 1.2,
                "occluder_side": "left",
            },
        ),
        (
            "occ_ped_003",
            "A jogger hidden behind a bus on the right side rushes toward the ego path.",
            {
                "language_actor_detail": "jogger",
                "auxiliary_entity": "bus_occluder",
                "source_region": "right_side",
                "target_region": "ego_path",
                "occluder_side": "right",
            },
        ),
        (
            "occ_ped_004",
            "A person becomes visible from behind a roadside van and crosses into the ego lane at 1.6 m/s.",
            {
                "language_actor_detail": "adult_or_unspecified",
                "auxiliary_entity": "van_occluder",
                "source_region": "curbside",
                "target_region": "ego_lane",
                "actor_speed_mps": 1.6,
                "occluder_side_mode": "sample_once",
            },
        ),
        (
            "occ_ped_005",
            "A wheelchair user emerges from behind a road barrier on the left and moves into the ego path.",
            {
                "language_actor_detail": "wheelchair_user",
                "auxiliary_entity": "barrier_occluder",
                "source_region": "left_side",
                "target_region": "ego_path",
                "occluder_side": "left",
            },
        ),
        (
            "occ_ped_006",
            "A pedestrian is fully hidden by a parked truck at the curb, then abruptly enters the ego lane.",
            {
                "language_actor_detail": "adult_or_unspecified",
                "auxiliary_entity": "parked_truck_occluder",
                "source_region": "curbside",
                "target_region": "ego_lane",
                "visibility": "fully_occluded",
                "risk_level": "aggressive",
                "occluder_side_mode": "sample_once",
            },
        ),
        (
            "occ_ped_007",
            "A slowly moving pedestrian appears from behind a parked car on the right roadside and enters the ego lane at 0.8 m/s.",
            {
                "auxiliary_entity": "parked_car_occluder",
                "source_region": "right_side",
                "target_region": "ego_lane",
                "actor_speed_mps": 0.8,
                "risk_level": "mild",
                "occluder_side": "right",
            },
        ),
        (
            "occ_ped_008",
            "A pedestrian suddenly comes out from behind a bus on the left side and heads for the ego path at 1.9 m/s.",
            {
                "auxiliary_entity": "bus_occluder",
                "source_region": "left_side",
                "target_region": "ego_path",
                "actor_speed_mps": 1.9,
                "risk_level": "aggressive",
                "occluder_side": "left",
            },
        ),
        (
            "occ_ped_009",
            "A schoolboy is obscured by a parked van and then steps into the ego lane.",
            {
                "language_actor_detail": "child",
                "auxiliary_entity": "van_occluder",
                "source_region": "curbside",
                "target_region": "ego_lane",
                "occluder_side_mode": "sample_once",
            },
        ),
        (
            "occ_ped_010",
            "A runner hidden behind a roadside barrier becomes visible and moves across the ego path.",
            {
                "language_actor_detail": "jogger",
                "auxiliary_entity": "barrier_occluder",
                "source_region": "curbside",
                "target_region": "ego_path",
                "occluder_side_mode": "sample_once",
            },
        ),
        (
            "occ_ped_011",
            "A pedestrian partially concealed behind a parked truck on the right begins entering the ego lane.",
            {
                "auxiliary_entity": "parked_truck_occluder",
                "source_region": "right_side",
                "target_region": "ego_lane",
                "visibility": "partially_occluded",
                "occluder_side": "right",
            },
        ),
        (
            "occ_ped_012",
            "A girl emerges from behind a parked car at the curb and moves toward the ego path; the side is not specified.",
            {
                "language_actor_detail": "child",
                "auxiliary_entity": "parked_car_occluder",
                "source_region": "curbside",
                "target_region": "ego_path",
                "occluder_side_mode": "sample_once",
            },
        ),
    ]

    common = {
        "primary_actor_type": "pedestrian",
        "tracked_object_type": "TrackedObjectType.PEDESTRIAN",
        "sledge_collection": "pedestrians",
        "hazard_interaction": "occluded_emergence",
        "motion_direction": "occluder_to_ego_path",
    }
    return [
        OccludedPromptCase(case_id=case_id, prompt=prompt, expected={**common, **expected})
        for case_id, prompt, expected in rows
    ]


def write_prompt_cases(path: Path, cases: Optional[Iterable[OccludedPromptCase]] = None) -> Path:
    """Write one JSON object per line."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(cases or default_prompt_cases())
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    return output


def read_prompt_cases(path: Path) -> List[OccludedPromptCase]:
    """Read JSONL generated by :func:`write_prompt_cases`."""

    rows: List[OccludedPromptCase] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(
                OccludedPromptCase(
                    case_id=str(payload["case_id"]),
                    prompt=str(payload["prompt"]),
                    expected=dict(payload.get("expected", {})),
                )
            )
    return rows
