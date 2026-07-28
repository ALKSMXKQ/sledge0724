"""Batch construction of position-aware occluded-pedestrian B1 scenes."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sledge.semantic_control.io import save_json
from sledge.semantic_control.occluded_pedestrian_pipeline.language.eventframe_adapter import (
    ControlOverrides,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.pipeline import (
    OccludedPedestrianPipeline,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.position_control import (
    install_position_aware_layout_patch,
    normalize_occluder_position,
    position_matches,
    requested_occluder_position,
)


@dataclass(frozen=True)
class SemanticRetentionCase:
    sample_id: str
    condition_id: str
    input_raw: str
    source_relative_path: str
    source_scenario_type: str
    prompt: str
    occluder_type: str
    occluder_position: str
    direction: str
    pedestrian_speed_mps: float
    risk_level: str
    replicate: int

    @property
    def overrides(self) -> ControlOverrides:
        return ControlOverrides(
            occluder_type=self.occluder_type,
            direction=self.direction,
            pedestrian_speed_mps=self.pedestrian_speed_mps,
            risk_level=self.risk_level,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_semantic_retention_matrix(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def discover_raw_scenes(input_root: Path, glob_pattern: str = "**/sledge_raw.gz") -> List[Path]:
    scenes = sorted(Path(input_root).glob(glob_pattern))
    if not scenes:
        raise FileNotFoundError(f"No scenes matched {glob_pattern!r} under {input_root}")
    return scenes


def build_semantic_retention_cases(
    *,
    input_root: Path,
    profile: str,
    matrix_config: Dict[str, Any],
    glob_pattern: str = "**/sledge_raw.gz",
    max_cases: Optional[int] = None,
    total_samples_override: Optional[int] = None,
) -> List[SemanticRetentionCase]:
    """Build a deterministic, approximately balanced experiment matrix."""

    input_root = Path(input_root).resolve()
    scenes = discover_raw_scenes(input_root, glob_pattern)
    profiles = matrix_config.get("profiles", {})
    if profile not in profiles:
        raise KeyError(f"Unknown profile={profile!r}; available={sorted(profiles)}")
    cfg = dict(profiles[profile])

    occluders = [str(value) for value in cfg["occluder_types"]]
    positions = [normalize_occluder_position(value) for value in cfg["occluder_positions"]]
    directions = [str(value) for value in cfg["directions"]]
    risks = [str(value) for value in cfg["risk_levels"]]
    speeds = [float(value) for value in cfg["pedestrian_speeds_mps"]]
    conditions = list(itertools.product(occluders, positions, directions, risks, speeds))
    if not conditions:
        raise ValueError("Semantic-retention matrix has no conditions")

    total_samples = (
        int(total_samples_override)
        if total_samples_override is not None
        else int(cfg.get("total_samples", len(conditions)))
    )
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    if max_cases is not None:
        total_samples = min(total_samples, max(0, int(max_cases)))

    templates = list(matrix_config.get("prompt_templates", []))
    if not templates:
        raise ValueError("Semantic-retention matrix needs at least one prompt template")

    scenes_by_type: Dict[str, List[Path]] = defaultdict(list)
    for scene in scenes:
        rel = scene.relative_to(input_root)
        scenes_by_type[_source_scenario_type(rel)].append(scene)
    source_types = sorted(scenes_by_type)

    condition_usage: Counter = Counter()
    axis_usage = {
        "occluder": Counter(),
        "position": Counter(),
        "direction": Counter(),
        "risk": Counter(),
        "speed": Counter(),
    }
    source_usage: Counter = Counter()
    cases: List[SemanticRetentionCase] = []

    for global_index in range(total_samples):
        condition_index, condition = min(
            enumerate(conditions),
            key=lambda item: (
                axis_usage["occluder"][item[1][0]],
                axis_usage["position"][item[1][1]],
                axis_usage["direction"][item[1][2]],
                axis_usage["risk"][item[1][3]],
                axis_usage["speed"][item[1][4]],
                condition_usage[item[1]],
                item[0],
            ),
        )
        del condition_index
        occluder, position, direction, risk, speed = condition
        replicate = int(condition_usage[condition])
        condition_usage[condition] += 1
        axis_usage["occluder"][occluder] += 1
        axis_usage["position"][position] += 1
        axis_usage["direction"][direction] += 1
        axis_usage["risk"][risk] += 1
        axis_usage["speed"][speed] += 1

        source_type = min(source_types, key=lambda value: (source_usage[value], value))
        source_usage[source_type] += 1
        typed_scenes = scenes_by_type[source_type]
        scene_offset = _stable_index(
            f"{occluder}|{position}|{direction}|{risk}|{speed:.2f}|{source_type}",
            len(typed_scenes),
        )
        scene = typed_scenes[(scene_offset + replicate) % len(typed_scenes)]
        rel = scene.relative_to(input_root)

        condition_id = (
            f"occ-{occluder}__pos-{position}__dir-{direction}__"
            f"risk-{risk}__speed-{speed:.1f}"
        )
        token = hashlib.sha1(
            f"{rel}|{condition_id}|{replicate}".encode("utf-8")
        ).hexdigest()[:10]
        sample_id = f"{condition_id}__r{replicate:03d}__{token}"
        template = templates[global_index % len(templates)]
        prompt = template.format(
            occluder=_display_occluder(occluder),
            position=_display_position(position),
            direction=_display_direction(direction),
            risk=risk,
            speed=f"{speed:.1f}",
        )
        cases.append(
            SemanticRetentionCase(
                sample_id=sample_id,
                condition_id=condition_id,
                input_raw=str(scene),
                source_relative_path=str(rel),
                source_scenario_type=source_type,
                prompt=prompt,
                occluder_type=occluder,
                occluder_position=position,
                direction=direction,
                pedestrian_speed_mps=speed,
                risk_level=risk,
                replicate=replicate,
            )
        )

    return cases


def summarize_semantic_retention_cases(
    cases: Iterable[SemanticRetentionCase],
) -> Dict[str, Any]:
    rows = list(cases)
    return {
        "num_cases": len(rows),
        "num_conditions": len({row.condition_id for row in rows}),
        "axis_counts": {
            "occluder_type": dict(sorted(Counter(row.occluder_type for row in rows).items())),
            "occluder_position": dict(
                sorted(Counter(row.occluder_position for row in rows).items())
            ),
            "direction": dict(sorted(Counter(row.direction for row in rows).items())),
            "risk_level": dict(sorted(Counter(row.risk_level for row in rows).items())),
            "pedestrian_speed_mps": dict(
                sorted(Counter(f"{row.pedestrian_speed_mps:.1f}" for row in rows).items())
            ),
            "source_scenario_type": dict(
                sorted(Counter(row.source_scenario_type for row in rows).items())
            ),
        },
    }


class PositionAwareOccludedPedestrianPipeline(OccludedPedestrianPipeline):
    """Existing B1 editor plus a required occluder-position band."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        install_position_aware_layout_patch()
        super().__init__(*args, **kwargs)

    def run_case(self, case: SemanticRetentionCase) -> Dict[str, Any]:
        with requested_occluder_position(case.occluder_position):
            row = super().run_case(case)
        return self._append_position_validation(case, row)

    def _append_position_validation(
        self,
        case: SemanticRetentionCase,
        row: Dict[str, Any],
    ) -> Dict[str, Any]:
        sample_id = case.sample_id
        b1_dir = self.layout.b1_cache / sample_id
        artifact_root = self.layout.artifacts / sample_id
        metrics_path = artifact_root / "04_evaluation/b1_metrics.json"
        label_path = b1_dir / "scenario_label.json"
        summary_path = artifact_root / "sample_summary.json"
        semantic_path = b1_dir / "semantic_report.json"

        metrics = _read_json(metrics_path)
        ratio = float(metrics.get("occluder", {}).get("projection_ratio", -1.0))
        position_ok = position_matches(case.occluder_position, ratio)
        checks = dict(metrics.get("checks", {}))
        checks["occluder_position_match"] = bool(position_ok)
        metrics["checks"] = checks
        metrics["requested_occluder_position"] = case.occluder_position
        metrics["occluder_projection_ratio"] = ratio
        metrics["overall_pass"] = bool(all(bool(value) for value in checks.values()))
        metrics["semantic_satisfaction_rate"] = (
            sum(bool(value) for value in checks.values()) / len(checks) if checks else 0.0
        )
        save_json(metrics_path, metrics)

        label = _read_json(label_path)
        label.update(
            {
                "occluder_position": case.occluder_position,
                "occluder_projection_ratio": ratio,
                "occluder_position_match": bool(position_ok),
                "accepted": bool(label.get("accepted", False) and metrics["overall_pass"]),
            }
        )
        if semantic_path.exists():
            semantic_payload = _read_json(semantic_path)
            layout = (
                semantic_payload.get("report", {})
                .get("extra", {})
                .get("occluder_layout", {})
            )
            if layout:
                label["occluder_layout"] = layout
        save_json(label_path, label)

        row.update(
            {
                "occluder_position": case.occluder_position,
                "occluder_projection_ratio": ratio,
                "occluder_position_match": bool(position_ok),
                "b1_pass": bool(row.get("b1_pass", False) and metrics["overall_pass"]),
                "b1_semantic_satisfaction_rate": float(
                    metrics.get("semantic_satisfaction_rate", 0.0)
                ),
            }
        )
        save_json(summary_path, row)
        return row


def _stable_index(text: str, modulus: int) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16) % modulus


def _source_scenario_type(rel: Path) -> str:
    parts = rel.parts
    if len(parts) >= 3:
        return parts[-3]
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def _display_occluder(value: str) -> str:
    return {
        "vehicle": "stationary vehicle",
        "bicycle": "stationary bicycle",
        "generic_object": "generic roadside object",
        "traffic_cone": "traffic cone",
        "barrier": "road barrier",
        "czone_sign": "construction-zone sign",
    }.get(value, value.replace("_", " "))


def _display_position(value: str) -> str:
    return {
        "near_ego": "closer to ego",
        "midway": "midway between ego and the pedestrian",
        "near_pedestrian": "closer to the pedestrian",
    }[normalize_occluder_position(value)]


def _display_direction(value: str) -> str:
    return value.replace("_", " ")


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)
