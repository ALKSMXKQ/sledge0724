"""Deterministic debug and diversity experiment matrices."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List

from sledge.semantic_control.occluded_pedestrian_pipeline.language.eventframe_adapter import (
    ControlOverrides,
)


@dataclass(frozen=True)
class ExperimentCase:
    sample_id: str
    condition_id: str
    input_raw: str
    source_relative_path: str
    source_scenario_type: str
    prompt: str
    occluder_type: str
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


def load_matrix_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def discover_raw_scenes(input_root: Path, glob_pattern: str = "**/sledge_raw.gz") -> List[Path]:
    scenes = sorted(input_root.glob(glob_pattern))
    if not scenes:
        raise FileNotFoundError(f"No scenes matched {glob_pattern!r} under {input_root}")
    return scenes


def build_experiment_cases(
    *,
    input_root: Path,
    profile: str,
    matrix_config: Dict[str, Any],
    glob_pattern: str = "**/sledge_raw.gz",
    max_cases: int | None = None,
) -> List[ExperimentCase]:
    scenes = discover_raw_scenes(input_root, glob_pattern)
    profiles = matrix_config.get("profiles", {})
    if profile not in profiles:
        raise KeyError(f"Unknown profile={profile!r}; available={sorted(profiles)}")
    cfg = dict(profiles[profile])

    occluders = list(cfg["occluder_types"])
    directions = list(cfg["directions"])
    speeds = [float(v) for v in cfg["pedestrian_speeds_mps"]]
    risk = str(cfg.get("risk_level", "moderate"))
    samples_per_condition = int(cfg.get("samples_per_condition", 0))
    total_samples = int(cfg.get("total_samples", 0))
    if samples_per_condition <= 0 and total_samples <= 0:
        raise ValueError("profile requires a positive samples_per_condition or total_samples")
    templates = list(matrix_config.get("prompt_templates", []))
    if not templates:
        raise ValueError("experiment matrix needs at least one prompt template")

    conditions = list(itertools.product(occluders, directions, speeds))
    scenes_by_type: Dict[str, List[Path]] = defaultdict(list)
    for scene in scenes:
        rel = scene.relative_to(input_root)
        scenes_by_type[_source_scenario_type(rel)].append(scene)
    source_types = sorted(
        scenes_by_type,
        key=lambda value: hashlib.sha1(f"{profile}|{value}".encode("utf-8")).hexdigest(),
    )
    scheduled = []
    if total_samples > 0:
        condition_replicates: Counter = Counter()
        occluder_usage: Counter = Counter()
        direction_usage: Counter = Counter()
        speed_usage: Counter = Counter()
        for global_index in range(total_samples):
            condition_index, condition = min(
                enumerate(conditions),
                key=lambda item: (
                    occluder_usage[item[1][0]],
                    direction_usage[item[1][1]],
                    speed_usage[item[1][2]],
                    condition_replicates[item[1]],
                    item[0],
                ),
            )
            replicate = int(condition_replicates[condition])
            condition_replicates[condition] += 1
            occluder_usage[condition[0]] += 1
            direction_usage[condition[1]] += 1
            speed_usage[condition[2]] += 1
            scheduled.append((global_index, condition_index, condition, replicate))
    else:
        for condition_index, condition in enumerate(conditions):
            for replicate in range(samples_per_condition):
                scheduled.append((len(scheduled), condition_index, condition, replicate))

    cases: List[ExperimentCase] = []
    used_scenes = set()
    for global_index, condition_index, (occluder, direction, speed), replicate in scheduled:
        condition_id = f"occ-{occluder}__dir-{direction}__speed-{speed:.1f}"
        # Original scenario type is an explicit diversity axis. Round-robin
        # over source families and enforce a globally unique training scene for
        # finite pilot profiles such as pilot100.
        source_type = source_types[(condition_index + global_index) % len(source_types)]
        typed_scenes = scenes_by_type[source_type]
        typed_offset = _stable_index(f"{condition_id}|{source_type}", len(typed_scenes))
        if total_samples > 0:
            scene = _pick_unused_scene(typed_scenes, typed_offset + replicate, used_scenes)
        else:
            scene = typed_scenes[(typed_offset + replicate) % len(typed_scenes)]
        rel = scene.relative_to(input_root)
        template = templates[global_index % len(templates)]
        prompt = template.format(
            occluder=_display_occluder(occluder),
            direction=_display_direction(direction),
            speed=f"{speed:.1f}",
            risk=risk,
        )
        token = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:8]
        sample_id = f"{condition_id}__r{replicate:02d}__{token}"
        cases.append(
            ExperimentCase(
                sample_id=sample_id,
                condition_id=condition_id,
                input_raw=str(scene),
                source_relative_path=str(rel),
                source_scenario_type=source_type,
                prompt=prompt,
                occluder_type=occluder,
                direction=direction,
                pedestrian_speed_mps=speed,
                risk_level=risk,
                replicate=replicate,
            )
        )
        if total_samples > 0:
            used_scenes.add(scene)
        if max_cases is not None and len(cases) >= max_cases:
            return cases
    return cases


def summarize_case_matrix(cases: Iterable[ExperimentCase]) -> Dict[str, Any]:
    rows = list(cases)
    conditions = sorted({row.condition_id for row in rows})
    source_types = sorted({row.source_scenario_type for row in rows})
    source_type_counts = Counter(row.source_scenario_type for row in rows)
    occluder_counts = Counter(row.occluder_type for row in rows)
    direction_counts = Counter(row.direction for row in rows)
    speed_counts = Counter(f"{row.pedestrian_speed_mps:.1f}" for row in rows)
    return {
        "num_cases": len(rows),
        "num_conditions": len(conditions),
        "conditions": conditions,
        "source_scenario_types": source_types,
        "axis_counts": {
            "occluder_type": dict(sorted(occluder_counts.items())),
            "direction": dict(sorted(direction_counts.items())),
            "pedestrian_speed_mps": dict(sorted(speed_counts.items())),
            "source_scenario_type": dict(sorted(source_type_counts.items())),
        },
    }


def _stable_index(text: str, modulus: int) -> int:
    value = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)
    return value % modulus


def _pick_unused_scene(scenes: List[Path], start: int, used_scenes: set) -> Path:
    for offset in range(len(scenes)):
        candidate = scenes[(start + offset) % len(scenes)]
        if candidate not in used_scenes:
            return candidate
    raise RuntimeError("Unable to select a unique source scene from the requested source scenario type")


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


def _display_direction(value: str) -> str:
    return value.replace("_", " ")
