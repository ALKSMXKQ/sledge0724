"""Deterministic experiment matrices using occluder side, not semantic direction."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sledge.semantic_control.occluded_pedestrian_pipeline.language.eventframe_adapter import (
    ControlOverrides,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.language.occluded_prompt_matrix import (
    OccludedPromptCase,
)


SEMANTIC_DIRECTION = "occluder_to_ego_path"


@dataclass(frozen=True)
class ExperimentCase:
    sample_id: str
    condition_id: str
    input_raw: str
    source_relative_path: str
    source_scenario_type: str
    prompt: str
    occluder_type: Optional[str]
    occluder_side: str
    pedestrian_speed_mps: Optional[float]
    risk_level: Optional[str]
    replicate: int
    prompt_case_id: str = ""
    template_seed: Optional[int] = None
    semantic_direction: str = SEMANTIC_DIRECTION

    @property
    def direction(self) -> str:
        """Concrete B1 direction, available only after a side is fixed."""

        if self.occluder_side == "left":
            return "left_to_right"
        if self.occluder_side == "right":
            return "right_to_left"
        return "derived_after_sampling"

    @property
    def overrides(self) -> ControlOverrides:
        return ControlOverrides(
            occluder_type=self.occluder_type,
            occluder_side=(
                self.occluder_side
                if self.occluder_side in {"left", "right"}
                else None
            ),
            pedestrian_speed_mps=self.pedestrian_speed_mps,
            risk_level=self.risk_level,
            seed=self.template_seed,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["direction"] = self.direction
        return payload


def load_matrix_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def discover_raw_scenes(
    input_root: Path,
    glob_pattern: str = "**/sledge_raw.gz",
) -> List[Path]:
    scenes = sorted(Path(input_root).glob(glob_pattern))
    if not scenes:
        raise FileNotFoundError(
            f"No scenes matched {glob_pattern!r} under {input_root}"
        )
    return scenes


def build_experiment_cases(
    *,
    input_root: Path,
    profile: str,
    matrix_config: Dict[str, Any],
    glob_pattern: str = "**/sledge_raw.gz",
    max_cases: int | None = None,
) -> List[ExperimentCase]:
    """Build legacy-scale matrices with a side axis.

    Old configurations containing ``directions`` are accepted and converted:
    ``left_to_right -> left`` and ``right_to_left -> right``.
    """

    scenes = discover_raw_scenes(input_root, glob_pattern)
    profiles = matrix_config.get("profiles", {})
    if profile not in profiles:
        raise KeyError(f"Unknown profile={profile!r}; available={sorted(profiles)}")
    cfg = dict(profiles[profile])

    occluders = list(cfg["occluder_types"])
    sides = _profile_sides(cfg)
    speeds = [float(value) for value in cfg["pedestrian_speeds_mps"]]
    risk = str(cfg.get("risk_level", "moderate"))
    samples_per_condition = int(cfg.get("samples_per_condition", 0))
    total_samples = int(cfg.get("total_samples", 0))
    if samples_per_condition <= 0 and total_samples <= 0:
        raise ValueError(
            "profile requires positive samples_per_condition or total_samples"
        )

    templates = list(matrix_config.get("prompt_templates", []))
    if not templates:
        raise ValueError("experiment matrix needs at least one prompt template")

    conditions = list(itertools.product(occluders, sides, speeds))
    scenes_by_type = _group_scenes_by_type(input_root, scenes)
    source_types = _stable_source_types(profile, scenes_by_type)
    scheduled = _schedule_conditions(
        conditions,
        total_samples=total_samples,
        samples_per_condition=samples_per_condition,
    )

    cases: List[ExperimentCase] = []
    used_scenes = set()
    for global_index, condition_index, condition, replicate in scheduled:
        occluder, side, speed = condition
        concrete_direction = _direction_from_side(side)
        condition_id = (
            f"occ-{occluder}__side-{side}__speed-{speed:.1f}"
        )
        source_type = source_types[
            (condition_index + global_index) % len(source_types)
        ]
        typed_scenes = scenes_by_type[source_type]
        typed_offset = _stable_index(
            f"{condition_id}|{source_type}",
            len(typed_scenes),
        )
        if total_samples > 0:
            scene = _pick_unused_scene(
                typed_scenes,
                typed_offset + replicate,
                used_scenes,
            )
        else:
            scene = typed_scenes[(typed_offset + replicate) % len(typed_scenes)]
        rel = scene.relative_to(input_root)

        template = templates[global_index % len(templates)]
        prompt = template.format(
            occluder=_display_occluder(occluder),
            side=_display_side(side),
            direction=_display_direction(concrete_direction),
            speed=f"{speed:.1f}",
            risk=risk,
        )
        token = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:8]
        seed = _stable_seed(condition_id, str(rel), replicate)
        cases.append(
            ExperimentCase(
                sample_id=f"{condition_id}__r{replicate:02d}__{token}",
                condition_id=condition_id,
                input_raw=str(scene),
                source_relative_path=str(rel),
                source_scenario_type=source_type,
                prompt=prompt,
                occluder_type=occluder,
                occluder_side=side,
                pedestrian_speed_mps=speed,
                risk_level=risk,
                replicate=replicate,
                template_seed=seed,
            )
        )
        if total_samples > 0:
            used_scenes.add(scene)
        if max_cases is not None and len(cases) >= max_cases:
            return cases
    return cases


def build_prompt_matrix_cases(
    *,
    input_root: Path,
    prompt_cases: Sequence[OccludedPromptCase],
    scenes_per_prompt: int = 2,
    glob_pattern: str = "**/sledge_raw.gz",
    max_cases: Optional[int] = None,
) -> List[ExperimentCase]:
    """Pair each diagnostic prompt with distinct source scenes.

    Language-derived occluder type, speed and risk are intentionally not
    overridden unless the expected matrix explicitly contains a numeric speed.
    """

    if scenes_per_prompt <= 0:
        raise ValueError("scenes_per_prompt must be positive")
    scenes = discover_raw_scenes(input_root, glob_pattern)
    scenes_by_type = _group_scenes_by_type(input_root, scenes)
    source_types = _stable_source_types("language_matrix", scenes_by_type)
    used = set()
    output: List[ExperimentCase] = []

    for prompt_index, prompt_case in enumerate(prompt_cases):
        expected = dict(prompt_case.expected)
        expected_side = str(expected.get("occluder_side", "sample_once"))
        speed = expected.get("actor_speed_mps")
        risk = expected.get("risk_level")
        for replicate in range(scenes_per_prompt):
            source_type = source_types[
                (prompt_index + replicate) % len(source_types)
            ]
            typed = scenes_by_type[source_type]
            start = _stable_index(
                f"{prompt_case.case_id}|{replicate}|{source_type}",
                len(typed),
            )
            scene = _pick_unused_scene(typed, start, used)
            used.add(scene)
            rel = scene.relative_to(input_root)
            token = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:8]
            seed = _stable_seed(prompt_case.case_id, str(rel), replicate)
            condition_id = f"language-{prompt_case.case_id}"
            output.append(
                ExperimentCase(
                    sample_id=(
                        f"{condition_id}__r{replicate:02d}__{token}"
                    ),
                    condition_id=condition_id,
                    input_raw=str(scene),
                    source_relative_path=str(rel),
                    source_scenario_type=source_type,
                    prompt=prompt_case.prompt,
                    # Keep None to test language-derived type.
                    occluder_type=None,
                    occluder_side=(
                        expected_side
                        if expected_side in {"left", "right"}
                        else "sample_once"
                    ),
                    pedestrian_speed_mps=(
                        float(speed) if speed is not None else None
                    ),
                    risk_level=str(risk) if risk is not None else None,
                    replicate=replicate,
                    prompt_case_id=prompt_case.case_id,
                    template_seed=seed,
                )
            )
            if max_cases is not None and len(output) >= max_cases:
                return output
    return output


def summarize_case_matrix(cases: Iterable[ExperimentCase]) -> Dict[str, Any]:
    rows = list(cases)
    conditions = sorted({row.condition_id for row in rows})
    source_types = sorted({row.source_scenario_type for row in rows})
    source_type_counts = Counter(row.source_scenario_type for row in rows)
    occluder_counts = Counter(str(row.occluder_type or "language_derived") for row in rows)
    side_counts = Counter(row.occluder_side for row in rows)
    speed_counts = Counter(
        "language_derived"
        if row.pedestrian_speed_mps is None
        else f"{row.pedestrian_speed_mps:.1f}"
        for row in rows
    )
    return {
        "num_cases": len(rows),
        "num_conditions": len(conditions),
        "conditions": conditions,
        "source_scenario_types": source_types,
        "semantic_direction": SEMANTIC_DIRECTION,
        "axis_counts": {
            "occluder_type": dict(sorted(occluder_counts.items())),
            "occluder_side": dict(sorted(side_counts.items())),
            "pedestrian_speed_mps": dict(sorted(speed_counts.items())),
            "source_scenario_type": dict(sorted(source_type_counts.items())),
        },
    }


def _profile_sides(cfg: Dict[str, Any]) -> List[str]:
    if cfg.get("occluder_sides"):
        sides = [str(value) for value in cfg["occluder_sides"]]
    else:
        mapping = {
            "left_to_right": "left",
            "right_to_left": "right",
        }
        sides = [
            mapping[str(value)]
            for value in cfg.get("directions", [])
            if str(value) in mapping
        ]
    if not sides:
        raise ValueError("profile needs occluder_sides or supported directions")
    invalid = sorted(set(sides) - {"left", "right"})
    if invalid:
        raise ValueError(f"invalid occluder sides: {invalid}")
    return sides


def _group_scenes_by_type(
    input_root: Path,
    scenes: Sequence[Path],
) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = defaultdict(list)
    for scene in scenes:
        grouped[_source_scenario_type(scene.relative_to(input_root))].append(scene)
    return grouped


def _stable_source_types(
    profile: str,
    scenes_by_type: Dict[str, List[Path]],
) -> List[str]:
    return sorted(
        scenes_by_type,
        key=lambda value: hashlib.sha1(
            f"{profile}|{value}".encode("utf-8")
        ).hexdigest(),
    )


def _schedule_conditions(
    conditions,
    *,
    total_samples: int,
    samples_per_condition: int,
):
    scheduled = []
    if total_samples > 0:
        condition_replicates: Counter = Counter()
        occluder_usage: Counter = Counter()
        side_usage: Counter = Counter()
        speed_usage: Counter = Counter()
        for global_index in range(total_samples):
            condition_index, condition = min(
                enumerate(conditions),
                key=lambda item: (
                    occluder_usage[item[1][0]],
                    side_usage[item[1][1]],
                    speed_usage[item[1][2]],
                    condition_replicates[item[1]],
                    item[0],
                ),
            )
            replicate = int(condition_replicates[condition])
            condition_replicates[condition] += 1
            occluder_usage[condition[0]] += 1
            side_usage[condition[1]] += 1
            speed_usage[condition[2]] += 1
            scheduled.append(
                (global_index, condition_index, condition, replicate)
            )
        return scheduled

    for condition_index, condition in enumerate(conditions):
        for replicate in range(samples_per_condition):
            scheduled.append(
                (len(scheduled), condition_index, condition, replicate)
            )
    return scheduled


def _stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def _stable_index(text: str, modulus: int) -> int:
    value = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)
    return value % modulus


def _pick_unused_scene(
    scenes: List[Path],
    start: int,
    used_scenes: set,
) -> Path:
    for offset in range(len(scenes)):
        candidate = scenes[(start + offset) % len(scenes)]
        if candidate not in used_scenes:
            return candidate
    raise RuntimeError(
        "Unable to select a unique source scene from the requested source type"
    )


def _source_scenario_type(rel: Path) -> str:
    parts = rel.parts
    if len(parts) >= 3:
        return parts[-3]
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def _direction_from_side(side: str) -> str:
    if side == "left":
        return "left_to_right"
    if side == "right":
        return "right_to_left"
    return "derived_after_sampling"


def _display_occluder(value: str) -> str:
    return {
        "vehicle": "stationary vehicle",
        "bicycle": "stationary bicycle",
        "generic_object": "generic roadside object",
        "traffic_cone": "traffic cone",
        "barrier": "road barrier",
        "czone_sign": "construction-zone sign",
    }.get(value, value.replace("_", " "))


def _display_side(value: str) -> str:
    return {
        "left": "on the left roadside",
        "right": "on the right roadside",
    }.get(value, "at the roadside")


def _display_direction(value: str) -> str:
    return value.replace("_", " ")
