"""Simplified matrix: coarse B0 topology filter + high-level hazard controls only."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from sledge.semantic_control.io import load_raw_scene
from sledge.semantic_control.occluded_pedestrian_pipeline.experiment_matrix import ExperimentCase
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.topology_filter import (
    normalize_topology_family,
    scene_matches_topology,
)


def build_simple_experiment_cases(
    *,
    input_root: Path,
    config: Dict[str, Any],
    max_cases: int | None = None,
    glob_pattern: str = "**/sledge_raw.gz",
) -> List[ExperimentCase]:
    """Build cases without editing B0 topology.

    ``topology_family`` is used only to retain/drop source scenes. Numeric hazard
    parameters such as exact pedestrian speed are left ``None`` so the existing
    hazard sampler resolves them from the risk-level ranges.
    """
    input_root = Path(input_root)
    topology_family = normalize_topology_family(config.get("topology_family", "any"))
    occluder_types = list(config.get("occluder_types", ["vehicle"]))
    sides = list(config.get("occluder_sides", ["left", "right"]))
    risk_levels = list(config.get("risk_levels", [config.get("risk_level", "moderate")]))
    total_samples = int(config.get("total_samples", 30))
    prompt_template = str(
        config.get(
            "prompt_template",
            "A {risk} occluded-pedestrian event: a {occluder} on the {side} hides a pedestrian who emerges toward the ego path.",
        )
    )

    candidates: List[Path] = []
    for path in sorted(input_root.glob(glob_pattern)):
        scene, _ = load_raw_scene(path)
        if scene_matches_topology(scene, topology_family):
            candidates.append(path)
    if not candidates:
        raise ValueError(
            f"No B0 scenes match topology_family={topology_family!r} under {input_root}"
        )

    cases: List[ExperimentCase] = []
    for idx in range(total_samples):
        occluder = str(occluder_types[idx % len(occluder_types)])
        side = str(sides[(idx // len(occluder_types)) % len(sides)])
        risk = str(risk_levels[(idx // max(1, len(occluder_types) * len(sides))) % len(risk_levels)])
        scene = candidates[idx % len(candidates)]
        rel = scene.relative_to(input_root)
        prompt = prompt_template.format(occluder=occluder, side=side, risk=risk)
        cases.append(
            ExperimentCase(
                sample_id=f"simple_{topology_family}_{idx:05d}",
                condition_id=f"topo-{topology_family}__occ-{occluder}__side-{side}__risk-{risk}",
                input_raw=str(scene),
                source_relative_path=str(rel),
                source_scenario_type=topology_family,
                prompt=prompt,
                occluder_type=occluder,
                occluder_side=side,
                pedestrian_speed_mps=None,
                risk_level=risk,
                replicate=idx,
                prompt_case_id="simple_coarse_topology",
                template_seed=idx,
            )
        )
        if max_cases is not None and len(cases) >= max_cases:
            break
    return cases


__all__ = ["build_simple_experiment_cases"]
