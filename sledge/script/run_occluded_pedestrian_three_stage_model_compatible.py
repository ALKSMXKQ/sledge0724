"""Model-compatible launcher for the occluded-pedestrian three-stage study.

This wrapper keeps the existing B1 -> RVAE -> diffusion pipeline unchanged, but
materializes pedestrian speeds from the *actual* SLEDGE/RVAE input config before
B1 construction.  It also installs a fail-closed controlled-slot matcher so a
truncated pedestrian/occluder can never be silently replaced by an unrelated
background object.

Why this launcher exists
------------------------
``sledge_raw_feature_processing`` clamps pedestrian velocity to
``pedestrian_max_velocity``.  If an experiment asks for a faster pedestrian,
the dangerous scene changes before it reaches the RVAE.  Counting that change
as RVAE/diffusion semantic loss would be scientifically incorrect.

The model-compatible matrix therefore stores speed fractions in [0, 1].  This
launcher resolves them to concrete m/s values from the loaded model input
configuration, writes the resolved matrix into the run manifest directory, and
then delegates to ``run_occluded_pedestrian_three_stage``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional

from sledge.script.build_paired_original_edited_vector_caches import (
    build_sledge_config,
)
from sledge.script import run_occluded_pedestrian_three_stage as base_runner
from sledge.semantic_control.io import save_json
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation import (
    pilot_export,
    stage_comparison,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.processed_slot_guard import (
    strict_match_processed_slot,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.refinement_runner import (
    OccludedPedestrianHalfDenoiseRunner,
)


DEFAULT_MODEL_COMPATIBLE_MATRIX = (
    base_runner.SLEDGE_ROOT
    / "semantic_control/occluded_pedestrian_pipeline/configs/"
    "semantic_retention_matrix_model_compatible.json"
)


def build_probe_parser() -> argparse.ArgumentParser:
    """Parse only the arguments needed before delegating to the base runner."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--matrix-config",
        type=Path,
        default=DEFAULT_MODEL_COMPATIBLE_MATRIX,
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    probe_args, _ = build_probe_parser().parse_known_args(forwarded)

    run_root = probe_args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    manifests = run_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    sledge_config = build_sledge_config(str(probe_args.config))
    pedestrian_max_velocity = float(sledge_config.pedestrian_max_velocity)
    if pedestrian_max_velocity <= 0.0:
        raise ValueError(
            "Loaded SLEDGE config has non-positive pedestrian_max_velocity="
            f"{pedestrian_max_velocity}"
        )

    matrix = _read_json(probe_args.matrix_config)
    resolved_matrix, speed_audit = _materialize_model_compatible_speeds(
        matrix,
        pedestrian_max_velocity=pedestrian_max_velocity,
    )

    resolved_matrix_path = manifests / "resolved_semantic_retention_matrix.json"
    save_json(resolved_matrix_path, resolved_matrix)
    save_json(
        manifests / "model_input_contract.json",
        {
            "schema_version": "occluded_pedestrian_model_input_contract_v1",
            "source_config": str(probe_args.config.resolve()),
            "source_matrix": str(probe_args.matrix_config.resolve()),
            "resolved_matrix": str(resolved_matrix_path),
            "frame": list(sledge_config.frame),
            "num_vehicles": int(sledge_config.num_vehicles),
            "num_pedestrians": int(sledge_config.num_pedestrians),
            "num_static_objects": int(sledge_config.num_static_objects),
            "vehicle_max_velocity_mps": float(
                sledge_config.vehicle_max_velocity
            ),
            "pedestrian_max_velocity_mps": pedestrian_max_velocity,
            "speed_policy": (
                "materialize_fraction_of_model_max_and_fail_on_"
                "out_of_range_explicit_speed"
            ),
            "processed_slot_policy": (
                "exact_geometry_match_or_fail_closed"
            ),
            "speed_profiles": speed_audit,
        },
    )

    _install_fail_closed_slot_matcher()

    print(
        "[MODEL INPUT] pedestrian_max_velocity="
        f"{pedestrian_max_velocity:.3f} m/s"
    )
    for profile_name, row in speed_audit.items():
        print(
            f"[MODEL INPUT] {profile_name}: "
            f"pedestrian_speeds_mps={row['resolved_speeds_mps']}"
        )
    print(
        "[MODEL INPUT] resolved matrix -> "
        f"{resolved_matrix_path}"
    )

    forwarded = _set_option(
        forwarded,
        "--matrix-config",
        str(resolved_matrix_path),
    )
    base_runner.main(forwarded)


def _materialize_model_compatible_speeds(
    matrix: Mapping[str, Any],
    *,
    pedestrian_max_velocity: float,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve each profile to concrete speeds accepted by model preprocessing."""

    resolved = deepcopy(dict(matrix))
    profiles = resolved.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Experiment matrix must contain a non-empty profiles object")

    audit: Dict[str, Any] = {}
    for profile_name, raw_profile in profiles.items():
        if not isinstance(raw_profile, dict):
            raise TypeError(
                f"Profile {profile_name!r} must be a JSON object"
            )
        profile = dict(raw_profile)
        fractions = profile.get(
            "pedestrian_speed_fractions_of_model_max"
        )
        explicit_speeds = profile.get("pedestrian_speeds_mps")

        if fractions is not None:
            if not isinstance(fractions, list) or not fractions:
                raise ValueError(
                    f"Profile {profile_name!r} has an empty/invalid "
                    "pedestrian_speed_fractions_of_model_max"
                )
            normalized_fractions: List[float] = []
            concrete: List[float] = []
            for value in fractions:
                fraction = float(value)
                if not 0.0 < fraction <= 1.0:
                    raise ValueError(
                        f"Profile {profile_name!r} speed fraction {fraction} "
                        "must be in (0, 1]"
                    )
                speed = round(
                    pedestrian_max_velocity * fraction,
                    3,
                )
                if speed <= 0.0 or speed > pedestrian_max_velocity + 1e-9:
                    raise ValueError(
                        f"Resolved speed {speed} for profile {profile_name!r} "
                        "is outside the model input range"
                    )
                normalized_fractions.append(fraction)
                concrete.append(speed)

            concrete = sorted(set(concrete))
            if not concrete:
                raise ValueError(
                    f"Profile {profile_name!r} resolved to zero speeds"
                )
            profile["pedestrian_speeds_mps"] = concrete
            profile[
                "resolved_pedestrian_max_velocity_mps"
            ] = pedestrian_max_velocity
            profile[
                "resolved_speed_source"
            ] = "fraction_of_loaded_model_max"
            profiles[profile_name] = profile
            audit[profile_name] = {
                "source": "fraction_of_loaded_model_max",
                "fractions": normalized_fractions,
                "resolved_speeds_mps": concrete,
            }
            continue

        if explicit_speeds is None:
            raise ValueError(
                f"Profile {profile_name!r} must define either "
                "pedestrian_speed_fractions_of_model_max or "
                "pedestrian_speeds_mps"
            )
        if not isinstance(explicit_speeds, list) or not explicit_speeds:
            raise ValueError(
                f"Profile {profile_name!r} pedestrian_speeds_mps is empty"
            )

        concrete = sorted(set(float(value) for value in explicit_speeds))
        bad = [
            speed
            for speed in concrete
            if speed <= 0.0
            or speed > pedestrian_max_velocity + 1e-9
        ]
        if bad:
            raise ValueError(
                f"Profile {profile_name!r} requests pedestrian speeds {bad} "
                f"outside loaded model limit {pedestrian_max_velocity:.3f} m/s. "
                "Use pedestrian_speed_fractions_of_model_max instead."
            )
        profile["pedestrian_speeds_mps"] = concrete
        profile[
            "resolved_pedestrian_max_velocity_mps"
        ] = pedestrian_max_velocity
        profile[
            "resolved_speed_source"
        ] = "explicit_model_compatible_speed"
        profiles[profile_name] = profile
        audit[profile_name] = {
            "source": "explicit_model_compatible_speed",
            "fractions": None,
            "resolved_speeds_mps": concrete,
        }

    resolved["profiles"] = profiles
    resolved[
        "resolved_model_pedestrian_max_velocity_mps"
    ] = pedestrian_max_velocity
    resolved[
        "resolved_speed_policy"
    ] = "model_input_compatible_fail_closed"
    return resolved, audit


def _install_fail_closed_slot_matcher() -> None:
    """Use one strict matcher at B1 export, RVAE reconstruction and diffusion."""

    # pilot_export imported the historical helper into its own module globals,
    # so replace that binding explicitly.
    pilot_export._match_processed_slot = strict_match_processed_slot

    # Keep other stage-comparison utilities consistent when they are called in
    # the same process.
    stage_comparison._match_processed_slot = strict_match_processed_slot

    # RVAE reconstruction and diffusion resolve slots through this class method.
    OccludedPedestrianHalfDenoiseRunner._match_slot = staticmethod(
        strict_match_processed_slot
    )


def _set_option(argv: List[str], name: str, value: str) -> List[str]:
    """Replace one CLI option while preserving every unrelated base-runner arg."""

    output = list(argv)
    if name in output:
        index = output.index(name)
        if index + 1 >= len(output):
            raise ValueError(f"Missing value for {name}")
        output[index + 1] = value
        return output
    output.extend([name, value])
    return output


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


if __name__ == "__main__":
    main()
