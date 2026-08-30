"""One-command occluded-pedestrian B1 -> RVAE -> diffusion experiment.

This is the single supported entry point for the semantic-retention study. It
reuses the hierarchical B1 editor and protected diffusion path, persists an RVAE
bottleneck reconstruction, validates simulator-readable gzip caches, and keeps
raw learned-model retention separate from semantic-protected outputs.

Before B1 construction, this runner resolves pedestrian speed fractions against
the actual SLEDGE/RVAE ``pedestrian_max_velocity`` and also selects source scenes
with enough standard preprocessing capacity to retain one controlled pedestrian
and one controlled occluder.  The latter deliberately preserves the original
SLEDGE nearest-N preprocessing policy: we change the dataset source scene, not
the trained model's input-selection rule.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_utils import (
    coords_in_frame,
)
from sledge.script.build_paired_original_edited_vector_caches import build_sledge_config
from sledge.semantic_control.io import load_raw_scene, save_json
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.pilot_export import (
    export_b1_simulation_cache,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.experiment_matrix import (
    ExperimentCase,
    build_experiment_cases,
    load_matrix_config,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.diffusion_modes import (
    RAW_DIFFUSION_BASELINE,
    SEMANTIC_PROTECTED,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.reconstruction_runner import (
    run_rvae_reconstruction,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.object_types import (
    element_name_for_occluder,
    embed_type_overrides,
    make_type_override,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.pipeline import (
    OccludedPedestrianPipeline,
    RunLayout,
    run_diffusion_comparison,
    run_half_denoise,
)
from sledge.simulation.scenarios.sledge_scenario.sledge_scenario import SledgeScenario


REPO_ROOT = Path(__file__).resolve().parents[2]
SLEDGE_ROOT = REPO_ROOT / "sledge"
DEFAULT_MATRIX = (
    SLEDGE_ROOT
    / "semantic_control/occluded_pedestrian_pipeline/configs/semantic_retention_matrix.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-build occluded pedestrian dart-out scenes and persist "
            "simulation-ready B1, RVAE and protected-diffusion gzip caches."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root containing source **/sledge_raw.gz scenes",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Output root for all three stages",
    )
    parser.add_argument(
        "--matrix-config",
        type=Path,
        default=DEFAULT_MATRIX,
    )
    parser.add_argument(
        "--profiles",
        default="retention_moderate",
        help=(
            "Comma-separated matrix profiles. For risk-level diversity use "
            "retention_mild,retention_moderate,retention_aggressive"
        ),
    )
    parser.add_argument(
        "--glob-pattern",
        default="**/sledge_raw.gz",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit the interleaved multi-profile case list for smoke runs",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Expanded semantic_img2img / SLEDGE OmegaConf yaml",
    )
    parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--diffusion-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--diffusion-run",
        choices=["protected", "both"],
        default="both",
        help=(
            "both also saves an unprotected scientific baseline; protected "
            "always runs because it is the official semantic-guaranteed output"
        ),
    )
    parser.add_argument(
        "--llm-provider",
        choices=["none", "ollama"],
        default="none",
    )
    parser.add_argument(
        "--llm-model",
        default="qwen2.5:7b",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
    )
    parser.add_argument(
        "--no-b1-visuals",
        action="store_true",
    )
    parser.add_argument(
        "--no-b1-previews",
        action="store_true",
    )
    parser.add_argument(
        "--save-diffusion-visuals",
        action="store_true",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    matrix = load_matrix_config(args.matrix_config)
    matrix, model_input_contract = _resolve_model_input_matrix(
        matrix,
        config=args.config,
        run_root=run_root,
        source_matrix=args.matrix_config,
    )

    profiles = [
        value.strip()
        for value in str(args.profiles).split(",")
        if value.strip()
    ]
    if not profiles:
        raise ValueError("--profiles must contain at least one profile name")

    print(
        "[MODEL INPUT] pedestrian_max_velocity="
        f"{model_input_contract['pedestrian_max_velocity_mps']:.3f} m/s"
    )
    for profile in profiles:
        if profile in model_input_contract["speed_profiles"]:
            print(
                f"[MODEL INPUT] {profile}: pedestrian_speeds_mps="
                f"{model_input_contract['speed_profiles'][profile]['resolved_speeds_mps']}"
            )

    cases = _build_profile_cases(
        input_root=args.input_root,
        matrix=matrix,
        profiles=profiles,
        glob_pattern=args.glob_pattern,
        max_cases=args.max_cases,
    )
    if not cases:
        raise RuntimeError("Parameter matrix produced no experiment cases")

    cases, source_selection = _select_model_input_compatible_sources(
        cases,
        input_root=args.input_root,
        glob_pattern=args.glob_pattern,
        model_input_contract=model_input_contract,
        run_root=run_root,
    )
    model_input_contract = dict(model_input_contract)
    model_input_contract["source_selection"] = source_selection
    save_json(run_root / "manifests/model_input_contract.json", model_input_contract)

    print(
        "[MODEL INPUT] source selection: "
        f"retargeted={source_selection['num_retargeted']} "
        f"cross_type_fallbacks={source_selection['num_cross_type_fallbacks']}"
    )

    pipeline = OccludedPedestrianPipeline(
        run_root,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        ollama_url=args.ollama_url,
        strict_check=True,
        save_visuals=not args.no_b1_visuals,
    )
    print(f"[1/5] B1 semantic editing: {len(cases)} parameterized cases")
    b1_summary = pipeline.run_batch(cases)
    accepted_ids = _accepted_sample_ids(run_root)
    if not accepted_ids:
        raise RuntimeError(
            "B1 produced zero accepted occluded-pedestrian scenes; "
            "diffusion is not allowed to start"
        )

    print(f"[2/5] Exporting {len(accepted_ids)} B1 simulator gzip caches")
    b1_export = export_b1_simulation_cache(
        run_root,
        args.config,
        limit=None,
        save_previews=not args.no_b1_previews,
    )

    print("[3/5] Persisting deterministic RVAE encode(mu)->decode checkpoint")
    rvae_summary = run_rvae_reconstruction(
        run_root=run_root,
        config=args.config,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        device=args.device,
        max_scenes=None,
        strict_protected=True,
    )

    print("[4/5] Running diffusion generation with semantic acceptance gate")
    if args.diffusion_run == "both":
        diffusion_summary = run_diffusion_comparison(
            run_root=run_root,
            config=args.config,
            autoencoder_checkpoint=args.autoencoder_checkpoint,
            diffusion_checkpoint=args.diffusion_checkpoint,
            device=args.device,
            max_scenes=None,
            repair_attempts=args.repair_attempts,
            save_visuals=args.save_diffusion_visuals,
        )
        raw_gzip = _finalize_diffusion_cache(
            run_root,
            RAW_DIFFUSION_BASELINE,
            require_semantic_pass=False,
        )
    else:
        diffusion_summary = run_half_denoise(
            run_root=run_root,
            config=args.config,
            autoencoder_checkpoint=args.autoencoder_checkpoint,
            diffusion_checkpoint=args.diffusion_checkpoint,
            device=args.device,
            diffusion_mode=SEMANTIC_PROTECTED,
            max_scenes=None,
            repair_attempts=args.repair_attempts,
            save_visuals=args.save_diffusion_visuals,
        )
        raw_gzip = None

    protected_gzip = _finalize_diffusion_cache(
        run_root,
        SEMANTIC_PROTECTED,
        require_semantic_pass=True,
    )

    print("[5/5] Verifying the three official simulator-readable gzip stages")
    contract = _build_three_stage_contract(
        run_root,
        accepted_ids,
    )
    payload = {
        "schema_version": "occluded_pedestrian_three_stage_run_v3_source_compatible",
        "profiles": profiles,
        "num_parameter_cases": len(cases),
        "num_accepted_b1": len(accepted_ids),
        "model_input_contract": model_input_contract,
        "b1": b1_summary,
        "b1_simulation_export": b1_export,
        "rvae": rvae_summary,
        "diffusion": diffusion_summary,
        "raw_diffusion_gzip_finalize": raw_gzip,
        "protected_diffusion_gzip_finalize": protected_gzip,
        "three_stage_contract": contract,
    }
    save_json(run_root / "manifests/three_stage_run_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _resolve_model_input_matrix(
    matrix: Mapping[str, Any],
    *,
    config: Path,
    run_root: Path,
    source_matrix: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Materialize model-compatible pedestrian speeds before B1 construction."""

    sledge_config = build_sledge_config(str(config))
    pedestrian_max_velocity = float(sledge_config.pedestrian_max_velocity)
    if pedestrian_max_velocity <= 0.0:
        raise ValueError(
            "Loaded SLEDGE config has non-positive pedestrian_max_velocity="
            f"{pedestrian_max_velocity}"
        )

    resolved = deepcopy(dict(matrix))
    profiles = resolved.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Experiment matrix must contain a non-empty profiles object")

    speed_profiles: Dict[str, Any] = {}
    for profile_name, raw_profile in profiles.items():
        if not isinstance(raw_profile, dict):
            raise TypeError(f"Profile {profile_name!r} must be a JSON object")
        profile = dict(raw_profile)
        fractions = profile.get("pedestrian_speed_fractions_of_model_max")
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
                speed = round(pedestrian_max_velocity * fraction, 3)
                if speed <= 0.0 or speed > pedestrian_max_velocity + 1e-9:
                    raise ValueError(
                        f"Resolved speed {speed} for profile {profile_name!r} "
                        "is outside the loaded model input range"
                    )
                normalized_fractions.append(fraction)
                concrete.append(speed)

            concrete = sorted(set(concrete))
            if not concrete:
                raise ValueError(f"Profile {profile_name!r} resolved to zero speeds")
            profile["pedestrian_speeds_mps"] = concrete
            profile["resolved_pedestrian_max_velocity_mps"] = pedestrian_max_velocity
            profile["resolved_speed_source"] = "fraction_of_loaded_model_max"
            profiles[profile_name] = profile
            speed_profiles[profile_name] = {
                "source": "fraction_of_loaded_model_max",
                "fractions": normalized_fractions,
                "resolved_speeds_mps": concrete,
            }
            continue

        if explicit_speeds is None:
            raise ValueError(
                f"Profile {profile_name!r} must define either "
                "pedestrian_speed_fractions_of_model_max or pedestrian_speeds_mps"
            )
        if not isinstance(explicit_speeds, list) or not explicit_speeds:
            raise ValueError(f"Profile {profile_name!r} pedestrian_speeds_mps is empty")

        concrete = sorted(set(float(value) for value in explicit_speeds))
        bad = [
            speed
            for speed in concrete
            if speed <= 0.0 or speed > pedestrian_max_velocity + 1e-9
        ]
        if bad:
            raise ValueError(
                f"Profile {profile_name!r} requests pedestrian speeds {bad} "
                f"outside loaded model limit {pedestrian_max_velocity:.3f} m/s. "
                "Use pedestrian_speed_fractions_of_model_max instead."
            )
        profile["pedestrian_speeds_mps"] = concrete
        profile["resolved_pedestrian_max_velocity_mps"] = pedestrian_max_velocity
        profile["resolved_speed_source"] = "explicit_model_compatible_speed"
        profiles[profile_name] = profile
        speed_profiles[profile_name] = {
            "source": "explicit_model_compatible_speed",
            "fractions": None,
            "resolved_speeds_mps": concrete,
        }

    resolved["profiles"] = profiles
    resolved["resolved_model_pedestrian_max_velocity_mps"] = pedestrian_max_velocity
    resolved["resolved_speed_policy"] = "model_input_compatible_fail_closed"

    manifest_dir = run_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = manifest_dir / "resolved_semantic_retention_matrix.json"
    save_json(resolved_path, resolved)

    contract = {
        "schema_version": "occluded_pedestrian_model_input_contract_v2_source_capacity",
        "source_config": str(Path(config).resolve()),
        "source_matrix": str(Path(source_matrix).resolve()),
        "resolved_matrix": str(resolved_path),
        "frame": list(sledge_config.frame),
        "num_vehicles": int(sledge_config.num_vehicles),
        "num_pedestrians": int(sledge_config.num_pedestrians),
        "num_static_objects": int(sledge_config.num_static_objects),
        "vehicle_max_velocity_mps": float(sledge_config.vehicle_max_velocity),
        "pedestrian_max_velocity_mps": pedestrian_max_velocity,
        "speed_policy": (
            "materialize_fraction_of_model_max_and_fail_on_out_of_range_explicit_speed"
        ),
        "processed_slot_policy": "standard_nearest_N_preprocessing_unchanged",
        "source_selection_policy": (
            "reserve_one_standard_preprocessing_slot_for_controlled_pedestrian_and_occluder"
        ),
        "speed_profiles": speed_profiles,
    }
    save_json(manifest_dir / "model_input_contract.json", contract)
    return resolved, contract


def _select_model_input_compatible_sources(
    cases: List[ExperimentCase],
    *,
    input_root: Path,
    glob_pattern: str,
    model_input_contract: Mapping[str, Any],
    run_root: Path,
) -> Tuple[List[ExperimentCase], Dict[str, Any]]:
    """Choose B0 scenes with guaranteed nearest-N capacity for the hazard.

    The current RVAE preprocessing keeps the nearest ``num_pedestrians`` and
    nearest ``num_vehicles`` entities, while static objects have their own fixed
    capacity.  A controlled object can therefore disappear even when it is
    inside the spatial frame.  We do not alter that trained preprocessing rule.

    Instead, each experiment condition is assigned a source B0 that has one
    free slot in the relevant collections *before* the controlled hazard is
    inserted.  This is conservative but gives an explicit guarantee that the
    B1 controlled pedestrian/occluder cannot be removed solely by capacity.
    """

    root = Path(input_root).resolve()
    frame = list(model_input_contract["frame"])
    capacities = {
        "pedestrians": int(model_input_contract["num_pedestrians"]),
        "vehicles": int(model_input_contract["num_vehicles"]),
        "static_objects": int(model_input_contract["num_static_objects"]),
    }
    if capacities["pedestrians"] < 1:
        raise ValueError("Model input has no pedestrian capacity")

    all_sources = sorted(Path(input_root).glob(glob_pattern))
    if not all_sources:
        raise FileNotFoundError(
            f"No source scenes matched {glob_pattern!r} under {input_root}"
        )

    by_type: Dict[str, List[Path]] = {}
    for path in all_sources:
        rel = path.resolve().relative_to(root)
        source_type = _source_scenario_type_from_relative(rel)
        by_type.setdefault(source_type, []).append(path.resolve())

    cache: Dict[str, Dict[str, Any]] = {}
    used: set[str] = set()
    selected_cases: List[ExperimentCase] = []
    selection_rows: List[Dict[str, Any]] = []

    for case in cases:
        occ_element = element_name_for_occluder(case.occluder_type or "vehicle")
        required = {
            "pedestrians": capacities["pedestrians"] - 1,
            occ_element: capacities[occ_element] - 1,
        }

        preferred = list(by_type.get(case.source_scenario_type, []))
        fallback = [
            path
            for source_type, paths in sorted(by_type.items())
            if source_type != case.source_scenario_type
            for path in paths
        ]

        def _ordered(paths: List[Path], salt: str) -> List[Path]:
            current = Path(case.input_raw).resolve()
            unique = list(dict.fromkeys(paths))
            unique.sort(
                key=lambda path: hashlib.sha1(
                    f"{case.condition_id}|{salt}|{path}".encode("utf-8")
                ).hexdigest()
            )
            if current in unique:
                unique.remove(current)
                unique.insert(0, current)
            return unique

        chosen: Optional[Path] = None
        chosen_info: Optional[Dict[str, Any]] = None
        cross_type_fallback = False

        for candidate_group, is_fallback in (
            (_ordered(preferred, "same_type"), False),
            (_ordered(fallback, "cross_type"), True),
        ):
            for candidate in candidate_group:
                key = str(candidate)
                if key in used:
                    continue
                info = cache.get(key)
                if info is None:
                    info = _source_capacity_info(candidate, frame)
                    cache[key] = info
                if not bool(info.get("readable", False)):
                    continue
                if int(info["counts"]["pedestrians"]) > required["pedestrians"]:
                    continue
                if int(info["counts"][occ_element]) > required[occ_element]:
                    continue
                chosen = candidate
                chosen_info = info
                cross_type_fallback = is_fallback
                break
            if chosen is not None:
                break

        if chosen is None or chosen_info is None:
            raise RuntimeError(
                "Could not find a standard-preprocessing-capacity-compatible B0 "
                f"for condition={case.condition_id!r}. Required in-frame counts: "
                f"pedestrians<={required['pedestrians']}, "
                f"{occ_element}<={required[occ_element]}. "
                "The experiment will not change RVAE nearest-N preprocessing."
            )

        used.add(str(chosen))
        rel = chosen.relative_to(root)
        actual_source_type = _source_scenario_type_from_relative(rel)
        token = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:8]
        sample_prefix = case.sample_id.rsplit("__", 1)[0]
        selected_case = replace(
            case,
            sample_id=f"{sample_prefix}__{token}",
            input_raw=str(chosen),
            source_relative_path=str(rel),
            source_scenario_type=actual_source_type,
        )
        selected_cases.append(selected_case)

        original_path = Path(case.input_raw).resolve()
        selection_rows.append(
            {
                "condition_id": case.condition_id,
                "original_sample_id": case.sample_id,
                "selected_sample_id": selected_case.sample_id,
                "original_input_raw": str(original_path),
                "selected_input_raw": str(chosen),
                "original_source_scenario_type": case.source_scenario_type,
                "selected_source_scenario_type": actual_source_type,
                "retargeted": chosen != original_path,
                "cross_type_fallback": cross_type_fallback,
                "occluder_element": occ_element,
                "in_frame_counts": dict(chosen_info["counts"]),
                "capacity_limits_before_hazard": required,
                "model_frame": frame,
            }
        )

    payload = {
        "schema_version": "occluded_pedestrian_model_input_source_selection_v1",
        "policy": (
            "standard_preprocessing_unchanged; reserve one capacity slot for "
            "controlled pedestrian and one for controlled occluder"
        ),
        "num_cases": len(selected_cases),
        "num_retargeted": sum(bool(row["retargeted"]) for row in selection_rows),
        "num_cross_type_fallbacks": sum(
            bool(row["cross_type_fallback"]) for row in selection_rows
        ),
        "rows": selection_rows,
    }
    save_json(run_root / "manifests/model_input_source_selection.json", payload)
    return selected_cases, payload


def _source_capacity_info(path: Path, frame: List[float]) -> Dict[str, Any]:
    try:
        scene, _ = load_raw_scene(path)
        counts = {
            "pedestrians": _count_states_in_frame(scene.pedestrians.states, frame),
            "vehicles": _count_states_in_frame(scene.vehicles.states, frame),
            "static_objects": _count_states_in_frame(scene.static_objects.states, frame),
        }
        return {"readable": True, "counts": counts}
    except Exception as exc:
        return {
            "readable": False,
            "counts": {"pedestrians": 10**9, "vehicles": 10**9, "static_objects": 10**9},
            "error": f"{type(exc).__name__}: {exc}",
        }


def _count_states_in_frame(states: Any, frame: List[float]) -> int:
    arr = np.asarray(states)
    if arr.size == 0:
        return 0
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] < 2:
        return 0
    return int(np.count_nonzero(coords_in_frame(arr[..., :2], frame)))


def _source_scenario_type_from_relative(rel: Path) -> str:
    parts = rel.parts
    if len(parts) >= 3:
        return str(parts[-3])
    if len(parts) >= 2:
        return str(parts[-2])
    return "unknown"


def _build_profile_cases(
    *,
    input_root: Path,
    matrix: Dict[str, Any],
    profiles: List[str],
    glob_pattern: str,
    max_cases: Optional[int],
) -> List[ExperimentCase]:
    """Interleave profiles so early smoke limits still cover multiple risks."""

    grouped: List[List[ExperimentCase]] = []
    for profile in profiles:
        rows = build_experiment_cases(
            input_root=input_root,
            profile=profile,
            matrix_config=matrix,
            glob_pattern=glob_pattern,
            max_cases=None,
        )
        grouped.append(
            [
                replace(
                    case,
                    sample_id=f"{profile}__{case.sample_id}",
                    condition_id=f"{profile}__{case.condition_id}",
                )
                for case in rows
            ]
        )

    output: List[ExperimentCase] = []
    max_len = max((len(rows) for rows in grouped), default=0)
    for index in range(max_len):
        for rows in grouped:
            if index < len(rows):
                output.append(rows[index])
                if max_cases is not None and len(output) >= int(max_cases):
                    return output
    return output


def _accepted_sample_ids(run_root: Path) -> List[str]:
    output: List[str] = []
    for label_path in sorted(
        (run_root / "b1_edited_cache").glob("*/scenario_label.json")
    ):
        label = _read_json(label_path)
        if bool(label.get("accepted", False)):
            output.append(str(label.get("sample_id", label_path.parent.name)))
    return output


def _finalize_diffusion_cache(
    run_root: Path,
    mode: str,
    *,
    require_semantic_pass: bool,
) -> Dict[str, Any]:
    """Reattach visible object type metadata and verify SledgeScenario round-trip.

    The learned SLEDGE vector stores geometry in vehicle/static slots but does
    not intrinsically encode every nuPlan subtype (e.g. BICYCLE). B1 already
    stores that subtype as gzip metadata. Reattaching the same metadata after
    geometry generation changes no diffusion geometry and makes visualization
    faithful to the requested occluder type.
    """

    layout = RunLayout(Path(run_root).resolve())
    cache_root = layout.b2_cache_for(mode)
    metrics_by_sample = {
        str(row["sample_id"]): row
        for row in _read_jsonl_if_exists(
            layout.manifests / f"b2_{mode}_results.jsonl"
        )
        if row.get("sample_id")
    }
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for label_path in sorted(cache_root.glob("**/scenario_label.json")):
        label = _read_json(label_path)
        sample_id = str(label.get("sample_id", ""))
        if not sample_id:
            edited = Path(str(label.get("edited_scene_path", "")))
            sample_id = edited.parent.name
        if not sample_id:
            continue
        b1_label_path = layout.b1_cache / sample_id / "scenario_label.json"
        if not b1_label_path.exists():
            continue
        b1_label = _read_json(b1_label_path)
        tracked_type_name = str(
            b1_label.get("occluder_tracked_object_type", "VEHICLE")
        )
        metrics = metrics_by_sample.get(sample_id, {})

        if mode == SEMANTIC_PROTECTED:
            protected = dict(label.get("protected_slots", {}) or {})
            element = str(protected.get("occluder_element", ""))
            index = int(protected.get("occluder_index", -1))
        else:
            occluder = dict(metrics.get("occluder", {}) or {})
            element = str(occluder.get("element", ""))
            index = int(occluder.get("index", -1))

        overrides: Dict[str, Dict[str, str]] = {}
        if element in {"vehicles", "static_objects"} and index >= 0:
            overrides = make_type_override(element, index, tracked_type_name)

        vector_path = label_path.parent / "sledge_vector.gz"
        try:
            if not vector_path.exists():
                raise FileNotFoundError(vector_path)
            if overrides:
                embed_type_overrides(vector_path, overrides)
            _assert_sledge_round_trip(label_path.parent / "sledge_vector", overrides)
            semantic_pass = bool(metrics.get("overall_pass", False))
            if require_semantic_pass and not semantic_pass:
                raise RuntimeError("canonical B2 semantic metrics did not pass")
            label.update(
                {
                    "sample_id": sample_id,
                    "object_type_overrides": overrides,
                    "object_type_metadata_source": "B1_semantic_intent",
                    "gzip_round_trip_pass": True,
                    "canonical_semantic_pass": semantic_pass,
                }
            )
            save_json(label_path, label)
            rows.append(
                {
                    "sample_id": sample_id,
                    "vector_gz": str(vector_path),
                    "semantic_pass": semantic_pass,
                    "gzip_round_trip_pass": True,
                    "object_type_overrides": overrides,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "sample_id": sample_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    payload = {
        "schema_version": "occluded_pedestrian_diffusion_gzip_finalize_v1",
        "diffusion_mode": mode,
        "cache_root": str(cache_root),
        "num_validated": len(rows),
        "num_failures": len(failures),
        "rows": rows,
        "failures": failures,
    }
    save_json(layout.manifests / f"b2_{mode}_gzip_finalize.json", payload)
    if require_semantic_pass and failures:
        raise RuntimeError(
            "Protected B2 gzip/semantic finalization failed for "
            f"{[row['sample_id'] for row in failures]}"
        )
    return payload


def _build_three_stage_contract(
    run_root: Path,
    accepted_ids: Iterable[str],
) -> Dict[str, Any]:
    """Require B1, protected RVAE and protected diffusion gzip for every B1 pass."""

    run_root = Path(run_root).resolve()
    layout = RunLayout(run_root)
    b1_by_sample = _index_cache_labels(run_root / "b1_simulation_cache")
    rvae_raw_by_sample = _index_cache_labels(
        run_root / "rvae_reconstruction/raw_cache"
    )
    rvae_protected_by_sample = _index_cache_labels(
        run_root / "rvae_reconstruction/semantic_protected_cache"
    )
    b2_raw_by_sample = _index_cache_labels(
        layout.b2_cache_for(RAW_DIFFUSION_BASELINE)
    )
    b2_protected_by_sample = _index_cache_labels(
        layout.b2_cache_for(SEMANTIC_PROTECTED)
    )
    b2_metrics = {
        str(row["sample_id"]): row
        for row in _read_jsonl_if_exists(
            layout.manifests / f"b2_{SEMANTIC_PROTECTED}_results.jsonl"
        )
        if row.get("sample_id")
    }

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for sample_id in accepted_ids:
        b1_label = b1_by_sample.get(sample_id)
        rvae_protected_label = rvae_protected_by_sample.get(sample_id)
        b2_protected_label = b2_protected_by_sample.get(sample_id)
        missing = [
            name
            for name, value in (
                ("B1", b1_label),
                ("RVAE_PROTECTED", rvae_protected_label),
                ("B2_PROTECTED", b2_protected_label),
            )
            if value is None
        ]
        if missing:
            failures.append(
                {
                    "sample_id": sample_id,
                    "error": f"missing official cache stages: {missing}",
                }
            )
            continue

        b1_data = _read_json(b1_label)
        rvae_data = _read_json(rvae_protected_label)
        b2_data = _read_json(b2_protected_label)
        b2_metric = b2_metrics.get(sample_id, {})
        checks = {
            "b1_gzip_round_trip": bool(b1_data.get("gzip_round_trip_pass", False)),
            "rvae_semantic_pass": bool(rvae_data.get("semantic_pass", False)),
            "rvae_gzip_round_trip": bool(
                rvae_data.get("gzip_round_trip_pass", False)
            ),
            "b2_semantic_pass": bool(b2_metric.get("overall_pass", False)),
            "b2_gzip_round_trip": bool(b2_data.get("gzip_round_trip_pass", False)),
        }
        official_pass = all(checks.values())
        row = {
            "sample_id": sample_id,
            "official_pass": official_pass,
            "checks": checks,
            "B1_edited_gz": str(b1_label.parent / "sledge_vector.gz"),
            "RVAE_raw_gz": (
                str(rvae_raw_by_sample[sample_id].parent / "sledge_vector.gz")
                if sample_id in rvae_raw_by_sample
                else None
            ),
            "RVAE_semantic_protected_gz": str(
                rvae_protected_label.parent / "sledge_vector.gz"
            ),
            "B2_raw_diffusion_gz": (
                str(b2_raw_by_sample[sample_id].parent / "sledge_vector.gz")
                if sample_id in b2_raw_by_sample
                else None
            ),
            "B2_semantic_protected_gz": str(
                b2_protected_label.parent / "sledge_vector.gz"
            ),
        }
        rows.append(row)
        if not official_pass:
            failures.append(
                {
                    "sample_id": sample_id,
                    "error": "official semantic/gzip contract failed",
                    "checks": checks,
                }
            )

    accepted_list = list(accepted_ids) if not isinstance(accepted_ids, list) else accepted_ids
    payload = {
        "schema_version": "occluded_pedestrian_three_stage_contract_v1",
        "num_accepted_b1": len(accepted_list),
        "num_complete": len(rows),
        "num_official_pass": sum(bool(row["official_pass"]) for row in rows),
        "num_failures": len(failures),
        "official_stages": [
            "B1_EDITED_SIMULATION_GZ",
            "RVAE_SEMANTIC_PROTECTED_GZ",
            "B2_SEMANTIC_PROTECTED_DIFFUSION_GZ",
        ],
        "diagnostic_stages": ["RVAE_RAW_GZ", "B2_RAW_DIFFUSION_GZ"],
        "rows": rows,
        "failures": failures,
    }
    save_json(run_root / "manifests/three_stage_gz_contract.json", payload)
    if failures:
        raise RuntimeError(
            "Three-stage semantic/gzip contract failed for "
            f"{[row['sample_id'] for row in failures]}"
        )
    return payload


def _index_cache_labels(cache_root: Path) -> Dict[str, Path]:
    output: Dict[str, Path] = {}
    for label_path in Path(cache_root).glob("**/scenario_label.json"):
        label = _read_json(label_path)
        sample_id = str(label.get("sample_id", ""))
        if not sample_id:
            edited = Path(str(label.get("edited_scene_path", "")))
            sample_id = edited.parent.name
        if sample_id:
            output[sample_id] = label_path
    return output


def _assert_sledge_round_trip(
    cache_base: Path,
    overrides: Mapping[str, Mapping[str, str]],
) -> None:
    scenario = SledgeScenario(cache_base)
    detections = scenario.initial_tracked_objects.tracked_objects
    by_token = {obj.track_token: obj for obj in detections}
    for entries in overrides.values():
        for index_text, type_name in entries.items():
            expected_type = TrackedObjectType[str(type_name)]
            token = f"{expected_type.value}_{int(index_text)}"
            obj = by_token.get(token)
            if obj is None or obj.tracked_object_type != expected_type:
                observed = [
                    (item.track_token, item.tracked_object_type.name)
                    for item in detections
                ]
                raise RuntimeError(
                    f"Expected typed object {(token, expected_type.name)}, "
                    f"observed={observed}"
                )


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl_if_exists(path: Path) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


if __name__ == "__main__":
    main()
