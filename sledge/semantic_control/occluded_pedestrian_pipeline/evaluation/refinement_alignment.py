"""Occlusion-aware alignment adapter for the existing half-denoise engine."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from sledge.semantic_control.generation.legacy.evaluators.crossing_evaluator import (
    PromptAlignmentResult,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.metrics import (
    evaluate_occluded_pedestrian_scene,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.language.eventframe_adapter import (
    OccludedPedestrianEventFrameAdapter,
)


class OccludedPedestrianRefinementAlignmentEvaluator:
    """Expose strict occlusion semantics through the legacy alignment interface."""

    def __init__(self, projection_time_s: float = 2.1) -> None:
        self.projection_time_s = float(projection_time_s)
        self.adapter = OccludedPedestrianEventFrameAdapter(llm_provider="none")
        self._spec_cache: Dict[str, Any] = {}
        self.preferred_pedestrian_index = None
        self.preferred_occluder_index = None
        self.preferred_occluder_elem_name = "vehicles"
        self.lane_center_y = 0.0
        self.reference_scene = None

    def set_preferred_slots(self, pedestrian_index: int, occluder_index: int, occluder_elem_name: str) -> None:
        self.preferred_pedestrian_index = pedestrian_index if pedestrian_index >= 0 else None
        self.preferred_occluder_index = occluder_index if occluder_index >= 0 else None
        self.preferred_occluder_elem_name = occluder_elem_name

    def set_reference_scene(self, scene: Any) -> None:
        """Set the B1 vector whose road topology B2 must preserve."""
        self.reference_scene = scene

    def set_lane_center_y(self, lane_center_y: float) -> None:
        self.lane_center_y = float(
            lane_center_y
        )

    def evaluate(self, sledge_vector: Any, prompt_spec: Any = None) -> PromptAlignmentResult:
        prompt = str(getattr(prompt_spec, "raw_prompt", "") or getattr(prompt_spec, "normalized_prompt", ""))
        if prompt not in self._spec_cache:
            self._spec_cache[prompt] = self.adapter.adapt(prompt).hazard_spec
        metrics = evaluate_occluded_pedestrian_scene(
            sledge_vector,
            self._spec_cache[prompt],
            preferred_pedestrian_index=self.preferred_pedestrian_index,
            preferred_occluder_index=self.preferred_occluder_index,
            preferred_occluder_elem_name=self.preferred_occluder_elem_name,
            projection_time_s=self.projection_time_s,
            lane_center_y=self.lane_center_y,
        )
        checks = metrics.get("checks", {})
        details = {
            "pedestrian_presence_score": float(bool(checks.get("pedestrian_exists", False))),
            "roadside_emergence_score": float(
                bool(checks.get("occluder_exists", False))
                and bool(checks.get("occluder_between_ego_and_actor", False))
                and bool(checks.get("line_of_sight_occlusion", False))
            ),
            "crossing_direction_score": float(
                bool(checks.get("direction_match", False)) and bool(checks.get("speed_match", False))
            ),
            "ego_lane_conflict_score": float(bool(checks.get("crossing_reaches_ego_lane", False))),
            "immediacy_score": float(bool(checks.get("interaction_timing_match", False))),
        }
        failed = [name for name, passed in checks.items() if not passed]
        topology = evaluate_road_topology_preservation(self.reference_scene, sledge_vector)
        if not topology["passed"]:
            failed.append("road_topology_preservation")
        semantic_total = float(metrics.get("semantic_satisfaction_rate", 0.0))
        total = semantic_total if topology["passed"] else 0.0
        details["road_topology_score"] = float(topology["score"])
        return PromptAlignmentResult(
            total=total,
            details=details,
            notes=[
                "occluded-pedestrian strict checks: " + (", ".join(failed) if failed else "all passed"),
                "road topology: "
                f"source_to_generated={topology['source_to_generated_mean_m']:.3f}m, "
                f"generated_to_source={topology['generated_to_source_mean_m']:.3f}m, "
                f"source_p95={topology['source_to_generated_p95_m']:.3f}m, "
                f"generated_p95={topology['generated_to_source_p95_m']:.3f}m, "
                f"line_ratio={topology['line_point_ratio']:.3f}",
            ],
            accepted=bool(metrics.get("overall_pass", False) and topology["passed"]),
        )


def evaluate_road_topology_preservation(reference_scene: Any, candidate_scene: Any) -> Dict[str, Any]:
    """Symmetric nearest-line distance rejects tangled diffusion road graphs."""
    if reference_scene is None:
        return {
            "passed": True,
            "score": 1.0,
            "source_to_generated_mean_m": 0.0,
            "generated_to_source_mean_m": 0.0,
            "source_to_generated_p95_m": 0.0,
            "generated_to_source_p95_m": 0.0,
            "line_point_ratio": 1.0,
        }
    source = _valid_line_points(reference_scene)
    candidate = _valid_line_points(candidate_scene)
    if not len(source) or not len(candidate):
        return {
            "passed": False,
            "score": 0.0,
            "source_to_generated_mean_m": float("inf"),
            "generated_to_source_mean_m": float("inf"),
            "source_to_generated_p95_m": float("inf"),
            "generated_to_source_p95_m": float("inf"),
            "line_point_ratio": 0.0,
        }
    # At most 1000x1000 points for the configured vector representation.
    distances = np.linalg.norm(source[:, None, :] - candidate[None, :, :], axis=-1)
    source_distances = np.min(distances, axis=1)
    candidate_distances = np.min(distances, axis=0)
    source_to_generated = float(source_distances.mean())
    generated_to_source = float(candidate_distances.mean())
    source_to_generated_p95 = float(np.percentile(source_distances, 95))
    generated_to_source_p95 = float(np.percentile(candidate_distances, 95))
    ratio = float(len(candidate) / len(source))
    passed = (
        source_to_generated <= 3.0
        and generated_to_source <= 2.0
        and source_to_generated_p95 <= 3.5
        and generated_to_source_p95 <= 3.0
        and 0.55 <= ratio <= 1.45
    )
    normalized_error = 0.5 * (source_to_generated / 3.0 + generated_to_source / 2.0)
    score = float(max(0.0, 1.0 - normalized_error))
    return {
        "passed": bool(passed),
        "score": score,
        "source_to_generated_mean_m": source_to_generated,
        "generated_to_source_mean_m": generated_to_source,
        "source_to_generated_p95_m": source_to_generated_p95,
        "generated_to_source_p95_m": generated_to_source_p95,
        "line_point_ratio": ratio,
    }


def _valid_line_points(scene: Any) -> np.ndarray:
    states = np.asarray(scene.lines.states, dtype=np.float32)
    masks = np.asarray(scene.lines.mask).reshape(-1) >= 0.3
    if states.ndim != 3 or not np.any(masks):
        return np.zeros((0, 2), dtype=np.float32)
    return states[masks, :, :2].reshape(-1, 2)
