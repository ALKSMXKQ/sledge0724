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
    """Expose the canonical occlusion semantics through the legacy interface.

    Two contracts are deliberately separated:

    * hazard semantics are evaluated against the *effective B1 HazardSemanticSpec*
      whenever the runner can recover it from the B1 artifact;
    * B1-vs-B2 road proximity is always measured for diagnostics, but is only a
      hard acceptance gate when the experiment explicitly copies/protects B1
      road geometry.

    This separation is essential for ``generated_with_guidance`` experiments:
    the generated road is allowed to differ from B1, so exact/near-exact B1 road
    preservation cannot also be used as the semantic acceptance criterion.
    """

    def __init__(self, projection_time_s: float = 2.1) -> None:
        self.projection_time_s = float(projection_time_s)
        self.adapter = OccludedPedestrianEventFrameAdapter(llm_provider="none")
        self._spec_cache: Dict[str, Any] = {}
        self.reference_spec: Any = None
        self.preferred_pedestrian_index = None
        self.preferred_occluder_index = None
        self.preferred_occluder_elem_name = "vehicles"
        self.lane_center_y = 0.0
        self.reference_scene = None
        self.topology_gate_enabled = True

    def set_preferred_slots(
        self,
        pedestrian_index: int,
        occluder_index: int,
        occluder_elem_name: str,
    ) -> None:
        self.preferred_pedestrian_index = pedestrian_index if pedestrian_index >= 0 else None
        self.preferred_occluder_index = occluder_index if occluder_index >= 0 else None
        self.preferred_occluder_elem_name = occluder_elem_name

    def set_reference_scene(self, scene: Any) -> None:
        """Set B1 road geometry used for diagnostic B1-vs-B2 comparison."""
        self.reference_scene = scene

    def set_reference_spec(self, spec: Any) -> None:
        """Set the exact effective B1 hazard spec; ``None`` enables fallback parsing."""
        self.reference_spec = spec

    def set_projection_time_s(self, projection_time_s: float) -> None:
        self.projection_time_s = float(projection_time_s)

    def set_lane_center_y(self, lane_center_y: float) -> None:
        self.lane_center_y = float(lane_center_y)

    def set_topology_gate_enabled(self, enabled: bool) -> None:
        self.topology_gate_enabled = bool(enabled)

    def evaluate(self, sledge_vector: Any, prompt_spec: Any = None) -> PromptAlignmentResult:
        prompt = str(
            getattr(prompt_spec, "raw_prompt", "")
            or getattr(prompt_spec, "normalized_prompt", "")
        )
        spec = self.reference_spec
        semantic_contract_source = "effective_b1_hazard_spec"
        if spec is None:
            semantic_contract_source = "prompt_reparsed_fallback"
            if prompt not in self._spec_cache:
                self._spec_cache[prompt] = self.adapter.adapt(prompt).hazard_spec
            spec = self._spec_cache[prompt]

        metrics = evaluate_occluded_pedestrian_scene(
            sledge_vector,
            spec,
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
                bool(checks.get("direction_match", False))
                and bool(checks.get("speed_match", False))
            ),
            "ego_lane_conflict_score": float(
                bool(checks.get("crossing_reaches_ego_lane", False))
            ),
            "immediacy_score": float(
                bool(checks.get("interaction_timing_match", False))
            ),
        }

        failed = [name for name, passed in checks.items() if not passed]
        topology = evaluate_road_topology_preservation(
            self.reference_scene,
            sledge_vector,
        )
        topology_gate_pass = bool(topology["passed"]) if self.topology_gate_enabled else True
        if self.topology_gate_enabled and not topology["passed"]:
            failed.append("road_topology_preservation")

        semantic_total = float(metrics.get("semantic_satisfaction_rate", 0.0))
        total = semantic_total if topology_gate_pass else 0.0
        details["road_topology_score"] = float(topology["score"])
        details["road_topology_pass"] = float(bool(topology["passed"]))
        details["road_topology_gate_enabled"] = float(self.topology_gate_enabled)

        topology_role = "hard_gate" if self.topology_gate_enabled else "diagnostic_only"
        return PromptAlignmentResult(
            total=total,
            details=details,
            notes=[
                "occluded-pedestrian strict checks: "
                + (", ".join(failed) if failed else "all passed"),
                f"semantic contract: {semantic_contract_source}; "
                f"projection_time_s={self.projection_time_s:.3f}; "
                f"lane_center_y={self.lane_center_y:.3f}",
                f"road topology ({topology_role}): "
                f"source_to_generated={topology['source_to_generated_mean_m']:.3f}m, "
                f"generated_to_source={topology['generated_to_source_mean_m']:.3f}m, "
                f"source_p95={topology['source_to_generated_p95_m']:.3f}m, "
                f"generated_p95={topology['generated_to_source_p95_m']:.3f}m, "
                f"line_ratio={topology['line_point_ratio']:.3f}",
            ],
            accepted=bool(metrics.get("overall_pass", False) and topology_gate_pass),
        )


def evaluate_road_topology_preservation(
    reference_scene: Any,
    candidate_scene: Any,
) -> Dict[str, Any]:
    """Symmetric nearest-line distance used as a diagnostic/legacy road gate."""
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
    normalized_error = 0.5 * (
        source_to_generated / 3.0 + generated_to_source / 2.0
    )
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
