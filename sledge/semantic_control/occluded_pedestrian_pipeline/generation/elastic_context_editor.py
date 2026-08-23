"""Road-fixed elastic B0 editing for occluded-pedestrian construction.

The simplified EDIT_EXISTING policy is intentionally asymmetric:

* road/lane geometry is immutable;
* ego, target pedestrian and occluder are semantic controls;
* unrelated background vehicles/pedestrians/static objects are elastic;
* exact local blocker handling happens inside ``HazardClearancePrimitiveOps``:
  relocate first, otherwise delete;
* hard hazard semantics are never weakened to obtain a higher success rate.

This module therefore owns high-level retries (ego + hazard envelope) and audit
logging, while low-level nuisance-entity clearance is performed exactly where
the occluder candidate is known.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    EgoIndex,
    SledgeVectorElement,
    SledgeVectorRaw,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_spec import (
    HazardSemanticSpec,
)


@dataclass(frozen=True)
class ElasticContextPolicy:
    """High-level retry policy for road-fixed semantic editing.

    Background clearance itself is not guessed here anymore. Every editor
    attempt uses the candidate-aware move-then-delete policy implemented by
    ``HazardClearancePrimitiveOps``.
    """

    lateral_gap_floor_m: Tuple[float, ...] = (0.0, 0.0, 0.0)
    longitudinal_extension_m: Tuple[float, ...] = (0.0, 6.0, 12.0)
    min_ego_speed_mps: float = 2.5
    max_ego_speed_mps: float = 15.0
    allow_background_reposition: bool = True
    allow_background_removal: bool = True
    background_policy: str = "move_then_delete_local_blockers"

    @property
    def num_attempts(self) -> int:
        return max(
            len(self.lateral_gap_floor_m),
            len(self.longitudinal_extension_m),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ElasticContextHazardConstructor:
    """Construct one EDIT_EXISTING B1 while freezing source road/lane lines."""

    def __init__(self, policy: Optional[ElasticContextPolicy] = None) -> None:
        self.policy = policy or ElasticContextPolicy()

    def construct(
        self,
        scene: SledgeVectorRaw,
        spec: HazardSemanticSpec,
        sampled_parameters: Mapping[str, Any],
        editor: Any,
    ) -> Tuple[
        SledgeVectorRaw,
        Any,
        Dict[str, Any],
        Dict[str, Any],
        HazardSemanticSpec,
    ]:
        """Return the first metric-valid B1, or the best executable B1.

        Every attempt starts from the same B0. The only pre-editor changes are
        semantic ego control and a bounded hazard-geometry envelope. Once the
        editor finds a hard-valid occluder candidate, nuisance background actors
        in that candidate's reserved region are moved; if relocation fails they
        are deleted. The resulting actions are pulled from ``editor_report`` and
        stored in ``context_edit_report.json``.
        """

        source_road = self._snapshot_element(scene.lines)
        attempt_reports: List[Dict[str, Any]] = []
        best: Optional[
            Tuple[
                float,
                SledgeVectorRaw,
                Any,
                Dict[str, Any],
                HazardSemanticSpec,
                Dict[str, Any],
            ]
        ] = None

        for attempt_index in range(self.policy.num_attempts):
            working = deepcopy(scene)
            attempt_spec = deepcopy(spec)
            preparation = self.prepare_scene_for_attempt(
                working,
                attempt_spec,
                sampled_parameters,
                attempt_index=attempt_index,
            )

            try:
                edited_scene, edit_result, editor_report = editor.edit(
                    working,
                    attempt_spec,
                )
            except Exception as exc:
                attempt_reports.append(
                    {
                        **preparation,
                        "editor_finished": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            if not self._element_matches_snapshot(edited_scene.lines, source_road):
                raise RuntimeError(
                    "EDIT_EXISTING violated immutable road/lane geometry: "
                    "scene.lines changed during hazard construction"
                )

            background_edits = self._extract_background_edits(editor_report)
            removal_count = sum(
                1 for row in background_edits if row.get("operation") == "delete"
            )
            reposition_count = sum(
                1
                for row in background_edits
                if row.get("operation") == "reposition"
            )

            projection_time_s = float(
                editor_report.get("extra", {}).get(
                    "semantic_validation_time_offset_s",
                    0.0,
                )
            )
            # Lazy import avoids the evaluation -> generation package cycle.
            from sledge.semantic_control.occluded_pedestrian_pipeline.evaluation.metrics import (
                evaluate_occluded_pedestrian_scene,
            )

            metrics = evaluate_occluded_pedestrian_scene(
                edited_scene,
                attempt_spec,
                preferred_pedestrian_index=getattr(
                    edit_result,
                    "pedestrian_index",
                    None,
                ),
                preferred_occluder_index=getattr(
                    edit_result,
                    "occluder_index",
                    None,
                ),
                preferred_occluder_elem_name=str(
                    getattr(edit_result, "occluder_elem_name", "vehicles")
                ),
                projection_time_s=projection_time_s,
                lane_center_y=float(
                    editor_report.get("extra", {}).get(
                        "conflict_lane_y",
                        0.0,
                    )
                ),
            )
            score = float(metrics.get("semantic_satisfaction_rate", 0.0))
            report = {
                **preparation,
                "editor_finished": True,
                "hard_road_lane_preserved": True,
                "background_actor_edits": background_edits,
                "background_actor_edit_count": len(background_edits),
                "background_reposition_count": int(reposition_count),
                "background_removal_count": int(removal_count),
                "removal_used": bool(removal_count > 0),
                "b1_metrics_preview": metrics,
                "preview_pass": bool(metrics.get("overall_pass", False)),
            }
            attempt_reports.append(report)

            if best is None or score > best[0]:
                best = (
                    score,
                    edited_scene,
                    edit_result,
                    editor_report,
                    attempt_spec,
                    report,
                )

            if metrics.get("overall_pass", False):
                context_report = self._final_report(
                    sampled_parameters=sampled_parameters,
                    attempt_reports=attempt_reports,
                    selected_attempt=attempt_index,
                    selected_report=report,
                    accepted_by_preview=True,
                )
                editor_report = dict(editor_report)
                editor_report["elastic_context"] = context_report
                return (
                    edited_scene,
                    edit_result,
                    editor_report,
                    context_report,
                    attempt_spec,
                )

        if best is None:
            error_lines = [
                f"attempt={row.get('attempt_index')}: "
                f"{row.get('error', 'unknown failure')}"
                for row in attempt_reports
            ]
            raise RuntimeError(
                "Elastic EDIT_EXISTING failed on every attempt. "
                + " | ".join(error_lines)
            )

        score, edited_scene, edit_result, editor_report, best_spec, selected_report = best
        selected_attempt = int(selected_report["attempt_index"])
        context_report = self._final_report(
            sampled_parameters=sampled_parameters,
            attempt_reports=attempt_reports,
            selected_attempt=selected_attempt,
            selected_report=selected_report,
            accepted_by_preview=False,
        )
        context_report["best_semantic_satisfaction_rate"] = float(score)
        editor_report = dict(editor_report)
        editor_report["elastic_context"] = context_report
        return (
            edited_scene,
            edit_result,
            editor_report,
            context_report,
            best_spec,
        )

    def prepare_scene_for_attempt(
        self,
        scene: SledgeVectorRaw,
        spec: HazardSemanticSpec,
        sampled_parameters: Mapping[str, Any],
        *,
        attempt_index: int,
    ) -> Dict[str, Any]:
        """Apply high-level semantic controls without guessing background blockers."""

        attempt_index = max(0, int(attempt_index))
        lateral_floor = float(
            self._schedule_value(
                self.policy.lateral_gap_floor_m,
                attempt_index,
            )
        )
        longitudinal_extension = float(
            self._schedule_value(
                self.policy.longitudinal_extension_m,
                attempt_index,
            )
        )

        ego_edit = self._apply_semantic_ego_state(scene, sampled_parameters)
        geometry_edit = self._tune_hazard_geometry(
            spec,
            lateral_gap_floor_m=lateral_floor,
            longitudinal_extension_m=longitudinal_extension,
        )
        execution_hints = self._inject_execution_hints(
            spec,
            sampled_parameters,
        )

        return {
            "attempt_index": int(attempt_index),
            "policy": self.policy.to_dict(),
            "ego_edit": ego_edit,
            "geometry_edit": geometry_edit,
            "execution_hints": execution_hints,
            # Background is intentionally untouched before the exact occluder
            # candidate is known.
            "background_actor_edits": [],
            "background_actor_edit_count": 0,
            "background_reposition_count": 0,
            "background_removal_count": 0,
            "removal_used": False,
        }

    def _apply_semantic_ego_state(
        self,
        scene: SledgeVectorRaw,
        sampled: Mapping[str, Any],
    ) -> Dict[str, Any]:
        before = self._ego_state_payload(scene)
        requested_speed = self._float(
            sampled.get("ego_speed_mps"),
            default=max(before["speed_mps"], self.policy.min_ego_speed_mps),
        )
        requested_accel = self._float(
            sampled.get("ego_acceleration_mps2"),
            default=before["acceleration_mps2"],
        )
        speed = float(
            np.clip(
                requested_speed,
                self.policy.min_ego_speed_mps,
                self.policy.max_ego_speed_mps,
            )
        )
        self._set_ego_state(scene, speed, requested_accel)
        after = self._ego_state_payload(scene)
        return {
            "operation": "semantic_control",
            "before": before,
            "after": after,
            "requested_speed_mps": float(requested_speed),
            "requested_acceleration_mps2": float(requested_accel),
            "reason": (
                "ego is a semantic control in EDIT_EXISTING and may be further "
                "synchronized by the occluded-interaction geometry solver"
            ),
        }

    @staticmethod
    def _tune_hazard_geometry(
        spec: HazardSemanticSpec,
        *,
        lateral_gap_floor_m: float,
        longitudinal_extension_m: float,
    ) -> Dict[str, Any]:
        risk = spec.risk_layer
        old_lateral = tuple(float(v) for v in risk.lateral_gap_range_m)
        old_longitudinal = tuple(
            float(v) for v in risk.longitudinal_distance_range_m
        )

        # Do not push the pedestrian progressively farther from the ego path.
        # Pedestrian lateral distance is now solved directly from the target
        # lane-entry time inside HazardClearancePrimitiveOps.  The language/risk
        # lateral prior remains untouched and is kept only as metadata.
        risk.lateral_gap_range_m = old_lateral

        new_longitudinal_low = max(4.0, old_longitudinal[0])
        new_longitudinal_high = min(
            40.0,
            old_longitudinal[1] + longitudinal_extension_m,
        )
        risk.longitudinal_distance_range_m = (
            float(new_longitudinal_low),
            float(max(new_longitudinal_low, new_longitudinal_high)),
        )

        return {
            "lateral_gap_range_before_m": list(old_lateral),
            "lateral_gap_range_after_m": list(risk.lateral_gap_range_m),
            "longitudinal_distance_range_before_m": list(old_longitudinal),
            "longitudinal_distance_range_after_m": list(
                risk.longitudinal_distance_range_m
            ),
            "reason": (
                "keep lateral risk prior unchanged; only widen longitudinal "
                "search because pedestrian lateral position is timing-derived"
            ),
        }

    @staticmethod
    def _inject_execution_hints(
        spec: HazardSemanticSpec,
        sampled: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Carry sampled physical occluder dimensions into PrimitiveOps.

        nuPlan category projection can remain VEHICLE while a parked car, truck
        or bus still keeps its sampled physical width/length.
        """

        debug = dict(getattr(spec, "debug", {}) or {})
        hints: Dict[str, Any] = {}
        for key in (
            "occluder_width_m",
            "occluder_length_m",
            "occluder_lateral_offset_m",
        ):
            value = sampled.get(key)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            debug[key] = float(value)
            hints[key] = float(value)
        debug["semantic_lane_center_y"] = 0.0
        hints["semantic_lane_center_y"] = 0.0
        spec.debug = debug
        return hints

    @staticmethod
    def _extract_background_edits(editor_report: Mapping[str, Any]) -> List[Dict[str, Any]]:
        extra = dict(editor_report.get("extra", {}) or {})
        rows = list(extra.get("background_clearance_edits", []) or [])
        # Some reports also embed the same information in the final layout.
        if not rows:
            layout = dict(extra.get("occluder_layout", {}) or {})
            rows = list(layout.get("background_clearance_edits", []) or [])
        return [dict(row) for row in rows]

    @staticmethod
    def _snapshot_element(elem: SledgeVectorElement) -> Dict[str, np.ndarray]:
        return {
            "states": np.asarray(elem.states).copy(),
            "mask": np.asarray(elem.mask).copy(),
        }

    @staticmethod
    def _element_matches_snapshot(
        elem: SledgeVectorElement,
        snapshot: Mapping[str, np.ndarray],
    ) -> bool:
        return bool(
            np.array_equal(np.asarray(elem.states), snapshot["states"])
            and np.array_equal(np.asarray(elem.mask), snapshot["mask"])
        )

    @staticmethod
    def _set_ego_state(
        scene: SledgeVectorRaw,
        speed_mps: float,
        acceleration_mps2: float,
    ) -> None:
        states = np.asarray(scene.ego.states)
        if states.size == 0:
            scene.ego.states = np.asarray(
                [speed_mps, 0.0, acceleration_mps2, 0.0],
                dtype=np.float32,
            )
        else:
            flat = states.reshape(-1)
            if flat.size > EgoIndex.VELOCITY_X:
                flat[EgoIndex.VELOCITY_X] = float(speed_mps)
            if flat.size > EgoIndex.VELOCITY_Y:
                flat[EgoIndex.VELOCITY_Y] = 0.0
            if flat.size > EgoIndex.ACCELERATION_X:
                flat[EgoIndex.ACCELERATION_X] = float(acceleration_mps2)
            if flat.size > EgoIndex.ACCELERATION_Y:
                flat[EgoIndex.ACCELERATION_Y] = 0.0
        mask = np.asarray(scene.ego.mask)
        if mask.size:
            mask.reshape(-1)[0] = True

    @staticmethod
    def _ego_state_payload(scene: SledgeVectorRaw) -> Dict[str, float]:
        states = np.asarray(scene.ego.states, dtype=np.float32).reshape(-1)
        vx = (
            float(states[EgoIndex.VELOCITY_X])
            if states.size > EgoIndex.VELOCITY_X
            else 0.0
        )
        vy = (
            float(states[EgoIndex.VELOCITY_Y])
            if states.size > EgoIndex.VELOCITY_Y
            else 0.0
        )
        ax = (
            float(states[EgoIndex.ACCELERATION_X])
            if states.size > EgoIndex.ACCELERATION_X
            else 0.0
        )
        ay = (
            float(states[EgoIndex.ACCELERATION_Y])
            if states.size > EgoIndex.ACCELERATION_Y
            else 0.0
        )
        return {
            "velocity_x_mps": vx,
            "velocity_y_mps": vy,
            "speed_mps": float(math.hypot(vx, vy)),
            "acceleration_x_mps2": ax,
            "acceleration_y_mps2": ay,
            "acceleration_mps2": float(math.hypot(ax, ay)),
        }

    @staticmethod
    def _schedule_value(values: Sequence[Any], index: int) -> Any:
        if not values:
            raise ValueError("empty elastic-context schedule")
        return values[min(index, len(values) - 1)]

    @staticmethod
    def _float(value: Any, *, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = float(default)
        if not math.isfinite(parsed):
            parsed = float(default)
        return float(parsed)

    def _final_report(
        self,
        *,
        sampled_parameters: Mapping[str, Any],
        attempt_reports: List[Dict[str, Any]],
        selected_attempt: int,
        selected_report: Mapping[str, Any],
        accepted_by_preview: bool,
    ) -> Dict[str, Any]:
        edits = list(selected_report.get("background_actor_edits", []) or [])
        removals = sum(1 for row in edits if row.get("operation") == "delete")
        repositions = sum(
            1 for row in edits if row.get("operation") == "reposition"
        )
        return {
            "schema_version": "elastic_context_edit_v2_move_then_delete",
            "policy": self.policy.to_dict(),
            "construction_contract": {
                "hard_preserve": ["road_geometry", "lane_geometry"],
                "semantic_controls": [
                    "ego_state",
                    "primary_pedestrian",
                    "occluder",
                ],
                "elastic_context": [
                    "background_vehicles",
                    "background_pedestrians",
                    "background_static_objects",
                ],
                "background_priority": [
                    "preserve_if_not_blocking",
                    "reposition_if_blocking",
                    "delete_if_relocation_fails",
                ],
                "hazard_constraints_remain_hard": True,
                "background_removal_allowed": True,
                "traffic_lights_removed": False,
            },
            "sampled_ego_target_mps": sampled_parameters.get("ego_speed_mps"),
            "selected_attempt": int(selected_attempt),
            "accepted_by_metrics_preview": bool(accepted_by_preview),
            "hard_road_lane_preserved": bool(
                selected_report.get("hard_road_lane_preserved", False)
            ),
            "ego_edit": dict(selected_report.get("ego_edit", {})),
            "geometry_edit": dict(selected_report.get("geometry_edit", {})),
            "execution_hints": dict(
                selected_report.get("execution_hints", {}) or {}
            ),
            "background_actor_edits": edits,
            "background_actor_edit_count": len(edits),
            "background_reposition_count": int(repositions),
            "background_removal_count": int(removals),
            "removal_used": bool(removals > 0),
            "attempts": attempt_reports,
        }


__all__ = [
    "ElasticContextPolicy",
    "ElasticContextHazardConstructor",
]