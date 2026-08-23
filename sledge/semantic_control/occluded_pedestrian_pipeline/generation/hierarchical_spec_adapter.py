"""Bridge sampled hierarchical parameters to the existing B1 editor specification."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_spec import (
    HazardSemanticSpec,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hierarchical_template_sampler import (
    EDIT_EXISTING,
    SYNTHESIZE_NEW,
    ConcreteOccludedPedestrianParameters,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.spec_presets import (
    apply_risk_preset,
)


RISK_TTC_WINDOWS = {
    "mild": (3.0, 4.2),
    "moderate": (2.0, 3.0),
    "aggressive": (1.2, 2.0),
    "critical": (0.9, 1.5),
}


class HierarchicalHazardSpecAdapter:
    """Convert one valid concrete sample into ``HazardSemanticSpec``.

    Global road handling is mode-aware:

    * edit_existing: road/map geometry is preserved from B0 and road-template
      parameters are only compatibility metadata;
    * synthesize_new: road/ego geometry has already been built by
      ``TemplateSceneSynthesizer`` from the sampled template.

    The compositional editor therefore never needs to silently regenerate road
    geometry for this occluded-pedestrian family.
    """

    def adapt(
        self,
        *,
        prompt: str,
        hierarchical_spec: Mapping[str, Any],
        sample: ConcreteOccludedPedestrianParameters,
        spec_id: str,
        construction_plan: Optional[Mapping[str, Any]] = None,
    ) -> HazardSemanticSpec:
        if not sample.valid:
            raise ValueError(
                "Cannot create HazardSemanticSpec from invalid sampled parameters: "
                + "; ".join(sample.issues)
            )

        hierarchy = dict(hierarchical_spec.get("hierarchy_layer", {}) or {})
        values = dict(hierarchy.get("path_values", {}) or {})
        plan = dict(
            construction_plan
            or hierarchical_spec.get("scene_construction", {})
            or {}
        )
        mode = str(plan.get("mode", sample.construction_mode))
        if mode not in {EDIT_EXISTING, SYNTHESIZE_NEW}:
            raise ValueError(f"Unsupported construction mode: {mode!r}")

        road_topology = self._road_topology(sample.road_topology)

        # ``time_to_collision_s`` in the hierarchy is explicitly declared as a
        # DERIVED variable.  The previous adapter incorrectly converted the
        # pre-construction estimate ``ego_distance / ego_speed`` into a hard
        # target interval.  That made the final pedestrian/ego interaction
        # chase an arbitrary number unrelated to pedestrian lane-entry timing.
        #
        # For ordinary prompts, risk level supplies the construction window.
        # If a future prompt provides an explicit user TTC value, preserve that
        # direct instruction instead.
        ttc_range, ttc_control_source = self._resolve_ttc_range(
            hierarchical_spec=hierarchical_spec,
            sample=sample,
        )
        gap_center = max(sample.ego_distance_to_conflict_m, 1.0)
        gap_range = (
            max(1.0, gap_center - 2.0),
            gap_center + 2.0,
        )

        edit_existing = mode == EDIT_EXISTING
        payload: Dict[str, Any] = {
            "spec_id": spec_id,
            "description": (
                "A nuPlan pedestrian starts behind an occluder and moves from "
                "the resolved occluder side toward the ego path."
            ),
            "canonical_type": "Occluded-Pedestrian",
            "raw_prompt": prompt,
            "road_layer": {
                # In edit mode this value describes the language template only;
                # the actual B0 lines remain untouched.
                "road_topology": road_topology,
                "lane_context": "ego_path",
                "anchor_type": "ego_future_path",
                "anchor_region": "front",
                "num_lanes": (
                    None if edit_existing else sample.lane_count
                ),
                # Metrics still need a lane-width scalar; in edit mode this is
                # the B0-context estimate, not a value used to rewrite B0.
                "lane_width_m": sample.lane_width_m,
                "require_lane_continuity": True,
                "require_drivable_route": True,
                "allow_lane_generation": False,
                "generated_road_layout": "none",
            },
            "actor_layer": {
                "primary_actor": "pedestrian",
                "actor_role": "crossing_actor",
                "secondary_actor": sample.executable_occluder_type,
                "allow_actor_insertion": True,
                "prefer_existing_actor": False,
            },
            "object_layer": {
                "occlusion": {
                    "enabled": True,
                    "occluder_type": sample.executable_occluder_type,
                    "occlusion_position": "between_ego_and_actor",
                    "occlusion_level": (
                        "partial"
                        if values.get("visibility") == "partially_occluded"
                        else "full"
                    ),
                },
                "static_obstacle": {"enabled": False},
            },
            "interaction_layer": {
                "conflict_type": "lateral_conflict",
                # Absolute execution direction is derived only after the side
                # has been resolved once.
                "conflict_direction": sample.concrete_direction,
                "distance_relation": "close",
                "speed_relation": "normal",
                "interaction_goal": "near_miss",
            },
            "risk_layer": {
                "risk_level": sample.risk_level,
                "ttc_range_s": ttc_range,
                "gap_range_m": gap_range,
                "target_actor_speed_mps": sample.actor_speed_mps,
                "target_decel_range_mps2": (
                    max(1.0, sample.braking_deceleration_mps2 - 1.0),
                    sample.braking_deceleration_mps2 + 1.0,
                ),
                "collision_allowed": False,
            },
            "validation_layer": {
                "require_actor_match": True,
                # A local edit prompt did not ask us to rewrite or prove a
                # global road topology. Synthesis prompts did.
                "require_road_context_match": not edit_existing,
                "require_conflict_relation": True,
                "require_direction_match": True,
                "require_visibility_match": True,
                "require_lane_validity": True,
                "require_no_initial_collision": True,
                "require_ttc_in_range": True,
                "require_gap_in_range": True,
            },
            "protection_layer": {
                "protect_primary_actor": True,
                "protect_secondary_actor": True,
                "protect_static_obstacle": True,
                "protect_conflict_corridor": True,
                "protect_road_anchor": True,
            },
            "tags": [
                "hierarchical_eventframe",
                "occluded_pedestrian",
                mode,
                sample.executable_occluder_type,
                sample.occluder_side,
                sample.concrete_direction,
                sample.risk_level,
            ],
            "debug": {
                "construction_mode": mode,
                "scene_construction": plan,
                "road_parameter_source": sample.road_parameter_source,
                "ego_state_source": sample.ego_state_source,
                "semantic_direction": sample.semantic_direction,
                "occluder_side": sample.occluder_side,
                "language_actor_detail": sample.language_actor_detail,
                "semantic_occluder_type": sample.semantic_occluder_type,
                "sampled_parameters": sample.to_dict(),
                "hierarchical_path": values,
                "preconstruction_ttc_estimate_s": float(sample.time_to_collision_s),
                "ttc_target_range_s": list(ttc_range),
                "ttc_control_source": ttc_control_source,
            },
        }

        spec = HazardSemanticSpec.from_dict(payload)
        spec = apply_risk_preset(spec, overwrite=False)
        # Preserve concrete values after filling any remaining preset defaults.
        spec.risk_layer.target_actor_speed_mps = sample.actor_speed_mps
        spec.risk_layer.ttc_range_s = ttc_range
        spec.risk_layer.gap_range_m = gap_range
        return spec

    @classmethod
    def _resolve_ttc_range(
        cls,
        *,
        hierarchical_spec: Mapping[str, Any],
        sample: ConcreteOccludedPedestrianParameters,
    ) -> tuple[tuple[float, float], str]:
        completed = dict(
            hierarchical_spec.get("parameter_layer", {})
            .get("completed", {})
            or {}
        )
        entry = completed.get("time_to_collision_s")
        if isinstance(entry, Mapping):
            source = str(entry.get("source", ""))
            value = entry.get("value")
            if source == "user_input":
                parsed = cls._parse_explicit_ttc(value)
                if parsed is not None:
                    return parsed, "explicit_user_ttc"

        risk = str(sample.risk_level or "moderate")
        return (
            tuple(float(v) for v in RISK_TTC_WINDOWS.get(risk, RISK_TTC_WINDOWS["moderate"])),
            f"risk_level:{risk}",
        )

    @staticmethod
    def _parse_explicit_ttc(value: Any) -> Optional[tuple[float, float]]:
        if isinstance(value, (int, float)):
            center = max(float(value), 0.2)
            return (max(0.2, center - 0.25), center + 0.25)
        if isinstance(value, (list, tuple)) and value:
            vals = [float(v) for v in value[:2]]
            if len(vals) == 1:
                center = max(vals[0], 0.2)
                return (max(0.2, center - 0.25), center + 0.25)
            low, high = sorted(vals)
            return (max(0.2, low), max(max(0.2, low), high))
        return None

    @staticmethod
    def _road_topology(value: Any) -> str:
        mapping = {
            "straight_segment": "straight",
            "straight": "straight",
            "intersection": "intersection",
            "roundabout": "roundabout",
            "merge_diverge": "merge",
            "merge": "merge",
            "work_zone": "construction_zone",
            "construction_zone": "construction_zone",
        }
        return mapping.get(str(value), "straight")