"""Sample one concrete occluded-pedestrian scene from a hierarchical template."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence


SEMANTIC_DIRECTION = "occluder_to_ego_path"
EDIT_EXISTING = "edit_existing"
SYNTHESIZE_NEW = "synthesize_new"


@dataclass(frozen=True)
class SamplingOverrides:
    occluder_type: Optional[str] = None
    occluder_side: Optional[str] = None
    pedestrian_speed_mps: Optional[float] = None
    risk_level: Optional[str] = None
    seed: Optional[int] = None


@dataclass
class ConcreteOccludedPedestrianParameters:
    seed: int
    semantic_direction: str
    occluder_side: str
    concrete_direction: str
    language_actor_detail: str
    semantic_occluder_type: str
    executable_occluder_type: str
    risk_level: str
    lane_width_m: float
    lane_count: int
    road_curvature: float
    ego_speed_mps: float
    ego_acceleration_mps2: float
    ego_distance_to_conflict_m: float
    actor_speed_mps: float
    actor_acceleration_mps2: float
    actor_start_time_s: float
    occluder_longitudinal_m: float
    occluder_lateral_offset_m: float
    occluder_length_m: float
    occluder_width_m: float
    reveal_distance_m: float
    minimum_clearance_m: float
    braking_deceleration_mps2: float
    time_to_collision_s: float
    construction_mode: str = EDIT_EXISTING
    road_topology: str = "straight_segment"
    road_parameter_source: str = "hierarchical_template"
    ego_state_source: str = "hierarchical_template"
    valid: bool = True
    issues: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def concrete_parameter_sample_ready(self) -> bool:
        return self.valid

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["concrete_parameter_sample_ready"] = (
            self.concrete_parameter_sample_ready
        )
        # A concrete SLEDGE scene does not exist until B1 editing and geometric
        # validation have completed.
        payload["sampled_scene_ready"] = False
        payload["actor_heading_definition"] = (
            "unit_vector(occluder_position, nearest_point_on_ego_path)"
        )
        payload["conflict_point_xy"] = [self.ego_distance_to_conflict_m, 0.0]
        lateral_sign = 1.0 if self.occluder_side == "left" else -1.0
        payload["occluder_position_xy"] = [
            self.occluder_longitudinal_m,
            lateral_sign
            * (0.5 * self.lane_width_m + self.occluder_lateral_offset_m),
        ]
        return payload


class HierarchicalTemplateSampler:
    """Resolve template leaves while respecting the construction mode.

    ``edit_existing`` preserves road/lane parameters extracted from B0, but ego
    state is a semantic control sampled from the hierarchical hazard template.
    This lets the ego motion adapt to the requested dangerous interaction while
    the original road/lane geometry remains fixed.

    ``synthesize_new`` samples both road and ego parameters from the hierarchy
    and then applies any explicit global constraints found by the construction
    router.
    """

    def sample(
        self,
        payload: Mapping[str, Any],
        *,
        prompt: str = "",
        case_id: str = "",
        overrides: Optional[SamplingOverrides] = None,
        construction_plan: Optional[Mapping[str, Any]] = None,
        scene_context: Optional[Mapping[str, Any]] = None,
    ) -> ConcreteOccludedPedestrianParameters:
        spec = self._spec(payload)
        hierarchy = dict(spec.get("hierarchy_layer", {}) or {})
        values = dict(hierarchy.get("path_values", {}) or {})
        attributes = dict(hierarchy.get("attributes", {}) or {})
        completed = dict(spec.get("parameter_layer", {}).get("completed", {}) or {})
        overrides = overrides or SamplingOverrides()

        plan = dict(
            construction_plan
            or spec.get("scene_construction", {})
            or {}
        )
        mode = str(plan.get("mode", EDIT_EXISTING))
        if mode not in {EDIT_EXISTING, SYNTHESIZE_NEW}:
            raise ValueError(f"unsupported construction mode: {mode!r}")
        explicit = dict(plan.get("explicit_global_constraints", {}) or {})
        context = dict(scene_context or {})

        seed = (
            int(overrides.seed)
            if overrides.seed is not None
            else self._stable_seed(case_id, prompt, values)
        )
        rng = random.Random(seed)

        side, side_source = self._sample_side(completed, values, overrides, rng)
        concrete_direction = "left_to_right" if side == "left" else "right_to_left"

        semantic_occluder = str(
            values.get("auxiliary_entity", "generic_vehicle_occluder")
        )
        executable_occluder = (
            self._normalize_executable_occluder(overrides.occluder_type)
            if overrides.occluder_type
            else self._execution_occluder(semantic_occluder)
        )
        risk = self._risk(
            overrides.risk_level or values.get("risk_level", "moderate")
        )

        (
            lane_width,
            lane_count,
            road_curvature,
            ego_speed,
            ego_acceleration,
            road_parameter_source,
            ego_state_source,
        ) = self._global_parameters(
            mode=mode,
            explicit=explicit,
            context=context,
            completed=completed,
            rng=rng,
        )

        road_topology = str(
            explicit.get(
                "road_topology",
                values.get("road_topology", "straight_segment"),
            )
        )
        if mode == SYNTHESIZE_NEW:
            directionality = str(explicit.get("directionality", ""))
            lane_layout = list(explicit.get("lane_layout", []) or [])
            if directionality == "bidirectional" and "lane_count" not in explicit:
                lane_count = max(2, lane_count)
            if "minimum_lane_count" in explicit and "lane_count" not in explicit:
                lane_count = max(int(explicit["minimum_lane_count"]), lane_count)
            if lane_layout and "lane_count" not in explicit:
                lane_count = max(2, lane_count)
            if explicit.get("road_shape") == "curved" and abs(road_curvature) < 1e-5:
                road_curvature = 0.004

        ego_distance = self._sample_number(
            completed, "ego_distance_to_conflict_m", rng, 10.0
        )
        actor_speed = (
            float(overrides.pedestrian_speed_mps)
            if overrides.pedestrian_speed_mps is not None
            else self._sample_number(completed, "actor_speed_mps", rng, 1.6)
        )
        actor_acceleration = self._sample_number(
            completed, "actor_acceleration_mps2", rng, 0.0
        )
        actor_start_time = self._sample_number(
            completed, "actor_start_time_s", rng, 0.8
        )
        occ_x = self._nested_range_sample(
            self._entry_value(completed, "occluder_position", {}),
            "x_m",
            rng,
            7.0,
        )
        occ_offset = self._sample_number(
            completed, "occluder_lateral_offset_m", rng, 2.0
        )
        occ_length = self._sample_number(
            completed, "occluder_length_m", rng, 5.0
        )
        occ_width = self._sample_number(
            completed, "occluder_width_m", rng, 2.0
        )
        reveal_distance = self._sample_number(
            completed, "reveal_distance_m", rng, min(6.0, ego_distance)
        )
        minimum_clearance = self._sample_number(
            completed, "minimum_clearance_m", rng, 1.0
        )
        braking = self._sample_number(
            completed, "braking_deceleration_mps2", rng, 4.5
        )

        reveal_distance = min(
            max(reveal_distance, 0.5),
            max(ego_distance, 0.5),
        )
        ttc = ego_distance / max(ego_speed, 1e-3)
        issues = self._validate(
            values=values,
            side=side,
            actor_speed=actor_speed,
            ego_speed=ego_speed,
            ego_distance=ego_distance,
            reveal_distance=reveal_distance,
            occluder_offset=occ_offset,
            lane_width=lane_width,
            lane_count=lane_count,
            mode=mode,
        )

        return ConcreteOccludedPedestrianParameters(
            seed=seed,
            semantic_direction=SEMANTIC_DIRECTION,
            occluder_side=side,
            concrete_direction=concrete_direction,
            language_actor_detail=str(
                attributes.get("language_actor_detail", "adult_or_unspecified")
            ),
            semantic_occluder_type=semantic_occluder,
            executable_occluder_type=executable_occluder,
            risk_level=risk,
            lane_width_m=lane_width,
            lane_count=lane_count,
            road_curvature=road_curvature,
            ego_speed_mps=ego_speed,
            ego_acceleration_mps2=ego_acceleration,
            ego_distance_to_conflict_m=ego_distance,
            actor_speed_mps=actor_speed,
            actor_acceleration_mps2=actor_acceleration,
            actor_start_time_s=actor_start_time,
            occluder_longitudinal_m=occ_x,
            occluder_lateral_offset_m=occ_offset,
            occluder_length_m=occ_length,
            occluder_width_m=occ_width,
            reveal_distance_m=reveal_distance,
            minimum_clearance_m=minimum_clearance,
            braking_deceleration_mps2=braking,
            time_to_collision_s=ttc,
            construction_mode=mode,
            road_topology=road_topology,
            road_parameter_source=road_parameter_source,
            ego_state_source=ego_state_source,
            valid=not issues,
            issues=issues,
            provenance={
                "seed": seed,
                "construction_mode": mode,
                "road_parameters": road_parameter_source,
                "ego_state": ego_state_source,
                "occluder_side": side_source,
                "semantic_direction": "hierarchical_geometric_constraint",
                "concrete_direction": "derived_from_occluder_side",
                "pedestrian_speed_mps": (
                    "control_override"
                    if overrides.pedestrian_speed_mps is not None
                    else self._parameter_source(completed, "actor_speed_mps")
                ),
                "risk_level": (
                    "control_override"
                    if overrides.risk_level is not None
                    else "hierarchical_template"
                ),
            },
        )

    def _global_parameters(
        self,
        *,
        mode: str,
        explicit: Mapping[str, Any],
        context: Mapping[str, Any],
        completed: Mapping[str, Any],
        rng: random.Random,
    ) -> tuple[float, int, float, float, float, str, str]:
        if mode == EDIT_EXISTING and context:
            lane_width = self._positive_float(
                context.get("lane_width_m"),
                self._sample_number(completed, "lane_width_m", rng, 3.5),
            )
            lane_count = max(
                1,
                int(round(self._positive_float(context.get("lane_count"), 1.0))),
            )
            road_curvature = self._finite_float(
                context.get("road_curvature"), 0.0
            )

            # Road/lane stays B0-conditioned, but ego is deliberately editable.
            # Use the hierarchical language-conditioned prior instead of the
            # source ego state; the elastic geometry solver may refine this
            # target further after pedestrian/occluder placement.
            ego_speed = self._sample_number(
                completed, "ego_speed_mps", rng,
                self._positive_float(context.get("ego_speed_mps"), 8.0),
            )
            ego_acceleration = self._sample_number(
                completed, "ego_acceleration_mps2", rng,
                self._finite_float(context.get("ego_acceleration_mps2"), 0.0),
            )
            return (
                lane_width,
                lane_count,
                road_curvature,
                ego_speed,
                ego_acceleration,
                "b0_scene_context",
                "hierarchical_template_semantic_control",
            )

        lane_width = self._sample_number(completed, "lane_width_m", rng, 3.5)
        lane_count = max(
            1,
            int(round(self._sample_number(completed, "lane_count", rng, 1.0))),
        )
        road_curvature = self._sample_number(
            completed, "road_curvature", rng, 0.0
        )
        ego_speed = self._sample_number(completed, "ego_speed_mps", rng, 8.0)
        ego_acceleration = self._sample_number(
            completed, "ego_acceleration_mps2", rng, 0.0
        )
        source = (
            "hierarchical_template"
            if mode == SYNTHESIZE_NEW
            else "hierarchical_fallback_no_b0_context"
        )

        if mode == SYNTHESIZE_NEW:
            if "lane_width_m" in explicit:
                lane_width = float(explicit["lane_width_m"])
            if "lane_count" in explicit:
                lane_count = max(1, int(explicit["lane_count"]))
            if "road_curvature" in explicit:
                road_curvature = float(explicit["road_curvature"])
            return (
                lane_width,
                lane_count,
                road_curvature,
                ego_speed,
                ego_acceleration,
                "explicit_language_plus_hierarchical_template"
                if explicit
                else "hierarchical_template",
                "hierarchical_template",
            )

        return (
            lane_width,
            lane_count,
            road_curvature,
            ego_speed,
            ego_acceleration,
            source,
            source,
        )

    @staticmethod
    def _spec(payload: Mapping[str, Any]) -> Dict[str, Any]:
        if "spec" in payload and isinstance(payload["spec"], Mapping):
            return dict(payload["spec"])
        return dict(payload)

    @staticmethod
    def _stable_seed(
        case_id: str,
        prompt: str,
        values: Mapping[str, Any],
    ) -> int:
        text = f"{case_id}|{prompt}|{sorted(values.items())}"
        return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)

    def _sample_side(
        self,
        completed: Mapping[str, Any],
        values: Mapping[str, Any],
        overrides: SamplingOverrides,
        rng: random.Random,
    ) -> tuple[str, str]:
        if overrides.occluder_side:
            side = str(overrides.occluder_side).lower()
            if side not in {"left", "right"}:
                raise ValueError("occluder_side must be left or right")
            return side, "control_override"

        source_region = str(values.get("source_region", "curbside"))
        if source_region == "left_side":
            return "left", "hierarchical_source_region"
        if source_region == "right_side":
            return "right", "hierarchical_source_region"

        value = self._entry_value(completed, "occluder_side", None)
        if isinstance(value, str) and value in {"left", "right"}:
            return value, "hierarchical_parameter"
        if isinstance(value, Mapping):
            candidates = list(value.get("values", []) or [])
            candidates = [
                str(item)
                for item in candidates
                if str(item) in {"left", "right"}
            ]
            if candidates:
                return rng.choice(candidates), "categorical_sample_once"
        return rng.choice(["left", "right"]), "deterministic_default_sample_once"

    @staticmethod
    def _entry_value(
        completed: Mapping[str, Any],
        name: str,
        default: Any,
    ) -> Any:
        entry = completed.get(name, default)
        return entry.get("value", default) if isinstance(entry, Mapping) else entry

    @staticmethod
    def _parameter_source(completed: Mapping[str, Any], name: str) -> str:
        entry = completed.get(name)
        if isinstance(entry, Mapping):
            return str(entry.get("source", "hierarchical_template"))
        return "hierarchical_template"

    def _sample_number(
        self,
        completed: Mapping[str, Any],
        name: str,
        rng: random.Random,
        default: float,
    ) -> float:
        return self._number(
            self._entry_value(completed, name, default),
            rng,
            default,
        )

    def _nested_range_sample(
        self,
        value: Any,
        key: str,
        rng: random.Random,
        default: float,
    ) -> float:
        if isinstance(value, Mapping):
            return self._number(value.get(key, default), rng, default)
        return default

    @staticmethod
    def _number(
        value: Any,
        rng: random.Random,
        default: float,
    ) -> float:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items = list(value)
            if not items:
                return float(default)
            if len(items) == 1:
                return float(items[0])
            low, high = sorted((float(items[0]), float(items[1])))
            return rng.uniform(low, high)
        return float(default)

    @staticmethod
    def _positive_float(value: Any, default: float) -> float:
        try:
            output = float(value)
        except (TypeError, ValueError):
            return float(default)
        return output if output > 0.0 else float(default)

    @staticmethod
    def _finite_float(value: Any, default: float) -> float:
        try:
            output = float(value)
        except (TypeError, ValueError):
            return float(default)
        if output != output or output in {float("inf"), float("-inf")}:
            return float(default)
        return output

    @staticmethod
    def _risk(value: Any) -> str:
        risk = str(value or "moderate").lower()
        if risk == "critical":
            return "aggressive"
        return risk if risk in {"mild", "moderate", "aggressive"} else "moderate"

    @staticmethod
    def _execution_occluder(semantic: str) -> str:
        value = str(semantic)
        if value in {
            "parked_car_occluder",
            "parked_truck_occluder",
            "bus_occluder",
            "van_occluder",
            "generic_vehicle_occluder",
        }:
            return "vehicle"
        if value == "barrier_occluder":
            return "barrier"
        return "generic_object"

    @staticmethod
    def _normalize_executable_occluder(value: Any) -> str:
        raw = str(value).lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "car": "vehicle",
            "truck": "vehicle",
            "bus": "vehicle",
            "van": "vehicle",
            "parked_car": "vehicle",
            "parked_truck": "vehicle",
            "static_object": "generic_object",
            "barrier_occluder": "barrier",
        }
        normalized = aliases.get(raw, raw)
        supported = {
            "vehicle",
            "bicycle",
            "generic_object",
            "traffic_cone",
            "barrier",
            "czone_sign",
        }
        if normalized not in supported:
            raise ValueError(
                f"unsupported executable occluder type: {value!r}"
            )
        return normalized

    @staticmethod
    def _validate(
        *,
        values: Mapping[str, Any],
        side: str,
        actor_speed: float,
        ego_speed: float,
        ego_distance: float,
        reveal_distance: float,
        occluder_offset: float,
        lane_width: float,
        lane_count: int,
        mode: str,
    ) -> List[str]:
        issues: List[str] = []
        if values.get("primary_actor_type") != "pedestrian":
            issues.append("primary_actor_type must be pedestrian")
        if values.get("hazard_interaction") != "occluded_emergence":
            issues.append("hazard_interaction must be occluded_emergence")
        if values.get("motion_direction") != SEMANTIC_DIRECTION:
            issues.append("motion_direction must be occluder_to_ego_path")
        if side not in {"left", "right"}:
            issues.append("one concrete occluder side is required")
        if not 0.5 <= actor_speed <= 2.0:
            issues.append(
                "actor_speed_mps is outside the current RVAE range [0.5, 2.0]"
            )
        if ego_speed <= 0.0:
            issues.append("ego_speed_mps must be positive")
        if ego_distance <= 0.0:
            issues.append("ego_distance_to_conflict_m must be positive")
        if not 0.0 < reveal_distance <= ego_distance:
            issues.append(
                "reveal_distance_m must be within ego distance to conflict"
            )
        if occluder_offset <= 0.0:
            issues.append("occluder_lateral_offset_m must be positive")
        if lane_width <= 0.0:
            issues.append("lane_width_m must be positive")
        if lane_count <= 0:
            issues.append("lane_count must be positive")
        if mode not in {EDIT_EXISTING, SYNTHESIZE_NEW}:
            issues.append("construction mode must be edit_existing or synthesize_new")
        return issues


__all__ = [
    "SEMANTIC_DIRECTION",
    "EDIT_EXISTING",
    "SYNTHESIZE_NEW",
    "SamplingOverrides",
    "ConcreteOccludedPedestrianParameters",
    "HierarchicalTemplateSampler",
]