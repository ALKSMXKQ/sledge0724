"""Route hierarchical language specs to B0 editing or full scene synthesis.

The router intentionally separates *global road structure* from *local hazard
semantics*.  Merely mentioning the ego lane/path, a roadside, or a pedestrian
crossing a lane never triggers full synthesis.  Full synthesis is selected only
when a global road constraint is explicitly provided by the user.

The hierarchy always contains a value for every node, therefore routing must use
both ``value`` and ``source``.  A hierarchical default such as
``road_topology=straight_segment`` is not evidence that the prompt requested a
new road.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


class SceneConstructionMode(str, Enum):
    """How B1 should be constructed."""

    EDIT_EXISTING = "edit_existing"
    SYNTHESIZE_NEW = "synthesize_new"


# Provenance values that mean the user really supplied the value.  ``normalized``
# is deliberately excluded: in the current hierarchy it can also be produced
# from parser normalization/defaulting and therefore is not a safe synthesis
# trigger by itself.
EXPLICIT_SOURCES = frozenset(
    {
        "explicit",
        "user_input",
        "llm_explicit",
        "prompt_explicit",
        "control_override",
    }
)

# ``road_topology`` is inherently global.  ``ego_traffic_space`` is mixed: some
# values describe global lane organization while others merely locate the local
# hazard.  Only the structural subset may trigger synthesis.
GLOBAL_HIERARCHY_NODE_VALUES: Mapping[str, frozenset[str] | None] = {
    "road_topology": None,
    "ego_traffic_space": frozenset(
        {
            "same_direction_multi_lane",
            "bidirectional_road",
            "ramp_merge",
            "lane_drop",
            "diverge",
            "open_lane",
            "partially_blocked_lane",
            "closed_lane",
        }
    ),
}

# Global road geometry/layout parameters live in ``parameter_layer.completed``
# rather than the hierarchy.  They only trigger synthesis when their provenance
# is explicit.  The same parameters remain in the template in edit mode but are
# marked inactive so that B0 geometry wins.
GLOBAL_ROAD_PARAMETER_NAMES = frozenset(
    {
        "lane_count",
        "lane_width_m",
        "road_width_m",
        "road_curvature",
        "road_curvature_1pm",
        "road_heading",
        "road_heading_rad",
        "road_directionality",
        "lane_directionality",
        "lane_configuration",
        "lane_layout",
        "lane_center_offset_m",
        "merge_length_m",
        "closed_lane_count",
        "turn_lane_count",
    }
)

# These hierarchy nodes describe the hazardous interaction itself and are always
# local with respect to construction routing.  In particular target_region may
# be ``ego_lane`` / ``ego_path`` without requesting a new road.
LOCAL_HAZARD_NODE_NAMES = (
    "primary_actor_group",
    "primary_actor_type",
    "hazard_interaction",
    "auxiliary_entity",
    "source_region",
    "target_region",
    "anchor_region",
    "visibility",
    "motion_direction",
    "trigger_event",
    "ego_required_response",
    "risk_level",
)

LOCAL_HAZARD_PARAMETER_NAMES = frozenset(
    {
        "actor_speed_mps",
        "actor_acceleration_mps2",
        "actor_start_time_s",
        "actor_initial_position",
        "actor_heading",
        "crossing_direction",
        "occlusion_enabled",
        "occluder_type",
        "occluder_side",
        "occluder_position",
        "occluder_lateral_offset_m",
        "occluder_length_m",
        "occluder_width_m",
        "reveal_distance_m",
        "ego_distance_to_conflict_m",
        "conflict_point_xy",
        "minimum_clearance_m",
        "initial_gap_m",
        "time_to_collision_s",
        "braking_deceleration_mps2",
    }
)

DEFAULTISH_SOURCES = frozenset(
    {
        "hierarchical_default",
        "hierarchical_prior",
        "categorical_prior",
        "deterministic_default",
        "default",
        "not_applicable",
    }
)

UNKNOWN_VALUES = {None, "", "unknown", "unknown_side", "unspecified"}


@dataclass(frozen=True)
class RoutingEvidence:
    """One constraint used to explain a routing decision."""

    name: str
    value: Any
    source: str
    kind: str

    def label(self) -> str:
        return f"{self.name}={self.value}"


@dataclass(frozen=True)
class SceneConstructionDecision:
    """Serializable routing decision attached to the final language spec."""

    mode: SceneConstructionMode
    reason: str
    explicit_global_constraints: Tuple[str, ...] = ()
    local_hazard_constraints: Tuple[str, ...] = ()
    routing_evidence: Tuple[RoutingEvidence, ...] = ()
    inherits_b0_road: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "reason": self.reason,
            "explicit_global_constraints": list(self.explicit_global_constraints),
            "local_hazard_constraints": list(self.local_hazard_constraints),
            "routing_evidence": [asdict(item) for item in self.routing_evidence],
            "inherits_b0_road": self.inherits_b0_road,
        }


class SceneConstructionRouter:
    """Make a deterministic, provenance-aware scene construction decision."""

    def route(self, spec: Mapping[str, Any]) -> SceneConstructionDecision:
        nodes = self._hierarchy_nodes(spec)
        parameters = self._completed_parameters(spec)

        global_evidence: List[RoutingEvidence] = []
        for node in nodes:
            node_type = str(node.get("node_type", ""))
            if node_type not in GLOBAL_HIERARCHY_NODE_VALUES:
                continue
            value = node.get("value")
            source = str(node.get("source", "unknown"))
            allowed_values = GLOBAL_HIERARCHY_NODE_VALUES[node_type]
            if not self._is_explicit(source):
                continue
            if value in UNKNOWN_VALUES:
                continue
            if allowed_values is not None and str(value) not in allowed_values:
                continue
            global_evidence.append(
                RoutingEvidence(
                    name=node_type,
                    value=value,
                    source=source,
                    kind="hierarchy_node",
                )
            )

        for name in sorted(GLOBAL_ROAD_PARAMETER_NAMES):
            entry = parameters.get(name)
            if not isinstance(entry, Mapping):
                continue
            source = str(entry.get("source", "unknown"))
            value = entry.get("value")
            if not self._is_explicit(source) or value in UNKNOWN_VALUES:
                continue
            global_evidence.append(
                RoutingEvidence(
                    name=name,
                    value=value,
                    source=source,
                    kind="road_parameter",
                )
            )

        local_constraints = self._local_hazard_constraints(nodes, parameters)
        if global_evidence:
            return SceneConstructionDecision(
                mode=SceneConstructionMode.SYNTHESIZE_NEW,
                reason="explicit_global_road_structure",
                explicit_global_constraints=tuple(item.label() for item in global_evidence),
                local_hazard_constraints=tuple(local_constraints),
                routing_evidence=tuple(global_evidence),
                inherits_b0_road=False,
            )

        return SceneConstructionDecision(
            mode=SceneConstructionMode.EDIT_EXISTING,
            reason="no_explicit_global_road_structure",
            explicit_global_constraints=(),
            local_hazard_constraints=tuple(local_constraints),
            routing_evidence=(),
            inherits_b0_road=True,
        )

    def attach(self, spec: Mapping[str, Any]) -> Dict[str, Any]:
        """Return a copied spec with routing and parameter execution policy."""

        out: Dict[str, Any] = deepcopy(dict(spec))
        decision = self.route(out)
        out["scene_construction"] = decision.to_dict()
        self._attach_parameter_execution_policy(out, decision.mode)
        return out

    @staticmethod
    def _is_explicit(source: str) -> bool:
        return str(source).strip().lower() in EXPLICIT_SOURCES

    @staticmethod
    def _hierarchy_nodes(spec: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        layer = spec.get("hierarchy_layer", {})
        if not isinstance(layer, Mapping):
            return []
        raw_nodes = layer.get("selected_path", [])
        if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, (str, bytes)):
            return []
        return [node for node in raw_nodes if isinstance(node, Mapping)]

    @staticmethod
    def _completed_parameters(spec: Mapping[str, Any]) -> Mapping[str, Any]:
        layer = spec.get("parameter_layer", {})
        if not isinstance(layer, Mapping):
            return {}
        completed = layer.get("completed", {})
        return completed if isinstance(completed, Mapping) else {}

    def _local_hazard_constraints(
        self,
        nodes: Iterable[Mapping[str, Any]],
        parameters: Mapping[str, Any],
    ) -> List[str]:
        constraints: List[str] = []
        by_name = {str(node.get("node_type", "")): node for node in nodes}

        for name in LOCAL_HAZARD_NODE_NAMES:
            node = by_name.get(name)
            if not node:
                continue
            value = node.get("value")
            source = str(node.get("source", "unknown"))
            if value in UNKNOWN_VALUES:
                continue
            # Keep semantically meaningful inferred values, but omit pure
            # hierarchy defaults so the list remains an explanation of the
            # actual local hazard rather than a dump of the full tree.
            if source in DEFAULTISH_SOURCES:
                continue
            constraints.append(f"{name}={value}")

        for name in sorted(LOCAL_HAZARD_PARAMETER_NAMES):
            entry = parameters.get(name)
            if not isinstance(entry, Mapping):
                continue
            value = entry.get("value")
            source = str(entry.get("source", "unknown"))
            if value in UNKNOWN_VALUES or not self._is_explicit(source):
                continue
            label = f"{name}={value}"
            if label not in constraints:
                constraints.append(label)

        return constraints

    @staticmethod
    def _attach_parameter_execution_policy(
        spec: MutableMapping[str, Any], mode: SceneConstructionMode
    ) -> None:
        parameter_layer = spec.setdefault("parameter_layer", {})
        if not isinstance(parameter_layer, MutableMapping):
            return
        completed = parameter_layer.get("completed", {})
        completed_names = sorted(completed) if isinstance(completed, Mapping) else []
        global_names = sorted(name for name in completed_names if name in GLOBAL_ROAD_PARAMETER_NAMES)

        if mode == SceneConstructionMode.EDIT_EXISTING:
            inactive = global_names
            active = [name for name in completed_names if name not in GLOBAL_ROAD_PARAMETER_NAMES]
            road_geometry_source = "inherit_b0"
        else:
            inactive = []
            active = completed_names
            road_geometry_source = "parameter_template"

        parameter_layer["execution_policy"] = {
            "scene_construction_mode": mode.value,
            "road_geometry_source": road_geometry_source,
            "active_parameter_names": active,
            "inactive_parameter_names": inactive,
            "global_road_parameter_names": global_names,
            "local_hazard_parameters_always_active": True,
        }


def attach_scene_construction(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Convenience function for callers that do not need a router instance."""

    return SceneConstructionRouter().attach(spec)
