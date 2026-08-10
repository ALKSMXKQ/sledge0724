"""Route hierarchical language specs to B0 editing or full scene synthesis.

The router intentionally separates *global road structure* from *local hazard
semantics*. Merely mentioning the ego lane/path, a roadside, or a pedestrian
crossing a lane never triggers full synthesis. Full synthesis is selected only
when a global road constraint is explicitly provided by the user.

The hierarchy always contains a value for every node, therefore routing must use
both ``value`` and ``source``. A hierarchical default such as
``road_topology=straight_segment`` is not evidence that the prompt requested a
new road.

The current EventFrame schema does not expose every global road property (for
example ``lane_count``). A deliberately narrow prompt-provenance bridge extracts
only global road structure expressions and records them as ``prompt_explicit``.
It never parses actor/occluder/ego-lane hazard semantics.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
import re
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


class SceneConstructionMode(str, Enum):
    """How B1 should be constructed."""

    EDIT_EXISTING = "edit_existing"
    SYNTHESIZE_NEW = "synthesize_new"


EXPLICIT_SOURCES = frozenset(
    {
        "explicit",
        "user_input",
        "llm_explicit",
        "prompt_explicit",
        "control_override",
    }
)

GLOBAL_HIERARCHY_NODE_VALUES: Mapping[str, Optional[FrozenSet[str]]] = {
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

UNKNOWN_TEXT_VALUES = frozenset({"", "unknown", "unknown_side", "unspecified"})
LANE_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}


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

    def route(
        self,
        spec: Mapping[str, Any],
        *,
        prompt: Optional[str] = None,
    ) -> SceneConstructionDecision:
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
            if self._is_unknown(value):
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
            if not self._is_explicit(source) or self._is_unknown(value):
                continue
            global_evidence.append(
                RoutingEvidence(
                    name=name,
                    value=value,
                    source=source,
                    kind="road_parameter",
                )
            )

        if prompt:
            global_evidence.extend(self._prompt_global_road_evidence(prompt))
        global_evidence = self._deduplicate_evidence(global_evidence)

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

    def attach(
        self,
        spec: Mapping[str, Any],
        *,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a copied spec with routing and parameter execution policy."""

        out: Dict[str, Any] = deepcopy(dict(spec))
        decision = self.route(out, prompt=prompt)
        out["scene_construction"] = decision.to_dict()
        self._attach_parameter_execution_policy(out, decision.mode)
        return out

    @staticmethod
    def _is_explicit(source: str) -> bool:
        return str(source).strip().lower() in EXPLICIT_SOURCES

    @staticmethod
    def _is_unknown(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().lower() in UNKNOWN_TEXT_VALUES
        return False

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
            if self._is_unknown(value):
                continue
            if source in DEFAULTISH_SOURCES:
                continue
            constraints.append(f"{name}={value}")

        for name in sorted(LOCAL_HAZARD_PARAMETER_NAMES):
            entry = parameters.get(name)
            if not isinstance(entry, Mapping):
                continue
            value = entry.get("value")
            source = str(entry.get("source", "unknown"))
            if self._is_unknown(value) or not self._is_explicit(source):
                continue
            label = f"{name}={value}"
            if label not in constraints:
                constraints.append(label)

        return constraints

    @staticmethod
    def _prompt_global_road_evidence(prompt: str) -> List[RoutingEvidence]:
        """Extract only explicit global road structure missing from EventFrame.

        This is intentionally conservative. Generic mentions of ``lane``,
        ``ego lane``, ``ego path``, crossing, left/right roadside, and vehicle
        merging do not match these patterns.
        """

        text = " ".join(str(prompt).lower().replace("_", " ").split())
        evidence: List[RoutingEvidence] = []

        def add(name: str, value: Any) -> None:
            evidence.append(
                RoutingEvidence(
                    name=name,
                    value=value,
                    source="prompt_explicit",
                    kind="prompt_global_constraint",
                )
            )

        lane_count_match = re.search(
            r"\b(one|two|three|four|five|six|[1-6])\s*[- ]\s*lane\b",
            text,
        )
        if lane_count_match:
            token = lane_count_match.group(1)
            add("lane_count", LANE_COUNT_WORDS.get(token, int(token) if token.isdigit() else token))

        if re.search(r"\b(?:bidirectional|bi-directional|two-way)\s+(?:road|street|traffic)\b", text):
            add("road_directionality", "bidirectional")
        elif re.search(r"\bone-way\s+(?:road|street|traffic)\b", text):
            add("road_directionality", "one_way")

        if re.search(r"\b(?:four-way|three-way|t-junction|t junction|cross)\s+intersection\b|\bintersection\b|\bjunction\b", text):
            add("road_topology", "intersection")
        if re.search(r"\b(?:roundabout|traffic circle|rotary)\b", text):
            add("road_topology", "roundabout")
        if re.search(r"\b(?:highway\s+merge|merge\s+ramp|merging\s+ramp|merge\s+area|diverge|lane\s+drop)\b", text):
            add("road_topology", "merge_diverge")
        if re.search(r"\b(?:work\s*zone|roadwork|road\s+works|construction\s+zone)\b", text):
            add("road_topology", "work_zone")
        if re.search(r"\b(?:closed\s+lane|lane\s+closed|partially\s+blocked\s+lane)\b", text):
            add("lane_configuration", "closed_or_blocked_lane")
        if re.search(r"\bdedicated\s+(?:left|right)[- ]turn\s+lane\b", text):
            add("lane_configuration", "dedicated_turn_lane")
        if re.search(r"\b(?:multi-lane|multilane)\s+(?:road|street)\b", text):
            add("lane_configuration", "multi_lane")

        width_match = re.search(
            r"\blane\s+width\s*(?:is|of|=|:)??\s*(\d+(?:\.\d+)?)\s*m(?:eter|eters)?\b",
            text,
        )
        if width_match:
            add("lane_width_m", float(width_match.group(1)))
        else:
            wide_lane_match = re.search(
                r"\b(\d+(?:\.\d+)?)\s*m(?:eter|eters)?[- ]wide\s+lane\b",
                text,
            )
            if wide_lane_match:
                add("lane_width_m", float(wide_lane_match.group(1)))

        if re.search(r"\bcurved\s+road\b|\broad\s+curves\b|\broad\s+curvature\b", text):
            add("road_geometry", "curved")
        elif re.search(r"\bstraight\s+road\b|\bstraight\s+roadway\b", text):
            add("road_geometry", "straight")

        return evidence

    @staticmethod
    def _deduplicate_evidence(items: Iterable[RoutingEvidence]) -> List[RoutingEvidence]:
        result: List[RoutingEvidence] = []
        seen = set()
        for item in items:
            key = (item.name, repr(item.value))
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

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


def attach_scene_construction(
    spec: Mapping[str, Any],
    *,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function for callers that do not need a router instance."""

    return SceneConstructionRouter().attach(spec, prompt=prompt)
