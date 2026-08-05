"""Recursive hierarchy for natural-language traffic-scene understanding.

The hierarchy deliberately separates two concerns:

1. linguistic detail used to explain what the prompt said, and
2. executable nuPlan/SLEDGE categories used by downstream scene generation.

For example, ``child``, ``jogger`` and ``wheelchair user`` are retained as
``language_actor_detail`` metadata, but every human-on-foot actor is projected
to the executable category ``pedestrian`` because SLEDGE stores those actors in
its ``pedestrians`` collection.

The selected hierarchy is:

    road topology -> ego traffic space -> actor group -> executable actor type
    -> hazard interaction -> auxiliary entity -> spatial relations
    -> visibility -> relative motion -> temporal trigger -> provisional response
    -> semantic risk -> executable parameters.

Every selected node contains both the legal values at its own level and the
legal values for the next level.  This avoids the ambiguous old use of
``allowed_children`` for same-level sibling candidates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from sledge.semantic_control.language.event_frame import EventFrame


UNKNOWN_VALUES = {None, "", "unknown", "unknown_side", "unspecified"}

NODE_ORDER: Tuple[str, ...] = (
    "road_topology",
    "ego_traffic_space",
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


@dataclass(frozen=True)
class HierarchyNode:
    """One selected node in the recursive parameter hierarchy."""

    level: int
    node_type: str
    value: str
    source: str = "inferred"
    confidence: float = 0.0
    evidence: str = ""
    parent_type: str = ""
    parent_value: str = ""
    allowed_values_at_level: Tuple[str, ...] = ()
    next_node_type: str = ""
    allowed_children: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["allowed_values_at_level"] = list(self.allowed_values_at_level)
        data["allowed_children"] = list(self.allowed_children)
        return data


@dataclass
class HierarchicalScenePath:
    """A root-to-leaf decision path plus validation diagnostics."""

    nodes: List[HierarchyNode] = field(default_factory=list)
    executable_parameter_groups: Dict[str, List[str]] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    valid: bool = True
    issues: List[str] = field(default_factory=list)
    schema_version: str = "eventframe_v6_nuplan_hierarchical_tree"

    @property
    def values(self) -> Dict[str, str]:
        return {node.node_type: node.value for node in self.nodes}

    def value(self, node_type: str, default: str = "unknown") -> str:
        return self.values.get(node_type, default)

    def node(self, node_type: str) -> Optional[HierarchyNode]:
        return next((node for node in self.nodes if node.node_type == node_type), None)

    def condition_path(self, *, through: Optional[str] = None) -> str:
        parts: List[str] = []
        for node in self.nodes:
            parts.append(f"{node.node_type}={node.value}")
            if through and node.node_type == through:
                break
        return " -> ".join(parts)

    def to_nested_tree(self) -> Dict[str, Any]:
        """Serialize the selected root-to-leaf path.

        This is intentionally named a selected path tree; it is not presented as
        the complete ontology.  Each node exposes its real next-level branch
        options through ``allowed_children``.
        """

        root: Dict[str, Any] = {
            "node_type": "scene_root",
            "value": "ego_centered_hazard_scene",
            "tree_kind": "selected_root_to_leaf_path",
            "children": [],
        }
        cursor = root
        for node in self.nodes:
            child = node.to_dict()
            child["children"] = []
            cursor["children"] = [child]
            cursor = child
        cursor["executable_parameter_groups"] = self.executable_parameter_groups
        return root

    def nuplan_projection(self) -> Dict[str, Any]:
        actor_type = self.value("primary_actor_type")
        auxiliary = self.value("auxiliary_entity", "none")

        if actor_type == "pedestrian":
            actor_category = "pedestrian"
            tracked_object_type = "TrackedObjectType.PEDESTRIAN"
            sledge_collection = "pedestrians"
        elif actor_type in {
            "lead_vehicle",
            "adjacent_vehicle",
            "merging_vehicle",
            "oncoming_vehicle",
            "cross_traffic_vehicle",
            "circulating_vehicle",
            "generic_vehicle",
        }:
            actor_category = "vehicle"
            tracked_object_type = "TrackedObjectType.VEHICLE"
            sledge_collection = "vehicles"
        elif actor_type == "cyclist":
            actor_category = "cyclist"
            tracked_object_type = "unsupported_by_current_sledge_builder"
            sledge_collection = "unsupported"
        else:
            actor_category = "static_object"
            tracked_object_type = "static_object"
            sledge_collection = "static_objects"

        occluder_category = "none"
        occluder_collection = "none"
        if auxiliary in {
            "parked_car_occluder",
            "parked_truck_occluder",
            "bus_occluder",
            "van_occluder",
            "generic_vehicle_occluder",
        }:
            occluder_category = "vehicle"
            occluder_collection = "vehicles"
        elif auxiliary.endswith("_occluder"):
            occluder_category = "static_object"
            occluder_collection = "static_objects"

        compatible = sledge_collection != "unsupported"
        return {
            "actor_category": actor_category,
            "tracked_object_type": tracked_object_type,
            "sledge_collection": sledge_collection,
            "language_actor_detail": self.attributes.get("language_actor_detail", "unspecified"),
            "language_actor_text": self.attributes.get("language_actor_text", ""),
            "occluder_category": occluder_category,
            "occluder_sledge_collection": occluder_collection,
            "coordinate_frame": "ego_local",
            "compatible": compatible,
            "warnings": [] if compatible else ["actor_category_requires_downstream_projection"],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "selected_path": [node.to_dict() for node in self.nodes],
            "path_values": self.values,
            "selected_tree": self.to_nested_tree(),
            "tree": self.to_nested_tree(),
            "tree_kind": "selected_root_to_leaf_path",
            "branch_options": {
                node.node_type: list(node.allowed_children) for node in self.nodes
            },
            "attributes": dict(self.attributes),
            "nuplan_projection": self.nuplan_projection(),
            "executable_parameter_groups": self.executable_parameter_groups,
            "valid": self.valid,
            "issues": list(self.issues),
        }


ROAD_TO_TRAFFIC_SPACES: Dict[str, Set[str]] = {
    "straight_segment": {
        "single_lane",
        "same_direction_multi_lane",
        "bidirectional_road",
        "curbside_zone",
        "crosswalk_zone",
    },
    "intersection": {
        "straight_through",
        "left_turn_path",
        "right_turn_path",
        "cross_traffic_zone",
    },
    "roundabout": {"entry_path", "circulating_lane", "exit_path"},
    "merge_diverge": {"ramp_merge", "lane_drop", "diverge"},
    "work_zone": {"open_lane", "partially_blocked_lane", "closed_lane"},
}

TRAFFIC_SPACE_TO_ACTOR_GROUPS: Dict[str, Set[str]] = {
    key: {"vehicle", "static_object", "vulnerable_road_user"}
    for key in {
        "single_lane",
        "bidirectional_road",
        "curbside_zone",
        "straight_through",
        "left_turn_path",
        "right_turn_path",
        "cross_traffic_zone",
        "entry_path",
        "exit_path",
        "partially_blocked_lane",
    }
}
TRAFFIC_SPACE_TO_ACTOR_GROUPS.update(
    {
        "same_direction_multi_lane": {"vehicle", "static_object"},
        "crosswalk_zone": {"vehicle", "vulnerable_road_user"},
        "circulating_lane": {"vehicle"},
        "ramp_merge": {"vehicle"},
        "lane_drop": {"vehicle", "static_object"},
        "diverge": {"vehicle"},
        "open_lane": {"vehicle", "static_object"},
        "closed_lane": {"static_object"},
    }
)

ACTOR_GROUP_TO_TYPES: Dict[str, Set[str]] = {
    "vulnerable_road_user": {"pedestrian", "cyclist"},
    "vehicle": {
        "lead_vehicle",
        "adjacent_vehicle",
        "merging_vehicle",
        "oncoming_vehicle",
        "cross_traffic_vehicle",
        "circulating_vehicle",
        "generic_vehicle",
    },
    "static_object": {
        "barrier",
        "traffic_cone",
        "debris",
        "parked_vehicle",
        "generic_obstacle",
    },
}

ACTOR_TYPE_TO_INTERACTIONS: Dict[str, Set[str]] = {
    "pedestrian": {
        "path_crossing",
        "enter_ego_lane",
        "occluded_emergence",
        "longitudinal_occupation",
    },
    "cyclist": {
        "path_crossing",
        "enter_ego_lane",
        "occluded_emergence",
        "longitudinal_occupation",
    },
    "lead_vehicle": {"gradual_braking", "hard_braking", "sudden_stop", "stationary_lead"},
    "adjacent_vehicle": {"lane_change", "cut_in", "aggressive_cut_in", "lane_encroachment"},
    "merging_vehicle": {"lane_change", "cut_in", "aggressive_cut_in", "roundabout_entry_conflict"},
    "oncoming_vehicle": {"left_turn_across_oncoming", "oncoming_path_conflict", "wrong_way_approach"},
    "cross_traffic_vehicle": {"intersection_crossing", "red_light_intrusion", "unprotected_crossing"},
    "circulating_vehicle": {"roundabout_entry_conflict"},
    "generic_vehicle": {
        "lane_change",
        "cut_in",
        "path_crossing",
        "intersection_crossing",
        "oncoming_path_conflict",
    },
    "barrier": {"lane_blocking", "partial_lane_occupation"},
    "traffic_cone": {"lane_blocking", "partial_lane_occupation"},
    "debris": {"lane_blocking", "partial_lane_occupation", "sudden_obstacle_appearance"},
    "parked_vehicle": {"lane_blocking", "partial_lane_occupation"},
    "generic_obstacle": {"lane_blocking", "partial_lane_occupation", "sudden_obstacle_appearance"},
}

INTERACTION_TO_AUXILIARIES: Dict[str, Set[str]] = {
    "occluded_emergence": {
        "parked_car_occluder",
        "parked_truck_occluder",
        "bus_occluder",
        "van_occluder",
        "barrier_occluder",
        "vegetation_occluder",
        "building_edge_occluder",
        "generic_vehicle_occluder",
    },
    "path_crossing": {"none", "crosswalk", "traffic_light"},
    "enter_ego_lane": {"none", "curb", "crosswalk"},
    "longitudinal_occupation": {"none", "lane_marking"},
    "gradual_braking": {"none", "traffic_flow", "traffic_light"},
    "hard_braking": {"none", "traffic_flow", "traffic_light"},
    "sudden_stop": {"none", "traffic_flow", "traffic_light"},
    "stationary_lead": {"none", "traffic_flow"},
    "lane_change": {"none", "lane_marking", "traffic_flow"},
    "cut_in": {"none", "lane_marking", "traffic_flow"},
    "aggressive_cut_in": {"none", "lane_marking", "traffic_flow"},
    "lane_encroachment": {"none", "lane_marking"},
    "left_turn_across_oncoming": {"none", "traffic_light", "intersection_center"},
    "oncoming_path_conflict": {"none", "lane_marking", "median"},
    "wrong_way_approach": {"none", "lane_marking", "median"},
    "intersection_crossing": {"none", "traffic_light", "stop_sign"},
    "red_light_intrusion": {"traffic_light"},
    "unprotected_crossing": {"none", "stop_sign"},
    "roundabout_entry_conflict": {"none", "yield_sign", "circulating_traffic"},
    "lane_blocking": {"none", "traffic_cone", "temporary_barrier"},
    "partial_lane_occupation": {"none", "traffic_cone", "temporary_barrier"},
    "sudden_obstacle_appearance": {"none"},
}

INTERACTION_TO_SOURCE_REGIONS: Dict[str, Set[str]] = {
    "path_crossing": {"left_side", "right_side", "curbside", "crosswalk_side"},
    "enter_ego_lane": {"left_side", "right_side", "curbside"},
    "occluded_emergence": {"left_side", "right_side", "curbside"},
    "longitudinal_occupation": {"front_same_lane", "curbside"},
    "gradual_braking": {"front_same_lane"},
    "hard_braking": {"front_same_lane"},
    "sudden_stop": {"front_same_lane"},
    "stationary_lead": {"front_same_lane"},
    "lane_change": {"adjacent_left_lane", "adjacent_right_lane"},
    "cut_in": {"adjacent_left_lane", "adjacent_right_lane"},
    "aggressive_cut_in": {"adjacent_left_lane", "adjacent_right_lane"},
    "lane_encroachment": {"adjacent_left_lane", "adjacent_right_lane"},
    "left_turn_across_oncoming": {"opposite_lane"},
    "oncoming_path_conflict": {"opposite_lane"},
    "wrong_way_approach": {"opposite_lane"},
    "intersection_crossing": {"left_side", "right_side", "cross_traffic_lane"},
    "red_light_intrusion": {"cross_traffic_lane"},
    "unprotected_crossing": {"cross_traffic_lane"},
    "roundabout_entry_conflict": {"circulating_lane"},
    "lane_blocking": {"front_same_lane"},
    "partial_lane_occupation": {"front_same_lane"},
    "sudden_obstacle_appearance": {"front_same_lane"},
}

INTERACTION_TO_TARGET_REGIONS: Dict[str, Set[str]] = {
    "path_crossing": {"ego_lane", "ego_path"},
    "enter_ego_lane": {"ego_lane"},
    "occluded_emergence": {"ego_lane", "ego_path"},
    "longitudinal_occupation": {"ego_lane", "same_lane"},
    "gradual_braking": {"same_lane"},
    "hard_braking": {"same_lane"},
    "sudden_stop": {"same_lane"},
    "stationary_lead": {"same_lane"},
    "lane_change": {"ego_lane"},
    "cut_in": {"ego_lane"},
    "aggressive_cut_in": {"ego_lane"},
    "lane_encroachment": {"ego_lane"},
    "left_turn_across_oncoming": {"left_turn_path"},
    "oncoming_path_conflict": {"ego_path"},
    "wrong_way_approach": {"ego_lane"},
    "intersection_crossing": {"intersection_conflict_zone", "ego_path"},
    "red_light_intrusion": {"intersection_conflict_zone"},
    "unprotected_crossing": {"intersection_conflict_zone"},
    "roundabout_entry_conflict": {"roundabout_entry"},
    "lane_blocking": {"ego_lane"},
    "partial_lane_occupation": {"ego_lane"},
    "sudden_obstacle_appearance": {"ego_lane"},
}

INTERACTION_TO_VISIBILITY: Dict[str, Set[str]] = {
    "occluded_emergence": {"partially_occluded", "fully_occluded"},
}

INTERACTION_TO_MOTION_DIRECTIONS: Dict[str, Set[str]] = {
    "occluded_emergence": {"occluder_to_ego_path"},
    "path_crossing": {"left_to_right", "right_to_left", "toward_ego_path"},
    "enter_ego_lane": {"left_to_right", "right_to_left", "toward_ego_path"},
    "longitudinal_occupation": {"longitudinal_same_direction"},
    "gradual_braking": {"longitudinal_same_direction"},
    "hard_braking": {"longitudinal_same_direction"},
    "sudden_stop": {"longitudinal_same_direction"},
    "stationary_lead": {"stationary"},
    "lane_change": {"into_ego_lane"},
    "cut_in": {"into_ego_lane"},
    "aggressive_cut_in": {"into_ego_lane"},
    "lane_encroachment": {"into_ego_lane"},
    "left_turn_across_oncoming": {"opposite_direction"},
    "oncoming_path_conflict": {"opposite_direction"},
    "wrong_way_approach": {"opposite_direction"},
    "intersection_crossing": {"toward_ego_path"},
    "red_light_intrusion": {"toward_ego_path"},
    "unprotected_crossing": {"toward_ego_path"},
    "roundabout_entry_conflict": {"circulating_across_entry"},
    "lane_blocking": {"stationary"},
    "partial_lane_occupation": {"stationary"},
    "sudden_obstacle_appearance": {"stationary"},
}

INTERACTION_TO_TRIGGERS: Dict[str, Set[str]] = {
    "occluded_emergence": {"occluded_actor_becomes_visible"},
    "path_crossing": {"actor_starts_moving", "actor_enters_ego_lane"},
    "enter_ego_lane": {"actor_enters_ego_lane"},
    "longitudinal_occupation": {"ego_reaches_conflict_area"},
    "gradual_braking": {"lead_vehicle_decelerates"},
    "hard_braking": {"lead_vehicle_decelerates"},
    "sudden_stop": {"lead_vehicle_decelerates"},
    "stationary_lead": {"ego_reaches_conflict_area"},
    "lane_change": {"actor_crosses_lane_boundary"},
    "cut_in": {"actor_crosses_lane_boundary"},
    "aggressive_cut_in": {"actor_crosses_lane_boundary"},
    "lane_encroachment": {"actor_crosses_lane_boundary"},
    "left_turn_across_oncoming": {"ego_reaches_conflict_area"},
    "oncoming_path_conflict": {"ego_reaches_conflict_area"},
    "wrong_way_approach": {"ego_reaches_conflict_area"},
    "intersection_crossing": {"actor_enters_ego_lane", "ego_reaches_conflict_area"},
    "red_light_intrusion": {"actor_enters_ego_lane"},
    "unprotected_crossing": {"actor_enters_ego_lane"},
    "roundabout_entry_conflict": {"ego_reaches_conflict_area"},
    "lane_blocking": {"ego_reaches_conflict_area"},
    "partial_lane_occupation": {"ego_reaches_conflict_area"},
    "sudden_obstacle_appearance": {"ego_reaches_conflict_area"},
}

INTERACTION_TO_RESPONSES: Dict[str, Set[str]] = {
    "path_crossing": {"brake", "yield", "brake_or_emergency_brake"},
    "enter_ego_lane": {"brake", "yield", "brake_or_emergency_brake"},
    "occluded_emergence": {"brake_or_emergency_brake", "steer"},
    "longitudinal_occupation": {"brake", "yield", "steer"},
    "gradual_braking": {"brake"},
    "hard_braking": {"brake_or_emergency_brake"},
    "sudden_stop": {"brake_or_emergency_brake", "steer"},
    "stationary_lead": {"brake", "stop", "steer"},
    "lane_change": {"brake", "yield"},
    "cut_in": {"brake", "yield", "steer"},
    "aggressive_cut_in": {"brake_or_emergency_brake", "steer"},
    "lane_encroachment": {"brake", "steer"},
    "left_turn_across_oncoming": {"yield", "stop", "brake"},
    "oncoming_path_conflict": {"brake", "steer", "stop"},
    "wrong_way_approach": {"brake_or_emergency_brake", "steer", "stop"},
    "intersection_crossing": {"brake", "yield", "stop"},
    "red_light_intrusion": {"brake_or_emergency_brake", "stop"},
    "unprotected_crossing": {"yield", "brake", "stop"},
    "roundabout_entry_conflict": {"yield", "stop", "brake"},
    "lane_blocking": {"brake", "stop", "steer"},
    "partial_lane_occupation": {"brake", "steer"},
    "sudden_obstacle_appearance": {"brake_or_emergency_brake", "steer"},
}

EXECUTABLE_PARAMETER_GROUPS: Dict[str, List[str]] = {
    "road_parameters": ["lane_width_m", "lane_count", "road_curvature", "conflict_point_xy"],
    "ego_parameters": ["ego_speed_mps", "ego_acceleration_mps2", "ego_distance_to_conflict_m"],
    "actor_parameters": [
        "actor_speed_mps",
        "actor_acceleration_mps2",
        "actor_initial_position",
        "actor_heading",
        "actor_start_time_s",
    ],
    "auxiliary_parameters": [
        "occluder_position",
        "occluder_length_m",
        "occluder_width_m",
        "occluder_lateral_offset_m",
    ],
    "risk_parameters": [
        "reveal_distance_m",
        "initial_gap_m",
        "time_to_collision_s",
        "minimum_clearance_m",
        "braking_deceleration_mps2",
    ],
}


def _contains(text: str, terms: Sequence[str]) -> bool:
    value = (text or "").lower()
    return any(term in value for term in terms)


def _clean(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        return default if normalized in UNKNOWN_VALUES else normalized or default
    if isinstance(value, (list, tuple, set)):
        return _clean(next(iter(value)), default) if len(value) == 1 else default
    return str(value).strip().lower().replace("-", "_").replace(" ", "_") or default


def _first_allowed(candidate: str, allowed: Iterable[str], fallback: str) -> str:
    options = set(allowed)
    if candidate in options:
        return candidate
    if fallback in options:
        return fallback
    return sorted(options)[0] if options else candidate


def _span(sentence: str, terms: Sequence[str]) -> str:
    lower = (sentence or "").lower()
    for term in terms:
        index = lower.find(term.lower())
        if index >= 0:
            return sentence[index : index + len(term)]
    return ""


class HierarchicalSceneResolver:
    """Resolve a flat EventFrame spec into one constrained root-to-leaf path."""

    def resolve(self, frame: EventFrame, spec: Mapping[str, Any]) -> HierarchicalScenePath:
        slots = dict(spec.get("semantic_slots", {}) or {})
        sentence = frame.sentence or str(spec.get("evidence", {}).get("sentence", ""))

        road, road_source, road_conf = self._road_topology(frame, slots, sentence)
        traffic_space, traffic_source, traffic_conf = self._traffic_space(frame, slots, sentence, road)
        actor_group, actor_type, actor_detail, actor_source, actor_conf = self._actor(
            frame, slots, sentence, traffic_space
        )
        interaction, interaction_source, interaction_conf = self._interaction(
            frame, slots, sentence, actor_type
        )
        auxiliary, auxiliary_source, auxiliary_conf = self._auxiliary(
            frame, slots, sentence, interaction
        )
        source_region, source_source, source_conf = self._source_region(
            frame, slots, sentence, interaction
        )
        target_region, target_source, target_conf = self._target_region(
            frame, slots, sentence, interaction
        )
        anchor_region, anchor_source, anchor_conf = self._anchor_region(
            frame, slots, sentence, interaction
        )
        visibility, visibility_source, visibility_conf = self._visibility(
            frame, slots, sentence, interaction
        )
        motion_direction, direction_source, direction_conf = self._motion_direction(
            frame, slots, sentence, interaction, source_region
        )
        trigger, trigger_source, trigger_conf = self._trigger(frame, slots, sentence, interaction)
        response, response_source, response_conf = self._response(frame, slots, sentence, interaction)
        risk, risk_source, risk_conf = self._risk(frame, slots, sentence, interaction)

        values = {
            "road_topology": road,
            "ego_traffic_space": traffic_space,
            "primary_actor_group": actor_group,
            "primary_actor_type": actor_type,
            "hazard_interaction": interaction,
            "auxiliary_entity": auxiliary,
            "source_region": source_region,
            "target_region": target_region,
            "anchor_region": anchor_region,
            "visibility": visibility,
            "motion_direction": motion_direction,
            "trigger_event": trigger,
            "ego_required_response": response,
            "risk_level": risk,
        }
        source_meta = {
            "road_topology": (road_source, road_conf),
            "ego_traffic_space": (traffic_source, traffic_conf),
            "primary_actor_group": (actor_source, actor_conf),
            "primary_actor_type": (actor_source, actor_conf),
            "hazard_interaction": (interaction_source, interaction_conf),
            "auxiliary_entity": (auxiliary_source, auxiliary_conf),
            "source_region": (source_source, source_conf),
            "target_region": (target_source, target_conf),
            "anchor_region": (anchor_source, anchor_conf),
            "visibility": (visibility_source, visibility_conf),
            "motion_direction": (direction_source, direction_conf),
            "trigger_event": (trigger_source, trigger_conf),
            "ego_required_response": (response_source, response_conf),
            "risk_level": (risk_source, risk_conf),
        }

        path = HierarchicalScenePath(
            executable_parameter_groups=EXECUTABLE_PARAMETER_GROUPS,
            attributes={
                "language_actor_detail": actor_detail,
                "language_actor_text": frame.main_actor.text,
                "absolute_direction_hint": self._absolute_direction_hint(source_region),
                "response_status": (
                    "provisional_until_kinematic_check"
                    if response == "brake_or_emergency_brake"
                    else "resolved_from_semantics"
                ),
            },
        )

        for index, node_type in enumerate(NODE_ORDER):
            level = index + 1
            parent_type = "scene_root" if index == 0 else NODE_ORDER[index - 1]
            parent_value = "ego_centered_hazard_scene" if index == 0 else values[parent_type]
            next_node_type = NODE_ORDER[index + 1] if index + 1 < len(NODE_ORDER) else "executable_parameters"
            allowed_at_level = tuple(
                sorted(self.allowed_values(node_type, parent_type, parent_value, values))
            )
            allowed_children = tuple(
                sorted(self.allowed_values(next_node_type, node_type, values[node_type], values))
            )
            source, confidence = source_meta[node_type]
            path.nodes.append(
                HierarchyNode(
                    level=level,
                    node_type=node_type,
                    value=values[node_type],
                    source=source,
                    confidence=max(0.0, min(float(confidence), 1.0)),
                    evidence=self._node_evidence(frame, node_type, values[node_type]),
                    parent_type=parent_type,
                    parent_value=parent_value,
                    allowed_values_at_level=allowed_at_level,
                    next_node_type=next_node_type,
                    allowed_children=allowed_children,
                )
            )

        path.issues = self.validate(path)
        path.valid = not path.issues
        return path

    def allowed_values(
        self,
        node_type: str,
        parent_type: str,
        parent_value: str,
        selected: Mapping[str, str],
    ) -> Set[str]:
        if node_type == "road_topology":
            return set(ROAD_TO_TRAFFIC_SPACES)
        if node_type == "ego_traffic_space":
            return set(ROAD_TO_TRAFFIC_SPACES.get(parent_value, set()))
        if node_type == "primary_actor_group":
            return set(TRAFFIC_SPACE_TO_ACTOR_GROUPS.get(parent_value, set()))
        if node_type == "primary_actor_type":
            return set(ACTOR_GROUP_TO_TYPES.get(parent_value, set()))
        if node_type == "hazard_interaction":
            return set(ACTOR_TYPE_TO_INTERACTIONS.get(parent_value, set()))
        if node_type == "auxiliary_entity":
            return set(INTERACTION_TO_AUXILIARIES.get(parent_value, {"none"}))

        interaction = selected.get("hazard_interaction", "unknown")
        if node_type == "source_region":
            return set(INTERACTION_TO_SOURCE_REGIONS.get(interaction, {"front_same_lane"}))
        if node_type == "target_region":
            return set(INTERACTION_TO_TARGET_REGIONS.get(interaction, {"ego_path"}))
        if node_type == "anchor_region":
            return {
                "near_front",
                "far_front",
                "intersection_center",
                "lane_boundary",
                "crosswalk",
                "roundabout_entry",
            }
        if node_type == "visibility":
            return set(
                INTERACTION_TO_VISIBILITY.get(
                    interaction, {"fully_visible", "partially_occluded", "fully_occluded"}
                )
            )
        if node_type == "motion_direction":
            return set(INTERACTION_TO_MOTION_DIRECTIONS.get(interaction, {"toward_ego_path"}))
        if node_type == "trigger_event":
            return set(INTERACTION_TO_TRIGGERS.get(interaction, {"ego_reaches_conflict_area"}))
        if node_type == "ego_required_response":
            return set(INTERACTION_TO_RESPONSES.get(interaction, {"brake"}))
        if node_type == "risk_level":
            return {"mild", "moderate", "aggressive", "critical"}
        if node_type == "executable_parameters":
            return set(EXECUTABLE_PARAMETER_GROUPS)
        return set()

    def allowed_children(
        self,
        parent_type: str,
        parent_value: str,
        child_type: str,
        selected: Mapping[str, str],
    ) -> Set[str]:
        return self.allowed_values(child_type, parent_type, parent_value, selected)

    def validate(self, path: HierarchicalScenePath) -> List[str]:
        issues: List[str] = []
        values = path.values

        for index, node in enumerate(path.nodes):
            expected_type = NODE_ORDER[index]
            if node.node_type != expected_type:
                issues.append(f"hierarchy_order_mismatch:{index + 1}:{node.node_type}!={expected_type}")
            if node.value not in set(node.allowed_values_at_level):
                issues.append(f"invalid_value_at_level:{node.node_type}={node.value}")
            if index + 1 < len(path.nodes):
                child = path.nodes[index + 1]
                if child.value not in set(node.allowed_children):
                    issues.append(
                        f"invalid_transition:{node.node_type}={node.value}"
                        f"->{child.node_type}={child.value}"
                    )

        interaction = values.get("hazard_interaction", "unknown")
        visibility = values.get("visibility", "unknown")
        auxiliary = values.get("auxiliary_entity", "none")
        direction = values.get("motion_direction", "unknown")
        actor_type = values.get("primary_actor_type", "unknown")

        if actor_type not in ACTOR_GROUP_TO_TYPES.get(values.get("primary_actor_group", ""), set()):
            issues.append("actor_type_not_nuplan_projection_category")
        if interaction == "occluded_emergence":
            if visibility == "fully_visible":
                issues.append("occluded_emergence_requires_occluded_visibility")
            if not auxiliary.endswith("_occluder"):
                issues.append("occluded_emergence_requires_occluder")
            if direction != "occluder_to_ego_path":
                issues.append("occluded_emergence_requires_unique_occluder_to_ego_path_direction")
        return issues

    @staticmethod
    def _road_topology(frame: EventFrame, slots: Mapping[str, Any], sentence: str) -> Tuple[str, str, float]:
        raw = _clean(slots.get("road_topology", frame.road_context.road_type))
        if raw == "intersection" or _contains(sentence, ["intersection", "junction"]):
            return "intersection", "explicit" if raw == "intersection" else "inferred", 0.95
        if raw == "roundabout" or _contains(sentence, ["roundabout", "rotary", "traffic circle"]):
            return "roundabout", "explicit" if raw == "roundabout" else "inferred", 0.97
        if raw in {"construction_zone", "work_zone"} or _contains(sentence, ["construction", "work zone", "roadwork"]):
            return "work_zone", "normalized" if raw != "unknown" else "inferred", 0.94
        if raw in {"merge_diverge", "ramp_merge"} or _contains(sentence, ["ramp", "merge area", "lane drop", "diverge"]):
            return "merge_diverge", "inferred", 0.82
        return "straight_segment", "normalized" if raw != "unknown" else "hierarchical_default", 0.9 if raw != "unknown" else 0.72

    @staticmethod
    def _traffic_space(
        frame: EventFrame,
        slots: Mapping[str, Any],
        sentence: str,
        road: str,
    ) -> Tuple[str, str, float]:
        motion = _clean(slots.get("motion_geometry"))
        layout = _clean(slots.get("road_layout", frame.road_context.lane_context))
        source = _clean(slots.get("source_side", frame.main_event.source_relation))
        if road == "intersection":
            if frame.ego_event.ego_maneuver == "left_turn" or _contains(sentence, ["left turn", "turns left"]):
                return "left_turn_path", "explicit", 0.96
            if motion == "lateral" and _clean(slots.get("actor_type")) == "vehicle":
                return "cross_traffic_zone", "inferred", 0.86
            return "straight_through", "hierarchical_default", 0.65
        if road == "roundabout":
            return ("entry_path", "inferred", 0.95) if layout == "roundabout_entry" else ("circulating_lane", "hierarchical_default", 0.62)
        if road == "merge_diverge":
            return "ramp_merge", "inferred", 0.82
        if road == "work_zone":
            return ("closed_lane", "explicit", 0.92) if _contains(sentence, ["closed lane", "lane closed"]) else ("partially_blocked_lane", "inferred", 0.8)
        if _contains(sentence, ["crosswalk", "zebra crossing"]):
            return "crosswalk_zone", "explicit", 0.94
        if source in {"curbside", "from_curb"} or _contains(sentence, ["curb", "roadside", "sidewalk", "parked car", "parked truck"]):
            return "curbside_zone", "inferred", 0.84
        if motion == "merging" or layout in {"adjacent_lane_cut_in", "multi_lane_road"}:
            return "same_direction_multi_lane", "inferred", 0.9
        if motion == "oncoming" or _contains(sentence, ["oncoming", "opposite lane", "opposing"]):
            return "bidirectional_road", "inferred", 0.9
        return "single_lane", "hierarchical_default", 0.62

    @staticmethod
    def _actor(
        frame: EventFrame,
        slots: Mapping[str, Any],
        sentence: str,
        traffic_space: str,
    ) -> Tuple[str, str, str, str, float]:
        actor_base = _clean(slots.get("actor_type"))
        actor_text = " ".join([frame.main_actor.text, sentence]).lower()
        role = _clean(slots.get("actor_role"))
        motion = _clean(slots.get("motion_geometry"))

        if actor_base == "pedestrian" or frame.main_actor.actor_class == "human_on_foot":
            detail = "child" if _contains(actor_text, ["child", "kid", "schoolkid", "boy", "girl"]) else "jogger" if _contains(actor_text, ["jogger", "runner", "running person"]) else "wheelchair_user" if _contains(actor_text, ["wheelchair"]) else "adult_or_unspecified"
            return "vulnerable_road_user", "pedestrian", detail, "normalized_to_nuplan", 0.99
        if actor_base == "cyclist" or frame.main_actor.actor_class == "cyclist":
            detail = "ebike" if _contains(actor_text, ["e-bike", "ebike", "electric bicycle"]) else "scooter" if _contains(actor_text, ["scooter"]) else "bicycle"
            return "vulnerable_road_user", "cyclist", detail, "normalized", 0.94
        if actor_base in {"traffic_object", "static_obstacle"} or frame.main_actor.actor_class == "traffic_object":
            if _contains(actor_text, ["cone"]):
                actor_type = "traffic_cone"
            elif _contains(actor_text, ["barrier"]):
                actor_type = "barrier"
            elif _contains(actor_text, ["debris"]):
                actor_type = "debris"
            elif _contains(actor_text, ["parked vehicle", "parked car"]):
                actor_type = "parked_vehicle"
            else:
                actor_type = "generic_obstacle"
            return "static_object", actor_type, "none", "inferred", 0.88
        if motion == "longitudinal" or role == "braking_actor":
            return "vehicle", "lead_vehicle", "none", "inferred", 0.95
        if motion == "oncoming" or role == "approaching_actor":
            return "vehicle", "oncoming_vehicle", "none", "inferred", 0.94
        if traffic_space == "cross_traffic_zone" or _contains(sentence, ["cross traffic", "from the side road"]):
            return "vehicle", "cross_traffic_vehicle", "none", "inferred", 0.87
        if traffic_space in {"entry_path", "ramp_merge"} or _contains(sentence, ["merging vehicle", "merge into"]):
            return "vehicle", "merging_vehicle", "none", "inferred", 0.9
        if motion == "merging" or role == "merging_actor":
            return "vehicle", "adjacent_vehicle", "none", "inferred", 0.91
        return "vehicle", "generic_vehicle", "none", "hierarchical_default", 0.58

    @staticmethod
    def _interaction(
        frame: EventFrame,
        slots: Mapping[str, Any],
        sentence: str,
        actor_type: str,
    ) -> Tuple[str, str, float]:
        motion = _clean(slots.get("motion_geometry", frame.main_event.motion_axis))
        event = _clean(slots.get("fine_grained_conflict_type", frame.main_event.event_type))
        risk = _clean(slots.get("risk_level"), "moderate")
        occluded = bool(slots.get("occlusion_enabled", frame.occlusion.enabled))
        allowed = ACTOR_TYPE_TO_INTERACTIONS.get(actor_type, set())

        if event == "roundabout_entry_conflict" and "roundabout_entry_conflict" in allowed:
            return "roundabout_entry_conflict", "inferred", 0.97
        if occluded and "occluded_emergence" in allowed:
            explicit = frame.occlusion.enabled or _contains(sentence, ["behind", "occluded", "emerges", "appears from"])
            return "occluded_emergence", "explicit" if explicit else "inferred", 0.98
        if actor_type in {"pedestrian", "cyclist"}:
            candidate = "enter_ego_lane" if event == "enter_ego_lane" else "path_crossing"
            return _first_allowed(candidate, allowed, "path_crossing"), "inferred", 0.93
        if actor_type == "lead_vehicle":
            if event == "hard_stop_ahead" or _contains(sentence, ["hard brake", "brakes hard", "panic", "slams", "suddenly stops"]):
                return "hard_braking", "explicit" if _contains(sentence, ["hard", "panic", "slams", "suddenly"]) else "inferred", 0.95
            return "gradual_braking", "hierarchical_default", 0.73
        if actor_type in {"adjacent_vehicle", "merging_vehicle", "generic_vehicle"} and motion == "merging":
            candidate = "aggressive_cut_in" if risk in {"aggressive", "critical"} or _contains(sentence, ["aggressive", "tight gap", "no room", "squeezes"]) else "cut_in"
            return _first_allowed(candidate, allowed, "lane_change"), "inferred", 0.9
        if actor_type in {"oncoming_vehicle", "generic_vehicle"} and motion == "oncoming":
            candidate = "left_turn_across_oncoming" if frame.ego_event.ego_maneuver == "left_turn" or _contains(sentence, ["left turn", "turns left"]) else "oncoming_path_conflict"
            return _first_allowed(candidate, allowed, "oncoming_path_conflict"), "inferred", 0.94
        if actor_type == "cross_traffic_vehicle" or (motion == "lateral" and actor_type.endswith("vehicle")):
            return _first_allowed("intersection_crossing", allowed, "path_crossing"), "inferred", 0.84
        if actor_type == "circulating_vehicle":
            return "roundabout_entry_conflict", "inferred", 0.96
        if actor_type in ACTOR_GROUP_TO_TYPES["static_object"] or motion == "static":
            return _first_allowed("lane_blocking", allowed, "partial_lane_occupation"), "inferred", 0.9
        return (sorted(allowed)[0], "hierarchical_default", 0.4) if allowed else ("path_crossing", "hierarchical_default", 0.4)

    @staticmethod
    def _auxiliary(
        frame: EventFrame,
        slots: Mapping[str, Any],
        sentence: str,
        interaction: str,
    ) -> Tuple[str, str, float]:
        allowed = INTERACTION_TO_AUXILIARIES.get(interaction, {"none"})
        if interaction == "occluded_emergence":
            raw = _clean(slots.get("occluder_type", frame.occlusion.occluder_type))
            if raw == "truck" or _contains(sentence, ["truck", "lorry"]):
                candidate = "parked_truck_occluder"
            elif raw == "bus" or _contains(sentence, ["bus"]):
                candidate = "bus_occluder"
            elif raw == "van" or _contains(sentence, ["van"]):
                candidate = "van_occluder"
            elif raw == "parked_vehicle" or _contains(sentence, ["parked car", "parked vehicle"]):
                candidate = "parked_car_occluder"
            elif _contains(sentence, ["barrier", "wall"]):
                candidate = "barrier_occluder"
            elif _contains(sentence, ["tree", "bush", "vegetation"]):
                candidate = "vegetation_occluder"
            elif _contains(sentence, ["building", "corner"]):
                candidate = "building_edge_occluder"
            else:
                candidate = "generic_vehicle_occluder"
            explicit = raw != "unknown" or bool(_span(sentence, ["parked truck", "truck", "parked car", "bus", "van", "barrier", "tree", "building"]))
            return _first_allowed(candidate, allowed, "generic_vehicle_occluder"), "normalized_explicit" if explicit else "hierarchical_default", 0.98 if explicit else 0.62
        if _contains(sentence, ["crosswalk", "zebra crossing"]) and "crosswalk" in allowed:
            return "crosswalk", "explicit", 0.96
        if _contains(sentence, ["traffic light", "signal"]) and "traffic_light" in allowed:
            return "traffic_light", "explicit", 0.93
        if _contains(sentence, ["cone"]) and "traffic_cone" in allowed:
            return "traffic_cone", "explicit", 0.95
        if _contains(sentence, ["barrier"]) and "temporary_barrier" in allowed:
            return "temporary_barrier", "explicit", 0.93
        return _first_allowed("none", allowed, sorted(allowed)[0]), "hierarchical_default", 0.7

    @staticmethod
    def _source_region(
        frame: EventFrame,
        slots: Mapping[str, Any],
        sentence: str,
        interaction: str,
    ) -> Tuple[str, str, float]:
        allowed = INTERACTION_TO_SOURCE_REGIONS[interaction]
        relation = _clean(slots.get("source_side", frame.main_event.source_relation))
        mapping = {
            "left": "left_side",
            "from_left": "left_side",
            "right": "right_side",
            "from_right": "right_side",
            "curbside": "curbside",
            "from_curb": "curbside",
            "front": "front_same_lane",
            "opposite": "opposite_lane",
            "from_opposite_direction": "opposite_lane",
            "roundabout_inside": "circulating_lane",
            "from_circulating_lane": "circulating_lane",
        }
        candidate = mapping.get(relation, "unknown")
        if candidate == "unknown":
            if _contains(sentence, ["from the left", "left side", "left curb"]):
                candidate = "left_side"
            elif _contains(sentence, ["from the right", "right side", "right curb"]):
                candidate = "right_side"
            elif _contains(sentence, ["curb", "roadside", "sidewalk", "parked truck", "parked car"]):
                candidate = "curbside"
        if candidate in allowed:
            return candidate, "explicit" if relation != "unknown" or candidate in {"left_side", "right_side"} else "inferred", 0.9
        return sorted(allowed)[0], "hierarchical_default", 0.55

    @staticmethod
    def _target_region(
        frame: EventFrame,
        slots: Mapping[str, Any],
        sentence: str,
        interaction: str,
    ) -> Tuple[str, str, float]:
        allowed = INTERACTION_TO_TARGET_REGIONS[interaction]
        raw = _clean(slots.get("target_path", frame.main_event.target_relation))
        mapping = {
            "ego_lane": "ego_lane",
            "ego_path": "ego_path",
            "same_lane": "same_lane",
            "ego_turn_path": "left_turn_path",
            "intersection": "intersection_conflict_zone",
            "roundabout_entry": "roundabout_entry",
        }
        candidate = mapping.get(raw, raw)
        if candidate in allowed:
            return candidate, "explicit" if raw != "unknown" else "inferred", 0.92
        fallback = "ego_lane" if "ego_lane" in allowed else sorted(allowed)[0]
        return fallback, "hierarchical_default", 0.65

    @staticmethod
    def _anchor_region(
        frame: EventFrame,
        slots: Mapping[str, Any],
        sentence: str,
        interaction: str,
    ) -> Tuple[str, str, float]:
        raw = _clean(slots.get("anchor_region", frame.main_event.event_location_relation))
        if raw in {"intersection", "at_intersection"} or interaction in {"left_turn_across_oncoming", "intersection_crossing", "red_light_intrusion", "unprotected_crossing"}:
            return "intersection_center", "inferred", 0.9
        if raw == "roundabout_entry" or interaction == "roundabout_entry_conflict":
            return "roundabout_entry", "inferred", 0.95
        if _contains(sentence, ["crosswalk"]):
            return "crosswalk", "explicit", 0.94
        if interaction in {"lane_change", "cut_in", "aggressive_cut_in", "lane_encroachment"}:
            return "lane_boundary", "inferred", 0.9
        if _contains(sentence, ["just ahead", "near ego", "close ahead", "directly ahead"]):
            return "near_front", "explicit", 0.9
        if _contains(sentence, ["far ahead", "in the distance"]):
            return "far_front", "explicit", 0.9
        if raw in {"front", "ahead_of", "in_front_of"}:
            return "near_front", "inferred", 0.72
        return "near_front", "hierarchical_default", 0.58

    @staticmethod
    def _visibility(
        frame: EventFrame,
        slots: Mapping[str, Any],
        sentence: str,
        interaction: str,
    ) -> Tuple[str, str, float]:
        occluded = bool(slots.get("occlusion_enabled", frame.occlusion.enabled))
        if interaction == "occluded_emergence" or occluded:
            if _contains(sentence, ["partial", "partially"]):
                return "partially_occluded", "explicit", 0.96
            explicit = frame.occlusion.enabled or _contains(sentence, ["behind", "hidden", "occluded"])
            return "fully_occluded", "explicit" if explicit else "inferred", 0.96 if explicit else 0.9
        return "fully_visible", "hierarchical_default", 0.8

    @staticmethod
    def _motion_direction(
        frame: EventFrame,
        slots: Mapping[str, Any],
        sentence: str,
        interaction: str,
        source_region: str,
    ) -> Tuple[str, str, float]:
        if interaction == "occluded_emergence":
            return "occluder_to_ego_path", "geometric_constraint", 1.0
        raw = _clean(slots.get("conflict_direction", frame.main_event.motion_direction))
        if raw in {"left_to_right", "right_to_left"}:
            return raw, "explicit", 0.95
        if source_region == "left_side":
            return "left_to_right", "inferred", 0.86
        if source_region == "right_side":
            return "right_to_left", "inferred", 0.86
        return sorted(INTERACTION_TO_MOTION_DIRECTIONS.get(interaction, {"toward_ego_path"}))[0], "inferred", 0.88

    @staticmethod
    def _trigger(
        frame: EventFrame,
        slots: Mapping[str, Any],
        sentence: str,
        interaction: str,
    ) -> Tuple[str, str, float]:
        allowed = INTERACTION_TO_TRIGGERS[interaction]
        if interaction == "occluded_emergence":
            return "occluded_actor_becomes_visible", "inferred", 0.96
        if interaction in {"gradual_braking", "hard_braking", "sudden_stop"}:
            return "lead_vehicle_decelerates", "inferred", 0.97
        if interaction in {"lane_change", "cut_in", "aggressive_cut_in", "lane_encroachment"}:
            return "actor_crosses_lane_boundary", "inferred", 0.95
        if "actor_enters_ego_lane" in allowed and _contains(sentence, ["enter", "into the lane", "steps into"]):
            return "actor_enters_ego_lane", "explicit", 0.91
        return sorted(allowed)[0], "hierarchical_default", 0.68

    @staticmethod
    def _response(
        frame: EventFrame,
        slots: Mapping[str, Any],
        sentence: str,
        interaction: str,
    ) -> Tuple[str, str, float]:
        allowed = INTERACTION_TO_RESPONSES[interaction]
        if _contains(sentence, ["emergency brake", "hard brake", "panic brake"]):
            return _first_allowed("brake_or_emergency_brake", allowed, "brake"), "explicit", 0.95
        if _contains(sentence, ["yield"]):
            return _first_allowed("yield", allowed, "brake"), "explicit", 0.94
        if _contains(sentence, ["steer", "swerve", "evade"]):
            return _first_allowed("steer", allowed, "brake"), "explicit", 0.94
        if "brake_or_emergency_brake" in allowed:
            return "brake_or_emergency_brake", "kinematic_pending", 0.68
        if "yield" in allowed and interaction in {"left_turn_across_oncoming", "roundabout_entry_conflict", "unprotected_crossing"}:
            return "yield", "inferred", 0.8
        return ("brake", "hierarchical_default", 0.7) if "brake" in allowed else (sorted(allowed)[0], "hierarchical_default", 0.65)

    @staticmethod
    def _risk(
        frame: EventFrame,
        slots: Mapping[str, Any],
        sentence: str,
        interaction: str,
    ) -> Tuple[str, str, float]:
        raw = _clean(slots.get("risk_level"), "moderate")
        if _contains(sentence, ["critical", "imminent collision", "unavoidable"]):
            return "critical", "explicit", 0.95
        if _contains(sentence, ["aggressive", "dangerous", "suddenly", "abruptly", "near miss", "near-miss"]):
            return "aggressive", "explicit_semantic_modifier", 0.9
        if _contains(sentence, ["mild", "slowly", "comfortable"]):
            return "mild", "explicit_semantic_modifier", 0.9
        if raw in {"mild", "moderate", "aggressive", "critical"}:
            return raw, "inferred", 0.74
        if interaction in {"occluded_emergence", "aggressive_cut_in", "hard_braking", "sudden_stop", "red_light_intrusion", "wrong_way_approach"}:
            return "aggressive", "hierarchical_prior", 0.72
        return "moderate", "hierarchical_default", 0.65

    @staticmethod
    def _absolute_direction_hint(source_region: str) -> str:
        if source_region == "left_side":
            return "left_to_right"
        if source_region == "right_side":
            return "right_to_left"
        return "derived_after_occluder_side_sampling"

    @staticmethod
    def _node_evidence(frame: EventFrame, node_type: str, value: str) -> str:
        sentence = frame.sentence
        if node_type in {"primary_actor_group", "primary_actor_type"}:
            return frame.main_actor.text or _span(sentence, ["child", "pedestrian", "person", "cyclist"])
        if node_type == "hazard_interaction" and value == "occluded_emergence":
            return _span(sentence, ["emerges from behind", "appears from behind", "comes out from behind", "behind"])
        if node_type == "auxiliary_entity":
            return _span(sentence, ["parked truck", "truck", "parked car", "bus", "van", "barrier", "tree", "building"])
        if node_type == "target_region":
            return _span(sentence, ["ego lane", "ego path", "same lane"])
        if node_type == "visibility":
            return _span(sentence, ["from behind", "behind", "occluded", "hidden"])
        if node_type == "risk_level":
            return _span(sentence, ["suddenly", "abruptly", "dangerous", "aggressive", "critical", "mild"])
        if node_type == "motion_direction" and value == "occluder_to_ego_path":
            return "derived: selected occluder -> ego path"
        if node_type == "ego_required_response" and value == "brake_or_emergency_brake":
            return "derived: response pending sampled TTC and stopping distance"
        return ""
