"""Recursive hierarchy for natural-language traffic-scene understanding.

The existing EventFrame mapper emits mostly flat semantic slots.  This module
projects those slots onto one *selected path* through a constrained scene tree:

    road topology -> ego traffic space -> actor group -> actor type
    -> hazard interaction -> auxiliary entity -> spatial relations
    -> temporal trigger -> ego response -> risk -> executable parameters.

Each child is validated against its parent.  Missing fine-grained values may be
inferred from an already selected parent path, but the provenance remains
explicit so downstream experiments can distinguish user evidence from a prior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from sledge.semantic_control.language.event_frame import EventFrame


UNKNOWN_VALUES = {None, "", "unknown", "unknown_side", "unspecified"}


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
    allowed_children: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["allowed_children"] = list(self.allowed_children)
        return data


@dataclass
class HierarchicalScenePath:
    """A root-to-leaf decision path plus validation diagnostics."""

    nodes: List[HierarchyNode] = field(default_factory=list)
    executable_parameter_groups: Dict[str, List[str]] = field(default_factory=dict)
    valid: bool = True
    issues: List[str] = field(default_factory=list)
    schema_version: str = "eventframe_v5_hierarchical_tree"

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
        root: Dict[str, Any] = {
            "node_type": "scene_root",
            "value": "ego_centered_hazard_scene",
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "selected_path": [node.to_dict() for node in self.nodes],
            "path_values": self.values,
            "tree": self.to_nested_tree(),
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
    "single_lane": {"vulnerable_road_user", "vehicle", "static_object"},
    "same_direction_multi_lane": {"vehicle", "static_object"},
    "bidirectional_road": {"vehicle", "vulnerable_road_user", "static_object"},
    "curbside_zone": {"vulnerable_road_user", "vehicle", "static_object"},
    "crosswalk_zone": {"vulnerable_road_user", "vehicle"},
    "straight_through": {"vehicle", "vulnerable_road_user", "static_object"},
    "left_turn_path": {"vehicle", "vulnerable_road_user"},
    "right_turn_path": {"vehicle", "vulnerable_road_user"},
    "cross_traffic_zone": {"vehicle", "vulnerable_road_user"},
    "entry_path": {"vehicle", "vulnerable_road_user"},
    "circulating_lane": {"vehicle"},
    "exit_path": {"vehicle", "vulnerable_road_user"},
    "ramp_merge": {"vehicle"},
    "lane_drop": {"vehicle", "static_object"},
    "diverge": {"vehicle"},
    "open_lane": {"vehicle", "static_object"},
    "partially_blocked_lane": {"static_object", "vehicle", "vulnerable_road_user"},
    "closed_lane": {"static_object"},
}

ACTOR_GROUP_TO_TYPES: Dict[str, Set[str]] = {
    "vulnerable_road_user": {
        "pedestrian",
        "child_pedestrian",
        "jogger",
        "wheelchair_user",
        "cyclist",
        "ebike_rider",
        "scooter_rider",
    },
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
    "pedestrian": {"path_crossing", "enter_ego_lane", "occluded_emergence", "longitudinal_occupation"},
    "child_pedestrian": {"path_crossing", "enter_ego_lane", "occluded_emergence"},
    "jogger": {"path_crossing", "enter_ego_lane", "occluded_emergence"},
    "wheelchair_user": {"path_crossing", "enter_ego_lane", "longitudinal_occupation"},
    "cyclist": {"path_crossing", "enter_ego_lane", "occluded_emergence", "longitudinal_occupation"},
    "ebike_rider": {"path_crossing", "enter_ego_lane", "occluded_emergence"},
    "scooter_rider": {"path_crossing", "enter_ego_lane", "occluded_emergence"},
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
        "generic_occluder",
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

INTERACTION_TO_TRIGGERS: Dict[str, Set[str]] = {
    "occluded_emergence": {"occluded_actor_becomes_visible", "actor_enters_ego_lane"},
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
    "path_crossing": {"brake", "yield", "emergency_brake"},
    "enter_ego_lane": {"brake", "yield", "emergency_brake"},
    "occluded_emergence": {"brake", "emergency_brake", "steer"},
    "longitudinal_occupation": {"brake", "yield", "steer"},
    "gradual_braking": {"brake"},
    "hard_braking": {"brake", "emergency_brake"},
    "sudden_stop": {"emergency_brake", "steer"},
    "stationary_lead": {"brake", "stop", "steer"},
    "lane_change": {"brake", "yield"},
    "cut_in": {"brake", "yield", "steer"},
    "aggressive_cut_in": {"emergency_brake", "steer"},
    "lane_encroachment": {"brake", "steer"},
    "left_turn_across_oncoming": {"yield", "stop", "brake"},
    "oncoming_path_conflict": {"brake", "steer", "stop"},
    "wrong_way_approach": {"emergency_brake", "steer", "stop"},
    "intersection_crossing": {"brake", "yield", "stop"},
    "red_light_intrusion": {"emergency_brake", "stop"},
    "unprotected_crossing": {"yield", "brake", "stop"},
    "roundabout_entry_conflict": {"yield", "stop", "brake"},
    "lane_blocking": {"brake", "stop", "steer"},
    "partial_lane_occupation": {"brake", "steer"},
    "sudden_obstacle_appearance": {"emergency_brake", "steer"},
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
        if value in UNKNOWN_VALUES:
            return default
        return value.strip().lower().replace("-", "_").replace(" ", "_") or default
    if isinstance(value, (list, tuple, set)):
        if len(value) == 1:
            return _clean(next(iter(value)), default)
        return default
    return str(value).strip().lower().replace("-", "_").replace(" ", "_") or default


def _first_allowed(candidate: str, allowed: Iterable[str], fallback: str) -> str:
    options = set(allowed)
    return candidate if candidate in options else fallback if fallback in options else sorted(options)[0]


def _evidence(frame: EventFrame, *parts: str) -> str:
    values = [part for part in parts if part]
    return " | ".join(values) or frame.sentence


class HierarchicalSceneResolver:
    """Resolve a flat EventFrame spec into one constrained root-to-leaf path."""

    def resolve(self, frame: EventFrame, spec: Mapping[str, Any]) -> HierarchicalScenePath:
        slots = dict(spec.get("semantic_slots", {}) or {})
        sentence = frame.sentence or str(spec.get("evidence", {}).get("sentence", ""))

        road, road_source, road_conf = self._road_topology(frame, slots, sentence)
        traffic_space, traffic_source, traffic_conf = self._traffic_space(frame, slots, sentence, road)
        actor_group, actor_type, actor_source, actor_conf = self._actor(frame, slots, sentence, traffic_space)
        interaction, interaction_source, interaction_conf = self._interaction(frame, slots, sentence, actor_type)
        auxiliary, auxiliary_source, auxiliary_conf = self._auxiliary(frame, slots, sentence, interaction)
        source_region, source_source, source_conf = self._source_region(frame, slots, sentence, interaction)
        target_region, target_source, target_conf = self._target_region(frame, slots, sentence, interaction)
        anchor_region, anchor_source, anchor_conf = self._anchor_region(frame, slots, sentence, interaction)
        visibility, visibility_source, visibility_conf = self._visibility(frame, slots, sentence, interaction)
        motion_direction, direction_source, direction_conf = self._motion_direction(
            frame, slots, sentence, interaction, source_region
        )
        trigger, trigger_source, trigger_conf = self._trigger(frame, slots, sentence, interaction)
        response, response_source, response_conf = self._response(frame, slots, sentence, interaction)
        risk, risk_source, risk_conf = self._risk(frame, slots, sentence, interaction)

        selected = [
            ("road_topology", road, road_source, road_conf, frame.road_context.evidence_text),
            ("ego_traffic_space", traffic_space, traffic_source, traffic_conf, frame.road_context.evidence_text),
            ("primary_actor_group", actor_group, actor_source, actor_conf, frame.main_actor.evidence_text),
            ("primary_actor_type", actor_type, actor_source, actor_conf, frame.main_actor.evidence_text),
            ("hazard_interaction", interaction, interaction_source, interaction_conf, frame.main_event.evidence_text),
            ("auxiliary_entity", auxiliary, auxiliary_source, auxiliary_conf, frame.occlusion.evidence_text),
            ("source_region", source_region, source_source, source_conf, frame.main_event.evidence_text),
            ("target_region", target_region, target_source, target_conf, frame.main_event.evidence_text),
            ("anchor_region", anchor_region, anchor_source, anchor_conf, frame.main_event.evidence_text),
            ("visibility", visibility, visibility_source, visibility_conf, frame.occlusion.evidence_text),
            ("motion_direction", motion_direction, direction_source, direction_conf, frame.main_event.evidence_text),
            ("trigger_event", trigger, trigger_source, trigger_conf, frame.main_event.evidence_text),
            ("ego_required_response", response, response_source, response_conf, frame.ego_event.evidence_text),
            ("risk_level", risk, risk_source, risk_conf, sentence),
        ]

        path = HierarchicalScenePath(executable_parameter_groups=EXECUTABLE_PARAMETER_GROUPS)
        previous_type = "scene_root"
        previous_value = "ego_centered_hazard_scene"
        for level, (node_type, value, source, confidence, evidence_text) in enumerate(selected, start=1):
            allowed = tuple(sorted(self.allowed_children(previous_type, previous_value, node_type, path.values)))
            path.nodes.append(
                HierarchyNode(
                    level=level,
                    node_type=node_type,
                    value=value,
                    source=source,
                    confidence=max(0.0, min(float(confidence), 1.0)),
                    evidence=_evidence(frame, evidence_text),
                    parent_type=previous_type,
                    parent_value=previous_value,
                    allowed_children=allowed,
                )
            )
            previous_type = node_type
            previous_value = value

        path.issues = self.validate(path)
        path.valid = not path.issues
        return path

    def allowed_children(
        self,
        parent_type: str,
        parent_value: str,
        child_type: str,
        selected: Mapping[str, str],
    ) -> Set[str]:
        if child_type == "road_topology":
            return set(ROAD_TO_TRAFFIC_SPACES)
        if child_type == "ego_traffic_space":
            return set(ROAD_TO_TRAFFIC_SPACES.get(parent_value, set()))
        if child_type == "primary_actor_group":
            return set(TRAFFIC_SPACE_TO_ACTOR_GROUPS.get(parent_value, ACTOR_GROUP_TO_TYPES))
        if child_type == "primary_actor_type":
            return set(ACTOR_GROUP_TO_TYPES.get(parent_value, set()))
        if child_type == "hazard_interaction":
            return set(ACTOR_TYPE_TO_INTERACTIONS.get(parent_value, set()))
        if child_type == "auxiliary_entity":
            return set(INTERACTION_TO_AUXILIARIES.get(parent_value, {"none"}))
        interaction = selected.get("hazard_interaction", "unknown")
        if child_type == "source_region":
            return set(INTERACTION_TO_SOURCE_REGIONS.get(interaction, {"front_same_lane"}))
        if child_type == "target_region":
            return set(INTERACTION_TO_TARGET_REGIONS.get(interaction, {"ego_path"}))
        if child_type == "anchor_region":
            return {"near_front", "far_front", "intersection_center", "lane_boundary", "crosswalk", "roundabout_entry"}
        if child_type == "visibility":
            return set(INTERACTION_TO_VISIBILITY.get(interaction, {"fully_visible", "partially_occluded", "fully_occluded"}))
        if child_type == "motion_direction":
            return {
                "left_to_right",
                "right_to_left",
                "into_ego_lane",
                "toward_ego",
                "opposite_direction",
                "longitudinal_same_direction",
                "stationary",
                "circulating_across_entry",
                "crossing_unspecified",
            }
        if child_type == "trigger_event":
            return set(INTERACTION_TO_TRIGGERS.get(interaction, {"ego_reaches_conflict_area"}))
        if child_type == "ego_required_response":
            return set(INTERACTION_TO_RESPONSES.get(interaction, {"brake"}))
        if child_type == "risk_level":
            return {"mild", "moderate", "aggressive", "critical"}
        return set()

    def validate(self, path: HierarchicalScenePath) -> List[str]:
        issues: List[str] = []
        values = path.values
        transitions = [
            ("road_topology", "ego_traffic_space", ROAD_TO_TRAFFIC_SPACES),
            ("ego_traffic_space", "primary_actor_group", TRAFFIC_SPACE_TO_ACTOR_GROUPS),
            ("primary_actor_group", "primary_actor_type", ACTOR_GROUP_TO_TYPES),
            ("primary_actor_type", "hazard_interaction", ACTOR_TYPE_TO_INTERACTIONS),
            ("hazard_interaction", "auxiliary_entity", INTERACTION_TO_AUXILIARIES),
        ]
        for parent_key, child_key, table in transitions:
            parent = values.get(parent_key, "unknown")
            child = values.get(child_key, "unknown")
            allowed = table.get(parent, set())
            if child not in allowed:
                issues.append(f"invalid_transition:{parent_key}={parent}->{child_key}={child}")

        interaction = values.get("hazard_interaction", "unknown")
        for child_key, table in [
            ("source_region", INTERACTION_TO_SOURCE_REGIONS),
            ("target_region", INTERACTION_TO_TARGET_REGIONS),
            ("trigger_event", INTERACTION_TO_TRIGGERS),
            ("ego_required_response", INTERACTION_TO_RESPONSES),
        ]:
            child = values.get(child_key, "unknown")
            if child not in table.get(interaction, set()):
                issues.append(f"invalid_interaction_child:{interaction}->{child_key}={child}")

        visibility = values.get("visibility", "unknown")
        if interaction == "occluded_emergence" and visibility == "fully_visible":
            issues.append("occluded_emergence_requires_occluded_visibility")
        auxiliary = values.get("auxiliary_entity", "none")
        if interaction == "occluded_emergence" and not auxiliary.endswith("_occluder"):
            issues.append("occluded_emergence_requires_occluder")
        return issues

    @staticmethod
    def _road_topology(frame: EventFrame, slots: Mapping[str, Any], sentence: str) -> Tuple[str, str, float]:
        raw = _clean(slots.get("road_topology", frame.road_context.road_type))
        if raw in {"intersection"} or _contains(sentence, ["intersection", "junction"]):
            return "intersection", "explicit" if raw == "intersection" else "inferred", 0.95
        if raw == "roundabout" or _contains(sentence, ["roundabout", "rotary", "traffic circle"]):
            return "roundabout", "explicit" if raw == "roundabout" else "inferred", 0.97
        if raw in {"construction_zone", "work_zone"} or _contains(sentence, ["construction", "work zone", "roadwork"]):
            return "work_zone", "explicit" if raw in {"construction_zone", "work_zone"} else "inferred", 0.94
        if raw in {"merge_diverge", "ramp_merge"} or _contains(sentence, ["ramp", "merge area", "lane drop", "diverge"]):
            return "merge_diverge", "inferred", 0.82
        return "straight_segment", "hierarchical_default" if raw == "unknown" else "normalized", 0.72 if raw == "unknown" else 0.9

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
            if frame.ego_event.ego_maneuver == "enter_roundabout" or layout == "roundabout_entry":
                return "entry_path", "inferred", 0.95
            return "circulating_lane", "hierarchical_default", 0.62
        if road == "merge_diverge":
            return "ramp_merge", "inferred", 0.82
        if road == "work_zone":
            if _contains(sentence, ["closed lane", "lane closed"]):
                return "closed_lane", "explicit", 0.92
            return "partially_blocked_lane", "inferred", 0.8
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
    ) -> Tuple[str, str, str, float]:
        actor_base = _clean(slots.get("actor_type"))
        actor_text = " ".join([frame.main_actor.text, sentence]).lower()
        role = _clean(slots.get("actor_role"))
        motion = _clean(slots.get("motion_geometry"))

        if actor_base == "pedestrian" or frame.main_actor.actor_class == "human_on_foot":
            if _contains(actor_text, ["child", "kid", "schoolkid", "boy", "girl"]):
                return "vulnerable_road_user", "child_pedestrian", "explicit", 0.97
            if _contains(actor_text, ["jogger", "runner", "running person"]):
                return "vulnerable_road_user", "jogger", "explicit", 0.93
            if _contains(actor_text, ["wheelchair"]):
                return "vulnerable_road_user", "wheelchair_user", "explicit", 0.96
            return "vulnerable_road_user", "pedestrian", "normalized", 0.95
        if actor_base == "cyclist" or frame.main_actor.actor_class == "cyclist":
            if _contains(actor_text, ["e-bike", "ebike", "electric bicycle"]):
                return "vulnerable_road_user", "ebike_rider", "explicit", 0.94
            if _contains(actor_text, ["scooter"]):
                return "vulnerable_road_user", "scooter_rider", "explicit", 0.9
            return "vulnerable_road_user", "cyclist", "normalized", 0.94
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
            return "static_object", actor_type, "inferred", 0.88

        if motion == "longitudinal" or role == "braking_actor":
            return "vehicle", "lead_vehicle", "inferred", 0.95
        if motion == "oncoming" or role == "approaching_actor":
            return "vehicle", "oncoming_vehicle", "inferred", 0.94
        if traffic_space == "cross_traffic_zone" or _contains(sentence, ["cross traffic", "from the side road"]):
            return "vehicle", "cross_traffic_vehicle", "inferred", 0.87
        if traffic_space in {"entry_path", "ramp_merge"} or _contains(sentence, ["merging vehicle", "merge into"]):
            return "vehicle", "merging_vehicle", "inferred", 0.9
        if motion == "merging" or role == "merging_actor":
            return "vehicle", "adjacent_vehicle", "inferred", 0.91
        return "vehicle", "generic_vehicle", "hierarchical_default", 0.58

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
            return "occluded_emergence", "explicit", 0.98
        if actor_type in {"pedestrian", "child_pedestrian", "jogger", "wheelchair_user", "cyclist", "ebike_rider", "scooter_rider"}:
            candidate = "enter_ego_lane" if event == "enter_ego_lane" else "path_crossing"
            return _first_allowed(candidate, allowed, "path_crossing"), "inferred", 0.93
        if actor_type == "lead_vehicle":
            if event == "hard_stop_ahead" or _contains(sentence, ["hard brake", "brakes hard", "panic", "slams", "suddenly stops"]):
                return "hard_braking", "explicit" if _contains(sentence, ["hard", "panic", "slams", "suddenly"]) else "inferred", 0.95
            return "gradual_braking", "hierarchical_default", 0.73
        if actor_type in {"adjacent_vehicle", "merging_vehicle", "generic_vehicle"} and motion == "merging":
            if risk in {"aggressive", "critical"} or _contains(sentence, ["aggressive", "tight gap", "no room", "squeezes"]):
                return _first_allowed("aggressive_cut_in", allowed, "cut_in"), "inferred", 0.9
            return _first_allowed("cut_in", allowed, "lane_change"), "inferred", 0.87
        if actor_type in {"oncoming_vehicle", "generic_vehicle"} and motion == "oncoming":
            if frame.ego_event.ego_maneuver == "left_turn" or _contains(sentence, ["left turn", "turns left"]):
                return _first_allowed("left_turn_across_oncoming", allowed, "oncoming_path_conflict"), "inferred", 0.96
            return _first_allowed("oncoming_path_conflict", allowed, "path_crossing"), "inferred", 0.88
        if actor_type == "cross_traffic_vehicle" or (motion == "lateral" and actor_type.endswith("vehicle")):
            return _first_allowed("intersection_crossing", allowed, "path_crossing"), "inferred", 0.84
        if actor_type == "circulating_vehicle" or event == "roundabout_entry_conflict":
            return "roundabout_entry_conflict", "inferred", 0.96
        if actor_type in ACTOR_GROUP_TO_TYPES["static_object"] or motion == "static":
            return _first_allowed("lane_blocking", allowed, "partial_lane_occupation"), "inferred", 0.9
        return sorted(allowed)[0] if allowed else "path_crossing", "hierarchical_default", 0.4

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
            parked = _contains(sentence, ["parked", "stopped roadside"])
            if raw == "truck" or _contains(sentence, ["truck", "lorry"]):
                candidate = "parked_truck_occluder" if parked else "parked_truck_occluder"
            elif raw == "bus" or "bus" in sentence.lower():
                candidate = "bus_occluder"
            elif raw == "van" or "van" in sentence.lower():
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
                candidate = "generic_occluder"
            return _first_allowed(candidate, allowed, "generic_occluder"), "inferred", 0.95 if candidate != "generic_occluder" else 0.62
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
            elif _contains(sentence, ["curb", "roadside", "sidewalk"]):
                candidate = "curbside"
        if candidate in allowed:
            return candidate, "explicit" if relation != "unknown" else "inferred", 0.9
        fallback = sorted(allowed)[0]
        return fallback, "hierarchical_default", 0.55

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
        if raw in {"intersection", "at_intersection"} or interaction in {
            "left_turn_across_oncoming",
            "intersection_crossing",
            "red_light_intrusion",
            "unprotected_crossing",
        }:
            return "intersection_center", "inferred", 0.9
        if raw == "roundabout_entry" or interaction == "roundabout_entry_conflict":
            return "roundabout_entry", "inferred", 0.95
        if _contains(sentence, ["crosswalk"]):
            return "crosswalk", "explicit", 0.94
        if interaction in {"lane_change", "cut_in", "aggressive_cut_in", "lane_encroachment"}:
            return "lane_boundary", "inferred", 0.9
        if raw in {"front", "ahead_of", "in_front_of"} or _contains(sentence, ["just ahead", "near ego", "close ahead"]):
            return "near_front", "explicit" if raw != "unknown" else "inferred", 0.86
        return "far_front" if _contains(sentence, ["far ahead", "in the distance"]) else "near_front", "hierarchical_default", 0.6

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
            return "fully_occluded", "inferred", 0.9
        return "fully_visible", "hierarchical_default", 0.8

    @staticmethod
    def _motion_direction(
        frame: EventFrame,
        slots: Mapping[str, Any],
        sentence: str,
        interaction: str,
        source_region: str,
    ) -> Tuple[str, str, float]:
        raw = _clean(slots.get("conflict_direction", frame.main_event.motion_direction))
        if raw in {"left_to_right", "right_to_left"}:
            return raw, "explicit", 0.95
        if source_region in {"left_side", "adjacent_left_lane"}:
            return "left_to_right" if interaction in {"path_crossing", "enter_ego_lane", "occluded_emergence"} else "into_ego_lane", "inferred", 0.86
        if source_region in {"right_side", "adjacent_right_lane"}:
            return "right_to_left" if interaction in {"path_crossing", "enter_ego_lane", "occluded_emergence"} else "into_ego_lane", "inferred", 0.86
        if interaction in {"lane_change", "cut_in", "aggressive_cut_in", "lane_encroachment"}:
            return "into_ego_lane", "inferred", 0.93
        if interaction in {"gradual_braking", "hard_braking", "sudden_stop", "stationary_lead"}:
            return "stationary" if interaction == "stationary_lead" else "longitudinal_same_direction", "inferred", 0.93
        if interaction in {"left_turn_across_oncoming", "oncoming_path_conflict", "wrong_way_approach"}:
            return "opposite_direction", "inferred", 0.95
        if interaction == "roundabout_entry_conflict":
            return "circulating_across_entry", "inferred", 0.95
        if interaction in {"lane_blocking", "partial_lane_occupation", "sudden_obstacle_appearance"}:
            return "stationary", "inferred", 0.96
        return "crossing_unspecified", "distributional_default", 0.5

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
        risk = _clean(slots.get("risk_level"), "moderate")
        if _contains(sentence, ["emergency brake", "hard brake", "panic brake"]):
            candidate = "emergency_brake"
            source = "explicit"
        elif _contains(sentence, ["yield"]):
            candidate = "yield"
            source = "explicit"
        elif _contains(sentence, ["steer", "swerve", "evade"]):
            candidate = "steer"
            source = "explicit"
        elif risk in {"aggressive", "critical"} and "emergency_brake" in allowed:
            candidate = "emergency_brake"
            source = "inferred"
        elif "yield" in allowed and interaction in {"left_turn_across_oncoming", "roundabout_entry_conflict", "unprotected_crossing"}:
            candidate = "yield"
            source = "inferred"
        else:
            candidate = "brake" if "brake" in allowed else sorted(allowed)[0]
            source = "hierarchical_default"
        return candidate, source, 0.92 if source == "explicit" else 0.78

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
        if raw in {"mild", "moderate", "aggressive", "critical"}:
            source = "explicit" if _contains(sentence, [raw, "dangerous", "aggressive", "mild"]) else "inferred"
            return raw, source, 0.9 if source == "explicit" else 0.76
        if interaction in {"occluded_emergence", "aggressive_cut_in", "hard_braking", "sudden_stop", "red_light_intrusion", "wrong_way_approach"}:
            return "aggressive", "hierarchical_prior", 0.75
        return "moderate", "hierarchical_default", 0.65
