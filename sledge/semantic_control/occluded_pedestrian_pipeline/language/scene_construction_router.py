"""Route hierarchical language templates to B0 editing or full scene synthesis.

The router implements one deliberately conservative rule:

* local hazard language (pedestrian, occluder, side, speed, ego lane/path,
  occlusion, risk) edits an existing B0 scene;
* explicit *global* road/lane structure routes to scene synthesis.

The hierarchy itself is not used as proof that the user specified a global
road structure because the hierarchy contains inferred/default road nodes even
when the prompt is silent about road topology. Routing therefore uses the
original prompt as the source of truth for explicit global structure and uses
hierarchical values only to describe the local hazard that will be executed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


EDIT_EXISTING = "edit_existing"
SYNTHESIZE_NEW = "synthesize_new"
ROUTING_POLICY = "explicit_global_road_structure_routes_to_synthesis_v3_move_then_delete"


@dataclass(frozen=True)
class SceneConstructionPlan:
    """Serializable execution plan produced after language parsing."""

    mode: str
    reason: str
    explicit_global_constraints: Dict[str, Any] = field(default_factory=dict)
    local_hazard_constraints: Dict[str, Any] = field(default_factory=dict)
    # Legacy compatibility field. In EDIT_EXISTING this now contains only
    # immutable road/lane structure, not ego/background state.
    preserve_from_b0: List[str] = field(default_factory=list)
    hard_preserve_from_b0: List[str] = field(default_factory=list)
    semantic_edit_controls: List[str] = field(default_factory=list)
    elastic_context: List[str] = field(default_factory=list)
    context_edit_policy: Dict[str, Any] = field(default_factory=dict)
    sample_from_template: List[str] = field(default_factory=list)
    derive_from_scene: List[str] = field(default_factory=list)
    trigger_evidence: List[str] = field(default_factory=list)
    source_scene_usage: str = "semantic_base_scene"
    policy: str = ROUTING_POLICY

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SceneConstructionRouter:
    """Choose between road-fixed elastic editing and full scene synthesis."""

    _LANE_WORDS = {
        "single": 1,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
    }

    # These patterns describe GLOBAL scene structure. Expressions such as
    # "ego lane", "ego path", "left roadside", "roadside barrier" and
    # "enters the lane" are intentionally absent.
    _TOPOLOGY_PATTERNS: Sequence[Tuple[str, Sequence[str]]] = (
        (
            "intersection",
            (
                r"\b(?:at|in|inside|near|approaching)\s+(?:an?\s+)?intersection\b",
                r"\b(?:at|in|inside|near|approaching)\s+(?:an?\s+)?junction\b",
                r"\b(?:at|in|inside|near)\s+(?:the\s+)?crossroads?\b",
                r"\bfour[- ]way\s+(?:intersection|junction)\b",
                r"\bt[- ]junction\b",
                r"(?:在|位于|接近)(?:一个)?(?:十字路口|交叉口|路口)",
            ),
        ),
        (
            "roundabout",
            (
                r"\b(?:at|in|inside|entering|approaching)\s+(?:a\s+)?roundabout\b",
                r"\btraffic circle\b",
                r"(?:在|进入|接近)(?:一个)?环岛",
            ),
        ),
        (
            "merge_diverge",
            (
                r"\b(?:highway\s+)?(?:merge|merging|diverge|diverging)(?:\s+(?:area|section|road|lane))?\b",
                r"\b(?:on[- ]?ramp|off[- ]?ramp|merge ramp|exit ramp)\b",
                r"\blane drop\b",
                r"(?:在|位于)(?:匝道|合流区|分流区)",
            ),
        ),
        (
            "work_zone",
            (
                r"\b(?:construction zone|work zone|roadworks?|road works?)\b",
                r"\b(?:lane closure|closed lane)\b",
                r"(?:在|位于)(?:施工区|道路施工区|封闭车道)",
            ),
        ),
        (
            "straight_segment",
            (
                r"\bon\s+(?:a|the)\s+straight\s+(?:road|street|segment)\b",
                r"\balong\s+(?:a|the)\s+straight\s+(?:road|street|segment)\b",
                r"(?:在|沿着)(?:一条)?直路",
            ),
        ),
    )

    _CURVED_ROAD_PATTERNS: Sequence[str] = (
        r"\bon\s+(?:a|the)\s+(?:curved|winding)\s+(?:road|street|segment)\b",
        r"\balong\s+(?:a|the)\s+(?:curved|winding)\s+(?:road|street|segment)\b",
        r"(?:在|沿着)(?:一条)?(?:弯道|弯曲道路)",
    )

    _LANE_LAYOUT_PATTERNS: Sequence[Tuple[str, str]] = (
        ("dedicated_left_turn_lane", r"\bdedicated\s+left[- ]turn\s+lane\b"),
        ("dedicated_right_turn_lane", r"\bdedicated\s+right[- ]turn\s+lane\b"),
        ("turn_lane", r"\bturn\s+lane\b"),
        ("closed_lane", r"\b(?:closed lane|lane closure)\b"),
        ("dedicated_left_turn_lane", r"专用左转车道"),
        ("dedicated_right_turn_lane", r"专用右转车道"),
        ("closed_lane", r"(?:封闭车道|车道封闭)"),
    )

    def route(
        self,
        *,
        prompt: str,
        hierarchical_spec: Mapping[str, Any],
    ) -> SceneConstructionPlan:
        text = str(prompt or "")
        normalized = text.lower()
        global_constraints: Dict[str, Any] = {}
        trigger_evidence: List[str] = []

        topology = self._match_topology(normalized, trigger_evidence)
        if topology:
            global_constraints["road_topology"] = topology

        if self._first_match(normalized, self._CURVED_ROAD_PATTERNS):
            evidence = self._first_match(normalized, self._CURVED_ROAD_PATTERNS)
            if evidence:
                trigger_evidence.append(evidence)
            global_constraints["road_shape"] = "curved"
            # The hierarchy currently has no separate curve leaf. Keep the
            # executable topology as a road segment and carry curvature as an
            # explicit global constraint.
            global_constraints.setdefault("road_topology", "straight_segment")

        lane_count = self._extract_lane_count(normalized)
        if lane_count is not None:
            global_constraints["lane_count"] = lane_count[0]
            trigger_evidence.append(lane_count[1])

        multi_lane = self._first_match(
            normalized,
            (
                r"\b(?:multi[- ]lane|multilane)\s+(?:road|street|carriageway)\b",
                r"\bsame[- ]direction\s+multi[- ]lane\s+(?:road|street|carriageway)\b",
                r"(?:多车道道路|多车道公路)",
            ),
        )
        if multi_lane:
            global_constraints["minimum_lane_count"] = 2
            trigger_evidence.append(multi_lane)

        directionality = self._extract_directionality(normalized)
        if directionality is not None:
            global_constraints["directionality"] = directionality[0]
            trigger_evidence.append(directionality[1])

        lane_width = self._extract_lane_width(normalized)
        if lane_width is not None:
            global_constraints["lane_width_m"] = lane_width[0]
            trigger_evidence.append(lane_width[1])

        curvature = self._extract_road_curvature(normalized)
        if curvature is not None:
            global_constraints["road_curvature"] = curvature[0]
            trigger_evidence.append(curvature[1])

        lane_layout = self._extract_lane_layout(normalized)
        if lane_layout:
            global_constraints["lane_layout"] = lane_layout[0]
            trigger_evidence.extend(lane_layout[1])

        hierarchy = dict(hierarchical_spec.get("hierarchy_layer", {}) or {})
        values = dict(hierarchy.get("path_values", {}) or {})
        completed = dict(
            hierarchical_spec.get("parameter_layer", {}).get("completed", {}) or {}
        )
        local_constraints = self._local_hazard_constraints(values, completed)

        trigger_evidence = self._deduplicate(trigger_evidence)
        if global_constraints:
            return SceneConstructionPlan(
                mode=SYNTHESIZE_NEW,
                reason="explicit_global_road_structure",
                explicit_global_constraints=global_constraints,
                local_hazard_constraints=local_constraints,
                preserve_from_b0=[],
                hard_preserve_from_b0=[],
                semantic_edit_controls=[
                    "road_geometry",
                    "lane_geometry",
                    "ego_state",
                    "primary_pedestrian",
                    "occluder",
                ],
                elastic_context=[],
                context_edit_policy={
                    "mode": "template_synthesis",
                    "hazard_constraints_remain_hard": True,
                },
                sample_from_template=[
                    "lane_width_m",
                    "lane_count",
                    "road_curvature",
                    "ego_speed_mps",
                    "ego_acceleration_mps2",
                    "ego_distance_to_conflict_m",
                    "actor_speed_mps",
                    "actor_acceleration_mps2",
                    "actor_start_time_s",
                    "occluder_side",
                    "occluder_length_m",
                    "occluder_width_m",
                    "occluder_lateral_offset_m",
                    "reveal_distance_m",
                    "minimum_clearance_m",
                    "braking_deceleration_mps2",
                ],
                derive_from_scene=[
                    "conflict_point_xy",
                    "occluder_position",
                    "actor_initial_position",
                    "actor_heading",
                    "initial_gap_m",
                    "time_to_collision_s",
                ],
                trigger_evidence=trigger_evidence,
                source_scene_usage="blank_capacity_scaffold_only",
            )

        return SceneConstructionPlan(
            mode=EDIT_EXISTING,
            reason="no_explicit_global_road_structure",
            explicit_global_constraints={},
            local_hazard_constraints=local_constraints,
            preserve_from_b0=[
                "road_geometry",
                "lane_geometry",
            ],
            hard_preserve_from_b0=[
                "road_geometry",
                "lane_geometry",
                "road_topology",
            ],
            semantic_edit_controls=[
                "ego_state",
                "primary_pedestrian",
                "occluder",
            ],
            elastic_context=[
                "background_vehicles",
                "background_pedestrians",
                "background_static_objects",
            ],
            context_edit_policy={
                "mode": "move_then_delete_local_blockers",
                "preserve_unrelated_context_when_feasible": True,
                "allow_reposition": True,
                "allow_velocity_adjustment": False,
                "allow_removal": True,
                "removal_scope": "local_occluder_reserved_region_only",
                "traffic_lights_removable": False,
                "hazard_constraints_remain_hard": True,
            },
            sample_from_template=[
                "ego_speed_mps",
                "ego_acceleration_mps2",
                "ego_distance_to_conflict_m",
                "occluder_side",
                "actor_speed_mps",
                "actor_acceleration_mps2",
                "actor_start_time_s",
                "occluder_length_m",
                "occluder_width_m",
                "occluder_lateral_offset_m",
                "reveal_distance_m",
                "minimum_clearance_m",
                "braking_deceleration_mps2",
            ],
            derive_from_scene=[
                "conflict_point_xy",
                "occluder_position",
                "actor_initial_position",
                "actor_heading",
                "initial_gap_m",
                "time_to_collision_s",
            ],
            trigger_evidence=[],
            source_scene_usage="road_fixed_elastic_semantic_base_scene",
        )

    def attach_plan(
        self,
        *,
        prompt: str,
        hierarchical_spec: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Return a shallowly copied spec with a serialized construction plan."""

        spec = dict(hierarchical_spec)
        spec["scene_construction"] = self.route(
            prompt=prompt,
            hierarchical_spec=spec,
        ).to_dict()
        return spec

    def _match_topology(
        self,
        text: str,
        evidence_out: List[str],
    ) -> Optional[str]:
        for topology, patterns in self._TOPOLOGY_PATTERNS:
            fragment = self._first_match(text, patterns)
            if fragment:
                evidence_out.append(fragment)
                return topology
        return None

    def _extract_lane_count(self, text: str) -> Optional[Tuple[int, str]]:
        # "two-lane road", "3 lane street", but not "ego lane".
        match = re.search(
            r"\b(single|one|two|three|four|five|six|[1-6])[- ]lane\s+"
            r"(?:road|street|carriageway|avenue|segment)\b",
            text,
        )
        if match:
            token = match.group(1)
            count = self._LANE_WORDS.get(token, int(token) if token.isdigit() else 1)
            return int(count), match.group(0)

        # "road with two lanes".
        match = re.search(
            r"\b(?:road|street|carriageway|avenue)\s+with\s+"
            r"(single|one|two|three|four|five|six|[1-6])\s+lanes?\b",
            text,
        )
        if match:
            token = match.group(1)
            count = self._LANE_WORDS.get(token, int(token) if token.isdigit() else 1)
            return int(count), match.group(0)

        cn = re.search(r"([一二三四五六1-6])车道(?:道路|公路|街道)?", text)
        if cn:
            mapping = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}
            token = cn.group(1)
            return int(mapping.get(token, int(token) if token.isdigit() else 1)), cn.group(0)
        return None

    @staticmethod
    def _extract_directionality(text: str) -> Optional[Tuple[str, str]]:
        patterns = (
            ("bidirectional", r"\b(?:bidirectional|two[- ]way)\s+(?:road|street|carriageway)\b"),
            ("one_way", r"\bone[- ]way\s+(?:road|street|carriageway)\b"),
            ("bidirectional", r"(?:双向道路|双向车道)"),
            ("one_way", r"(?:单向道路|单行道)"),
        )
        for value, pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return value, match.group(0)
        return None

    @staticmethod
    def _extract_lane_width(text: str) -> Optional[Tuple[float, str]]:
        patterns = (
            r"\blane\s+width\s*(?:is|=|of|:)??\s*(\d+(?:\.\d+)?)\s*(?:m|meter|meters)\b",
            r"\b(\d+(?:\.\d+)?)\s*(?:m|meter|meters)[- ]wide\s+lane\b",
            r"车道宽度(?:为|是|=)?\s*(\d+(?:\.\d+)?)\s*米",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = float(match.group(1))
                if 2.0 <= value <= 6.0:
                    return value, match.group(0)
        return None

    @staticmethod
    def _extract_road_curvature(text: str) -> Optional[Tuple[float, str]]:
        patterns = (
            r"\broad\s+curvature\s*(?:is|=|of|:)??\s*"
            r"(-?\d+(?:\.\d+)?)\s*(?:1/m|m\^-1|per meter)?",
            r"道路曲率(?:为|是|=)?\s*(-?\d+(?:\.\d+)?)\s*(?:1/m|每米)?",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = float(match.group(1))
                if abs(value) <= 0.2:
                    return value, match.group(0)
        return None

    def _extract_lane_layout(self, text: str) -> Optional[Tuple[List[str], List[str]]]:
        values: List[str] = []
        evidence: List[str] = []
        for value, pattern in self._LANE_LAYOUT_PATTERNS:
            match = re.search(pattern, text)
            if match:
                values.append(value)
                evidence.append(match.group(0))
        if not values:
            return None
        return self._deduplicate(values), self._deduplicate(evidence)

    @staticmethod
    def _local_hazard_constraints(
        values: Mapping[str, Any],
        completed: Mapping[str, Any],
    ) -> Dict[str, Any]:
        keys = (
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
        output = {key: values.get(key) for key in keys if values.get(key) is not None}
        actor_speed = completed.get("actor_speed_mps")
        if isinstance(actor_speed, Mapping):
            if actor_speed.get("source") == "user_input":
                output["actor_speed_mps"] = actor_speed.get("value")
        return output

    @staticmethod
    def _first_match(text: str, patterns: Sequence[str]) -> Optional[str]:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    @staticmethod
    def _deduplicate(values: Sequence[str]) -> List[str]:
        seen = set()
        output: List[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            output.append(value)
        return output


__all__ = [
    "EDIT_EXISTING",
    "SYNTHESIZE_NEW",
    "ROUTING_POLICY",
    "SceneConstructionPlan",
    "SceneConstructionRouter",
]