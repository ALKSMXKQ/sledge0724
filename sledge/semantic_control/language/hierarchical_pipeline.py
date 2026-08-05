"""Default EventFrame pipeline with recursive parent-constrained semantics.

This module keeps the established EventFrame parser and legacy hazard-spec
layers intact, then adds a selected hierarchy path and hierarchy-conditioned
parameter completion. Existing consumers can continue reading actor_layer,
interaction_layer, motion_layer, and parameter_layer, while new consumers can
read hierarchy_layer.tree or hierarchy_layer.selected_path.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from sledge.semantic_control.language.event_frame import EventFrame
from sledge.semantic_control.language.event_frame_mapper import (
    EventFrameToHazardSpecMapper,
    validate_spec,
)
from sledge.semantic_control.language.event_frame_parser import EventFrameParser
from sledge.semantic_control.language.event_frame_verifier import EventFrameVerifier
from sledge.semantic_control.language.event_sequence_builder import EventSequenceBuilder
from sledge.semantic_control.language.hierarchical_ontology import (
    HierarchicalScenePath,
    HierarchicalSceneResolver,
)
from sledge.semantic_control.language.missing_info_filler import MissingInfoFiller


@dataclass
class HierarchicalPipelineResult:
    """Structured output returned by :class:`HierarchicalEventFramePipeline`."""

    frame: EventFrame
    spec: Dict[str, Any]
    frame_issues: List[str]
    spec_issues: List[str]
    hierarchy_issues: List[str]

    @property
    def valid(self) -> bool:
        return not self.frame_issues and not self.spec_issues and not self.hierarchy_issues


class HierarchicalParameterFiller:
    """Complete numeric leaves using the already selected parent path.

    The legacy semantic-slot filler runs first. This class refines only values
    whose source is not explicit user input and records the complete parent path
    that conditioned each refinement.
    """

    def __init__(self, base_filler: Optional[MissingInfoFiller] = None) -> None:
        self.base_filler = base_filler or MissingInfoFiller()

    def fill(
        self,
        spec: Dict[str, Any],
        frame: EventFrame,
        hierarchy: HierarchicalScenePath,
    ) -> Dict[str, Any]:
        out = self.base_filler.fill(spec, frame)
        params = out.setdefault("parameter_layer", {})
        completed: Dict[str, Dict[str, Any]] = dict(params.get("completed", {}) or {})
        values = hierarchy.values

        for name, entry in list(completed.items()):
            if not isinstance(entry, dict):
                entry = {"value": entry}
                completed[name] = entry
            entry.setdefault("unit", "")
            entry.setdefault("source", "unknown")
            entry.setdefault("reason", "")
            entry.setdefault("confidence", 0.6 if entry["source"] != "user_input" else 1.0)
            entry.setdefault("evidence", [frame.sentence] if entry["source"] == "user_input" else [])
            entry.setdefault("conditioned_on", {})
            entry.setdefault("is_assumption", entry["source"] != "user_input")
            entry.setdefault("alternatives", [])

        def put(
            name: str,
            value: Any,
            *,
            unit: str = "",
            reason: str,
            through: str,
            confidence: float,
            alternatives: Optional[List[Any]] = None,
            overwrite_inferred: bool = True,
        ) -> None:
            current = completed.get(name)
            if current and current.get("source") == "user_input":
                return
            if current and not overwrite_inferred:
                return
            conditioned_on = self._conditions_through(hierarchy, through)
            completed[name] = {
                "value": value,
                "unit": unit,
                "source": "hierarchical_prior",
                "reason": reason,
                "confidence": max(0.0, min(float(confidence), 1.0)),
                "evidence": [],
                "conditioned_on": conditioned_on,
                "condition_path": hierarchy.condition_path(through=through),
                "is_assumption": True,
                "alternatives": list(alternatives or []),
            }

        actor = values.get("primary_actor_type", "unknown")
        interaction = values.get("hazard_interaction", "unknown")
        traffic_space = values.get("ego_traffic_space", "unknown")
        auxiliary = values.get("auxiliary_entity", "none")
        direction = values.get("motion_direction", "crossing_unspecified")
        risk = values.get("risk_level", "moderate")
        visibility = values.get("visibility", "fully_visible")

        actor_speed = self._actor_speed_prior(actor, risk)
        if actor_speed is not None:
            put(
                "actor_speed_mps",
                actor_speed,
                unit="m/s",
                reason=f"actor-speed prior for {actor} under {risk} risk",
                through="risk_level",
                confidence=0.84 if actor != "generic_vehicle" else 0.62,
            )

        ego_speed = self._ego_speed_prior(traffic_space, risk)
        if ego_speed is not None:
            put(
                "ego_speed_mps",
                ego_speed,
                unit="m/s",
                reason=f"ego-speed prior for {traffic_space} under {risk} risk",
                through="risk_level",
                confidence=0.76,
            )

        if interaction in {"path_crossing", "enter_ego_lane", "occluded_emergence"}:
            if direction in {"left_to_right", "right_to_left"}:
                put(
                    "crossing_direction",
                    direction,
                    reason="direction inherited from the selected source-region branch",
                    through="motion_direction",
                    confidence=0.9,
                )
            else:
                put(
                    "crossing_direction",
                    ["left_to_right", "right_to_left"],
                    reason="parent path determines crossing but not the originating side",
                    through="motion_direction",
                    confidence=0.5,
                    alternatives=["left_to_right", "right_to_left"],
                )

        if interaction == "occluded_emergence" or visibility in {"partially_occluded", "fully_occluded"}:
            reveal = [3.0, 8.0] if risk in {"aggressive", "critical"} else [5.0, 12.0]
            put(
                "reveal_distance_m",
                reveal,
                unit="m",
                reason="reveal-distance prior conditioned on occlusion and risk branch",
                through="risk_level",
                confidence=0.82,
            )
            put(
                "occlusion_enabled",
                True,
                reason="occluded-emergence branch requires an active occluder",
                through="visibility",
                confidence=0.99,
            )
            put(
                "occluder_type",
                self._occluder_parameter_type(auxiliary),
                reason="auxiliary-entity leaf determines the simulator occluder class",
                through="auxiliary_entity",
                confidence=0.92 if auxiliary != "generic_occluder" else 0.62,
            )
            put(
                "occluder_lateral_offset_m",
                [1.0, 4.0],
                unit="m",
                reason="roadside/adjacent placement prior for the selected occluder branch",
                through="auxiliary_entity",
                confidence=0.72,
            )
            length, width = self._occluder_size_prior(auxiliary)
            put(
                "occluder_length_m",
                length,
                unit="m",
                reason=f"size prior for {auxiliary}",
                through="auxiliary_entity",
                confidence=0.76,
            )
            put(
                "occluder_width_m",
                width,
                unit="m",
                reason=f"size prior for {auxiliary}",
                through="auxiliary_entity",
                confidence=0.76,
            )

        if interaction in {"cut_in", "aggressive_cut_in", "lane_change", "lane_encroachment"}:
            gap = [3.0, 10.0] if interaction == "aggressive_cut_in" or risk in {"aggressive", "critical"} else [7.0, 20.0]
            put(
                "initial_longitudinal_gap_m",
                gap,
                unit="m",
                reason="gap prior inherited from the vehicle cut-in branch",
                through="risk_level",
                confidence=0.8,
            )
            source_region = values.get("source_region", "unknown")
            side = "left" if source_region == "adjacent_left_lane" else "right" if source_region == "adjacent_right_lane" else ["left", "right"]
            put(
                "source_side",
                side,
                reason="source lane inherited from the spatial branch",
                through="source_region",
                confidence=0.9 if isinstance(side, str) else 0.5,
                alternatives=[] if isinstance(side, str) else ["left", "right"],
            )

        if interaction in {"gradual_braking", "hard_braking", "sudden_stop", "stationary_lead"}:
            deceleration = [4.0, 9.0] if interaction in {"hard_braking", "sudden_stop"} else [2.0, 5.0]
            if interaction != "stationary_lead":
                put(
                    "lead_deceleration_mps2",
                    deceleration,
                    unit="m/s^2",
                    reason=f"deceleration prior for {interaction}",
                    through="hazard_interaction",
                    confidence=0.84,
                )
            headway = [6.0, 18.0] if risk in {"aggressive", "critical"} else [12.0, 30.0]
            put(
                "initial_headway_m",
                headway,
                unit="m",
                reason="headway prior conditioned on the longitudinal risk branch",
                through="risk_level",
                confidence=0.79,
            )

        if interaction in {"left_turn_across_oncoming", "oncoming_path_conflict", "wrong_way_approach"}:
            put(
                "initial_oncoming_distance_m",
                [12.0, 30.0] if risk in {"aggressive", "critical"} else [20.0, 45.0],
                unit="m",
                reason="oncoming-distance prior inherited from the oncoming branch",
                through="risk_level",
                confidence=0.8,
            )

        params["completed"] = completed
        params["completion_policy"] = "recursive_parent_path_conditioned_completion"
        params["hierarchical_context"] = dict(values)
        params["hierarchical_path_signature"] = hierarchy.condition_path()
        out["parameter_layer"] = params
        return out

    @staticmethod
    def _conditions_through(path: HierarchicalScenePath, through: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for node in path.nodes:
            result[node.node_type] = node.value
            if node.node_type == through:
                break
        return result

    @staticmethod
    def _actor_speed_prior(actor: str, risk: str) -> Optional[List[float]]:
        priors: Dict[str, List[float]] = {
            "pedestrian": [1.0, 2.2],
            "child_pedestrian": [1.8, 3.2] if risk in {"aggressive", "critical"} else [1.2, 2.4],
            "jogger": [2.5, 4.5],
            "wheelchair_user": [0.6, 1.5],
            "cyclist": [3.0, 7.0],
            "ebike_rider": [4.0, 9.0],
            "scooter_rider": [3.0, 8.0],
            "lead_vehicle": [5.0, 20.0],
            "adjacent_vehicle": [8.0, 22.0],
            "merging_vehicle": [8.0, 22.0],
            "oncoming_vehicle": [8.0, 22.0],
            "cross_traffic_vehicle": [6.0, 18.0],
            "circulating_vehicle": [4.0, 12.0],
            "generic_vehicle": [7.0, 22.0],
            "barrier": [0.0, 0.0],
            "traffic_cone": [0.0, 0.0],
            "debris": [0.0, 0.0],
            "parked_vehicle": [0.0, 0.0],
            "generic_obstacle": [0.0, 0.0],
        }
        return priors.get(actor)

    @staticmethod
    def _ego_speed_prior(traffic_space: str, risk: str) -> Optional[List[float]]:
        priors: Dict[str, List[float]] = {
            "curbside_zone": [5.0, 10.0],
            "crosswalk_zone": [4.0, 9.0],
            "single_lane": [5.0, 15.0],
            "same_direction_multi_lane": [8.0, 20.0],
            "bidirectional_road": [7.0, 18.0],
            "straight_through": [5.0, 13.0],
            "left_turn_path": [3.0, 8.0],
            "right_turn_path": [3.0, 8.0],
            "cross_traffic_zone": [4.0, 10.0],
            "entry_path": [3.0, 9.0],
            "circulating_lane": [4.0, 12.0],
            "exit_path": [4.0, 12.0],
            "ramp_merge": [8.0, 20.0],
            "lane_drop": [6.0, 16.0],
            "diverge": [7.0, 18.0],
            "open_lane": [4.0, 12.0],
            "partially_blocked_lane": [3.0, 9.0],
            "closed_lane": [0.0, 5.0],
        }
        value = priors.get(traffic_space)
        if value and risk == "critical":
            return [value[0], max(value[0], value[1] * 0.85)]
        return value

    @staticmethod
    def _occluder_parameter_type(auxiliary: str) -> str:
        mapping = {
            "parked_car_occluder": "parked_vehicle",
            "parked_truck_occluder": "truck",
            "bus_occluder": "bus",
            "van_occluder": "van",
            "barrier_occluder": "barrier",
            "vegetation_occluder": "vegetation",
            "building_edge_occluder": "building_edge",
            "generic_occluder": "vehicle",
        }
        return mapping.get(auxiliary, "vehicle")

    @staticmethod
    def _occluder_size_prior(auxiliary: str) -> Tuple[List[float], List[float]]:
        priors: Dict[str, Tuple[List[float], List[float]]] = {
            "parked_car_occluder": ([3.8, 5.2], [1.7, 2.1]),
            "parked_truck_occluder": ([6.0, 12.0], [2.2, 2.8]),
            "bus_occluder": ([8.0, 14.0], [2.3, 2.7]),
            "van_occluder": ([4.8, 7.0], [1.9, 2.3]),
            "barrier_occluder": ([2.0, 8.0], [0.3, 1.0]),
            "vegetation_occluder": ([2.0, 8.0], [1.0, 4.0]),
            "building_edge_occluder": ([5.0, 20.0], [2.0, 8.0]),
            "generic_occluder": ([4.0, 8.0], [1.8, 2.5]),
        }
        return priors.get(auxiliary, priors["generic_occluder"])


class HierarchicalEventFramePipeline:
    """End-to-end natural-language pipeline using the recursive hierarchy."""

    def __init__(
        self,
        *,
        llm_provider: str = "none",
        llm_model: str = "qwen2.5:7b",
        ollama_url: str = "http://127.0.0.1:11434",
        allow_fallback: bool = True,
        no_repair: bool = False,
        respect_llm_event_sequence: bool = False,
    ) -> None:
        self.parser = EventFrameParser(
            llm_provider=llm_provider,
            llm_model=llm_model,
            ollama_url=ollama_url,
            allow_fallback=allow_fallback,
        )
        self.sequence_builder = EventSequenceBuilder()
        self.frame_verifier = EventFrameVerifier()
        self.mapper = EventFrameToHazardSpecMapper()
        self.hierarchy_resolver = HierarchicalSceneResolver()
        self.parameter_filler = HierarchicalParameterFiller()
        self.no_repair = no_repair
        self.respect_llm_event_sequence = respect_llm_event_sequence

    def parse_to_result(self, sentence: str) -> HierarchicalPipelineResult:
        frame = self.parser.parse(sentence)
        frame = self.sequence_builder.build(frame, overwrite=not self.respect_llm_event_sequence)
        verify0 = self.frame_verifier.verify_frame(frame)
        if not verify0.passed and not self.no_repair:
            frame = self.frame_verifier.repair_frame(frame)
            frame = self.sequence_builder.build(frame, overwrite=not self.respect_llm_event_sequence)
        verify1 = self.frame_verifier.verify_frame(frame)

        spec = self.mapper.map(frame)
        hierarchy = self.hierarchy_resolver.resolve(frame, spec)
        spec = attach_hierarchy(spec, hierarchy)
        spec = self.parameter_filler.fill(spec, frame, hierarchy)

        legacy_ok, legacy_errors = validate_spec(spec)
        hierarchy_errors = list(hierarchy.issues)
        validation = spec.setdefault("validation_layer", {})
        validation["legacy_spec_valid"] = legacy_ok
        validation["legacy_spec_errors"] = list(legacy_errors)
        validation["hierarchy_valid"] = hierarchy.valid
        validation["hierarchy_issues"] = hierarchy_errors
        validation["pipeline_design"] = "eventframe_recursive_hierarchical_tree"

        return HierarchicalPipelineResult(
            frame=frame,
            spec=spec,
            frame_issues=list(verify1.issues),
            spec_issues=list(legacy_errors),
            hierarchy_issues=hierarchy_errors,
        )

    def parse_to_spec(self, sentence: str) -> Tuple[EventFrame, Dict[str, Any]]:
        result = self.parse_to_result(sentence)
        return result.frame, result.spec


def attach_hierarchy(spec: Dict[str, Any], hierarchy: HierarchicalScenePath) -> Dict[str, Any]:
    """Attach a hierarchy without removing legacy spec layers."""

    out = deepcopy(spec)
    out["schema_version"] = hierarchy.schema_version
    out["hierarchy_layer"] = hierarchy.to_dict()
    out.setdefault("validation_layer", {})["composition_policy"] = (
        "recursive_parent_constrained_hierarchy_with_legacy_projection"
    )
    return out


def validate_hierarchical_spec(spec: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    """Validate the serialized hierarchy layer independently of EventFrame."""

    layer = dict(spec.get("hierarchy_layer", {}) or {})
    issues: List[str] = list(layer.get("issues", []) or [])
    path = list(layer.get("selected_path", []) or [])
    if not path:
        issues.append("hierarchy_selected_path_required")
        return False, issues

    required_order = [
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
    ]
    actual_order = [str(node.get("node_type", "")) for node in path]
    if actual_order != required_order:
        issues.append(f"hierarchy_order_mismatch:{actual_order}")

    for index, node in enumerate(path, start=1):
        if int(node.get("level", -1)) != index:
            issues.append(f"hierarchy_level_mismatch:{node.get('node_type')}={node.get('level')}")
        if node.get("value") in {None, "", "unknown"}:
            issues.append(f"hierarchy_unknown_value:{node.get('node_type')}")

    return not issues, issues


DefaultLanguageUnderstandingPipeline = HierarchicalEventFramePipeline
