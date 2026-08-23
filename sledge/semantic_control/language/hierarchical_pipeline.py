"""EventFrame pipeline with nuPlan-compatible hierarchical semantics.

The module keeps the established EventFrame parser, verifier and legacy mapper,
then adds four strict stages:

1. resolve one parent-constrained semantic path;
2. project linguistic actor descriptions to nuPlan/SLEDGE categories;
3. complete a full executable *parameter template* with provenance and hard
   geometric constraints; and
4. report separately whether the semantic template is complete and whether a
   concrete sampled scene has already passed kinematic checks.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import re

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


PARAMETER_ALIASES: Dict[str, str] = {
    "actor_speed": "actor_speed_mps",
    "pedestrian_speed": "actor_speed_mps",
    "ego_speed": "ego_speed_mps",
    "initial_distance": "ego_distance_to_conflict_m",
    "distance_to_conflict": "ego_distance_to_conflict_m",
    "occluder_length": "occluder_length_m",
    "occluder_width": "occluder_width_m",
}


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
        return (
            not self.frame_issues
            and not self.spec_issues
            and not self.hierarchy_issues
        )


class HierarchicalParameterFiller:
    """Complete executable leaves from an already selected hierarchy path.

    Ranges and derived expressions form a sampling template. They are not
    misreported as a concrete scene. A separate downstream sampler must choose
    one value per range and run the hard constraints before the scene is ready
    for nuPlan simulation.
    """

    def __init__(
        self, base_filler: Optional[MissingInfoFiller] = None
    ) -> None:
        self.base_filler = base_filler or MissingInfoFiller()

    def fill(
        self,
        spec: Dict[str, Any],
        frame: EventFrame,
        hierarchy: HierarchicalScenePath,
    ) -> Dict[str, Any]:
        out = self.base_filler.fill(spec, frame)
        params = out.setdefault("parameter_layer", {})
        completed: Dict[str, Dict[str, Any]] = self._normalize_completed(
            dict(params.get("completed", {}) or {}), frame
        )
        values = hierarchy.values
        projection = hierarchy.nuplan_projection()

        for original_name, parameter in frame.completed_parameters.items():
            source = str(parameter.source or "unknown")
            if source not in {"user_input", "explicit", "llm_explicit"}:
                continue
            name = PARAMETER_ALIASES.get(original_name, original_name)
            completed[name] = self._entry(
                parameter.value,
                unit=parameter.unit,
                source="user_input",
                reason=parameter.reason
                or "explicit parameter from EventFrame",
                confidence=1.0,
                evidence=[frame.sentence],
                conditioned_on={},
                condition_path="",
                is_assumption=False,
            )

        def put(
            name: str,
            value: Any,
            *,
            unit: str = "",
            reason: str,
            through: str,
            confidence: float,
            source: str = "hierarchical_prior",
            is_assumption: bool = True,
            evidence: Optional[List[str]] = None,
            alternatives: Optional[List[Any]] = None,
            overwrite_inferred: bool = True,
        ) -> None:
            current = completed.get(name)
            if current and current.get("source") == "user_input":
                return
            if current and not overwrite_inferred:
                return
            completed[name] = self._entry(
                value,
                unit=unit,
                source=source,
                reason=reason,
                confidence=confidence,
                evidence=list(evidence or []),
                conditioned_on=self._conditions_through(
                    hierarchy, through
                ),
                condition_path=hierarchy.condition_path(through=through),
                is_assumption=is_assumption,
                alternatives=list(alternatives or []),
            )

        actor = values.get("primary_actor_type", "unknown")
        interaction = values.get("hazard_interaction", "unknown")
        traffic_space = values.get("ego_traffic_space", "unknown")
        auxiliary = values.get("auxiliary_entity", "none")
        source_region = values.get("source_region", "unknown")
        direction = values.get("motion_direction", "toward_ego_path")
        risk = values.get("risk_level", "moderate")
        visibility = values.get("visibility", "fully_visible")

        # Final explicit-speed preservation and correction.
        #
        # MissingInfoFiller may already have interpreted a bare explicit speed
        # as ego_speed_mps. For vulnerable-road-user prompts that is wrong
        # unless the sentence explicitly binds the speed to ego. Therefore:
        #
        #   pedestrian ... at 1.2 m/s
        #       -> actor_speed_mps = 1.2
        #       -> remove erroneous ego_speed_mps = 1.2
        #       -> later restore ego_speed_mps from the hierarchy prior
        #
        #   Ego travels at 8 m/s
        #       -> ego_speed_mps = 8.0
        #
        # Explicit user values always dominate hierarchical priors.
        explicit_speed_mps = self._extract_explicit_speed_mps(
            frame.sentence
        )
        if explicit_speed_mps is not None:
            speed_is_ego_scoped = self._speed_is_explicitly_ego_scoped(
                frame.sentence
            )

            if speed_is_ego_scoped:
                completed["ego_speed_mps"] = self._entry(
                    explicit_speed_mps,
                    unit="m/s",
                    source="user_input",
                    reason=(
                        "explicit ego speed detected directly from "
                        "the original prompt"
                    ),
                    confidence=1.0,
                    evidence=[frame.sentence],
                    conditioned_on={},
                    condition_path="",
                    is_assumption=False,
                )

            elif actor in {"pedestrian", "cyclist"}:
                completed["actor_speed_mps"] = self._entry(
                    explicit_speed_mps,
                    unit="m/s",
                    source="user_input",
                    reason=(
                        "explicit vulnerable-road-user speed detected "
                        "directly from the original prompt"
                    ),
                    confidence=1.0,
                    evidence=[frame.sentence],
                    conditioned_on={
                        "primary_actor_type": actor,
                        "hazard_interaction": interaction,
                    },
                    condition_path=hierarchy.condition_path(
                        through="primary_actor_type"
                    ),
                    is_assumption=False,
                )

                # MissingInfoFiller can still leave the same bare numeric speed
                # under ego_speed_mps. Remove only that duplicated value. Once
                # removed, the normal hierarchy ego-speed prior below will be
                # inserted by put().
                existing_ego_speed = completed.get("ego_speed_mps")
                if isinstance(existing_ego_speed, Mapping):
                    existing_value = existing_ego_speed.get("value")
                    try:
                        same_as_explicit_speed = (
                            abs(
                                float(existing_value)
                                - float(explicit_speed_mps)
                            )
                            <= 1e-9
                        )
                    except (TypeError, ValueError):
                        same_as_explicit_speed = False

                    if same_as_explicit_speed:
                        completed.pop("ego_speed_mps", None)

        put(
            "lane_width_m",
            [3.2, 3.8],
            unit="m",
            reason="nuPlan-compatible lane-width sampling range",
            through="ego_traffic_space",
            confidence=0.78,
        )
        put(
            "lane_count",
            (
                [1, 2]
                if traffic_space
                in {
                    "single_lane",
                    "curbside_zone",
                    "bidirectional_road",
                }
                else [2, 4]
            ),
            reason=f"lane-count prior for {traffic_space}",
            through="ego_traffic_space",
            confidence=0.7,
        )
        put(
            "road_curvature",
            [-0.005, 0.005],
            unit="1/m",
            reason="near-straight local road curvature prior",
            through="road_topology",
            confidence=0.68,
        )

        ego_distance = (
            [6.0, 14.0]
            if risk in {"aggressive", "critical"}
            else [10.0, 24.0]
        )
        put(
            "ego_distance_to_conflict_m",
            ego_distance,
            unit="m",
            reason="distance from ego origin to the selected conflict point",
            through="risk_level",
            confidence=0.78,
        )
        put(
            "conflict_point_xy",
            {
                "frame": "ego_local",
                "x_m": {"reference": "ego_distance_to_conflict_m"},
                "y_m": [-0.5, 0.5],
            },
            unit="m",
            reason="conflict point lies ahead on the ego path",
            through="target_region",
            confidence=0.95,
            source="derived_constraint",
            is_assumption=False,
        )
        put(
            "ego_acceleration_mps2",
            [0.0, 0.0],
            unit="m/s^2",
            reason="ego initially follows a constant-speed baseline",
            through="ego_traffic_space",
            confidence=0.8,
        )

        actor_speed = self._actor_speed_prior(actor, risk)
        if actor_speed is not None:
            put(
                "actor_speed_mps",
                actor_speed,
                unit="m/s",
                reason=(
                    "nuPlan/SLEDGE actor-speed prior for executable "
                    f"category {actor}"
                ),
                through="risk_level",
                confidence=0.86,
            )
        ego_speed = self._ego_speed_prior(traffic_space, risk)
        if ego_speed is not None:
            put(
                "ego_speed_mps",
                ego_speed,
                unit="m/s",
                reason=(
                    f"ego-speed prior for {traffic_space} "
                    f"under {risk} semantic risk"
                ),
                through="risk_level",
                confidence=0.76,
            )

        # --------------------------------------------------------------
        # Final speed-binding contract.
        #
        # This block deliberately runs AFTER both actor and ego speed
        # priors have been considered.  Earlier layers may have preserved a
        # bare numeric speed as ``ego_speed_mps`` with source=user_input.
        # Because ``put`` correctly protects user_input, that stale binding
        # can otherwise survive even after the same number has been assigned
        # to the pedestrian.  The benchmark semantics are unambiguous:
        #
        #   "pedestrian ... at 1.2 m/s"
        #       -> actor_speed_mps = 1.2 (user_input)
        #       -> ego_speed_mps   = hierarchy prior
        #
        # A speed remains attached to ego only when the text explicitly says
        # that ego travels/moves/drives at that speed.
        # --------------------------------------------------------------
        if (
            explicit_speed_mps is not None
            and actor in {"pedestrian", "cyclist"}
            and not self._speed_is_explicitly_ego_scoped(frame.sentence)
        ):
            completed["actor_speed_mps"] = self._entry(
                explicit_speed_mps,
                unit="m/s",
                source="user_input",
                reason=(
                    "explicit vulnerable-road-user speed detected "
                    "directly from the original prompt"
                ),
                confidence=1.0,
                evidence=[frame.sentence],
                conditioned_on={
                    "primary_actor_type": actor,
                    "hazard_interaction": interaction,
                },
                condition_path=hierarchy.condition_path(
                    through="primary_actor_type"
                ),
                is_assumption=False,
            )

            corrected_ego_speed = self._ego_speed_prior(
                traffic_space, risk
            )
            if corrected_ego_speed is not None:
                completed["ego_speed_mps"] = self._entry(
                    corrected_ego_speed,
                    unit="m/s",
                    source="hierarchical_prior",
                    reason=(
                        f"ego-speed prior for {traffic_space} "
                        f"under {risk} semantic risk; explicit prompt "
                        "speed belongs to the vulnerable road user"
                    ),
                    confidence=0.76,
                    evidence=[],
                    conditioned_on=self._conditions_through(
                        hierarchy, "risk_level"
                    ),
                    condition_path=hierarchy.condition_path(
                        through="risk_level"
                    ),
                    is_assumption=True,
                )

        put(
            "actor_acceleration_mps2",
            [0.0, 1.0] if actor == "pedestrian" else [-1.0, 1.0],
            unit="m/s^2",
            reason="bounded initial actor acceleration prior",
            through="primary_actor_type",
            confidence=0.68,
        )
        put(
            "actor_start_time_s",
            [0.2, 2.0],
            unit="s",
            reason=(
                "actor motion begins after the ego baseline is established"
            ),
            through="trigger_event",
            confidence=0.72,
        )
        put(
            "minimum_clearance_m",
            [0.5, 2.0],
            unit="m",
            reason="non-collision safety clearance range",
            through="risk_level",
            confidence=0.7,
        )
        put(
            "braking_deceleration_mps2",
            [3.0, 7.0],
            unit="m/s^2",
            reason=(
                "candidate ego braking envelope; exact response is "
                "selected after kinematic evaluation"
            ),
            through="ego_required_response",
            confidence=0.72,
        )

        if interaction in {
            "path_crossing",
            "enter_ego_lane",
            "occluded_emergence",
        }:
            if interaction == "occluded_emergence":
                put(
                    "crossing_direction",
                    "occluder_to_ego_path",
                    reason=(
                        "the actor moves from its selected occluder "
                        "toward the ego path"
                    ),
                    through="motion_direction",
                    confidence=1.0,
                    source="derived_constraint",
                    is_assumption=False,
                )
                put(
                    "actor_heading",
                    {
                        "frame": "ego_local",
                        "definition": (
                            "unit_vector(occluder_position, "
                            "nearest_point_on_ego_path)"
                        ),
                    },
                    reason=(
                        "heading is uniquely derived from occluder "
                        "position toward the ego path"
                    ),
                    through="motion_direction",
                    confidence=1.0,
                    source="derived_constraint",
                    is_assumption=False,
                )
            else:
                put(
                    "crossing_direction",
                    direction,
                    reason=(
                        "direction inherited from the selected spatial branch"
                    ),
                    through="motion_direction",
                    confidence=0.88,
                    source=(
                        "derived_constraint"
                        if direction == "toward_ego_path"
                        else "hierarchical_prior"
                    ),
                    is_assumption=direction != "toward_ego_path",
                )
                put(
                    "actor_heading",
                    {
                        "frame": "ego_local",
                        "definition": direction,
                    },
                    reason="heading follows the selected crossing direction",
                    through="motion_direction",
                    confidence=0.85,
                )

        if interaction == "occluded_emergence" or visibility in {
            "partially_occluded",
            "fully_occluded",
        }:
            occluder_explicit = bool(frame.occlusion.enabled) and str(
                frame.occlusion.occluder_type or "unknown"
            ) not in {"", "unknown"}
            occluder_type = self._occluder_parameter_type(auxiliary)

            put(
                "occlusion_enabled",
                True,
                reason=(
                    "the prompt explicitly defines an "
                    "occluded-emergence relation"
                ),
                through="visibility",
                confidence=1.0,
                source=(
                    "user_input"
                    if frame.occlusion.enabled
                    else "derived_constraint"
                ),
                is_assumption=False,
                evidence=[
                    frame.occlusion.evidence_text or frame.sentence
                ],
            )
            put(
                "occluder_type",
                occluder_type,
                reason=(
                    "the prompt occluder is normalized to a "
                    "nuPlan/SLEDGE-supported category"
                ),
                through="auxiliary_entity",
                confidence=0.99 if occluder_explicit else 0.72,
                source=(
                    "user_input"
                    if occluder_explicit
                    else "hierarchical_prior"
                ),
                is_assumption=not occluder_explicit,
                evidence=(
                    [frame.occlusion.evidence_text or frame.sentence]
                    if occluder_explicit
                    else []
                ),
            )
            put(
                "occluder_side",
                self._occluder_side_value(source_region),
                reason=(
                    "one occluder side is sampled once; direction is "
                    "then derived from that position toward ego"
                ),
                through="source_region",
                confidence=(
                    0.92
                    if source_region in {"left_side", "right_side"}
                    else 0.65
                ),
                source=(
                    "derived_constraint"
                    if source_region in {"left_side", "right_side"}
                    else "categorical_prior"
                ),
                alternatives=(
                    []
                    if source_region in {"left_side", "right_side"}
                    else ["left", "right"]
                ),
            )
            put(
                "occluder_lateral_offset_m",
                [1.0, 4.0],
                unit="m",
                reason="roadside offset from the ego-path boundary",
                through="auxiliary_entity",
                confidence=0.72,
            )
            put(
                "occluder_position",
                {
                    "frame": "ego_local",
                    "x_m": [4.0, 12.0],
                    "side": {"reference": "occluder_side"},
                    "lateral_offset_m": {
                        "reference": "occluder_lateral_offset_m"
                    },
                    "placement": "outside_ego_lane",
                },
                unit="m",
                reason="occluder is placed roadside and ahead of ego",
                through="auxiliary_entity",
                confidence=0.82,
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
            put(
                "actor_initial_position",
                {
                    "frame": "ego_local",
                    "relation": (
                        "behind_occluder_away_from_ego_path"
                    ),
                    "occluder_reference": "occluder_position",
                    "hidden_offset_m": [0.5, 2.0],
                },
                unit="m",
                reason=(
                    "pedestrian begins behind the occluder on the "
                    "side away from the ego path"
                ),
                through="auxiliary_entity",
                confidence=0.96,
                source="derived_constraint",
                is_assumption=False,
            )
            reveal = (
                [3.0, 8.0]
                if risk in {"aggressive", "critical"}
                else [5.0, 12.0]
            )
            put(
                "reveal_distance_m",
                reveal,
                unit="m",
                reason=(
                    "ego-to-conflict distance when the actor first "
                    "clears the occluder"
                ),
                through="risk_level",
                confidence=0.82,
            )
            put(
                "initial_gap_m",
                {
                    "definition": (
                        "distance(ego_position, actor_initial_position)"
                    )
                },
                unit="m",
                reason=(
                    "initial gap is derived after actor and occluder "
                    "positions are sampled"
                ),
                through="auxiliary_entity",
                confidence=1.0,
                source="derived_constraint",
                is_assumption=False,
            )
            put(
                "time_to_collision_s",
                {
                    "definition": (
                        "ego_distance_to_conflict_m / "
                        "max(ego_speed_mps, epsilon)"
                    ),
                    "evaluate_at": "actor_reveal_time",
                },
                unit="s",
                reason=(
                    "TTC is computed after sampling rather than "
                    "independently sampled"
                ),
                through="risk_level",
                confidence=1.0,
                source="derived_constraint",
                is_assumption=False,
            )

        if "actor_initial_position" not in completed:
            put(
                "actor_initial_position",
                {
                    "frame": "ego_local",
                    "relation": "source_region",
                    "source_region": source_region,
                },
                unit="m",
                reason=(
                    "actor start position is sampled inside the "
                    "selected source region"
                ),
                through="source_region",
                confidence=0.7,
            )
        if "actor_heading" not in completed:
            put(
                "actor_heading",
                {
                    "frame": "ego_local",
                    "definition": direction,
                },
                reason="actor heading follows the selected motion direction",
                through="motion_direction",
                confidence=0.72,
            )
        if "occluder_position" not in completed:
            put(
                "occluder_position",
                None,
                unit="m",
                reason="no occluder is active for this branch",
                through="auxiliary_entity",
                confidence=1.0,
                source="not_applicable",
                is_assumption=False,
            )
        if "occluder_length_m" not in completed:
            put(
                "occluder_length_m",
                0.0,
                unit="m",
                reason="no occluder is active for this branch",
                through="auxiliary_entity",
                confidence=1.0,
                source="not_applicable",
                is_assumption=False,
            )
        if "occluder_width_m" not in completed:
            put(
                "occluder_width_m",
                0.0,
                unit="m",
                reason="no occluder is active for this branch",
                through="auxiliary_entity",
                confidence=1.0,
                source="not_applicable",
                is_assumption=False,
            )
        if "occluder_lateral_offset_m" not in completed:
            put(
                "occluder_lateral_offset_m",
                0.0,
                unit="m",
                reason="no occluder is active for this branch",
                through="auxiliary_entity",
                confidence=1.0,
                source="not_applicable",
                is_assumption=False,
            )
        if "reveal_distance_m" not in completed:
            put(
                "reveal_distance_m",
                None,
                unit="m",
                reason="no reveal event is active for this branch",
                through="visibility",
                confidence=1.0,
                source="not_applicable",
                is_assumption=False,
            )
        if "initial_gap_m" not in completed:
            put(
                "initial_gap_m",
                {
                    "definition": (
                        "distance(ego_position, actor_initial_position)"
                    )
                },
                unit="m",
                reason="initial gap is derived from sampled states",
                through="source_region",
                confidence=1.0,
                source="derived_constraint",
                is_assumption=False,
            )
        if "time_to_collision_s" not in completed:
            put(
                "time_to_collision_s",
                {
                    "definition": (
                        "relative_distance / max(closing_speed, epsilon)"
                    )
                },
                unit="s",
                reason="TTC is derived from sampled states",
                through="risk_level",
                confidence=1.0,
                source="derived_constraint",
                is_assumption=False,
            )

        if interaction in {
            "cut_in",
            "aggressive_cut_in",
            "lane_change",
            "lane_encroachment",
        }:
            put(
                "initial_longitudinal_gap_m",
                (
                    [3.0, 10.0]
                    if interaction == "aggressive_cut_in"
                    else [7.0, 20.0]
                ),
                unit="m",
                reason=(
                    "gap prior inherited from the vehicle cut-in branch"
                ),
                through="risk_level",
                confidence=0.8,
            )
        if interaction in {
            "gradual_braking",
            "hard_braking",
            "sudden_stop",
        }:
            put(
                "lead_deceleration_mps2",
                (
                    [4.0, 9.0]
                    if interaction in {"hard_braking", "sudden_stop"}
                    else [2.0, 5.0]
                ),
                unit="m/s^2",
                reason=f"deceleration prior for {interaction}",
                through="hazard_interaction",
                confidence=0.84,
            )

        params["completed"] = completed
        params["completion_policy"] = (
            "nuplan_projected_parent_path_conditioned_template"
        )
        params["hierarchical_context"] = dict(values)
        params["hierarchical_path_signature"] = hierarchy.condition_path()
        params["parameter_constraints"] = self._parameter_constraints(
            interaction
        )
        self._synchronize_missing(params, completed)

        required_parameters = {
            name
            for names in hierarchy.executable_parameter_groups.values()
            for name in names
        }
        missing_template_parameters = sorted(
            required_parameters - set(completed)
        )
        params["template_required_parameters"] = sorted(
            required_parameters
        )
        params["template_missing_parameters"] = (
            missing_template_parameters
        )
        params["parameter_template_complete"] = (
            not missing_template_parameters
        )

        out["parameter_layer"] = params
        out["nuplan_layer"] = self._nuplan_layer(
            projection, completed, auxiliary
        )

        base_template_ready = bool(
            hierarchy.valid
            and projection.get("compatible", False)
            and not missing_template_parameters
        )
        occluded_contract = self._occluded_pedestrian_contract(
            frame=frame,
            hierarchy=hierarchy,
            projection=projection,
            completed=completed,
        )
        contract_applicable = bool(occluded_contract["applicable"])
        contract_passed = bool(occluded_contract["passed"])
        scene_template_ready = (
            base_template_ready
            and (not contract_applicable or contract_passed)
        )

        out["readiness"] = {
            "semantic_understanding": (
                "passed"
                if hierarchy.valid and (not contract_applicable or contract_passed)
                else "failed"
            ),
            "hierarchical_path": (
                "passed" if hierarchy.valid else "failed"
            ),
            "nuplan_category_projection": (
                "passed" if projection.get("compatible") else "failed"
            ),
            "parameter_template": (
                "complete"
                if not missing_template_parameters
                else "incomplete"
            ),
            "occluded_pedestrian_contract": (
                "passed"
                if contract_applicable and contract_passed
                else "failed"
                if contract_applicable
                else "not_applicable"
            ),
            "occluded_pedestrian_contract_issues": list(
                occluded_contract["issues"]
            ),
            "kinematic_consistency": "pending_concrete_sampling",
            "scene_template_ready": scene_template_ready,
            "sampled_scene_ready": False,
        }
        return out

    @staticmethod
    def _looks_like_occluded_pedestrian_prompt(frame: EventFrame) -> bool:
        """Detect the benchmark family from raw wording, independent of parsing.

        This guard prevents an internally consistent but semantically wrong
        vehicle/cut-in parse from being marked ``scene_template_ready=True`` for
        a prompt that clearly describes an occluded human-on-foot hazard.
        """

        sentence = (frame.sentence or "").lower()
        human_terms = (
            "pedestrian",
            "walker",
            "person",
            "adult",
            "child",
            "kid",
            "schoolkid",
            "schoolboy",
            "schoolgirl",
            "boy",
            "girl",
            "jogger",
            "runner",
            "wheelchair user",
            "on foot",
        )
        occlusion_terms = (
            "behind",
            "hidden",
            "concealed",
            "obscured",
            "occluded",
            "screened",
            "masked",
            "from behind",
            "emerges from",
            "comes out from",
        )
        return (
            any(term in sentence for term in human_terms)
            and any(term in sentence for term in occlusion_terms)
        )

    @classmethod
    def _occluded_pedestrian_contract(
        cls,
        *,
        frame: EventFrame,
        hierarchy: HierarchicalScenePath,
        projection: Mapping[str, Any],
        completed: Mapping[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Family-specific semantic gate used by the occluded-pedestrian study."""

        applicable = cls._looks_like_occluded_pedestrian_prompt(frame)
        if not applicable:
            return {"applicable": False, "passed": True, "issues": []}

        values = hierarchy.values
        issues: List[str] = []
        if values.get("primary_actor_type") != "pedestrian":
            issues.append("primary_actor_type_must_be_pedestrian")
        if values.get("hazard_interaction") != "occluded_emergence":
            issues.append("hazard_interaction_must_be_occluded_emergence")
        if values.get("motion_direction") != "occluder_to_ego_path":
            issues.append("motion_direction_must_be_occluder_to_ego_path")
        if not str(values.get("auxiliary_entity", "")).endswith("_occluder"):
            issues.append("occluder_entity_required")
        if projection.get("tracked_object_type") != "TrackedObjectType.PEDESTRIAN":
            issues.append("nuplan_projection_must_be_pedestrian")
        if projection.get("sledge_collection") != "pedestrians":
            issues.append("sledge_collection_must_be_pedestrians")

        crossing = completed.get("crossing_direction", {})
        crossing_value = (
            crossing.get("value")
            if isinstance(crossing, Mapping)
            else crossing
        )
        if crossing_value != "occluder_to_ego_path":
            issues.append("crossing_direction_must_be_unique_relative_direction")

        return {
            "applicable": True,
            "passed": not issues,
            "issues": issues,
        }

    @staticmethod
    def _entry(
        value: Any,
        *,
        unit: str,
        source: str,
        reason: str,
        confidence: float,
        evidence: List[str],
        conditioned_on: Mapping[str, str],
        condition_path: str,
        is_assumption: bool,
        alternatives: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "value": value,
            "unit": unit,
            "source": source,
            "reason": reason,
            "confidence": max(
                0.0, min(float(confidence), 1.0)
            ),
            "evidence": list(evidence),
            "conditioned_on": dict(conditioned_on),
            "condition_path": condition_path,
            "is_assumption": bool(is_assumption),
            "alternatives": list(alternatives or []),
        }

    def _normalize_completed(
        self,
        completed: Dict[str, Any],
        frame: EventFrame,
    ) -> Dict[str, Dict[str, Any]]:
        normalized: Dict[str, Dict[str, Any]] = {}
        for original_name, raw_entry in completed.items():
            name = PARAMETER_ALIASES.get(
                original_name, original_name
            )
            entry = (
                dict(raw_entry)
                if isinstance(raw_entry, dict)
                else {"value": raw_entry}
            )
            source = str(entry.get("source", "unknown"))
            normalized[name] = self._entry(
                entry.get("value"),
                unit=str(entry.get("unit", "")),
                source=source,
                reason=str(entry.get("reason", "")),
                confidence=float(
                    entry.get(
                        "confidence",
                        1.0 if source == "user_input" else 0.6,
                    )
                ),
                evidence=list(
                    entry.get(
                        "evidence",
                        (
                            [frame.sentence]
                            if source == "user_input"
                            else []
                        ),
                    )
                    or []
                ),
                conditioned_on=dict(
                    entry.get("conditioned_on", {}) or {}
                ),
                condition_path=str(
                    entry.get("condition_path", "")
                ),
                is_assumption=bool(
                    entry.get(
                        "is_assumption",
                        source != "user_input",
                    )
                ),
                alternatives=list(
                    entry.get("alternatives", []) or []
                ),
            )
        return normalized

    @staticmethod
    def _conditions_through(
        path: HierarchicalScenePath,
        through: str,
    ) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for node in path.nodes:
            result[node.node_type] = node.value
            if node.node_type == through:
                break
        return result

    @staticmethod
    def _extract_explicit_speed_mps(
        sentence: str,
    ) -> Optional[float]:
        """Extract one explicit numeric speed and normalize it to m/s.

        Supported forms include:
        - 1.2 m/s
        - 1.2 meter per second
        - 1.2 meters per second
        - 20 mph
        - 30 km/h
        - 30 kph
        - 30 kmph
        """

        text = str(sentence or "").lower()

        match = re.search(
            (
                r"(\d+(?:\.\d+)?)"
                r"\s*"
                r"(m/s|meter per second|meters per second)"
            ),
            text,
        )
        if match:
            return float(match.group(1))

        match = re.search(
            r"(\d+(?:\.\d+)?)\s*mph",
            text,
        )
        if match:
            return round(float(match.group(1)) * 0.44704, 3)

        match = re.search(
            (
                r"(\d+(?:\.\d+)?)"
                r"\s*"
                r"(km/h|kph|kmph)"
            ),
            text,
        )
        if match:
            return round(float(match.group(1)) / 3.6, 3)

        return None

    @staticmethod
    def _speed_is_explicitly_ego_scoped(
        sentence: str,
    ) -> bool:
        """Return True only when the speed is clearly bound to ego.

        A bare speed in an occluded-pedestrian prompt must not be interpreted
        as ego speed.

        Examples classified as ego speed:
            "Ego travels at 8 m/s."
            "The ego vehicle moves at 10 m/s."
            "Ego speed is 9 m/s."

        Examples classified as actor speed:
            "A pedestrian enters the lane at 1.2 m/s."
            "A jogger crosses at 1.9 m/s."
        """

        text = str(sentence or "").lower()
        patterns = [
            (
                r"\bego\b"
                r".{0,60}"
                r"(travels?|moves?|drives?|speed|velocity)"
                r".{0,40}"
                r"\d+(?:\.\d+)?"
            ),
            (
                r"\bego vehicle\b"
                r".{0,60}"
                r"\d+(?:\.\d+)?"
            ),
            (
                r"\bego car\b"
                r".{0,60}"
                r"\d+(?:\.\d+)?"
            ),
            (
                r"\bego speed\b"
                r".{0,20}"
                r"\d+(?:\.\d+)?"
            ),
            (
                r"\bego velocity\b"
                r".{0,20}"
                r"\d+(?:\.\d+)?"
            ),
        ]
        return any(
            re.search(pattern, text) is not None
            for pattern in patterns
        )

    @staticmethod
    def _actor_speed_prior(
        actor: str, risk: str
    ) -> Optional[List[float]]:
        priors: Dict[str, List[float]] = {
            "pedestrian": (
                [1.0, 2.0]
                if risk in {"aggressive", "critical"}
                else [0.8, 1.6]
            ),
            "cyclist": [3.0, 7.0],
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
    def _ego_speed_prior(
        traffic_space: str, risk: str
    ) -> Optional[List[float]]:
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
            return [
                value[0],
                max(value[0], value[1] * 0.85),
            ]
        return value

    @staticmethod
    def _occluder_parameter_type(auxiliary: str) -> str:
        mapping = {
            "parked_car_occluder": "vehicle",
            "parked_truck_occluder": "vehicle",
            "bus_occluder": "vehicle",
            "van_occluder": "vehicle",
            "generic_vehicle_occluder": "vehicle",
            "barrier_occluder": "static_object",
            "vegetation_occluder": "static_object",
            "building_edge_occluder": "static_object",
        }
        return mapping.get(auxiliary, "vehicle")

    @staticmethod
    def _occluder_size_prior(
        auxiliary: str,
    ) -> Tuple[List[float], List[float]]:
        priors: Dict[
            str,
            Tuple[List[float], List[float]],
        ] = {
            "parked_car_occluder": (
                [3.8, 5.2],
                [1.7, 2.1],
            ),
            "parked_truck_occluder": (
                [6.0, 12.0],
                [2.2, 2.8],
            ),
            "bus_occluder": (
                [8.0, 14.0],
                [2.3, 2.7],
            ),
            "van_occluder": (
                [4.8, 7.0],
                [1.9, 2.3],
            ),
            "barrier_occluder": (
                [2.0, 8.0],
                [0.3, 1.0],
            ),
            "vegetation_occluder": (
                [2.0, 8.0],
                [1.0, 4.0],
            ),
            "building_edge_occluder": (
                [5.0, 20.0],
                [2.0, 8.0],
            ),
            "generic_vehicle_occluder": (
                [4.0, 8.0],
                [1.8, 2.5],
            ),
        }
        return priors.get(
            auxiliary,
            priors["generic_vehicle_occluder"],
        )

    @staticmethod
    def _occluder_side_value(source_region: str) -> Any:
        if source_region == "left_side":
            return "left"
        if source_region == "right_side":
            return "right"
        return {
            "distribution": "categorical",
            "values": ["left", "right"],
            "sample_once": True,
        }

    @staticmethod
    def _parameter_constraints(
        interaction: str,
    ) -> List[Dict[str, Any]]:
        constraints: List[Dict[str, Any]] = [
            {
                "id": "conflict_on_ego_path",
                "type": "hard",
                "expression": (
                    "conflict_point_xy.y_m lies within "
                    "ego lane boundaries"
                ),
            },
            {
                "id": "ttc_is_derived",
                "type": "derived",
                "expression": (
                    "time_to_collision_s is calculated from "
                    "sampled positions and velocities"
                ),
            },
            {
                "id": "response_after_kinematics",
                "type": "decision",
                "expression": (
                    "choose brake vs emergency_brake after "
                    "TTC and stopping-distance evaluation"
                ),
            },
        ]
        if interaction == "occluded_emergence":
            constraints.extend(
                [
                    {
                        "id": "nuplan_pedestrian_category",
                        "type": "hard",
                        "expression": (
                            "primary actor is "
                            "TrackedObjectType.PEDESTRIAN "
                            "and is stored in "
                            "SledgeVectorRaw.pedestrians"
                        ),
                    },
                    {
                        "id": "occluder_outside_ego_lane",
                        "type": "hard",
                        "expression": (
                            "occluder footprint does not "
                            "overlap the ego lane center corridor"
                        ),
                    },
                    {
                        "id": "actor_hidden_before_reveal",
                        "type": "hard",
                        "expression": (
                            "line_of_sight(ego, actor) "
                            "intersects occluder before "
                            "reveal_time_s"
                        ),
                    },
                    {
                        "id": "unique_relative_direction",
                        "type": "hard",
                        "expression": (
                            "actor velocity points from "
                            "occluder_position to "
                            "nearest_point_on_ego_path"
                        ),
                    },
                    {
                        "id": "actor_path_intersects_ego_path",
                        "type": "hard",
                        "expression": (
                            "pedestrian trajectory intersects "
                            "ego path at conflict_point_xy"
                        ),
                    },
                    {
                        "id": "reveal_before_conflict",
                        "type": "hard",
                        "expression": (
                            "0 < reveal_distance_m <= "
                            "ego_distance_to_conflict_m"
                        ),
                    },
                ]
            )
        return constraints

    @staticmethod
    def _synchronize_missing(
        params: Dict[str, Any],
        completed: Mapping[str, Dict[str, Any]],
    ) -> None:
        original_required = list(
            params.get("required_missing", []) or []
        )
        original_defaultable = list(
            params.get("defaultable_missing", []) or []
        )
        resolved: List[Dict[str, str]] = []
        unresolved: List[str] = []
        for original_name in original_required:
            canonical = PARAMETER_ALIASES.get(
                str(original_name),
                str(original_name),
            )
            if canonical in completed:
                resolved.append(
                    {
                        "original_name": str(original_name),
                        "resolved_as": canonical,
                    }
                )
            else:
                unresolved.append(canonical)
        params["original_required_missing"] = original_required
        params["original_defaultable_missing"] = original_defaultable
        params["resolved_missing"] = resolved
        params["required_missing"] = sorted(set(unresolved))
        params["unresolved_required"] = sorted(set(unresolved))

    @staticmethod
    def _nuplan_layer(
        projection: Mapping[str, Any],
        completed: Mapping[str, Dict[str, Any]],
        auxiliary: str,
    ) -> Dict[str, Any]:
        del completed
        return {
            "schema_version": "sledge_vector_raw_projection_v1",
            "coordinate_frame": "ego_local",
            "actor": {
                "semantic_detail": projection.get(
                    "language_actor_detail",
                    "unspecified",
                ),
                "nuplan_category": projection.get(
                    "actor_category"
                ),
                "tracked_object_type": projection.get(
                    "tracked_object_type"
                ),
                "sledge_collection": projection.get(
                    "sledge_collection"
                ),
                "state_mapping": {
                    "position": (
                        "parameter_layer.completed."
                        "actor_initial_position.value"
                    ),
                    "heading": (
                        "parameter_layer.completed."
                        "actor_heading.value"
                    ),
                    "speed": (
                        "parameter_layer.completed."
                        "actor_speed_mps.value"
                    ),
                    "acceleration": (
                        "parameter_layer.completed."
                        "actor_acceleration_mps2.value"
                    ),
                },
            },
            "occluder": {
                "semantic_type": auxiliary,
                "nuplan_category": projection.get(
                    "occluder_category"
                ),
                "sledge_collection": projection.get(
                    "occluder_sledge_collection"
                ),
                "state_mapping": {
                    "position": (
                        "parameter_layer.completed."
                        "occluder_position.value"
                    ),
                    "length": (
                        "parameter_layer.completed."
                        "occluder_length_m.value"
                    ),
                    "width": (
                        "parameter_layer.completed."
                        "occluder_width_m.value"
                    ),
                },
            },
            "compatible": bool(
                projection.get("compatible", False)
            ),
            "warnings": list(
                projection.get("warnings", []) or []
            ),
        }


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
        self.respect_llm_event_sequence = (
            respect_llm_event_sequence
        )

    def parse_to_result(
        self, sentence: str
    ) -> HierarchicalPipelineResult:
        frame = self.parser.parse(sentence)
        frame = self.sequence_builder.build(
            frame,
            overwrite=not self.respect_llm_event_sequence,
        )
        verify0 = self.frame_verifier.verify_frame(frame)
        if not verify0.passed and not self.no_repair:
            frame = self.frame_verifier.repair_frame(frame)
            frame = self.sequence_builder.build(
                frame,
                overwrite=not self.respect_llm_event_sequence,
            )
        verify1 = self.frame_verifier.verify_frame(frame)

        spec = self.mapper.map(frame)
        hierarchy = self.hierarchy_resolver.resolve(frame, spec)
        spec = attach_hierarchy(spec, hierarchy)
        spec = self._project_legacy_layers_to_nuplan(
            spec, hierarchy
        )
        spec = self._refine_event_sequence(
            spec, frame, hierarchy
        )
        spec = self.parameter_filler.fill(
            spec, frame, hierarchy
        )

        legacy_ok, legacy_errors = validate_spec(spec)
        hierarchy_errors = list(hierarchy.issues)
        hierarchy_ok, serialized_errors = (
            validate_hierarchical_spec(spec)
        )
        hierarchy_errors.extend(
            error
            for error in serialized_errors
            if error not in hierarchy_errors
        )

        readiness = dict(spec.get("readiness", {}) or {})
        if readiness.get("occluded_pedestrian_contract") == "failed":
            for issue in list(
                readiness.get("occluded_pedestrian_contract_issues", []) or []
            ):
                tagged = f"occluded_pedestrian_contract:{issue}"
                if tagged not in hierarchy_errors:
                    hierarchy_errors.append(tagged)

        validation = spec.setdefault("validation_layer", {})
        validation["legacy_spec_valid"] = legacy_ok
        validation["legacy_spec_errors"] = list(
            legacy_errors
        )
        validation["hierarchy_valid"] = hierarchy_ok
        validation["hierarchy_issues"] = hierarchy_errors
        validation["nuplan_projection_valid"] = bool(
            spec.get("nuplan_layer", {}).get(
                "compatible", False
            )
        )
        validation["pipeline_design"] = (
            "eventframe_v6_nuplan_parent_constrained_tree"
        )

        return HierarchicalPipelineResult(
            frame=frame,
            spec=spec,
            frame_issues=list(verify1.issues),
            spec_issues=list(legacy_errors),
            hierarchy_issues=hierarchy_errors,
        )

    def parse_to_spec(
        self, sentence: str
    ) -> Tuple[EventFrame, Dict[str, Any]]:
        result = self.parse_to_result(sentence)
        return result.frame, result.spec

    @staticmethod
    def _project_legacy_layers_to_nuplan(
        spec: Dict[str, Any],
        hierarchy: HierarchicalScenePath,
    ) -> Dict[str, Any]:
        out = deepcopy(spec)
        projection = hierarchy.nuplan_projection()
        values = hierarchy.values

        semantic_slots = out.setdefault("semantic_slots", {})
        semantic_slots["actor_type"] = projection[
            "actor_category"
        ]
        semantic_slots["nuplan_actor_category"] = projection[
            "actor_category"
        ]
        semantic_slots["language_actor_detail"] = projection[
            "language_actor_detail"
        ]
        semantic_slots["motion_direction"] = values.get(
            "motion_direction", "unknown"
        )

        actor_layer = out.setdefault("actor_layer", {})
        actor_layer["primary_actor"] = projection[
            "actor_category"
        ]
        actor_layer["base_actor_type"] = projection[
            "actor_category"
        ]
        actor_layer["nuplan_tracked_object_type"] = projection[
            "tracked_object_type"
        ]
        actor_layer["sledge_collection"] = projection[
            "sledge_collection"
        ]
        actor_layer["language_actor_detail"] = projection[
            "language_actor_detail"
        ]

        motion_layer = out.setdefault("motion_layer", {})
        motion_layer["motion_direction"] = values.get(
            "motion_direction", "unknown"
        )
        return out

    @staticmethod
    def _refine_event_sequence(
        spec: Dict[str, Any],
        frame: EventFrame,
        hierarchy: HierarchicalScenePath,
    ) -> Dict[str, Any]:
        out = deepcopy(spec)
        if (
            hierarchy.value("hazard_interaction")
            != "occluded_emergence"
        ):
            return out

        actor = hierarchy.nuplan_projection().get(
            "actor_category", "pedestrian"
        )
        auxiliary = hierarchy.value("auxiliary_entity")
        response = hierarchy.value("ego_required_response")
        evidence = frame.sentence
        sequence = [
            {
                "order": 1,
                "actor": "ego",
                "event_type": "ego_driving",
                "action": (
                    frame.ego_event.ego_maneuver
                    or "drive_forward"
                ),
                "relation_to_previous": "start",
                "evidence_text": (
                    frame.ego_event.evidence_text
                ),
            },
            {
                "order": 2,
                "actor": actor,
                "event_type": "actor_occluded",
                "action": (
                    f"hidden_behind_{auxiliary}"
                ),
                "relation_to_previous": (
                    "during_ego_baseline"
                ),
                "evidence_text": (
                    frame.occlusion.evidence_text
                    or evidence
                ),
            },
            {
                "order": 3,
                "actor": actor,
                "event_type": (
                    "occluded_actor_becomes_visible"
                ),
                "action": (
                    "emerge_from_occluder_toward_ego_path"
                ),
                "relation_to_previous": (
                    "after_hidden_state"
                ),
                "evidence_text": (
                    frame.main_event.evidence_text
                    or evidence
                ),
            },
            {
                "order": 4,
                "actor": actor,
                "event_type": "enter_ego_lane",
                "action": (
                    "enter_ego_lane_from_occluder"
                ),
                "relation_to_previous": "after_reveal",
                "evidence_text": (
                    frame.main_event.evidence_text
                    or evidence
                ),
            },
            {
                "order": 5,
                "actor": "ego_and_pedestrian",
                "event_type": "conflict_point_approach",
                "action": "approach_shared_conflict_point",
                "relation_to_previous": "after_lane_entry",
                "evidence_text": (
                    "derived: paths intersect at "
                    "conflict_point_xy"
                ),
            },
            {
                "order": 6,
                "actor": "ego",
                "event_type": "ego_response",
                "action": response,
                "relation_to_previous": (
                    "after_kinematic_evaluation"
                ),
                "evidence_text": (
                    "derived: select brake severity after "
                    "TTC and stopping-distance check"
                ),
            },
        ]
        event_layer = out.setdefault("event_layer", {})
        event_layer["event_sequence"] = sequence
        event_layer["event_sequence_labels"] = [
            (
                f"{step['order']}:{step['actor']}:"
                f"{step['event_type']}:{step['action']}"
            )
            for step in sequence
        ]
        event_layer["num_events"] = len(sequence)
        event_layer["sequence_policy"] = (
            "occlusion_reveal_lane_entry_are_distinct_events"
        )
        return out


def attach_hierarchy(
    spec: Dict[str, Any],
    hierarchy: HierarchicalScenePath,
) -> Dict[str, Any]:
    """Attach a hierarchy without removing legacy spec layers."""

    out = deepcopy(spec)
    out["schema_version"] = hierarchy.schema_version
    out["hierarchy_layer"] = hierarchy.to_dict()
    out.setdefault(
        "validation_layer", {}
    )["composition_policy"] = (
        "nuplan_projected_recursive_parent_constrained_hierarchy"
    )
    return out


def validate_hierarchical_spec(
    spec: Mapping[str, Any],
) -> Tuple[bool, List[str]]:
    """Validate the serialized hierarchy and nuPlan projection."""

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
    actual_order = [
        str(node.get("node_type", ""))
        for node in path
    ]
    if actual_order != required_order:
        issues.append(
            f"hierarchy_order_mismatch:{actual_order}"
        )

    for index, node in enumerate(path, start=1):
        if int(node.get("level", -1)) != index:
            issues.append(
                "hierarchy_level_mismatch:"
                f"{node.get('node_type')}="
                f"{node.get('level')}"
            )
        if node.get("value") in {
            None,
            "",
            "unknown",
        }:
            issues.append(
                "hierarchy_unknown_value:"
                f"{node.get('node_type')}"
            )
        allowed_at_level = set(
            node.get("allowed_values_at_level", []) or []
        )
        if (
            allowed_at_level
            and node.get("value") not in allowed_at_level
        ):
            issues.append(
                "hierarchy_value_not_allowed:"
                f"{node.get('node_type')}="
                f"{node.get('value')}"
            )
        if index < len(path):
            next_value = path[index].get("value")
            allowed_children = set(
                node.get("allowed_children", []) or []
            )
            if (
                allowed_children
                and next_value not in allowed_children
            ):
                issues.append(
                    "hierarchy_child_not_allowed:"
                    f"{node.get('node_type')}="
                    f"{node.get('value')}->{next_value}"
                )

    values = dict(layer.get("path_values", {}) or {})
    if values.get("primary_actor_type") == "pedestrian":
        projection = dict(
            layer.get("nuplan_projection", {}) or {}
        )
        if (
            projection.get("tracked_object_type")
            != "TrackedObjectType.PEDESTRIAN"
        ):
            issues.append(
                "pedestrian_requires_nuplan_pedestrian_projection"
            )
        if projection.get("sledge_collection") != "pedestrians":
            issues.append(
                "pedestrian_requires_sledge_pedestrians_collection"
            )
    if (
        values.get("hazard_interaction")
        == "occluded_emergence"
    ):
        if (
            values.get("motion_direction")
            != "occluder_to_ego_path"
        ):
            issues.append(
                "occluded_emergence_direction_must_be_"
                "occluder_to_ego_path"
            )

    return not issues, issues


DefaultLanguageUnderstandingPipeline = (
    HierarchicalEventFramePipeline
)
