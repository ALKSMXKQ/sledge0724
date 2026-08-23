"""Validation for hierarchical occluded-pedestrian parameter templates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
)


SEMANTIC_DIRECTION = (
    "occluder_to_ego_path"
)

REQUIRED_EVENT_ORDER = (
    "actor_occluded",
    "occluded_actor_becomes_visible",
    "enter_ego_lane",
)

REQUIRED_TEMPLATE_PARAMETERS = {
    "lane_width_m",
    "lane_count",
    "road_curvature",
    "conflict_point_xy",
    "ego_speed_mps",
    "ego_acceleration_mps2",
    "ego_distance_to_conflict_m",
    "actor_speed_mps",
    "actor_acceleration_mps2",
    "actor_initial_position",
    "actor_heading",
    "actor_start_time_s",
    "occluder_position",
    "occluder_length_m",
    "occluder_width_m",
    "occluder_lateral_offset_m",
    "reveal_distance_m",
    "initial_gap_m",
    "time_to_collision_s",
    "minimum_clearance_m",
    "braking_deceleration_mps2",
}


@dataclass
class HierarchicalTemplateValidation:
    passed: bool
    issues: List[str] = field(
        default_factory=list
    )
    warnings: List[str] = field(
        default_factory=list
    )
    observed: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return asdict(self)


class HierarchicalOccludedTemplateValidator:
    """Validate the semantic template before concrete scene sampling."""

    def validate(
        self,
        payload: Mapping[str, Any],
        expected: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> HierarchicalTemplateValidation:
        spec = self._spec(payload)

        hierarchy = dict(
            spec.get(
                "hierarchy_layer",
                {},
            )
            or {}
        )

        values = dict(
            hierarchy.get(
                "path_values",
                {},
            )
            or {}
        )

        attributes = dict(
            hierarchy.get(
                "attributes",
                {},
            )
            or {}
        )

        projection = dict(
            hierarchy.get(
                "nuplan_projection",
                {},
            )
            or {}
        )

        params = dict(
            spec.get(
                "parameter_layer",
                {},
            )
            or {}
        )

        completed = dict(
            params.get(
                "completed",
                {},
            )
            or {}
        )

        readiness = dict(
            spec.get(
                "readiness",
                {},
            )
            or {}
        )

        event_layer = dict(
            spec.get(
                "event_layer",
                {},
            )
            or {}
        )

        events = list(
            event_layer.get(
                "event_sequence",
                [],
            )
            or []
        )

        issues: List[str] = []
        warnings: List[str] = []

        # --------------------------------------------------------------
        # Core semantic contract
        # --------------------------------------------------------------
        required_values = {
            "primary_actor_type":
                "pedestrian",
            "hazard_interaction":
                "occluded_emergence",
            "motion_direction":
                SEMANTIC_DIRECTION,
        }

        for (
            key,
            wanted,
        ) in required_values.items():
            actual = values.get(key)

            if key == "motion_direction":
                if not (
                    self
                    ._semantic_direction_matches(
                        actual
                    )
                ):
                    issues.append(
                        f"{key}={actual!r}, "
                        "expected only "
                        f"{wanted!r}"
                    )

            elif actual != wanted:
                issues.append(
                    f"{key}={actual!r}, "
                    f"expected {wanted!r}"
                )

        auxiliary = str(
            values.get(
                "auxiliary_entity",
                "none",
            )
        )

        if (
            auxiliary == "none"
            or not auxiliary.endswith(
                "_occluder"
            )
        ):
            issues.append(
                "occluded pedestrian template "
                "must contain an auxiliary "
                "occluder"
            )

        # --------------------------------------------------------------
        # nuPlan/SLEDGE projection
        # --------------------------------------------------------------
        if (
            projection.get(
                "tracked_object_type"
            )
            != (
                "TrackedObjectType."
                "PEDESTRIAN"
            )
        ):
            issues.append(
                "nuPlan projection must use "
                "TrackedObjectType.PEDESTRIAN"
            )

        if (
            projection.get(
                "sledge_collection"
            )
            != "pedestrians"
        ):
            issues.append(
                "SLEDGE projection must use "
                "pedestrians"
            )

        if not bool(
            projection.get(
                "compatible",
                False,
            )
        ):
            issues.append(
                "nuPlan/SLEDGE projection "
                "is not compatible"
            )

        # --------------------------------------------------------------
        # Semantic direction
        # --------------------------------------------------------------
        direction_entry = (
            completed.get(
                "crossing_direction",
                {},
            )
        )

        direction_value = (
            self._value(
                direction_entry
            )
        )

        alternatives: List[Any] = []

        if isinstance(
            direction_entry,
            Mapping,
        ):
            alternatives = (
                self._as_list(
                    direction_entry.get(
                        "alternatives",
                        [],
                    )
                )
            )

        direction_values = (
            self._extract_semantic_values(
                direction_value
            )
        )

        invalid_direction_values = [
            value
            for value
            in direction_values
            if value
            != SEMANTIC_DIRECTION
        ]

        if invalid_direction_values:
            issues.append(
                "crossing_direction must "
                "contain only "
                f"{SEMANTIC_DIRECTION!r}, "
                "observed "
                f"{direction_value!r}"
            )

        forbidden = {
            "left_to_right",
            "right_to_left",
        }

        alternative_values: List[
            str
        ] = []

        for value in alternatives:
            alternative_values.extend(
                self._extract_semantic_values(
                    value
                )
            )

        if any(
            value in forbidden
            for value
            in alternative_values
        ):
            issues.append(
                "semantic template still "
                "exposes absolute crossing "
                "direction alternatives"
            )

        if any(
            value in forbidden
            for value
            in direction_values
        ):
            issues.append(
                "crossing_direction contains "
                "an absolute execution "
                "direction"
            )

        # --------------------------------------------------------------
        # Event ordering
        # --------------------------------------------------------------
        event_types = [
            str(
                item.get(
                    "event_type",
                    "",
                )
            )
            for item
            in events
            if isinstance(
                item,
                Mapping,
            )
        ]

        positions: List[int] = []

        for event_type in (
            REQUIRED_EVENT_ORDER
        ):
            if (
                event_type
                not in event_types
            ):
                issues.append(
                    "missing event: "
                    f"{event_type}"
                )
            else:
                positions.append(
                    event_types.index(
                        event_type
                    )
                )

        if (
            len(positions)
            == len(
                REQUIRED_EVENT_ORDER
            )
            and positions
            != sorted(positions)
        ):
            issues.append(
                "occlusion/reveal/lane-entry "
                "events are out of order"
            )

        # --------------------------------------------------------------
        # Parameter completeness
        # --------------------------------------------------------------
        missing = sorted(
            REQUIRED_TEMPLATE_PARAMETERS
            - set(completed)
        )

        declared_missing = list(
            params.get(
                "template_missing_parameters",
                [],
            )
            or []
        )

        unresolved = list(
            params.get(
                "unresolved_required",
                [],
            )
            or []
        )

        if missing:
            issues.append(
                "missing completed parameters: "
                f"{missing}"
            )

        if declared_missing:
            issues.append(
                "template_missing_parameters="
                f"{declared_missing}"
            )

        if unresolved:
            issues.append(
                "unresolved_required="
                f"{unresolved}"
            )

        if not bool(
            params.get(
                "parameter_template_complete",
                False,
            )
        ):
            issues.append(
                "parameter_template_complete "
                "is false"
            )

        # --------------------------------------------------------------
        # Readiness
        # --------------------------------------------------------------
        if (
            readiness.get(
                "scene_template_ready"
            )
            is not True
        ):
            issues.append(
                "scene_template_ready "
                "is not true"
            )

        contract_status = (
            readiness.get(
                "occluded_pedestrian_contract"
            )
        )

        if (
            contract_status
            is not None
            and contract_status
            != "passed"
        ):
            issues.append(
                "occluded_pedestrian_contract="
                f"{contract_status!r}"
            )

        contract_issues = list(
            readiness.get(
                "occluded_pedestrian_contract_issues",
                [],
            )
            or []
        )

        if contract_issues:
            issues.append(
                "occluded_pedestrian_contract_issues="
                f"{contract_issues}"
            )

        if (
            readiness.get(
                "sampled_scene_ready"
            )
            is not False
        ):
            warnings.append(
                "template output should "
                "normally have "
                "sampled_scene_ready=false"
            )

        if (
            readiness.get(
                "kinematic_consistency"
            )
            != (
                "pending_concrete_sampling"
            )
        ):
            warnings.append(
                "kinematic_consistency "
                "should remain "
                "pending_concrete_sampling "
                "before B1 sampling"
            )

        expected = dict(
            expected or {}
        )

        observed = {
            "path_values":
                values,
            "nuplan_projection":
                projection,
            "language_actor_detail":
                attributes.get(
                    "language_actor_detail"
                ),
            "event_types":
                event_types,
            "parameter_count":
                len(completed),
            "crossing_direction":
                direction_value,
            "crossing_direction_values":
                direction_values,
            "crossing_direction_alternatives":
                alternatives,
            "readiness":
                readiness,
        }

        self._validate_expected(
            expected=expected,
            values=values,
            projection=projection,
            completed=completed,
            hierarchy=hierarchy,
            issues=issues,
        )

        return (
            HierarchicalTemplateValidation(
                passed=not issues,
                issues=issues,
                warnings=warnings,
                observed=observed,
            )
        )

    @staticmethod
    def _spec(
        payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if (
            "spec" in payload
            and isinstance(
                payload["spec"],
                Mapping,
            )
        ):
            return dict(
                payload["spec"]
            )

        return dict(payload)

    @staticmethod
    def _value(
        entry: Any,
    ) -> Any:
        if isinstance(
            entry,
            Mapping,
        ):
            return entry.get(
                "value"
            )

        return entry

    @staticmethod
    def _as_list(
        value: Any,
    ) -> List[Any]:
        if value is None:
            return []

        if (
            isinstance(
                value,
                Sequence,
            )
            and not isinstance(
                value,
                (
                    str,
                    bytes,
                ),
            )
        ):
            return list(value)

        return [value]

    @classmethod
    def _extract_semantic_values(
        cls,
        value: Any,
    ) -> List[str]:
        if value is None:
            return []

        if isinstance(
            value,
            Mapping,
        ):
            if "values" in value:
                return (
                    cls
                    ._extract_semantic_values(
                        value.get(
                            "values"
                        )
                    )
                )

            if "value" in value:
                return (
                    cls
                    ._extract_semantic_values(
                        value.get(
                            "value"
                        )
                    )
                )

            return []

        if (
            isinstance(
                value,
                Sequence,
            )
            and not isinstance(
                value,
                (
                    str,
                    bytes,
                ),
            )
        ):
            normalized: List[
                str
            ] = []

            for item in value:
                normalized.extend(
                    cls
                    ._extract_semantic_values(
                        item
                    )
                )

            return normalized

        return [str(value)]

    @classmethod
    def _semantic_direction_matches(
        cls,
        value: Any,
    ) -> bool:
        values = (
            cls
            ._extract_semantic_values(
                value
            )
        )

        return (
            len(values) == 1
            and values[0]
            == SEMANTIC_DIRECTION
        )

    def _validate_expected(
        self,
        *,
        expected: Mapping[str, Any],
        values: Mapping[str, Any],
        projection: Mapping[str, Any],
        completed: Mapping[
            str,
            Any,
        ],
        hierarchy: Mapping[str, Any],
        issues: List[str],
    ) -> None:
        for (
            key,
            wanted,
        ) in expected.items():
            actual: Any

            if key in values:
                actual = values.get(key)

                if (
                    key
                    == "motion_direction"
                    and wanted
                    == SEMANTIC_DIRECTION
                ):
                    normalized = (
                        self
                        ._extract_semantic_values(
                            actual
                        )
                    )

                    if (
                        len(normalized)
                        == 1
                        and normalized[0]
                        == SEMANTIC_DIRECTION
                    ):
                        continue

            elif key in projection:
                actual = projection.get(
                    key
                )

            elif (
                key
                == "language_actor_detail"
            ):
                actual = (
                    hierarchy.get(
                        "attributes",
                        {},
                    )
                    .get(
                        "language_actor_detail"
                    )
                )

                if (
                    wanted
                    == "adult_or_unspecified"
                    and actual
                    in {
                        "adult",
                        "unspecified",
                        "pedestrian",
                    }
                ):
                    continue

            # ----------------------------------------------------------
            # Explicit pedestrian speed:
            # value must be exact AND provenance must be user_input.
            # ----------------------------------------------------------
            elif (
                key
                == "actor_speed_mps"
            ):
                entry = completed.get(
                    "actor_speed_mps"
                )

                actual = self._value(
                    entry
                )

                source = (
                    str(
                        entry.get(
                            "source",
                            "",
                        )
                    )
                    if isinstance(
                        entry,
                        Mapping,
                    )
                    else ""
                )

                try:
                    exact_match = (
                        abs(
                            float(actual)
                            - float(wanted)
                        )
                        <= 1e-6
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    exact_match = False

                if not exact_match:
                    issues.append(
                        "expected explicit "
                        "actor_speed_mps="
                        f"{wanted!r}, observed "
                        f"{actual!r}"
                    )

                if (
                    source
                    != "user_input"
                ):
                    issues.append(
                        "explicit "
                        "actor_speed_mps must "
                        "have source='user_input', "
                        "observed source="
                        f"{source!r}"
                    )

                continue

            elif key == "occluder_side":
                actual = self._value(
                    completed.get(
                        "occluder_side"
                    )
                )

                if isinstance(
                    actual,
                    Mapping,
                ):
                    allowed = [
                        str(item)
                        for item
                        in (
                            actual.get(
                                "values",
                                [],
                            )
                            or []
                        )
                    ]

                    if (
                        str(wanted)
                        in allowed
                    ):
                        continue

                elif (
                    isinstance(
                        actual,
                        Sequence,
                    )
                    and not isinstance(
                        actual,
                        (
                            str,
                            bytes,
                        ),
                    )
                ):
                    if (
                        str(wanted)
                        in [
                            str(item)
                            for item
                            in actual
                        ]
                    ):
                        continue

            elif (
                key
                == "occluder_side_mode"
            ):
                side = self._value(
                    completed.get(
                        "occluder_side"
                    )
                )

                if isinstance(
                    side,
                    Mapping,
                ):
                    if bool(
                        side.get(
                            "sample_once",
                            False,
                        )
                    ):
                        actual = (
                            "sample_once"
                        )
                    else:
                        actual = side
                else:
                    actual = side

            else:
                continue

            if actual != wanted:
                issues.append(
                    f"expected {key}="
                    f"{wanted!r}, "
                    "observed "
                    f"{actual!r}"
                )