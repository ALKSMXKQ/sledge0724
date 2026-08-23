"""Research-oriented natural-language resolution for occluded-pedestrian scenes.

This module deliberately separates language understanding from executable scene
editing.  Natural language fills a structured control template; researchers can
inspect, edit, lock, and persist that template before generation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)


SUPPORTED_OCCLUDERS: Set[str] = {
    "vehicle",
    "bicycle",
    "generic_object",
    "traffic_cone",
    "barrier",
    "czone_sign",
}

SUPPORTED_DIRECTIONS: Set[str] = {
    "left_to_right",
    "right_to_left",
}

SUPPORTED_RISK_LEVELS: Set[str] = {
    "mild",
    "moderate",
    "aggressive",
}

SUPPORTED_CONTROL_FIELDS: Tuple[str, ...] = (
    "occluder_type",
    "direction",
    "pedestrian_speed_mps",
    "risk_level",
)

DEFAULT_CONTROLS: Dict[str, Any] = {
    "occluder_type": "vehicle",
    "direction": "right_to_left",
    "pedestrian_speed_mps": 1.6,
    "risk_level": "moderate",
}


# Values are canonical nuPlan-visible categories.
# Longer and more specific terms must appear before short generic aliases.
OCCLUDER_ALIASES: Dict[
    str,
    Tuple[str, ...],
] = {
    "traffic_cone": (
        "traffic cone",
        "road cone",
        "safety cone",
        "交通锥",
        "交通锥桶",
        "锥桶",
        "路锥",
    ),
    "czone_sign": (
        "construction-zone sign",
        "construction zone sign",
        "construction sign",
        "roadwork sign",
        "施工区域标志",
        "施工标志",
        "施工牌",
    ),
    "generic_object": (
        "generic roadside object",
        "generic obstacle",
        "generic object",
        "roadside object",
        "通用道路物体",
        "通用障碍物",
        "通用物体",
    ),
    "barrier": (
    "concrete traffic barrier",
    "traffic barrier",
    "road barrier",
    "construction barrier",
    "guardrail",
    "barrier",
    "隔离护栏",
    "道路围挡",
    "施工围挡",
    "隔离栏",
    "护栏",
    "路障",
    ),
    "bicycle": (
        "parked bicycle",
        "stationary bicycle",
        "bicycle",
        "bike",
        "停放的自行车",
        "自行车",
        "单车",
    ),
    "vehicle": (
        "parked delivery van",
        "stationary delivery van",
        "parked vehicle",
        "stationary vehicle",
        "parked car",
        "delivery van",
        "vehicle",
        "minivan",
        "van",
        "truck",
        "lorry",
        "bus",
        "coach",
        "car",
        "停放的厢式货车",
        "停放的货车",
        "停放的公交车",
        "停放车辆",
        "停靠车辆",
        "厢式货车",
        "面包车",
        "公交车",
        "大巴车",
        "货车",
        "卡车",
        "轿车",
        "车辆",
    ),
}


NEGATED_OCCLUSION_PATTERNS: Tuple[
    str,
    ...,
] = (
    (
        r"\b(?:not|never|without)\s+"
        r"(?:being\s+)?"
        r"(?:hidden|occluded|blocked)\b"
    ),
    (
        r"\b(?:no|without)\s+"
        r"(?:visual\s+)?occlusion\b"
    ),
    (
        r"行人.{0,12}(?:没有|未被|不是)"
        r"(?:车辆|物体|障碍物)?.{0,8}"
        r"(?:遮挡|挡住)"
    ),
    (
        r"(?:无遮挡|没有遮挡|"
        r"无视线遮挡|视线未受阻)"
    ),
)


PEDESTRIAN_TERMS: Tuple[str, ...] = (
    "pedestrian",
    "person",
    "walker",
    "行人",
    "路人",
)

CROSSING_TERMS: Tuple[str, ...] = (
    "cross",
    "crossing",
    "rush",
    "emerge",
    "enter the ego path",
    "横穿",
    "冲出",
    "冲入",
    "穿过",
    "进入车道",
)

OCCLUSION_TERMS: Tuple[str, ...] = (
    "hidden behind",
    "hide a pedestrian",
    "hides a pedestrian",
    "comes out from behind",
    "emerges from behind",
    "blocks ego's view",
    "blocks the view",
    "occluded",
    "occlusion",
    "遮挡",
    "挡住视线",
    "视线受阻",
    "从后面冲出",
)


QUALITATIVE_SPEEDS: Tuple[
    Tuple[
        str,
        Tuple[str, ...],
        Tuple[float, float],
        float,
    ],
    ...,
] = (
    (
        "slow",
        (
            "slowly",
            "slow walking",
            "walking slowly",
            "缓慢",
            "慢速",
            "慢慢",
        ),
        (0.5, 1.0),
        0.8,
    ),
    (
        "fast",
        (
            "very fast",
            "fast walking",
            "running speed",
            "rushes",
            "sprints",
            "快速",
            "高速冲出",
            "飞快",
            "奔跑",
        ),
        (1.6, 2.0),
        1.9,
    ),
    (
        "normal",
        (
            "normal walking speed",
            "pedestrian speed",
            "walking speed",
            "正常步行",
            "正常速度",
            "步行速度",
        ),
        (1.0, 1.6),
        1.4,
    ),
)


@dataclass(frozen=True)
class FieldEvidence:
    """One parser's evidence for a control field."""

    value: Any
    source: str
    parser: str
    confidence: Optional[float] = None
    original_text: Optional[str] = None
    normalized_from: Optional[str] = None
    value_range: Optional[
        Tuple[float, float]
    ] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)

        if self.value_range is not None:
            payload["value_range"] = list(
                self.value_range
            )

        return payload


@dataclass(frozen=True)
class ResolutionIssue:
    """A semantic ambiguity, contradiction, unsupported input, or warning."""

    code: str
    severity: str
    message: str
    fields: Tuple[str, ...] = ()
    details: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["fields"] = list(
            self.fields
        )
        return payload


@dataclass(frozen=True)
class ResolvedField:
    """Final executable value plus traceability metadata."""

    value: Any
    source: str
    confidence: Optional[float]
    locked: bool
    adjustable: bool
    original_text: Optional[str] = None
    normalized_from: Optional[str] = None
    value_range: Optional[
        Tuple[float, float]
    ] = None
    alternatives: Tuple[
        Dict[str, Any],
        ...,
    ] = ()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)

        if self.value_range is not None:
            payload["value_range"] = list(
                self.value_range
            )

        payload["alternatives"] = list(
            self.alternatives
        )

        return payload


@dataclass
class RuleParseResult:
    """Deterministic domain parser output used beside EventFrame parsing."""

    fields: Dict[
        str,
        FieldEvidence,
    ] = field(
        default_factory=dict
    )

    issues: List[
        ResolutionIssue
    ] = field(
        default_factory=list
    )

    emergence_side: Optional[str] = None
    reference_frame: str = "ego"

    excluded_occluders: Set[str] = field(
        default_factory=set
    )

    target_family_detected: bool = False
    negated_occlusion: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fields": {
                key: value.to_dict()
                for key, value in self.fields.items()
            },
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
            "emergence_side": (
                self.emergence_side
            ),
            "reference_frame": (
                self.reference_frame
            ),
            "excluded_occluders": sorted(
                self.excluded_occluders
            ),
            "target_family_detected": (
                self.target_family_detected
            ),
            "negated_occlusion": (
                self.negated_occlusion
            ),
        }


@dataclass
class ResolutionResult:
    """Complete editable parameter template produced before scene generation."""

    prompt: str

    fields: Dict[
        str,
        ResolvedField,
    ]

    issues: List[
        ResolutionIssue
    ]

    rule_parse: RuleParseResult

    eventframe_candidates: Dict[
        str,
        FieldEvidence,
    ]

    reference_frame: str = "ego"

    scenario_family: str = (
        "occluded_pedestrian"
    )

    semantic_valid: bool = True
    requires_confirmation: bool = False

    defaults_used: Tuple[str, ...] = ()
    locked_fields: Tuple[str, ...] = ()
    adjustable_fields: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": (
                "occluded_pedestrian_resolution_v2"
            ),
            "prompt": self.prompt,
            "scenario_family": (
                self.scenario_family
            ),
            "reference_frame": (
                self.reference_frame
            ),
            "fields": {
                key: value.to_dict()
                for key, value in self.fields.items()
            },
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
            "semantic_valid": (
                self.semantic_valid
            ),
            "requires_confirmation": (
                self.requires_confirmation
            ),
            "defaults_used": list(
                self.defaults_used
            ),
            "locked_fields": list(
                self.locked_fields
            ),
            "adjustable_fields": list(
                self.adjustable_fields
            ),
            "rule_parse": (
                self.rule_parse.to_dict()
            ),
            "eventframe_candidates": {
                key: value.to_dict()
                for key, value
                in self.eventframe_candidates.items()
            },
        }

    def values(self) -> Dict[str, Any]:
        return {
            key: item.value
            for key, item in self.fields.items()
        }


class RuleBasedPromptParser:
    """Deterministic bilingual parser for the four executable control fields."""

    def parse(
        self,
        prompt: str,
    ) -> RuleParseResult:
        text = str(
            prompt or ""
        ).strip()

        lower = text.lower()

        result = RuleParseResult()

        has_pedestrian = _contains_any(
            lower,
            PEDESTRIAN_TERMS,
        )

        has_crossing = _contains_any(
            lower,
            CROSSING_TERMS,
        )

        has_occlusion = (
            _contains_any(
                lower,
                OCCLUSION_TERMS,
            )
            or _mentions_behind_occluder(
                text
            )
        )

        result.negated_occlusion = any(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            for pattern
            in NEGATED_OCCLUSION_PATTERNS
        )

        result.target_family_detected = bool(
            has_pedestrian
            and has_crossing
            and has_occlusion
        )

        if result.negated_occlusion:
            result.issues.append(
                ResolutionIssue(
                    code="negated_occlusion",
                    severity="error",
                    message=(
                        "The prompt explicitly states "
                        "that the pedestrian is not "
                        "occluded."
                    ),
                    fields=(
                        "scenario_family",
                    ),
                )
            )

        elif not result.target_family_detected:
            result.issues.append(
                ResolutionIssue(
                    code=(
                        "target_family_not_explicit"
                    ),
                    severity="warning",
                    message=(
                        "The deterministic parser did "
                        "not find explicit pedestrian, "
                        "crossing, and occlusion evidence "
                        "together. EventFrame evidence "
                        "must confirm the family."
                    ),
                    fields=(
                        "scenario_family",
                    ),
                )
            )

        result.excluded_occluders = (
            self._extract_excluded_occluders(
                text
            )
        )

        occluder = self._extract_occluder(
            text,
            result.excluded_occluders,
        )

        if occluder is not None:
            result.fields[
                "occluder_type"
            ] = occluder

        explicit_direction = (
            self._extract_explicit_direction(
                text
            )
        )

        emergence_side = (
            self._extract_emergence_side(
                text
            )
        )

        result.emergence_side = (
            emergence_side
        )

        derived_direction: Optional[
            FieldEvidence
        ] = None

        if emergence_side == "left":
            derived_direction = FieldEvidence(
                value="left_to_right",
                source="prompt_evidence",
                parser="rule",
                confidence=0.93,
                original_text=(
                    "emergence_side=left"
                ),
                normalized_from=(
                    "occluder-side relation"
                ),
            )

        elif emergence_side == "right":
            derived_direction = FieldEvidence(
                value="right_to_left",
                source="prompt_evidence",
                parser="rule",
                confidence=0.93,
                original_text=(
                    "emergence_side=right"
                ),
                normalized_from=(
                    "occluder-side relation"
                ),
            )

        if (
            explicit_direction is not None
            and derived_direction is not None
        ):
            if (
                explicit_direction.value
                != derived_direction.value
            ):
                result.issues.append(
                    ResolutionIssue(
                        code=(
                            "emergence_direction_conflict"
                        ),
                        severity="error",
                        message=(
                            "The pedestrian emergence "
                            "side contradicts the stated "
                            "crossing direction. Under the "
                            "ego reference frame, left-side "
                            "emergence requires "
                            "left_to_right, and right-side "
                            "emergence requires "
                            "right_to_left."
                        ),
                        fields=(
                            "direction",
                            "emergence_side",
                        ),
                        details={
                            "emergence_side": (
                                emergence_side
                            ),
                            "derived_direction": (
                                derived_direction.value
                            ),
                            "stated_direction": (
                                explicit_direction.value
                            ),
                        },
                    )
                )

            result.fields[
                "direction"
            ] = explicit_direction

        elif explicit_direction is not None:
            result.fields[
                "direction"
            ] = explicit_direction

        elif derived_direction is not None:
            result.fields[
                "direction"
            ] = derived_direction

        speed = self._extract_speed(
            text
        )

        if speed is not None:
            result.fields[
                "pedestrian_speed_mps"
            ] = speed

        risk = self._extract_risk(
            text
        )

        if risk is not None:
            result.fields[
                "risk_level"
            ] = risk

        if (
            result.excluded_occluders
            and "occluder_type"
            not in result.fields
        ):
            result.issues.append(
                ResolutionIssue(
                    code=(
                        "occluder_excluded_without_replacement"
                    ),
                    severity="warning",
                    message=(
                        "The prompt excludes one or more "
                        "occluder types but does not "
                        "clearly name a supported "
                        "replacement."
                    ),
                    fields=(
                        "occluder_type",
                    ),
                    details={
                        "excluded": sorted(
                            result.excluded_occluders
                        )
                    },
                )
            )

        return result

    @staticmethod
    def _extract_excluded_occluders(
        text: str,
    ) -> Set[str]:
        lower = text.lower()
        excluded: Set[str] = set()

        for (
            canonical,
            aliases,
        ) in OCCLUDER_ALIASES.items():
            for alias in aliases:
                escaped = re.escape(
                    alias.lower()
                )

                english = (
                    rf"(?:do\s+not|don't|without|"
                    rf"exclude|not)\s+"
                    rf"(?:use\s+)?"
                    rf"(?:a\s+|an\s+|the\s+)?"
                    rf"{escaped}"
                )

                chinese = (
                    rf"(?:不要|不使用|排除|"
                    rf"禁用|不能用).{{0,8}}"
                    rf"{re.escape(alias)}"
                )

                if (
                    re.search(
                        english,
                        lower,
                    )
                    or re.search(
                        chinese,
                        text,
                    )
                ):
                    excluded.add(
                        canonical
                    )
                    break

        return excluded

    @staticmethod
    def _extract_occluder(
        text: str,
        excluded: Set[str],
    ) -> Optional[FieldEvidence]:
        lower = text.lower()

        candidates: List[
            Tuple[
                int,
                int,
                str,
                str,
            ]
        ] = []

        for (
            canonical,
            aliases,
        ) in OCCLUDER_ALIASES.items():
            if canonical in excluded:
                continue

            for alias in aliases:
                index = lower.find(
                    alias.lower()
                )

                if index >= 0:
                    candidates.append(
                        (
                            index,
                            -len(alias),
                            canonical,
                            alias,
                        )
                    )

        if not candidates:
            return None

        (
            _,
            _,
            canonical,
            alias,
        ) = sorted(candidates)[0]

        return FieldEvidence(
            value=canonical,
            source="prompt_evidence",
            parser="rule",
            confidence=0.98,
            original_text=alias,
            normalized_from=alias,
        )

    @staticmethod
    def _extract_explicit_direction(
        text: str,
    ) -> Optional[FieldEvidence]:
        lower = text.lower().replace(
            "-",
            " ",
        )

        rtl_patterns = (
            (
                r"(?:from\s+)?right"
                r"(?:\s+side)?\s+to\s+left"
            ),
            (
                r"right\s*(?:→|->)\s*left"
            ),
            (
                r"由?右(?:侧|边)?"
                r"(?:向|到|往)左"
            ),
            (
                r"从右(?:侧|边)?"
                r"(?:向|到|往)左"
            ),
            r"右向左",
        )

        ltr_patterns = (
            (
                r"(?:from\s+)?left"
                r"(?:\s+side)?\s+to\s+right"
            ),
            (
                r"left\s*(?:→|->)\s*right"
            ),
            (
                r"由?左(?:侧|边)?"
                r"(?:向|到|往)右"
            ),
            (
                r"从左(?:侧|边)?"
                r"(?:向|到|往)右"
            ),
            r"左向右",
        )

        rtl = [
            pattern
            for pattern in rtl_patterns
            if re.search(
                pattern,
                (
                    lower
                    if "right" in pattern
                    else text
                ),
            )
        ]

        ltr = [
            pattern
            for pattern in ltr_patterns
            if re.search(
                pattern,
                (
                    lower
                    if "left" in pattern
                    else text
                ),
            )
        ]

        if rtl and ltr:
            return FieldEvidence(
                value="ambiguous",
                source="prompt_evidence",
                parser="rule",
                confidence=0.0,
                original_text=text,
            )

        if rtl:
            return FieldEvidence(
                value="right_to_left",
                source="prompt_evidence",
                parser="rule",
                confidence=0.99,
                original_text=(
                    _first_matching_fragment(
                        text,
                        rtl_patterns,
                    )
                ),
            )

        if ltr:
            return FieldEvidence(
                value="left_to_right",
                source="prompt_evidence",
                parser="rule",
                confidence=0.99,
                original_text=(
                    _first_matching_fragment(
                        text,
                        ltr_patterns,
                    )
                ),
            )

        return None

    @staticmethod
    def _extract_emergence_side(
        text: str,
    ) -> Optional[str]:
        lower = text.lower().replace(
            "-",
            " ",
        )

        left_patterns = (
            (
                r"(?:from|at)\s+the\s+"
                r"(?:vehicle|car|truck|bus|"
                r"van|occluder)'?s?\s+"
                r"left\s+side"
            ),
            (
                r"(?:from|at)\s+the\s+"
                r"left\s+side\s+of\s+"
                r"(?:the\s+)?"
                r"(?:vehicle|car|truck|bus|"
                r"van|occluder)"
            ),
            (
                r"从(?:车辆|汽车|货车|卡车|"
                r"公交车|大巴车|遮挡物)"
                r"左(?:侧|边)"
                r"(?:冲出|出现|露出|进入)"
            ),
            (
                r"在(?:车辆|汽车|货车|卡车|"
                r"公交车|大巴车|遮挡物)"
                r"左(?:侧|边)"
                r"(?:冲出|出现|露出)"
            ),
        )

        right_patterns = (
            (
                r"(?:from|at)\s+the\s+"
                r"(?:vehicle|car|truck|bus|"
                r"van|occluder)'?s?\s+"
                r"right\s+side"
            ),
            (
                r"(?:from|at)\s+the\s+"
                r"right\s+side\s+of\s+"
                r"(?:the\s+)?"
                r"(?:vehicle|car|truck|bus|"
                r"van|occluder)"
            ),
            (
                r"从(?:车辆|汽车|货车|卡车|"
                r"公交车|大巴车|遮挡物)"
                r"右(?:侧|边)"
                r"(?:冲出|出现|露出|进入)"
            ),
            (
                r"在(?:车辆|汽车|货车|卡车|"
                r"公交车|大巴车|遮挡物)"
                r"右(?:侧|边)"
                r"(?:冲出|出现|露出)"
            ),
        )

        has_left = any(
            re.search(
                pattern,
                (
                    lower
                    if pattern.startswith(
                        "(?:from"
                    )
                    else text
                ),
            )
            for pattern in left_patterns
        )

        has_right = any(
            re.search(
                pattern,
                (
                    lower
                    if pattern.startswith(
                        "(?:from"
                    )
                    else text
                ),
            )
            for pattern in right_patterns
        )

        if has_left and has_right:
            return "ambiguous"

        if has_left:
            return "left"

        if has_right:
            return "right"

        return None

    @staticmethod
    def _extract_speed(
        text: str,
    ) -> Optional[FieldEvidence]:
        mps_patterns = (
            (
                r"(?P<value>\d+(?:\.\d+)?)"
                r"\s*(?:m\s*/\s*s|mps|"
                r"m/s|米每秒)"
            ),
            (
                r"(?:速度|speed)\s*"
                r"(?:is|为|=|:|of)?\s*"
                r"(?P<value>\d+(?:\.\d+)?)"
                r"\s*(?:m\s*/?\s*s|"
                r"米每秒)?"
            ),
        )

        for pattern in mps_patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                value = float(
                    match.group("value")
                )

                return FieldEvidence(
                    value=value,
                    source="prompt_evidence",
                    parser="rule",
                    confidence=0.99,
                    original_text=(
                        match.group(0)
                    ),
                    normalized_from="m/s",
                )

        kmh_patterns = (
            (
                r"(?P<value>\d+(?:\.\d+)?)"
                r"\s*(?:km\s*/\s*h|kmph|"
                r"kph|公里每小时|千米每小时)"
            ),
            (
                r"(?:速度|speed)\s*"
                r"(?:is|为|=|:|of)?\s*"
                r"(?P<value>\d+(?:\.\d+)?)"
                r"\s*(?:km\s*/?\s*h|"
                r"公里每小时|千米每小时)"
            ),
        )

        for pattern in kmh_patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                raw = float(
                    match.group("value")
                )

                return FieldEvidence(
                    value=raw / 3.6,
                    source="prompt_evidence",
                    parser="rule",
                    confidence=0.99,
                    original_text=(
                        match.group(0)
                    ),
                    normalized_from=(
                        f"{raw:g} km/h"
                    ),
                )

        lower = text.lower()

        for (
            label,
            terms,
            value_range,
            representative,
        ) in QUALITATIVE_SPEEDS:
            if any(
                term in lower
                for term in terms
            ):
                return FieldEvidence(
                    value=representative,
                    source=(
                        "semantic_range_completion"
                    ),
                    parser="rule",
                    confidence=0.75,
                    original_text=label,
                    normalized_from=(
                        "qualitative_speed"
                    ),
                    value_range=value_range,
                )

        return None

    @staticmethod
    def _extract_risk(
        text: str,
    ) -> Optional[FieldEvidence]:
        lower = text.lower()

        mapping = (
            (
                "aggressive",
                (
                    "aggressive",
                    "high risk",
                    "high-risk",
                    "critical",
                    "dangerous",
                    "高危",
                    "高风险",
                    "激进",
                    "极危险",
                ),
                0.92,
            ),
            (
                "mild",
                (
                    "mild",
                    "low risk",
                    "low-risk",
                    "gentle",
                    "轻度",
                    "低风险",
                    "温和",
                ),
                0.92,
            ),
            (
                "moderate",
                (
                    "moderate",
                    "medium risk",
                    "medium-risk",
                    "中度",
                    "中风险",
                    "适中",
                ),
                0.92,
            ),
        )

        for (
            value,
            terms,
            confidence,
        ) in mapping:
            matched = next(
                (
                    term
                    for term in terms
                    if term in lower
                ),
                None,
            )

            if matched:
                return FieldEvidence(
                    value=value,
                    source="prompt_evidence",
                    parser="rule",
                    confidence=confidence,
                    original_text=matched,
                )

        return None


def extract_eventframe_candidates(
    mapped: Mapping[str, Any],
) -> Dict[str, FieldEvidence]:
    """Convert mapped EventFrame slots into comparable control candidates."""

    candidates: Dict[
        str,
        FieldEvidence,
    ] = {}

    object_layer = _dict(
        mapped.get(
            "object_layer"
        )
    )

    occlusion = _dict(
        object_layer.get(
            "occlusion"
        )
    )

    raw_occluder = occlusion.get(
        "occluder_type"
    )

    normalized_occluder = (
        normalize_occluder(
            raw_occluder
        )
    )

    if normalized_occluder is not None:
        candidates[
            "occluder_type"
        ] = FieldEvidence(
            value=normalized_occluder,
            source=(
                "eventframe_inference"
            ),
            parser="eventframe",
            confidence=0.70,
            original_text=str(
                raw_occluder
            ),
            normalized_from=str(
                raw_occluder
            ),
        )

    interaction = _dict(
        mapped.get(
            "interaction_layer"
        )
    )

    direction = normalize_direction(
        interaction.get(
            "conflict_direction"
        )
    )

    source_relation = str(
        interaction.get(
            "source_relation",
            "",
        )
    ).strip().lower()

    if direction is None:
        if source_relation == "from_left":
            direction = "left_to_right"

        elif source_relation == "from_right":
            direction = "right_to_left"

    if direction is not None:
        candidates[
            "direction"
        ] = FieldEvidence(
            value=direction,
            source=(
                "eventframe_inference"
            ),
            parser="eventframe",
            confidence=0.68,
            original_text=str(
                interaction.get(
                    "conflict_direction"
                )
                or source_relation
            ),
        )

    completed = _dict(
        _dict(
            mapped.get(
                "parameter_layer"
            )
        ).get(
            "completed"
        )
    )

    speed_slot = completed.get(
        "actor_speed_mps"
    )

    speed = _slot_midpoint(
        speed_slot
    )

    if speed is not None:
        candidates[
            "pedestrian_speed_mps"
        ] = FieldEvidence(
            value=speed,
            source=(
                "eventframe_completion"
            ),
            parser="eventframe",
            confidence=0.55,
            original_text=str(
                speed_slot
            ),
            value_range=(
                _slot_range(
                    speed_slot
                )
            ),
        )

    risk_layer = _dict(
        mapped.get(
            "risk_layer"
        )
    )

    risk = normalize_risk(
        risk_layer.get(
            "risk_level"
        )
    )

    if risk is not None:
        candidates[
            "risk_level"
        ] = FieldEvidence(
            value=risk,
            source=(
                "eventframe_inference"
            ),
            parser="eventframe",
            confidence=0.65,
            original_text=str(
                risk_layer.get(
                    "risk_level"
                )
            ),
        )

    return candidates


def resolve_prompt_controls(
    prompt: str,
    mapped: Mapping[str, Any],
    *,
    overrides: Optional[Any] = None,
    strict_conflicts: bool = True,
    fail_on_warning: bool = False,
) -> ResolutionResult:
    """Resolve executable controls using explicit, rule, EventFrame, and defaults.

    Priority:
        explicit user override
        > deterministic prompt evidence
        > EventFrame inference/completion
        > deterministic template default

    ``overrides`` is duck-typed to avoid a circular import. It may expose the
    four control attributes, ``locked_fields``, and ``source_by_field``.
    """

    rule = RuleBasedPromptParser().parse(
        prompt
    )

    eventframe = (
        extract_eventframe_candidates(
            mapped
        )
    )

    issues = list(
        rule.issues
    )

    override_values = {
        field_name: (
            getattr(
                overrides,
                field_name,
                None,
            )
            if overrides is not None
            else None
        )
        for field_name
        in SUPPORTED_CONTROL_FIELDS
    }

    source_by_field = dict(
        getattr(
            overrides,
            "source_by_field",
            {},
        )
        or {}
    )

    requested_locks = set(
        getattr(
            overrides,
            "locked_fields",
            (),
        )
        or ()
    )

    requested_locks.update(
        key
        for key, value
        in override_values.items()
        if value is not None
    )

    fields: Dict[
        str,
        ResolvedField,
    ] = {}

    defaults_used: List[str] = []

    for field_name in SUPPORTED_CONTROL_FIELDS:
        prompt_evidence = (
            rule.fields.get(
                field_name
            )
        )

        event_evidence = (
            eventframe.get(
                field_name
            )
        )

        override_value = (
            override_values[
                field_name
            ]
        )

        if (
            prompt_evidence is not None
            and prompt_evidence.value
            == "ambiguous"
        ):
            issues.append(
                ResolutionIssue(
                    code=(
                        f"ambiguous_{field_name}"
                    ),
                    severity="error",
                    message=(
                        "The prompt contains mutually "
                        "incompatible values for "
                        f"{field_name}."
                    ),
                    fields=(
                        field_name,
                    ),
                )
            )

            prompt_evidence = None

        if (
            prompt_evidence is not None
            and event_evidence is not None
            and event_evidence.source
            != "eventframe_completion"
        ):
            if not values_equivalent(
                field_name,
                prompt_evidence.value,
                event_evidence.value,
            ):
                issues.append(
                    ResolutionIssue(
                        code=(
                            "dual_parser_disagreement_"
                            f"{field_name}"
                        ),
                        severity="warning",
                        message=(
                            "The deterministic parser "
                            "and EventFrame parser "
                            f"disagree on {field_name}; "
                            "deterministic prompt "
                            "evidence is used unless "
                            "explicitly overridden."
                        ),
                        fields=(
                            field_name,
                        ),
                        details={
                            "rule_value": (
                                prompt_evidence.value
                            ),
                            "eventframe_value": (
                                event_evidence.value
                            ),
                        },
                    )
                )

        alternatives = tuple(
            evidence.to_dict()
            for evidence in (
                prompt_evidence,
                event_evidence,
            )
            if evidence is not None
        )

        if override_value is not None:
            normalized = (
                normalize_control_value(
                    field_name,
                    override_value,
                )
            )

            source = source_by_field.get(
                field_name,
                "explicit_override",
            )

            if (
                prompt_evidence is not None
                and not values_equivalent(
                    field_name,
                    normalized,
                    prompt_evidence.value,
                )
            ):
                issues.append(
                    ResolutionIssue(
                        code=(
                            "override_prompt_conflict_"
                            f"{field_name}"
                        ),
                        severity="warning",
                        message=(
                            "The explicit value for "
                            f"{field_name} overrides a "
                            "different value stated in "
                            "the prompt."
                        ),
                        fields=(
                            field_name,
                        ),
                        details={
                            "explicit_value": (
                                normalized
                            ),
                            "prompt_value": (
                                prompt_evidence.value
                            ),
                        },
                    )
                )

            evidence = FieldEvidence(
                value=normalized,
                source=source,
                parser="explicit",
                confidence=1.0,
                original_text=str(
                    override_value
                ),
            )

        elif prompt_evidence is not None:
            evidence = prompt_evidence

        elif event_evidence is not None:
            evidence = event_evidence

        else:
            defaults_used.append(
                field_name
            )

            evidence = FieldEvidence(
                value=DEFAULT_CONTROLS[
                    field_name
                ],
                source="template_default",
                parser="default",
                confidence=None,
            )

        normalized_value = (
            normalize_control_value(
                field_name,
                evidence.value,
            )
        )

        locked = (
            field_name
            in requested_locks
        )

        fields[
            field_name
        ] = ResolvedField(
            value=normalized_value,
            source=evidence.source,
            confidence=evidence.confidence,
            locked=locked,
            adjustable=not locked,
            original_text=(
                evidence.original_text
            ),
            normalized_from=(
                evidence.normalized_from
            ),
            value_range=(
                evidence.value_range
            ),
            alternatives=alternatives,
        )

    chosen_occluder = str(
        fields[
            "occluder_type"
        ].value
    )

    if (
        chosen_occluder
        in rule.excluded_occluders
    ):
        issues.append(
            ResolutionIssue(
                code=(
                    "excluded_occluder_selected"
                ),
                severity="error",
                message=(
                    "The resolved occluder "
                    f"{chosen_occluder!r} "
                    "was explicitly excluded "
                    "by the prompt."
                ),
                fields=(
                    "occluder_type",
                ),
            )
        )

    _validate_side_direction_consistency(
        rule,
        fields,
        issues,
    )

    _validate_supported_family(
        rule,
        mapped,
        issues,
    )

    error_count = sum(
        issue.severity == "error"
        for issue in issues
    )

    warning_count = sum(
        issue.severity == "warning"
        for issue in issues
    )

    semantic_valid = (
        error_count == 0
        and (
            not fail_on_warning
            or warning_count == 0
        )
    )

    requires_confirmation = bool(
        defaults_used
        or warning_count
    )

    if (
        strict_conflicts
        and error_count
    ):
        semantic_valid = False

    locked_fields = tuple(
        field_name
        for field_name, item
        in fields.items()
        if item.locked
    )

    adjustable_fields = tuple(
        field_name
        for field_name, item
        in fields.items()
        if item.adjustable
    )

    return ResolutionResult(
        prompt=prompt,
        fields=fields,
        issues=issues,
        rule_parse=rule,
        eventframe_candidates=eventframe,
        semantic_valid=semantic_valid,
        requires_confirmation=(
            requires_confirmation
        ),
        defaults_used=tuple(
            defaults_used
        ),
        locked_fields=locked_fields,
        adjustable_fields=(
            adjustable_fields
        ),
    )


def resolution_from_payload(
    payload: Mapping[str, Any],
) -> ResolutionResult:
    """Load an edited resolution JSON file produced by ``ResolutionResult``."""

    fields_payload = _dict(
        payload.get(
            "fields"
        )
    )

    missing = [
        name
        for name in SUPPORTED_CONTROL_FIELDS
        if name not in fields_payload
    ]

    if missing:
        raise ValueError(
            "Resolution payload is "
            f"missing fields: {missing}"
        )

    fields: Dict[
        str,
        ResolvedField,
    ] = {}

    for name in SUPPORTED_CONTROL_FIELDS:
        item = _dict(
            fields_payload[name]
        )

        value = normalize_control_value(
            name,
            item.get("value"),
        )

        value_range_raw = item.get(
            "value_range"
        )

        value_range = None

        if (
            isinstance(
                value_range_raw,
                Sequence,
            )
            and not isinstance(
                value_range_raw,
                (str, bytes),
            )
        ):
            values = list(
                value_range_raw
            )

            if len(values) >= 2:
                value_range = (
                    float(values[0]),
                    float(values[1]),
                )

        fields[name] = ResolvedField(
            value=value,
            source=str(
                item.get(
                    "source",
                    "edited_resolution",
                )
            ),
            confidence=_optional_float(
                item.get(
                    "confidence"
                )
            ),
            locked=bool(
                item.get(
                    "locked",
                    False,
                )
            ),
            adjustable=bool(
                item.get(
                    "adjustable",
                    not bool(
                        item.get(
                            "locked",
                            False,
                        )
                    ),
                )
            ),
            original_text=item.get(
                "original_text"
            ),
            normalized_from=item.get(
                "normalized_from"
            ),
            value_range=value_range,
            alternatives=tuple(
                item.get(
                    "alternatives",
                    (),
                )
                or ()
            ),
        )

    issues = [
        ResolutionIssue(
            code=str(
                item.get(
                    "code",
                    "loaded_issue",
                )
            ),
            severity=str(
                item.get(
                    "severity",
                    "warning",
                )
            ),
            message=str(
                item.get(
                    "message",
                    "",
                )
            ),
            fields=tuple(
                item.get(
                    "fields",
                    (),
                )
                or ()
            ),
            details=dict(
                item.get(
                    "details",
                    {},
                )
                or {}
            ),
        )
        for item in (
            payload.get(
                "issues",
                (),
            )
            or ()
        )
    ]

    empty_rule = RuleParseResult()

    semantic_valid = bool(
        payload.get(
            "semantic_valid",
            True,
        )
    ) and not any(
        issue.severity == "error"
        for issue in issues
    )

    locked_fields = tuple(
        name
        for name, item
        in fields.items()
        if item.locked
    )

    adjustable_fields = tuple(
        name
        for name, item
        in fields.items()
        if item.adjustable
    )

    return ResolutionResult(
        prompt=str(
            payload.get(
                "prompt",
                "",
            )
        ),
        fields=fields,
        issues=issues,
        rule_parse=empty_rule,
        eventframe_candidates={},
        reference_frame=str(
            payload.get(
                "reference_frame",
                "ego",
            )
        ),
        scenario_family=str(
            payload.get(
                "scenario_family",
                "occluded_pedestrian",
            )
        ),
        semantic_valid=semantic_valid,
        requires_confirmation=bool(
            payload.get(
                "requires_confirmation",
                False,
            )
        ),
        defaults_used=tuple(
            payload.get(
                "defaults_used",
                (),
            )
            or ()
        ),
        locked_fields=locked_fields,
        adjustable_fields=(
            adjustable_fields
        ),
    )


def merge_resolution_with_explicit_values(
    resolution: ResolutionResult,
    explicit_values: Mapping[str, Any],
    explicit_locks: Iterable[str] = (),
) -> ResolutionResult:
    """Apply direct CLI values to an edited or loaded resolution template."""

    fields = dict(
        resolution.fields
    )

    locks = set(
        resolution.locked_fields
    )

    locks.update(
        explicit_locks
    )

    issues = list(
        resolution.issues
    )

    for name in SUPPORTED_CONTROL_FIELDS:
        raw = explicit_values.get(
            name
        )

        if raw is None:
            continue

        value = normalize_control_value(
            name,
            raw,
        )

        previous = fields[name]

        if not values_equivalent(
            name,
            value,
            previous.value,
        ):
            issues.append(
                ResolutionIssue(
                    code=(
                        "edited_resolution_override_"
                        f"{name}"
                    ),
                    severity="warning",
                    message=(
                        "A direct explicit value "
                        f"replaced {name} from the "
                        "resolution file."
                    ),
                    fields=(
                        name,
                    ),
                    details={
                        "previous": previous.value,
                        "explicit": value,
                    },
                )
            )

        locks.add(name)

        fields[name] = ResolvedField(
            value=value,
            source="explicit_override",
            confidence=1.0,
            locked=True,
            adjustable=False,
            original_text=str(raw),
            alternatives=(
                previous.to_dict(),
            ),
        )

    for (
        name,
        item,
    ) in list(
        fields.items()
    ):
        locked = name in locks

        fields[name] = ResolvedField(
            value=item.value,
            source=item.source,
            confidence=item.confidence,
            locked=locked,
            adjustable=not locked,
            original_text=(
                item.original_text
            ),
            normalized_from=(
                item.normalized_from
            ),
            value_range=(
                item.value_range
            ),
            alternatives=(
                item.alternatives
            ),
        )

    return ResolutionResult(
        prompt=resolution.prompt,
        fields=fields,
        issues=issues,
        rule_parse=(
            resolution.rule_parse
        ),
        eventframe_candidates=(
            resolution.eventframe_candidates
        ),
        reference_frame=(
            resolution.reference_frame
        ),
        scenario_family=(
            resolution.scenario_family
        ),
        semantic_valid=not any(
            issue.severity == "error"
            for issue in issues
        ),
        requires_confirmation=bool(
            resolution.defaults_used
            or any(
                issue.severity == "warning"
                for issue in issues
            )
        ),
        defaults_used=(
            resolution.defaults_used
        ),
        locked_fields=tuple(
            name
            for name, item
            in fields.items()
            if item.locked
        ),
        adjustable_fields=tuple(
            name
            for name, item
            in fields.items()
            if item.adjustable
        ),
    )


def normalize_control_value(
    field_name: str,
    value: Any,
) -> Any:
    if field_name == "occluder_type":
        normalized = normalize_occluder(
            value
        )

        if normalized is None:
            raise ValueError(
                "Unsupported "
                f"occluder_type={value!r}"
            )

        return normalized

    if field_name == "direction":
        normalized = normalize_direction(
            value
        )

        if normalized is None:
            raise ValueError(
                "Unsupported "
                f"direction={value!r}"
            )

        return normalized

    if (
        field_name
        == "pedestrian_speed_mps"
    ):
        return validate_speed(
            value
        )

    if field_name == "risk_level":
        normalized = normalize_risk(
            value
        )

        if normalized is None:
            raise ValueError(
                "Unsupported "
                f"risk_level={value!r}"
            )

        return normalized

    raise KeyError(
        "Unsupported control "
        f"field={field_name!r}"
    )


def normalize_occluder(
    value: Any,
) -> Optional[str]:
    text = (
        str(
            value or ""
        )
        .strip()
        .lower()
        .replace(
            "-",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )

    if (
        not text
        or text in {
            "none",
            "unknown",
            "null",
        }
    ):
        return None

    aliases = {
        "car": "vehicle",
        "parked_car": "vehicle",
        "parked_vehicle": "vehicle",
        "stationary_vehicle": "vehicle",
        "van": "vehicle",
        "delivery_van": "vehicle",
        "truck": "vehicle",
        "bus": "vehicle",
        "bike": "bicycle",
        "road_cone": "traffic_cone",
        "cone": "traffic_cone",
        "guardrail": "barrier",
        "road_barrier": "barrier",
        "construction_sign": "czone_sign",
        "construction_zone_sign": (
            "czone_sign"
        ),
        "generic_obstacle": (
            "generic_object"
        ),
    }

    text = aliases.get(
        text,
        text,
    )

    return (
        text
        if text in SUPPORTED_OCCLUDERS
        else None
    )


def normalize_direction(
    value: Any,
) -> Optional[str]:
    text = (
        str(
            value or ""
        )
        .strip()
        .lower()
        .replace(
            "-",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )

    mapping = {
        "lefttoright": "left_to_right",
        "ltr": "left_to_right",
        "from_left": "left_to_right",
        "righttoleft": "right_to_left",
        "rtl": "right_to_left",
        "from_right": "right_to_left",
    }

    text = mapping.get(
        text,
        text,
    )

    return (
        text
        if text in SUPPORTED_DIRECTIONS
        else None
    )


def normalize_risk(
    value: Any,
) -> Optional[str]:
    text = (
        str(
            value or ""
        )
        .strip()
        .lower()
        .replace(
            "-",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )

    mapping = {
        "low": "mild",
        "low_risk": "mild",
        "medium": "moderate",
        "medium_risk": "moderate",
        "high": "aggressive",
        "high_risk": "aggressive",
    }

    text = mapping.get(
        text,
        text,
    )

    return (
        text
        if text in SUPPORTED_RISK_LEVELS
        else None
    )


def validate_speed(
    value: Any,
) -> float:
    speed = float(value)

    if not math.isfinite(
        speed
    ):
        raise ValueError(
            "pedestrian_speed_mps "
            "must be finite"
        )

    if not 0.5 <= speed <= 2.0:
        raise ValueError(
            "pedestrian_speed_mps="
            f"{speed} is outside [0.5, 2.0]; "
            "the current RVAE configuration "
            "supports at most 2.0 m/s"
        )

    return speed


def values_equivalent(
    field_name: str,
    first: Any,
    second: Any,
) -> bool:
    try:
        left = normalize_control_value(
            field_name,
            first,
        )

        right = normalize_control_value(
            field_name,
            second,
        )

    except (
        TypeError,
        ValueError,
        KeyError,
    ):
        return False

    if (
        field_name
        == "pedestrian_speed_mps"
    ):
        return (
            abs(
                float(left)
                - float(right)
            )
            <= 0.05
        )

    return left == right


def _validate_side_direction_consistency(
    rule: RuleParseResult,
    fields: Mapping[
        str,
        ResolvedField,
    ],
    issues: List[
        ResolutionIssue
    ],
) -> None:
    if rule.emergence_side not in {
        "left",
        "right",
    }:
        return

    expected = (
        "left_to_right"
        if rule.emergence_side == "left"
        else "right_to_left"
    )

    actual = str(
        fields[
            "direction"
        ].value
    )

    if (
        actual != expected
        and not any(
            issue.code
            == "emergence_direction_conflict"
            for issue in issues
        )
    ):
        issues.append(
            ResolutionIssue(
                code=(
                    "resolved_side_direction_conflict"
                ),
                severity="error",
                message=(
                    "emergence_side="
                    f"{rule.emergence_side} "
                    "requires direction="
                    f"{expected}, but the "
                    "resolved direction is "
                    f"{actual}."
                ),
                fields=(
                    "direction",
                    "emergence_side",
                ),
            )
        )


def _validate_supported_family(
    rule: RuleParseResult,
    mapped: Mapping[str, Any],
    issues: List[
        ResolutionIssue
    ],
) -> None:
    actor = str(
        _dict(
            mapped.get(
                "actor_layer"
            )
        ).get(
            "primary_actor",
            "",
        )
    ).lower()

    conflict = str(
        _dict(
            mapped.get(
                "interaction_layer"
            )
        ).get(
            "conflict_type",
            "",
        )
    ).lower()

    occluded = bool(
        _dict(
            _dict(
                mapped.get(
                    "object_layer"
                )
            ).get(
                "occlusion"
            )
        ).get(
            "enabled",
            False,
        )
    )

    eventframe_supports = (
        actor == "pedestrian"
        and conflict in {
            "lateral_conflict",
            "crossing_path_conflict",
        }
        and occluded
    )

    if rule.negated_occlusion:
        return

    if (
        not rule.target_family_detected
        and not eventframe_supports
    ):
        issues.append(
            ResolutionIssue(
                code=(
                    "unsupported_prompt_family"
                ),
                severity="error",
                message=(
                    "The prompt is not a "
                    "supported occluded-pedestrian "
                    "crossing scene."
                ),
                fields=(
                    "scenario_family",
                ),
                details={
                    "eventframe_actor": actor,
                    "eventframe_conflict": (
                        conflict
                    ),
                    "eventframe_occlusion": (
                        occluded
                    ),
                },
            )
        )


def _slot_midpoint(
    slot: Any,
) -> Optional[float]:
    value = (
        slot.get(
            "value"
        )
        if isinstance(
            slot,
            Mapping,
        )
        else slot
    )

    if isinstance(
        value,
        (int, float),
    ):
        return float(value)

    if (
        isinstance(
            value,
            Sequence,
        )
        and not isinstance(
            value,
            (str, bytes),
        )
    ):
        values = list(value)

        if len(values) >= 2:
            return 0.5 * (
                float(values[0])
                + float(values[1])
            )

        if len(values) == 1:
            return float(
                values[0]
            )

    return None


def _slot_range(
    slot: Any,
) -> Optional[
    Tuple[float, float]
]:
    value = (
        slot.get(
            "value"
        )
        if isinstance(
            slot,
            Mapping,
        )
        else slot
    )

    if (
        isinstance(
            value,
            Sequence,
        )
        and not isinstance(
            value,
            (str, bytes),
        )
    ):
        values = list(value)

        if len(values) >= 2:
            first = float(
                values[0]
            )
            second = float(
                values[1]
            )

            return (
                min(
                    first,
                    second,
                ),
                max(
                    first,
                    second,
                ),
            )

    return None


def _dict(
    value: Any,
) -> Dict[str, Any]:
    return (
        dict(value)
        if isinstance(
            value,
            Mapping,
        )
        else {}
    )


def _optional_float(
    value: Any,
) -> Optional[float]:
    if value is None:
        return None

    return float(value)


def _contains_any(
    text: str,
    terms: Iterable[str],
) -> bool:
    return any(
        term.lower() in text
        for term in terms
    )


def _mentions_behind_occluder(
    text: str,
) -> bool:
    lower = text.lower()

    behind = bool(
        re.search(
            (
                r"(?:behind|from behind).{0,40}"
                r"(?:parked\s+)?"
                r"(?:vehicle|car|truck|bus|"
                r"van|barrier|bicycle)"
            ),
            lower,
        )
        or re.search(
            (
                r"(?:from|从|在).{0,20}"
                r"(?:parked\s+vehicle|车辆|"
                r"货车|公交车|护栏|自行车|遮挡物)"
                r"\s*(?:后|后方|背后)"
            ),
            text,
            flags=re.IGNORECASE,
        )
    )

    reverse = bool(
        re.search(
            (
                r"(?:vehicle|car|truck|bus|"
                r"van|barrier|bicycle).{0,40}"
                r"(?:hides|occludes|blocks)"
            ),
            lower,
        )
        or re.search(
            (
                r"(?:车辆|货车|公交车|护栏|"
                r"自行车|遮挡物).{0,20}"
                r"(?:遮住|遮挡|挡住)"
            ),
            text,
        )
    )

    side_emergence = bool(
        re.search(
            (
                r"(?:from|at)\s+the\s+"
                r"(?:left|right)\s+side\s+of\s+"
                r"(?:the\s+)?"
                r"(?:vehicle|car|truck|bus|"
                r"van|occluder)"
            ),
            lower,
        )
        or re.search(
            (
                r"从(?:(?:停放的?|停靠的?)?)"
                r"(?:车辆|汽车|货车|卡车|"
                r"公交车|大巴车|遮挡物)"
                r"(?:左|右)(?:侧|边)"
                r"(?:冲出|出现|露出|进入)"
            ),
            text,
        )
    )

    return (
        behind
        or reverse
        or side_emergence
    )


def _first_matching_fragment(
    text: str,
    patterns: Sequence[str],
) -> Optional[str]:
    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(0)

    return None