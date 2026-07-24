"""EventFrame-to-executable-spec adapter for occluded pedestrian scenes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re
from typing import Any, Dict, Optional

from sledge.semantic_control.language import (
    EventFrameParser,
    EventFrameToHazardSpecMapper,
    EventFrameVerifier,
    EventSequenceBuilder,
    MissingInfoFiller,
)

from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_spec import (
    HazardSemanticSpec,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.spec_presets import (
    apply_risk_preset,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.object_types import (
    SUPPORTED_OCCLUDER_TYPES,
    normalize_occluder_type,
)


SUPPORTED_OCCLUDERS = set(SUPPORTED_OCCLUDER_TYPES)
SUPPORTED_DIRECTIONS = {"left_to_right", "right_to_left"}
SUPPORTED_RISK_LEVELS = {"mild", "moderate", "aggressive"}


@dataclass
class ControlOverrides:
    """Optional executable controls with explicit provenance."""

    occluder_type: Optional[str] = None
    direction: Optional[str] = None
    pedestrian_speed_mps: Optional[float] = None
    risk_level: Optional[str] = None


@dataclass
class AdaptationResult:
    prompt: str
    event_frame: Dict[str, Any]
    mapped_eventframe_spec: Dict[str, Any]
    hazard_spec: HazardSemanticSpec
    frame_verification: Dict[str, Any]
    spec_verification: Dict[str, Any]
    provenance: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "event_frame": self.event_frame,
            "mapped_eventframe_spec": self.mapped_eventframe_spec,
            "hazard_spec": self.hazard_spec.to_dict(),
            "frame_verification": self.frame_verification,
            "spec_verification": self.spec_verification,
            "provenance": self.provenance,
        }


class OccludedPedestrianEventFrameAdapter:
    """Compile one supported EventFrame family into ``HazardSemanticSpec``.

    The adapter is intentionally strict. Unsupported or non-occluded prompts
    fail instead of silently falling back to an ordinary pedestrian crossing.
    """

    def __init__(
        self,
        *,
        llm_provider: str = "none",
        llm_model: str = "qwen2.5:7b",
        ollama_url: str = "http://127.0.0.1:11434",
    ) -> None:
        self.parser = EventFrameParser(
            llm_provider="ollama" if llm_provider == "ollama" else "none",
            llm_model=llm_model,
            ollama_url=ollama_url,
            allow_fallback=True,
        )
        self.sequence_builder = EventSequenceBuilder()
        self.mapper = EventFrameToHazardSpecMapper()
        self.verifier = EventFrameVerifier()
        self.filler = MissingInfoFiller()

    def adapt(self, prompt: str, overrides: Optional[ControlOverrides] = None) -> AdaptationResult:
        overrides = overrides or ControlOverrides()
        frame = self.parser.parse(prompt)
        frame = self.sequence_builder.build(frame, overwrite=True)
        first_check = self.verifier.verify_frame(frame)
        if not first_check.passed:
            frame = self.verifier.repair_frame(frame)
            frame = self.sequence_builder.build(frame, overwrite=True)
        frame_check = self.verifier.verify_frame(frame)

        mapped = self.mapper.map(frame)
        mapped = self.filler.fill(mapped, frame)
        mapped = self._repair_supported_prompt_semantics(prompt, mapped)
        mapped_check = self.verifier.verify_spec(mapped)

        self._require_supported_semantics(mapped)
        provenance: Dict[str, Dict[str, Any]] = {}

        occluder, source = self._resolve_occluder(prompt, mapped, overrides)
        provenance["occluder_type"] = {"value": occluder, "source": source}

        direction, source = self._resolve_direction(prompt, mapped, overrides)
        provenance["direction"] = {"value": direction, "source": source}

        speed, source = self._resolve_pedestrian_speed(prompt, mapped, overrides)
        provenance["pedestrian_speed_mps"] = {"value": speed, "source": source}

        risk, source = self._resolve_risk(prompt, mapped, overrides)
        provenance["risk_level"] = {"value": risk, "source": source}

        digest = hashlib.sha1(
            f"{prompt}|{occluder}|{direction}|{speed:.3f}|{risk}".encode("utf-8")
        ).hexdigest()[:12]

        road_topology = str(mapped.get("road_layer", {}).get("road_topology", "straight"))
        if road_topology in {"unknown", "straight_lane", "curbside"}:
            road_topology = "straight"

        spec = HazardSemanticSpec.from_dict(
            {
                "spec_id": f"occluded_pedestrian_{digest}",
                "description": "A pedestrian emerges from behind a nuPlan-visible occluder and crosses the ego path.",
                "canonical_type": "Occluded-Pedestrian",
                "raw_prompt": prompt,
                "road_layer": {
                    "road_topology": road_topology,
                    "lane_context": "ego_path",
                    "anchor_type": "ego_future_path",
                    "anchor_region": "front",
                    "has_crosswalk": bool(mapped.get("road_layer", {}).get("has_crosswalk", False)),
                    "require_lane_continuity": True,
                    "require_drivable_route": True,
                },
                "actor_layer": {
                    "primary_actor": "pedestrian",
                    "actor_role": "crossing_actor",
                    "secondary_actor": occluder,
                    "allow_actor_insertion": True,
                    "prefer_existing_actor": False,
                },
                "object_layer": {
                    "occlusion": {
                        "enabled": True,
                        "occluder_type": occluder,
                        "occlusion_position": "between_ego_and_actor",
                        "occlusion_level": "full",
                    },
                    "static_obstacle": {"enabled": False},
                },
                "interaction_layer": {
                    "conflict_type": "lateral_conflict",
                    "conflict_direction": direction,
                    "distance_relation": "close",
                    "speed_relation": "normal",
                    "interaction_goal": "near_miss",
                },
                "risk_layer": {
                    "risk_level": risk,
                    "target_actor_speed_mps": speed,
                    "collision_allowed": False,
                },
                "validation_layer": {
                    "require_actor_match": True,
                    "require_road_context_match": True,
                    "require_conflict_relation": True,
                    "require_direction_match": True,
                    "require_visibility_match": True,
                    "require_lane_validity": True,
                    "require_no_initial_collision": True,
                    "require_ttc_in_range": True,
                    "require_gap_in_range": True,
                },
                "protection_layer": {
                    "protect_primary_actor": True,
                    "protect_secondary_actor": True,
                    "protect_static_obstacle": True,
                    "protect_conflict_corridor": True,
                    "protect_road_anchor": True,
                },
                "tags": ["eventframe", "occluded_pedestrian", occluder, direction, risk],
                "debug": {"adapter_provenance": provenance},
            }
        )
        spec = apply_risk_preset(spec, overwrite=True)
        spec.risk_layer.target_actor_speed_mps = speed

        return AdaptationResult(
            prompt=prompt,
            event_frame=frame.to_dict(),
            mapped_eventframe_spec=mapped,
            hazard_spec=spec,
            frame_verification=asdict(frame_check),
            spec_verification=asdict(mapped_check),
            provenance=provenance,
        )

    @staticmethod
    def _require_supported_semantics(mapped: Dict[str, Any]) -> None:
        actor = mapped.get("actor_layer", {}).get("primary_actor")
        conflict = mapped.get("interaction_layer", {}).get("conflict_type")
        occluded = bool(mapped.get("object_layer", {}).get("occlusion", {}).get("enabled", False))
        errors = []
        if actor != "pedestrian":
            errors.append(f"primary_actor={actor!r}, expected 'pedestrian'")
        if conflict not in {"lateral_conflict", "crossing_path_conflict"}:
            errors.append(f"conflict_type={conflict!r}, expected lateral crossing")
        if not occluded:
            errors.append("occlusion.enabled is false")
        if errors:
            raise ValueError("Unsupported prompt for occluded-pedestrian pipeline: " + "; ".join(errors))

    @staticmethod
    def _repair_supported_prompt_semantics(prompt: str, mapped: Dict[str, Any]) -> Dict[str, Any]:
        """Repair parser misses only when the prompt itself contains clear evidence."""

        lower = prompt.lower()
        has_pedestrian = any(word in lower for word in ["pedestrian", "person", "walker", "行人"])
        has_crossing = any(word in lower for word in ["cross", "across", "rush", "enter", "横穿", "冲出", "穿过"])
        has_occlusion = any(
            word in lower
            for word in [
                "hidden behind",
                "comes out from behind",
                "emerges from behind",
                "blocks ego's view",
                "blocks the view",
                "occlud",
                "遮挡",
                "视线",
            ]
        )
        repairs = []
        if has_pedestrian:
            actor = mapped.setdefault("actor_layer", {})
            if actor.get("primary_actor") != "pedestrian":
                actor["primary_actor"] = "pedestrian"
                repairs.append("primary_actor<-prompt_evidence")
        if has_crossing:
            interaction = mapped.setdefault("interaction_layer", {})
            if interaction.get("conflict_type") not in {"lateral_conflict", "crossing_path_conflict"}:
                interaction["conflict_type"] = "lateral_conflict"
                repairs.append("conflict_type<-prompt_evidence")
        if has_occlusion:
            occlusion = mapped.setdefault("object_layer", {}).setdefault("occlusion", {})
            if not bool(occlusion.get("enabled", False)):
                occlusion["enabled"] = True
                repairs.append("occlusion.enabled<-prompt_evidence")
            prompt_occluder = _extract_occluder(prompt)
            if prompt_occluder:
                occlusion["occluder_type"] = prompt_occluder
        if repairs:
            mapped.setdefault("debug", {})["pipeline_semantic_repairs"] = repairs
        return mapped

    @staticmethod
    def _resolve_occluder(prompt: str, mapped: Dict[str, Any], overrides: ControlOverrides):
        if overrides.occluder_type:
            value = _normalize_occluder(overrides.occluder_type)
            return value, "control_override"
        text_value = _extract_occluder(prompt)
        if text_value:
            return text_value, "prompt_evidence"
        mapped_value = mapped.get("object_layer", {}).get("occlusion", {}).get("occluder_type")
        value = _normalize_occluder(mapped_value)
        return value, "eventframe"

    @staticmethod
    def _resolve_direction(prompt: str, mapped: Dict[str, Any], overrides: ControlOverrides):
        if overrides.direction:
            value = str(overrides.direction)
            if value not in SUPPORTED_DIRECTIONS:
                raise ValueError(f"Unsupported direction={value!r}")
            return value, "control_override"
        text_value = _extract_direction(prompt)
        if text_value:
            return text_value, "prompt_evidence"
        relation = str(mapped.get("interaction_layer", {}).get("source_relation", ""))
        if relation == "from_left":
            return "left_to_right", "eventframe"
        if relation == "from_right":
            return "right_to_left", "eventframe"
        return "right_to_left", "deterministic_default"

    @staticmethod
    def _resolve_pedestrian_speed(prompt: str, mapped: Dict[str, Any], overrides: ControlOverrides):
        if overrides.pedestrian_speed_mps is not None:
            return _validate_speed(overrides.pedestrian_speed_mps), "control_override"
        text_value = _extract_pedestrian_speed(prompt)
        if text_value is not None:
            return _validate_speed(text_value), "prompt_evidence"
        completed = mapped.get("parameter_layer", {}).get("completed", {})
        slot = completed.get("actor_speed_mps", {})
        value = slot.get("value") if isinstance(slot, dict) else None
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return _validate_speed(0.5 * (float(value[0]) + float(value[1]))), "eventframe_default"
        return 1.6, "deterministic_default"

    @staticmethod
    def _resolve_risk(prompt: str, mapped: Dict[str, Any], overrides: ControlOverrides):
        if overrides.risk_level:
            value = str(overrides.risk_level).lower()
            if value not in SUPPORTED_RISK_LEVELS:
                raise ValueError(f"Unsupported risk_level={value!r}")
            return value, "control_override"
        explicit = _extract_explicit_risk(prompt)
        if explicit:
            return explicit, "prompt_evidence"
        value = str(mapped.get("risk_layer", {}).get("risk_level", "moderate")).lower()
        return (value if value in SUPPORTED_RISK_LEVELS else "moderate"), "eventframe"


def _normalize_occluder(value: Any) -> str:
    return normalize_occluder_type(value)


def _extract_occluder(text: str) -> Optional[str]:
    lower = text.lower()
    patterns = [
        ("traffic_cone", ["traffic cone", "road cone", "锥桶", "交通锥", "路锥"]),
        ("czone_sign", ["czone sign", "construction zone sign", "construction-zone sign", "construction sign", "施工标志", "施工牌"]),
        ("generic_object", ["generic object", "generic obstacle", "通用物体", "通用障碍物"]),
        ("barrier", ["road barrier", "barrier", "护栏", "路障"]),
        ("bicycle", ["bicycle", "bike", "自行车", "单车"]),
        ("vehicle", [
            "parked vehicle", "parked car", "vehicle", "car", "van", "truck", "bus",
            "停放车辆", "停靠车辆", "轿车", "面包车", "卡车", "货车", "公交", "大巴",
        ]),
    ]
    for value, words in patterns:
        if any(word in lower for word in words):
            return value
    return None


def _extract_direction(text: str) -> Optional[str]:
    lower = text.lower().replace("-", " ")
    if re.search(r"(?:from\s+)?right\s+to\s+left", lower) or re.search(r"右(?:侧)?.{0,12}(?:向|到)左", text):
        return "right_to_left"
    if re.search(r"(?:from\s+)?left\s+to\s+right", lower) or re.search(r"左(?:侧)?.{0,12}(?:向|到)右", text):
        return "left_to_right"
    return None


def _extract_pedestrian_speed(text: str) -> Optional[float]:
    patterns = [
        r"(?:pedestrian|person|walker|行人).{0,100}?(?:at\s+)?(?:speed\s*)?(\d+(?:\.\d+)?)\s*m\s*/?\s*s",
        r"(?:行人速度|pedestrian speed)\s*(?:is|为|=|:)?\s*(\d+(?:\.\d+)?)",
        r"(?:at\s+speed|速度)\s*(?:of|为|=|:)?\s*(\d+(?:\.\d+)?)\s*m\s*/?\s*s",
        r"(?:at|以)\s*(\d+(?:\.\d+)?)\s*(?:m\s*/?\s*s|米每秒)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _extract_explicit_risk(text: str) -> Optional[str]:
    lower = text.lower()
    if any(word in lower for word in ["aggressive", "high risk", "高危", "激进", "危险等级高"]):
        return "aggressive"
    if any(word in lower for word in ["mild", "low risk", "轻度", "温和"]):
        return "mild"
    if any(word in lower for word in ["moderate", "medium risk", "中度", "适中"]):
        return "moderate"
    return None


def _validate_speed(value: Any) -> float:
    speed = float(value)
    if not 0.5 <= speed <= 2.0:
        raise ValueError(
            f"pedestrian_speed_mps={speed} is outside [0.5, 2.0]; "
            "the current RVAE config clips pedestrian speed at 2.0 m/s"
        )
    return speed
