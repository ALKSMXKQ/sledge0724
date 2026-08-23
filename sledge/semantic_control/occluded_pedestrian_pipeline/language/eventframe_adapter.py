"""Hierarchical EventFrame-to-executable-spec adapter for occluded pedestrians."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, Mapping, Optional

from sledge.semantic_control.language.hierarchical_pipeline import (
    HierarchicalEventFramePipeline,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_spec import (
    HazardSemanticSpec,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hierarchical_spec_adapter import (
    HierarchicalHazardSpecAdapter,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hierarchical_template_sampler import (
    HierarchicalTemplateSampler,
    SamplingOverrides,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.language.hierarchical_template_validator import (
    HierarchicalOccludedTemplateValidator,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.language.scene_construction_router import (
    SceneConstructionRouter,
)


@dataclass
class ControlOverrides:
    """Optional concrete controls.

    ``direction`` is retained only for backward compatibility. It is converted
    to one occluder side before sampling and is never inserted into the semantic
    hierarchy as a pair of alternative directions.
    """

    occluder_type: Optional[str] = None
    occluder_side: Optional[str] = None
    direction: Optional[str] = None
    pedestrian_speed_mps: Optional[float] = None
    risk_level: Optional[str] = None
    seed: Optional[int] = None

    def resolved_side(self) -> Optional[str]:
        side = str(self.occluder_side).lower() if self.occluder_side else None
        if side and side not in {"left", "right"}:
            raise ValueError("occluder_side must be 'left' or 'right'")
        if not self.direction:
            return side

        direction = str(self.direction).lower()
        mapping = {
            "left_to_right": "left",
            "right_to_left": "right",
        }
        if direction not in mapping:
            raise ValueError(
                "direction is a deprecated execution override and must be "
                "'left_to_right' or 'right_to_left'"
            )
        derived = mapping[direction]
        if side and side != derived:
            raise ValueError(
                f"occluder_side={side!r} conflicts with direction={direction!r}"
            )
        return derived


@dataclass
class AdaptationResult:
    prompt: str
    event_frame: Dict[str, Any]
    mapped_eventframe_spec: Dict[str, Any]
    hierarchical_spec: Dict[str, Any]
    template_validation: Dict[str, Any]
    scene_construction: Dict[str, Any]
    sampled_parameters: Dict[str, Any]
    hazard_spec: HazardSemanticSpec
    frame_verification: Dict[str, Any]
    spec_verification: Dict[str, Any]
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "event_frame": self.event_frame,
            "mapped_eventframe_spec": self.mapped_eventframe_spec,
            "hierarchical_spec": self.hierarchical_spec,
            "template_validation": self.template_validation,
            "scene_construction": self.scene_construction,
            "sampled_parameters": self.sampled_parameters,
            "hazard_spec": self.hazard_spec.to_dict(),
            "frame_verification": self.frame_verification,
            "spec_verification": self.spec_verification,
            "provenance": self.provenance,
        }


class OccludedPedestrianEventFrameAdapter:
    """Compile one natural-language prompt through the hierarchical template.

    The adapter fails closed: a prompt must resolve to a nuPlan pedestrian,
    ``occluded_emergence`` and the relative direction
    ``occluder_to_ego_path`` before any B1 construction is attempted.
    """

    def __init__(
        self,
        *,
        llm_provider: str = "none",
        llm_model: str = "qwen2.5:7b",
        ollama_url: str = "http://127.0.0.1:11434",
    ) -> None:
        self.pipeline = HierarchicalEventFramePipeline(
            llm_provider=llm_provider,
            llm_model=llm_model,
            ollama_url=ollama_url,
            allow_fallback=True,
        )
        self.validator = HierarchicalOccludedTemplateValidator()
        self.router = SceneConstructionRouter()
        self.sampler = HierarchicalTemplateSampler()
        self.spec_adapter = HierarchicalHazardSpecAdapter()

    def adapt(
        self,
        prompt: str,
        overrides: Optional[ControlOverrides] = None,
        *,
        case_id: str = "",
        b0_scene_context: Optional[Mapping[str, Any]] = None,
    ) -> AdaptationResult:
        overrides = overrides or ControlOverrides()
        result = self.pipeline.parse_to_result(prompt)
        validation = self.validator.validate(result.spec)
        if not validation.passed:
            raise ValueError(
                "Unsupported or incomplete occluded-pedestrian template: "
                + "; ".join(validation.issues)
            )

        # Route only after the hierarchy has been validated. The routing rule
        # uses the original prompt for explicit global road evidence so inferred
        # hierarchy defaults can never accidentally force full synthesis.
        construction = self.router.route(
            prompt=prompt,
            hierarchical_spec=result.spec,
        )
        hierarchical_spec = deepcopy(result.spec)
        hierarchical_spec["scene_construction"] = construction.to_dict()

        side = overrides.resolved_side()
        sample = self.sampler.sample(
            hierarchical_spec,
            prompt=prompt,
            case_id=case_id,
            overrides=SamplingOverrides(
                occluder_type=overrides.occluder_type,
                occluder_side=side,
                pedestrian_speed_mps=overrides.pedestrian_speed_mps,
                risk_level=overrides.risk_level,
                seed=overrides.seed,
            ),
            construction_plan=construction.to_dict(),
            scene_context=b0_scene_context,
        )
        if not sample.valid:
            raise ValueError(
                "Concrete hierarchical sampling failed: "
                + "; ".join(sample.issues)
            )

        digest = hashlib.sha1(
            (
                f"{case_id}|{prompt}|{construction.mode}|"
                f"{sample.semantic_occluder_type}|{sample.occluder_side}|"
                f"{sample.actor_speed_mps:.3f}|{sample.risk_level}|{sample.seed}"
            ).encode("utf-8")
        ).hexdigest()[:12]
        hazard_spec = self.spec_adapter.adapt(
            prompt=prompt,
            hierarchical_spec=hierarchical_spec,
            sample=sample,
            spec_id=f"occluded_pedestrian_{digest}",
            construction_plan=construction.to_dict(),
        )

        provenance = {
            "construction_mode": {
                "value": construction.mode,
                "source": "scene_construction_router",
                "reason": construction.reason,
                "trigger_evidence": construction.trigger_evidence,
            },
            "primary_actor_type": {
                "value": "pedestrian",
                "source": "hierarchical_nuplan_projection",
            },
            "language_actor_detail": {
                "value": sample.language_actor_detail,
                "source": "hierarchical_language_metadata",
            },
            "semantic_direction": {
                "value": sample.semantic_direction,
                "source": "geometric_constraint",
            },
            "occluder_side": {
                "value": sample.occluder_side,
                "source": sample.provenance.get("occluder_side"),
            },
            "concrete_direction": {
                "value": sample.concrete_direction,
                "source": "derived_from_occluder_side",
            },
            "occluder_type": {
                "value": sample.executable_occluder_type,
                "semantic_value": sample.semantic_occluder_type,
                "source": (
                    "control_override"
                    if overrides.occluder_type
                    else "hierarchical_nuplan_projection"
                ),
            },
            "pedestrian_speed_mps": {
                "value": sample.actor_speed_mps,
                "source": sample.provenance.get("pedestrian_speed_mps"),
            },
            "road_parameters": {
                "source": sample.road_parameter_source,
            },
            "ego_state": {
                "source": sample.ego_state_source,
            },
            "risk_level": {
                "value": sample.risk_level,
                "source": sample.provenance.get("risk_level"),
            },
        }

        frame_verification = {
            "passed": not result.frame_issues,
            "issues": list(result.frame_issues),
        }
        spec_verification = {
            "passed": validation.passed and not result.hierarchy_issues,
            "issues": list(validation.issues) + list(result.hierarchy_issues),
            "warnings": list(validation.warnings),
        }
        return AdaptationResult(
            prompt=prompt,
            event_frame=result.frame.to_dict(),
            # Kept for compatibility with existing artifact readers.
            mapped_eventframe_spec=hierarchical_spec,
            hierarchical_spec=hierarchical_spec,
            template_validation=validation.to_dict(),
            scene_construction=construction.to_dict(),
            sampled_parameters=sample.to_dict(),
            hazard_spec=hazard_spec,
            frame_verification=frame_verification,
            spec_verification=spec_verification,
            provenance=provenance,
        )