from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVectorRaw
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_spec import HazardSemanticSpec
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.primitive_compiler import compile_spec_to_ops
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.primitive_executor import PrimitiveExecutor
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.edit_models import SceneEditResult
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.spec_checker import check_spec
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.spec_presets import normalize_spec


class CompositionalSemanticSceneEditor:
    """
    Primitive-based compositional semantic editor.

    It does not route by canonical_type. It normalizes and checks the layered
    HazardSemanticSpec, compiles it into primitive ops, then executes them.
    """

    def __init__(self, strict_check: bool = False) -> None:
        self.strict_check = strict_check
        self.executor = PrimitiveExecutor()

    def edit(self, scene: SledgeVectorRaw, spec: HazardSemanticSpec) -> Tuple[SledgeVectorRaw, SceneEditResult, Dict[str, Any]]:
        normalized_spec = normalize_spec(spec)
        check_report = check_spec(normalized_spec, strict=self.strict_check)
        if not check_report.valid:
            raise ValueError(f"Invalid HazardSemanticSpec: {check_report.errors}")
        primitive_ops = compile_spec_to_ops(normalized_spec)
        working_scene = copy.deepcopy(scene)
        edited_scene, edit_result, report = self.executor.execute(working_scene, normalized_spec, primitive_ops)
        report["spec_check"] = check_report.to_dict()
        return edited_scene, edit_result, report

