from dataclasses import dataclass

import numpy as np

from sledge.semantic_control.occluded_pedestrian_pipeline.generation.template_scene_synthesizer import (
    TemplateSceneSynthesizer,
)


@dataclass
class _Element:
    states: np.ndarray
    mask: np.ndarray


@dataclass
class _Scene:
    lines: _Element
    vehicles: _Element
    pedestrians: _Element
    static_objects: _Element
    green_lights: _Element
    red_lights: _Element
    ego: _Element


def _scaffold() -> _Scene:
    return _Scene(
        lines=_Element(
            states=np.ones((20, 12, 2), dtype=np.float32),
            mask=np.ones((20, 12), dtype=bool),
        ),
        vehicles=_Element(
            states=np.ones((8, 6), dtype=np.float32),
            mask=np.ones((8,), dtype=bool),
        ),
        pedestrians=_Element(
            states=np.ones((5, 6), dtype=np.float32),
            mask=np.ones((5,), dtype=bool),
        ),
        static_objects=_Element(
            states=np.ones((6, 5), dtype=np.float32),
            mask=np.ones((6,), dtype=bool),
        ),
        green_lights=_Element(
            states=np.ones((4, 2), dtype=np.float32),
            mask=np.ones((4,), dtype=bool),
        ),
        red_lights=_Element(
            states=np.ones((4, 2), dtype=np.float32),
            mask=np.ones((4,), dtype=bool),
        ),
        ego=_Element(
            states=np.ones((4,), dtype=np.float32),
            mask=np.ones((1,), dtype=bool),
        ),
    )


def test_synthesis_clears_source_semantics_and_builds_new_base() -> None:
    scene, report = TemplateSceneSynthesizer().synthesize(
        scaffold_scene=_scaffold(),
        hierarchical_spec={"hierarchy_layer": {"path_values": {}}},
        sampled_parameters={
            "road_topology": "intersection",
            "lane_count": 2,
            "lane_width_m": 3.6,
            "road_curvature": 0.0,
            "ego_distance_to_conflict_m": 12.0,
            "ego_speed_mps": 8.0,
            "ego_acceleration_mps2": 0.0,
        },
        construction_plan={
            "mode": "synthesize_new",
            "explicit_global_constraints": {
                "road_topology": "intersection",
            },
        },
    )

    assert report["source_scene_usage"] == "blank_capacity_scaffold_only"
    assert report["source_semantic_content_preserved"] is False
    assert report["generated_polyline_count"] > 0
    assert not np.any(scene.vehicles.mask)
    assert not np.any(scene.pedestrians.mask)
    assert not np.any(scene.static_objects.mask)
    assert np.any(scene.lines.mask)
    assert float(np.asarray(scene.ego.states).reshape(-1)[0]) == 8.0