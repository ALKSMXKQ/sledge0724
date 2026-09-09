from sledge.semantic_control.occluded_pedestrian_pipeline.generation.topology_adaptive_projection import (
    LegacyTopologyAdaptiveHazardProjector,
    PathRelativeTopologyAdaptiveHazardProjector,
    TopologyAdaptiveHazardProjector,
)


def test_default_topology_projector_is_path_relative_v2() -> None:
    assert TopologyAdaptiveHazardProjector is PathRelativeTopologyAdaptiveHazardProjector
    assert TopologyAdaptiveHazardProjector is not LegacyTopologyAdaptiveHazardProjector


def test_legacy_projector_remains_available_for_ablation() -> None:
    legacy = LegacyTopologyAdaptiveHazardProjector(projection_time_s=2.1)
    active = TopologyAdaptiveHazardProjector(projection_time_s=2.1)
    assert type(legacy).__name__ == "TopologyAdaptiveHazardProjector"
    assert type(active).__name__ == "PathRelativeTopologyAdaptiveHazardProjector"
