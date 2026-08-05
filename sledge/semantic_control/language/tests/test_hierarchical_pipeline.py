"""Tests for recursive parent-constrained language semantics."""

from sledge.semantic_control.language.event_frame import (
    ActorFrame,
    CompletedParameter,
    EgoEventFrame,
    EventFrame,
    MainEventFrame,
    OcclusionFrame,
    RoadContextFrame,
)
from sledge.semantic_control.language.hierarchical_ontology import (
    HierarchicalScenePath,
    HierarchicalSceneResolver,
    HierarchyNode,
)
from sledge.semantic_control.language.hierarchical_pipeline import (
    HierarchicalParameterFiller,
    attach_hierarchy,
    validate_hierarchical_spec,
)


def _occluded_child_frame() -> EventFrame:
    return EventFrame(
        sentence="A child suddenly emerges from behind a parked truck into the ego lane.",
        main_actor=ActorFrame(
            text="child",
            actor_class="human_on_foot",
            actor_role="hazard_actor",
            evidence_text="child",
        ),
        ego_event=EgoEventFrame(ego_maneuver="drive_forward"),
        main_event=MainEventFrame(
            event_type="enter_ego_lane",
            predicate_text="emerges into",
            path_or_object="ego lane",
            event_location_relation="ahead_of",
            location_relation_function="event_anchor",
            source_relation="from_curb",
            target_relation="ego_lane",
            motion_axis="lateral",
            motion_direction="unknown",
            evidence_text="emerges from behind a parked truck into the ego lane",
        ),
        road_context=RoadContextFrame(road_type="straight_lane", lane_context="unknown"),
        occlusion=OcclusionFrame(
            enabled=True,
            occluder_type="truck",
            relation_to_actor="behind",
            evidence_text="behind a parked truck",
        ),
        confidence=0.92,
    )


def _occluded_child_spec():
    return {
        "semantic_slots": {
            "road_topology": "straight_lane",
            "road_layout": "unknown",
            "actor_type": "pedestrian",
            "actor_role": "crossing_actor",
            "motion_geometry": "lateral",
            "fine_grained_conflict_type": "enter_ego_lane",
            "occlusion_enabled": True,
            "occluder_type": "truck",
            "source_side": "curbside",
            "target_path": "ego_lane",
            "anchor_region": "front",
            "risk_level": "aggressive",
            "conflict_direction": "unknown",
            "visibility": "occluded",
        },
        "motion_layer": {
            "hazard_event_type": "enter_ego_lane",
            "motion_axis": "lateral",
        },
        "parameter_layer": {
            "required_missing": [],
            "defaultable_missing": [],
            "distributional_defaults": {},
            "completed": {},
        },
    }


def test_resolver_builds_recursive_occluded_child_path() -> None:
    path = HierarchicalSceneResolver().resolve(_occluded_child_frame(), _occluded_child_spec())

    assert path.valid, path.issues
    assert path.value("road_topology") == "straight_segment"
    assert path.value("ego_traffic_space") == "curbside_zone"
    assert path.value("primary_actor_group") == "vulnerable_road_user"
    assert path.value("primary_actor_type") == "child_pedestrian"
    assert path.value("hazard_interaction") == "occluded_emergence"
    assert path.value("auxiliary_entity") == "parked_truck_occluder"
    assert path.value("target_region") == "ego_lane"
    assert path.value("visibility") == "fully_occluded"
    assert path.value("ego_required_response") == "emergency_brake"

    nested = path.to_nested_tree()
    cursor = nested
    selected_types = []
    while cursor.get("children"):
        cursor = cursor["children"][0]
        selected_types.append(cursor["node_type"])
    assert selected_types[:5] == [
        "road_topology",
        "ego_traffic_space",
        "primary_actor_group",
        "primary_actor_type",
        "hazard_interaction",
    ]


def test_validator_rejects_pedestrian_cut_in_transition() -> None:
    path = HierarchicalScenePath(
        nodes=[
            HierarchyNode(1, "road_topology", "straight_segment"),
            HierarchyNode(2, "ego_traffic_space", "curbside_zone"),
            HierarchyNode(3, "primary_actor_group", "vulnerable_road_user"),
            HierarchyNode(4, "primary_actor_type", "pedestrian"),
            HierarchyNode(5, "hazard_interaction", "aggressive_cut_in"),
            HierarchyNode(6, "auxiliary_entity", "none"),
            HierarchyNode(7, "source_region", "adjacent_left_lane"),
            HierarchyNode(8, "target_region", "ego_lane"),
            HierarchyNode(9, "anchor_region", "lane_boundary"),
            HierarchyNode(10, "visibility", "fully_visible"),
            HierarchyNode(11, "motion_direction", "into_ego_lane"),
            HierarchyNode(12, "trigger_event", "actor_crosses_lane_boundary"),
            HierarchyNode(13, "ego_required_response", "emergency_brake"),
            HierarchyNode(14, "risk_level", "aggressive"),
        ]
    )

    issues = HierarchicalSceneResolver().validate(path)
    assert any("primary_actor_type=pedestrian->hazard_interaction=aggressive_cut_in" in issue for issue in issues)


def test_hierarchical_completion_uses_full_parent_path() -> None:
    frame = _occluded_child_frame()
    spec = _occluded_child_spec()
    path = HierarchicalSceneResolver().resolve(frame, spec)

    completed_spec = HierarchicalParameterFiller().fill(spec, frame, path)
    completed = completed_spec["parameter_layer"]["completed"]

    assert completed["actor_speed_mps"]["value"] == [1.8, 3.2]
    assert completed["actor_speed_mps"]["source"] == "hierarchical_prior"
    assert completed["actor_speed_mps"]["conditioned_on"]["primary_actor_type"] == "child_pedestrian"
    assert completed["reveal_distance_m"]["value"] == [3.0, 8.0]
    assert completed["occluder_type"]["value"] == "truck"
    assert completed["occluder_length_m"]["value"] == [6.0, 12.0]
    assert completed_spec["parameter_layer"]["completion_policy"] == (
        "recursive_parent_path_conditioned_completion"
    )


def test_explicit_parameter_is_never_overwritten_by_hierarchy() -> None:
    frame = _occluded_child_frame()
    frame.completed_parameters["actor_speed_mps"] = CompletedParameter(
        value=4.2,
        unit="m/s",
        source="user_input",
        reason="explicit actor speed",
    )
    spec = _occluded_child_spec()
    path = HierarchicalSceneResolver().resolve(frame, spec)

    completed = HierarchicalParameterFiller().fill(spec, frame, path)["parameter_layer"]["completed"]

    assert completed["actor_speed_mps"]["value"] == 4.2
    assert completed["actor_speed_mps"]["source"] == "user_input"
    assert completed["actor_speed_mps"]["is_assumption"] is False


def test_distributional_source_side_list_is_safe_for_cut_in() -> None:
    frame = EventFrame(
        sentence="A vehicle cuts in aggressively from an adjacent lane with almost no room.",
        main_actor=ActorFrame(text="vehicle", actor_class="vehicle"),
        main_event=MainEventFrame(
            event_type="lane_change_into_ego_lane",
            motion_axis="merging",
            source_relation="from_adjacent_lane",
            target_relation="ego_lane",
            event_location_relation="ahead_of",
        ),
        road_context=RoadContextFrame(road_type="straight_lane", lane_context="adjacent_lane"),
    )
    spec = {
        "semantic_slots": {
            "road_topology": "multi_lane_road",
            "road_layout": "adjacent_lane_cut_in",
            "actor_type": "vehicle",
            "actor_role": "merging_actor",
            "motion_geometry": "merging",
            "fine_grained_conflict_type": "lane_change_into_ego_lane",
            "source_side": ["left", "right"],
            "target_path": "ego_lane",
            "risk_level": "aggressive",
        }
    }

    path = HierarchicalSceneResolver().resolve(frame, spec)

    assert path.valid, path.issues
    assert path.value("hazard_interaction") == "aggressive_cut_in"
    assert path.value("source_region") in {"adjacent_left_lane", "adjacent_right_lane"}


def test_roundabout_event_is_not_collapsed_to_generic_cut_in() -> None:
    frame = EventFrame(
        sentence="Ego enters a roundabout as a circulating vehicle closes the entry gap.",
        main_actor=ActorFrame(text="circulating vehicle", actor_class="vehicle"),
        ego_event=EgoEventFrame(ego_maneuver="enter_roundabout"),
        main_event=MainEventFrame(
            event_type="roundabout_entry_conflict",
            motion_axis="merging",
            source_relation="from_circulating_lane",
            target_relation="roundabout_entry",
            event_location_relation="at_roundabout_entry",
        ),
        road_context=RoadContextFrame(road_type="roundabout", lane_context="roundabout_entry"),
    )
    spec = {
        "semantic_slots": {
            "road_topology": "roundabout",
            "road_layout": "roundabout_entry",
            "actor_type": "vehicle",
            "actor_role": "merging_actor",
            "motion_geometry": "merging",
            "fine_grained_conflict_type": "roundabout_entry_conflict",
            "source_side": "roundabout_inside",
            "target_path": "roundabout_entry",
            "risk_level": "moderate",
        }
    }

    path = HierarchicalSceneResolver().resolve(frame, spec)

    assert path.valid, path.issues
    assert path.value("hazard_interaction") == "roundabout_entry_conflict"
    assert path.value("target_region") == "roundabout_entry"


def test_serialized_hierarchy_keeps_legacy_spec_compatible() -> None:
    frame = _occluded_child_frame()
    base_spec = _occluded_child_spec()
    base_spec["actor_layer"] = {"primary_actor": "pedestrian"}
    hierarchy = HierarchicalSceneResolver().resolve(frame, base_spec)

    output = attach_hierarchy(base_spec, hierarchy)
    ok, issues = validate_hierarchical_spec(output)

    assert ok, issues
    assert output["actor_layer"]["primary_actor"] == "pedestrian"
    assert output["schema_version"] == "eventframe_v5_hierarchical_tree"
    assert output["hierarchy_layer"]["path_values"]["hazard_interaction"] == "occluded_emergence"
