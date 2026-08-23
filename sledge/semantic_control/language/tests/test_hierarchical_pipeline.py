"""Tests for the nuPlan-compatible recursive language hierarchy."""

from sledge.semantic_control.language.event_frame import (
    ActorFrame,
    CompletedParameter,
    EgoEventFrame,
    EventFrame,
    MainEventFrame,
    MissingInformationFrame,
    OcclusionFrame,
    RoadContextFrame,
)
from sledge.semantic_control.language.hierarchical_ontology import (
    HierarchicalScenePath,
    HierarchicalSceneResolver,
    HierarchyNode,
)
from sledge.semantic_control.language.hierarchical_pipeline import (
    HierarchicalEventFramePipeline,
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
        road_context=RoadContextFrame(
            road_type="straight_lane",
            lane_context="unknown",
            evidence_text="",
        ),
        occlusion=OcclusionFrame(
            enabled=True,
            occluder_type="truck",
            relation_to_actor="behind",
            evidence_text="behind a parked truck",
        ),
        missing_information=MissingInformationFrame(
            required=["actor_speed", "ego_speed", "initial_distance"]
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
        "actor_layer": {
            "primary_actor": "pedestrian",
            "base_actor_type": "pedestrian",
        },
        "motion_layer": {
            "hazard_event_type": "enter_ego_lane",
            "motion_axis": "lateral",
        },
        "event_layer": {
            "event_sequence": [],
            "event_sequence_labels": [],
            "num_events": 1,
        },
        "parameter_layer": {
            "required_missing": ["actor_speed", "ego_speed", "initial_distance"],
            "defaultable_missing": [],
            "distributional_defaults": {},
            "completed": {},
        },
    }


def test_human_subtype_is_metadata_but_nuplan_actor_is_pedestrian() -> None:
    path = HierarchicalSceneResolver().resolve(_occluded_child_frame(), _occluded_child_spec())

    assert path.valid, path.issues
    assert path.value("primary_actor_type") == "pedestrian"
    assert path.attributes["language_actor_detail"] == "child"

    projection = path.nuplan_projection()
    assert projection["actor_category"] == "pedestrian"
    assert projection["tracked_object_type"] == "TrackedObjectType.PEDESTRIAN"
    assert projection["sledge_collection"] == "pedestrians"


def test_occluded_emergence_has_one_relative_direction() -> None:
    path = HierarchicalSceneResolver().resolve(_occluded_child_frame(), _occluded_child_spec())

    assert path.value("motion_direction") == "occluder_to_ego_path"
    motion_node = path.node("motion_direction")
    assert motion_node is not None
    assert motion_node.allowed_values_at_level == ("occluder_to_ego_path",)
    assert motion_node.source == "geometric_constraint"


def test_allowed_children_are_real_next_level_values() -> None:
    path = HierarchicalSceneResolver().resolve(_occluded_child_frame(), _occluded_child_spec())

    road = path.node("road_topology")
    actor_group = path.node("primary_actor_group")
    actor_type = path.node("primary_actor_type")
    assert road is not None and actor_group is not None and actor_type is not None

    assert "curbside_zone" in road.allowed_children
    assert "straight_segment" not in road.allowed_children
    assert "pedestrian" in actor_group.allowed_children
    assert "vulnerable_road_user" not in actor_group.allowed_children
    assert "occluded_emergence" in actor_type.allowed_children


def test_hierarchical_completion_is_nuplan_compatible_and_complete() -> None:
    frame = _occluded_child_frame()
    spec = _occluded_child_spec()
    path = HierarchicalSceneResolver().resolve(frame, spec)

    completed_spec = HierarchicalParameterFiller().fill(spec, frame, path)
    completed = completed_spec["parameter_layer"]["completed"]

    assert completed["actor_speed_mps"]["value"] == [1.0, 2.0]
    assert completed["crossing_direction"]["value"] == "occluder_to_ego_path"
    assert completed["crossing_direction"]["alternatives"] == []
    assert completed["actor_heading"]["source"] == "derived_constraint"
    assert completed["occlusion_enabled"]["source"] == "user_input"
    assert completed["occlusion_enabled"]["is_assumption"] is False
    assert completed["occluder_type"]["value"] == "vehicle"
    assert completed["occluder_type"]["source"] == "user_input"
    assert completed["occluder_type"]["is_assumption"] is False

    parameter_layer = completed_spec["parameter_layer"]
    assert parameter_layer["parameter_template_complete"] is True
    assert parameter_layer["template_missing_parameters"] == []
    assert parameter_layer["required_missing"] == []
    assert {item["resolved_as"] for item in parameter_layer["resolved_missing"]} == {
        "actor_speed_mps",
        "ego_speed_mps",
        "ego_distance_to_conflict_m",
    }
    assert completed_spec["readiness"]["scene_template_ready"] is True
    assert completed_spec["readiness"]["sampled_scene_ready"] is False


def test_occluder_side_can_vary_without_creating_two_motion_directions() -> None:
    frame = _occluded_child_frame()
    path = HierarchicalSceneResolver().resolve(frame, _occluded_child_spec())
    output = HierarchicalParameterFiller().fill(_occluded_child_spec(), frame, path)
    completed = output["parameter_layer"]["completed"]

    side = completed["occluder_side"]["value"]
    assert side["distribution"] == "categorical"
    assert side["values"] == ["left", "right"]
    assert side["sample_once"] is True
    assert completed["crossing_direction"]["value"] == "occluder_to_ego_path"


def test_event_sequence_separates_hidden_reveal_and_lane_entry() -> None:
    frame = _occluded_child_frame()
    path = HierarchicalSceneResolver().resolve(frame, _occluded_child_spec())
    spec = attach_hierarchy(_occluded_child_spec(), path)

    output = HierarchicalEventFramePipeline._refine_event_sequence(spec, frame, path)
    event_types = [step["event_type"] for step in output["event_layer"]["event_sequence"]]

    assert event_types.index("actor_occluded") < event_types.index("occluded_actor_becomes_visible")
    assert event_types.index("occluded_actor_becomes_visible") < event_types.index("enter_ego_lane")
    assert output["event_layer"]["sequence_policy"] == (
        "occlusion_reveal_lane_entry_are_distinct_events"
    )


def test_anchor_is_not_marked_explicit_without_near_front_words() -> None:
    path = HierarchicalSceneResolver().resolve(_occluded_child_frame(), _occluded_child_spec())
    anchor = path.node("anchor_region")
    assert anchor is not None
    assert anchor.value == "near_front"
    assert anchor.source in {"inferred", "hierarchical_default"}


def test_serialized_hierarchy_validates_nuplan_projection() -> None:
    frame = _occluded_child_frame()
    spec = _occluded_child_spec()
    path = HierarchicalSceneResolver().resolve(frame, spec)
    output = attach_hierarchy(spec, path)

    ok, issues = validate_hierarchical_spec(output)
    assert ok, issues
    assert output["schema_version"] == "eventframe_v6_nuplan_hierarchical_tree"
    assert output["hierarchy_layer"]["nuplan_projection"]["sledge_collection"] == "pedestrians"
    assert output["hierarchy_layer"]["tree_kind"] == "selected_root_to_leaf_path"


def test_explicit_numeric_parameter_is_not_overwritten() -> None:
    frame = _occluded_child_frame()
    frame.completed_parameters["ego_speed"] = CompletedParameter(
        value=8.5,
        unit="m/s",
        source="user_input",
        reason="explicit speed in prompt",
    )
    spec = _occluded_child_spec()
    path = HierarchicalSceneResolver().resolve(frame, spec)

    output = HierarchicalParameterFiller().fill(spec, frame, path)
    ego_speed = output["parameter_layer"]["completed"]["ego_speed_mps"]
    assert ego_speed["value"] == 8.5
    assert ego_speed["source"] == "user_input"
    assert ego_speed["is_assumption"] is False


def test_validator_rejects_pedestrian_cut_in_transition() -> None:
    path = HierarchicalScenePath(
        nodes=[
            HierarchyNode(
                1,
                "road_topology",
                "straight_segment",
                allowed_values_at_level=("straight_segment",),
                next_node_type="ego_traffic_space",
                allowed_children=("curbside_zone",),
            ),
            HierarchyNode(
                2,
                "ego_traffic_space",
                "curbside_zone",
                allowed_values_at_level=("curbside_zone",),
                next_node_type="primary_actor_group",
                allowed_children=("vulnerable_road_user",),
            ),
            HierarchyNode(
                3,
                "primary_actor_group",
                "vulnerable_road_user",
                allowed_values_at_level=("vulnerable_road_user",),
                next_node_type="primary_actor_type",
                allowed_children=("pedestrian",),
            ),
            HierarchyNode(
                4,
                "primary_actor_type",
                "pedestrian",
                allowed_values_at_level=("pedestrian",),
                next_node_type="hazard_interaction",
                allowed_children=("path_crossing", "occluded_emergence"),
            ),
            HierarchyNode(
                5,
                "hazard_interaction",
                "aggressive_cut_in",
                allowed_values_at_level=("path_crossing", "occluded_emergence"),
            ),
        ]
    )

    issues = HierarchicalSceneResolver().validate(path)
    assert any("invalid_value_at_level:hazard_interaction=aggressive_cut_in" in issue for issue in issues)