# EventFrame language understanding

This package owns natural-language decomposition and semantic representation.
It must not import scene editors, diffusion code, or simulation builders.

## Default hierarchical path

New integrations should use `HierarchicalEventFramePipeline` (also exported as
`DefaultLanguageUnderstandingPipeline`). It keeps the existing EventFrame and
legacy spec layers, then selects one recursive, parent-constrained path:

```text
road_topology
-> ego_traffic_space
-> primary_actor_group
-> primary_actor_type
-> hazard_interaction
-> auxiliary_entity
-> source_region
-> target_region
-> anchor_region
-> visibility
-> motion_direction
-> trigger_event
-> ego_required_response
-> risk_level
-> executable parameters
```

## nuPlan/SLEDGE category projection

The hierarchy separates linguistic descriptions from executable actor classes.
All human-on-foot descriptions (`child`, `adult`, `jogger`, wheelchair user,
etc.) are normalized to:

```text
primary_actor_type = pedestrian
tracked_object_type = TrackedObjectType.PEDESTRIAN
sledge_collection = pedestrians
```

The original wording remains available as
`hierarchy_layer.attributes.language_actor_detail`; it does not create a new
nuPlan or SLEDGE actor category.

For an occluded pedestrian emergence, the direction is represented once in a
relative geometric frame:

```text
motion_direction = occluder_to_ego_path
```

When the prompt does not state the roadside, `occluder_side` may sample left or
right once for scene diversity. After that placement is sampled, pedestrian
heading is deterministically derived from the occluder position toward the ego
path. The output never creates two alternative pedestrian motion directions for
one scene.

## Tree and parameter semantics

`allowed_values_at_level` contains legal sibling values for the current node.
`allowed_children` contains the actual legal values for the next node type.
`hierarchy_layer.tree` is explicitly a selected root-to-leaf path, not the full
ontology.

The parameter layer now:

- completes every declared executable parameter group;
- distinguishes explicit values, priors, and derived constraints;
- keeps occlusion and occluder type as non-assumptions when stated in the text;
- separates hidden, reveal, lane-entry, and conflict events;
- computes TTC from sampled states instead of independently sampling it;
- records hard visibility, placement, path-intersection, and direction constraints;
- reports `scene_template_ready` separately from `sampled_scene_ready`.

A complete parameter template is not yet one concrete scene. Downstream code
must sample one value for each numeric/categorical range, resolve derived
references, and evaluate every hard constraint. Only then should
`sampled_scene_ready` become true and the concrete states be written into
`SledgeVectorRaw`.

Single prompt example:

```bash
python -m sledge.script.language.run_hierarchical_language_pipeline \
  --prompt "A child suddenly emerges from behind a parked truck into the ego lane."
```

JSONL example:

```bash
python -m sledge.script.language.run_hierarchical_language_pipeline \
  --input-jsonl cases.jsonl \
  --output hierarchical_results.jsonl
```

## Main modules

- `narrative_semantics.py`: clauses, candidate events, and hazard focus.
- `event_frame.py`: serializable intermediate representation.
- `event_frame_parser.py`: LLM/fallback parsing into EventFrame.
- `event_sequence_builder.py`: temporal event reconstruction.
- `event_frame_verifier.py`: consistency checks and deterministic repair.
- `event_frame_mapper.py`: compositional hazard specification.
- `missing_info_filler.py`: legacy semantic-slot default completion.
- `hierarchical_ontology.py`: recursive tree, nuPlan projection, legal transitions, and path resolver.
- `hierarchical_pipeline.py`: hierarchy-aware orchestration, event refinement, parameter completion, and constraints.
- `direct_template_baseline.py`: no-EventFrame comparison baseline.
