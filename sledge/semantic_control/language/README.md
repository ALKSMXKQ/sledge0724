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

A child value is legal only when it is registered under its selected parent.
For example, `pedestrian -> aggressive_cut_in` is rejected, while
`child_pedestrian -> occluded_emergence -> parked_truck_occluder` is valid.
Missing numeric leaves are conditioned on the full selected parent path and are
stored with `source`, `confidence`, `conditioned_on`, and `is_assumption`.

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
- `hierarchical_ontology.py`: recursive tree, legal transitions, and path resolver.
- `hierarchical_pipeline.py`: default hierarchy-aware orchestration and leaf completion.
- `direct_template_baseline.py`: no-EventFrame comparison baseline.
