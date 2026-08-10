# Dual-mode scene construction routing

This branch separates language understanding from B1 scene construction.

```text
Natural language
    -> hierarchical EventFrame pipeline
    -> provenance-aware SceneConstructionRouter
        -> edit_existing
        -> synthesize_new
    -> B1
```

## Routing rule

`edit_existing` is the default. It preserves the B0 road/map and applies only
local hazard edits.

Local constraints do **not** trigger synthesis, including:

- `ego_lane` / `ego_path`
- left/right roadside or curbside source regions
- pedestrian/child/jogger actor semantics
- occluder type and occluded emergence
- actor speed
- risk severity
- pedestrian crossing/entering a lane
- an actor merging into the ego lane

`synthesize_new` is selected only when the user explicitly specifies global road
structure, for example:

- lane count (`two-lane`, `three-lane`, ...)
- bidirectional/one-way road directionality
- intersection or roundabout topology
- highway merge ramp / diverge / lane drop
- work-zone or closed-lane topology
- dedicated turn-lane layout
- explicit lane width
- explicit straight/curved road geometry

Hierarchy values are never sufficient by themselves. The router requires an
explicit provenance source. `hierarchical_default`, `hierarchical_prior`, and
`inferred` values do not trigger synthesis.

Because the current EventFrame schema does not expose every road property (most
notably `lane_count`), the router contains a narrow global-road provenance bridge.
It extracts only global road structure from the original prompt and records the
source as `prompt_explicit`. It does not parse local hazard semantics.

## Output spec

The routed language pipeline adds:

```json
{
  "scene_construction": {
    "mode": "edit_existing",
    "reason": "no_explicit_global_road_structure",
    "explicit_global_constraints": [],
    "local_hazard_constraints": [
      "hazard_interaction=occluded_emergence",
      "source_region=right_side",
      "target_region=ego_path"
    ],
    "routing_evidence": [],
    "inherits_b0_road": true
  }
}
```

For an explicit two-lane bidirectional road, the same field becomes
`synthesize_new` and records `lane_count=2` and
`road_directionality=bidirectional` as explicit global constraints.

## Parameter execution policy

The original hierarchy parameter template is preserved in both modes.

In `edit_existing` mode:

- road parameters such as `lane_width_m`, `lane_count`, and `road_curvature` stay
  in `parameter_layer.completed` but are listed in
  `parameter_layer.execution_policy.inactive_parameter_names`;
- `road_geometry_source` is `inherit_b0`;
- local hazard parameters stay active.

In `synthesize_new` mode:

- the road parameters become active;
- `road_geometry_source` is `parameter_template`.

This avoids maintaining two language templates.

## Language CLI

The existing hierarchy CLI now uses `RoutedHierarchicalEventFramePipeline`:

```bash
python -m sledge.script.language.run_hierarchical_language_pipeline \
  --prompt "A child emerges from behind a parked truck into the ego lane."
```

Expected mode: `edit_existing`.

```bash
python -m sledge.script.language.run_hierarchical_language_pipeline \
  --prompt "On a two-lane bidirectional road, a child emerges from behind a parked truck."
```

Expected mode: `synthesize_new`.

## B1 dispatcher

Downstream code must not inspect the prompt again. Use
`dispatch_scene_construction` and provide the two construction backends:

```python
result = dispatch_scene_construction(
    spec,
    b0_scene=b0,
    edit_existing=edit_b0,
    synthesize_new=synthesize_from_template,
)
```

`edit_existing` receives B0 plus the routed spec. `synthesize_new` receives only
the routed spec, so it cannot accidentally inherit B0 road geometry.

The repository currently has a mature compositional B0 editor but does not yet
contain a complete full-road-from-template synthesis backend. The dispatcher
therefore defines the correct interface without pretending that road synthesis
already exists.

## Validation

Run the focused tests on the server:

```bash
python -m pytest \
  sledge/semantic_control/language/tests/test_scene_construction_router.py \
  sledge/semantic_control/occluded_pedestrian_pipeline/tests/test_scene_construction_dispatcher.py \
  -vv
```

Then rerun the existing hierarchy regression tests:

```bash
python -m pytest \
  sledge/semantic_control/language/tests/test_hierarchical_pipeline.py \
  sledge/semantic_control/occluded_pedestrian_pipeline/tests/test_eventframe_adapter.py \
  -vv
```

## Experiment reporting

Persist `scene_construction.mode` into the B1/B2 manifest. Report semantic
retention separately for:

- edited B1 (`edit_existing`)
- synthesized B1 (`synthesize_new`)
- overall retention

The primary diffusion-retention experiment should continue to use edited B1;
full synthesis is best treated as an extension experiment because it changes the
entire scene distribution rather than only the local hazard semantics.
