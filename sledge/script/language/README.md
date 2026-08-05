# Language experiment scripts

- `compare_language_control_experiments.py` compares EventFrame parsing with the
  direct-template baseline.
- `build_fars_language_eval_dataset.py` builds FARS-derived benchmark cases.
- `run_hierarchical_language_pipeline.py` runs the nuPlan-compatible recursive
  hierarchy and returns both semantic metadata and the SLEDGE projection.

For human-on-foot prompts, the executable output always uses
`TrackedObjectType.PEDESTRIAN` / `SledgeVectorRaw.pedestrians`; words such as
`child` remain language metadata only. Occluded emergence uses the single
relative direction `occluder_to_ego_path`, while the occluder side may be
sampled once for left/right scene diversity.

The files with the same names one directory above are compatibility entry
points. Both old and new command paths are supported.

Example:

```bash
python -m sledge.script.language.run_hierarchical_language_pipeline \
  --prompt "A child emerges from behind a parked truck into the ego lane."
```
