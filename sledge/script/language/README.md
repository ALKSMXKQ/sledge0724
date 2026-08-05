# Language experiment scripts

- `compare_language_control_experiments.py` compares EventFrame parsing with the
  direct-template baseline.
- `build_fars_language_eval_dataset.py` builds FARS-derived benchmark cases.
- `run_hierarchical_language_pipeline.py` runs the recursive parent-constrained
  language tree for one prompt or a JSONL dataset.

The files with the same names one directory above are compatibility entry
points. Both old and new command paths are supported.

Example:

```bash
python -m sledge.script.language.run_hierarchical_language_pipeline \
  --prompt "A child emerges from behind a parked truck into the ego lane."
```
