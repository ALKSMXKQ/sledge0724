# Language evaluation data

This directory contains inputs for language-understanding experiments only. It
must not contain generated scene caches or diffusion outputs.

- Small `*_cases.jsonl` files are hand-authored robustness and control suites.
- `fars2024_real_hazard_cases*.jsonl` are generated benchmark splits.
- `*_summary.json` records source/split metadata.
- Raw downloaded archives, when needed, belong under `raw/` and should remain
  ignored by Git.

Each case should keep its prompt, expected slots, category, group, and stable ID
together so train/test leakage can be audited.

