# Phase 2 — B1 Hazard Anchor Audit

## Scientific question

Before any diffusion experiment, quantify whether the existing B0 -> B1 builder actually creates the intended occluded-pedestrian hazard.

The frozen Phase-1 contract is:

\[
H(B1) = O \land E \land C \land T.
\]

- `O`: genuine pre-reveal occlusion.
- `E`: stable reveal.
- `C`: spatiotemporal ego-pedestrian conflict.
- `T`: safety-critical reveal-to-conflict timing.

Phase 2 does **not** change any Phase-1 thresholds.

## Non-negotiable protocol

1. Select and freeze `N=20..50` unique real B0 scenes before observing any B1 outcome. Use `N=50` for the main audit.
2. Run the existing B0 -> B1 pipeline exactly once for each frozen case under the pre-fixed condition/seed.
3. Keep editor errors, adapter errors, and H-failures in the denominator.
4. Never use `target accepted`, rejection sampling, or repeated attempts until a B1 passes when estimating the Phase-2 success probability.
5. Report both:
   - `P(H | B1 built)`: diagnostic quality conditional on successful construction.
   - `P(H(B1) | frozen B0)`: end-to-end hazard-anchor yield. **This is the Phase-2 gate metric.**

The distinction prevents a builder with a low yield from looking strong after failed samples are silently discarded.

## Temporal adapter

Current SLEDGE B1 artifacts are single-frame semantic vectors, while the Phase-1 evaluator needs a visibility sequence and ego/pedestrian trajectories.

`phase2_b1_hazard_audit.py` therefore uses an explicit constant-velocity rollout adapter:

- horizon: `6.0 s`
- step: `0.2 s`
- ego velocity: B1 ego velocity
- pedestrian velocity: B1 speed + heading
- vehicle occluders: their B1 speed + heading
- static occluders: stationary
- each rollout frame calls the unchanged Phase-1 BEV LOS evaluator
- the resulting visibility sequence and trajectories are passed unchanged to `OccludedPedestrianContract.verify()`

This adapter is an explicit Phase-2 approximation, not part of the definition of H. For the paper-quality final audit, replace the adapter with real temporal replay if the original nuPlan trajectory/history is available; the H contract should remain unchanged.

## Run

From repository root on branch `phase2-b1-hazard-audit`:

```bash
export PHASE2=sledge/script/evaluation/phase2_b1_hazard_audit.py
export DATA_ROOT=/ABS/PATH/TO/REAL/SLEDGE/CACHE
export MATRIX=/ABS/PATH/TO/YOUR/EXISTING/OCCLUDED_PEDESTRIAN_MATRIX.json
export PROFILE=<existing-real-data-profile>
export OUT=/ABS/PATH/TO/phase2_b1_audit

# Step 1: freeze 50 unique real B0s before running the editor.
PYTHONPATH=. python "$PHASE2" freeze \
  --input-root "$DATA_ROOT" \
  --matrix-config "$MATRIX" \
  --profile "$PROFILE" \
  --output-manifest "$OUT/frozen_b0_50.jsonl" \
  --n 50

# Step 2: one B0 -> B1 attempt per frozen case, then H(B1).
PYTHONPATH=. python "$PHASE2" run \
  --manifest "$OUT/frozen_b0_50.jsonl" \
  --output-root "$OUT/pipeline" \
  --audit-output "$OUT/phase2_b1_audit.jsonl"
```

For a smoke test only, use `--n 20`. The main result should use 50 unless data availability forces a smaller prespecified N.

## Outputs

- `frozen_b0_50.jsonl`: immutable selected B0 case list.
- `frozen_b0_50.meta.json`: selection protocol metadata.
- `phase2_b1_audit.jsonl`: full per-case diagnostics, including visibility sequence.
- `phase2_b1_audit.csv`: compact table for analysis.
- `phase2_b1_audit.summary.json`: aggregate rates, failure distribution, Wilson interval, and gate decision.

The first paper-facing table is generated from the summary:

| Stage | N | O | E | C | T | H |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 (built only) | ... | ... | ... | ... | ... | ... |
| B0 -> B1 end-to-end | 50 | ... | ... | ... | ... | ... |

Also report the primary failure distribution (`no_pre_reveal_occlusion`, `no_stable_reveal`, `no_spatiotemporal_conflict`, `conflict_not_after_reveal`, `non_critical_timing`, builder/adapter error).

## Gate

Main gate:

\[
P(H(B1) \mid \text{frozen B0}) \ge 0.90.
\]

Preferred result:

\[
P(H(B1) \mid \text{frozen B0}) > 0.95.
\]

- `< 0.90`: **FIX_B1_BUILDER_BEFORE_DIFFUSION**.
- `>= 0.90`: eligible to proceed, but inspect O/E/C/T failure modes before moving on.
- `> 0.95`: preferred builder reliability level.

With `N=50`, also report the 95% Wilson confidence interval; do not present 45/50 or 48/50 as a population-level certainty.

## Unit checks

```bash
PYTHONPATH=. python -m pytest -q \
  sledge/script/evaluation/phase2_b1_hazard_audit_tests.py
```

These checks verify that failed B1 attempts remain in the denominator and that the 90% / >95% gate semantics are implemented as specified.
