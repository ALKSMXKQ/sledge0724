# Occluded-pedestrian three-stage `.gz` pipeline

This document describes the end-to-end experiment for converting ordinary
source scenes into **occluder-hidden pedestrian dart-out hazards**, observing
what the RVAE and diffusion model preserve, and producing simulator-readable
SLEDGE gzip caches at every checkpoint.

## Scientific contract

The experiment deliberately separates **measurement** from **guarantee**.

1. **B1 edited scene**
   - The existing hierarchical parameter/template editor constructs the hazard.
   - The canonical strict semantic validator must accept the scene.
   - The accepted raw edit is also exported as a typed `sledge_vector.gz` and
     opened with `SledgeScenario`.
2. **RVAE raw reconstruction**
   - B1 raster -> RVAE encoder -> deterministic latent `mu` -> RVAE decoder.
   - No semantic slot restoration is applied.
   - This gzip measures whether the autoencoder bottleneck itself retains the
     occluder / pedestrian / crossing geometry.
3. **RVAE semantic-protected reconstruction**
   - Starts from the same raw RVAE reconstruction.
   - Restores only the B1 road/ego contract and controlled pedestrian/occluder
     slots, using the same slot-resolution logic as protected diffusion.
   - The canonical occluded-pedestrian metrics must pass before the official
     gzip is admitted.
4. **Raw diffusion baseline** (optional but default)
   - One unprotected diffusion generation.
   - No semantic compositing and no semantic selection are allowed.
   - It is retained even when semantics fail so model failure modes remain
     measurable.
5. **Semantic-protected diffusion**
   - Runs multiple low-noise candidates.
   - Protects the controlled semantic slots after decoding.
   - Applies the canonical semantic acceptance gate.
   - Only accepted outputs become official B2 gzip scenes.

The protected path is therefore an **output contract**, not a claim that the
unconditional learned model always preserves the hazard by itself. Compare the
raw checkpoints to the protected checkpoints when reporting model retention.

## Recommended parameter matrix

Use:

```text
sledge/semantic_control/occluded_pedestrian_pipeline/configs/semantic_retention_matrix.json
```

The matrix varies:

- occluder type: vehicle, bicycle, generic object, traffic cone, barrier,
  construction-zone sign;
- occluder side: left / right;
- pedestrian speed;
- source scenario type through deterministic stratified source sampling;
- risk severity through the `retention_mild`, `retention_moderate`, and
  `retention_aggressive` profiles.

`retention_smoke` is a small first-run profile.

## One-command run

Run from the repository root / active `sledge` environment:

```bash
python -m sledge.script.run_occluded_pedestrian_three_stage \
  --input-root /path/to/exp/caches/autoencoder_cache \
  --run-root /path/to/exp/occluded_pedestrian_three_stage \
  --matrix-config sledge/semantic_control/occluded_pedestrian_pipeline/configs/semantic_retention_matrix.json \
  --profiles retention_smoke \
  --config /path/to/semantic_img2img_cfg.yaml \
  --autoencoder-checkpoint /path/to/rvae.ckpt \
  --diffusion-checkpoint /path/to/diffusion_checkpoint \
  --device cuda
```

After the smoke run succeeds, run the balanced three-risk experiment:

```bash
python -m sledge.script.run_occluded_pedestrian_three_stage \
  --input-root /path/to/exp/caches/autoencoder_cache \
  --run-root /path/to/exp/occluded_pedestrian_semantic_retention \
  --profiles retention_mild,retention_moderate,retention_aggressive \
  --config /path/to/semantic_img2img_cfg.yaml \
  --autoencoder-checkpoint /path/to/rvae.ckpt \
  --diffusion-checkpoint /path/to/diffusion_checkpoint \
  --device cuda
```

Use `--diffusion-run protected` when you only need the guaranteed output and do
not need the raw scientific diffusion baseline. The default is `both`.

## `.gz` products

For each accepted B1 sample the run writes:

### 1. Modified scene for simulation

```text
<run_root>/b1_simulation_cache/log/sudden_pedestrian_crossing/<sample_id>/sledge_vector.gz
```

This is the processed form of the parameter-edited scene, with the requested
nuPlan occluder type embedded in the gzip.

### 2. Actual RVAE reconstruction

```text
<run_root>/rvae_reconstruction/raw_cache/log/sudden_pedestrian_crossing/<sample_id>/sledge_vector.gz
```

Use this to visually inspect whether the encoder/decoder itself lost the hidden
pedestrian, occluder, direction, or dangerous timing geometry.

### 3. Semantic-protected RVAE reconstruction

```text
<run_root>/rvae_reconstruction/semantic_protected_cache/log/sudden_pedestrian_crossing/<sample_id>/sledge_vector.gz
```

This is the official RVAE-stage scene when downstream processing requires the
hazard semantics to remain valid.

### 4. Raw diffusion baseline

```text
<run_root>/b2_diffusion/raw_diffusion_baseline/generated_cache/**/sledge_vector.gz
```

This is diagnostic. It may fail the hazard semantics by design.

### 5. Semantic-protected diffusion result

```text
<run_root>/b2_diffusion/semantic_protected/generated_cache/**/sledge_vector.gz
```

This is the official diffusion result. The pipeline does not admit it to the
final contract unless the canonical hazard metrics pass and `SledgeScenario`
can round-trip the gzip.

Because the diffusion runner's cache token is not always identical to
`sample_id`, use the final manifest below rather than guessing B2 paths.

## The file to use for visual inspection

The most useful index is:

```text
<run_root>/manifests/three_stage_gz_contract.json
```

Each row contains exact paths for:

- `B1_edited_gz`
- `RVAE_raw_gz`
- `RVAE_semantic_protected_gz`
- `B2_raw_diffusion_gz`
- `B2_semantic_protected_gz`

and hard pass/fail checks for the three official simulator stages.

Other useful reports are:

```text
<run_root>/manifests/b1_summary.json
<run_root>/manifests/b1_simulation_cache_summary.json
<run_root>/manifests/rvae_reconstruction_summary.json
<run_root>/manifests/b2_raw_diffusion_baseline_results.jsonl
<run_root>/manifests/b2_semantic_protected_results.jsonl
<run_root>/manifests/diffusion_semantic_retention_comparison.json
<run_root>/manifests/three_stage_run_summary.json
```

## What is actually guaranteed

For an **official protected result**, all of the following must hold:

- a pedestrian exists;
- an occluder exists;
- the occluder is between ego and pedestrian;
- ego-to-pedestrian line of sight is blocked;
- the occluder stays clear of the ego corridor;
- the pedestrian moves from the requested side toward the ego path;
- pedestrian speed matches the parameter template tolerance;
- the pedestrian reaches the ego lane;
- pedestrian/ego conflict timing matches the requested risk window;
- pedestrian and occluder do not overlap initially;
- the occluder satisfies the stationary-occluder contract;
- the gzip can be loaded by `SledgeScenario`;
- the requested nuPlan-visible occluder type survives gzip round-trip.

If any protected RVAE or protected diffusion scene fails these checks, the
three-stage command exits with an error and records the failing `sample_id`
instead of silently treating that scene as valid.
