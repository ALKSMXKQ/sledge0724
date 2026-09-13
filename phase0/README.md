# Phase 0 — Representation Audit

`phase0/` is a standalone research-stage workspace. It does **not** modify the existing `sledge/` package.

## Scientific goal

Before implementing LOS / reveal / conflict / RTTC or any new diffusion method, establish exactly what the current code and data represent.

Gate 0 is passed only when we can trust the mapping

`real SLEDGE representation -> canonical representation -> real SLEDGE representation`

and know which information is preserved or irreversibly lost.

## Current structure

```text
phase0/
├── README.md
├── run_tests.sh
├── run_source_audit.sh
├── src/
│   └── phase0_audit/
│       ├── __init__.py
│       ├── canonical.py
│       ├── audit.py
│       ├── adapter.py              # generic diagnostic adapter (legacy helper)
│       ├── sledge_adapter.py       # adapter for the real processed SledgeVector layout
│       ├── sledge_contract.py      # source-verified constants + runtime assertions
│       └── source_audit.py         # CLI source/runtime contract verifier
├── tests/
│   ├── test_phase0.py
│   └── test_sledge_adapter.py
├── docs/
│   ├── data_contract.md
│   └── source_audit.md
└── configs/
    └── schema.example.json         # generic format example only; not the real SLEDGE schema
```

## Step 0A — unit tests

From repository root:

```bash
bash phase0/run_tests.sh
```

These tests are standalone and check canonical-state logic plus the real six-dimensional SLEDGE actor layout.

## Step 0B — verify assumptions against the checked-out SLEDGE installation

Run this in the same Python/conda environment that you normally use for SLEDGE:

```bash
bash phase0/run_source_audit.sh
```

It writes:

```text
phase0/outputs/source_audit.json
```

A non-zero exit means the checked-out repository/environment disagrees with an audited assumption and Gate 0 must stop there.

## What source audit has already established

See `docs/source_audit.md`. Key findings include:

- vehicle/pedestrian processed state is `[x, y, heading, width, length, scalar_speed]`;
- poses are converted to ego-local coordinates by the feature builder;
- original actor track IDs are not stored in processed `SledgeVector` actor states;
- processed line vectors preserve sampled geometry but not lane IDs/successor topology;
- default RVAE input is 12 x 256 x 256 and default latent is 64 x 8 x 8;
- the diffusion dataset explicitly trains on cached RVAE `mu` and no extra latent scaling was observed in the loader/training path;
- simulation observations retain object-level identity through nuPlan `DetectionsTracks`/`TrackedObjects`.

## Gate 0 is not closed yet

We still need runtime evidence from representative real B0 scenes and at least one simulation rollout. Do not start Phase 1 until these checks are complete.
