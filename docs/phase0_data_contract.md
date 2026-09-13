# Phase 0 — Representation/Data Audit Contract

## What data Phase 0 uses

Phase 0 is **not** a generation experiment and does not require B2. Its main input is a
small, representative set of **existing real scenes (B0)** together with the *same scenes*
as they appear at the important interfaces of the existing SLEDGE pipeline:

1. dataset / feature-builder output;
2. actor/lane representation consumed by the RVAE;
3. RVAE input tensor and decoded output shape/semantics;
4. diffusion scheduler latent/noise interface (shape, dtype, scaling only in Phase 0);
5. simulation rollout state/trajectory representation.

If an existing B1 editor already exists, a few B1 samples can be audited as an additional
sanity check, especially to make sure pedestrian fields use the same semantics. B1 is not
required to pass Gate 0.

## Recommended audit sample

Use ~10–20 real scenes, not thousands. Cover at least:
- straight road, curve, intersection;
- scenes with vehicles;
- scenes with pedestrians if available;
- scenes with static/parked objects if available;
- at least one scene with lane successors/predecessors.

The goal is schema coverage, not statistical estimation.

## What must be written down

For every representation interface, record:
- Python class / dict key / tensor name;
- exact shape and dtype;
- coordinate frame (world vs ego-local);
- units (m, m/s, radians or degrees);
- actor row/slot ordering;
- padding and validity convention;
- actor-type coding;
- exact x/y/heading/velocity/length/width fields;
- lane centerline representation;
- lane successor/predecessor representation;
- static-object representation;
- time axis and sampling interval;
- whether a field is full extent or half extent;
- normalization/scaling before RVAE;
- latent shape and scheduler scaling.

## Gate 0

You pass Gate 0 only after an adapter can execute:

`legacy scene -> Canonical SceneState -> legacy scene`

without changing key mapped actor fields beyond numerical tolerance, and you can answer
where pedestrian x/y/heading/speed/size and lane topology are stored.

## How to use the included code

### 1. Print structures from inside existing code

```python
import json
from sledge.hazard_equivalence.audit import structure_report

report = structure_report(feature_or_scene_object)
print(json.dumps(report, indent=2, ensure_ascii=False))
```

Run this at the dataset/feature, RVAE, scheduler, and simulation interfaces.

### 2. After you have verified the real actor column schema

Create a JSON file such as `phase0_schema.json` **using your verified repository fields**:

```json
{
  "frame": "ego_local",
  "columns": {
    "x": 0,
    "y": 1,
    "heading": 2,
    "length": 3,
    "width": 4,
    "actor_type": 7,
    "track_id": 8,
    "valid": 9,
    "vx": 5,
    "vy": 6
  },
  "actor_type_map": {
    "0": "vehicle",
    "1": "pedestrian",
    "2": "static",
    "3": "ego"
  }
}
```

The numbers above are an **example format only**. Do not copy them into the project unless
the repository proves they are correct.

### 3. Run round-trip audit

```bash
python -m sledge.hazard_equivalence.scripts.phase0_audit \
  --npz /path/to/one_scene.npz \
  --actors-key actors \
  --schema /path/to/phase0_schema.json \
  --scene-id example_001 \
  --out /tmp/phase0_report.json
```

Exit code 0 means the canonical round trip passed. Exit code 2 means Gate 0 failed.
