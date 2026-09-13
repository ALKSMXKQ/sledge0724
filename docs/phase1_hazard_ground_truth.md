# Phase 1 — Hazard Ground Truth

## Goal

Phase 1 establishes a deterministic, auditable evaluator for an occluded-pedestrian hazard before any generation experiment is allowed.

The contract is

\[
H = O \land E \land C \land T.
\]

- `O`: a pedestrian is genuinely occluded in ego line of sight before reveal.
- `E`: a stable visibility reveal occurs.
- `C`: ego and pedestrian have a spatial conflict region and temporally close occupancy of that region.
- `T`: reveal-to-conflict time is safety-critical.

All thresholds are explicit configuration values, not hidden constants in experiment code.

## 1. LOS visibility

`compute_visibility(ego, pedestrian, occluders)` uses the canonical `ActorState` representation and a 2-D BEV geometric line-of-sight test.

The pedestrian footprint is sampled on a regular grid. For each target sample, an occluder blocks visibility only when it intersects the finite ego-to-target segment before the pedestrian sample. An object behind the pedestrian therefore cannot be counted as an occluder.

Outputs:

- `visibility_fraction`
- `occluding_object`
- `occlusion_type`: `visible`, `partially_occluded`, or `fully_occluded`

This is deliberately a deterministic BEV ground-truth baseline. It is not a camera-image visibility estimator.

## 2. Reveal

For visibility sequence `V(t)`, the reveal detector returns the earliest index satisfying

\[
V(t^- ) < \tau,
\]

followed by

\[
V(t),\ldots,V(t+K-1) > \tau.
\]

The inequalities are strict. A one-frame visibility flicker does not constitute a reveal.

Default configuration:

- visibility threshold `tau = 0.5`
- consecutive visible frames `K = 3`
- minimum preceding occluded frames `= 1`

## 3. Spatiotemporal conflict

`compute_spatiotemporal_conflict(ego_trajectory, pedestrian_trajectory)` first finds path intersections (or a near-intersection within a small configurable spatial tolerance), then defines a circular conflict region around the candidate point.

Arrival times are obtained by continuous linear interpolation along trajectory segments.

For each actor, entry/exit intervals in the conflict region are computed continuously. PET is the temporal separation between these occupancy intervals; overlapping intervals give `PET = 0`.

`conflict_exists` requires both a spatial candidate and `PET <= max_pet_s`.

Outputs:

- `conflict_exists`
- `conflict_region`
- `ego_arrival_time`
- `ped_arrival_time`
- `arrival_time_gap`
- `PET`
- `minimum_distance`

`minimum_distance` is the continuous synchronized Euclidean separation over the common time support, rather than the distance between independently sampled path points.

Default configuration:

- path tolerance `0.25 m`
- conflict-region radius `1.0 m`
- maximum PET `2.0 s`

## 4. Critical timing

\[
RTTC = t_{conflict} - t_{reveal}.
\]

The current Gate-1 defaults are:

- `aggressive`: `0 < RTTC <= 1.0 s`
- `moderate`: `1.0 < RTTC <= 3.0 s`
- `safe`: `RTTC > 3.0 s`

`T=True` for aggressive or moderate timing and `T=False` for safe timing. These thresholds are explicit configuration and must later be calibrated/validated on real data; they are not claimed as universal safety constants.

## 5. Hazard contract

`OccludedPedestrianContract.verify(...)` returns the four predicates, final hazard label, intermediate results, and a single primary failure reason.

Failure-reason priority is:

1. no pre-reveal occlusion
2. no stable reveal
3. no spatiotemporal conflict
4. conflict does not occur after reveal
5. non-critical timing
6. none (valid hazard)

## Gate 1

The evaluator must pass all toy cases before any generation experiment begins:

| Case | Expected result |
| --- | --- |
| real occlusion + reveal + conflict | `H=True` |
| unoccluded crossing | `H=False` |
| occluded but never reveals | `H=False` |
| reveal without path conflict | `H=False` |
| conflict 5.0 s after reveal | `safe`, `H=False` |
| conflict 0.8 s after reveal | `aggressive`, `H=True` |

Run from the repository root:

```bash
PYTHONPATH=. python -m pytest -q tests/hazard_equivalence/ground_truth
```

Do not proceed to diffusion/generation experiments unless Gate 1 is 100% correct.
