# Phase 0C — Temporal / Object Identity Audit

## Method and evidence boundaries

After 0B-3 PASS and 0B-4 COMPLETE, the audit inspects this installation's actual `AbstractScenario`, `NuPlanScenario`, `DetectionsTracks`, and `TrackedObjects` classes. Signatures, source locations, and hashes are recorded in `api_inspection.json`. It constructs actual `NuPlanScenario` instances against existing local log databases and map files. It never substitutes a synthetic scenario for real data.

The cohort reuses prior audited tokens whose corresponding local database exists. Selection prioritizes the 12 available 0B-3 scenarios, then adds three per target type from 0B-4 (seed 9102026). Unavailable log databases are recorded as cohort eligibility constraints, before running the experiment. No failed rollout is removed. Each scenario spans 5 seconds forward with `subsample_ratio=1`, plus separate real past/future API probes of 10 samples over one second each. The contiguous rollout uses `get_number_of_iterations`, `get_time_point`, `get_ego_state_at_iteration`, and `get_tracked_objects_at_iteration`. Probe timestamps are aligned with corresponding ego and detection outputs.

For every observed object, it records global position, heading, velocity when defined, box dimensions, track token, detection token, integer runtime ID, timestamp, and representation status. `actor_observations.csv.gz` retains the full observations; `per_track.csv` and `per_frame.csv` supply summaries. All numeric output uses explicit units in field names or the contract below.

Persistent identity is checked using repeated track tokens, per-frame duplicate detection, runtime-ID/category consistency, and an independent SQLite join of each `lidar_box.prev_token` / `next_token` to its linked row's `track_token`. This validates annotation consistency; it cannot prove that annotators never swapped two physical objects. Gaps are measured as observation discontinuities, not automatically ID changes.

## Coordinate and unit contract

- Object x/y and ego pose x/y are map/global coordinates in meters. Heading is radians. Boxes store full width/length in meters; velocity is m/s. Static objects have no measured velocity field in the native API.
- SLEDGE agent-local origin is **ego center**, while nuPlan DB ego position and the returned `rear_axle` refer to the rear axle. Both are saved, preventing origin confusion.
- Independent transformation uses `dx=x_actor-x_ego_center`, `dy=y_actor-y_ego_center`, `x_local=cos(h)*dx+sin(h)*dy`, `y_local=-sin(h)*dx+cos(h)*dy`. Heading uses `atan2(sin(h_actor-h_ego), cos(h_actor-h_ego))`, because the native **raw feature builder** explicitly performs that normalization. 0B-3 still compares processed heading without adding wrapping.
- Native raw actor features cast to float32; the independent transform remains float64 for reporting. Speed is `hypot(vx,vy)`, then the validated processor applies its category limit. Width/length and speed are included in the state correspondence tests.
- Vehicles undergo the native drivable-area predicate and radius filtering before raw features; pedestrians/static undergo radius filtering. The audit calls the actual map helper and native raw feature builders, checks independently derived rows, and applies the validated replay plus native actor processors for every frame.
- At the cached anchor, a full-state assignment matches current native raw rows to cached raw rows. This is a **raw correspondence** check because the DB query does not promise actor ordering; it does not reorder B3 comparisons. Ambiguous full-state matches are counted explicitly and do not provide unique ID evidence.
- Native scenario/object timestamps are absolute microseconds tied to `lidar_pc`. The nuPlan ego query explicitly substitutes the lidar timestamp for the linked `ego_pose` timestamp, which can differ. Both the public timestamp consistency and underlying offsets are measured.

## Enter / exit semantics

The trace separates `kept`, `cap_excluded`, `outside_frame`, `raw_drivable_area_exclusion`, `raw_radius_exclusion`, and unsupported actor categories. For actors present in both successive frames, it distinguishes geometric frame entry/exit from a kept-to-cap-excluded change. Observation disappearance is `observation_disappearance_unknown_cause`, and first appearance is censored: absence from detections alone does not establish a physical spatial exit. The initial frame is left-censored and no events are invented before it. Missing and reappearing tracks produce recorded observation gaps.

## Scope of the Gate

A Phase 0C PASS means identity, timestamps, coordinates, units and representation transitions can be established for this cohort with no structural validation failures. It is not a guarantee of exhaustive annotations, physical identity correctness, or continuous-time event truth. No O/E/C/T, LOS, Reveal, Conflict, RTTC, or hazard evaluator is implemented.

## Real-data results

**PASS**, scoped to the sampled annotation identity/time/coordinate contract.

| Measure | Result |
|---|---:|
| Rollouts | 30 (12 from B3, 18 from B4) |
| Contiguous frames | 3,017 |
| Scenario-local tracks | 4,089 |
| Object observations | 285,188 |
| Tracks seen in multiple frames | 4,069 |
| Database previous/next identity-link checks | 567,822 |
| Duplicate/missing IDs, runtime ID/category discontinuities, broken identity links | 0 |
| Tracks with observation gaps in these windows | 0 |
| Expected detection-token changes | 281,099 |
| Mean / std dt | 49.990217 / 0.038893 ms |
| Min / max dt | 49.803 / 50.058 ms |
| Independent float64 to native float32 maximum state error | 3.8146785e-6 |
| Cached raw actor correspondences at anchors | 2,451 |
| Maximum current raw vs cached raw state error | 0 |
| Ambiguous cache correspondences | 0 |
| Cached vs rebuilt ego-feature error | 0 |

All 60 past/future API probes returned 10 aligned samples each. Rollouts contain either 100 or 101 frames because real timestamps and inclusive extraction bounds determine membership; no fixed-count replacement was used. All 30 selected scenarios completed. The eligible union contained 937 previously audited scenes, 136 with local log databases and 801 unavailable; eligibility was fixed before experiment execution.

Observed common-track spatial entries/exits: 461 / 394. Kept→cap-excluded transitions: 249; cap-excluded→kept: 241. These counts include repeated transitions, not distinct actors. Of actors leaving a previously kept representation, 175 moved outside the square, 249 were capacity-excluded, 1 failed raw drivable-area filtering, and 245 disappeared from the object observations (cause unknown). These are separate mechanisms. Total supported-category observation appearances/disappearances across the windows were 1,347 / 1,156; they are censored observations, not proof of physical entry/exit.

## Velocity and timestamp caveats established by data

Ego velocity must not be conflated with global actor velocity. Against ego global rear-axle displacement, the mean residual is **6.2026 m/s** if API vx/vy are used directly as global components, versus **0.2279 m/s** after rotating them from ego/body axes. This provides empirical evidence for body-axis ego velocity in this installation/data. SLEDGE `compute_ego_features()` copies those components without an extra rotation.

Agent velocity favors global axes: vehicle residual median/p95 = 0.3191/1.6640 m/s; pedestrian = 0.1225/0.6887 m/s. Treating agent velocities as body-frame components gives much larger p95 residuals (11.7997 and 2.5316 m/s respectively). These finite-difference diagnostics are not exact physical laws for smoothed/discrete annotations.

**Data-quality observations remain:** the maximum dynamic velocity/displacement residual is 53.1895 m/s for vehicles and 18.4889 m/s for pedestrians. The ten largest measured-agent examples, including both detection tokens, track token, timestamp, displacement and velocity, are saved in `diagnostic_summary.json`. They are not silently removed, nor automatically labeled physical identity switches. Native DB track-link consistency does not establish perfect physical trajectories. Follow-up quality inspection is required before treating every raw trajectory as physically accurate event ground truth.

This installed `StaticObject` API exposes a **dummy zero velocity**, although static SLEDGE state remains five-dimensional. The original all-object residual is therefore not a measured-agent statistic. The supplementary CLI separates 163,077 dynamic observations from 122,111 static dummy-velocity observations and reports dynamic-only distributions. There is no static speed clipping or measured static-speed claim.

Underlying ego_pose-minus-lidar timestamps range from **−5,444 to +5,315 µs**, with mean +20.569 µs. Public scenario ego/object timestamps agree because the query substitutes the lidar clock. This millisecond-scale alignment offset and 50 ms sampling interval must remain explicit in future timing interpretation.

## Answers to the four research questions

**Q1. Persistent identity?** Yes at the annotated object level in this cohort, using `(log_name, track_token)`. Detection `token` changes normally; integer `track_id` is assigned by a process-local cached counter. Structural identity failures were zero, but physical annotation errors cannot be excluded by token/link checks.

**Q2. Temporal sampling?** Approximately 20 Hz with small measured jitter and strictly increasing timestamps. This supports a sampled temporal evaluation substrate. It does not yet validate the accuracy of t_reveal, t_conflict or RTTC, which were not defined or implemented; sub-frame event precision cannot be assumed.

**Q3. Is processed SledgeVector sufficient temporal hazard ground truth?** No as a standalone source in this implementation. Its schema lacks persistent IDs and timestamps, and real same-track kept-to-cap-excluded transitions occur. It cannot distinguish these causes from the vector alone. This is a representation sufficiency conclusion, not a model failure claim.

**Q4. Best ground-truth layer?** The object-level nuPlan temporal state is the strongest available substrate, with original timestamps, persistent annotated track tokens, global geometry and measured agent velocities, plus explicit annotation-quality checks. A derived ledger can reference raw/processed states while keeping that provenance.

| Candidate layer | Identity / clock | Evidence and limitation |
|---|---|---|
| nuPlan object-level temporal state | Annotated track tokens and timestamps validated | Preferred substrate; preserve quality flags, body/global velocity conventions and ego/lidar offset |
| SledgeVectorRaw | No stored persistent track ID or timestamp | Useful geometry correspondence; earlier radius/drivable-area selection; cached snapshot is insufficient alone |
| Processed SledgeVector | No stored persistent track ID or timestamp | Additional spatial crop/top-K and velocity clipping; observed capacity-induced disappearance |
| RVAE latent | No explicit validated identity/time contract | Not established as temporal ground truth by these audits; no latent model experiment was run |

## Gate

Phase 0C: **PASS** for the tested structural identity/time/coordinate/transition contract. Structural anomalies: none. Trajectory velocity/position inconsistencies are separately recorded data-quality observations, not hidden under an all-data-perfect claim. Combined with B3 PASS and B4 COMPLETE, the requested Phase 0 audit Gate can close. This does not close the open questions about annotation fidelity or future hazard-event definitions.
