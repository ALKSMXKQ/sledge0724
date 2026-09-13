# Phase 0 review package

Repository: `ALKSMXKQ/sledge0724`. Working branch: `phase0-standalone`.
Baseline commit: `ce5a8a9`. All added implementation, tests, reports, and evidence are under `phase0/`.

## Review entry points

- [0B-3 native equivalence report](phase0b3_processor_equivalence.md)
- [0B-4 targeted retention report](phase0b4_targeted_retention.md)
- [0C temporal identity report](phase0c_temporal_identity.md)
- [Actual execution commands](phase0_execution_commands.md)
- Machine-readable results: `../outputs/processor_equivalence/summary.json`, `../outputs/targeted_retention/summary.json`, `../outputs/temporal_identity/summary.json`.

The execution record preserves the full experiment and test commands. Each executable audit module supports `--help`, `--cache-root`, `--output-dir`, `--sample-size`, and `--seed`. Outputs refuse overwrite. CLI database/map paths are explicit; no server-specific paths are embedded in implementation defaults. Native config parameters are read from the actual model YAML. Original source, cache, and data were not edited.

## Environment

Python executable: `/home/leitingting/anaconda3/envs/sledge/bin/python` (conda environment `sledge`).

| Package | Version |
|---|---|
| numpy | 1.23.4 |
| torch | 2.3.0 (runtime build 2.3.0+cu121) |
| opencv-python | 4.9.0.80 |
| hydra-core | 1.3.2 |
| nuplan-devkit | 1.2.2 |
| scipy | 1.13.1 |
| shapely | 2.0.7 |

`SLEDGE_EXP_ROOT=/home16T/home8T_1/leitingting/sledge_workspace/exp`.
nuPlan source: `/home16T/home8T_1/leitingting/nuplan-devkit`, git commit `e9241677997dd86bfc0bcd44817ab04fe631405b`. Each experiment records config, processor and replay hashes; 0C also records inspected API hashes. Audit computations use CPU; the import-time NVML warning does not indicate a failed processor call.

## Code changes

Implementation:

- `phase0/src/phase0_audit/native_common.py`: native configuration, independent replay, read-only cache loading, deterministic sampling, provenance and serialization.
- `phase0/src/phase0_audit/processor_equivalence.py`: native processor comparisons, selection evidence and enforced 0B-3 Gate.
- `phase0/src/phase0_audit/scenario_index.py`: read-only DB lookup and authoritative unknown-type accounting.
- `phase0/src/phase0_audit/targeted_retention.py`: gated target-type sampling, capacity/crop metrics and distance distributions.
- `phase0/src/phase0_audit/temporal_state.py`: independent coordinates, native raw/processed actor correspondence, track and transition diagnostics.
- `phase0/src/phase0_audit/temporal_identity.py`: actual nuPlan rollouts and API probes, DB identity-link checks, timestamps and evidence output.
- `phase0/src/phase0_audit/temporal_diagnostics.py`: read saved observations, distinguish dummy static velocity, and quantify velocity-coordinate hypotheses and representation departures.

Tests: `test_processor_equivalence.py`, `test_targeted_retention.py`, `test_temporal_identity.py` under `phase0/tests/`. Full suite: 18 tests passed; synthetic cases exercise all three actor types and the requested boundary/overflow/tie/speed/mask cases. They do not substitute for real runs.

Documentation and outputs are listed in the file manifest accompanying this review. The three pre-existing untracked files `phase0/outputs/source_audit.json`, `cache_audit.json`, and `retention_audit.json` are preserved without modification or inclusion in the new commits.

## Phase 0B-3 result

**PASS**. 100 scenes, 300 category comparisons. In-frame, count, ordering, mask, speed-clipping, and original retention-count agreement all 100%; max absolute state error 0; mismatches `[]`. Real selected vehicles included 11 speeds above the configured limit; selected real pedestrians did not exercise clipping, so pedestrian clipping is additionally covered by synthetic native tests. The sample covers ordinary scenes, both overflow categories, and every requested target type. Reasons, raw indices, full state arrays, masks and speed triples are saved.

The first launch failed before reading data because Hydra's ListConfig could not be serialized. The wrapper was fixed to use the model's `all` conversion policy. Both launch logs are retained. No tolerance was relaxed, failed sample removed, or preprocessing changed.

## Phase 0B-4 result

**COMPLETE**. Six target types × 150 scenes = 900. All vehicles retained in-frame (zero vehicle capacity drops). Pedestrian/static capacity metrics, signed and positive overflow distributions, `d_K`, `d_first_drop`, and boundary gaps appear in the stage report and JSON, with per-scene records retained.

Unknown accounting covers all 500,356 cached scenarios: 137,977 remain unknown. Of these, 40,684 are verified as untagged in matching local DBs; 97,293 lack matching local DBs. Zero guessed labels, zero unknown scenes in targeted statistics, and zero known-label disagreements where local DB verification was possible. Exact recovery of the historical 295/1,000 subset is not claimed because the old script did not save its full token list.

A first capacity-dropped pedestrian is as close as 8.664 m in this cohort. Sampling is log-balanced and not a uniform estimate of the entire cache. Nearby actor omission is not a hazard or planner-failure label.

## Scientific interpretation

### Confirmed Facts

The current native processor filters actor centers to the configured square, sorts by ego distance using default NumPy argsort, retains the configured cap, and casts state to float32 before speed clipping. The old count measurement agrees on this cohort. Raw dummy actor masks do not mean invalid actors. Processor outputs do not encode persistent track identity or timestamps.

The full raw builder has an earlier selection layer: vehicle drivable-area filtering and actor radius filtering. The local coordinate origin is ego center. The Python config class and model YAML have different decoder/query defaults, so this audit uses the actual model YAML; it does not silently assume dataclass defaults describe a trained model.

`unknown` is also an upstream nuPlan label for untagged frames, not just a metadata parser failure. Unknown labels without evidence remain unresolved and excluded from target groups.

### Observed Phenomena

Pedestrian and static capacity drops occur at substantially different rates across the six selected target groups. Some dropped pedestrian centers are within approximately 9–13 m of ego. Vehicles do not overflow in either of these new cohorts. These are cohort observations, with log-balanced sampling and possible within-log correlation.

### Open Questions

How complete and physically correct are the source annotations? Can corresponding missing log DBs be supplied for the 97,293 unknown cache entries lacking local source data? Untagged frames will still have no authoritative target label even when source DBs are available. What event-time precision would a future hazard definition require relative to sampling jitter and ego/lidar timestamp alignment? These audits do not answer hazard semantics or planner outcomes.

### Unsupported Claims

Dropped actor equals hazard actor; generic representation loss is safety-critical information loss; actor retention rates establish planner difficulty; zero observed vehicle overflow means overflow never occurs; stable annotated IDs prove no physical ID switches anywhere; RVAE latent or processed slot index supplies persistent identity; a 30 m radial distance necessarily indicates proximity to a square frame boundary. None of these claims is supported.

## Differences from assumptions and remaining constraints

No SLEDGE-core change was required. Inherited raw deserialization returns the processed base class, so the audit constructs the real raw dataclass explicitly. The YAML rather than bare dataclass supplies the audited config. Existing raw vehicle filtering precedes the frame/top-K stages. Scalar radius and distance do not locate the boundary of a square. Unknown tags are partly confirmed absent and partly unverified due to missing local DBs; they were not fabricated into known labels.

B3 hashes were rechecked after later experiments: all 100 source cache files were unchanged. `git diff ce5a8a9 -- sledge` is empty. Existing untracked baseline outputs remain untouched.


## Phase 0C result

**PASS** on 30 real rollouts / 3,017 contiguous frames / 4,089 scenario-local tracks / 285,188 object observations. 567,822 independent database previous/next links agree on track identity. Structural identity failures and observed track gaps: zero. Mean dt = 49.990217 ms, std = 0.038893 ms, range = 49.803–50.058 ms. Maximum independent-transform error = 3.8146785e-6; all 2,451 anchor raw actor correspondences are exact and unambiguous.

461 spatial entries, 394 spatial exits; 249 kept-to-cap-excluded transitions and 241 capacity readmissions. Of previously kept objects, 175 leave spatially, 249 through capacity, 1 through raw drivable-area filtering and 245 disappear from detections with unknown cause. Full per-track/per-frame/observation and transition files are retained.

**Ground-truth choice:** object-level nuPlan temporal state, retaining `(log_name, track_token)`, original timestamps, coordinates and annotation-quality flags. Raw/processed vectors alone lack the persistent identity/time fields required for a temporal evaluator. No hazard evaluator is implemented.

**Material qualifications:** ego vx/vy empirically use body axes, whereas agent velocities favor global axes; static API velocity is dummy zero. Ego-pose/lidar timestamps differ by −5.444 to +5.315 ms beneath the aligned public API. Dynamic velocity/position residuals include large outliers (vehicle max 53.1895 m/s, pedestrian max 18.4889 m/s), saved with tokens in diagnostic_summary.json. These are open annotation-quality observations, not proven identity switches and not model failures. The structural Gate PASS does not certify every source trajectory's physical accuracy.


## Git and final Gate

| Stage | Commit | Commit message |
|---|---|---|
| 0B-3 | `be87b9c55d5181716b5b4c54381ef093b00153c3` | phase0: validate retention replay against native processor on 100 real scenes |
| 0B-4 | `85f8a18fdf82b684d4d8f4161ea35f09509f6e93` | phase0: audit 900 targeted scenes and quantify authoritative unknown labels |
| 0C | `42c62b2b741ab8b2984b307c046da3ff128e241d` | phase0: validate nuPlan temporal identity and coordinates on 30 real rollouts |

A final packaging commit records this review, the file manifest and Gate status. Find its exact hash with `git log -1 --format=%H` at delivery. All commits are on `phase0-standalone`. Complete changed-file list: `../outputs/phase0_changed_files.txt`.

- Phase 0B-3: **PASS**
- Phase 0B-4: **COMPLETE**
- Phase 0C: **PASS**
- Gate 0: **CLOSED** for the requested Phase 0 audit scope.

The measurement tool was validated against native processing, targeted statistics and unknown reasons were quantified, and the temporal identity/time/coordinate/transition contract was checked on real rollouts. This closure is not a declaration that unknown labels have been invented or that all physical source trajectories are flawless. The documented annotation-quality and temporal-resolution questions remain open for future work. No model or hazard-performance conclusion is claimed.
