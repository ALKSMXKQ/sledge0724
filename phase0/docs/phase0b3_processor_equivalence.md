# Phase 0B-3 — Native Processor Equivalence

This is measurement-instrument validation, not a model performance experiment.
The machine-readable Gate is `phase0/outputs/processor_equivalence/summary.json`.

## Implementation and scope

The audit imports and calls the complete `sledge_raw_feature_processing()` on deserialized real `sledge_raw.gz` files, including its native line processing and rasterization. It constructs the actual `SledgeVectorRaw` dataclass with native element deserialization: this checkout's inherited `SledgeVectorRaw.deserialize()` returns a `SledgeVector`, so explicitly constructing the raw subclass avoids relying on that type mismatch. No monkey patch or SLEDGE source change is used.

Configuration comes from `sledge/script/config/common/autoencoder_model/rvae_model.yaml`, instantiated as `RVAEConfig` with Hydra's `all` conversion. This matters because the Python dataclass and model YAML differ in decoder/query defaults, although the actor preprocessing parameters agree. The complete instantiated config and source hashes are retained.

The new replay is independent of the native processor. For every category it compares the original retention script's count metrics as well as the complete native ordered state/mask, pre-cap native frame predicate, and post-float32 speed clipping. Native `coords_in_frame()` exposes the intermediate frame mask without instrumentation. No heading wrapping is added. `np.argsort` retains NumPy's default algorithm; no secondary sort is introduced.

The numeric error is computed in float64 from the predicted and native arrays, including zero padding, per scene/category/dimension. Full states and masks make selected-only errors independently recoverable. `speed_error=null` for static objects. A completely identical pair of actor states has no observable ID distinction in the processed vector; ordered state agreement is not evidence of persistent actor identity.

## Selection and reproducibility

The native cache schema is `log/scenario_type/token/sledge_raw.gz` (`nuplan/planning/training/preprocessing/utils/utils_cache.py`). The full sorted cache inventory is sampled with seed 9102026 into a 1,000-scene screening pool, supplemented by up to 40 candidates per target type with log-stratified sampling. Overflow screening is performed only to build a validation set, not to estimate prevalence. Stratum quotas and seeded fill select 100 unique cache paths. Availability of overflow strata refers to the screened pool; complete target-type counts refer to the whole inventory. The selection JSON preserves every candidate and every selected scene's reason. Short strata use all available candidates and do not fabricate replacements of that stratum.

Evidence files: `selected_tokens.txt`, `selection.json`, `per_scene.csv`, `state_comparisons.json`, `mismatches.json`, `environment.json`, and `summary.json` under `phase0/outputs/processor_equivalence/`.

## Synthetic checks

Tests cover all three categories, empty actors, exactly K, K+1, equal-distance actors with distinct headings, float32/float64 inputs, speed beyond the configured maximum, all-False/all-True raw masks, exact frame boundaries including a corner, and just-outside actors. Negative controls perturb state and mask and require comparison failure. The full Phase 0 test suite passed (10 tests); real-data Gate is evaluated separately.

## Limitations

This validates the observed processor/config/software version and selected real cases. It does not establish model reconstruction fidelity, hazard membership, temporal identity, or the safety significance of dropped actors. The original 1,000-scene retention output remains untouched. Its reservoir traversal was not sorted and did not save the full selected token list, so exact retrospective reconstruction cannot be guaranteed if filesystem enumeration changes.

## Real-data result and Gate

**PASS** — 100 scenes; 300 category checks.

- in_frame_agreement: 1.0
- count_agreement: 1.0
- ordering_agreement: 1.0
- mask_agreement: 1.0
- speed_clipping_agreement: 1.0
- legacy_count_agreement: 1.0
- max_abs_error: 0.0
- mismatch_count: 0

Selected overlapping coverage: `{'normal_no_overflow': 49, 'traversing_pickup_dropoff': 14, 'pedestrians_overflow': 32, 'near_pedestrian_on_crosswalk': 13, 'static_objects_overflow': 47, 'on_pickup_dropoff': 16, 'on_traffic_light_intersection': 15, 'traversing_intersection': 12, 'traversing_crosswalk': 11}`.

0B-3 permits entering 0B-4. Gate 0 remains NOT CLOSED until 0C has completed.

Real-data clipping coverage (from saved state comparisons): `{'vehicles_selected': 783, 'vehicles_clipped': 11, 'pedestrians_selected': 980, 'pedestrians_clipped': 0}`. Every reported clipped value agrees with native output.
