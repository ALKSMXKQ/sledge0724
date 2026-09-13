# Phase 0B-4 — Safety-Relevant Targeted Retention

## Method

This stage started after the real-data 0B-3 PASS. Its CLI checks that the passed Gate's config, native processor, replay hashes and package versions still match. It uses the validated replay with the native YAML configuration. No SLEDGE preprocessing, capacity, or data is modified.

Each of the six target scenario types contributes 150 scenes, unless fewer are available (then all). Fixed seed 9102026; equal allocation across logs, shuffled within each log, with redistribution when a log is exhausted. This estimates behavior in the selected log-balanced cohort; it is not a scene-uniform prevalence estimate over the entire cache. Nearby samples can still be correlated. These are representation statistics, not model performance metrics.

The unit of the per-scene CSV is scenario × actor category. `overflow=in_frame_count-cap` is signed; the report supplies both all-scene signed overflow and positive-overflow-only distributions. `d_K` is the last retained actor's center distance for any nonempty retained set, even below capacity. `d_first_drop` and `boundary_gap` are null when there is no capacity drop. All spatial decisions use actor center positions, inclusive square bounds, not oriented-box intersection.

`P(kept | inside frame)` uses totals across actors. Cap-drop scene fraction uses all selected scenes, including empty scenes. Raw-nonempty and in-frame-nonempty counts are both retained. Spatial crop and capacity drop are separate columns. Distribution sample counts are explicit; static objects have no speed dimension.

## Scenario-type provenance

The directory schema is not a filename heuristic: `nuplan/planning/training/preprocessing/utils/utils_cache.py` writes `cache_path / scenario.log_name / scenario.scenario_type / scenario.token`. Cache metadata also contains the original `unknown` labels and original machine paths. Labels are checked against local SQLite databases opened with `mode=ro`.

`get_scenarios_from_db()` left-joins `scenario_tag` and uses `MAX(st.type)` for a selected type. `nuplan_scenario_filter_utils.py` replaces null labels with `DEFAULT_SCENARIO_NAME`, whose value is `unknown`. Unknown is therefore a legitimate upstream absence of annotation, not automatically a parsing bug. The audit resolves only a single unambiguous authoritative tag; multiple tags remain explicit ambiguity. Missing databases/tokens remain unresolved and quantified. No token-to-type guess is used.

The old 1,000-scene output includes 295 unknown scenarios but did not preserve its full sampled token list; exact retrospective membership cannot be recovered reliably. This audit instead enumerates and accounts for every unknown in the current 500,356-entry cache. It preserves token-level evidence in `unknown_resolution_per_token.csv`. No unknown enters the six target groups.

## Interpretation boundaries

Observation and interpretation are reported separately below. A small `d_first_drop` establishes a nearby actor center omitted by capacity, not that the actor defines a hazard. A center distance of 30 m does not, by itself, prove proximity to a **square** frame boundary: direction matters. A 64×64 square contains corners up to approximately 45.25 m from ego. No hazard actor label, planner effect, or safety-critical information-loss claim is established here.

## Observation

COMPLETE: 900 real scenes, 150 per target type. Unknown in target statistics: 0.

Across the full cache: 137,977 unknown, of which 40,684 are confirmed untagged in the local authoritative DB and 97,293 lack a local matching log database. No unknown label was reassigned; all 137,977 remain semantically unresolved, with exact reasons. Known cache labels had zero disagreements where the corresponding local DB was available.

| Scenario type | Actor | n | cap-drop scene % | P(kept given inside) | signed overflow median / p95 / max | d_K median m | d_first_drop median / min m | gap median m |
|---|---|---:|---:|---:|---|---:|---|---:|
| near_pedestrian_on_crosswalk | vehicles | 150 | 0.00 | 1.0000 | -44.000 / -31.000 / -21.000 | 32.198 | null / null | null |
| near_pedestrian_on_crosswalk | pedestrians | 150 | 38.67 | 0.5614 | -8.000 / 48.000 / 99.000 | 25.543 | 22.429 / 8.664 | 0.328 |
| near_pedestrian_on_crosswalk | static_objects | 150 | 56.00 | 0.6719 | 5.500 / 31.650 / 57.000 | 24.948 | 23.802 / 17.645 | 0.530 |
| traversing_crosswalk | vehicles | 150 | 0.00 | 1.0000 | -47.000 / -33.450 / -23.000 | 30.805 | null / null | null |
| traversing_crosswalk | pedestrians | 150 | 8.00 | 0.5935 | -18.000 / 20.550 / 73.000 | 28.680 | 23.048 / 11.104 | 0.147 |
| traversing_crosswalk | static_objects | 150 | 15.33 | 0.6822 | -26.000 / 44.200 / 54.000 | 27.330 | 20.454 / 17.571 | 1.528 |
| on_pickup_dropoff | vehicles | 150 | 0.00 | 1.0000 | -41.000 / -31.450 / -15.000 | 34.196 | null / null | null |
| on_pickup_dropoff | pedestrians | 150 | 30.00 | 0.5757 | -7.500 / 50.100 / 84.000 | 27.143 | 20.748 / 10.839 | 0.281 |
| on_pickup_dropoff | static_objects | 150 | 56.00 | 0.7365 | 4.000 / 32.100 / 39.000 | 31.288 | 22.936 / 17.029 | 0.420 |
| traversing_pickup_dropoff | vehicles | 150 | 0.00 | 1.0000 | -41.000 / -30.450 / -19.000 | 35.821 | null / null | null |
| traversing_pickup_dropoff | pedestrians | 150 | 33.33 | 0.5915 | -7.000 / 38.550 / 87.000 | 26.219 | 20.306 / 10.667 | 0.322 |
| traversing_pickup_dropoff | static_objects | 150 | 58.67 | 0.7559 | 5.000 / 25.550 / 32.000 | 29.273 | 26.423 / 15.235 | 0.375 |
| on_traffic_light_intersection | vehicles | 150 | 0.00 | 1.0000 | -40.000 / -24.000 / -17.000 | 33.906 | null / null | null |
| on_traffic_light_intersection | pedestrians | 150 | 30.67 | 0.5601 | -11.500 / 41.000 / 68.000 | 27.169 | 22.018 / 9.780 | 0.222 |
| on_traffic_light_intersection | static_objects | 150 | 50.67 | 0.6295 | 1.500 / 45.000 / 74.000 | 24.664 | 22.972 / 14.620 | 0.369 |
| traversing_intersection | vehicles | 150 | 0.00 | 1.0000 | -45.000 / -30.450 / -20.000 | 32.547 | null / null | null |
| traversing_intersection | pedestrians | 150 | 26.67 | 0.6346 | -17.000 / 30.650 / 68.000 | 27.197 | 26.992 / 12.956 | 0.333 |
| traversing_intersection | static_objects | 150 | 28.00 | 0.8511 | -15.000 / 20.000 / 26.000 | 29.367 | 28.729 / 18.596 | 0.694 |

All median/p75/p90/p95/max distributions, conditional sample counts, actor totals, and per-scene metrics are in summary.json and per_scene.csv. Nearest dropped examples are preserved per type/category in examples.json.

## Scientific interpretation

Capacity truncation differs across these selected target groups and can remove actor centers relatively close to ego. Scene type is a sampling stratum, not hazard semantics. These data establish neither which omitted actor participates in a conflict nor whether a planner is challenged. Missing local source DBs constrain retrospective unknown-label verification; relabeling them without evidence would be invalid.

No model-performance PASS/FAIL applies. 0B-4 is COMPLETE with unknown precisely quantified; Gate 0 remains NOT CLOSED pending 0C.
