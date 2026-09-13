# Phase 0 — Source Audit Results

This document records what is **actually supported by the current repository source**. It is not a design assumption.

## 1. Agent representation

`AgentIndex` is shared by vehicles and pedestrians and has exactly six fields:

| Index | Meaning | Unit / convention |
|---:|---|---|
| 0 | x | metres, ego-local after feature construction |
| 1 | y | metres, ego-local after feature construction |
| 2 | heading | radians, ego-local |
| 3 | width | full bounding-box width in metres |
| 4 | length | full bounding-box length in metres |
| 5 | velocity | scalar speed magnitude in m/s |

The feature builder first reads a nuPlan tracked object's center/box/velocity magnitude and then converts the SE(2) pose from absolute coordinates into the ego frame.

### Critical limitation: identity

The six-dimensional SLEDGE actor state does **not** contain the original nuPlan track token/id. Therefore a processed `SledgeVector` cannot support a true persistent `track_id` round trip. Phase 0 uses explicit slot IDs such as `vehicle:0` only as representation-local identifiers.

For future temporal hazard evaluation, persistent identity must come from the nuPlan/simulation object representation, not be invented from SLEDGE vector slots.

## 2. Static-object representation

`StaticObjectIndex` has exactly five fields:

`[x, y, heading, width, length]`.

The SE(2) state is likewise transformed to the ego-local frame.

## 3. Processed masks

During `process_agents` / `process_static_objects`, fixed-size arrays are allocated and a boolean output mask is set to `True` for populated slots. Therefore for a processed `SledgeVector`, `mask == True` means that slot is valid.

Do not confuse this with the raw feature builder: the raw agent helper allocates an all-False mask, while the processing path uses the raw states and constructs a new valid-slot mask.

## 4. Line / lane representation

The map feature builder does construct a directed lane/lane-connector graph from map IDs and outgoing edges. Raw path geometry contains `(x, y, heading)` points.

However, after SLEDGE feature processing, the vector line representation is reduced to sampled `(x, y)` geometry with a per-line validity mask.

### Critical limitation: topology

The processed `SledgeVector.lines` does **not** preserve the original lane ID, lane-vs-connector identity, predecessor IDs, or successor IDs. It should therefore be treated as summarized map-line geometry, not as a topology-complete lane graph.

Phase 1 must obtain topology from the map API / a separate topology adapter if successor information is needed.

## 5. Raster input to RVAE

Default RVAE configuration:

- metric frame: `64 m x 64 m`;
- pixel size: `0.25 m`;
- raster size: `256 x 256`;
- raster channels: 12;
- input becomes channel-first when converted to a torch feature tensor.

Channel order:

0. line x-orientation channel
1. line y-orientation channel
2. vehicle x-velocity channel
3. vehicle y-velocity channel
4. pedestrian x-velocity channel
5. pedestrian y-velocity channel
6. static-object x-orientation channel
7. static-object y-orientation channel
8. green-light x-orientation channel
9. green-light y-orientation channel
10. red-light x-orientation channel
11. red-light y-orientation channel

## 6. RVAE latent

The encoder produces `2 * latent_channel` channels and splits them into `mu` and `log_var`.

With the default config:

- latent channels: 64;
- latent spatial size: `8 x 8`;
- distribution fields: `mu`, `log_var`.

During RVAE training/forward, decoding uses a reparameterized sample `mu + eps * std`.

### Diffusion-specific finding

The diffusion latent dataset does **not** use that stochastic sample. It explicitly loads `data["mu"]` as the clean diffusion training feature.

No additional latent normalization/scaling was observed between the latent dataset loader and diffusion training: the dataset converts that array directly to `float32`, then DDPM noise is added.

## 7. Diffusion interface

Default DiT-B configuration expects:

- input channels: 64;
- output channels: 64;
- sample size: `8 x 8`;
- patch size: 1.

Default DDPM scheduler:

- 1000 train timesteps;
- beta start 0.0015;
- beta end 0.015;
- linear beta schedule;
- `clip_sample = False`.

The current `LDMPipeline` also supports an encoded source latent via `init_latents` and uses `scheduler.add_noise(...)` at the selected start timestep.

## 8. Simulation representation

The simulation side is object-based rather than the six-dimensional processed SLEDGE vector:

- ego is a nuPlan `EgoState`;
- observations are `DetectionsTracks` / `TrackedObjects`;
- vehicle, pedestrian and static-object managers retain dictionary keys/IDs;
- propagation uses the actual simulation interval `next_iteration.time_s - iteration.time_s`.

This is the better representation for a future temporal `TrajectoryState`, because persistent identity survives there.

## 9. What is still NOT verified

The static source audit is not Gate 0 by itself. The following must still be checked on the server with real B0 samples:

1. actual cached array/tensor shapes and dtypes for the experiment configuration you use;
2. actual valid mask counts for vehicles/pedestrians/static objects/lines;
3. whether your checkpoint or Hydra experiment overrides any default RVAE/DiT settings;
4. the exact local-axis sign convention needed by the geometric hazard evaluator;
5. a real processed `SledgeVector -> canonical -> SledgeVector` round trip on representative B0 scenes;
6. a simulation-side identity/trajectory probe on at least one rollout.

Until those runtime checks pass, **Gate 0 remains open**.
