# Phase 0 Data Contract

Phase 0 uses a small representative set of real B0 scenes and inspects the same scenes at key interfaces of the existing SLEDGE pipeline.

Audit points:

1. dataset / feature-builder output;
2. actor and lane representation;
3. RVAE encoder input;
4. RVAE latent output;
5. RVAE decoder output;
6. diffusion scheduler interface;
7. simulation rollout input/output.

For every interface, record:

- class / dict key / tensor name;
- shape and dtype;
- coordinate frame;
- units;
- actor ordering and padding convention;
- actor type coding;
- x/y/heading/velocity/length/width fields;
- lane centerline and successor/predecessor representation;
- static-object representation;
- time axis / sampling interval;
- normalization and scaling.

Gate 0 passes only when the verified legacy representation can round-trip through the canonical representation without changing mapped actor states beyond tolerance.

Important: `phase0/configs/schema.example.json` is only a format example. Its indices are not claims about the real SLEDGE schema.
