# Phase 0 execution record

Working directory: `/home16T/home8T_1/leitingting/sledge_workspace/sledge`.
All database/cache reads are read-only. Existing outputs are preserved; audit CLIs refuse nonempty output directories.

## 0B-3

```bash
MPLCONFIGDIR=/tmp/phase0-mpl PYTHONPATH=phase0/src:$PYTHONPATH /home/leitingting/anaconda3/envs/sledge/bin/python -m unittest discover -s phase0/tests -p 'test_processor_equivalence.py' -v
MPLCONFIGDIR=/tmp/phase0-mpl PYTHONPATH=phase0/src:$PYTHONPATH /home/leitingting/anaconda3/envs/sledge/bin/python -m phase0_audit.processor_equivalence --cache-root /home16T/home8T_1/leitingting/sledge_workspace/exp/caches/autoencoder_cache --output-dir phase0/outputs/processor_equivalence --sample-size 100 --seed 9102026 > phase0/outputs/processor_equivalence_run.log 2>&1
MPLCONFIGDIR=/tmp/phase0-mpl PYTHONPATH=phase0/src:$PYTHONPATH /home/leitingting/anaconda3/envs/sledge/bin/python -m phase0_audit.processor_equivalence --cache-root /home16T/home8T_1/leitingting/sledge_workspace/exp/caches/autoencoder_cache --output-dir phase0/outputs/processor_equivalence --sample-size 100 --seed 9102026 > phase0/outputs/processor_equivalence_run_v2.log 2>&1
MPLCONFIGDIR=/tmp/phase0-mpl PYTHONPATH=phase0/src:$PYTHONPATH /home/leitingting/anaconda3/envs/sledge/bin/python -m pytest -q phase0/tests > phase0/outputs/phase0b3_tests.log 2>&1
```

The first real-audit launch failed before reading scenes: Hydra instantiated `frame` as a `ListConfig` that JSON cannot serialize. The wrapper now requests `_convert_='all'`, matching the model YAML's conversion policy. The first log is retained; no scene mismatch was discarded.

## 0B-4 (started only after 0B-3 PASS)

```bash
MPLCONFIGDIR=/tmp/phase0-mpl PYTHONPATH=phase0/src:$PYTHONPATH /home/leitingting/anaconda3/envs/sledge/bin/python -m phase0_audit.targeted_retention --cache-root /home16T/home8T_1/leitingting/sledge_workspace/exp/caches/autoencoder_cache --db-root /home16T/home8T_1/leitingting/sledge_workspace/dataset/nuplan-v1.1/splits --output-dir phase0/outputs/targeted_retention --sample-size 150 --seed 9102026 > phase0/outputs/targeted_retention_run.log 2>&1
MPLCONFIGDIR=/tmp/phase0-mpl PYTHONPATH=phase0/src:$PYTHONPATH /home/leitingting/anaconda3/envs/sledge/bin/python -m pytest -q phase0/tests > phase0/outputs/phase0b4_tests.log 2>&1
```

## 0C (started only after 0B-4 COMPLETE)

```bash
MPLCONFIGDIR=/tmp/phase0-mpl PYTHONPATH=phase0/src:$PYTHONPATH /home/leitingting/anaconda3/envs/sledge/bin/python -m phase0_audit.temporal_identity --cache-root /home16T/home8T_1/leitingting/sledge_workspace/exp/caches/autoencoder_cache --db-root /home16T/home8T_1/leitingting/sledge_workspace/dataset/nuplan-v1.1/splits --map-root /home16T/home8T_1/leitingting/sledge_workspace/dataset/maps --output-dir phase0/outputs/temporal_identity --sample-size 30 --seed 9102026 --duration 5 > phase0/outputs/temporal_identity_run.log 2>&1
MPLCONFIGDIR=/tmp/phase0-mpl PYTHONPATH=phase0/src:$PYTHONPATH /home/leitingting/anaconda3/envs/sledge/bin/python -m pytest -q phase0/tests > phase0/outputs/phase0c_tests.log 2>&1
```

For a fresh run on another machine, change the CLI paths and use an empty output directory for each stage. `PYTHONPATH` must include this repository, `phase0/src`, and the installed nuPlan checkout; `SLEDGE_EXP_ROOT` supplies the default cache root. No script contains these server paths as defaults. The recorded environment JSON captures the actual executable, packages, config and processor hashes.

## Temporal evidence diagnostics and final tests

The source inspection found that StaticObject exposes a dummy zero velocity. The following command reanalyzes the saved real observations to separate it from measured dynamic velocity; it does not rerun, replace, or drop any real scenario.

```bash
MPLCONFIGDIR=/tmp/phase0-mpl PYTHONPATH=phase0/src:$PYTHONPATH /home/leitingting/anaconda3/envs/sledge/bin/python -m phase0_audit.temporal_diagnostics --input-dir phase0/outputs/temporal_identity --output-dir phase0/outputs/temporal_identity > phase0/outputs/temporal_diagnostics_run.log 2>&1
MPLCONFIGDIR=/tmp/phase0-mpl PYTHONPATH=phase0/src:$PYTHONPATH /home/leitingting/anaconda3/envs/sledge/bin/python -m pytest -q phase0/tests > phase0/outputs/phase0_final_tests.log 2>&1
```
