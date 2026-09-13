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
