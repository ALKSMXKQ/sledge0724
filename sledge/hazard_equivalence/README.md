# Hazard Equivalence — Phase 0

Phase 0 audits the existing SLEDGE scene representation before any hazard evaluator or diffusion method is added.

Main rule: **do not guess legacy tensor indices**. Verify them from the real repository/data pipeline, encode them in an explicit adapter, then require a legacy -> canonical -> legacy round trip to pass.

See `docs/phase0_data_contract.md`.

Run the unit tests from the repository root with:

```bash
PYTHONPATH=. python -m pytest -q tests/hazard_equivalence
```
