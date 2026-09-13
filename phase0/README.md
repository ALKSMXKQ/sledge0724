# Phase 0 — Representation Audit

`phase0/` is a completely standalone research-stage folder. It does **not** modify the existing `sledge/` package.

## Directory

```text
phase0/
├── README.md
├── run_tests.sh
├── src/
│   └── phase0_audit/
│       ├── __init__.py
│       ├── canonical.py
│       ├── adapter.py
│       ├── audit.py
│       └── run_audit.py
├── tests/
│   └── test_phase0.py
├── docs/
│   └── data_contract.md
└── configs/
    └── schema.example.json
```

## Goal

Phase 0 only answers: **what data representation does the current SLEDGE pipeline really use?**

Do not implement hazard evaluation or diffusion modifications here.

## Test

From repository root:

```bash
bash phase0/run_tests.sh
```

## Rule

The schema in `configs/schema.example.json` is only an example of the file format. Never assume its indices are the real SLEDGE indices until they are verified from the repository/data pipeline.
