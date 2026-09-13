#!/bin/bash
set -e
cd "$(git rev-parse --show-toplevel)"
PYTHONPATH=phase0/src python -m phase0_audit.retention_audit --sample-size "${PHASE0_RETENTION_SAMPLES:-1000}" --out phase0/outputs/retention_audit.json "$@"
