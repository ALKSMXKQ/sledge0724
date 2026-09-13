#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHONPATH=phase0/src python -m phase0_audit.cache_audit \
  --max-samples "${PHASE0_CACHE_SAMPLES:-12}" \
  --out phase0/outputs/cache_audit.json \
  "$@"
