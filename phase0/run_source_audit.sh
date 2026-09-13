#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p phase0/outputs

PYTHONPATH="phase0/src:." \
python -m phase0_audit.source_audit \
  --out phase0/outputs/source_audit.json

echo
echo "Phase-0 source contract verified."
echo "Report: phase0/outputs/source_audit.json"
