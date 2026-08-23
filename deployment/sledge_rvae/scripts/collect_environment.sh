#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
OUTPUT=${1:-"${PACKAGE_DIR}/reports/environment.txt"}
mkdir -p "$(dirname -- "${OUTPUT}")"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uname -a
  cat /etc/os-release
  nvidia-smi
  nvcc --version
  if command -v trtexec >/dev/null 2>&1; then trtexec --version; else echo "trtexec=not-installed (Python builder used)"; fi
  cmake --version
  c++ --version
} >"${OUTPUT}" 2>&1

echo "Wrote ${OUTPUT}"
