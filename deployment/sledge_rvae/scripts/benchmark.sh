#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${PACKAGE_DIR}/../.." && pwd)
RUNNER=${RUNNER:-"${PACKAGE_DIR}/cpp/build/sledge_rvae_trt"}
CONFIG=${CONFIG:-"${PACKAGE_DIR}/configs/runtime.ini"}
GPU_CSV=${GPU_CSV:-"${PACKAGE_DIR}/reports/gpu_samples.csv"}
PYTHON=${PYTHON:-"${PACKAGE_DIR}/.venv/bin/python"}

if [[ ! -x "${RUNNER}" ]]; then
  echo "Runner not found or not executable: ${RUNNER}" >&2
  exit 2
fi

cd "${REPO_ROOT}"

"${SCRIPT_DIR}/collect_environment.sh"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -lms 100 >"${GPU_CSV}" &
SAMPLER_PID=$!
cleanup() {
  kill "${SAMPLER_PID}" 2>/dev/null || true
  wait "${SAMPLER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"${RUNNER}" --config "${CONFIG}"
cleanup
trap - EXIT INT TERM

"${PYTHON}" -m deployment.sledge_rvae.python.performance_report
