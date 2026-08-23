#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${PACKAGE_DIR}/../.." && pwd)
source "${SCRIPT_DIR}/activate_deploy.sh"
PYTHON=${PYTHON:-"${PACKAGE_DIR}/.venv/bin/python"}
cd "${REPO_ROOT}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Deployment virtual environment missing; run ${SCRIPT_DIR}/bootstrap_env.sh first" >&2
  exit 2
fi

"${SCRIPT_DIR}/preflight_target.sh"
"${PYTHON}" -m deployment.sledge_rvae.python.prepare_sample --device cpu
"${PYTHON}" -m deployment.sledge_rvae.python.export_onnx
PRECISION=fp32 "${SCRIPT_DIR}/build_engine.sh"
cmake -S "${PACKAGE_DIR}/cpp" -B "${PACKAGE_DIR}/cpp/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${PACKAGE_DIR}/cpp/build" --parallel
"${PYTHON}" -m deployment.sledge_rvae.python.validate \
  --trt-runner "${PACKAGE_DIR}/cpp/build/sledge_rvae_trt" \
  --runner-config "${PACKAGE_DIR}/configs/runtime.ini" \
  --precision fp32
"${SCRIPT_DIR}/benchmark.sh"
