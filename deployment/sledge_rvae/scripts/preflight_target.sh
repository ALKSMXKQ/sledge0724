#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${PACKAGE_DIR}/../.." && pwd)
source "${SCRIPT_DIR}/activate_deploy.sh"
PYTHON=${PYTHON:-"${PACKAGE_DIR}/.venv/bin/python"}
TRTEXEC=${TRTEXEC:-trtexec}

fail() { echo "TARGET PREFLIGHT FAILED: $*" >&2; exit 2; }
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found"
command -v nvcc >/dev/null 2>&1 || fail "nvcc not found"
[[ -x "${PYTHON}" ]] || fail "deployment Python not found: ${PYTHON}"

nvidia-smi -L >/dev/null 2>&1 || fail "NVIDIA driver/GPU unavailable"
CUDA_VERSION=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)
[[ "${CUDA_VERSION}" == "12.3" ]] || fail "CUDA 12.3 required, found ${CUDA_VERSION:-unknown}"
TRT_PY_VERSION=$("${PYTHON}" -c 'import tensorrt as trt; print(trt.__version__)')
[[ "${TRT_PY_VERSION}" == "8.6.1" ]] || fail "TensorRT Python 8.6.1 required, found ${TRT_PY_VERSION}"
if command -v "${TRTEXEC}" >/dev/null 2>&1; then
  TRTEXEC_VERSION=$("${TRTEXEC}" --version 2>&1 || true)
  [[ "${TRTEXEC_VERSION}" == *"8.6.1"* || "${TRTEXEC_VERSION}" == *"v8601"* ]] \
    || fail "trtexec is not TensorRT 8.6.1: ${TRTEXEC_VERSION}"
else
  echo "trtexec not found; target will use TensorRT Python builder"
fi

cd "${REPO_ROOT}"
"${PYTHON}" -m deployment.sledge_rvae.python.check_environment >/dev/null
"${PYTHON}" -m deployment.sledge_rvae.python.inspect_onnx >/dev/null

echo "Target preflight PASS: CUDA ${CUDA_VERSION}, TensorRT ${TRT_PY_VERSION}"
