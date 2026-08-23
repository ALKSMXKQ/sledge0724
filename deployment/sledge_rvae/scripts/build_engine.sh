#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${PACKAGE_DIR}/../.." && pwd)
source "${SCRIPT_DIR}/activate_deploy.sh"
cd "${REPO_ROOT}"
ONNX_PATH=${ONNX_PATH:-"${PACKAGE_DIR}/artifacts/sledge_rvae.onnx"}
PRECISION=${PRECISION:-fp32}
TRTEXEC=${TRTEXEC:-trtexec}
PYTHON=${PYTHON:-"${PACKAGE_DIR}/.venv/bin/python"}
WORKSPACE_MIB=${WORKSPACE_MIB:-4096}

case "${PRECISION}" in
  fp32) PRECISION_ARGS=() ;;
  fp16) PRECISION_ARGS=(--fp16) ;;
  *) echo "PRECISION must be fp32 or fp16" >&2; exit 2 ;;
esac

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "ONNX file not found: ${ONNX_PATH}" >&2
  exit 2
fi
CUDA_VERSION=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)
[[ "${CUDA_VERSION}" == "12.3" ]] || { echo "CUDA 12.3 required, found ${CUDA_VERSION:-unknown}" >&2; exit 2; }
TRT_VERSION=$("${PYTHON}" -c 'import tensorrt as trt; print(trt.__version__)')
[[ "${TRT_VERSION}" == "8.6.1" ]] || { echo "TensorRT 8.6.1 required, found ${TRT_VERSION}" >&2; exit 2; }

ENGINE_PATH=${ENGINE_PATH:-"${PACKAGE_DIR}/artifacts/sledge_rvae_${PRECISION}.engine"}
LOG_PATH=${LOG_PATH:-"${PACKAGE_DIR}/reports/build_${PRECISION}.log"}
TIMING_CACHE=${TIMING_CACHE:-"${PACKAGE_DIR}/artifacts/timing.cache"}
mkdir -p "$(dirname -- "${ENGINE_PATH}")" "$(dirname -- "${LOG_PATH}")"

if ! command -v "${TRTEXEC}" >/dev/null 2>&1; then
  echo "trtexec not found; using TensorRT 8.6.1 Python builder" | tee "${LOG_PATH}"
  "${PYTHON}" -m deployment.sledge_rvae.python.build_engine \
    --onnx "${ONNX_PATH}" --engine "${ENGINE_PATH}" --precision "${PRECISION}" \
    --workspace-mib "${WORKSPACE_MIB}" --timing-cache "${TIMING_CACHE}" 2>&1 | tee -a "${LOG_PATH}"
  test -s "${ENGINE_PATH}"
  exit 0
fi

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "precision=${PRECISION}"
  echo "onnx=${ONNX_PATH}"
  echo "engine=${ENGINE_PATH}"
  echo "fixed_profile=raster:1x12x256x256"
  echo "workspace_mib=${WORKSPACE_MIB}"
  echo "custom_plugins=none"
  uname -a
  nvidia-smi
  "${TRTEXEC}" --version
  "${TRTEXEC}" \
    --onnx="${ONNX_PATH}" \
    --saveEngine="${ENGINE_PATH}" \
    --minShapes=raster:1x12x256x256 \
    --optShapes=raster:1x12x256x256 \
    --maxShapes=raster:1x12x256x256 \
    --memPoolSize="workspace:${WORKSPACE_MIB}" \
    --timingCacheFile="${TIMING_CACHE}" \
    --builderOptimizationLevel=5 \
    --verbose \
    "${PRECISION_ARGS[@]}"
} 2>&1 | tee "${LOG_PATH}"

test -s "${ENGINE_PATH}"
sha256sum "${ONNX_PATH}" "${ENGINE_PATH}" | tee -a "${LOG_PATH}"
echo "Built ${ENGINE_PATH}"
