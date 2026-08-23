#!/usr/bin/env bash
# Source this file; do not execute it as a child process.

SLEDGE_DEPLOY_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SLEDGE_DEPLOY_PACKAGE_DIR=$(cd -- "${SLEDGE_DEPLOY_SCRIPT_DIR}/.." && pwd)

source "${SLEDGE_DEPLOY_PACKAGE_DIR}/.venv/bin/activate"

# NVIDIA pip wheels keep shared objects below site-packages/nvidia/*/lib rather
# than a system loader path. Include all inherited and venv-local CUDA 12 paths.
SLEDGE_NVIDIA_LIBS=$(python -c 'import pathlib,sys; paths=[]; [paths.extend(str(p) for p in pathlib.Path(root,"nvidia").glob("*/lib") if p.is_dir()) for root in sys.path]; print(":".join(dict.fromkeys(paths)))')
SLEDGE_TRT_LIBS="${SLEDGE_DEPLOY_PACKAGE_DIR}/.venv/lib/python3.9/site-packages/tensorrt_libs"
export LD_LIBRARY_PATH="${SLEDGE_NVIDIA_LIBS}:${SLEDGE_TRT_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export SLEDGE_DEPLOY_PYTHON="${SLEDGE_DEPLOY_PACKAGE_DIR}/.venv/bin/python"

unset SLEDGE_DEPLOY_SCRIPT_DIR SLEDGE_NVIDIA_LIBS SLEDGE_TRT_LIBS

