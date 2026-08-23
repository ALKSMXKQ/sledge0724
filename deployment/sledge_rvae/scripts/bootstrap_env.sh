#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${PACKAGE_DIR}/../.." && pwd)
BASE_PYTHON=${BASE_PYTHON:-python}
VENV_DIR=${VENV_DIR:-"${PACKAGE_DIR}/.venv"}

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install -r "${PACKAGE_DIR}/requirements-export.txt"
"${VENV_DIR}/bin/python" -m pip install -r "${PACKAGE_DIR}/requirements-target.txt"
cd "${REPO_ROOT}"
source "${PACKAGE_DIR}/scripts/activate_deploy.sh"
MPLCONFIGDIR=/tmp/sledge_deploy_mpl "${VENV_DIR}/bin/python" -m deployment.sledge_rvae.python.check_environment

echo "Deployment environment ready: ${VENV_DIR}"
echo "Activate with: source ${PACKAGE_DIR}/scripts/activate_deploy.sh"
