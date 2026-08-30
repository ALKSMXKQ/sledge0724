#!/usr/bin/env bash
set -euo pipefail

# Rebuild the strict-additive pilot100 dataset:
# natural language -> EventFrame -> parameter template -> B1 edit
# -> typed, simulation-readable sledge_vector.gz.
#
# Set PILOT100_RUN_B2=1 to continue through the GPU diffusion stage. The
# default stops after the inspectable B0/B1 and simulation-cache stages.

REPO_ROOT="/home16T/home8T_1/leitingting/sledge_workspace/sledge"
WORKSPACE_ROOT="/home16T/home8T_1/leitingting/sledge_workspace"
SLEDGE_PYTHON="/home/leitingting/anaconda3/envs/sledge/bin/python"
INPUT_ROOT="${WORKSPACE_ROOT}/exp/caches/autoencoder_cache"
OUTPUT_ROOT="${WORKSPACE_ROOT}/exp/occluded_pedestrian_runs/pilot100_nuplan_types_v1"
SLEDGE_CONFIG="${WORKSPACE_ROOT}/semantic_img2img_cfg.yaml"
CONTROL_MODE="${PILOT100_CONTROL_MODE:-controlled}"
CANDIDATE_POOL_SIZE="${PILOT100_CANDIDATE_POOL_SIZE:-800}"
RUN_B2="${PILOT100_RUN_B2:-0}"

if [[ ! -x "${SLEDGE_PYTHON}" ]]; then
  echo "Missing SLEDGE Python: ${SLEDGE_PYTHON}" >&2
  exit 1
fi

if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "Missing input cache: ${INPUT_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${SLEDGE_CONFIG}" ]]; then
  echo "Missing SLEDGE config: ${SLEDGE_CONFIG}" >&2
  exit 1
fi

if [[ "${CONTROL_MODE}" != "controlled" && "${CONTROL_MODE}" != "prompt_only" ]]; then
  echo "PILOT100_CONTROL_MODE must be controlled or prompt_only" >&2
  exit 1
fi

# Refuse to mix a new run with a partial or existing batch.
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Output already exists: ${OUTPUT_ROOT}" >&2
  echo "Move or rename it before rebuilding; this script will not overwrite it." >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs"
export MPLCONFIGDIR="${OUTPUT_ROOT}/logs/matplotlib_cache"
mkdir -p "${MPLCONFIGDIR}"
cd "${REPO_ROOT}"

echo "[1/4] Running focused pipeline tests"
"${SLEDGE_PYTHON}" -m pytest -q \
  sledge/semantic_control/occluded_pedestrian_pipeline/tests \
  2>&1 | tee "${OUTPUT_ROOT}/logs/00_tests.log"

echo "[2/4] Building 100 strictly accepted B0/B1 pairs (${CONTROL_MODE})"
PYTHONUNBUFFERED=1 "${SLEDGE_PYTHON}" \
  -m sledge.semantic_control.occluded_pedestrian_pipeline.cli batch \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --profile pilot100 \
  --target-accepted 100 \
  --candidate-pool-size "${CANDIDATE_POOL_SIZE}" \
  --control-mode "${CONTROL_MODE}" \
  --llm-provider none \
  --accept-defaults \
  2>&1 | tee "${OUTPUT_ROOT}/logs/01_batch.log"

echo "[3/4] Exporting typed simulation gzip caches and previews"
PYTHONUNBUFFERED=1 "${SLEDGE_PYTHON}" \
  -m sledge.semantic_control.occluded_pedestrian_pipeline.cli export-b1 \
  --run-root "${OUTPUT_ROOT}" \
  --config "${SLEDGE_CONFIG}" \
  --limit 100 \
  2>&1 | tee "${OUTPUT_ROOT}/logs/02_export_b1.log"

echo "[4/4] Verifying counts and summaries"
export PILOT100_OUTPUT_ROOT="${OUTPUT_ROOT}"
"${SLEDGE_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PILOT100_OUTPUT_ROOT"])

counts = {
    "attempted_cases": sum(1 for line in (root / "manifests/cases.jsonl").open() if line.strip()),
    "b0_raw": len(list((root / "b0_original_cache").glob("*/sledge_raw.gz"))),
    "b1_raw": len(list((root / "b1_edited_cache").glob("*/sledge_raw.gz"))),
    "b1_vector": len(list((root / "b1_simulation_cache").glob("**/sledge_vector.gz"))),
}

b1_summary = json.loads((root / "manifests/b1_summary.json").read_text())
export_summary = json.loads(
    (root / "manifests/b1_simulation_cache_summary.json").read_text()
)

checks = {
    "b0_raw_100": counts["b0_raw"] == 100,
    "b1_raw_100": counts["b1_raw"] == 100,
    "b1_vector_100": counts["b1_vector"] == 100,
    "target_reached": b1_summary["target_reached"] is True,
    "accepted_100": b1_summary["accepted_count"] == 100,
    "b1_pass_100": b1_summary["b1"]["overall_pass_count"] == 100,
    "exported_100": export_summary["num_exported"] == 100,
    "type_round_trip_100": export_summary["gzip_round_trip_pass_count"] == 100,
}

print(json.dumps({"counts": counts, "checks": checks}, indent=2))

failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"pilot100 verification failed: {failed}")
PY

if [[ "${RUN_B2}" == "1" ]]; then
  echo "[R1] Running RVAE reconstruction and writing candidate/protected gzip caches"
  PYTHONUNBUFFERED=1 "${SLEDGE_PYTHON}" \
    -m sledge.semantic_control.occluded_pedestrian_pipeline.cli reconstruct \
    --run-root "${OUTPUT_ROOT}" \
    --config "${SLEDGE_CONFIG}" \
    --device cuda \
    --max-scenes 100 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/03_reconstruct_rvae.log"

  echo "[B2] Running strict semantic-protected half-denoise"
  PYTHONUNBUFFERED=1 "${SLEDGE_PYTHON}" \
    -m sledge.semantic_control.occluded_pedestrian_pipeline.cli refine \
    --run-root "${OUTPUT_ROOT}" \
    --config "${SLEDGE_CONFIG}" \
    --device cuda \
    --max-refine-scenes 100 \
    --mode semantic_protected \
    2>&1 | tee "${OUTPUT_ROOT}/logs/04_refine_b2.log"

  PYTHONUNBUFFERED=1 "${SLEDGE_PYTHON}" \
    -m sledge.semantic_control.occluded_pedestrian_pipeline.cli audit-gz \
    --run-root "${OUTPUT_ROOT}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/05_audit_gz.log"

  "${SLEDGE_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PILOT100_OUTPUT_ROOT"])
rvae = json.loads((root / "manifests/rvae_reconstruction_summary.json").read_text())
b2 = json.loads((root / "manifests/b2_semantic_protected_summary.json").read_text())
gzip_manifest = json.loads((root / "manifests/generated_gzip_stages.json").read_text())
checks = {
    "rvae_expected_100": rvae["num_expected"] == 100,
    "rvae_generated_100": rvae["num_generated"] == 100,
    "rvae_semantics_100": rvae["all_semantics_preserved"] is True,
    "b2_expected_100": b2["num_expected"] == 100,
    "b2_generated_100": b2["num_generated"] == 100,
    "b2_pass_100": b2["stage_metrics"]["overall_pass_count"] == 100,
    "all_gzip_stages_complete": gzip_manifest["all_complete"] is True,
}
print(json.dumps({"b2_checks": checks}, indent=2))
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"pilot100 B2 verification failed: {failed}")
PY
elif [[ "${RUN_B2}" != "0" ]]; then
  echo "PILOT100_RUN_B2 must be 0 or 1" >&2
  exit 1
fi

echo "pilot100 rebuild completed successfully"
echo "Output: ${OUTPUT_ROOT}"
echo "Simulation cache: ${OUTPUT_ROOT}/b1_simulation_cache"
echo "RVAE protected cache: ${OUTPUT_ROOT}/rvae_reconstruction/generated_cache"
echo "Diffusion protected cache: ${OUTPUT_ROOT}/b2_diffusion/semantic_protected/generated_cache"
echo "Gzip manifest: ${OUTPUT_ROOT}/manifests/generated_gzip_stages.csv"
echo "Type montage: ${OUTPUT_ROOT}/visualizations/b1_typed_cache/occluder_type_montage.png"
