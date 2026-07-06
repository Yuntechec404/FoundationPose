#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run FoundationPose on BOP LM and LMO.
# ============================================================

FP_DIR="${FP_DIR:-/home/user/FoundationPose}"
PY_SCRIPT="${PY_SCRIPT:-${FP_DIR}/run_linemod_sam2_bbox.py}"
CONDA_ENV="${CONDA_ENV:-foundationpose}"

BOP_ROOT="${BOP_ROOT:-${FP_DIR}/demo_data/bop}"
LM_DIR="${LM_DIR:-${BOP_ROOT}/lm}"
LMO_DIR="${LMO_DIR:-${BOP_ROOT}/lmo}"

OUT_ROOT="${OUT_ROOT:-${FP_DIR}/debug/debug_lmo_gt_trans_80}"
LOG_ROOT="${LOG_ROOT:-${FP_DIR}/debug/logs_lmo_gt_trans_80}"

MAX_FRAMES_PER_OBJ="${MAX_FRAMES_PER_OBJ:-80}"
TOP_K="${TOP_K:-50}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

# ------------------------------------------------------------
# Activate Conda.
# ------------------------------------------------------------
if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] Cannot find conda.sh"
  exit 1
fi

conda activate "${CONDA_ENV}"

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "[ERROR] CONDA_PREFIX is empty after activating ${CONDA_ENV}"
  exit 1
fi

# Conda C++ runtime must precede Ubuntu's system libstdc++.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH="${CUDA_HOME}/bin:${CONDA_PREFIX}/bin:${PATH}"

cd "${FP_DIR}"

# ------------------------------------------------------------
# Preflight checks.
# ------------------------------------------------------------
if [ ! -f "${PY_SCRIPT}" ]; then
  echo "[ERROR] Python runner not found: ${PY_SCRIPT}"
  exit 1
fi

for dataset_dir in "${LM_DIR}" "${LMO_DIR}"; do
  if [ ! -d "${dataset_dir}/test" ]; then
    echo "[ERROR] BOP test directory not found: ${dataset_dir}/test"
    exit 1
  fi
done

echo "[INFO] Python       : $(command -v python)"
echo "[INFO] Conda env     : ${CONDA_ENV}"
echo "[INFO] CONDA_PREFIX  : ${CONDA_PREFIX}"
echo "[INFO] FoundationPose: ${FP_DIR}"
echo "[INFO] Runner        : ${PY_SCRIPT}"
echo "[INFO] Output root   : ${OUT_ROOT}"
echo "[INFO] Log root      : ${LOG_ROOT}"
echo "[INFO] max frames/obj: ${MAX_FRAMES_PER_OBJ}"
echo "[INFO] mask sources  : gt"
echo "[INFO] refine modes  : none,icp,ndt,gicp,vgicp"
echo "[INFO] refine apply  : trans"

python - <<'PY'
import cv2
import torch

print("[CHECK] cv2 version       :", cv2.__version__)
print("[CHECK] torch version     :", torch.__version__)
print("[CHECK] torch CUDA        :", torch.version.cuda)
print("[CHECK] CUDA available    :", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")
print("[CHECK] CUDA device       :", torch.cuda.get_device_name(0))
PY

run_dataset() {
  local dataset_name="$1"
  local dataset_dir="$2"
  local output_dir="${OUT_ROOT}/${dataset_name}"
  local log_file="${LOG_ROOT}/${dataset_name}.log"

  mkdir -p "${output_dir}"

  echo ""
  echo "============================================================"
  echo "[START] Dataset : ${dataset_name}"
  echo "[DATA ]         : ${dataset_dir}"
  echo "[DEBUG]         : ${output_dir}"
  echo "[LOG  ]         : ${log_file}"
  echo "============================================================"

  python "${PY_SCRIPT}" \
    --linemod_dir "${dataset_dir}" \
    --debug 0 \
    --debug_dir "${output_dir}" \
    --obj_ids all \
    --max_frames_per_obj "${MAX_FRAMES_PER_OBJ}" \
    --mask_sources gt \
    --depth_refine_modes none,icp,ndt,gicp,vgicp \
    --depth_refine_apply trans \
    --top_k "${TOP_K}" \
    --top_flag \
    2>&1 | tee "${log_file}"

  echo ""
  echo "[DONE] Dataset : ${dataset_name}"
  echo "[OUT ]         : ${output_dir}"
  echo "[LOG ]         : ${log_file}"
}

# run_dataset "lm" "${LM_DIR}"
run_dataset "lmo" "${LMO_DIR}"

echo ""
echo "============================================================"
echo "All LM and LMO runs finished."
echo "Results: ${OUT_ROOT}"
echo "Logs   : ${LOG_ROOT}"
echo "============================================================"
