#!/usr/bin/env bash
set -Eeuo pipefail

# Run all custom BOP experiments sequentially.
# Execute this script from any directory.
#
# Usage:
#   chmod +x run_all_custom_bop_sam2.sh
#   ./run_all_custom_bop_sam2.sh
#
# Run after activating the FoundationPose Conda environment:
#   conda activate foundationpose

FP_ROOT="/home/user/FoundationPose"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROGRAM="${FP_ROOT}/run_custom_bop_sam2_bbox.py"
DATA_ROOT="${FP_ROOT}/demo_data/custom_datasets"
DEBUG_ROOT="${FP_ROOT}/debug"
SAM_CKPT="${FP_ROOT}/sam2.1_l.pt"
LOG_ROOT="${DEBUG_ROOT}/logs_custom_bop"

mkdir -p "${DEBUG_ROOT}" "${LOG_ROOT}"

if [[ ! -f "${PROGRAM}" ]]; then
    echo "[ERROR] Program not found: ${PROGRAM}" >&2
    exit 1
fi

if [[ ! -f "${SAM_CKPT}" ]]; then
    echo "[ERROR] SAM2 checkpoint not found: ${SAM_CKPT}" >&2
    exit 1
fi

echo "[INFO] Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "[INFO] FoundationPose root: ${FP_ROOT}"
echo "[INFO] SAM2 checkpoint: ${SAM_CKPT}"

COMMON_ARGS=(
    --scene_ids all
    --custom_model_unit auto
    --max_frames_per_obj 80
    --mask_sources gt
    --depth_refine_modes none,icp,ndt,gicp,vgicp
    --depth_refine_apply trans
    --sam_ckpt "${SAM_CKPT}"
    --sam_imgsz 640
)

run_dataset() {
    local dataset_name="$1"
    local debug_name="$2"

    local dataset_dir="${DATA_ROOT}/${dataset_name}"
    local debug_dir="${DEBUG_ROOT}/${debug_name}"
    local log_file="${LOG_ROOT}/${dataset_name}.log"

    if [[ ! -d "${dataset_dir}" ]]; then
        echo "[ERROR] Dataset directory not found: ${dataset_dir}" >&2
        return 1
    fi

    mkdir -p "${debug_dir}"

    echo
    echo "============================================================"
    echo "[START] Dataset: ${dataset_name}"
    echo "[DATA ] ${dataset_dir}"
    echo "[DEBUG] ${debug_dir}"
    echo "[LOG  ] ${log_file}"
    echo "============================================================"

    "${PYTHON_BIN}" "${PROGRAM}" \
        --linemod_dir "${dataset_dir}" \
        --debug_dir "${debug_dir}" \
        "${COMMON_ARGS[@]}" \
        2>&1 | tee "${log_file}"

    echo "[DONE] Dataset: ${dataset_name}"
}

cd "${FP_ROOT}"

run_dataset "uneven_illumination_2" "debug_uneven_illumination_2"
run_dataset "partial_occlusion_2"   "debug_partial_occlusion_2"
run_dataset "dynamic_target_2"      "debug_dynamic_target_2"
run_dataset "complex_background_2"  "debug_complex_background_2"

echo
echo "============================================================"
echo "[DONE] All custom BOP experiments completed successfully."
echo "[LOGS] ${LOG_ROOT}"
echo "============================================================"
