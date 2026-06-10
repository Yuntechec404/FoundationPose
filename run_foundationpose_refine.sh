#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Step 1: Run FoundationPose only
# Output: BOP CSVs under OUT_ROOT/<experiment>/
# Later use eval_bop_depth_refine_all.sh to calculate BOP scores.
# ============================================================

# ---------- Paths ----------
FP_DIR="/home/user/FoundationPose"
BOP_ROOT="${FP_DIR}/demo_data/bop"
REFINER_CFG="${FP_DIR}/weights/2023-10-28-18-33-37/config.yml"
OUT_ROOT="${FP_DIR}/debug/depth_refine_auto"

# ---------- Conda env ----------
FP_ENV="foundationpose"

# ---------- FoundationPose ----------
BOP_TIME_MODE="sum"

# ---------- Depth refine common parameters ----------
DEPTH_REFINE_ACCEPT_IF_BETTER="True"
DEPTH_REFINE_MAX_POINTS=2048
DEPTH_REFINE_MIN_POINTS=50
DEPTH_REFINE_MAX_CORR_DIST=0.015
DEPTH_REFINE_ICP_ITER=2
DEPTH_REFINE_DEPTH_DIFF_THRESH=0.02
DEPTH_REFINE_TRANS_CLAMP=0.005
DEPTH_REFINE_ROT_CLAMP_DEG=5.0
DEPTH_REFINE_LOG="True"

# ---------- Conda activate helper ----------
if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] Cannot find conda.sh. Please edit this script and source conda.sh manually."
  exit 1
fi

mkdir -p "${OUT_ROOT}"

backup_cfg_once() {
  if [ ! -f "${REFINER_CFG}.bak_depth_refine" ]; then
    cp "${REFINER_CFG}" "${REFINER_CFG}.bak_depth_refine"
    echo "[INFO] Backup config saved to ${REFINER_CFG}.bak_depth_refine"
  fi
}

set_refiner_config() {
  local mode="$1"   # none / icp / ndp
  local apply="$2"  # trans / se3

  python - <<PY
from pathlib import Path
import yaml

cfg_path = Path(r"${REFINER_CFG}")
with cfg_path.open("r") as f:
    cfg = yaml.safe_load(f)
if cfg is None:
    cfg = {}

# Keep checkpoint-compatible neural representation.
cfg["rot_rep"] = "axis_angle"
cfg["trans_rep"] = "tracknet"

# Test-time depth geometry refinement.
cfg["depth_refine_mode"] = "${mode}"
cfg["depth_refine_apply"] = "${apply}"
cfg["depth_refine_accept_if_better"] = ${DEPTH_REFINE_ACCEPT_IF_BETTER}
cfg["depth_refine_max_points"] = ${DEPTH_REFINE_MAX_POINTS}
cfg["depth_refine_min_points"] = ${DEPTH_REFINE_MIN_POINTS}
cfg["depth_refine_max_corr_dist"] = ${DEPTH_REFINE_MAX_CORR_DIST}
cfg["depth_refine_icp_iter"] = ${DEPTH_REFINE_ICP_ITER}
cfg["depth_refine_depth_diff_thresh"] = ${DEPTH_REFINE_DEPTH_DIFF_THRESH}
cfg["depth_refine_trans_clamp"] = ${DEPTH_REFINE_TRANS_CLAMP}
cfg["depth_refine_rot_clamp_deg"] = ${DEPTH_REFINE_ROT_CLAMP_DEG}
cfg["depth_refine_log"] = ${DEPTH_REFINE_LOG}

with cfg_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"[CONFIG] depth_refine_mode={cfg['depth_refine_mode']}, depth_refine_apply={cfg['depth_refine_apply']}")
print(f"[CONFIG] rot_rep={cfg['rot_rep']}, trans_rep={cfg['trans_rep']}")
PY
}

run_fp_one() {
  local dataset="$1"   # lm / lmo
  local mode="$2"      # none / icp / ndp
  local apply="$3"     # trans / se3
  local exp_name="$4"

  local dataset_dir="${BOP_ROOT}/${dataset}"
  local out_dir="${OUT_ROOT}/${exp_name}"
  local csv_name="foundationpose_${dataset}-test.csv"

  echo ""
  echo "============================================================"
  echo "[FOUNDATIONPOSE] ${exp_name}"
  echo "[FOUNDATIONPOSE] dataset=${dataset}, mode=${mode}, apply=${apply}"
  echo "[FOUNDATIONPOSE] output=${out_dir}"
  echo "============================================================"

  mkdir -p "${out_dir}"

  conda activate "${FP_ENV}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  export BOP_DIR="${BOP_ROOT}"
  export BOP_PATH="${BOP_ROOT}"

  set_refiner_config "${mode}" "${apply}"

  cd "${FP_DIR}"
  python run_linemod.py \
    --linemod_dir "${dataset_dir}" \
    --debug 0 \
    --debug_dir "${out_dir}" \
    --bop_time_mode "${BOP_TIME_MODE}"

  if [ ! -f "${out_dir}/${csv_name}" ]; then
    echo "[ERROR] Missing BOP CSV: ${out_dir}/${csv_name}"
    exit 1
  fi

  echo "[DONE] FoundationPose finished: ${exp_name}"
  echo "[CSV] ${out_dir}/${csv_name}"
}

main() {
  backup_cfg_once

  # LMO: occluded dataset. Baseline + SE(3) tests.
  # run_fp_one "lmo" "none" "trans" "lmo_tracknet_axis_angle_none"
  run_fp_one "lmo" "icp"  "se3"   "lmo_tracknet_axis_angle_icp_se3"
  run_fp_one "lmo" "ndp"  "se3"   "lmo_tracknet_axis_angle_ndp_se3"

  # LM: less occlusion. Translation-only and SE(3) tests.
  run_fp_one "lm" "none" "trans" "lm_tracknet_axis_angle_none"
  run_fp_one "lm"  "icp"  "trans" "lm_tracknet_axis_angle_icp_trans"
  run_fp_one "lm"  "ndp"  "trans" "lm_tracknet_axis_angle_ndp_trans"
  run_fp_one "lm"  "icp"  "se3"   "lm_tracknet_axis_angle_icp_se3"
  run_fp_one "lm"  "ndp"  "se3"   "lm_tracknet_axis_angle_ndp_se3"

  echo ""
  echo "All FoundationPose runs finished."
  echo "Next: run eval_bop_depth_refine_all.sh in bop_toolkit env."
}

main "$@"
