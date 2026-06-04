#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import time
import signal
import sys
import itertools

# ==========================================
# 測試環境設定
# ==========================================
ROSBAG_PATH = "/home/user/catkin_ws/2026-01-08-14-00-59.bag"
LAUNCH_PKG = "ros_foundationpose"
LAUNCH_FILE = "foundationpose_oilpalm.launch"

# 模型路徑定義
DET_MODEL_DETECT = "/home/user/catkin_ws/src/FoundationPose/data/oilpalm_stem.onnx"

# ==========================================
# 固定分割模型：SAM2.1-L
# ==========================================
FIXED_SEG_BACKEND = "sam2"
FIXED_SAM_CKPT = "/home/user/catkin_ws/src/FoundationPose/data/sam2/sam2.1_l.pt"

# ==========================================
# 初始姿態估測 / Coarse Estimation 測試參數
# ==========================================
COARSE_MIN_N_VIEWS_LIST = [40]
COARSE_INPLANE_STEP_LIST = [40, 30]
# [60, 30]
COARSE_ORIENTATION_MODE_LIST = ["uniform", "inverted"]
COARSE_ORIENTATION_TILT_DEG_LIST = [90, 80, 70]
EST_REFINE_ITER_LIST = [5, 10, 2]

# 定義要測試的組合清單
TEST_SUITES = []
for min_n_views, inplane_step, orientation_mode, tilt_deg, est_refine_iter in itertools.product(
    COARSE_MIN_N_VIEWS_LIST,
    COARSE_INPLANE_STEP_LIST,
    COARSE_ORIENTATION_MODE_LIST,
    COARSE_ORIENTATION_TILT_DEG_LIST,
    EST_REFINE_ITER_LIST,
):
    TEST_SUITES.append({
        "seg_backend": FIXED_SEG_BACKEND,
        "det_model": DET_MODEL_DETECT,
        "sam_ckpt": FIXED_SAM_CKPT,
        "coarse_min_n_views": min_n_views,
        "coarse_inplane_step": inplane_step,
        "coarse_orientation_mode": orientation_mode,
        "coarse_orientation_tilt_deg": tilt_deg,
        "est_refine_iter": est_refine_iter,
    })


def run_single_test(test_config, test_idx, total_tests):
    print("\n" + "=" * 80)
    print(f"   開始測試 {test_idx}/{total_tests}")
    print(f"   seg_backend                  : {test_config['seg_backend']}")
    print(f"   sam_ckpt                     : {os.path.basename(test_config['sam_ckpt'])}")
    print(f"   coarse_min_n_views           : {test_config['coarse_min_n_views']}")
    print(f"   coarse_inplane_step          : {test_config['coarse_inplane_step']}")
    print(f"   coarse_orientation_mode      : {test_config['coarse_orientation_mode']}")
    print(f"   coarse_orientation_tilt_deg  : {test_config['coarse_orientation_tilt_deg']}")
    print(f"   est_refine_iter              : {test_config['est_refine_iter']}")
    print("=" * 80)

    # 1. 準備 roslaunch 指令
    launch_cmd = [
        "roslaunch", LAUNCH_PKG, LAUNCH_FILE,
        "perf_eval_enable:=true",

        # 固定使用 SAM2.1-L
        f"seg_backend:={test_config['seg_backend']}",
        f"det_model:={test_config['det_model']}",
        f"sam_ckpt:={test_config['sam_ckpt']}",

        # Coarse Estimation 參數
        f"coarse_min_n_views:={test_config['coarse_min_n_views']}",
        f"coarse_inplane_step:={test_config['coarse_inplane_step']}",
        f"coarse_orientation_mode:={test_config['coarse_orientation_mode']}",
        f"coarse_orientation_tilt_deg:={test_config['coarse_orientation_tilt_deg']}",

        # Initial pose refinement iteration
        f"est_refine_iter:={test_config['est_refine_iter']}",
    ]

    print("[TEST] roslaunch command:")
    print(" ".join(launch_cmd))

    # 啟動 Tracker Node
    print("[TEST] 啟動 Tracker Node...")
    tracker_process = subprocess.Popen(launch_cmd)

    # 等待模型載入
    print("[TEST] 等待 30 秒讓模型載入 GPU 預熱...")
    time.sleep(30)

    # 2. 啟動 Rosbag
    print(f"[TEST] 開始播放 Rosbag: {ROSBAG_PATH}")
    bag_cmd = ["rosbag", "play", ROSBAG_PATH]
    bag_process = subprocess.Popen(bag_cmd)

    # 3. 等待 Rosbag 播放完畢
    try:
        bag_process.wait()
    except KeyboardInterrupt:
        print("[TEST] 使用者中斷測試！")
        bag_process.terminate()
        tracker_process.terminate()
        sys.exit(0)

    # 4. Rosbag 播完後清理
    print("[TEST] Rosbag 播放完畢，準備關閉 Tracker 並保存 CSV。")
    tracker_process.send_signal(signal.SIGINT)

    try:
        tracker_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("[TEST] Tracker 未回應，強制擊殺。")
        tracker_process.kill()

    print("[TEST] 本輪測試結束。休息 5 秒釋放記憶體...\n")
    time.sleep(5)


if __name__ == "__main__":
    if not os.path.exists(ROSBAG_PATH):
        print(f"[ERROR] 找不到 Rosbag 檔案: {ROSBAG_PATH}")
        sys.exit(1)

    total = len(TEST_SUITES)
    print(f"[INFO] 總測試組合數: {total}")
    print("[INFO] 固定使用 seg_backend=sam2, sam_ckpt=sam2.1_l.pt")
    print("[INFO] 測試參數：")
    print(f"       coarse_min_n_views          = {COARSE_MIN_N_VIEWS_LIST}")
    print(f"       coarse_inplane_step         = {COARSE_INPLANE_STEP_LIST}")
    print(f"       coarse_orientation_mode     = {COARSE_ORIENTATION_MODE_LIST}")
    print(f"       coarse_orientation_tilt_deg = {COARSE_ORIENTATION_TILT_DEG_LIST}")
    print(f"       est_refine_iter             = {EST_REFINE_ITER_LIST}")

    for idx, config in enumerate(TEST_SUITES, start=1):
        run_single_test(config, idx, total)

    print("所有測試項目已順利執行完畢！")
