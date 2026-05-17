#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import time
import signal
import sys

# ==========================================
# 測試環境設定
# ==========================================
ROSBAG_PATH = "/home/user/catkin_ws/2026-01-08-14-00-59.bag" 
LAUNCH_PKG = "ros_foundationpose"
LAUNCH_FILE = "foundationpose_oilpalm_benchmark.launch"

# 模型路徑定義 (根據你的路徑設定)
DET_MODEL_DETECT = "/home/user/catkin_ws/src/FoundationPose/data/oilpalm_stem.onnx"
DET_MODEL_SEG = "/home/user/catkin_ws/src/FoundationPose/data/bunch_stem-seg.onnx"

# 定義要測試的模型組合清單
TEST_SUITES = [
    # 1. BBox
    {"seg_backend": "bbox", "det_model": DET_MODEL_DETECT, "sam_ckpt": ""},
    
    # 2. YOLO11-Seg (必須使用 Seg 模型)
    {"seg_backend": "yolo11-seg", "det_model": DET_MODEL_SEG, "sam_ckpt": ""},
    
    # 3. SAM
    {"seg_backend": "sam", "det_model": DET_MODEL_DETECT, "sam_ckpt": "/home/user/catkin_ws/src/FoundationPose/data/sam/sam_l.pt"},
    {"seg_backend": "sam", "det_model": DET_MODEL_DETECT, "sam_ckpt": "/home/user/catkin_ws/src/FoundationPose/data/sam/sam_b.pt"},
    
    # 4. SAM2
    {"seg_backend": "sam2", "det_model": DET_MODEL_DETECT, "sam_ckpt": "/home/user/catkin_ws/src/FoundationPose/data/sam2/sam2_l.pt"},
    {"seg_backend": "sam2", "det_model": DET_MODEL_DETECT, "sam_ckpt": "/home/user/catkin_ws/src/FoundationPose/data/sam2/sam2_t.pt"},
    
    {"seg_backend": "sam2", "det_model": DET_MODEL_DETECT, "sam_ckpt": "/home/user/catkin_ws/src/FoundationPose/data/sam2/sam2.1_l.pt"},
    {"seg_backend": "sam2", "det_model": DET_MODEL_DETECT, "sam_ckpt": "/home/user/catkin_ws/src/FoundationPose/data/sam2/sam2.1_t.pt"},

    # 5. FastSAM
    {"seg_backend": "fastsam", "det_model": DET_MODEL_DETECT, "sam_ckpt": "/home/user/catkin_ws/src/FoundationPose/data/fastsam/FastSAM-x.pt"},
    {"seg_backend": "fastsam", "det_model": DET_MODEL_DETECT, "sam_ckpt": "/home/user/catkin_ws/src/FoundationPose/data/fastsam/FastSAM-s.pt"},
]

def run_single_test(test_config):
    print("\n" + "="*60)
    print(f"🚀 開始測試 Backend: {test_config['seg_backend']}")
    if test_config['sam_ckpt']:
        print(f"   使用的權重: {os.path.basename(test_config['sam_ckpt'])}")
    print("="*60)

    # 1. 準備 roslaunch 指令
    launch_cmd = [
        "roslaunch", LAUNCH_PKG, LAUNCH_FILE,
        "perf_eval_enable:=true",
        f"seg_backend:={test_config['seg_backend']}",
        f"det_model:={test_config['det_model']}"
    ]
    # 動態傳入權重
    if test_config['sam_ckpt']:
        launch_cmd.append(f"sam_ckpt:={test_config['sam_ckpt']}")

    # 啟動 Tracker Node
    print("[TEST] 啟動 Tracker Node...")
    tracker_process = subprocess.Popen(launch_cmd)
    
    # 等待模型載入 (給予 GPU 30 秒的暖機時間)
    print("[TEST] 等待 30 秒讓模型載入 GPU 預熱...")
    time.sleep(30)

    # 2. 啟動 Rosbag
    print(f"[TEST] 開始播放 Rosbag: {ROSBAG_PATH}")
    bag_cmd = ["rosbag", "play", ROSBAG_PATH]
    bag_process = subprocess.Popen(bag_cmd)

    # 3. 等待 Rosbag 播放完畢
    try:
        bag_process.wait() # 程式會卡在這裡，直到 bag 播完
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

    for config in TEST_SUITES:
        run_single_test(config)

    print("🎉 所有 Benchmark 測試項目已順利執行完畢！")