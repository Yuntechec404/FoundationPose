#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import subprocess
import signal
import sys

# ==========================================
# 測試環境設定 (請依據實際環境修改)
# ==========================================
ROSBAG_PATH = "/home/user/catkin_ws/2026-01-08-14-00-59.bag" 
PACKAGE_NAME = "ros_foundationpose"
LAUNCH_FILE = "stem_pruning_benchmark.launch"
LOG_DIR = os.path.expanduser("/home/user/catkin_ws/src/FoundationPose/log/test_bunch_0512")

os.makedirs(LOG_DIR, exist_ok=True)
# ==========================================
# 定義要測試的模型組合清單
# pipeline_mode 說明:
# 1: YOLO + SAM + Cutie
# 2: YOLO + SAM2 + Cutie
# 3: YOLO + FastSAM + Cutie
# 4: YOLO + SAM2
# ==========================================
TEST_SUITES = [
    # 1: YOLO + SAM + Cutie
    {"mode": 1, "name": "YOLO_SAM_b_Cutie", "sam_ckpt": "/home/user/catkin_ws/src/FoundationPose/data/sam/sam_b.pt"},
    {"mode": 1, "name": "YOLO_SAM_l_Cutie", "sam_ckpt": "/home/user/catkin_ws/src/FoundationPose/data/sam/sam_l.pt"},
    
    # 2: YOLO + SAM2 + Cutie
    {"mode": 2, "name": "YOLO_SAM2_t_Cutie", "sam2_ckpt": "/home/user/catkin_ws/src/FoundationPose/data/sam2/sam2_t.pt"},
    {"mode": 2, "name": "YOLO_SAM2_l_Cutie", "sam2_ckpt": "/home/user/catkin_ws/src/FoundationPose/data/sam2/sam2_l.pt"},
    {"mode": 2, "name": "YOLO_SAM2.1_t_Cutie", "sam2_ckpt": "/home/user/catkin_ws/src/FoundationPose/data/sam2/sam2.1_t.pt"},
    {"mode": 2, "name": "YOLO_SAM2.1_l_Cutie", "sam2_ckpt": "/home/user/catkin_ws/src/FoundationPose/data/sam2/sam2.1_l.pt"},
    
    # 3: YOLO + FastSAM + Cutie
    {"mode": 3, "name": "YOLO_FastSAM-s_Cutie", "fastsam_ckpt": "/home/user/catkin_ws/src/FoundationPose/data/fastsam/FastSAM-s.pt"},
    {"mode": 3, "name": "YOLO_FastSAM-x_Cutie", "fastsam_ckpt": "/home/user/catkin_ws/src/FoundationPose/data/fastsam/FastSAM-x.pt"},

    # 4: YOLO + SAM2
    {"mode": 4, "name": "YOLO_SAM2_t", "sam2_ckpt": "/home/user/catkin_ws/src/FoundationPose/data/sam2/sam2_t.pt"},
    {"mode": 4, "name": "YOLO_SAM2_l", "sam2_ckpt": "/home/user/catkin_ws/src/FoundationPose/data/sam2/sam2_l.pt"},
    {"mode": 4, "name": "YOLO_SAM2.1_t", "sam2_ckpt": "/home/user/catkin_ws/src/FoundationPose/data/sam2/sam2.1_t.pt"},
    {"mode": 4, "name": "YOLO_SAM2.1_l", "sam2_ckpt": "/home/user/catkin_ws/src/FoundationPose/data/sam2/sam2.1_l.pt"},
]

def run_single_test(test_config):
    mode = test_config["mode"]
    test_name = f"Mode{mode}_{test_config['name']}"
    
    print("\n" + "="*60)
    print(f"開始測試: {test_name}")
    print(f"Pipeline Mode: {mode}")
    
    # 決定要傳入哪種 SAM 權重變數
    weight_arg = ""
    weight_val = ""
    if "sam_ckpt" in test_config:
        weight_arg = "sam_ckpt"
        weight_val = test_config["sam_ckpt"]
    elif "sam2_ckpt" in test_config:
        weight_arg = "sam2_ckpt"
        weight_val = test_config["sam2_ckpt"]
        
    if weight_val:
         print(f"   使用的權重 ({weight_arg}): {weight_val}")
    print("="*60)

    # 建立該次測試專屬的 log 資料夾
    test_log_dir = os.path.join(LOG_DIR, test_name)
    os.makedirs(test_log_dir, exist_ok=True)
    
    # 1. 準備 roslaunch 指令
    launch_cmd = [
        "roslaunch", PACKAGE_NAME, LAUNCH_FILE,
        f"pipeline_mode:={mode}",
        "perf_eval_enable:=true",
        f"log_root_dir:={test_log_dir}"
    ]
    
    if weight_arg and weight_val:
        launch_cmd.append(f"{weight_arg}:={weight_val}")

    # 啟動 Tracker Node (使用 preexec_fn=os.setsid 以便後續能乾淨擊殺整個 Process Group)
    print(f"[TEST] 執行 Launch: {' '.join(launch_cmd)}")
    tracker_process = subprocess.Popen(launch_cmd, preexec_fn=os.setsid)
    
    # 等待模型載入 (給予 GPU 15 秒的暖機時間)
    print("[TEST] 等待 15 秒讓模型載入 GPU 預熱...")
    time.sleep(15)

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
        os.killpg(os.getpgid(tracker_process.pid), signal.SIGINT)
        sys.exit(0)

    # 4. Rosbag 播完後清理
    print("[TEST] Rosbag 播放完畢，準備關閉 Tracker 並保存資料。")
    # 發送 SIGINT 給整個 process group，確保底層節點也被徹底關閉
    os.killpg(os.getpgid(tracker_process.pid), signal.SIGINT)
    
    try:
        tracker_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("[TEST] Tracker 未回應，強制擊殺 Process Group。")
        os.killpg(os.getpgid(tracker_process.pid), signal.SIGKILL)
        
    print(f"[TEST] {test_name} 測試結束。休息 5 秒釋放記憶體...\n")
    time.sleep(5)

if __name__ == "__main__":
    if not os.path.exists(ROSBAG_PATH):
        print(f"[ERROR] 找不到 Rosbag 檔案: {ROSBAG_PATH}")
        sys.exit(1)

    for config in TEST_SUITES:
        run_single_test(config)

    print("所有 Stem Tracking Benchmark 測試項目已順利執行完畢！")
