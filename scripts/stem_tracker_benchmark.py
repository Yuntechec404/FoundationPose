#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, csv
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import rospy
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as F

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import message_filters
import tf
from ultralytics import YOLO, SAM, FastSAM
from ultralytics.models.sam import SAM2DynamicInteractivePredictor

CUTIE_PATH = os.path.expanduser("~/Cutie") 
if CUTIE_PATH not in sys.path:
    sys.path.append(CUTIE_PATH)

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

class StemPoseEstimatorBenchmark:
    def __init__(self):
        rospy.init_node('stem_tracker_benchmark', anonymous=True)
        self.bridge = CvBridge()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # ROS 參數
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/color/camera_info')
        self.image_raw_topic = rospy.get_param('~image_raw_topic', '/camera/color/image_raw')
        self.depth_topic = rospy.get_param('~depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.camera_tf = rospy.get_param('~camera_tf', 'camera_color_optical_frame')
        self.child_tf = rospy.get_param('~child_tf', 'stem_estimated_frame')
        self.pose_topic = rospy.get_param('~pose_topic', '/stem/pruning_pose')
        self.pose_vis_topic = rospy.get_param('~pose_vis_topic', '/stem/debug_vis_image')
        
        self.det_select_mode = rospy.get_param("~det_select_mode", "nearest_depth")
        self.pipeline_mode = rospy.get_param("~pipeline_mode", 1)
        self.perf_eval_enable = rospy.get_param("~perf_eval_enable", False)
        self.log_root_dir = rospy.get_param("~log_root_dir", "/tmp/fp_benchmark")
        
        self.sam_ckpt = rospy.get_param("~sam_ckpt", "sam_b.pt")
        self.fastsam_ckpt = rospy.get_param("~fastsam_ckpt", "FastSAM-s.pt") # 【修正】：補回參數
        self.sam2_ckpt = rospy.get_param("~sam2_ckpt", "sam2_b.pt")
        self.model_path = rospy.get_param("~model_path", "yolov11n.pt")
        
        self.target_cls = rospy.get_param("~target_cls", 0)
        self.conf_thresh = rospy.get_param("~conf_thresh", 0.3)
        self.erode_iterations = rospy.get_param("~erode_iterations", 2)
        self.depth_min = rospy.get_param("~depth_min", 0.01) 
        self.depth_max = rospy.get_param("~depth_max", 3.0) 
        self.assumed_stem_radius = rospy.get_param("~assumed_stem_radius", 0.0)
        self.erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        self.mode_names = {
            1: "YOLO + SAM + Cutie",
            2: "YOLO + SAM2 + Cutie",
            3: "YOLO + FastSAM + Cutie",
            4: "YOLO + SAM2",
        }
        self.mode_str = self.mode_names.get(self.pipeline_mode, f"Unknown Mode {self.pipeline_mode}")

        self.K_ready = False
        self.fx = self.fy = self.cx = self.cy = 0.0
        self.latest_color_img = None
        self.latest_depth_img = None
        self.latest_header = None
        self.new_frame_available = False
        
        # --- Benchmark 變數 ---
        self.track_start_time = 0.0
        self.cycle_count = 1
        self.save_first_pose_flag = False
        self.latest_pose_img = None

        self.cv_window_name = f"Benchmark: {self.mode_str}"
        cv2.namedWindow(self.cv_window_name, cv2.WINDOW_NORMAL)
        
        self.csv_file = None
        self.csv_writer = None
        self._setup_run_debug_dir()

        # 1. 載入 YOLO
        rospy.loginfo(f"載入 YOLO 模型: {self.model_path}")
        self.yolo = YOLO(self.model_path, task="detect")
        
        # 2. 載入 SAM 家族
        self.prompt_model = None
        if self.pipeline_mode == 1:
            rospy.loginfo(f"載入 SAM: {self.sam_ckpt}")
            self.prompt_model = SAM(self.sam_ckpt)
        elif self.pipeline_mode == 2:
            rospy.loginfo(f"載入 SAM2: {self.sam2_ckpt}")
            self.prompt_model = SAM(self.sam2_ckpt)
        elif self.pipeline_mode == 3:
            rospy.loginfo(f"載入 FastSAM: {self.fastsam_ckpt}")
            self.prompt_model = FastSAM(self.fastsam_ckpt)
        elif self.pipeline_mode == 4:
            rospy.loginfo(f"載入 SAM2 Dynamic Interactive Predictor: {self.sam2_ckpt}")
            overrides = dict(
                conf=0.01,
                task="segment",
                mode="predict",
                imgsz=1024,
                model=self.sam2_ckpt,
                save=False
            )
            self.prompt_model = SAM2DynamicInteractivePredictor(
                overrides=overrides,
                max_obj_num=3
            )
            
        if self.prompt_model and self.pipeline_mode != 4:
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.prompt_model(dummy_img, bboxes=[[10,10,100,100]], imgsz=640, device=self.device, verbose=False)
        
        # 3. 載入 Cutie
        if self.pipeline_mode in [1, 2, 3]:
            rospy.loginfo("載入 Cutie 模型...")
            network = get_default_model()
            network.eval().to(self.device)
            self.processor = InferenceCore(network, cfg=network.cfg)
            self.processor.max_internal_size = 480 
        
        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=1)
        self.vis_pub = rospy.Publisher(self.pose_vis_topic, Image, queue_size=1)
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_cb)
        color_sub = message_filters.Subscriber(self.image_raw_topic, Image)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=2, slop=0.05)
        self.ts.registerCallback(self.sync_callback)
        rospy.loginfo("系統已啟動，等待相機影像...")

    def _setup_run_debug_dir(self):
        if not self.perf_eval_enable: return
        os.makedirs(self.log_root_dir, exist_ok=True)
        self.csv_log_path = os.path.join(self.log_root_dir, "perf_eval.csv")
        self.csv_file = open(self.csv_log_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'Timestamp', 'Mode', 'Total_Time(ms)', 'Detect_Time(ms)', 
            'Seg_Time(ms)', 'Track_Time(ms)', 'Depth_Filter_Time(ms)', 'Root_Z(m)'
        ])

    def camera_info_cb(self, msg):
        if not self.K_ready:
            self.fx = msg.K[0]; self.cx = msg.K[2]
            self.fy = msg.K[4]; self.cy = msg.K[5]
            self.K_ready = True

    def sync_callback(self, color_msg, depth_msg):
        if not self.K_ready: return
        try:
            self.latest_color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            self.latest_depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            self.latest_header = depth_msg.header
            self.new_frame_available = True
        except Exception: pass

    def cv2_to_tensor(self, img_bgr):
        return F.to_tensor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).to(self.device)

    def get_best_yolo_bbox(self, r, target_indices, W, H, depth_img):
        best_idx = None
        if self.det_select_mode == "middle":
            min_dist = float('inf')
            cx_img, cy_img = W / 2.0, H / 2.0
            for idx in target_indices:
                box = r.boxes.xywh[idx].cpu().numpy()
                dist = (box[0] - cx_img)**2 + (box[1] - cy_img)**2
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
        elif self.det_select_mode == "nearest_depth":
            min_depth = float('inf')
            temp_depth = depth_img.copy()
            temp_depth[temp_depth == 0] = np.max(temp_depth)
            temp_depth = cv2.GaussianBlur(temp_depth, (3, 3), 0)

            for idx in target_indices:
                box = r.boxes.xyxy[idx].cpu().numpy().astype(int)
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
                roi_depth = temp_depth[y1:y2, x1:x2] / 1000.0
                valid_z = roi_depth[(roi_depth > self.depth_min) & (roi_depth < self.depth_max)]
                if len(valid_z) > 0:
                    med_z = np.median(valid_z)
                    if med_z < min_depth:
                        min_depth = med_z
                        best_idx = idx
            if best_idx is None:
                best_idx = target_indices[0]
        else:
            best_conf = -1.0
            for idx in target_indices:
                conf = float(r.boxes.conf[idx].cpu().numpy())
                if conf > best_conf:
                    best_conf = conf
                    best_idx = idx
        return best_idx

    def run_spin(self):
        STATE_SEARCHING = 0
        STATE_TRACKING = 1
        current_state = STATE_SEARCHING
        
        while not rospy.is_shutdown():
            if not self.new_frame_available or self.latest_color_img is None:
                cv2.waitKey(1); continue
            
            color_img = self.latest_color_img.copy()
            depth_img = self.latest_depth_img.copy()
            header = self.latest_header
            self.new_frame_available = False
            
            vis_img = color_img.copy()
            H, W = color_img.shape[:2]
            mask_img = None
            
            t_total_start = time.perf_counter()
            t_det, t_seg, t_track, t_depth = 0.0, 0.0, 0.0, 0.0

            if current_state == STATE_TRACKING and (time.time() - self.track_start_time) > 5.0:
                rospy.loginfo(f"穩定追蹤超過 5 秒，強制進行第 {self.cycle_count+1} 次重置檢測！")
                
                if self.latest_pose_img is not None and self.perf_eval_enable:
                    timestamp_str = datetime.now().strftime("%H%M%S")
                    save_path = os.path.join(self.log_root_dir, f"cycle{self.cycle_count}_pose_last_{timestamp_str}.jpg")
                    cv2.imwrite(save_path, self.latest_pose_img)
                    rospy.loginfo(f"[TEST] 已儲存第 {self.cycle_count} 週期的最後一幀影像: {save_path}")

                if self.pipeline_mode in [1, 2, 3]:
                    self.processor.clear_memory()
                elif self.pipeline_mode == 4:
                    try: self.prompt_model.reset_memory()
                    except: pass
                        
                torch.cuda.empty_cache()
                current_state = STATE_SEARCHING
                self.cycle_count += 1

            with torch.inference_mode(), torch.cuda.amp.autocast():
                
                if current_state == STATE_SEARCHING:
                    t_det_start = time.perf_counter()
                    results = self.yolo.predict(color_img, imgsz=640, conf=self.conf_thresh, device=self.device, verbose=False)
                    t_det = (time.perf_counter() - t_det_start) * 1000.0
                    
                    best_mask_uint8 = None

                    if results and len(results[0].boxes) > 0:
                        r = results[0]
                        classes = r.boxes.cls.cpu().numpy()
                        target_indices = np.where(classes == self.target_cls)[0]
                        
                        if len(target_indices) > 0:
                            best_idx = self.get_best_yolo_bbox(r, target_indices, W, H, depth_img)
                            bbox_xyxy = r.boxes.xyxy[best_idx].cpu().numpy() 
                            x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
                            
                            t_seg_start = time.perf_counter()
                            if self.prompt_model is not None:
                                if self.pipeline_mode == 4:
                                    seg_results = self.prompt_model(
                                        source=color_img,
                                        bboxes=[[x1, y1, x2, y2]],
                                        obj_ids=[1],
                                        update_memory=True
                                    )
                                else:
                                    # Mode 1, 2, 3
                                    seg_results = self.prompt_model(color_img, bboxes=[[x1, y1, x2, y2]], imgsz=640, device=self.device, verbose=False)
                                
                                if seg_results and seg_results[0].masks is not None:
                                    m = seg_results[0].masks.data.cpu().numpy()
                                    if m.size > 0 and m.shape[0] > 0:
                                        m2 = m[0] if m.ndim == 3 else m
                                        if m2.shape != (H, W):
                                            m2 = cv2.resize(m2.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                                        best_mask_uint8 = (m2 > 0).astype(np.uint8)
                                    else:
                                        rospy.logwarn(f"[{self.mode_str}] 初始化階段未檢測到有效 Mask (陣列為空)")
                                        
                            t_seg = (time.perf_counter() - t_seg_start) * 1000.0

                            if self.perf_eval_enable and best_mask_uint8 is not None:
                                timestamp_str = datetime.now().strftime("%H%M%S")
                                bbox_img = color_img.copy()
                                cv2.rectangle(bbox_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.imwrite(os.path.join(self.log_root_dir, f"cycle{self.cycle_count}_bbox_{timestamp_str}.jpg"), bbox_img)
                                
                                seg_img = color_img.copy()
                                seg_img[best_mask_uint8 > 0] = [0, 255, 0]
                                cv2.imwrite(os.path.join(self.log_root_dir, f"cycle{self.cycle_count}_seg_{timestamp_str}.jpg"), seg_img)

                    if best_mask_uint8 is not None:
                        mask_img = best_mask_uint8 * 255
                        
                        if self.pipeline_mode in [1, 2, 3]:
                            init_mask = torch.from_numpy(best_mask_uint8).to(self.device).long()
                            img_tensor = self.cv2_to_tensor(color_img)
                            self.processor.clear_memory()
                            self.processor.step(img_tensor, init_mask, objects=[1])
                            
                        current_state = STATE_TRACKING
                        self.track_start_time = time.time()
                        self.save_first_pose_flag = True 
                        cv2.putText(vis_img, f"Init: {self.mode_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        cv2.putText(vis_img, f"Searching... {self.mode_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        
                elif current_state == STATE_TRACKING:
                    tracked_mask = None
                    
                    if self.pipeline_mode in [1, 2, 3]:
                        t_track_start = time.perf_counter()
                        img_tensor = self.cv2_to_tensor(color_img)
                        output_prob = self.processor.step(img_tensor)
                        pred_mask_tensor = self.processor.output_prob_to_mask(output_prob)
                        raw_tracked_mask = (pred_mask_tensor.cpu().numpy() == 1).astype(np.uint8)
                        tracked_mask = cv2.erode(raw_tracked_mask, self.erode_kernel, iterations=self.erode_iterations)
                        t_track = (time.perf_counter() - t_track_start) * 1000.0
                    
                    elif self.pipeline_mode == 4 and self.prompt_model is not None:
                        t_track_start = time.perf_counter()
                        track_results = self.prompt_model(source=color_img)

                        if track_results and track_results[0].masks is not None:
                            m = track_results[0].masks.data.cpu().numpy()
                            if m.size > 0 and m.shape[0] > 0:
                                m2 = m[0] if m.ndim == 3 else m
                                if m2.shape != (H, W):
                                    m2 = cv2.resize(m2.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                                tracked_mask = (m2 > 0).astype(np.uint8)
                                tracked_mask = cv2.erode(tracked_mask, self.erode_kernel, iterations=self.erode_iterations)
                            else:
                                rospy.logwarn(f"[{self.mode_str}] 連續追蹤丟失目標 (Mask陣列為空)")
                                tracked_mask = None 
                                
                        t_track = (time.perf_counter() - t_track_start) * 1000.0

                    if tracked_mask is None or tracked_mask.sum() < 50:
                        if self.pipeline_mode in [1, 2, 3]: self.processor.clear_memory()
                        elif self.pipeline_mode == 4:
                            try: self.prompt_model.reset_memory()
                            except: pass
                        
                        current_state = STATE_SEARCHING
                        mask_img = None
                    else:
                        colored_mask = np.zeros_like(vis_img)
                        colored_mask[tracked_mask > 0] = [0, 0, 255]
                        cv2.addWeighted(vis_img, 1.0, colored_mask, 0.5, 0, vis_img)
                        mask_img = tracked_mask * 255
                        cv2.putText(vis_img, f"Tracking: {self.mode_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # ==================================================
            # 計算 3D 深度與濾波耗時
            # ==================================================
            target_z = 0.0
            if mask_img is not None:
                t_depth_start = time.perf_counter()
                center_3d, quat, projected_2d, status = self.calculate_direct_root_pose(depth_img, mask_img)
                t_depth = (time.perf_counter() - t_depth_start) * 1000.0

                if center_3d is not None:
                    target_z = center_3d[2]
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = header.stamp
                    pose_msg.header.frame_id = self.camera_tf 
                    pose_msg.pose.position.x = center_3d[0]
                    pose_msg.pose.position.y = center_3d[1]
                    pose_msg.pose.position.z = center_3d[2]
                    pose_msg.pose.orientation.x = quat[0]
                    pose_msg.pose.orientation.y = quat[1]
                    pose_msg.pose.orientation.z = quat[2]
                    pose_msg.pose.orientation.w = quat[3]
                    
                    self.pose_pub.publish(pose_msg)
                    self.tf_broadcaster.sendTransform(
                        (center_3d[0], center_3d[1], center_3d[2]),
                        (quat[0], quat[1], quat[2], quat[3]),
                        header.stamp, self.child_tf, self.camera_tf
                    )
                    
                    cv2.circle(vis_img, projected_2d, 8, (255, 0, 0), -1) 
                    cv2.putText(vis_img, f"Z: {target_z:.3f}m", (projected_2d[0] + 15, projected_2d[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) 
                    
                    self.latest_pose_img = vis_img.copy()
                    
                    if self.save_first_pose_flag and self.perf_eval_enable:
                        timestamp_str = datetime.now().strftime("%H%M%S")
                        cv2.imwrite(os.path.join(self.log_root_dir, f"cycle{self.cycle_count}_pose_first_{timestamp_str}.jpg"), vis_img)
                        self.save_first_pose_flag = False

            # --- 計算總時長與寫入 CSV ---
            t_total = (time.perf_counter() - t_total_start) * 1000.0
            
            if self.perf_eval_enable and self.csv_file is not None and not self.csv_file.closed:
                self.csv_writer.writerow([
                    header.stamp.to_sec(), self.mode_str, f"{t_total:.2f}", 
                    f"{t_det:.2f}", f"{t_seg:.2f}", f"{t_track:.2f}", f"{t_depth:.2f}", f"{target_z:.4f}"
                ])

            cv2.imshow(self.cv_window_name, vis_img)
            try: self.vis_pub.publish(self.bridge.cv2_to_imgmsg(vis_img, "bgr8"))
            except Exception: pass
            cv2.waitKey(1)

    def calculate_direct_root_pose(self, depth_img, mask):
        filtered_depth = depth_img.copy()
        filtered_depth = cv2.GaussianBlur(filtered_depth, (3, 3), 0)

        v, u = np.where(mask > 0)
        if len(v) == 0: return None, None, None, "No Mask Points"
            
        sorted_indices = np.argsort(v)[::-1] 
        top_k = min(20, len(sorted_indices))
        bottom_indices = sorted_indices[:top_k]
        
        u_bottom_group = u[bottom_indices]
        v_bottom_group = v[bottom_indices]
        
        target_u = int(np.median(u_bottom_group))
        target_v = int(np.median(v_bottom_group))
        projected_2d = (target_u, target_v)

        z_raw_all = filtered_depth[v, u] / 1000.0
        valid_global_mask = (z_raw_all > self.depth_min) & (z_raw_all < self.depth_max)
        
        if np.sum(valid_global_mask) == 0: return None, None, None, "No valid depth in entire stem"
            
        z_bottom_raw = filtered_depth[v_bottom_group, u_bottom_group] / 1000.0
        valid_bottom_mask = (z_bottom_raw > self.depth_min) & (z_bottom_raw < self.depth_max)
        
        if np.sum(valid_bottom_mask) > 0: target_z = float(np.median(z_bottom_raw[valid_bottom_mask]))
        else: target_z = float(np.median(z_raw_all[valid_global_mask]))
        
        x_surface = (target_u - self.cx) * target_z / self.fx
        y_surface = (target_v - self.cy) * target_z / self.fy
        target_z_center = target_z + self.assumed_stem_radius
        
        center_3d = np.array([x_surface, y_surface, target_z_center])
        quat = [0.0, 0.0, 0.0, 1.0]

        return center_3d, quat, projected_2d, "OK"

    def cleanup(self):
        if self.perf_eval_enable and self.latest_pose_img is not None and self.log_root_dir is not None:
            timestamp_str = datetime.now().strftime("%H%M%S")
            save_path = os.path.join(self.log_root_dir, f"cycle{self.cycle_count}_pose_last_{timestamp_str}.jpg")
            cv2.imwrite(save_path, self.latest_pose_img)
            rospy.loginfo(f"[TEST] 程式關閉，已儲存最終幀影像: {save_path}")

        if self.csv_file is not None and not self.csv_file.closed:
            self.csv_file.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        estimator = StemPoseEstimatorBenchmark()
        rospy.on_shutdown(estimator.cleanup) 
        estimator.run_spin()
    except rospy.ROSInterruptException:
        pass
