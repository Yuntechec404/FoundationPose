#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys

# ==========================================
# [記憶體優化] 解決 CUDA 記憶體碎片化問題
# ==========================================
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

from ultralytics import YOLO

# ==========================================
# 匯入 Cutie 核心組件 (請確認路徑正確)
# ==========================================
CUTIE_PATH = os.path.expanduser("~/Cutie") 
if CUTIE_PATH not in sys.path:
    sys.path.append(CUTIE_PATH)

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

class StemPoseEstimatorROI:
    def __init__(self):
        rospy.init_node('stem_tracker_roi', anonymous=True)
        self.bridge = CvBridge()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # ROS 參數 (Topic)
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/color/camera_info')
        self.image_raw_topic = rospy.get_param('~image_raw_topic', '/camera/color/image_raw')
        self.depth_topic = rospy.get_param('~depth_topic', '/camera/aligned_depth_to_color/image_raw')
        
        self.camera_tf = rospy.get_param('~camera_tf', 'camera_color_optical_frame')
        self.child_tf = rospy.get_param('~child_tf', 'stem_estimated_frame')
        self.pose_topic = rospy.get_param('~pose_topic', '/stem/pruning_pose')
        self.pose_vis_topic = rospy.get_param('~pose_vis_topic', '/stem/debug_vis_image')

        # ==========================================
        # 🎯 新增：ROI 選擇策略 (middle | score | nearest_depth)
        # ==========================================
        self.det_select_mode = rospy.get_param("~det_select_mode", "score")

        # 模型參數
        self.model_path = rospy.get_param("~model_path", "yolov11n-seg.pt")
        self.target_cls = rospy.get_param("~target_cls", 0)
        self.conf_thresh = rospy.get_param("~conf_thresh", 0.5)
        self.erode_iterations = rospy.get_param("~erode_iterations", 2)
        
        # 深度設定
        self.depth_min = rospy.get_param("~depth_min", 0.1) 
        self.depth_max = rospy.get_param("~depth_max", 3.0) 
        self.assumed_stem_radius = rospy.get_param("~assumed_stem_radius", 0.04)
        
        # 狀態變數
        self.K_ready = False
        self.fx = self.fy = self.cx = self.cy = 0.0
        self.latest_color_img = None
        self.latest_depth_img = None
        self.latest_header = None
        self.new_frame_available = False
        self.is_tracking = False
        
        self.cv_window_name = "YOLO + Cutie Root Tracker (ROI Mode)"
        cv2.namedWindow(self.cv_window_name, cv2.WINDOW_NORMAL)
        
        # 1. 載入 YOLO 模型
        rospy.loginfo(f"正在載入YOLO-Seg模型: {self.model_path}")
        rospy.loginfo(f"目前的 ROI 選擇策略為: {self.det_select_mode}")
        self.yolo = YOLO(self.model_path)
        
        with torch.inference_mode(), torch.cuda.amp.autocast():
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            self.yolo.predict(dummy_img, device=self.device, imgsz=480, verbose=False)
        
        # 2. 載入 Cutie 模型
        rospy.loginfo("正在載入 Cutie 模型...")
        network = get_default_model()
        network.eval().to(self.device)
        self.processor = InferenceCore(network, cfg=network.cfg)
        
        self.processor.max_internal_size = 480 
        self.erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        # 通訊設定
        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=1)
        self.vis_pub = rospy.Publisher(self.pose_vis_topic, Image, queue_size=1)
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_cb)
        color_sub = message_filters.Subscriber(self.image_raw_topic, Image)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=2, slop=0.05)
        self.ts.registerCallback(self.sync_callback)
        
        rospy.loginfo("系統已啟動，等待相機影像...")

    def camera_info_cb(self, msg):
        if not self.K_ready:
            self.fx = msg.K[0]
            self.cx = msg.K[2]
            self.fy = msg.K[4]
            self.cy = msg.K[5]
            self.K_ready = True

    def sync_callback(self, color_msg, depth_msg):
        if not self.K_ready:
            return
        try:
            self.latest_color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            self.latest_depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            self.latest_header = depth_msg.header
            self.new_frame_available = True
        except Exception as e:
            rospy.logwarn(f"影像轉換失敗: {e}")

    def cv2_to_tensor(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(img_rgb).to(self.device)
        return tensor

    def run_spin(self):
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            if not self.new_frame_available or self.latest_color_img is None:
                cv2.waitKey(1)
                rate.sleep()
                continue
            
            color_img = self.latest_color_img.copy()
            depth_img = self.latest_depth_img.copy()
            header = self.latest_header
            self.new_frame_available = False
            
            vis_img = color_img.copy()
            H, W = color_img.shape[:2]
            mask_img = None

            with torch.inference_mode(), torch.cuda.amp.autocast():
                
                # ==================================================
                # 模式 1：YOLO 初始化與 ROI 挑選
                # ==================================================
                if not self.is_tracking:
                    results = self.yolo.predict(color_img, imgsz=480, conf=self.conf_thresh, device=self.device, verbose=False)
                    best_mask_uint8 = None

                    if results and results[0].masks is not None:
                        r = results[0]
                        classes = r.boxes.cls.cpu().numpy()
                        target_indices = np.where(classes == self.target_cls)[0]
                        
                        if len(target_indices) > 0:
                            best_idx = None
                            
                            # 策略 1: 最靠近畫面中心 (middle)
                            if self.det_select_mode == "middle":
                                min_dist = float('inf')
                                cx_img, cy_img = W / 2.0, H / 2.0
                                for idx in target_indices:
                                    # YOLO bbox 格式為 [x_center, y_center, width, height]
                                    box = r.boxes.xywh[idx].cpu().numpy()
                                    dist = (box[0] - cx_img)**2 + (box[1] - cy_img)**2
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_idx = idx

                            # 策略 2: 物理距離最近 (nearest_depth)
                            elif self.det_select_mode == "nearest_depth":
                                min_depth = float('inf')
                                # 製作一個暫時的平滑深度圖來讀取
                                temp_depth = depth_img.copy()
                                temp_depth[temp_depth == 0] = np.max(temp_depth)
                                temp_depth = cv2.GaussianBlur(temp_depth, (3, 3), 0)

                                for idx in target_indices:
                                    poly = r.masks.xy[idx]
                                    if len(poly) > 2:
                                        # 產生單一實例的 Mask
                                        instance_mask = np.zeros((H, W), dtype=np.uint8)
                                        cv2.fillPoly(instance_mask, [np.int32(poly)], 1)
                                        v_inst, u_inst = np.where(instance_mask > 0)
                                        
                                        if len(v_inst) > 0:
                                            # 取出這個實例內的所有深度值
                                            z_vals = temp_depth[v_inst, u_inst] / 1000.0
                                            valid_z = z_vals[(z_vals > self.depth_min) & (z_vals < self.depth_max)]
                                            
                                            if len(valid_z) > 0:
                                                med_z = np.median(valid_z)
                                                if med_z < min_depth:
                                                    min_depth = med_z
                                                    best_idx = idx
                                                    
                                # 如果深度因為某種原因全算不出來，退回給第一筆資料以防當機
                                if best_idx is None:
                                    best_idx = target_indices[0]

                            # 策略 3: 信心度最高 (score) - 預設
                            else:
                                best_conf = -1.0
                                for idx in target_indices:
                                    conf = float(r.boxes.conf[idx].cpu().numpy())
                                    if conf > best_conf:
                                        best_conf = conf
                                        best_idx = idx

                            # --- 確定要追蹤的候選者後，生成 Mask ---
                            poly = r.masks.xy[best_idx]
                            temp_mask = np.zeros((H, W), dtype=np.uint8)
                            if len(poly) > 2:
                                cv2.fillPoly(temp_mask, [np.int32(poly)], 1) 
                                best_mask_uint8 = temp_mask
                    
                    if best_mask_uint8 is not None:
                        colored_mask = np.zeros_like(vis_img)
                        colored_mask[best_mask_uint8 > 0] = [0, 255, 0]
                        cv2.addWeighted(vis_img, 1.0, colored_mask, 0.4, 0, vis_img)
                        
                        torch.cuda.empty_cache()
                        
                        init_mask = torch.from_numpy(best_mask_uint8).to(self.device).long()
                        img_tensor = self.cv2_to_tensor(color_img)
                        self.processor.clear_memory()
                        self.processor.step(img_tensor, init_mask, objects=[1])
                        self.is_tracking = True
                        mask_img = best_mask_uint8 * 255 
                        
                        cv2.putText(vis_img, f"MODE: YOLO Init ({self.det_select_mode}) -> Cutie", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        cv2.putText(vis_img, f"MODE: YOLO Searching... ({self.det_select_mode})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # ==================================================
                # 模式 2：Cutie 連續追蹤
                # ==================================================
                else:
                    img_tensor = self.cv2_to_tensor(color_img)
                    output_prob = self.processor.step(img_tensor)
                    pred_mask_tensor = self.processor.output_prob_to_mask(output_prob)
                    raw_tracked_mask = (pred_mask_tensor.cpu().numpy() == 1).astype(np.uint8)
                    
                    tracked_mask = cv2.erode(raw_tracked_mask, self.erode_kernel, iterations=self.erode_iterations)
                    
                    if tracked_mask.sum() < 50:
                        rospy.logwarn("Cutie lost target! Switching back to YOLO.")
                        self.processor.clear_memory()
                        torch.cuda.empty_cache() 
                        self.is_tracking = False
                    else:
                        colored_mask = np.zeros_like(vis_img)
                        colored_mask[tracked_mask > 0] = [0, 0, 255]
                        cv2.addWeighted(vis_img, 1.0, colored_mask, 0.5, 0, vis_img)
                        
                        mask_img = tracked_mask * 255
                        cv2.putText(vis_img, f"MODE: CUTIE Tracking (Erode:{self.erode_iterations})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # ==================================================
            # 根部 3D 座標與發布 (已套用破洞修補與 2D 尋底邏輯)
            # ==================================================
            if mask_img is not None:
                center_3d, quat, projected_2d, status = self.calculate_direct_root_pose(depth_img, mask_img)

                if center_3d is not None:
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
                    
                    depth_text = f"Root Z: {center_3d[2]:.3f} m"
                    text_org = (projected_2d[0] + 15, projected_2d[1] - 15)
                    cv2.putText(vis_img, depth_text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4) 
                    cv2.putText(vis_img, depth_text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2) 
                    
                    cv2.putText(vis_img, "TF Published: OK", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(vis_img, f"Mapping Failed: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(self.cv_window_name, vis_img)
            try:
                self.vis_pub.publish(self.bridge.cv2_to_imgmsg(vis_img, "bgr8"))
            except Exception:
                pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.processor.clear_memory()
                torch.cuda.empty_cache()
                self.is_tracking = False
                rospy.loginfo("已手動重置為 YOLO 搜尋模式")
            elif key == ord('q'):
                rospy.signal_shutdown("Quit")
                break
                
            rate.sleep()

    def calculate_direct_root_pose(self, depth_img, mask):
        filtered_depth = depth_img.copy()
        filtered_depth = cv2.GaussianBlur(filtered_depth, (3, 3), 0)

        v, u = np.where(mask > 0)
        if len(v) == 0:
            return None, None, None, "No Mask Points"
            
        # 1. 確保 2D 點位落在 Mask 最下緣
        sorted_indices = np.argsort(v)[::-1] 
        top_k = min(20, len(sorted_indices))
        bottom_indices = sorted_indices[:top_k]
        
        u_bottom_group = u[bottom_indices]
        v_bottom_group = v[bottom_indices]
        
        target_u = int(np.median(u_bottom_group))
        target_v = int(np.median(v_bottom_group))
        projected_2d = (target_u, target_v)

        # 2. 智慧處理深度破洞
        z_raw_all = filtered_depth[v, u] / 1000.0
        valid_global_mask = (z_raw_all > self.depth_min) & (z_raw_all < self.depth_max)
        
        if np.sum(valid_global_mask) < 10: 
            return None, None, None, "No valid depth in entire stem"
            
        z_bottom_raw = filtered_depth[v_bottom_group, u_bottom_group] / 1000.0
        valid_bottom_mask = (z_bottom_raw > self.depth_min) & (z_bottom_raw < self.depth_max)
        
        if np.sum(valid_bottom_mask) > 0:
            # 底部有成功算出深度
            target_z = float(np.median(z_bottom_raw[valid_bottom_mask]))
        else:
            # 底部破洞，借用整條莖的中位數有效深度
            target_z = float(np.median(z_raw_all[valid_global_mask]))
        
        # 3. 投影到 3D 空間
        x_surface = (target_u - self.cx) * target_z / self.fx
        y_surface = (target_v - self.cy) * target_z / self.fy
        
        target_z_center = target_z + self.assumed_stem_radius
        
        center_3d = np.array([x_surface, y_surface, target_z_center])
        quat = [0.0, 0.0, 0.0, 1.0]

        return center_3d, quat, projected_2d, "OK"

    def cleanup(self):
        cv2.destroyAllWindows()
        rospy.loginfo("CV2 windows destroyed.")

if __name__ == '__main__':
    try:
        estimator = StemPoseEstimatorROI()
        rospy.on_shutdown(estimator.cleanup) 
        estimator.run_spin()
    except rospy.ROSInterruptException:
        pass
