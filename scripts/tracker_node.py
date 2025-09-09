#!/home/user/anaconda3/envs/foundationpose/bin/python3
# coding :utf-8
# ROS1
import os
import sys
# Set megapose environment variables
sys.path.append('/home/user/anaconda3/envs/foundationpose/lib/python3.9/site-packages')
os.environ.setdefault("ULTRALYTICS_NO_INSTALL", "1")

import rospy
import numpy as np
import cv2
import trimesh

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import message_filters

# 你的 FoundationPose/Scorer/Refiner 與繪圖工具，直接沿用你現有專案的模組路徑
# 假設已能 from estimater import FoundationPose, draw_posed_3d_box, draw_xyz_axis
# 假設已能 from datareader import ScorePredictor, PoseRefinePredictor, dr
from estimater import FoundationPose, draw_posed_3d_box, draw_xyz_axis, dr
from datareader import ScorePredictor, PoseRefinePredictor
from ultralytics import YOLO

bridge = CvBridge()

# ────────────────────────
# 來自你原始腳本的部分小工具（裁切、IoU等）
# ────────────────────────

def clip_xyxy(xyxy, W, H):
    if xyxy is None:
        return None
    x1, y1, x2, y2 = map(float, xyxy)
    return np.array([max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)], dtype=np.float32)


def rect_to_mask(depth, xyxy, expand=0.0):
    if xyxy is None:
        return None
    H, W = depth.shape[:2]
    x1, y1, x2, y2 = xyxy.astype(np.int32)
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    x1 = max(0, int(x1 - w * expand))
    y1 = max(0, int(y1 - h * expand))
    x2 = min(W - 1, int(x2 + w * expand))
    y2 = min(H - 1, int(y2 + h * expand))
    m = np.zeros((H, W), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def iou_xyxy(a, b):
    if a is None or b is None:
        return 0.0
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (ua + ub - inter + 1e-6))


def yolo_det_all(detector: YOLO, img_bgr, imgsz=640, conf=0.25):
    r = detector.predict(source=img_bgr, imgsz=imgsz, conf=conf, verbose=False)[0]
    if len(r.boxes) == 0:
        return (np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32))
    xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
    sc   = r.boxes.conf.cpu().numpy().astype(np.float32)
    cl   = r.boxes.cls.cpu().numpy().astype(np.int32)
    return xyxy, sc, cl


def project_3d_bbox_xyxy(K, center_pose, bbox_minmax, img_shape):
    H, W = img_shape[:2]
    mn, mx = bbox_minmax
    xs = [mn[0], mx[0]]
    ys = [mn[1], mx[1]]
    zs = [mn[2], mx[2]]
    corners = np.array([[x, y, z, 1.0] for x in xs for y in ys for z in zs], dtype=np.float64)  # (8,4)
    Pc = (center_pose @ corners.T).T
    Z = Pc[:, 2]
    valid = Z > 1e-6
    if not np.any(valid):
        return None
    X = Pc[valid, 0] / Z[valid]
    Y = Pc[valid, 1] / Z[valid]
    u = K[0, 0] * X + K[0, 2]
    v = K[1, 1] * Y + K[1, 2]
    return clip_xyxy(np.array([u.min(), v.min(), u.max(), v.max()], dtype=np.float32), W, H)


class FoundationPoseTracker(object):
    def __init__(self):
        # 讀參數
        self.mesh_file       = rospy.get_param("~model/mesh_file")
        self.det_onnx        = rospy.get_param("~model/det_onnx")
        self.det_imgsz       = int(rospy.get_param("~model/det_imgsz", 640))
        self.det_conf        = float(rospy.get_param("~model/det_conf", 0.25))
        self.det_class       = int(rospy.get_param("~model/det_class", 0))
        self.roi_expand      = float(rospy.get_param("~model/roi_expand", 0.01))
        self.est_refine_iter = int(rospy.get_param("~model/est_refine_iter", 5))
        self.track_refine_iter = int(rospy.get_param("~model/track_refine_iter", 2))

        self.iou_stride   = int(rospy.get_param("~iou/stride", 1))
        self.iou_thresh   = float(rospy.get_param("~iou/thresh", 0.2))
        self.iou_patience = int(rospy.get_param("~iou/patience", 2))
        self.iou_log      = bool(rospy.get_param("~iou/log", False))

        self.publish_viz  = bool(rospy.get_param("~vis/publish_viz", True))
        self.viz_topic    = rospy.get_param("~vis/viz_topic", "/foundationpose/viz")
        self.init_mode    = rospy.get_param("~runtime/init_mode", "yolo")

        # 資源初始化
        self.mesh = trimesh.load(self.mesh_file)
        to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.to_origin = to_origin
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

        self.scorer  = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx   = dr.RasterizeCudaContext()
        self.est = FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            debug_dir=None,
            debug=0,
            glctx=self.glctx,
        )
        rospy.loginfo("Estimator initialization done")

        self.detector = YOLO(self.det_onnx, task='detect')
        self.prefer_cls = None if self.det_class < 0 else self.det_class

        # ROS I/O
        image_sub = message_filters.Subscriber("image", Image)
        depth_sub = message_filters.Subscriber("depth", Image)
        info_sub  = message_filters.Subscriber("camera_info", CameraInfo)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, depth_sub, info_sub], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.cb)

        self.pose_pub = rospy.Publisher("pose", PoseStamped, queue_size=1)
        self.viz_pub  = rospy.Publisher(self.viz_topic, Image, queue_size=1) if self.publish_viz else None

        # 狀態
        self.first_frame = True
        self.pose = np.eye(4, dtype=np.float64)
        self.mask = None
        self.frame_count = 0
        self.iou_bad_count = 0
        self.iou_val = None
        self.K = None

    def cb(self, img_msg, depth_msg, info_msg):
        # 轉影像
        color = bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        depth = depth.astype(np.float32) * 0.001  # mm -> m (與你的腳本一致)
        H, W = color.shape[:2]

        # 相機內參 K
        if self.K is None:
            K = np.array([[info_msg.K[0], 0, info_msg.K[2]],
                          [0, info_msg.K[4], info_msg.K[5]],
                          [0, 0, 1]], dtype=np.float64)
            self.K = K
        self.frame_count += 1

        # 初始化：用 YOLO 取 ROI
        if self.init_mode == 'yolo':
            if self.first_frame:
                # 偵測一次
                r = self.detector.predict(source=color, imgsz=self.det_imgsz, conf=self.det_conf, verbose=False)[0]
                if len(r.boxes) == 0:
                    return
                # 取指定類別或最高分
                xyxy = r.boxes.xyxy.cpu().numpy()
                sc   = r.boxes.conf.cpu().numpy()
                cl   = r.boxes.cls.cpu().numpy().astype(int)
                if self.prefer_cls is not None and (cl == self.prefer_cls).any():
                    idx = int(np.argmax(sc * (cl == self.prefer_cls)))
                else:
                    idx = int(np.argmax(sc))
                det_xyxy = clip_xyxy(xyxy[idx], W, H)
                self.mask = rect_to_mask(depth, det_xyxy, expand=self.roi_expand)
                self.pose = self.est.register(K=self.K, rgb=color, depth=depth, ob_mask=self.mask, iteration=self.est_refine_iter)
                self.first_frame = False
        else:
            # click 模式在 ROS 中不常用，此處略過
            if self.first_frame:
                return

        # 追蹤
        self.pose = self.est.track_one(rgb=color, depth=depth, K=self.K, iteration=self.track_refine_iter)

        # 計算/檢查 IoU（每 N 幀）
        vis_bgr = color.copy()
        if self.pose is not None:
            center_pose = self.pose @ np.linalg.inv(self.to_origin)

            if (self.frame_count % max(1, self.iou_stride) == 0):
                est_xyxy = project_3d_bbox_xyxy(self.K, center_pose, self.bbox, img_shape=color.shape)
                if est_xyxy is not None:
                    xyxy_all, sc_all, cl_all = yolo_det_all(self.detector, color, imgsz=self.det_imgsz, conf=self.det_conf)
                    if len(xyxy_all) > 0:
                        use_mask = np.ones(len(xyxy_all), dtype=bool)
                        if self.prefer_cls is not None and (cl_all == self.prefer_cls).any():
                            use_mask = (cl_all == self.prefer_cls)
                        xyxy_use = xyxy_all[use_mask]
                        ious = []
                        for bb in xyxy_use:
                            bb_c = clip_xyxy(bb, W, H)
                            ious.append(iou_xyxy(bb_c, est_xyxy))
                        if len(ious) > 0:
                            self.iou_val = float(np.max(ious))
                            if self.iou_log:
                                rospy.loginfo(f"[IOU] frame={self.frame_count} val={self.iou_val:.3f}")
                            if self.iou_val < self.iou_thresh:
                                self.iou_bad_count += 1
                            else:
                                self.iou_bad_count = 0
                            if self.iou_bad_count >= self.iou_patience:
                                self.iou_bad_count = 0
                                self.first_frame = True
                                self.pose = None
                                rospy.logwarn("Re-init requested (low IoU)")

            # 視覺化與發布 Pose
            color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            vis = draw_posed_3d_box(self.K, img=color_rgb, ob_in_cam=center_pose, bbox=self.bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=self.pose, scale=0.05, K=self.K, thickness=3, transparency=0, is_input_rgb=True)
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

            # 發布 PoseStamped（相機座標系）
            ps = PoseStamped()
            ps.header = img_msg.header  # 使用影像的時間戳/座標系
            # 將 4x4 轉成 position + quaternion
            t = center_pose[:3, 3]
            R = center_pose[:3, :3]
            # 旋轉矩陣轉四元數
            qw = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
            qx = (R[2, 1] - R[1, 2]) / (4.0 * qw)
            qy = (R[0, 2] - R[2, 0]) / (4.0 * qw)
            qz = (R[1, 0] - R[0, 1]) / (4.0 * qw)
            ps.pose.position.x = float(t[0])
            ps.pose.position.y = float(t[1])
            ps.pose.position.z = float(t[2])
            ps.pose.orientation.x = float(qx)
            ps.pose.orientation.y = float(qy)
            ps.pose.orientation.z = float(qz)
            ps.pose.orientation.w = float(qw)
            self.pose_pub.publish(ps)

        if self.publish_viz:
            self.viz_pub.publish(bridge.cv2_to_imgmsg(vis_bgr, encoding='bgr8'))


def main():
    rospy.init_node('foundationpose_tracker', anonymous=False)
    _ = FoundationPoseTracker()
    rospy.loginfo("foundationpose_tracker is running")
    rospy.spin()

if __name__ == '__main__':
    main()

