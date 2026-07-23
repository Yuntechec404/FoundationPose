#!/home/user/anaconda3/envs/foundationpose/bin/python3
# -*- coding: utf-8 -*-

import os, sys
import time  
import csv   
from datetime import datetime 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
import rospy
import numpy as np
import cv2
import trimesh
import torch

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped, Transform
from cv_bridge import CvBridge
from forklift_msg.msg import Confidence, Detection
from std_msgs.msg import Empty

from collections import deque
import tf

sys.path.append('/home/user/anaconda3/envs/foundationpose/lib/python3.8/site-packages')
FOUNDATIONPOSE_SRC = "/home/user/FoundationPose"
if FOUNDATIONPOSE_SRC not in sys.path:
    sys.path.append(FOUNDATIONPOSE_SRC)

os.environ.setdefault("ULTRALYTICS_NO_INSTALL", "1")

from estimater import FoundationPose, draw_posed_3d_box, draw_xyz_axis, ScorePredictor, PoseRefinePredictor, dr
from ultralytics import YOLO, SAM, FastSAM

selecting_bbox = False
box_points = []

class FoundationPoseTracker:
    def __init__(self):
        self.init_parameter()
        
        self.det_model_name = os.path.basename(self.det_model)
        if self.seg_backend in ['sam', 'sam2', 'fastsam']:
            self.seg_model_name = f"{self.seg_backend}_{os.path.basename(self.sam_ckpt)}"
        elif self.seg_backend == 'yolo11-seg':
            self.seg_model_name = f"yolo11-seg_{self.det_model_name}"
        else:
            self.seg_model_name = self.seg_backend
            
        self._setup_run_debug_dir()
        
        if self.perf_eval_enable and self.debug_dir:
            self.perf_csv_path = os.path.join(self.debug_dir, "perf_eval.csv")
            with open(self.perf_csv_path, mode='w', newline='') as f:
                f.write("frame_count,det_model,seg_model,detection_ms,segmentation_ms,initial_pose_ms,refine_pose_ms\n")
        else:
            self.perf_csv_path = None

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
            
        self.bridge = CvBridge()
        self.color = None
        self.depth = None
        self.depth_encoding = None
        self.got_depth = False
        self.got_rgb = False
        self.depth_size = (0, 0)
        self.rgb_size = (0, 0)
        self.pose = np.eye(4, dtype=np.float64)
        self.mask = None
        self.frame_count = 0
        self.iou_bad_count = 0
        self.iou_val = None
        self.K = None
        self._last_yolo_text = ""
        
        self.consecutive_det_count = 0

        self.time_det = 0.0  
        self.time_seg = 0.0  
        self.time_init = 0.0 
        self.time_refine = 0.0 
        
        self.ui_time_det = 0.0 
        self.ui_time_seg = 0.0
        self.ui_time_init = 0.0
        self.ui_time_refine = 0.0

        self._rgb_win_created = False
        self._depth_win_created = False
        self._rgb_win_sized = False
        self._depth_win_sized = False
        self._rgb_initial_size = (900, 720)
        self._depth_initial_size = (900, 720)

        self.tracking_start_time = None
        self.track_save_count = 0        
        self.first_track_saved = False 

        # Pub/Sub
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.imageCallback, queue_size=1)
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self.depthCallback, queue_size=1)
        self.info_sub = rospy.Subscriber(self.info_topic,  CameraInfo, self.infoCallback, queue_size=1)
        
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.pose_pub = rospy.Publisher(self.object_name, Pose, queue_size=1, latch=True)
        self.conf_pub = rospy.Publisher(self.object_name + "_confidence", Confidence, queue_size=1, latch=True)
        
        if self.yolo_start_mode == "wait":
            self._ready_sub = rospy.Subscriber(self.object_name + "_detection", Detection, self.detectionCallback, queue_size=1)
                    
        self.window_create()

        # foundationpose初始化
        self.mesh = trimesh.load(self.mesh_file)
        self.to_origin, self.extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.to_origin = self.to_origin
        self.bbox = np.stack([-self.extents/2, self.extents/2], axis=0).reshape(2, 3)

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        
        safe_debug_dir = self.debug_dir if self.debug_dir else getattr(self, "debug_root", "/tmp/fp_debug")
        
        self.est = FoundationPose(
            model_pts=self.mesh.vertices, 
            model_normals=self.mesh.vertex_normals, 
            mesh=self.mesh, 
            scorer=self.scorer, 
            refiner=self.refiner, 
            debug_dir=safe_debug_dir,
            debug=0, 
            glctx=self.glctx,
            coarse_min_n_views=self.coarse_min_n_views,
            coarse_inplane_step=self.coarse_inplane_step,
            coarse_orientation_mode=self.coarse_orientation_mode,
            coarse_orientation_tilt_deg=self.coarse_orientation_tilt_deg,
            coarse_object_up_axis=self.coarse_object_up_axis
        )
        rospy.loginfo("Estimator initialization done")

        # 決定 YOLO 任務
        task_type = 'segment' if self.seg_backend == 'yolo11-seg' else 'detect'
        self.detector, self.det_device = self.load_detector(self.det_model, task=task_type)
        is_gpu, yolo_desc = self.yolo_uses_gpu(self.detector)
        rospy.loginfo(f"[YOLO] GPU enabled: {is_gpu}  ({yolo_desc})")
        
        # ==========================================
        # SAM / SAM2 / FastSAM 的載入邏輯
        # ==========================================
        self.prompt_model = None
        if self.seg_backend in ['sam', 'sam2', 'fastsam']:
            try:
                # FastSAM 和 SAM/SAM2 在 Ultralytics 使用的 Class 稍微不同，但 API 相同
                if self.seg_backend == 'fastsam':
                    self.prompt_model = FastSAM(self.sam_ckpt)
                else:
                    self.prompt_model = SAM(self.sam_ckpt)
                    
                rospy.loginfo(f"[{self.seg_backend.upper()}] Loaded {self.sam_ckpt}")
                dummy = np.zeros((self.det_imgsz, self.det_imgsz, 3), np.uint8)
                # 統一預熱
                _ = self.prompt_model(dummy, bboxes=[[10,10,100,100]], imgsz=self.det_imgsz, device="cuda:0" if torch.cuda.is_available() else "cpu", verbose=False)
            except Exception as e:
                rospy.logwarn(f"[{self.seg_backend.upper()}] init failed: {e}")
                self.prompt_model = None

    def init_parameter(self):
        ns = rospy.get_name()
        gp = lambda k, d: rospy.get_param(ns + "/" + k, d)

        self.image_topic = gp("image_topic", "/camera/color/image_raw")
        self.info_topic = gp("info_topic",  "/camera/color/camera_info")
        self.depth_topic = gp("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.depth_info_topic = gp("depth_info_topic", "")
        self.camera_tf = gp("camera_tf", "")
        self.object_name = gp("object_name", "tracked_object")

        self.mesh_file = gp("mesh_file", "")
        self.det_model = gp("det_model", "yolo11n-seg.pt")
        self.init_mode = gp("init_mode", "yolo")
        self.yolo_start_mode = gp("yolo_start_mode", "immediate").strip().lower()
        self.det_conf = float(gp("det_conf", 0.25))
        self.det_class = int(gp("det_class", -1))
        self.init_det_patience = int(gp("init_det_patience", 3))
        self.est_top_k = int(gp("est_top_k", 5))
        self.est_refine_iter = int(gp("est_refine_iter", 5))
        self.track_refine_iter = int(gp("track_refine_iter", 2))

        # Coarse initial pose sampling parameters for FoundationPose.register().
        self.coarse_min_n_views = int(gp("coarse/min_n_views", 40))
        self.coarse_inplane_step = int(gp("coarse/inplane_step", 60))
        self.coarse_orientation_mode = gp("coarse/orientation_mode", "uniform").strip().lower()
        self.coarse_orientation_tilt_deg = float(gp("coarse/orientation_tilt_deg", 30.0))
        self.coarse_object_up_axis = int(gp("coarse/object_up_axis", 1))

        if self.coarse_orientation_mode in ("none", "all"):
            self.coarse_orientation_mode = "uniform"
        if self.coarse_orientation_mode not in ("uniform", "upright", "inverted"):
            rospy.logwarn(
                f"[CoarsePoseGrid] Unknown orientation_mode={self.coarse_orientation_mode}, fallback to uniform"
            )
            self.coarse_orientation_mode = "uniform"
        if self.coarse_object_up_axis not in (0, 1, 2):
            rospy.logwarn(
                f"[CoarsePoseGrid] Unknown object_up_axis={self.coarse_object_up_axis}, fallback to 1(Y)"
            )
            self.coarse_object_up_axis = 1
        if self.coarse_inplane_step <= 0 or self.coarse_inplane_step > 360:
            rospy.logwarn(
                f"[CoarsePoseGrid] Invalid inplane_step={self.coarse_inplane_step}, fallback to 60"
            )
            self.coarse_inplane_step = 60
        if self.coarse_min_n_views <= 0:
            rospy.logwarn(
                f"[CoarsePoseGrid] Invalid min_n_views={self.coarse_min_n_views}, fallback to 40"
            )
            self.coarse_min_n_views = 40

        rospy.loginfo(
            f"[CoarsePoseGrid] launch params: min_n_views={self.coarse_min_n_views}, "
            f"inplane_step={self.coarse_inplane_step}, "
            f"orientation_mode={self.coarse_orientation_mode}, "
            f"orientation_tilt_deg={self.coarse_orientation_tilt_deg}, "
            f"object_up_axis={self.coarse_object_up_axis}"
        )
        
        self.debug_root = gp("debug_dir", "/tmp/fp_debug")

        self.perf_eval_enable = bool(gp("perf_eval_enable", True))
        self.seg_backend = gp("seg_backend", "yolo11-seg").strip().lower() 
        self.sam_ckpt = gp("sam_ckpt", "FastSAM-x.pt").strip()

        self.roi_expand = float(gp("roi_expand", 0.02))
        self.iou_stride = int(gp("iou_stride", 1))
        self.iou_log = bool(gp("iou_log", True))
        self.iou_thresh = float(gp("iou_thresh", 0.2))
        self.iou_patience = int(gp("iou_patience", 3))
        self.det_imgsz = int(gp("det_imgsz", 640))
        self.prefer_cls = None if self.det_class < 0 else self.det_class

        self.det_select_mode = gp("det_select_mode", "score").strip().lower()
        if self.det_select_mode not in ("score", "middle", "nearest_depth"):
            self.det_select_mode = "score"

        self.show_depth_win = bool(gp("show_depth_window", False))
        self.show_rgb_win = bool(gp("show_rgb_window", True))
        self.depth_win_name = gp("depth_win_name", "Depth Image")
        self.rgb_win_name = gp("rgb_win_name", "RGB Image")
        self.depth_win_xy = gp("depth_window_xy", [100, 100])  
        self.rgb_win_xy = gp("rgb_window_xy", [100, 500])  
        self.max_depth_mm = float(gp("max_depth_mm", 10000.0))  
        self.colormap_id = int(gp("colormap", int(cv2.COLORMAP_JET)))  
        self.invert_colormap = bool(gp("invert_colormap", False))

        # Re-initialization strategy:
        #   "yolo_iou"         : 週期性跑 YOLO，計算 YOLO bbox 與 projected 3D bbox 的 IoU
        #   "depth_confidence" : 使用 track_one() 回傳的 Inlier Ratio 與 Depth MAE 判斷是否需要 reinit
        self.reinit_mode = gp("reinit_mode", "yolo_iou").strip().lower()
        if self.reinit_mode in ("iou", "yolo", "yolo-iou"):
            self.reinit_mode = "yolo_iou"
        elif self.reinit_mode in ("geom", "geometry", "depth", "depth_conf", "depth-confidence"):
            self.reinit_mode = "depth_confidence"
        elif self.reinit_mode not in ("yolo_iou", "depth_confidence"):
            rospy.logwarn(f"[Reinit] Unknown reinit_mode={self.reinit_mode}, fallback to yolo_iou")
            self.reinit_mode = "yolo_iou"

        self.geom_mae_thresh = float(gp("geom_mae_thresh", 0.03))
        self.geom_inlier_thresh = float(gp("geom_inlier_thresh", 0.5))
        self.geom_patience = int(gp("geom_patience", 3))
        self.geom_bad_count = 0
        self.depth_mae = None
        self.inlier_ratio = None
        self.depth_conf_score = None

    def _setup_run_debug_dir(self):
        if not self.perf_eval_enable:
            self.debug_dir = None
            return
        root = getattr(self, "debug_root", None) or getattr(self, "debug_dir", "/tmp/fp_debug")
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        base = os.path.join(root, ts)
        run_dir = base
        k = 1
        while os.path.exists(run_dir):
            run_dir = f"{base}_{k:02d}"
            k += 1
        self.debug_dir = run_dir
        os.makedirs(self.debug_dir, exist_ok=True)

    def _ensure_dir(self, d=None):
        if d is None: d = getattr(self, "debug_dir", None)
        if not d: return
        try: os.makedirs(d, exist_ok=True)
        except Exception: pass

    def _ensure_parent(self, path: str):
        try:
            parent = os.path.dirname(path)
            if parent: os.makedirs(parent, exist_ok=True)
        except Exception: pass

    def yolo_backend_info(self,detector):
        info = {"engine": None, "torch_device": None, "ort_providers": None}
        m = getattr(detector, "model", None)
        if m is None: return info
        info["engine"] = str(getattr(m, "backend", getattr(m, "engine", "")))
        try:
            if hasattr(m, "parameters"):
                p = next(m.parameters(), None)
                if p is not None: info["torch_device"] = str(p.device)
        except Exception: pass
        sess = None
        for attr in ("session", "ort_session", "session_ort"):
            s = getattr(m, attr, None)
            if s is not None:
                sess = s
                break
        if sess is not None:
            try: info["ort_providers"] = list(sess.get_providers())
            except Exception: pass
        else:
            prov = getattr(m, "providers", None)
            if prov is not None: info["ort_providers"] = list(prov)
        return info

    def yolo_uses_gpu(self,detector):
        info = self.yolo_backend_info(detector)
        eng = (info["engine"] or "").lower()
        dev = (info["torch_device"] or "").lower()
        prov = info["ort_providers"] or []
        if any("CUDAExecutionProvider" == p for p in prov):
            return True, f"engine={eng or 'onnxruntime'} providers={prov}"
        if "tensorrt" in eng: return True, f"engine={eng}"
        if dev.startswith("cuda"): return True, f"engine={eng or 'pytorch'} device={dev}"
        return False, f"engine={eng or 'unknown'} providers={prov or 'None'} device={dev or 'None'}"

    def load_detector(self, model_path: str, task='detect'):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"detector model not found: {model_path}")
        def _onnx_sibling(p):
            frond, ext = os.path.splitext(p)
            return frond + ".onnx"

        ext = os.path.splitext(model_path)[1].lower()
        det_device = "cpu"

        if ext == ".pt":
            try:
                det = YOLO(model_path, task=task)
                if torch.cuda.is_available():
                    try:
                        det.to("cuda:0")
                        det_device = 0 
                        return det, det_device
                    except Exception as ge: pass
                try:
                    det.to("cpu")
                    det_device = "cpu"
                    return det, det_device
                except Exception as ce: pass
            except Exception as e: pass

            onnx_fallback = _onnx_sibling(model_path)
            if os.path.isfile(onnx_fallback):
                det, det_device = self._load_onnx_with_gpu_fallback(onnx_fallback, task=task)
                return det, det_device
            else:
                raise RuntimeError(f"Failed to load PT '{model_path}' and no sibling ONNX found.")
        elif ext == ".onnx":
            det, det_device = self._load_onnx_with_gpu_fallback(model_path, task=task)
            return det, det_device
        else:
            raise ValueError(f"Unsupported detector extension: {ext}. Use .pt or .onnx")
        
    def _load_onnx_with_gpu_fallback(self, onnx_path: str, task='detect'):
        det = YOLO(onnx_path, task=task)
        sess = None
        for attr in ("session", "ort_session", "session_ort"):
            s = getattr(det.model, attr, None)
            if s is not None:
                sess = s
                break
        det_device = "cpu(ORT)"
        if sess is not None:
            try:
                provs = list(sess.get_providers())
            except Exception:
                provs = []
            if "CUDAExecutionProvider" in provs:
                det_device = "cuda(ORT)"
                return det, det_device
            try:
                sess.set_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
                provs2 = list(sess.get_providers())
                if "CUDAExecutionProvider" in provs2:
                    det_device = "cuda(ORT)"
                    return det, det_device
            except Exception as ge: pass
            try: sess.set_providers(["CPUExecutionProvider"])
            except Exception: pass
            det_device = "cpu(ORT)"
            return det, det_device
        return det, det_device

    def clip_xyxy(self, xyxy, W, H):
        if xyxy is None: return None
        x1, y1, x2, y2 = map(float, xyxy)
        return np.array([max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)], dtype=np.float32)

    def rect_to_mask(self, depth, xyxy, expand=0.0):
        if xyxy is None: return None
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

    def iou_xyxy(self, a, b):
        if a is None or b is None: return 0.0
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

    def yolo_det_xyxy_mask(self, detector: YOLO, img_bgr, imgsz=640, conf=0.25, prefer_cls=None, mode="score"):
        t0 = time.perf_counter()
        r = detector.predict(source=img_bgr, imgsz=imgsz, conf=conf, device=self.det_device, verbose=False)[0]
        if self.perf_eval_enable: 
            elapsed = (time.perf_counter() - t0) * 1000.0
            if self.seg_backend == 'yolo11-seg':
                self.time_seg = elapsed
                self.time_det = 0.0
            else:
                self.time_det = elapsed

        if len(r.boxes) == 0:
            return None, 0.0, None, None

        xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)   
        sc = r.boxes.conf.cpu().numpy().astype(np.float32)   
        cl = r.boxes.cls.cpu().numpy().astype(int)            
        masks = r.masks.data.cpu().numpy() if r.masks is not None else None

        mask = (sc >= float(conf))
        if prefer_cls is not None: mask = mask & (cl == int(prefer_cls))

        idxs = np.where(mask)[0]
        if idxs.size == 0: return None, 0.0, None, None

        mode = (mode or "score").lower()
        H, W = img_bgr.shape[:2]

        if mode == "middle":
            cx_img, cy_img = W / 2.0, H / 2.0
            centers_x = (xyxy[idxs, 0] + xyxy[idxs, 2]) / 2.0
            centers_y = (xyxy[idxs, 1] + xyxy[idxs, 3]) / 2.0
            dists = (centers_x - cx_img)**2 + (centers_y - cy_img)**2
            pick = idxs[np.argmin(dists)]
        elif mode == "nearest_depth" and hasattr(self, "depth_m"):
            min_depth = float('inf')
            pick = idxs[0] 
            temp_depth = self.depth_m.copy()
            temp_depth[temp_depth == 0] = np.max(temp_depth)
            temp_depth = cv2.GaussianBlur(temp_depth, (3, 3), 0)
            for idx in idxs:
                if masks is not None:
                    m = masks[idx]
                    if m.shape != (H, W): m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                    v_inst, u_inst = np.where(m > 0)
                else:
                    m = self.rect_to_mask(temp_depth, xyxy[idx])
                    v_inst, u_inst = np.where(m > 0)
                if len(v_inst) > 0:
                    z_vals = temp_depth[v_inst, u_inst]
                    valid_z = z_vals[(z_vals > 0.1) & (z_vals < 5.0)]
                    if len(valid_z) > 0:
                        med_z = np.median(valid_z)
                        if med_z < min_depth:
                            min_depth = med_z
                            pick = idx
        else: 
            pick = idxs[np.argmax(sc[idxs])]

        target_mask = masks[pick] if masks is not None else None
        return xyxy[pick], float(sc[pick]), int(cl[pick]), target_mask

    def yolo_det_all(self, detector: YOLO, img_bgr, imgsz=640, conf=0.25):
        t0 = time.perf_counter()
        r = detector.predict(source=img_bgr, imgsz=imgsz, conf=conf, device=self.det_device, verbose=False)[0]
        if self.perf_eval_enable: 
            elapsed = (time.perf_counter() - t0) * 1000.0
            if self.seg_backend == 'yolo11-seg':
                self.time_seg = elapsed
                self.time_det = 0.0
            else:
                self.time_det = elapsed

        if len(r.boxes) == 0:
            return (np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32))
        xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        sc   = r.boxes.conf.cpu().numpy().astype(np.float32)
        cl   = r.boxes.cls.cpu().numpy().astype(np.int32)
        return xyxy, sc, cl

    def project_3d_bbox_xyxy(self, K, center_pose, bbox_minmax, img_shape):
        H, W = img_shape[:2]
        mn, mx = bbox_minmax
        xs = [mn[0], mx[0]]
        ys = [mn[1], mx[1]]
        zs = [mn[2], mx[2]]
        corners = np.array([[x, y, z, 1.0] for x in xs for y in ys for z in zs], dtype=np.float64)  
        Pc = (center_pose @ corners.T).T
        Z = Pc[:, 2]
        valid = Z > 1e-6
        if not np.any(valid): return None
        X = Pc[valid, 0] / Z[valid]
        Y = Pc[valid, 1] / Z[valid]
        u = K[0, 0] * X + K[0, 2]
        v = K[1, 1] * Y + K[1, 2]
        return self.clip_xyxy(np.array([u.min(), v.min(), u.max(), v.max()], dtype=np.float32), W, H)

    def get_segmentation_mask(self, color, depth, xyxy, yolo_mask_data=None):
        H, W = color.shape[:2]
        if xyxy is None: return np.zeros((H, W), dtype=bool)

        x1, y1, x2, y2 = self.clip_xyxy(xyxy, W, H).astype(int)

        if self.seg_backend == 'yolo11-seg' and yolo_mask_data is not None:
            if yolo_mask_data.shape != (H, W):
                mask = cv2.resize(yolo_mask_data.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            else: mask = yolo_mask_data
            return mask.astype(bool)

        # ==========================================
        # 統一呼叫 SAM / SAM2 / FastSAM
        # ==========================================
        elif self.seg_backend in ['sam', 'sam2', 'fastsam'] and self.prompt_model is not None:
            t0 = time.perf_counter()
            results = self.prompt_model(color, bboxes=[[int(x1), int(y1), int(x2), int(y2)]], imgsz=self.det_imgsz, device="cuda:0" if torch.cuda.is_available() else "cpu", verbose=False)
            if self.perf_eval_enable: 
                self.time_seg = (time.perf_counter() - t0) * 1000.0
            
            if results and results[0].masks is not None:
                m = results[0].masks.data
                if torch.is_tensor(m): m = m.detach().cpu().numpy()
                m = np.asarray(m)
                if m.ndim == 3: m2 = m[0]
                else: m2 = m
                if m2.shape != (H, W): m2 = cv2.resize(m2.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                return (m2 > 0).astype(bool)

        if self.perf_eval_enable and self.seg_backend == 'bbox':
            self.time_seg = 0.0
        return self.rect_to_mask(depth, xyxy, expand=self.roi_expand)

    def _open_window(self, name, pos_xy, init_size, is_rgb=True):
        w, h = init_size
        try: cv2.destroyWindow(name)
        except Exception: pass
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, int(w), int(h))
        cv2.moveWindow(name, int(pos_xy[0]), int(pos_xy[1]))
        if is_rgb:
            self._rgb_win_created = True
            self._rgb_win_sized = False
        else:
            self._depth_win_created = True
            self._depth_win_sized = False

    def window_create(self):
        if self.show_rgb_win and not self._rgb_win_created:
            self._open_window(self.rgb_win_name, self.rgb_win_xy, self._rgb_initial_size, is_rgb=True)
        if self.show_depth_win and not self._depth_win_created:
            self._open_window(self.depth_win_name, self.depth_win_xy, self._depth_initial_size, is_rgb=False)

    def pump_windows(self, rgb_frame=None, depth_frame=None):
        if self.show_rgb_win and not self._rgb_win_created:
            self._open_window(self.rgb_win_name, self.rgb_win_xy, self._rgb_initial_size, is_rgb=True)
        if self.show_depth_win and not self._depth_win_created:
            self._open_window(self.depth_win_name, self.depth_win_xy, self._depth_initial_size, is_rgb=False)

        if rgb_frame is None and self.show_rgb_win:
            rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(rgb_frame, "Waiting for detection / click init...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        if depth_frame is None and self.show_depth_win:
            depth_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        if self.show_rgb_win and rgb_frame is not None:
            if self.perf_eval_enable:
                if getattr(self, 'time_det', 0) > 0: self.ui_time_det = self.time_det
                if getattr(self, 'time_seg', 0) > 0: self.ui_time_seg = self.time_seg
                if getattr(self, 'time_init', 0) > 0: self.ui_time_init = self.time_init
                if getattr(self, 'time_refine', 0) > 0: self.ui_time_refine = self.time_refine
                
                det_text = "N/A" if self.seg_backend == 'yolo11-seg' else f"{self.ui_time_det:.1f}ms"
                cv2.putText(rgb_frame, f"Det: {det_text} | Seg: {self.ui_time_seg:.1f}ms | Init: {self.ui_time_init:.1f}ms | Refine: {self.ui_time_refine:.1f}ms", 
                            (10, rgb_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow(self.rgb_win_name, rgb_frame)
            if not self._rgb_win_sized:
                w, h = self._rgb_initial_size
                cv2.resizeWindow(self.rgb_win_name, int(w), int(h))
                cv2.moveWindow(self.rgb_win_name, int(self.rgb_win_xy[0]), int(self.rgb_win_xy[1]))
                self._rgb_win_sized = True

        if self.show_depth_win and depth_frame is not None:
            cv2.imshow(self.depth_win_name, depth_frame)
            if not self._depth_win_sized:
                w, h = self._depth_initial_size
                cv2.resizeWindow(self.depth_win_name, int(w), int(h))
                cv2.moveWindow(self.depth_win_name, int(self.depth_win_xy[0]), int(self.depth_win_xy[1]))
                self._depth_win_sized = True

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): rospy.signal_shutdown("user quit")

    def click_bbox(self, event, x, y, flags, param):
        global box_points
        if event == cv2.EVENT_LBUTTONDOWN:
            box_points.append((x, y))

    def select_bbox(self, color):
        global box_points, selecting_bbox
        box_points.clear()
        selecting_bbox = True
        cv2.setMouseCallback(self.rgb_win_name, self.click_bbox)
        return True

    def update_bbox_selection(self, color):
        global box_points, selecting_bbox
        if not selecting_bbox: return True
        if len(box_points) < 1:
            display_img = color.copy()
            self.pump_windows(display_img, self.depth_vis if self.got_depth else None)
            return False
        elif len(box_points) < 2:
            display_img = color.copy()
            for pt in box_points: cv2.circle(display_img, pt, radius=5, color=(0, 255, 0), thickness=-1)
            self.pump_windows(display_img, self.depth_vis if self.got_depth else None)
            return False
        else:
            selecting_bbox = False
            return True

    def create_mask(self, depth, bbox_points):
        x1, y1 = bbox_points[0]
        x2, y2 = bbox_points[1]
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        mask = np.zeros_like(depth, dtype=bool)
        mask[y_min:y_max, x_min:x_max] = True
        return mask

    def draw_conf_bar(self, img, value, label="confidence", origin=(10, 30), size=(220, 18), max_val=1.0):
        val = float(value.get(label, 0.0))
        mv = max(1e-6, float(max_val))
        v = max(0.0, min(val / mv, 1.0))
        x, y = origin
        w, h = size
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), thickness=-1)
        cv2.rectangle(img, (x, y), (x + int(w * v), y + h), (60, 180, 75), thickness=-1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (220, 220, 220), thickness=1)
        cv2.putText(img, f"{label}: {val:.3f}", (x, y - 6),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    def init_via_yolo_roi(self, detector, color, depth, K, est, est_refine_iter, roi_expand, det_imgsz, det_conf, prefer_cls, do_register=True, lock_count=0, lock_target=0):
        H, W = color.shape[:2]
        
        det_xyxy, det_score, det_cls, det_mask_data = self.yolo_det_xyxy_mask(
            detector, color, imgsz=det_imgsz, conf=det_conf, prefer_cls=prefer_cls, mode=self.det_select_mode
        )

        if det_xyxy is None:
            vis = color.copy()
            cv2.putText(vis, "YOLO ROI not found - waiting...", (20, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            self.pump_windows(vis, self.depth_vis if self.got_depth else None)
            return None, None, None, None

        if not do_register:
            vis = color.copy()
            x1, y1, x2, y2 = det_xyxy.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(vis, f"Locking ROI... {lock_count}/{lock_target}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            self.pump_windows(vis, self.depth_vis if self.got_depth else None)
            return "locking", None, det_xyxy, det_score

        mask = self.get_segmentation_mask(color, depth, det_xyxy, det_mask_data)
        
        t0 = time.perf_counter()
        pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter, top_k=self.est_top_k)
        if self.perf_eval_enable: 
            self.time_init = (time.perf_counter() - t0) * 1000.0

        init_vis = color.copy()
        x1, y1, x2, y2 = det_xyxy.astype(int)
        cv2.rectangle(init_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if self.seg_backend != 'bbox':
            init_vis[mask] = init_vis[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
            
        cv2.putText(init_vis, f"INIT ROI s={det_score:.2f} ({self.det_select_mode})", (x1, max(0, y1 - 8)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        self.pump_windows(init_vis, self.depth_vis if self.got_depth else None)
        return pose, mask, det_xyxy, det_score

    def periodic_yolo_iou(self, frame_count, stride, detector, color, center_pose, bbox_minmax, K, det_imgsz, det_conf, prefer_cls, vis_bgr, log=False):
        iou_val = None
        if detector is None or center_pose is None or (frame_count % max(1, stride) != 0):
            if self._last_yolo_text:
                cv2.putText(vis_bgr, self._last_yolo_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)
            return vis_bgr, None
    
        H, W = color.shape[:2]
        est_xyxy = self.project_3d_bbox_xyxy(K, center_pose, bbox_minmax, img_shape=color.shape)
        if est_xyxy is None:
            xyxy_all, sc_all, cl_all = self.yolo_det_all(detector, color, imgsz=det_imgsz, conf=det_conf)
            self._last_yolo_text = f"YOLO det={len(xyxy_all)}"
            cv2.putText(vis_bgr, self._last_yolo_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
            return vis_bgr, None

        xyxy_all, sc_all, cl_all = self.yolo_det_all(detector, color, imgsz=det_imgsz, conf=det_conf)
        if len(xyxy_all) == 0:
            self._last_yolo_text = "YOLO det=0"
            cv2.putText(vis_bgr, self._last_yolo_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
            return vis_bgr, 0.0

        use_mask = np.ones(len(xyxy_all), dtype=bool)
        if prefer_cls is not None and (cl_all == prefer_cls).any(): use_mask = (cl_all == prefer_cls)

        xyxy_use = xyxy_all[use_mask]
        sc_use = sc_all[use_mask]
        cl_use = cl_all[use_mask]

        ious = []
        for bb in xyxy_use:
            bb_clipped = self.clip_xyxy(bb, W, H)
            ious.append(self.iou_xyxy(bb_clipped, est_xyxy))
        ious = np.array(ious, dtype=float)

        if len(ious) == 0:
            self._last_yolo_text = f"YOLO det={len(xyxy_all)} (no valid)"
            cv2.putText(vis_bgr, self._last_yolo_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)
            return vis_bgr, None

        best_idx = int(np.argmax(ious))
        iou_val  = float(ious[best_idx])
        best_cls = int(cl_use[best_idx])
        best_sc  = float(sc_use[best_idx])

        self._last_yolo_text = f"YOLO det={len(xyxy_all)} best cls={best_cls} s={best_sc:.2f} IoU={iou_val:.3f}"
        cv2.putText(vis_bgr, self._last_yolo_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)
        return vis_bgr, iou_val

    def mat4_to_translation_quat(self, T: np.ndarray):
        assert T.shape == (4, 4)
        t = (float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
        qx, qy, qz, qw = tf.transformations.quaternion_from_matrix(T)
        q = (float(qx), float(qy), float(qz), float(qw))
        return t, q

    def mat4_to_pose(self, T: np.ndarray) -> Pose:
        t, q = self.mat4_to_translation_quat(T)
        msg = Pose()
        msg.position.x, msg.position.y, msg.position.z = t
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = q
        return msg
    
    def broadcast_transform_and_pose(self, transform: Transform, object_name: str, camera_tf: str):
        t = (transform.translation.x, transform.translation.y, transform.translation.z)
        q = (transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w)
        self.tf_broadcaster.sendTransform(t, q, rospy.Time.now(), object_name, camera_tf)
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = t
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
        self.pose_pub.publish(pose)

    def imageCallback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e: return
        self.color = img
        self.got_rgb = True
        self.rgb_size = (img.shape[1], img.shape[0])

    def infoCallback(self, msg: CameraInfo):
        self.info_msg = msg
        if self.K is None:
            self.K = np.array([[msg.K[0], 0, msg.K[2]],
                               [0, msg.K[4], msg.K[5]],
                               [0, 0, 1]], dtype=np.float64)

    def depthCallback(self, msg: Image):
        self.depth_encoding = msg.encoding
        try: d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.float32)
        except Exception as e: return

        if msg.encoding.upper() in ("16UC1", "TYPE_16UC1"): depth_m = d * 0.001 
        else: depth_m = d if np.nanmax(d) <= 10.0 else d * 0.001
        self.depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)

        maxd_mm = max(1.0, float(self.max_depth_mm))
        depth_mm_for_vis = np.clip(self.depth_m * 1000.0, 0.0, maxd_mm)
        depth_8u = (depth_mm_for_vis * (255.0 / maxd_mm)).astype(np.uint8)
        if self.invert_colormap: depth_8u = 255 - depth_8u
        self.depth_vis = cv2.applyColorMap(depth_8u, self.colormap_id)

        self.got_depth = True
        self.depth_size = (self.depth_vis.shape[1], self.depth_vis.shape[0])

    def detectionCallback(self, msg: Detection):
        self.ready_received.detection_allowed = msg.detection_allowed
        self.ready_received.layer = msg.layer
        if msg.layer == 0.0: self.det_select_mode = "score"
        elif msg.layer == 1.0: self.det_select_mode = "nearest_depth"
        elif msg.layer == 2.0: self.det_select_mode = "middle"
        else: self.det_select_mode = "score"

    def confidence_publish(self, score: float, detection: bool):
        conf_msg = Confidence()
        conf_msg.stamp = rospy.Time.now()
        conf_msg.frame_id = self.object_name
        conf_msg.object_IoU = float(score)
        conf_msg.object_detection = detection
        self.conf_pub.publish(conf_msg)

    def spin(self):
        self.frame_count = 0
        first_frame = True
        self.ready_received = Detection()
        self.ready_received.detection_allowed = False
        self.ready_received.layer = 0.0
        
        while not rospy.is_shutdown():
            self.time_det = 0.0
            self.time_seg = 0.0
            self.time_init = 0.0
            self.time_refine = 0.0

            if not (self.got_depth and self.got_rgb and self.K is not None and hasattr(self, "depth_m")):
                self.pump_windows(self.color if self.got_rgb else None, self.depth_vis if self.got_depth else None)
                rospy.sleep(0.01)
                continue

            if (self.init_mode == 'yolo' and self.yolo_start_mode == 'wait' and not self.ready_received.detection_allowed):
                self.pose = None
                first_frame = True
                self.consecutive_det_count = 0 
                self.tracking_start_time = None
                self.iou_bad_count = 0
                self.iou_val = None

                vis_rgb = self.color.copy() if self.color is not None else np.zeros((480,640,3), np.uint8)
                cv2.putText(vis_rgb, "DETECTION DISABLED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                self.pump_windows(vis_rgb if self.show_rgb_win else None, self.depth_vis if (self.show_depth_win and hasattr(self, "depth_vis")) else None)

                self.confidence_publish(0.0, False)
                continue

                
            self.frame_count += 1
            
            if self.init_mode == 'click':
                if selecting_bbox:
                    if not self.update_bbox_selection(self.color): continue
                    else:
                        self.mask = self.create_mask(self.depth_m, box_points)
                        t0 = time.perf_counter()
                        self.pose = self.est.register(K=self.K, rgb=self.color, depth=self.depth_m, ob_mask=self.mask, iteration=self.est_refine_iter, top_k=self.est_top_k)
                        if self.perf_eval_enable: 
                            self.time_init = (time.perf_counter() - t0) * 1000.0
                            self.time_det = 0.0
                            self.time_seg = 0.0 
                        box_points.clear()
                        first_frame = False
                elif first_frame:
                    self.select_bbox(self.color)
                    continue 
            else:
                if first_frame:
                    self.consecutive_det_count += 1
                    do_register = (self.consecutive_det_count >= self.init_det_patience)
                    
                    init = self.init_via_yolo_roi(self.detector, self.color, self.depth_m, self.K, self.est,self.est_refine_iter, self.roi_expand,self.det_imgsz, self.det_conf, self.prefer_cls,do_register=do_register,lock_count=self.consecutive_det_count,lock_target=self.init_det_patience)
                    
                    if init[0] is None:
                        self.consecutive_det_count = 0 
                        continue 
                    elif init[0] == "locking":
                        continue
                        
                    # 獲取 BBox 數據
                    self.pose, self.mask, det_xyxy, det_score = init
                    
                    # 儲存第一幀的 BBox 與 Seg 影像
                    if self.perf_eval_enable and self.mask is not None and self.debug_dir:
                        timestamp_str = datetime.now().strftime("%H%M%S")
                        cycle_idx = self.track_save_count + 1
                        
                        # 1. 存 BBox
                        bbox_img = self.color.copy()
                        if det_xyxy is not None:
                            x1, y1, x2, y2 = det_xyxy.astype(int)
                            cv2.rectangle(bbox_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.imwrite(os.path.join(self.debug_dir, f"cycle{cycle_idx}_bbox_{timestamp_str}.jpg"), bbox_img)
                        
                        # 2. 存 Segmentation
                        seg_img = self.color.copy()
                        seg_img[self.mask > 0] = [0, 255, 0]
                        cv2.imwrite(os.path.join(self.debug_dir, f"cycle{cycle_idx}_seg_{timestamp_str}.jpg"), seg_img)

                    first_frame = False
                    self.consecutive_det_count = 0

            # === 追蹤 ===
            t1 = time.perf_counter()
            track_extra = {}
            enable_depth_self_check = (self.reinit_mode == "depth_confidence")
            
            self.pose = self.est.track_one(rgb=self.color,depth=self.depth_m,K=self.K,iteration=self.track_refine_iter,extra=track_extra,enable_self_check=enable_depth_self_check)

            self.depth_mae = track_extra.get("depth_mae", None)
            self.inlier_ratio = track_extra.get("inlier_ratio", None)
            if self.depth_mae is not None and self.inlier_ratio is not None:
                tau_d = max(float(self.geom_mae_thresh), 1e-6)
                self.depth_conf_score = float(self.inlier_ratio) * float(np.exp(-float(self.depth_mae) / tau_d))
            else:
                self.depth_conf_score = None

            if self.perf_eval_enable: 
                self.time_refine = (time.perf_counter() - t1) * 1000.0

            vis_bgr = self.color.copy()
            if self.pose is not None:
                center_pose = self.pose @ np.linalg.inv(self.to_origin)
                color_rgb = cv2.cvtColor(self.color, cv2.COLOR_BGR2RGB)
                
                t, q = self.mat4_to_translation_quat(self.pose)
                tr = Transform()
                tr.translation.x, tr.translation.y, tr.translation.z = t
                tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w = q
                parent_frame = self.camera_tf if self.camera_tf else "camera_color_optical_frame"
                self.broadcast_transform_and_pose(tr, self.object_name, parent_frame)

                vis = draw_posed_3d_box(self.K, img=color_rgb, ob_in_cam=center_pose, bbox=self.bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=self.pose, scale=0.05, K=self.K, thickness=3, transparency=0, is_input_rgb=True)
                vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

                # ==========================================
                # 5 秒自動重置計時器 & 第一幀儲存邏輯
                # ==========================================
                if self.tracking_start_time is None:
                    self.tracking_start_time = time.time()

                if not self.first_track_saved and self.debug_dir:
                    self.track_save_count += 1
                    timestamp_str = datetime.now().strftime("%H%M%S")
                    save_path = os.path.join(self.debug_dir, f"cycle{self.track_save_count}_pose_{timestamp_str}.jpg")
                    cv2.imwrite(save_path, vis_bgr)
                    rospy.loginfo(f"[TEST] 已儲存第 {self.track_save_count} 次循環的 BBox/Seg/Pose 截圖")
                    self.first_track_saved = True

                # 只有開啟效能評估模式時，才啟用 5 秒自動 reinit
                if self.perf_eval_enable:
                    if time.time() - self.tracking_start_time >= 5.0:
                        rospy.logwarn(f"[TEST] {self.seg_backend} 追蹤已達 5 秒，觸發自動重置！")
                        first_frame = True
                        self.pose = None
                        self.consecutive_det_count = 0
                        self.tracking_start_time = None
                        self.first_track_saved = False  # 允許下次重置後存圖
                        continue
                
                # ==========================================
                # Reinit 判斷模式：
                #   1) yolo_iou：YOLO bbox 與 projected 3D bbox 計算 IoU
                #   2) depth_confidence：使用 Inlier Ratio 與 Depth MAE
                # ==========================================
                confidence_score = None

                if self.reinit_mode == "yolo_iou":
                    if self.init_mode == 'yolo':
                        vis_bgr, _new_iou = self.periodic_yolo_iou(frame_count=self.frame_count,stride=self.iou_stride,detector=self.detector,color=self.color,center_pose=center_pose,bbox_minmax=self.bbox,K=self.K,det_imgsz=self.det_imgsz,det_conf=self.det_conf,prefer_cls=self.prefer_cls,vis_bgr=vis_bgr,log=self.iou_log)

                        if _new_iou is not None:
                            self.iou_val = float(_new_iou)

                        if self.iou_val is not None:
                            confidence_score = float(self.iou_val)

                            # 左下角顯示 YOLO-IoU 數值，不使用 draw_conf_bar
                            cv2.putText(vis_bgr,f"YOLO-IoU={self.iou_val:.3f} bad={self.iou_bad_count}/{self.iou_patience}",(10, vis_bgr.shape[0] - 18),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0, 0, 255),1,cv2.LINE_AA)

                            if self.iou_val < self.iou_thresh:
                                self.iou_bad_count += 1
                            else:
                                self.iou_bad_count = 0

                            if self.iou_bad_count >= self.iou_patience:
                                rospy.logwarn(
                                    f"[Reinit][YOLO-IoU] IoU={self.iou_val:.3f} < {self.iou_thresh:.3f} "
                                    f"for {self.iou_bad_count} frames, reinit."
                                )
                                self.iou_bad_count = 0
                                self.geom_bad_count = 0
                                first_frame = True
                                self.pose = None
                                self.consecutive_det_count = 0
                                self.tracking_start_time = None
                                self.first_track_saved = False

                elif self.reinit_mode == "depth_confidence":
                    if self.inlier_ratio is not None and self.depth_mae is not None:
                        confidence_score = float(self.depth_conf_score) if self.depth_conf_score is not None else float(self.inlier_ratio)

                        # 左下角顯示 Inlier Ratio 與 Depth MAE，不使用 draw_conf_bar
                        cv2.putText(vis_bgr,f"Inlier={self.inlier_ratio:.3f} MAE={self.depth_mae*100.0:.1f}cm bad={self.geom_bad_count}/{self.geom_patience}",(10, vis_bgr.shape[0] - 18),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0, 0, 255),1,cv2.LINE_AA)

                        geom_bad = (
                            float(self.inlier_ratio) < float(self.geom_inlier_thresh)
                            or float(self.depth_mae) > float(self.geom_mae_thresh)
                        )

                        if geom_bad:
                            self.geom_bad_count += 1
                        else:
                            self.geom_bad_count = 0

                        if self.geom_bad_count >= self.geom_patience:
                            rospy.logwarn(
                                f"[Reinit][Depth] inlier={self.inlier_ratio:.3f} < {self.geom_inlier_thresh:.3f} "
                                f"or mae={self.depth_mae:.4f}m > {self.geom_mae_thresh:.4f}m "
                                f"for {self.geom_bad_count} frames, reinit."
                            )
                            self.iou_bad_count = 0
                            self.geom_bad_count = 0
                            first_frame = True
                            self.pose = None
                            self.consecutive_det_count = 0
                            self.tracking_start_time = None
                            self.first_track_saved = False
                    else:
                        cv2.putText(vis_bgr,"DepthConf unavailable: update estimater.track_one(extra=..., enable_self_check=True)",(10, 25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),1,cv2.LINE_AA)
            
            self.pump_windows(vis_bgr if (self.show_rgb_win and self.color is not None) else None,
                              self.depth_vis if (self.show_depth_win and self.got_depth and self.depth_vis is not None) else None)
            if self.reinit_mode == "yolo_iou" and self.iou_val is not None:
                self.confidence_publish(float(self.iou_val), not first_frame)
            elif self.reinit_mode == "depth_confidence" and self.depth_conf_score is not None:
                self.confidence_publish(float(self.depth_conf_score), not first_frame)
            else:
                self.confidence_publish(0.0, False)

            if self.perf_eval_enable and getattr(self, "perf_csv_path", None) is not None:
                with open(self.perf_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    fmt = lambda t: round(t, 2) if t > 0 else ""
                    writer.writerow([self.frame_count, self.det_model_name, self.seg_model_name, fmt(self.time_det), fmt(self.time_seg), fmt(self.time_init), fmt(self.time_refine)])
                
        cv2.destroyAllWindows()

if __name__ == "__main__":
    rospy.init_node("foundationpose_tracker", anonymous=False)
    node = FoundationPoseTracker()
    node.spin()
