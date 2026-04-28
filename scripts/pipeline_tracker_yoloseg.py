#!/home/user/anaconda3/envs/foundationpose/bin/python3
# -*- coding: utf-8 -*-

import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import rospy
import numpy as np
import cv2
import trimesh
import torch
import psutil
import time

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Transform, Point, Quaternion, Twist
from cv_bridge import CvBridge
from forklift_msg.msg import Confidence, Detection
from std_msgs.msg import Bool
from datetime import datetime
import torchvision.transforms.functional as F

import tf

# --- 專案路徑 ---
import sys
EXTRA_PATHS = [
    "/home/user/anaconda3/envs/foundationpose/lib/python3.8/site-packages",
    "/home/user/FoundationPose",
]

for p in EXTRA_PATHS:
    if p not in sys.path:
        sys.path.append(p)

# --- 載入 Cutie VOS 模組 ---
CUTIE_PATH = "/home/user/Cutie" 
if CUTIE_PATH not in sys.path:
    sys.path.append(CUTIE_PATH)
try:
    from cutie.inference.inference_core import InferenceCore
    from cutie.utils.get_default_model import get_default_model
    CUTIE_AVAILABLE = True
except ImportError:
    CUTIE_AVAILABLE = False
    print("[WARNING] Cutie module not found. Stem Servoing will fallback or fail.")

os.environ.setdefault("ULTRALYTICS_NO_INSTALL", "1")

from estimater import FoundationPose, draw_posed_3d_box, draw_xyz_axis, ScorePredictor, PoseRefinePredictor, dr
from ultralytics import YOLO

selecting_bbox = False
box_points = []

class FoundationPosePipelineTracker:
    def __init__(self):
        self.init_parameter()
        # debug_dir: {debug_root}/{yyyyMMdd-HHmm}/
        if self.iou_log:
            self._setup_run_debug_dir()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # 基本狀態
        self.bridge = CvBridge()
        self.color = None
        self.depth = None
        self.depth_encoding = None
        self.got_depth = False
        self.got_rgb = False
        self.depth_size = (0, 0)
        self.rgb_size = (0, 0)
        self.pose_bunch = None
        self.pose_stem  = None
        self.mask = None
        self.frame_count = 0
        self.iou_bad_count = 0
        self.iou_val = None
        self.iou_bad_count_bunch = 0
        self.last_iou_update_bunch = -1
        self.K = None
        self._last_yolo_text = ""
        
        self.ready_received = Detection()
        self.ready_received.detection_allowed = False
        self.det_select_mode_current = getattr(self, "det_select_mode", "score")
        self._force_bunch_detect = False   # reinit 後強制果串檢測直到被 block
        self._stem_lock = False
        self._pause_hold = False
        self._last_allowed = False

        # --- Cutie 初始化 ---
        if CUTIE_AVAILABLE:
            rospy.loginfo("[Cutie] Loading weights...")
            self.cutie_net = get_default_model()
            self.cutie_net.eval().cuda()
            self.cutie_processor = InferenceCore(self.cutie_net, cfg=self.cutie_net.cfg)
            self.cutie_processor.max_internal_size = 640 # 限制解析度防斷電

        self.bunch_cutie_state = "CRUISING"  # CRUISING / SERVOING
        self.stem_cutie_state = "CRUISING"   # CRUISING / SERVOING
        self.erode_iterations = 2             # 腐蝕次數(防多割)
        self.erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        # --- 葉莖持續追蹤 (Tracking) 狀態 ---        
        self.last_stem_3d_pos = None  # np.array([x, y, z]) 紀錄最後一次算出的葉莖相機座標
        self.last_stem_time = None    # rospy.Time 紀錄最後一次更新的時間
        self.stem_lost_timeout = rospy.Duration(self.stem_lost) 
        
        # YOLO 延時 debounce 狀態 (只保留 Bunch 的，Stem 已改用預測追蹤)
        self._yolo_delay_left_bunch = 0
        self._yolo_delay_bbox_bunch = None

        # Seg 暖機幀數計數器
        self.seg_warmup_left_bunch = 0

        # 遮蔽率效能分析器
        self.perf_nn_times = []     # 紀錄 AI 推論時間
        self.perf_mesh_times = []   # 紀錄網格交集時間
        self.perf_occ_cpu = []      # 紀錄整體 CPU 佔用
        self.perf_process = psutil.Process(os.getpid()) if 'psutil' in sys.modules else None

        # GUI 狀態
        self._rgb_win_created = False
        self._depth_win_created = False
        self._rgb_win_sized = False
        self._depth_win_sized = False
        self._rgb_initial_size = (900, 720)
        self._depth_initial_size = (900, 720)

        # 後處理 debounce 狀態
        self._post_pending = False
        self._post_fail_time = None

        # ROS Pub/Sub
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.imageCallback, queue_size=1)
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self.depthCallback, queue_size=1)
        self.info_sub  = rospy.Subscriber(self.info_topic,  CameraInfo, self.infoCallback, queue_size=1)
        self.cmd_vel_sub = rospy.Subscriber(self.cmd_vel_topic, Twist, self.cmd_vel_callback, queue_size=1)
        self.current_twist = Twist()

        self.tf_broadcaster = tf.TransformBroadcaster()

        self.conf_pub = rospy.Publisher("FFB_state", Confidence, queue_size=1, latch=True)
        self.harvest_done_sub = rospy.Subscriber("FFB_state" + "_harvest_done", Bool, self.harvestDoneCallback, queue_size=1)
        if self.yolo_start_mode == "wait":
            self._ready_sub = rospy.Subscriber("FFB_state" + "_detection", Detection, self.detectionCallback, queue_size=1)

        self.window_create()

        # FoundationPose 初始化
        os.makedirs(self.debug_dir, exist_ok=True)
        self.mesh_bunch = trimesh.load(self.mesh_file)
        self.to_origin_bunch, self.extents_bunch = trimesh.bounds.oriented_bounds(self.mesh_bunch) # np.eye(4),self.mesh_bunch.extents
        self.gt_to_origin_bunch, self.gt_extents_bunch = np.eye(4),self.mesh_bunch.extents
        self.bbox_bunch = np.stack([-self.extents_bunch/2, self.extents_bunch/2], axis=0).reshape(2, 3)

        # Score/Refine/Raster
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()

        self.est_bunch = FoundationPose(model_pts=self.mesh_bunch.vertices,
                                        model_normals=self.mesh_bunch.vertex_normals,
                                        mesh=self.mesh_bunch, scorer=self.scorer, refiner=self.refiner,
                                        debug_dir=self.debug_dir, debug=0, glctx=self.glctx)
        
        # YOLOv11
        self.detector, self.det_device = self.load_detector(self.det_model)
        # --- YOLO warmup (force ORT session creation) ---
        try:
            dummy = np.zeros((self.det_imgsz, self.det_imgsz, 3), np.uint8)
            _ = self.detector.predict(source=dummy, imgsz=self.det_imgsz, conf=0.01, device=self.det_device, verbose=False)
        except Exception as e:
            rospy.logwarn(f"[YOLO warmup] failed: {e}")

        is_gpu, yolo_desc = self.yolo_uses_gpu(self.detector)
        rospy.loginfo(f"[YOLO] predict device hint: {self.det_device}")
        rospy.loginfo(f"[YOLO] GPU enabled: {is_gpu}  ({yolo_desc})")
        rospy.loginfo("Detector initialization done")
        
        # YOLOv11 seg (YOLO11-seg)
        self.seg_detector = None
        if self.seg_backend == "yolo_seg":
            try:
                # task="segment"
                self.seg_detector = YOLO(self.seg_model, task="segment")
                rospy.loginfo(f"[YOLO-SEG] loaded: {self.seg_model}")

                # warmup
                dummy = np.zeros((self.seg_imgsz, self.seg_imgsz, 3), np.uint8)
                _ = self.seg_detector.predict(
                    source=dummy, imgsz=self.seg_imgsz, conf=0.01,
                    device=self.det_device, verbose=False
                )
                rospy.loginfo("[YOLO-SEG] warmup done")
            except Exception as e:
                rospy.logwarn(f"[YOLO-SEG] init failed: {e}")
                self.seg_detector = None

        # 狀態機
        self.mode = "bunch"     # bunch / stem
        self._hi_cnt = 0
        self._registering_until = 0
        self._reinit_until = 0

    # ---------------------------
    # 參數
    # ---------------------------
    def init_parameter(self):
        ns = rospy.get_name()
        gp = lambda k, d: rospy.get_param(ns + "/" + k, d)

        # Topics / frames
        self.image_topic = gp("image_topic", "/camera/color/image_raw")
        self.info_topic = gp("info_topic",  "/camera/color/camera_info")
        self.depth_topic = gp("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.depth_info_topic = gp("depth_info_topic", "")
        self.camera_tf = gp("camera_tf", "")
        self.bunch_name = gp("bunch_name", "oilpalm")
        self.stem_name = gp("stem_name", "stem")
        self.camera_side = gp("camera_side", "right").strip().lower()
        self.camera_tag_offset_x = float(gp("camera_tag_offset_x", 0.0))

        # Files & modes
        self.mesh_file = gp("mesh_file", "")
        self.det_model = gp("det_model", "yolov11n.onnx")
        self.init_mode = gp("init_mode", "yolo")
        self.yolo_start_mode = gp("yolo_start_mode", "immediate").strip().lower()
        self.debug_root = gp("debug_root", gp("debug_dir", "/tmp/fp_debug"))
        self.debug_dir = self.debug_root

        # YOLO
        self.det_conf = float(gp("det_conf", 0.25))
        self.det_class = int(gp("det_class", -1))
        self.det_imgsz = int(gp("det_imgsz", 640))
        self.prefer_cls = None if self.det_class < 0 else self.det_class
        self.det_select_mode = gp("det_select_mode", "score").strip().lower()

        # FoundationPose iters
        self.est_refine_iter = int(gp("est_refine_iter", 5))
        self.track_refine_iter = int(gp("track_refine_iter", 2))

        # IoU check / ROI
        self.roi_expand = float(gp("roi_expand", 0.01))
        self.iou_stride = int(gp("iou_stride", 3))
        self.iou_log = bool(gp("iou_log", False))
        self.iou_thresh = float(gp("iou_thresh", 0.25))
        self.iou_patience = int(gp("iou_patience", 3))

        # Windows
        self.show_depth_win = bool(gp("show_depth_window", False))
        self.show_rgb_win = bool(gp("show_rgb_window", True))
        self.depth_win_name = gp("depth_win_name", "Depth")
        self.rgb_win_name = gp("rgb_win_name", "RGB")
        self.depth_win_xy = gp("depth_window_xy", [100,100])
        self.rgb_win_xy = gp("rgb_window_xy", [100,500])
        self.max_depth_mm = float(gp("max_depth_mm", 2000.0))
        self.colormap_id = int(gp("colormap", int(cv2.COLORMAP_JET)))
        self.invert_colormap= bool(gp("invert_colormap", False))

        # 後處理（共用）
        self.pp_enable = bool(gp("postproc/enable", True))
        self.pp_orient_center_tol_px = float(gp("postproc/orient_center_tol_px", 20.0))

        # 果串（BUNCH）後處理
        self.bunch_expect_orientation = gp("postproc/bunch/expect_orientation", "inverted").strip().lower()
        self.bunch_size_mode = gp("postproc/bunch/size_mode", "bbox_mm").strip().lower()
        self.bunch_expect_bbox_w_mm = float(gp("postproc/bunch/expect_bbox_w_mm", 115.0))
        self.bunch_expect_bbox_h_mm = float(gp("postproc/bunch/expect_bbox_h_mm", 80.0))
        self.bunch_size_ratio_min = float(gp("postproc/bunch/size_ratio_min", 0.6))
        self.bunch_expect_depth_m = float(gp("postproc/bunch/expect_depth_m", 1.2))
        self.bunch_depth_tol_m = float(gp("postproc/bunch/depth_tolerance_m", 0.25))

        # 葉莖（STEM）參數
        self.cmd_vel_topic = gp("postproc/stem/cmd_vel_topic", "/cmd_vel")
        self.stem_lost = float(gp("postproc/stem/stem_lost", 3.0))
        self.stem_depth_min = float(gp("postproc/stem/stem_depth_min", 0.1))
        self.stem_depth_max = float(gp("postproc/stem/stem_depth_max", 3.0))
        self.stem_assumed_radius = float(gp("postproc/stem/stem_assumed_radius", 0.02))
        self.tracker_type = gp("tracker_type", "bytetrack.yaml")
        self.stem_max_jump_m = float(gp("postproc/stem/max_jump_m", 0.06)) 
        self.stem_ema_alpha = float(gp("postproc/stem/ema_alpha", 0.6))

        # 打包尺寸設定
        self.cfg_bunch = dict(
            size_mode=self.bunch_size_mode,
            expect_bbox_w_mm=self.bunch_expect_bbox_w_mm,
            expect_bbox_h_mm=self.bunch_expect_bbox_h_mm,
            size_ratio_min=self.bunch_size_ratio_min,
            expect_depth_m=self.bunch_expect_depth_m,
            depth_tol_m=self.bunch_depth_tol_m,
        )

        # 類別 id
        self.cls_bunch = int(gp("classes/bunch", 0))
        self.cls_stem = int(gp("classes/stem",  1))

        # 遮蔽率策略
        self.policy_occ_hi = float(gp("policy/occ_thresh_high", 0.60))
        self.policy_hi_pat = int(gp("policy/high_patience", 3))

        # 分割後端
        self.seg_backend = gp("postproc/seg_backend", "yolo_seg").strip().lower()  # sam | bbox | yolo_seg
        self.seg_model = gp("postproc/seg_model", "vit_h").strip()
        self.seg_imgsz = int(gp("postproc/seg_imgsz", 640))
        self.occ_seg_warmup_n = int(gp("occ/seg_warmup_n", 8))

        # reinit debounce 秒數
        self.pp_retry_delay_sec = float(gp("postproc/retry_delay_sec", 1.0))

    # =========================
    # YOLO / 幾何 / 工具
    # =========================
    def yolo_backend_info(self,detector):
        """獲取 YOLO 模型當前運行的後端資訊 (如 PyTorch / ONNXRuntime 裝置)。"""
        info = {"engine": None, "torch_device": None, "ort_providers": None}
        m = getattr(detector, "model", None)
        if m is None:
            return info
        info["engine"] = str(getattr(m, "backend", getattr(m, "engine", "")))
        try:
            if hasattr(m, "parameters"):
                p = next(m.parameters(), None)
                if p is not None:
                    info["torch_device"] = str(p.device)
        except Exception:
            pass
        sess = None
        for attr in ("session", "ort_session", "session_ort"):
            s = getattr(m, attr, None)
            if s is not None:
                sess = s; break
        if sess is not None:
            try:
                info["ort_providers"] = list(sess.get_providers())
            except Exception:
                pass
        else:
            prov = getattr(m, "providers", None)
            if prov is not None:
                info["ort_providers"] = list(prov)
        return info

    def yolo_uses_gpu(self,detector):
        """判斷 YOLO 模型是否成功使用了 GPU 加速。"""
        info = self.yolo_backend_info(detector)
        eng = (info["engine"] or "").lower()
        dev = (info["torch_device"] or "").lower()
        prov = info["ort_providers"] or []
        if any("CUDAExecutionProvider" == p for p in prov):
            return True, f"engine={eng or 'onnxruntime'} providers={prov}"
        if "tensorrt" in eng:
            return True, f"engine={eng}"
        if dev.startswith("cuda"):
            return True, f"engine={eng or 'pytorch'} device={dev}"
        return False, f"engine={eng or 'unknown'} providers={prov or 'None'} device={dev or 'None'}"

    def load_detector(self, model_path: str):
        """
        載入 YOLO 模型。
        支援 .pt (PyTorch) 與 .onnx (ONNXRuntime)。
        若讀取 .pt 失敗，會自動尋找同名的 .onnx 作為備案。
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"detector model not found: {model_path}")
        def _onnx_sibling(p):
            stem, ext = os.path.splitext(p)
            return stem + ".onnx"
        ext = os.path.splitext(model_path)[1].lower()
        det_device = "cpu"
        if ext == ".pt":
            rospy.loginfo(f"[YOLO Loader] Loading PyTorch (.pt): {model_path}")
            try:
                det = YOLO(model_path)
                if torch.cuda.is_available():
                    try:
                        det.to("cuda:0"); det_device = 0
                        return det, det_device
                    except Exception as ge:
                        rospy.logwarn(f"[YOLO Loader] PT move to GPU failed: {ge}. Will try PT on CPU.")
                else:
                    rospy.logwarn("[YOLO Loader] No CUDA available; PT will run on CPU.")
                try:
                    det.to("cpu"); det_device = "cpu"
                    return det, det_device
                except Exception as ce:
                    rospy.logwarn(f"[YOLO Loader] PT on CPU failed: {ce}")
            except Exception as e:
                rospy.logwarn(f"[YOLO Loader] PT load failed: {e}")
            onnx_fallback = _onnx_sibling(model_path)
            if os.path.isfile(onnx_fallback):
                det, det_device = self._load_onnx_with_gpu_fallback(onnx_fallback)
                return det, det_device
            else:
                raise RuntimeError(f"Failed to load PT '{model_path}' and no sibling ONNX found.")
        elif ext == ".onnx":
            det, det_device = self._load_onnx_with_gpu_fallback(model_path)
            return det, det_device
        else:
            raise ValueError(f"Unsupported detector extension: {ext}. Use .pt or .onnx")
        
    def _load_onnx_with_gpu_fallback(self, onnx_path: str):
        """載入 ONNX YOLO 模型，並嘗試強制使用 CUDAExecutionProvider。"""
        det = YOLO(onnx_path, task="detect")
        sess = None
        for attr in ("session","ort_session","session_ort"):
            s = getattr(det.model, attr, None)
            if s is not None:
                sess = s; break
        device_hint = 0 if torch.cuda.is_available() else "cpu"
        device_desc = "cuda" if device_hint == 0 else "cpu"
        if sess is not None:
            try:
                provs = list(sess.get_providers())
            except Exception:
                provs = []

            if "CUDAExecutionProvider" not in provs:
                try:
                    sess.set_providers(["CUDAExecutionProvider","CPUExecutionProvider"])
                    provs = list(sess.get_providers())
                except Exception:
                    pass

            if "CUDAExecutionProvider" in provs:
                device_hint = 0
                device_desc = f"ORT CUDA providers={provs}"
            else:
                device_hint = "cpu"
                device_desc = f"ORT CPU providers={provs}"

        rospy.loginfo(f"[YOLO Loader] ONNX runtime: {device_desc}")
        return det, device_hint

    def clip_xyxy(self, xyxy, W, H):
        """將 Bounding Box 座標限制在圖片邊界 (W, H) 內，避免越界。"""
        if xyxy is None:
            return None
        x1, y1, x2, y2 = map(float, xyxy)
        return np.array([max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)], dtype=np.float32)

    def rect_to_mask(self, depth, xyxy, expand=0.0):
        """根據 2D Bounding Box 產生對應的二值化 (Boolean) Mask。可透過 expand 參數向外擴張。"""
        if xyxy is None: return None
        H, W = depth.shape[:2]
        x1,y1,x2,y2 = xyxy.astype(np.int32)
        w,h = max(1,x2-x1), max(1,y2-y1)
        x1 = max(0, int(x1 - w*expand))
        y1 = max(0, int(y1 - h*expand))
        x2 = min(W-1, int(x2 + w*expand))
        y2 = min(H-1, int(y2 + h*expand))
        m = np.zeros((H,W), dtype=bool)
        m[y1:y2, x1:x2] = True
        return m

    def yolo_det_all(self, detector: YOLO, img_bgr, imgsz=640, conf=0.25):
        """執行 YOLO Tracking，並回傳 BBox、信心度、類別與 Track ID。"""
        r = detector.track(source=img_bgr, imgsz=imgsz, conf=conf, device=self.det_device, 
                           persist=True, tracker=getattr(self, "tracker_type", "bytetrack.yaml"), verbose=False)[0]
        
        if len(r.boxes) == 0:
            return (np.empty((0,4), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int32), np.empty((0,), np.int32))

        xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        sc = r.boxes.conf.cpu().numpy().astype(np.float32)
        cl = r.boxes.cls.cpu().numpy().astype(np.int32)

        if r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(np.int32)
        else:
            ids = np.full((len(cl),), -1, dtype=np.int32)
            
        return xyxy, sc, cl, ids

    def pick_top1(self, img_bgr, cls_id, conf_thresh=None):
        """單獨對圖片跑 YOLO，並回傳指定類別中信心度最高的單一 BBox。"""
        xyxy, sc, cl, ids = self.yolo_det_all(self.detector, img_bgr, imgsz=self.det_imgsz,
                                         conf=self.det_conf if conf_thresh is None else conf_thresh)
        if len(xyxy)==0: return None, None
        mask = (cl==cls_id) if cls_id is not None else np.ones(len(cl), dtype=bool)
        idx = np.where(mask)[0]
        if idx.size==0: return None, None
        j = idx[np.argmax(sc[idx])]
        return xyxy[j], sc[j]

    def project_3d_bbox_xyxy(self, K, center_pose, bbox_minmax, img_shape):
        """將 3D Bounding Box 的 8 個頂點依照給定的 6D 姿態 (Pose) 與相機內參 (K) 投影到 2D 畫面，求取其 2D 外包矩形。"""
        H, W = img_shape[:2]
        mn, mx = bbox_minmax
        xs = [mn[0], mx[0]]
        ys = [mn[1], mx[1]]
        zs = [mn[2], mx[2]]
        corners = np.array([[x,y,z,1.0] for x in xs for y in ys for z in zs], dtype=np.float64)  # (8,4)
        Pc = (center_pose @ corners.T).T
        Z = Pc[:,2]
        valid = Z > 1e-6
        if not np.any(valid): return None
        X = Pc[valid,0] / Z[valid]
        Y = Pc[valid,1] / Z[valid]
        u = K[0,0]*X + K[0,2]
        v = K[1,1]*Y + K[1,2]
        return self.clip_xyxy(np.array([u.min(), v.min(), u.max(), v.max()], dtype=np.float32), W, H)

    def _bbox_depth_distance_m(self, bbox_xyxy, depth_m, sample_ratio=0.35, min_valid=30, use="median", z_min=0.05, z_max=10.0):
        """計算給定 BBox 區域內 (扣除邊緣的中心區域) 深度值的代表數 (中位數或最小值)。用於判斷物體遠近。"""
        if depth_m is None: return None
        x1, y1, x2, y2 = bbox_xyxy
        H, W = depth_m.shape[:2]
        x1, x2 = int(np.clip(x1, 0, W - 1)), int(np.clip(x2, 0, W - 1))
        y1, y2 = int(np.clip(y1, 0, H - 1)), int(np.clip(y2, 0, H - 1))
        if x2 <= x1 or y2 <= y1: return None

        bw, bh = x2 - x1, y2 - y1
        cx1 = int(x1 + (1 - sample_ratio) * 0.5 * bw)
        cx2 = int(x2 - (1 - sample_ratio) * 0.5 * bw)
        cy1 = int(y1 + (1 - sample_ratio) * 0.5 * bh)
        cy2 = int(y2 - (1 - sample_ratio) * 0.5 * bh)

        roi = depth_m[cy1:cy2, cx1:cx2]
        if roi.size == 0: return None
        vals = roi[(roi > z_min) & (roi < z_max)]
        if vals.size < min_valid: return None
        return float(np.min(vals)) if use == "min" else float(np.median(vals))

    def select_yolo_bbox(self, xyxy, scores, classes, img_shape, prefer_cls=None, select_mode="score", conf_th=0.0):
        """
        根據不同的選擇模式從多個 YOLO 檢測框中挑選一個最合適的：
        - score: 挑選信心度最高者。
        - middle: 挑選中心點最靠近畫面正中央者。
        - nearest_depth: 挑選深度最近者。
        """
        if xyxy is None or len(xyxy) == 0: return None, None
        H, W = img_shape[:2]
        idx = np.arange(len(xyxy))
        if prefer_cls is not None:
            idx = idx[classes == prefer_cls]
        if idx.size == 0: return None, None

        xyxy_f = xyxy[idx]
        scores_f = scores[idx]

        if conf_th is not None and conf_th > 0:
            keep = scores_f >= float(conf_th)
            if not np.any(keep): return None, None
            xyxy_f = xyxy_f[keep]
            scores_f = scores_f[keep]

        if select_mode == "score":
            j = int(np.argmax(scores_f))
            return xyxy_f[j], float(scores_f[j])
        elif select_mode == "middle":
            cx_img, cy_img = W * 0.5, H * 0.5
            best_d, best_j = 1e18, -1
            for i, bb in enumerate(xyxy_f):
                cx, cy = 0.5 * (bb[0] + bb[2]), 0.5 * (bb[1] + bb[3])
                d = (cx - cx_img) ** 2 + (cy - cy_img) ** 2
                if d < best_d:
                    best_d, best_j = d, i
            return (xyxy_f[best_j], float(scores_f[best_j])) if best_j >= 0 else (None, None)
        elif select_mode == "nearest_depth":
            depth_m = getattr(self, "depth_m", None)
            if depth_m is None:
                j = int(np.argmax(scores_f))
                return xyxy_f[j], float(scores_f[j])
            best_dist, best_j = 1e18, -1
            z_max = float(self.max_depth_mm) * 0.001 if hasattr(self, "max_depth_mm") else 10.0
            for i, bb in enumerate(xyxy_f):
                dist = self._bbox_depth_distance_m(bb, depth_m, z_max=z_max)
                if dist is not None and dist < best_dist:
                    best_dist, best_j = dist, i
            if best_j >= 0:
                return xyxy_f[best_j], float(scores_f[best_j])
            j = int(np.argmax(scores_f))
            return xyxy_f[j], float(scores_f[j])
        else:
            j = int(np.argmax(scores_f))
            return xyxy_f[j], float(scores_f[j])

    # =========================
    # IoU / 遮蔽率（暖機後）
    # =========================
    def iou_vs_projection_for_class_from_dets(self, color_bgr, K, center_pose, bbox_minmax, prefer_cls, xyxy_all, sc_all, cl_all, tag="bunch"):
        """計算 FoundationPose 當前預估姿態投影出的 2D Box 與該幀 YOLO 偵測到的 2D Box 之間的 IoU (交集比聯集)。"""
        if center_pose is None or K is None: return None, None
        H, W = color_bgr.shape[:2]
        est_xyxy = self.project_3d_bbox_xyxy(K, center_pose, bbox_minmax, img_shape=color_bgr.shape)
        if est_xyxy is None: return None, None
        if xyxy_all is None or len(xyxy_all) == 0: return 0.0, None

        use_mask = np.ones(len(xyxy_all), dtype=bool)
        if prefer_cls is not None and (cl_all == prefer_cls).any():
            use_mask = (cl_all == prefer_cls)

        xyxy_use = xyxy_all[use_mask]
        if len(xyxy_use) == 0: return 0.0, None

        ious = []
        for bb in xyxy_use:
            bb_c = self.clip_xyxy(bb, W, H)
            ious.append(self.iou_xyxy(bb_c, est_xyxy))
        ious = np.asarray(ious, dtype=float)
        if ious.size == 0: return 0.0, None

        j = int(np.argmax(ious))
        return float(ious[j]), xyxy_use[j]

    def maybe_regrab_roi_by_iou_from_dets(self, mode, center_pose, xyxy_all, sc_all, cl_all):
        """如果追蹤過程中，投影框與 YOLO 的檢測框的 IoU 過低並持續若干幀，則利用 YOLO 框重新初始化 FoundationPose。"""
        if center_pose is None or self.K is None: return False
        if (self.frame_count % max(1, self.iou_stride)) != 0: return False

        if mode == "bunch":
            prefer_cls = self.cls_bunch
            bbox_mm    = self.bbox_bunch
            bad_count_attr = "iou_bad_count_bunch"
            last_upd_attr  = "last_iou_update_bunch"
        else:
            return False # Stem 不再使用這個機制

        iou_val, best_xyxy = self.iou_vs_projection_for_class_from_dets(
            self.color, self.K, center_pose, bbox_mm,
            prefer_cls=prefer_cls,
            xyxy_all=xyxy_all, sc_all=sc_all, cl_all=cl_all,
            tag=mode
        )
        self.iou_val = iou_val

        if iou_val is None: return False

        setattr(self, last_upd_attr, self.frame_count)
        if iou_val < float(self.iou_thresh):
            setattr(self, bad_count_attr, getattr(self, bad_count_attr) + 1)
        else:
            setattr(self, bad_count_attr, 0)

        if getattr(self, bad_count_attr) < int(self.iou_patience): return False

        # 觸發 reinit
        setattr(self, bad_count_attr, 0)

        if best_xyxy is not None:
            m = self.rect_to_mask(self.depth_m, self.clip_xyxy(best_xyxy, *self.rgb_size), expand=self.roi_expand)
            new_pose = self.est_bunch.register(
                K=self.K, rgb=self.color, depth=self.depth_m,
                ob_mask=m, iteration=self.est_refine_iter
            )
            if new_pose is not None:
                self.pose_bunch = new_pose
                if getattr(self, "det_select_mode_current", self.det_select_mode) == "middle":
                    self._force_bunch_detect = True
                    self._hi_cnt = 0
                self.seg_warmup_left_bunch = int(self.occ_seg_warmup_n)

            vis = self.color.copy()
            cv2.putText(vis, f"Re-init ROI ({mode}, low IoU)", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
            self.pump_windows(vis if self.show_rgb_win else None,
                            self.depth_vis if (self.show_depth_win and hasattr(self, "depth_vis")) else None)
            return True

        self.pose_bunch = None
        vis = self.color.copy()
        cv2.putText(vis, f"Re-init needed ({mode}, low IoU, no det)", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
        self.pump_windows(vis if self.show_rgb_win else None,
                        self.depth_vis if (self.show_depth_win and hasattr(self, "depth_vis")) else None)
        return True

    def iou_xyxy(self, a, b):
        """計算兩個 2D Bounding Box 的 IoU。"""
        if a is None or b is None: return 0.0
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        aw, ah = max(0.0, ax2 - ax1), max(0.0, ay2 - ay1)
        bw, bh = max(0.0, bx2 - bx1), max(0.0, by2 - by1)
        union = aw*ah + bw*bh - inter
        return float(inter / union) if union > 0 else 0.0

    def _bunch_skip_occlusion(self) -> bool:
        """檢查目前的檢測模式是否為 middle，若是則跳過遮蔽率運算"""
        mode = getattr(self, "det_select_mode_current", self.det_select_mode)
        return (str(mode).strip().lower() == "middle")

    def pose_2d_box_xyxy(self, which:str):
        """利用目前的 6D 姿態，投影出 2D 的提示框，用來輔助 YOLO-Seg 切割果串"""
        if which.lower() == "bunch":
            if self.pose_bunch is None or self.K is None: return None
            center_pose = self.pose_bunch @ np.linalg.inv(self.to_origin_bunch)
            bbox = self.bbox_bunch
            return self.project_3d_bbox_xyxy(self.K, center_pose, bbox, self.color.shape)
        return None

    def compute_occ_and_iou(self, which: str, xyxy_all=None, sc_all=None, cl_all=None):
        """
        暖機期間：計算實體遮罩 (YOLO-Seg) 與 3D 模型投影面積的交集，推算真實遮蔽率 (Occ)。
        暖機後：只計算快速的 2D 框 IoU，不浪費效能算遮蔽率。
        """
        occ = 1.0
        if self.color is None or self.K is None or which.lower() != "bunch":
            return 1.0, 0.0, False

        if self.pose_bunch is None:
            return 1.0, 0.0, False

        center_pose = self.pose_bunch @ np.linalg.inv(self.to_origin_bunch)

        # 1. 永遠計算 IoU
        iou_val, best_xyxy = self.iou_vs_projection_for_class_from_dets(
            self.color, self.K, center_pose, self.bbox_bunch,
            prefer_cls=self.cls_bunch,
            xyxy_all=xyxy_all, sc_all=sc_all, cl_all=cl_all,
            tag="bunch"
        )

        skip_occlusion_check = self._force_bunch_detect

        # 2. 暖機階段 (Warmup): 執行 YOLO-Seg 分割與精確的物理遮蔽率計算
        if (not skip_occlusion_check) and (self.seg_warmup_left_bunch > 0):
            self.seg_warmup_left_bunch -= 1
            if self.seg_warmup_left_bunch == 0:
                rospy.loginfo("[bunch] Seg warmup finished, switch to IoU-only mode (skip occ)")

            prompt_box = self.pose_2d_box_xyxy("bunch")
            if self.seg_backend == "yolo_seg":
                seg = self.seg_mask_from_yolo_seg(self.color, prompt_box, target_cls=self.cls_bunch)
            else:
                seg = self.render_bbox_mask_proxy(self.K, center_pose, self.bbox_bunch, self.color.shape)
            
            occ = self.occ_from_gt_mesh_vs_seg("bunch", seg.astype(bool))

            # Debug 存檔
            if getattr(self, "iou_log", False):
                try:
                    self.save_occ_seg_debug(
                        self.color, seg.astype(bool) if seg is not None else None,
                        self.K, center_pose, self.bbox_bunch, occ, best_xyxy, tag="bunch"
                    )
                except Exception:
                    pass

            return occ, (0.0 if iou_val is None else float(iou_val)), True

        # 3. 非暖機階段: IoU-only (不耗費 GPU 算遮蔽率，省效能)
        if iou_val is None:
            occ = None
            iou_show = 0.0
        else:
            occ = None
            iou_show = float(iou_val)

        # Debug 存檔
        if getattr(self, "iou_log", False):
            try:
                self.save_iou_debug(self.color, self.K, center_pose, self.bbox_bunch, best_xyxy, iou_val, tag="bunch")
            except Exception:
                pass

        return occ, iou_show, False
    
    # =========================
    # Debug helpers
    # =========================
    def _setup_run_debug_dir(self):
        """建立本次執行存放 Debug 圖片的專屬時間戳資料夾。"""
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
        rospy.loginfo(f"[DBG] debug_dir (this run) = {self.debug_dir}")

    def _ensure_dir(self, d=None):
        """確保目錄存在，若無則建立。"""
        if d is None: d = getattr(self, "debug_dir", None)
        if not d: return
        try: os.makedirs(d, exist_ok=True)
        except Exception: pass

    def _ensure_parent(self, path: str):
        """確保檔案的上層目錄存在，若無則建立。"""
        try:
            parent = os.path.dirname(path)
            if parent: os.makedirs(parent, exist_ok=True)
        except Exception: pass

    def _dbg_path(self, subdir: str, prefix: str):
        """產生除錯圖片的完整存檔路徑 (自動帶上當前的 frame_count)。"""
        root = os.path.join(self.debug_dir, subdir)
        self._ensure_dir(root)
        p = os.path.join(root, f"{prefix}_{self.frame_count:06d}.png")
        self._ensure_parent(p)
        return p

    def _overlay_mask(self, img_bgr, mask_bool, alpha=0.45, color=(0, 0, 255)):
        """將 Boolean Mask 疊加半透明的指定顏色到 BGR 影像上。"""
        if mask_bool is None: return img_bgr
        vis = img_bgr.copy()
        if mask_bool.dtype != np.uint8:
            m = (mask_bool.astype(np.uint8) * 255)
        else:
            m = mask_bool
        color_img = np.zeros_like(vis)
        color_img[:] = color
        color_img = cv2.bitwise_and(color_img, color_img, mask=m)
        vis = cv2.addWeighted(vis, 1.0, color_img, alpha, 0)
        return vis

    def _draw_rect(self, img, xyxy, color=(0, 255, 255), thick=2, label=None):
        """在影像上繪製矩形外框與文字標籤。"""
        if xyxy is None: return img
        x1, y1, x2, y2 = [int(t) for t in xyxy]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        if label is not None:
            cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return img

    def _draw_pose_box(self, img_bgr, K, pose_obj_in_cam, bbox_minmax, which, axis_scale=0.05):
        """根據物件 6D 姿態繪製 3D 邊界框 (Cuboid) 以及 XYZ 座標軸。"""
        if which.lower() != "bunch": return img_bgr
        try:
            to_origin = self.to_origin_bunch
            center_pose = pose_obj_in_cam @ np.linalg.inv(to_origin)
            rgb = draw_posed_3d_box(
                K, img=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                ob_in_cam=center_pose, bbox=bbox_minmax
            )
            rgb = draw_xyz_axis(rgb, ob_in_cam=pose_obj_in_cam, scale=axis_scale, K=K,
                                thickness=3, transparency=0, is_input_rgb=True)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return img_bgr

    def save_occ_seg_debug(self, color_bgr, seg_mask, K, center_pose, bbox_minmax, occ, yolo_xyxy, tag="bunch"):
        """暖機期間：儲存遮蔽率計算的除錯影像 (顯示 Seg 遮罩、YOLO框、投影框)。"""
        if not getattr(self, "iou_log", False) or tag.lower() != "bunch": return
        vis = color_bgr.copy()
        proj_xyxy = self.project_3d_bbox_xyxy(K, center_pose, bbox_minmax, color_bgr.shape)
        vis = self._overlay_mask(vis, seg_mask, alpha=0.45, color=(0, 0, 255))
        vis = self._draw_rect(vis, proj_xyxy, color=(0, 255, 0), thick=2, label="proj-2D bbox") 
        vis = self._draw_rect(vis, self.clip_xyxy(yolo_xyxy, vis.shape[1], vis.shape[0]) if yolo_xyxy is not None else None,
                              color=(0, 255, 255), thick=2, label="YOLO bbox") 

        to_origin = self.to_origin_bunch 
        vis = self._draw_pose_box(vis, K, pose_obj_in_cam=center_pose @ to_origin,
            bbox_minmax=bbox_minmax, which="bunch", axis_scale=0.05)

        cv2.putText(vis, f"[{tag}] OCC(Seg)={occ:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 220), 2, cv2.LINE_AA)

        outp = self._dbg_path("occ_seg", f"occ_seg_{tag.lower()}")
        cv2.imwrite(outp, vis)

    def save_iou_debug(self, color_bgr, K, center_pose, bbox_minmax, yolo_xyxy, iou, tag="bunch"):
        """追蹤期間：儲存 IoU 計算的除錯影像。"""
        if not getattr(self, "iou_log", False) or tag.lower() != "bunch": return
        vis = color_bgr.copy()
        proj_xyxy = self.project_3d_bbox_xyxy(K, center_pose, bbox_minmax, color_bgr.shape)
        vis = self._draw_rect(vis, proj_xyxy, color=(0, 255, 0), thick=2, label="proj-2D bbox") 
        vis = self._draw_rect(vis, self.clip_xyxy(yolo_xyxy, vis.shape[1], vis.shape[0]) if yolo_xyxy is not None else None,
                              color=(0, 255, 255), thick=2, label="YOLO bbox") 

        to_origin = self.to_origin_bunch
        vis = self._draw_pose_box(vis, K, pose_obj_in_cam=center_pose @ to_origin,
            bbox_minmax=bbox_minmax, which="bunch", axis_scale=0.05)

        occ = 1.0 - (float(iou) if iou is not None else 0.0)
        cv2.putText(vis, f"[{tag}] IoU={0.0 if iou is None else float(iou):.3f}  OCC={occ:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 220), 2, cv2.LINE_AA)

        outp = self._dbg_path(f"iou_{tag.lower()}", f"iou_{tag.lower()}")
        cv2.imwrite(outp, vis)
    
    def save_binary_mask(self, mask_bool: np.ndarray, subdir: str, fname: str):
        """將 Boolean Mask 存成黑白二值化的 PNG 圖片檔供除錯。"""
        if not getattr(self, "iou_log", False): return
        try:
            if mask_bool is None: return
            m = (mask_bool.astype(np.uint8) * 255)
            outp = self._dbg_path(subdir, fname) 
            cv2.imwrite(outp, m)
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"[DBG] save_binary_mask failed: {e}")

    def render_mesh_silhouette_mask(self, K, pose_center_in_cam, mesh, img_shape):
        """
        以高速向量化運算，將 3D Mesh 依據相機內參與 6D 姿態，
        精確投影到 2D 影像上並填滿三角形，生成物體理論上的無遮擋遮罩 (Silhouette)。
        """
        H, W = img_shape[:2]
        V = mesh.vertices.astype(np.float64)             
        Vh = np.concatenate([V, np.ones((V.shape[0],1))], axis=1)  
        Pc = (pose_center_in_cam @ Vh.T).T               

        Z = Pc[:, 2]
        valid_Z = Z > 1e-6
        Z_safe = Z.copy()
        Z_safe[~valid_Z] = 1.0

        u = K[0,0] * (Pc[:, 0] / Z_safe) + K[0,2]
        v = K[1,1] * (Pc[:, 1] / Z_safe) + K[1,2]

        uv = np.stack([u, v], axis=1)
        F = mesh.faces
        triangles = uv[F]
        valid_faces = valid_Z[F].all(axis=1)
        
        if not np.any(valid_faces):
            return np.zeros((H, W), dtype=bool)

        valid_triangles = triangles[valid_faces]
        MARGIN = 10
        min_uv = valid_triangles.min(axis=1) 
        max_uv = valid_triangles.max(axis=1) 
        
        in_screen = ~(
            (min_uv[:, 0] > W + MARGIN) | 
            (max_uv[:, 0] < -MARGIN) | 
            (min_uv[:, 1] > H + MARGIN) | 
            (max_uv[:, 1] < -MARGIN)
        )
        
        final_triangles = valid_triangles[in_screen]
        mask = np.zeros((H, W), np.uint8)
        if final_triangles.shape[0] > 0:
            final_triangles = np.clip(final_triangles, -2048, W + 2048).astype(np.int32)
            cv2.fillPoly(mask, final_triangles, 255, lineType=cv2.LINE_AA)

        return (mask > 0)

    def occ_from_gt_mesh_vs_seg(self, which:str, seg_mask_bool: np.ndarray):
        """
        計算物體遮蔽率。
        公式為：1.0 - (理論上 3D 投影的無遮擋面積 與 實際 YOLO-Seg 切出的面積之交集) / 3D 理論面積。
        """
        if self.color is None or self.K is None or which.lower() != "bunch": return 1.0
        
        seg_mask_bool = seg_mask_bool.astype(bool)
        if seg_mask_bool.shape[:2] != self.color.shape[:2]:
            seg_mask_bool = cv2.resize(seg_mask_bool.astype(np.uint8), (self.color.shape[1], self.color.shape[0]),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
        if self.pose_bunch is None: return 1.0
        center_pose = self.pose_bunch @ np.linalg.inv(self.gt_to_origin_bunch)
        mesh = self.mesh_bunch
        tag  = "bunch"

        gt = self.render_mesh_silhouette_mask(self.K, center_pose, mesh, self.color.shape)
        if getattr(self, "iou_log", False):
            self.save_binary_mask(gt, "gt_mask", f"gt_{tag}")
            self.save_binary_mask(seg_mask_bool, "seg_mask", f"seg_{tag}")
        
        area = int(gt.sum())
        if area < 100: return 1.0

        inter = int(np.logical_and(gt, seg_mask_bool).sum())
        visible_ratio = inter / float(area)
        occ = float(1.0 - visible_ratio)
                        
        if getattr(self, "iou_log", False):
            try:
                os.makedirs(os.path.join(self.debug_dir, "occ_seg"), exist_ok=True)
                vis = self.color.copy()
                seg_u8 = (seg_mask_bool.astype(np.uint8)*255)
                gt_u8  = (gt.astype(np.uint8)*255)
                cnts_seg,_ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts_gt,_  = cv2.findContours(gt_u8,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts_seg, -1, (0,255,0), 2)
                cv2.drawContours(vis, cnts_gt, -1, (255,0,0), 2)
                cv2.putText(vis, f"[{tag.upper()}] OCC(GTvsSeg)={occ:.3f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                fn = os.path.join(self.debug_dir, "occ_seg", f"occ_seg_{tag}_{self.frame_count:06d}.png")
                cv2.imwrite(fn, vis)
            except Exception as e:
                rospy.logwarn_throttle(1.0, f"[DBG] save occ_seg failed: {e}")

        return occ

    def render_bbox_mask_proxy(self, K, center_pose, bbox_minmax, img_shape):
        """(Fallback機制) 用 3D bounding box 的 2D 投影外框直接填滿充當遮罩，省去複雜 Mesh 渲染。"""
        est_xyxy = self.project_3d_bbox_xyxy(K, center_pose, bbox_minmax, img_shape)
        if est_xyxy is None: return np.zeros(img_shape[:2], dtype=bool)
        return self.rect_to_mask(np.zeros(img_shape[:2], np.uint8), est_xyxy, expand=0.0)

    def seg_mask_from_yolo_seg(self, bgr, xyxy, target_cls=None):
        """
        利用 YOLO11-Seg 模型產生畫素級的二值化實例遮罩 (Instance Mask)。
        這能濾除 Bounding Box 中不屬於目標物體的背景部分。
        """
        H, W = bgr.shape[:2]
        if xyxy is None: return np.zeros((H, W), dtype=bool)
        x1, y1, x2, y2 = xyxy.astype(np.int32)
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        cx1 = max(0, int(x1 - w * self.roi_expand)); cy1 = max(0, int(y1 - h * self.roi_expand))
        cx2 = min(W - 1, int(x2 + w * self.roi_expand)); cy2 = min(H - 1, int(y2 + h * self.roi_expand))

        if self.seg_detector is None: return self.rect_to_mask(bgr, xyxy, expand=self.roi_expand)

        try:
            pred_kwargs = {"source": bgr, "imgsz": self.seg_imgsz, "conf": self.det_conf, "device": self.det_device, "verbose": False}
            if target_cls is not None: pred_kwargs["classes"] = [int(target_cls)]
            r = self.seg_detector.predict(**pred_kwargs)[0]
            if r.masks is None or len(r.masks.xy) == 0: return self.rect_to_mask(bgr, xyxy, expand=self.roi_expand)

            best_mask, best_score = None, -1.0
            for poly in r.masks.xy:
                if len(poly) < 3: continue
                mm_bin = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(mm_bin, [np.int32(poly)], 1)
                mm_bin = mm_bin.astype(bool)
                inter = mm_bin[cy1:cy2, cx1:cx2].sum()
                score = float(inter / (mm_bin.sum() + 1e-6))
                if score > best_score:
                    best_score = score
                    best_mask = mm_bin
            if best_mask is None or best_score < 0.1: return self.rect_to_mask(bgr, xyxy, expand=self.roi_expand)
            return best_mask
        except Exception as e:
            return self.rect_to_mask(bgr, xyxy, expand=self.roi_expand)

    # =========================
    # stem YOLO-Seg
    # =========================
    def calculate_direct_root_pose(self, depth_m, mask_bool):
        """
        從深度圖中提取 Mask 區域內的有效點，利用中位數抗噪。
        """
        v, u = np.where(mask_bool > 0)
        if len(v) == 0:
            return None, None, None, "No Mask Points"
            
        # 1. 只提取有 Mask 像素的深度值
        z_raw = depth_m[v, u]
        
        # 2. 過濾無效深度 (太近、太遠、或 0)
        valid_mask = (z_raw > self.stem_depth_min) & (z_raw < self.stem_depth_max)
        if np.sum(valid_mask) < 10: 
            return None, None, None, "No valid depth in mask"
            
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = z_raw[valid_mask]

        # 3. 找最下方 (V 座標最大) 的 Top K 個點
        sorted_indices = np.argsort(v_valid)[::-1]
        top_k = min(20, len(sorted_indices))
        bottom_indices = sorted_indices[:top_k]
        
        target_u = int(np.median(u_valid[bottom_indices]))
        target_v = int(np.median(v_valid[bottom_indices]))
        target_z = float(np.median(z_valid[bottom_indices])) # 中位數天生抗噪，不需全圖高斯模糊
        
        # 4. 座標轉換
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        x_surface = (target_u - cx) * target_z / fx
        y_surface = (target_v - cy) * target_z / fy
        target_z_center = target_z + self.stem_assumed_radius
        
        center_3d = np.array([x_surface, y_surface, target_z_center])
        projected_2d = (target_u, target_v)
        
        return center_3d, [0.0, 0.0, 0.0, 1.0], projected_2d, "OK"

    def cmd_vel_callback(self, msg: Twist):
        """讀取 ROS cmd_vel Topic (車體速度)，用於預測葉莖在畫面中的位移。"""
        self.current_twist = msg

    def get_predicted_stem_pos(self, now):
        """
        運動預測
        運算過程：相機座標 -> 車體座標 -> 扣除車體位移 -> 相機座標
        相機光學座標系 (X向右, Y向下, Z向前)。
        """
        if self.last_stem_3d_pos is None or self.last_stem_time is None:
            return None
            
        dt = (now - self.last_stem_time).to_sec()
        if dt <= 0:
            return self.last_stem_3d_pos

        # 讀取車體速度 (base_link)
        vx = self.current_twist.linear.x
        vy = self.current_twist.linear.y
        wz = self.current_twist.angular.z

        # 讀取相機安裝參數 (如果沒設定，給個安全預設值)
        side_sign = -1 if getattr(self, "camera_side", "right") == "right" else 1
        offset_x = getattr(self, "camera_tag_offset_x", 0.0)

        # 舊的相機座標 (X向右, Y向下, Z向前)
        X_c, Y_c, Z_c = self.last_stem_3d_pos

        # ====================================================
        # 1. 轉換為車體座標系 (base_link)
        # 對應你的邏輯： marker_x = pz, marker_y = side_sign*px + offset
        # ====================================================
        X_b = Z_c
        Y_b = side_sign * X_c + offset_x
        # Z_b (高度) 在 2D 移動中不影響平面預測，忽略不管

        # ====================================================
        # 2. 扣除車體的相對運動 (Inverse Kinematics)
        # 車體前進/旋轉，等於目標物體相對車體後退/反向旋轉
        # ====================================================
        dx = vx * dt
        dy = vy * dt
        dtheta = wz * dt

        # 先平移
        X_b_shifted = X_b - dx
        Y_b_shifted = Y_b - dy

        # 再旋轉 (因為座標系轉了 dtheta，等於物體相對於座標系轉了 -dtheta)
        cos_t = np.cos(-dtheta)
        sin_t = np.sin(-dtheta)

        X_b_new = X_b_shifted * cos_t - Y_b_shifted * sin_t
        Y_b_new = X_b_shifted * sin_t + Y_b_shifted * cos_t

        # ====================================================
        # 3. 轉回相機座標系 (Camera Optical Frame)
        # 反推：X_b_new = Z_c_new, Y_b_new = side_sign*X_c_new + offset
        # ====================================================
        Z_c_new = X_b_new
        X_c_new = (Y_b_new - offset_x) * side_sign  # 乘以 side_sign (1 或 -1) 等同除以 side_sign
        Y_c_new = Y_c  # 相機 Y 軸 (對應真實世界的高低) 假設在移動中不變

        return np.array([X_c_new, Y_c_new, Z_c_new])

    def pick_nearest_to_2d_point(self, xyxy, sc, cl, ids, target_2d, cls_id):
        """尋找畫面上與目標 2D 預測座標點距離最近的 YOLO Bounding Box。不設像素上限，交由後續 3D 深度防呆處理。"""
        if xyxy is None or len(xyxy) == 0 or target_2d is None:
            return None, None, -1
        idx = np.where(cl == int(cls_id))[0]
        if idx.size == 0:
            return None, None, -1
        tx, ty = target_2d
        best_dist, bid = float('inf'), -1
        
        for i in idx:
            cx = 0.5 * (xyxy[i][0] + xyxy[i][2])
            cy = 0.5 * (xyxy[i][1] + xyxy[i][3])
            d_sq = (cx - tx)**2 + (cy - ty)**2
            if d_sq < best_dist:
                best_dist = d_sq
                bid = int(i)

        if bid < 0:
            return None, None, -1
        
        return xyxy[bid], float(sc[bid]), int(ids[bid])

    # =============== GUI ===============
    def _open_window(self, name, pos_xy, init_size, is_rgb=True):
        """開啟 OpenCV 視窗。"""
        w, h = init_size
        try: cv2.destroyWindow(name)
        except Exception: pass
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, int(w), int(h))
        cv2.moveWindow(name, int(pos_xy[0]), int(pos_xy[1]))
        if is_rgb:
            self._rgb_win_created = True; self._rgb_win_sized = False
        else:
            self._depth_win_created = True; self._depth_win_sized = False

    def window_create(self):
        """初次啟動時建立設定的 RGB 和 Depth 視窗。"""
        if self.show_rgb_win and not self._rgb_win_created:
            self._open_window(self.rgb_win_name, self.rgb_win_xy, self._rgb_initial_size, is_rgb=True)
        if self.show_depth_win and not self._depth_win_created:
            self._open_window(self.depth_win_name, self.depth_win_xy, self._depth_initial_size, is_rgb=False)

    def pump_windows(self, rgb_frame=None, depth_frame=None):
        """將影像寫入 OpenCV 視窗並處理鍵盤輸入事件 ('q' 退出)。"""
        if self.show_rgb_win and not self._rgb_win_created:
            self._open_window(self.rgb_win_name, self.rgb_win_xy, self._rgb_initial_size, is_rgb=True)
        if self.show_depth_win and not self._depth_win_created:
            self._open_window(self.depth_win_name, self.depth_win_xy, self._depth_initial_size, is_rgb=False)
        if rgb_frame is None and self.show_rgb_win:
            rgb_frame = np.zeros((480,640,3), np.uint8)
            cv2.putText(rgb_frame, "Waiting for detection / click init...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
        if depth_frame is None and self.show_depth_win:
            depth_frame = np.zeros((480,640,3), np.uint8)
        if self.show_rgb_win and rgb_frame is not None:
            cv2.imshow(self.rgb_win_name, rgb_frame)
            if not self._rgb_win_sized:
                w,h = self._rgb_initial_size
                cv2.resizeWindow(self.rgb_win_name, int(w), int(h))
                cv2.moveWindow(self.rgb_win_name, int(self.rgb_win_xy[0]), int(self.rgb_win_xy[1]))
                self._rgb_win_sized = True
        if self.show_depth_win and depth_frame is not None:
            cv2.imshow(self.depth_win_name, depth_frame)
            if not self._depth_win_sized:
                w,h = self._depth_initial_size
                cv2.resizeWindow(self.depth_win_name, int(w), int(h))
                cv2.moveWindow(self.depth_win_name, int(self.depth_win_xy[0]), int(self.depth_win_xy[1]))
                self._depth_win_sized = True
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rospy.signal_shutdown("user quit")
        elif key == ord('d'):
            self.show_depth_win = not self.show_depth_win
        elif key == ord('r'):
            self.show_rgb_win = not self.show_rgb_win

    def draw_conf_bar(self, img, value, label="confidence", origin=(10, 30), size=(220, 18), max_val=1.0):
        """在畫面上繪製帶有進度條視覺效果的信心度 (Confidence / IoU) 。"""
        val = float(value.get(label, 0.0))
        mv = max(1e-6, float(max_val))
        v = max(0.0, min(val / mv, 1.0))

        x, y = origin
        w, h = size
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), thickness=-1)
        cv2.rectangle(img, (x, y), (x + int(w * v), y + h), (60, 180, 75), thickness=-1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (220, 220, 220), thickness=1)
        cv2.putText(img, f"{label}: {val:.3f}", (x, y - 6),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
    # =========================
    # 方向/尺寸 後處理（僅供 Bunch）
    # =========================
    def _orientation_ok(self, center_pose_cam: np.ndarray, origin_pose_cam: np.ndarray,
                        expect_orientation: str, tol_px: float):
        """檢查 FoundationPose 算出的果串上下朝向是否符合預期。"""
        if center_pose_cam is None or origin_pose_cam is None or self.K is None:
            return True, 0.0
        Xc, Yc, Zc = map(float, center_pose_cam[:3, 3])
        Xo, Yo, Zo = map(float, origin_pose_cam[:3, 3])
        if Zc <= 1e-6 or Zo <= 1e-6:
            return True, 0.0
        fy, cy = float(self.K[1,1]), float(self.K[1,2])
        vc = fy*(Yc/Zc) + cy
        vo = fy*(Yo/Zo) + cy
        dv = float(vo - vc)    
        if dv < -tol_px:
            measured = "upright"
        elif dv > tol_px:
            measured = "inverted"
        else:
            measured = "neutral"
        expect = (expect_orientation or "upright").strip().lower()
        ok = True if measured == "neutral" else (measured == expect)
        return ok, dv

    def _size_ok(self, center_pose_cam: np.ndarray, bbox, cfg: dict):
        """檢查推估出來的 3D Box 尺寸是否合理，防止算出的物體過大或過小(被背景干擾)。"""
        if center_pose_cam is None:
            return True, 0.0, "none"
        size_mode = (cfg.get("size_mode") or "bbox_mm").lower()
        if size_mode == "bbox_mm":
            expected_w = float(cfg.get("expect_bbox_w_mm", 9999))
            expected_h = float(cfg.get("expect_bbox_h_mm", 9999))
            ratio_min  = float(cfg.get("size_ratio_min", 0.8))
            bbox_min, bbox_max = bbox
            corners = np.array([[x, y, z, 1.0] for x in [bbox_min[0], bbox_max[0]]
                                            for y in [bbox_min[1], bbox_max[1]]
                                            for z in [bbox_min[2], bbox_max[2]]], dtype=np.float64)
            world_pts = (center_pose_cam @ corners.T).T[:, :3] * 1000.0
            diff = world_pts.max(axis=0) - world_pts.min(axis=0)
            actual_w = float(np.linalg.norm(diff[[0, 2]]))
            actual_h = float(abs(diff[1]))
            ok_w = (actual_w >= ratio_min * expected_w)
            ok_h = (actual_h >= ratio_min * expected_h)
            ok = ok_w and ok_h
            return ok, (actual_w, actual_h), "bbox_mm>=min(w,h)"
        if size_mode == "depth":
            z = float(center_pose_cam[2, 3])
            expect_z = float(cfg.get("expect_depth_m", 1.2))
            tol_z    = float(cfg.get("depth_tol_m", 0.25))
            ok = (abs(z - expect_z) <= tol_z)
            return ok, z, "depth_m"
        return True, 0.0, "none"

    def postprocess_and_maybe_reinit(self, pose_obj_in_cam: np.ndarray, which="bunch"):
        """統合所有後處理檢查：若方向顛倒或尺寸不對，則判斷是否需要重新初始化追蹤器。"""
        if not self.pp_enable:
            return True, False
            
        # 葉莖現在不再走任何姿態與尺寸檢查（純粹看 YOLO/Depth)
        if which == "stem":
            return True, False
            
        to_origin = self.to_origin_bunch
        bbox      = self.bbox_bunch
        expect_ori = self.bunch_expect_orientation
        size_cfg   = self.cfg_bunch

        center_pose = pose_obj_in_cam @ np.linalg.inv(to_origin)

        orient_ok, dv_px = self._orientation_ok(
            center_pose, pose_obj_in_cam,
            expect_orientation=expect_ori,
            tol_px=self.pp_orient_center_tol_px
        )
        size_ok, metric_val, metric_name = self._size_ok(center_pose, bbox, size_cfg)

        if isinstance(metric_val, (tuple, list, np.ndarray)) and len(metric_val) >= 2:
            metric_str = f"({float(metric_val[0]):.1f},{float(metric_val[1]):.1f})"
        else:
            metric_str = f"{float(metric_val):.3f}"
        if self.iou_log:
            rospy.loginfo_throttle(1.0, f"[POST-{which}] orient_ok={orient_ok} dv_px={dv_px:.1f} | size_ok={size_ok} {metric_name}={metric_str}")

        if orient_ok and size_ok:
            self._post_pending = False
            self._post_fail_time = None
            return True, False

        if not self._post_pending:
            self._post_pending = True
            self._post_fail_time = rospy.Time.now()
            rospy.logwarn_throttle(1.0, f"[POST-{which}] Fail. Debounce {self.pp_retry_delay_sec:.1f}s before reinit.")
        return False, True

    # =========================
    # ROS 工具 / Callbacks
    # =========================
    def mat4_to_translation_quat(self, T: np.ndarray):
        """將 4x4 旋轉平移矩陣轉換為 ROS 格式的三維平移與四元數。"""
        t = (float(T[0,3]), float(T[1,3]), float(T[2,3]))
        qx,qy,qz,qw = tf.transformations.quaternion_from_matrix(T)
        return t, (float(qx),float(qy),float(qz),float(qw))

    def broadcast_transform_and_pose(self, T: np.ndarray, which: str, parent: str):
        """透過 ROS TF Broadcast 發布當前的追蹤物件 (果串或葉莖) 座標系。"""
        child = self.bunch_name if which.lower() == "bunch" else self.stem_name
        t = (float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
        qx, qy, qz, qw = tf.transformations.quaternion_from_matrix(T)
        self.tf_broadcaster.sendTransform(t, (qx, qy, qz, qw), rospy.Time.now(), child, parent)

    def imageCallback(self, msg: Image):
        """處理相機的 RGB 影像輸入。"""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("rgb decode failed: %r", e); return
        self.color = img; self.got_rgb = True
        self.rgb_size = (img.shape[1], img.shape[0])

    def infoCallback(self, msg: CameraInfo):
        """處理相機內參 (CameraIntrinsics) 輸入，建立 3x3 相機矩陣。"""
        self.info_msg = msg
        if self.K is None:
            self.K = np.array([[msg.K[0], 0, msg.K[2]],
                               [0, msg.K[4], msg.K[5]],
                               [0, 0, 1]], dtype=np.float64)

    def depthCallback(self, msg: Image):
        """處理相機深度圖，將其轉換為公尺，同時產生顏色標示圖供顯示。"""
        self.depth_encoding = msg.encoding
        try:
            d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.float32)
        except Exception as e:
            rospy.logwarn("depth decode failed: %r", e); return
        if msg.encoding.upper() in ("16UC1","TYPE_16UC1"):
            depth_m = d * 0.001
        else:
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
            mx = float(d.max())
            depth_m = d if mx <= 10.0 else d * 0.001
        self.depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        maxd_mm = max(1.0, float(self.max_depth_mm))
        depth_mm_for_vis = np.clip(self.depth_m * 1000.0, 0.0, maxd_mm)
        depth_8u = (depth_mm_for_vis * (255.0 / maxd_mm)).astype(np.uint8)
        if self.invert_colormap: depth_8u = 255 - depth_8u
        self.depth_vis = cv2.applyColorMap(depth_8u, self.colormap_id)
        self.got_depth = True
        self.depth_size = (self.depth_vis.shape[1], self.depth_vis.shape[0])

    def detectionCallback(self, msg: Detection):
        """接收外部發送的開始或停止檢測訊號 (允許/暫停)。"""
        mode = (getattr(msg, "det_select_mode", "") or "").strip().lower()
        if mode not in ("score", "middle", "nearest_depth"):
            rospy.logwarn_throttle(1.0, f"[DETECTION] invalid det_select_mode={mode}, fallback score")
            mode = "score"
        self.det_select_mode_current = mode
        self.ready_received.detection_allowed = bool(getattr(msg, "detection_allowed", False))
    
    def harvestDoneCallback(self, msg: Bool):
        """接收機械手臂夾取完成訊號，觸發系統重置，準備下一輪抓取果串。"""
        if not bool(msg.data):
            return
        rospy.logwarn("[HARVEST_DONE] received True -> hard reset to bunch")
        self._reset_all_to_bunch()
        self._publish_zero_current(used_seg=False)

    def _tag(self, which: str) -> str:
        """產生當前模式字串標籤。"""
        return "bunch" if which.lower() == "bunch" else "stem"

    def _state(self, which: str, used_seg: bool = False) -> str:
        """建立自定義的系統當前狀態字串。"""
        part = self._tag(which)
        allowed = bool(getattr(self.ready_received, "detection_allowed", False))

        # 1. 系統暫停
        if self.yolo_start_mode == "wait" and (not allowed):
            return f"{part}:PAUSED"

        # 2. 相機或影像資料尚未準備好
        if not (self.got_depth and self.got_rgb and self.K is not None and hasattr(self, "depth_m")):
            return f"{part}:INITIALIZING"

        # 3. 根據目標姿態判定狀態
        if which.lower() == "bunch":
            if self.pose_bunch is None:
                return f"{part}:YOLO"
            if used_seg and self.seg_warmup_left_bunch > 0:
                return f"{part}:YOLOSEG"
            return f"{part}:STABLE"
        else:
            if self.pose_stem is None:
                return f"{part}:YOLOSEG"
            return f"{part}:STABLE"

    def confidence_publish(self, which: str, iou: float, detection: bool, used_seg: bool = False):
        """封裝當前檢測的各種指標與位置資訊，並以自訂的 Confidence Message 發布。"""
        conf_msg = Confidence()
        conf_msg.stamp = rospy.Time.now()
        conf_msg.state = self._state(which, used_seg=used_seg)
        conf_msg.frame_id = self.bunch_name if which.lower() == "bunch" else self.stem_name
        conf_msg.object_IoU = float(iou)
        conf_msg.object_detection = bool(detection)

        T = self.pose_bunch if which.lower() == "bunch" else self.pose_stem
        if T is not None:
            t = (float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
            qx, qy, qz, qw = tf.transformations.quaternion_from_matrix(T)
        else:
            t = (0.0, 0.0, 0.0)
            qx, qy, qz, qw = (0.0, 0.0, 0.0, 1.0)

        conf_msg.position = Point(x=t[0], y=t[1], z=t[2])
        conf_msg.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        self.conf_pub.publish(conf_msg)

    def _publish_zero_current(self, used_seg: bool = False):
        """發布空的 Confidence 訊息 (當作沒有追蹤到任何物體時的預設動作)。"""
        which = "stem" if self.mode == "stem" else "bunch"
        self.confidence_publish(which, 0.0, False, used_seg=used_seg)

    def _reset_pipeline_state(self):
        """軟重置：清除姿態與計數器，讓系統重新使用 YOLO 尋找果串。"""
        self.mode = "bunch"
        self.pose_bunch = None
        self.pose_stem  = None
        self._hi_cnt = 0
        self._stem_lock = False
        self.target_stem_id = -1

        self.iou_bad_count = 0
        self.iou_val = None
        self.iou_bad_count_bunch = 0

        self.seg_warmup_left_bunch = 0
        self.last_stem_3d_pos = None
        self.last_stem_time = None

        self._force_bunch_detect = False
        self.last_bunch_3d_pos = None

        # 重置 Cutie 多目標
        self.bunch_cutie_state = "CRUISING"
        self.stem_cutie_state = "CRUISING"
        if self.cutie_processor is not None:
            self.cutie_processor.clear_memory()

    def _reset_all_to_bunch(self):
        """硬重置：除了清除姿態，也將延遲過濾和畫面文字一併清空。"""
        self._reset_pipeline_state()
        self._yolo_delay_left_bunch = 0
        self._yolo_delay_bbox_bunch = None
        self._post_pending = False
        self._post_fail_time = None
        self._last_yolo_text = ""
        self._registering_until = 0
        self._reinit_until = 0
    
    def _handle_detection_paused(self):
        """處理當使用者從外部發送暫停訊號時的邏輯：畫面印出提示，發布 0 的 Confidence。"""
        if self.color is not None:
            vis_rgb = self.color.copy()
            cv2.putText(vis_rgb, "PAUSED (detection_allowed=FALSE)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
        else:
            vis_rgb = None

        self._publish_zero_current(used_seg=False)
        self.pump_windows(
            vis_rgb if self.show_rgb_win else None,
            self.depth_vis if (self.show_depth_win and hasattr(self, "depth_vis")) else None
        )

    def pick_nearest_from_dets(self, xyxy, sc, cl, ids, target_xyxy, cls_id):
        """從 YOLO 已檢測的所有框之中，尋找與目標 BBox 中心點距離最近的特定類別 BBox。"""
        if xyxy is None or len(xyxy) == 0 or target_xyxy is None:
            return None, None, -1
        idx = np.where(cl == int(cls_id))[0]
        if idx.size == 0:
            return None, None, -1
        tx = 0.5 * (target_xyxy[0] + target_xyxy[2])
        ty = 0.5 * (target_xyxy[1] + target_xyxy[3])
        best, bid = 1e18, -1
        for i in idx:
            cx = 0.5 * (xyxy[i][0] + xyxy[i][2])
            cy = 0.5 * (xyxy[i][1] + xyxy[i][3])
            d = (cx - tx) ** 2 + (cy - ty) ** 2
            if d < best:
                best, bid = d, int(i)
        if bid < 0:
            return None, None, -1
        return xyxy[bid], float(sc[bid]), int(ids[bid])

    def _yolo_delay_update(self, which: str, xyxy_now):
        """
        確認 YOLO 是否在連續 N 幀都穩定偵測到目標，
        以此避免追蹤器因為單一幀的殘影雜訊而錯誤初始化。
        """
        self.yolo_delay_frames = 5
        n = max(0, int(getattr(self, "yolo_delay_frames", 0)))
        
        # 目前僅有 Bunch 需要，Stem 已移除這段需求
        if which != "bunch": return True, xyxy_now 

        left_attr = "_yolo_delay_left_bunch"
        bbox_attr = "_yolo_delay_bbox_bunch"

        left = int(getattr(self, left_attr, 0))
        if n <= 0: return (xyxy_now is not None), xyxy_now

        if xyxy_now is None:
            setattr(self, left_attr, 0)
            setattr(self, bbox_attr, None)
            return False, None

        if left <= 0:
            setattr(self, left_attr, n)
            setattr(self, bbox_attr, xyxy_now.copy())
            return False, None

        setattr(self, bbox_attr, xyxy_now.copy())
        left -= 1
        setattr(self, left_attr, left)

        if left <= 0:
            bbox_use = getattr(self, bbox_attr, xyxy_now)
            return True, bbox_use
        return False, None

    # =========================
    # 主循環
    # =========================
    def spin(self):
        """
        ROS 節點的主循環函數：
        協調各模組的運行順序，等候影像資料 -> 檢查系統是否被暫停 -> 執行 YOLO -> 
        套用 FoundationPose 姿態追蹤 (bunch) 或直接投影估算 (stem) -> 
        發布 TF / 執行 GUI 可視化。
        """
        self.frame_count = 0
        used_seg = False

        while not rospy.is_shutdown():
            iou_for_bar = 0.0
            used_seg = False
            
            if not (self.got_depth and self.got_rgb and self.K is not None and hasattr(self, "depth_m")):
                self.pump_windows(self.color if self.got_rgb else None, self.depth_vis if self.got_depth else None)
                continue

            allowed = bool(getattr(self.ready_received, "detection_allowed", False))
            if self.yolo_start_mode == "wait" and (not allowed):
                self._handle_detection_paused()
                continue

            self.frame_count += 1
            now = rospy.Time.now()
            
            img_rgb = cv2.cvtColor(self.color, cv2.COLOR_BGR2RGB)
            img_tensor = F.to_tensor(img_rgb).unsqueeze(0).cuda().float()
            vis_bgr = self.color.copy()

            # --- 決定是否需要跑 YOLO ---
            run_yolo = False
            if self.bunch_cutie_state == "CRUISING":
                run_yolo = True
            elif self.mode == "stem" and self.stem_cutie_state == "CRUISING":
                run_yolo = True

            xyxy_all, sc_all, cl_all, ids_all = None, None, None, None
            if run_yolo:
                xyxy_all, sc_all, cl_all, ids_all = self.yolo_det_all(self.detector, self.color, imgsz=self.det_imgsz, conf=self.det_conf)

            # 準備提供給 Cutie 初始化的陣列 (0=背景, 1=果串, 2=葉莖)
            cutie_init_mask = np.zeros((self.rgb_size[1], self.rgb_size[0]), dtype=np.uint8)
            cutie_init_objs = []

            # ==========================================
            # 1. BUNCH CRUISING (尋找果串)
            # ==========================================
            if self.bunch_cutie_state == "CRUISING":
                bunch_xyxy, bunch_conf = self.select_yolo_bbox(
                    xyxy_all, sc_all, cl_all, img_shape=self.color.shape,
                    prefer_cls=self.cls_bunch, select_mode=self.det_select_mode_current, conf_th=self.det_conf
                )
                ready, bb_use = self._yolo_delay_update(bunch_xyxy)
                
                if not ready:
                    if bunch_xyxy is not None:
                        cv2.putText(vis_bgr, f"YOLO detected. Delay... ({self._yolo_delay_left_bunch} frames left)",
                                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    self._publish_zero_current(used_seg=False)
                else:
                    # 取得果串 Mask 並寫入 Cutie 初始化陣列
                    used_mask = self.seg_mask_from_yolo_seg(self.color, bb_use, target_cls=self.cls_bunch)
                    if used_mask.sum() > 50:
                        rospy.loginfo("[Hand-off] YOLO -> Cutie Locked BUNCH (ID=1)!")
                        cutie_init_mask[used_mask] = 1
                        cutie_init_objs.append(1)
                        
                        # 註冊 FoundationPose
                        self.pose_bunch = self.est_bunch.register(
                            K=self.K, rgb=self.color, depth=self.depth_m, ob_mask=used_mask.astype(bool), iteration=self.est_refine_iter
                        )
                        self.bunch_cutie_state = "SERVOING"

            # ==========================================
            # 2. STEM CRUISING (尋找葉莖)
            # ==========================================
            predicted_2d = None
            if self.mode == "stem":
                predicted_3d = self.get_predicted_stem_pos(now)
                if predicted_3d is not None and predicted_3d[2] > 0:
                    u = int(self.K[0,0] * predicted_3d[0] / predicted_3d[2] + self.K[0,2])
                    v = int(self.K[1,1] * predicted_3d[1] / predicted_3d[2] + self.K[1,2])
                    predicted_2d = (u, v)
                    if self.show_rgb_win: cv2.circle(vis_bgr, (u, v), 5, (255, 0, 0), -1)

                if self.stem_cutie_state == "CRUISING" and run_yolo:
                    stem_xyxy = None
                    if self.target_stem_id != -1 and ids_all is not None:
                        idx = np.where(ids_all == self.target_stem_id)[0]
                        if len(idx) > 0: stem_xyxy = xyxy_all[idx[0]]

                    if stem_xyxy is None:
                        bunch_xyxy, _ = self.select_yolo_bbox(
                            xyxy_all, sc_all, cl_all, img_shape=self.color.shape,
                            prefer_cls=self.cls_bunch, select_mode="score", conf_th=self.det_conf
                        )
                        if bunch_xyxy is not None:
                            stem_xyxy, _, f_id = self.pick_nearest_from_dets(xyxy_all, sc_all, cl_all, ids_all, bunch_xyxy, self.cls_stem)
                            if stem_xyxy is not None: self.target_stem_id = f_id
                        elif predicted_2d is not None:
                            stem_xyxy, _, f_id = self.pick_nearest_to_2d_point(xyxy_all, sc_all, cl_all, ids_all, predicted_2d, self.cls_stem)
                            if stem_xyxy is not None: self.target_stem_id = f_id
                            elif (now - self.last_stem_time) > self.stem_lost_timeout:
                                self.last_stem_3d_pos = None; self.last_stem_time = None; self.target_stem_id = -1

                    if stem_xyxy is not None:
                        used_mask = self.seg_mask_from_yolo_seg(self.color, stem_xyxy, target_cls=self.cls_stem)
                        if used_mask.sum() > 50:
                            rospy.loginfo("[Hand-off] YOLO -> Cutie Locked STEM (ID=2)!")
                            cutie_init_mask[used_mask] = 2
                            cutie_init_objs.append(2)
                            self.stem_cutie_state = "SERVOING"

            # ==========================================
            # 3. 執行 Cutie 推論 (初始化 或 追蹤)
            # ==========================================
            pred_mask_tensor = None
            if self.cutie_processor is not None:
                if len(cutie_init_objs) > 0:
                    init_mask_tensor = torch.from_numpy(cutie_init_mask).unsqueeze(0).cuda().long()
                    with torch.cuda.amp.autocast(), torch.inference_mode():
                        output_prob = self.cutie_processor.step(img_tensor, init_mask_tensor, objects=cutie_init_objs)
                        pred_mask_tensor = self.cutie_processor.output_prob_to_mask(output_prob)
                elif self.bunch_cutie_state == "SERVOING" or self.stem_cutie_state == "SERVOING":
                    with torch.cuda.amp.autocast(), torch.inference_mode():
                        output_prob = self.cutie_processor.step(img_tensor)
                        pred_mask_tensor = self.cutie_processor.output_prob_to_mask(output_prob)

            # ==========================================
            # 4. BUNCH SERVOING (處理果串 Cutie 結果與 3D 姿態)
            # ==========================================
            iou_for_bar = 0.0
            if self.bunch_cutie_state == "SERVOING" and pred_mask_tensor is not None:
                # 擷取 ID 1 的 Mask
                bunch_mask_bool = (pred_mask_tensor.squeeze().cpu().numpy() == 1)
                
                if bunch_mask_bool.sum() < 100:
                    rospy.logwarn("[BUNCH] Cutie track lost! Resetting all.")
                    self._reset_all_to_bunch()
                else:
                    used_seg = True
                    # 1. 繼續追蹤 6D 姿態
                    self.pose_bunch = self.est_bunch.track_one(rgb=self.color, depth=self.depth_m, K=self.K, iteration=self.track_refine_iter)
                    
                    # 2. 使用 Cutie Mask 計算遮蔽率 OCC
                    occ = self.occ_from_gt_mesh_vs_seg("bunch", bunch_mask_bool)
                    
                    # 3. 檢查 FP 是否飄走 (FoundationPose 救援機制)
                    center_pose = self.pose_bunch @ np.linalg.inv(self.to_origin_bunch)
                    fp_est_xyxy = self.project_3d_bbox_xyxy(self.K, center_pose, self.bbox_bunch, self.color.shape)
                    fp_mask = self.rect_to_mask(self.depth_m, fp_est_xyxy, expand=0.0)
                    
                    iou = 0.0
                    if fp_mask is not None:
                        inter = np.logical_and(fp_mask, bunch_mask_bool).sum()
                        union = np.logical_or(fp_mask, bunch_mask_bool).sum()
                        iou = inter / union if union > 0 else 0.0
                    iou_for_bar = iou
                    
                    ok_to_publish, pending = self.postprocess_and_maybe_reinit(self.pose_bunch, "bunch")
                    
                    # 如果 FP 的投影與 Cutie Mask 不匹配 (IoU 過低)，或姿態顛倒，強制使用 Cutie 的 ROI 重新對齊 FP
                    if iou < self.iou_thresh or pending:
                        rospy.logwarn(f"[BUNCH] FP Drifted (IoU={iou:.2f}) or Pose Bad. Rescuing with Cutie ROI!")
                        self.pose_bunch = self.est_bunch.register(
                            K=self.K, rgb=self.color, depth=self.depth_m, ob_mask=bunch_mask_bool, iteration=self.est_refine_iter
                        )
                        self._post_pending = False # 獲救後取消等待
                        
                    # 4. 判斷是否準備切換 Stem 模式
                    if occ >= self.policy_occ_hi: self._hi_cnt += 1
                    else: self._hi_cnt = 0

                    if self._hi_cnt >= self.policy_hi_pat:
                        self.mode = "stem"

                    # 發布 TF 與影像
                    if self.pose_bunch is not None:
                        self.last_bunch_3d_pos = self.pose_bunch[:3, 3].copy()
                        parent_frame = self.camera_tf if self.camera_tf else "camera_color_optical_frame"
                        self.broadcast_transform_and_pose(self.pose_bunch, "bunch", parent_frame)
                        self.confidence_publish("bunch", iou_for_bar, True, used_seg=True)
                        
                        if self.show_rgb_win:
                            vis_bgr = self._overlay_mask(vis_bgr, bunch_mask_bool, alpha=0.4, color=(0, 0, 255)) # 果串紅色
                            center_pose = self.pose_bunch @ np.linalg.inv(self.to_origin_bunch)
                            vis_bgr = draw_posed_3d_box(self.K, img=cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB), ob_in_cam=center_pose, bbox=self.bbox_bunch)
                            vis_bgr = draw_xyz_axis(vis_bgr, ob_in_cam=self.pose_bunch, scale=0.05, K=self.K, thickness=3, transparency=0, is_input_rgb=True)
                            vis_bgr = cv2.cvtColor(vis_bgr, cv2.COLOR_RGB2BGR)

            # ==========================================
            # 5. STEM SERVOING (處理葉莖 Cutie 結果與 3D)
            # ==========================================
            if self.mode == "stem" and self.stem_cutie_state == "SERVOING" and pred_mask_tensor is not None:
                # 擷取 ID 2 的 Mask
                stem_mask_255 = (pred_mask_tensor.squeeze().cpu().numpy() == 2).astype(np.uint8) * 255
                
                # 腐蝕運算防多割
                used_mask = cv2.erode(stem_mask_255, self.erode_kernel, iterations=self.erode_iterations)

                if (used_mask / 255.0).sum() < 50:
                    rospy.logwarn("[STEM] Cutie track lost! Reverting to YOLO Cruising.")
                    self.stem_cutie_state = "CRUISING"
                    self.pose_stem = None
                    self.confidence_publish("stem", 0.0, False, used_seg=False)
                else:
                    center_3d, quat, projected_2d, status = self.calculate_direct_root_pose(self.depth_m, used_mask)

                    if center_3d is not None:
                        # [防呆 1] Z 軸生長約束 (消除前景葉片)
                        if self.last_bunch_3d_pos is not None and center_3d[2] < (self.last_bunch_3d_pos[2] - 0.05):
                            rospy.logwarn_throttle(0.5, "[stem] REJECTED: Foreground leaf (Z too close).")
                            center_3d = None 

                        # [防呆 2] 3D 空間跳動限制 (消除旁枝干擾)
                        if center_3d is not None and predicted_3d is not None:
                            jump_dist = float(np.linalg.norm(center_3d - predicted_3d))
                            if jump_dist > self.stem_max_jump_m:  
                                rospy.logwarn_throttle(0.5, f"[stem] REJECTED: Jumped {jump_dist:.2f}m")
                                center_3d = None 

                        # [平滑 3] EMA 濾波
                        if center_3d is not None:
                            if self.last_stem_3d_pos is not None:
                                smoothed_3d = (self.stem_ema_alpha * center_3d) + ((1.0 - self.stem_ema_alpha) * self.last_stem_3d_pos)
                            else:
                                smoothed_3d = center_3d.copy()
                                
                            self.last_stem_3d_pos = smoothed_3d.copy()
                            self.last_stem_time = now

                            T = tf.transformations.quaternion_matrix(quat)
                            T[0, 3] = smoothed_3d[0]; T[1, 3] = smoothed_3d[1]; T[2, 3] = smoothed_3d[2]
                            self.pose_stem = T

                            if self.show_rgb_win:
                                vis_bgr = self._overlay_mask(vis_bgr, used_mask, alpha=0.5, color=(0, 255, 255)) # 葉莖黃色
                                cv2.circle(vis_bgr, projected_2d, 8, (0, 0, 255), -1) 
                                cv2.putText(vis_bgr, f"Root Z:{smoothed_3d[2]:.2f}m", (projected_2d[0] + 10, projected_2d[1] - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
                            parent_frame = self.camera_tf if self.camera_tf else "camera_color_optical_frame"
                            self.broadcast_transform_and_pose(self.pose_stem, "stem", parent_frame)
                            self.confidence_publish("stem", 1.0, True, used_seg=True)
                            
                    if center_3d is None:
                        self.pose_stem = None
                        self.confidence_publish("stem", 0.0, False, used_seg=False)

            # ==========================================
            # 6. 更新 GUI
            # ==========================================
            if self.show_rgb_win:
                if self.bunch_cutie_state == "SERVOING":
                    cv2.putText(vis_bgr, "BUNCH: CUTIE LOCKED", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if self.mode == "stem" and self.stem_cutie_state == "SERVOING":
                    cv2.putText(vis_bgr, f"STEM: CUTIE LOCKED (Erode:{self.erode_iterations})", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                self.draw_conf_bar(vis_bgr, {"IoU": iou_for_bar}, label="IoU vs Mask",
                                   origin=(10, vis_bgr.shape[0] - 28), size=(220, 18), max_val=1.0)
                
            self.pump_windows(
                vis_bgr if (self.show_rgb_win and self.color is not None) else None,
                self.depth_vis if (self.show_depth_win and self.got_depth and self.depth_vis is not None) else None
            )

        cv2.destroyAllWindows()

if __name__ == "__main__":
    rospy.init_node("pipeline_tracker", anonymous=False)
    node = FoundationPosePipelineTracker()
    node.spin()
