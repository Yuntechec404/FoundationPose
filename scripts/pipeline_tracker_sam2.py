#!/home/user/anaconda3/envs/foundationpose/bin/python3
# -*- coding: utf-8 -*-

import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import rospy
import numpy as np
import cv2
import trimesh
import torch
import time
import psutil

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Transform, Point, Quaternion
from cv_bridge import CvBridge
from forklift_msg.msg import Confidence, Detection
from std_msgs.msg import Bool
from datetime import datetime

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


os.environ.setdefault("ULTRALYTICS_NO_INSTALL", "1")

from estimater import FoundationPose, draw_posed_3d_box, draw_xyz_axis, ScorePredictor, PoseRefinePredictor, dr
from ultralytics import YOLO, SAM
from ultralytics.models.sam import SAM2DynamicInteractivePredictor

# SAM
# try:
#     from segment_anything import sam_model_registry, SamPredictor
#     _SAM_AVAILABLE = True
# except Exception:
#     _SAM_AVAILABLE = False

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
        self.iou_bad_count_stem  = 0
        self.last_iou_update_bunch = -1
        self.last_iou_update_stem  = -1
        self.K = None
        self._last_yolo_text = ""
        # Unified detection gating (bunch+stem share the same topic)
        self.ready_received = Detection()
        self.ready_received.detection_allowed = False
        # current detection select mode (affects BUNCH only)
        self.det_select_mode_current = getattr(self, "det_select_mode", "score")
        self._force_bunch_detect = False   # reinit 後強制果串檢測直到被 block
        self._stem_lock = False
        self._pause_hold = False
        self._last_allowed = False
        
        # YOLO 延時 debounce 狀態
        self._yolo_delay_left_bunch = 0
        self._yolo_delay_left_stem  = 0
        self._yolo_delay_bbox_bunch = None
        self._yolo_delay_bbox_stem  = None

        # SAM 暖機幀數計數器
        self.sam_warmup_left_bunch = 0
        self.sam_warmup_left_stem = 0

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

        self.tf_broadcaster = tf.TransformBroadcaster()

        self.conf_pub = rospy.Publisher(self.bunch_name, Confidence, queue_size=1, latch=True)
        self.harvest_done_sub = rospy.Subscriber(self.bunch_name + "_harvest_done", Bool, self.harvestDoneCallback, queue_size=1)
        if self.yolo_start_mode == "wait":
            self._ready_sub = rospy.Subscriber(self.bunch_name + "_detection", Detection, self.detectionCallback, queue_size=1)

        self.window_create()

        # FoundationPose 初始化（果串 + 葉莖）
        os.makedirs(self.debug_dir, exist_ok=True)
        self.mesh_bunch = trimesh.load(self.mesh_file)
        self.mesh_stem  = trimesh.load(self.mesh_file_stem)

        self.to_origin_bunch, self.extents_bunch = trimesh.bounds.oriented_bounds(self.mesh_bunch) # np.eye(4),self.mesh_bunch.extents
        self.gt_to_origin_bunch, self.gt_extents_bunch = np.eye(4),self.mesh_bunch.extents
        self.bbox_bunch = np.stack([-self.extents_bunch/2, self.extents_bunch/2], axis=0).reshape(2, 3)

        self.to_origin_stem, self.extents_stem = trimesh.bounds.oriented_bounds(self.mesh_stem)
        self.gt_to_origin_stem, self.gt_extents_stem = np.eye(4),self.mesh_stem.extents
        self.bbox_stem = np.stack([-self.extents_stem/2, self.extents_stem/2], axis=0).reshape(2, 3)

        # Score/Refine/Raster
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()

        self.est_bunch = FoundationPose(model_pts=self.mesh_bunch.vertices,
                                        model_normals=self.mesh_bunch.vertex_normals,
                                        mesh=self.mesh_bunch, scorer=self.scorer, refiner=self.refiner,
                                        debug_dir=self.debug_dir, debug=0, glctx=self.glctx)
        self.est_stem = FoundationPose(model_pts=self.mesh_stem.vertices,
                                        model_normals=self.mesh_stem.vertex_normals,
                                        mesh=self.mesh_stem, scorer=self.scorer, refiner=self.refiner,
                                        debug_dir=self.debug_dir, debug=0, glctx=self.glctx)
        rospy.loginfo("Estimator initialization done (bunch+stem)")

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

        # SAM predictor
        # self.sam_predictor = None
        # if self.seg_backend == "sam" and _SAM_AVAILABLE:
        #     try:
        #         sam = sam_model_registry[self.sam_model](checkpoint=self.sam_ckpt)
        #         if torch.cuda.is_available():
        #             sam.to(device="cuda")
        #         self.sam_predictor = SamPredictor(sam)
        #         rospy.loginfo("SAM loaded: %s", self.sam_model)
        #     except Exception as e:
        #         rospy.logwarn("SAM init failed: %r. Fallback to bbox mask.", e)
        # else:
        #     if self.seg_backend == "sam":
        #         rospy.logwarn("segment_anything not available. Fallback to bbox mask.")

        # SAM2 Ultralytics (Video segmentation w/ memory)
        self.sam2_model = None               # kept for backward-compat (unused)
        self.sam2_predictor = None           # SAM2DynamicInteractivePredictor
        self.sam2_prev_masks = {}            # {obj_id: np.ndarray(H,W) bool}
        self.sam2_obj_ids = {"bunch": 1, "stem": 2}

        if self.seg_backend == "sam2":
            try:
                # Use the dynamic interactive predictor so we can:
                #  - Box prompt: initial spatial constraint
                #  - Point prompt: fg/bg refinement
                #  - Mask prompt: previous-frame mask as constraint (and to update memory)
                overrides = dict(
                    conf=0.01,
                    task="segment",
                    mode="predict",
                    imgsz=self.sam_imgsz,
                    model=self.sam_ckpt,
                    save=False,
                )
                self.sam2_predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=3)
                rospy.loginfo(f"[SAM2] DynamicInteractivePredictor loaded: {self.sam_ckpt}")

                # warmup
                dummy = np.zeros((self.sam_imgsz, self.sam_imgsz, 3), np.uint8)
                _ = self.sam2_predictor(
                    source=dummy,
                    bboxes=[[0, 0, self.sam_imgsz - 1, self.sam_imgsz - 1]],
                    obj_ids=[self.sam2_obj_ids["bunch"]],
                    update_memory=True,
                )
                rospy.loginfo("[SAM2] warmup done")
            except Exception as e:
                rospy.logwarn(f"[SAM2] init failed: {e}")
                self.sam2_predictor = None

        # 狀態機
        self.mode = "BUNCH"     # BUNCH / STEM
        self._hi_cnt = 0
        self._registering_until = 0
        self._reinit_until = 0
        # self._lo_cnt = 0

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

        # Files & modes
        self.mesh_file = gp("mesh_file", "")
        self.mesh_file_stem = gp("mesh_file_stem", "")
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

        # 葉莖（STEM）後處理
        self.stem_expect_orientation = gp("postproc/stem/expect_orientation", "upright").strip().lower()
        self.stem_size_mode = gp("postproc/stem/size_mode", "bbox_mm").strip().lower()
        self.stem_expect_bbox_w_mm = float(gp("postproc/stem/expect_bbox_w_mm", 115.0))
        self.stem_expect_bbox_h_mm = float(gp("postproc/stem/expect_bbox_h_mm", 80.0))
        self.stem_size_ratio_min = float(gp("postproc/stem/size_ratio_min", 0.6))
        self.stem_expect_depth_m = float(gp("postproc/stem/expect_depth_m", 1.2))
        self.stem_depth_tol_m = float(gp("postproc/stem/depth_tolerance_m", 0.25))

        # 打包尺寸設定
        self.cfg_bunch = dict(
            size_mode=self.bunch_size_mode,
            expect_bbox_w_mm=self.bunch_expect_bbox_w_mm,
            expect_bbox_h_mm=self.bunch_expect_bbox_h_mm,
            size_ratio_min=self.bunch_size_ratio_min,
            expect_depth_m=self.bunch_expect_depth_m,
            depth_tol_m=self.bunch_depth_tol_m,
        )
        self.cfg_stem = dict(
            size_mode=self.stem_size_mode,
            expect_bbox_w_mm=self.stem_expect_bbox_w_mm,
            expect_bbox_h_mm=self.stem_expect_bbox_h_mm,
            size_ratio_min=self.stem_size_ratio_min,
            expect_depth_m=self.stem_expect_depth_m,
            depth_tol_m=self.stem_depth_tol_m,
        )

        # 類別 id
        self.cls_bunch = int(gp("classes/bunch", 0))
        self.cls_stem = int(gp("classes/stem",  1))

        # 遮蔽率策略
        self.policy_occ_hi = float(gp("policy/occ_thresh_high", 0.60))
        self.policy_hi_pat = int(gp("policy/high_patience", 3))

        # 分割後端
        self.seg_backend = gp("postproc/seg_backend", "sam").strip().lower()  # sam | bbox
        self.sam_model = gp("postproc/sam_model", "vit_h").strip()
        self.sam_ckpt = gp("postproc/sam_ckpt", gp("postproc/sam_ckpt_path", "/home/user/.cache/sam_vit_h.pth")).strip()
        self.sam_imgsz = int(gp("postproc/sam_imgsz", 640))
        self.occ_sam_warmup_n = int(gp("occ/sam_warmup_n", 8))

        # reinit debounce 秒數
        self.pp_retry_delay_sec = float(gp("postproc/retry_delay_sec", 1.0))
        self.pp_on_fail = gp("postproc/on_fail", "reinit").strip().lower()

    # =========================
    # YOLO / 幾何 / 工具
    # =========================
    def yolo_backend_info(self,detector):
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
                        rospy.loginfo("[YOLO Loader] PT on GPU OK")
                        return det, det_device
                    except Exception as ge:
                        rospy.logwarn(f"[YOLO Loader] PT move to GPU failed: {ge}. Will try PT on CPU.")
                else:
                    rospy.logwarn("[YOLO Loader] No CUDA available; PT will run on CPU.")
                try:
                    det.to("cpu"); det_device = "cpu"
                    rospy.loginfo("[YOLO Loader] PT on CPU OK")
                    return det, det_device
                except Exception as ce:
                    rospy.logwarn(f"[YOLO Loader] PT on CPU failed: {ce}")
            except Exception as e:
                rospy.logwarn(f"[YOLO Loader] PT load failed: {e}")
            onnx_fallback = _onnx_sibling(model_path)
            if os.path.isfile(onnx_fallback):
                rospy.loginfo(f"[YOLO Loader] Trying fallback ONNX: {onnx_fallback}")
                det, det_device = self._load_onnx_with_gpu_fallback(onnx_fallback)
                return det, det_device
            else:
                raise RuntimeError(f"Failed to load PT '{model_path}' and no sibling ONNX found.")
        elif ext == ".onnx":
            rospy.loginfo(f"[YOLO Loader] Loading ONNX: {model_path}")
            det, det_device = self._load_onnx_with_gpu_fallback(model_path)
            return det, det_device
        else:
            raise ValueError(f"Unsupported detector extension: {ext}. Use .pt or .onnx")
        
    def _load_onnx_with_gpu_fallback(self, onnx_path: str):
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
        if xyxy is None:
            return None
        x1, y1, x2, y2 = map(float, xyxy)
        return np.array([max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)], dtype=np.float32)

    def rect_to_mask(self, depth, xyxy, expand=0.0):
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
        r = detector.predict(source=img_bgr, imgsz=imgsz, conf=conf, device=self.det_device, verbose=False)[0]
        if len(r.boxes) == 0:
            return (np.empty((0,4), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int32))
        xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        sc   = r.boxes.conf.cpu().numpy().astype(np.float32)
        cl   = r.boxes.cls.cpu().numpy().astype(np.int32)
        return xyxy, sc, cl

    def pick_top1(self, img_bgr, cls_id, conf_thresh=None):
        xyxy, sc, cl = self.yolo_det_all(self.detector, img_bgr, imgsz=self.det_imgsz,
                                         conf=self.det_conf if conf_thresh is None else conf_thresh)
        if len(xyxy)==0: return None, None
        mask = (cl==cls_id) if cls_id is not None else np.ones(len(cl), dtype=bool)
        idx = np.where(mask)[0]
        if idx.size==0: return None, None
        j = idx[np.argmax(sc[idx])]
        return xyxy[j], sc[j]

    def project_3d_bbox_xyxy(self, K, center_pose, bbox_minmax, img_shape):
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

    def _bbox_depth_distance_m(self, bbox_xyxy, depth_m,
                            sample_ratio=0.35,
                            min_valid=30,
                            use="median",
                            z_min=0.05, z_max=10.0):
        if depth_m is None:
            return None

        x1, y1, x2, y2 = bbox_xyxy
        H, W = depth_m.shape[:2]

        x1 = int(np.clip(x1, 0, W - 1))
        x2 = int(np.clip(x2, 0, W - 1))
        y1 = int(np.clip(y1, 0, H - 1))
        y2 = int(np.clip(y2, 0, H - 1))
        if x2 <= x1 or y2 <= y1:
            return None

        # 取 bbox 中心區域，降低邊緣混到背景
        bw = x2 - x1
        bh = y2 - y1
        cx1 = int(x1 + (1 - sample_ratio) * 0.5 * bw)
        cx2 = int(x2 - (1 - sample_ratio) * 0.5 * bw)
        cy1 = int(y1 + (1 - sample_ratio) * 0.5 * bh)
        cy2 = int(y2 - (1 - sample_ratio) * 0.5 * bh)

        roi = depth_m[cy1:cy2, cx1:cx2]
        if roi.size == 0:
            return None

        vals = roi[(roi > z_min) & (roi < z_max)]
        if vals.size < min_valid:
            return None

        if use == "min":
            return float(np.min(vals))
        else:
            return float(np.median(vals))

    def select_yolo_bbox(self, xyxy, scores, classes, img_shape,
                     prefer_cls=None, select_mode="score", conf_th=0.0):
        """
        select_mode:
            - "score" : confidence 最大
            - "middle" : bbox 中心最靠近影像中心
            - "nearest_depth" : (scores >= conf_th) 且用 depth_m 選最近
        """
        if xyxy is None or len(xyxy) == 0:
            return None, None

        H, W = img_shape[:2]

        # 類別過濾
        idx = np.arange(len(xyxy))
        if prefer_cls is not None:
            idx = idx[classes == prefer_cls]
        if idx.size == 0:
            return None, None

        xyxy_f = xyxy[idx]
        scores_f = scores[idx]

        # conf門檻
        if conf_th is not None and conf_th > 0:
            keep = scores_f >= float(conf_th)
            if not np.any(keep):
                return None, None
            xyxy_f = xyxy_f[keep]
            scores_f = scores_f[keep]

        if select_mode == "score":
            j = int(np.argmax(scores_f))
            return xyxy_f[j], float(scores_f[j])

        elif select_mode == "middle":
            cx_img = W * 0.5
            cy_img = H * 0.5
            best_d, best_j = 1e18, -1
            for i, bb in enumerate(xyxy_f):
                cx = 0.5 * (bb[0] + bb[2])
                cy = 0.5 * (bb[1] + bb[3])
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
                dist = self._bbox_depth_distance_m(
                    bb, depth_m,
                    sample_ratio=0.35,
                    min_valid=30,
                    use="median",
                    z_min=0.05, z_max=z_max
                )
                if dist is None:
                    continue
                if dist < best_dist:
                    best_dist, best_j = dist, i

            if best_j >= 0:
                return xyxy_f[best_j], float(scores_f[best_j])

            # 深度都無效 → 退回 score
            j = int(np.argmax(scores_f))
            return xyxy_f[j], float(scores_f[j])

        else:
            rospy.logwarn_throttle(1.0, f"[YOLO] Unknown det_select_mode={select_mode}, fallback to score")
            j = int(np.argmax(scores_f))
            return xyxy_f[j], float(scores_f[j])

    # =========================
    # IoU / 遮蔽率（暖機後）
    # =========================
    def iou_vs_projection_for_class(self, color_bgr, K, center_pose, bbox_minmax, prefer_cls, det_imgsz, det_conf):
        """
        以 YOLO 偵測框（過濾 prefer_cls 後）對 3D 投影外框計算 IoU，回傳 (best_iou, best_xyxy)。
        若此幀未能計算（例如投影失效且也沒偵測），回傳 (None, None)；若有偵測但都無效，回 0.0 與 None。
        """
        if center_pose is None or K is None:
            return None, None

        H, W = color_bgr.shape[:2]
        est_xyxy = self.project_3d_bbox_xyxy(K, center_pose, bbox_minmax, img_shape=color_bgr.shape)
        # 投影失效 → 即便有偵測也無從比較，視為本幀無 IoU
        if est_xyxy is None:
            return None, None

        xyxy_all, sc_all, cl_all = self.yolo_det_all(self.detector, color_bgr, imgsz=det_imgsz, conf=det_conf)
        if len(xyxy_all) == 0:
            # 明確「沒偵測」→ 視為 IoU=0（可選：若想不計數，就改回 (None,None)）
            return 0.0, None

        use_mask = np.ones(len(xyxy_all), dtype=bool)
        if prefer_cls is not None and (cl_all == prefer_cls).any():
            use_mask = (cl_all == prefer_cls)

        xyxy_use = xyxy_all[use_mask]
        if len(xyxy_use) == 0:
            return 0.0, None

        # 計算對每個 YOLO 框的 IoU，取最大
        ious = []
        for bb in xyxy_use:
            bb_c = self.clip_xyxy(bb, W, H)
            ious.append(self.iou_xyxy(bb_c, est_xyxy))
        ious = np.array(ious, dtype=float)
        if ious.size == 0:
            return 0.0, None
        j = int(np.argmax(ious))
        return float(ious[j]), xyxy_use[j]
    
    def iou_vs_projection_for_class_from_dets(self, color_bgr, K, center_pose, bbox_minmax, prefer_cls, xyxy_all, sc_all, cl_all, tag="BUNCH"):
        """
        以「本幀已算好的 det arrays」對 3D 投影外框計算 IoU。
        return (best_iou, best_xyxy)
        - 投影失效 → (None, None)
        - 沒偵測 → (0.0, None)
        """
        if center_pose is None or K is None:
            return None, None

        H, W = color_bgr.shape[:2]
        est_xyxy = self.project_3d_bbox_xyxy(K, center_pose, bbox_minmax, img_shape=color_bgr.shape)
        if est_xyxy is None:
            return None, None

        if xyxy_all is None or len(xyxy_all) == 0:
            return 0.0, None

        use_mask = np.ones(len(xyxy_all), dtype=bool)
        if prefer_cls is not None and (cl_all == prefer_cls).any():
            use_mask = (cl_all == prefer_cls)

        xyxy_use = xyxy_all[use_mask]
        if len(xyxy_use) == 0:
            return 0.0, None

        ious = []
        for bb in xyxy_use:
            bb_c = self.clip_xyxy(bb, W, H)
            ious.append(self.iou_xyxy(bb_c, est_xyxy))
        ious = np.asarray(ious, dtype=float)
        if ious.size == 0:
            return 0.0, None

        j = int(np.argmax(ious))
        return float(ious[j]), xyxy_use[j]

    def maybe_regrab_roi_by_iou_from_dets(self, mode, center_pose, xyxy_all, sc_all, cl_all):
        """
        mode: "BUNCH" / "STEM"
        return True 表示已 reinit/清空，呼叫端應 continue
        """
        if center_pose is None or self.K is None:
            return False

        if (self.frame_count % max(1, self.iou_stride)) != 0:
            return False

        if mode == "BUNCH":
            prefer_cls = self.cls_bunch
            bbox_mm    = self.bbox_bunch
            bad_count_attr = "iou_bad_count_bunch"
            last_upd_attr  = "last_iou_update_bunch"
        else:
            prefer_cls = self.cls_stem
            bbox_mm    = self.bbox_stem
            bad_count_attr = "iou_bad_count_stem"
            last_upd_attr  = "last_iou_update_stem"

        iou_val, best_xyxy = self.iou_vs_projection_for_class_from_dets(
            self.color, self.K, center_pose, bbox_mm,
            prefer_cls=prefer_cls,
            xyxy_all=xyxy_all, sc_all=sc_all, cl_all=cl_all,
            tag=mode
        )
        self.iou_val = iou_val

        if iou_val is None:
            return False

        setattr(self, last_upd_attr, self.frame_count)
        if iou_val < float(self.iou_thresh):
            setattr(self, bad_count_attr, getattr(self, bad_count_attr) + 1)
        else:
            setattr(self, bad_count_attr, 0)

        if getattr(self, bad_count_attr) < int(self.iou_patience):
            return False

        # 觸發 reinit
        setattr(self, bad_count_attr, 0)

        if best_xyxy is not None:
            m = self.rect_to_mask(self.depth_m, self.clip_xyxy(best_xyxy, *self.rgb_size), expand=self.roi_expand)

            if mode == "BUNCH":
                new_pose = self.est_bunch.register(
                    K=self.K, rgb=self.color, depth=self.depth_m,
                    ob_mask=m, iteration=self.est_refine_iter
                )
                if new_pose is not None:
                    self.pose_bunch = new_pose
                    if getattr(self, "det_select_mode_current", self.det_select_mode) == "middle":
                        self._force_bunch_detect = True
                        self._hi_cnt = 0
                    # BUNCH 才需要 SAM warmup
                    self.sam_warmup_left_bunch = int(self.occ_sam_warmup_n)
            else:
                new_pose = self.est_stem.register(
                    K=self.K, rgb=self.color, depth=self.depth_m,
                    ob_mask=m, iteration=self.est_refine_iter
                )
                if new_pose is not None:
                    self.pose_stem = new_pose
                    # STEM 永遠不做 SAM
                    self.sam_warmup_left_stem = 0

            vis = self.color.copy()
            cv2.putText(vis, f"Re-init ROI ({mode}, low IoU)", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
            self.pump_windows(vis if self.show_rgb_win else None,
                            self.depth_vis if (self.show_depth_win and hasattr(self, "depth_vis")) else None)
            self._set_reinit(3)
            return True

        # 沒框 → 清空 pose
        if mode == "BUNCH":
            self.pose_bunch = None
        else:
            self.pose_stem = None

        vis = self.color.copy()
        cv2.putText(vis, f"Re-init needed ({mode}, low IoU, no det)", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
        self.pump_windows(vis if self.show_rgb_win else None,
                        self.depth_vis if (self.show_depth_win and hasattr(self, "depth_vis")) else None)
        self._set_reinit(3)
        return True

    def iou_xyxy(self, a, b):
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

    def occ_from_iou(self, yolo_xyxy, K, center_pose, bbox_minmax, img_shape, color_bgr):
        proj_xyxy = self.project_3d_bbox_xyxy(K, center_pose, bbox_minmax, img_shape)
        if proj_xyxy is None or yolo_xyxy is None:
            if getattr(self, "iou_log", False):
                self.save_iou_debug(color_bgr, K, center_pose, self.bbox_bunch, yolo_xyxy, iou=None, tag="BUNCH")
            return 1.0
        iou = self.iou_xyxy(self.clip_xyxy(yolo_xyxy, img_shape[1], img_shape[0]), proj_xyxy)
        if getattr(self, "iou_log", False):
            self.save_iou_debug(color_bgr, K, center_pose, self.bbox_bunch, yolo_xyxy, iou, tag="BUNCH")
        return 1.0 - iou
    # =========================
    # Debug helpers
    # =========================
    def _setup_run_debug_dir(self):
        """
        建立本次執行的 run folder：
        {debug_root}/{yyyyMMdd-HHmm}/
        若同分鐘重啟且已存在，會自動加尾碼 _01, _02...
        """
        root = getattr(self, "debug_root", None) or getattr(self, "debug_dir", "/tmp/fp_debug")
        ts = datetime.now().strftime("%Y%m%d-%H%M")  # yyyyMMdd-hhmm (24h)

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
        """Create directory d. If d is None, use self.debug_dir (for backward-compat old no-arg calls)."""
        if d is None:
            d = getattr(self, "debug_dir", None)
        if not d:
            return
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass

    def _ensure_parent(self, path: str):
        """Ensure parent directory for a file path exists."""
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        except Exception:
            pass

    def _dbg_path(self, subdir: str, prefix: str):
        """
        subdir: 子資料夾名稱（如 'occ_sam', 'iou_bunch', 'pose_stem'）
        prefix: 檔名前綴（如 'occ_sam_bunch', 'iou_bunch', 'pose_stem'）
        """
        root = os.path.join(self.debug_dir, subdir)
        self._ensure_dir(root)
        p = os.path.join(root, f"{prefix}_{self.frame_count:06d}.png")
        self._ensure_parent(p)
        return p

    def _overlay_mask(self, img_bgr, mask_bool, alpha=0.45, color=(0, 0, 255)):
        if mask_bool is None:
            return img_bgr
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
        if xyxy is None: return img
        x1, y1, x2, y2 = [int(t) for t in xyxy]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        if label is not None:
            cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return img

    def _draw_pose_box(self, img_bgr, K, pose_obj_in_cam, bbox_minmax, which, axis_scale=0.05):
        try:
            to_origin = self.to_origin_bunch if which.lower()=="bunch" else self.to_origin_stem
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

    def save_occ_sam_debug(self, color_bgr, sam_mask, K, center_pose, bbox_minmax, occ, yolo_xyxy, tag="BUNCH"):
        """暖機期（SAM）下的遮蔽率可視化：SAM mask、YOLO 框、3D bbox 投影框與 cuboid。"""
        if not getattr(self, "iou_log", False):
            return
        vis = color_bgr.copy()

        # 3D 投影外框（矩形）
        proj_xyxy = self.project_3d_bbox_xyxy(K, center_pose, bbox_minmax, color_bgr.shape)
        vis = self._overlay_mask(vis, sam_mask, alpha=0.45, color=(0, 0, 255))  # red overlay
        vis = self._draw_rect(vis, proj_xyxy, color=(0, 255, 0), thick=2, label="proj-2D bbox")  # green
        vis = self._draw_rect(vis, self.clip_xyxy(yolo_xyxy, vis.shape[1], vis.shape[0]) if yolo_xyxy is not None else None,
                              color=(0, 255, 255), thick=2, label="YOLO bbox")  # yellow

        # 3D cuboid
        # 這裡傳入的是幾何中心與物體座標的 pose，_draw_pose_box 內會補 axis
        kind = (tag or "BUNCH").lower()
        to_origin = self.to_origin_bunch if kind=="bunch" else self.to_origin_stem
        vis = self._draw_pose_box(vis, K, pose_obj_in_cam=center_pose @ to_origin,
            bbox_minmax=bbox_minmax, which=("bunch" if kind=="bunch" else "stem"), axis_scale=0.05)

        # 文字
        cv2.putText(vis, f"[{tag}] OCC(SAM)={occ:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 220), 2, cv2.LINE_AA)

        # 存檔
        outp = self._dbg_path("occ_sam", f"occ_sam_{tag.lower()}")
        cv2.imwrite(outp, vis)

    def save_iou_debug(self, color_bgr, K, center_pose, bbox_minmax, yolo_xyxy, iou, tag="BUNCH"):
        """暖機後（IoU）下的遮蔽率可視化：YOLO 框 vs 3D 投影外框，外加 cuboid。"""
        if not getattr(self, "iou_log", False):
            return
        vis = color_bgr.copy()

        proj_xyxy = self.project_3d_bbox_xyxy(K, center_pose, bbox_minmax, color_bgr.shape)
        vis = self._draw_rect(vis, proj_xyxy, color=(0, 255, 0), thick=2, label="proj-2D bbox")  # green
        vis = self._draw_rect(vis, self.clip_xyxy(yolo_xyxy, vis.shape[1], vis.shape[0]) if yolo_xyxy is not None else None,
                              color=(0, 255, 255), thick=2, label="YOLO bbox")  # yellow

        # 3D cuboid
        kind = (tag or "BUNCH").lower()
        to_origin = self.to_origin_bunch if kind=="bunch" else self.to_origin_stem
        vis = self._draw_pose_box(vis, K, pose_obj_in_cam=center_pose @ to_origin,
            bbox_minmax=bbox_minmax, which=("bunch" if kind=="bunch" else "stem"), axis_scale=0.05)

        occ = 1.0 - (float(iou) if iou is not None else 0.0)
        cv2.putText(vis, f"[{tag}] IoU={0.0 if iou is None else float(iou):.3f}  OCC={occ:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 220), 2, cv2.LINE_AA)

        outp = self._dbg_path(f"iou_{tag.lower()}", f"iou_{tag.lower()}")
        cv2.imwrite(outp, vis)

    def save_pose_bbox_debug(self, vis_bgr, tag="BUNCH"):
        if not getattr(self, "iou_log", False):
            return
        outp = self._dbg_path(f"pose_{tag.lower()}", f"pose_{tag.lower()}")
        cv2.imwrite(outp, vis_bgr)
    
    def save_binary_mask(self, mask_bool: np.ndarray, subdir: str, fname: str):
        """
        將 bool mask 以 0/255 PNG 存檔到 {debug_dir}/{subdir}/{fname}_{frame:06d}.png
        """
        if not getattr(self, "iou_log", False):
            return
        try:
            if mask_bool is None:
                return
            m = (mask_bool.astype(np.uint8) * 255)
            outp = self._dbg_path(subdir, fname)  # 自動帶 frame_count 與副檔名
            cv2.imwrite(outp, m)
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"[DBG] save_binary_mask failed: {e}")

    def save_gt_mask_debug(self, which: str):
        """
        存「完整投影」的 GT mask（= BOP 的 mask，不是 mask_visib）。
        不與 SAM 相交、不做 z-test，只是把 mesh 以當前 6D pose 投影成 2D 填面。
        """
        if not getattr(self, "iou_log", False):
            return
        if self.color is None or self.K is None:
            return
        H, W = self.color.shape[:2]

        if which.lower() == "bunch":
            if self.pose_bunch is None: return
            center_pose = self.pose_bunch @ np.linalg.inv(self.gt_to_origin_bunch)
            mesh = self.mesh_bunch
            tag  = "bunch"
        else:
            if self.pose_stem is None: return
            center_pose = self.pose_stem @ np.linalg.inv(self.gt_to_origin_stem)
            mesh = self.mesh_stem
            tag  = "stem"

        # 直接用你已有的投影成「輪廓遮罩」的函式（無可見性裁切）
        gt_mask_bool = self.render_mesh_silhouette_mask(self.K, center_pose, mesh, self.color.shape)

        # （可選）把小洞補起來，避免網格小破洞造成孔洞
        gt_u8 = (gt_mask_bool.astype(np.uint8) * 255)
        gt_u8 = cv2.morphologyEx(gt_u8, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

        outp = os.path.join(self.debug_dir, "gt_mask", f"gt_mask_{tag}_{self.frame_count:06d}.png")
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        cv2.imwrite(outp, gt_u8)

    # =========================
    # SAM / 遮蔽率（暖機期）
    # =========================
    def _get_warmup_left(self, which: str) -> int:
        return self.sam_warmup_left_bunch if which.lower()=="bunch" else self.sam_warmup_left_stem

    def _bunch_skip_occlusion(self) -> bool:
        mode = getattr(self, "det_select_mode_current", self.det_select_mode)
        return (str(mode).strip().lower() == "middle")

    def _dec_warmup_left(self, which: str):
        if which.lower()=="bunch":
            if self.sam_warmup_left_bunch > 0: self.sam_warmup_left_bunch -= 1
        else:
            if self.sam_warmup_left_stem  > 0: self.sam_warmup_left_stem  -= 1

    def compute_occ_and_iou(self, which: str, xyxy_all=None, sc_all=None, cl_all=None):
        occ = 1.0
        if self.color is None or self.K is None:
            return 1.0, 0.0, False

        which_l = which.lower()
        if which_l == "bunch":
            pose = self.pose_bunch
            bbox = self.bbox_bunch
            prefer_cls = self.cls_bunch
            to_origin = self.to_origin_bunch
        else:
            pose = self.pose_stem
            bbox = self.bbox_stem
            prefer_cls = self.cls_stem
            to_origin = self.to_origin_stem

        if pose is None:
            return 1.0, 0.0, False

        center_pose = pose @ np.linalg.inv(to_origin)

        # IoU：使用本幀 det arrays
        iou_val, best_xyxy = self.iou_vs_projection_for_class_from_dets(
            self.color, self.K, center_pose, bbox,
            prefer_cls=prefer_cls,
            xyxy_all=xyxy_all, sc_all=sc_all, cl_all=cl_all,
            tag=which_l.upper()
        )

        skip_occlusion_check = (which_l == "bunch" and self._force_bunch_detect)

        # ==========================================
        # 暖機階段 (Warmup): 執行 SAM 分割與遮蔽率計算
        # ==========================================
        if (not skip_occlusion_check) and (which_l == "bunch") and (self._get_warmup_left(which_l) > 0):
            self._dec_warmup_left(which_l)
            
            # 暖機剛好結束的瞬間：觸發效能結算報告
            if self._get_warmup_left(which_l) == 0:
                rospy.loginfo(f"[{which_l.upper()}] SAM warmup finished, switch to IoU-only mode (skip occ)")
                
                # ===== [嚴謹效能結算與輸出 - 僅在 iou_log == True 時執行] =====
                if getattr(self, "iou_log", False) and hasattr(self, 'perf_nn_times') and len(self.perf_nn_times) > 0:
                    avg_nn = float(np.mean(self.perf_nn_times))
                    avg_mesh = float(np.mean(self.perf_mesh_times))
                    avg_cpu = float(np.mean(self.perf_occ_cpu))
                    gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
                    
                    rospy.loginfo("====== [SAM2 OCCLUSION PERFORMANCE] ======")
                    rospy.loginfo(f"Warmup Frames     : {len(self.perf_nn_times)}")
                    rospy.loginfo(f"Avg AI Infer Time : {avg_nn:.2f} ms")
                    rospy.loginfo(f"Avg Mesh Calc Time: {avg_mesh:.2f} ms")
                    rospy.loginfo(f"Total Avg Time/Fr : {avg_nn + avg_mesh:.2f} ms")
                    rospy.loginfo(f"Avg CPU Usage     : {avg_cpu:.2f} %")
                    rospy.loginfo(f"Peak GPU VRAM     : {gpu_mem_mb:.2f} MB")
                    rospy.loginfo("==========================================")
                    
                    try:
                        import csv
                        from datetime import datetime
                        csv_path = os.path.join(self.debug_dir, "occ_sam_performance.csv")
                        file_exists = os.path.isfile(csv_path)
                        with open(csv_path, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow(["Timestamp", "Frames", "Avg_AI_ms", "Avg_Mesh_ms", "Avg_CPU_Percent", "Peak_GPU_Mem_MB"])
                            writer.writerow([datetime.now().strftime("%Y%m%d_%H%M%S"), len(self.perf_nn_times), round(avg_nn,2), round(avg_mesh,2), round(avg_cpu,2), round(gpu_mem_mb,2)])
                    except Exception as e:
                        rospy.logwarn(f"Failed to write sam performance CSV: {e}")
                    
                    # 清空數據
                    self.perf_nn_times.clear()
                    self.perf_mesh_times.clear()
                    self.perf_occ_cpu.clear()
                # ===============================

                return None, (0.0 if iou_val is None else float(iou_val)), False

            # --- [開始測量：僅在 iou_log 為 True 時耗費資源測量] ---
            if getattr(self, "iou_log", False):
                import time
                t_start = time.perf_counter()
                cpu_start = self.perf_process.cpu_times() if hasattr(self, 'perf_process') and self.perf_process else None

            # [第一階段] 產生提示框與 AI 推論
            prompt_box = self.pose_2d_box_xyxy(which_l)
            seg = self.seg_mask_from_sam(self.color, prompt_box)
            
            # --- [中途點：AI 推論結束] ---
            if getattr(self, "iou_log", False):
                t_mid = time.perf_counter()
                nn_time_ms = (t_mid - t_start) * 1000.0
                if hasattr(self, 'perf_nn_times'):
                    self.perf_nn_times.append(nn_time_ms)

            # [第二階段] 網格投影與物理交集計算
            occ = self.occ_from_gt_mesh_vs_sam(which_l, seg.astype(bool))

            # --- [結束點：物理計算結束] ---
            if getattr(self, "iou_log", False):
                t_end = time.perf_counter()
                mesh_time_ms = (t_end - t_mid) * 1000.0
                
                if cpu_start is not None:
                    cpu_end = self.perf_process.cpu_times()
                    cpu_time_spent = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
                    cpu_percent = (cpu_time_spent / (t_end - t_start)) * 100.0 if (t_end - t_start) > 0 else 0.0
                else:
                    cpu_percent = 0.0
                    
                if hasattr(self, 'perf_mesh_times'):
                    self.perf_mesh_times.append(mesh_time_ms)
                    self.perf_occ_cpu.append(cpu_percent)
            # ------------------------------------

            if getattr(self, "iou_log", False):
                try:
                    self.save_occ_sam_debug(
                        self.color, seg.astype(bool) if seg is not None else None,
                        self.K, center_pose, bbox, occ, best_xyxy, tag=which_l.upper()
                    )
                except Exception:
                    pass

            return occ, (0.0 if iou_val is None else float(iou_val)), True

        # ==========================================
        # 非暖機階段: IoU-only (完全跳過遮蔽率運算)
        # ==========================================
        if iou_val is None:
            occ = None
            iou_show = 0.0
        else:
            occ = None
            iou_show = float(iou_val)

        if getattr(self, "iou_log", False):
            try:
                self.save_iou_debug(self.color, self.K, center_pose, bbox, best_xyxy, iou_val, tag=which_l.upper())
            except Exception:
                pass

        return occ, iou_show, False

    def pose_2d_box_xyxy(self, which:str):
        """用 6D pose 的 3D bbox 投影出 2D 外框（當作 SAM 提示框）"""
        if which.lower() == "bunch":
            if self.pose_bunch is None or self.K is None: return None
            center_pose = self.pose_bunch @ np.linalg.inv(self.to_origin_bunch)
            bbox = self.bbox_bunch
        else:
            if self.pose_stem is None or self.K is None: return None
            center_pose = self.pose_stem @ np.linalg.inv(self.to_origin_stem)
            bbox = self.bbox_stem
        return self.project_3d_bbox_xyxy(self.K, center_pose, bbox, self.color.shape)

    def render_mesh_silhouette_mask(self, K, pose_center_in_cam, mesh, img_shape):
        """
        以 6D pose 將 mesh 投影為 2D 二值 mask。
        【超高速向量化版本】利用 NumPy 與 OpenCV 批次處理，避免 Python for 迴圈。
        """
        H, W = img_shape[:2]

        V = mesh.vertices.astype(np.float64)             # (N,3)
        Vh = np.concatenate([V, np.ones((V.shape[0],1))], axis=1)  # (N,4)
        Pc = (pose_center_in_cam @ Vh.T).T               # (N,4)

        Z = Pc[:, 2]
        valid_Z = Z > 1e-6
        
        # 預防除以零的警告
        Z_safe = Z.copy()
        Z_safe[~valid_Z] = 1.0

        # 一次性計算所有頂點的 u, v 投影
        u = K[0,0] * (Pc[:, 0] / Z_safe) + K[0,2]
        v = K[1,1] * (Pc[:, 1] / Z_safe) + K[1,2]

        # 將 u, v 結合成 (N, 2) 的陣列
        uv = np.stack([u, v], axis=1)

        # 取得網格的三角形面索引 (M, 3)
        F = mesh.faces
        
        # 利用索引快速映射出所有三角形的 2D 座標，形狀變為 (M, 3, 2)
        triangles = uv[F]

        # 找出哪些三角形是「有效」的（三個頂點都在相機前方）
        # valid_Z[F] 會產生 (M, 3) 的布林陣列，.all(axis=1) 表示三個頂點都要 True
        valid_faces = valid_Z[F].all(axis=1)
        
        if not np.any(valid_faces):
            return np.zeros((H, W), dtype=bool)

        valid_triangles = triangles[valid_faces]

        # 快速視域測試：過濾掉「完全在螢幕外」的三角形
        MARGIN = 10
        min_uv = valid_triangles.min(axis=1) # (M_valid, 2)
        max_uv = valid_triangles.max(axis=1) # (M_valid, 2)
        
        in_screen = ~(
            (min_uv[:, 0] > W + MARGIN) | 
            (max_uv[:, 0] < -MARGIN) | 
            (min_uv[:, 1] > H + MARGIN) | 
            (max_uv[:, 1] < -MARGIN)
        )
        
        final_triangles = valid_triangles[in_screen]

        mask = np.zeros((H, W), np.uint8)
        if final_triangles.shape[0] > 0:
            # OpenCV 繪圖要求整數，並加上裁切避免內部記憶體溢位
            final_triangles = np.clip(final_triangles, -2048, W + 2048).astype(np.int32)
            
            # 【效能核彈】用 cv2.fillPoly 一次性把所有的三角形畫上去！
            cv2.fillPoly(mask, final_triangles, 255, lineType=cv2.LINE_AA)

        return (mask > 0)    

    def occ_from_gt_mesh_vs_sam(self, which:str, sam_mask_bool: np.ndarray):
        """
        occ = 1 - | GT(mesh@pose) ∩ SAM | / | GT(mesh@pose) |
        which in {"bunch","stem"}
        """
        if self.color is None or self.K is None:
            return 1.0
        
        sam_mask_bool = sam_mask_bool.astype(bool)
        if sam_mask_bool.shape[:2] != self.color.shape[:2]:
            sam_mask_bool = cv2.resize(sam_mask_bool.astype(np.uint8), (self.color.shape[1], self.color.shape[0]),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
        if which.lower() == "bunch":
            if self.pose_bunch is None: return 1.0
            center_pose = self.pose_bunch @ np.linalg.inv(self.gt_to_origin_bunch)
            mesh = self.mesh_bunch
            tag  = "bunch"
        else:
            if self.pose_stem is None: return 1.0
            center_pose = self.pose_stem @ np.linalg.inv(self.gt_to_origin_stem)
            mesh = self.mesh_stem
            tag  = "stem"

        gt = self.render_mesh_silhouette_mask(self.K, center_pose, mesh, self.color.shape)
        if getattr(self, "iou_log", False):
            self.save_binary_mask(gt, "gt_mask", f"gt_{tag}")
            self.save_binary_mask(sam_mask_bool, "sam_mask", f"sam_{tag}")
        
        area = int(gt.sum())
        if area < 100:  # 幾乎看不到
            return 1.0

        inter = int(np.logical_and(gt, sam_mask_bool).sum())
        visible_ratio = inter / float(area)
        occ = float(1.0 - visible_ratio)
                        
        # —— Debug 輸出：{debug_dir}/occ_sam/occ_sam_<which>_XXXXXX.png
        if getattr(self, "iou_log", False):
            try:
                os.makedirs(os.path.join(self.debug_dir, "occ_sam"), exist_ok=True)
                vis = self.color.copy()
                # 疊 SAM（綠）與 GT（藍）輪廓
                sam_u8 = (sam_mask_bool.astype(np.uint8)*255)
                gt_u8  = (gt.astype(np.uint8)*255)
                cnts_sam,_ = cv2.findContours(sam_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts_gt,_  = cv2.findContours(gt_u8,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts_sam, -1, (0,255,0), 2)
                cv2.drawContours(vis, cnts_gt, -1, (255,0,0), 2)
                cv2.putText(vis, f"[{tag.upper()}] OCC(GTvsSAM)={occ:.3f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                fn = os.path.join(self.debug_dir, "occ_sam", f"occ_sam_{tag}_{self.frame_count:06d}.png")
                cv2.imwrite(fn, vis)
                self.save_gt_mask_debug(which=tag)
            except Exception as e:
                rospy.logwarn_throttle(1.0, f"[DBG] save occ_sam failed: {e}")

        return occ

    def render_bbox_mask_proxy(self, K, center_pose, bbox_minmax, img_shape):
        """用 3D bbox 投影外框矩形近似遮罩（快速、不渲染mesh）"""
        est_xyxy = self.project_3d_bbox_xyxy(K, center_pose, bbox_minmax, img_shape)
        if est_xyxy is None:
            return np.zeros(img_shape[:2], dtype=bool)
        return self.rect_to_mask(np.zeros(img_shape[:2], np.uint8), est_xyxy, expand=0.0)
    
    # def seg_mask_from_sam(self, bgr, xyxy):
    #     """用 SAM 基於 bbox 產生二值 mask（bool）。若不可用則回 bbox mask。"""
    #     H, W = bgr.shape[:2]
    #     x1,y1,x2,y2 = self.clip_xyxy(xyxy, W, H).astype(np.int32)
    #     if self.sam_predictor is None:
    #         m = np.zeros((H,W), dtype=bool); m[y1:y2, x1:x2] = True
    #         return m
    #     self.sam_predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    #     box = np.array([x1,y1,x2,y2], dtype=np.int32)
    #     masks, scores, _ = self.sam_predictor.predict(point_coords=None, point_labels=None, box=box[None,:], multimask_output=True)
    #     if masks is None or len(masks)==0:
    #         m = np.zeros((H,W), dtype=bool); m[y1:y2, x1:x2] = True
    #         return m
    #     return masks[int(np.argmax(scores))].astype(bool)

    def _unletterbox_mask_to_orig(self, m_sq: np.ndarray, orig_hw, imgsz: int):
        """m_sq: (imgsz, imgsz) or (H',W') from SAM in letterbox-square space"""
        H, W = orig_hw

        # letterbox scale
        r = min(imgsz / W, imgsz / H)
        new_w = int(round(W * r))
        new_h = int(round(H * r))

        pad_w = int((imgsz - new_w) / 2)
        pad_h = int((imgsz - new_h) / 2)

        # crop out padding area
        m_crop = m_sq[pad_h:pad_h + new_h, pad_w:pad_w + new_w]

        # resize back to original
        m_orig = cv2.resize(m_crop.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        return m_orig.astype(bool)

    def seg_mask_from_sam(self, bgr, xyxy, obj_key: str = "bunch", points=None, labels=None, use_prev_mask: bool = True):
        """
        SAM2 video-style segmentation using Ultralytics SAM2DynamicInteractivePredictor.

        Prompts:
          a) Box prompt  : initial spatial constraint (e.g., YOLO bbox)
          b) Point prompt: optional fg/bg refinement (points + labels)
          c) Mask prompt : previous-frame mask as constraint + memory update

        Args:
            bgr: np.ndarray(H,W,3) BGR image
            xyxy: [x1,y1,x2,y2] bbox in image coordinates
            obj_key: "bunch" or "stem" (maps to stable obj_id)
            points: list[[x,y], ...] or list[list[list[x,y]]] (Ultralytics SAM prompt format)
            labels: list[int] matching points (1=fg, 0=bg)
            use_prev_mask: whether to feed previous mask as mask prompt
        """
        H, W = bgr.shape[:2]
        if xyxy is None:
            return np.zeros((H, W), dtype=bool)

        x1, y1, x2, y2 = self.clip_xyxy(xyxy, W, H).astype(np.int32)

        # fallback: bbox mask
        def _bbox_mask():
            m = np.zeros((H, W), dtype=bool)
            m[y1:y2, x1:x2] = True
            return m

        if self.sam2_predictor is None:
            return _bbox_mask()

        obj_id = int(self.sam2_obj_ids.get(obj_key, 0))

        # ---- build mask prompt from previous frame ----
        mask_prompt = None
        if use_prev_mask and (obj_id in self.sam2_prev_masks):
            pm = self.sam2_prev_masks[obj_id]
            if pm.shape != (H, W):
                pm = cv2.resize(pm.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            # SAM2DynamicInteractivePredictor.update_memory expects (N,H,W) for N objects
            mask_prompt = pm.astype(np.uint8)[None, :, :]

        # ---- (optional) auto points: center fg + 4 bg outside bbox ----
        if points is None and labels is None:
            cx = int(0.5 * (x1 + x2))
            cy = int(0.5 * (y1 + y2))
            # 1 fg point at center, 4 bg points near corners (clipped)
            points = [[cx, cy],
                      [max(0, x1 - 5), max(0, y1 - 5)],
                      [min(W - 1, x2 + 5), max(0, y1 - 5)],
                      [max(0, x1 - 5), min(H - 1, y2 + 5)],
                      [min(W - 1, x2 + 5), min(H - 1, y2 + 5)]]
            labels = [1, 0, 0, 0, 0]

        # ---- normalize prompt shapes for Ultralytics SAM2 predictor ----
        # Ultralytics expects points/labels to be batched (B, K, 2) / (B, K) when using interactive predictor.
        # If caller provides a flat list of points like [[x,y], ...], wrap it into an outer list for B=1.
        if points is not None and len(points) > 0 and isinstance(points[0], (list, tuple)) and len(points[0]) == 2 and isinstance(points[0][0], (int, float, np.number)):
            points = [points]
        if labels is not None and len(labels) > 0 and isinstance(labels[0], (int, np.integer)):
            labels = [labels]

        # ---- run predictor ----
        try:
            # If we have a previous mask, we can just update memory with mask prompt (cheap constraints).
            # If not, use bbox as initial prompt to bootstrap.
            use_bbox = mask_prompt is None
            results = self.sam2_predictor(
                source=bgr,
                bboxes=[[int(x1), int(y1), int(x2), int(y2)]] if use_bbox else None,
                points=points,
                labels=labels,
                masks=mask_prompt,
                obj_ids=[obj_id],
                update_memory=True,
            )

            # results[0].masks.data can be torch.Tensor or np.ndarray, usually (N,H,W)
            m = results[0].masks.data
            if torch.is_tensor(m):
                m = m.detach().cpu().numpy()
            m = np.asarray(m)

            if m.ndim == 2:
                cand = [m]
            elif m.ndim == 3:
                cand = [m[i] for i in range(m.shape[0])]
            else:
                return _bbox_mask()

            # pick the mask that overlaps bbox most (robust when predictor returns multiple objs)
            best = None
            best_score = -1.0
            for mm in cand:
                mm_bin = (mm > 0)
                inter = mm_bin[y1:y2, x1:x2].sum()
                area = mm_bin.sum() + 1e-6
                score = float(inter / area)
                if score > best_score:
                    best_score = score
                    best = mm_bin

            if best is None:
                return _bbox_mask()

            # cache for next frame (mask prompt)
            self.sam2_prev_masks[obj_id] = best

            return best

        except Exception as e:
            rospy.logwarn_throttle(1.0, f"[SAM2] seg failed: {e}")
            return _bbox_mask()

    def occ_score_bunch(self, color_bgr, K, pose_obj_in_cam):
        # 6D pose → 2D 提示框（優先）
        prompt_box = self.pose_2d_box_xyxy("bunch")
        if prompt_box is None:
            # 沒有 pose 時就無法做 GT 對比，保守回完全遮蔽
            return 1.0

        # SAM 分割
        yolo_xyxy, _ = self.pick_top1(color_bgr, self.cls_bunch)  # 備用
        try:
            seg = self.seg_mask_from_sam(color_bgr, prompt_box)
        except Exception:
            # SAM 出錯時退回 YOLO 框當作 mask
            seg = self.rect_to_mask(self.depth_m, self.clip_xyxy(yolo_xyxy, *self.rgb_size), expand=0.0) if yolo_xyxy is not None else None
            if seg is None:
                return 1.0
        seg_bool = seg.astype(bool)

        # 用 GT(mask from mesh@pose) vs SAM
        occ = self.occ_from_gt_mesh_vs_sam("bunch", seg_bool)
        return occ

    # =============== GUI ===============
    def _open_window(self, name, pos_xy, init_size, is_rgb=True):
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
        val = float(value.get(label, 0.0))
        mv = max(1e-6, float(max_val))
        v = max(0.0, min(val / mv, 1.0))

        x, y = origin
        w, h = size
        # 背景
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), thickness=-1)
        # 進度
        cv2.rectangle(img, (x, y), (x + int(w * v), y + h), (60, 180, 75), thickness=-1)
        # 邊框
        cv2.rectangle(img, (x, y), (x + w, y + h), (220, 220, 220), thickness=1)
        # 文字
        cv2.putText(img, f"{label}: {val:.3f}", (x, y - 6),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    # =========================
    # 方向/尺寸 後處理（同前）
    # =========================
    def _orientation_ok(self, center_pose_cam: np.ndarray, origin_pose_cam: np.ndarray,
                        expect_orientation: str, tol_px: float):
        if center_pose_cam is None or origin_pose_cam is None or self.K is None:
            return True, 0.0
        Xc, Yc, Zc = map(float, center_pose_cam[:3, 3])
        Xo, Yo, Zo = map(float, origin_pose_cam[:3, 3])
        if Zc <= 1e-6 or Zo <= 1e-6:
            return True, 0.0
        fy, cy = float(self.K[1,1]), float(self.K[1,2])
        vc = fy*(Yc/Zc) + cy
        vo = fy*(Yo/Zo) + cy
        dv = float(vo - vc)     # >0: 原點更低；<0: 原點更高
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
        if not self.pp_enable:
            return True, False
        if which == "bunch":
            to_origin = self.to_origin_bunch
            bbox      = self.bbox_bunch
            expect_ori = self.bunch_expect_orientation
            size_cfg   = self.cfg_bunch
        else:
            to_origin = self.to_origin_stem
            bbox      = self.bbox_stem
            expect_ori = self.stem_expect_orientation
            size_cfg   = self.cfg_stem

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
        t = (float(T[0,3]), float(T[1,3]), float(T[2,3]))
        qx,qy,qz,qw = tf.transformations.quaternion_from_matrix(T)
        return t, (float(qx),float(qy),float(qz),float(qw))

    def broadcast_transform_and_pose(self, T: np.ndarray, which: str, parent: str):
        """Broadcast TF only.

        NOTE: Pose is now carried in Confidence.msg and published on a unified topic.
        """
        child = self.bunch_name if which.lower() == "bunch" else self.stem_name

        t = (float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
        qx, qy, qz, qw = tf.transformations.quaternion_from_matrix(T)
        self.tf_broadcaster.sendTransform(t, (qx, qy, qz, qw), rospy.Time.now(), child, parent)

    def imageCallback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("rgb decode failed: %r", e); return
        self.color = img; self.got_rgb = True
        self.rgb_size = (img.shape[1], img.shape[0])

    def infoCallback(self, msg: CameraInfo):
        self.info_msg = msg
        if self.K is None:
            self.K = np.array([[msg.K[0], 0, msg.K[2]],
                               [0, msg.K[4], msg.K[5]],
                               [0, 0, 1]], dtype=np.float64)

    def depthCallback(self, msg: Image):
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
        """Unified detection gating callback.

        - detection_allowed: whether we are allowed to run YOLO/pose pipeline
        - det_select_mode: affects BUNCH only. When already in STEM, occlusion check is skipped.
        """
        mode = (getattr(msg, "det_select_mode", "") or "").strip().lower()
        if mode not in ("score", "middle", "nearest_depth"):
            rospy.logwarn_throttle(1.0, f"[DETECTION] invalid det_select_mode={mode}, fallback score")
            mode = "score"
        self.det_select_mode_current = mode
        self.ready_received.detection_allowed = bool(getattr(msg, "detection_allowed", False))
    
    def harvestDoneCallback(self, msg: Bool):
        if not bool(msg.data):
            return

        rospy.logwarn("[HARVEST_DONE] received True -> hard reset to BUNCH")
        self._reset_all_to_bunch()
        self._publish_zero_current(used_sam=False)

    def _tag(self, which: str) -> str:
        return "BUNCH" if which.lower() == "bunch" else "STEM"

    def _set_registering(self, n_frames: int = 2):
        self._registering_until = self.frame_count + max(1, int(n_frames))

    def _set_reinit(self, n_frames: int = 3):
        self._reinit_until = self.frame_count + max(1, int(n_frames))

    def _state(self, which: str, used_sam: bool = False) -> str:
        part = self._tag(which)

        allowed = bool(getattr(self.ready_received, "detection_allowed", False))

        if self.yolo_start_mode == "wait" and (not allowed):
            return f"{part}:PAUSED"

        if not (self.got_depth and self.got_rgb and self.K is not None and hasattr(self, "depth_m")):
            return f"{part}:INITIALIZING"

        if getattr(self, "_post_pending", False):
            return f"{part}:POSTPROCESS"

        if getattr(self, "_reinit_until", 0) >= self.frame_count:
            return f"{part}:REINITIALIZING"

        if getattr(self, "_registering_until", 0) >= self.frame_count:
            return f"{part}:REGISTERING"

        if which.lower() == "bunch":
            if self.pose_bunch is None:
                return f"{part}:YOLO"
            if used_sam and self.sam_warmup_left_bunch > 0:
                return f"{part}:SAM"
            return f"{part}:STABLE"
        else:
            if self.pose_stem is None:
                return f"{part}:YOLO"
            if used_sam and self.sam_warmup_left_stem > 0:
                return f"{part}:SAM"
            return f"{part}:STABLE"

    def confidence_publish(self, which: str, iou: float, detection: bool, used_sam: bool = False):
        """Publish unified Confidence.msg for either bunch or stem.

        Confidence.msg fields used:
        - stamp
        - frame_id: self.bunch_name or self.stem_name
        - object_IoU
        - object_detection
        - state
        - position / orientation: from current estimated pose (if available)
        """
        conf_msg = Confidence()
        conf_msg.stamp = rospy.Time.now()
        conf_msg.state = self._state(which, used_sam=used_sam)
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

    def _publish_zero_current(self, used_sam: bool = False):
        """Publish a single zero-confidence message on the unified topic."""
        which = "stem" if self.mode == "STEM" else "bunch"
        self.confidence_publish(which, 0.0, False, used_sam=used_sam)

    def _reset_pipeline_state(self):
        """Reset the whole pipeline to initial waiting state (same as your duplicated logic)."""
        self.mode = "BUNCH"
        self.pose_bunch = None
        self.pose_stem  = None
        self._hi_cnt = 0
        self._stem_lock = False

        # IoU / counters
        self.iou_bad_count = 0
        self.iou_val = None
        self.iou_bad_count_bunch = 0
        self.iou_bad_count_stem  = 0

        # SAM warmup
        self.sam_warmup_left_bunch = 0
        self.sam_warmup_left_stem  = 0

        # Force re-detect flag
        self._force_bunch_detect = False

    def _reset_all_to_bunch(self):
        """Hard reset: clear all cached data and return to BUNCH."""
        self._reset_pipeline_state()

        # Clear YOLO debounce buffers
        self._yolo_delay_left_bunch = 0
        self._yolo_delay_left_stem = 0
        self._yolo_delay_bbox_bunch = None
        self._yolo_delay_bbox_stem = None

        # Clear SAM2 mask memory
        try:
            self.sam2_prev_masks.clear()
        except Exception:
            self.sam2_prev_masks = {}

        # Clear postproc debounce
        self._post_pending = False
        self._post_fail_time = None

        # Clear transient windows text
        self._last_yolo_text = ""

        # Clear scheduling guards
        self._registering_until = 0
        self._reinit_until = 0
    
    def _handle_detection_paused(self):
        """Pause: do NOT reset any state. Just hold and publish zero(optional)."""
        if self.color is not None:
            vis_rgb = self.color.copy()
            cv2.putText(vis_rgb, "PAUSED (detection_allowed=FALSE)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
        else:
            vis_rgb = None

        # 你要不要 publish 0：我建議要（讓上游知道暫停）
        self._publish_zero_current(used_sam=False)

        self.pump_windows(
            vis_rgb if self.show_rgb_win else None,
            self.depth_vis if (self.show_depth_win and hasattr(self, "depth_vis")) else None
        )

    def pick_nearest_from_dets(self, xyxy, sc, cl, target_xyxy, cls_id):
        """
        Pick nearest bbox (by bbox center distance) from already computed det arrays.
        Avoids re-running YOLO in the same frame.
        """
        if xyxy is None or len(xyxy) == 0 or target_xyxy is None:
            return None, None

        idx = np.where(cl == int(cls_id))[0]
        if idx.size == 0:
            return None, None

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
            return None, None
        return xyxy[bid], float(sc[bid])

    def _yolo_delay_update(self, which: str, xyxy_now):
        """
        which: "bunch" or "stem"
        xyxy_now: 本幀對應 bbox (np.array[4]) 或 None
        return: (ready: bool, bbox_to_use)
        - ready=True 表示倒數結束，可以拿 bbox_to_use 去 register()
        - 如果中途 bbox 消失，會重置倒數
        """
        self.yolo_delay_frames = 5
        n = max(0, int(getattr(self, "yolo_delay_frames", 0)))

        if which == "bunch":
            left_attr = "_yolo_delay_left_bunch"
            bbox_attr = "_yolo_delay_bbox_bunch"
        else:
            left_attr = "_yolo_delay_left_stem"
            bbox_attr = "_yolo_delay_bbox_stem"

        left = int(getattr(self, left_attr, 0))
        bbox_hold = getattr(self, bbox_attr, None)

        # 不需要 delay
        if n <= 0:
            return (xyxy_now is not None), xyxy_now

        # 本幀沒偵測到 → 重置
        if xyxy_now is None:
            setattr(self, left_attr, 0)
            setattr(self, bbox_attr, None)
            return False, None

        # 本幀有偵測到
        if left <= 0:
            # 第一次進入 delay
            setattr(self, left_attr, n)
            setattr(self, bbox_attr, xyxy_now.copy())
            return False, None

        # 持續偵測到：更新 hold bbox（用最新的）
        setattr(self, bbox_attr, xyxy_now.copy())

        # 倒數
        left -= 1
        setattr(self, left_attr, left)

        if left <= 0:
            # 倒數結束 → 回傳 bbox
            bbox_use = getattr(self, bbox_attr, xyxy_now)
            return True, bbox_use

        return False, None

    # =========================
    # 主循環
    # =========================
    def spin(self):
        self.frame_count = 0
        used_sam = False

        while not rospy.is_shutdown():
            iou_for_bar = 0.0
            used_sam = False
            # 1) 等資料齊
            if not (self.got_depth and self.got_rgb and self.K is not None and hasattr(self, "depth_m")):
                self.pump_windows(self.color if self.got_rgb else None,
                                self.depth_vis if self.got_depth else None)
                rospy.sleep(0.1)
                continue

            # 2) STEM lock gating：一旦切到 STEM 就永遠只跑 STEM，直到 external reset / harvest done
            allowed = bool(getattr(self.ready_received, "detection_allowed", False))
            if self._stem_lock:
                if self.yolo_start_mode == "wait" and (not allowed):
                    # pause: keep stem lock + keep mode, do NOT reset
                    self.mode = "STEM"
                    self._handle_detection_paused()
                    rospy.sleep(0.05)
                    continue

                # still allowed
                self.mode = "STEM"

            # 3) wait 模式 gating：統一用同一個 allow
            if self.yolo_start_mode == "wait" and (not allowed):
                self._handle_detection_paused()
                rospy.sleep(0.05)
                continue

            # 4) postprocess debounce：倒數中暫停
            if getattr(self, "_post_pending", False):
                now = rospy.Time.now()
                start_t = self._post_fail_time or now
                elapsed = (now - start_t).to_sec()
                remaining = max(0.0, float(self.pp_retry_delay_sec) - elapsed)

                vis_rgb = self.color.copy()
                cv2.putText(vis_rgb, f"Post-check pending... reinit in {remaining:.1f}s",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

                self.pump_windows(vis_rgb if self.show_rgb_win else None,
                                self.depth_vis if (self.show_depth_win and hasattr(self, "depth_vis")) else None)
                self._publish_zero_current(used_sam=used_sam)

                if elapsed >= float(self.pp_retry_delay_sec):
                    rospy.logwarn("[POST] Debounce timeout reached. Reinit now.")
                    self._set_reinit(3)
                    self._post_pending = False
                    self._post_fail_time = None
                    self.pose_bunch = None
                    self.pose_stem  = None

                rospy.sleep(0.01)
                continue

            # ---- 進入本幀處理 ----
            self.frame_count += 1

            # 5) 本幀只跑一次 YOLO
            xyxy_all, sc_all, cl_all = self.yolo_det_all(
                self.detector, self.color, imgsz=self.det_imgsz, conf=self.det_conf
            )

            bunch_xyxy, bunch_conf = self.select_yolo_bbox(
                xyxy_all, sc_all, cl_all,
                img_shape=self.color.shape,
                prefer_cls=self.cls_bunch,
                select_mode=getattr(self, "det_select_mode_current", self.det_select_mode),
                conf_th=self.det_conf
            )

            vis_bgr = self.color.copy()

            # =========================
            # BUNCH mode
            # =========================
            if self.mode == "BUNCH":
                # (A) init bunch pose
                if self.pose_bunch is None:
                    # 先做 YOLO 延時：需要連續看到 bbox 才會 ready
                    ready, bb_use = self._yolo_delay_update("bunch", bunch_xyxy)
                    vis_bgr = self.color.copy()

                    if bunch_xyxy is None:
                        self.pump_windows(vis_bgr, self.depth_vis if self.got_depth else None)
                        self._publish_zero_current(used_sam=used_sam)
                        continue

                    # 顯示倒數文字，不要register
                    left = int(getattr(self, "_yolo_delay_left_bunch", 0))
                    if not ready:
                        cv2.putText(vis_bgr, f"YOLO detected. Delay register... ({left} frames left)",
                                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
                        self.pump_windows(vis_bgr, self.depth_vis if self.got_depth else None)
                        self._publish_zero_current(used_sam=used_sam)
                        continue

                    m = self.rect_to_mask(self.depth_m, self.clip_xyxy(bb_use, *self.rgb_size), expand=self.roi_expand)
                    self.pose_bunch = self.est_bunch.register(
                        K=self.K, rgb=self.color, depth=self.depth_m,
                        ob_mask=m, iteration=self.est_refine_iter
                    )
                    if self.pose_bunch is None:
                        self._publish_zero_current(used_sam=used_sam)
                        continue

                    self._set_registering(2)
                    if not self._bunch_skip_occlusion():
                        self.sam_warmup_left_bunch = int(self.occ_sam_warmup_n)
                    else:
                        self.sam_warmup_left_bunch = 0

                # (B) track
                self.pose_bunch = self.est_bunch.track_one(
                    rgb=self.color, depth=self.depth_m, K=self.K, iteration=self.track_refine_iter
                )

                # (C) occ/iou
                if self._bunch_skip_occlusion():
                    used_sam = False
                    self.sam_warmup_left_bunch = 0  # 保險：避免之前殘留 warmup
                    center_pose = self.pose_bunch @ np.linalg.inv(self.to_origin_bunch)
                    iou_val, _ = self.iou_vs_projection_for_class_from_dets(
                        self.color, self.K, center_pose, self.bbox_bunch,
                        prefer_cls=self.cls_bunch,
                        xyxy_all=xyxy_all, sc_all=sc_all, cl_all=cl_all,
                        tag="BUNCH"
                    )
                    iou_for_bar = 0.0 if iou_val is None else float(iou_val)
                    occ = None
                else:
                    occ, iou_for_bar, used_sam = self.compute_occ_and_iou("bunch", xyxy_all, sc_all, cl_all)

                # (D) IoU-based regrab ROI (bunch)
                if self.pose_bunch is not None:
                    center_pose = self.pose_bunch @ np.linalg.inv(self.to_origin_bunch)
                    if self.maybe_regrab_roi_by_iou_from_dets("BUNCH", center_pose, xyxy_all, sc_all, cl_all):
                        self._publish_zero_current(used_sam=used_sam)
                        continue

                # (E) hi_cnt logic (same as yours)
                if self._bunch_skip_occlusion():    # middle：永遠不進遮蔽率狀態機
                    self._hi_cnt = 0
                    rospy.loginfo_throttle(1.0, f"[BUNCH] det_select_mode=middle iou={iou_for_bar:.2f} (skip occlusion)")
                else:
                    if used_sam:
                        # 只在 SAM warmup 期間才做遮蔽判斷 / 切 STEM
                        rospy.loginfo_throttle(1.0, f"[BUNCH] occ={float(occ):.2f} warmup=Y iou={iou_for_bar:.2f}")
                        if occ is not None and occ >= self.policy_occ_hi:
                            self._hi_cnt += 1
                    else:
                        # warmup 結束後：不算 occ、不使用 occ、也不切 stem
                        self._hi_cnt = 0
                        rospy.loginfo_throttle(1.0, f"[BUNCH] warmup=N iou={iou_for_bar:.2f} (skip occ & stem-switch)")

                    rospy.loginfo_throttle(1.0, f"[BUNCH] hi_cnt={self._hi_cnt}")

                    # (F) switch to STEM only during warmup (used_sam)
                    if used_sam and self._hi_cnt >= self.policy_hi_pat:
                        stem_xyxy, _ = self.pick_nearest_from_dets(xyxy_all, sc_all, cl_all, bunch_xyxy, self.cls_stem)
                        if stem_xyxy is not None:
                            m = self.rect_to_mask(self.depth_m, self.clip_xyxy(stem_xyxy, *self.rgb_size), expand=self.roi_expand)
                            self.pose_stem = self.est_stem.register(
                                K=self.K, rgb=self.color, depth=self.depth_m,
                                ob_mask=m, iteration=self.est_refine_iter
                            )
                            if self.pose_stem is not None:
                                self.mode = "STEM"
                                self._stem_lock = True
                                self.pose_bunch = None
                                self.sam_warmup_left_bunch = 0
                                try:
                                    self.sam2_prev_masks.pop(self.sam2_obj_ids["bunch"], None)
                                except Exception:
                                    pass
                                self._hi_cnt = 0
                                self._publish_zero_current(used_sam=used_sam)
                                self.pump_windows(vis_bgr if self.show_rgb_win else None,
                                                self.depth_vis if (self.show_depth_win and hasattr(self, "depth_vis")) else None)
                                continue

                # (G) postprocess (bunch)
                ok_to_publish, pending = self.postprocess_and_maybe_reinit(self.pose_bunch, which="bunch")
                if pending:
                    cv2.putText(vis_bgr, "Post-check pending...", (10,90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                    self.pump_windows(vis_bgr if self.show_rgb_win else None,
                                    self.depth_vis if (self.show_depth_win and hasattr(self, "depth_vis")) else None)
                    self._publish_zero_current(used_sam=used_sam)
                    continue

                # (H) publish TF/pose + visualize (bunch)
                if ok_to_publish and self.pose_bunch is not None:
                    center_pose = self.pose_bunch @ np.linalg.inv(self.to_origin_bunch)
                    vis = draw_posed_3d_box(self.K, img=cv2.cvtColor(self.color, cv2.COLOR_BGR2RGB),
                                            ob_in_cam=center_pose, bbox=self.bbox_bunch)
                    vis = draw_xyz_axis(vis, ob_in_cam=self.pose_bunch, scale=0.05, K=self.K,
                                        thickness=3, transparency=0, is_input_rgb=True)
                    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

                    parent_frame = self.camera_tf if self.camera_tf else "camera_color_optical_frame"
                    self.broadcast_transform_and_pose(self.pose_bunch, "bunch", parent_frame)
                    self.confidence_publish("bunch", iou_for_bar, True, used_sam=used_sam)

            # =========================
            # STEM mode
            # =========================
            else:
                self._hi_cnt = 0
                used_sam = False  # STEM不用SAM

                # (A) init stem pose if needed
                if self.pose_stem is None:
                    if bunch_xyxy is None:
                        self.pump_windows(vis_bgr, self.depth_vis if self.got_depth else None)
                        self._publish_zero_current(used_sam=used_sam)
                        continue

                    stem_xyxy, _ = self.pick_nearest_from_dets(xyxy_all, sc_all, cl_all, bunch_xyxy, self.cls_stem)
                    if stem_xyxy is None:
                        # stem 沒偵測到 → reset delay
                        self._yolo_delay_update("stem", None)
                        self.pump_windows(vis_bgr, self.depth_vis if self.got_depth else None)
                        self._publish_zero_current(used_sam=used_sam)
                        continue
                    # stem bbox delay
                    ready, bb_use = self._yolo_delay_update("stem", stem_xyxy)
                    left = int(getattr(self, "_yolo_delay_left_stem", 0))
                    if not ready:
                        cv2.putText(vis_bgr, f"STEM YOLO detected. Delay register... ({left} frames left)",
                                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
                        self.pump_windows(vis_bgr, self.depth_vis if self.got_depth else None)
                        self._publish_zero_current(used_sam=used_sam)
                        continue

                    m = self.rect_to_mask(self.depth_m, self.clip_xyxy(bb_use, *self.rgb_size), expand=self.roi_expand)
                    self.pose_stem = self.est_stem.register(
                        K=self.K, rgb=self.color, depth=self.depth_m,
                        ob_mask=m, iteration=self.est_refine_iter
                    )
                    if self.pose_stem is None:
                        self._publish_zero_current(used_sam=used_sam)
                        continue

                    self._set_registering(2)

                # (B) track stem
                self.pose_stem = self.est_stem.track_one(
                    rgb=self.color, depth=self.depth_m, K=self.K, iteration=self.track_refine_iter
                )

                if self.pose_stem is not None:
                    center_pose = self.pose_stem @ np.linalg.inv(self.to_origin_stem)
                    iou_val, _ = self.iou_vs_projection_for_class_from_dets(
                        self.color, self.K, center_pose, self.bbox_stem,
                        prefer_cls=self.cls_stem,
                        xyxy_all=xyxy_all, sc_all=sc_all, cl_all=cl_all,
                        tag="STEM"
                    )
                    iou_for_bar = 0.0 if iou_val is None else float(iou_val)
                else:
                    iou_for_bar = 0.0

                # (D) if track lost → 重新抓一次（但不做遮蔽率判斷）
                if self.pose_stem is None:
                    if bunch_xyxy is not None:
                        stem_xyxy, _ = self.pick_nearest_from_dets(xyxy_all, sc_all, cl_all, bunch_xyxy, self.cls_stem)
                        if stem_xyxy is not None:
                            m = self.rect_to_mask(self.depth_m, self.clip_xyxy(stem_xyxy, *self.rgb_size), expand=self.roi_expand)
                            self.pose_stem = self.est_stem.register(
                                K=self.K, rgb=self.color, depth=self.depth_m,
                                ob_mask=m, iteration=self.est_refine_iter
                            )
                    if self.pose_stem is None:
                        self.confidence_publish("stem", 0.0, False, used_sam=False)
                        self.pump_windows(vis_bgr if self.show_rgb_win else None,
                                        self.depth_vis if (self.show_depth_win and hasattr(self, "depth_vis")) else None)
                        continue

                # (E) postprocess (stem)
                ok_to_publish, pending = self.postprocess_and_maybe_reinit(self.pose_stem, which="stem")
                if pending:
                    cv2.putText(vis_bgr, "Post-check pending...", (10,90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                    self.pump_windows(vis_bgr if self.show_rgb_win else None,
                                    self.depth_vis if (self.show_depth_win and hasattr(self, "depth_vis")) else None)
                    self._publish_zero_current(used_sam=used_sam)
                    continue

                # (F) IoU-based regrab ROI (stem)
                if self.pose_stem is not None:
                    center_pose = self.pose_stem @ np.linalg.inv(self.to_origin_stem)
                    if self.maybe_regrab_roi_by_iou_from_dets("STEM", center_pose, xyxy_all, sc_all, cl_all):
                        self._publish_zero_current(used_sam=used_sam)
                        continue

                # (G) publish TF/pose + visualize (stem)
                if ok_to_publish and self.pose_stem is not None:
                    center_pose = self.pose_stem @ np.linalg.inv(self.to_origin_stem)
                    vis = draw_posed_3d_box(self.K, img=cv2.cvtColor(self.color, cv2.COLOR_BGR2RGB),
                                            ob_in_cam=center_pose, bbox=self.bbox_stem)
                    vis = draw_xyz_axis(vis, ob_in_cam=self.pose_stem, scale=0.05, K=self.K,
                                        thickness=3, transparency=0, is_input_rgb=True)
                    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

                    parent_frame = self.camera_tf if self.camera_tf else "camera_color_optical_frame"
                    self.broadcast_transform_and_pose(self.pose_stem, "stem", parent_frame)
                    self.confidence_publish("stem", iou_for_bar, True, used_sam=used_sam)

            # 5) IoU bar
            bar_w, bar_h, margin = 220, 18, 10
            iou_show = float(iou_for_bar) if 'iou_for_bar' in locals() else 0.0
            self.draw_conf_bar(
                vis_bgr, {"IoU": iou_show},
                label="IoU",
                origin=(10, vis_bgr.shape[0] - margin - bar_h),
                size=(bar_w, bar_h),
                max_val=1.0
            )

            # 6) show
            self.pump_windows(
                vis_bgr if (self.show_rgb_win and self.color is not None) else None,
                self.depth_vis if (self.show_depth_win and self.got_depth and self.depth_vis is not None) else None
            )

        cv2.destroyAllWindows()

if __name__ == "__main__":
    rospy.init_node("pipeline_tracker", anonymous=False)
    node = FoundationPosePipelineTracker()
    node.spin()
