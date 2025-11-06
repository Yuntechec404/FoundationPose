#!/home/user/anaconda3/envs/foundationpose/bin/python3
# -*- coding: utf-8 -*-

import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import rospy
import numpy as np
import cv2
import trimesh
import torch

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Transform
from cv_bridge import CvBridge
from ros_foundationpose.msg import Confidence
from forklift_server.msg import Detection

import tf

# --- 專案路徑 ---
sys.path.append('/home/user/anaconda3/envs/foundationpose/lib/python3.8/site-packages')
FOUNDATIONPOSE_SRC = "/home/user/FoundationPose"
if FOUNDATIONPOSE_SRC not in sys.path:
    sys.path.append(FOUNDATIONPOSE_SRC)

os.environ.setdefault("ULTRALYTICS_NO_INSTALL", "1")

from estimater import FoundationPose, draw_posed_3d_box, draw_xyz_axis, ScorePredictor, PoseRefinePredictor, dr
from ultralytics import YOLO

# SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
    _SAM_AVAILABLE = True
except Exception:
    _SAM_AVAILABLE = False

selecting_bbox = False
box_points = []

class FoundationPosePipelineTracker:
    def __init__(self):
        self.init_parameter()
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
        self.ready_received = Detection()
        self.ready_received.detection_allowed = False
        self.ready_received.layer = 0.0

        # SAM 暖機幀數計數器
        self.sam_warmup_left_bunch = 0
        self.sam_warmup_left_stem  = 0

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
        # Pose publishers
        self.pose_pub_bunch = rospy.Publisher(self.bunch_name, Pose, queue_size=1, latch=True)
        self.pose_pub_stem  = rospy.Publisher(self.stem_name,  Pose, queue_size=1, latch=True)
        # Confidence publishers
        self.conf_pub_bunch = rospy.Publisher(self.bunch_name + "_confidence", Confidence, queue_size=1, latch=True)
        self.conf_pub_stem  = rospy.Publisher(self.stem_name  + "_confidence", Confidence, queue_size=1, latch=True)
        # Optional: detection gating topic → 掛在 bunch 名稱上
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
        self.scorer  = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx   = dr.RasterizeCudaContext()

        self.est_bunch = FoundationPose(model_pts=self.mesh_bunch.vertices,
                                        model_normals=self.mesh_bunch.vertex_normals,
                                        mesh=self.mesh_bunch, scorer=self.scorer, refiner=self.refiner,
                                        debug_dir=self.debug_dir, debug=0, glctx=self.glctx)
        self.est_stem  = FoundationPose(model_pts=self.mesh_stem.vertices,
                                        model_normals=self.mesh_stem.vertex_normals,
                                        mesh=self.mesh_stem, scorer=self.scorer, refiner=self.refiner,
                                        debug_dir=self.debug_dir, debug=0, glctx=self.glctx)
        rospy.loginfo("Estimator initialization done (bunch+stem)")

        # YOLOv11
        self.detector, self.det_device = self.load_detector(self.det_model)
        is_gpu, yolo_desc = self.yolo_uses_gpu(self.detector)
        rospy.loginfo(f"[YOLO] GPU enabled: {is_gpu}  ({yolo_desc})")
        rospy.loginfo(f"[YOLO] predict device hint: {self.det_device}")
        rospy.loginfo("Detector initialization done")

        # SAM predictor
        self.sam_predictor = None
        if self.seg_backend == "sam" and _SAM_AVAILABLE:
            try:
                sam = sam_model_registry[self.sam_model](checkpoint=self.sam_ckpt)
                if torch.cuda.is_available():
                    sam.to(device="cuda")
                self.sam_predictor = SamPredictor(sam)
                rospy.loginfo("SAM loaded: %s", self.sam_model)
            except Exception as e:
                rospy.logwarn("SAM init failed: %r. Fallback to bbox mask.", e)
        else:
            if self.seg_backend == "sam":
                rospy.logwarn("segment_anything not available. Fallback to bbox mask.")

        # 狀態機
        self.mode = "BUNCH"     # BUNCH / STEM
        self._hi_cnt = 0
        self._lo_cnt = 0

    # ---------------------------
    # 參數
    # ---------------------------
    def init_parameter(self):
        ns = rospy.get_name()
        gp = lambda k, d: rospy.get_param(ns + "/" + k, d)

        # Topics / frames
        self.image_topic = gp("image_topic", "/camera/color/image_raw")
        self.info_topic  = gp("info_topic",  "/camera/color/camera_info")
        self.depth_topic = gp("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.depth_info_topic = gp("depth_info_topic", "")
        self.camera_tf   = gp("camera_tf", "")
        self.bunch_name  = gp("bunch_name", "oilpalm")
        self.stem_name   = gp("stem_name", "stem")

        # Files & modes
        self.mesh_file       = gp("mesh_file", "")
        self.mesh_file_stem  = gp("mesh_file_stem", "")
        self.det_model       = gp("det_model", "yolov11n.onnx")
        self.init_mode       = gp("init_mode", "yolo")
        self.yolo_start_mode = gp("yolo_start_mode", "immediate").strip().lower()
        self.debug_dir       = gp("debug_dir", "/tmp/fp_debug")

        # YOLO
        self.det_conf   = float(gp("det_conf", 0.25))
        self.det_class  = int(gp("det_class", -1))
        self.det_imgsz  = int(gp("det_imgsz", 640))
        self.det_select_mode = gp("det_select_mode", "score").strip().lower()
        if self.det_select_mode not in ("score","top","bottom"):
            rospy.logwarn("Unknown det_select_mode=%s, fallback to 'score'", self.det_select_mode)
            self.det_select_mode = "score"
        self.prefer_cls = None if self.det_class < 0 else self.det_class

        # FoundationPose iters
        self.est_refine_iter   = int(gp("est_refine_iter", 5))
        self.track_refine_iter = int(gp("track_refine_iter", 2))

        # IoU check / ROI
        self.roi_expand   = float(gp("roi_expand", 0.01))
        self.iou_stride   = int(gp("iou_stride", 3))
        self.iou_log      = bool(gp("iou_log", False))
        self.iou_thresh   = float(gp("iou_thresh", 0.25))
        self.iou_patience = int(gp("iou_patience", 3))

        # Windows
        self.show_depth_win = bool(gp("show_depth_window", False))
        self.show_rgb_win   = bool(gp("show_rgb_window", True))
        self.depth_win_name = gp("depth_win_name", "Depth")
        self.rgb_win_name   = gp("rgb_win_name", "RGB")
        self.depth_win_xy   = gp("depth_window_xy", [100,100])
        self.rgb_win_xy     = gp("rgb_window_xy", [100,500])
        self.max_depth_mm   = float(gp("max_depth_mm", 2000.0))
        self.colormap_id    = int(gp("colormap", int(cv2.COLORMAP_JET)))
        self.invert_colormap= bool(gp("invert_colormap", False))

        # 後處理（共用）
        self.pp_enable  = bool(gp("postproc/enable", True))
        self.pp_orient_center_tol_px = float(gp("postproc/orient_center_tol_px", 20.0))

        # 果串（BUNCH）後處理
        self.bunch_expect_orientation = gp("postproc/bunch/expect_orientation", "inverted").strip().lower()
        self.bunch_size_mode          = gp("postproc/bunch/size_mode", "bbox_mm").strip().lower()
        self.bunch_expect_bbox_w_mm   = float(gp("postproc/bunch/expect_bbox_w_mm", 115.0))
        self.bunch_expect_bbox_h_mm   = float(gp("postproc/bunch/expect_bbox_h_mm", 80.0))
        self.bunch_size_ratio_min     = float(gp("postproc/bunch/size_ratio_min", 0.6))
        self.bunch_expect_depth_m     = float(gp("postproc/bunch/expect_depth_m", 1.2))
        self.bunch_depth_tol_m        = float(gp("postproc/bunch/depth_tolerance_m", 0.25))

        # 葉莖（STEM）後處理
        self.stem_expect_orientation   = gp("postproc/stem/expect_orientation", "upright").strip().lower()
        self.stem_size_mode            = gp("postproc/stem/size_mode", "bbox_mm").strip().lower()
        self.stem_expect_bbox_w_mm     = float(gp("postproc/stem/expect_bbox_w_mm", 115.0))
        self.stem_expect_bbox_h_mm     = float(gp("postproc/stem/expect_bbox_h_mm", 80.0))
        self.stem_size_ratio_min       = float(gp("postproc/stem/size_ratio_min", 0.6))
        self.stem_expect_depth_m       = float(gp("postproc/stem/expect_depth_m", 1.2))
        self.stem_depth_tol_m          = float(gp("postproc/stem/depth_tolerance_m", 0.25))

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
        self.cls_stem  = int(gp("classes/stem",  1))

        # 遮蔽率策略
        self.policy_occ_hi   = float(gp("policy/occ_thresh_high", 0.60))
        self.policy_occ_lo   = float(gp("policy/occ_thresh_low",  0.40))
        self.policy_hi_pat   = int(gp("policy/high_patience", 3))
        self.policy_recheckN = int(gp("policy/recheck_interval", 10))
        self.policy_light_it = int(gp("policy/light_init_iters", 1))
        self.policy_allow_bunch_recheck = bool(gp("policy/allow_bunch_recheck_in_stem", False))

        # 分割後端
        self.seg_backend = gp("postproc/seg_backend", "sam").strip().lower()  # sam | bbox
        self.sam_model   = gp("postproc/sam_model", "vit_h").strip()
        self.sam_ckpt    = gp("postproc/sam_ckpt", gp("postproc/sam_ckpt_path", "/home/user/.cache/sam_vit_h.pth")).strip()
        self.occ_sam_warmup_n = int(gp("occ/sam_warmup_n", 8))

        # reinit debounce 秒數
        self.pp_retry_delay_sec  = float(gp("postproc/retry_delay_sec", 1.0))
        self.pp_on_fail          = gp("postproc/on_fail", "reinit").strip().lower()

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
        det_device = "cpu(ORT)"
        if sess is not None:
            try:
                provs = list(sess.get_providers())
            except Exception:
                provs = []
            if "CUDAExecutionProvider" in provs:
                rospy.loginfo(f"[YOLO Loader] ONNX providers={provs} (GPU OK)")
                return det, "cuda(ORT)"
            try:
                sess.set_providers(["CUDAExecutionProvider","CPUExecutionProvider"])
                provs2 = list(sess.get_providers())
                if "CUDAExecutionProvider" in provs2:
                    rospy.loginfo(f"[YOLO Loader] ONNX switched to providers={provs2} (GPU OK)")
                    return det, "cuda(ORT)"
                else:
                    rospy.logwarn(f"[YOLO Loader] ONNX CUDA provider not available, using CPU providers={provs2}")
            except Exception as ge:
                rospy.logwarn(f"[YOLO Loader] set_providers CUDA failed: {ge}. Use CPU.")
            try:
                sess.set_providers(["CPUExecutionProvider"])
            except Exception:
                pass
            rospy.loginfo("[YOLO Loader] ONNX on CPUExecutionProvider")
            return det, "cpu(ORT)"
        rospy.logwarn("[YOLO Loader] ONNX session not exposed; provider control skipped.")
        return det, det_device

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

    def pick_nearest(self, img_bgr, target_xyxy, cls_id):
        xyxy, sc, cl = self.yolo_det_all(self.detector, img_bgr, imgsz=self.det_imgsz, conf=self.det_conf)
        if len(xyxy)==0 or target_xyxy is None: return None, None
        idx = np.where(cl==cls_id)[0]
        if idx.size==0: return None, None
        tx = 0.5*(target_xyxy[0]+target_xyxy[2]); ty = 0.5*(target_xyxy[1]+target_xyxy[3])
        best, bid = 1e18, -1
        for i in idx:
            cx = 0.5*(xyxy[i][0]+xyxy[i][2]); cy = 0.5*(xyxy[i][1]+xyxy[i][3])
            d = (cx-tx)**2 + (cy-ty)**2
            if d<best: best, bid = d, i
        if bid<0: return None, None
        return xyxy[bid], sc[bid]

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
    
    def maybe_regrab_roi_by_iou(self, mode, center_pose):
        """
        針對當前 mode ('BUNCH' or 'STEM')：
        - 每 iou_stride 幀計算一次 IoU（對應類別）；
        - 若連續 iou_patience 次 < iou_thresh，則重新抓 ROI：
            - 有當幀 YOLO 框：直接以該框 expand 成 mask 後 register（re-init）
            - 沒框：清空該 mode 的 pose，交由下輪初始化
        回傳 True 表示本函式已做 re-init（呼叫端應該 continue 跳過本幀後續）。
        """
        if self.detector is None or center_pose is None:
            return False

        if (self.frame_count % max(1, self.iou_stride)) != 0:
            return False  # 非檢查幀

        if mode == "BUNCH":
            prefer_cls = self.cls_bunch
            bbox_mm    = self.bbox_bunch
            to_origin  = self.to_origin_bunch
            bad_count_attr = "iou_bad_count_bunch"
            last_upd_attr  = "last_iou_update_bunch"
        else:
            prefer_cls = self.cls_stem
            bbox_mm    = self.bbox_stem
            to_origin  = self.to_origin_stem
            bad_count_attr = "iou_bad_count_stem"
            last_upd_attr  = "last_iou_update_stem"

        # 投影需以幾何中心姿勢
        center_pose_cam = center_pose  # 已是幾何中心

        iou_val, best_xyxy = self.iou_vs_projection_for_class(
            self.color, self.K, center_pose_cam, bbox_mm,
            prefer_cls=prefer_cls, det_imgsz=self.det_imgsz, det_conf=self.det_conf
        )
        self.iou_val = iou_val  # 紀錄供外部查詢
        
        # 沒有新 IoU（None）→ 不計數
        if iou_val is None:
            return False

        # 紀錄最新 IoU 與更新幀
        setattr(self, last_upd_attr, self.frame_count)
        # 判斷高/低
        if iou_val < float(self.iou_thresh):
            setattr(self, bad_count_attr, getattr(self, bad_count_attr) + 1)
        else:
            setattr(self, bad_count_attr, 0)

        # 連續低於門檻 → re-grab ROI
        if getattr(self, bad_count_attr) >= int(self.iou_patience):
            setattr(self, bad_count_attr, 0)

            # 有當幀偵測框 → 直接以該框 ROI register
            if best_xyxy is not None:
                m = self.rect_to_mask(self.depth_m, self.clip_xyxy(best_xyxy, *self.rgb_size), expand=self.roi_expand)
                if mode == "BUNCH":
                    new_pose = self.est_bunch.register(K=self.K, rgb=self.color, depth=self.depth_m,
                                                       ob_mask=m, iteration=self.est_refine_iter)
                    if new_pose is not None:
                        self.pose_bunch = new_pose
                        # 重新暖機 SAM（BUNCH）
                        self.sam_warmup_left_bunch = int(self.occ_sam_warmup_n)
                else:
                    new_pose = self.est_stem.register(K=self.K, rgb=self.color, depth=self.depth_m,
                                                      ob_mask=m, iteration=self.est_refine_iter)
                    if new_pose is not None:
                        self.pose_stem = new_pose
                        self.sam_warmup_left_stem = int(self.occ_sam_warmup_n)
                # 畫字提示
                vis = self.color.copy()
                cv2.putText(vis, f"Re-init ROI ({mode}, low IoU)", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                self.pump_windows(vis if self.show_rgb_win else None,
                                  self.depth_vis if (self.show_depth_win and hasattr(self, "depth_vis")) else None)
                # 本幀已處理 re-init → 呼叫端應跳過後續
                return True

            # 沒框 → 清空該 mode 的 pose，交由下輪初始化
            if mode == "BUNCH":
                self.pose_bunch = None
            else:
                self.pose_stem = None

            vis = self.color.copy()
            cv2.putText(vis, f"Re-init needed ({mode}, low IoU, no det)", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
            self.pump_windows(vis if self.show_rgb_win else None,
                              self.depth_vis if (self.show_depth_win and hasattr(self, "depth_vis")) else None)
            return True

        return False

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

    def _dec_warmup_left(self, which: str):
        if which.lower()=="bunch":
            if self.sam_warmup_left_bunch > 0: self.sam_warmup_left_bunch -= 1
        else:
            if self.sam_warmup_left_stem  > 0: self.sam_warmup_left_stem  -= 1

    def compute_occ_and_iou(self, which: str):
        """
        回傳 (occ, iou, used_sam)
        - 暖機期：用 GTvsSAM 算 occ，另外也算 iou 供顯示；used_sam=True
        - 暖機結束：只算 iou，occ = 1 - iou；used_sam=False
        """
        occ = 1.0
        if self.color is None or self.K is None:
            return 1.0, 0.0, False

        which_l = which.lower()
        if which_l == "bunch":
            pose = self.pose_bunch
            bbox = self.bbox_bunch
            prefer_cls = self.cls_bunch
        else:
            pose = self.pose_stem
            bbox = self.bbox_stem
            prefer_cls = self.cls_stem

        if pose is None:
            return 1.0, 0.0, False

        # 幾何中心姿勢
        center_pose = pose @ np.linalg.inv(self.to_origin_bunch if which_l=="bunch" else self.to_origin_stem)

        # 先計算 IoU（即使在暖機，用於 bar 顯示 & debug）
        iou_val, best_xyxy = self.iou_vs_projection_for_class(
            self.color, self.K, center_pose, bbox,
            prefer_cls=prefer_cls, det_imgsz=self.det_imgsz, det_conf=self.det_conf
        )
        # iou_val 可能為 None（例如投影失效），畫面 bar 時我們會處理預設值

        # 暖機：用 SAM + GT 算 occ
        if self._get_warmup_left(which_l) > 0:
            self._dec_warmup_left(which_l)  # 暖機計數 -1
            if self._get_warmup_left(which_l) == 0:
                rospy.loginfo(f"[{which_l.upper()}] SAM warmup finished, switch to IoU-only mode")
                return 1.0, (0.0 if iou_val is None else float(iou_val)), False

            # 否則繼續 SAM 計算
            prompt_box = self.pose_2d_box_xyxy(which_l)
            seg = self.seg_mask_from_sam(self.color, prompt_box)
            occ = self.occ_from_gt_mesh_vs_sam(which_l, seg.astype(bool))

            # Debug 圖（SAM）
            try:
                if getattr(self, "iou_log", False):
                    self.save_occ_sam_debug(
                        self.color, seg.astype(bool) if 'seg' in locals() and seg is not None else None,
                        self.K, center_pose, bbox, occ, best_xyxy, tag=which_l.upper()
                    )
            except Exception:
                pass

            return occ, (0.0 if iou_val is None else float(iou_val)), True

        # 非暖機：只用 IoU
        if iou_val is None:
            # 無法計算 IoU → 視為高遮蔽
            occ = 1.0
            iou_show = 0.0
        else:
            occ = 1.0 - float(iou_val)
            iou_show = float(iou_val)

        # Debug 圖（IoU）
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
        以 6D pose 將 mesh 投影為 2D 二值 mask（不做 z-buffer，但會只繪製 Z>0 的三角形）。
        這樣可避免索引錯亂與 inf/NaN 造成的大量漏畫。
        """
        H, W = img_shape[:2]

        V = mesh.vertices.astype(np.float64)             # (N,3)
        Vh = np.concatenate([V, np.ones((V.shape[0],1))], axis=1)  # (N,4)
        Pc = (pose_center_in_cam @ Vh.T).T               # (N,4)

        Z = Pc[:, 2]
        # 投影：僅對 Z>0 的頂點計算 u,v；其餘先填 NaN
        u = np.full((V.shape[0],), np.nan, dtype=np.float64)
        v = np.full((V.shape[0],), np.nan, dtype=np.float64)
        valid = Z > 1e-6
        u[valid] = K[0,0] * (Pc[valid, 0] / Z[valid]) + K[0,2]
        v[valid] = K[1,1] * (Pc[valid, 1] / Z[valid]) + K[1,2]

        mask = np.zeros((H, W), np.uint8)
        F = mesh.faces  # (M,3), 使用原索引，不壓縮

        # 小幅邊界裕度，避免邊界三角形被錯誤判定在畫面外
        MARGIN = 10

        for f in F:
            i0, i1, i2 = int(f[0]), int(f[1]), int(f[2])

            # 只畫三個頂點都在相機前方的三角形
            if (not valid[i0]) or (not valid[i1]) or (not valid[i2]):
                continue

            pts = np.array([
                [u[i0], v[i0]],
                [u[i1], v[i1]],
                [u[i2], v[i2]],
            ], dtype=np.float64)

            # 有 NaN 的話跳過
            if not np.isfinite(pts).all():
                continue

            # 快速視域測試（全部在螢幕外才略過）
            if ((pts[:,0] < -MARGIN).all() or (pts[:,0] > W + MARGIN).all() or
                (pts[:,1] < -MARGIN).all() or (pts[:,1] > H + MARGIN).all()):
                continue

            # 轉為 int 並裁切寬容邊界，避免 OpenCV 溢位
            pts_i = pts.copy()
            pts_i[:,0] = np.clip(pts_i[:,0], -2048, W + 2048)
            pts_i[:,1] = np.clip(pts_i[:,1], -2048, H + 2048)
            pts_i = pts_i.astype(np.int32)

            cv2.fillConvexPoly(mask, pts_i, 255, lineType=cv2.LINE_AA)

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

    def seg_mask_from_posebox(self, bgr, which: str):
        """用 6D pose 投影框作為 SAM/bbox 的提示框，產生 mask。"""
        box = self.pose_2d_box_xyxy(which)
        if box is None: 
            return None
        return self.seg_mask_from_sam(bgr, box)  # 內部會 fallback 成矩形 mask

    def render_bbox_mask_proxy(self, K, center_pose, bbox_minmax, img_shape):
        """用 3D bbox 投影外框矩形近似遮罩（快速、不渲染mesh）"""
        est_xyxy = self.project_3d_bbox_xyxy(K, center_pose, bbox_minmax, img_shape)
        if est_xyxy is None:
            return np.zeros(img_shape[:2], dtype=bool)
        return self.rect_to_mask(np.zeros(img_shape[:2], np.uint8), est_xyxy, expand=0.0)
    
    def seg_mask_from_sam(self, bgr, xyxy):
        """用 SAM 基於 bbox 產生二值 mask（bool）。若不可用則回 bbox mask。"""
        H, W = bgr.shape[:2]
        x1,y1,x2,y2 = self.clip_xyxy(xyxy, W, H).astype(np.int32)
        if self.sam_predictor is None:
            m = np.zeros((H,W), dtype=bool); m[y1:y2, x1:x2] = True
            return m
        self.sam_predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        box = np.array([x1,y1,x2,y2], dtype=np.int32)
        masks, scores, _ = self.sam_predictor.predict(point_coords=None, point_labels=None, box=box[None,:], multimask_output=True)
        if masks is None or len(masks)==0:
            m = np.zeros((H,W), dtype=bool); m[y1:y2, x1:x2] = True
            return m
        return masks[int(np.argmax(scores))].astype(bool)

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

    def occ_score_stem(self, color_bgr, K, pose_obj_in_cam):
        prompt_box = self.pose_2d_box_xyxy("stem")
        if prompt_box is None:
            return 1.0
        yolo_xyxy, _ = self.pick_top1(color_bgr, self.cls_stem)  # 備用
        try:
            seg = self.seg_mask_from_sam(color_bgr, prompt_box)
        except Exception:
            seg = self.rect_to_mask(self.depth_m, self.clip_xyxy(yolo_xyxy, *self.rgb_size), expand=0.0) if yolo_xyxy is not None else None
            if seg is None:
                return 1.0
        seg_bool = seg.astype(bool)
        return self.occ_from_gt_mesh_vs_sam("stem", seg_bool)

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
        """同時送 TF + 對應 Pose topic"""
        if which.lower() == "bunch":
            child = self.bunch_name
            pub   = self.pose_pub_bunch
        else:
            child = self.stem_name
            pub   = self.pose_pub_stem

        # TF
        t = (float(T[0,3]), float(T[1,3]), float(T[2,3]))
        qx,qy,qz,qw = tf.transformations.quaternion_from_matrix(T)
        self.tf_broadcaster.sendTransform(t, (qx,qy,qz,qw), rospy.Time.now(), child, parent)

        # Pose
        msg = Pose()
        msg.position.x, msg.position.y, msg.position.z = t
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = (qx,qy,qz,qw)
        pub.publish(msg)

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
            depth_m = d if np.nanmax(d) <= 10.0 else d * 0.001
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
        elif msg.layer == 1.0: self.det_select_mode = "bottom"
        elif msg.layer == 2.0: self.det_select_mode = "top"
        else: self.det_select_mode = "score"

    def confidence_publish(self, which: str, score: float, detection: bool):
        """which: 'bunch' or 'stem'"""
        conf_msg = Confidence()
        conf_msg.stamp = rospy.Time.now()
        if which.lower() == "bunch":
            conf_msg.frame_id = self.bunch_name
            conf_msg.object_IoU = float(score)
            conf_msg.object_detection = detection
            self.conf_pub_bunch.publish(conf_msg)
        else:
            conf_msg.frame_id = self.stem_name
            conf_msg.object_IoU = float(score)
            conf_msg.object_detection = detection
            self.conf_pub_stem.publish(conf_msg)

    # =========================
    # 主循環
    # =========================
    def spin(self):
        self.frame_count = 0

        while not rospy.is_shutdown():
            if not (self.got_depth and self.got_rgb and self.K is not None and hasattr(self, "depth_m")):
                self.pump_windows(self.color if self.got_rgb else None,
                                  self.depth_vis if self.got_depth else None)
                rospy.sleep(0.01); continue
            # 若收到採收完成訊號，重設為果串模式
            if not self.ready_received.detection_allowed:
                self.mode = "BUNCH"
                self.pose_stem = None
                self._hi_cnt = self._lo_cnt = 0
                self.confidence_publish("bunch", 0.0, False)
                self.confidence_publish("stem",  0.0, False)

            if (self.init_mode == 'yolo' and self.yolo_start_mode == 'wait'
                and not self.ready_received.detection_allowed):
                self.pose_bunch = None; self.pose_stem = None
                self.iou_bad_count = 0; self.iou_val = None
                vis_rgb = self.color.copy()
                cv2.putText(vis_rgb, "DETECTION DISABLED", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                self.pump_windows(vis_rgb if self.show_rgb_win else None,
                                  self.depth_vis if (self.show_depth_win and hasattr(self, "depth_vis")) else None)
                self.confidence_publish("bunch", 0.0, False)
                self.confidence_publish("stem",  0.0, False)
                continue

            # 後處理 debounce：倒數中，暫停估測
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
                self.confidence_publish("bunch", 0.0, False)
                self.confidence_publish("stem",  0.0, False)
                if elapsed >= float(self.pp_retry_delay_sec):
                    rospy.logwarn("[POST] Debounce timeout reached. Reinit now.")
                    self._post_pending = False; self._post_fail_time = None
                    self.pose_bunch = None; self.pose_stem = None
                rospy.sleep(0.01)
                continue

            self.frame_count += 1

            # 先抓果串 Top-1（供兩種模式使用）
            bunch_xyxy, bunch_conf = self.pick_top1(self.color, self.cls_bunch)
            vis_bgr = self.color.copy()

            if self.mode == "BUNCH":
                # 初始化
                if self.pose_bunch is None:
                    if bunch_xyxy is None:
                        self.pump_windows(vis_bgr, self.depth_vis if self.got_depth else None)
                        self.confidence_publish("bunch", 0.0, False)
                        self.confidence_publish("stem",  0.0, False)
                        continue
                    m = self.rect_to_mask(self.depth_m, self.clip_xyxy(bunch_xyxy, *self.rgb_size), expand=self.roi_expand)
                    self.pose_bunch = self.est_bunch.register(K=self.K, rgb=self.color, depth=self.depth_m,
                                                              ob_mask=m, iteration=self.est_refine_iter)
                    if self.pose_bunch is None:
                        self.confidence_publish("bunch", 0.0, False)
                        self.confidence_publish("stem",  0.0, False)
                        continue
                    # 初始化成功後，開啟 SAM 暖機期
                    self.sam_warmup_left_bunch = int(self.occ_sam_warmup_n)

                # 追蹤果串
                self.pose_bunch = self.est_bunch.track_one(rgb=self.color, depth=self.depth_m,
                                                           K=self.K, iteration=self.track_refine_iter)

                # 遮蔽率：暖機→SAM；之後→IoU
                occ, iou_for_bar, used_sam = self.compute_occ_and_iou("bunch")
                # if self.iou_log:
                rospy.loginfo_throttle(1.0, f"[BUNCH] occ={occ:.2f}  iou={iou_for_bar:.2f}  warmup={'Y' if used_sam else 'N'}")

                # 門檻與遲滯
                if occ >= self.policy_occ_hi:
                    self._hi_cnt += 1; self._lo_cnt = 0
                elif occ <= self.policy_occ_lo:
                    self._lo_cnt += 1; self._hi_cnt = 0
                else:
                    self._hi_cnt = self._lo_cnt = 0

                # 遮蔽連續高 → 切 STEM
                if self._hi_cnt >= self.policy_hi_pat:
                    stem_xyxy, _ = self.pick_nearest(self.color, bunch_xyxy, self.cls_stem)
                    if stem_xyxy is not None:
                        m = self.rect_to_mask(self.depth_m, self.clip_xyxy(stem_xyxy, *self.rgb_size), expand=self.roi_expand)
                        self.pose_stem = self.est_stem.register(K=self.K, rgb=self.color, depth=self.depth_m,
                                                                ob_mask=m, iteration=self.est_refine_iter)
                        if self.pose_stem is not None:
                            self.mode = "STEM"
                            self._hi_cnt = self._lo_cnt = 0
                            self.confidence_publish("bunch", 0.0, False)
                            self.confidence_publish("stem",  0.0, False)
                            self.pump_windows(vis_bgr if self.show_rgb_win else None,
                                              self.depth_vis if (self.show_depth_win and hasattr(self, "depth_vis")) else None)
                            continue

                # 後處理（方向/尺寸）
                ok_to_publish, pending = self.postprocess_and_maybe_reinit(self.pose_bunch, which="bunch")
                if pending:
                    cv2.putText(vis_bgr, "Post-check pending...", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                    self.pump_windows(vis_bgr if self.show_rgb_win else None,
                                      self.depth_vis if (self.show_depth_win and hasattr(self, "depth_vis")) else None)
                    self.confidence_publish("bunch", 0.0, False)
                    self.confidence_publish("stem",  0.0, False)
                    continue

                # === IoU 檢查：連續低於門檻 → 重新抓 ROI ===
                if self.pose_bunch is not None:
                    center_pose = self.pose_bunch @ np.linalg.inv(self.to_origin_bunch)
                    if self.maybe_regrab_roi_by_iou("BUNCH", center_pose):
                        # 本幀已重新抓 ROI 或清空等待 → 跳過後續
                        self.confidence_publish("bunch", 0.0, False)
                        self.confidence_publish("stem",  0.0, False)
                        continue

                # 可視化 + TF（果串）
                if ok_to_publish and self.pose_bunch is not None:
                    center_pose = self.pose_bunch @ np.linalg.inv(self.to_origin_bunch)
                    vis = draw_posed_3d_box(self.K, img=cv2.cvtColor(self.color, cv2.COLOR_BGR2RGB),
                                            ob_in_cam=center_pose, bbox=self.bbox_bunch)
                    vis = draw_xyz_axis(vis, ob_in_cam=self.pose_bunch, scale=0.05, K=self.K,
                                        thickness=3, transparency=0, is_input_rgb=True)
                    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                    parent_frame = self.camera_tf if self.camera_tf else "camera_color_optical_frame"
                    self.broadcast_transform_and_pose(self.pose_bunch, "bunch", parent_frame)
                    self.confidence_publish("bunch", 1.0, True)
                    # self.save_pose_bbox_debug(vis_bgr, tag="BUNCH")

            else:  # STEM 模式
                # 初始化
                if self.pose_stem is None:
                    if bunch_xyxy is None:
                        self.pump_windows(vis_bgr, self.depth_vis if self.got_depth else None)
                        self.confidence_publish("bunch", 0.0, False)
                        self.confidence_publish("stem",  0.0, False)
                        continue
                    stem_xyxy, _ = self.pick_nearest(self.color, bunch_xyxy, self.cls_stem)
                    if stem_xyxy is None:
                        self.pump_windows(vis_bgr, self.depth_vis if self.got_depth else None)
                        self.confidence_publish("bunch", 0.0, False)
                        self.confidence_publish("stem",  0.0, False)
                        continue
                    m = self.rect_to_mask(self.depth_m, self.clip_xyxy(stem_xyxy, *self.rgb_size), expand=self.roi_expand)
                    self.pose_stem = self.est_stem.register(K=self.K, rgb=self.color, depth=self.depth_m,
                                                            ob_mask=m, iteration=self.est_refine_iter)
                    if self.pose_stem is None:
                        self.confidence_publish("bunch", 0.0, False)
                        self.confidence_publish("stem",  0.0, False)
                        continue
                    # 切 STEM 後，回到 BUNCH 的 recheck 依然會走 IoU（不使用 SAM）

                # 追蹤葉莖
                self.pose_stem = self.est_stem.track_one(rgb=self.color, depth=self.depth_m,
                                                         K=self.K, iteration=self.track_refine_iter)

                # STEM 追蹤完成後加入遮蔽率評估
                occ, iou_for_bar, used_sam = self.compute_occ_and_iou("stem")
                if self.iou_log:
                    rospy.loginfo_throttle(1.0, f"[STEM] occ={occ:.2f}  iou={iou_for_bar:.2f}  warmup={'Y' if used_sam else 'N'}")

                if self.pose_stem is None:
                    stem_xyxy, _ = self.pick_nearest(self.color, bunch_xyxy, self.cls_stem)
                    if stem_xyxy is not None:
                        m = self.rect_to_mask(self.depth_m, self.clip_xyxy(stem_xyxy, *self.rgb_size), expand=self.roi_expand)
                        self.pose_stem = self.est_stem.register(K=self.K, rgb=self.color, depth=self.depth_m,
                                                                ob_mask=m, iteration=self.est_refine_iter)
                    if self.pose_stem is None:
                        # 還是沒抓到就下個迭代再試，維持 STEM 與 0 conf
                        self.confidence_publish("stem", 0.0, False)
                        self.pump_windows(vis_bgr if self.show_rgb_win else None,
                                        self.depth_vis if (self.show_depth_win and hasattr(self, "depth_vis")) else None)
                        continue
                # 週期性輕量檢查果串是否已不遮蔽 → 回 BUNCH（用 IoU）
                if (self.policy_allow_bunch_recheck and self.ready_received.detection_allowed and (self.frame_count % max(1, self.policy_recheckN)) == 0 and bunch_xyxy is not None):
                    mm = self.rect_to_mask(self.depth_m, self.clip_xyxy(bunch_xyxy, *self.rgb_size), expand=self.roi_expand)
                    pose_test = self.est_bunch.register(K=self.K, rgb=self.color, depth=self.depth_m,
                                                        ob_mask=mm, iteration=max(1, self.policy_light_it))
                    if pose_test is not None:
                        try:
                            center_pose = pose_test @ np.linalg.inv(self.to_origin_bunch)
                            occ = self.occ_from_iou(bunch_xyxy, self.K, center_pose, self.bbox_bunch, self.color.shape, self.color)
                            rospy.loginfo(f"[RECHECK] occ={occ:.2f}")
                            if occ <= self.policy_occ_lo:
                                self.mode = "BUNCH"
                                self.pose_bunch = pose_test
                                self._hi_cnt = self._lo_cnt = 0
                                # 回到 BUNCH 時重新開啟一小段暖機（可選：不開亦可）
                                self.sam_warmup_left_bunch = int(self.occ_sam_warmup_n)
                        except Exception as e:
                            rospy.logwarn_throttle(1.0, f"[RECHECK] occ failed: {e}")

                # 後處理（方向/尺寸）
                ok_to_publish, pending = self.postprocess_and_maybe_reinit(self.pose_stem, which="stem")
                if pending:
                    cv2.putText(vis_bgr, "Post-check pending...", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                    self.pump_windows(vis_bgr if self.show_rgb_win else None,
                                      self.depth_vis if (self.show_depth_win and hasattr(self, "depth_vis")) else None)
                    self.confidence_publish("bunch", 0.0, False)
                    self.confidence_publish("stem",  0.0, False)
                    continue

                # === IoU 檢查：連續低於門檻 → 重新抓 ROI ===
                if self.pose_stem is not None:
                    center_pose = self.pose_stem @ np.linalg.inv(self.to_origin_stem)
                    if self.maybe_regrab_roi_by_iou("STEM", center_pose):
                        self.confidence_publish("bunch", 0.0, False)
                        self.confidence_publish("stem",  0.0, False)
                        continue

                # 可視化 + TF（葉莖）
                if ok_to_publish and self.pose_stem is not None:
                    center_pose = self.pose_stem @ np.linalg.inv(self.to_origin_stem)
                    vis = draw_posed_3d_box(self.K, img=cv2.cvtColor(self.color, cv2.COLOR_BGR2RGB),
                                            ob_in_cam=center_pose, bbox=self.bbox_stem)
                    vis = draw_xyz_axis(vis, ob_in_cam=self.pose_stem, scale=0.05, K=self.K,
                                        thickness=3, transparency=0, is_input_rgb=True)
                    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                    parent_frame = self.camera_tf if self.camera_tf else "camera_color_optical_frame"
                    self.broadcast_transform_and_pose(self.pose_stem, "stem", parent_frame)
                    self.confidence_publish("stem",  1.0, True)
                    # self.save_pose_bbox_debug(vis_bgr, tag="STEM")
            # 只在 self.iou_val 有值時才畫，或給預設值
            bar_w, bar_h, margin = 220, 18, 10
            if hasattr(self, "draw_conf_bar"):
                # iou_for_bar 來自 compute_occ_and_iou()
                iou_show = float(iou_for_bar) if 'iou_for_bar' in locals() else 0.0
                self.draw_conf_bar(
                    vis_bgr, {"IoU": iou_show},
                    label="IoU",
                    origin=(10, vis_bgr.shape[0] - margin - bar_h),
                    size=(bar_w, bar_h),
                    max_val=1.0
                )
            # 顯示
            self.pump_windows(vis_bgr if (self.show_rgb_win and self.color is not None) else None,
                              self.depth_vis if (self.show_depth_win and self.got_depth and self.depth_vis is not None) else None)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    rospy.init_node("pipeline_tracker", anonymous=False)
    node = FoundationPosePipelineTracker()
    node.spin()
