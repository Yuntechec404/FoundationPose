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

# ==========================================
# 專案路徑設定
# ==========================================
EXTRA_PATHS = [
    "/home/user/anaconda3/envs/foundationpose/lib/python3.8/site-packages",
    "/home/user/FoundationPose",
]
for p in EXTRA_PATHS:
    if p not in sys.path:
        sys.path.append(p)

# ==========================================
# 載入 Cutie VOS (影片物件分割) 模組
# ==========================================
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

# ==========================================
# 載入 FoundationPose 與 AI 基石模型
# ==========================================
from estimater import FoundationPose, draw_posed_3d_box, draw_xyz_axis, ScorePredictor, PoseRefinePredictor, dr
from ultralytics import YOLO, SAM, FastSAM
from ultralytics.models.sam import SAM2DynamicInteractivePredictor

selecting_bbox = False
box_points = []

class FoundationPosePipelineTracker:
    def __init__(self):
        self.init_parameter()
        
        # 建立 Debug 影像存檔目錄
        if self.iou_log:
            self._setup_run_debug_dir()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # ------------------------------------------
        # 系統基本狀態初始化
        # ------------------------------------------
        self.bridge = CvBridge()
        self.color = None
        self.depth = None
        self.depth_encoding = None
        self.got_depth = False
        self.got_rgb = False
        self.depth_size = (0, 0)
        self.rgb_size = (0, 0)
        
        # 紀錄果串(bunch)與葉莖(stem)的 6D 姿態矩陣
        self.pose_bunch = None
        self.pose_stem  = None
        self.mask = None
        
        # 計數器與旗標
        self.frame_count = 0
        self.iou_bad_count = 0
        self.iou_val = None
        self.iou_bad_count_bunch = 0
        self.last_iou_update_bunch = -1
        self.K = None # 相機內參矩陣
        self._last_yolo_text = ""
        
        self.ready_received = Detection()
        self.ready_received.detection_allowed = False
        self.det_select_mode_current = getattr(self, "det_select_mode", "score")
        self._force_bunch_detect = False   # 重新初始化後強制進行果串檢測
        self._stem_lock = False
        self.target_stem_id = -1
        self.last_bunch_3d_pos = None
        self._pause_hold = False
        self._last_allowed = False

        # ------------------------------------------
        # Cutie VOS 初始化
        # ------------------------------------------
        if CUTIE_AVAILABLE:
            rospy.loginfo("[Cutie] Loading weights...")
            self.cutie_net = get_default_model()
            self.cutie_net.eval().cuda()
            self.cutie_processor = InferenceCore(self.cutie_net, cfg=self.cutie_net.cfg)
            self.cutie_processor.max_internal_size = 640 # 限制解析度以防止記憶體溢出 (OOM)

        # 狀態機：分別紀錄 Bunch 與 Stem 是在「巡航尋找(CRUISING)」還是「伺服追蹤(SERVOING)」
        self.bunch_cutie_state = "CRUISING"
        self.stem_cutie_state = "CRUISING"
        
        # ------------------------------------------
        # 葉莖 (Stem) 持續追蹤狀態 (用於運動預測)
        # ------------------------------------------
        self.last_stem_3d_pos = None  # 紀錄最後一次算出的葉莖 3D 相機座標
        self.last_stem_time = None    # 紀錄最後一次更新的時間，計算 dt 用
        self.stem_lost_timeout = rospy.Duration(self.stem_lost) 
        
        # YOLO 延時防呆狀態 (防止單幀雜訊造成錯誤鎖定)
        self._yolo_delay_left_bunch = 0
        self._yolo_delay_bbox_bunch = None
        self.seg_warmup_left_bunch = 0 # 暖機幀數倒數

        # GUI 視窗狀態
        self._rgb_win_created = False
        self._depth_win_created = False
        self._rgb_win_sized = False
        self._depth_win_sized = False
        self._rgb_initial_size = (900, 720)
        self._depth_initial_size = (900, 720)

        # 後處理防呆延遲狀態
        self._post_pending = False
        self._post_fail_time = None

        # ------------------------------------------
        # ROS 節點與 Topic 訂閱設定
        # ------------------------------------------
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.imageCallback, queue_size=1)
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self.depthCallback, queue_size=1)
        self.info_sub  = rospy.Subscriber(self.info_topic,  CameraInfo, self.infoCallback, queue_size=1)
        self.cmd_vel_sub = rospy.Subscriber(self.cmd_vel_topic, Twist, self.cmd_vel_callback, queue_size=1)
        self.current_twist = Twist() # 車體速度

        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()

        # 發布追蹤信心度與結果
        self.conf_pub = rospy.Publisher("FFB_state", Confidence, queue_size=1, latch=True)
        self.harvest_done_sub = rospy.Subscriber("FFB_state" + "_harvest_done", Bool, self.harvestDoneCallback, queue_size=1)
        if self.yolo_start_mode == "wait":
            self._ready_sub = rospy.Subscriber("FFB_state" + "_detection", Detection, self.detectionCallback, queue_size=1)

        self.window_create()

        # ------------------------------------------
        # FoundationPose 模型初始化
        # ------------------------------------------
        os.makedirs(self.debug_dir, exist_ok=True)
        self.mesh_bunch = trimesh.load(self.mesh_file)
        self.to_origin_bunch, self.extents_bunch = trimesh.bounds.oriented_bounds(self.mesh_bunch) 
        self.gt_to_origin_bunch, self.gt_extents_bunch = np.eye(4),self.mesh_bunch.extents
        self.bbox_bunch = np.stack([-self.extents_bunch/2, self.extents_bunch/2], axis=0).reshape(2, 3)

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()

        self.est_bunch = FoundationPose(model_pts=self.mesh_bunch.vertices,
                                        model_normals=self.mesh_bunch.vertex_normals,
                                        mesh=self.mesh_bunch, scorer=self.scorer, refiner=self.refiner,
                                        debug_dir=self.debug_dir, debug=0, glctx=self.glctx)
        
        # ------------------------------------------
        # YOLOv11 (物件偵測) 初始化
        # ------------------------------------------
        self.detector, self.det_device = self.load_detector(self.det_model)
        # 強制 ONNXRuntime 建立 session (Warmup)
        try:
            dummy = np.zeros((self.det_imgsz, self.det_imgsz, 3), np.uint8)
            _ = self.detector.predict(source=dummy, imgsz=self.det_imgsz, conf=0.01, device=self.det_device, verbose=False)
        except Exception as e:
            rospy.logwarn(f"[YOLO warmup] failed: {e}")

        is_gpu, yolo_desc = self.yolo_uses_gpu(self.detector)
        rospy.loginfo(f"[YOLO] predict device hint: {self.det_device}")
        rospy.loginfo(f"[YOLO] GPU enabled: {is_gpu}  ({yolo_desc})")
        rospy.loginfo("Detector initialization done")
        
        # ==========================================
        # [核心架構] 動態載入分割後端模型 (SAM / SAM2 / FastSAM / YOLO-seg)
        # 依據 launch 檔中設定的 pipeline_mode (1~4) 決定載入哪些模型以節省記憶體
        # ==========================================
        self.seg_detector = None
        # 如果果串模式設為 4，代表使用輕量級的 YOLO-seg
        if self.bunch_pipeline_mode == 4:
            try:
                self.seg_detector = YOLO(self.seg_model, task="segment")
                dummy = np.zeros((self.seg_imgsz, self.seg_imgsz, 3), np.uint8)
                _ = self.seg_detector.predict(source=dummy, imgsz=self.seg_imgsz, conf=0.01, device=self.det_device, verbose=False)
                rospy.loginfo("[YOLO-SEG] warmup done")
            except Exception as e:
                rospy.logwarn(f"[YOLO-SEG] init failed: {e}")

        # 存放 Prompt 基礎模型 (透過 YOLO 提供的 BBox 作為提示進行分割)
        self.prompt_models = {}
        def load_sam_family(mode_id, ckpt):
            if mode_id not in self.prompt_models:
                try:
                    # mode 3: FastSAM, 其他為 SAM/SAM2
                    model = FastSAM(ckpt) if mode_id == 3 else SAM(ckpt)
                    dummy = np.zeros((self.det_imgsz, self.det_imgsz, 3), np.uint8)
                    model(dummy, bboxes=[[10,10,100,100]], imgsz=self.det_imgsz, device=self.det_device, verbose=False)
                    self.prompt_models[mode_id] = model
                    name = "sam" if mode_id == 1 else "sam2" if mode_id == 2 else "fastsam"
                    rospy.loginfo(f"[{name.upper()}] loaded: {ckpt}")
                except Exception as e:
                    rospy.logwarn(f"[SAM_FAMILY_{mode_id}] init failed: {e}")

        # 根據 bunch_pipeline_mode 或 stem_pipeline_mode 的需求，動態載入對應的模型
        if 1 in [self.bunch_pipeline_mode, self.stem_pipeline_mode]: load_sam_family(1, self.sam_ckpt)
        if 2 in [self.bunch_pipeline_mode, self.stem_pipeline_mode]: load_sam_family(2, self.sam2_ckpt)
        if 3 in [self.bunch_pipeline_mode, self.stem_pipeline_mode]: load_sam_family(3, self.fastsam_ckpt)

        # 針對 Stem 模式 4：不使用 Cutie，純依賴 SAM2 Dynamic Interactive Predictor 進行影像追蹤
        self.sam2_tracker = None
        if self.stem_pipeline_mode == 4:
            rospy.loginfo(f"載入 SAM2 Dynamic Interactive Predictor: {self.sam2_ckpt}")
            overrides = dict(conf=0.01, task="segment", mode="predict", imgsz=1024, model=self.sam2_ckpt, save=False)
            try:
                self.sam2_tracker = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=3)
            except Exception as e:
                rospy.logwarn(f"[SAM2_TRACKER] init failed: {e}")

        # 總控狀態機模式
        self.mode = "bunch"     # 初始尋找果串 (bunch)，後續依遮蔽率切換為葉莖 (stem)
        self._hi_cnt = 0
        self._registering_until = 0
        self._reinit_until = 0

    # ---------------------------
    # 從 ROS Parameter Server 讀取參數
    # ---------------------------
    def init_parameter(self):
        ns = rospy.get_name()
        gp = lambda k, d: rospy.get_param(ns + "/" + k, d)

        self.image_topic = gp("image_topic", "/camera/color/image_raw")
        self.info_topic = gp("info_topic",  "/camera/color/camera_info")
        self.depth_topic = gp("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.depth_info_topic = gp("depth_info_topic", "")
        self.camera_tf = gp("camera_tf", "")
        self.bunch_name = gp("bunch_name", "oilpalm")
        self.stem_name = gp("stem_name", "stem")

        self.mesh_file = gp("mesh_file", "")
        self.det_model = gp("det_model", "yolov11n.onnx")
        self.yolo_start_mode = gp("yolo_start_mode", "immediate").strip().lower()
        self.debug_root = gp("debug_root", gp("debug_dir", "/tmp/fp_debug"))
        self.debug_dir = self.debug_root

        # YOLO 偵測參數
        self.det_conf = float(gp("det_conf", 0.25))
        self.det_class = int(gp("det_class", -1))
        self.det_imgsz = int(gp("det_imgsz", 640))
        self.prefer_cls = None if self.det_class < 0 else self.det_class
        self.det_select_mode = gp("det_select_mode", "score").strip().lower()

        # FoundationPose 最佳化迭代次數
        self.est_refine_iter = int(gp("est_refine_iter", 5))
        self.track_refine_iter = int(gp("track_refine_iter", 2))

        # IoU 檢查閾值與容忍度
        self.roi_expand = float(gp("roi_expand", 0.01))
        self.iou_stride = int(gp("iou_stride", 3))
        self.iou_log = bool(gp("iou_log", False))
        self.iou_thresh = float(gp("iou_thresh", 0.25))
        self.iou_patience = int(gp("iou_patience", 3))

        # 視窗顯示設定
        self.show_depth_win = bool(gp("show_depth_window", False))
        self.show_rgb_win = bool(gp("show_rgb_window", True))
        self.depth_win_name = gp("depth_win_name", "Depth")
        self.rgb_win_name = gp("rgb_win_name", "RGB")
        self.depth_win_xy = gp("depth_window_xy", [100,100])
        self.rgb_win_xy = gp("rgb_window_xy", [100,500])
        self.max_depth_mm = float(gp("max_depth_mm", 2000.0))
        self.colormap_id = int(gp("colormap", int(cv2.COLORMAP_JET)))
        self.invert_colormap= bool(gp("invert_colormap", False))

        self.pp_enable = bool(gp("postproc/enable", True))
        self.pp_orient_center_tol_px = float(gp("postproc/orient_center_tol_px", 20.0))

        # ------------------------------------------
        # [核心設定] 果串 (BUNCH) 追蹤管線設定
        # 1: YOLO+SAM, 2: YOLO+SAM2, 3: YOLO+FastSAM, 4: YOLO-seg
        # ------------------------------------------
        self.bunch_pipeline_mode = int(gp("postproc/bunch/pipeline_mode", 4)) 
        self.bunch_expect_orientation = gp("postproc/bunch/expect_orientation", "inverted").strip().lower()
        self.bunch_size_mode = gp("postproc/bunch/size_mode", "bbox_mm").strip().lower()
        self.bunch_expect_bbox_w_mm = float(gp("postproc/bunch/expect_bbox_w_mm", 115.0))
        self.bunch_expect_bbox_h_mm = float(gp("postproc/bunch/expect_bbox_h_mm", 80.0))
        self.bunch_size_ratio_min = float(gp("postproc/bunch/size_ratio_min", 0.6))
        self.bunch_expect_depth_m = float(gp("postproc/bunch/expect_depth_m", 1.2))
        self.bunch_depth_tol_m = float(gp("postproc/bunch/depth_tolerance_m", 0.25))

        # ------------------------------------------
        # [核心設定] 葉莖 (STEM) 追蹤管線設定
        # 1: YOLO+SAM+Cutie, 2: YOLO+SAM2+Cutie, 3: YOLO+FastSAM+Cutie, 4: YOLO+SAM2純追蹤(無Cutie)
        # ------------------------------------------
        self.stem_pipeline_mode = int(gp("postproc/stem/pipeline_mode", 1)) 
        self.cmd_vel_topic = gp("postproc/stem/cmd_vel_topic", "/cmd_vel")
        self.stem_lost = float(gp("postproc/stem/stem_lost", 3.0))
        self.stem_depth_min = float(gp("postproc/stem/stem_depth_min", 0.1))
        self.stem_depth_max = float(gp("postproc/stem/stem_depth_max", 3.0))
        self.stem_assumed_radius = float(gp("postproc/stem/stem_assumed_radius", 0.02))
        # YOLO tracker 已停用；YOLO 只使用 predict() 做純偵測，不使用 ByteTrack / BoT-SORT。
        self.stem_max_jump_m = float(gp("postproc/stem/max_jump_m", 0.06)) 
        self.stem_ema_alpha = float(gp("postproc/stem/ema_alpha", 0.6))

        self.cfg_bunch = dict(
            size_mode=self.bunch_size_mode,
            expect_bbox_w_mm=self.bunch_expect_bbox_w_mm,
            expect_bbox_h_mm=self.bunch_expect_bbox_h_mm,
            size_ratio_min=self.bunch_size_ratio_min,
            expect_depth_m=self.bunch_expect_depth_m,
            depth_tol_m=self.bunch_depth_tol_m,
        )

        self.cls_bunch = int(gp("classes/bunch", 0))
        self.cls_stem = int(gp("classes/stem",  1))

        # 遮蔽率切換策略 (當果串被樹葉遮擋 > 90% 時切換尋找葉莖)
        self.policy_occ_hi = float(gp("policy/occ_thresh_high", 0.60))
        self.policy_hi_pat = int(gp("policy/high_patience", 3))

        # ------------------------------------------
        # 分割模型權重路徑設定
        # ------------------------------------------
        self.seg_model = gp("postproc/seg_model", "yolov11n-seg.pt").strip()
        self.sam_ckpt = gp("postproc/sam_ckpt", "sam_b.pt").strip()
        self.sam2_ckpt = gp("postproc/sam2_ckpt", "sam2_b.pt").strip()
        self.fastsam_ckpt = gp("postproc/fastsam_ckpt", "FastSAM-s.pt").strip()
        self.seg_imgsz = int(gp("postproc/seg_imgsz", 640))
        self.occ_seg_warmup_n = int(gp("occ/seg_warmup_n", 8))

        self.pp_retry_delay_sec = float(gp("postproc/retry_delay_sec", 1.0))

    # =========================
    # YOLO 與 幾何工具函數
    # =========================
    def yolo_backend_info(self,detector):
        """獲取 YOLO 模型當前運行的硬體後端資訊 (PyTorch / ONNXRuntime)"""
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
        """判斷 YOLO 是否成功使用 GPU 加速"""
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
        """載入 YOLO 權重檔，支援 .pt 與 .onnx 格式"""
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
        """強制讓 ONNXRuntime 嘗試使用 CUDA Provider"""
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
        """限制 BBox 不得超出影像邊界"""
        if xyxy is None:
            return None
        x1, y1, x2, y2 = map(float, xyxy)
        return np.array([max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)], dtype=np.float32)

    def rect_to_mask(self, depth, xyxy, expand=0.0):
        """將 2D BBox 轉換為粗略的布林遮罩 (Fallback 用)"""
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
        """
        執行 YOLO 純偵測，回傳所有物件的 BBox、信心度、類別與 dummy IDs。

        重要設計：
        - 不使用 detector.track()。
        - 不使用 ByteTrack / BoT-SORT / 任何 YOLO tracker。
        - YOLO 只負責 CRUISING 階段的重新偵測與初始化。
        - 後續追蹤交給 Cutie；stem_pipeline_mode == 4 時交給 SAM2 tracker。
        - ids 全部填 -1，避免後續邏輯依賴 YOLO tracker ID。
        """
        empty = (
            np.empty((0, 4), np.float32),
            np.empty((0,), np.float32),
            np.empty((0,), np.int32),
            np.empty((0,), np.int32),
        )

        if img_bgr is None:
            return empty

        try:
            r = detector.predict(
                source=img_bgr,
                imgsz=imgsz,
                conf=conf,
                device=self.det_device,
                verbose=False
            )[0]
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"[YOLO] predict failed: {e}")
            return empty

        if r.boxes is None or len(r.boxes) == 0:
            return empty

        xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        sc = r.boxes.conf.cpu().numpy().astype(np.float32)
        cl = r.boxes.cls.cpu().numpy().astype(np.int32)

        H, W = img_bgr.shape[:2]
        x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
        bw, bh = x2 - x1, y2 - y1

        valid = (
            np.isfinite(xyxy).all(axis=1) &
            np.isfinite(sc) &
            (bw > 1.0) &
            (bh > 1.0) &
            (x2 > 0) &
            (y2 > 0) &
            (x1 < W - 1) &
            (y1 < H - 1)
        )

        if not np.any(valid):
            return empty

        xyxy = xyxy[valid]
        sc = sc[valid]
        cl = cl[valid]
        ids = np.full((len(cl),), -1, dtype=np.int32)

        return xyxy, sc, cl, ids

    def project_3d_bbox_xyxy(self, K, center_pose, bbox_minmax, img_shape):
        """將 FP 推估的 3D 姿態轉換並投影回 2D 畫面的外接矩形"""
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
        """計算 BBox 區域內的代表深度值 (過濾邊緣雜訊)"""
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
        """根據策略 (最高分、最置中、深度最近) 挑選最佳的 YOLO 偵測框"""
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

    # ==========================================
    # 統一的動態 Mask 生成函數
    # 根據輸入的 mode (1, 2, 3, 4) 決定使用哪個神經網路來提取像素級遮罩
    # ==========================================
    def get_mask_by_mode(self, bgr, xyxy, mode, target_cls=None):
        """
        根據指定的管線編號，從 YOLO BBox 提示中提取高精度 Mask：
        mode 1: SAM (Segment Anything Model)
        mode 2: SAM2 (Segment Anything 2)
        mode 3: FastSAM (Fast Segment Anything)
        mode 4: YOLO11-seg (YOLOv11 內建分割)
        """
        if xyxy is None: return np.zeros(bgr.shape[:2], dtype=bool)
        
        # 模式 4: 交由 YOLO-seg 處理
        if mode == 4:
            return self.seg_mask_from_yolo_seg(bgr, xyxy, target_cls)
        
        # 模式 1~3: 取得對應的 SAM Prompt 模型
        model = self.prompt_models.get(mode)
        if model is None: 
            return self.rect_to_mask(bgr, xyxy, expand=self.roi_expand)
            
        x1, y1, x2, y2 = xyxy.astype(int)
        H, W = bgr.shape[:2]
        try:
            # 將 BBox 轉為 Prompt 餵給 SAM 家族模型
            results = model(bgr, bboxes=[[x1, y1, x2, y2]], imgsz=self.det_imgsz, device=self.det_device, verbose=False)
            if results and results[0].masks is not None:
                m = results[0].masks.data.cpu().numpy()
                if m.size > 0:
                    m2 = m[0] if m.ndim == 3 else m
                    # SAM 輸出的 mask 解析度可能與原圖不同，需要做 INTER_NEAREST 放縮
                    if m2.shape != (H, W):
                        m2 = cv2.resize(m2.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                    return (m2 > 0).astype(bool)
        except Exception as e:
            rospy.logwarn(f"[PromptModel Mode {mode}] inference error: {e}")
            
        # 如果模型推論失敗，退回使用傳統的 2D Bounding Box 填滿作為 Mask
        return self.rect_to_mask(bgr, xyxy, expand=self.roi_expand)

    # =========================
    # 遮蔽率 (Occ) 與 IoU 計算
    # =========================
    def iou_vs_projection_for_class_from_dets(self, color_bgr, K, center_pose, bbox_minmax, prefer_cls, xyxy_all, sc_all, cl_all, tag="bunch"):
        """計算 FP 3D 姿態的 2D 投影，與實際 YOLO 偵測框之間的交集比聯集 (IoU)"""
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
        """如果 IoU 連續過低，代表追蹤模型可能飄移，觸發 Re-init 強制重對齊"""
        if center_pose is None or self.K is None: return False
        if (self.frame_count % max(1, self.iou_stride)) != 0: return False

        if mode == "bunch":
            prefer_cls = self.cls_bunch
            bbox_mm    = self.bbox_bunch
            bad_count_attr = "iou_bad_count_bunch"
            last_upd_attr  = "last_iou_update_bunch"
        else:
            return False # 葉莖 (Stem) 不使用這套邏輯

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
        """計算兩矩形的 IoU 交集比例"""
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
        mode = getattr(self, "det_select_mode_current", self.det_select_mode)
        return (str(mode).strip().lower() == "middle")

    def pose_2d_box_xyxy(self, which:str):
        """投影 FP 的 6D 姿態為 2D 矩形 (做為 SAM 提示框)"""
        if which.lower() == "bunch":
            if self.pose_bunch is None or self.K is None: return None
            center_pose = self.pose_bunch @ np.linalg.inv(self.to_origin_bunch)
            bbox = self.bbox_bunch
            return self.project_3d_bbox_xyxy(self.K, center_pose, bbox, self.color.shape)
        return None

    def compute_occ_and_iou(self, which: str, xyxy_all=None, sc_all=None, cl_all=None):
        """
        暖機期間 (Warmup)：計算實體遮罩與 3D 模型投影的交集，推算真實遮蔽率 (Occ)。
        暖機後：為省 GPU 效能，只計算快速的 2D 框 IoU，不浪費算力。
        """
        occ = 1.0
        if self.color is None or self.K is None or which.lower() != "bunch":
            return 1.0, 0.0, False

        if self.pose_bunch is None:
            return 1.0, 0.0, False

        center_pose = self.pose_bunch @ np.linalg.inv(self.to_origin_bunch)

        # 1. 永遠計算快速 IoU
        iou_val, best_xyxy = self.iou_vs_projection_for_class_from_dets(
            self.color, self.K, center_pose, self.bbox_bunch,
            prefer_cls=self.cls_bunch,
            xyxy_all=xyxy_all, sc_all=sc_all, cl_all=cl_all,
            tag="bunch"
        )

        skip_occlusion_check = self._force_bunch_detect

        # 2. 暖機階段: 執行精確的分割與物理遮蔽率計算
        if (not skip_occlusion_check) and (self.seg_warmup_left_bunch > 0):
            self.seg_warmup_left_bunch -= 1
            if self.seg_warmup_left_bunch == 0:
                rospy.loginfo("[bunch] Seg warmup finished, switch to IoU-only mode (skip occ)")

            prompt_box = self.pose_2d_box_xyxy("bunch")
            
            # [核心修改] 統一透過管線編號調用分割模型生成遮罩
            seg = self.get_mask_by_mode(self.color, prompt_box, self.bunch_pipeline_mode, target_cls=self.cls_bunch)
            
            occ = self.occ_from_gt_mesh_vs_seg("bunch", seg.astype(bool))

            if getattr(self, "iou_log", False):
                try:
                    self.save_occ_seg_debug(
                        self.color, seg.astype(bool) if seg is not None else None,
                        self.K, center_pose, self.bbox_bunch, occ, best_xyxy, tag="bunch"
                    )
                except Exception:
                    pass

            return occ, (0.0 if iou_val is None else float(iou_val)), True

        # 3. 非暖機階段: IoU-only (不耗費 GPU 算遮蔽率)
        if iou_val is None:
            occ = None
            iou_show = 0.0
        else:
            occ = None
            iou_show = float(iou_val)

        if getattr(self, "iou_log", False):
            try:
                self.save_iou_debug(self.color, self.K, center_pose, self.bbox_bunch, best_xyxy, iou_val, tag="bunch")
            except Exception:
                pass

        return occ, iou_show, False
    
    # =========================
    # Debug / 繪圖與除錯工具
    # =========================
    def _setup_run_debug_dir(self):
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
        if d is None: d = getattr(self, "debug_dir", None)
        if not d: return
        try: os.makedirs(d, exist_ok=True)
        except Exception: pass

    def _ensure_parent(self, path: str):
        try:
            parent = os.path.dirname(path)
            if parent: os.makedirs(parent, exist_ok=True)
        except Exception: pass

    def _dbg_path(self, subdir: str, prefix: str):
        root = os.path.join(self.debug_dir, subdir)
        self._ensure_dir(root)
        p = os.path.join(root, f"{prefix}_{self.frame_count:06d}.png")
        self._ensure_parent(p)
        return p

    def _overlay_mask(self, img_bgr, mask_bool, alpha=0.45, color=(0, 0, 255)):
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
        if xyxy is None: return img
        x1, y1, x2, y2 = [int(t) for t in xyxy]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        if label is not None:
            cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return img

    def _draw_pose_box(self, img_bgr, K, pose_obj_in_cam, bbox_minmax, which, axis_scale=0.05):
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
        if not getattr(self, "iou_log", False): return
        try:
            if mask_bool is None: return
            m = (mask_bool.astype(np.uint8) * 255)
            outp = self._dbg_path(subdir, fname) 
            cv2.imwrite(outp, m)
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"[DBG] save_binary_mask failed: {e}")

    def render_mesh_silhouette_mask(self, K, pose_center_in_cam, mesh, img_shape):
        """利用矩陣運算將 3D Mesh 投影回 2D 生成理想無遮擋遮罩"""
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
        """計算面積比率： (Mesh投影聯集 Seg Mask) / Mesh投影面積"""
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
        est_xyxy = self.project_3d_bbox_xyxy(K, center_pose, bbox_minmax, img_shape)
        if est_xyxy is None: return np.zeros(img_shape[:2], dtype=bool)
        return self.rect_to_mask(np.zeros(img_shape[:2], np.uint8), est_xyxy, expand=0.0)

    def seg_mask_from_yolo_seg(self, bgr, xyxy, target_cls=None):
        """利用 YOLO-Seg 輸出高精度的畫素分割"""
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
    # 葉莖 (Stem) 實體追蹤與深度計算
    # =========================
    def _backproject_uvz(self, u, v, z):
        """將 pixel + depth 反投影到 camera_color_optical_frame。"""
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x = (float(u) - cx) * float(z) / fx
        y = (float(v) - cy) * float(z) / fy
        return np.array([x, y, float(z)], dtype=np.float64)

    def _normalize_vec(self, v, fallback):
        v = np.asarray(v, dtype=np.float64)
        n = np.linalg.norm(v)
        if n < 1e-9 or not np.isfinite(n):
            return np.asarray(fallback, dtype=np.float64)
        return v / n

    def calculate_direct_root_pose(self, depth_m, mask_bool):
        """
        從 stem mask + depth 建立與 FoundationPose 相同定義的 4x4 pose。

        統一定義：回傳的 pose 是 T_camera_stem，也就是 stem 物件座標系在
        camera_color_optical_frame 下的位置與姿態。矩陣欄向量定義如下：
            T[:3, 0] = +X_stem，顯示為紅色
            T[:3, 1] = +Y_stem，顯示為綠色
            T[:3, 2] = +Z_stem，顯示為藍色

        純分割沒有 CAD mesh，因此這裡用穩定的幾何規則定義 stem frame：
            origin  = mask 最下方的 root / cut point
            +Y      = root 指向 stem 上方的中心線方向
            +Z      = camera ray 在垂直於 +Y 的平面上投影後的方向
            +X      = +Y 與 +Z 決定的右手座標軸
        """
        v, u = np.where(mask_bool > 0)
        if len(v) == 0:
            return None, None, None, "No Mask Points"

        z_raw = depth_m[v, u]
        valid_mask = (z_raw > self.stem_depth_min) & (z_raw < self.stem_depth_max)
        if np.sum(valid_mask) < 10:
            return None, None, None, "No valid depth in mask"

        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = z_raw[valid_mask]

        # 1) origin：畫面最下方的 root / cut point
        sorted_indices = np.argsort(v_valid)[::-1]
        bottom_k = min(20, len(sorted_indices))
        bottom_indices = sorted_indices[:bottom_k]

        target_u = int(np.median(u_valid[bottom_indices]))
        target_v = int(np.median(v_valid[bottom_indices]))
        target_z = float(np.median(z_valid[bottom_indices]))

        p_surface = self._backproject_uvz(target_u, target_v, target_z)
        center_3d = p_surface.copy()
        center_3d[2] += float(self.stem_assumed_radius)
        projected_2d = (target_u, target_v)

        # 2) +Y_stem：root 指向 stem 上方中心線。
        #    取 mask 較上方的一小群點，比單點 PCA 更穩。
        top_k = min(30, len(sorted_indices))
        top_indices = sorted_indices[-top_k:]
        top_u = float(np.median(u_valid[top_indices]))
        top_v = float(np.median(v_valid[top_indices]))
        top_z = float(np.median(z_valid[top_indices]))
        p_top = self._backproject_uvz(top_u, top_v, top_z)
        y_axis = self._normalize_vec(p_top - p_surface, fallback=np.array([0.0, -1.0, 0.0]))

        # 3) +Z_stem：用相機視線方向固定 stem 的 roll，避免每幀亂轉。
        #    先把 camera ray 投影到垂直於 +Y 的平面，確保與 +Y 正交。
        cam_ray = self._normalize_vec(center_3d, fallback=np.array([0.0, 0.0, 1.0]))
        z_axis = cam_ray - np.dot(cam_ray, y_axis) * y_axis
        z_axis = self._normalize_vec(z_axis, fallback=np.array([0.0, 0.0, 1.0]))

        # 4) 右手座標系：X × Y = Z。
        x_axis = self._normalize_vec(np.cross(y_axis, z_axis), fallback=np.array([1.0, 0.0, 0.0]))
        z_axis = self._normalize_vec(np.cross(x_axis, y_axis), fallback=z_axis)

        R = np.eye(4, dtype=np.float64)
        R[:3, 0] = x_axis  # red
        R[:3, 1] = y_axis  # green
        R[:3, 2] = z_axis  # blue
        quat = tf.transformations.quaternion_from_matrix(R)

        return center_3d.astype(np.float64), quat, projected_2d, "OK"

    def cmd_vel_callback(self, msg: Twist):
        """讀取 ROS 車體線速度與角速度"""
        self.current_twist = msg

    def get_predicted_stem_pos(self, now):
        """利用 TF Tree 自動將歷史相機點轉到車體系，扣除車體位移後再轉回相機系"""
        if self.last_stem_3d_pos is None or self.last_stem_time is None:
            return None
            
        dt = (now - self.last_stem_time).to_sec()
        if dt <= 0: return self.last_stem_3d_pos

        # 讀取車體速度
        vx = self.current_twist.linear.x
        vy = self.current_twist.linear.y
        wz = self.current_twist.angular.z

        parent_frame = self.camera_tf if self.camera_tf else "camera_color_optical_frame"

        try:
            # 從 TF 樹直接動態查詢相機相對於機器人中心的安裝姿態
            # 這會自動吸收相機裝在左邊、右邊、正裝或倒裝的所有物理狀態
            self.tf_listener.waitForTransform('base_link', parent_frame, rospy.Time(0), rospy.Duration(0.1))
            (trans, rot) = self.tf_listener.lookupTransform('base_link', parent_frame, rospy.Time(0))
            
            # 建立相機到車體系的 4x4 轉換矩陣
            T_b_c = tf.transformations.quaternion_matrix(rot)
            T_b_c[0:3, 3] = trans
            
            # 1. 將上一幀在相機系下的 3D 座標轉換到車體系 (base_link)
            P_c = np.append(self.last_stem_3d_pos, 1.0)
            P_b = T_b_c @ P_c
            
            # 2. 在車體系下，扣除這段時間內車體產生的相對運動量 (逆運動學預測)
            dx = vx * dt
            dy = vy * dt
            dtheta = wz * dt

            P_b[0] -= dx
            P_b[1] -= dy

            cos_t = np.cos(-dtheta)
            sin_t = np.sin(-dtheta)
            nx = P_b[0] * cos_t - P_b[1] * sin_t
            ny = P_b[0] * sin_t + P_b[1] * cos_t
            P_b[0] = nx
            P_b[1] = ny

            # 3. 透過逆矩陣，將預測完的新車體座標，重新投影回當前的相記光學系
            T_c_b = np.linalg.inv(T_b_c)
            P_c_new = T_c_b @ P_b
            
            return P_c_new[:3]

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, f"[Kinematics TF] Lookup failed, fallback to static: {e}")
            return self.last_stem_3d_pos

    def pick_nearest_to_2d_point(self, xyxy, sc, cl, ids, target_2d, cls_id):
        """從 YOLO 所有框中，挑選與預測點最接近的 BBox"""
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
        
        return xyxy[bid], float(sc[bid]), -1

    def _bbox_center(self, xyxy):
        if xyxy is None:
            return None
        return np.array([
            0.5 * (float(xyxy[0]) + float(xyxy[2])),
            0.5 * (float(xyxy[1]) + float(xyxy[3]))
        ], dtype=np.float32)

    def _bbox_depth_median(self, bbox_xyxy, depth_m=None):
        """回傳 bbox 中央區域的 median depth；若沒有有效深度則回傳 None。"""
        if depth_m is None:
            depth_m = getattr(self, "depth_m", None)

        if depth_m is None or bbox_xyxy is None:
            return None

        z_max = float(self.max_depth_mm) * 0.001 if hasattr(self, "max_depth_mm") else 10.0
        return self._bbox_depth_distance_m(
            bbox_xyxy, depth_m,
            sample_ratio=0.45, min_valid=20, use="median",
            z_min=0.05, z_max=z_max
        )

    def select_nearest_depth_bbox_for_cls(self, xyxy, scores, classes, cls_id, conf_th=0.0):
        """從指定 class 中選出 bbox 中央區域深度最近者。"""
        if xyxy is None or len(xyxy) == 0:
            return None, None, None

        idx = np.where(classes == int(cls_id))[0]
        if idx.size == 0:
            return None, None, None

        best_i = -1
        best_depth = float("inf")

        for i in idx:
            if scores is not None and conf_th is not None and float(scores[i]) < float(conf_th):
                continue

            d = self._bbox_depth_median(xyxy[i])
            if d is None:
                continue

            if d < best_depth:
                best_depth = d
                best_i = int(i)

        if best_i < 0:
            return None, None, None

        return xyxy[best_i], float(scores[best_i]), float(best_depth)

    def select_stem_bbox_priority(self, xyxy_all, sc_all, cl_all):
        """
        STEM 抓取候選選擇策略。
        優先順序：
        1. 距離相機最近的 stem。
        2. 若 stem 深度不可用，選離最近距離 bunch 最近的 stem。
        3. 若沒有有效 bunch，退回最高分 stem。
        """
        if xyxy_all is None or len(xyxy_all) == 0:
            return None, None

        # 第一順位：最近深度 stem。
        stem_xyxy, stem_score, stem_depth = self.select_nearest_depth_bbox_for_cls(
            xyxy_all, sc_all, cl_all, self.cls_stem, conf_th=self.det_conf
        )

        if stem_xyxy is not None:
            rospy.loginfo_throttle(
                0.5,
                f"[STEM SELECT] nearest-depth stem selected: depth={stem_depth:.3f}m, score={stem_score:.3f}"
            )
            return stem_xyxy, stem_score

        # 第二順位：找最近距離 bunch，再選離該 bunch 中心最近的 stem。
        bunch_xyxy, bunch_score, bunch_depth = self.select_nearest_depth_bbox_for_cls(
            xyxy_all, sc_all, cl_all, self.cls_bunch, conf_th=self.det_conf
        )

        stem_idx = np.where(cl_all == int(self.cls_stem))[0]
        if stem_idx.size == 0:
            return None, None

        # 如果沒有有效 bunch depth，改用最高分 bunch 作為幾何參考。
        if bunch_xyxy is None:
            bunch_xyxy, bunch_score = self.select_yolo_bbox(
                xyxy_all, sc_all, cl_all,
                img_shape=self.color.shape,
                prefer_cls=self.cls_bunch,
                select_mode="score",
                conf_th=self.det_conf
            )

        if bunch_xyxy is not None:
            bc = self._bbox_center(bunch_xyxy)
            best_i = -1
            best_d2 = float("inf")

            for i in stem_idx:
                if sc_all is not None and float(sc_all[i]) < float(self.det_conf):
                    continue

                scenter = self._bbox_center(xyxy_all[i])
                if scenter is None or bc is None:
                    continue

                d2 = float(np.sum((scenter - bc) ** 2))
                if d2 < best_d2:
                    best_d2 = d2
                    best_i = int(i)

            if best_i >= 0:
                rospy.loginfo_throttle(
                    0.5,
                    f"[STEM SELECT] fallback nearest-to-nearest-bunch selected: "
                    f"stem_score={float(sc_all[best_i]):.3f}, "
                    f"bunch_depth={bunch_depth if bunch_depth is not None else -1:.3f}m"
                )
                return xyxy_all[best_i], float(sc_all[best_i])

        # 第三順位：最高分 stem。
        stem_xyxy, stem_score = self.select_yolo_bbox(
            xyxy_all, sc_all, cl_all,
            img_shape=self.color.shape,
            prefer_cls=self.cls_stem,
            select_mode="score",
            conf_th=self.det_conf
        )

        if stem_xyxy is not None:
            rospy.loginfo_throttle(
                0.5,
                f"[STEM SELECT] fallback highest-score stem selected: score={stem_score:.3f}"
            )

        return stem_xyxy, stem_score

    # ==========================================
    # GUI 與畫面管理
    # ==========================================
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
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), thickness=-1)
        cv2.rectangle(img, (x, y), (x + int(w * v), y + h), (60, 180, 75), thickness=-1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (220, 220, 220), thickness=1)
        cv2.putText(img, f"{label}: {val:.3f}", (x, y - 6),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
    # =========================
    # 方向/尺寸 後處理防呆
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
        t = (float(T[0,3]), float(T[1,3]), float(T[2,3]))
        qx,qy,qz,qw = tf.transformations.quaternion_from_matrix(T)
        return t, (float(qx),float(qy),float(qz),float(qw))

    def broadcast_transform_and_pose(self, T: np.ndarray, which: str, parent: str):
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
        mode = (getattr(msg, "det_select_mode", "") or "").strip().lower()
        if mode not in ("score", "middle", "nearest_depth"):
            rospy.logwarn_throttle(1.0, f"[DETECTION] invalid det_select_mode={mode}, fallback score")
            mode = "score"
        self.det_select_mode_current = mode
        self.ready_received.detection_allowed = bool(getattr(msg, "detection_allowed", False))
    
    def harvestDoneCallback(self, msg: Bool):
        if not bool(msg.data):
            return
        rospy.logwarn("[HARVEST_DONE] received True -> hard reset to bunch")
        self._reset_all_to_bunch()
        self._publish_zero_current(used_seg=False)

    def _tag(self, which: str) -> str:
        return "bunch" if which.lower() == "bunch" else "stem"

    def _state(self, which: str, used_seg: bool = False) -> str:
        part = self._tag(which)
        allowed = bool(getattr(self.ready_received, "detection_allowed", False))

        if self.yolo_start_mode == "wait" and (not allowed):
            return f"{part}:PAUSED"

        if not (self.got_depth and self.got_rgb and self.K is not None and hasattr(self, "depth_m")):
            return f"{part}:INITIALIZING"

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
        which = "stem" if self.mode == "stem" else "bunch"
        self.confidence_publish(which, 0.0, False, used_seg=used_seg)

    def _reset_cutie_processor(self, reason=""):
        """
        完整重置 Cutie InferenceCore
        """
        if not CUTIE_AVAILABLE:
            self.cutie_processor = None
            return

        try:
            if getattr(self, "cutie_processor", None) is not None:
                try:
                    self.cutie_processor.clear_memory()
                except Exception as e:
                    rospy.logwarn(f"[Cutie Reset] clear_memory failed before rebuild: {e}")

            self.cutie_processor = InferenceCore(self.cutie_net, cfg=self.cutie_net.cfg)
            self.cutie_processor.max_internal_size = 640

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            rospy.logwarn(f"[Cutie Reset] Re-created InferenceCore. reason={reason}")

        except Exception as e:
            rospy.logerr(f"[Cutie Reset] Failed to re-create InferenceCore: {e}")
            self.cutie_processor = None

    def _reset_pipeline_state(self):
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

        self.bunch_cutie_state = "CRUISING"
        self.stem_cutie_state = "CRUISING"
        self._reset_cutie_processor(reason="_reset_pipeline_state")

        if getattr(self, "sam2_tracker", None) is not None:
            try:
                self.sam2_tracker.reset_memory()
            except Exception as e:
                rospy.logwarn(f"[SAM2 Reset] reset_memory failed: {e}")

    def _reset_all_to_bunch(self):
        rospy.logwarn("[RESET] reset all pipeline state to BUNCH / CRUISING")
        self._reset_pipeline_state()
        self._yolo_delay_left_bunch = 0
        self._yolo_delay_bbox_bunch = None
        self._post_pending = False
        self._post_fail_time = None
        self._last_yolo_text = ""
        self._registering_until = 0
        self._reinit_until = 0

    def _reset_stem_to_cruising(self, reason=""):
        """
        STEM tracking loss 時使用。
        重點：
        - 不切回 bunch，仍停留在 self.mode == "stem"。
        - 清掉目前 stem pose / tracking memory。
        - 下一幀回到 STEM CRUISING，重新執行 YOLO + SAM/SAM2/FastSAM，
          重新給 Cutie 第一幀 mask。
        """
        rospy.logwarn(f"[STEM RESET] reset stem to CRUISING. reason={reason}")

        self.pose_stem = None
        self._stem_lock = False
        self.target_stem_id = -1
        self.last_stem_3d_pos = None
        self.last_stem_time = None
        self.stem_cutie_state = "CRUISING"

        self.confidence_publish("stem", 0.0, False, used_seg=False)

        # Mode 1~3：Cutie tracking loss，重建 InferenceCore，避免 memory / sensory state 殘留。
        if self.stem_pipeline_mode != 4:
            try:
                self._reset_cutie_processor(reason=f"stem lost: {reason}")
            except Exception as e:
                rospy.logwarn(f"[STEM RESET] Cutie reset failed: {e}")
                try:
                    if getattr(self, "cutie_processor", None) is not None:
                        self.cutie_processor.clear_memory()
                except Exception:
                    pass

        # Mode 4：SAM2 tracker memory reset。
        if self.stem_pipeline_mode == 4 and getattr(self, "sam2_tracker", None) is not None:
            try:
                self.sam2_tracker.reset_memory()
            except Exception as e:
                rospy.logwarn(f"[STEM RESET] SAM2 reset_memory failed: {e}")
    
    def _handle_detection_paused(self):
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
        """
        從 YOLO 純偵測結果中，挑選最靠近 target_xyxy 中心的指定類別 bbox。
        不依賴 YOLO tracker ID；第三個回傳值固定為 -1。
        """
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
        return xyxy[bid], float(sc[bid]), -1

    def _yolo_delay_update(self, which: str, xyxy_now):
        self.yolo_delay_frames = 5
        n = max(0, int(getattr(self, "yolo_delay_frames", 0)))
        
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

    # ==========================================
    # ROS 節點主循環 (Main Spin)
    # 控管所有硬體與演算法的互動流程
    # ==========================================
    def spin(self):
        self.frame_count = 0
        used_seg = False

        while not rospy.is_shutdown():
            iou_for_bar = 0.0
            used_seg = False
            
            # 等候所有感測器與相機資料就緒
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
            img_tensor = F.to_tensor(img_rgb).cuda().float()
            vis_bgr = self.color.copy()

            # --- 判斷目前狀態是否需要運行 YOLO 來重新尋找物體 ---
            run_yolo = False
            if self.mode == "bunch" and self.bunch_cutie_state == "CRUISING":
                run_yolo = True
            elif self.mode == "stem" and self.stem_cutie_state == "CRUISING":
                run_yolo = True

            xyxy_all, sc_all, cl_all, ids_all = None, None, None, None
            if run_yolo:
                xyxy_all, sc_all, cl_all, ids_all = self.yolo_det_all(self.detector, self.color, imgsz=self.det_imgsz, conf=self.det_conf)

            # Cutie 初始遮罩矩陣 (0=背景, 1=果串, 2=葉莖)
            cutie_init_mask = np.zeros((self.rgb_size[1], self.rgb_size[0]), dtype=np.uint8)
            cutie_init_objs = []

            # ==========================================
            # 1. BUNCH CRUISING (全域搜尋果串)
            # ==========================================
            if self.mode == "bunch" and self.bunch_cutie_state == "CRUISING":
                bunch_xyxy, bunch_conf = self.select_yolo_bbox(
                    xyxy_all, sc_all, cl_all, img_shape=self.color.shape,
                    prefer_cls=self.cls_bunch, select_mode=self.det_select_mode_current, conf_th=self.det_conf
                )
                ready, bb_use = self._yolo_delay_update("bunch", bunch_xyxy)
                
                if not ready:
                    if bunch_xyxy is not None:
                        cv2.putText(vis_bgr, f"YOLO detected. Delay... ({self._yolo_delay_left_bunch} frames left)",
                                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    self._publish_zero_current(used_seg=False)
                else:
                    # [核心修改] 透過 bunch_pipeline_mode (1~4) 產生精確的 Cutie 初始化 Mask
                    used_mask = self.get_mask_by_mode(self.color, bb_use, self.bunch_pipeline_mode, target_cls=self.cls_bunch)
                    if used_mask.sum() > 50:
                        rospy.loginfo(f"[Hand-off] YOLO -> Cutie Locked BUNCH (ID=1) via Mode {self.bunch_pipeline_mode}!")
                        cutie_init_mask[used_mask] = 1
                        cutie_init_objs.append(1)
                        
                        # 首次向 FoundationPose 註冊果串
                        self.pose_bunch = self.est_bunch.register(
                            K=self.K, rgb=self.color, depth=self.depth_m, ob_mask=used_mask.astype(bool), iteration=self.est_refine_iter
                        )
                        self.bunch_cutie_state = "SERVOING" # 狀態進入伺服追蹤

            # ==========================================
            # 2. STEM CRUISING (根據機器人軌跡搜尋葉莖)
            # ==========================================
            predicted_2d = None
            if self.mode == "stem":
                # 計算先前的物理預測落點
                predicted_3d = self.get_predicted_stem_pos(now)
                if predicted_3d is not None and predicted_3d[2] > 0:
                    u = int(self.K[0,0] * predicted_3d[0] / predicted_3d[2] + self.K[0,2])
                    v = int(self.K[1,1] * predicted_3d[1] / predicted_3d[2] + self.K[1,2])
                    predicted_2d = (u, v)
                    if self.show_rgb_win: cv2.circle(vis_bgr, (u, v), 5, (255, 0, 0), -1)

                if self.stem_cutie_state == "CRUISING" and run_yolo:
                    # STEM 重新初始化不使用 YOLO tracker ID。
                    # 優先順序：
                    #   1) 距離相機最近的 stem。
                    #   2) 若 stem 深度不可用，選離最近距離 bunch 最近的 stem。
                    #   3) 若都不可用，退回最高分 stem。
                    stem_xyxy, stem_score = self.select_stem_bbox_priority(
                        xyxy_all, sc_all, cl_all
                    )

                    if stem_xyxy is None and predicted_2d is not None:
                        # 最後補救：若 YOLO 有 stem 但深度/果串參考都失敗，
                        # 用上一幀運動預測點找最接近的 stem。
                        stem_xyxy, stem_score, _ = self.pick_nearest_to_2d_point(
                            xyxy_all, sc_all, cl_all, ids_all, predicted_2d, self.cls_stem
                        )

                        if stem_xyxy is None and self.last_stem_time is not None and (now - self.last_stem_time) > self.stem_lost_timeout:
                            self.last_stem_3d_pos = None
                            self.last_stem_time = None

                    if stem_xyxy is not None:
                        # [核心修改] 根據 Stem Mode (1~4) 進行分流處理
                        # Mode 4: 捨棄 Cutie，完全交由 SAM2 內建的 Video Tracker 追蹤
                        if self.stem_pipeline_mode == 4 and getattr(self, "sam2_tracker", None) is not None:
                            x1, y1, x2, y2 = stem_xyxy.astype(int)
                            # 使用 SAM2 的 update_memory 功能進行首次記憶體寫入
                            seg_results = self.sam2_tracker(source=self.color, bboxes=[[x1, y1, x2, y2]], obj_ids=[2], update_memory=True)
                            if seg_results and seg_results[0].masks is not None:
                                m = seg_results[0].masks.data.cpu().numpy()
                                m2 = m[0] if m.ndim == 3 else m
                                if m2.shape != self.color.shape[:2]:
                                    m2 = cv2.resize(m2.astype(np.uint8), (self.color.shape[1], self.color.shape[0]), interpolation=cv2.INTER_NEAREST)
                                used_mask = (m2 > 0).astype(bool)
                            else:
                                used_mask = np.zeros(self.color.shape[:2], dtype=bool)
                        else:
                            # Mode 1, 2, 3: 提取 Prompt 遮罩，準備交給下方的 Cutie 處理
                            used_mask = self.get_mask_by_mode(self.color, stem_xyxy, self.stem_pipeline_mode, target_cls=self.cls_stem)

                        if used_mask.sum() > 50:
                            rospy.loginfo(
                                f"[Hand-off] YOLO -> Locked STEM "
                                f"(Mode: {self.stem_pipeline_mode}, score={stem_score})!"
                            )
                            self.stem_cutie_state = "SERVOING"
                            self.pose_stem = None
                            # 只有 Mode 1~3 需要把 Mask 塞進 Cutie 陣列
                            if self.stem_pipeline_mode != 4:
                                cutie_init_mask[used_mask] = 2
                                cutie_init_objs.append(2)

            # ==========================================
            # 3. 執行 Cutie 推論 (初始化 或 畫面傳播追蹤)
            # ==========================================
            pred_mask_tensor = None
            if self.cutie_processor is not None:
                # 情況 A: YOLO 剛剛交接，需要餵入 Prompt Mask 來寫入 Cutie 記憶體
                if len(cutie_init_objs) > 0:
                    init_mask_tensor = torch.from_numpy(cutie_init_mask).cuda().long()
                    with torch.cuda.amp.autocast(), torch.inference_mode():
                        output_prob = self.cutie_processor.step(img_tensor, init_mask_tensor, objects=cutie_init_objs)
                        pred_mask_tensor = self.cutie_processor.output_prob_to_mask(output_prob)
                
                # 情況 B: 系統已在追蹤狀態 (Bunch 追蹤中，或 Stem 採用 Mode 1~3 追蹤中)
                elif self.bunch_cutie_state == "SERVOING" or (self.stem_cutie_state == "SERVOING" and self.stem_pipeline_mode != 4):
                    with torch.cuda.amp.autocast(), torch.inference_mode():
                        # 純依賴 Cutie 內部的時序記憶體推論當前幀
                        output_prob = self.cutie_processor.step(img_tensor)
                        pred_mask_tensor = self.cutie_processor.output_prob_to_mask(output_prob)

            # ==========================================
            # 4. BUNCH SERVOING (結合 FP 推算 3D 果串姿態)
            # ==========================================
            iou_for_bar = 0.0
            if self.mode == "bunch" and self.bunch_cutie_state == "SERVOING" and pred_mask_tensor is not None:
                bunch_mask_bool = (pred_mask_tensor.squeeze().cpu().numpy() == 1)
                
                if bunch_mask_bool.sum() < 100:
                    rospy.logwarn("[BUNCH] Cutie track lost! Resetting all.")
                    self._reset_all_to_bunch()
                else:
                    used_seg = True
                    # 將 Cutie 計算出的 2D 像素區域丟給 FoundationPose 執行 6D Pose 追蹤
                    self.pose_bunch = self.est_bunch.track_one(rgb=self.color, depth=self.depth_m, K=self.K, iteration=self.track_refine_iter)
                    
                    occ = self.occ_from_gt_mesh_vs_seg("bunch", bunch_mask_bool)
                    
                    # 姿態救援機制：比對 Cutie 與 FP 的重合度
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
                    
                    # FP 姿態飄移時，強制用 Cutie 高精度 Mask 重新抓取 3D 深度
                    if iou < self.iou_thresh or pending:
                        rospy.logwarn(f"[BUNCH] FP Drifted (IoU={iou:.2f}) or Pose Bad. Rescuing with Cutie ROI!")
                        self.pose_bunch = self.est_bunch.register(
                            K=self.K, rgb=self.color, depth=self.depth_m, ob_mask=bunch_mask_bool, iteration=self.est_refine_iter
                        )
                        self._post_pending = False 
                        
                    # 若遮蔽率超標持續 N 幀，切換為 Stem 模式
                    if occ >= self.policy_occ_hi: self._hi_cnt += 1
                    else: self._hi_cnt = 0

                    if self._hi_cnt >= self.policy_hi_pat:
                        self.mode = "stem"
                        self.bunch_cutie_state = "CRUISING"
                        self.stem_cutie_state = "CRUISING"

                        # BUNCH object id=1 轉 STEM object id=2 前，重建 Cutie
                        self._reset_cutie_processor(reason="mode switch bunch -> stem")

                        rospy.logwarn("[MODE SWITCH] 遮蔽率過高，切換至 STEM 模式，Cutie重建。")

                    # ROS TF 發送
                    if self.pose_bunch is not None:
                        self.last_bunch_3d_pos = self.pose_bunch[:3, 3].copy()
                        parent_frame = self.camera_tf if self.camera_tf else "camera_color_optical_frame"
                        self.broadcast_transform_and_pose(self.pose_bunch, "bunch", parent_frame)
                        self.confidence_publish("bunch", iou_for_bar, True, used_seg=True)
                        
                        if self.show_rgb_win:
                            vis_bgr = self._overlay_mask(vis_bgr, bunch_mask_bool, alpha=0.4, color=(0, 0, 255)) 
                            center_pose = self.pose_bunch @ np.linalg.inv(self.to_origin_bunch)
                            vis_bgr = draw_posed_3d_box(self.K, img=cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB), ob_in_cam=center_pose, bbox=self.bbox_bunch)
                            vis_bgr = draw_xyz_axis(vis_bgr, ob_in_cam=self.pose_bunch, scale=0.05, K=self.K, thickness=3, transparency=0, is_input_rgb=True)
                            vis_bgr = cv2.cvtColor(vis_bgr, cv2.COLOR_RGB2BGR)

            # ==========================================
            # 5. STEM SERVOING (依據 2D Mask 計算葉莖 3D 切割點)
            # ==========================================
            if self.mode == "stem" and self.stem_cutie_state == "SERVOING":
                stem_mask_255 = None
                
                # Mode 4 分流: 提取 SAM2 Tracker 的結果
                if self.stem_pipeline_mode == 4 and getattr(self, "sam2_tracker", None) is not None:
                    track_results = self.sam2_tracker(source=self.color)
                    if track_results and track_results[0].masks is not None:
                        m = track_results[0].masks.data.cpu().numpy()
                        if m.size > 0:
                            m2 = m[0] if m.ndim == 3 else m
                            if m2.shape != self.color.shape[:2]:
                                m2 = cv2.resize(m2.astype(np.uint8), (self.color.shape[1], self.color.shape[0]), interpolation=cv2.INTER_NEAREST)
                            stem_mask_255 = (m2 > 0).astype(np.uint8) * 255
                
                # Mode 1, 2, 3 分流: 提取 Cutie Processor 的結果
                elif pred_mask_tensor is not None:
                    stem_mask_255 = (pred_mask_tensor.squeeze().cpu().numpy() == 2).astype(np.uint8) * 255

                if stem_mask_255 is not None:
                    used_mask = stem_mask_255
                    mask_area = int((used_mask.astype(bool)).sum())
                    if mask_area < 50:
                        self._reset_stem_to_cruising(
                            reason=f"mask too small, area={mask_area}"
                        )
                    else:
                        center_3d, quat, projected_2d, status = self.calculate_direct_root_pose(
                            self.depth_m,
                            used_mask
                        )

                        if center_3d is not None:
                            # 3D 防呆 1: 防前景誤判 (若 Z 值比果串還要近，則駁回)
                            # if self.last_bunch_3d_pos is not None and center_3d[2] < (self.last_bunch_3d_pos[2] - 0.05):
                            #     rospy.logwarn_throttle(0.5, "[stem] REJECTED: Foreground leaf (Z too close).")
                            #     center_3d = None 

                            # 3D 防呆 2: 防止空間跳躍 (過濾旁邊干擾的樹枝)
                            if center_3d is not None and predicted_3d is not None:
                                jump_dist = float(np.linalg.norm(center_3d - predicted_3d))
                                if jump_dist > self.stem_max_jump_m:  
                                    rospy.logwarn_throttle(0.5, f"[stem] REJECTED: Jumped {jump_dist:.2f}m")
                                    center_3d = None 

                            if center_3d is not None:
                                # EMA 低通濾波讓機械手臂夾取更平滑
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
                                    vis_bgr = self._overlay_mask(vis_bgr, used_mask, alpha=0.5, color=(0, 255, 255))
                                    cv2.circle(vis_bgr, projected_2d, 8, (0, 0, 255), -1) 
                                    cv2.putText(vis_bgr, f"Depth:{smoothed_3d[2]:.2f}m", (projected_2d[0] + 10, projected_2d[1] - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                
                                parent_frame = self.camera_tf if self.camera_tf else "camera_color_optical_frame"
                                self.broadcast_transform_and_pose(self.pose_stem, "stem", parent_frame)
                                self.confidence_publish("stem", 1.0, True, used_seg=True)
                                
                        if center_3d is None:
                            self._reset_stem_to_cruising(
                                reason=f"calculate_direct_root_pose failed or rejected: {status}"
                            )
                else:
                    self._reset_stem_to_cruising(
                        reason="no mask output from Cutie/SAM2 tracker"
                    )

            # ==========================================
            # 6. GUI 資訊更新
            # ==========================================
            if self.show_rgb_win:
                if self.mode == "bunch" and self.bunch_cutie_state == "SERVOING":
                    cv2.putText(vis_bgr, f"BUNCH: CUTIE & FP LOCKED (Mode {self.bunch_pipeline_mode})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif self.mode == "stem":
                    cv2.putText(vis_bgr, f"BUNCH: PAUSED (STEM MODE)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

                if self.mode == "stem" and self.stem_cutie_state == "SERVOING":
                    tracker_name = "SAM2" if self.stem_pipeline_mode == 4 else "CUTIE"
                    cv2.putText(vis_bgr, f"STEM: {tracker_name} LOCKED (Mode {self.stem_pipeline_mode})", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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
