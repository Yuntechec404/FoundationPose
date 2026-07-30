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
import math

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
    print("[WARNING] Cutie module not found. Frond Servoing will fallback or fail.")

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
        if self.self_eval_log:
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
        
        # 紀錄果串(fruit)與葉莖(frond)的 6D 姿態矩陣
        self.pose_bunch = None
        self.pose_stem  = None
        self.mask = None
        
        # 計數器與旗標
        self.frame_count = 0
        self.K = None # 相機內參矩陣
        self._last_yolo_text = ""

        # FoundationPose 幾何自我評估狀態。
        # MAE 越低越好；Inlier Ratio 越高越好。
        self.depth_mae = None
        self.inlier_ratio = None
        self.depth_conf_score = None
        self.geom_bad_count = 0
        self.geom_state = "UNAVAILABLE"
        self.geom_high_occlusion = False
        
        self.ready_received = Detection()
        self.ready_received.detection_allowed = False
        self.det_select_mode_current = self.det_select_mode
        self._stem_lock = False
        self.target_stem_id = -1
        self.last_bunch_3d_pos = None
        self._pause_hold = False
        self._last_allowed = False
        self.target_mode_locked = False
        self.locked_target_mode = None

        # ------------------------------------------
        # Cutie VOS 初始化
        # ------------------------------------------
        if CUTIE_AVAILABLE:
            rospy.loginfo("[Cutie] Loading weights...")
            self.cutie_net = get_default_model()
            self.cutie_net.eval().cuda()
            self.cutie_processor = InferenceCore(self.cutie_net, cfg=self.cutie_net.cfg)
            self.cutie_processor.max_internal_size = 640 # 限制解析度以防止記憶體溢出 (OOM)

        # Frond 維持原本的 Cutie / SAM2 追蹤狀態。
        # Fruit 不再使用 Cutie；是否已進入 FoundationPose tracking 直接由 pose_bunch 判斷。
        self.stem_cutie_state = "CRUISING"
        
        # ------------------------------------------
        # 葉莖 (Frond) 持續追蹤狀態 (用於運動預測)
        # ------------------------------------------
        self.last_stem_3d_pos = None  # 紀錄最後一次算出的葉莖 3D 相機座標
        self.last_stem_time = None    # 紀錄最後一次更新的時間，計算 dt 用
        self.stem_lost_timeout = rospy.Duration(self.stem_lost) 
        
        # YOLO 穩定偵測狀態：連續偵測 init_det_patience 幀後才交給 FoundationPose。
        # 命名與 foundationpose_tracker.py 保持一致。
        self.consecutive_det_count = 0

        # GUI 視窗狀態
        self._rgb_win_created = False
        self._depth_win_created = False
        self._rgb_win_sized = False
        self._depth_win_sized = False
        self._rgb_initial_size = (900, 720)
        self._depth_initial_size = (900, 720)

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
        self.bbox_bunch = np.stack([-self.extents_bunch/2, self.extents_bunch/2], axis=0).reshape(2, 3)

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.est_bunch = FoundationPose(
            model_pts=self.mesh_bunch.vertices,
            model_normals=self.mesh_bunch.vertex_normals,
            mesh=self.mesh_bunch,
            scorer=self.scorer,
            refiner=self.refiner,
            debug_dir=self.debug_dir,
            debug=0,
            glctx=self.glctx,
            coarse_min_n_views=self.coarse_min_n_views,
            coarse_inplane_step=self.coarse_inplane_step,
            coarse_orientation_mode=self.coarse_orientation_mode,
            coarse_orientation_tilt_deg=self.coarse_orientation_tilt_deg,
            coarse_object_up_axis=self.coarse_object_up_axis,
        )
        
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

        # 針對 Frond 模式 4：不使用 Cutie，純依賴 SAM2 Dynamic Interactive Predictor 進行影像追蹤
        self.sam2_tracker = None
        if self.stem_pipeline_mode == 4:
            rospy.loginfo(f"載入 SAM2 Dynamic Interactive Predictor: {self.sam2_ckpt}")
            overrides = dict(conf=0.01, task="segment", mode="predict", imgsz=1024, model=self.sam2_ckpt, save=False)
            try:
                self.sam2_tracker = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=3)
            except Exception as e:
                rospy.logwarn(f"[SAM2_TRACKER] init failed: {e}")

        # 總控狀態機模式
        self.mode = "fruit"     # 初始尋找果串 (fruit)，後續依遮蔽率切換為葉莖 (frond)
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
        self.camera_tf = gp("camera_tf", "")
        self.bunch_name = gp("bunch_name", "oilpalm")
        self.stem_name = gp("stem_name", "frond")

        self.mesh_file = gp("mesh_file", "")
        self.det_model = gp("det_model", "yolov11n.onnx")
        self.yolo_start_mode = gp("yolo_start_mode", "immediate").strip().lower()
        self.debug_root = gp("debug_dir", "/tmp/fp_debug")
        self.debug_dir = self.debug_root

        # YOLO 偵測參數
        self.det_conf = float(gp("det_conf", 0.25))
        self.det_imgsz = int(gp("det_imgsz", 640))
        self.det_select_mode = gp("det_select_mode", "score").strip().lower()
        if self.det_select_mode not in ("score", "middle", "nearest_depth"):
            rospy.logwarn(
                f"[YOLO] Unknown det_select_mode={self.det_select_mode}, fallback to score"
            )
            self.det_select_mode = "score"
        self.init_det_patience = max(1, int(gp("init_det_patience", 10)))
        self.middle_depth_tie_px = float(gp("selection/middle_depth_tie_px", 40.0))

        # FoundationPose 初始化／追蹤參數
        self.est_top_k = max(1, int(gp("est_top_k", 5)))
        self.est_refine_iter = int(gp("est_refine_iter", 5))
        self.track_refine_iter = int(gp("track_refine_iter", 2))

        # Coarse initial pose sampling parameters for FoundationPose.register().
        self.coarse_min_n_views = int(gp("coarse/min_n_views", 40))
        self.coarse_inplane_step = int(gp("coarse/inplane_step", 60))
        self.coarse_orientation_mode = gp(
            "coarse/orientation_mode", "inverted"
        ).strip().lower()
        self.coarse_orientation_tilt_deg = float(
            gp("coarse/orientation_tilt_deg", 80.0)
        )
        self.coarse_object_up_axis = int(gp("coarse/object_up_axis", 1))

        if self.coarse_orientation_mode in ("none", "all"):
            self.coarse_orientation_mode = "uniform"
        if self.coarse_orientation_mode not in ("uniform", "upright", "inverted"):
            rospy.logwarn(
                f"[CoarsePoseGrid] Unknown orientation_mode={self.coarse_orientation_mode}, "
                "fallback to uniform"
            )
            self.coarse_orientation_mode = "uniform"
        if self.coarse_object_up_axis not in (0, 1, 2):
            rospy.logwarn(
                f"[CoarsePoseGrid] Unknown object_up_axis={self.coarse_object_up_axis}, "
                "fallback to 1(Y)"
            )
            self.coarse_object_up_axis = 1
        if self.coarse_inplane_step <= 0 or self.coarse_inplane_step > 360:
            rospy.logwarn(
                f"[CoarsePoseGrid] Invalid inplane_step={self.coarse_inplane_step}, "
                "fallback to 60"
            )
            self.coarse_inplane_step = 60
        if self.coarse_min_n_views <= 0:
            rospy.logwarn(
                f"[CoarsePoseGrid] Invalid min_n_views={self.coarse_min_n_views}, "
                "fallback to 40"
            )
            self.coarse_min_n_views = 40

        rospy.loginfo(
            f"[CoarsePoseGrid] launch params: min_n_views={self.coarse_min_n_views}, "
            f"inplane_step={self.coarse_inplane_step}, "
            f"orientation_mode={self.coarse_orientation_mode}, "
            f"orientation_tilt_deg={self.coarse_orientation_tilt_deg}, "
            f"object_up_axis={self.coarse_object_up_axis}, "
            f"est_top_k={self.est_top_k}"
        )

        # FoundationPose 深度幾何自我評估。
        # 重新初始化：MAE > geom_mae_thresh 或 Inlier Ratio < geom_inlier_thresh。
        # 低遮蔽：MAE <= occ_mae_thresh 且 Inlier Ratio >= occ_inlier_thresh。
        # 介於兩組門檻之間：高遮蔽。
        self.roi_expand = float(gp("roi_expand", 0.01))
        self.self_eval_log = bool(gp("self_eval_log", False))
        self.geom_mae_thresh = float(gp("geom_mae_thresh", 0.10))
        self.geom_inlier_thresh = float(gp("geom_inlier_thresh", 0.50))
        self.occ_mae_thresh = float(gp("occ_mae_thresh", 0.05))
        self.occ_inlier_thresh = float(gp("occ_inlier_thresh", 0.75))
        self.geom_patience = max(1, int(gp("geom_patience", 5)))

        if self.occ_mae_thresh > self.geom_mae_thresh:
            rospy.logwarn(
                f"[SelfCheck] occ_mae_thresh={self.occ_mae_thresh:.4f} > "
                f"geom_mae_thresh={self.geom_mae_thresh:.4f}; clamp to reinit threshold."
            )
            self.occ_mae_thresh = self.geom_mae_thresh
        if self.occ_inlier_thresh < self.geom_inlier_thresh:
            rospy.logwarn(
                f"[SelfCheck] occ_inlier_thresh={self.occ_inlier_thresh:.3f} < "
                f"geom_inlier_thresh={self.geom_inlier_thresh:.3f}; clamp to reinit threshold."
            )
            self.occ_inlier_thresh = self.geom_inlier_thresh

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

        # ------------------------------------------
        # [核心設定] 果串 (FRUIT) 追蹤管線設定
        # 1: YOLO+SAM, 2: YOLO+SAM2, 3: YOLO+FastSAM, 4: YOLO-seg
        # ------------------------------------------
        self.bunch_pipeline_mode = int(gp("postproc/fruit/pipeline_mode", 4)) 

        # ------------------------------------------
        # [核心設定] 葉莖 (FROND) 追蹤管線設定
        # 1: YOLO+SAM+Cutie, 2: YOLO+SAM2+Cutie, 3: YOLO+FastSAM+Cutie, 4: YOLO+SAM2純追蹤(無Cutie)
        # ------------------------------------------
        self.stem_pipeline_mode = int(gp("postproc/frond/pipeline_mode", 1)) 
        self.cmd_vel_topic = gp("postproc/frond/cmd_vel_topic", "/cmd_vel")
        self.stem_lost = float(gp("postproc/frond/stem_lost", 3.0))
        self.stem_depth_min = float(gp("postproc/frond/stem_depth_min", 0.1))
        self.stem_depth_max = float(gp("postproc/frond/stem_depth_max", 3.0))
        self.stem_assumed_radius = float(gp("postproc/frond/stem_assumed_radius", 0.02))
        # YOLO tracker 已停用；YOLO 只使用 predict() 做純偵測，不使用 ByteTrack / BoT-SORT。
        self.stem_max_jump_m = float(gp("postproc/frond/max_jump_m", 0.06)) 
        self.stem_ema_alpha = float(gp("postproc/frond/ema_alpha", 0.6))


        self.cls_bunch = int(gp("classes/fruit", 0))
        self.cls_stem = int(gp("classes/frond",  1))

        # 高遮蔽狀態連續出現 N 幀後切換至葉莖模式。
        self.policy_hi_pat = max(1, int(gp("policy/high_patience", 3)))

        # ------------------------------------------
        # 分割模型權重路徑設定
        # ------------------------------------------
        self.seg_model = gp("postproc/seg_model", "yolov11n-seg.pt").strip()
        self.sam_ckpt = gp("postproc/sam_ckpt", "sam_b.pt").strip()
        self.sam2_ckpt = gp("postproc/sam2_ckpt", "sam2_b.pt").strip()
        self.fastsam_ckpt = gp("postproc/fastsam_ckpt", "FastSAM-s.pt").strip()
        self.seg_imgsz = int(gp("postproc/seg_imgsz", 640))


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
            frond, ext = os.path.splitext(p)
            return frond + ".onnx"
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
        - Fruit 後續追蹤交給 FoundationPose.track_one()；Stem mode 1~3 交給 Cutie，mode 4 交給 SAM2 tracker。
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

    def _select_nearest_depth_from_indices(self,xyxy,scores,candidate_indices):
        best_i = None
        best_depth = float("inf")

        for i in candidate_indices:
            depth = self._bbox_depth_median(xyxy[i])

            if depth is not None and depth < best_depth:
                best_depth = depth
                best_i = int(i)

        if best_i is not None:
            return best_i

        # 全部沒有有效深度時用 confidence。
        return int(max(candidate_indices,key=lambda i: float(scores[i])))

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
            cx_img = W * 0.5
            cy_img = H * 0.5
            contains_center = []
            center_distances = []

            for i, bb in enumerate(xyxy_f):
                x1, y1, x2, y2 = map(float, bb)

                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)

                d = math.hypot(cx - cx_img,cy - cy_img)
                center_distances.append(d)

                if (x1 <= cx_img <= x2 and y1 <= cy_img <= y2):
                    contains_center.append(i)

            # 第一順位：包含畫面中心的 bbox。
            if contains_center:
                best_j = self._select_nearest_depth_from_indices(xyxy_f,scores_f,contains_center)

                return xyxy_f[best_j], float(scores_f[best_j])

            # 沒有 bbox 包含中心：
            # 找最靠近中心的一組候選。
            min_dist = min(center_distances)

            tie_px = float(getattr(self, "middle_depth_tie_px", 40.0))

            middle_candidates = [i for i, d in enumerate(center_distances)if d <= min_dist + tie_px]

            best_j = self._select_nearest_depth_from_indices(xyxy_f,scores_f,middle_candidates)

            return xyxy_f[best_j], float(scores_f[best_j])
        elif select_mode == "nearest_depth":
            depth_m = getattr(self, "depth_m", None)
            if depth_m is None:
                j = int(np.argmax(scores_f))
                return xyxy_f[j], float(scores_f[j])
            best_dist, best_j = 1e18, -1
            z_max = float(self.max_depth_mm) * 0.001 if hasattr(self, "max_depth_mm") else 10.0
            for i, bb in enumerate(xyxy_f):
                distance = self._bbox_depth_distance_m(bb, depth_m, z_max=z_max)
                if distance is not None and distance < best_dist:
                    best_dist, best_j = distance, i
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

    def init_via_yolo_roi(
        self,
        detector,
        color,
        depth,
        K,
        est,
        est_refine_iter,
        roi_expand,
        det_imgsz,
        det_conf,
        prefer_cls,
        do_register=True,
        lock_count=0,
        lock_target=0,
        det_xyxy=None,
        det_score=None,
        pipeline_mode=None,
    ):
        """
        以 YOLO ROI 初始化 FoundationPose。

        前段參數名稱與 foundationpose_tracker.py 的 init_via_yolo_roi() 保持一致；
        Pipeline 可額外傳入已選好的 det_xyxy / det_score，避免同一幀重複執行 YOLO。
        """
        if det_xyxy is None:
            xyxy_all, sc_all, cl_all, _ = self.yolo_det_all(
                detector, color, imgsz=det_imgsz, conf=det_conf
            )
            det_xyxy, det_score = self.select_yolo_bbox(
                xyxy_all,
                sc_all,
                cl_all,
                img_shape=color.shape,
                prefer_cls=prefer_cls,
                select_mode=self.det_select_mode_current,
                conf_th=det_conf,
            )

        if det_xyxy is None:
            return None, None, None, None

        if not do_register:
            return "locking", None, det_xyxy, det_score

        mode = self.bunch_pipeline_mode if pipeline_mode is None else int(pipeline_mode)
        mask = self.get_mask_by_mode(
            color, det_xyxy, mode, target_cls=prefer_cls
        ).astype(bool)
        if int(mask.sum()) <= 50:
            rospy.logwarn_throttle(1.0, "[INIT] Segmentation mask too small; waiting for next detection.")
            return None, mask, det_xyxy, det_score

        pose = est.register(
            K=K,
            rgb=color,
            depth=depth,
            ob_mask=mask,
            iteration=est_refine_iter,
            top_k=self.est_top_k,
        )
        return pose, mask, det_xyxy, det_score

    # =========================
    # FoundationPose 自我評估
    # =========================
    # =========================
    # Debug / visualization helpers
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

    def _class_name(self, cls_id: int) -> str:
        names = getattr(self.detector, "names", {})
        try:
            if isinstance(names, dict):
                return str(names.get(int(cls_id), int(cls_id)))
            return str(names[int(cls_id)])
        except Exception:
            return str(int(cls_id))

    def draw_yolo_detections(self, img_bgr, xyxy_all, sc_all, cl_all, selected_xyxy=None, target_cls=None):
        """
        顯示指定類別的 YOLO 偵測框。
        target_cls:
            None 顯示全部類別
            self.cls_bunch 只顯示果串
            self.cls_stem 只顯示葉莖
        """
        if img_bgr is None or xyxy_all is None:
            return img_bgr
        vis = img_bgr
        selected = (
            None
            if selected_xyxy is None
            else np.asarray(selected_xyxy, dtype=np.float32)
        )

        for bb, score, cls_id in zip(xyxy_all, sc_all, cl_all):
            # 只畫指定類別
            if target_cls is not None and int(cls_id) != int(target_cls):
                continue

            bb = self.clip_xyxy(bb,vis.shape[1],vis.shape[0],)

            is_selected = (
                selected is not None
                and np.allclose(bb, selected, atol=2.0)
            )

            color = (0, 255, 255) if is_selected else (0, 255, 0)
            thick = 3 if is_selected else 2
            label = (
                f"{self._class_name(int(cls_id))} "
                f"{float(score):.2f}"
            )

            self._draw_rect(vis,bb,color=color,thick=thick,label=label,)

        return vis

    def evaluate_depth_self_check(self, depth_mae, inlier_ratio):
        """
        將 FoundationPose 自我評估結果分成三類：
        - REINIT_CANDIDATE：MAE 太高或 Inlier Ratio 太低。
        - HIGH_OCCLUSION：介於重新初始化門檻與低遮蔽門檻之間。
        - LOW_OCCLUSION：MAE 低且 Inlier Ratio 高。
        """
        if depth_mae is None or inlier_ratio is None:
            return "UNAVAILABLE", False, False, None

        mae = float(depth_mae)
        inlier = float(inlier_ratio)
        tau_d = max(float(self.geom_mae_thresh), 1e-6)
        score = float(np.clip(inlier * np.exp(-mae / tau_d), 0.0, 1.0))

        reinit_candidate = (
            inlier < float(self.geom_inlier_thresh)
            or mae > float(self.geom_mae_thresh)
        )
        low_occlusion = (
            inlier >= float(self.occ_inlier_thresh)
            and mae <= float(self.occ_mae_thresh)
        )

        if reinit_candidate:
            return "REINIT_CANDIDATE", True, False, score
        if low_occlusion:
            return "LOW_OCCLUSION", False, False, score
        return "HIGH_OCCLUSION", False, True, score

    def _draw_pose_box(self, img_bgr, K, pose_obj_in_cam, bbox_minmax, which, axis_scale=0.05):
        if which.lower() != "fruit": return img_bgr
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
    # 葉莖 (Frond) 實體追蹤與深度計算
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
        從 frond mask + depth 建立與 FoundationPose 相同定義的 4x4 pose。

        統一定義：回傳的 pose 是 T_camera_stem，也就是 frond 物件座標系在
        camera_color_optical_frame 下的位置與姿態。矩陣欄向量定義如下：
            T[:3, 0] = +X_stem，顯示為紅色
            T[:3, 1] = +Y_stem，顯示為綠色
            T[:3, 2] = +Z_stem，顯示為藍色

        純分割沒有 CAD mesh，因此這裡用穩定的幾何規則定義 frond frame：
            origin  = mask 最下方的 root / cut point
            +Y      = root 指向 frond 上方的中心線方向
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

        # 2) +Y_stem：root 指向 frond 上方中心線。
        #    取 mask 較上方的一小群點，比單點 PCA 更穩。
        top_k = min(30, len(sorted_indices))
        top_indices = sorted_indices[-top_k:]
        top_u = float(np.median(u_valid[top_indices]))
        top_v = float(np.median(v_valid[top_indices]))
        top_z = float(np.median(z_valid[top_indices]))
        p_top = self._backproject_uvz(top_u, top_v, top_z)
        y_axis = self._normalize_vec(p_top - p_surface, fallback=np.array([0.0, -1.0, 0.0]))

        # 3) +Z_stem：用相機視線方向固定 frond 的 roll，避免每幀亂轉。
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

        except Exception as e:
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
        FROND 抓取候選選擇策略。
        優先順序：
        1. 距離相機最近的 stem。
        2. 若 frond 深度不可用，選離最近距離 fruit 最近的 stem。
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
                f"[FROND SELECT] nearest-depth frond selected: depth={stem_depth:.3f}m, score={stem_score:.3f}"
            )
            return stem_xyxy, stem_score

        # 第二順位：找最近距離 bunch，再選離該 fruit 中心最近的 stem。
        bunch_xyxy, bunch_score, bunch_depth = self.select_nearest_depth_bbox_for_cls(
            xyxy_all, sc_all, cl_all, self.cls_bunch, conf_th=self.det_conf
        )

        stem_idx = np.where(cl_all == int(self.cls_stem))[0]
        if stem_idx.size == 0:
            return None, None

        # 如果沒有有效 fruit depth，改用最高分 fruit 作為幾何參考。
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
                    f"[FROND SELECT] fallback nearest-to-nearest-fruit selected: "
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
                f"[FROND SELECT] fallback highest-score frond selected: score={stem_score:.3f}"
            )

        return stem_xyxy, stem_score

    # ==========================================
    # GUI 與畫面管理
    # ==========================================
    def _open_window(self, name, pos_xy, init_size, is_rgb=True):
        w, h = init_size
        try: cv2.destroyWindow(name)
        except Exception: pass
        # print(f"準備建立視窗: {name}")
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        # print(f"視窗建立成功: {name}")
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
    # ROS 工具 / Callbacks
    # =========================
    def mat4_to_translation_quat(self, T: np.ndarray):
        t = (float(T[0,3]), float(T[1,3]), float(T[2,3]))
        qx,qy,qz,qw = tf.transformations.quaternion_from_matrix(T)
        return t, (float(qx),float(qy),float(qz),float(qw))

    def broadcast_transform_and_pose(self, T: np.ndarray, which: str, parent: str):
        child = self.bunch_name if which.lower() == "fruit" else self.stem_name
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

        allowed = bool(getattr(msg, "detection_allowed", False))

        previous_select_mode = self.det_select_mode_current
        self.det_select_mode_current = mode
        self.ready_received.detection_allowed = allowed

        # middle：車體對位完成，鎖定當下 fruit / frond
        if allowed and mode == "middle":
            if not self.target_mode_locked:
                locked_mode = (self.mode if self.mode in ("fruit", "frond")else "fruit")

                self.target_mode_locked = True
                self.locked_target_mode = locked_mode
                self.mode = locked_mode

                # middle 後不再累積遮蔽率切換計數。
                self._hi_cnt = 0
                rospy.logwarn(
                    f"[TARGET MODE LOCK] middle received -> "
                    f"lock mode={self.locked_target_mode}"
                )
            else:
                self.mode = self.locked_target_mode

        # nearest_depth / score：回到找目標階段，解除鎖定
        elif allowed and mode in ("nearest_depth", "score"):
            if self.target_mode_locked:
                rospy.logwarn(
                    f"[TARGET MODE UNLOCK] "
                    f"{previous_select_mode} -> {mode}"
                )

            self.target_mode_locked = False
            self.locked_target_mode = None

    def harvestDoneCallback(self, msg: Bool):
        if not bool(msg.data):
            return
        rospy.logwarn("[HARVEST_DONE] received True -> hard reset to fruit")
        self._reset_all_to_bunch()
        self._publish_zero_current(used_seg=False)

    def _tag(self, which: str) -> str:
        return "fruit" if which.lower() == "fruit" else "frond"

    def _state(self, which: str, used_seg: bool = False) -> str:
        part = self._tag(which)
        allowed = bool(getattr(self.ready_received, "detection_allowed", False))

        if self.yolo_start_mode == "wait" and (not allowed):
            return f"{part}:PAUSED"

        if not (self.got_depth and self.got_rgb and self.K is not None and hasattr(self, "depth_m")):
            return f"{part}:INITIALIZING"

        if which.lower() == "fruit":
            if self.pose_bunch is None:
                return f"{part}:YOLO"
            return f"{part}:STABLE"
        else:
            if self.pose_stem is None:
                return f"{part}:YOLOSEG"
            return f"{part}:STABLE"

    def confidence_publish(self, which: str, confidence_score: float, detection: bool, used_seg: bool = False):
        conf_msg = Confidence()
        conf_msg.stamp = rospy.Time.now()
        conf_msg.state = self._state(which, used_seg=used_seg)
        conf_msg.frame_id = self.bunch_name if which.lower() == "fruit" else self.stem_name
        # forklift_msg/Confidence 欄位名稱仍為 object_IoU；此處改放深度幾何自我評估分數。
        conf_msg.object_IoU = float(confidence_score)
        conf_msg.object_detection = bool(detection)

        T = self.pose_bunch if which.lower() == "fruit" else self.pose_stem
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
        which = "frond" if self.mode == "frond" else "fruit"
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
        self.mode = "fruit"
        self.pose_bunch = None
        self.pose_stem  = None
        self._hi_cnt = 0
        self.target_mode_locked = False
        self.locked_target_mode = None
        self._stem_lock = False
        self.target_stem_id = -1

        self.depth_mae = None
        self.inlier_ratio = None
        self.depth_conf_score = None
        self.geom_bad_count = 0
        self.geom_state = "UNAVAILABLE"
        self.geom_high_occlusion = False

        self.last_stem_3d_pos = None
        self.last_stem_time = None

        self.last_bunch_3d_pos = None

        self.stem_cutie_state = "CRUISING"
        # Cutie 僅服務 Stem；整體重置時仍需清除 Frond 的時序記憶。
        self._reset_cutie_processor(reason="_reset_pipeline_state")

        if getattr(self, "sam2_tracker", None) is not None:
            try:
                self.sam2_tracker.reset_memory()
            except Exception as e:
                rospy.logwarn(f"[SAM2 Reset] reset_memory failed: {e}")

    def _reset_all_to_bunch(self):
        rospy.logwarn("[RESET] reset all pipeline state to FRUIT / CRUISING")
        self._reset_pipeline_state()
        self.consecutive_det_count = 0
        self._last_yolo_text = ""
        self._registering_until = 0
        self._reinit_until = 0

    def _reset_stem_to_cruising(self, reason=""):
        """
        FROND tracking loss 時使用。
        重點：
        - 不切回 bunch，仍停留在 self.mode == "frond"。
        - 清掉目前 frond pose / tracking memory。
        - 下一幀回到 FROND CRUISING，重新執行 YOLO + SAM/SAM2/FastSAM，
          重新給 Cutie 第一幀 mask。
        """
        rospy.logwarn(f"[FROND RESET] reset frond to CRUISING. reason={reason}")

        self.pose_stem = None
        self._stem_lock = False
        self.target_stem_id = -1
        self.last_stem_3d_pos = None
        self.last_stem_time = None
        self.stem_cutie_state = "CRUISING"

        self.confidence_publish("frond", 0.0, False, used_seg=False)

        # Mode 1~3：Cutie tracking loss，重建 InferenceCore，避免 memory / sensory state 殘留。
        if self.stem_pipeline_mode != 4:
            try:
                self._reset_cutie_processor(reason=f"frond lost: {reason}")
            except Exception as e:
                rospy.logwarn(f"[FROND RESET] Cutie reset failed: {e}")
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
                rospy.logwarn(f"[FROND RESET] SAM2 reset_memory failed: {e}")
    
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
        """
        使用與 foundationpose_tracker.py 相同的 init_det_patience 邏輯。

        YOLO 必須連續偵測到果串 N 幀才允許 FoundationPose.register()；
        任一幀未偵測到目標即將 consecutive_det_count 歸零。
        """
        if which != "fruit":
            return (xyxy_now is not None), xyxy_now, 0

        if xyxy_now is None:
            self.consecutive_det_count = 0
            return False, None, 0

        self.consecutive_det_count += 1
        lock_count = self.consecutive_det_count
        bbox_use = np.asarray(xyxy_now, dtype=np.float32).copy()
        do_register = lock_count >= self.init_det_patience

        if do_register:
            self.consecutive_det_count = 0

        return do_register, bbox_use, lock_count

    # ==========================================
    # ROS 節點主循環 (Main Spin)
    # 控管所有硬體與演算法的互動流程
    # ==========================================
    def spin(self):
        self.frame_count = 0
        used_seg = False

        while not rospy.is_shutdown():
            self_eval_for_publish = 0.0
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
            
            # Fruit 的 FoundationPose 追蹤不需要 Cutie，也不需要建立 Cutie 輸入 tensor。
            # 只有 Frond mode 1~3 真正執行 Cutie 時，才在下方建立 img_tensor。
            vis_bgr = self.color.copy()

            # --- 判斷目前狀態是否需要運行 YOLO 來重新尋找物體 ---
            run_yolo = False
            if self.mode == "fruit" and self.pose_bunch is None:
                run_yolo = True
            elif self.mode == "frond" and self.stem_cutie_state == "CRUISING":
                run_yolo = True

            xyxy_all, sc_all, cl_all, ids_all = None, None, None, None
            if run_yolo:
                xyxy_all, sc_all, cl_all, ids_all = self.yolo_det_all(self.detector, self.color, imgsz=self.det_imgsz, conf=self.det_conf)
                
                # if self.show_rgb_win:
                #     vis_bgr = self.draw_yolo_detections(vis_bgr, xyxy_all, sc_all, cl_all)

            # Cutie 僅供 Frond mode 1~3 使用；標籤 2 代表葉莖。
            cutie_init_mask = np.zeros((self.rgb_size[1], self.rgb_size[0]), dtype=np.uint8)
            cutie_init_objs = []

            # ==========================================
            # 1. FRUIT CRUISING (全域搜尋果串)
            # ==========================================
            if self.mode == "fruit" and self.pose_bunch is None:
                bunch_xyxy, bunch_conf = self.select_yolo_bbox(
                    xyxy_all, sc_all, cl_all, img_shape=self.color.shape,
                    prefer_cls=self.cls_bunch, select_mode=self.det_select_mode_current, conf_th=self.det_conf
                )
                do_register, bb_use, lock_count = self._yolo_delay_update(
                    "fruit", bunch_xyxy
                )

                if self.show_rgb_win:
                    vis_bgr = self.draw_yolo_detections(vis_bgr,xyxy_all,sc_all,cl_all,selected_xyxy=bunch_xyxy,target_cls=self.cls_bunch,)
                    cv2.putText(vis_bgr,f"Locking ROI... {lock_count}/{self.init_det_patience}",(10, 70),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 255),2,cv2.LINE_AA,)

                if bunch_xyxy is not None:
                    init = self.init_via_yolo_roi(
                        self.detector,
                        self.color,
                        self.depth_m,
                        self.K,
                        self.est_bunch,
                        self.est_refine_iter,
                        self.roi_expand,
                        self.det_imgsz,
                        self.det_conf,
                        self.cls_bunch,
                        do_register=do_register,
                        lock_count=lock_count,
                        lock_target=self.init_det_patience,
                        det_xyxy=bb_use,
                        det_score=bunch_conf,
                        pipeline_mode=self.bunch_pipeline_mode,
                    )
                else:
                    init = (None, None, None, None)

                if not do_register:
                    self._publish_zero_current(used_seg=False)
                elif init[0] is not None:
                    self.pose_bunch, used_mask, _, _ = init
                    self.mask = used_mask
                    self.geom_bad_count = 0
                    self.geom_state = "REGISTERED"
                    self.depth_mae = None
                    self.inlier_ratio = None
                    self.depth_conf_score = None
                    rospy.loginfo(
                        f"[FRUIT INIT] YOLO -> segmentation mode {self.bunch_pipeline_mode} "
                        f"-> FoundationPose.register(top_k={self.est_top_k}); "
                        f"Coarse={self.coarse_orientation_mode}."
                    )

            # ==========================================
            # 2. FROND CRUISING (根據機器人軌跡搜尋葉莖)
            # ==========================================
            predicted_2d = None
            if self.mode == "frond":
                # 計算先前的物理預測落點
                predicted_3d = self.get_predicted_stem_pos(now)
                if predicted_3d is not None and predicted_3d[2] > 0:
                    u = int(self.K[0,0] * predicted_3d[0] / predicted_3d[2] + self.K[0,2])
                    v = int(self.K[1,1] * predicted_3d[1] / predicted_3d[2] + self.K[1,2])
                    predicted_2d = (u, v)
                    if self.show_rgb_win: cv2.circle(vis_bgr, (u, v), 5, (255, 0, 0), -1)

                if self.stem_cutie_state == "CRUISING" and run_yolo:
                    # FROND 重新初始化不使用 YOLO tracker ID。
                    # 優先順序：
                    #   1) 距離相機最近的 stem。
                    #   2) 若 frond 深度不可用，選離最近距離 fruit 最近的 stem。
                    #   3) 若都不可用，退回最高分 stem。
                    stem_xyxy, stem_score = self.select_stem_bbox_priority(xyxy_all, sc_all, cl_all)

                    if stem_xyxy is None and predicted_2d is not None:
                        # 最後補救：若 YOLO 有 frond 但深度/果串參考都失敗，
                        # 用上一幀運動預測點找最接近的 stem。
                        stem_xyxy, stem_score, _ = self.pick_nearest_to_2d_point(xyxy_all, sc_all, cl_all, ids_all, predicted_2d, self.cls_stem)

                        if self.show_rgb_win:
                            vis_bgr = self.draw_yolo_detections(vis_bgr,xyxy_all,sc_all,cl_all,selected_xyxy=stem_xyxy,target_cls=self.cls_stem,)

                        if stem_xyxy is None and self.last_stem_time is not None and (now - self.last_stem_time) > self.stem_lost_timeout:
                            self.last_stem_3d_pos = None
                            self.last_stem_time = None

                    if stem_xyxy is not None:
                        # [核心修改] 根據 Frond Mode (1~4) 進行分流處理
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
                                f"[Hand-off] YOLO -> Locked FROND "
                                f"(Mode: {self.stem_pipeline_mode}, score={stem_score})!"
                            )
                            self.stem_cutie_state = "SERVOING"
                            self.pose_stem = None
                            # 只有 Mode 1~3 需要把 Mask 塞進 Cutie 陣列
                            if self.stem_pipeline_mode != 4:
                                cutie_init_mask[used_mask] = 2
                                cutie_init_objs.append(2)

            # ==========================================
            # 3. 執行 Frond Cutie 推論（Bunch 完全不使用 Cutie）
            # ==========================================
            pred_mask_tensor = None
            stem_needs_cutie = (
                self.mode == "frond"
                and self.stem_pipeline_mode != 4
                and (len(cutie_init_objs) > 0 or self.stem_cutie_state == "SERVOING")
            )
            if stem_needs_cutie and getattr(self, "cutie_processor", None) is not None:
                img_rgb = cv2.cvtColor(self.color, cv2.COLOR_BGR2RGB)
                img_tensor = F.to_tensor(img_rgb).cuda().float()

                # 情況 A：YOLO + segmentation 剛交接，將 Frond 第一幀 mask 寫入 Cutie。
                if len(cutie_init_objs) > 0:
                    init_mask_tensor = torch.from_numpy(cutie_init_mask).cuda().long()
                    with torch.cuda.amp.autocast(), torch.inference_mode():
                        output_prob = self.cutie_processor.step(
                            img_tensor, init_mask_tensor, objects=cutie_init_objs
                        )
                        pred_mask_tensor = self.cutie_processor.output_prob_to_mask(output_prob)

                # 情況 B：Stem mode 1~3 的後續 Cutie 傳播追蹤。
                elif self.stem_cutie_state == "SERVOING":
                    with torch.cuda.amp.autocast(), torch.inference_mode():
                        output_prob = self.cutie_processor.step(img_tensor)
                        pred_mask_tensor = self.cutie_processor.output_prob_to_mask(output_prob)

            # ==========================================
            # 4. FRUIT FOUNDATIONPOSE TRACKING
            #    register() 後直接 track_one(enable_self_check=True)，不使用 Cutie。
            # ==========================================
            self_eval_for_publish = 0.0
            if self.mode == "fruit" and self.pose_bunch is not None:
                used_seg = False  # segmentation 只在 register / re-register 時使用
                track_extra = {}

                try:
                    tracked_pose = self.est_bunch.track_one(
                        rgb=self.color,
                        depth=self.depth_m,
                        K=self.K,
                        iteration=self.track_refine_iter,
                        extra=track_extra,
                        enable_self_check=True,
                    )
                except Exception as e:
                    rospy.logerr_throttle(1.0, f"[FRUIT][FoundationPose] track_one failed: {e}")
                    tracked_pose = None

                if tracked_pose is None:
                    # track_one 本身失敗也視為一幀異常。
                    self.geom_state = "TRACK_FAILED"
                    self.depth_mae = None
                    self.inlier_ratio = None
                    self.depth_conf_score = None
                    self.geom_bad_count += 1
                    reinit_candidate = True
                    # track_one 失敗歸類為重新初始化候選，不先切換 Stem。
                    high_occlusion = False
                else:
                    self.pose_bunch = tracked_pose
                    self.depth_mae = track_extra.get("depth_mae", None)
                    self.inlier_ratio = track_extra.get("inlier_ratio", None)
                    (
                        self.geom_state,
                        reinit_candidate,
                        high_occlusion,
                        self.depth_conf_score,
                    ) = self.evaluate_depth_self_check(
                        self.depth_mae, self.inlier_ratio
                    )

                    if self.depth_conf_score is not None:
                        self_eval_for_publish = float(self.depth_conf_score)

                    if reinit_candidate:
                        self.geom_bad_count += 1
                    else:
                        self.geom_bad_count = 0

                reinit_by_geometry = self.geom_bad_count >= self.geom_patience
                if reinit_by_geometry:
                    self._hi_cnt += 1
                    mae_text = (
                        "None" if self.depth_mae is None else f"{self.depth_mae:.4f}m"
                    )
                    inlier_text = (
                        "None" if self.inlier_ratio is None else f"{self.inlier_ratio:.3f}"
                    )
                    rospy.logwarn(
                        f"[FRUIT][SelfCheck] consecutive abnormal frames "
                        f"{self.geom_bad_count}/{self.geom_patience}: "
                        f"Inlier={inlier_text}, MAE={mae_text}. "
                        "Discard pose and restart YOLO -> segmentation -> FoundationPose.register()."
                    )

                    # 不使用 Cutie mask 直接重註冊。清除姿態後，下一幀由 run_yolo
                    # 自動重新走 init_det_patience、分割與 register(top_k)。
                    self.pose_bunch = None
                    self.mask = None
                    self.consecutive_det_count = 0
                    self.geom_bad_count = 0
                    self.geom_state = "REINIT_REQUIRED"
                    self.geom_high_occlusion = True
                    self.depth_conf_score = None
                    self.confidence_publish("fruit", 0.0, False, used_seg=False)

                    if self.show_rgb_win:
                        cv2.putText(vis_bgr,"FRUIT REINIT: waiting for YOLO + segmentation",(10, 100),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0, 0, 255),2,cv2.LINE_AA)
                else:
                    # 中間品質區間視為高遮蔽；低 MAE 且高 Inlier Ratio 視為低遮蔽。
                    # middle 後不再用遮蔽率切換 fruit / stem。
                    if self.target_mode_locked:
                        self._hi_cnt = 0
                        if self.locked_target_mode in ("fruit", "frond"):
                            self.mode = self.locked_target_mode
                        rospy.loginfo_throttle(1.0,f"[OCCLUSION POLICY DISABLED] target mode locked={self.locked_target_mode}")

                    else:
                        if high_occlusion:
                            self._hi_cnt += 1
                        else:
                            self._hi_cnt = 0

                        if self._hi_cnt >= self.policy_hi_pat:
                            self.mode = "frond"
                            self.stem_cutie_state = "CRUISING"
                            self._reset_cutie_processor(reason="mode switch fruit -> frond")

                            rospy.logwarn("[MODE SWITCH] MAE / Inlier Ratio 判定為高遮蔽，切換至 FROND 模式。")

                    if self.pose_bunch is not None:
                        self.last_bunch_3d_pos = self.pose_bunch[:3, 3].copy()
                        parent_frame = (
                            self.camera_tf
                            if self.camera_tf
                            else "camera_color_optical_frame"
                        )
                        self.broadcast_transform_and_pose(
                            self.pose_bunch, "fruit", parent_frame
                        )
                        self.confidence_publish(
                            "fruit", self_eval_for_publish, True, used_seg=False
                        )

                        if self.show_rgb_win:
                            center_pose = self.pose_bunch @ np.linalg.inv(
                                self.to_origin_bunch
                            )
                            vis_bgr = draw_posed_3d_box(
                                self.K,
                                img=cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB),
                                ob_in_cam=center_pose,
                                bbox=self.bbox_bunch,
                            )
                            vis_bgr = draw_xyz_axis(
                                vis_bgr,
                                ob_in_cam=self.pose_bunch,
                                scale=0.05,
                                K=self.K,
                                thickness=3,
                                transparency=0,
                                is_input_rgb=True,
                            )
                            vis_bgr = cv2.cvtColor(vis_bgr, cv2.COLOR_RGB2BGR)

                            if self.depth_mae is not None and self.inlier_ratio is not None:
                                cv2.putText(
                                    vis_bgr,
                                    f"SelfCheck={self.geom_state}  "
                                    f"Inlier={self.inlier_ratio:.3f}  "
                                    f"MAE={self.depth_mae * 100.0:.1f}cm  "
                                    f"bad={self.geom_bad_count}/{self.geom_patience}",
                                    (10, vis_bgr.shape[0] - 48),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (0, 0, 255),
                                    1,
                                    cv2.LINE_AA,
                                )
                            else:
                                cv2.putText(
                                    vis_bgr,
                                    "SelfCheck unavailable or track failed",
                                    (10, vis_bgr.shape[0] - 48),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 255),
                                    1,
                                    cv2.LINE_AA,
                                )

            # ==========================================
            # 5. FROND SERVOING (依據 2D Mask 計算葉莖 3D 切割點)
            # ==========================================
            if self.mode == "frond" and self.stem_cutie_state == "SERVOING":
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
                            #     rospy.logwarn_throttle(0.5, "[frond] REJECTED: Foreground leaf (Z too close).")
                            #     center_3d = None 

                            # 3D 防呆 2: 防止空間跳躍 (過濾旁邊干擾的樹枝)
                            if center_3d is not None and predicted_3d is not None:
                                jump_dist = float(np.linalg.norm(center_3d - predicted_3d))
                                if jump_dist > self.stem_max_jump_m:  
                                    rospy.logwarn_throttle(0.5, f"[frond] REJECTED: Jumped {jump_dist:.2f}m")
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
                                    
                                    # 修正：計算相機到目標的三維直線距離 (歐式距離 = 根號(X^2 + Y^2 + Z^2))
                                    true_dist = float(np.linalg.norm(smoothed_3d))
                                    
                                    cv2.putText(vis_bgr, f"Distance: {true_dist:.2f}m", (projected_2d[0] + 10, projected_2d[1] - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                
                                parent_frame = self.camera_tf if self.camera_tf else "camera_color_optical_frame"
                                self.broadcast_transform_and_pose(self.pose_stem, "frond", parent_frame)
                                self.confidence_publish("frond", 1.0, True, used_seg=True)
                                
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
                if self.mode == "fruit" and self.pose_bunch is not None:
                    cv2.putText(
                        vis_bgr,
                        f"FRUIT: FOUNDATIONPOSE TRACKING (Init Mode {self.bunch_pipeline_mode})",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                elif self.mode == "frond":
                    cv2.putText(vis_bgr, f"FRUIT: PAUSED (FROND MODE)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

                if self.mode == "frond" and self.stem_cutie_state == "SERVOING":
                    tracker_name = "SAM2" if self.stem_pipeline_mode == 4 else "CUTIE"
                    cv2.putText(vis_bgr, f"FROND: {tracker_name} LOCKED (Mode {self.stem_pipeline_mode})", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                self.draw_conf_bar(vis_bgr,{"Inlier Ratio": 0.0 if self.inlier_ratio is None else self.inlier_ratio},label="Inlier Ratio",origin=(10, vis_bgr.shape[0] - 28),size=(220, 18),max_val=1.0,)
                
            self.pump_windows(
                vis_bgr if (self.show_rgb_win and self.color is not None) else None,
                self.depth_vis if (self.show_depth_win and self.got_depth and self.depth_vis is not None) else None
            )

        cv2.destroyAllWindows()

if __name__ == "__main__":
    rospy.init_node("pipeline_tracker", anonymous=False)
    node = FoundationPosePipelineTracker()
    node.spin()
