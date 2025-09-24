#!/home/user/anaconda3/envs/foundationpose/bin/python3
# -*- coding: utf-8 -*-

import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用哪一個GPU
import rospy
import numpy as np
import cv2
import trimesh
import torch

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped, Transform
from cv_bridge import CvBridge
from ros_foundationpose.msg import Confidence
from forklift_server.msg import Detection

from collections import deque
import tf

# --- 第三方與專案路徑（依你的環境調整） ---
sys.path.append('/home/user/anaconda3/envs/foundationpose/lib/python3.8/site-packages')
FOUNDATIONPOSE_SRC = "/home/user/FoundationPose"
if FOUNDATIONPOSE_SRC not in sys.path:
    sys.path.append(FOUNDATIONPOSE_SRC)

os.environ.setdefault("ULTRALYTICS_NO_INSTALL", "1")

from estimater import FoundationPose, draw_posed_3d_box, draw_xyz_axis, ScorePredictor, PoseRefinePredictor, dr
from ultralytics import YOLO

selecting_bbox = False
box_points = []

class FoundationPoseTracker:
    def __init__(self):
        self.init_parameter()
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

        self._rgb_win_created = False
        self._depth_win_created = False
        self._rgb_win_sized = False
        self._depth_win_sized = False
        self._rgb_initial_size = (900, 720)
        self._depth_initial_size = (900, 720)

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
        os.makedirs(self.debug_dir, exist_ok=True)
        self.mesh = trimesh.load(self.mesh_file)
        self.to_origin, self.extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.to_origin = self.to_origin
        self.bbox = np.stack([-self.extents/2, self.extents/2], axis=0).reshape(2, 3)

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(model_pts=self.mesh.vertices, model_normals=self.mesh.vertex_normals, mesh=self.mesh, scorer=self.scorer, refiner=self.refiner, debug_dir=self.debug_dir, debug=0, glctx=self.glctx,)
        rospy.loginfo("Estimator initialization done")

        self.detector, self.det_device = self.load_detector(self.det_model)
        is_gpu, yolo_desc = self.yolo_uses_gpu(self.detector)
        rospy.loginfo(f"[YOLO] GPU enabled: {is_gpu}  ({yolo_desc})")
        rospy.loginfo(f"[YOLO] predict device hint: {self.det_device}")
        rospy.loginfo("Detector initialization done")

    def init_parameter(self):
        # 參數
        ns = rospy.get_name()
        gp = lambda k, d: rospy.get_param(ns + "/" + k, d)

        self.image_topic = gp("image_topic", "/camera/color/image_raw")
        self.info_topic = gp("info_topic",  "/camera/color/camera_info")
        self.depth_topic = gp("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.depth_info_topic = gp("depth_info_topic", "")
        self.camera_tf = gp("camera_tf", "")
        self.object_name = gp("object_name", "tracked_object")

        self.mesh_file = gp("mesh_file", "")
        self.det_model = gp("det_model", "yolov11n.onnx")
        self.init_mode = gp("init_mode", "yolo")
        self.yolo_start_mode = gp("yolo_start_mode", "immediate").strip().lower()
        self.det_conf = float(gp("det_conf", 0.25))
        self.det_class = int(gp("det_class", -1))
        self.est_refine_iter = int(gp("est_refine_iter", 5))
        self.track_refine_iter = int(gp("track_refine_iter", 2))
        self.debug_dir = gp("debug_dir", "/tmp/fp_debug")

        self.roi_expand = float(gp("roi_expand", 0.02))
        self.iou_stride = int(gp("iou_stride", 1))
        self.iou_log = bool(gp("iou_log", True))
        self.iou_thresh = float(gp("iou_thresh", 0.2))
        self.iou_patience = int(gp("iou_patience", 3))
        self.det_imgsz = int(gp("det_imgsz", 640))
        self.prefer_cls = None if self.det_class < 0 else self.det_class

        self.det_select_mode = gp("det_select_mode", "score").strip().lower()
        if self.det_select_mode not in ("score", "top", "bottom"):
            rospy.logwarn("Unknown det_select_mode=%s, fallback to 'score'", self.det_select_mode)
            self.det_select_mode = "score"

        self.show_depth_win = bool(gp("show_depth_window", False))
        self.show_rgb_win = bool(gp("show_rgb_window", True))
        self.depth_win_name = gp("depth_win_name", "Depth Image")
        self.rgb_win_name = gp("rgb_win_name", "RGB Image")
        self.depth_win_xy = gp("depth_window_xy", [100, 100])  # [x, y]
        self.rgb_win_xy = gp("rgb_window_xy", [100, 500])  # [x, y]

        self.max_depth_mm = float(gp("max_depth_mm", 10000.0))  # 0~10000mm 會映射到 0~255
        self.colormap_id = int(gp("colormap", int(cv2.COLORMAP_JET)))  # OpenCV colormap 常數
        self.invert_colormap = bool(gp("invert_colormap", False))
    
    # =========================
    # YOLO 後端/裝置資訊工具
    # =========================
    def yolo_backend_info(self,detector):
        """回傳 (engine, torch_device, ort_providers) 三項資訊，涵蓋 .pt/.onnx/等"""
        info = {"engine": None, "torch_device": None, "ort_providers": None}
        m = getattr(detector, "model", None)
        if m is None:
            return info

        # 可能存在的後端描述
        info["engine"] = str(getattr(m, "backend", getattr(m, "engine", "")))

        # PyTorch：檢查參數是否在CUDA
        try:
            if hasattr(m, "parameters"):
                p = next(m.parameters(), None)
                if p is not None:
                    info["torch_device"] = str(p.device)
        except Exception:
            pass

        # ONNX Runtime：檢查 providers
        sess = None
        for attr in ("session", "ort_session", "session_ort"):
            s = getattr(m, attr, None)
            if s is not None:
                sess = s
                break
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
        """回傳 (is_gpu: bool, 描述字串)"""
        info = self.yolo_backend_info(detector)
        eng = (info["engine"] or "").lower()
        dev = (info["torch_device"] or "").lower()
        prov = info["ort_providers"] or []

        # ONNX Runtime：看到 CUDAExecutionProvider 就代表用 GPU
        if any("CUDAExecutionProvider" == p for p in prov):
            return True, f"engine={eng or 'onnxruntime'} providers={prov}"

        # TensorRT（若用 trt 匯出）
        if "tensorrt" in eng:
            return True, f"engine={eng}"

        # PyTorch：參數在 cuda 裝置上
        if dev.startswith("cuda"):
            return True, f"engine={eng or 'pytorch'} device={dev}"

        return False, f"engine={eng or 'unknown'} providers={prov or 'None'} device={dev or 'None'}"

    def load_detector(self, model_path: str):
        """
        載入偵測模型（僅支援 .pt / .onnx）
        規則：
          - .pt：優先 GPU（model.to('cuda') / predict(device=0)），失敗則 CPU；若載入本身失敗且旁邊有同名 .onnx，切到 .onnx
          - .onnx：優先 CUDAExecutionProvider，失敗則 CPUExecutionProvider
        回傳: (detector, det_device_str)
           det_device_str 會給 predict() 當 device 參數（對 .pt 有效；.onnx 會被忽略）
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
            # 先試 GPU
            try:
                det = YOLO(model_path)   # 會自動推斷任務
                if torch.cuda.is_available():
                    try:
                        det.to("cuda:0")
                        det_device = 0  # predict(device=0)
                        rospy.loginfo("[YOLO Loader] PT on GPU OK")
                        return det, det_device
                    except Exception as ge:
                        rospy.logwarn(f"[YOLO Loader] PT move to GPU failed: {ge}. Will try PT on CPU.")
                else:
                    rospy.logwarn("[YOLO Loader] No CUDA available; PT will run on CPU.")
                # 再試 CPU
                try:
                    det.to("cpu")
                    det_device = "cpu"
                    rospy.loginfo("[YOLO Loader] PT on CPU OK")
                    return det, det_device
                except Exception as ce:
                    rospy.logwarn(f"[YOLO Loader] PT on CPU failed: {ce}")

            except Exception as e:
                rospy.logwarn(f"[YOLO Loader] PT load failed: {e}")

            # 到這裡表示 .pt 失敗，若旁邊有同名 .onnx 就用它
            onnx_fallback = _onnx_sibling(model_path)
            if os.path.isfile(onnx_fallback):
                rospy.loginfo(f"[YOLO Loader] Trying fallback ONNX: {onnx_fallback}")
                det, det_device = self._load_onnx_with_gpu_fallback(onnx_fallback)
                return det, det_device
            else:
                raise RuntimeError(f"Failed to load PT '{model_path}' and no sibling ONNX found.\n"
                                    "Hint: 這通常是 .pt 與 ultralytics 版本不相容；請改用 .onnx 或調整 ultralytics 版本。")
        elif ext == ".onnx":
            rospy.loginfo(f"[YOLO Loader] Loading ONNX: {model_path}")
            det, det_device = self._load_onnx_with_gpu_fallback(model_path)
            return det, det_device
        else:
            raise ValueError(f"Unsupported detector extension: {ext}. Use .pt or .onnx")
        
    def _load_onnx_with_gpu_fallback(self, onnx_path: str):
        """
        優先用 CUDAExecutionProvider；若不可用或失敗則改 CPUExecutionProvider。
        回傳: (detector, det_device_str)；對 ONNX 來說 det_device_str 只是提示字串。
        """
        # 先照常載入
        det = YOLO(onnx_path, task="detect")

        # 嘗試檢查/切換 providers
        sess = None
        for attr in ("session", "ort_session", "session_ort"):
            s = getattr(det.model, attr, None)
            if s is not None:
                sess = s
                break

        # 預設提示
        det_device = "cpu(ORT)"

        if sess is not None:
            try:
                provs = list(sess.get_providers())
            except Exception:
                provs = []

            # 若已經有 CUDA provider 就直接用
            if "CUDAExecutionProvider" in provs:
                rospy.loginfo(f"[YOLO Loader] ONNX providers={provs} (GPU OK)")
                det_device = "cuda(ORT)"
                return det, det_device

            # 沒有 CUDA provider → 試著設定成 GPU，再不行用 CPU
            try:
                sess.set_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
                provs2 = list(sess.get_providers())
                if "CUDAExecutionProvider" in provs2:
                    rospy.loginfo(f"[YOLO Loader] ONNX switched to providers={provs2} (GPU OK)")
                    det_device = "cuda(ORT)"
                    return det, det_device
                else:
                    rospy.logwarn(f"[YOLO Loader] ONNX CUDA provider not available, using CPU providers={provs2}")
            except Exception as ge:
                rospy.logwarn(f"[YOLO Loader] ONNX set_providers to CUDA failed: {ge}. Will use CPUExecutionProvider.")

            # 最後：CPU
            try:
                sess.set_providers(["CPUExecutionProvider"])
            except Exception:
                pass
            rospy.loginfo("[YOLO Loader] ONNX on CPUExecutionProvider")
            det_device = "cpu(ORT)"
            return det, det_device

        # 如果拿不到 session，就照預設（Ultralytics 內部會自己決定）
        rospy.logwarn("[YOLO Loader] ONNX session not exposed; provider control skipped.")
        return det, det_device

    # =========================
    # 偵測 / BBox / 幾何工具
    # =========================
    def clip_xyxy(self, xyxy, W, H):
        if xyxy is None:
            return None
        x1, y1, x2, y2 = map(float, xyxy)
        return np.array([max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)], dtype=np.float32)

    def rect_to_mask(self, depth, xyxy, expand=0.0):
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

    def iou_xyxy(self, a, b):
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

    def yolo_det_xyxy(self, detector: YOLO, img_bgr, imgsz=640, conf=0.25, prefer_cls=None, mode="score"):
        """
        回傳單一最佳框 (xyxy, score, cls)，挑選策略由mode控制：
        - 'score': 分數最高
        - 'top': 最靠上 (y1 最小)
        - 'bottom':最靠下 (y2 最大)
        會先以conf做門檻過濾；若有prefer_cls，會再以類別過濾。
        """
        r = detector.predict(source=img_bgr, imgsz=imgsz, conf=conf, device=self.det_device, verbose=False)[0]
        if len(r.boxes) == 0:
            return None, 0.0, None

        xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)   # (N,4)
        sc = r.boxes.conf.cpu().numpy().astype(np.float32)   # (N,)
        cl = r.boxes.cls.cpu().numpy().astype(int)           # (N,)

        # 先以分數門檻過濾
        mask = (sc >= float(conf))

        # 若指定類別，則進一步過濾
        if prefer_cls is not None:
            mask = mask & (cl == int(prefer_cls))

        idxs = np.where(mask)[0]
        if idxs.size == 0:
            return None, 0.0, None

        mode = (mode or "score").lower()
        if mode == "top":
            # y1 最小（越靠上）
            y1 = xyxy[idxs, 1]
            pick = idxs[np.argmin(y1)]
        elif mode == "bottom":
            # y2 最大（越靠下）
            y2 = xyxy[idxs, 3]
            pick = idxs[np.argmax(y2)]
        else:
            # 'score'：分數最高
            pick = idxs[np.argmax(sc[idxs])]

        return xyxy[pick], float(sc[pick]), int(cl[pick])

    def yolo_det_all(self, detector: YOLO, img_bgr, imgsz=640, conf=0.25):
        """
        回傳所有偵測：
        xyxy: (N,4) float32
        sc:   (N,)  float32
        cl:   (N,)  int32
        若沒有偵測，三者皆為長度 0 的陣列。
        """
        r = detector.predict(source=img_bgr, imgsz=imgsz, conf=conf, device=self.det_device, verbose=False)[0]
        if len(r.boxes) == 0:
            return (np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=np.int32))
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
        return self.clip_xyxy(np.array([u.min(), v.min(), u.max(), v.max()], dtype=np.float32), W, H)

    # =========================
    # 滑鼠點選ROI
    # =========================
    box_points = []
    selecting_bbox = False

    def _open_window(self, name, pos_xy, init_size, is_rgb=True):
        w, h = init_size
        try:
            cv2.destroyWindow(name)
        except Exception:
            pass
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, int(w), int(h))
        cv2.moveWindow(name, int(pos_xy[0]), int(pos_xy[1]))

        if is_rgb:
            self._rgb_win_created = True
            self._rgb_win_sized = False
        else:
            self._depth_win_created = True
            self._depth_win_sized = False

    def _close_window(self, name, is_rgb=True):
        try:
            cv2.destroyWindow(name)
        except Exception:
            pass
        if is_rgb:
            self._rgb_win_created = False
            self._rgb_win_sized = False
        else:
            self._depth_win_created = False
            self._depth_win_sized = False

    def window_create(self):
        if self.show_rgb_win and not self._rgb_win_created:
            self._open_window(self.rgb_win_name, self.rgb_win_xy, self._rgb_initial_size, is_rgb=True)
        if self.show_depth_win and not self._depth_win_created:
            self._open_window(self.depth_win_name, self.depth_win_xy, self._depth_initial_size, is_rgb=False)

    def pump_windows(self, rgb_frame=None, depth_frame=None):
        """統一處理視窗建立/移動/縮放/顯示/事件，確保即使尚未偵測也會持續更新。"""
        # 先確保 window 存在
        if self.show_rgb_win and not self._rgb_win_created:
            self._open_window(self.rgb_win_name, self.rgb_win_xy, self._rgb_initial_size, is_rgb=True)
        if self.show_depth_win and not self._depth_win_created:
            self._open_window(self.depth_win_name, self.depth_win_xy, self._depth_initial_size, is_rgb=False)

        # 若沒給影像，用空畫面頂住，避免 GUI 停住
        if rgb_frame is None and self.show_rgb_win:
            rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(rgb_frame, "Waiting for detection / click init...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        if depth_frame is None and self.show_depth_win:
            depth_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 顯示 + 初始定位/尺寸（確保 move/resize 真的跑到）
        if self.show_rgb_win and rgb_frame is not None:
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

        # 永遠跑 waitKey 讓 GUI 有事件循環
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rospy.signal_shutdown("user quit")
        elif key == ord('d'):
            self.toggle_depth_window()
        elif key == ord('r'):
            self.toggle_rgb_window()
        
    def ensure_window(self, name, xy, size=(900, 720)):
        try:
            cv2.getWindowProperty(name, 0)  # 會丟例外代表視窗不存在
        except cv2.error:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(name, int(xy[0]), int(xy[1]))
            cv2.resizeWindow(name, size[0], size[1])

    def toggle_depth_window(self):
        self.show_depth_win = not self.show_depth_win
        if self.show_depth_win:
            self.ensure_window(self.depth_win_name, self.depth_win_xy)
        else:
            try:
                cv2.destroyWindow(self.depth_win_name)
            except cv2.error:
                pass

    def toggle_rgb_window(self):
        self.show_rgb_win = not self.show_rgb_win
        if self.show_rgb_win:
            self.ensure_window(self.rgb_win_name, self.rgb_win_xy)
        else:
            try:
                cv2.destroyWindow(self.rgb_win_name)
            except cv2.error:
                pass

    def click_bbox(self, event, x, y, flags, param):
        global box_points
        if event == cv2.EVENT_LBUTTONDOWN:
            box_points.append((x, y))
            rospy.loginfo(f"Clicked point: {x}, {y}")

    def select_bbox(self, color):
        """開始手動選框流程"""
        global box_points, selecting_bbox
        box_points.clear()
        selecting_bbox = True
        cv2.setMouseCallback(self.rgb_win_name, self.click_bbox)
        rospy.loginfo("Please click on the upper left and lower right corners of the object")
        return True

    def update_bbox_selection(self, color):
        """更新手動選框流程狀態"""
        global box_points, selecting_bbox
        if not selecting_bbox:
            return True

        # 階段1：等第一點
        if len(box_points) < 1:
            display_img = color.copy()
            self.pump_windows(display_img, self.depth_vis if self.got_depth else None)
            return False

        # 階段2：等第二點
        elif len(box_points) < 2:
            display_img = color.copy()
            for pt in box_points:
                cv2.circle(display_img, pt, radius=5, color=(0, 255, 0), thickness=-1)
            self.pump_windows(display_img, self.depth_vis if self.got_depth else None)
            return False

        # 完成
        else:
            selecting_bbox = False
            rospy.loginfo(f"Bounding Box selected: {box_points}")
            return True

    def create_mask(self, depth, bbox_points):
        """依手動 bbox 產生 mask"""
        x1, y1 = bbox_points[0]
        x2, y2 = bbox_points[1]
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        mask = np.zeros_like(depth, dtype=bool)
        mask[y_min:y_max, x_min:x_max] = True
        return mask

    # =========================
    # 視覺化 / 文字疊圖
    # =========================
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
    # 初始化 / 追蹤輔助（YOLO + IoU）
    # =========================
    def init_via_yolo_roi(self, detector, color, depth, K, est, est_refine_iter, roi_expand, det_imgsz, det_conf, prefer_cls):
        """用 YOLO 取 ROI，建立 mask 後呼叫 register"""
        H, W = color.shape[:2]
        det_xyxy, det_score, det_cls = self.yolo_det_xyxy(detector, color, imgsz=det_imgsz, conf=det_conf, prefer_cls=prefer_cls, mode=self.det_select_mode)
        if det_xyxy is None:
            vis = color.copy()
            cv2.putText(vis, "YOLO ROI not found - waiting...", (20, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            self.pump_windows(vis, self.depth_vis if self.got_depth else None)
            return None, None, None, None

        det_xyxy = self.clip_xyxy(det_xyxy, W, H)
        mask = self.rect_to_mask(depth, det_xyxy, expand=roi_expand)
        pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)

        # 可視化初始化 ROI
        init_vis = color.copy()
        x1, y1, x2, y2 = det_xyxy.astype(int)
        cv2.rectangle(init_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(init_vis, f"INIT ROI s={det_score:.2f}", (x1, max(0, y1 - 8)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        self.pump_windows(init_vis, self.depth_vis if self.got_depth else None)
        return pose, mask, det_xyxy, det_score

    def periodic_yolo_iou(self, frame_count, stride, detector, color, center_pose, bbox_minmax, K,
                        det_imgsz, det_conf, prefer_cls, vis_bgr, log=False):
        """
        每 N 幀用 YOLO 算一次 bbox，和 3D 投影 bbox 比 IoU，並在左上角印 YOLO 資訊。
        回傳 (更新後影像, iou_val)
        """
        iou_val = None
        if detector is None or frame_count % max(1, stride) != 0 or center_pose is None:
            return vis_bgr, iou_val

        H, W = color.shape[:2]

        # 投影 bbox
        est_xyxy = self.project_3d_bbox_xyxy(K, center_pose, bbox_minmax, img_shape=color.shape)
        if est_xyxy is None:    # 投影失敗就只顯示偵測數
            xyxy_all, sc_all, cl_all = self.yolo_det_all(detector, color, imgsz=det_imgsz, conf=det_conf)
            cv2.putText(vis_bgr, f"YOLO det={len(xyxy_all)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
            return vis_bgr, iou_val
        
        # YOLO 偵測
        xyxy_all, sc_all, cl_all = self.yolo_det_all(detector, color, imgsz=det_imgsz, conf=det_conf)
        if len(xyxy_all) == 0:
            cv2.putText(vis_bgr, "YOLO det=0", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
            return vis_bgr, 0.0
        
        # 若有指定 prefer_cls，且該類別有偵測，僅在該類別中挑；否則全體比較
        use_mask = np.ones(len(xyxy_all), dtype=bool)
        if prefer_cls is not None and (cl_all == prefer_cls).any():
            use_mask = (cl_all == prefer_cls)

        xyxy_use = xyxy_all[use_mask]
        sc_use = sc_all[use_mask]
        cl_use = cl_all[use_mask]

        # 算每一個偵測與投影框的 IoU，取最大
        ious = []
        for bb in xyxy_use:
            bb_clipped = self.clip_xyxy(bb, W, H)
            ious.append(self.iou_xyxy(bb_clipped, est_xyxy))
        ious = np.array(ious, dtype=float)

        if len(ious) == 0:
            cv2.putText(vis_bgr, f"YOLO det={len(xyxy_all)} (no valid)", (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)
            return vis_bgr, None

        best_idx = int(np.argmax(ious))
        iou_val  = float(ious[best_idx])
        best_cls = int(cl_use[best_idx])
        best_sc  = float(sc_use[best_idx])

        # 左上角顯示最佳配對資訊
        cv2.putText(vis_bgr, f"YOLO det={len(xyxy_all)} best cls={best_cls} s={best_sc:.2f} IoU={iou_val:.3f}",(10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)

        if log:
            rospy.loginfo(f"[IOU] frame={frame_count} best_iou={iou_val:.4f} (cls={best_cls}, score={best_sc:.3f})")

        return vis_bgr, iou_val

    # =========================
    # ROS 轉換工具
    # =========================
    def mat4_to_translation_quat(self, T: np.ndarray):
        """4x4 → (tx,ty,tz), (qx,qy,qz,qw)"""
        assert T.shape == (4, 4)
        t = (float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
        qx, qy, qz, qw = tf.transformations.quaternion_from_matrix(T)
        q = (float(qx), float(qy), float(qz), float(qw))
        return t, q

    def mat4_to_pose(self, T: np.ndarray) -> Pose:
        """4x4 → geometry_msgs/Pose"""
        t, q = self.mat4_to_translation_quat(T)
        msg = Pose()
        msg.position.x, msg.position.y, msg.position.z = t
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = q
        return msg
    
    def broadcast_transform_and_pose(self, transform: Transform, object_name: str, camera_tf: str):
        # --- send TF (parent= camera_tf, child= object_name)
        t = (transform.translation.x, transform.translation.y, transform.translation.z)
        q = (transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w)
        self.tf_broadcaster.sendTransform(t, q, rospy.Time.now(), object_name, camera_tf)

        # --- publish Pose on <object_name>
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = t
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
        self.pose_pub.publish(pose)

    def imageCallback(self, msg: Image):
        try:
            # 轉成 BGR8
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("rgb decode failed: %r", e)
            return
        self.color = img
        self.got_rgb = True
        self.rgb_size = (img.shape[1], img.shape[0])  # (W, H)

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
            rospy.logwarn("depth decode failed: %r", e)
            return

        # 轉公尺（演算法用）
        if msg.encoding.upper() in ("16UC1", "TYPE_16UC1"):
            depth_m = d * 0.001  # mm -> m
        else:
            # 可能已是 m；若最大值遠小於 10，視為 m；否則視為 mm
            depth_m = d if np.nanmax(d) <= 10.0 else d * 0.001

        # 存給演算法
        self.depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)

        # 另外做可視化彩圖（僅顯示）
        maxd_mm = max(1.0, float(self.max_depth_mm))
        depth_mm_for_vis = np.clip(self.depth_m * 1000.0, 0.0, maxd_mm)
        depth_8u = (depth_mm_for_vis * (255.0 / maxd_mm)).astype(np.uint8)
        if self.invert_colormap:
            depth_8u = 255 - depth_8u
        self.depth_vis = cv2.applyColorMap(depth_8u, self.colormap_id)

        self.got_depth = True
        self.depth_size = (self.depth_vis.shape[1], self.depth_vis.shape[0])

    def detectionCallback(self, msg: Detection):
        self.ready_received.detection_allowed = msg.detection_allowed
        self.ready_received.layer = msg.layer
        if msg.layer == 0.0:
            self.det_select_mode = "score"
        elif msg.layer == 1.0:
            self.det_select_mode = "bottom"
        elif msg.layer == 2.0:
            self.det_select_mode = "top"
        else:
            self.det_select_mode = "score"

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
            if not (self.got_depth and self.got_rgb and self.K is not None and hasattr(self, "depth_m")):
                self.pump_windows(self.color if self.got_rgb else None, self.depth_vis if self.got_depth else None)
                rospy.sleep(0.01)
                continue

            if (self.init_mode == 'yolo' and self.yolo_start_mode == 'wait' and not self.ready_received.detection_allowed):
                self.pose = None
                first_frame = True
                self.iou_bad_count = 0
                self.iou_val = None

                # 視窗上顯示暫停提示
                vis_rgb = self.color.copy() if self.color is not None else np.zeros((480,640,3), np.uint8)
                cv2.putText(vis_rgb, "DETECTION DISABLED", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                self.pump_windows(vis_rgb if self.show_rgb_win else None,
                                self.depth_vis if (self.show_depth_win and hasattr(self, "depth_vis")) else None)

                # 發布信心為 0
                self.confidence_publish(0.0, False)
                continue

            self.frame_count += 1
            # === 初始化 ROI 階段 ===
            if self.init_mode == 'click':
                # 點選流程
                if selecting_bbox:
                    if not self.update_bbox_selection(self.color):
                        continue
                    else:
                        self.mask = self.create_mask(self.depth_m, box_points)
                        self.pose = self.est.register(K=self.K, rgb=self.color, depth=self.depth_m, ob_mask=self.mask, iteration=self.est_refine_iter)
                        box_points.clear()
                        first_frame = False
                elif first_frame:
                    self.select_bbox(self.color)
                    continue  # 等點選完成
            else:
                # YOLO
                if first_frame:
                    init = self.init_via_yolo_roi(self.detector, self.color, self.depth_m, self.K, self.est,
                                             self.est_refine_iter, self.roi_expand,
                                             self.det_imgsz, self.det_conf, self.prefer_cls)
                    if init[0] is None:
                        continue  # 等到偵到為止
                    self.pose, self.mask, _, _ = init
                    first_frame = False

            # === 追蹤 ===
            self.pose = self.est.track_one(rgb=self.color, depth=self.depth_m, K=self.K, iteration=self.track_refine_iter)

            vis_bgr = self.color.copy()
            if self.pose is not None:
                center_pose = self.pose @ np.linalg.inv(self.to_origin)
                color_rgb = cv2.cvtColor(self.color, cv2.COLOR_BGR2RGB)
                # rospy.loginfo(f"pose={self.pose})")
                # 將 4x4 轉成 geometry_msgs/Transform
                t, q = self.mat4_to_translation_quat(self.pose)
                tr = Transform()
                tr.translation.x, tr.translation.y, tr.translation.z = t
                tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w = q

                # 廣播 + 發布（topic = self.object_name, frame = self.camera_tf）
                parent_frame = self.camera_tf if self.camera_tf else "camera_color_optical_frame"
                self.broadcast_transform_and_pose(tr, self.object_name, parent_frame)

                # 可視化
                vis = draw_posed_3d_box(self.K, img=color_rgb, ob_in_cam=center_pose, bbox=self.bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=self.pose, scale=0.05, K=self.K, thickness=3, transparency=0, is_input_rgb=True)
                vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                if self.iou_val is not None:
                    bar_w, bar_h, margin = 220, 18, 10
                    self.draw_conf_bar(vis_bgr,{"IoU": float(self.iou_val)},label="IoU",origin=(10, vis_bgr.shape[0] - margin - bar_h),size=(bar_w, bar_h),max_val=1.0)
                if self.init_mode == 'yolo':
                    # === 每 N 幀做一次 YOLO vs. 估測框 IoU===
                    vis_bgr, self.iou_val = self.periodic_yolo_iou(
                        frame_count=self.frame_count, stride=self.iou_stride, detector=self.detector,
                        color=self.color, center_pose=center_pose, bbox_minmax=self.bbox, K=self.K,
                        det_imgsz=self.det_imgsz, det_conf=self.det_conf, prefer_cls=self.prefer_cls,
                        vis_bgr=vis_bgr, log=self.iou_log
                    )
                    # IoU 連續低於閾值則觸發重新流程
                    if self.iou_val is not None:
                        if self.iou_val < self.iou_thresh:
                            self.iou_bad_count += 1
                        else:
                            self.iou_bad_count = 0

                        if self.iou_bad_count >= self.iou_patience:
                            self.iou_bad_count = 0
                            first_frame = True    # 下一輪走初始化（依 init_mode 決定是 yolo 或 click）
                            self.pose = None
                            cv2.putText(vis_bgr, "Re-init (low IoU)", (10, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

                if self.got_depth and self.got_rgb:
                    if self.depth_size != self.rgb_size:
                        rospy.logwarn_throttle(2.0, "Depth and RGB image sizes differ: depth=%s rgb=%s", str(self.depth_size), str(self.rgb_size))
            
            self.pump_windows(vis_bgr if (self.show_rgb_win and self.color is not None) else None,
                              self.depth_vis if (self.show_depth_win and self.got_depth and self.depth_vis is not None) else None)
            if self.iou_val is not None:
                self.confidence_publish(float(self.iou_val), not first_frame)
            else:
                self.confidence_publish(0.0, False)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    rospy.init_node("foundationpose_tracker", anonymous=False)
    node = FoundationPoseTracker()
    node.spin()
