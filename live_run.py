import pyrealsense2 as rs
import numpy as np
import cv2
import trimesh
import os
os.environ["ULTRALYTICS_NO_INSTALL"] = "1"
import argparse
import logging

from estimater import *
from datareader import *

from ultralytics import YOLO

# =========================
# YOLO 後端/裝置資訊工具
# =========================
def yolo_backend_info(detector):
    """回傳 (engine, torch_device, ort_providers) 三項資訊，涵蓋 .pt/.onnx/等"""
    info = {"engine": None, "torch_device": None, "ort_providers": None}
    m = getattr(detector, "model", None)
    if m is None:
        return info

    # 可能存在的後端描述
    info["engine"] = str(getattr(m, "backend", getattr(m, "engine", "")))

    # PyTorch：檢查參數是否在CUDA
    try:
        import torch  # noqa
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

def yolo_uses_gpu(detector):
    """回傳 (is_gpu: bool, 描述字串)"""
    info = yolo_backend_info(detector)
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

# =========================
# 偵測 / BBox / 幾何工具
# =========================
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

def yolo_det_xyxy(detector: YOLO, img_bgr, imgsz=640, conf=0.25, prefer_cls=None):
    r = detector.predict(source=img_bgr, imgsz=imgsz, conf=conf, verbose=False)[0]
    if len(r.boxes) == 0:
        return None, 0.0, None
    xyxy = r.boxes.xyxy.cpu().numpy()
    sc = r.boxes.conf.cpu().numpy()
    cl = r.boxes.cls.cpu().numpy().astype(int)
    if prefer_cls is not None and (cl == prefer_cls).any():
        idx = np.argmax(sc * (cl == prefer_cls))
    else:
        idx = int(np.argmax(sc))
    return xyxy[idx], float(sc[idx]), int(cl[idx])

def yolo_det_all(detector: YOLO, img_bgr, imgsz=640, conf=0.25):
    """
    回傳所有偵測：
      xyxy: (N,4) float32
      sc:   (N,)  float32
      cl:   (N,)  int32
    若沒有偵測，三者皆為長度 0 的陣列。
    """
    r = detector.predict(source=img_bgr, imgsz=imgsz, conf=conf, verbose=False)[0]
    if len(r.boxes) == 0:
        return (np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32))
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

# =========================
# 滑鼠點選ROI
# =========================
box_points = []
selecting_bbox = False
window_name = "Object Tracking"

def setup_window():
    """設置視窗尺寸（使用全域 image_H / image_W）"""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, image_H, image_W)

def click_bbox(event, x, y, flags, param):
    global box_points
    if event == cv2.EVENT_LBUTTONDOWN:
        box_points.append((x, y))
        logging.info(f"Clicked point: {x}, {y}")

def select_bbox(color):
    """開始手動選框流程"""
    global box_points, selecting_bbox
    box_points.clear()
    selecting_bbox = True
    cv2.setMouseCallback(window_name, click_bbox)
    logging.info("Please click on the upper left and lower right corners of the object")
    return True

def update_bbox_selection(color):
    """更新手動選框流程狀態"""
    global box_points, selecting_bbox
    if not selecting_bbox:
        return True

    # 階段1：等第一點
    if len(box_points) < 1:
        display_img = color.copy()
        cv2.imshow(window_name, display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exit")
            exit()
        return False

    # 階段2：凍結畫面，等第二點
    elif len(box_points) < 2:
        display_img = color.copy()
        for pt in box_points:
            cv2.circle(display_img, pt, radius=5, color=(0, 255, 0), thickness=-1)
        cv2.imshow(window_name, display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exit")
            exit()
        return False

    # 完成
    else:
        selecting_bbox = False
        logging.info(f"Bounding Box selected: {box_points}")
        return True

def create_mask(depth, bbox_points):
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
def draw_conf_bar(img, value, label="confidence", origin=(10, 30), size=(220, 18), max_val=1.0):
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
def init_via_yolo_roi(detector, color, depth, K, est, est_refine_iter, roi_expand, det_imgsz, det_conf, prefer_cls):
    """用 YOLO 取 ROI，建立 mask 後呼叫 register"""
    H, W = color.shape[:2]
    det_xyxy, det_score, det_cls = yolo_det_xyxy(detector, color, imgsz=det_imgsz, conf=det_conf, prefer_cls=prefer_cls)
    if det_xyxy is None:
        vis = color.copy()
        cv2.putText(vis, "YOLO ROI not found - waiting...", (20, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, vis)
        cv2.waitKey(1)
        return None, None, None, None

    det_xyxy = clip_xyxy(det_xyxy, W, H)
    mask = rect_to_mask(depth, det_xyxy, expand=roi_expand)
    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)

    # 可視化初始化 ROI
    init_vis = color.copy()
    x1, y1, x2, y2 = det_xyxy.astype(int)
    cv2.rectangle(init_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(init_vis, f"INIT ROI s={det_score:.2f}", (x1, max(0, y1 - 8)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow(window_name, init_vis)
    cv2.waitKey(1)
    return pose, mask, det_xyxy, det_score

def periodic_yolo_iou(frame_count, stride, detector, color, center_pose, bbox_minmax, K,
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
    est_xyxy = project_3d_bbox_xyxy(K, center_pose, bbox_minmax, img_shape=color.shape)
    if est_xyxy is None:    # 投影失敗就只顯示偵測數
        xyxy_all, sc_all, cl_all = yolo_det_all(detector, color, imgsz=det_imgsz, conf=det_conf)
        cv2.putText(vis_bgr, f"YOLO det={len(xyxy_all)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
        return vis_bgr, iou_val
    
    # YOLO 偵測
    xyxy_all, sc_all, cl_all = yolo_det_all(detector, color, imgsz=det_imgsz, conf=det_conf)
    if len(xyxy_all) == 0:
        cv2.putText(vis_bgr, "YOLO det=0", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
        return vis_bgr, iou_val
    
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
        bb_clipped = clip_xyxy(bb, W, H)
        ious.append(iou_xyxy(bb_clipped, est_xyxy))
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
        logging.info(f"[IOU] frame={frame_count} best_iou={iou_val:.4f} (cls={best_cls}, score={best_sc:.3f})")

    return vis_bgr, iou_val

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--init_mode', type=str, default='yolo', choices=['click','yolo'],help='初始ROI模式：click=手動點兩角；yolo=用ONNX自動抓')
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/models/oilpalm/oilpalm.obj')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    parser.add_argument('--image_H', type=int, default=640)
    parser.add_argument('--image_W', type=int, default=480)
    parser.add_argument('--image_fps', type=int, default=30)
    
    parser.add_argument('--det_onnx', type=str, default=f'{code_dir}/demo_data/models/oilpalm_pallet.onnx',help='YOLOv11 ONNX 路徑（用來抓 ROI 與 IoU 對比）')
    parser.add_argument('--det_imgsz', type=int, default=640)
    parser.add_argument('--det_conf', type=float, default=0.25)
    parser.add_argument('--det_class', type=int, default=0, help='只取特定類別；<0 表示不限制')
    parser.add_argument('--roi_expand', type=float, default=0.01, help='初始化 ROI 外擴比例')
    parser.add_argument('--iou_stride', type=int, default=1, help='每 N 幀做一次YOLO偵測與IoU計算')
    parser.add_argument('--iou_log', action='store_true', help='將IoU印到log')
    parser.add_argument('--iou_thresh', type=float, default=0.2, help='IoU 閾值，低於此值將重新初始化')
    parser.add_argument('--iou_patience', type=int, default=2, help='連續低於閾值幀數後才重啟，避免誤觸')
    
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)
    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                         scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("Estimator initialization done")

    #### YOLO ONNX 偵測器
    prefer_cls = None if args.det_class < 0 else int(args.det_class)
    detector = YOLO(args.det_onnx, task='detect')
    is_gpu, yolo_desc = yolo_uses_gpu(detector)
    logging.info(f"[YOLO] GPU enabled: {is_gpu}  ({yolo_desc})")

    #### RealSense 初始化 ####
    image_H = args.image_H
    image_W = args.image_W
    image_fps = args.image_fps
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, image_H, image_W, rs.format.bgr8, image_fps)
    config.enable_stream(rs.stream.depth, image_H, image_W, rs.format.z16, image_fps)
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    #### 相機內參數 K ####
    color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([
        [color_intrin.fx, 0, color_intrin.ppx],
        [0, color_intrin.fy, color_intrin.ppy],
        [0, 0, 1]
    ], dtype=np.float64)

    #### 主循環 ####
    setup_window()
    first_frame = True
    pose = np.eye(4, dtype=np.float64)
    mask = None
    frame_count = 0
    iou_bad_count = 0
    iou_val = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue
            
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color = np.asanyarray(color_frame.get_data())
            depth = depth_image * 0.001  # Convert from mm to meters

            H, W = color.shape[:2]
            frame_count += 1

            # === 初始化 ROI 階段 ===
            if args.init_mode == 'click':
                # 點選流程
                if selecting_bbox:
                    if not update_bbox_selection(color):
                        continue
                    else:
                        mask = create_mask(depth, box_points)
                        pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
                        box_points.clear()
                        first_frame = False
                elif first_frame:
                    select_bbox(color)
                    continue  # 等點選完成
            else:
                # YOLO
                if first_frame:
                    init = init_via_yolo_roi(detector, color, depth, K, est,
                                             args.est_refine_iter, args.roi_expand,
                                             args.det_imgsz, args.det_conf, prefer_cls)
                    if init[0] is None:
                        continue  # 等到偵到為止
                    pose, mask, _, _ = init
                    first_frame = False

            # === 追蹤 ===
            pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)

            # === 視覺化 ===
            vis_bgr = color.copy()
            if pose is not None:
                center_pose = pose @ np.linalg.inv(to_origin)
                color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                vis = draw_posed_3d_box(K, img=color_rgb, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=pose, scale=0.05, K=K, thickness=3, transparency=0, is_input_rgb=True)
                vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                if iou_val is not None:
                    bar_w, bar_h, margin = 220, 18, 10
                    draw_conf_bar(vis_bgr,{"IoU": float(iou_val)},label="IoU",origin=(10, vis_bgr.shape[0] - margin - bar_h),size=(bar_w, bar_h),max_val=1.0)
                if args.init_mode == 'yolo':
                    # === 每 N 幀做一次 YOLO vs. 估測框 IoU（副程式）===
                    vis_bgr, iou_val = periodic_yolo_iou(
                        frame_count=frame_count, stride=args.iou_stride, detector=detector,
                        color=color, center_pose=center_pose, bbox_minmax=bbox, K=K,
                        det_imgsz=args.det_imgsz, det_conf=args.det_conf, prefer_cls=prefer_cls,
                        vis_bgr=vis_bgr, log=args.iou_log
                    )
                    # IoU 連續低於閾值則觸發重新流程
                    if iou_val is not None:
                        if iou_val < args.iou_thresh:
                            iou_bad_count += 1
                        else:
                            iou_bad_count = 0

                        if iou_bad_count >= args.iou_patience:
                            iou_bad_count = 0
                            first_frame = True    # 下一輪走初始化（依 init_mode 決定是 yolo 或 click）
                            pose = None
                            cv2.putText(vis_bgr, "Re-init (low IoU)", (10, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow(window_name, vis_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
