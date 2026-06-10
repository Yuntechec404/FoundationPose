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
# YOLO 與 ROI 工具 (保留用於首次初始化)
# =========================
def clip_xyxy(xyxy, W, H):
    if xyxy is None: return None
    x1, y1, x2, y2 = map(float, xyxy)
    return np.array([max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)], dtype=np.float32)

def rect_to_mask(depth, xyxy, expand=0.0):
    if xyxy is None: return None
    H, W = depth.shape[:2]
    x1, y1, x2, y2 = xyxy.astype(np.int32)
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    x1, y1 = max(0, int(x1 - w * expand)), max(0, int(y1 - h * expand))
    x2, y2 = min(W - 1, int(x2 + w * expand)), min(H - 1, int(y2 + h * expand))
    m = np.zeros((H, W), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m

def yolo_det_xyxy(detector: YOLO, img_bgr, imgsz=640, conf=0.25, prefer_cls=None):
    r = detector.predict(source=img_bgr, imgsz=imgsz, conf=conf, verbose=False)[0]
    if len(r.boxes) == 0: return None, 0.0, None
    xyxy, sc, cl = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)
    idx = np.argmax(sc * (cl == prefer_cls)) if prefer_cls is not None and (cl == prefer_cls).any() else int(np.argmax(sc))
    return xyxy[idx], float(sc[idx]), int(cl[idx])

# =========================
# 滑鼠點選ROI
# =========================
box_points = []
selecting_bbox = False
window_name = "Object Tracking - Self Aware Mode"

def setup_window():
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, image_H, image_W)

def click_bbox(event, x, y, flags, param):
    global box_points
    if event == cv2.EVENT_LBUTTONDOWN:
        box_points.append((x, y))

def select_bbox(color):
    global box_points, selecting_bbox
    box_points.clear()
    selecting_bbox = True
    cv2.setMouseCallback(window_name, click_bbox)
    logging.info("Please click on the upper left and lower right corners.")
    return True

def update_bbox_selection(color):
    global box_points, selecting_bbox
    if not selecting_bbox: return True
    display_img = color.copy()
    if len(box_points) < 1:
        cv2.imshow(window_name, display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'): exit()
        return False
    elif len(box_points) < 2:
        for pt in box_points: cv2.circle(display_img, pt, 5, (0, 255, 0), -1)
        cv2.imshow(window_name, display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'): exit()
        return False
    else:
        selecting_bbox = False
        return True

def create_mask(depth, bbox_points):
    x1, y1, x2, y2 = *bbox_points[0], *bbox_points[1]
    mask = np.zeros_like(depth, dtype=bool)
    mask[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)] = True
    return mask

# =========================
# 視覺化 UI
# =========================
def draw_conf_bar(img, val, label="confidence", origin=(10, 30), size=(220, 18), color=(60, 180, 75)):
    v = max(0.0, min(float(val), 1.0))
    x, y, w, h = *origin, *size
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + int(w * v), y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (220, 220, 220), 1)
    cv2.putText(img, f"{label}: {val:.3f}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# =========================
# 初始化 (YOLO)
# =========================
def init_via_yolo_roi(detector, color, depth, K, est, est_refine_iter, roi_expand, det_imgsz, det_conf, prefer_cls):
    H, W = color.shape[:2]
    det_xyxy, det_score, _ = yolo_det_xyxy(detector, color, imgsz=det_imgsz, conf=det_conf, prefer_cls=prefer_cls)
    if det_xyxy is None:
        vis = color.copy()
        cv2.putText(vis, "YOLO ROI not found - waiting...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow(window_name, vis)
        cv2.waitKey(1)
        return None, None
    det_xyxy = clip_xyxy(det_xyxy, W, H)
    mask = rect_to_mask(depth, det_xyxy, expand=roi_expand)
    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter, top_k=5)
    return pose, mask

# =========================
# 主程式
# =========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--init_mode', type=str, default='yolo', choices=['click','yolo'])
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/models/oilpalm/oilpalm.obj')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    
    # YOLO 初始設定
    parser.add_argument('--det_onnx', type=str, default=f'{code_dir}/demo_data/models/oilpalm_stem.onnx')
    parser.add_argument('--det_imgsz', type=int, default=640)
    parser.add_argument('--det_conf', type=float, default=0.4)
    parser.add_argument('--det_class', type=int, default=0)
    parser.add_argument('--roi_expand', type=float, default=0.01)
    
    # 自我檢視與 Re-Init 參數
    parser.add_argument('--depth_mae_thresh', type=float, default=0.03, help='MAE閾值，低於此值認為追蹤失敗 m')
    parser.add_argument('--inlier_ratio_thresh', type=float, default=0.5, help='Inlier Ratio 閾值，低於此值認為追蹤失敗')
    parser.add_argument('--bad_patience', type=int, default=5, help='連續 N 幀低於閾值後才重啟')
    
    args = parser.parse_args()
    set_logging_format()
    set_seed(0)

    # 載入模型與估測器
    mesh = trimesh.load(args.mesh_file)
    os.system(f'rm -rf {args.debug_dir}/* && mkdir -p {args.debug_dir}/track_vis {args.debug_dir}/ob_in_cam')
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                         scorer=ScorePredictor(), refiner=PoseRefinePredictor(), 
                         debug_dir=args.debug_dir, debug=args.debug, glctx=glctx)
    detector = YOLO(args.det_onnx, task='detect')
    prefer_cls = None if args.det_class < 0 else int(args.det_class)

    # 相機設定
    image_H, image_W, image_fps = 640, 480, 30
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, image_H, image_W, rs.format.bgr8, image_fps)
    config.enable_stream(rs.stream.depth, image_H, image_W, rs.format.z16, image_fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    
    color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[color_intrin.fx, 0, color_intrin.ppx], [0, color_intrin.fy, color_intrin.ppy], [0, 0, 1]], dtype=np.float64)

    setup_window()
    first_frame = True
    pose = None
    bad_count = 0

    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            if not frames.get_depth_frame() or not frames.get_color_frame(): continue
            
            depth = np.asanyarray(frames.get_depth_frame().get_data()) * 0.001
            color = np.asanyarray(frames.get_color_frame().get_data())

            # === 初始化 ===
            if first_frame:
                if args.init_mode == 'click':
                    if selecting_bbox:
                        if update_bbox_selection(color):
                            pose = est.register(K=K, rgb=color, depth=depth, ob_mask=create_mask(depth, box_points), iteration=args.est_refine_iter, top_k=5)
                            box_points.clear()
                            first_frame = False
                    else:
                        select_bbox(color)
                    continue
                else:
                    res = init_via_yolo_roi(detector, color, depth, K, est, args.est_refine_iter, args.roi_expand, args.det_imgsz, args.det_conf, prefer_cls)
                    if res[0] is None: continue
                    pose, _ = res
                    first_frame = False

            # === 追蹤與自我檢視 ===
            extra_info = {}
            # 將 extra_info 傳入，獲取 Mask IoU 與 Feat Cosine
            pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter, extra=extra_info)

            depth_mae_val = extra_info.get('depth_mae', None)
            inlier_ratio_val = extra_info.get('inlier_ratio', None)
            feat_cos_val = extra_info.get('feat_cos', None)
            iters_used = extra_info.get('actual_iters', args.track_refine_iter)

            # === 視覺化 ===
            vis_bgr = color.copy()
            if pose is not None:
                center_pose = pose @ np.linalg.inv(to_origin)
                vis_rgb = draw_posed_3d_box(K, cv2.cvtColor(color, cv2.COLOR_BGR2RGB), center_pose, bbox)
                vis_rgb = draw_xyz_axis(vis_rgb, pose, scale=0.05, K=K, thickness=3, is_input_rgb=True)
                vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

                # 左上角顯示迭代次數 (Early Stopping 效益展示)
                cv2.putText(vis_bgr, f"Iters: {iters_used}/{args.track_refine_iter}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 繪製信心指數 Bar (UI)
                if depth_mae_val is not None:
                    mae_cm = depth_mae_val * 100
                    cv2.putText(vis_bgr, f"Depth Error: {mae_cm:.1f} cm", (10, image_W - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
                if inlier_ratio_val is not None:
                    draw_conf_bar(vis_bgr, inlier_ratio_val, "Inlier Ratio", (10, image_W - 35), (200, 15), color=(0, 200, 255))

                # === 追蹤丟失判定 (Tracking Loss Detection) ===
                is_mae_bad = (depth_mae_val is not None and depth_mae_val > args.depth_mae_thresh)
                is_inlier_bad = (inlier_ratio_val is not None and inlier_ratio_val < args.inlier_ratio_thresh)

                if is_mae_bad or is_inlier_bad:
                    bad_count += 1
                    cv2.putText(vis_bgr, f"WARNING: Low Confidence! ({bad_count}/{args.bad_patience})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    bad_count = 0

                # 觸發重新初始化 (Re-Init)
                if bad_count >= args.bad_patience:
                    bad_count = 0
                    first_frame = True
                    pose = None
                    cv2.putText(vis_bgr, "RE-INIT TRIGGERED!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            cv2.imshow(window_name, vis_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
