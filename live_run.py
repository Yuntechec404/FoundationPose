import pyrealsense2 as rs
import numpy as np
import cv2
import trimesh
import os
import argparse
import logging

from estimater import *
from datareader import *

# 用來存儲使用者點選的兩個角點
box_points = []
selecting_bbox = False  # 新增：追踪是否正在選擇bbox狀態
window_name = "Object Tracking"

def setup_window():
    """設置統一窗口並移動到左上角"""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, image_H, image_W)  # 設置窗口大小

def click_bbox(event, x, y, flags, param):
    global box_points
    if event == cv2.EVENT_LBUTTONDOWN:
        box_points.append((x, y))
        print(f"Clicked point: {x}, {y}")

def select_bbox(color):
    """
    用戶手動點選bbox
    影像實時更新直到第一個點選後 freeze
    """
    global box_points, selecting_bbox
    box_points.clear()
    selecting_bbox = True  # 設置選擇狀態
    
    cv2.setMouseCallback(window_name, click_bbox)
    print("Please click on the upper left and lower right corners of the object")
    
    return True  # 返回True表示開始選擇流程

def update_bbox_selection(color):
    """
    更新bbox選擇流程
    返回：True表示選擇完成，False表示仍在選擇中
    """
    global box_points, selecting_bbox
    
    if not selecting_bbox:
        return True
    
    # 第一階段：即時更新畫面直到第一個點
    if len(box_points) < 1:
        display_img = color.copy()
        cv2.imshow(window_name, display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit")
            exit()
        return False
    
    # 第二階段：freeze畫面，等待第二點
    elif len(box_points) < 2:
        display_img = color.copy()  # 使用當前幀而不是clone
        for pt in box_points:
            cv2.circle(display_img, pt, radius=5, color=(0,255,0), thickness=-1)
        cv2.imshow(window_name, display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit")
            exit()
        return False
    
    # 選擇完成
    else:
        selecting_bbox = False
        print(f"Bounding Box selected: {box_points}")
        return True

def create_mask(depth, bbox_points):
    """
    建立 mask，根據 bbox_points (list of two tuples) 設定 mask
    """
    x1, y1 = bbox_points[0]
    x2, y2 = bbox_points[1]
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    mask = np.zeros_like(depth, dtype=bool)
    mask[y_min:y_max, x_min:x_max] = True
    return mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/models/cube/cube.obj')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    parser.add_argument('--image_H', type=int, default=640)
    parser.add_argument('--image_W', type=int, default=480)
    parser.add_argument('--image_fps', type=int, default=30)
    parser.add_argument('--score_threshold', type=float, default=0.0)
    
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

            # 處理bbox選擇流程
            if selecting_bbox:
                bbox_selection_complete = update_bbox_selection(color)
                if not bbox_selection_complete:
                    continue  # 繼續選擇流程，不進行pose estimation
                else:
                    # 選擇完成，建立mask並初始化pose
                    mask = create_mask(depth, box_points)
                    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
                    box_points.clear()
                    first_frame = False

            elif first_frame:
                # 開始選擇bbox流程
                select_bbox(color)
                continue  # 跳到下一輪開始選擇流程

            else:
                # 正常追踪流程
                pose, score = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)
                
                if score < args.score_threshold:
                    logging.warning(f"Pose estimation score too high ({score:}), resetting tracker.")
                    
                    first_frame = True
                    pose = None
                    continue  # 跳到下一輪重新選擇bbox
            
            # 只有在不選擇bbox且pose有效時才顯示tracking結果
            if debug >= 1 and pose is not None and not selecting_bbox:
                center_pose = pose @ np.linalg.inv(to_origin)
                color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                vis = draw_posed_3d_box(K, img=color_rgb, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
                # vis = draw_posed_3d_origin_axis(K, img=color_rgb, ob_in_cam=center_pose, scale=0.1, thickness=2, is_input_rgb=False)
                cv2.imshow(window_name, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if debug >= 2 and pose is not None and not selecting_bbox:
                frame_idx = int(rs.frame_number())
                os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
                cv2.imwrite(f'{debug_dir}/track_vis/{frame_idx}.png', vis)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
