# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
from datareader import *
import itertools
from learning.training.predict_score import *
from learning.training.predict_pose_refine import *
import yaml


class FoundationPose:
  def __init__(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer:ScorePredictor=None, refiner:PoseRefinePredictor=None, glctx=None, debug=0, debug_dir='/home/user/debug/novel_pose_debug/',
               coarse_min_n_views=40, coarse_inplane_step=60, coarse_orientation_mode="uniform", coarse_orientation_tilt_deg=30.0, coarse_object_up_axis=1):
    self.gt_pose = None
    self.ignore_normal_flip = True
    self.debug = debug
    self.debug_dir = debug_dir
    os.makedirs(debug_dir, exist_ok=True)

    self.reset_object(model_pts, model_normals, symmetry_tfs=symmetry_tfs, mesh=mesh)
    self.make_rotation_grid(min_n_views=coarse_min_n_views, inplane_step=coarse_inplane_step, orientation_mode=coarse_orientation_mode, orientation_tilt_deg=coarse_orientation_tilt_deg, object_up_axis=coarse_object_up_axis)

    self.glctx = glctx

    if scorer is not None:
      self.scorer = scorer
    else:
      self.scorer = ScorePredictor()

    if refiner is not None:
      self.refiner = refiner
    else:
      self.refiner = PoseRefinePredictor()

    self.pose_last = None   # Used for tracking; per the centered mesh


  def reset_object(self, model_pts, model_normals, symmetry_tfs=None, mesh=None):
    max_xyz = mesh.vertices.max(axis=0)
    min_xyz = mesh.vertices.min(axis=0)
    self.model_center = (min_xyz+max_xyz)/2
    if mesh is not None:
      self.mesh_ori = mesh.copy()
      mesh = mesh.copy()
      mesh.vertices = mesh.vertices - self.model_center.reshape(1,3)

    model_pts = mesh.vertices
    self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
    self.vox_size = max(self.diameter/20.0, 0.003)
    # logging.info(f'self.diameter:{self.diameter}, vox_size:{self.vox_size}')
    self.dist_bin = self.vox_size/2
    self.angle_bin = 20  # Deg
    pcd = toOpen3dCloud(model_pts, normals=model_normals)
    pcd = pcd.voxel_down_sample(self.vox_size)
    self.max_xyz = np.asarray(pcd.points).max(axis=0)
    self.min_xyz = np.asarray(pcd.points).min(axis=0)
    self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
    self.normals = F.normalize(torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device='cuda'), dim=-1)
    # logging.info(f'self.pts:{self.pts.shape}')
    self.mesh_path = None
    self.mesh = mesh
    if self.mesh is not None:
      self.mesh_path = f'/tmp/{uuid.uuid4()}.obj'
      self.mesh.export(self.mesh_path)
    self.mesh_tensors = make_mesh_tensors(self.mesh)

    if symmetry_tfs is None:
      self.symmetry_tfs = torch.eye(4).float().cuda()[None]
    else:
      self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)

    logging.info("reset done")



  def get_tf_to_centered_mesh(self):
    tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
    tf_to_center[:3,3] = -torch.as_tensor(self.model_center, device='cuda', dtype=torch.float)
    return tf_to_center


  def to_device(self, s='cuda:0'):
    for k in self.__dict__:
      self.__dict__[k] = self.__dict__[k]
      if torch.is_tensor(self.__dict__[k]) or isinstance(self.__dict__[k], nn.Module):
        # logging.info(f"Moving {k} to device {s}")
        self.__dict__[k] = self.__dict__[k].to(s)
    for k in self.mesh_tensors:
      # logging.info(f"Moving {k} to device {s}")
      self.mesh_tensors[k] = self.mesh_tensors[k].to(s)
    if self.refiner is not None:
      self.refiner.model.to(s)
    if self.scorer is not None:
      self.scorer.model.to(s)

  def _coarse_orientation_ok(self,ob_in_cam: np.ndarray,orientation_mode: str = "uniform",orientation_tilt_deg: float = 30.0,object_up_axis: int = 2):
    """
    orientation_mode:
      "uniform"  : keep original uniform sampling, no filtering
      "upright"  : keep upright + slightly tilted upright poses
      "inverted" : keep inverted + slightly tilted inverted poses

    object_up_axis:
      0 -> object local X is semantic up
      1 -> object local Y is semantic up
      2 -> object local Z is semantic up

    OpenCV camera convention:
      +X right, +Y down, +Z forward.
      Therefore screen up is [0, -1, 0], screen down is [0, 1, 0].
    """
    mode = (orientation_mode or "uniform").strip().lower()

    if mode in ["uniform", "all", "none"]:
      return True, "uniform", 0.0

    if mode not in ["upright", "inverted"]:
      raise ValueError(f"Unknown orientation_mode={orientation_mode}. "f"Use 'uniform', 'upright', or 'inverted'.")

    obj_up_in_cam = ob_in_cam[:3, object_up_axis].astype(np.float64)
    norm = np.linalg.norm(obj_up_in_cam)
    if norm < 1e-12:
      return True, "neutral", 0.0

    obj_up_in_cam = obj_up_in_cam / norm

    screen_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    screen_down = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    if mode == "upright":
      target = screen_up
    else:
      target = screen_down

    cos_angle = float(np.dot(obj_up_in_cam, target))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_deg = float(np.rad2deg(np.arccos(cos_angle)))

    ok = angle_deg <= float(orientation_tilt_deg)
    return ok, mode, angle_deg

  def make_rotation_grid(self,min_n_views=40,inplane_step=60,orientation_mode="uniform",orientation_tilt_deg=30.0,object_up_axis=2):
    """
    Coarse pose sampling.

    orientation_mode:
      "uniform": 原本均勻採樣，不篩姿態
      "upright": 正放 + 正放微微傾斜，orientation_tilt_deg 控制
      "inverted": 倒放 + 倒放微微傾斜，orientation_tilt_deg 控制

    orientation_tilt_deg:
      允許物體 local up 軸偏離目標方向的角度。
      例如 30 表示正放/倒放方向 ±30 度內都保留。

    object_up_axis:
      0:X, 1:Y, 2:Z。
      預設 2，代表物體模型座標 local +Z 是物體上方。
    """
    orientation_mode = (orientation_mode or "uniform").strip().lower()
    cam_in_obs = sample_views_icosphere(n_views=min_n_views)

    rot_grid = []
    orientation_counter = {"kept": 0,"dropped": 0,"uniform": 0,"upright": 0,"inverted": 0,"neutral": 0}

    for i in range(len(cam_in_obs)):
      for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
        cam_in_ob = cam_in_obs[i].copy()
        R_inplane = euler_matrix(0, 0, inplane_rot)
        cam_in_ob = cam_in_ob @ R_inplane

        ob_in_cam = np.linalg.inv(cam_in_ob)

        ok, measured_mode, angle_deg = self._coarse_orientation_ok(ob_in_cam=ob_in_cam,orientation_mode=orientation_mode,orientation_tilt_deg=orientation_tilt_deg,object_up_axis=object_up_axis,)

        if measured_mode in orientation_counter:
          orientation_counter[measured_mode] += 1

        if not ok:
          orientation_counter["dropped"] += 1
          continue

        orientation_counter["kept"] += 1
        rot_grid.append(ob_in_cam)

    rot_grid = np.asarray(rot_grid)

    if len(rot_grid) == 0:
      raise RuntimeError(
        f"No coarse pose left after orientation filtering. "
        f"orientation_mode={orientation_mode}, "
        f"orientation_tilt_deg={orientation_tilt_deg}, "
        f"object_up_axis={object_up_axis}"
      )

    rot_grid = mycpp.cluster_poses(30,99999,rot_grid,self.symmetry_tfs.data.cpu().numpy())

    rot_grid = np.asarray(rot_grid)

    logging.info(f"after cluster, rot_grid:{rot_grid.shape}")

    self.rot_grid = torch.as_tensor(rot_grid, device='cuda', dtype=torch.float)

  def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None):
    '''
    @scene_pts: torch tensor (N,3)
    '''
    ob_in_cams = self.rot_grid.clone()
    center = self.guess_translation(depth=depth, mask=mask, K=K)
    ob_in_cams[:,:3,3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1,3)
    return ob_in_cams


  def guess_translation(self, depth, mask, K):
    vs,us = np.where(mask>0)
    if len(us)==0:
      logging.info(f'mask is all zero')
      return np.zeros((3))
    uc = (us.min()+us.max())/2.0
    vc = (vs.min()+vs.max())/2.0
    valid = mask.astype(bool) & (depth>=0.001)
    if not valid.any():
      logging.info(f"valid is empty")
      return np.zeros((3))

    zc = np.median(depth[valid])
    center = (np.linalg.inv(K)@np.asarray([uc,vc,1]).reshape(3,1))*zc

    if self.debug>=2:
      pcd = toOpen3dCloud(center.reshape(1,3))
      o3d.io.write_point_cloud(f'{self.debug_dir}/init_center.ply', pcd)

    return center.reshape(3)


  def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5, top_k=5, top_flag=False):
    '''Copmute pose from given pts to self.pcd
    @pts: (N,3) np array, downsampled scene points
    '''
    set_seed(0)
    # logging.info('Welcome')

    if self.glctx is None:
      if glctx is None:
        self.glctx = dr.RasterizeCudaContext()
        # self.glctx = dr.RasterizeGLContext()
      else:
        self.glctx = glctx

    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')

    if self.debug>=2:
      xyz_map = depth2xyzmap(depth, K)
      valid = xyz_map[...,2]>=0.001
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply',pcd)
      cv2.imwrite(f'{self.debug_dir}/ob_mask.png', (ob_mask*255.0).clip(0,255))

    normal_map = None
    valid = (depth>=0.001) & (ob_mask>0)
    if valid.sum()<4:
      # logging.info(f'valid too small, return')
      pose = np.eye(4)
      pose[:3,3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
      return pose

    if self.debug>=2:
      imageio.imwrite(f'{self.debug_dir}/color.png', rgb)
      cv2.imwrite(f'{self.debug_dir}/depth.png', (depth*1000).astype(np.uint16))
      valid = xyz_map[...,2]>=0.001
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply',pcd)

    self.H, self.W = depth.shape[:2]
    self.K = K
    self.ob_id = ob_id
    self.ob_mask = ob_mask

    # 1. 產生候選姿態 (Generate random pose hypotheses)
    poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
    poses = poses.data.cpu().numpy()
    # logging.info(f'poses:{poses.shape}')
    center = self.guess_translation(depth=depth, mask=ob_mask, K=K)

    poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
    poses[:,:3,3] = torch.as_tensor(center.reshape(1,3), device='cuda')

    add_errs = self.compute_add_err_to_gt_pose(poses)
    # logging.info(f"after viewpoint, add_errs min:{add_errs.min()}")

    xyz_map = depth2xyzmap(depth, K)
    if top_flag == True:

      # 2. 所有候選姿態送入比對學習 (Scoring all hypotheses without refinement)
      scores_init, _ = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter, get_vis=False)
      
      # 3. 找出相對 TOP-X
      # 避免候選姿態數量少於 top_k，取最小值
      actual_top_k = min(top_k, len(poses))
      ids_init = torch.as_tensor(scores_init).argsort(descending=True)
      top_ids = ids_init[:actual_top_k]
      poses = poses[top_ids]  # 只保留 Top-X 的姿態

    # 4. TOP-X refinement 5 次
    poses_refined, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map, glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration, get_vis=self.debug>=2)
    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)

    # 4.5 Score the learned refined candidates FIRST.
    # Important: do not duplicate candidates here. The previous design scored
    # original+geometry-corrected poses (e.g. 252+252=504), which doubled the
    # scorer memory and could cause CUDA OOM.
    poses = poses_refined

    # 5. Send only RefineNet candidates to the learned scorer and select the best one.
    scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter, get_vis=self.debug>=2)
    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)

    add_errs = self.compute_add_err_to_gt_pose(poses)
    # logging.info(f"final, add_errs min:{add_errs.min()}")

    ids = torch.as_tensor(scores).argsort(descending=True)
    scores = scores[ids]
    poses = poses[ids]

    # Keep the scorer-best pose as fallback.
    best_centered_pose = poses[0:1]
    self.best_id = ids[0]

    # 5.5 Optional depth geometry correction AFTER scorer, using top-k reranking.
    #
    # Safer and more useful than correcting only the single best pose:
    #   candidates -> RefineNet -> scorer -> top-k poses
    #   -> geometry correction for top-k
    #   -> scorer re-ranks [original top-k + corrected top-k]
    #   -> final pose
    #
    # top_k is passed by run_linemod.py:
    #   est.register(..., top_k=20, top_flag=True)
    #
    # If top_flag=True, the same top_k is also used earlier to keep only the
    # best initial hypotheses before RefineNet.  If top_flag=False, this block
    # still uses top_k after the refined scorer.
    depth_mode = str(getattr(self.refiner.cfg, 'depth_refine_mode', 'none')).lower() if hasattr(self.refiner, 'cfg') else 'none'
    if depth_mode not in ['none', 'off', 'false', '0'] and hasattr(self.refiner, 'depth_geometry_refine_poses'):
      actual_refine_top_k = int(max(1, min(int(top_k), len(poses))))
      original_topk = poses[:actual_refine_top_k]

      corrected_topk = self.refiner.depth_geometry_refine_poses(
        mesh=self.mesh,
        mesh_tensors=self.mesh_tensors,
        rgb=rgb,
        depth=depth,
        K=K,
        ob_in_cams=original_topk,
        normal_map=normal_map,
        xyz_map=xyz_map,
        glctx=self.glctx,
        mesh_diameter=self.diameter,
        depth_roi_mask=ob_mask
      )

      if torch.is_tensor(corrected_topk) and len(corrected_topk) > 0:
        corrected_topk = corrected_topk.reshape(-1, 4, 4)

        # Re-rank only a small set to avoid the previous 252+252 OOM problem.
        rerank_poses = torch.cat([original_topk, corrected_topk], dim=0)

        rerank_scores, vis = self.scorer.predict(
          mesh=self.mesh,
          rgb=rgb,
          depth=depth,
          K=K,
          ob_in_cams=rerank_poses.data.cpu().numpy(),
          normal_map=normal_map,
          mesh_tensors=self.mesh_tensors,
          glctx=self.glctx,
          mesh_diameter=self.diameter,
          get_vis=self.debug>=2
        )
        if vis is not None:
          imageio.imwrite(f'{self.debug_dir}/vis_score_depth_topk_rerank.png', vis)

        rerank_ids = torch.as_tensor(rerank_scores).argsort(descending=True)
        rerank_scores_sorted = rerank_scores[rerank_ids]
        rerank_poses_sorted = rerank_poses[rerank_ids]

        best_centered_pose = rerank_poses_sorted[0:1]

        # Store final candidates/scores as reranked small candidate set.
        self.poses = rerank_poses_sorted
        self.scores = rerank_scores_sorted
        self.best_id = rerank_ids[0]

        n_corrected = int(len(corrected_topk))
        chosen = int(rerank_ids[0].item())
        chosen_src = "original_topk" if chosen < actual_refine_top_k else "corrected_topk"
        logging.info(
          f"[DepthRefine] Applied post-score top-k rerank: mode={depth_mode}, "
          f"top_k={actual_refine_top_k}, rerank_candidates={len(rerank_poses)}, "
          f"corrected={n_corrected}, final_from={chosen_src}, "
          f"candidates_scored_before_rerank={len(poses_refined)}"
        )
      else:
        logging.info(f"[DepthRefine] Post-score {depth_mode} returned no corrected top-k poses; keep scorer best pose.")
        self.poses = poses
        self.scores = scores
    else:
      self.poses = poses
      self.scores = scores

    best_pose = best_centered_pose[0] @ self.get_tf_to_centered_mesh()
    self.pose_last = best_centered_pose[0]

    return best_pose.data.cpu().numpy()

  def compute_add_err_to_gt_pose(self, poses):
    '''
    @poses: wrt. the centered mesh
    '''
    return -torch.ones(len(poses), device='cuda', dtype=torch.float)

  def _get_pose_crop_window(self, pose_current, K, H, W):
    """
    計算 Pose-Conditioned Cropping 的視窗範圍。
    回傳: (top, bottom, left, right, center_u, center_v, z_distance) 或 None
    """
    t_est = pose_current[0, :3, 3].cpu().numpy()
    z_distance = t_est[2]
    
    if z_distance <= 0.001:
      return None

    center_2d = K @ t_est
    u = center_2d[0] / z_distance
    v = center_2d[1] / z_distance
    
    scale = (self.diameter / z_distance) * K[0, 0] * 1.2 
    half_s = int(scale / 2)
    
    left, right = max(0, int(u) - half_s), min(W, int(u) + half_s)
    top, bottom = max(0, int(v) - half_s), min(H, int(v) + half_s)
    
    if right <= left or bottom <= top:
      return None 

    return (top, bottom, left, right, int(u - left), int(v - top), z_distance)
  
  def _compute_depth_confidence(self, crop_real_depth, crop_render_depth, z_distance):
    """
    計算MAE與Inlier Ratio。
    回傳: (depth_mae, inlier_ratio, inliers, crop_render_mask)
    """
    import torch
    import numpy as np

    crop_render_mask = (crop_render_depth > 1e-3)
    valid_area = crop_render_mask.sum().float()
    
    if valid_area <= 0:
      return 999.0, 0.0, None, None

    depth_diff = torch.abs(crop_real_depth - crop_render_depth)
    
    # 自適應閾值公式
    dynamic_tolerance = np.clip((z_distance * 0.015) + (self.diameter * 0.05), 0.01, 0.08)
    
    # 計算 MAE
    valid_obs = (crop_real_depth > 1e-3) & crop_render_mask
    if valid_obs.sum() > 0:
      depth_mae = depth_diff[valid_obs].mean().item()
    else:
      depth_mae = 999.0
        
    # 計算 Inlier Ratio
    inliers = (depth_diff < dynamic_tolerance) & (crop_real_depth > 1e-3) & crop_render_mask
    inlier_ratio = (inliers.sum().float() / valid_area).item()

    return depth_mae, inlier_ratio, inliers, crop_render_mask
  
  def _save_debug_overlay(self, rgb, top, bottom, left, right, center_u, center_v, inliers, crop_render_mask, inlier_ratio, depth_mae):
    """
    將局部裁切框與信心指標繪製成除錯影像。
    """
    import time

    inliers_np = inliers.cpu().numpy().astype(np.uint8) * 255
    render_mask_np = crop_render_mask.cpu().numpy().astype(np.uint8) * 255
    
    overlay = rgb.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 255), 2)
    crop_roi = overlay[top:bottom, left:right]
    
    red_layer = np.zeros_like(crop_roi)
    red_layer[:, :, 2] = 255  
    green_layer = np.zeros_like(crop_roi)
    green_layer[:, :, 1] = 255 
    
    cv2.addWeighted(crop_roi, 1.0, red_layer, 0.7, 0, crop_roi, dtype=cv2.CV_8U)
    crop_roi[inliers_np == 0] = rgb[top:bottom, left:right][inliers_np == 0] 
    
    temp_roi = crop_roi.copy()
    cv2.addWeighted(temp_roi, 1.0, green_layer, 0.4, 0, temp_roi, dtype=cv2.CV_8U)
    crop_roi[render_mask_np > 0] = temp_roi[render_mask_np > 0]
    
    cv2.drawMarker(crop_roi, (center_u, center_v), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
    
    overlay[top:bottom, left:right] = crop_roi
    
    cv2.putText(overlay, f"Crop Inlier: {inlier_ratio:.2f}", (left, max(20, top - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"Crop MAE: {depth_mae*100:.1f}cm", (left, max(45, top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
    
    timestamp = int(time.time() * 1000)
    save_path = f"{self.debug_dir}/crop_conf_{timestamp}.png"
    cv2.imwrite(save_path, overlay)

  def track_one(self, rgb, depth, K, iteration, extra={}, enable_self_check=True):
    if self.pose_last is None:
      logging.info("Please init pose by register first")
      raise RuntimeError

    depth_tensor = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    depth_tensor = erode_depth(depth_tensor, radius=2, device='cuda')
    depth_tensor = bilateral_filter_depth(depth_tensor, radius=2, device='cuda')

    xyz_map = depth2xyzmap_batch(depth_tensor[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]

    pose_current = self.pose_last.reshape(1, 4, 4)
    H, W = rgb.shape[:2]

    # Refinement
    pose_next, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=pose_current.data.cpu().numpy() if torch.is_tensor(pose_current) else pose_current, normal_map=None, xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx, iteration=1, get_vis=self.debug>=2)
    pose_current = torch.as_tensor(pose_next, device='cuda', dtype=torch.float)

    if enable_self_check:
      _, render_depth, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pose_current, glctx=self.glctx, mesh_tensors=self.mesh_tensors)
      render_depth_2d = render_depth[0, ..., 0] if render_depth.ndim == 4 else render_depth[0]
      
      # 取得裁切視窗
      crop_info = self._get_pose_crop_window(pose_current, K, H, W)
      
      if crop_info is not None:
        top, bottom, left, right, center_u, center_v, z_dist = crop_info
        
        # 計算信心分數
        depth_mae, inlier_ratio, inliers, render_mask = self._compute_depth_confidence(depth_tensor[top:bottom, left:right], render_depth_2d[top:bottom, left:right], z_dist)
        
        # 視覺化除錯
        # if self.debug >= 1 and inliers is not None:
        #   self._save_debug_overlay(rgb, top, bottom, left, right, center_u, center_v, inliers, render_mask, inlier_ratio, depth_mae)
      else:
        depth_mae, inlier_ratio = 999.0, 0.0

      extra['depth_mae'] = depth_mae
      extra['inlier_ratio'] = inlier_ratio

    if self.debug >= 2:
        extra['vis'] = vis
    
    self.pose_last = pose_current[0]
    
    return (self.pose_last @ self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4)
