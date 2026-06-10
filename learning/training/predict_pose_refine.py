# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import functools
import os,sys,kornia
import time
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../')
import numpy as np
import torch
from omegaconf import OmegaConf
from learning.models.refine_network import RefineNet
from learning.datasets.h5_dataset import *
from Utils import *
from datareader import *
import scipy.spatial as spatial


@torch.inference_mode()
def make_crop_data_batch(render_size, ob_in_cams, mesh, rgb, depth, K, crop_ratio, xyz_map, normal_map=None, mesh_diameter=None, cfg=None, glctx=None, mesh_tensors=None, dataset:PoseRefinePairH5Dataset=None):
  # logging.info("Welcome make_crop_data_batch")
  H,W = depth.shape[:2]
  args = []
  method = 'box_3d'
  tf_to_crops = compute_crop_window_tf_batch(pts=mesh.vertices, H=H, W=W, poses=ob_in_cams, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), method=method, mesh_diameter=mesh_diameter)

  # logging.info("make tf_to_crops done")

  B = len(ob_in_cams)
  poseA = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')

  bs = 512
  rgb_rs = []
  depth_rs = []
  normal_rs = []
  xyz_map_rs = []

  bbox2d_crop = torch.as_tensor(np.array([0, 0, cfg['input_resize'][0]-1, cfg['input_resize'][1]-1]).reshape(2,2), device='cuda', dtype=torch.float)
  bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()).reshape(-1,4)

  for b in range(0,len(poseA),bs):
    extra = {}
    rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=poseA[b:b+bs], context='cuda', get_normal=cfg['use_normal'], glctx=glctx, mesh_tensors=mesh_tensors, output_size=cfg['input_resize'], bbox2d=bbox2d_ori[b:b+bs], use_light=True, extra=extra)
    rgb_rs.append(rgb_r)
    depth_rs.append(depth_r[...,None])
    normal_rs.append(normal_r)
    xyz_map_rs.append(extra['xyz_map'])
  rgb_rs = torch.cat(rgb_rs, dim=0).permute(0,3,1,2) * 255
  depth_rs = torch.cat(depth_rs, dim=0).permute(0,3,1,2)  #(B,1,H,W)
  xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0,3,1,2)  #(B,3,H,W)
  Ks = torch.as_tensor(K, device='cuda', dtype=torch.float).reshape(1,3,3)
  if cfg['use_normal']:
    normal_rs = torch.cat(normal_rs, dim=0).permute(0,3,1,2)  #(B,3,H,W)

  # logging.info("render done")

  rgbBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(rgb, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
  if rgb_rs.shape[-2:]!=cfg['input_resize']:
    rgbAs = kornia.geometry.transform.warp_perspective(rgb_rs, tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
  else:
    rgbAs = rgb_rs
  if xyz_map_rs.shape[-2:]!=cfg['input_resize']:
    xyz_mapAs = kornia.geometry.transform.warp_perspective(xyz_map_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    xyz_mapAs = xyz_map_rs
  xyz_mapBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(xyz_map, device='cuda', dtype=torch.float).permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)  #(B,3,H,W)

  if cfg['use_normal']:
    normalAs = kornia.geometry.transform.warp_perspective(normal_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
    normalBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(normal_map, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    normalAs = None
    normalBs = None

  # logging.info("warp done")

  mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device='cuda')*mesh_diameter
  pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=None, depthBs=None, normalAs=normalAs, normalBs=normalBs, poseA=poseA, poseB=None, xyz_mapAs=xyz_mapAs, xyz_mapBs=xyz_mapBs, tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters)
  pose_data = dataset.transform_batch(batch=pose_data, H_ori=H, W_ori=W, bound=1)

  # logging.info("pose batch data done")

  return pose_data

# ============================================================
# Test-time depth geometry refinement
# ------------------------------------------------------------
# These modules do NOT change the trained RefineNet output.
# The network remains:
#   rot_rep   = axis_angle
#   trans_rep = tracknet
#
# Optional post-refinement corrections:
#   none : original FoundationPose refiner output
#   ndp  : Non-iterative Depth Projection correction using pixel-wise
#          rendered xyz vs observed xyz in the pose-conditioned crop.
#   icp  : nearest-neighbor ICP correction using rendered xyz and observed
#          xyz point clouds in the pose-conditioned crop.
#
# depth_refine_apply:
#   trans : only apply translation correction
#   se3   : apply rotation + translation geometric correction.
#           This can be described in experiments as axis_angle+ICP/NDP,
#           but it does NOT change neural rot_rep; it only adds a geometric
#           SE(3) correction after the axis-angle network update.
# ============================================================

def _cfg_get(cfg, key, default):
  try:
    if key in cfg:
      return cfg[key]
  except Exception:
    pass
  return default


def _as_numpy(x):
  if torch.is_tensor(x):
    return x.detach().cpu().numpy()
  return np.asarray(x)


def _rotation_angle_deg(R):
  R = np.asarray(R, dtype=np.float64)
  c = (np.trace(R) - 1.0) * 0.5
  c = np.clip(c, -1.0, 1.0)
  return float(np.degrees(np.arccos(c)))


def _rotation_log(R):
  R = np.asarray(R, dtype=np.float64)
  c = (np.trace(R) - 1.0) * 0.5
  c = np.clip(c, -1.0, 1.0)
  theta = np.arccos(c)
  if theta < 1e-8:
    return np.zeros(3, dtype=np.float64)
  w = np.array([
    R[2,1] - R[1,2],
    R[0,2] - R[2,0],
    R[1,0] - R[0,1],
  ], dtype=np.float64) * (0.5 / np.sin(theta))
  return w * theta


def _rotation_exp(w):
  w = np.asarray(w, dtype=np.float64).reshape(3)
  theta = np.linalg.norm(w)
  if theta < 1e-12:
    return np.eye(3, dtype=np.float64)
  k = w / theta
  K = np.array([
    [0, -k[2], k[1]],
    [k[2], 0, -k[0]],
    [-k[1], k[0], 0],
  ], dtype=np.float64)
  return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _clamp_se3(R, t, max_rot_deg=5.0, max_trans=0.03):
  R = np.asarray(R, dtype=np.float64)
  t = np.asarray(t, dtype=np.float64).reshape(3)

  if max_rot_deg is not None and max_rot_deg > 0:
    angle = _rotation_angle_deg(R)
    if angle > max_rot_deg:
      w = _rotation_log(R)
      w = w * (float(max_rot_deg) / max(angle, 1e-9))
      R = _rotation_exp(w)

  if max_trans is not None and max_trans > 0:
    n = np.linalg.norm(t)
    if n > max_trans:
      t = t * (float(max_trans) / max(n, 1e-9))

  return R, t


def _kabsch(A, B):
  # Solve R,t such that B ~= R @ A + t.
  A = np.asarray(A, dtype=np.float64)
  B = np.asarray(B, dtype=np.float64)
  if len(A) < 3:
    return np.eye(3), np.zeros(3), False

  ca = A.mean(axis=0)
  cb = B.mean(axis=0)
  AA = A - ca
  BB = B - cb
  H = AA.T @ BB

  try:
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
      Vt[-1, :] *= -1
      R = Vt.T @ U.T
    t = cb - R @ ca
    return R, t, True
  except Exception:
    return np.eye(3), np.zeros(3), False


def _sample_points_pair(A_pts, B_pts, max_points=2048):
  n = len(A_pts)
  if n <= max_points:
    return A_pts, B_pts
  ids = np.random.choice(n, size=max_points, replace=False)
  return A_pts[ids], B_pts[ids]


def _extract_crop_xyz_pairs(pose_data,idx,z_min=1e-6,z_max=np.inf,depth_diff_thresh=0.05,max_points=2048,roi_mask_crop=None,):
  xyzA = _as_numpy(pose_data.xyz_mapAs[idx]).transpose(1, 2, 0)
  xyzB = _as_numpy(pose_data.xyz_mapBs[idx]).transpose(1, 2, 0)

  valid = np.isfinite(xyzA).all(axis=-1) & np.isfinite(xyzB).all(axis=-1)
  valid &= xyzA[..., 2] > z_min
  valid &= xyzB[..., 2] > z_min

  if roi_mask_crop is not None:
    m = _as_numpy(roi_mask_crop)
    if m.ndim == 3:
        m = m[0]
    valid &= m > 0.5

  if np.isfinite(z_max):
    valid &= xyzA[..., 2] < z_max
    valid &= xyzB[..., 2] < z_max

  if depth_diff_thresh is not None and depth_diff_thresh > 0:
    valid &= np.abs(xyzA[..., 2] - xyzB[..., 2]) < depth_diff_thresh

  A = xyzA[valid].reshape(-1, 3)
  B = xyzB[valid].reshape(-1, 3)

  if len(A) == 0:
    return A, B

  return _sample_points_pair(A, B, max_points=max_points)


def _pose_error_mean(A, B, R=None, t=None):
  if len(A) == 0:
    return np.inf
  if R is not None and t is not None:
    A2 = (R @ A.T).T + t.reshape(1,3)
  else:
    A2 = A
  return float(np.mean(np.linalg.norm(A2 - B, axis=1)))


def _ndp_correction_from_pairs(A, B, apply='trans', max_rot_deg=5.0, max_trans=0.03):
  # NDP: non-iterative pixel-wise depth projection correction.
  if len(A) < 3:
    return np.eye(3), np.zeros(3), False

  if apply == 'trans':
    residual = B - A
    t = np.median(residual, axis=0)
    R = np.eye(3, dtype=np.float64)
    R, t = _clamp_se3(R, t, max_rot_deg=max_rot_deg, max_trans=max_trans)
    return R, t, True

  R, t, ok = _kabsch(A, B)
  if not ok:
    return np.eye(3), np.zeros(3), False
  R, t = _clamp_se3(R, t, max_rot_deg=max_rot_deg, max_trans=max_trans)
  return R, t, True


def _icp_correction(A, B, apply='trans', max_corr_dist=0.03, max_iter=5, max_rot_deg=5.0, max_trans=0.03, min_points=30):
  # ICP correction using rendered crop xyz as source and observed crop xyz as target.
  if len(A) < min_points or len(B) < min_points:
    return np.eye(3), np.zeros(3), False

  src0 = np.asarray(A, dtype=np.float64)
  tgt = np.asarray(B, dtype=np.float64)

  if len(src0) > len(tgt):
    src0 = src0[np.random.choice(len(src0), size=len(tgt), replace=False)]

  R_total = np.eye(3, dtype=np.float64)
  t_total = np.zeros(3, dtype=np.float64)

  tree = spatial.cKDTree(tgt)

  ok_any = False
  for _ in range(int(max_iter)):
    src = (R_total @ src0.T).T + t_total.reshape(1,3)
    dists, ids = tree.query(src, k=1, workers=-1)

    valid = np.isfinite(dists)
    if max_corr_dist is not None and max_corr_dist > 0:
      valid &= dists < max_corr_dist

    if valid.sum() < min_points:
      break

    A_corr = src[valid]
    B_corr = tgt[ids[valid]]

    if apply == 'trans':
      step_t = np.median(B_corr - A_corr, axis=0)
      step_R = np.eye(3, dtype=np.float64)
      step_ok = True
    else:
      step_R, step_t, step_ok = _kabsch(A_corr, B_corr)

    if not step_ok:
      break

    step_R, step_t = _clamp_se3(step_R, step_t, max_rot_deg=max_rot_deg, max_trans=max_trans)

    R_total = step_R @ R_total
    t_total = step_R @ t_total + step_t
    ok_any = True

    if np.linalg.norm(step_t) < 1e-5 and _rotation_angle_deg(step_R) < 0.05:
      break

  if not ok_any:
    return np.eye(3), np.zeros(3), False

  R_total, t_total = _clamp_se3(R_total, t_total, max_rot_deg=max_rot_deg, max_trans=max_trans)
  return R_total, t_total, True


def _apply_corr_to_pose(T, R_corr, t_corr, apply='trans'):
  T = np.asarray(T, dtype=np.float64).copy()
  if apply == 'trans':
    T[:3, 3] = T[:3, 3] + np.asarray(t_corr, dtype=np.float64).reshape(3)
    return T

  Tc = np.eye(4, dtype=np.float64)
  Tc[:3,:3] = R_corr
  Tc[:3,3] = t_corr
  return Tc @ T

class PoseRefinePredictor:
  def __init__(self,):
    logging.info("welcome")
    self.amp = True
    self.run_name = "2023-10-28-18-33-37"
    model_name = 'model_best.pth'
    code_dir = os.path.dirname(os.path.realpath(__file__))
    ckpt_dir = f'{code_dir}/../../weights/{self.run_name}/{model_name}'

    self.cfg = OmegaConf.load(f'{code_dir}/../../weights/{self.run_name}/config.yml')

    self.cfg['ckpt_dir'] = ckpt_dir
    self.cfg['enable_amp'] = True

    ########## Defaults, to be backward compatible
    if 'use_normal' not in self.cfg:
      self.cfg['use_normal'] = False
    if 'use_mask' not in self.cfg:
      self.cfg['use_mask'] = False
    if 'use_BN' not in self.cfg:
      self.cfg['use_BN'] = False
    if 'c_in' not in self.cfg:
      self.cfg['c_in'] = 4
    if 'crop_ratio' not in self.cfg or self.cfg['crop_ratio'] is None:
      self.cfg['crop_ratio'] = 1.2
    if 'n_view' not in self.cfg:
      self.cfg['n_view'] = 1
    if 'trans_rep' not in self.cfg:
      self.cfg['trans_rep'] = 'tracknet'
    if 'rot_rep' not in self.cfg:
      self.cfg['rot_rep'] = 'axis_angle'
    if 'zfar' not in self.cfg:
      self.cfg['zfar'] = 3
    if 'normalize_xyz' not in self.cfg:
      self.cfg['normalize_xyz'] = False
    # ============================================================
    if 'depth_refine_mode' not in self.cfg:
      self.cfg['depth_refine_mode'] = os.environ.get('FP_DEPTH_REFINE_MODE', 'none')  # none / icp / ndp
    if 'depth_refine_apply' not in self.cfg:
      self.cfg['depth_refine_apply'] = os.environ.get('FP_DEPTH_REFINE_APPLY', 'trans')  # trans / se3
    if 'depth_refine_accept_if_better' not in self.cfg:
      self.cfg['depth_refine_accept_if_better'] = True
    if 'depth_refine_max_points' not in self.cfg:
      self.cfg['depth_refine_max_points'] = 2048
    if 'depth_refine_min_points' not in self.cfg:
      self.cfg['depth_refine_min_points'] = 50
    if 'depth_refine_max_corr_dist' not in self.cfg:
      self.cfg['depth_refine_max_corr_dist'] = 0.03
    if 'depth_refine_icp_iter' not in self.cfg:
      self.cfg['depth_refine_icp_iter'] = 5
    if 'depth_refine_depth_diff_thresh' not in self.cfg:
      self.cfg['depth_refine_depth_diff_thresh'] = 0.05
    if 'depth_refine_trans_clamp' not in self.cfg:
      self.cfg['depth_refine_trans_clamp'] = 0.03
    if 'depth_refine_rot_clamp_deg' not in self.cfg:
      self.cfg['depth_refine_rot_clamp_deg'] = 5.0
    if 'depth_refine_log' not in self.cfg:
      self.cfg['depth_refine_log'] = True
    # ============================================================
    if isinstance(self.cfg['zfar'], str) and 'inf' in self.cfg['zfar'].lower():
      self.cfg['zfar'] = np.inf
    if 'normal_uint8' not in self.cfg:
      self.cfg['normal_uint8'] = False
    # logging.info(f"self.cfg: \n {OmegaConf.to_yaml(self.cfg)}")

    self.dataset = PoseRefinePairH5Dataset(cfg=self.cfg, h5_file='', mode='test')
    self.model = RefineNet(cfg=self.cfg, c_in=self.cfg['c_in']).cuda()

    # logging.info(f"Using pretrained model from {ckpt_dir}")
    ckpt = torch.load(ckpt_dir)
    if 'model' in ckpt:
      ckpt = ckpt['model']
    self.model.load_state_dict(ckpt, strict=False)

    self.model.cuda().eval()
    # logging.info("init done")
    self.last_trans_update = None
    self.last_rot_update = None

  def set_depth_refine(self, mode=None, apply=None, **kwargs):
    """
    Runtime setter for test-time depth geometry refinement.

    Example:
      refiner.set_depth_refine(mode='icp', apply='trans')
      refiner.set_depth_refine(mode='ndp', apply='se3', depth_refine_trans_clamp=0.02)

    mode:
      none / icp / ndp
    apply:
      trans / se3
    """
    if mode is not None:
      self.cfg['depth_refine_mode'] = str(mode).lower()
    if apply is not None:
      self.cfg['depth_refine_apply'] = str(apply).lower()
    for k, v in kwargs.items():
      self.cfg[k] = v

  def _depth_geometry_refine_batch(self, B_in_cams, rgb_tensor, depth_tensor, K, xyz_map_tensor, normal_map, mesh_centered, glctx, mesh_tensors, mesh_diameter, depth_roi_mask=None):
    mode = str(_cfg_get(self.cfg, 'depth_refine_mode', 'none')).lower()
    apply = str(_cfg_get(self.cfg, 'depth_refine_apply', 'trans')).lower()

    if mode in ['none', 'off', 'false', '0']:
      return B_in_cams

    if mode not in ['icp', 'ndp']:
      logging.warning(f"[DepthRefine] Unknown mode={mode}, skip.")
      return B_in_cams

    if apply not in ['trans', 'se3']:
      logging.warning(f"[DepthRefine] Unknown apply={apply}, use trans.")
      apply = 'trans'

    # Build pose-conditioned crops for the final network-refined poses.
    pose_data = make_crop_data_batch(
      self.cfg.input_resize,
      B_in_cams,
      mesh_centered,
      rgb_tensor,
      depth_tensor,
      K,
      crop_ratio=self.cfg['crop_ratio'],
      normal_map=normal_map,
      xyz_map=xyz_map_tensor,
      cfg=self.cfg,
      glctx=glctx,
      mesh_tensors=mesh_tensors,
      dataset=self.dataset,
      mesh_diameter=mesh_diameter
    )
    roi_mask_crops = None
    if depth_roi_mask is not None:
      mask_tensor = torch.as_tensor(depth_roi_mask, device='cuda', dtype=torch.float)

      if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor[None, None]  # 1,1,H,W
      elif mask_tensor.ndim == 3:
        mask_tensor = mask_tensor[:, None]

      mask_tensor = mask_tensor.expand(len(B_in_cams), -1, -1, -1)

      roi_mask_crops = kornia.geometry.transform.warp_perspective(
        mask_tensor,
        pose_data.tf_to_crops,
        dsize=self.cfg.input_resize,
        mode='nearest',
        align_corners=False
      )

    poses_np = _as_numpy(B_in_cams).copy()

    max_points = int(_cfg_get(self.cfg, 'depth_refine_max_points', 2048))
    min_points = int(_cfg_get(self.cfg, 'depth_refine_min_points', 50))
    max_corr_dist = float(_cfg_get(self.cfg, 'depth_refine_max_corr_dist', 0.03))
    icp_iter = int(_cfg_get(self.cfg, 'depth_refine_icp_iter', 5))
    depth_diff_thresh = float(_cfg_get(self.cfg, 'depth_refine_depth_diff_thresh', 0.05))
    max_trans = float(_cfg_get(self.cfg, 'depth_refine_trans_clamp', 0.03))
    max_rot_deg = float(_cfg_get(self.cfg, 'depth_refine_rot_clamp_deg', 5.0))
    accept_if_better = bool(_cfg_get(self.cfg, 'depth_refine_accept_if_better', True))
    do_log = bool(_cfg_get(self.cfg, 'depth_refine_log', True))

    n_ok = 0
    n_try = 0

    for i in range(len(poses_np)):
      roi_mask_crop = None
      if roi_mask_crops is not None:
        roi_mask_crop = roi_mask_crops[i]

      A, B = _extract_crop_xyz_pairs(pose_data,i,z_min=1e-6,z_max=_cfg_get(self.cfg, 'zfar', np.inf),depth_diff_thresh=depth_diff_thresh,max_points=max_points,roi_mask_crop=roi_mask_crop)

      if len(A) < min_points or len(B) < min_points:
        continue

      n_try += 1
      before_err = _pose_error_mean(A, B)

      if mode == 'ndp':
        R_corr, t_corr, ok = _ndp_correction_from_pairs(
          A, B,
          apply=apply,
          max_rot_deg=max_rot_deg,
          max_trans=max_trans
        )
      else:
        R_corr, t_corr, ok = _icp_correction(
          A, B,
          apply=apply,
          max_corr_dist=max_corr_dist,
          max_iter=icp_iter,
          max_rot_deg=max_rot_deg,
          max_trans=max_trans,
          min_points=min_points
        )

      if not ok:
        continue

      after_err = _pose_error_mean(A, B, R=R_corr, t=t_corr)

      if accept_if_better and not (after_err < before_err):
        continue

      poses_np[i] = _apply_corr_to_pose(poses_np[i], R_corr, t_corr, apply=apply)
      n_ok += 1

    if do_log:
      logging.info(
        f"[DepthRefine] mode={mode}, apply={apply}, "
        f"accepted={n_ok}/{n_try}, candidates={len(poses_np)}, "
        f"max_trans={max_trans}m, max_rot={max_rot_deg}deg"
      )

    return torch.as_tensor(poses_np, device='cuda', dtype=torch.float)

  @torch.inference_mode()
  def predict(self, rgb, depth, K, ob_in_cams, xyz_map, normal_map=None,get_vis=False, mesh=None, mesh_tensors=None, glctx=None,mesh_diameter=None, iteration=5, depth_roi_mask=None):
    '''
    @rgb: np array (H,W,3)
    @ob_in_cams: np array (N,4,4)
    '''
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # logging.info(f'ob_in_cams:{ob_in_cams.shape}')
    tf_to_center = np.eye(4)
    ob_centered_in_cams = ob_in_cams
    mesh_centered = mesh

    # logging.info(f'self.cfg.use_normal:{self.cfg.use_normal}')
    if not self.cfg.use_normal:
      normal_map = None

    crop_ratio = self.cfg['crop_ratio']
    # logging.info(f"trans_normalizer:{self.cfg['trans_normalizer']}, rot_normalizer:{self.cfg['rot_normalizer']}")
    bs = 1024

    B_in_cams = torch.as_tensor(ob_centered_in_cams, device='cuda', dtype=torch.float)


    if mesh_tensors is None:
      mesh_tensors = make_mesh_tensors(mesh_centered)

    rgb_tensor = torch.as_tensor(rgb, device='cuda', dtype=torch.float)
    depth_tensor = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    xyz_map_tensor = torch.as_tensor(xyz_map, device='cuda', dtype=torch.float)
    trans_normalizer = self.cfg['trans_normalizer']
    if not isinstance(trans_normalizer, float):
      trans_normalizer = torch.as_tensor(list(trans_normalizer), device='cuda', dtype=torch.float).reshape(1,3)
    
    for _ in range(iteration):
      # logging.info("making cropped data")
      pose_data = make_crop_data_batch(self.cfg.input_resize, B_in_cams, mesh_centered, rgb_tensor, depth_tensor, K, crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor, cfg=self.cfg, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter)
      B_in_cams = []
      for b in range(0, pose_data.rgbAs.shape[0], bs):
        A = torch.cat([pose_data.rgbAs[b:b+bs].cuda(), pose_data.xyz_mapAs[b:b+bs].cuda()], dim=1).float()
        B = torch.cat([pose_data.rgbBs[b:b+bs].cuda(), pose_data.xyz_mapBs[b:b+bs].cuda()], dim=1).float()
        # logging.info("forward start")
        with torch.cuda.amp.autocast(enabled=self.amp):
          output = self.model(A,B)
        for k in output:
          output[k] = output[k].float()
        # logging.info("forward done")

        if self.cfg['trans_rep']=='tracknet':
          if not self.cfg['normalize_xyz']:
            trans_delta = torch.tanh(output["trans"])*trans_normalizer
          else:
            trans_delta = output["trans"]

        elif self.cfg['trans_rep']=='deepim':
          def project_and_transform_to_crop(centers):
            uvs = (pose_data.Ks[b:b+bs]@centers.reshape(-1,3,1)).reshape(-1,3)
            uvs = uvs/uvs[:,2:3]
            uvs = (pose_data.tf_to_crops[b:b+bs]@uvs.reshape(-1,3,1)).reshape(-1,3)
            return uvs[:,:2]

          rot_delta = output["rot"]
          z_pred = output['trans'][:,2]*pose_data.poseA[b:b+bs][...,2,3]
          uvA_crop = project_and_transform_to_crop(pose_data.poseA[b:b+bs][...,:3,3])
          uv_pred_crop = uvA_crop + output['trans'][:,:2]*self.cfg['input_resize'][0]
          uv_pred = transform_pts(uv_pred_crop, pose_data.tf_to_crops[b:b+bs].inverse().cuda())
          center_pred = torch.cat([uv_pred, torch.ones((len(rot_delta),1), dtype=torch.float, device='cuda')], dim=-1)
          center_pred = (pose_data.Ks[b:b+bs].inverse().cuda()@center_pred.reshape(len(rot_delta),3,1)).reshape(len(rot_delta),3) * z_pred.reshape(len(rot_delta),1)
          trans_delta = center_pred-pose_data.poseA[b:b+bs][...,:3,3]

        else:
          trans_delta = output["trans"]

        if self.cfg['rot_rep']=='axis_angle':
          rot_mat_delta = torch.tanh(output["rot"])*self.cfg['rot_normalizer']
          rot_mat_delta = so3_exp_map(rot_mat_delta).permute(0,2,1)
        elif self.cfg['rot_rep']=='6d':
          rot_mat_delta = rotation_6d_to_matrix(output['rot']).permute(0,2,1)
        else:
          raise RuntimeError

        if self.cfg['normalize_xyz']:
          trans_delta *= (mesh_diameter/2)

        B_in_cam = egocentric_delta_pose_to_pose(pose_data.poseA[b:b+bs], trans_delta=trans_delta, rot_mat_delta=rot_mat_delta)
        B_in_cams.append(B_in_cam)

      B_in_cams = torch.cat(B_in_cams, dim=0).reshape(len(ob_in_cams),4,4)
    # Optional test-time depth geometry correction.
    # Keep neural representation unchanged:
    #   rot_rep   = axis_angle
    #   trans_rep = tracknet
    # Then optionally add:
    #   none / ICP / NDP, either translation-only or SE(3).
    B_in_cams = self._depth_geometry_refine_batch(
      B_in_cams=B_in_cams,
      rgb_tensor=rgb_tensor,
      depth_tensor=depth_tensor,
      K=K,
      xyz_map_tensor=xyz_map_tensor,
      normal_map=normal_map,
      mesh_centered=mesh_centered,
      glctx=glctx,
      mesh_tensors=mesh_tensors,
      mesh_diameter=mesh_diameter,
      depth_roi_mask=depth_roi_mask
    )

    B_in_cams_out = B_in_cams@torch.tensor(tf_to_center[None], device='cuda', dtype=torch.float)
    
    torch.cuda.empty_cache()

    self.last_trans_update = trans_delta
    self.last_rot_update = rot_mat_delta
    # logging.info(f"model: {self.model}")
    if get_vis:
      # logging.info("get_vis...")
      canvas = []
      padding = 2
      pose_data = make_crop_data_batch(self.cfg.input_resize, torch.as_tensor(ob_centered_in_cams), mesh_centered, rgb, depth, K, crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor, cfg=self.cfg, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter)
      for id in range(0, len(B_in_cams)):
        rgbA_vis = (pose_data.rgbAs[id]*255).permute(1,2,0).data.cpu().numpy()
        rgbB_vis = (pose_data.rgbBs[id]*255).permute(1,2,0).data.cpu().numpy()
        row = [rgbA_vis, rgbB_vis]
        H,W = rgbA_vis.shape[:2]
        if pose_data.depthAs is not None:
          depthA = pose_data.depthAs[id].data.cpu().numpy().reshape(H,W)
          depthB = pose_data.depthBs[id].data.cpu().numpy().reshape(H,W)
        elif pose_data.xyz_mapAs is not None:
          depthA = pose_data.xyz_mapAs[id][2].data.cpu().numpy().reshape(H,W)
          depthB = pose_data.xyz_mapBs[id][2].data.cpu().numpy().reshape(H,W)
        zmin = min(depthA.min(), depthB.min())
        zmax = max(depthA.max(), depthB.max())
        depthA_vis = depth_to_vis(depthA, zmin=zmin, zmax=zmax, inverse=False)
        depthB_vis = depth_to_vis(depthB, zmin=zmin, zmax=zmax, inverse=False)
        row += [depthA_vis, depthB_vis]
        if pose_data.normalAs is not None:
          pass
        row = make_grid_image(row, nrow=len(row), padding=padding, pad_value=255)
        row = cv_draw_text(row, text=f'id:{id}', uv_top_left=(10,10), color=(0,255,0), fontScale=0.5)
        canvas.append(row)
      canvas = make_grid_image(canvas, nrow=1, padding=padding, pad_value=255)

      pose_data = make_crop_data_batch(self.cfg.input_resize, B_in_cams, mesh_centered, rgb, depth, K, crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor, cfg=self.cfg, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter)
      canvas_refined = []
      for id in range(0, len(B_in_cams)):
        rgbA_vis = (pose_data.rgbAs[id]*255).permute(1,2,0).data.cpu().numpy()
        rgbB_vis = (pose_data.rgbBs[id]*255).permute(1,2,0).data.cpu().numpy()
        row = [rgbA_vis, rgbB_vis]
        H,W = rgbA_vis.shape[:2]
        if pose_data.depthAs is not None:
          depthA = pose_data.depthAs[id].data.cpu().numpy().reshape(H,W)
          depthB = pose_data.depthBs[id].data.cpu().numpy().reshape(H,W)
        elif pose_data.xyz_mapAs is not None:
          depthA = pose_data.xyz_mapAs[id][2].data.cpu().numpy().reshape(H,W)
          depthB = pose_data.xyz_mapBs[id][2].data.cpu().numpy().reshape(H,W)
        zmin = min(depthA.min(), depthB.min())
        zmax = max(depthA.max(), depthB.max())
        depthA_vis = depth_to_vis(depthA, zmin=zmin, zmax=zmax, inverse=False)
        depthB_vis = depth_to_vis(depthB, zmin=zmin, zmax=zmax, inverse=False)
        row += [depthA_vis, depthB_vis]
        row = make_grid_image(row, nrow=len(row), padding=padding, pad_value=255)
        canvas_refined.append(row)

      canvas_refined = make_grid_image(canvas_refined, nrow=1, padding=padding, pad_value=255)
      canvas = make_grid_image([canvas, canvas_refined], nrow=2, padding=padding, pad_value=255)
      torch.cuda.empty_cache()
      return B_in_cams_out, canvas

    return B_in_cams_out, None

