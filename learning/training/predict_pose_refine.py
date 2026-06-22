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
from pathlib import Path
import yaml
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

  # Keep raw camera-frame xyz maps before dataset.transform_batch().
  # The network needs the transformed tensors, but geometry refinement must
  # estimate metric corrections in the camera coordinate system.
  raw_xyz_mapAs = xyz_mapAs.clone()
  raw_xyz_mapBs = xyz_mapBs.clone()
  raw_tf_to_crops = tf_to_crops.clone()

  pose_data = dataset.transform_batch(batch=pose_data, H_ori=H, W_ori=W, bound=1)

  # Dynamically attach raw camera-frame crop tensors for ICP/NDT/GICP/VGICP.
  pose_data.raw_xyz_mapAs = raw_xyz_mapAs
  pose_data.raw_xyz_mapBs = raw_xyz_mapBs
  pose_data.raw_tf_to_crops = raw_tf_to_crops

  # logging.info("pose batch data done")

  return pose_data

# ============================================================
# Test-time depth geometry refinement
# ------------------------------------------------------------
# These modules do NOT change the trained RefineNet output.
# The network remains checkpoint-compatible:
#   rot_rep   = axis_angle
#   trans_rep = tracknet
#
# Corrected design:
#   1. RefineNet first outputs pose hypotheses.
#   2. register()/track_one() optionally calls depth_geometry_refine_poses().
#   3. For register(), original and geometry-corrected poses can be sent to
#      the FoundationPose scorer together, so bad geometry corrections are not
#      forced to replace the learned refiner output.
#
# Supported algorithms:
#   icp   : Iterative Closest Point, point-to-point nearest-neighbor update.
#   ndt   : Normal Distributions Transform update using target voxel
#           Gaussian statistics and Mahalanobis distance.
#   gicp  : Generalized ICP update using source/target local covariances
#           and Mahalanobis distance.
#   vgicp : Voxelized GICP update using target voxel covariance + source
#           local covariance Mahalanobis distance.
#
# depth_refine_apply:
#   trans   : update X,Y,Z translation only.
#   trans_z : update camera Z translation only. This is the safest option for
#             depth-assisted distance refinement.
#   se3     : update rotation + translation. Kept for ablation, but usually
#             riskier on occluded/noisy depth.
# ============================================================



def _depth_refine_project_root():
  # predict_pose_refine.py is expected at FoundationPose/learning/training/.
  # Therefore code_dir/../../ is the FoundationPose project root.
  return Path(code_dir).resolve().parents[1]


def _depth_refine_default_yml_path():
  return _depth_refine_project_root() / "config" / "depth_refine.yml"


def _depth_refine_normalize_value(v):
  if isinstance(v, str):
    s = v.strip()
    sl = s.lower()
    if sl in ["true", "yes", "on"]:
      return True
    if sl in ["false", "no", "off"]:
      return False
    if sl in ["auto", "adaptive", "best", "opt", "optimal"]:
      return "auto"
    if sl in ["none", "null"]:
      return "none"
    try:
      if any(c in s for c in [".", "e", "E"]):
        return float(s)
      return int(s)
    except Exception:
      return s
  return v


def _depth_refine_flatten_yml(raw):
  """
  Read a standalone YAML file and map it into PoseRefinePredictor cfg keys.

  Supported YAML format:
    depth_refine:
      mode: icp
      apply: trans_z
      accept_if_better: true
      auto_sweep: true
      log: true

  Optional numeric overrides are still supported, but the recommended default
  is to omit them and let the code choose adaptive per-pose values.
  """
  if raw is None:
    return {}
  if not isinstance(raw, dict):
    return {}

  node = raw.get("depth_refine", raw)
  if node is None or not isinstance(node, dict):
    return {}

  alias = {
    "mode": "depth_refine_mode",
    "apply": "depth_refine_apply",
    "accept_if_better": "depth_refine_accept_if_better",
    "score_with_original": "depth_refine_score_with_original",
    "in_predict": "depth_refine_in_predict",
    "max_points": "depth_refine_max_points",
    "min_points": "depth_refine_min_points",
    "max_corr_dist": "depth_refine_max_corr_dist",
    "icp_iter": "depth_refine_icp_iter",
    "depth_diff_thresh": "depth_refine_depth_diff_thresh",
    "trans_clamp": "depth_refine_trans_clamp",
    "rot_clamp_deg": "depth_refine_rot_clamp_deg",
    "voxel_size": "depth_refine_voxel_size",
    "voxel_min_points": "depth_refine_voxel_min_points",
    "knn": "depth_refine_knn",
    "auto_sweep": "depth_refine_auto_sweep",
    "log": "depth_refine_log",
  }

  out = {}

  for k, v in node.items():
    if k == "parameters" and isinstance(v, dict):
      for pk, pv in v.items():
        key = alias.get(pk, pk)
        if not str(key).startswith("depth_refine_"):
          key = "depth_refine_" + str(key)
        out[key] = _depth_refine_normalize_value(pv)
      continue

    key = alias.get(k, k)
    if not str(key).startswith("depth_refine_"):
      key = "depth_refine_" + str(key)
    out[key] = _depth_refine_normalize_value(v)

  return out


def _load_depth_refine_yml_to_cfg(cfg):
  """
  Load FoundationPose/config/depth_refine.yml.

  Priority:
    standalone depth_refine.yml > checkpoint config.yml defaults.

  This intentionally does NOT use environment variables.  The runner writes
  this small YAML file before launching each Python process.
  """
  yml_path = _depth_refine_default_yml_path()

  # Backward-compatible aliases if you manually place the file elsewhere
  # inside the project root.
  candidates = [
    yml_path,
    _depth_refine_project_root() / "depth_refine.yml",
    _depth_refine_project_root() / "config" / "depth_refine_config.yml",
  ]

  loaded_path = None
  loaded = {}
  for p in candidates:
    try:
      if p.is_file():
        with p.open("r") as f:
          raw = yaml.safe_load(f) or {}
        loaded = _depth_refine_flatten_yml(raw)
        loaded_path = p
        break
    except Exception as e:
      logging.warning(f"[DepthRefineYAML] Failed to read {p}: {e}")

  for k, v in loaded.items():
    cfg[k] = v

  if loaded_path is not None:
    logging.info(
      f"[DepthRefineYAML] loaded={loaded_path}, "
      f"mode={cfg.get('depth_refine_mode', 'none')}, "
      f"apply={cfg.get('depth_refine_apply', 'trans')}"
    )
  else:
    logging.info(f"[DepthRefineYAML] no standalone YAML found; default to mode=none if not set.")

  return cfg


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


def _clamp_se3(R, t, max_rot_deg=5.0, max_trans=0.03, apply='trans'):
  R = np.asarray(R, dtype=np.float64)
  t = np.asarray(t, dtype=np.float64).reshape(3)

  if apply == 'trans_z':
    t = np.array([0.0, 0.0, t[2]], dtype=np.float64)
  elif apply == 'trans':
    R = np.eye(3, dtype=np.float64)

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


def _weighted_kabsch(A, B, weights=None):
  """Solve R,t such that B ~= R @ A + t."""
  A = np.asarray(A, dtype=np.float64)
  B = np.asarray(B, dtype=np.float64)
  if len(A) < 3:
    return np.eye(3), np.zeros(3), False

  if weights is None:
    weights = np.ones(len(A), dtype=np.float64)
  weights = np.asarray(weights, dtype=np.float64).reshape(-1)
  weights = np.maximum(weights, 1e-12)
  weights = weights / np.sum(weights)

  ca = np.sum(A * weights[:,None], axis=0)
  cb = np.sum(B * weights[:,None], axis=0)
  AA = A - ca
  BB = B - cb
  H = (AA * weights[:,None]).T @ BB

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


def _weighted_translation(A, B, weights=None, robust='median'):
  residual = np.asarray(B, dtype=np.float64) - np.asarray(A, dtype=np.float64)
  if len(residual) == 0:
    return np.zeros(3, dtype=np.float64), False
  if robust == 'median' or weights is None:
    return np.median(residual, axis=0), True
  weights = np.asarray(weights, dtype=np.float64).reshape(-1)
  weights = np.maximum(weights, 1e-12)
  weights = weights / np.sum(weights)
  return np.sum(residual * weights[:,None], axis=0), True


def _sample_points_pair(A_pts, B_pts, max_points=2048):
  n = len(A_pts)
  if n <= max_points:
    return A_pts, B_pts
  ids = np.random.choice(n, size=max_points, replace=False)
  return A_pts[ids], B_pts[ids]


def _extract_crop_xyz_pairs(pose_data, idx, z_min=1e-6, z_max=np.inf, depth_diff_thresh=0.05, max_points=2048, roi_mask_crop=None):
  """
  Extract pixel-wise rendered/observed 3D pairs in camera metric coordinates.

  Important:
  - Use raw_xyz_mapAs/raw_xyz_mapBs if available. These are saved before
    dataset.transform_batch(), so they remain in camera meters.
  - Using pose_data.xyz_mapAs/Bs after transform_batch() can be normalized and
    must NOT be directly used to update pose translation.
  """
  xyzA_t = getattr(pose_data, 'raw_xyz_mapAs', pose_data.xyz_mapAs)
  xyzB_t = getattr(pose_data, 'raw_xyz_mapBs', pose_data.xyz_mapBs)
  xyzA = _as_numpy(xyzA_t[idx]).transpose(1, 2, 0)
  xyzB = _as_numpy(xyzB_t[idx]).transpose(1, 2, 0)

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


def _pose_error_mean(A, B, R=None, t=None, weights=None):
  if len(A) == 0:
    return np.inf
  if R is not None and t is not None:
    A2 = (R @ A.T).T + t.reshape(1, 3)
  else:
    A2 = A
  err = np.linalg.norm(A2 - B, axis=1)
  if weights is None:
    return float(np.mean(err))
  weights = np.asarray(weights, dtype=np.float64).reshape(-1)
  weights = np.maximum(weights, 1e-12)
  return float(np.sum(err * weights) / np.sum(weights))


def _apply_mode_to_corr(R, t, apply, max_rot_deg, max_trans):
  if apply == 'trans':
    R = np.eye(3, dtype=np.float64)
  elif apply == 'trans_z':
    R = np.eye(3, dtype=np.float64)
    t = np.array([0.0, 0.0, np.asarray(t).reshape(3)[2]], dtype=np.float64)
  elif apply == 'se3':
    pass
  else:
    R = np.eye(3, dtype=np.float64)
  return _clamp_se3(R, t, max_rot_deg=max_rot_deg, max_trans=max_trans, apply=apply)


def _icp_correction(A, B, apply='trans', max_corr_dist=0.03, max_iter=5, max_rot_deg=5.0, max_trans=0.03, min_points=30):
  """Iterative Closest Point (point-to-point)."""
  if len(A) < min_points or len(B) < min_points:
    return np.eye(3), np.zeros(3), False

  src0 = np.asarray(A, dtype=np.float64)
  tgt = np.asarray(B, dtype=np.float64)
  R_total = np.eye(3, dtype=np.float64)
  t_total = np.zeros(3, dtype=np.float64)
  tree = spatial.cKDTree(tgt)

  ok_any = False
  for _ in range(int(max_iter)):
    src = (R_total @ src0.T).T + t_total.reshape(1, 3)
    dists, ids = tree.query(src, k=1, workers=-1)
    valid = np.isfinite(dists)
    if max_corr_dist is not None and max_corr_dist > 0:
      valid &= dists < max_corr_dist
    if valid.sum() < min_points:
      break

    A_corr = src[valid]
    B_corr = tgt[ids[valid]]
    if apply in ['trans', 'trans_z']:
      step_t, step_ok = _weighted_translation(A_corr, B_corr, robust='median')
      step_R = np.eye(3, dtype=np.float64)
    else:
      step_R, step_t, step_ok = _weighted_kabsch(A_corr, B_corr)
    if not step_ok:
      break

    step_R, step_t = _apply_mode_to_corr(step_R, step_t, apply, max_rot_deg, max_trans)
    R_total = step_R @ R_total
    t_total = step_R @ t_total + step_t
    ok_any = True

    if np.linalg.norm(step_t) < 1e-5 and _rotation_angle_deg(step_R) < 0.05:
      break

  if not ok_any:
    return np.eye(3), np.zeros(3), False
  R_total, t_total = _apply_mode_to_corr(R_total, t_total, apply, max_rot_deg, max_trans)
  return R_total, t_total, True


def _voxel_stats(points, voxel_size=0.01, min_points=5):
  points = np.asarray(points, dtype=np.float64)
  if len(points) == 0:
    return np.empty((0,3)), np.empty((0,3,3)), np.empty((0,), dtype=np.int64)

  keys = np.floor(points / float(voxel_size)).astype(np.int64)
  buckets = {}
  for i, k in enumerate(map(tuple, keys)):
    buckets.setdefault(k, []).append(i)

  means, covs, counts = [], [], []
  for ids in buckets.values():
    if len(ids) < min_points:
      continue
    P = points[ids]
    mu = P.mean(axis=0)
    X = P - mu
    C = (X.T @ X) / max(len(P) - 1, 1)
    C = _regularize_cov(C)
    means.append(mu)
    covs.append(C)
    counts.append(len(P))

  if len(means) == 0:
    return np.empty((0,3)), np.empty((0,3,3)), np.empty((0,), dtype=np.int64)
  return np.asarray(means), np.asarray(covs), np.asarray(counts)


def _regularize_cov(C, eps=1e-6, min_eig=1e-6, max_cond=1e6):
  """
  Make a 3x3 covariance safely invertible.

  For NDT/GICP/VGICP we need covariance inverse matrices.  Local object
  depth crops can be nearly planar or have too few points, so raw covariance
  is often rank-deficient.  This function symmetrizes, eigen-clamps, and adds
  a small diagonal term.
  """
  C = np.asarray(C, dtype=np.float64).reshape(3, 3)
  C = 0.5 * (C + C.T)
  C = C + np.eye(3, dtype=np.float64) * float(eps)

  try:
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, float(min_eig))
    vmax = float(vals.max())
    if vmax > 0 and max_cond is not None and max_cond > 0:
      vals = np.maximum(vals, vmax / float(max_cond))
    C = (vecs * vals.reshape(1, 3)) @ vecs.T
    C = 0.5 * (C + C.T)
  except Exception:
    C = C + np.eye(3, dtype=np.float64) * float(min_eig)
  return C


def _invert_covariances(covs, eps=1e-6):
  covs = np.asarray(covs, dtype=np.float64)
  if len(covs) == 0:
    return np.empty((0, 3, 3), dtype=np.float64)
  invs = []
  for C in covs:
    C = _regularize_cov(C, eps=eps)
    try:
      invs.append(np.linalg.inv(C))
    except np.linalg.LinAlgError:
      invs.append(np.linalg.pinv(C))
  return np.asarray(invs)


def _skew(v):
  v = np.asarray(v, dtype=np.float64).reshape(3)
  return np.array([
    [0.0, -v[2], v[1]],
    [v[2], 0.0, -v[0]],
    [-v[1], v[0], 0.0],
  ], dtype=np.float64)


def _solve_linear_system(H, g, damping=1e-9):
  H = np.asarray(H, dtype=np.float64)
  g = np.asarray(g, dtype=np.float64)
  H = 0.5 * (H + H.T)
  H = H + np.eye(H.shape[0], dtype=np.float64) * float(damping)
  try:
    return np.linalg.solve(H, g)
  except np.linalg.LinAlgError:
    return np.linalg.lstsq(H, g, rcond=None)[0]


def _mahalanobis_objective(A, B, Omegas, R=None, t=None):
  if len(A) == 0:
    return np.inf
  A = np.asarray(A, dtype=np.float64)
  B = np.asarray(B, dtype=np.float64)
  Omegas = np.asarray(Omegas, dtype=np.float64)
  if R is not None and t is not None:
    A2 = (R @ A.T).T + np.asarray(t, dtype=np.float64).reshape(1, 3)
  else:
    A2 = A
  r = A2 - B
  val = 0.0
  for i in range(len(r)):
    val += float(r[i].T @ Omegas[i] @ r[i])
  return val / max(len(r), 1)


def _solve_mahalanobis_delta(A, B, Omegas, apply='trans', max_rot_deg=5.0, max_trans=0.03):
  """
  Solve a single Mahalanobis least-squares update.

  Residual:
    r_i = A_i - B_i

  Objective:
    sum_i r_i^T Omega_i r_i

  Modes:
    trans:
      minimize over Delta t in R^3.
    trans_z:
      minimize over Delta z only.
    se3:
      one Gauss-Newton step on left-multiplicative SE(3):
        x' = exp(xi^) x, J = [-[x]x, I].
  """
  A = np.asarray(A, dtype=np.float64)
  B = np.asarray(B, dtype=np.float64)
  Omegas = np.asarray(Omegas, dtype=np.float64)

  if len(A) < 3:
    return np.eye(3), np.zeros(3), False

  if apply in ['trans', 'trans_z']:
    # r = A + dt - B.  Solve H dt = -g.
    if apply == 'trans':
      H = np.zeros((3, 3), dtype=np.float64)
      g = np.zeros(3, dtype=np.float64)
      for a, b, Om in zip(A, B, Omegas):
        r = a - b
        H += Om
        g += Om @ r
      dt = -_solve_linear_system(H, g)
      R = np.eye(3, dtype=np.float64)
      R, dt = _apply_mode_to_corr(R, dt, apply, max_rot_deg, max_trans)
      return R, dt, True

    # Z-only update.  J = e_z.
    ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    Hzz = 0.0
    gz = 0.0
    for a, b, Om in zip(A, B, Omegas):
      r = a - b
      Hzz += float(ez.T @ Om @ ez)
      gz += float(ez.T @ Om @ r)
    if abs(Hzz) < 1e-12:
      return np.eye(3), np.zeros(3), False
    dz = -gz / Hzz
    R = np.eye(3, dtype=np.float64)
    dt = np.array([0.0, 0.0, dz], dtype=np.float64)
    R, dt = _apply_mode_to_corr(R, dt, apply, max_rot_deg, max_trans)
    return R, dt, True

  # SE(3) one-step Gauss-Newton.
  H = np.zeros((6, 6), dtype=np.float64)
  g = np.zeros(6, dtype=np.float64)
  for a, b, Om in zip(A, B, Omegas):
    r = a - b
    J = np.zeros((3, 6), dtype=np.float64)
    J[:, :3] = -_skew(a)
    J[:, 3:] = np.eye(3, dtype=np.float64)
    H += J.T @ Om @ J
    g += J.T @ Om @ r

  dxi = -_solve_linear_system(H, g)
  w = dxi[:3]
  dt = dxi[3:]
  R = _rotation_exp(w)
  R, dt = _apply_mode_to_corr(R, dt, apply, max_rot_deg, max_trans)
  return R, dt, True


def _local_covariances(points, k=10):
  points = np.asarray(points, dtype=np.float64)
  if len(points) == 0:
    return np.empty((0, 3, 3), dtype=np.float64)
  if len(points) < 3:
    return np.tile(np.eye(3, dtype=np.float64)[None] * 1e-6, (len(points), 1, 1))

  k = int(max(3, min(k, len(points))))
  tree = spatial.cKDTree(points)
  _, idxs = tree.query(points, k=k, workers=-1)

  covs = []
  for ids in idxs:
    P = points[np.asarray(ids).reshape(-1)]
    mu = P.mean(axis=0)
    X = P - mu
    C = (X.T @ X) / max(len(P) - 1, 1)
    covs.append(_regularize_cov(C))
  return np.asarray(covs)


def _ndt_correction(A, B, apply='trans', voxel_size=0.01, max_corr_dist=0.03, max_rot_deg=5.0, max_trans=0.03, min_points=30, voxel_min_points=5, max_iter=3):
  """
  NDT-like correction using the original NDT Mahalanobis objective.

  Target observed points are voxelized into Gaussian cells.  A transformed
  rendered/source point is associated with the nearest target voxel mean.
  The update minimizes:
      sum_i (x_i - mu_v)^T Sigma_v^{-1} (x_i - mu_v)

  This is still constrained by `apply`:
    trans   : optimize translation only.
    trans_z : optimize z translation only.
    se3     : one small SE(3) Gauss-Newton update per iteration.
  """
  if len(A) < min_points or len(B) < min_points:
    return np.eye(3), np.zeros(3), False

  src0 = np.asarray(A, dtype=np.float64)
  means, covs, counts = _voxel_stats(B, voxel_size=voxel_size, min_points=voxel_min_points)
  if len(means) < 3:
    return np.eye(3), np.zeros(3), False

  inv_covs = _invert_covariances(covs)
  tree = spatial.cKDTree(means)

  R_total = np.eye(3, dtype=np.float64)
  t_total = np.zeros(3, dtype=np.float64)
  ok_any = False

  for _ in range(int(max_iter)):
    src = (R_total @ src0.T).T + t_total.reshape(1, 3)
    dists, ids = tree.query(src, k=1, workers=-1)

    valid = np.isfinite(dists)
    if max_corr_dist is not None and max_corr_dist > 0:
      valid &= dists < max_corr_dist
    if valid.sum() < min_points:
      break

    A_corr = src[valid]
    B_corr = means[ids[valid]]
    Omegas = inv_covs[ids[valid]]

    before = _mahalanobis_objective(A_corr, B_corr, Omegas)
    step_R, step_t, step_ok = _solve_mahalanobis_delta(
      A_corr, B_corr, Omegas,
      apply=apply,
      max_rot_deg=max_rot_deg,
      max_trans=max_trans
    )
    if not step_ok:
      break
    after = _mahalanobis_objective(A_corr, B_corr, Omegas, R=step_R, t=step_t)
    if not np.isfinite(after) or after > before:
      break

    R_total = step_R @ R_total
    t_total = step_R @ t_total + step_t
    ok_any = True

    if np.linalg.norm(step_t) < 1e-5 and _rotation_angle_deg(step_R) < 0.05:
      break

  if not ok_any:
    return np.eye(3), np.zeros(3), False
  R_total, t_total = _apply_mode_to_corr(R_total, t_total, apply, max_rot_deg, max_trans)
  return R_total, t_total, True


def _gicp_correction(A, B, apply='trans', max_corr_dist=0.03, max_iter=3, max_rot_deg=5.0, max_trans=0.03, min_points=30, knn=10):
  """
  Generalized ICP correction with covariance Mahalanobis objective.

  Correspondence:
    transformed source point -> nearest target point.

  Objective:
    sum_i r_i^T (C_Bi + R C_Ai R^T)^-1 r_i
  where:
    r_i = x_i - q_i

  Unlike the previous trace-weight approximation, this uses full 3x3
  information matrices and source + target local covariances.
  """
  if len(A) < min_points or len(B) < min_points:
    return np.eye(3), np.zeros(3), False

  src0 = np.asarray(A, dtype=np.float64)
  tgt = np.asarray(B, dtype=np.float64)

  cov_src0 = _local_covariances(src0, k=knn)
  cov_tgt = _local_covariances(tgt, k=knn)

  tree = spatial.cKDTree(tgt)
  R_total = np.eye(3, dtype=np.float64)
  t_total = np.zeros(3, dtype=np.float64)
  ok_any = False

  for _ in range(int(max_iter)):
    src = (R_total @ src0.T).T + t_total.reshape(1, 3)
    dists, ids = tree.query(src, k=1, workers=-1)

    valid = np.isfinite(dists)
    if max_corr_dist is not None and max_corr_dist > 0:
      valid &= dists < max_corr_dist
    if valid.sum() < min_points:
      break

    src_ids = np.where(valid)[0]
    tgt_ids = ids[valid]

    A_corr = src[src_ids]
    B_corr = tgt[tgt_ids]

    # Full GICP information:
    #   Omega_i = (C_B + R C_A R^T)^-1
    cov_src_cur = np.asarray([R_total @ cov_src0[j] @ R_total.T for j in src_ids])
    cov_pair = cov_tgt[tgt_ids] + cov_src_cur
    Omegas = _invert_covariances(cov_pair)

    before = _mahalanobis_objective(A_corr, B_corr, Omegas)
    step_R, step_t, step_ok = _solve_mahalanobis_delta(
      A_corr, B_corr, Omegas,
      apply=apply,
      max_rot_deg=max_rot_deg,
      max_trans=max_trans
    )
    if not step_ok:
      break
    after = _mahalanobis_objective(A_corr, B_corr, Omegas, R=step_R, t=step_t)
    if not np.isfinite(after) or after > before:
      break

    R_total = step_R @ R_total
    t_total = step_R @ t_total + step_t
    ok_any = True

    if np.linalg.norm(step_t) < 1e-5 and _rotation_angle_deg(step_R) < 0.05:
      break

  if not ok_any:
    return np.eye(3), np.zeros(3), False
  R_total, t_total = _apply_mode_to_corr(R_total, t_total, apply, max_rot_deg, max_trans)
  return R_total, t_total, True


def _vgicp_correction(A, B, apply='trans', voxel_size=0.01, max_corr_dist=0.03, max_iter=3, max_rot_deg=5.0, max_trans=0.03, min_points=30, voxel_min_points=5, knn=10):
  """
  Voxelized Generalized ICP correction with voxel covariance Mahalanobis cost.

  Target points are summarized by voxel means/covariances. Source points keep
  local source covariances. A source point is associated with the nearest target
  voxel mean.

  Objective:
    sum_i r_i^T (Sigma_v + R C_Ai R^T)^-1 r_i
  where:
    r_i = x_i - mu_v
  """
  if len(A) < min_points or len(B) < min_points:
    return np.eye(3), np.zeros(3), False

  src0 = np.asarray(A, dtype=np.float64)
  means, covs, counts = _voxel_stats(B, voxel_size=voxel_size, min_points=voxel_min_points)
  if len(means) < 3:
    return np.eye(3), np.zeros(3), False

  cov_src0 = _local_covariances(src0, k=knn)
  tree = spatial.cKDTree(means)

  R_total = np.eye(3, dtype=np.float64)
  t_total = np.zeros(3, dtype=np.float64)
  ok_any = False

  for _ in range(int(max_iter)):
    src = (R_total @ src0.T).T + t_total.reshape(1, 3)
    dists, ids = tree.query(src, k=1, workers=-1)

    valid = np.isfinite(dists)
    if max_corr_dist is not None and max_corr_dist > 0:
      valid &= dists < max_corr_dist
    if valid.sum() < min_points:
      break

    src_ids = np.where(valid)[0]
    voxel_ids = ids[valid]

    A_corr = src[src_ids]
    B_corr = means[voxel_ids]

    cov_src_cur = np.asarray([R_total @ cov_src0[j] @ R_total.T for j in src_ids])
    cov_pair = covs[voxel_ids] + cov_src_cur
    Omegas = _invert_covariances(cov_pair)

    before = _mahalanobis_objective(A_corr, B_corr, Omegas)
    step_R, step_t, step_ok = _solve_mahalanobis_delta(
      A_corr, B_corr, Omegas,
      apply=apply,
      max_rot_deg=max_rot_deg,
      max_trans=max_trans
    )
    if not step_ok:
      break
    after = _mahalanobis_objective(A_corr, B_corr, Omegas, R=step_R, t=step_t)
    if not np.isfinite(after) or after > before:
      break

    R_total = step_R @ R_total
    t_total = step_R @ t_total + step_t
    ok_any = True

    if np.linalg.norm(step_t) < 1e-5 and _rotation_angle_deg(step_R) < 0.05:
      break

  if not ok_any:
    return np.eye(3), np.zeros(3), False
  R_total, t_total = _apply_mode_to_corr(R_total, t_total, apply, max_rot_deg, max_trans)
  return R_total, t_total, True


def _apply_corr_to_pose(T, R_corr, t_corr, apply='trans'):
  T = np.asarray(T, dtype=np.float64).copy()
  if apply in ['trans', 'trans_z']:
    T[:3, 3] = T[:3, 3] + np.asarray(t_corr, dtype=np.float64).reshape(3)
    return T

  Tc = np.eye(4, dtype=np.float64)
  Tc[:3, :3] = R_corr
  Tc[:3, 3] = t_corr
  return Tc @ T


# ============================================================
# Adaptive parameter selection for post-score depth refinement
# ------------------------------------------------------------
# In full BOP / real robot runs there is no ground truth available at test
# time, so there is no single globally optimal fixed parameter set.  The
# following helpers estimate per-pose parameters from the rendered/observed
# depth residual statistics, then optionally run a tiny local sweep and keep
# the correction that minimizes the internal depth alignment error.
# ============================================================

def _is_auto_value(v):
  if v is None:
    return True
  if isinstance(v, str):
    return v.strip().lower() in ['auto', 'adaptive', 'best', 'opt', 'optimal', 'none', '']
  return False


def _cfg_raw(cfg, key, default='auto'):
  try:
    if key in cfg:
      return cfg[key]
  except Exception:
    pass
  return default


def _to_bool(v, default=True):
  if isinstance(v, str):
    s = v.strip().lower()
    if s in ['1', 'true', 'yes', 'y', 'on']:
      return True
    if s in ['0', 'false', 'no', 'n', 'off']:
      return False
    if s in ['auto', 'adaptive']:
      return default
  if v is None:
    return default
  return bool(v)


def _to_float_or_auto(v):
  if _is_auto_value(v):
    return 'auto'
  try:
    return float(v)
  except Exception:
    return 'auto'


def _to_int_or_auto(v):
  if _is_auto_value(v):
    return 'auto'
  try:
    return int(float(v))
  except Exception:
    return 'auto'


def _robust_sigma(x):
  x = np.asarray(x, dtype=np.float64).reshape(-1)
  if len(x) == 0:
    return 0.0
  med = np.median(x)
  mad = np.median(np.abs(x - med))
  return float(1.4826 * mad + 1e-12)


def _safe_quantile(x, q, default=0.0):
  x = np.asarray(x, dtype=np.float64).reshape(-1)
  x = x[np.isfinite(x)]
  if len(x) == 0:
    return float(default)
  return float(np.quantile(x, q))


def _subsample_pair(A, B, max_points):
  max_points = int(max_points)
  if max_points <= 0 or len(A) <= max_points:
    return A, B
  # Deterministic subsampling to keep experiments repeatable.
  ids = np.linspace(0, len(A) - 1, max_points).round().astype(np.int64)
  return A[ids], B[ids]


def _filter_pair_by_adaptive_depth(A, B, min_points=30, user_thresh='auto'):
  """
  Keep reliable rendered/observed depth pairs using robust per-pose statistics.
  This replaces a fixed depth_diff_thresh when cfg value is 'auto'.
  """
  if len(A) == 0:
    return A, B, 0.0

  dz = np.abs(A[:, 2] - B[:, 2])
  if not _is_auto_value(user_thresh):
    th = float(user_thresh)
    if th > 0:
      m = dz < th
      if m.sum() >= min_points:
        return A[m], B[m], th
    return A, B, th

  med = float(np.median(dz))
  sig = _robust_sigma(dz)
  q80 = _safe_quantile(dz, 0.80, med)
  q90 = _safe_quantile(dz, 0.90, q80)

  # Robust but not too restrictive.  Bounds are in meters.
  th = max(q80, med + 3.0 * sig)
  th = float(np.clip(th, 0.008, 0.080))

  m = dz < th
  if m.sum() < min_points:
    th = float(np.clip(q90, th, 0.100))
    m = dz < th
  if m.sum() < min_points:
    return A, B, th
  return A[m], B[m], th


def _adaptive_depth_refine_params(A, B, mode, apply, mesh_diameter=None, cfg=None, user_values=None):
  """
  Estimate per-pose numeric parameters from point statistics.

  Returns a dict containing:
    max_points, min_points, max_corr_dist, icp_iter, depth_diff_thresh,
    max_trans, max_rot_deg, voxel_size, voxel_min_points, knn.
  """
  A = np.asarray(A, dtype=np.float64)
  B = np.asarray(B, dtype=np.float64)
  n = int(min(len(A), len(B)))
  diam = float(mesh_diameter) if mesh_diameter is not None and np.isfinite(mesh_diameter) and mesh_diameter > 0 else 0.10

  if n > 0:
    residual = B[:n] - A[:n]
    dist = np.linalg.norm(residual, axis=1)
    dz = np.abs(residual[:, 2])
    p50 = _safe_quantile(dist, 0.50, 0.01)
    p70 = _safe_quantile(dist, 0.70, p50)
    p85 = _safe_quantile(dist, 0.85, p70)
    p90 = _safe_quantile(dist, 0.90, p85)
    dz85 = _safe_quantile(dz, 0.85, p70)
    bbox_diag = float(np.linalg.norm(np.percentile(B[:n], 95, axis=0) - np.percentile(B[:n], 5, axis=0))) if n >= 10 else diam
  else:
    p50 = p70 = p85 = p90 = dz85 = 0.01
    bbox_diag = diam

  user_values = user_values or {}

  def choose_float(key, auto_value):
    v = user_values.get(key, 'auto')
    if _is_auto_value(v):
      return float(auto_value)
    return float(v)

  def choose_int(key, auto_value):
    v = user_values.get(key, 'auto')
    if _is_auto_value(v):
      return int(auto_value)
    return int(float(v))

  max_points_auto = 8192 if n >= 4096 else max(2048, n)
  min_points_auto = int(np.clip(max(20, 0.01 * max(n, 1)), 20, 80))

  # Correspondence radius should cover most inlier residuals, but avoid
  # admitting too many background/occluder points.  Use residual quantile and
  # object scale jointly.
  max_corr_auto = max(0.012, 1.5 * p85, 0.04 * diam)
  max_corr_auto = float(np.clip(max_corr_auto, 0.012, min(0.100, max(0.030, 0.35 * diam))))

  # Limit the final physical correction.  The clamp is intentionally smaller
  # than max_corr_dist so the local optimizer cannot jump to a wrong surface.
  if apply == 'trans_z':
    trans_auto = max(0.006, 1.5 * dz85, 0.030 * diam)
  else:
    trans_auto = max(0.006, 1.25 * p70, 0.030 * diam)
  trans_auto = float(np.clip(trans_auto, 0.006, min(0.060, max(0.015, 0.20 * diam))))

  # Voxel size uses object/crop scale plus point count.  For small object crops,
  # too small voxels make covariance rank-deficient; too large voxels erase shape.
  voxel_auto = max(0.004, bbox_diag / 28.0, diam / 45.0)
  voxel_auto = float(np.clip(voxel_auto, 0.004, min(0.035, max(0.012, diam / 6.0))))

  # More points -> allow larger KNN.  Keep KNN bounded for speed and locality.
  knn_auto = int(np.clip(round(np.sqrt(max(n, 1)) / 2.0), 10, 40))
  if mode in ['gicp', 'vgicp']:
    knn_auto = int(np.clip(knn_auto, 15, 45))

  # voxel_min_points should be low for object crops to avoid empty voxel maps.
  voxel_min_auto = 3 if n < 4096 else 4

  # Iterations: post-score best-pose correction only; use moderate local GN/ICP.
  if mode == 'icp':
    iter_auto = 8
  elif mode in ['gicp', 'vgicp']:
    iter_auto = 5
  else:
    iter_auto = 4

  rot_auto = 3.0 if apply == 'se3' else 0.0

  return {
    'max_points': choose_int('max_points', max_points_auto),
    'min_points': choose_int('min_points', min_points_auto),
    'max_corr_dist': choose_float('max_corr_dist', max_corr_auto),
    'icp_iter': choose_int('icp_iter', iter_auto),
    'max_trans': choose_float('max_trans', trans_auto),
    'max_rot_deg': choose_float('max_rot_deg', rot_auto if rot_auto > 0 else 3.0),
    'voxel_size': choose_float('voxel_size', voxel_auto),
    'voxel_min_points': choose_int('voxel_min_points', voxel_min_auto),
    'knn': choose_int('knn', knn_auto),
  }


def _depth_refine_candidates_from_params(params, mode, apply, auto_sweep=True):
  """
  Build a small local parameter sweep around the adaptive estimate and let the
  internal residual choose the best correction.  This is a practical no-GT
  approximation to "best" parameters at test time.
  """
  base = dict(params)
  if not auto_sweep:
    return [base]

  # Keep sweep tiny because register() now corrects only the scorer-best pose.
  corr_scales = [0.75, 1.0, 1.35]
  clamp_scales = [0.75, 1.0, 1.35]

  if mode in ['ndt', 'vgicp']:
    voxel_scales = [0.75, 1.0, 1.35]
  else:
    voxel_scales = [1.0]

  out = []
  seen = set()
  for cs in corr_scales:
    for ts in clamp_scales:
      for vs in voxel_scales:
        p = dict(base)
        p['max_corr_dist'] = float(np.clip(base['max_corr_dist'] * cs, 0.006, 0.120))
        p['max_trans'] = float(np.clip(base['max_trans'] * ts, 0.003, 0.080))
        p['voxel_size'] = float(np.clip(base['voxel_size'] * vs, 0.003, 0.050))
        key = (
          round(p['max_corr_dist'], 5),
          round(p['max_trans'], 5),
          round(p['voxel_size'], 5),
          p['min_points'],
          p['voxel_min_points'],
          p['knn'],
          p['icp_iter'],
        )
        if key not in seen:
          seen.add(key)
          out.append(p)
  return out


def _run_depth_correction_with_params(mode, A, B, apply, params):
  if mode == 'icp':
    return _icp_correction(
      A, B,
      apply=apply,
      max_corr_dist=params['max_corr_dist'],
      max_iter=params['icp_iter'],
      max_rot_deg=params['max_rot_deg'],
      max_trans=params['max_trans'],
      min_points=params['min_points'],
    )
  if mode == 'ndt':
    return _ndt_correction(
      A, B,
      apply=apply,
      voxel_size=params['voxel_size'],
      max_corr_dist=params['max_corr_dist'],
      max_iter=params['icp_iter'],
      max_rot_deg=params['max_rot_deg'],
      max_trans=params['max_trans'],
      min_points=params['min_points'],
      voxel_min_points=params['voxel_min_points'],
    )
  if mode == 'gicp':
    return _gicp_correction(
      A, B,
      apply=apply,
      max_corr_dist=params['max_corr_dist'],
      max_iter=params['icp_iter'],
      max_rot_deg=params['max_rot_deg'],
      max_trans=params['max_trans'],
      min_points=params['min_points'],
      knn=params['knn'],
    )
  if mode == 'vgicp':
    return _vgicp_correction(
      A, B,
      apply=apply,
      voxel_size=params['voxel_size'],
      max_corr_dist=params['max_corr_dist'],
      max_iter=params['icp_iter'],
      max_rot_deg=params['max_rot_deg'],
      max_trans=params['max_trans'],
      min_points=params['min_points'],
      voxel_min_points=params['voxel_min_points'],
      knn=params['knn'],
    )
  return np.eye(3), np.zeros(3), False


def _select_best_depth_correction(mode, A, B, apply, params, auto_sweep=True):
  before_err = _pose_error_mean(A, B)
  best = {
    'ok': False,
    'R': np.eye(3, dtype=np.float64),
    't': np.zeros(3, dtype=np.float64),
    'score': before_err,
    'params': dict(params),
  }

  for p in _depth_refine_candidates_from_params(params, mode, apply, auto_sweep=auto_sweep):
    R_corr, t_corr, ok = _run_depth_correction_with_params(mode, A, B, apply, p)
    if not ok:
      continue
    score = _pose_error_mean(A, B, R=R_corr, t=t_corr)
    if np.isfinite(score) and score < best['score']:
      best.update({'ok': True, 'R': R_corr, 't': t_corr, 'score': score, 'params': dict(p)})

  return best, before_err

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
    # Load standalone depth refinement YAML.
    # Do not use environment variables for mode/apply.  The runner writes:
    #   FoundationPose/config/depth_refine.yml
    # before each experiment.
    self.cfg = _load_depth_refine_yml_to_cfg(self.cfg)

    if 'depth_refine_mode' not in self.cfg:
      self.cfg['depth_refine_mode'] = 'none'  # none / icp / ndt / gicp / vgicp
    if 'depth_refine_apply' not in self.cfg:
      self.cfg['depth_refine_apply'] = 'trans'  # trans / trans_z / se3
    if 'depth_refine_accept_if_better' not in self.cfg:
      self.cfg['depth_refine_accept_if_better'] = True
    if 'depth_refine_score_with_original' not in self.cfg:
      self.cfg['depth_refine_score_with_original'] = False
    if 'depth_refine_in_predict' not in self.cfg:
      self.cfg['depth_refine_in_predict'] = False
    # 'auto' enables per-pose adaptive values based on depth residual statistics.
    if 'depth_refine_max_points' not in self.cfg:
      self.cfg['depth_refine_max_points'] = 'auto'
    if 'depth_refine_min_points' not in self.cfg:
      self.cfg['depth_refine_min_points'] = 'auto'
    if 'depth_refine_max_corr_dist' not in self.cfg:
      self.cfg['depth_refine_max_corr_dist'] = 'auto'
    if 'depth_refine_icp_iter' not in self.cfg:
      self.cfg['depth_refine_icp_iter'] = 'auto'
    if 'depth_refine_depth_diff_thresh' not in self.cfg:
      self.cfg['depth_refine_depth_diff_thresh'] = 'auto'
    if 'depth_refine_trans_clamp' not in self.cfg:
      self.cfg['depth_refine_trans_clamp'] = 'auto'
    if 'depth_refine_rot_clamp_deg' not in self.cfg:
      self.cfg['depth_refine_rot_clamp_deg'] = 'auto'
    if 'depth_refine_voxel_size' not in self.cfg:
      self.cfg['depth_refine_voxel_size'] = 'auto'
    if 'depth_refine_voxel_min_points' not in self.cfg:
      self.cfg['depth_refine_voxel_min_points'] = 'auto'
    if 'depth_refine_knn' not in self.cfg:
      self.cfg['depth_refine_knn'] = 'auto'
    if 'depth_refine_auto_sweep' not in self.cfg:
      self.cfg['depth_refine_auto_sweep'] = True
    if 'depth_refine_log' not in self.cfg:
      self.cfg['depth_refine_log'] = True

    self.cfg['depth_refine_mode'] = str(self.cfg['depth_refine_mode']).lower()
    self.cfg['depth_refine_apply'] = str(self.cfg['depth_refine_apply']).lower()
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

    mode:
      none / icp / ndt / gicp / vgicp
    apply:
      trans / trans_z / se3
    """
    if mode is not None:
      self.cfg['depth_refine_mode'] = str(mode).lower()
    if apply is not None:
      self.cfg['depth_refine_apply'] = str(apply).lower()
    for k, v in kwargs.items():
      self.cfg[k] = v

  def _build_depth_refine_crop_data(self, B_in_cams, rgb_tensor, depth_tensor, K, xyz_map_tensor, normal_map, mesh_centered, glctx, mesh_tensors, mesh_diameter):
    return make_crop_data_batch(
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

  def depth_geometry_refine_poses(self, rgb, depth, K, ob_in_cams, xyz_map, normal_map=None, mesh=None, mesh_tensors=None, glctx=None, mesh_diameter=None, depth_roi_mask=None):
    """
    Apply optional test-time depth geometry correction to already refined poses.

    This function is intended to be called from FoundationPose.register()
    after RefineNet.predict(). It returns corrected poses only; register() can
    concatenate original + corrected poses and let the learned scorer select.
    """
    mode = str(_cfg_get(self.cfg, 'depth_refine_mode', 'none')).lower()
    apply = str(_cfg_get(self.cfg, 'depth_refine_apply', 'trans')).lower()

    if mode in ['none', 'off', 'false', '0']:
      return torch.as_tensor(ob_in_cams, device='cuda', dtype=torch.float)

    if mode not in ['icp', 'ndt', 'gicp', 'vgicp']:
      logging.warning(f"[DepthRefine] Unknown mode={mode}, skip.")
      return torch.as_tensor(ob_in_cams, device='cuda', dtype=torch.float)

    if apply not in ['trans', 'trans_z', 'se3']:
      logging.warning(f"[DepthRefine] Unknown apply={apply}, use trans.")
      apply = 'trans'

    if mesh_tensors is None:
      mesh_tensors = make_mesh_tensors(mesh)

    B_in_cams = torch.as_tensor(ob_in_cams, device='cuda', dtype=torch.float)
    rgb_tensor = torch.as_tensor(rgb, device='cuda', dtype=torch.float)
    depth_tensor = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    xyz_map_tensor = torch.as_tensor(xyz_map, device='cuda', dtype=torch.float)

    if not self.cfg.use_normal:
      normal_map = None

    pose_data = self._build_depth_refine_crop_data(
      B_in_cams=B_in_cams,
      rgb_tensor=rgb_tensor,
      depth_tensor=depth_tensor,
      K=K,
      xyz_map_tensor=xyz_map_tensor,
      normal_map=normal_map,
      mesh_centered=mesh,
      glctx=glctx,
      mesh_tensors=mesh_tensors,
      mesh_diameter=mesh_diameter
    )

    roi_mask_crops = None
    if depth_roi_mask is not None:
      mask_tensor = torch.as_tensor(depth_roi_mask, device='cuda', dtype=torch.float)
      if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor[None, None]
      elif mask_tensor.ndim == 3:
        mask_tensor = mask_tensor[:, None]
      mask_tensor = mask_tensor.expand(len(B_in_cams), -1, -1, -1)
      tf_to_crops = getattr(pose_data, 'raw_tf_to_crops', pose_data.tf_to_crops)
      roi_mask_crops = kornia.geometry.transform.warp_perspective(
        mask_tensor,
        tf_to_crops,
        dsize=self.cfg.input_resize,
        mode='nearest',
        align_corners=False
      )

    poses_np = _as_numpy(B_in_cams).copy()

    raw_user = {
      'max_points': _cfg_raw(self.cfg, 'depth_refine_max_points', 'auto'),
      'min_points': _cfg_raw(self.cfg, 'depth_refine_min_points', 'auto'),
      'max_corr_dist': _cfg_raw(self.cfg, 'depth_refine_max_corr_dist', 'auto'),
      'icp_iter': _cfg_raw(self.cfg, 'depth_refine_icp_iter', 'auto'),
      'depth_diff_thresh': _cfg_raw(self.cfg, 'depth_refine_depth_diff_thresh', 'auto'),
      'max_trans': _cfg_raw(self.cfg, 'depth_refine_trans_clamp', 'auto'),
      'max_rot_deg': _cfg_raw(self.cfg, 'depth_refine_rot_clamp_deg', 'auto'),
      'voxel_size': _cfg_raw(self.cfg, 'depth_refine_voxel_size', 'auto'),
      'voxel_min_points': _cfg_raw(self.cfg, 'depth_refine_voxel_min_points', 'auto'),
      'knn': _cfg_raw(self.cfg, 'depth_refine_knn', 'auto'),
    }

    # Extraction needs a numeric max_points.  If auto, keep enough points for
    # robust statistics, then sub-sample after adaptive filtering.
    max_points_raw = _to_int_or_auto(raw_user['max_points'])
    extract_max_points = 12000 if max_points_raw == 'auto' else int(max_points_raw)

    min_points_raw = _to_int_or_auto(raw_user['min_points'])
    pre_min_points = 20 if min_points_raw == 'auto' else int(min_points_raw)

    depth_diff_raw = _to_float_or_auto(raw_user['depth_diff_thresh'])
    extract_depth_thresh = None if depth_diff_raw == 'auto' else float(depth_diff_raw)

    accept_if_better = _to_bool(_cfg_get(self.cfg, 'depth_refine_accept_if_better', True), default=True)
    do_log = _to_bool(_cfg_get(self.cfg, 'depth_refine_log', True), default=True)
    auto_sweep = _to_bool(_cfg_get(self.cfg, 'depth_refine_auto_sweep', True), default=True)

    n_ok = 0
    n_try = 0
    last_params = None
    used_depth_thresh = None

    for i in range(len(poses_np)):
      roi_mask_crop = roi_mask_crops[i] if roi_mask_crops is not None else None

      A, B = _extract_crop_xyz_pairs(
        pose_data,
        i,
        z_min=1e-6,
        z_max=_cfg_get(self.cfg, 'zfar', np.inf),
        depth_diff_thresh=extract_depth_thresh,
        max_points=extract_max_points,
        roi_mask_crop=roi_mask_crop
      )

      A, B, used_depth_thresh = _filter_pair_by_adaptive_depth(
        A, B,
        min_points=pre_min_points,
        user_thresh=raw_user['depth_diff_thresh']
      )

      if len(A) < pre_min_points or len(B) < pre_min_points:
        continue

      params = _adaptive_depth_refine_params(
        A, B,
        mode=mode,
        apply=apply,
        mesh_diameter=mesh_diameter,
        cfg=self.cfg,
        user_values=raw_user
      )

      A, B = _subsample_pair(A, B, params['max_points'])

      if len(A) < params['min_points'] or len(B) < params['min_points']:
        continue

      n_try += 1
      best, before_err = _select_best_depth_correction(
        mode, A, B, apply, params, auto_sweep=auto_sweep
      )

      if not best['ok']:
        continue

      if accept_if_better and not (best['score'] < before_err):
        continue

      poses_np[i] = _apply_corr_to_pose(poses_np[i], best['R'], best['t'], apply=apply)
      last_params = best['params']
      n_ok += 1

    if do_log:
      if last_params is None:
        logging.info(
          f"[DepthRefine] mode={mode}, apply={apply}, accepted={n_ok}/{n_try}, "
          f"candidates={len(poses_np)}, adaptive=True, no accepted correction"
        )
      else:
        logging.info(
          f"[DepthRefine] mode={mode}, apply={apply}, accepted={n_ok}/{n_try}, "
          f"candidates={len(poses_np)}, adaptive=True, auto_sweep={auto_sweep}, "
          f"depth_th={used_depth_thresh:.4f}, corr={last_params['max_corr_dist']:.4f}m, "
          f"clamp={last_params['max_trans']:.4f}m, voxel={last_params['voxel_size']:.4f}m, "
          f"knn={last_params['knn']}, iter={last_params['icp_iter']}"
        )

    return torch.as_tensor(poses_np, device='cuda', dtype=torch.float)

  # Backward-compatible wrapper name.
  def _depth_geometry_refine_batch(self, B_in_cams, rgb_tensor, depth_tensor, K, xyz_map_tensor, normal_map, mesh_centered, glctx, mesh_tensors, mesh_diameter, depth_roi_mask=None):
    return self.depth_geometry_refine_poses(
      rgb=_as_numpy(rgb_tensor),
      depth=_as_numpy(depth_tensor),
      K=K,
      ob_in_cams=B_in_cams,
      xyz_map=_as_numpy(xyz_map_tensor),
      normal_map=normal_map,
      mesh=mesh_centered,
      mesh_tensors=mesh_tensors,
      glctx=glctx,
      mesh_diameter=mesh_diameter,
      depth_roi_mask=depth_roi_mask
    )

  @torch.inference_mode()
  def predict(self, rgb, depth, K, ob_in_cams, xyz_map, normal_map=None, get_vis=False, mesh=None, mesh_tensors=None, glctx=None, mesh_diameter=None, iteration=5):
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
