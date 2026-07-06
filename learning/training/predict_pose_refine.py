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


@torch.inference_mode()
def make_crop_data_batch(render_size, ob_in_cams, mesh, rgb, depth, K, crop_ratio, xyz_map, normal_map=None, mesh_diameter=None, cfg=None, glctx=None, mesh_tensors=None, dataset:PoseRefinePairH5Dataset=None, keep_raw_geometry=False):
  # logging.info("Welcome make_crop_data_batch")
  raw_xyz_mapAs = None
  raw_xyz_mapBs = None
  raw_tf_to_crops = None

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

  if keep_raw_geometry:
    # Keep raw camera-frame xyz maps before dataset.transform_batch().
    # The network needs the transformed tensors, but geometry refinement must
    # estimate metric corrections in the camera coordinate system.
    raw_xyz_mapAs = xyz_mapAs.clone()
    raw_xyz_mapBs = xyz_mapBs.clone()
    raw_tf_to_crops = tf_to_crops.clone()

  pose_data = dataset.transform_batch(batch=pose_data, H_ori=H, W_ori=W, bound=1)

  if keep_raw_geometry:
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
    "nn_chunk": "depth_refine_nn_chunk",
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


def _as_cuda_float_tensor(x, device=None):
  """Return a contiguous float32 tensor without moving CUDA tensors to CPU."""
  if torch.is_tensor(x):
    if device is None:
      device = x.device
    return x.to(device=device, dtype=torch.float32).contiguous()
  if device is None:
    device = torch.device('cuda', torch.cuda.current_device())
  return torch.as_tensor(x, device=device, dtype=torch.float32).contiguous()


def _rotation_angle_deg_torch(R):
  c = (torch.trace(R) - 1.0) * 0.5
  c = torch.clamp(c, -1.0, 1.0)
  return torch.rad2deg(torch.acos(c))


def _rotation_log_torch(R):
  c = torch.clamp((torch.trace(R) - 1.0) * 0.5, -1.0, 1.0)
  theta = torch.acos(c)
  vee = torch.stack([
    R[2, 1] - R[1, 2],
    R[0, 2] - R[2, 0],
    R[1, 0] - R[0, 1],
  ])
  denom = torch.clamp(2.0 * torch.sin(theta), min=1e-8)
  w = vee / denom * theta
  return torch.where(theta < 1e-7, torch.zeros_like(w), w)


def _rotation_exp_torch(w):
  w = w.reshape(3)
  theta = torch.linalg.vector_norm(w)
  safe_theta = torch.clamp(theta, min=1e-12)
  k = w / safe_theta
  K = torch.zeros((3, 3), device=w.device, dtype=w.dtype)
  K[0, 1] = -k[2]
  K[0, 2] = k[1]
  K[1, 0] = k[2]
  K[1, 2] = -k[0]
  K[2, 0] = -k[1]
  K[2, 1] = k[0]
  I = torch.eye(3, device=w.device, dtype=w.dtype)
  R = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)
  return torch.where((theta < 1e-7).reshape(1, 1), I, R)


def _clamp_se3_torch(R, t, max_rot_deg=5.0, max_trans=0.03, apply='trans'):
  R = R.reshape(3, 3)
  t = t.reshape(3)
  I = torch.eye(3, device=R.device, dtype=R.dtype)

  if apply == 'trans_z':
    t = torch.stack([t.new_zeros(()), t.new_zeros(()), t[2]])
    R = I
  elif apply == 'trans':
    R = I

  if max_rot_deg is not None and float(max_rot_deg) > 0 and apply == 'se3':
    angle = _rotation_angle_deg_torch(R)
    scale = torch.clamp(
      R.new_tensor(float(max_rot_deg)) / torch.clamp(angle, min=1e-9),
      max=1.0,
    )
    R = _rotation_exp_torch(_rotation_log_torch(R) * scale)

  if max_trans is not None and float(max_trans) > 0:
    norm_t = torch.linalg.vector_norm(t)
    scale = torch.clamp(
      t.new_tensor(float(max_trans)) / torch.clamp(norm_t, min=1e-9),
      max=1.0,
    )
    t = t * scale

  return R, t


def _weighted_kabsch_torch(A, B, weights=None):
  """Solve R,t on CUDA such that B ~= A @ R.T + t."""
  if A.shape[0] < 3:
    return torch.eye(3, device=A.device, dtype=A.dtype), A.new_zeros(3), False

  if weights is None:
    weights = A.new_ones(A.shape[0])
  weights = torch.clamp(weights.reshape(-1), min=1e-12)
  weights = weights / torch.clamp(weights.sum(), min=1e-12)

  ca = torch.sum(A * weights[:, None], dim=0)
  cb = torch.sum(B * weights[:, None], dim=0)
  AA = A - ca
  BB = B - cb
  H = (AA * weights[:, None]).transpose(0, 1) @ BB

  U, _, Vh = torch.linalg.svd(H, full_matrices=False)
  det = torch.det(Vh.transpose(0, 1) @ U.transpose(0, 1))
  D = torch.eye(3, device=A.device, dtype=A.dtype)
  D[2, 2] = torch.where(det < 0, det.new_tensor(-1.0), det.new_tensor(1.0))
  R = Vh.transpose(0, 1) @ D @ U.transpose(0, 1)
  t = cb - R @ ca
  ok = bool(torch.isfinite(R).all().item() and torch.isfinite(t).all().item())
  return R, t, ok


def _weighted_translation_torch(A, B, weights=None, robust='median'):
  residual = B - A
  if residual.shape[0] == 0:
    return A.new_zeros(3), False
  if robust == 'median' or weights is None:
    return torch.median(residual, dim=0).values, True
  weights = torch.clamp(weights.reshape(-1), min=1e-12)
  weights = weights / torch.clamp(weights.sum(), min=1e-12)
  return torch.sum(residual * weights[:, None], dim=0), True


def _subsample_pair_torch(A, B, max_points):
  max_points = int(max_points)
  if max_points <= 0 or A.shape[0] <= max_points:
    return A, B
  ids = torch.linspace(
    0,
    A.shape[0] - 1,
    steps=max_points,
    device=A.device,
  ).round().long()
  return A.index_select(0, ids), B.index_select(0, ids)


def _extract_crop_xyz_pairs_torch(
    pose_data,
    idx,
    z_min=1e-6,
    z_max=float('inf'),
    depth_diff_thresh=0.05,
    max_points=2048,
    roi_mask_crop=None,
):
  """Extract camera-frame metric point pairs directly on the CUDA device."""
  xyzA_t = getattr(pose_data, 'raw_xyz_mapAs', pose_data.xyz_mapAs)
  xyzB_t = getattr(pose_data, 'raw_xyz_mapBs', pose_data.xyz_mapBs)
  xyzA = xyzA_t[idx].permute(1, 2, 0).contiguous()
  xyzB = xyzB_t[idx].permute(1, 2, 0).contiguous()

  valid = torch.isfinite(xyzA).all(dim=-1) & torch.isfinite(xyzB).all(dim=-1)
  valid &= xyzA[..., 2] > float(z_min)
  valid &= xyzB[..., 2] > float(z_min)

  if roi_mask_crop is not None:
    m = roi_mask_crop
    if m.ndim == 3:
      m = m[0]
    valid &= m > 0.5

  if z_max is not None and np.isfinite(float(z_max)):
    valid &= xyzA[..., 2] < float(z_max)
    valid &= xyzB[..., 2] < float(z_max)

  if depth_diff_thresh is not None and float(depth_diff_thresh) > 0:
    valid &= torch.abs(xyzA[..., 2] - xyzB[..., 2]) < float(depth_diff_thresh)

  A = xyzA[valid].reshape(-1, 3)
  B = xyzB[valid].reshape(-1, 3)
  return _subsample_pair_torch(A, B, max_points=max_points)


def _pose_error_mean_torch(A, B, R=None, t=None, weights=None):
  if A.shape[0] == 0:
    return A.new_tensor(float('inf'))
  if R is not None and t is not None:
    A2 = A @ R.transpose(0, 1) + t.reshape(1, 3)
  else:
    A2 = A
  err = torch.linalg.vector_norm(A2 - B, dim=1)
  if weights is None:
    return err.mean()
  weights = torch.clamp(weights.reshape(-1), min=1e-12)
  return torch.sum(err * weights) / torch.clamp(weights.sum(), min=1e-12)


def _apply_mode_to_corr_torch(R, t, apply, max_rot_deg, max_trans):
  I = torch.eye(3, device=R.device, dtype=R.dtype)
  if apply == 'trans':
    R = I
  elif apply == 'trans_z':
    R = I
    t = torch.stack([t.new_zeros(()), t.new_zeros(()), t.reshape(3)[2]])
  elif apply != 'se3':
    R = I
  return _clamp_se3_torch(
    R,
    t,
    max_rot_deg=max_rot_deg,
    max_trans=max_trans,
    apply=apply,
  )


def _chunked_knn_torch(query, reference, k=1, chunk_size=1024):
  """Exact CUDA KNN using chunked torch.cdist to bound temporary memory."""
  if query.shape[0] == 0 or reference.shape[0] == 0:
    if k == 1:
      return query.new_empty((0,)), torch.empty((0,), device=query.device, dtype=torch.long)
    return query.new_empty((0, k)), torch.empty((0, k), device=query.device, dtype=torch.long)

  k = int(max(1, min(int(k), int(reference.shape[0]))))
  chunk_size = int(max(1, chunk_size))
  dist_parts = []
  id_parts = []
  for start in range(0, query.shape[0], chunk_size):
    q = query[start:start + chunk_size]
    distances = torch.cdist(q, reference, p=2.0)
    values, indices = torch.topk(distances, k=k, dim=1, largest=False, sorted=True)
    dist_parts.append(values)
    id_parts.append(indices)
  dists = torch.cat(dist_parts, dim=0)
  ids = torch.cat(id_parts, dim=0)
  if k == 1:
    return dists[:, 0], ids[:, 0]
  return dists, ids


def _regularize_covariances_torch(covs, eps=1e-6, min_diag=1e-6, relative_jitter=1e-3):
  """Regularize batched 3x3 PSD covariances without CUDA eigendecomposition."""
  if covs.numel() == 0:
    return covs.reshape(0, 3, 3)
  C = 0.5 * (covs + covs.transpose(-1, -2))
  scale = torch.diagonal(C, dim1=-2, dim2=-1).sum(dim=-1) / 3.0
  jitter = torch.clamp(scale * float(relative_jitter), min=float(min_diag)) + float(eps)
  I = torch.eye(3, device=C.device, dtype=C.dtype).unsqueeze(0)
  return C + jitter[:, None, None] * I


def _invert_covariances_torch(covs, eps=1e-6):
  """Batched SPD inverse on CUDA via Cholesky; no NumPy or SciPy path."""
  if covs.numel() == 0:
    return covs.reshape(0, 3, 3)
  C = _regularize_covariances_torch(covs, eps=eps)
  L, info = torch.linalg.cholesky_ex(C, check_errors=False)
  bad = info.ne(0).to(dtype=C.dtype)
  if bad.ndim == 0:
    bad = bad.reshape(1)
  I = torch.eye(3, device=C.device, dtype=C.dtype).unsqueeze(0)
  C_retry = C + bad[:, None, None] * 1e-3 * I
  L_retry, _ = torch.linalg.cholesky_ex(C_retry, check_errors=False)
  L = torch.where(bad[:, None, None].bool(), L_retry, L)
  return torch.cholesky_inverse(L)


def _voxel_stats_torch(points, voxel_size=0.01, min_points=5):
  """GPU voxel means/covariances using torch.unique and index_add_."""
  if points.shape[0] == 0:
    return (
      points.new_empty((0, 3)),
      points.new_empty((0, 3, 3)),
      torch.empty((0,), device=points.device, dtype=torch.long),
    )

  keys = torch.floor(points / float(voxel_size)).to(torch.int64)
  _, inverse, counts = torch.unique(
    keys,
    dim=0,
    sorted=True,
    return_inverse=True,
    return_counts=True,
  )
  n_voxels = int(counts.shape[0])

  sums = points.new_zeros((n_voxels, 3))
  sums.index_add_(0, inverse, points)
  counts_f = counts.to(points.dtype)
  means = sums / torch.clamp(counts_f[:, None], min=1.0)

  outer = points[:, :, None] * points[:, None, :]
  second = points.new_zeros((n_voxels, 3, 3))
  second_flat = second.reshape(n_voxels, 9)
  second_flat.index_add_(0, inverse, outer.reshape(-1, 9))
  second = second_flat.reshape(n_voxels, 3, 3)

  centered_ss = second - counts_f[:, None, None] * (
    means[:, :, None] * means[:, None, :]
  )
  denom = torch.clamp(counts_f - 1.0, min=1.0)
  covs = centered_ss / denom[:, None, None]

  keep = counts >= int(min_points)
  means = means[keep]
  covs = _regularize_covariances_torch(covs[keep])
  counts = counts[keep]
  return means, covs, counts


def _solve_linear_system_torch(H, g, damping=1e-8):
  H = 0.5 * (H + H.transpose(-1, -2))
  I = torch.eye(H.shape[-1], device=H.device, dtype=H.dtype)
  H1 = H + I * float(damping)
  x1, info1 = torch.linalg.solve_ex(H1, g.reshape(-1, 1), check_errors=False)
  H2 = H + I * max(float(damping) * 1000.0, 1e-5)
  x2, _ = torch.linalg.solve_ex(H2, g.reshape(-1, 1), check_errors=False)
  bad = info1.ne(0)
  x = torch.where(bad.reshape(1, 1), x2, x1)
  return x.reshape(-1)


def _mahalanobis_objective_torch(A, B, Omegas, R=None, t=None):
  if A.shape[0] == 0:
    return A.new_tensor(float('inf'))
  if R is not None and t is not None:
    A = A @ R.transpose(0, 1) + t.reshape(1, 3)
  residual = A - B
  values = torch.einsum('ni,nij,nj->n', residual, Omegas, residual)
  return values.mean()


def _solve_mahalanobis_delta_torch(
    A,
    B,
    Omegas,
    apply='trans',
    max_rot_deg=5.0,
    max_trans=0.03,
):
  if A.shape[0] < 3:
    return torch.eye(3, device=A.device, dtype=A.dtype), A.new_zeros(3), False

  residual = A - B
  I3 = torch.eye(3, device=A.device, dtype=A.dtype)

  if apply == 'trans':
    H = Omegas.sum(dim=0)
    g = torch.einsum('nij,nj->i', Omegas, residual)
    dt = -_solve_linear_system_torch(H, g)
    R, dt = _apply_mode_to_corr_torch(I3, dt, apply, max_rot_deg, max_trans)
    return R, dt, True

  if apply == 'trans_z':
    Hzz = Omegas[:, 2, 2].sum()
    gz = torch.sum(Omegas[:, 2, :] * residual)
    if abs(float(Hzz.item())) < 1e-12:
      return I3, A.new_zeros(3), False
    dz = -gz / Hzz
    dt = torch.stack([dz.new_zeros(()), dz.new_zeros(()), dz])
    R, dt = _apply_mode_to_corr_torch(I3, dt, apply, max_rot_deg, max_trans)
    return R, dt, True

  J = A.new_zeros((A.shape[0], 3, 6))
  J[:, 0, 1] = A[:, 2]
  J[:, 0, 2] = -A[:, 1]
  J[:, 1, 0] = -A[:, 2]
  J[:, 1, 2] = A[:, 0]
  J[:, 2, 0] = A[:, 1]
  J[:, 2, 1] = -A[:, 0]
  J[:, :, 3:] = I3.unsqueeze(0)

  H = torch.einsum('nki,nkl,nlj->ij', J, Omegas, J)
  g = torch.einsum('nki,nkl,nl->i', J, Omegas, residual)
  dxi = -_solve_linear_system_torch(H, g)
  R = _rotation_exp_torch(dxi[:3])
  dt = dxi[3:]
  R, dt = _apply_mode_to_corr_torch(R, dt, apply, max_rot_deg, max_trans)
  return R, dt, True


def _local_covariances_torch(points, k=10, chunk_size=1024):
  if points.shape[0] == 0:
    return points.new_empty((0, 3, 3))
  if points.shape[0] < 3:
    I = torch.eye(3, device=points.device, dtype=points.dtype) * 1e-6
    return I.unsqueeze(0).expand(points.shape[0], -1, -1).clone()

  k = int(max(3, min(int(k), int(points.shape[0]))))
  _, idxs = _chunked_knn_torch(
    points,
    points,
    k=k,
    chunk_size=chunk_size,
  )
  neighborhoods = points[idxs]
  centered = neighborhoods - neighborhoods.mean(dim=1, keepdim=True)
  covs = torch.einsum('nki,nkj->nij', centered, centered)
  covs = covs / float(max(k - 1, 1))
  return _regularize_covariances_torch(covs)


def _icp_correction_torch(
    A,
    B,
    apply='trans',
    max_corr_dist=0.03,
    max_iter=5,
    max_rot_deg=5.0,
    max_trans=0.03,
    min_points=30,
    cache=None,
    nn_chunk=1024,
):
  if A.shape[0] < min_points or B.shape[0] < min_points:
    return torch.eye(3, device=A.device, dtype=A.dtype), A.new_zeros(3), False

  src0 = A
  tgt = B
  R_total = torch.eye(3, device=A.device, dtype=A.dtype)
  t_total = A.new_zeros(3)
  ok_any = False

  for _ in range(int(max_iter)):
    src = src0 @ R_total.transpose(0, 1) + t_total.reshape(1, 3)
    dists, ids = _chunked_knn_torch(src, tgt, k=1, chunk_size=nn_chunk)
    valid = torch.isfinite(dists)
    if max_corr_dist is not None and float(max_corr_dist) > 0:
      valid &= dists < float(max_corr_dist)
    if int(valid.sum().item()) < int(min_points):
      break

    A_corr = src[valid]
    B_corr = tgt[ids[valid]]
    if apply in ['trans', 'trans_z']:
      step_t, step_ok = _weighted_translation_torch(A_corr, B_corr, robust='median')
      step_R = torch.eye(3, device=A.device, dtype=A.dtype)
    else:
      step_R, step_t, step_ok = _weighted_kabsch_torch(A_corr, B_corr)
    if not step_ok:
      break

    step_R, step_t = _apply_mode_to_corr_torch(
      step_R,
      step_t,
      apply,
      max_rot_deg,
      max_trans,
    )
    R_total = step_R @ R_total
    t_total = step_R @ t_total + step_t
    ok_any = True

    trans_small = float(torch.linalg.vector_norm(step_t).item()) < 1e-5
    rot_small = float(_rotation_angle_deg_torch(step_R).item()) < 0.05
    if trans_small and rot_small:
      break

  if not ok_any:
    return torch.eye(3, device=A.device, dtype=A.dtype), A.new_zeros(3), False
  R_total, t_total = _apply_mode_to_corr_torch(
    R_total,
    t_total,
    apply,
    max_rot_deg,
    max_trans,
  )
  return R_total, t_total, True


def _ndt_correction_torch(
    A,
    B,
    apply='trans',
    voxel_size=0.01,
    max_corr_dist=0.03,
    max_rot_deg=5.0,
    max_trans=0.03,
    min_points=30,
    voxel_min_points=5,
    max_iter=3,
    cache=None,
    nn_chunk=1024,
):
  if A.shape[0] < min_points or B.shape[0] < min_points:
    return torch.eye(3, device=A.device, dtype=A.dtype), A.new_zeros(3), False

  cache = {} if cache is None else cache
  voxel_key = ('target_voxel', round(float(voxel_size), 8), int(voxel_min_points))
  if voxel_key not in cache:
    means, covs, counts = _voxel_stats_torch(
      B,
      voxel_size=voxel_size,
      min_points=voxel_min_points,
    )
    cache[voxel_key] = (means, covs, counts)
  means, covs, _ = cache[voxel_key]
  if means.shape[0] < 3:
    return torch.eye(3, device=A.device, dtype=A.dtype), A.new_zeros(3), False

  inv_key = ('target_voxel_inv', round(float(voxel_size), 8), int(voxel_min_points))
  if inv_key not in cache:
    cache[inv_key] = _invert_covariances_torch(covs)
  inv_covs = cache[inv_key]

  R_total = torch.eye(3, device=A.device, dtype=A.dtype)
  t_total = A.new_zeros(3)
  ok_any = False

  for _ in range(int(max_iter)):
    src = A @ R_total.transpose(0, 1) + t_total.reshape(1, 3)
    dists, ids = _chunked_knn_torch(src, means, k=1, chunk_size=nn_chunk)
    valid = torch.isfinite(dists)
    if max_corr_dist is not None and float(max_corr_dist) > 0:
      valid &= dists < float(max_corr_dist)
    if int(valid.sum().item()) < int(min_points):
      break

    A_corr = src[valid]
    voxel_ids = ids[valid]
    B_corr = means[voxel_ids]
    Omegas = inv_covs[voxel_ids]

    before = _mahalanobis_objective_torch(A_corr, B_corr, Omegas)
    step_R, step_t, step_ok = _solve_mahalanobis_delta_torch(
      A_corr,
      B_corr,
      Omegas,
      apply=apply,
      max_rot_deg=max_rot_deg,
      max_trans=max_trans,
    )
    if not step_ok:
      break
    after = _mahalanobis_objective_torch(A_corr, B_corr, Omegas, R=step_R, t=step_t)
    if not bool(torch.isfinite(after).item()) or float(after.item()) > float(before.item()):
      break

    R_total = step_R @ R_total
    t_total = step_R @ t_total + step_t
    ok_any = True
    if float(torch.linalg.vector_norm(step_t).item()) < 1e-5 and float(_rotation_angle_deg_torch(step_R).item()) < 0.05:
      break

  if not ok_any:
    return torch.eye(3, device=A.device, dtype=A.dtype), A.new_zeros(3), False
  R_total, t_total = _apply_mode_to_corr_torch(R_total, t_total, apply, max_rot_deg, max_trans)
  return R_total, t_total, True


def _gicp_correction_torch(
    A,
    B,
    apply='trans',
    max_corr_dist=0.03,
    max_iter=3,
    max_rot_deg=5.0,
    max_trans=0.03,
    min_points=30,
    knn=10,
    cache=None,
    nn_chunk=1024,
):
  if A.shape[0] < min_points or B.shape[0] < min_points:
    return torch.eye(3, device=A.device, dtype=A.dtype), A.new_zeros(3), False

  cache = {} if cache is None else cache
  src_key = ('source_cov', int(knn))
  tgt_key = ('target_cov', int(knn))
  if src_key not in cache:
    cache[src_key] = _local_covariances_torch(A, k=knn, chunk_size=nn_chunk)
  if tgt_key not in cache:
    cache[tgt_key] = _local_covariances_torch(B, k=knn, chunk_size=nn_chunk)
  cov_src0 = cache[src_key]
  cov_tgt = cache[tgt_key]

  R_total = torch.eye(3, device=A.device, dtype=A.dtype)
  t_total = A.new_zeros(3)
  ok_any = False

  for _ in range(int(max_iter)):
    src = A @ R_total.transpose(0, 1) + t_total.reshape(1, 3)
    dists, ids = _chunked_knn_torch(src, B, k=1, chunk_size=nn_chunk)
    valid = torch.isfinite(dists)
    if max_corr_dist is not None and float(max_corr_dist) > 0:
      valid &= dists < float(max_corr_dist)
    if int(valid.sum().item()) < int(min_points):
      break

    src_ids = torch.nonzero(valid, as_tuple=False).reshape(-1)
    tgt_ids = ids[valid]
    A_corr = src[src_ids]
    B_corr = B[tgt_ids]

    Rb = R_total.unsqueeze(0)
    cov_src_cur = Rb @ cov_src0[src_ids] @ R_total.transpose(0, 1).unsqueeze(0)
    cov_pair = cov_tgt[tgt_ids] + cov_src_cur
    Omegas = _invert_covariances_torch(cov_pair)

    before = _mahalanobis_objective_torch(A_corr, B_corr, Omegas)
    step_R, step_t, step_ok = _solve_mahalanobis_delta_torch(
      A_corr,
      B_corr,
      Omegas,
      apply=apply,
      max_rot_deg=max_rot_deg,
      max_trans=max_trans,
    )
    if not step_ok:
      break
    after = _mahalanobis_objective_torch(A_corr, B_corr, Omegas, R=step_R, t=step_t)
    if not bool(torch.isfinite(after).item()) or float(after.item()) > float(before.item()):
      break

    R_total = step_R @ R_total
    t_total = step_R @ t_total + step_t
    ok_any = True
    if float(torch.linalg.vector_norm(step_t).item()) < 1e-5 and float(_rotation_angle_deg_torch(step_R).item()) < 0.05:
      break

  if not ok_any:
    return torch.eye(3, device=A.device, dtype=A.dtype), A.new_zeros(3), False
  R_total, t_total = _apply_mode_to_corr_torch(R_total, t_total, apply, max_rot_deg, max_trans)
  return R_total, t_total, True


def _vgicp_correction_torch(
    A,
    B,
    apply='trans',
    voxel_size=0.01,
    max_corr_dist=0.03,
    max_iter=3,
    max_rot_deg=5.0,
    max_trans=0.03,
    min_points=30,
    voxel_min_points=5,
    knn=10,
    cache=None,
    nn_chunk=1024,
):
  if A.shape[0] < min_points or B.shape[0] < min_points:
    return torch.eye(3, device=A.device, dtype=A.dtype), A.new_zeros(3), False

  cache = {} if cache is None else cache
  voxel_key = ('target_voxel', round(float(voxel_size), 8), int(voxel_min_points))
  if voxel_key not in cache:
    cache[voxel_key] = _voxel_stats_torch(
      B,
      voxel_size=voxel_size,
      min_points=voxel_min_points,
    )
  means, covs, _ = cache[voxel_key]
  if means.shape[0] < 3:
    return torch.eye(3, device=A.device, dtype=A.dtype), A.new_zeros(3), False

  src_key = ('source_cov', int(knn))
  if src_key not in cache:
    cache[src_key] = _local_covariances_torch(A, k=knn, chunk_size=nn_chunk)
  cov_src0 = cache[src_key]

  R_total = torch.eye(3, device=A.device, dtype=A.dtype)
  t_total = A.new_zeros(3)
  ok_any = False

  for _ in range(int(max_iter)):
    src = A @ R_total.transpose(0, 1) + t_total.reshape(1, 3)
    dists, ids = _chunked_knn_torch(src, means, k=1, chunk_size=nn_chunk)
    valid = torch.isfinite(dists)
    if max_corr_dist is not None and float(max_corr_dist) > 0:
      valid &= dists < float(max_corr_dist)
    if int(valid.sum().item()) < int(min_points):
      break

    src_ids = torch.nonzero(valid, as_tuple=False).reshape(-1)
    voxel_ids = ids[valid]
    A_corr = src[src_ids]
    B_corr = means[voxel_ids]

    Rb = R_total.unsqueeze(0)
    cov_src_cur = Rb @ cov_src0[src_ids] @ R_total.transpose(0, 1).unsqueeze(0)
    cov_pair = covs[voxel_ids] + cov_src_cur
    Omegas = _invert_covariances_torch(cov_pair)

    before = _mahalanobis_objective_torch(A_corr, B_corr, Omegas)
    step_R, step_t, step_ok = _solve_mahalanobis_delta_torch(
      A_corr,
      B_corr,
      Omegas,
      apply=apply,
      max_rot_deg=max_rot_deg,
      max_trans=max_trans,
    )
    if not step_ok:
      break
    after = _mahalanobis_objective_torch(A_corr, B_corr, Omegas, R=step_R, t=step_t)
    if not bool(torch.isfinite(after).item()) or float(after.item()) > float(before.item()):
      break

    R_total = step_R @ R_total
    t_total = step_R @ t_total + step_t
    ok_any = True
    if float(torch.linalg.vector_norm(step_t).item()) < 1e-5 and float(_rotation_angle_deg_torch(step_R).item()) < 0.05:
      break

  if not ok_any:
    return torch.eye(3, device=A.device, dtype=A.dtype), A.new_zeros(3), False
  R_total, t_total = _apply_mode_to_corr_torch(R_total, t_total, apply, max_rot_deg, max_trans)
  return R_total, t_total, True


def _apply_corr_to_pose_torch(T, R_corr, t_corr, apply='trans'):
  T = T.clone()
  if apply in ['trans', 'trans_z']:
    T[:3, 3] = T[:3, 3] + t_corr.reshape(3)
    return T
  Tc = torch.eye(4, device=T.device, dtype=T.dtype)
  Tc[:3, :3] = R_corr
  Tc[:3, 3] = t_corr
  return Tc @ T


# ============================================================
# Adaptive parameter selection for CUDA depth refinement.
# Only scalar configuration values leave the GPU; point clouds and all
# registration matrices remain CUDA tensors.
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


def _safe_quantile_torch(x, q, default=0.0):
  if x.numel() == 0:
    return float(default)
  x = x[torch.isfinite(x)]
  if x.numel() == 0:
    return float(default)
  return float(torch.quantile(x, float(q)).item())


def _robust_sigma_torch(x):
  if x.numel() == 0:
    return 0.0
  med = torch.median(x)
  mad = torch.median(torch.abs(x - med))
  return float((1.4826 * mad + 1e-12).item())


def _filter_pair_by_adaptive_depth_torch(A, B, min_points=30, user_thresh='auto'):
  if A.shape[0] == 0:
    return A, B, 0.0

  dz = torch.abs(A[:, 2] - B[:, 2])
  if not _is_auto_value(user_thresh):
    th = float(user_thresh)
    if th > 0:
      mask = dz < th
      if int(mask.sum().item()) >= int(min_points):
        return A[mask], B[mask], th
    return A, B, th

  med = float(torch.median(dz).item())
  sig = _robust_sigma_torch(dz)
  q80 = _safe_quantile_torch(dz, 0.80, med)
  q90 = _safe_quantile_torch(dz, 0.90, q80)
  th = float(np.clip(max(q80, med + 3.0 * sig), 0.008, 0.080))

  mask = dz < th
  if int(mask.sum().item()) < int(min_points):
    th = float(np.clip(q90, th, 0.100))
    mask = dz < th
  if int(mask.sum().item()) < int(min_points):
    return A, B, th
  return A[mask], B[mask], th


def _adaptive_depth_refine_params_torch(
    A,
    B,
    mode,
    apply,
    mesh_diameter=None,
    cfg=None,
    user_values=None,
):
  n = int(min(A.shape[0], B.shape[0]))
  diam = (
    float(mesh_diameter)
    if mesh_diameter is not None and np.isfinite(float(mesh_diameter)) and float(mesh_diameter) > 0
    else 0.10
  )

  if n > 0:
    residual = B[:n] - A[:n]
    dist = torch.linalg.vector_norm(residual, dim=1)
    dz = torch.abs(residual[:, 2])
    p50 = _safe_quantile_torch(dist, 0.50, 0.01)
    p70 = _safe_quantile_torch(dist, 0.70, p50)
    p85 = _safe_quantile_torch(dist, 0.85, p70)
    dz85 = _safe_quantile_torch(dz, 0.85, p70)
    if n >= 10:
      lo = torch.quantile(B[:n], 0.05, dim=0)
      hi = torch.quantile(B[:n], 0.95, dim=0)
      bbox_diag = float(torch.linalg.vector_norm(hi - lo).item())
    else:
      bbox_diag = diam
  else:
    p70 = p85 = dz85 = 0.01
    bbox_diag = diam

  user_values = user_values or {}

  def choose_float(key, auto_value):
    value = user_values.get(key, 'auto')
    return float(auto_value) if _is_auto_value(value) else float(value)

  def choose_int(key, auto_value):
    value = user_values.get(key, 'auto')
    return int(auto_value) if _is_auto_value(value) else int(float(value))

  max_points_auto = 8192 if n >= 4096 else max(2048, n)
  min_points_auto = int(np.clip(max(20, 0.01 * max(n, 1)), 20, 80))
  max_corr_auto = float(np.clip(
    max(0.012, 1.5 * p85, 0.04 * diam),
    0.012,
    min(0.100, max(0.030, 0.35 * diam)),
  ))

  if apply == 'trans_z':
    trans_auto = max(0.006, 1.5 * dz85, 0.030 * diam)
  else:
    trans_auto = max(0.006, 1.25 * p70, 0.030 * diam)
  trans_auto = float(np.clip(
    trans_auto,
    0.006,
    min(0.060, max(0.015, 0.20 * diam)),
  ))

  voxel_auto = float(np.clip(
    max(0.004, bbox_diag / 28.0, diam / 45.0),
    0.004,
    min(0.035, max(0.012, diam / 6.0)),
  ))

  knn_auto = int(np.clip(round(np.sqrt(max(n, 1)) / 2.0), 10, 40))
  if mode in ['gicp', 'vgicp']:
    knn_auto = int(np.clip(knn_auto, 15, 45))
  voxel_min_auto = 3 if n < 4096 else 4

  if mode == 'icp':
    iter_auto = 8
  elif mode in ['gicp', 'vgicp']:
    iter_auto = 5
  else:
    iter_auto = 4

  return {
    'max_points': choose_int('max_points', max_points_auto),
    'min_points': choose_int('min_points', min_points_auto),
    'max_corr_dist': choose_float('max_corr_dist', max_corr_auto),
    'icp_iter': choose_int('icp_iter', iter_auto),
    'max_trans': choose_float('max_trans', trans_auto),
    'max_rot_deg': choose_float('max_rot_deg', 3.0),
    'voxel_size': choose_float('voxel_size', voxel_auto),
    'voxel_min_points': choose_int('voxel_min_points', voxel_min_auto),
    'knn': choose_int('knn', knn_auto),
  }


def _run_depth_correction_with_params_torch(
    mode,
    A,
    B,
    apply,
    params,
    cache=None,
    nn_chunk=1024,
):
  common = dict(
    apply=apply,
    max_corr_dist=params['max_corr_dist'],
    max_iter=params['icp_iter'],
    max_rot_deg=params['max_rot_deg'],
    max_trans=params['max_trans'],
    min_points=params['min_points'],
    cache=cache,
    nn_chunk=nn_chunk,
  )
  if mode == 'icp':
    return _icp_correction_torch(A, B, **common)
  if mode == 'ndt':
    return _ndt_correction_torch(
      A,
      B,
      voxel_size=params['voxel_size'],
      voxel_min_points=params['voxel_min_points'],
      **common,
    )
  if mode == 'gicp':
    return _gicp_correction_torch(A, B, knn=params['knn'], **common)
  if mode == 'vgicp':
    return _vgicp_correction_torch(
      A,
      B,
      voxel_size=params['voxel_size'],
      voxel_min_points=params['voxel_min_points'],
      knn=params['knn'],
      **common,
    )
  return torch.eye(3, device=A.device, dtype=A.dtype), A.new_zeros(3), False


def _select_best_depth_correction_torch(
    mode,
    A,
    B,
    apply,
    params,
    auto_sweep=True,
    nn_chunk=1024,
):
  """Staged GPU search: 3 runs for ICP/GICP, at most 5 for NDT/VGICP."""
  before_score_t = _pose_error_mean_torch(A, B)
  before_score = float(before_score_t.item())
  cache = {}
  best = {
    'ok': False,
    'R': torch.eye(3, device=A.device, dtype=A.dtype),
    't': A.new_zeros(3),
    'score': before_score,
    'params': dict(params),
  }
  evaluated = set()

  def evaluate(candidate):
    key = (
      round(float(candidate['max_corr_dist']), 8),
      round(float(candidate['max_trans']), 8),
      round(float(candidate['voxel_size']), 8),
      int(candidate['knn']),
      int(candidate['icp_iter']),
    )
    if key in evaluated:
      return None
    evaluated.add(key)
    R_corr, t_corr, ok = _run_depth_correction_with_params_torch(
      mode,
      A,
      B,
      apply,
      candidate,
      cache=cache,
      nn_chunk=nn_chunk,
    )
    if not ok:
      return None
    score_t = _pose_error_mean_torch(A, B, R=R_corr, t=t_corr)
    score = float(score_t.item())
    if np.isfinite(score) and score < best['score']:
      best.update({
        'ok': True,
        'R': R_corr,
        't': t_corr,
        'score': score,
        'params': dict(candidate),
      })
    return R_corr, t_corr, score, dict(candidate)

  if not auto_sweep:
    evaluate(dict(params))
    return best, before_score

  base = dict(params)
  for scale in (0.75, 1.0, 1.35):
    candidate = dict(base)
    candidate['max_corr_dist'] = float(np.clip(
      base['max_corr_dist'] * scale,
      0.006,
      0.120,
    ))
    evaluate(candidate)

  if mode in ['ndt', 'vgicp']:
    corr_best = best['params']['max_corr_dist'] if best['ok'] else base['max_corr_dist']
    for scale in (0.75, 1.35):
      candidate = dict(base)
      candidate['max_corr_dist'] = float(corr_best)
      candidate['voxel_size'] = float(np.clip(
        base['voxel_size'] * scale,
        0.003,
        0.050,
      ))
      evaluate(candidate)

  if best['ok']:
    raw_R = best['R']
    raw_t = best['t']
    raw_params = dict(best['params'])
    for scale in (0.75, 1.0, 1.35):
      candidate = dict(raw_params)
      candidate['max_trans'] = float(np.clip(
        base['max_trans'] * scale,
        0.003,
        0.080,
      ))
      R_clamped, t_clamped = _apply_mode_to_corr_torch(
        raw_R,
        raw_t,
        apply,
        candidate['max_rot_deg'],
        candidate['max_trans'],
      )
      score = float(_pose_error_mean_torch(A, B, R=R_clamped, t=t_clamped).item())
      if np.isfinite(score) and score < best['score']:
        best.update({
          'ok': True,
          'R': R_clamped,
          't': t_clamped,
          'score': score,
          'params': candidate,
        })

  return best, before_score



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
    if 'depth_refine_nn_chunk' not in self.cfg:
      self.cfg['depth_refine_nn_chunk'] = 1024
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

  # def set_depth_refine(self, mode=None, apply=None, **kwargs):
  #   """
  #   Runtime setter for test-time depth geometry refinement.

  #   mode:
  #     none / icp / ndt / gicp / vgicp
  #   apply:
  #     trans / trans_z / se3
  #   """
  #   if mode is not None:
  #     self.cfg['depth_refine_mode'] = str(mode).lower()
  #   if apply is not None:
  #     self.cfg['depth_refine_apply'] = str(apply).lower()
  #   for k, v in kwargs.items():
  #     self.cfg[k] = v

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
      mesh_diameter=mesh_diameter,
      keep_raw_geometry=True
    )

  @torch.inference_mode()
  def depth_geometry_refine_poses(self, rgb, depth, K, ob_in_cams, xyz_map, normal_map=None, mesh=None, mesh_tensors=None, glctx=None, mesh_diameter=None, depth_roi_mask=None):
    """
    Apply test-time depth correction with a PyTorch CUDA backend.

    Point extraction, exact nearest-neighbour search, voxel statistics, local
    covariances, Mahalanobis costs, Gauss-Newton solves, and pose updates remain
    on the same CUDA device. Only small scalar values are synchronized for
    adaptive parameter selection, stopping conditions, and logging.
    """
    mode = str(_cfg_get(self.cfg, 'depth_refine_mode', 'none')).lower()
    apply = str(_cfg_get(self.cfg, 'depth_refine_apply', 'trans')).lower()

    if torch.is_tensor(ob_in_cams) and ob_in_cams.is_cuda:
      device = ob_in_cams.device
    else:
      if not torch.cuda.is_available():
        raise RuntimeError(
          'Depth refinement backend requires CUDA, but torch.cuda.is_available() is False.'
        )
      device = torch.device('cuda', torch.cuda.current_device())

    B_in_cams = _as_cuda_float_tensor(ob_in_cams, device=device)

    if mode in ['none', 'off', 'false', '0']:
      return B_in_cams

    if mode not in ['icp', 'ndt', 'gicp', 'vgicp']:
      logging.warning(f"[DepthRefine] Unknown mode={mode}, skip.")
      return B_in_cams

    if apply not in ['trans', 'trans_z', 'se3']:
      logging.warning(f"[DepthRefine] Unknown apply={apply}, use trans.")
      apply = 'trans'

    if mesh_tensors is None:
      mesh_tensors = make_mesh_tensors(mesh)

    rgb_tensor = _as_cuda_float_tensor(rgb, device=device)
    depth_tensor = _as_cuda_float_tensor(depth, device=device)
    xyz_map_tensor = _as_cuda_float_tensor(xyz_map, device=device)

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
      mesh_diameter=mesh_diameter,
    )

    roi_mask_crops = None
    if depth_roi_mask is not None:
      mask_tensor = _as_cuda_float_tensor(depth_roi_mask, device=device)
      if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor[None, None]
      elif mask_tensor.ndim == 3:
        mask_tensor = mask_tensor[:, None]
      mask_tensor = mask_tensor.expand(B_in_cams.shape[0], -1, -1, -1)
      tf_to_crops = getattr(pose_data, 'raw_tf_to_crops', pose_data.tf_to_crops)
      roi_mask_crops = kornia.geometry.transform.warp_perspective(
        mask_tensor,
        tf_to_crops,
        dsize=self.cfg.input_resize,
        mode='nearest',
        align_corners=False,
      )

    poses_t = B_in_cams.clone()

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

    max_points_raw = _to_int_or_auto(raw_user['max_points'])
    extract_max_points = 12000 if max_points_raw == 'auto' else int(max_points_raw)

    min_points_raw = _to_int_or_auto(raw_user['min_points'])
    pre_min_points = 20 if min_points_raw == 'auto' else int(min_points_raw)

    depth_diff_raw = _to_float_or_auto(raw_user['depth_diff_thresh'])
    extract_depth_thresh = None if depth_diff_raw == 'auto' else float(depth_diff_raw)

    accept_if_better = _to_bool(
      _cfg_get(self.cfg, 'depth_refine_accept_if_better', True),
      default=True,
    )
    do_log = _to_bool(
      _cfg_get(self.cfg, 'depth_refine_log', True),
      default=True,
    )
    auto_sweep = _to_bool(
      _cfg_get(self.cfg, 'depth_refine_auto_sweep', True),
      default=True,
    )
    nn_chunk = int(_cfg_get(self.cfg, 'depth_refine_nn_chunk', 1024))
    nn_chunk = max(128, nn_chunk)

    n_ok = 0
    n_try = 0
    last_params = None
    used_depth_thresh = None

    for i in range(poses_t.shape[0]):
      roi_mask_crop = roi_mask_crops[i] if roi_mask_crops is not None else None

      A, B = _extract_crop_xyz_pairs_torch(
        pose_data,
        i,
        z_min=1e-6,
        z_max=_cfg_get(self.cfg, 'zfar', float('inf')),
        depth_diff_thresh=extract_depth_thresh,
        max_points=extract_max_points,
        roi_mask_crop=roi_mask_crop,
      )

      A, B, used_depth_thresh = _filter_pair_by_adaptive_depth_torch(
        A,
        B,
        min_points=pre_min_points,
        user_thresh=raw_user['depth_diff_thresh'],
      )

      if A.shape[0] < pre_min_points or B.shape[0] < pre_min_points:
        continue

      params = _adaptive_depth_refine_params_torch(
        A,
        B,
        mode=mode,
        apply=apply,
        mesh_diameter=mesh_diameter,
        cfg=self.cfg,
        user_values=raw_user,
      )

      A, B = _subsample_pair_torch(A, B, params['max_points'])
      if A.shape[0] < params['min_points'] or B.shape[0] < params['min_points']:
        continue

      n_try += 1
      best, before_err = _select_best_depth_correction_torch(
        mode,
        A,
        B,
        apply,
        params,
        auto_sweep=auto_sweep,
        nn_chunk=nn_chunk,
      )

      if not best['ok']:
        continue
      if accept_if_better and not (best['score'] < before_err):
        continue

      poses_t[i] = _apply_corr_to_pose_torch(
        poses_t[i],
        best['R'],
        best['t'],
        apply=apply,
      )
      last_params = best['params']
      n_ok += 1

    if do_log:
      if last_params is None:
        logging.info(
          f"[DepthRefine] backend=torch_cuda, device={device}, mode={mode}, "
          f"apply={apply}, accepted={n_ok}/{n_try}, candidates={poses_t.shape[0]}, "
          f"adaptive=True, auto_sweep={auto_sweep}, nn_chunk={nn_chunk}, "
          f"no accepted correction"
        )
      else:
        logging.info(
          f"[DepthRefine] backend=torch_cuda, device={device}, mode={mode}, "
          f"apply={apply}, accepted={n_ok}/{n_try}, candidates={poses_t.shape[0]}, "
          f"adaptive=True, auto_sweep={auto_sweep}, nn_chunk={nn_chunk}, "
          f"depth_th={used_depth_thresh:.4f}, "
          f"corr={last_params['max_corr_dist']:.4f}m, "
          f"clamp={last_params['max_trans']:.4f}m, "
          f"voxel={last_params['voxel_size']:.4f}m, "
          f"knn={last_params['knn']}, iter={last_params['icp_iter']}"
        )

    return poses_t

  # Backward-compatible wrapper name. Keep tensors on CUDA when supplied.
  def _depth_geometry_refine_batch(self, B_in_cams, rgb_tensor, depth_tensor, K, xyz_map_tensor, normal_map, mesh_centered, glctx, mesh_tensors, mesh_diameter, depth_roi_mask=None):
    return self.depth_geometry_refine_poses(
      rgb=rgb_tensor,
      depth=depth_tensor,
      K=K,
      ob_in_cams=B_in_cams,
      xyz_map=xyz_map_tensor,
      normal_map=normal_map,
      mesh=mesh_centered,
      mesh_tensors=mesh_tensors,
      glctx=glctx,
      mesh_diameter=mesh_diameter,
      depth_roi_mask=depth_roi_mask,
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

    self.last_trans_update = trans_delta
    self.last_rot_update = rot_mat_delta
    # logging.info(f"model: {self.model}")
    if get_vis:
      # logging.info("get_vis...")
      canvas = []
      padding = 2
      pose_data = make_crop_data_batch(self.cfg.input_resize, torch.as_tensor(ob_centered_in_cams), mesh_centered, rgb, depth, K, crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor, cfg=self.cfg, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter, keep_raw_geometry=False)
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

      pose_data = make_crop_data_batch(self.cfg.input_resize, B_in_cams, mesh_centered, rgb, depth, K, crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor, cfg=self.cfg, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter, keep_raw_geometry=False)
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
