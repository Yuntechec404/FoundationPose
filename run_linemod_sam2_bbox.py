# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# Modified for GT-bbox -> SAM2-mask segmentation ablation, depth-refine ablation, and geometry-guided SAM2 mask refinement.

from Utils import *
import json, uuid, joblib, os, sys
import csv, time
import glob, copy
import scipy.spatial as spatial
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat
import itertools
import argparse
from pathlib import Path

from datareader import *
from estimater import *

try:
  from ultralytics import SAM
except Exception:
  SAM = None

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/mycpp/build')
import yaml


# ============================================================
# BOP reader patch:
# 部分 BOP 測試集 rgb 數量可能多於 scene_gt.json 的 key。
# 原版 BopBaseReader 會 assert len(scene_gt)==len(color_files)，
# 這裡改成以 scene_gt.json 的 key 為主，只保留有 GT 標註的 rgb frame。
# 同時支援 BOP 標準 models/obj_XXXXXX.ply 的 model-based mesh 路徑。
# ============================================================

def patch_bop_reader_filter_by_scene_gt():
  def _patched_bop_init(self, base_dir, zfar=np.inf, resize=1):
    self.base_dir = base_dir
    self.resize = resize
    self.dataset_name = None

    self.color_files = sorted(glob.glob(f"{self.base_dir}/rgb/*"))
    if len(self.color_files) == 0:
      self.color_files = sorted(glob.glob(f"{self.base_dir}/gray/*"))

    self.zfar = zfar

    self.K_table = {}
    with open(f'{self.base_dir}/scene_camera.json', 'r') as ff:
      info = json.load(ff)

    for k in info:
      self.K_table[f'{int(k):06d}'] = np.array(info[k]['cam_K']).reshape(3, 3)
      self.bop_depth_scale = info[k]['depth_scale']

    if os.path.exists(f'{self.base_dir}/scene_gt.json'):
      with open(f'{self.base_dir}/scene_gt.json', 'r') as ff:
        self.scene_gt = json.load(ff)
      self.scene_gt = copy.deepcopy(self.scene_gt)

      gt_ids = set([f"{int(k):06d}" for k in self.scene_gt.keys()])
      before_n = len(self.color_files)
      self.color_files = [
        p for p in self.color_files
        if os.path.splitext(os.path.basename(p))[0] in gt_ids
      ]
      after_n = len(self.color_files)

      if before_n != after_n:
        logging.info(
          f"[BOPReaderPatch] {self.base_dir}: rgb files {before_n} -> {after_n}, "
          f"filtered by scene_gt.json keys."
        )

      if len(self.color_files) == 0:
        raise RuntimeError(
          f"[BOPReaderPatch] No rgb files matched scene_gt.json keys in {self.base_dir}"
        )
    else:
      self.scene_gt = None

    self.make_id_strs()

  def _patched_linemod_get_gt_mesh_file(self, ob_id):
    root = self.base_dir
    while True:
      mesh_file = f'{root}/models/obj_{ob_id:06d}.ply'
      if os.path.exists(mesh_file):
        return mesh_file

      mesh_file = f'{root}/lm_models/models/obj_{ob_id:06d}.ply'
      if os.path.exists(mesh_file):
        return mesh_file

      parent = os.path.abspath(f'{root}/../')
      if parent == root:
        raise FileNotFoundError(
          f"Cannot find model-based mesh obj_{ob_id:06d}.ply from base_dir={self.base_dir}"
        )
      root = parent

  BopBaseReader.__init__ = _patched_bop_init
  LinemodReader.get_gt_mesh_file = _patched_linemod_get_gt_mesh_file
  if 'LinemodOcclusionReader' in globals():
    LinemodOcclusionReader.get_gt_mesh_file = _patched_linemod_get_gt_mesh_file


patch_bop_reader_filter_by_scene_gt()


def infer_bop_dataset_name(dataset_root):
  return os.path.basename(os.path.abspath(dataset_root)).lower()


def get_bop_reader_class(dataset_name):
  dataset_name = str(dataset_name).lower()
  if dataset_name == 'lmo' and 'LinemodOcclusionReader' in globals():
    return LinemodOcclusionReader
  return LinemodReader


def parse_obj_ids_arg(obj_ids_arg):
  if obj_ids_arg is None:
    return None
  s = str(obj_ids_arg).strip()
  if s == "" or s.lower() in ["all", "none", "null"]:
    return None
  s = s.replace(",", " ")
  ids = []
  for token in s.split():
    token = token.strip()
    if token == "":
      continue
    ids.append(int(token))
  return sorted(set(ids))




def parse_str_list(s, default=None):
  if s is None:
    return default if default is not None else []
  s = str(s).strip()
  if s == "":
    return default if default is not None else []
  return [x.strip().lower() for x in s.replace(";", ",").replace(" ", ",").split(",") if x.strip()]


def parse_depth_refine_applies(s, default=None):
  """
  Parse --depth_refine_apply.

  Supported:
    --depth_refine_apply trans_z
    --depth_refine_apply trans,trans_z,se3
    --depth_refine_apply all
  """
  valid = ["trans", "trans_z", "se3"]
  items = parse_str_list(s, default=default or ["trans_z"])
  if len(items) == 0:
    items = default or ["trans_z"]

  expanded = []
  for x in items:
    x = str(x).strip().lower()
    if x in ["all", "*"]:
      expanded.extend(valid)
    else:
      expanded.append(x)

  out = []
  for x in expanded:
    if x not in valid:
      raise ValueError(f"Unknown depth_refine_apply={x}. Use trans, trans_z, se3, or all.")
    if x not in out:
      out.append(x)
  return out


def iter_depth_refine_mode_apply(depth_modes, depth_applies):
  """
  Return experiment pairs.

  Important:
    mode=none does not depend on apply, so it is generated only once.
    Otherwise running all applies would duplicate the same *_none CSV rows.
  """
  pairs = []
  seen_none = False
  for mode in depth_modes:
    mode = str(mode).strip().lower()
    if mode in ["none", "off", "false", "0"]:
      if not seen_none:
        pairs.append(("none", "none"))
        seen_none = True
      continue

    for apply in depth_applies:
      pairs.append((mode, apply))

  if len(pairs) == 0:
    pairs.append(("none", "none"))
  return pairs


def apply_for_cfg(mode, apply):
  """
  predict_pose_refine.py only needs apply when mode is not none.
  Use trans as a safe placeholder for none.
  """
  mode = str(mode).lower()
  apply = str(apply).lower()
  if mode in ["none", "off", "false", "0"] or apply in ["none", "off", "false", "0"]:
    return "trans"
  return apply



def make_reader(reader_cls, video_dir):
  try:
    return reader_cls(video_dir, split=None)
  except TypeError as e:
    if 'split' in str(e):
      return reader_cls(video_dir)
    raise


def pose_to_bop_row(scene_id, im_id, obj_id, pose, score=1.0, runtime_sec=0.0, translation_scale=1000.0):
  pose = np.asarray(pose, dtype=np.float64)
  if pose.shape != (4, 4):
    raise ValueError(f"pose must be 4x4, got {pose.shape}")

  R = pose[:3, :3].reshape(-1)
  t = pose[:3, 3] * float(translation_scale)

  return {
    "scene_id": int(scene_id),
    "im_id": int(im_id),
    "obj_id": int(obj_id),
    "score": float(score),
    "R": " ".join([f"{x:.8f}" for x in R]),
    "t": " ".join([f"{x:.8f}" for x in t]),
    "time": float(runtime_sec),
  }


def make_bop_times_consistent(rows, mode="sum"):
  if mode == "zero":
    for r in rows:
      r["time"] = 0.0
    return rows

  time_table = {}
  for r in rows:
    key = (int(r["scene_id"]), int(r["im_id"]))
    if key not in time_table:
      time_table[key] = []
    time_table[key].append(float(r["time"]))

  for r in rows:
    key = (int(r["scene_id"]), int(r["im_id"]))
    if mode == "max":
      r["time"] = max(time_table[key])
    else:
      r["time"] = sum(time_table[key])
  return rows


def write_bop_results_csv(csv_path, rows, time_mode="sum"):
  rows = make_bop_times_consistent(rows, mode=time_mode)
  os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
    writer.writeheader()
    for r in rows:
      writer.writerow(r)
  logging.info(f"[DONE] BOP19 CSV saved to {csv_path}, rows={len(rows)}, time_mode={time_mode}")


# ============================================================
# Mask / bbox / SAM2 helpers.
# ============================================================

def clip_xyxy(xyxy, W, H):
  x1, y1, x2, y2 = [float(v) for v in xyxy]
  return np.array([
    max(0.0, min(W - 1.0, x1)),
    max(0.0, min(H - 1.0, y1)),
    max(0.0, min(W - 1.0, x2)),
    max(0.0, min(H - 1.0, y2)),
  ], dtype=np.float32)


def gt_mask_to_expanded_bbox(gt_mask, expand=0.10):
  """Use GT mask only to generate a bbox prompt, then expand it."""
  if gt_mask is None:
    return None
  gt = np.asarray(gt_mask) > 0
  H, W = gt.shape[:2]
  vs, us = np.where(gt)
  if len(vs) == 0:
    return None

  x1, x2 = float(us.min()), float(us.max() + 1)
  y1, y2 = float(vs.min()), float(vs.max() + 1)
  bw = max(1.0, x2 - x1)
  bh = max(1.0, y2 - y1)
  x1 -= bw * float(expand)
  x2 += bw * float(expand)
  y1 -= bh * float(expand)
  y2 += bh * float(expand)
  return clip_xyxy([x1, y1, x2, y2], W, H)


def bbox_to_mask(xyxy, H, W):
  m = np.zeros((H, W), dtype=bool)
  if xyxy is None:
    return m
  x1, y1, x2, y2 = clip_xyxy(xyxy, W, H).astype(np.int32)
  if x2 <= x1 or y2 <= y1:
    return m
  m[y1:y2, x1:x2] = True
  return m


def compute_mask_metrics(pred_mask, gt_mask):
  pred = np.asarray(pred_mask).astype(bool) if pred_mask is not None else None
  gt = np.asarray(gt_mask).astype(bool) if gt_mask is not None else None
  if pred is None or gt is None or pred.shape != gt.shape:
    return {
      "mask_valid": 0,
      "mask_iou": 0.0,
      "mask_dice": 0.0,
      "mask_precision": 0.0,
      "mask_recall": 0.0,
      "mask_f1": 0.0,
      "mask_pred_area": 0,
      "mask_gt_area": int(gt.sum()) if gt is not None else 0,
      "mask_area_ratio": 0.0,
    }

  inter = int(np.logical_and(pred, gt).sum())
  union = int(np.logical_or(pred, gt).sum())
  pred_area = int(pred.sum())
  gt_area = int(gt.sum())
  fp = int(np.logical_and(pred, ~gt).sum())
  fn = int(np.logical_and(~pred, gt).sum())

  precision = inter / max(inter + fp, 1)
  recall = inter / max(inter + fn, 1)
  dice = (2.0 * inter) / max(pred_area + gt_area, 1)
  f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
  return {
    "mask_valid": 1 if pred_area > 0 and gt_area > 0 else 0,
    "mask_iou": float(inter / max(union, 1)),
    "mask_dice": float(dice),
    "mask_precision": float(precision),
    "mask_recall": float(recall),
    "mask_f1": float(f1),
    "mask_pred_area": pred_area,
    "mask_gt_area": gt_area,
    "mask_area_ratio": float(pred_area / max(gt_area, 1)),
  }


def write_dict_rows_csv(path, rows):
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  if len(rows) == 0:
    with open(path, "w", newline="") as f:
      f.write("")
    return
  keys = []
  for r in rows:
    for k in r.keys():
      if k not in keys:
        keys.append(k)
  with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
  logging.info(f"[DONE] CSV saved: {path}, rows={len(rows)}")


def load_sam2_model(sam_ckpt, imgsz=640):
  if SAM is None:
    raise RuntimeError("ultralytics.SAM is not available. Install/activate ultralytics environment.")
  if sam_ckpt is None or str(sam_ckpt).strip() == "":
    raise RuntimeError("--sam_ckpt is required for SAM2 segmentation.")
  model = SAM(str(sam_ckpt))
  dev = "cuda:0" if torch.cuda.is_available() else "cpu"
  try:
    dummy = np.zeros((int(imgsz), int(imgsz), 3), np.uint8)
    _ = model(dummy, bboxes=[[10, 10, 100, 100]], imgsz=int(imgsz), device=dev, verbose=False)
  except Exception as e:
    logging.info(f"[SAM2] warmup warning: {e}")
  logging.info(f"[SAM2] loaded ckpt={sam_ckpt}, device={dev}, imgsz={imgsz}")
  return model


def sam2_segment_from_bbox(prompt_model, color, xyxy, imgsz=640):
  H, W = color.shape[:2]
  if prompt_model is None or xyxy is None:
    return np.zeros((H, W), dtype=bool), 0.0, "no_model_or_bbox"
  x1, y1, x2, y2 = clip_xyxy(xyxy, W, H).astype(int)
  dev = "cuda:0" if torch.cuda.is_available() else "cpu"
  t0 = time.perf_counter()
  try:
    # Same Ultralytics SAM/SAM2 usage as foundationpose_tracker.py:
    # prompt_model(color, bboxes=[[x1,y1,x2,y2]], imgsz=..., device=..., verbose=False)
    results = prompt_model(color, bboxes=[[int(x1), int(y1), int(x2), int(y2)]], imgsz=int(imgsz), device=dev, verbose=False)
    seg_time = time.perf_counter() - t0
    if results and results[0].masks is not None:
      m = results[0].masks.data
      if torch.is_tensor(m):
        m = m.detach().cpu().numpy()
      m = np.asarray(m)
      if m.ndim == 3:
        m2 = m[0]
      else:
        m2 = m
      if m2.shape != (H, W):
        m2 = cv2.resize(m2.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
      return (m2 > 0).astype(bool), seg_time, "ok"
    return np.zeros((H, W), dtype=bool), seg_time, "empty_mask"
  except Exception as e:
    return np.zeros((H, W), dtype=bool), time.perf_counter() - t0, f"error:{e}"


def save_mask_vis(path, color, gt_mask, bbox_xyxy=None, sam2_mask=None):
  try:
    vis = color.copy()
    if gt_mask is not None:
      overlay = np.zeros_like(vis)
      overlay[np.asarray(gt_mask) > 0] = [0, 255, 0]
      vis = cv2.addWeighted(vis, 0.75, overlay, 0.25, 0)
    if sam2_mask is not None:
      overlay = np.zeros_like(vis)
      overlay[np.asarray(sam2_mask) > 0] = [0, 0, 255]
      vis = cv2.addWeighted(vis, 0.75, overlay, 0.25, 0)
    if bbox_xyxy is not None:
      x1, y1, x2, y2 = bbox_xyxy.astype(int)
      cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, vis)
  except Exception as e:
    logging.info(f"[VIS] failed to save {path}: {e}")


# ============================================================
# Geometry-guided SAM2 mask refinement helpers.
# ============================================================

def keep_largest_component(mask):
  mask = np.asarray(mask).astype(bool)
  if mask.sum() == 0:
    return mask
  num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
  if num <= 1:
    return mask
  # label 0 is background.
  areas = stats[1:, cv2.CC_STAT_AREA]
  keep_label = int(np.argmax(areas) + 1)
  return labels == keep_label


def pose_to_centered_pose(est, pose):
  """Convert returned FoundationPose pose (original mesh -> camera) to centered mesh pose."""
  pose_np = pose.detach().cpu().numpy() if torch.is_tensor(pose) else np.asarray(pose)
  pose_np = pose_np.reshape(4, 4).astype(np.float32)
  try:
    T_orig_to_center = est.get_tf_to_centered_mesh()
    if torch.is_tensor(T_orig_to_center):
      T_orig_to_center = T_orig_to_center.detach().cpu().numpy()
    centered_pose = pose_np @ np.linalg.inv(T_orig_to_center.astype(np.float32))
    return centered_pose.astype(np.float32)
  except Exception:
    # Fallback: if the pose was already centered, rendering will still work.
    return pose_np.astype(np.float32)


def render_cad_depth_mask(est, K, H, W, pose):
  """Render CAD depth/mask from a returned FoundationPose pose."""
  centered_pose = pose_to_centered_pose(est, pose)
  ob_in_cams = torch.as_tensor(centered_pose[None], device='cuda', dtype=torch.float32)
  with torch.no_grad():
    _, render_depth, _ = nvdiffrast_render(
      K=K,
      H=int(H),
      W=int(W),
      ob_in_cams=ob_in_cams,
      glctx=est.glctx,
      mesh_tensors=est.mesh_tensors,
    )
  if torch.is_tensor(render_depth):
    rd = render_depth.detach()
    if rd.ndim == 4:
      rd = rd[0, ..., 0]
    elif rd.ndim == 3:
      rd = rd[0]
    rd = rd.float().cpu().numpy()
  else:
    rd = np.asarray(render_depth)
    if rd.ndim == 4:
      rd = rd[0, ..., 0]
    elif rd.ndim == 3:
      rd = rd[0]
  rd = np.nan_to_num(rd.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
  rm = rd > 1e-6
  return rd, rm


def estimate_geometry_depth_tolerance(render_depth, render_mask):
  if float(opt.geo_depth_tol) > 0:
    return float(opt.geo_depth_tol)
  vals = np.asarray(render_depth)[np.asarray(render_mask).astype(bool)]
  vals = vals[np.isfinite(vals) & (vals > 1e-6)]
  if vals.size == 0:
    return float(opt.geo_depth_tol_max)
  z_med = float(np.median(vals))
  tol = float(opt.geo_depth_tol_ratio) * max(z_med, 1e-6)
  tol = max(float(opt.geo_depth_tol_min), min(float(opt.geo_depth_tol_max), tol))
  return float(tol)


def geometry_refine_sam2_mask(sam2_mask, depth, bbox_mask, render_depth, render_mask, method='none'):
  """
  Refine SAM2 mask using CAD-rendered geometry and observed depth.

  method is used for bookkeeping and for small method-specific morphology choices.
  The main difference among none/icp/ndt/gicp/vgicp comes from the pose used to
  render CAD depth/mask. That pose is generated by the corresponding depth refine
  mode before this function is called.
  """
  sam2 = np.asarray(sam2_mask).astype(bool)
  bbox = np.asarray(bbox_mask).astype(bool)
  rd = np.asarray(render_depth).astype(np.float32)
  rm = np.asarray(render_mask).astype(bool)
  obs = np.asarray(depth).astype(np.float32)

  if sam2.shape != obs.shape:
    raise ValueError(f"sam2_mask shape {sam2.shape} != depth shape {obs.shape}")

  kernel = np.ones((3, 3), dtype=np.uint8)
  rm_use = rm.astype(np.uint8)
  if int(opt.geo_render_dilate_iter) > 0:
    rm_use = cv2.dilate(rm_use, kernel, iterations=int(opt.geo_render_dilate_iter))
  rm_use = rm_use.astype(bool)

  tol = estimate_geometry_depth_tolerance(rd, rm)
  valid_obs = obs > 1e-6
  depth_diff = np.abs(obs - rd)
  depth_inlier = valid_obs & rm_use & (depth_diff <= tol)

  # Remove SAM2 pixels that are far from rendered CAD silhouette/depth.
  refined = sam2 & rm_use & valid_obs
  refined = refined & ((depth_diff <= tol) | (~rm))

  # Optionally add CAD-rendered visible pixels that agree with observed depth.
  # This recovers SAM2 false negatives inside the GT-derived bbox prompt.
  if bool(opt.geo_add_render_inliers):
    refined = refined | (bbox & depth_inlier)

  # Restrict to the GT-derived bbox prompt to avoid growing into unrelated objects.
  refined = refined & bbox

  if int(opt.geo_open_iter) > 0:
    refined = cv2.morphologyEx(refined.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=int(opt.geo_open_iter)).astype(bool)
  if int(opt.geo_close_iter) > 0:
    refined = cv2.morphologyEx(refined.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=int(opt.geo_close_iter)).astype(bool)

  if bool(opt.geo_keep_largest_cc):
    refined = keep_largest_component(refined)

  # Avoid catastrophic mask collapse. If geometry filtering removes too much,
  # fallback to SAM2 intersect rendered silhouette, then to original SAM2.
  min_area = int(max(1, float(opt.geo_min_area_ratio) * max(int(sam2.sum()), 1)))
  fallback_used = "none"
  if int(refined.sum()) < min_area:
    fallback = (sam2 & rm_use & bbox)
    if int(fallback.sum()) >= min_area:
      refined = fallback
      fallback_used = "sam2_and_render"
    else:
      refined = sam2 & bbox
      fallback_used = "sam2_original"

  # Extra acceptance gate to prevent over-deletion.
  # This gate uses only the original SAM2 mask and geometry statistics, not GT.
  # If geometry removes too much of the original SAM2 support, keep SAM2.
  accept_gate_used = "off"
  accept_gate_reason = "ok"
  sam2_area = max(int(sam2.sum()), 1)
  refined_area = int(refined.sum())
  removed_pixels_tmp = int(np.logical_and(sam2, ~refined).sum())
  added_pixels_tmp = int(np.logical_and(refined, ~sam2).sum())
  union_with_sam2 = int(np.logical_or(refined, sam2).sum())
  inter_with_sam2 = int(np.logical_and(refined, sam2).sum())
  area_ratio_vs_sam2 = float(refined_area / sam2_area)
  removed_ratio_vs_sam2 = float(removed_pixels_tmp / sam2_area)
  iou_with_sam2 = float(inter_with_sam2 / max(union_with_sam2, 1))
  depth_inlier_ratio_vs_sam2 = float(np.logical_and(depth_inlier, sam2).sum() / sam2_area)

  if bool(getattr(opt, "geo_accept_gate", True)):
    reasons = []
    if area_ratio_vs_sam2 < float(opt.geo_accept_min_area_ratio):
      reasons.append(f"area_ratio={area_ratio_vs_sam2:.3f}<min={float(opt.geo_accept_min_area_ratio):.3f}")
    if iou_with_sam2 < float(opt.geo_accept_min_iou_with_sam2):
      reasons.append(f"iou_sam2={iou_with_sam2:.3f}<min={float(opt.geo_accept_min_iou_with_sam2):.3f}")
    if removed_ratio_vs_sam2 > float(opt.geo_accept_max_removed_ratio):
      reasons.append(f"removed_ratio={removed_ratio_vs_sam2:.3f}>max={float(opt.geo_accept_max_removed_ratio):.3f}")
    if depth_inlier_ratio_vs_sam2 < float(opt.geo_accept_min_depth_inlier_ratio):
      reasons.append(f"depth_inlier_ratio={depth_inlier_ratio_vs_sam2:.3f}<min={float(opt.geo_accept_min_depth_inlier_ratio):.3f}")

    accept_gate_used = "accepted"
    if len(reasons) > 0:
      refined = sam2 & bbox
      fallback_used = "accept_gate_sam2_original"
      accept_gate_used = "rejected"
      accept_gate_reason = ";".join(reasons)

      # Recompute metrics after fallback.
      refined_area = int(refined.sum())
      removed_pixels_tmp = int(np.logical_and(sam2, ~refined).sum())
      added_pixels_tmp = int(np.logical_and(refined, ~sam2).sum())
      union_with_sam2 = int(np.logical_or(refined, sam2).sum())
      inter_with_sam2 = int(np.logical_and(refined, sam2).sum())
      area_ratio_vs_sam2 = float(refined_area / sam2_area)
      removed_ratio_vs_sam2 = float(removed_pixels_tmp / sam2_area)
      iou_with_sam2 = float(inter_with_sam2 / max(union_with_sam2, 1))

  stats = {
    "geo_method": str(method).lower(),
    "geo_depth_tol": float(tol),
    "geo_render_area": int(rm.sum()),
    "geo_render_dilate_area": int(rm_use.sum()),
    "geo_depth_inlier_area": int(depth_inlier.sum()),
    "geo_sam2_area_before": int(sam2.sum()),
    "geo_refined_area": int(refined.sum()),
    "geo_added_pixels": int(np.logical_and(refined, ~sam2).sum()),
    "geo_removed_pixels": int(np.logical_and(sam2, ~refined).sum()),
    "geo_fallback": fallback_used,
    "geo_accept_gate": accept_gate_used,
    "geo_accept_gate_reason": accept_gate_reason,
    "geo_area_ratio_vs_sam2": float(area_ratio_vs_sam2),
    "geo_removed_ratio_vs_sam2": float(removed_ratio_vs_sam2),
    "geo_iou_with_sam2": float(iou_with_sam2),
    "geo_depth_inlier_ratio_vs_sam2": float(depth_inlier_ratio_vs_sam2),
  }
  return refined.astype(bool), stats


def save_geo_mask_vis(path, color, gt_mask, sam2_mask, refined_mask, render_mask=None, bbox_xyxy=None):
  try:
    vis = color.copy()
    if gt_mask is not None:
      overlay = np.zeros_like(vis)
      overlay[np.asarray(gt_mask) > 0] = [0, 255, 0]      # GT: green
      vis = cv2.addWeighted(vis, 0.75, overlay, 0.25, 0)
    if sam2_mask is not None:
      overlay = np.zeros_like(vis)
      overlay[np.asarray(sam2_mask) > 0] = [0, 0, 255]    # SAM2: red
      vis = cv2.addWeighted(vis, 0.75, overlay, 0.25, 0)
    if render_mask is not None:
      overlay = np.zeros_like(vis)
      overlay[np.asarray(render_mask) > 0] = [255, 0, 0]  # Render: blue
      vis = cv2.addWeighted(vis, 0.85, overlay, 0.15, 0)
    if refined_mask is not None:
      contour_img = np.zeros_like(vis)
      contours, _ = cv2.findContours(np.asarray(refined_mask).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)  # refined boundary: white
    if bbox_xyxy is not None:
      x1, y1, x2, y2 = bbox_xyxy.astype(int)
      cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, vis)
  except Exception as e:
    logging.info(f"[VIS] failed to save {path}: {e}")


# ============================================================
# Depth refine configuration helpers.
# ============================================================

def depth_refine_exp_name(dataset_name, mask_source, mode, apply):
  mode = str(mode).lower()
  apply = str(apply).lower()
  mask_source = str(mask_source).lower()
  if mode in ["none", "off", "false", "0"]:
    return f"{dataset_name}_{mask_source}_none"
  return f"{dataset_name}_{mask_source}_{mode}_{apply}"


def write_depth_refine_yml(fp_dir, mode, apply):
  cfg_dir = os.path.join(fp_dir, "config")
  os.makedirs(cfg_dir, exist_ok=True)
  yml_path = os.path.join(cfg_dir, "depth_refine.yml")
  data = {
    "depth_refine": {
      "mode": str(mode).lower(),
      "apply": apply_for_cfg(mode, apply),
      "accept_if_better": True,
      "auto_sweep": True,
      "log": True,
    }
  }
  with open(yml_path, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)
  return yml_path


def set_depth_refine_cfg(est, mode, apply, fp_dir=None):
  mode = str(mode).lower()
  apply = str(apply).lower()
  cfg_apply = apply_for_cfg(mode, apply)
  if fp_dir is None:
    fp_dir = code_dir
  yml_path = write_depth_refine_yml(fp_dir, mode, apply)
  if hasattr(est, "refiner") and hasattr(est.refiner, "cfg"):
    est.refiner.cfg["depth_refine_mode"] = mode
    est.refiner.cfg["depth_refine_apply"] = cfg_apply
    est.refiner.cfg["depth_refine_accept_if_better"] = True
    est.refiner.cfg["depth_refine_auto_sweep"] = True
    est.refiner.cfg["depth_refine_log"] = True
  return yml_path


# ============================================================
# Main worker.
# ============================================================

def run_register_once(reader, est, color, depth, ob_mask, ob_id, video_id, id_str,
                      dataset_name, mask_source, mode, apply, rows_by_exp):
  mode = str(mode).lower()
  apply = str(apply).lower()
  exp_name = depth_refine_exp_name(dataset_name, mask_source, mode, apply)
  exp_dir = os.path.join(opt.debug_dir, exp_name)
  os.makedirs(exp_dir, exist_ok=True)
  est.debug_dir = exp_dir

  set_depth_refine_cfg(est, mode, apply, fp_dir=code_dir)

  logging.info(
    f"[RUN_ONE] exp={exp_name}, mask={mask_source}, mode={mode}, apply={apply}, "
    f"scene={video_id}, im={id_str}, obj={ob_id}, top_k={opt.top_k}, top_flag={opt.top_flag}"
  )

  try:
    torch.cuda.empty_cache()
  except Exception:
    pass

  t0 = time.perf_counter()
  pose = est.register(
    K=reader.K,
    rgb=color,
    depth=depth,
    ob_mask=ob_mask,
    ob_id=ob_id,
    top_k=int(opt.top_k),
    top_flag=bool(opt.top_flag),
  )
  runtime_sec = time.perf_counter() - t0

  rows_by_exp.setdefault(exp_name, [])
  rows_by_exp[exp_name].append(
    pose_to_bop_row(
      scene_id=video_id,
      im_id=int(id_str),
      obj_id=ob_id,
      pose=pose,
      score=1.0,
      runtime_sec=runtime_sec,
      translation_scale=1000.0,
    )
  )

  logging.info(f"[POSE] exp={exp_name}, time={runtime_sec:.4f}s")
  logging.info(f"pose:\n{pose}")
  return pose, runtime_sec, exp_name


def run_pose_estimation_worker(reader, i_frames, est: FoundationPose = None, debug=0, ob_id=None, device='cuda:0',
                               sam2_model=None, dataset_name='lm', rows_by_exp=None, mask_metric_rows=None):
  torch.cuda.set_device(device)
  est.to_device(device)
  est.glctx = dr.RasterizeCudaContext(device=device)

  result = NestDict()
  if rows_by_exp is None:
    rows_by_exp = {}
  if mask_metric_rows is None:
    mask_metric_rows = []

  mask_sources = parse_str_list(opt.mask_sources, default=["gt", "sam2", "sam2_geo"])
  depth_modes = parse_str_list(opt.depth_refine_modes, default=["none", "icp", "ndt", "gicp", "vgicp"])
  if len(depth_modes) == 0:
    depth_modes = ["none"]
  depth_applies = parse_depth_refine_applies(opt.depth_refine_apply, default=["trans_z"])
  mode_apply_pairs = iter_depth_refine_mode_apply(depth_modes, depth_applies)

  for i, i_frame in enumerate(i_frames):
    logging.info(f"{i}/{len(i_frames)}, i_frame:{i_frame}, ob_id:{ob_id}")
    video_id = reader.get_video_id()
    color = reader.get_color(i_frame)
    depth = reader.get_depth(i_frame)
    id_str = reader.id_strs[i_frame]
    H, W = color.shape[:2]

    gt_mask_raw = reader.get_mask(i_frame, ob_id)
    if gt_mask_raw is None:
      logging.info("GT mask not found, skip")
      result[video_id][id_str][ob_id] = np.eye(4)
      return result, rows_by_exp, mask_metric_rows
    gt_mask = np.asarray(gt_mask_raw) > 0

    bbox_xyxy = gt_mask_to_expanded_bbox(gt_mask, expand=opt.bbox_expand)
    if bbox_xyxy is None:
      logging.info("GT-derived bbox not found, skip")
      result[video_id][id_str][ob_id] = np.eye(4)
      return result, rows_by_exp, mask_metric_rows

    bbox_mask = bbox_to_mask(bbox_xyxy, H, W)
    need_sam2 = ("sam2" in mask_sources) or ("sam2_geo" in mask_sources)
    sam2_mask = np.zeros((H, W), dtype=bool)
    sam2_time = 0.0
    sam2_status = "not_requested"
    if need_sam2:
      sam2_mask, sam2_time, sam2_status = sam2_segment_from_bbox(sam2_model, color, bbox_xyxy, imgsz=opt.sam_imgsz)

    common_meta = {
      "dataset": dataset_name,
      "scene_id": int(video_id),
      "im_id": int(id_str),
      "obj_id": int(ob_id),
      "bbox_expand": float(opt.bbox_expand),
      "bbox_x1": float(bbox_xyxy[0]),
      "bbox_y1": float(bbox_xyxy[1]),
      "bbox_x2": float(bbox_xyxy[2]),
      "bbox_y2": float(bbox_xyxy[3]),
      "sam_ckpt": str(opt.sam_ckpt),
      "sam_imgsz": int(opt.sam_imgsz),
      "sam2_time_sec": float(sam2_time),
      "sam2_status": sam2_status,
    }

    bbox_metrics = compute_mask_metrics(bbox_mask, gt_mask)
    bbox_metrics.update(common_meta)
    bbox_metrics["mask_source"] = "bbox_prompt"
    bbox_metrics["geo_method"] = "bbox_only"
    mask_metric_rows.append(bbox_metrics)

    if need_sam2:
      sam2_metrics = compute_mask_metrics(sam2_mask, gt_mask)
      sam2_metrics.update(common_meta)
      sam2_metrics["mask_source"] = "sam2"
      sam2_metrics["geo_method"] = "none_before_geo"
      mask_metric_rows.append(sam2_metrics)

    if opt.save_mask_vis and (len(mask_metric_rows) <= opt.save_mask_vis_limit * 4):
      vis_path = os.path.join(opt.debug_dir, "mask_vis", f"{dataset_name}_s{int(video_id):06d}_i{int(id_str):06d}_o{int(ob_id):06d}_sam2.png")
      save_mask_vis(vis_path, color, gt_mask, bbox_xyxy=bbox_xyxy, sam2_mask=sam2_mask if need_sam2 else None)

    est.gt_pose = reader.get_gt_pose(i_frame, ob_id)

    # 1) GT / bbox baseline mask sources.
    base_mask_table = {
      "gt": gt_mask,
      "bbox": bbox_mask,
    }
    for mask_source in mask_sources:
      if mask_source not in base_mask_table:
        continue
      ob_mask = base_mask_table[mask_source]
      if ob_mask is None or int(np.asarray(ob_mask).sum()) == 0:
        logging.info(f"[MASK] Empty mask_source={mask_source}, skip frame={id_str}, obj={ob_id}")
        continue
      for mode, apply in mode_apply_pairs:
        pose, runtime_sec, exp_name = run_register_once(
          reader, est, color, depth, ob_mask, ob_id, video_id, id_str,
          dataset_name, mask_source, mode, apply, rows_by_exp,
        )
        result[video_id][id_str][ob_id][exp_name] = pose

    # 2) SAM2 mask source and geometry-guided SAM2 mask refinement.
    #    For every depth mode:
    #      SAM2 mask -> FoundationPose pose with mode -> CAD render depth/mask
    #      -> geometry consistency -> refined SAM2 mask -> mask IoU
    #      -> optionally run FoundationPose again with refined mask.
    if need_sam2:
      if sam2_mask is None or int(np.asarray(sam2_mask).sum()) == 0:
        logging.info(f"[SAM2] Empty mask, cannot run sam2/sam2_geo frame={id_str}, obj={ob_id}")
        continue

      for mode, apply in mode_apply_pairs:
        mode = str(mode).lower()
        apply = str(apply).lower()
        sam2_pose = None

        # Always run SAM2 pose once if SAM2 is requested or if geometry mask needs a pose.
        if ("sam2" in mask_sources) or ("sam2_geo" in mask_sources):
          sam2_pose, runtime_sec, exp_name = run_register_once(
            reader, est, color, depth, sam2_mask, ob_id, video_id, id_str,
            dataset_name, "sam2", mode, apply, rows_by_exp,
          )
          result[video_id][id_str][ob_id][exp_name] = sam2_pose

        if "sam2_geo" not in mask_sources:
          continue

        # Render CAD depth/mask from the SAM2+mode pose, then refine segmentation.
        t_geo0 = time.perf_counter()
        try:
          render_depth, render_mask = render_cad_depth_mask(est, reader.K, H, W, sam2_pose)
          geo_mask, geo_stats = geometry_refine_sam2_mask(
            sam2_mask=sam2_mask,
            depth=depth,
            bbox_mask=bbox_mask,
            render_depth=render_depth,
            render_mask=render_mask,
            method=f"{mode}_{apply}" if mode != "none" else "none",
          )
          geo_status = "ok"
        except Exception as e:
          logging.info(f"[SAM2_GEO] failed mode={mode}, frame={id_str}, obj={ob_id}: {e}")
          render_mask = np.zeros_like(sam2_mask, dtype=bool)
          geo_mask = sam2_mask.copy()
          geo_stats = {
            "geo_method": f"{mode}_{apply}" if mode != "none" else "none",
            "geo_depth_tol": -1.0,
            "geo_render_area": 0,
            "geo_render_dilate_area": 0,
            "geo_depth_inlier_area": 0,
            "geo_sam2_area_before": int(sam2_mask.sum()),
            "geo_refined_area": int(geo_mask.sum()),
            "geo_added_pixels": 0,
            "geo_removed_pixels": 0,
            "geo_fallback": "exception",
          }
          geo_status = f"error:{e}"
        geo_time = time.perf_counter() - t_geo0

        geo_metrics = compute_mask_metrics(geo_mask, gt_mask)
        geo_metrics.update(common_meta)
        geo_metrics.update(geo_stats)
        geo_metrics["mask_source"] = f"sam2_geo_{mode}" if mode == "none" else f"sam2_geo_{mode}_{apply}"
        geo_metrics["geo_mode"] = mode
        geo_metrics["geo_apply"] = apply
        geo_metrics["geo_status"] = geo_status
        geo_metrics["geo_refine_time_sec"] = float(geo_time)
        mask_metric_rows.append(geo_metrics)

        if opt.save_mask_vis and (len(mask_metric_rows) <= opt.save_mask_vis_limit * 4):
          vis_path = os.path.join(opt.debug_dir, "mask_vis", f"{dataset_name}_s{int(video_id):06d}_i{int(id_str):06d}_o{int(ob_id):06d}_geo_{mode}_{apply}.png")
          save_geo_mask_vis(vis_path, color, gt_mask, sam2_mask, geo_mask, render_mask=render_mask, bbox_xyxy=bbox_xyxy)

        if bool(opt.rerun_geo_mask):
          if geo_mask is None or int(np.asarray(geo_mask).sum()) == 0:
            logging.info(f"[SAM2_GEO] refined mask empty, skip FP rerun mode={mode}, frame={id_str}, obj={ob_id}")
            continue
          geo_pose, geo_runtime_sec, geo_exp_name = run_register_once(
            reader, est, color, depth, geo_mask, ob_id, video_id, id_str,
            dataset_name, "sam2_geo", mode, apply, rows_by_exp,
          )
          result[video_id][id_str][ob_id][geo_exp_name] = geo_pose

  return result, rows_by_exp, mask_metric_rows


def summarize_segmentation_metrics(rows, out_dir):
  summary = []
  if not rows:
    return summary
  sources = sorted(set(r.get("mask_source", "") for r in rows))
  for src in sources:
    sub = [r for r in rows if r.get("mask_source") == src]
    if len(sub) == 0:
      continue
    item = {"mask_source": src, "count": len(sub)}
    for k in ["mask_iou", "mask_dice", "mask_precision", "mask_recall", "mask_f1", "mask_area_ratio", "sam2_time_sec", "geo_refine_time_sec", "geo_depth_tol", "geo_added_pixels", "geo_removed_pixels", "geo_depth_inlier_area", "geo_refined_area", "geo_area_ratio_vs_sam2", "geo_removed_ratio_vs_sam2", "geo_iou_with_sam2", "geo_depth_inlier_ratio_vs_sam2"]:
      vals = []
      for r in sub:
        try:
          vals.append(float(r.get(k, 0.0)))
        except Exception:
          pass
      vals = np.asarray(vals, dtype=np.float64)
      vals = vals[np.isfinite(vals)]
      if len(vals) == 0:
        continue
      item[f"{k}_mean"] = float(np.mean(vals))
      item[f"{k}_median"] = float(np.median(vals))
      item[f"{k}_p25"] = float(np.quantile(vals, 0.25))
      item[f"{k}_p75"] = float(np.quantile(vals, 0.75))
    summary.append(item)
  write_dict_rows_csv(os.path.join(out_dir, "segmentation_metrics_summary.csv"), summary)
  return summary


def write_experiment_metadata(out_dir, dataset_name, mask_sources, depth_modes, depth_applies):
  rows = []
  mode_apply_pairs = iter_depth_refine_mode_apply(depth_modes, depth_applies)
  for mask_source in mask_sources:
    for mode, apply in mode_apply_pairs:
      exp_name = depth_refine_exp_name(dataset_name, mask_source, mode, apply)
      rows.append({
        "experiment": exp_name,
        "dataset": dataset_name,
        "mask_source": mask_source,
        "depth_refine_mode": mode,
        "depth_refine_apply": apply,
        "sam_ckpt": opt.sam_ckpt,
        "sam_imgsz": opt.sam_imgsz,
        "bbox_expand": opt.bbox_expand,
        "top_k": opt.top_k,
        "top_flag": opt.top_flag,
        "rerun_geo_mask": opt.rerun_geo_mask,
        "geo_depth_tol": opt.geo_depth_tol,
        "geo_depth_tol_min": opt.geo_depth_tol_min,
        "geo_depth_tol_max": opt.geo_depth_tol_max,
        "geo_depth_tol_ratio": opt.geo_depth_tol_ratio,
        "geo_add_render_inliers": opt.geo_add_render_inliers,
        "geo_keep_largest_cc": opt.geo_keep_largest_cc,
        "geo_accept_gate": opt.geo_accept_gate,
        "geo_accept_min_area_ratio": opt.geo_accept_min_area_ratio,
        "geo_accept_min_iou_with_sam2": opt.geo_accept_min_iou_with_sam2,
        "geo_accept_max_removed_ratio": opt.geo_accept_max_removed_ratio,
        "geo_accept_min_depth_inlier_ratio": opt.geo_accept_min_depth_inlier_ratio,
        "bop_csv": os.path.join(exp_name, f"foundationpose_{dataset_name}-test.csv"),
      })
  write_dict_rows_csv(os.path.join(out_dir, "experiment_metadata.csv"), rows)


def run_pose_estimation():
  wp.force_load(device='cuda')

  opt.linemod_dir = os.path.abspath(opt.linemod_dir)
  bop_root = os.path.abspath(os.path.join(opt.linemod_dir, '..'))
  os.environ.setdefault('BOP_DIR', bop_root)

  test_root = f'{opt.linemod_dir}/test'
  if not os.path.isdir(test_root):
    raise FileNotFoundError(f"Cannot find test root: {test_root}")

  debug = opt.debug
  debug_dir = opt.debug_dir
  os.makedirs(debug_dir, exist_ok=True)

  dataset_name = infer_bop_dataset_name(opt.linemod_dir)
  ReaderClass = get_bop_reader_class(dataset_name)
  logging.info(f"[INFO] dataset={dataset_name}, reader={ReaderClass.__name__}")

  reader_tmp = make_reader(ReaderClass, f'{test_root}/000002')

  available_obj_ids = sorted([int(x) for x in reader_tmp.ob_ids])
  requested_obj_ids = parse_obj_ids_arg(opt.obj_ids)
  if requested_obj_ids is None:
    run_obj_ids = available_obj_ids
  else:
    run_obj_ids = [oid for oid in available_obj_ids if oid in requested_obj_ids]
    missing = [oid for oid in requested_obj_ids if oid not in available_obj_ids]
    if len(missing) > 0:
      logging.info(f"[OBJ_FILTER] requested obj_ids not available in dataset={dataset_name}: {missing}")
    if len(run_obj_ids) == 0:
      raise RuntimeError(
        f"[OBJ_FILTER] No requested obj_ids exist in dataset={dataset_name}. "
        f"requested={requested_obj_ids}, available={available_obj_ids}. "
        f"Note: LMO object ids are usually [1,5,6,8,9,10,11,12], not 2/3."
      )

  mask_sources = parse_str_list(opt.mask_sources, default=["gt", "sam2", "sam2_geo"])
  depth_modes = parse_str_list(opt.depth_refine_modes, default=["none", "icp", "ndt", "gicp", "vgicp"])
  depth_applies = parse_depth_refine_applies(opt.depth_refine_apply, default=["trans_z"])
  mode_apply_pairs = iter_depth_refine_mode_apply(depth_modes, depth_applies)
  valid_mask_sources = {"gt", "sam2", "bbox", "sam2_geo"}
  for ms in mask_sources:
    if ms not in valid_mask_sources:
      raise ValueError(f"Unknown mask_source={ms}. Use gt,sam2,bbox,sam2_geo")

  logging.info(f"[OBJ_FILTER] available_obj_ids={available_obj_ids}")
  logging.info(f"[OBJ_FILTER] requested_obj_ids={requested_obj_ids if requested_obj_ids is not None else 'ALL'}")
  logging.info(f"[OBJ_FILTER] run_obj_ids={run_obj_ids}")
  logging.info(f"[ABLATION] mask_sources={mask_sources}")
  logging.info(f"[ABLATION] depth_refine_modes={depth_modes}, applies={depth_applies}")
  logging.info(f"[ABLATION] mode_apply_pairs={mode_apply_pairs}")
  logging.info(f"[SAM2] ckpt={opt.sam_ckpt}, imgsz={opt.sam_imgsz}, bbox_expand={opt.bbox_expand}")

  sam2_model = None
  if ("sam2" in mask_sources) or ("sam2_geo" in mask_sources):
    sam2_model = load_sam2_model(opt.sam_ckpt, imgsz=opt.sam_imgsz)

  res = NestDict()
  rows_by_exp = {}
  mask_metric_rows = []
  glctx = dr.RasterizeCudaContext()
  mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()

  est = FoundationPose(
    model_pts=mesh_tmp.vertices.copy(),
    model_normals=mesh_tmp.vertex_normals.copy(),
    symmetry_tfs=None,
    mesh=mesh_tmp,
    scorer=None,
    refiner=None,
    glctx=glctx,
    debug_dir=debug_dir,
    debug=debug,
  )

  for ob_id in run_obj_ids:
    ob_id = int(ob_id)

    try:
      mesh = reader_tmp.get_gt_mesh(ob_id)
      symmetry_tfs = reader_tmp.symmetry_tfs[ob_id]
    except Exception as e:
      logging.info(f"[SKIP] ob_id={ob_id}: cannot load model-based mesh/symmetry. error={e}")
      continue

    if dataset_name == 'lmo':
      video_dir = f'{test_root}/000002'
    else:
      candidate_video_dir = f'{test_root}/{ob_id:06d}'
      if os.path.isdir(candidate_video_dir):
        video_dir = candidate_video_dir
      else:
        video_dir = f'{test_root}/000002'

    reader = make_reader(ReaderClass, video_dir)
    video_id = reader.get_video_id()

    est.reset_object(
      model_pts=mesh.vertices.copy(),
      model_normals=mesh.vertex_normals.copy(),
      symmetry_tfs=symmetry_tfs,
      mesh=mesh,
    )

    args = []
    for i in range(len(reader.color_files)):
      instance_ids = reader.get_instance_ids_in_image(i)
      if ob_id not in instance_ids:
        continue
      args.append((reader, [i], est, debug, ob_id, "cuda:0"))
      if opt.max_frames_per_obj > 0 and len(args) >= opt.max_frames_per_obj:
        logging.info(f"[FRAME_FILTER] ob_id={ob_id}: limit to max_frames_per_obj={opt.max_frames_per_obj}")
        break

    logging.info(
      f"[RUN] ob_id={ob_id}, video_dir={video_dir}, video_id={video_id}, "
      f"frames_with_obj={len(args)}"
    )

    outs = []
    for arg in args:
      out, rows_by_exp, mask_metric_rows = run_pose_estimation_worker(
        *arg,
        sam2_model=sam2_model,
        dataset_name=dataset_name,
        rows_by_exp=rows_by_exp,
        mask_metric_rows=mask_metric_rows,
      )
      outs.append(out)

    for out in outs:
      for video_id in out:
        for id_str in out[video_id]:
          for _ob_id in out[video_id][id_str]:
            res[video_id][id_str][_ob_id] = out[video_id][id_str][_ob_id]

  with open(f'{opt.debug_dir}/linemod_res.yml', 'w') as ff:
    yaml.safe_dump(make_yaml_dumpable(res), ff)

  # Write one BOP CSV per experiment folder, compatible with eval_bop_refine.sh.
  for exp_name, rows in sorted(rows_by_exp.items()):
    exp_dir = os.path.join(opt.debug_dir, exp_name)
    csv_name = opt.bop_result_name if opt.bop_result_name else f"foundationpose_{dataset_name}-test.csv"
    csv_path = os.path.join(exp_dir, csv_name)
    write_bop_results_csv(csv_path, rows, time_mode=opt.bop_time_mode)

  write_dict_rows_csv(os.path.join(opt.debug_dir, "segmentation_metrics.csv"), mask_metric_rows)
  summarize_segmentation_metrics(mask_metric_rows, opt.debug_dir)
  write_experiment_metadata(opt.debug_dir, dataset_name, mask_sources, depth_modes, depth_applies)

  logging.info(f"[DONE] result saved to {opt.debug_dir}/linemod_res.yml")
  logging.info(f"[DONE] experiments saved under {opt.debug_dir}")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--linemod_dir', type=str, default="/home/user/FoundationPose/demo_data/bop/lmo", help="BOP LM or LMO root dir, e.g. /home/user/FoundationPose/demo_data/bop/lm")
  parser.add_argument('--debug', type=int, default=0)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_sam2_2')
  parser.add_argument('--bop_result_name', type=str, default='', help='Output BOP CSV filename. Default: foundationpose_{dataset}-test.csv')
  parser.add_argument('--bop_time_mode', type=str, default='sum', choices=['sum', 'max', 'zero'], help='Make BOP time consistent per image: sum/max/zero')
  parser.add_argument('--obj_ids', type=str, default='', help='Comma-separated object ids to run, e.g. "1,5". Empty or "all" means all objects.')
  parser.add_argument('--max_frames_per_obj', type=int, default=5, help='Limit number of frames per object for quick tests. 0 means all frames.')

  # New ablation controls.
  parser.add_argument('--mask_sources', type=str, default='gt,sam2,sam2_geo', help='Comma-separated mask sources to run FoundationPose with: gt,sam2,bbox,sam2_geo')
  parser.add_argument('--depth_refine_modes', type=str, default='none,icp,ndt,gicp,vgicp', help='Comma-separated depth refine modes: none,icp,ndt,gicp,vgicp')
  parser.add_argument('--depth_refine_apply', type=str, default='trans,trans_z', help='Comma-separated apply modes: trans,trans_z,se3, or all. Example: --depth_refine_apply all')
  parser.add_argument('--top_k', type=int, default=50, help='FoundationPose register top_k.')
  parser.add_argument('--top_flag', action='store_true', default=True, help='Use top-k filtering before RefineNet/scorer if estimater.py supports it.')
  parser.add_argument('--no_top_flag', dest='top_flag', action='store_false', help='Disable top_flag.')

  # GT bbox -> SAM2 prompt controls.
  parser.add_argument('--bbox_expand', type=float, default=0.10, help='Expansion ratio around GT-derived bbox before feeding SAM2.')
  parser.add_argument('--sam_ckpt', type=str, default='sam2.1_l.pt', help='Ultralytics SAM2 checkpoint, e.g. sam2.1_l.pt or sam2_t.pt')
  parser.add_argument('--sam_imgsz', type=int, default=640, help='Ultralytics SAM/SAM2 inference size.')
  parser.add_argument('--save_mask_vis', action='store_true', default=True, help='Save GT/SAM2/bbox overlay images.')
  parser.add_argument('--save_mask_vis_limit', type=int, default=30, help='Maximum number of frames for mask visualization.')
  # Geometry-guided SAM2 mask refinement controls.
  parser.add_argument('--rerun_geo_mask', action='store_true', default=True, help='After geometry-guided mask refinement, run FoundationPose again with the refined mask.')
  parser.add_argument('--no_rerun_geo_mask', dest='rerun_geo_mask', action='store_false', help='Only compute refined mask IoU; do not run FoundationPose with refined mask.')
  parser.add_argument('--geo_depth_tol', type=float, default=-1.0, help='Fixed observed-vs-rendered depth tolerance in meters. <=0 means adaptive.')
  parser.add_argument('--geo_depth_tol_min', type=float, default=0.008, help='Minimum adaptive depth tolerance in meters.')
  parser.add_argument('--geo_depth_tol_max', type=float, default=0.060, help='Maximum adaptive depth tolerance in meters.')
  parser.add_argument('--geo_depth_tol_ratio', type=float, default=0.020, help='Adaptive tolerance ratio relative to median rendered depth.')
  parser.add_argument('--geo_render_dilate_iter', type=int, default=1, help='Dilate rendered CAD mask before geometry filtering.')
  parser.add_argument('--geo_sam_dilate_iter', type=int, default=0, help='Optionally dilate SAM2 mask support before refinement bookkeeping.')
  parser.add_argument('--geo_open_iter', type=int, default=0, help='Morphological opening iterations after geometry refinement.')
  parser.add_argument('--geo_close_iter', type=int, default=1, help='Morphological closing iterations after geometry refinement.')
  parser.add_argument('--geo_add_render_inliers', action='store_true', default=True, help='Add rendered CAD pixels whose depth agrees with observed depth.')
  parser.add_argument('--no_geo_add_render_inliers', dest='geo_add_render_inliers', action='store_false', help='Only remove inconsistent SAM2 pixels; do not add rendered inliers.')
  parser.add_argument('--geo_keep_largest_cc', action='store_true', default=True, help='Keep only the largest connected component after geometry refinement.')
  parser.add_argument('--no_geo_keep_largest_cc', dest='geo_keep_largest_cc', action='store_false', help='Do not keep largest connected component.')
  parser.add_argument('--geo_min_area_ratio', type=float, default=0.15, help='Fallback if refined mask area is smaller than this ratio of original SAM2 area.')

  # Conservative acceptance gate to prevent over-deletion. These checks do not use GT.
  parser.add_argument('--geo_accept_gate', action='store_true', default=True, help='Reject geometry-refined mask and fallback to SAM2 if it deletes too much.')
  parser.add_argument('--no_geo_accept_gate', dest='geo_accept_gate', action='store_false', help='Disable refined-mask acceptance gate.')
  parser.add_argument('--geo_accept_min_area_ratio', type=float, default=0.50, help='Reject if refined_area / sam2_area is below this value.')
  parser.add_argument('--geo_accept_min_iou_with_sam2', type=float, default=0.45, help='Reject if IoU(refined, original SAM2) is below this value.')
  parser.add_argument('--geo_accept_max_removed_ratio', type=float, default=0.55, help='Reject if removed_pixels / sam2_area is above this value.')
  parser.add_argument('--geo_accept_min_depth_inlier_ratio', type=float, default=0.03, help='Reject if depth-inlier pixels inside original SAM2 are too few.')

  opt = parser.parse_args()
  set_seed(0)

  run_pose_estimation()
