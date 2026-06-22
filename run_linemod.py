# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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
from datareader import *
from estimater import *

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

      # 以 scene_gt.json 的 key 為主，只保留有 GT 的影像
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
      # BOP 標準格式：/.../bop/lm/models/obj_000001.ply
      mesh_file = f'{root}/models/obj_{ob_id:06d}.ply'
      if os.path.exists(mesh_file):
        return mesh_file

      # FoundationPose 舊格式：/.../LINEMOD/lm_models/models/obj_000001.ply
      mesh_file = f'{root}/lm_models/models/obj_{ob_id:06d}.ply'
      if os.path.exists(mesh_file):
        return mesh_file

      parent = os.path.abspath(f'{root}/../')
      if parent == root:
        raise FileNotFoundError(
          f"Cannot find model-based mesh obj_{ob_id:06d}.ply from base_dir={self.base_dir}"
        )
      root = parent

  # patch classes imported from datareader
  BopBaseReader.__init__ = _patched_bop_init
  LinemodReader.get_gt_mesh_file = _patched_linemod_get_gt_mesh_file
  if 'LinemodOcclusionReader' in globals():
    LinemodOcclusionReader.get_gt_mesh_file = _patched_linemod_get_gt_mesh_file


patch_bop_reader_filter_by_scene_gt()


def infer_bop_dataset_name(dataset_root):
  """Infer BOP dataset name from root folder, e.g. /.../bop/lm -> lm."""
  return os.path.basename(os.path.abspath(dataset_root)).lower()


def get_bop_reader_class(dataset_name):
  """Use LMO-specific reader for lmo; LinemodReader otherwise."""
  dataset_name = str(dataset_name).lower()
  if dataset_name == 'lmo' and 'LinemodOcclusionReader' in globals():
    return LinemodOcclusionReader
  return LinemodReader




def parse_obj_ids_arg(obj_ids_arg):
  """
  Parse comma-separated object ids from CLI.

  Examples:
    --obj_ids 2,3
    --obj_ids "1 5 6"
    --obj_ids ""      -> run all objects
  """
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


def make_reader(reader_cls, video_dir):
  """
  Create BOP reader safely.

  LinemodReader supports split=None, while LinemodOcclusionReader does not.
  This wrapper keeps LM and LMO compatible with the same run script.
  """
  try:
    return reader_cls(video_dir, split=None)
  except TypeError as e:
    if 'split' in str(e):
      return reader_cls(video_dir)
    raise


def pose_to_bop_row(scene_id, im_id, obj_id, pose, score=1.0, runtime_sec=0.0, translation_scale=1000.0):
  """
  Convert 4x4 FoundationPose pose to one BOP19 CSV row.

  BOP19 format:
    scene_id, im_id, obj_id, score, R, t, time

  Notes:
    - score is confidence, not MSSD/MSPD/VSD/AR. For GT-mask localization, 1.0 is OK.
    - FoundationPose translation is assumed to be meter; BOP uses millimeter.
  """
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
  """
  BOP toolkit requires all estimates from the same image to have identical time.
  For multi-object scenes such as LMO, multiple obj_id may share the same
  (scene_id, im_id). We first record per-estimate runtime, then convert it to
  per-image runtime.

  mode:
    sum  : image time = sum of estimate runtimes in that image.
    max  : image time = max estimate runtime in that image.
    zero : image time = 0.0, useful if only evaluating pose accuracy.
  """
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


def get_mask(reader, i_frame, ob_id, detect_type):
  if detect_type == 'box':
    mask = reader.get_mask(i_frame, ob_id)
    if mask is None:
      return None
    H, W = mask.shape[:2]
    vs, us = np.where(mask > 0)
    if len(vs) == 0 or len(us) == 0:
      return None
    umin = us.min()
    umax = us.max()
    vmin = vs.min()
    vmax = vs.max()
    valid = np.zeros((H, W), dtype=bool)
    valid[vmin:vmax, umin:umax] = 1

  elif detect_type == 'mask':
    mask = reader.get_mask(i_frame, ob_id)
    if mask is None:
      return None
    valid = mask > 0

  elif detect_type == 'detected':
    mask = cv2.imread(reader.color_files[i_frame].replace('rgb', 'mask_cosypose'), -1)
    valid = mask == ob_id

  else:
    raise RuntimeError

  return valid


def run_pose_estimation_worker(reader, i_frames, est: FoundationPose = None, debug=0, ob_id=None, device='cuda:0'):
  torch.cuda.set_device(device)
  est.to_device(device)
  est.glctx = dr.RasterizeCudaContext(device=device)

  result = NestDict()
  bop_rows = []

  for i, i_frame in enumerate(i_frames):
    logging.info(f"{i}/{len(i_frames)}, i_frame:{i_frame}, ob_id:{ob_id}")
    video_id = reader.get_video_id()
    color = reader.get_color(i_frame)
    depth = reader.get_depth(i_frame)
    id_str = reader.id_strs[i_frame]

    debug_dir = est.debug_dir

    ob_mask = get_mask(reader, i_frame, ob_id, detect_type=detect_type)
    if ob_mask is None:
      logging.info("ob_mask not found, skip")
      result[video_id][id_str][ob_id] = np.eye(4)
      return result, bop_rows

    est.gt_pose = reader.get_gt_pose(i_frame, ob_id)

    t0 = time.perf_counter()
    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, ob_id=ob_id, top_k=100, top_flag=True)
    runtime_sec = time.perf_counter() - t0

    # BOP CSV 的 score 是 confidence，不是 MSSD/MSPD/VSD/AR。
    # 這裡使用 GT mask / GT instance localization，因此固定 1.0 即可。
    score = 1.0

    bop_rows.append(
      pose_to_bop_row(
        scene_id=video_id,
        im_id=int(id_str),
        obj_id=ob_id,
        pose=pose,
        score=score,
        runtime_sec=runtime_sec,
        translation_scale=1000.0
      )
    )

    logging.info(f"score:{score:.3f}, time:{runtime_sec:.4f}s")
    logging.info(f"pose:\n{pose}")

    if debug >= 3:
      m = est.mesh_ori.copy()
      tmp = m.copy()
      tmp.apply_transform(pose)
      tmp.export(f'{debug_dir}/model_tf.obj')

    result[video_id][id_str][ob_id] = pose

  return result, bop_rows


def run_pose_estimation():
  wp.force_load(device='cuda')

  # BOP root：
  #   LM : /home/user/FoundationPose/demo_data/bop/lm
  #   LMO: /home/user/FoundationPose/demo_data/bop/lmo
  opt.linemod_dir = os.path.abspath(opt.linemod_dir)

  # 給 datareader 內部使用的 BOP_DIR
  # 若使用 /.../bop/lm，則 BOP_DIR=/.../bop
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

  # 先用 test/000002 建立 reader_tmp，LM/LMO 都有 000002。
  # LMO 必須使用 LinemodOcclusionReader，否則 LinemodReader 會嘗試讀 ob_id=2，
  # 但 lmo/models_info.json 沒有 obj_000002，會造成 KeyError: '2'。
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

  logging.info(f"[OBJ_FILTER] available_obj_ids={available_obj_ids}")
  logging.info(f"[OBJ_FILTER] requested_obj_ids={requested_obj_ids if requested_obj_ids is not None else 'ALL'}")
  logging.info(f"[OBJ_FILTER] run_obj_ids={run_obj_ids}")

  res = NestDict()
  all_bop_rows = []
  glctx = dr.RasterizeCudaContext()
  mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()

  est = FoundationPose(model_pts=mesh_tmp.vertices.copy(),model_normals=mesh_tmp.vertex_normals.copy(),symmetry_tfs=None,mesh=mesh_tmp,scorer=None,refiner=None,glctx=glctx,debug_dir=debug_dir,debug=debug)

  # ============================================================
  # Model-based only:
  # 強制使用 BOP models/obj_XXXXXX.ply，不使用 reconstructed mesh。
  # ============================================================
  for ob_id in run_obj_ids:
    ob_id = int(ob_id)

    try:
      mesh = reader_tmp.get_gt_mesh(ob_id)
      symmetry_tfs = reader_tmp.symmetry_tfs[ob_id]
    except Exception as e:
      logging.info(f"[SKIP] ob_id={ob_id}: cannot load model-based mesh/symmetry. error={e}")
      continue

    # LM: test/000001 ~ test/000015，每個物體通常有自己的 scene folder。
    # LMO: test/000002 單一遮擋 scene，所有物體都在同一個 scene。
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
      mesh=mesh
    )

    args = []
    for i in range(len(reader.color_files)):
      # 若該 frame 沒有此 ob_id，直接略過，避免 get_mask 找不到
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
      out, bop_rows = run_pose_estimation_worker(*arg)
      outs.append(out)
      all_bop_rows.extend(bop_rows)

    for out in outs:
      for video_id in out:
        for id_str in out[video_id]:
          for _ob_id in out[video_id][id_str]:
            res[video_id][id_str][_ob_id] = out[video_id][id_str][_ob_id]

  with open(f'{opt.debug_dir}/linemod_res.yml', 'w') as ff:
    yaml.safe_dump(make_yaml_dumpable(res), ff)

  csv_name = opt.bop_result_name if opt.bop_result_name else f"foundationpose_{dataset_name}-test.csv"
  csv_path = os.path.join(opt.debug_dir, csv_name)
  write_bop_results_csv(csv_path, all_bop_rows, time_mode=opt.bop_time_mode)

  logging.info(f"[DONE] result saved to {opt.debug_dir}/linemod_res.yml")
  logging.info(f"[DONE] BOP CSV saved to {csv_path}")

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--linemod_dir',type=str,default="/home/user/FoundationPose/demo_data/bop/lm",help="BOP LM or LMO root dir, e.g. /home/user/FoundationPose/demo_data/bop/lm")
  parser.add_argument('--debug', type=int, default=0)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--bop_result_name', type=str, default='', help='Output BOP CSV filename. Default: foundationpose_{dataset}-test.csv')
  parser.add_argument('--bop_time_mode', type=str, default='sum', choices=['sum', 'max', 'zero'], help='Make BOP time consistent per image: sum/max/zero')
  parser.add_argument('--obj_ids', type=str, default='', help='Comma-separated object ids to run, e.g. "2,3". Empty or "all" means all objects.')
  parser.add_argument('--max_frames_per_obj', type=int, default=0, help='Limit number of frames per object for quick tests. 0 means all frames.')
  opt = parser.parse_args()
  set_seed(0)

  detect_type = 'mask'   # mask / box / detected

  run_pose_estimation()
  