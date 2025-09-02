import trimesh
import numpy as np
import os
import argparse
parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))
parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/models/cube/cube.obj')
args = parser.parse_args()
mesh = trimesh.load(args.mesh_file)
verts = np.asarray(mesh.vertices)

print("頂點數:", len(verts))
print("最小值:", verts.min(axis=0))
print("最大值:", verts.max(axis=0))
print("質心 centroid:", verts.mean(axis=0))
print("原點是否在包圍盒內:", np.all(verts.min(axis=0) <= 0) and np.all(verts.max(axis=0) >= 0))
