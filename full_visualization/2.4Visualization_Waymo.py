
import os
import sys
import numpy as np
import open3d as o3d
import pickle as pkl 
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())
import json
from tools.kitti360Scripts.helpers.labels import name2label
from utils.o3d_utils import vis_world_bounds_o3d, vis_layout_o3d, vis_voxel_world_o3d, vis_camera_o3d, costum_visualizer_o3d
from utils.data_utils import convert_legacy_kitti360_layout, convert_legacy_kitti360_stuff
from utils.transform_utils import  create_c2w, create_R

from tools.Occ3D.vistool import *

if __name__ == '__main__':
    data_root = 'data/nuscene'
    scene_id = 61
    metadata_path = os.path.join(data_root, 'annotations.json')
    with open(metadata_path, 'rb') as fp:
        metadata = json.load(fp)

    a = np.load('data/waymo/797/000.npz')
    b = np.load('data/waymo/797/000_04.npz')

    print('Done')