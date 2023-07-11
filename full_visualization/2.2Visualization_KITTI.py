
import os
import sys
import numpy as np
import open3d as o3d
import pickle as pkl 
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())

from tools.kitti360Scripts.helpers.labels import name2label
from utils.o3d_utils import vis_world_bounds_o3d, vis_layout_o3d, vis_voxel_world_o3d, vis_camera_o3d, costum_visualizer_o3d
from utils.data_utils import convert_legacy_kitti360_layout, convert_legacy_kitti360_stuff
from utils.transform_utils import  create_c2w, create_R

# KITTI360_Stuff = {
#     # Major
#     'vegetation',
#     'terrain',
#     'ground',
#     'road', 
#     'sidewalk',
#     'parking',
#     'rail track',
#     'building',
#     'gate',
#     'garage', 
#     'bridge',
#     'tunnel',
#     # Vehicle
#     'wall',
#     'car',
#     'truck',
#     'train','caravan',
#     'bus',
#     # Minor
#     'trailer',
#     'fence',
#     'guard rail',
#     'trash bin',
#     'box',
#     'lamp',
#     'smallpole',
#     'polegroup',
#     'stop',
#     'pole',
#     'traffic sign',
#     'traffic light'
# }

KITTI_learning_map_inv={
# 0: "unlabeled", 
   1: "car",
  2: "bicycle",
  3:  "motorcycle",
  4: "truck",
  5:  "other-vehicle",
   6: "person",
  7: "bicyclist",
  8: "motorcyclist",
  9:  "road",
  10: "parking",
  11: "sidewalk",
  12: "ground",
  13: "building",
  14: "fence",
  15: "vegetation",
   16:  "trunk",
  17: "terrain",
   18:"pole",
   19:"traffic-sign"
}

KITTI_learning_map = {KITTI_learning_map_inv[k] : k for k in KITTI_learning_map_inv }
KITTI_Stuff = {KITTI_learning_map_inv[k] for k in KITTI_learning_map_inv if (KITTI_learning_map_inv[k] in name2label)}
# name2semantic_kitti = {id2name_semantic_kitti : k for k in id2name_semantic_kitti }


scene_size = np.array((51.2, 51.2, 6.4))
vox_size = np.array((0.2, 0.2,0.2))
vox_origin = np.array([0, -25.6, -2])
world_bounds = np.array([64, 0, 32, -32, 14, -2])
H, W = 370,1220

K = np.array([[707.0912,   0.    , 601.8873],
                 [  0.    , 707.0912, 183.1104],
                 [0., 0., 1.]])

# For open3d follow opencv camera coordiante

w2c = np.array([[-0.00185774, -0.99996597, -0.00803998, -0.00478403],
               [-0.00648147,  0.00805186, -0.9999466 , -0.07337429],
               [ 0.9999773 , -0.00180553, -0.0064962 , -0.3339968 ],
               [0., 0., 0., 1.]])
c2w = np.array(w2c)


def convert_ssc_kitti_stuff(semantic_voxel_path, scene_size, voxel_size,voxel_origin ,semantic_list):
    '''
    Load old version seamntic voxel
    '''
    with open(semantic_voxel_path, 'rb') as fp:
        # a = pkl.load(fp)
        stuff_semantic= pkl.load(fp)['y_pred']
    stuff_semantic = stuff_semantic.transpose(1,0,2)

    point_num = (scene_size / voxel_size).astype(int) # X, Y, Z
    H, W, D = 32, 256, 256
    H_o, W_o, D_o = point_num[2], point_num[1], point_num[0]
    t = np.zeros_like( stuff_semantic).astype(np.uint8)

    for i in KITTI_Stuff:
        t[stuff_semantic == KITTI_learning_map[i]] = name2label[i].id

    import torch.nn.functional as F
    import torch
    semantic_voxel = F.interpolate(torch.tensor(t, dtype=torch.uint8)[None,None,:,:,:], (W_o, D_o, H_o), mode = 'nearest')[0,0].numpy()

    vertices_gridx,  vertices_gridy, vertices_gridz= np.meshgrid(np.linspace(0, scene_size[0], point_num[0]), np.linspace(0, scene_size[1], point_num[1]), np.linspace(0, scene_size[2], point_num[2]), indexing='xy') 
    loc_voxel = np.concatenate((vertices_gridx[:,:,:,None], vertices_gridy[:,:,:,None], vertices_gridz[:,:,:,None]), axis = -1)
    loc_voxel += voxel_origin[None,None,None]

    stuff =  {
            'semantic_grid' : semantic_voxel, 
            'loc_grid' : loc_voxel,
            'scene_size' : scene_size,
            'voxel_size' : voxel_size,
            'voxel_origin' : voxel_origin,
            'index_order' : 'YXZ',
            'semantic_color' : {n : np.array(name2label[n].color) / 255. for n in semantic_list},
            'semantic_id' : {n : name2label[n].id for n in semantic_list},
            'semantic_name' : semantic_list,
        }
    return stuff



if __name__ == '__main__':   
    # Congig of semantic kitti scenes(from monoScene)
    save_panoptic_prior = False
    vis_3d = True
    data_root = 'data/semantic-kitti'
    frame_id = 0
    legacy_layout_path = os.path.join(data_root, 'layout', '2013_05_28_drive_0000_sync', '%010d.pkl'%frame_id)
    legacy_semantic_voxel_path = os.path.join(data_root, 'voxel', '%06d.pkl'%frame_id)
    image_path = os.path.join(data_root, 'image_2', '%06d.png'%frame_id)

    stuff_world = convert_ssc_kitti_stuff(legacy_semantic_voxel_path, voxel_origin= vox_origin,
    voxel_size= vox_size, 
    scene_size= scene_size,
    semantic_list= KITTI_Stuff) # H W D 
    # object_layout = convert_ssc_kitti_layout(legacy_layout_path)

    geo_group = []
    geo_group += vis_world_bounds_o3d(bounds=world_bounds)
    coordinate_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    coordinate_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1).transform(c2w)
    geo_group += [coordinate_world, coordinate_cam]
    geo_group += vis_voxel_world_o3d(stuff_world,voxelization=True)
    # geo_group += vis_layout_o3d(object_layout)
    geo_group += vis_camera_o3d(instrinsic=K, extrinsic=w2c, z_revrse=True)
    # o3d.visualization.draw_plotly(geo_group)
    vis = costum_visualizer_o3d(geo_group=geo_group, set_camera=True, instrinsic=K, extrinsic=w2c, visible = vis_3d)
    project_img = np.asarray(vis.capture_screen_float_buffer())

    if vis_3d:
        vis.run()
        vis.destroy_window()
    H_show , W_show = min(project_img.shape[0], H), min(project_img.shape[1], W)
    rgb = plt.imread(image_path)
    plt.imshow(project_img[:H_show,:W_show] +rgb[:H_show,:W_show])
    plt.show()

    panoptic_prior_dir = semantic_voxel_path = os.path.join(data_root, 'panoptic_prior', '2013_05_28_drive_0000_sync')

    # if save_panoptic_prior:
        # os.makedirs(panoptic_prior_dir, exist_ok=True)
        # panoptic_prior_path = os.path.join(panoptic_prior_dir, '%010d.pkl'%frame_id)
        # del stuff_world['loc_grid']
        # panoptic_prior = {
        #         'c2w_openGL' : c2w_gl,
        #         'c2w_openCV' : c2w,
        #         'K' : K,
        #         'HW': (H, W), 
        #         'stuff_world': stuff_world,
        #         'object_layout': object_layout,
        #         }

        # with open(panoptic_prior_path, 'wb+') as fp:
        #     pkl.dump(panoptic_prior, fp)
        # os.makedirs(panoptic_prior_dir, exist_ok=True)
        # panoptic_prior_path = os.path.join(panoptic_prior_dir, '%010d.pkl'%frame_id)
        # del stuff_world['loc_grid']
        # panoptic_prior = {
        #         'c2w_openGL' : c2w_gl,
        #         'c2w_openCV' : c2w,
        #         'K' : K,
        #         'HW': (H, W), 
        #         'stuff_world': stuff_world,
        #         'object_layout': object_layout,
        #         }

        # with open(panoptic_prior_path, 'wb+') as fp:
        #     pkl.dump(panoptic_prior, fp)