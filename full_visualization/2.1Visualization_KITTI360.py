
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

KITTI360_Stuff = {
    # Major
    'vegetation',
    'terrain',
    'ground',
    'road', 
    'sidewalk',
    'parking',
    'rail track',
    'building',
    'gate',
    'garage', 
    'bridge',
    'tunnel',
    # Vehicle
    'wall',
    'car',
    'truck',
    'train','caravan',
    'bus',
    # Minor
    'trailer',
    'fence',
    'guard rail',
    'trash bin',
    'box',
    'lamp',
    'smallpole',
    'polegroup',
    'stop',
    'pole',
    'traffic sign',
    'traffic light'
}

scene_size = np.array((64., 64, 16))
vox_size = np.array((0.25, 0.25,0.25))
vox_origin = np.array([0, -32, -2])
world_bounds = np.array([64, 0, 32, -32, 14, -2])
H, W = 376, 1408

K = np.array([[552.554261,   0.      , 682.049453],
                 [  0.      , 552.554261, 238.769549],
                 [0., 0., 1.]])
cam_T = np.array((0,0,1.55))
cam_R = create_R((0,np.deg2rad(5),0,))
c2w = create_c2w(cam_R_world=cam_R, cam_T_world=cam_T, cam_type='opencv')
c2w_gl = create_c2w(cam_R_world=cam_R, cam_T_world=cam_T, cam_type='opencv')
# For open3d follow opencv camera coordiante
w2c = np.linalg.inv(c2w)

if __name__ == '__main__':   
    # Congig of semantic kitti scenes(from monoScene)
    save_panoptic_prior = True 
    vis_3d = True
    data_root = 'data/kitti-360'
    frame_id = 7337
    legacy_layout_path = os.path.join(data_root, 'layout', '2013_05_28_drive_0000_sync', '%010d.pkl'%frame_id)
    legacy_semantic_voxel_path = os.path.join(data_root, 'voxel', '2013_05_28_drive_0000_sync', '%010d.pkl'%frame_id)
    image_path = os.path.join(data_root, 'data_2d_raw', '2013_05_28_drive_0000_sync','image_00', '%010d.png'%frame_id)

    stuff_world = convert_legacy_kitti360_stuff(legacy_semantic_voxel_path, voxel_origin= vox_origin,
    voxel_size= vox_size, 
    scene_size= scene_size,
    semantic_list= KITTI360_Stuff) # H W D 
    object_layout = convert_legacy_kitti360_layout(legacy_layout_path)

    geo_group = []
    geo_group += vis_world_bounds_o3d(bounds=world_bounds)
    coordinate_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    coordinate_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1).transform(c2w)
    geo_group += [coordinate_world, coordinate_cam]
    geo_group += vis_voxel_world_o3d(stuff_world,voxelization=True)
    geo_group += vis_layout_o3d(object_layout)
    geo_group += vis_camera_o3d(instrinsic=K, extrinsic=w2c, z_revrse=True)
    # o3d.visualization.draw_plotly(geo_group)
    vis = costum_visualizer_o3d(geo_group=geo_group, set_camera=True, instrinsic=K, extrinsic=w2c, visible = vis_3d)
    # # vis.capture_screen_image(os.path.join('tmp', '%06d'%idx + '.png'))
    project_img = np.asarray(vis.capture_screen_float_buffer())

    if vis_3d:
        vis.run()
        vis.destroy_window()
    H_show , W_show = min(project_img.shape[0], H), min(project_img.shape[1], W)
    rgb = plt.imread(image_path)
    plt.imshow(project_img[:H_show,:W_show] +rgb[:H_show,:W_show])
    plt.show()



    panoptic_prior_dir = semantic_voxel_path = os.path.join(data_root, 'panoptic_prior', '2013_05_28_drive_0000_sync')


    if save_panoptic_prior:
        os.makedirs(panoptic_prior_dir, exist_ok=True)
        panoptic_prior_path = os.path.join(panoptic_prior_dir, '%010d.pkl'%frame_id)
        del stuff_world['loc_grid']
        panoptic_prior = {
                'c2w_openGL' : c2w_gl,
                'c2w_openCV' : c2w,
                'K' : K,
                'HW': (H, W), 
                'stuff_world': stuff_world,
                'object_layout': object_layout,
                }

        with open(panoptic_prior_path, 'wb+') as fp:
            pkl.dump(panoptic_prior, fp)