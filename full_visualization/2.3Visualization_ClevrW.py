
import os
import sys
import numpy as np
import open3d as o3d
import pickle as pkl 
from matplotlib import pyplot as plt
import json
sys.path.append(os.getcwd())
from collections import OrderedDict
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


name2id = {'ground' : 6,
              'wall' : 11,
              'object' : 26 }
name2color = {'ground' : (128, 64,128), 
              'wall' : (102,102,156),
              'object': (  70,  70,70)}


def clevr_legacy_stuff_converter(legacy_stuff_path, semantic_list, scene_size, voxel_size, voxel_origin):
    with open(legacy_stuff_path, 'rb+') as fp: 
        voxel = pkl.load(fp)
    H, W, D = voxel['(H,W,L)']
    stuff_semantic_idx = voxel['semantic']
    # stuff_semantic = np.zeros(H * W * L)
    point_num = (scene_size / voxel_size).astype(int) # X, Y, Z
    H, W, D = 64, 64, 64
    H_o, W_o, D_o = point_num[2], point_num[1], point_num[0]
    stuff_semantic = np.zeros(H * W * D)
    for s in semantic_list:
        if stuff_semantic_idx[s].shape[0] == 0:
            continue
        stuff_semantic[stuff_semantic_idx[s]] = name2id[s]
        
    semantic_voxel = stuff_semantic.reshape(H, W, D)
    semantic_voxel = semantic_voxel.transpose(1, 2, 0) # W D H
    semantic_voxel = np.flip(semantic_voxel, axis=0) 
    # semantic_voxel = np.flip(semantic_voxel, axis=2)
    semantic_voxel = np.ascontiguousarray(semantic_voxel)

    import torch.nn.functional as F
    import torch
    semantic_voxel = F.interpolate(torch.tensor(semantic_voxel, dtype=torch.uint8)[None,None,:,:,:], (W_o, D_o, H_o), mode = 'nearest')[0,0].numpy()

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
            'semantic_color' : {n : np.array(name2color[n]) / 255. for n in semantic_list},
            'semantic_id' : {n : name2id[n] for n in semantic_list},
            'semantic_name' : semantic_list,
        }
    return stuff


def parse_bbox(tr, scale = 1.):
    pass

def clevr_legacy_layout_converter(legacy_layout_path):
    with open(legacy_layout_path, 'rb') as fp:
        metadata = json.load(fp)
    layout = OrderedDict()
    objects = metadata['objects']
    layout = []
    bbox_id = 0
    for o in objects:
        if o['shape'] == 'wall':
            continue
        tr = np.zeros((4,4))
        tr[3,3] = 1
        tr[0:3,3] = o['3d_coords']
        R = create_R(rotate=(0, 0., o['rotation'] * 3.14 / 180.), scale = np.array((2.,2.,2.)) * o['size'])
        tr[0:3,0:3] = R
        layout += [{'bbox_id': o['index'], 'bbox_tr' : tr, 'bbox_semantic': name2id['object'], 'bbox_shape':o['shape'], 'color':o['color']}]
        bbox_id += 1

    # 8 vertices, 12faces by default 
    # opencv2world = np.array([
    #         [0,0,1,0],
    #         [-1,0,0,0],
    #         [0,-1,0,0],
    #         [0,0,0,1]])
    for globalId,obj in layout_raw.items():
        # skip dynamic objects
        # tr = np.eye(4)
        # tr[:3,:3], tr[:3,3] = obj.R, obj.T
        # tr = tr @ opencv2world.T

        # vertices = np.concatenate((obj.vertices, np.ones_like(obj.vertices[:,0:1])), axis=-1)
        # vertices = vertices @ opencv2world.T

        layout[globalId] = OrderedDict({
            'R' : tr[:3,:3],
            'T' : tr[:3,3],
            'vertices' : vertices[:,:3],
            'faces' : obj.faces,
            'name' : obj.name,
            'color' : np.array(name2label[obj.name].color) / 255.,
            'semanticId' : obj.semanticId,
            'instanceId' : obj.instanceId,
            'annotationId' : obj.annotationId})
            
    return layout



if __name__ == '__main__':
    CLEVR_Stuff_Name = ['ground','wall', 'object']
    x_max, x_min, y_max, y_min, z_max, z_min = 8, -8, 4, 0, 8, -8
    bounds = np.array((8, -8, 8, -8, 4, 0))

    scene_size = np.array((16., 16, 4))
    vox_size = np.array((0.1, 0.1,0.1))
    vox_origin = np.array([-8, -8, 0])
    vis_3d = True
    data_root = 'data/clevrw'
    frame_id = 2002

    metadata_path = os.path.join(data_root, 'Metadata', 'CLEVRTEX_train_%06d.json'%frame_id)
    rgb_path = os.path.join(data_root, 'RGB', 'CLEVRTEX_train_%6d.png')
    legacy_layout_path = os.path.join(data_root, 'Metadata', '%06d'%frame_id)
    legacy_semantc_voxel_path = os.path.join(data_root, 'Voxel', 'CLEVRTEX_train_%06d.pkl'%frame_id)




    stuff_world = clevr_legacy_stuff_converter(legacy_semantc_voxel_path, name2id.keys(), scene_size=scene_size, voxel_size=vox_size, voxel_origin=vox_origin)
    object_layout = clevr_legacy_layout_converter(metadata_path)

    with open(metadata_path,'rb+') as f:
        metadata = json.load(f)

    K = np.array(metadata['intrinsic'])
    c2w = np.array(metadata['extrinsic'])
    w2c = np.linalg.inv(c2w )

    
    geo_group = []
    coordinate_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    coordinate_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1).transform(c2w)
    geo_group += [coordinate_world, coordinate_cam]
    geo_group += vis_voxel_world_o3d(voxel_world = stuff_world, voxelization=True)
    geo_group += vis_layout_o3d(object_layout)
    geo_group += vis_world_bounds_o3d(bounds=bounds)
    # geo_group += vis_bbox_layout_o3d(bbox_tr=bbox_tr, bbox_semantic=bbox_semantic)
    geo_group += vis_camera_o3d(instrinsic=K, extrinsic=w2c, z_revrse=False)
    vis = costum_visualizer_o3d(geo_group=geo_group, instrinsic=K, extrinsic=w2c, visible=vis_3d)

    if vis_3d:
        vis.run()

    print('Done')

        


