import numpy as np
import open3d as o3d
import torch
from math import cos, sin
from tools.kitti360Scripts.helpers.labels import  name2label
import pickle as pkl
from collections import OrderedDict, defaultdict
#----------------------------KITTI-360 helper------------------------------------------------
def convert_legacy_kitti360_layout(layout_path):
    with open(layout_path, 'rb') as fp:
        layout_raw = pkl.load(fp)
    layout = OrderedDict()
    
    opencv2world = np.array([
            [0,0,1,0],
            [-1,0,0,0],
            [0,-1,0,0],
            [0,0,0,1]])
    for globalId,obj in layout_raw.items():
        # skip dynamic objects

        tr = np.eye(4)
        tr[:3,:3], tr[:3,3] = obj.R, obj.T
        tr = tr @ opencv2world

        vertices = np.concatenate((obj.vertices, np.ones_like(obj.vertices[:,0:1])), axis=-1)
        vertices = vertices @ opencv2world.T

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

def convert_legacy_kitti360_stuff(semantic_voxel_path, scene_size, voxel_size,voxel_origin ,semantic_list):
    '''
    Load old version seamntic voxel
    '''
    with open(semantic_voxel_path, 'rb') as fp:
        stuff_semantic_idx= pkl.load(fp)

    point_num = (scene_size / voxel_size).astype(int) # X, Y, Z
    H, W, D = 64, 64, 64
    H_o, W_o, D_o = point_num[2], point_num[1], point_num[0]
    stuff_semantic = np.zeros(H * W * D)
    for s in semantic_list:
        if stuff_semantic_idx[s].shape[0] == 0:
            continue
        stuff_semantic[stuff_semantic_idx[s]] = name2label[s].id
        
    semantic_voxel = stuff_semantic.reshape(H, W, D)
    semantic_voxel = semantic_voxel.transpose(1, 2, 0) # W D H
    semantic_voxel = np.flip(semantic_voxel, axis=0)
    semantic_voxel = np.flip(semantic_voxel, axis=2)
    semantic_voxel = np.ascontiguousarray(semantic_voxel)

    import torch.nn.functional as F

    if H_o > H:
        semantic_voxel = F.interpolate(torch.tensor(semantic_voxel, dtype=torch.uint8)[None,None,:,:,:], (W_o, D_o, H_o), mode = 'nearest')[0,0].numpy()
    else:
        import torch.nn as nn
        occupancy_mask = torch.from_numpy(semantic_voxel != 0)[None].to(torch.float32)
        ks = (W // W_o, D // D_o,  H // H_o)
        down =  nn.MaxPool3d(kernel_size=ks, stride=ks, return_indices=True)
        _ , _idx = down(occupancy_mask)
        semantic_voxel = semantic_voxel.reshape(-1)[_idx][0]

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
            'semantic_labels' : {n : name2label[n]for n in semantic_list},
            'semantic_id' : {name2label[n].id : n for n in semantic_list},
        }
    return stuff


#--------------------------------Semantic KITTI helper---------------------------------------
def convert_semanticKITTI_layout(semantic_voxel_path):
    pass

def convert_ssc_kitti_stuff(semantic_voxel_path, scene_size, voxel_size,voxel_origin ,semantic_list, KITTI_learning_map):
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

    for i in semantic_list:
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
            'semantic_labels': {n : name2label[n] for n in semantic_list},
            'semantic_id' : {name2label[n].id : n for n in semantic_list},
        }
    return stuff
#-----------------------------------------ClevrW helper---------------------------------------

def convert_legacy_ClevrW_stuff(semantic_voxel_path):
    pass
