
import os
import sys
import numpy as np
import open3d as o3d
import pickle as pkl 
from matplotlib import pyplot as plt
import json
sys.path.append(os.getcwd())
from collections import OrderedDict
from utils.o3d_utils import vis_world_bounds_o3d, vis_layout_o3d, vis_voxel_world_o3d, vis_camera_o3d, costum_visualizer_o3d
from utils.transform_utils import  create_c2w, create_R
from collections import namedtuple
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

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class
    'id'          , 
    'color'       , # The color of this label
    ] )


name2label = {
    'ground' : Label(  'ground'            ,  6 ,      (128, 64,128) ),
    'wall' : Label(  'wall'          ,  11 ,       (102,102,156)) ,
    'object' : Label(  'object' ,  2 ,    (  70,  70,70) )
}
    #       name                     id     color


name2id = {'ground' : 6,
              'wall' : 11,
              'object' : 26 }
name2color = {'ground' : (128, 64,128), 
              'wall' : (102,102,156),
              'object': (  70,  70,70)}

idx2rgb = {
    1:[255, 0, 0],
    2:[0, 128, 0],
    3:[0, 0, 255],
    4:[255, 255, 0],
    5:[141, 211, 199],
    6:[255, 255, 179],
    7:[190, 186, 218],
    8:[251, 128, 114],
    9:[128, 177, 211]}

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
    semantic_voxel = semantic_voxel.transpose(2, 1, 0) # W D H
    semantic_voxel = np.flip(semantic_voxel, axis=0) 
    # semantic_voxel = np.flip(semantic_voxel, axis=1) 
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
            'semantic_labels' : {n : name2label[n] for n in semantic_list},
            'semantic_id' : {name2label[n].id : n for n in semantic_list}
        }
    return stuff


def CV2GL(cv_mat):
    '''' 
    CV (camera look at +Z, up +Y)
        -Y
        |
        |
        +-----> +X
       /
      /
    +Z

    GL (camera look at -Z, up +Y)
        +Y
        |
        |
        +-----> +X
       /
      /
    -Z
    '''
    cv2gl_mat = np.array(
        [[1,0,0,0],
        [0,-1,0,0],
        [0,0,-1,0],
        [0,0,0,1]])
    # using GL basis to describe CV basis
    gl_mat = cv_mat @ cv2gl_mat
    # gl_mat[:3,3] = cv_mat[:3,3]
    return gl_mat



def parse_bbox(tr,name, semanticId, instanceId, color, scale = 1.):
    s = scale / 2
    bbox_verts = np.array(
    [[s,s,s,1],
    [s,s,-s,1],
    [s,-s,s,1],
    [s,-s,-s,1],
    [-s,s,-s,1],
    [-s,s,s,1],
    [-s,-s,-s,1],
    [-s,-s,s,1]])
    # vertexes_color =  ((bbox_verts[:,0:3].clone() + s) / (s * 2)).clip(0,1)
    bbox_verts = bbox_verts @ tr.T
    bbox_faces = np.array(
    [[0. ,2., 1.],
    [2. ,3. ,1.],
    [4. ,6. ,5.],
    [6. ,7. ,5.],
    [4. ,5. ,1.],
    [5. ,0. ,1.],
    [7. ,6. ,2.],
    [6. ,3. ,2.],
    [5. ,7. ,0.],
    [7. ,2. ,0.],
    [1. ,3. ,4.],
    [3. ,6. ,4.]])

    bounding_box = OrderedDict(
        {
            'R' : tr[:3,:3],
            'T' : tr[:3,3],
            'vertices' : bbox_verts[:,:3],
            'faces' : bbox_faces,
            'name' : name,
            'color' : np.array(color) * 2 / 255.,
            'semanticId' : semanticId,
            'instanceId' : instanceId,
            'annotationId' : -1
            })

    return bounding_box

def clevr_legacy_layout_converter(legacy_layout_path):
    with open(legacy_layout_path, 'rb') as fp:
        metadata = json.load(fp)
    layout = OrderedDict()
    objects = metadata['objects']
    # bbox_id = 0
    for o in objects:
        name = 'object'
        if o['shape'] == 'wall':
            continue
        tr = np.eye((4))
        tr[0:3,3] = o['3d_coords']
        R = create_R(rotate=(0, 0., o['rotation'] * 3.14 / 180.), scale = np.array((2.,2.,2.)) * o['size'])
        tr[0:3,0:3] = R
        bbox_id  = o['index']
        # 8 vertices, 12faces by default 
        bbox = parse_bbox(tr = tr, scale=1, color = idx2rgb[bbox_id], name=name, semanticId= name2id[name], instanceId= bbox_id)
        layout[bbox_id] = bbox
        # bbox_id += 1

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
    rgb_path = os.path.join(data_root, 'RGB', 'CLEVRTEX_train_%06d.png'%frame_id)
    legacy_layout_path = os.path.join(data_root, 'Metadata', '%06d'%frame_id)
    legacy_semantc_voxel_path = os.path.join(data_root, 'Voxel', 'CLEVRTEX_train_%06d.pkl'%frame_id)




    stuff_world = clevr_legacy_stuff_converter(legacy_semantc_voxel_path, name2id.keys(), scene_size=scene_size, voxel_size=vox_size, voxel_origin=vox_origin)
    object_layout = clevr_legacy_layout_converter(metadata_path)

    with open(metadata_path,'rb+') as f:
        metadata = json.load(f)

    # K = np.array(metadata['intrinsic'])
    H, W = 256,256
    K = np.array([
            [280.0,0.0,128.0],
            [0.0,280,128.0],
            [0.0,0.0,1.0]])
    c2w_gl = np.array(metadata['extrinsic'])
    c2w = CV2GL(c2w_gl)
    w2c = np.linalg.inv(c2w)

    
    geo_group = []
    coordinate_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    coordinate_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1).transform(c2w)
    geo_group += [coordinate_world, coordinate_cam]
    geo_group += vis_voxel_world_o3d(voxel_world = stuff_world, voxelize=True)
    geo_group += vis_layout_o3d(object_layout)
    geo_group += vis_world_bounds_o3d(bounds=bounds)

    # geo_group += vis_bbox_layout_o3d(bbox_tr=bbox_tr, bbox_semantic=bbox_semantic)
    geo_group += vis_camera_o3d(instrinsic=K, extrinsic=w2c, z_revrse=False)
    vis = costum_visualizer_o3d(geo_group=geo_group, instrinsic=K, extrinsic=w2c, visible=vis_3d)

    project_img = np.asarray(vis.capture_screen_float_buffer())

    if vis_3d:
        vis.run()
        vis.destroy_window()
    H_show , W_show = min(project_img.shape[0], H), min(project_img.shape[1], W)
    rgb = plt.imread(rgb_path)[...,:3]
    plt.imshow(project_img[:H_show,:W_show] +rgb[:H_show,:W_show])
    plt.show()

    print('Done')

        


