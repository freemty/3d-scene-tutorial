
import torch
import numpy as np
from math import cos, sin

def create_R(rotate = (0,0,0), scale = (1,1,1)):
    """ Build R matrix from rotate(eular angle) and scale(xyz)
    Args:
        rotate: eular angle (alpha, beta, gamma)
        scale: (s_x, s_y, s_z)
    Return:
        R
    """
    alpha, beta, gamma = rotate[0], rotate[1], rotate[2]

    rx_mat = np.array([
        [1,0,0],
        [0,cos(alpha),-sin(alpha)],
        [0,sin(alpha),cos(alpha)],   
    ])
    ry_mat = np.array([
        [cos(beta),0,sin(beta)],
        [0,1,0],
        [-sin(beta),0,cos(beta)],
    ])

    rz_mat = np.array([
        [cos(gamma),-sin(gamma),0],
        [sin(gamma),cos(gamma),0],
        [0,0,1],
    ])

    scale_mat = np.array([
        [scale[0],0,0],
        [0,scale[1],0],
        [0,0,scale[2]],
    ])

    R = np.matmul(rz_mat, ry_mat).dot(rx_mat).dot(scale_mat)

    return R


def create_c2w(cam_R_world, cam_T_world, cam_type = 'opencv'):
    
    c2w = np.eye(4)
    c2w[:3,:3] = cam_R_world
    c2w[:3,3] = cam_T_world
    # cam_tr_world[:3,:3] = cam_R_world
    # cam_tr_world[:3,3] = cam_T_world
    if cam_type == 'opencv':
        # c2lidar: Transfrom points from camera coodriante to world coordinate (using lidar basis to describe camera basis)
        # > https://zhuanlan.zhihu.com/p/404773542
        # > https://www.zhihu.com/question/407150749
        c2lidar = np.array([
            [0,0,1,0],
            [-1,0,0,0],
            [0,-1,0,0],
            [0,0,0,1]])
    elif cam_type == 'opengl':
        c2lidar = np.array([
            [0,0,-1,0],
            [-1,0,0,0],
            [0,1,0,0],
            [0,0,0,1]
        ])
    else:
        raise TypeError
    
    c2w = c2w @ c2lidar
    # c2w[:3,3] += cam_T_world

    return c2w



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

# # Ray helpers
# Camera Coordinate System
# 这里我们约定z方向长度为1的方向向量为ray_direction, 归一化的方向向量成称之为viewdir(view = ray_direction / norm( ray_direction))
#? Difference between coordinate systems(opengl vs opencv) -> https://stackoverflow.com/questions/44375149/opencv-to-opengl-coordinate-system-transform
# The camera coordinates of OpenCV goes X right, Y down, Z forward. While the camera coordinates of OpenGL goes X right, Y up, Z inward.
# Ray helpers
def get_rays_torch(H, W, K, c2w, cam_type = 'opengl'):
    '''
    coordinata type: opengl, opencv
    '''
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    if cam_type == 'opengl':
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    elif cam_type == 'opencv':
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, K, c2w , cam_type = 'opengl'):
    '''
    coordinata type: opengl, opencv
    '''
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    if cam_type == 'opengl':
        dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    elif cam_type == 'opencv':
        dirs = np.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d
