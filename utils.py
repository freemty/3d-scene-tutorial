import numpy as np
import open3d as o3d
from math import cos, sin

pi = 3.1415

# Transform utils

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
    # cam_tr_world[:3,:3] = cam_R_world
    # cam_tr_world[:3,3] = cam_T_world
    if cam_type == 'opencv':
        # rectmat: Transfrom points from camera coodriante to world coordinate (using world basis to describe cmaera basis)
        # > https://zhuanlan.zhihu.com/p/404773542
        # > https://www.zhihu.com/question/407150749
        rectmat = np.array([
            [0,0,1,0],
            [-1,0,0,0],
            [0,-1,0,0],
            [0,0,0,1]])
    elif cam_type == 'opengl':
        rectmat = np.array([
            [0,0,-1,0],
            [-1,0,0,0],
            [0,1,0,0],
            [0,0,0,1]
        ])
    else:
        raise TypeError
    
    c2w = c2w @ rectmat
    c2w[:3,3] += cam_T_world

    return c2w

import open3d as o3d
import json
import numpy as np


# o3d utils
def vis_world_bounds_o3d(bounds, add_origin = False, vis = False):
    x_max, x_min, y_max, y_min, z_max, z_min = bounds.ravel()
    # print("Let\'s draw a cubic using o3d.geometry.LineSet")
    points = [
        [x_min, y_min, z_min], 
        [x_max, y_min, z_min], 
        [x_min, y_max, z_min], 
        [x_max, y_max, z_min], 
        [x_min, y_min, z_max], 
        [x_max, y_min, z_max],
        [x_min, y_max, z_max], 
        [x_max, y_max, z_max]]
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [[0, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([line_set])
    mark_group = [line_set]
    if add_origin:
        mark_group += [o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0., 0,0])]
    if vis:
        o3d.visualization.draw_geometries(mark_group)
    return mark_group

# def vis_camera():
#     get_camera_frustum(img_size=, K, W2C, )

def get_camera_frustum(img_size, K, W2C, frustum_length=0.5, color=[0., 1., 0.]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],                          # frustum origin
                               [-half_w, -half_h, frustum_length],    # top-left image corner
                               [half_w, -half_h, frustum_length],     # top-right image corner
                               [half_w, half_h, frustum_length],      # bottom-right image corner
                               [-half_w, half_h, frustum_length]])    # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    C2W = np.linalg.inv(W2C)
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    return frustum_points, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset

def draw_camera(H, W, K, w2c):
    return frustums2lineset([get_camera_frustum((H,W), K, w2c)])


def visualize_cameras(colored_camera_dicts, sphere_radius, camera_size=0.1, geometry_file=None, geometry_type='mesh'):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    things_to_draw = [sphere, coord_frame]

    idx = 0
    for color, camera_dict in colored_camera_dicts:
        idx += 1

        cnt = 0
        frustums = []
        for img_name in sorted(camera_dict.keys()):
            K = np.array(camera_dict[img_name]['K']).reshape((4, 4))
            W2C = np.array(camera_dict[img_name]['W2C']).reshape((4, 4))
            C2W = np.linalg.inv(W2C)
            img_size = camera_dict[img_name]['img_size']
            frustums.append(get_camera_frustum(img_size, K, W2C, frustum_length=camera_size, color=color))
            cnt += 1
        cameras = frustums2lineset(frustums)
        things_to_draw.append(cameras)

    if geometry_file is not None:
        if geometry_type == 'mesh':
            geometry = o3d.io.read_triangle_mesh(geometry_file)
            geometry.compute_vertex_normals()
        elif geometry_type == 'pointcloud':
            geometry = o3d.io.read_point_cloud(geometry_file)
        else:
            raise Exception('Unknown geometry_type: ', geometry_type)

        things_to_draw.append(geometry)

    o3d.visualization.draw_geometries(things_to_draw)