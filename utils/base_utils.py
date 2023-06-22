# import numpy as np
# import open3d as o3d
# import torch
# from math import cos, sin
# from tools.kitti360Scripts.helpers.labels import  name2label

# pi = 3.1415


# # Transform utils

# def create_R(rotate = (0,0,0), scale = (1,1,1)):
#     """ Build R matrix from rotate(eular angle) and scale(xyz)
#     Args:
#         rotate: eular angle (alpha, beta, gamma)
#         scale: (s_x, s_y, s_z)
#     Return:
#         R
#     """
#     alpha, beta, gamma = rotate[0], rotate[1], rotate[2]

#     rx_mat = np.array([
#         [1,0,0],
#         [0,cos(alpha),-sin(alpha)],
#         [0,sin(alpha),cos(alpha)],   
#     ])
#     ry_mat = np.array([
#         [cos(beta),0,sin(beta)],
#         [0,1,0],
#         [-sin(beta),0,cos(beta)],
#     ])

#     rz_mat = np.array([
#         [cos(gamma),-sin(gamma),0],
#         [sin(gamma),cos(gamma),0],
#         [0,0,1],
#     ])

#     scale_mat = np.array([
#         [scale[0],0,0],
#         [0,scale[1],0],
#         [0,0,scale[2]],
#     ])

#     R = np.matmul(rz_mat, ry_mat).dot(rx_mat).dot(scale_mat)

#     return R


# def create_c2w(cam_R_world, cam_T_world, cam_type = 'opencv'):
    
#     c2w = np.eye(4)
#     c2w[:3,:3] = cam_R_world
#     # cam_tr_world[:3,:3] = cam_R_world
#     # cam_tr_world[:3,3] = cam_T_world
#     if cam_type == 'opencv':
#         # rectmat: Transfrom points from camera coodriante to world coordinate (using world basis to describe cmaera basis)
#         # > https://zhuanlan.zhihu.com/p/404773542
#         # > https://www.zhihu.com/question/407150749
#         rectmat = np.array([
#             [0,0,1,0],
#             [-1,0,0,0],
#             [0,-1,0,0],
#             [0,0,0,1]])
#     elif cam_type == 'opengl':
#         rectmat = np.array([
#             [0,0,-1,0],
#             [-1,0,0,0],
#             [0,1,0,0],
#             [0,0,0,1]
#         ])
#     else:
#         raise TypeError
    
#     c2w = c2w @ rectmat
#     c2w[:3,3] += cam_T_world

#     return c2w




# def visualize_cameras(colored_camera_dicts, sphere_radius, camera_size=0.1, geometry_file=None, geometry_type='mesh'):
#     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
#     sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
#     sphere.paint_uniform_color((1, 0, 0))

#     coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
#     things_to_draw = [sphere, coord_frame]

#     idx = 0
#     for color, camera_dict in colored_camera_dicts:
#         idx += 1

#         cnt = 0
#         frustums = []
#         for img_name in sorted(camera_dict.keys()):
#             K = np.array(camera_dict[img_name]['K']).reshape((4, 4))
#             W2C = np.array(camera_dict[img_name]['W2C']).reshape((4, 4))
#             C2W = np.linalg.inv(W2C)
#             img_size = camera_dict[img_name]['img_size']
#             frustums.append(get_camera_frustum(img_size, K, W2C, frustum_length=camera_size, color=color))
#             cnt += 1
#         cameras = frustums2lineset(frustums)
#         things_to_draw.append(cameras)

#     if geometry_file is not None:
#         if geometry_type == 'mesh':
#             geometry = o3d.io.read_triangle_mesh(geometry_file)
#             geometry.compute_vertex_normals()
#         elif geometry_type == 'pointcloud':
#             geometry = o3d.io.read_point_cloud(geometry_file)
#         else:
#             raise Exception('Unknown geometry_type: ', geometry_type)

#         things_to_draw.append(geometry)

#     o3d.visualization.draw_geometries(things_to_draw)
