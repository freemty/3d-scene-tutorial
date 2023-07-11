import torch
import numpy as np
import open3d as o3d
from tools.kitti360Scripts.helpers.labels import  name2label
import math
# o3d utils

def voxel2points(voxel, voxelSize, range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4], ignore_labels=[17, 255]):
    if isinstance(voxel, np.ndarray): voxel = torch.from_numpy(voxel)
    mask = torch.zeros_like(voxel, dtype=torch.bool)
    for ignore_label in ignore_labels:
        mask = torch.logical_or(voxel == ignore_label, mask)
    mask = torch.logical_not(mask)
    occIdx = torch.where(mask)

    points = torch.cat((occIdx[1][:, None] * voxelSize[0] + voxelSize[1] / 2 + range[0], \
                        occIdx[0][:, None] * voxelSize[1] + voxelSize[1] / 2 + range[1], \
                        occIdx[2][:, None] * voxelSize[2] + voxelSize[2] / 2 + range[2]), dim=1)
    return points, voxel[occIdx]

def voxel_profile(voxel, voxel_size):
    centers = torch.cat((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)
    # centers = voxel
    wlh = torch.cat((torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None],
                          torch.tensor(voxel_size[1]).repeat(centers.shape[0])[:, None],
                          torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None]), dim=1)
    yaw = torch.full_like(centers[:, 0:1], 0)
    return torch.cat((centers, wlh, yaw), dim=1)

def my_compute_box_3d(center, size, heading_angle):
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    heading_angle = -heading_angle - math.pi / 2
    center[:, 2] = center[:, 2] + h / 2
    #R = rotz(1 * heading_angle)
    l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
    x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
    y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
    z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
    #corners_3d = R @ torch.vstack([x_corners, y_corners, z_corners])
    corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
    corners_3d[..., 0] += center[:, 0:1]
    corners_3d[..., 1] += center[:, 1:2]
    corners_3d[..., 2] += center[:, 2:3]
    return corners_3d

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

def vis_voxel_world_o3d(voxel_world, vis = False, voxelize = True,large_voxel = True ):
    geo_group = []

    voxels = voxel_world['semantic_grid']
    # semantic_list = voxel_world['semantic_name']
    semantic_labels = voxel_world['semantic_labels']
    semantic_id =  voxel_world['semantic_id']
    voxel_size = voxel_world['voxel_size']
    scene_range =  np.concatenate( (voxel_world['voxel_origin'], voxel_world['voxel_origin'] + voxel_world['scene_size']))
    # voxel2points(voxel=0)

    color = np.zeros((max(semantic_id) + 1, 3))
    for i, n in semantic_id.items():
        color[i] = np.array(semantic_labels[n].color, dtype=np.float32) / 255.
    

    points, labels = voxel2points(voxels, voxel_size, range=scene_range, ignore_labels=[0])
    points = points.numpy()
    labels = labels.numpy()
    pcd_colors = color[labels.astype(int) % len(color)]
    bboxes = voxel_profile(torch.tensor(points), voxel_size )
    bboxes_corners = my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])
    bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], 
                          [4, 5], [5, 6], [6, 7], [7, 4], 
                          [0, 4], [1, 5], [2, 6], [3, 7]])  # lines along y-axis
    edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)
    edges = edges + bases_[:, None, None]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors[:, :3])
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=1.6, origin=[0, 0, 0])


    pcd.points = o3d.utility.Vector3dVector(points)
    if large_voxel:
        pcd = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size.min())
    
    geo_group += [pcd]

    if voxelize:
        line_sets = o3d.geometry.LineSet()
        line_sets.points = o3d.open3d.utility.Vector3dVector(bboxes_corners.reshape((-1, 3)))
        line_sets.lines = o3d.open3d.utility.Vector2iVector(edges.reshape((-1, 2)))
        line_sets.paint_uniform_color((0, 0, 0))
        geo_group += [line_sets]
    return geo_group

# loc_voxel
def vis_layout_o3d(layout, vis = False):
    mesh_group = []
    for _, obj in layout.items():
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(obj['vertices'])
        mesh.triangles = o3d.utility.Vector3iVector(obj['faces'])
        mesh.paint_uniform_color(obj['color'])
        mesh.compute_vertex_normals()
        mesh_group += [mesh]

    if vis:
        mesh_group += [o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)]
        o3d.visualization.draw_geometries(mesh_group)
    return mesh_group

def costum_visualizer_o3d(geo_group, set_camera = True, instrinsic =None, extrinsic = None, visible= False, background_color = np.array([0,0,0])):
    vis = o3d.visualization.Visualizer()
    W = 2 * round(instrinsic[0,2] + 0.5)
    H = 2 * round(instrinsic[1,2] + 0.5)
    vis.create_window(width= W, height= H, visible= visible)
    for g in geo_group:
        vis.add_geometry(g)
    if set_camera:
        instrinsic[0,2] = W / 2 - 0.5
        instrinsic[1,2] = H / 2 - 0.5
        ctr = vis.get_view_control()
        cam = ctr.convert_to_pinhole_camera_parameters()
        cam.extrinsic = extrinsic
        cam.intrinsic.intrinsic_matrix = instrinsic
        is_su = ctr.convert_from_pinhole_camera_parameters(cam)

        cam_ = vis.get_view_control().convert_to_pinhole_camera_parameters()

    rd = vis.get_render_option()
    rd.light_on = False
    rd.background_color = background_color

    # vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    # vis.run()
    return vis

def vis_camera_o3d(instrinsic, extrinsic, img_size = (256,256),z_revrse = True, vis = False):
    frustums = []
    camera_size = 1
    K = np.eye(4)
    K[:3,:3] = instrinsic
    W2C = extrinsic
    C2W = np.linalg.inv(W2C)
    # img_size = (256,256)
    frustums.append(get_camera_frustum(img_size, K, W2C, frustum_length=camera_size, color=(0,0,0.)))
    # cnt += 1
    camera_group = [frustums2lineset(frustums)]
    if vis:
        o3d.visualization.draw_geometries(camera_group)

    return camera_group


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
