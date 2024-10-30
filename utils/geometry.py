import sys
import torch
import torch.nn.functional as F

EPS = sys.float_info.epsilon

def TransformRays(rays, pose):
    # need cam 2 world pose
    pose = pose.to(rays.device)
    
    rays_o = rays[:, :3] + pose[:3, 3]
    rays_d = rays[:, 3:6] @ pose[:3, :3].T
    
    rays = torch.cat([rays_o, rays_d, rays[:, 6:]], -1)
    
    return rays

def TransformPoints(points, pose):
    # need cam 2 world pose
    pose = pose.to(points.device)
    
    points = points @ pose[:3, :3].T + pose[:3, 3]
    
    return points

def FishProject(rays_3D_cam, ocam, ratio = 1.0):
    grid_2D = ocam.world2cam(rays_3D_cam)
    
    mask_x = (grid_2D[:, 0] < 0) | (grid_2D[:, 0] >= ocam.width)
    mask_y = (grid_2D[:, 1] < 0) | (grid_2D[:, 1] >= ocam.height)
    mask = mask_x | mask_y
    
    grid_2D *= ratio
    
    return grid_2D, mask

def RayDist(rays_o, rays_d):
    # return the distance from origin to ray (rays_o, rays_d)
    cross = torch.linalg.cross(rays_o, rays_d, dim = -1)
    dist = torch.linalg.norm(cross, dim = -1) / torch.linalg.norm(rays_d, dim = -1)
    
    return dist

# Generated rays_d is normalized to 1 
def GenEqrRays(W, H):
    u = torch.arange(W) + 0.5
    v = torch.arange(H) + 0.5
    u_grid, v_grid = torch.meshgrid(u, v, indexing = "xy")
    uv = torch.stack([u_grid, v_grid], dim = -1)
    uv = uv.view(-1 ,2)
    
    phi = 2 * torch.pi * (0.75 - uv[:, 0] / W)      # yaw angle
    theta = torch.pi * (uv[:, 1] / (W / 2))         # pitch angle
    
    x = torch.sin(theta) * torch.cos(phi)
    y = -torch.cos(theta)
    z = torch.sin(theta) * torch.sin(phi)
    
    rays_d = torch.stack([x, y, z], dim = -1)
    
    return rays_d

def GenPinRays(W, H, intrinsic):
    u = torch.arange(W) + 0.5
    v = torch.arange(H) + 0.5
    u_grid, v_grid = torch.meshgrid(u, v, indexing="xy")
    uv = torch.stack([u_grid, v_grid], dim = -1)
    uv = uv.view(-1 ,2)
    
    point_x = (uv[:, 0] - intrinsic[0][2]) / intrinsic[0][0]
    point_y = (uv[:, 1] - intrinsic[1][2]) / intrinsic[1][1]
    point_z = torch.ones_like(point_x)
    
    rays_d = torch.stack([point_x, point_y, point_z], -1)
    rays_d = rays_d / torch.linalg.norm(rays_d, dim = -1, keepdim = True)
    
    return rays_d

def GenFishRays(W, H, ocam):
    u = torch.arange(W) + 0.5
    v = torch.arange(H) + 0.5
    u_grid, v_grid = torch.meshgrid(u, v, indexing = "xy")
    ratio = ocam.height / H
    u_grid = u_grid * ratio
    v_grid = v_grid * ratio
    uv = torch.stack([u_grid, v_grid], dim = -1)
    uv = uv.view(-1 ,2)
    rays_d = ocam.cam2world(uv)
    
    return rays_d

def SampleRays(rays, N, rand = True):
    rays_len = rays.shape[0]
    
    if rand:
        index = torch.randint(0, rays_len, (N, ))
    else:
        step = (0 - rays_len) / (N - 1)
        index = torch.arange(0, rays_len + step - EPS, step)
    
    return rays[index], index

def EqualRec2Rays(er_image):
    '''
    input: 
    er_image: (B, C, H, W)
    output:
    rays: (B, H*W, 6+C) (norm = 1)
    '''
    
    B, C, H, W = er_image.shape
    device = er_image.device
     
    rays_d = GenEqrRays(W, H).to(device).unsqueeze(0).repeat(B, 1, 1)
    rays_o = torch.zeros_like(rays_d)
    
    feat = er_image.permute(0, 2, 3, 1)
    feat = feat.view(B, -1 ,C)
    rays = torch.concat([rays_o, rays_d, feat], dim = -1)
    
    return rays

def EqualRec2Pinhole(er_image, image_H, image_W, intrinsic, R):
    '''
    input: 
    er_image: (B, C, H, W)
    image_H, image_W: pinhole image size
    intrinsic: pinhole camera intrinsic (3, 3)
    R: pinhole to panorama rotation matrix (B, 3, 3)
    output:
    pinhole image: (B, C, image_H, image_W)
    '''
    
    B, _, H, W = er_image.shape
    device = er_image.device
    
    rays_d = GenPinRays(image_W, image_H, intrinsic)
    rays_d = rays_d.to(device).unsqueeze(0).view(1, -1, 3).repeat(B, 1, 1)
    rays_d = torch.bmm(rays_d, R.permute(0, 2, 1))
    rays_d = rays_d.view(B, image_H, image_W, 3)
    
    grid_phi = torch.atan2(rays_d[..., 2], rays_d[..., 0])
    grid_phi[grid_phi < -0.5 * torch.pi] += 2 * torch.pi
    grid_phi = - grid_phi / torch.pi + 0.5
    
    grid_theta = torch.asin(rays_d[..., 1])
    grid_theta = grid_theta / ((H / W) * torch.pi)
    
    grid = torch.stack([grid_phi, grid_theta], -1)
    grid = grid / torch.tensor([-torch.pi, -(H / W) * torch.pi], device = device)
    grid = grid + torch.tensor([0.5, 0.0], device = device)
    
    ph_image = F.grid_sample(er_image, grid, align_corners = True, padding_mode = "border")
    
    return ph_image

def EqualVol2XYZ(er_vol, xyz):
    '''
    input:
    er_vol: (B, D, C, H, W)
    xyz: (D, N, 3)
    output:
    sample: (B, D, N, C)
    '''
    B, C, D, H, W = er_vol.shape
    
    er_vol = er_vol.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
    xyz  = xyz / torch.linalg.norm(xyz, dim = -1, keepdim = True)
    
    grid_phi = torch.atan2(xyz[..., 2], xyz[..., 0])
    grid_phi[grid_phi < -0.5 * torch.pi] += 2 * torch.pi
    grid_phi = - grid_phi / torch.pi + 0.5
    
    grid_theta = torch.asin(xyz[..., 1])
    grid_theta = grid_theta / ((H / W) * torch.pi)
    
    grid = torch.stack([grid_phi, grid_theta], -1)
    grid = grid.view(1, D, -1, 2).repeat(B, 1, 1, 1).view(B*D, 1, -1, 2)
    
    sample = F.grid_sample(er_vol, grid, align_corners = True)
    sample = sample.view(B, D, C, -1).permute(0, 1, 3, 2)
    
    return sample
    
def RaySphereIntersect(rays, radius, center = torch.zeros(3)):
    '''
    input:
    ray: (N, 3+3)
    radius: (M)
    center: (3)
    output:
    z: (N, M)
    mask: (N, M)
    '''
    depth_layer = radius.shape[0]
    device = rays.device
    center = center.to(device)

    rays_o = rays[:, :3]
    rays_d = rays[:, 3:6]
    pseudo_o = rays_o - center
    
    a = torch.sum(rays_d ** 2, dim = -1).unsqueeze(-1).repeat(1, depth_layer)
    b = 2 * torch.sum(rays_d * pseudo_o, dim = -1).unsqueeze(-1) .repeat(1, depth_layer)
    c = torch.sum(pseudo_o ** 2, dim = -1).unsqueeze(-1) - radius.unsqueeze(0) ** 2
    
    delta = b ** 2 - 4 * a * c
    intersect_mask = delta > 0
    z = torch.zeros_like(delta)
    
    delta = delta[intersect_mask]
    a = a[intersect_mask]
    b = b[intersect_mask]
    
    z[intersect_mask] = (-b + torch.sqrt(delta)) / (2 * a)
    
    return z, intersect_mask
