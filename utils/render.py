import torch

from utils.geometry import *

def render(nerf, rays, volume, images, ocams, poses, min_inv, max_inv):
    B, C, D, _, _ = volume.shape
    device = volume.device
    
    # inference
    invdepth_step = (max_inv - min_inv) / (D - 1)
    depth = 1.0 / torch.arange(min_inv, max_inv + invdepth_step - EPS, 
                               invdepth_step, device = device).flip(0)
    
    z, _ = RaySphereIntersect(rays, depth)
    rays_o = rays[:, :3]
    rays_d = rays[:, 3:6]
    
    xyz = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z.unsqueeze(-1)
    xyz_sample = xyz.permute(1, 0, 2).reshape(D, -1, 3)
    xyz = xyz.view(-1, 3)
    
    c_projects = []
    for cam_num, ocam in enumerate(ocams):
        image = images[:, cam_num]
        pose = poses[cam_num]
        xyz_cam = TransformPoints(xyz, pose)
        
        ratio = image.shape[2] / ocam.height
        grid_2D, mask = FishProject(xyz_cam, ocam, ratio)
        
        grid_2D[:, 0] = 2 * grid_2D[:, 0] / image.shape[3] - 1.0
        grid_2D[:, 1] = 2 * grid_2D[:, 1] / image.shape[2] - 1.0
        grid_2D[mask] = 2.0
        grid_2D = grid_2D.view(1, 1, -1, 2).repeat(B, 1, 1, 1)
        
        c_project = F.grid_sample(image, grid_2D, align_corners=True)
        c_project = c_project.view(B, 3, -1).permute(0, 2, 1)
        c_projects.append(c_project)
    c_projects = torch.cat(c_projects, dim = -1)
    c_projects = c_projects.view(-1, 3 * images.shape[1])
    
    volume_f = EqualVol2XYZ(volume, xyz_sample)
    volume_f = volume_f.permute(0, 2, 1, 3).reshape(-1, C)
    
    rays_d = rays_d.view(1, -1, 1, 3).repeat(B, 1, D, 1).view(-1, 3)
    xyz = xyz.unsqueeze(0).repeat(B, 1, 1).view(-1, 3)
    
    sigma, color = nerf(xyz, rays_d, volume_f, c_projects)
    
    sigma = sigma.view(-1, D)
    color = color.view(-1, D, 3)
    z = z.unsqueeze(0).repeat(B, 1, 1).view(-1, D)
    
    # render occupancy
    alphas = torch.sigmoid(sigma)
    
    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)
    weights = alphas * torch.cumprod(alphas_shifted[:, :-1], -1)
    
    render_depth = torch.sum(weights * z, 1)
    render_color = torch.sum(weights.unsqueeze(-1) * color, 1)
    render_opacity = torch.sum(weights, 1)

    render_depth = render_depth.view(B, -1)
    render_color = render_color.view(B, -1, 3)
    render_opacity = render_opacity.view(B, -1)
    
    return render_color, render_depth, render_opacity

def generate_fiseye_rays(W, H, poses, ocams):
    # need world 2 cam pose
    rayss = []
    for ocam_num, ocam in enumerate(ocams):
        pose = poses[ocam_num]
        rays_d = GenFishRays(W, H, ocam)
        rays_o = torch.zeros_like(rays_d)
        rays = torch.cat([rays_o, rays_d], dim = -1)
        rays = TransformRays(rays, pose.inverse())
        rayss.append(rays)
    rayss = torch.stack(rayss, dim = 0)
    
    return rayss

def generate_eqr_rays(W, H, pose):
    # need world 2 cam pose
    rays_d = GenEqrRays(W, H)
    rays_o = torch.zeros_like(rays_d)
    rays = torch.cat([rays_o, rays_d], dim = -1)
    rays = TransformRays(rays, pose.inverse())
    
    return rays

def generate_pin_rays(W, H, f, pose):
    # need world 2 cam pose
    intrinsic = torch.tensor([
        [f, 0, W / 2], 
        [0, f, H / 2], 
        [0, 0,     1]
    ], device = pose.device)
    rays_d = GenPinRays(W, H, intrinsic)
    rays_o = torch.zeros_like(rays_d)
    rays = torch.cat([rays_o, rays_d], dim = -1)
    rays = TransformRays(rays, pose.inverse())
    
    return rays

# look around and circling
def generate_traj_0(frame_len = 100, radius = 0.2, center = torch.zeros(3)):
    traj = []
    for fram_num in range(frame_len):
        angle = fram_num / float(frame_len) * 2 * torch.pi
        angle = torch.tensor(angle)
        
        x = radius * torch.sin(angle) + center[0]
        y = center[1]
        z = radius * torch.cos(angle) + center[2]
        
        pose = torch.tensor([
            [torch.cos(angle), 0, -torch.sin(angle), x],
            [0,                1,                 0, y],
            [torch.sin(angle), 0,  torch.cos(angle), z],
            [0,                0,                 0, 1]
        ])
        
        traj.append(pose)
        
    return traj

# fixed rot and circling
def generate_traj_1(frame_len = 100, radius = 0.2, center = torch.zeros(3)):
    traj = []
    for fram_num in range(frame_len):
        angle = fram_num / float(frame_len) * 2 * torch.pi
        angle = torch.tensor(angle)
        
        x = radius * torch.sin(angle) + center[0]
        y = center[1]
        z = radius * torch.cos(angle) + center[2]
        
        pose = torch.tensor([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])
        
        traj.append(pose)
        
    return traj

# fixed rot and spiraling
def generate_traj_2(frame_len = 100, radius = 0.2, height = 0.2, center = torch.zeros(3)):
    traj = []
    for fram_num in range(frame_len):
        angle = fram_num / float(frame_len) * 2 * torch.pi
        angle = torch.tensor(angle)
        ratio = fram_num / float(frame_len)
        
        x = radius * torch.cos(angle) + center[0]
        y = radius * torch.sin(angle) + center[1]
        
        if ratio < 0.5:
            z = height * (ratio * 2 - 0.5) + center[2]
        else:
            z = height * (-(ratio - 0.5) * 2 + 0.5) + center[2]
        
        pose = torch.tensor([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])
        
        traj.append(pose)
        
    return traj