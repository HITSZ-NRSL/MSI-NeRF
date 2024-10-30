import torch
from torch import optim
import torch.optim.lr_scheduler as lrs
from pytorch_lightning import LightningModule
from scipy.spatial.transform import Rotation as R

from model.loss import DepthLoss, ColorLoss
from model.network import OmniVolume, NeRF
from utils.render import *
from utils.geometry import SampleRays, EPS
from utils.visualize import vis_depth_image, vis_image

class MInterface(LightningModule):
    def __init__(self, args, ocams, poses):
        super().__init__()

        self.data_path = args.data_path
        self.lr = args.lr
        self.lr_scheduler = args.lr_scheduler
        self.lr_decay_steps = args.lr_decay_steps
        self.lr_decay_rate = args.lr_decay_rate
        self.lr_decay_min_lr = args.lr_decay_min_lr
        self.depth_weight = args.depth_weight
        
        self.CH = args.CH
        self.out_CH = args.out_CH
        self.image_size = args.image_size
        self.depth_size = args.depth_size
        self.novel_size = args.novel_size
        self.novel_f = args.novel_f
        self.depth_layer = args.depth_layer
        self.min_depth = args.min_depth
        self.max_invdepth = 1 / self.min_depth
        self.max_depth = 1 / EPS
        self.min_invdepth = 1 / self.max_depth
        self.eqr_sample_num = args.eqr_sample_num
        self.fish_sample_num = args.fish_sample_num
        self.val_chunk_size = args.val_chunk_size
        self.render_novel_view = args.render_novel_view
        self.eval_nvs = args.eval_nvs
        
        self.ocams = ocams
        self.poses = poses
        self.model = OmniVolume(self.CH, self.out_CH, add_appr = True)
        self.nerf = NeRF(f_channel = self.out_CH)
        self.depth_loss = DepthLoss()
        self.color_loss = ColorLoss()
        
        self.fiseye_rayss = generate_fiseye_rays(self.image_size[1], self.image_size[0], poses, ocams)
        self.eqr_rays = generate_eqr_rays(self.depth_size[1], self.depth_size[0], torch.eye(4))
        
        if self.render_novel_view:
            self.pin_rays = generate_pin_rays(self.novel_size[1], self.novel_size[0], self.novel_f, torch.eye(4))
            if args.traj_type == 0:
                self.render_traj = generate_traj_0()
            elif args.traj_type == 1:
                self.render_traj = generate_traj_1()
            elif args.traj_type == 2:
                self.render_traj = generate_traj_2()

    def forward(self, images, grids):
        return self.model(images, grids)
    
    def configure_optimizers(self):
        optimizer = optim.Adam([{'params': self.model.parameters()}, 
                                {'params': self.nerf.parameters()}], lr = self.lr)
        if self.lr_scheduler is None:
                return optimizer
        else:
            if self.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer, 
                                       step_size=self.lr_decay_steps, gamma=self.lr_decay_rate)
            elif self.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer, 
                                                  T_max=self.lr_decay_steps, eta_min=self.lr_decay_min_lr)
        
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        images, gt_invdepth, grids, aug_ind, aug_rot = batch
        
        # augmentation
        aug_angle = aug_rot * 2 * torch.pi / self.depth_size[1]
        aug_T = torch.tensor([
            [torch.cos(aug_angle), 0, -torch.sin(aug_angle), 0],
            [0,                    1,                     0, 0],
            [torch.sin(aug_angle), 0,  torch.cos(aug_angle), 0],
            [0,                    0,                     0, 1]
        ], device = self.device)
        
        ocams = [self.ocams[int(i)] for i in aug_ind]
        poses = [aug_T.inverse() @ self.poses[int(i)].to(self.device) for i in aug_ind]
        
        # volume inference
        B = images.shape[0]
        volume = self.model(images, grids)
        
        # color render loss
        color_loss = 0
        for image_num in range(images.shape[1]):
            image = images[:, image_num]
            fiseye_rays = self.fiseye_rayss[int(aug_ind[image_num])].to(self.device)
            fiseye_rays, fish_index = SampleRays(fiseye_rays, self.fish_sample_num)
            fiseye_rays = TransformRays(fiseye_rays, aug_T)
            
            color, _, _ = render(self.nerf, fiseye_rays, volume, images, ocams, poses,
                                 self.min_invdepth, self.max_invdepth)
            
            gt_color = image.view(B, 3, -1).permute(0, 2, 1)[:, fish_index]
            color_loss += self.color_loss(color, gt_color)
        
        # depth render loss
        eqr_rays = self.eqr_rays.to(self.device)
        eqr_rays, eqr_index = SampleRays(eqr_rays, self.eqr_sample_num)
        eqr_rays = TransformRays(eqr_rays, aug_T)
            
        _, depth, _ = render(self.nerf, eqr_rays, volume, images, ocams, poses,
                             self.min_invdepth, self.max_invdepth)
        
        depth_loss = self.depth_loss(depth, gt_invdepth.view(B, -1)[:, eqr_index])
        
        loss = color_loss + self.depth_weight * depth_loss
        
        self.log("train color loss", color_loss.item(), sync_dist=True)
        self.log("train invdepth loss", depth_loss.item(), sync_dist=True)
        self.log("train loss", loss.item(), sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, gt_invdepth, grids, _, _ = batch
        
        # volume inference
        B = images.shape[0]
        volume = self.model(images, grids)
        
        # color render loss
        color_loss = 0
        for image_num in range(images.shape[1]):
            image = images[:, image_num]
            fiseye_rays = self.fiseye_rayss[image_num].to(self.device)
            color = []
            for chunk in range(0, fiseye_rays.shape[0], self.val_chunk_size):
                rays = fiseye_rays[chunk : chunk + self.val_chunk_size]
                color_chunk, _, _ = render(self.nerf, rays, volume, images, self.ocams, self.poses,
                                           self.min_invdepth, self.max_invdepth)
                color.append(color_chunk)
            color = torch.cat(color, 1)
            color_loss += self.color_loss(color, image.view(B, 3, -1).permute(0, 2, 1))
        
        # depth render loss
        eqr_rays = self.eqr_rays.to(self.device)
        depth = []
        for chunk in range(0, eqr_rays.shape[0], self.val_chunk_size):
            rays = eqr_rays[chunk : chunk + self.val_chunk_size]
            _, depth_chunk, _ = render(self.nerf, rays, volume, images, self.ocams, self.poses,
                                       self.min_invdepth, self.max_invdepth)
            depth.append(depth_chunk)
        depth = torch.cat(depth, 1)
        depth_loss = self.depth_loss(depth, gt_invdepth.view(B, -1))
        
        loss = color_loss + self.depth_weight * depth_loss
        
        self.log("val color loss", color_loss.item(), sync_dist=True)
        self.log("val invdepth loss", depth_loss.item(), sync_dist=True)
        self.log("val loss", loss.item(), sync_dist=True)
        
        # visualization
        color_v = color.view(B, self.image_size[0], self.image_size[1], 3).permute(0, 3, 1, 2)
        vis = torch.cat([image, color_v], dim = 2)
        vis = vis_image(vis[0], totensor=True)
        self.logger.experiment.add_image('gt val color vis', vis, self.global_step)
        
        depth_v = depth.view(B, 1, self.depth_size[0], self.depth_size[1])
        vis = torch.cat([gt_invdepth, 1.0 / depth_v], dim = 2)
        vis = vis_depth_image(vis[0], totensor=True)
        self.logger.experiment.add_image('gt val invdepth vis', vis, self.global_step)
        
        # novel visualization
        if self.render_novel_view:
            pin_rays = self.pin_rays.to(self.device)
            pose = torch.eye(4, device=self.device)
            rot = R.random().as_matrix()
            pose[:3, :3] = torch.from_numpy(rot).to(self.device)
            theta = torch.rand(1) * 2 * torch.pi
            phi = torch.rand(1) * torch.pi
            r = torch.rand(1) * self.min_depth
            pose[0, 3] = r * torch.sin(phi) * torch.cos(theta)
            pose[1, 3] = r * torch.sin(phi) * torch.sin(theta)
            pose[2, 3] = r * torch.cos(phi)
            pin_rays = TransformRays(pin_rays, pose)
            
            novel_color = []
            novel_depth = []
            for chunk in range(0, pin_rays.shape[0], self.val_chunk_size):
                rays = pin_rays[chunk : chunk + self.val_chunk_size]
                color_chunk, depth_chunk, _ = render(self.nerf, rays, volume, images, self.ocams, self.poses,
                                                     self.min_invdepth, self.max_invdepth)
                novel_color.append(color_chunk)
                novel_depth.append(depth_chunk)
            novel_color = torch.cat(novel_color, 1)
            novel_depth = torch.cat(novel_depth, 1)
            
            novel_color_v = novel_color.view(B, self.novel_size[0], self.novel_size[1], 3).permute(0, 3, 1, 2)
            vis = vis_image(novel_color_v[0], totensor=True)
            self.logger.experiment.add_image('novel color vis', vis, self.global_step)
            novel_depth_v = novel_depth.view(B, 1, self.novel_size[0], self.novel_size[1])
            vis = vis_depth_image((1.0 / novel_depth_v[0]), totensor=True)
            self.logger.experiment.add_image('novel invdepth vis', vis, self.global_step)
        
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx):
        if self.eval_nvs:
            images, _, eqr_poses, grids = batch
            B = images.shape[0]
            volume = self.model(images, grids)
            
            novel_vis_colors = []
            novel_vis_depths = []
            for eqr_pose_num in range(eqr_poses.shape[1]):
                eqr_pose = eqr_poses[0, eqr_pose_num]
                eqr_rays = self.eqr_rays.to(self.device)
                
                eqr_rays = TransformRays(eqr_rays, eqr_pose)
                
                novel_color = []
                novel_depth = []
                for chunk in range(0, eqr_rays.shape[0], self.val_chunk_size):
                    rays = eqr_rays[chunk : chunk + self.val_chunk_size]
                    color_chunk, depth_chunk, _ = render(self.nerf, rays, volume, images, self.ocams, self.poses,
                                                        self.min_invdepth, self.max_invdepth)
                    novel_color.append(color_chunk)
                    novel_depth.append(depth_chunk)
                novel_color = torch.cat(novel_color, 1)
                novel_depth = torch.cat(novel_depth, 1)
            
                novel_color_v = novel_color.view(B, self.depth_size[0], self.depth_size[1], 3).permute(0, 3, 1, 2)
                novel_vis_color = vis_image(novel_color_v[0], bgr=True)
                novel_vis_colors.append(novel_vis_color)
                
                novel_depth_v = novel_depth.view(B, 1, self.depth_size[0], self.depth_size[1])
                novel_vis_depth = vis_depth_image((1.0 / novel_depth_v[0]), self.min_invdepth, self.max_invdepth)
                novel_vis_depths.append(novel_vis_depth)
            
            return novel_vis_colors, novel_vis_depths
        
        else:
            images, gt_invdepth, grids, _, _ = batch
            
            # volume inference
            B = images.shape[0]
            volume = self.model(images, grids)
                    
            # novel visualization
            if self.render_novel_view:
                novel_vis_colors = []
                novel_vis_depths = []
                for pose in self.render_traj:
                    pin_rays = self.pin_rays.to(self.device)
                    pin_rays = TransformRays(pin_rays, pose)
                    
                    novel_color = []
                    novel_depth = []
                    for chunk in range(0, pin_rays.shape[0], self.val_chunk_size):
                        rays = pin_rays[chunk : chunk + self.val_chunk_size]
                        color_chunk, depth_chunk, _ = render(self.nerf, rays, volume, images, self.ocams, self.poses,
                                                            self.min_invdepth, self.max_invdepth)
                        novel_color.append(color_chunk)
                        novel_depth.append(depth_chunk)
                    novel_color = torch.cat(novel_color, 1)
                    novel_depth = torch.cat(novel_depth, 1)
                
                    novel_color_v = novel_color.view(B, self.novel_size[0], self.novel_size[1], 3).permute(0, 3, 1, 2)
                    novel_vis_color = vis_image(novel_color_v[0])
                    novel_vis_colors.append(novel_vis_color)
                    
                    novel_depth_v = novel_depth.view(B, 1, self.novel_size[0], self.novel_size[1])
                    novel_vis_depth = vis_depth_image((1.0 / novel_depth_v[0]), self.min_invdepth, self.max_invdepth)
                    novel_vis_depths.append(novel_vis_depth)
                    
                return novel_vis_colors, novel_vis_depths

            else:
                # color render
                color_loss = 0
                for image_num in range(images.shape[1]):
                    image = images[:, image_num]
                    fiseye_rays = self.fiseye_rayss[image_num].to(self.device)
                    color = []
                    for chunk in range(0, fiseye_rays.shape[0], self.val_chunk_size):
                        rays = fiseye_rays[chunk : chunk + self.val_chunk_size]
                        color_chunk, _, _ = render(self.nerf, rays, volume, images, self.ocams, self.poses,
                                                self.min_invdepth, self.max_invdepth)
                        color.append(color_chunk)
                    color = torch.cat(color, 1)
                    color_loss += self.color_loss(color, image.view(B, 3, -1).permute(0, 2, 1))

                # depth render
                eqr_rays = self.eqr_rays.to(self.device)
                color = []
                depth = []
                for chunk in range(0, eqr_rays.shape[0], self.val_chunk_size):
                    rays = eqr_rays[chunk : chunk + self.val_chunk_size]
                    color_chunk, depth_chunk, _ = render(self.nerf, rays, volume, images, self.ocams, self.poses,
                                            self.min_invdepth, self.max_invdepth)
                    color.append(color_chunk)
                    depth.append(depth_chunk)
                color = torch.cat(color, 1)
                depth = torch.cat(depth, 1)

                print(color.min())

                # visualization
                color_v = color.view(B, self.depth_size[0], self.depth_size[1], 3).permute(0, 3, 1, 2)
                vis_color = vis_image(color_v[0], bgr = True)
                
                depth_v = depth.view(B, 1, self.depth_size[0], self.depth_size[1])
                vis = torch.cat([gt_invdepth, 1.0 / depth_v], dim = 2)
                vis_depth = vis_depth_image(vis[0])
            
                return vis_color, vis_depth
