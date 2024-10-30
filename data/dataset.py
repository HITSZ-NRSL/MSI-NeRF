import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import random
import cv2
import torch
import torch.utils.data as data
import torchvision
from PIL import Image
import numpy as np

from utils.geometry import *

class Dataset(data.Dataset):
    def __init__(self, data_dir, data_type, image_size, depth_size, depth_layer, min_depth, 
                 split, split_ratio, ocams, poses, test_num = 0, permute_aug = False, rot_aug = False):
        super(Dataset, self).__init__()
        
        self.data_dir = data_dir
        self.image_size = image_size
        self.depth_size = depth_size
        self.depth_layer = depth_layer
        self.min_depth = min_depth
        self.max_invdepth = 1 / self.min_depth
        self.max_depth = 1 / EPS
        self.min_invdepth = 1 / self.max_depth
        self.ocams = ocams
        self.poses = poses
        self.test_num = test_num
        
        self.permute_aug = permute_aug
        self.rot_aug = rot_aug
        self.train = split == "train"
        
        if data_type == "omnithings":
            self.zfill_len = 5
            data_len = 10240
            sub_dataset_list = ["omnithings"]
        elif data_type == "omnihouse":
            data_len = 2560
            self.zfill_len = 4
            sub_dataset_list = ["omnihouse"]
        elif data_type == "synthetic_urban":
            data_len = 1000
            self.zfill_len = 4
            sub_dataset_list = ["sunny",
                                "cloudy",
                                "sunset"]
        
        self.start, self.end, self.step = 1, data_len, 1
        full_idx = list(range(self.start, self.end + 1, self.step))
        train_num = int(split_ratio * len(full_idx))
        
        if split == "train":
            idx = full_idx[:train_num]
        elif split == "val":
            if self.test_num == 0:
                idx = full_idx[train_num:]
            else:
                idx = random.sample(full_idx[train_num:], self.test_num)
        
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self.image_size)
        ])
        self.depth_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self.depth_size)
        ])
        
        self.item_list = []
        for sub_dataset in sub_dataset_list:
            for image_num in idx:
                self.item_list.append((sub_dataset, image_num))
        
        self.grid = self.generate_sweep_grid()
    
    def generate_sweep_grid(self):
        rays_d = GenEqrRays(self.depth_size[1], self.depth_size[0])
        # / 2.0 to match the feature map size
        invdepth_step = (self.max_invdepth - self.min_invdepth) / (int(self.depth_layer / 2) - 1)
        invdepth = torch.arange(self.min_invdepth, self.max_invdepth + invdepth_step - EPS, invdepth_step)
        depth = 1.0 / invdepth
        
        xyz = depth.view(-1, 1, 1) * rays_d.view(1, -1, 3)
        xyz = xyz.view(-1, 3)
 
        grids_2D = []
        for cam_num, ocam in enumerate(self.ocams):
            pose = self.poses[cam_num]
            xyz_cam = TransformPoints(xyz, pose)
            
            # / 2.0 to match the feature map size
            ratio = self.image_size[0] / 2.0 / ocam.height
            grid_2D, mask = FishProject(xyz_cam, ocam, ratio)

            grid_2D = grid_2D.view(int(self.depth_layer / 2), self.depth_size[0], self.depth_size[1], 2)
            mask = mask.view(int(self.depth_layer / 2), self.depth_size[0], self.depth_size[1])
        
            grid_2D[..., 0] = 2 * grid_2D[..., 0] / (self.image_size[1] / 2.0) - 1.0
            grid_2D[..., 1] = 2 * grid_2D[..., 1] / (self.image_size[0] / 2.0) - 1.0
            grid_2D[mask] = 2.0
            
            grids_2D.append(grid_2D)
            
        return grids_2D

    def __getitem__(self, index):
        images = []
        sub, idx = self.item_list[index]
        
        for cam_num in range(1, 5):
            image_file = os.path.join(self.data_dir, sub, f"cam{cam_num}",
                                      str(idx).zfill(self.zfill_len) + ".png")
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image)
            images.append(image)
        
        invdepth_file = os.path.join(self.data_dir, "omnidepth_gt_640",
                                     str(idx).zfill(5) + ".tiff")
        with Image.open(invdepth_file) as tiff:
            tiff.seek(1)
            invdepth = np.array(tiff) * 3.0
        invdepth = self.depth_transform(invdepth)
        
        grid = self.grid
        
        # randomly permute
        aug_ind = [0, 1, 2, 3]
        if self.permute_aug and self.train:
            random.shuffle(aug_ind)
            images = [images[i] for i in aug_ind]
            grid = [grid[i] for i in aug_ind]
        
        images = torch.stack(images, 0)
        grid = torch.stack(grid, 0)

        # random rotate coordinate yaw
        aug_rot = 0
        if self.rot_aug and self.train:
            aug_rot = random.randint(0, self.depth_size[1] - 1)
            grid = torch.cat([grid[:,:,:,aug_rot:,:], grid[:,:,:,:aug_rot,:]], dim = 3)
            invdepth = torch.cat([invdepth[:,:,aug_rot:], invdepth[:,:,:aug_rot]], dim = 2)
        
        return images, invdepth, grid, aug_ind, aug_rot
    
    def __len__(self):
        return len(self.item_list)

class NVSDataset(data.Dataset):
    def __init__(self, data_dir, scenes, fisheye_size, depth_size, 
                 depth_layer, min_depth, ocams, poses, test_num = 0):
        super(NVSDataset, self).__init__()
        
        self.data_dir = data_dir
        self.scenes = scenes
        self.image_size = fisheye_size
        self.depth_size = depth_size
        self.depth_layer = depth_layer
        self.min_depth = min_depth
        self.max_invdepth = 1 / self.min_depth
        self.max_depth = 1 / EPS
        self.min_invdepth = 1 / self.max_depth
        self.ocams = ocams
        self.poses = poses
        self.test_num = test_num
        
        self.item_list = []
        for scene in self.scenes:
            scene_dir = os.path.join(self.data_dir, scene)
            scene_eqr_dir = os.path.join(scene_dir, "eqr")
            
            scene_eqr_1_dir = os.path.join(scene_eqr_dir, "1", "pose")
            for sample_num, _ in enumerate(os.listdir(scene_eqr_1_dir)):
                self.item_list.append((scene, sample_num+1))
        
        if self.test_num > 0:
            self.item_list = random.sample(self.item_list, self.test_num)
        
        self.fe_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self.image_size)
        ])
        self.eqr_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self.depth_size)
        ])
        
        self.cam_trans = torch.tensor([[1, 0, 0, 0], 
                                       [0, -1, 0, 0], 
                                       [0, 0, -1, 0.2], 
                                       [0, 0, 0, 1]])
        
        self.eqr_num = len(os.listdir(scene_eqr_dir))
        self.grids = self.generate_sweep_grid()
    
    def generate_sweep_grid(self):
        rays_d = GenEqrRays(self.depth_size[1], self.depth_size[0])
        # / 2.0 to match the feature map size
        invdepth_step = (self.max_invdepth - self.min_invdepth) / (int(self.depth_layer / 2) - 1)
        invdepth = torch.arange(self.min_invdepth, self.max_invdepth + invdepth_step - EPS, invdepth_step)
        depth = 1.0 / invdepth
        
        xyz = depth.view(-1, 1, 1) * rays_d.view(1, -1, 3)
        xyz = xyz.view(-1, 3)
 
        grids_2D = []
        for cam_num, ocam in enumerate(self.ocams):
            pose = self.poses[cam_num]
            xyz_cam = TransformPoints(xyz, pose)
            
            # / 2.0 to match the feature map size
            ratio = self.image_size[0] / 2.0 / ocam.height
            grid_2D, mask = FishProject(xyz_cam, ocam, ratio)

            grid_2D = grid_2D.view(int(self.depth_layer / 2), self.depth_size[0], self.depth_size[1], 2)
            mask = mask.view(int(self.depth_layer / 2), self.depth_size[0], self.depth_size[1])
        
            grid_2D[..., 0] = 2 * grid_2D[..., 0] / (self.image_size[1] / 2.0) - 1.0
            grid_2D[..., 1] = 2 * grid_2D[..., 1] / (self.image_size[0] / 2.0) - 1.0
            grid_2D[mask] = 2.0
            
            grids_2D.append(grid_2D)
            
        return grids_2D

    def __getitem__(self, index):
        scene, scene_id = self.item_list[index]
        
        fisheyes = []
        for cam_num in range(1, 5):
            fe_file = os.path.join(self.data_dir, scene, "fisheye", f"cam{cam_num}", 
                                   "image", str(scene_id) + ".png")
            fisheye = cv2.imread(fe_file)
            fisheye = cv2.cvtColor(fisheye, cv2.COLOR_BGR2RGB)
            fisheye = self.fe_transform(fisheye)
            fisheyes.append(fisheye)
            
            if cam_num == 1:
                fe_pose_file = os.path.join(self.data_dir, scene, "fisheye", f"cam{cam_num}", 
                                            "pose", str(scene_id) + ".txt")
                fe_pose = np.loadtxt(fe_pose_file)
                fe_pose = torch.from_numpy(fe_pose).float().inverse()
                center_pose = self.cam_trans.mm(fe_pose)
        
        eqrs = []
        eqr_poses = []
        for eqr_id in range(0, self.eqr_num):
            eqr_file = os.path.join(self.data_dir, scene, "eqr", str(eqr_id), 
                                    "image", str(scene_id) + ".png")
            eqr = cv2.imread(eqr_file)
            eqr = cv2.cvtColor(eqr, cv2.COLOR_BGR2RGB)
            eqr = self.eqr_transform(eqr)
            eqrs.append(eqr)
            
            eqr_pose_file = os.path.join(self.data_dir, scene, "eqr", str(eqr_id),
                                         "pose", str(scene_id) + ".txt")
            eqr_pose = np.loadtxt(eqr_pose_file)
            eqr_pose = torch.from_numpy(eqr_pose).float()
            # eqr to world transform
            eqr_pose = center_pose.mm(eqr_pose)
            
            eqr_poses.append(eqr_pose)
        
        fisheyes = torch.stack(fisheyes, 0)
        eqrs = torch.stack(eqrs, 0)
        eqr_poses = torch.stack(eqr_poses, 0)
        grids = torch.stack(self.grids, 0)

        return fisheyes, eqrs, eqr_poses, grids
    
    def __len__(self):
        return len(self.item_list)