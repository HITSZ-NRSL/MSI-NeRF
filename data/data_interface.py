import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from data.dataset import Dataset, NVSDataset

class DInterface(LightningDataModule):

    def __init__(self, args, ocams, poses):
        super().__init__()
        self.data_path = args.data_path
        self.nvs_data_path = args.nvs_data_path
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.split_ratio = args.split_ratio
        self.data_type = args.data_type
        self.image_size = args.image_size
        self.depth_size = args.depth_size
        self.depth_layer = args.depth_layer
        self.min_depth = args.min_depth
        self.ocams = ocams
        self.poses = poses
        self.permute_aug = args.permute_aug
        self.rot_aug = args.rot_aug
        self.render_novel_view = args.render_novel_view
        if self.render_novel_view:
            self.test_num = args.test_num
        else:
            self.test_num = 0
        self.eval_nvs = args.eval_nvs
        
    def setup(self, stage = None):
        if stage == "fit":
            self.train_dataset = Dataset(self.data_path, self.data_type, self.image_size, self.depth_size, self.depth_layer,
                                         self.min_depth, "train", self.split_ratio, self.ocams, self.poses, 
                                         self.permute_aug, self.rot_aug)
            self.val_dataset = Dataset(self.data_path, self.data_type, self.image_size, self.depth_size, self.depth_layer,
                                       self.min_depth, "val", self.split_ratio, self.ocams, self.poses)
        else:
            if self.eval_nvs:
                # scenes = ["apartment_0", "apartment_1", "apartment_2", 
                #           "frl_apartment_4", "frl_apartment_5", "hotel_0", 
                #           "office_0", "office_1", "office_2", "office_3", 
                #           "room_0", "room_1", "room_2", "frl_apartment_0", 
                #           "frl_apartment_1", "frl_apartment_2", "frl_apartment_3"]
                scenes = ["apartment_0"]
                self.test_dataset = NVSDataset(self.nvs_data_path, scenes, self.image_size, self.depth_size, self.depth_layer,
                                               self.min_depth, self.ocams, self.poses, self.test_num)
                print(len(self.test_dataset))
            else:
                self.test_dataset = Dataset(self.data_path, self.data_type, self.image_size, self.depth_size, self.depth_layer,
                                            self.min_depth, "val", self.split_ratio, self.ocams, self.poses, self.test_num)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, 
                          num_workers = self.num_workers, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, 
                          num_workers = self.num_workers, shuffle = False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, 
                          num_workers = self.num_workers, shuffle = False)
        
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, 
                          num_workers = self.num_workers, shuffle = False)
        