import torch
import torch.nn as nn

from model.common import *

class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()
        self.loss_func = torch.nn.L1Loss()

    def forward(self, depth, gt_invdepth):
        invdepth = 1.0 / depth
        loss = self.loss_func(gt_invdepth, invdepth)
        
        return loss

class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.loss_func = torch.nn.L1Loss()
    
    def forward(self, colors, images):
        loss = 0
        for color_num, color in enumerate(colors):
            image = images[color_num]
            loss += self.loss_func(image, color)
        
        return loss