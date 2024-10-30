import cv2
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

def vis_depth_image(depth, min_val = None, max_val = None, totensor = False, cmap=cv2.COLORMAP_JET):
    if torch.is_tensor(depth):
        depth = depth.detach().cpu().numpy().squeeze()
    depth = np.nan_to_num(depth)
    
    if min_val == None:
        min_val = depth.min()
    if max_val == None:
        max_val = depth.max()
    depth = (depth - min_val) / (max_val-min_val + 1e-8)
    depth = (255 * depth).astype(np.uint8)
    depth_cmap = cv2.applyColorMap(depth, cmap)
    
    if totensor:
        depth_cmap = cv2.cvtColor(depth_cmap, cv2.COLOR_BGR2RGB)
        depth_cmap = Image.fromarray(depth_cmap)
        depth_cmap = transforms.ToTensor()(depth_cmap)
    
    return depth_cmap

def vis_image(image, bgr = False, totensor = False):
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy().squeeze().transpose((1, 2, 0))
    
    if bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
        
    if totensor:
        image = Image.fromarray(image)
        image = transforms.ToTensor()(image)
        
    return image

def vis_3D_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()