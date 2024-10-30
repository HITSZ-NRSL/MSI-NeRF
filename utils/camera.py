import os
import torch
from ocam import OcamCamera
from scipy.spatial.transform import Rotation as Rot

def load_cam_intrinsic(data_dir, data_type, fov):
    if data_type == "synthetic_urban":
        data_type = "sunny"
    data_dir = os.path.join(data_dir, data_type)
    ocams = []
    for i in range(1, 5):
        key = f'cam{i}'
        ocam_file = os.path.join(data_dir, f'o{key}.txt')
        ocams.append(OcamCamera(ocam_file, fov))

    return ocams

def load_cam_extrinsic(data_dir, data_type):
    if data_type == "synthetic_urban":
        data_type = "sunny"
    data_dir = os.path.join(data_dir, data_type)
    # camera to world transform
    poses = []
    pose_file = os.path.join(data_dir, "poses.txt")
    with open(pose_file) as f:
        data = f.readlines()
        
    for pose in data:
        pose = list(map(float, pose.split()))
        T = torch.eye(4)
        angle = pose[:3]
        trans = pose[3:]
        trans[2] += 20.0
        R = Rot.from_rotvec(angle).as_matrix()
        T[:3,:3] = torch.from_numpy(R)
        T[:3, 3] = torch.tensor(trans) / 100.0
        
        # world to camera transform
        T = T.inverse()
        poses.append(T)
        
    return poses

