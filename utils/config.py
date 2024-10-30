import os
import configargparse

def load_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, default="config/default.txt")
    
    # Setting arguments
    parser.add_argument('--exp_name', type = str, default = "test_exp")
    parser.add_argument('--data_type', type = str, default = "omnithings")
    parser.add_argument('--data_path', type = str, default = "/home/nros/Data/OmniMVS_dataset/OmniThings")
    parser.add_argument('--nvs_data_path', type = str, default = "/home/nros/Data/Replica_360_clean")
    parser.add_argument('--ckpts_dir', type=str, default="ckpts/")
    parser.add_argument('--prediction_dir', type=str, default="predictions/")
    parser.add_argument('--default_root_dir', type = str, default = os.getcwd())
    
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--ckpts_epoch', type=int, default=-1)
    parser.add_argument('--version', type=int, default=-1)
    parser.add_argument('--max_epochs', type = int, default = 20)
    parser.add_argument('--batch_size', type = int, default = 2)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--split_ratio', type = float, default = 0.95)
    parser.add_argument('--val_check_interval', type = int, default = 2432)
    parser.add_argument('--log_every_n_steps', type = int, default = 10)
    parser.add_argument('--num_sanity_val_steps', type = int, default = 2)
    parser.add_argument('--every_n_epochs', type = int, default = 1)
    parser.add_argument('--permute_aug', action = "store_true", default = False)
    parser.add_argument('--rot_aug', action = "store_true", default = False)
    parser.add_argument('--render_novel_view', action = "store_true", default = False)
    parser.add_argument('--test_num', type = int, default = 10)
    parser.add_argument('--eval_nvs', action = "store_true", default = False)
    parser.add_argument('--traj_type', type = int, default = 0)
    
    # Network arguments
    parser.add_argument('--image_size', type = tuple, default = (384, 400))
    parser.add_argument('--depth_size', type = tuple, default = (256, 512))
    parser.add_argument('--novel_size', type = tuple, default = (300, 400))
    parser.add_argument('--novel_f', type = float, default = 200)
    parser.add_argument('--depth_layer', type = int, default = 64)
    parser.add_argument('--min_depth', type = float, default = 0.5)
    parser.add_argument('--fov', type = float, default = 220)
    parser.add_argument('--CH', type = int, default = 16)
    parser.add_argument('--out_CH', type = int, default = 16)
    
    # Trainer arguments
    parser.add_argument('--accelerator', type = str, default = "gpu")
    parser.add_argument('--gpus', type = int, default = 4)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--lr_scheduler', type = str, choices=["step", "cosine"], default="step")
    parser.add_argument('--lr_decay_steps', type = int, default = 15)
    parser.add_argument('--lr_decay_rate', type = float, default = 0.3)
    parser.add_argument('--lr_decay_min_lr', type = float, default = 1e-5)
    parser.add_argument('--depth_weight', type = float, default = 5.0)
    parser.add_argument('--eqr_sample_num', type = int, default = 16384)
    parser.add_argument('--fish_sample_num', type = int, default = 4096)
    parser.add_argument('--val_chunk_size', type = int, default = 65536)
    
    args = parser.parse_args()
    
    return args
