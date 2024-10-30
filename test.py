import cv2
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from utils import config
from utils.camera import load_cam_intrinsic, load_cam_extrinsic
from model import MInterface
from data import DInterface

if __name__ == "__main__":
    args = config.load_parser()
    pl.seed_everything(args.seed)
    
    ocams = load_cam_intrinsic(args.data_path, args.data_type, args.fov)
    poses = load_cam_extrinsic(args.data_path, args.data_type)
    
    data_model = DInterface(args, ocams, poses)
    if args.version != 0:
        retrieve_ckpt = os.path.join(args.ckpts_dir, args.exp_name, 
                                     "epoch=" + str(args.ckpts_epoch) + "-v" + str(args.version) + ".ckpt")
    else:
        retrieve_ckpt = os.path.join(args.ckpts_dir, args.exp_name, 
                                     "epoch=" + str(args.ckpts_epoch) + ".ckpt")
        
    model = MInterface.load_from_checkpoint(retrieve_ckpt,
                                            args = args, ocams = ocams, poses = poses)

    model.eval()
    
    logger = TensorBoardLogger(save_dir = os.path.join(os.getcwd(), "logs"), 
                               name = args.exp_name, 
                               log_graph = False)
    
    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=False)
    
    trainer = Trainer(strategy = ddp, 
                      accelerator = args.accelerator, 
                      gpus = args.gpus, 
                      max_epochs = args.max_epochs, 
                      default_root_dir = args.default_root_dir, 
                      logger = logger, 
                      val_check_interval = args.val_check_interval, 
                      log_every_n_steps = args.log_every_n_steps)
    
    pred_dir = os.path.join(args.prediction_dir, args.exp_name)
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
            
    preds = trainer.predict(model, data_model)
    
    if args.render_novel_view:
        for pred_num, pred in enumerate(preds):
            novel_vis_colors, novel_vis_depths = pred
            color_dir = os.path.join(pred_dir, str(pred_num), "color")
            if not os.path.exists(color_dir):
                os.makedirs(color_dir)
            depth_dir = os.path.join(pred_dir, str(pred_num), "depth")
            if not os.path.exists(depth_dir):
                os.makedirs(depth_dir)
                
            for frame_num in range(len(novel_vis_colors)):
                novel_vis_color = novel_vis_colors[frame_num]
                novel_vis_depth = novel_vis_depths[frame_num]
                
                pred_file = os.path.join(color_dir, "{}.jpg".format(frame_num))
                cv2.imwrite(pred_file, novel_vis_color)
                
                pred_file = os.path.join(depth_dir, "{}.jpg".format(frame_num))
                cv2.imwrite(pred_file, novel_vis_depth)
    
    else:
        if args.eval_nvs:

            for pred_num, pred in enumerate(preds):
                pred_color, pred_depth = pred

                for i in range(len(pred_color)):
                    
                    color_dir = os.path.join(pred_dir, str(i), "color")
                    if not os.path.exists(color_dir):
                        os.makedirs(color_dir)
                    depth_dir = os.path.join(pred_dir, str(i), "depth")
                    if not os.path.exists(depth_dir):
                        os.makedirs(depth_dir)

                    pred_file = os.path.join(color_dir, "{}.jpg".format(pred_num))
                    cv2.imwrite(pred_file, pred_color[i])
                    pred_file = os.path.join(depth_dir, "{}.jpg".format(pred_num))
                    cv2.imwrite(pred_file, pred_depth[i])

        else:
            color_dir = os.path.join(pred_dir, "color")
            if not os.path.exists(color_dir):
                os.makedirs(color_dir)
            depth_dir = os.path.join(pred_dir, "depth")
            if not os.path.exists(depth_dir):
                os.makedirs(depth_dir)
            
            for pred_num, pred in enumerate(preds):
                pred_color, pred_depth = pred
                
                pred_file = os.path.join(color_dir, "{}.jpg".format(pred_num))
                cv2.imwrite(pred_file, pred_color)
                
                pred_file = os.path.join(depth_dir, "{}.jpg".format(pred_num))
                cv2.imwrite(pred_file, pred_depth)
