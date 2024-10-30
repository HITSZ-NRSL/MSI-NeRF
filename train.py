import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
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
    
    if args.ckpts_epoch != -1:
        retrieve_ckpt = os.path.join(args.ckpts_dir, args.exp_name, 
                                     "epoch=" + str(args.ckpts_epoch) + ".ckpt")
        model = MInterface.load_from_checkpoint(retrieve_ckpt,
                                                args = args, ocams = ocams, poses = poses)
    else:
        model = MInterface(args, ocams, poses)
    
    ckpt_cb = ModelCheckpoint(dirpath=f'{args.ckpts_dir}/{args.exp_name}', 
                              filename='{epoch:d}', 
                              save_top_k=-1, 
                              save_on_train_epoch_end = True, 
                              every_n_epochs = args.every_n_epochs)
    
    pbar = TQDMProgressBar(refresh_rate=1)
    
    logger = TensorBoardLogger(save_dir = os.path.join(os.getcwd(), "logs"), 
                               name = args.exp_name, 
                               log_graph = False)
    
    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=False)
    
    trainer = Trainer(callbacks = [ckpt_cb, pbar], 
                      strategy = ddp, 
                      accelerator = args.accelerator, 
                      gpus = args.gpus, 
                      max_epochs = args.max_epochs, 
                      default_root_dir = args.default_root_dir, 
                      logger = logger, 
                      val_check_interval = args.val_check_interval, 
                      log_every_n_steps = args.log_every_n_steps, 
                      num_sanity_val_steps = args.num_sanity_val_steps)
    
    trainer.fit(model, data_model)