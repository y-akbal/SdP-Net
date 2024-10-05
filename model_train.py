import os
import sys
os.environ["OMP_NUM_THREADS"] = "5"

import torch
from torch import nn as nn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.distributed import init_process_group, destroy_process_group
### end of torch ### 
import hydra
from omegaconf import DictConfig
from model import MainModel
from training_tools import Trainer, return_scheduler_optimizer
from hf_dataset_generator import hf_train_val_data_loader
from training_utilities import track_accuracy, distributed_loss_track
import wandb
# 
torch.set_float32_matmul_precision("medium")

"""try:
    torch.set_default_dtype(torch.bfloat16)
    print("Default dtype is torch.bfloat16")
except Exception as e:
    print("Something wrong the default dtype is torch.float16")
    torch.set_default_dtype(torch.float16)
"""

## We replaced function with DDP setup, (in which case we may use FSDP!!!)
class DDP_setup(object):
    def __init__(self, backend = "nccl"):
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.backend = backend
    
    def __enter__(self):
        init_process_group(backend=self.backend)

    def __exit__(self, *args):
        destroy_process_group()


@hydra.main(version_base=None, config_path=".", config_name="model_config_vit")
def main(cfg : DictConfig):
    ## We will do everything with the following context window
    with DDP_setup():
        ## model configuration ##
        model_config, optimizer_scheduler_config = cfg["model_config"], cfg["optimizer_scheduler_config"]
        trainer_config = cfg["trainer_config"]
        data_config = cfg["data"]
        ## --- ### 
        ## model_config -- optimizer config -- scheduler config ##
        torch.manual_seed(231424314)
        model = MainModel.from_dict(**model_config)
        optimizer, scheduler = return_scheduler_optimizer(model, **optimizer_scheduler_config)
        ## batched train and validation data loader ## 
        if bool(cfg["DEBUG_MODE"]):
            from dataset_generator import fake_data_loader
            train_images, test_images = fake_data_loader()
        else:
            train_images, test_images = hf_train_val_data_loader(**data_config)
    
        gpu_id = int(os.environ["LOCAL_RANK"]) ### this local rank is determined by torch run!!!
    
        if gpu_id == 0:
            print(f"One epoch #batches {len(train_images)}, test #batch {len(test_images)}")
            print(f"Model has {model.return_num_params()} params. There are {torch.cuda.device_count()} GPUs available on this machine!!!")
            print(f"Current setup is {model_config}")
    
        train_loss_tracker = distributed_loss_track(task="Train")
        val_loss_tracker = distributed_loss_track(task="Validation")
        val_acc_tracker = track_accuracy()
    
        trainer = Trainer(
            model = model,
            train_data= train_images,
            val_data = test_images,
            optimizer = optimizer,
            scheduler= scheduler,
            gpu_id = gpu_id,
            val_loss_logger=val_loss_tracker,
            train_loss_logger=train_loss_tracker,
            val_accuracy_logger=val_acc_tracker,
            **trainer_config
        )
        
        wandb.init(project = "Tiny-SdpNet", config = dict(cfg))
        trainer.train()



if __name__ == '__main__':
    main()