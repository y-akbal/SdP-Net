import os
import sys
os.environ["OMP_NUM_THREADS"] = "3"
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
from model import main_model
from training_tools import Trainer, return_scheduler_optimizer
from hf_dataset_generator import hf_train_val_data_loader
from training_utilities import track_accuracy, distributed_loss_track
#
torch.set_float32_matmul_precision("medium")

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
        torch.manual_seed(10)
        model = main_model.from_dict(**model_config)
        optimizer, scheduler = return_scheduler_optimizer(model, **optimizer_scheduler_config)
        ## batched train and validation data loader ## 
        train_images, test_images = hf_train_val_data_loader(**data_config)
    
        gpu_id = int(os.environ["LOCAL_RANK"]) ### this local rank is determined by torch run!!!
    
        if gpu_id == 0:
            print(f"One epoch #batches {len(train_images)}, test #batch {len(test_images)}")
            print(f"Model has {model.return_num_params()} params. There are {torch.cuda.device_count()} GPUs available on this machine!!!")
            print(f"Current setup is {model_config}")
    
        train_loss_tracker = distributed_loss_track()
        val_loss_tracker = distributed_loss_track()
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
        trainer.train()



if __name__ == '__main__':
    main()