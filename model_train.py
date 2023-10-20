### Here we go!!!
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import functional as F
from tqdm import tqdm

### --- ###
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
### --- ###

import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
### --- ###
import hydra
from omegaconf import DictConfig, OmegaConf

### import model and train and validation data ###
from model import main_model
from dataset_generator import test_data, train_data
## --- ###
from train_tools import trainer



def data_loader_train(**kwargs):
    return None
def data_loader_validation(**kwargs):
    return None



@hydra.main(version_base=None, config_path=".", config_name="model_config")
def main(cfg : DictConfig):
    ## model configuration ##
    model_config, optimizer_config, scheduler_config = cfg["model_config"], cfg["optimizer_config"], cfg["scheduler_config"]
    snapshot_path = cfg["snapshot_path"]
    save_every = cfg["save_every"]

    ## model_config -- optimizer config -- scheduler config ##
    torch.manual_seed(0)
    model = main_model.from_dict(**model_config)
    print(model(torch.randn(1, 3, 224,224)).shape)
    ## -- ##
    """
    ### We now do some data_stuff ###
    train_dataset, val_dataset = cfg["data"]["train_path"], cfg["data"]["val_path"]
    train_data_kwargs, val_data_kwargs = cfg["data"]["train_data_details"], cfg["data"]["val_data_details"]
    train_data = return_dataset(**train_dataset)
    validation_data = return_dataset(**val_dataset)
    train_dataloader = data_loader(train_data, **train_data_kwargs)
    val_dataloader = data_loader(validation_data, **val_data_kwargs)
    print(f"Note that the train dataset contains {len(train_dataloader)}! batches!!")
    ### --- End of data grabbing --- ###
    
    ### Optimizer ###
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
    ### Training Stuff Here it comes ###
    
    trainer = Trainer(model = model, 
            train_data= train_dataloader,
            val_data = val_dataloader,
            optimizer = optimizer, 
            scheduler = scheduler,
            save_every = save_every,
            snapshot_path=snapshot_path,
            compile_model=cfg["compile_model"],
            
    )
    
    trainer.train(max_epochs = 2)
    trainer.validate()


"""






if __name__ == '__main__':
    ## use here weights and biases!!!!
    ## use it properly dude!!!
    main()