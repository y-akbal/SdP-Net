####  TODO Download the dataset using huggingface api
### using dataset class wrap it with torch dataloader
### use dataset_generator
### do this for both validation and train
### I would like to use collate_fn in the dataloader because of the mixup and cutmix---

import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import os
import torchvision.transforms.v2 as transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
from torch.utils.data import default_collate



def get_cache_dir():
    try:
        cache_dir = os.environ["HF_DATASETS_CACHE"]
    except KeyError:
        cache_dir = os.environ["HOME"]
    return cache_dir


def val_transforms(crop_size = (224,224),
        mean = [0.485, 0.456, 0.406], 
        std = [0.229, 0.224, 0.225]):

    transforms_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean, std)
    ])
    return transforms_val

def train_trainsforms(crop_size = (224,224),
                    mean = [0.485, 0.456, 0.406], 
                    std = [0.229, 0.224, 0.225]):
        ### Here we define the transformation functions for training and testing
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandAugment(), ## RandAugment ---
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(),
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean, std)
    ])
    return transforms_train

class hf_dataset(Dataset):
    def __init__(self, 
                 huggingface_dataset, 
                 transform=None):
        
        self.dataset = huggingface_dataset
        self.transform = transform
        ### The question is to whether mix the transformations or not!
        ### Or maybe do something like n choose k kinda thing???

    def __len__(self):
        return len(self.dataset)
    
    @classmethod
    def load_dataset(cls, transform = None, **kwargs):
        dset = load_dataset(**kwargs)
        return cls(dset, transform)

    def __getitem__(self, idx):

        image = self.dataset[idx]['image']
        label = self.dataset[idx]['label'] 

        if self.transform:
            transformed_image = self.transform(image)

        return transformed_image, label


def batch_collate_function(batch):
    pass

def train_val_data_loader(train_data, test_data, **kwargs):
    ### This dude prepares the training and validation data ###
    root_dir_train = kwargs["train_path"]["root_dir"]
    root_dir_val = kwargs["val_path"]["root_dir"]
    csv_file_val = kwargs["val_path"]["csv_file"]
    ##
    train_image_generator, dict_val = train_data(root_dir = root_dir_train)
    test_image_generator = test_data(root_dir = root_dir_val,
              csv_file = csv_file_val,
              classes_dict = dict_val
    )
    ##
    kwargs_train = kwargs["train_data_details"]
    kwargs_test = kwargs["val_data_details"]
    ##
    train_sampler = DistributedSampler(train_image_generator, shuffle = True)
    val_sampler = DistributedSampler(test_image_generator, shuffle = False)
    ## --- MixUp and CutMix --- ##
    
    NUM_CLASSES = 1000
    cutmix = v2.CutMix(num_classes=NUM_CLASSES)
    mixup = v2.MixUp(num_classes=NUM_CLASSES, alpha = 0.8)

    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    collate_fn = lambda batch : cutmix_or_mixup(*default_collate(batch))


    train_data = DataLoader(
        dataset= train_image_generator,
        sampler = train_sampler,
        collate_fn=collate_fn,
        **kwargs_train,
    )
    test_data = DataLoader(
        dataset= test_image_generator,
        sampler = val_sampler,
        **kwargs_test,
    )
    
    return train_data, test_data

