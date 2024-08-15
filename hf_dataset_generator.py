####  TODO Download the dataset using huggingface api
### using dataset class wrap it with torch dataloader
### use dataset_generator
### do this for both validation and train
### I would like to use collate_fn in the dataloader because of the mixup and cutmix---

import datasets
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import os

def get_cache_dir():
    try:
        cache_dir = os.environ["HF_DATASETS_CACHE"]
    except KeyError:
        cache_dir = os.environ["HOME"]
    return cache_dir


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

