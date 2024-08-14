####  TODO Download the dataset using huggingface api
### using dataset class wrap it with torch dataloader
### use dataset_generator
### do this for both validation and train
### I would like to use collate_fn in the dataloader because of the mixup and cutmix---

import datasets
from torch.utils.data import DataLoader, Dataset

CACHE_DIR = "/Users/yildirimakbal/Desktop"

from datasets import load_dataset
dset = load_dataset('imagenet-1k', cache_dir = CACHE_DIR, streaming=True).with_format("torch")

class hf_dataset(Dataset):
    def __init__(self, 
                 huggingface_dataset, 
                 transform=None):
        
        self.dataset = huggingface_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        image = self.dataset[idx]['image']
        label = self.dataset[idx]['label'] 

        if self.transform:
            image = self.transform(image)

        return image, label

