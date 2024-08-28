####  TODO Download the dataset using huggingface api
### using dataset class wrap it with torch dataloader
### use dataset_generator
### do this for both validation and train
### I would like to use collate_fn in the dataloader because of the mixup and cutmix---

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
        self.transform = transform if transform else lambda x: x
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

        transformed_image = self.transform(image)
        return transformed_image, label


import datasets

from datasets import load_dataset
dset = load_dataset('imagenet-1k', 
                    trust_remote_code=True, 
                                        use_auth_token=True, cache_dir = "/media/sahmaran/60E6D899E6D870B0/IMGNET")

transforms_val = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToImage(), 
    ])


a = set()
for x in dset["validation"]:
    a.add(x["cls"])
    print(x)

ds = hf_dataset(dset["test"], transforms_val)    


import numpy as np

train_x = np.memmap("test_x.dat", dtype = np.uint8, shape = (len(ds), 3, 64, 64), mode = "w+")
train_y = np.memmap("test_y.dat", dtype = np.int64, shape = (len(ds), ), mode = "w+")

import tqdm

q = 0
for i, image_dat in enumerate(tqdm.tqdm(ds)):
    x,y = image_dat
    if x.ndim < 3:
      q+= 1
    train_x[i], train_y[i] = x[0].numpy(), y
    if i % 500 == 0:
        train_x.flush()
        train_y.flush()
    



"""

for i in range(1010000000):
    print(hf_dataset(dset, transforms_)[i])

ds = hf_dataset(dset["train"], transforms_)


NUM_CLASSES = 1000
cutmix = v2.CutMix(num_classes=NUM_CLASSES)
mixup = v2.MixUp(num_classes=NUM_CLASSES)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])


def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))

data_loader = DataLoader(ds, batch_size= 128, pin_memory=True, num_workers=8, collate_fn = collate_fn)
for  i,(x,y) in enumerate(data_loader):
    print(y[0,:].max(), 1-y[0,:].max(), i, y.shape)

"""




def hf_train_val_data_loader(**kwargs):
    ### 
    ### This dude prepares the training and validation data ###
    ### 
    ###
    cache_dir = get_cache_dir()
    print(f"The datasets is to be cached at {cache_dir}")
    dset = load_dataset('imagenet-1k', 
                    cache_dir = get_cache_dir())
    
    train_trainsforms_, val_transforms_ = train_trainsforms(), val_transforms()

    dset_train, dset_test = dset["train"], dset["test"]    
    dset_train, dset_test = hf_dataset(dset_train, train_trainsforms_), hf_dataset(dset_test, val_transforms_)

    kwargs_train = kwargs["train_data_details"]
    kwargs_test = kwargs["val_data_details"]
    ##
    train_sampler = DistributedSampler(dset_train, shuffle = True)
    val_sampler = DistributedSampler(dset_test, shuffle = False)
    ## 
    ## --- MixUp and CutMix --- ##
    ## 
    NUM_CLASSES = 1000

    cutmix = v2.CutMix(num_classes = NUM_CLASSES)
    mixup = v2.MixUp(num_classes = NUM_CLASSES, alpha = 0.8)

    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    collate_fn = lambda batch : cutmix_or_mixup(*default_collate(batch))


    train_data = DataLoader(
        dataset= dset_train,
        sampler = train_sampler,
        collate_fn = collate_fn,
        **kwargs_train,
    )
    test_data = DataLoader(
        dataset= dset_test,
        sampler = val_sampler,
        **kwargs_test,
    )
    
    return train_data, test_data

