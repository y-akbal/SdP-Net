
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, disable_progress_bar
import datasets
import os
import torchvision.transforms.v2 as transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
from torch.utils.data import default_collate

disable_progress_bar()
datasets.logging.set_verbosity(datasets.logging.INFO)

def get_cache_dir():
    try:
        cache_dir = os.environ["HF_DATASETS_CACHE"]
    except KeyError:
        cache_dir = os.environ["HOME"]
    return cache_dir


def val_transforms(image_size = (320,320),
                   crop_size = (224,224),

        mean = [0.485, 0.456, 0.406], 
        std = [0.229, 0.224, 0.225]):

    transforms_val = transforms.Compose([
        transforms.RGB(),
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean, std)
    ])
    return transforms_val

def train_transforms(image_size = (224,224),
                    mean = [0.485, 0.456, 0.406], 
                    std = [0.229, 0.224, 0.225]):
        ### Here we define the transformation functions for training and testing, and maybe repeated augmentations!!!
    transforms_train = transforms.Compose([
        transforms.RGB(),
        transforms.RandomResizedCrop(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(),
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25)
    ])
    return transforms_train


class hf_dataset(Dataset):
    def __init__(self, 
                 huggingface_dataset, 
                 transform=None,
                 return_originals = False):
        
        self.dataset = huggingface_dataset
        self.transform = transform if transform else lambda x: x
        self.return_originals = return_originals
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
        if not self.return_originals:
            return transformed_image, label
        return transformed_image, image, label

"""dataset = load_dataset("timm/imagenet-22k-wds", streaming=True)
for a in dataset:
    print(a)
"""

"""
from datasets import load_dataset
dset = load_dataset('imagenet-1k', 
                    trust_remote_code=True, num_proc = 4)


transforms_ = train_trainsforms()
dset["train"].set_transform(transforms_)
ds = hf_dataset(dset["validation"], transform = val_transforms())




NUM_CLASSES = 1000
cutmix = v2.CutMix(num_classes=NUM_CLASSES)
mixup = v2.MixUp(num_classes=NUM_CLASSES)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])


def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))

data_loader = DataLoader(ds, batch_size=128, pin_memory=True, num_workers=12, shuffle = True, prefetch_factor=8, persistent_workers=False, drop_last = True)

from torch import nn as nn
from torchvision.models import get_model, list_models
from torch import optim
list_models()


model = get_model("swin_b", weights='IMAGENET1K_V1').cuda()
model.eval()
model.state_dict()

#model = torch.compile(model)
optimizer = optim.AdamW(model.parameters(), lr=0.000001)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.0)
from model import main_model
model = main_model(embedding_dim = 768,
                   n_head = 24, conv_kernel_size=9, patch_size=32, max_image_size=[32,32],
                   num_blocks = 15, 
                   stochastic_depth= False).cuda()
model.eval()
x = torch.randn(256, 3, 320, 320).cuda()
from training_utilities import MeasureTime

with torch.no_grad():
    with MeasureTime():
        output = model(x)

len(ds)
#model = torch.compile(model)
# Creates a GradScaler once at the beginning of training.
scaler = torch.GradScaler()


for i in range(100):
    local_loss = 0.0
    local_truth = 0
    counter = 0


    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for x, y in data_loader:
        
        x, y = x.to("cuda", non_blocking = True), y.to("cuda", non_blocking = True)
        start.record()
        optimizer.zero_grad(set_to_none=True)
        # Runs the forward pass with autocasting.
        with torch.inference_mode():
            output = model(x)
            loss = loss_fn(output,y)
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        #scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        #scaler.step(optimizer)
        #optimizer.zero_grad(set_to_none=True)

        # Updates the scale for next iteration.
        #scaler.update()
        


        local_loss += loss.item()
        local_truth += (output.argmax(-1) == y).sum().item()

        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
 
        print(x.shape, y.shape, counter, loss)

        print(start.elapsed_time(end)/1000, len(data_loader))

        counter += 1

    print(local_truth/len(ds))
    break
"""

def hf_train_val_data_loader(**kwargs):
    ### 
    ### This dude prepares the training and validation data ###
    ### 
    ###
    cache_dir = get_cache_dir()
    print(f"The datasets is to be cached at {cache_dir}")

    dset = load_dataset('imagenet-1k', 
                        keep_in_memory=False,
                        cache_dir = get_cache_dir(),
                        num_proc = 4, 
                        )
    

    dset_train, dset_test = dset["train"], dset["validation"]    
    train_crop_size, val_image_size, val_crop_size = kwargs["train_image_size"], kwargs["val_image_size"], kwargs["val_crop_size"]

    try:
        NUM_CLASSES = kwargs["Num_Classes"]
    except Exception:
        NUM_CLASSES = 1000


    train_transforms_, val_transforms_ = train_transforms(image_size = train_crop_size), val_transforms(image_size = val_image_size, crop_size = val_crop_size)

    dset_train, dset_test = hf_dataset(dset_train, train_transforms_), hf_dataset(dset_test, val_transforms_)

    kwargs_train = kwargs["train_data_details"]
    kwargs_test = kwargs["val_data_details"]
    ##
    train_sampler = DistributedSampler(dset_train, shuffle = True)
    val_sampler = DistributedSampler(dset_test, shuffle = False)
    ## 
    ## --- MixUp and CutMix --- ##
    ## 

    cutmix = v2.CutMix(num_classes = NUM_CLASSES,)
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

