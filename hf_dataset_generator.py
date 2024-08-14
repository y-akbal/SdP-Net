####  TODO Download the dataset using huggingface api
### using dataset class wrap it with torch dataloader
### use dataset_generator
### do this for both validation and train
import datasets
from torch.utils.data import DataLoader

CACHE_DIR = "/Users/yildirimakbal/Desktop"

from datasets import load_dataset
dset = load_dataset('imagenet-1k', cache_dir = CACHE_DIR, streaming=True).with_format("torch")

dset_ = DataLoader(dataset = dset["train"], batch_size = 16)

for x in dset_:
    print(x)