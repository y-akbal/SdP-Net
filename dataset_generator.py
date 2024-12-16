import pandas as pd
import os
from torchvision.io import read_image
from PIL import Image 
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms
import torch
### Gatto define your transforms we may jit'em if needed 
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
from torch.utils.data import default_collate



def return_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    ### Here we define the transformation functions for training and testing
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop((224,224)),
        transforms.RandAugment(), ## RandAugment ---
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(),
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean, std)
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean, std)
    ])
    return transforms_train, transforms_test

transforms_train, transforms_test = return_transforms()

def train_data(root_dir:str, 
            transformations = transforms_train):
    
    
    Images = datasets.ImageFolder(root = root_dir,
                                  transform = transformations,
    )
    dict_ = Images.class_to_idx
    ### The dictionary above is pretty important as this will 
    return Images, dict_


class test_data(Dataset):
    def __init__(self, 
                 classes_dict:dict,
                 csv_file:str,
                 root_dir:str,
                 transformations = transforms_test,
                 ):
        super().__init__()
        self.root_dir = root_dir
        self.classes_dict = classes_dict
        ###
        self.file = pd.read_csv(csv_file)
        self.file_names = self.file.iloc[:,0]
        self.anotations = self.file.iloc[:,1].apply(self.__split__)
        ###
        self.transformations = transformations
        
    def __len__(self):
        return len(self.anotations)

    def __getitem__(self, index):
        ## First images
        image = os.path.join(self.root_dir, self.file_names[index]+ ".JPEG")
        ### We need to test where .rgb method introduces some latency in the case
        ### that the image already has 3 channels!!!
        image_ = Image.open(image).convert('RGB') 
        transformed_image = self.transformations(image_)
        ## now the labels
        anotations = self.anotations[index]
        classes = self.classes_dict[anotations]
        return transformed_image, classes, anotations
    def __split__(self, n):
        return n.split()[0]


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

def fake_data_loader(train_samples:int = 10000, 
                     val_samples:int = 500, **kwargs):  
    ## This is a fake dataset generator for debugging purposes
    train_data = torch.randn(train_samples, 3, 224, 224)
    train_labels = torch.randint(low = 0, high = 1000, size = (train_samples,))
    val_data = torch.randn(val_samples, 3, 224, 224)
    val_labels = torch.randint(low = 0, high = 1000, size = (val_samples,))

    class FakeDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            image, label = self.data[index], self.labels[index]
            return image, label

    train_dataset = FakeDataset(train_data, train_labels)
    val_dataset = FakeDataset(val_data, val_labels)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)    

    train_data_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size = 32, **kwargs)
    val_data_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size = 32, **kwargs)

    return train_data_loader, val_data_loader


if __name__ == '__main__':
    print("Ok boomer!!!")

