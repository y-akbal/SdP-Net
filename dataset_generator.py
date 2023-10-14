import pandas as pd
import os
import torchvision
from torchvision.io import read_image
from torch.utils.data import Dataset

import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

### Gotto define your transforms we may jit'em if needed 



class test_data(Dataset):
    def __init__(self, 
                 classes_dict:dict,
                 csv_file:str,
                 root_dir:str,
                 ):
        super().__init__()
        self.samples:int = 0
        self.root_dir = root_dir
        self.classes_dict = classes_dict
        ###
        file = pd.read_csv(csv_file)
        self.file_names = file.iloc[:,0]
        self.anotations = file.iloc[:,1].apply(self.__split__)
        ###
        
    def __len__(self):
        return len(self.anotations)
    def __getitem__(self, index):
        image = os.path.join(self.root_dir, self.file_names[index]+ ".JPEG")
        
        return read_image(image), self.classes_dict[self.anotations[index]]
    def __split__(self, n):
        return n.split()[0]
        

def return_train_set(root_dir:str, transformations):
    Images = datasets.ImageFolder(root = root_dir,
                                  transform = transformations,

    )
    return Images, image


    