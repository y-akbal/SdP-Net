import pandas as pd
import os
from torchvision.io import read_image
from PIL import Image 
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
### Gotto define your transforms we may jit'em if needed 



def return_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    ### Here we define the transformation functions for training and testing
    
    transforms_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
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
        self.samples:int = 0
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
        

"""
#train_set test ok
loc = "~/Desktop/ImageNet/ILSVRC/Data/CLS-LOC/train"
I, dict_= train_data(root_dir = loc)
I[0][0].shape == 3,224,224  


col = test_data(classes_dict = dict_,
          csv_file="~/Desktop/ImageNet/LOC_val_solution.csv",
          root_dir="/home/sahmaran/Desktop/ImageNet/ILSVRC/Data/CLS-LOC/val"
          )
"""

if __name__ == '__main__':
    print("Ok boomer")