import pandas as pd
import os
from torchvision.io import read_image


class test_data(Dataset):
    def __init__(self, csv_file = "/home/sahmaran/data/ImageNet/LOC_val_solution.csv",
                 root_dir = "/home/sahmaran/data/ImageNet/ILSVRC/Data/CLS-LOC/val",
                 classes_dict = classes_dict):
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
        
        
