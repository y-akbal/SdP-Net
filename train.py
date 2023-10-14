### Here we go!!!
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import functional as F
from tqdm import tqdm
### --- ###
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

## --- ##

import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

## --- ###
## We will do DDP training here!!!



from model import main_model

model = main_model(conv_mixer_repetation=10, transformer_encoder_repetation=10, patch_size=4, multiplication_factor=1, squeeze_ratio=4).cuda()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.001, momentum = 0.9)




class CustomDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.X = torch.randn(10000, 3, 224, 224)
        self.y = torch.randn(10000, 513, 196)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


data = DataLoader(CustomDataset(), batch_size = 8, shuffle = True, num_workers = 4)

data_ = tqdm(data)
loss_ = 0
for i, (x , y) in enumerate(data_):
    x = x.cuda()
    y = y.cuda()
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, y)
    loss.backward()
    optimizer.step()
    loss_ += (loss.item()-loss_)/(i+1)
    




if __name__ == '__main__':
    ## main()