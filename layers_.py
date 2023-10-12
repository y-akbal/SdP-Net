import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn.functional as F

## Here we will have layers to be used 
## We shall mostly use the optimized torch layers
## rather than coming up with our own implementations

class conv_int(nn.Module):
    def __init__(self, embedding_dim = 100, patch_size = 4, activation = nn.GELU()):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.activation = activation
        self.conv = nn.Conv2d(in_channels = 3, 
                              out_channels = embedding_dim,
                              kernel_size = patch_size,
                              stride = patch_size
                              )
        self.batch_norm = nn.BatchNorm2d(embedding_dim)
    def forward(self, x):
        x = self.conv(x)
        x = self.conv(x)
        return self.batch_norm(x) 
    
    def __init__(self):
        
        ## depthwise
        ## 

