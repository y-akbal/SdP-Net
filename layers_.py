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


class conv_mixer(nn.Module):
    def __init__(self, 
                 embedding_dim, 
                 kernel_size = 5, 
                 activation = nn.GELU()):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels = embedding_dim, 
                              out_channels = embedding_dim,
                              kernel_size = kernel_size,
                              groups = embedding_dim,
                              padding = "same"
                              )
        self.conv1d = nn.Conv2d(in_channels = embedding_dim,
                                out_channels = embedding_dim,
                                kernel_size =1,

                                )
        self.batch_norm_1 = nn.BatchNorm2d(embedding_dim)
        self.batch_norm_2 = nn.BatchNorm2d(embedding_dim)
        self.activation = activation
    def forward(self, x_):
        x = self.conv2d(x_)
        x = self.activation(x)
        x = self.batch_norm_1(x)
        x += x_
        x = self.conv1d(x)
        x = self.activation(x)
        x = self.batch_norm_2(x)
        return x

"""
conv_mixer(200)(torch.randn(10, 200, 224,224)).std(1)
"""

class squeezer(nn.Module):
    def __init__(self, embedding_dim,
                 squeeze_ratio = 5
                 ):
        super().__init__()
        self.conv = nn.Conv2d(embedding_dim, 
                              embedding_dim, 
                              kernel_size = squeeze_ratio,
                              stride = squeeze_ratio)
    def forward(self, x):
        return self.conv(x)

"""
squeezer(100)(torch.randn(1, 100, 224,224)).shape
"""    
class encoder_layer(nn.Module):
    def __init__(self, 
                 embedding_dim, 
                 n_head,
                 activation_func = nn.GELU(),
                 ):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model = embedding_dim,
            nhead = n_head,
            activation = activation_func,
            batch_first = True,
            dim_feedforward = 512,
        )
    def forward(self, x):
        batch_size, C, H, W = x.shape
        x = x.view(batch_size, C, H*W)
        x = self.transformer_layer(x)
        return x

"""
encoder_layer(100, 5)(torch.randn(1, 100, 10,10))

x = torch.randn(1, 100, 100)
nn.TransformerEncoderLayer(100,100)()
encoder_layer = nn.TransformerEncoderLayer(d_model=512, 
                                           nhead=8, 
                                           batch_first = True,
                                           activation = nn.GELU()
                                           )
src = torch.rand(1, 5, 512)
encoder_layer(src).std(2)
torch.randn(1, 10, 23, 23)
"""
