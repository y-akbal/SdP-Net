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
        x = self.activation(x)
        return self.batch_norm(x) 

class res_jump(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    def forward(self, x):
        return x + self.layer(x)

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


class squeezer(nn.Module):
    def __init__(self, 
                 embedding_dim = 512,
                 squeeze_ratio = 5,
                 activation = nn.GELU()
                 ):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(embedding_dim, 
                              embedding_dim, 
                              kernel_size = squeeze_ratio,
                              stride = squeeze_ratio,
                              groups = embedding_dim
                              ),
                              nn.BatchNorm2d(embedding_dim),
                              activation
        )
        
    def forward(self, x):
        return self.seq(x)
"""
squeezer(128, squeeze_ratio = 7)(torch.randn(1, 128, 224,224)).shape      
"""
class first_encoder_layer(nn.Layer):
    def __init__(self, embedding_shape:tuple[int,int],
                 n_head:int = 8,
                 num_registers:int = 5,
                 squeeze_ratio:int = 7, 
                 multiplication_factor:int = 2, 
                 activation_func = nn.GELU(),

                 ):
        super().__init__()
        ### -- ###
        self.embedding_dim = embedding_shape[0]*embedding_shape[1]
        self.cls_registers = nn.Embedding(num_registers,
                                          self.embedding_dim)
        ## -- ##
        
        ### --- ###
        self.squeezer = squeezer(self.embedding_dim, 
                                 squeeze_ratio=squeeze_ratio,
                                 activation = activation_func)
        ### --- ###
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model = self.embedding_dim,
            nhead = n_head,
            dim_feedforward = multiplication_factor*self.embedding_dim, 
            activation = activation_func,
        )
        ### ---- ###
    def forward(self, x, y = None):
        embeddings = self.embedding_dim(y)
        return embeddings
class encoder_layer(nn.Module):
    ## Here we embed H*W instead of the batch dimension 
    ## this may sound a bit better however
    ## it has some downsides as it is mosty affected by the size of the image
    def __init__(self, 
                 embedding_shape:tuple[int, int], 
                 n_head:int,
                 activation_func = nn.GELU(),
                 ):
        super().__init__()
        self.embedding_dim = embedding_shape[0]*embedding_shape[1]
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model = self.embedding_dim,
            nhead = n_head,
            activation = activation_func,
            batch_first = True,
            dim_feedforward = 2*self.embedding_dim,
        )
    def forward(self, x):
        batch_size, C, _, _ = x.shape
        x = x.view(batch_size, C, self.embedding_dim)
        x = self.transformer_layer(x)
        return x


encoder_layer((32,32), 4)(torch.randn(32, 1024, 32, 32)).shape

if __name__ == "__main__":
    print("pff")
