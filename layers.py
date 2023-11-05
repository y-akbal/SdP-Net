import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn.functional as F

## Here we will have layers to be used 
## We shall mostly use the optimized torch layers
## rather than coming up with our own implementations

class conv_int(nn.Module):
    def __init__(self, embedding_dim = 128, patch_size = 4, activation = nn.GELU()):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.activation = activation
        self.conv = nn.Conv2d(in_channels = 3, 
                              out_channels = embedding_dim,
                              kernel_size = patch_size,
                              stride = patch_size, 
                              
                              )
        self.batch_norm = nn.SyncBatchNorm(embedding_dim)
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return self.batch_norm(x) 


class conv_mixer(nn.Module):
    def __init__(self, 
                 embedding_dim = 512, 
                 kernel_size = 5, 
                 activation = nn.GELU()):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels = embedding_dim, 
                              out_channels = embedding_dim,
                              kernel_size = kernel_size,
                              groups = embedding_dim,
                              padding = "same",
                              )
        self.conv1d = nn.Conv2d(in_channels = embedding_dim,
                                out_channels = embedding_dim,
                                kernel_size =1,
                                
                                )
        self.batch_norm_1 = nn.SyncBatchNorm(embedding_dim)
        self.batch_norm_2 = nn.SyncBatchNorm(embedding_dim)
        self.activation = activation
    def forward(self, x_):
        x = self.conv2d(x_)
        x = self.activation(x)
        x = self.conv1d(x_+self.batch_norm_1(x))
        x = self.activation(x)
        x = self.batch_norm_2(x)
        return x
"""
con = conv_mixer(512)
k = 0
for i in con.parameters():
     k += i.shape.numel()
print(k)
""" 
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
                              nn.SyncBatchNorm(embedding_dim),
                              activation
        )
        
    def forward(self, x):
        return self.seq(x)
"""
squeezer(128, squeeze_ratio = 7)(torch.randn(1, 128, 224,224)).shape     
for i in con.parameters():
     k += i.shape.numel()
print(k) 
"""
"""
k = 0
for i in nn.TransformerEncoderLayer(512, nhead = 8, batch_first=True,
                                    dim_feedforward=256).parameters():
    k += i.shape.numel()
print(k) 
"""
class embedding_layer(nn.Module):
    def __init__(self, embedding_dim_in:int = 512,
                 embedding_dim_out:int = 512,
                 num_registers:int = 1,
                 ):
        super().__init__()
        ### -- ###
        self.embedding = nn.Embedding(num_registers, embedding_dim_in)
        self.num_registers = num_registers

        if embedding_dim_in != embedding_dim_out:
            self.Linear = nn.Linear(embedding_dim_in, embedding_dim_out)
        else:
            self.Linear = lambda x: x 
        ### --- ###
        ### --- ###
        ### --- ###
        ### ----###
        self.register_buffer(
            "num_register",
            torch.tensor(
                [i for i in range(self.num_registers)],
                dtype=torch.int,
                requires_grad=False,
            ),
        )
    def forward(self, x, y = None):
        ### Here y will be localtions as there will be 2 more inputs that we save for extra 
        
        B, C, H, W = x.shape
        # C here is the embedding dimension!!!
        x = x.view(B, C, H*W).contiguous().transpose(-1,-2)

        if y == None:
            embeddings = self.embedding(self.num_register)
        else:
            embeddings = self.embedding(y)
        
        x = torch.cat((embeddings.repeat(B, 1, 1), x), 1)
        return self.Linear(x)

class encoder_layer(nn.Module):
    ## Here we embed H*W instead of the batch dimension 
    ## this may sound a bit better however
    ## it has some downsides as it is mosty affected by the size of the image
    def __init__(self, 
                 embedding_shape:tuple[int, int], 
                 n_head:int,
                 activation_func = nn.GELU(),
                 multiplication_factor:int = 2,
                 dropout = 0.2
                 ):
        super().__init__()
        self.embedding_dim = embedding_shape[0]*embedding_shape[1]
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model = self.embedding_dim,
            nhead = n_head,
            activation = activation_func,
            batch_first = True,
            dim_feedforward = int(multiplication_factor*self.embedding_dim),
            dropout = dropout,
            norm_first= True)
        
    def forward(self, x):
        return self.transformer_layer(x)




if __name__ == "__main__":
    print("Okkayy!!")

