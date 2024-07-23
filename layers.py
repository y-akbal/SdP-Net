import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from typing import Callable



## Here we will have layers to be used 
## We shall mostly use the optimized torch layers
## rather than coming up with our own implementations

class conv_patcher(nn.Module):
    def __init__(self, 
                 embedding_dim = 128, 
                 patch_size = 4):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.conv = nn.Conv2d(in_channels = 3, 
                              out_channels = embedding_dim,
                              kernel_size = patch_size,
                              stride = patch_size,                               
                              )
    def forward(self, x:torch.Tensor)->torch.Tensor:
        ## B, 3, H, W --> B, 3, H*W ### 
        B, C, H, W = x.shape
        x = self.conv(x) 
        x = x.view(B, C, H*W).contiguous().transpose(-1,-2)
        return x



class classification_head(nn.Module):
    ## Here we embed C instead of the batch H*W
    ## this may sound a bit weirdo!!! 
    def __init__(self, 
                 embedding_dim:int = 768, 
                 output_classes:int=1000,
                 dropout:float = 0.2,
                 ):
        
        super().__init__()
        self.output_head = nn.Sequential(*[nn.LayerNorm(embedding_dim),
                                        nn.Linear(embedding_dim, output_classes),
                                        nn.Tanh(),
                                        nn.Dropout(dropout),
                                        nn.Linear(output_classes, output_classes)
                                        ])   

    def forward(self, x):
        return self.output_head(x)
    
class conv_mixer(nn.Module):
    def __init__(self, 
                 embedding_dim:int = 512, 
                 kernel_size:int = 5, 
                 activation:Callable = nn.GELU()):
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
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.activation = activation
    def forward(self, x_:torch.Tensor)->torch.Tensor:
        x = self.conv2d(x_)
        x = self.activation(x)
        x = self.conv1d(x_+self.layer_norm_1(x))
        x = self.activation(x)
        x = self.layer_norm_2(x)
        return x

class embedding_layer(nn.Module):
    ## We isolated this layer in the case that you want to 
    ## do something like enumerating the pixels...
    def __init__(self, 
                 embedding_dim: int = 768,
                 num_registers:int = 1,
                 image_size:list[int, int] = [14,14]
                 ):
        super().__init__()
        ### -- ###
        self.num_registers = num_registers
        self.vertical_im_size = image_size[0]
        self.horizontal_im_size = image_size[1]
        ###
        self.register_embedding_layer = nn.Embedding(num_registers, embedding_dim)
        self.vertical_embedding_layer = nn.Embedding(image_size[0], embedding_dim)
        self.horizontal_embedding_layer = nn.Embedding(image_size[1], embedding_dim)
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

        self.register_buffer(
            "vertical_embedding",
            torch.tensor(
                [i for i in range(image_size[0])],
                dtype=torch.int,
                requires_grad=False,
            ),
        )

        self.register_buffer(
            "horizontal_embedding",
            torch.tensor(
                [i for i in range(image_size[1])],
                dtype=torch.int,
                requires_grad=False,
            ),
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        ### Here y will be localtions as there will be 2 more inputs that we save for extra 
        
        B, C, HW = x.shape
        assert HW == self.horizontal_im_size*self.vertical_im_size
        # C here is the embedding dimension!!!
        x = x.view(B, C, self.horizontal_im_size, self.vertical_im_size)
        ## NO NEED TO COMPUTE THESE DUDES FOR EACH FORWARD PASS!!!!!
        register_embeddings = self.register_embedding_layer(self.num_register)
        horizontal_embeddings = self.horizontal_embedding_layer(self.horizontal_embedding).transpose(-1, -2)
        vertical_embeddings = self.vertical_embedding_layer(self.vertical_embedding).transpose(-1, -2)

        ## Do something here!!! to add the vectors both verticall and horizontally!!!
        ## Here you may repeat some tokens --- !!!
        x = x.view(B, C, HW).transpose(-1, -2)
        ## and the concat with register tokens, and give the output to the 
        return  x

class encoder_layer(nn.Module):
    def __init__(self, 
                 embedding_dim:int, 
                 n_head:int,
                 activation_func:Callable = nn.GELU(),
                 multiplication_factor:int = 2,
                 dropout:float = 0.2
                 ):
        assert embedding_dim*multiplication_factor > 1, "Come on dude, do not squeeze to much"
        super().__init__()
        ## #Can we do here flash attention??? Shal we write this layer from scratch????
        ### What ya think?
        self.embedding_dim = embedding_dim
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model = self.embedding_dim,
            nhead = n_head,
            activation = activation_func,
            batch_first = True,
            dim_feedforward = int(multiplication_factor*self.embedding_dim),
            dropout = dropout,
            norm_first= True
            )
        
    def forward(self, x):
        return self.transformer_layer(x)




if __name__ == "__main__":
    print("Okkayy!!")

