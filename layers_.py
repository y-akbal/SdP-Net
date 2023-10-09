import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn.functional as F

## Here we will have layers to be used 
## We shall mostly use the optimized torch layers
## rather than coming up with our own implementations


x = torch.rand(10, 3, 224, 224)

nn.Conv2d(3, 3, kernel_size = 4, groups = 3,  padding = "same")(x).shape



l = torch.rand(3, 25)

Q = torch.einsum("ij,bilk->bjlk", l, x) 
T = (x.transpose(-3, -1) @ l).transpose(-3, -1)



class MultiHeadAtttention(
    nn.Module
):  
    def __init__(self, C_in=128, heads=4, dropout=0.2, the_same_QKV_matrix = True):
        super().__init__()
        ## Some simple dimension stuff
        assert (C_in/heads).is_integer(),  "Waccha divisibility criterion buddy"
        self.heads = heads

        ### Dropout along channel dimension
        self.dropout_Q = nn.Dropout2d(p=dropout)
        self.dropout_K = nn.Dropout2d(p=dropout)
        self.dropout_V = nn.Dropout2d(p=dropout)
        ### Pick up attention parameters, 
        if the_same_QKV_matrix:
            W = nn.Parameter(
            torch.randn(C_in, C_in) * (C_in) ** (-0.5)
        )
            self.L_Q, self.L_K, self.L_V = W, W, W
        else:
            pass
            
        ### This is for mixin' the output channels after attention
        self.output_dense = nn.Parameter(torch.randn(C_in, C_in) * (C_in) ** (-0.5))
        # Use here flash attention ---- ####
        

    def forward(self, x):
        
                
        return x