import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn.functional as F


x = torch.rand(10, 3, 224, 224)

l = torch.rand(3, 23)

Q = torch.einsum("ij,bilk->bjlk", l, x) 
T = (x.transpose(-3, -1) @ l).transpose(-3, -1)





class MultiHeadAtttention(
    nn.Module
):  
    def __init__(self, C_in=128, heads=4, dropout=0.2, the_same_QKV_matrix = True):
        super().__init__()
        ## Some simple dimension stuff
        assert (C_in/heads).is_integer(),  "Waccha divisibility criterion buddy"
        self.embedding_dim = C_in
        self.heads = heads

        ### Dropout along channel dimension
        self.dropout_Q = nn.Dropout2d(p=dropout)
        self.dropout_K = nn.Dropout2d(p=dropout)
        self.dropout_V = nn.Dropout2d(p=dropout)

        if the_same_QKV_matrix:
            self.W = nn.Parameter(
            torch.randn(C_in, C_in) * (C_in) ** (-0.5)
        )
        
        ### This is for mixin' the output channels after attention
        self.output_dense = nn.Parameter(torch.randn(C_in, C_in) * (C_in) ** (-0.5))
        
        
            