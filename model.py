import torch
from torch import nn as nn
from torch.nn import functional as F
from layers import conv_patcher, embedding_layer, classification_head, block
from utility_layers import SdPModel, StochasticDepth
from typing import Callable, Optional, Any, Union
from numpy import arccos, cos

class main_model(SdPModel):
    def __init__(self, 
                 embedding_dim:int = 768,
                 num_blocks:int = 10,
                 n_head:int = 4,                 
                 activation:Callable = nn.GELU("tanh"),
                 conv_kernel_size:int = 5,
                 patch_size:int = 16,
                 ffn_dropout:float = 0.2,
                 attn_dropout:float = 0.2,
                 stochastic_depth:bool = True,
                 stochastic_depth_p:list[float] = [0.9, 0.55],
                 output_classes:int = 1000,
                 max_image_size:list[int, int] = [14,14],
                 max_num_registers:int = 5,
                 embedding_activation:Callable = None,
                 conv_first:bool = True,
                 ):
        super().__init__()
        ## Here we go again ##
        self.conv_init = conv_patcher(
                embedding_dim= embedding_dim,
                patch_size= patch_size,
        )
        
        self.embedding_layer = embedding_layer(
            embedding_dim= embedding_dim,
            max_num_registers=max_num_registers,
            max_image_size= max_image_size,
            activation = embedding_activation,

        )
        ### Helper functions -- Below we apply stochastic depth only when asked!!!
        ST = lambda x, p: StochasticDepth(x, p) if stochastic_depth else x
        ## The following function will adjust stochastic depth p value according to cosine schedule!!!
        ST_p = lambda i: cos(arccos(stochastic_depth_p[0])*(1 - i/num_blocks) + arccos(stochastic_depth_p[1])*(i/num_blocks))

        self.blocks = nn.ModuleList([
                        ST(block(embedding_dim  = embedding_dim,
                        n_head = n_head,
                        activation_func = activation,
                        ff_dropout = ffn_dropout,
                        att_dropout = attn_dropout,
                        conv_kernel_size = conv_kernel_size,
                        conv_activation = activation,
                        conv_first = conv_first), p = ST_p(i))
                        for i in range(num_blocks)])
        
        self.output_head = classification_head(embedding_dim, 
                                       output_classes,
                                       ffn_dropout)


    def forward(self, 
                x:torch.tensor, 
                num_registers:int = 3) -> tuple[torch.tensor, torch.tensor]:
        ## Patches ##B, 3, H, W --> B, embedding_dim, H//Patch_size, W//Patch_size
        x = self.conv_init(x)
        ## Add embeddings
        x_raw_output, registers = self.embedding_layer(x, num_registers)
        ## Mixing with convs
        for block in self.blocks:
            x_raw_output, registers = block(x, registers)
        
        ## Squeezer only works when the input dim is different than the ouput dim
        x_classification_head = self.output_head(registers[:, 0, :])

        return x_classification_head, x_raw_output, registers

"""
model = main_model(conv_first=True)
with torch.inference_mode():
    print(model(torch.randn(5, 3, 224,224), num_registers = 5)[1].std())
"""

if __name__ == "__main__":
    print("Ok boomer!!!")
