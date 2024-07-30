import torch
from torch import nn as nn
from torch.nn import functional as F
from layers import conv_int, embedding_layer, output_head
from utility_layers import SdPModel, StochasticDepth
from typing import Callable, Optional, Any, Union


class main_model(SdPModel):
    def __init__(self, 
                 embedding_dim:int = 512,
                 num_blocks:int = 10,
                 n_head:int = 4,                 
                 activation:Callable = nn.GELU("tanh"),
                 conv_kernel_size:int = 5,
                 patch_size:int = 4,
                 ffn_dropout:float = 0.2,
                 attn_dropout:float = 0.2,
                 max_num_register:int = 5,  
                 stochastic_depth:bool = True,
                 stochastic_depth_p:float = 0.1,
                 multiplication_factor:int = 1, 
                 output_classes:int = 1000,
                 max_num_registers:int = 5,
                 embedding_activation:Callable = None,
                 conv_first:bool = True,
                 ):
        super().__init__()
        ## Here we go again ##
        self.conv_init = None
        self.embedding_layer = None
        self.blocks = None
        self.output_head = None


    def forward(self, 
                x:torch.tensor, 
                num_registers:int = 3) -> tuple[torch.tensor, torch.tensor]:
        ## Patches ##B, 3, H, W --> B, embedding_dim, H//Patch_size, W//Patch_size
        x = self.conv_init(x)
        ## Add embeddings
        x, registers = self.embedding_layer(x, num_registers)
        ## Mixing with convs
        x_raw_output, registers = self.blocks(x, registers)
        ## Squeezer only works when the input dim is different than the ouput dim
        x_classification_head = self.output_head(registers[:, 0, :])

        return x_classification_head, x_raw_output, registers

if __name__ == "__main__":
    print("Ok boomer!!!")
