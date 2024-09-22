import torch
from torch import nn as nn
from torch.nn import functional as F
from layers import ConvMixer, EmbeddingLayer, ConvPatcher, Block, FinalBlock, ClassificationHead
from utility_layers import SdPModel, StochasticDepth
from typing import Callable, Optional, Any, Union
from numpy import arccos, cos

torch.set_float32_matmul_precision('high')



class MainModel(SdPModel):
    def __init__(self, 
                 embedding_dim:int = 128,
                 num_blocks:int = 10,
                 n_head:int = 4,                 
                 activation:Callable = nn.GELU("tanh"),
                 conv_kernel_size:int = 5,
                 patch_size:int = 16,
                 ffn_dropout:float = 0.2,
                 attn_dropout:float = 0.2,
                 stochastic_depth:bool = True,
                 stochastic_depth_p:list[float] = [0.1, 0.01],
                 output_classes:int = 1000,
                 ff_multiplication_factor:int = 4,
                 max_image_size:list[int, int] = [14,14],
                 max_num_registers:int = 5,
                 embedding_activation:Callable = None,
                 conv_first:bool = True,
                 head_output_from_register:bool = False,
                 simple_mlp_output:bool = False,
                 output_head_bias:bool = False,
                 normalize_qv:bool = True,
                 ):
        super().__init__()
        ## oh s***  here we go again ##
        ## RIP CJ!##
        ### -- ##
        self.conv_init = ConvPatcher(
                embedding_dim= embedding_dim,
                patch_size= patch_size,
        )
        
        self.embedding_layer = EmbeddingLayer(
            embedding_dim= embedding_dim,
            max_num_registers=max_num_registers,
            max_image_size= max_image_size,
            activation = embedding_activation,
        )
        ### Helper functions -- Below we apply stochastic depth only when asked!!!
        ST = lambda x, p: StochasticDepth(x, p) if stochastic_depth else x
        ## The following function will adjust stochastic depth p value according to cosine schedule!!! storchastic_dept_0 -> stochastic_depth_1
        ST_p = lambda i: cos(arccos(stochastic_depth_p[0])*(1 - i/num_blocks) + arccos(stochastic_depth_p[1])*(i/num_blocks))

        self.blocks = nn.ModuleList([
                        ST(Block(embedding_dim  = embedding_dim,
                        n_head = n_head,
                        activation_func = activation,
                        ff_dropout = ffn_dropout,
                        att_dropout = attn_dropout,
                        multiplication_factor = ff_multiplication_factor,
                        conv_kernel_size = conv_kernel_size,
                        conv_activation = activation,
                        conv_first = conv_first,
                        normalize_qv = normalize_qv), p = ST_p(i))
                        for i in range(num_blocks)])
        
        self.final_block = FinalBlock(embedding_dim  = embedding_dim,
                        n_head = n_head,
                        activation_func = activation,
                        ff_dropout = ffn_dropout,
                        att_dropout = attn_dropout                       
        )


        self.output_head = ClassificationHead(embedding_dim, 
                                       output_classes,
                                       ffn_dropout,
                                       from_register = head_output_from_register,
                                       simple_output = simple_mlp_output,
                                       bias = output_head_bias,
                                       )

    def forward(self, 
                x:torch.tensor, 
                num_registers:int = 3, 
                return_raw_outputs:bool = False) -> tuple[torch.tensor, torch.tensor]:
        ## Patches ##B, 3, H, W --> B, embedding_dim, H//Patch_size, W//Patch_size
        x = self.conv_init(x)
        ## Add embeddings
        x_raw_output, registers = self.embedding_layer(x, num_registers)

        ## Mixing with convs
        for block in self.blocks:
            x_raw_output, registers = block(x_raw_output, registers)
        
        ## The final output layer does not contain any convolutional blocks!!
        x_raw_output, registers = self.final_block(x_raw_output, registers)
        ## Here depending on your needs we may further put a head, because the last conv layer in which case will not be used!
        
        x_classification_head = self.output_head(x_raw_output, registers)
        if not return_raw_outputs:
            return x_classification_head    
        return x_classification_head, x_raw_output, registers
    
"""
model = main_model(num_blocks = 15, 
                   embedding_dim = 768, 
                   patch_size=16, 
                   conv_first=False, 
                   stochastic_depth=True, 
                   conv_kernel_size = 7, 
                   stochastic_depth_p=[0.5, 0.1],
                   head_output_from_register=False,
                   simple_mlp_output=True,
                   max_image_size = [16,16],
                   ff_multiplication_factor=4,
                   normalize_qv = True,
                   ).to("mps")

model.return_num_params()

inputs = torch.randn(16, 3, 128, 128).to("mps")
targets = torch.randint(0, 1000, (16,)).to("mps")


model.return_num_params()
from torch import optim                   
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(1000):

    optimizer.zero_grad()
    # Forward pass
    outputs = model(inputs)
    loss = F.binary_cross_entropy_with_logits(outputs, F.one_hot(targets,1000).to(torch.float32))
    #loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(loss.item(), epoch)

    
for name, lay in model.named_parameters(): 
    if lay.grad == None:
        print(name, lay.grad)

"""


if __name__ == "__main__":
    print("Ok boomer!!!")

