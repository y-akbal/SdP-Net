import torch
from torch import nn as nn
from torch.nn import functional as F
from layers import conv_patcher, embedding_layer, classification_head, block, final_block
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
                 stochastic_depth_p:list[float] = [0.1, 0.01],
                 output_classes:int = 1000,
                 max_image_size:list[int, int] = [14,14],
                 max_num_registers:int = 5,
                 embedding_activation:Callable = None,
                 conv_first:bool = True,
                 head_output_from_register:bool = False,
                 simple_mlp_output:bool = False,
                 output_head_bias:bool = False,
                 ):
        super().__init__()
        ## oh s***  here we go again ##
        ## RIP CJ!##
        ### -- ##
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
        ## The following function will adjust stochastic depth p value according to cosine schedule!!! storchastic_dept_0 -> stochastic_depth_1
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
        
        self.final_block = final_block(embedding_dim  = embedding_dim,
                        n_head = n_head,
                        activation_func = activation,
                        ff_dropout = ffn_dropout,
                        att_dropout = attn_dropout                       
        )


        self.output_head = classification_head(embedding_dim, 
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
model = main_model(num_blocks = 10, 
                   embedding_dim = 128, 
                   patch_size=16,
                   conv_first=True, 
                   stochastic_depth=False,  
                   stochastic_depth_p=[0.9, 0.01],
                   head_output_from_register=False,
                   simple_mlp_output=False)

inputs = torch.randn(30, 3, 224, 224)
targets = torch.randint(0, 1000, (30,))

model(inputs, return_raw_outputs = False, num_registers = 10).std()
model.return_num_params()
                   

from torch import optim                   
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
batch_size = 2


init_start = torch.cuda.Event(enable_timing=True)

for epoch in range(10):
    optimizer.zero_grad()
        
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
        
    loss.backward()
    optimizer.step()
        
    print(loss.item(), epoch)

torch.cuda.synchronize() 
init_end = torch.cuda.Event(enable_timing=True)



print(f"elapsed time: {init_start.elapsed_time(init_end) / 1000}secs")
    
for name, lay in model.named_parameters(): 
    if lay.grad == None:
        print(name, lay.grad)


with torch.inference_mode():
    x,y,z = model(torch.randn(5, 3, 224,224), num_registers = 1, return_raw_outputs = True)odel.output_head
x.shape
model.return_num_params()
a = model.layer_test()

"""
if __name__ == "__main__":
    print("Ok boomer!!!")

