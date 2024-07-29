import torch
from torch import nn as nn
from torch.nn import functional as F
from layers import conv_int, conv_mixer, embedding_layer, encoder_layer, output_head
from utility_layers import SdPModel, StochasticDepth



class main_model(nn.Module):
    def __init__(self, 
                 embedding_dim:int = 512,
                 num_blocks:int = 10,
                 n_head:int = 4,                 
                 activation = nn.GELU("tanh"),
                 conv_kernel_size:int = 5,
                 patch_size:int = 4,
                 ffn_dropout:float = 0.2,
                 attn_dropout:float = 0.2,
                 max_num_register:int = 5,  
                 stochastic_depth:bool = True,
                 stochastic_depth_p:float = 0.1,
                 multiplication_factor:int = 1, 
                 output_classes = 1000,
                 ):
        super().__init__()
        ## Here we go again ##
        self.patch_size = patch_size
        self.num_register = num_register        

        self.conv_init = conv_int(embedding_dim= embedding_dim_conv, 
                                  patch_size = patch_size
                                  )
        self.conv_mixer = nn.Sequential(*[conv_mixer(embedding_dim_conv, 
                                        kernel_size= conv_kernel_size, 
                                        activation = activation)
                                        for i in range(conv_mixer_repetition)])
        
        self.embedding_layer = embedding_layer(embedding_dim_in=embedding_dim_conv,
                                            embedding_dim_out=embedding_dim_trans,
                                            num_registers=num_register,
                                            )
        self.encoder_rest= nn.Sequential(*[encoder_layer(embedding_dim = embedding_dim_trans,
                                        n_head = n_head,
                                        multiplication_factor= multiplication_factor,
                                        activation_func= activation,
                                        dropout = dropout,
                                        ) for i in range(transformer_encoder_repetition)])
 
        
        self.output_head = output_head(embedding_dim = embedding_dim_trans,
                                       output_classes= output_classes,
                                       dropout = dropout)  
                                        ### Add here some normalization without which we would never existSssss!!! 

    def forward(self, x, y = None):
        ## Patches 
        x = self.conv_init(x)
        ## Mixing with convs
        x = self.conv_mixer(x)
        ## Squeezer only works when the input dim is different than the ouput dim
        x = self.squeezer(x)
        ## Together with embeddings
        x = self.embedding_layer(x,y)
        ## Transformers take the wheel!!
        x =  self.encoder_rest(x)
        
        return self.output_head(x[:,0,:])

if __name__ == "__main__":
    print("Ok boomer!!!")
