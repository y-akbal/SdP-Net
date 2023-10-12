import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn.functional as F


from layers import conv_int, conv_mixer, squeezer, first_encoder_layer, encoder_layer




class main_model(nn.Module):
    def __init__(self, 
                 embedding_dim_conv = 512,
                 embedding_trans_conv = 256,
                 n_head = 4,
                 conv_kernel_size = 5,
                 conv_mixer_repetation = 5, 
                 transformer_encoder_repetation = 5,
                 activation = nn.GELU("tanh"),
                 patch_size = 4,
                 dropout = 0.2,
                 squeeze_ratio = 15,
                 num_register = 2,  
                 multiplication_factor = 2, 
                 ):
        super().__init__()
        self.conv_init = conv_int(embedding_dim= embedding_dim_conv, 
                                  patch_size = patch_size)
        self.conv_mixer = nn.Sequential(*[
            conv_mixer(embedding_dim_conv, 
                       kernel_size= conv_kernel_size)
                   for i in range(conv_mixer_repetation)])
        self.squeezer = squeezer(embedding_dim= embedding_dim_conv,
                                 squeeze_ratio= squeeze_ratio,
                                 activation=  activation
                                 )
        self.encoder = encoder_layer(
            
        )


    def forward(self, x, y):
        x = self.conv_init(x)
        x = self.conv_mixer(x)
        x = self.squeezer(x)
        return x

main_model(conv_mixer_repetation=10, squeeze_ratio=3)(torch.randn(1, 3, 224, 224),1).shape



if __name__ == "__main__":
    print("Ok boomer!!!")