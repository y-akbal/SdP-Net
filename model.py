import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn.functional as F


from layers import conv_int, conv_mixer, squeezer, first_encoder_layer, encoder_layer




class main_model(nn.Module):
    def __init__(self, 
                 embedding_dim_conv,
                 embedding_trans_conv,
                 n_head = 4,
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
    def forward(self, x, y):
        pass





if __name__ == "__main__":
    print("Ok boomer!!!")