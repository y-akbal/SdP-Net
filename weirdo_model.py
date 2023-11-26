import torch
from torch import nn as nn
from torch.nn import functional as F

from layers import freak_attention_encoder, conv_mixer 
from model import main_model

class freak_model(nn.Module):
    def __init__(self,
                 embedding_dim_conv:int = 512,
                 embedding_dim_trans:int = 512,
                 n_head:int = 4,
                 conv_kernel_size:int = 5,
                 conv_mixer_repetition:int = 5, 
                 transformer_encoder_repetition:int = 5,
                 activation = nn.GELU("tanh"),
                 patch_size:int = 4,
                 dropout:float = 0.2,
                 multiplication_factor:int = 1, 
                 output_classes = 1000,
                 ):
        self.conv_init = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=embedding_dim_conv,
                                                 kernel_size=patch_size,bias=False))
        self.conv_mixer = nn.Sequential(*[conv_mixer(embedding_dim_conv, kernel_size=conv_kernel_size, activation=activation) for _ in range(conv_mixer_repetition)
        ])
    def forward(self, x):
        x = self.conv_init(x)
        x = self.conv_mixer(x)




nn.LayerNorm([5, 16, 16])(torch.randn(5, 5, 16, 16)).var(3)