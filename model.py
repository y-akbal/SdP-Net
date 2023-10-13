import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math


from layers import conv_int, conv_mixer, squeezer, first_encoder_layer, encoder_layer




class main_model(nn.Module):
    def __init__(self, 
                 embedding_dim_conv:int = 512,
                 image_size = (224,224),
                 n_head:int = 4,
                 conv_kernel_size:int = 5,
                 conv_mixer_repetation:int = 5, 
                 transformer_encoder_repetation:int = 5,
                 activation = nn.GELU("tanh"),
                 patch_size:int = 4,
                 dropout:float = 0.2,
                 num_register:int = 2,  
                 multiplication_factor:int = 2, 
                 squeeze_ratio:int = 1,
                 ):
        super().__init__()
        ## Here we go again ##
        self.patch_size = patch_size
        self.squeeze_ratio = squeeze_ratio
        self.encoder_embedding_dim = list(map(self.fun_encoder_dim, image_size))
        

        self.conv_init = conv_int(embedding_dim= embedding_dim_conv, 
                                  patch_size = patch_size,
                                  activation = activation
                                  )
        self.conv_mixer = nn.Sequential(*[conv_mixer(embedding_dim_conv, 
                                        kernel_size= conv_kernel_size)
                                        for i in range(conv_mixer_repetation)])
        if squeeze_ratio == 1:
            self.squeezer = lambda x: x
        else:
            self.squeezer = squeezer(embedding_dim= embedding_dim_conv,
                                 squeeze_ratio= squeeze_ratio,
                                 activation=  activation
                                 )
        
        self.encoder_init = first_encoder_layer(embedding_shape= self.encoder_embedding_dim,
                                                n_head = n_head,
                                                num_registers = num_register,
                                                multiplication_factor= multiplication_factor,
                                                activation_func= activation,
                                                dropout=dropout
                                                )
        self.encoder_rest= nn.Sequential(*[encoder_layer(embedding_shape= self.encoder_embedding_dim,
                                        n_head = n_head,
                                        multiplication_factor= multiplication_factor,
                                        activation_func= activation,
                                        dropout=dropout,
                                        ) for i in range(transformer_encoder_repetation-1)])

        
    def fun_encoder_dim(self, n:int)->int:
        return math.floor(n/(self.patch_size*self.squeeze_ratio))

    def forward(self, x, y = None):
        x = self.conv_init(x)
        x = self.conv_mixer(x)
        x = self.squeezer(x)
        x = self.encoder_init(x,y)
        return self.encoder_rest(x)

torch.manual_seed(0)
main_model(conv_mixer_repetation=5, transformer_encoder_repetation=10, patch_size=12)(torch.randn(32, 3, 224, 224), torch.tensor([[1]])).shape

model = main_model(conv_mixer_repetation=5, transformer_encoder_repetation=10, patch_size=8, multiplication_factor=1).cuda(1)

model(torch.randn(8, 3, 224, 224).cuda(1), torch.tensor([[1]]).cuda(1)).shape


k = 0
for i in main_model(conv_mixer_repetation=10, transformer_encoder_repetation=5, patch_size=12, multiplication_factor=1).parameters():
    k += i.shape.numel()
print(k)


if __name__ == "__main__":
    print("Ok boomer!!!")