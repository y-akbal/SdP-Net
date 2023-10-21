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
                 image_size = (224, 224),
                 n_head:int = 4,
                 conv_kernel_size:int = 5,
                 conv_mixer_repetition:int = 5, 
                 transformer_encoder_repetition:int = 5,
                 activation = nn.GELU("tanh"),
                 patch_size:int = 4,
                 dropout:float = 0.2,
                 num_register:int = 2,  
                 multiplication_factor:int = 2, 
                 squeeze_ratio:int = 1,
                 output_classes = 1000,
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
                                        for i in range(conv_mixer_repetition)])
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
                                        ) for i in range(transformer_encoder_repetition-1)])
 
        
        self.lazy_output = nn.Linear(196, output_classes)
        
    def forward(self, x, y = None, task = "classification"):
        x = self.conv_init(x)
        x_ = self.conv_mixer(x)
        x = self.squeezer(x_)
        x = self.encoder_init(x,y)
        if task == "classification":
            x =  self.encoder_rest(x)[:,0,:]
        x =  self.encoder_rest(x).mean(-2)
        return self.lazy_output(x)
        

    def fun_encoder_dim(self, n:int)->int:
        return math.floor(n/(self.patch_size*self.squeeze_ratio))
    
    def return_num_params(self)->int:
        ## This dude will return the number of parameters
        total_params:int = 0 
        for param in self.parameters():
            total_params += param.shape.numel()
        return total_params

    ## The methods will work in tandem with the methods from_dict ## 
    ## They may not function if you use __init__ method!!! ##
    @classmethod
    def from_dict(cls, **kwargs):
        cls.config = kwargs
        model = cls(**kwargs)
        return model
    
    @classmethod
    def from_pretrained(cls, file_name):
        try:
            dict_ = torch.load(file_name)
            config = dict_["config"]
            state_dict = dict_["state_dict"]
            model = cls.from_dict(**config)
            model.load_state_dict(state_dict)
            print(
                f"Model loaded successfully!!!! The current configuration is {config}"
            )

        except Exception as e:
            print(f"Something went wrong with {e}")
        return model

    def save_model(self, file_name):
        fn = "Model" if file_name == None else file_name
        model = {}
        model["state_dict"] = self.state_dict()
        model["config"] = self.config
        try:
            torch.save(model, f"{fn}")
            print(
                f"Model saved succesfully, see the file {fn} for the weights and config file!!!"
            )
        except Exception as exp:
            print(f"Something went wrong with {exp}!!!!!")


"""
model = main_model(embedding_dim_conv=512, conv_mixer_repetition=10, transformer_encoder_repetition=5, patch_size=16, multiplication_factor=4).cuda()
model(torch.randn(1, 3, 224, 224).cuda(), torch.tensor([[1]]).cuda()).shape
model.return_num_params()

list(model.state_dict().keys())[45:86]


import numpy as np
y = torch.tensor(np.random.randint(0, 1000, size = 32)).cuda()
X = 0.4*torch.randn(32, 3, 224,224).cuda()
l = model(X, task = None)
F.cross_entropy(l,y)

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
for i in range(1000):
    optimizer.zero_grad()
    loss = F.cross_entropy(model(X, task = None),y)
    loss.backward()
    print(loss.item())
    optimizer.step()

"""

if __name__ == "__main__":
    print("Ok boomer!!!")