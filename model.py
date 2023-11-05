import torch
from torch import nn as nn
from torch.nn import functional as F
from layers import conv_int, conv_mixer, squeezer, first_encoder_layer, encoder_layer
import math
import functools

class main_model(nn.Module):
    def __init__(self, 
                 embedding_dim_conv:int = 512,
                 image_size:tuple = (224, 224),
                 n_head:int = 4,
                 conv_kernel_size:int = 5,
                 conv_mixer_repetition:int = 5, 
                 transformer_encoder_repetition:int = 5,
                 activation = nn.GELU("tanh"),
                 patch_size:int = 4,
                 dropout:float = 0.2,
                 num_register:int = 2,  
                 multiplication_factor:int = 1, 
                 squeeze_ratio:int = 1,
                 output_classes = 1000,
                 ):
        super().__init__()
        ## Here we go again ##
        self.patch_size = patch_size
        self.squeeze_ratio = squeeze_ratio
        self.encoder_embedding_dim:tuple = list(map(self.fun_encoder_dim, image_size))
        self.encoder_embeddid_dim_:int = functools.reduce(lambda x,y:x*y, self.encoder_embedding_dim)

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
 
        
        self.output_head = nn.Sequential(*[nn.Linear(self.encoder_embeddid_dim_, output_classes),
                                        nn.Tanh(),
                                        nn.Linear(output_classes, output_classes)])
        
    def forward(self, x, y = None, task = "C"):
        x = self.conv_init(x)
        x_ = self.conv_mixer(x)
        x = self.squeezer(x_)
        x = self.encoder_init(x,y)
        x =  self.encoder_rest(x)
        if task == "C":
            return self.output_head(x[:,0,:])
        return x
        

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
        model = cls(**kwargs)
        model.config = kwargs
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
            torch.save(model, f"{fn}.pt")
            print(
                f"Model saved succesfully, see the file {fn} for the weights and config file!!!"
            )
        except Exception as exp:
            print(f"Something went wrong with {exp}!!!!!")


#### Below is just debugging purposses should be considered seriously useful ####
"""
model = main_model(embedding_dim_conv=512, 
                conv_mixer_repetition = 5,
                conv_kernel_size = 7,
                transformer_encoder_repetition = 5, 
                patch_size=8, 
                multiplication_factor=2,
                squeeze_ratio= 1)

model.fun_encoder_dim(224)
model.return_num_params()
model(torch.randn(10, 3, 224, 224)).shape

import numpy as np
y = torch.tensor(np.random.randint(0, 1000, size = 8), dtype = torch.long)
X = 0.4*torch.randn(8, 3, 224,224)
l = model(X, task = "C")
F.cross_entropy(l,y)

torch.argmax(l, dim = -1) == y


optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
for i in range(1000):
    optimizer.zero_grad()
    loss = F.cross_entropy(model(X, task = "C"),y)
    loss.backward()
    print(loss.item())
    optimizer.step()


"""
if __name__ == "__main__":
    print("Ok boomer!!!")