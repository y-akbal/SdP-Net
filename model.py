import torch
from torch import nn as nn

from layers import conv_int, conv_mixer, embedding_layer, encoder_layer, squeezer

class main_model(nn.Module):
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
                 num_register:int = 2,  
                 multiplication_factor:int = 1, 
                 output_classes = 1000,
                 squeeze_ratio = 2,
                 ):
        super().__init__()
        ## Here we go again ##
        self.patch_size = patch_size
        
        self.num_register = num_register        

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
            self.squeezer = squeezer(embedding_dim = embedding_dim_conv,
                                     squeeze_ratio=squeeze_ratio,
                                     activation=activation)
        
        self.embedding_layer = embedding_layer(embedding_dim_in=embedding_dim_conv,
                                            embedding_dim_out=embedding_dim_trans,
                                            num_registers=num_register,
                                            )
        self.encoder_rest= nn.Sequential(*[encoder_layer(embedding_dim = embedding_dim_trans,
                                        n_head = n_head,
                                        multiplication_factor= multiplication_factor,
                                        activation_func= activation,
                                        dropout=dropout,
                                        ) for i in range(transformer_encoder_repetition)])
 
        
        self.output_head = nn.Sequential(*[nn.Dropout(p = dropout),
                                        nn.Linear(embedding_dim_trans, output_classes),
                                        nn.Tanh(),
                                        nn.Dropout(p = dropout),
                                        nn.Linear(output_classes, output_classes)])
        
    def forward(self, x, y = None, task = "C"):
        ## Patches 
        x = self.conv_init(x)
        ## Mixing with convs
        x = self.conv_mixer(x)
        x = self.squeezer(x)
        ## Together with embeddings
        x = self.embedding_layer(x,y)
        ## Transformers take the wheel!!
        x =  self.encoder_rest(x)
        if task == "C":
            return self.output_head(x[:,0,:])
        return x
            
        if task == "C":
            return self.output_head(x[:,:self.num_register,:]).transpose(-1,-2)
        return x.transpose(-1,-2)
            
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
                   embedding_dim_trans=512,
                conv_mixer_repetition = 5,
                conv_kernel_size = 7,
                transformer_encoder_repetition = 5, 
                patch_size = 16, 
                num_register=5,
                multiplication_factor= 1,
                ).cuda()

model.return_num_params()
model(torch.randn(10, 3, 224, 224), task = "sd").shape
import numpy as np
y = torch.tensor(np.random.randint(0, 1000, size = (64,5)), dtype = torch.long).cuda()
X = 0.4*torch.randn(64, 3, 224,224).cuda()
l = model(X)
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