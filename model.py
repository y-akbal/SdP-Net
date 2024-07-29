import torch
from torch import nn as nn
from torch.nn import functional as F
from layers import conv_int, conv_mixer, embedding_layer, encoder_layer, output_head




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
                 max_num_register:int = 2,  
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
                                        dropout=dropout,
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
    ""
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

if __name__ == "__main__":
    print("Ok boomer!!!")
