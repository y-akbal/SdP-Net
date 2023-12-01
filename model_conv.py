from torch import nn as nn
from layers import weirdo_conv_mixer
import torch

class freak_mixer(nn.Module):
    def __init__(self,
                 embedding_dim:int = 512,
                 conv_kernel_size:int = 5,
                 conv_mixer_repetition:int = 5, 
                 activation = nn.GELU("tanh"),
                 patch_size:int = 14,
                 multiplication_factor:int = 1, 
                 output_classes = 1000,
                 **kwargs,
                 ):
        super().__init__(**kwargs)        
        self.conv_init = nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels=embedding_dim,
                                                 kernel_size=patch_size,bias=True),
                                                 activation,
                                                 nn.SyncBatchNorm(embedding_dim)])
        self.conv_mixer = nn.Sequential(*[weirdo_conv_mixer(embedding_dim=embedding_dim, 
                                                            kernel_size=conv_kernel_size, 
                                                            activation= activation,
                                                            multiplication_factor = multiplication_factor)
                                                            for _ in range(conv_mixer_repetition)])
        self.base = nn.Sequential(*[self.conv_init,
                                    self.conv_mixer])
        self.head = nn.Sequential(*[
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(embedding_dim, output_classes)
        ])
    def forward(self, x):
        x = self.base(x)
        x = self.head(x)
        return x
    
    def return_num_params(self)->int:
        ## This dude will return the number of parameters
        total_params:int = 0 
        for param in self.parameters():
            total_params += param.shape.numel()
        return total_params

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

"""

l = 0
for p in freak_mixer(embedding_dim=768,multiplication_factor=4, conv_mixer_repetition=15).parameters():
    l += p.shape.numel()
print(l)
"""
if __name__ == '__main__':
    pass
## oh sweet, again??

