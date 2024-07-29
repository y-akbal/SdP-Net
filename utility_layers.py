import torch
from torch import nn as nn

class StochasticDepth(torch.nn.Module):
    def __init__(self, 
                 module: torch.nn.Module, 
                 p: float = 0.5):
        super().__init__()
        assert 0<p<1, "p must be a positive number or <1"
        self.module: torch.nn.Module = module
        self.p: float = p
        self._sampler = torch.Tensor(1)

    def forward(self, inputs):
        if self.training and self._sampler.uniform_().item() < self.p:
            return inputs
        return self.module(inputs).div_(1-self.p)


class SdPModel(nn.Module):
    ## Though not abstract this class contains some utility functions to inherited 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def layer_init(self):
        ## This function inits the weights of the layers with a proper variance
        pass

    def layer_test(self):
        ## This function tests whether something blows up or vanishes by hooking on to interim values
        pass
            
    def return_num_params(self)->int:
        ## This dude will return the number of parameters
        return sum([param.numel() for param in self.parameters()])

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

    def save_model(self, file_name = None):
        fn = "Model" if file_name == None else file_name
        model = {
            "state_dict":self.state_dict(),
            "config":self.config
        }
        try:
            torch.save(model, f"{fn}.pt")
            print(
                f"Model saved succesfully, see the file {fn}.pt for the weights and config file!!!"
            )
        except Exception as exp:
            print(f"Something went wrong with {exp}!!!!!")

    