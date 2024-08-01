import torch
from torch import nn as nn

class StochasticDepth(torch.nn.Module):
    def __init__(self, 
                 module: torch.nn.Module, 
                 p: float = 0.2):
        super().__init__()
        assert 0<p<1, "p must be a positive number or <1"
        self.module: torch.nn.Module = module
        self.p: float = p
        self._sampler = torch.Tensor(1)

    def forward(self, 
                x:torch.tensor, 
                register:torch.tensor)->tuple[torch.tensor, torch.tensor]:
        if self.training and self._sampler.uniform_().item() < self.p:
            # Direct input 
            return x, register
        x, register = self.module(x, register)
        ## Expected value of the output will be 0, but we will change the variance if we divide the things by 1-p!!!
        return x, register
"""
model = nn.Sequential(*[StochasticDepth(nn.Linear(10,10)) for i in range(10)])
model(torch.randn(10))
"""
class SdPModel(nn.Module):
    ## Though not abstract this class contains some utility functions to inherited 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = {}  

    def layer_init(self):
        ## This function inits the weights of the layers with a proper variance
        pass

    def layer_test(self, input = None):
        ## This function tests whether something blows up or vanishes by hooking up to interim values
        ## 
        Means = []
        Stds = []

        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                output_1, output_2 = output
                mean_1, mean_2, std_1, std_2 = output_1.mean(), output_2.mean(), output_1.std(), output_2.mean()
                Means.append(mean_1)
                Means.append(mean_2)
                Stds.append(std_1)
                Stds.append(std_2)
                
            mean, std = output.mean(), output.std()
            Means.append(mean)
            Stds.append(std)
        
        
        for module in self.modules():
            module.register_forward_hook(forward_hook)
        
        if not input:
            y = self(torch.randn(1, 3, 224, 224))
        
        for module in self.modules():
            module.remove_forward_hook(forward_hook)
        
        return Means, Stds

    
            
    def return_num_params(self)->int:
        ## This dude will return the number of parameters
        ## Here we use complex numbers for no purpose a tall
        params = sum([param.numel()*1j if param.requires_grad else param.numel() for param in self.parameters()])
        return int(params.imag), int(params.real)

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
            ## We may need to carry all the weights to cpu then save it!!!
            "config":self.config
        }
        try:
            torch.save(model, f"{fn}.pt")
            print(
                f"Model saved succesfully, see the file {fn}.pt for the weights and config file!!!"
            )
        except Exception as exp:
            print(f"Something went wrong with {exp}!!!!!")



