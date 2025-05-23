import torch
from torch import nn as nn
from typing import Union



class StochasticDepth(torch.nn.Module):
    def __init__(self, 
                 p: float = 0.2,
                 ):
        super().__init__()
        """Thank you timm!!!"""
        assert 0<p<1, "p must be a positive number or <1"
        self.p = p

    def forward(self, 
                x:torch.Tensor)->torch.Tensor:
        if self.training:

            size = (x.shape[0],) + (1,)*(x.ndim-1)


            noise_x = torch.empty(size, dtype = x.dtype, device= x.device, requires_grad = False).bernoulli(1-self.p).div(1-self.p)
            
            return noise_x*x
        
        return x




class StochasticDepth_2(torch.nn.Module):
    def __init__(self, 
                 module: torch.nn.Module, 
                 p: float = 0.2,
                 ):
        super().__init__()
        """Thank you timm!!!"""
        assert 0<p<1, "p must be a positive number or <1"
        self.p = float(p)
        self.module: torch.nn.Module = module

    def forward(self, 
                x:torch.Tensor, 
                register:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:

        x_new, register_new = self.module(x, register)

        if self.training:

            size = [1]*x.ndim

            noise_x = torch.empty(size, dtype = x_new.dtype, device= x_new.device, requires_grad = False).bernoulli(1-self.p).div(1-self.p)
            
            noise_register = noise_x.squeeze([-1, -2])

            return x_new, noise_register * register_new
        
        return x_new, register_new


class TecherModel(object):
    def __init__(self, 
                 model:nn.Module,
                 compile_model:bool = False,
                 ):
        self.model = model if not compile_model else torch.compile(model)
    
    ## The below class method should not called in DDP setting, because each replica will attempt to download the model!!!
    ## To avoid this, you need to download the model in the main process and then broadcast it to all the replicas!!!
    @classmethod
    def from_torchub(cls,name:str, **kwargs):
        model = torch.hub.load("pytorch/vision", name, **kwargs)
        return cls(model)
    def __call__(self, x:torch.tensor)->torch.tensor:
        return self.model(x)
          
"""
class m(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2,2)
    def forward(self, x, register):
        return self.layer(x), self.layer(register)
        
x,y = torch.randn(512, 2,2,2), torch.randn(5, 2)
model = StochasticDepth(m(), p = 0.9)
model = torch.compile(model, mode ="max-autotune")
model.train()
model(x,y)
"""

class SdPModel(nn.Module):
    ## Though not abstract this class contains some utility functions to inherited 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = {}  

    def layer_init(self):
        ## This function inits the weights of the layers with a proper variance
        ## For deeper layers we may keep the variance a bit lower than, we expect!!!
        pass

    def layer_test(self, 
                   input = None, 
                   output = None, 
                   loss_fn = None):
        ## This function tests whether something blows up or vanishes by hooking up to interim values
        ## 
        Means_forward = []
        Stds_forward = []
        Norm_backward = []

        ## Gotta make sure that the model is in eval mode!!!
        self.eval()
        
        @torch.no_grad
        def forward_hook(module, input, output):
            if isinstance(output, Union[tuple, list]):
                for output_ in output:
                    Means_forward.append(output_.mean().item())
                    Stds_forward.append(output_.std().item())

            else:
                Means_forward.append(output.mean().item())
                Stds_forward.append(output.std().item())

        for module in self.modules():
            module.register_forward_hook(forward_hook)
        
        if not input:
            y = self(torch.randn(1, 3, 224, 224))
        else:
            y = self(input)
        
        ## Remove forward hook
        for module in self.modules():
            module._forward_hooks.clear()

        ## Put the model in the train mode!!!
        ## Shall fix a local seed here to make sure that the result is producible!!!
        self.train()
        ### Now do some training for a single example!!! to find out if there is exploding stuff in backward pass!!!

        @torch.no_grad
        def backward_hook(module, input, output):
            pass
        
        return {"forward_means": Means_forward, 
                "Forward_std": Stds_forward, 
                "Backward_norm": Norm_backward}

    
            
    def return_num_params(self)->int:
        ## This dude will return the number of parameters
        ## Here we use complex numbers for no purpose a tall
        params = sum([param.numel()*1j if param.requires_grad else param.numel() for param in self.parameters()])
        return {"Trainable_params": int(params.imag),  "Non_trainable_params": int(params.real)}

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

