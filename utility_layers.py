import torch

class StochasticDepth(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, p: float = 0.5):
        super().__init__()
        assert 0<p<1, "p must be a positive number or <1"
        self.module: torch.nn.Module = module
        self.p: float = p
        self._sampler = torch.Tensor(1)

    def forward(self, inputs):
        if self.training and self._sampler.uniform_().item() < self.p:
            return inputs
        return self.module(inputs).div_(1-self.p)
