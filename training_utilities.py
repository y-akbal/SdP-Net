import torch
from torch.distributed import (
    all_reduce,
    ReduceOp,
)
import torch.distributed as dist
import wandb
from typing import Callable

class distributed_loss_track:

    def __init__(self, 
                 wandb_log:bool = True,
                 task:str = "Train"):
        
        self.temp_loss:float = 0
        self.counter:int = 0
        self.epoch:int = 0
        self.task = task
        self.wandb_log = wandb_log
    
    def update(self, batch_loss:float):
        self.temp_loss += batch_loss
        self.counter += 1

    def reset(self):
        self.temp_loss = 0.0
        self.counter = 0
        self.epoch += 1
    
    def __all_reduce__(self):
        loss_tensor = torch.tensor([self.temp_loss, self.counter], dtype=torch.float32).cuda()
        all_reduce(loss_tensor, ReduceOp.SUM, async_op=False)
        self.temp_loss, self.counter = loss_tensor.cpu().tolist()
    
    def log(self):
        try:
            self.__all_reduce__()
        except:
            print("Error in all reducing loss")

        if self.wandb_log:
                ## Only the main worker will log the loss -- else all workers will log the loss!!!
                wandb.log({"epoch": self.epoch, f"{self.task}_loss":self.temp_loss/self.counter})
        loss = self.temp_loss/self.counter
        self.reset()
        return loss
        

class track_accuracy:
    def __init__(self, 
                 task = "Validation", 
                 wandb_log = True):
        self.correct = torch.tensor(0.0, device='cuda', requires_grad=False)
        self.total = torch.tensor(0.0, device='cuda', requires_grad=False)
        self.epoch = 0
        self.task = task
        self.wandb_log = wandb_log

    def update(self, 
               batch_accuracy: float, 
               batch_size: float):
        self.correct += batch_accuracy
        self.total += batch_size
    
    def reset(self):
        self.correct.zero_()
        self.total.zero_()
        self.epoch += 1
    
    def synchronize(self):
        dist.all_reduce(self.correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.total, op=dist.ReduceOp.SUM)
    
    def log(self):
        self.synchronize()
        
        if self.total > 0:
            acc = (self.correct / self.total).item()
        else:
            acc = 0.0

        if self.wandb_log and dist.get_rank() == 0:
            wandb.log({"epoch": self.epoch, f"{self.task}_acc": acc})
        
        result = acc
        self.reset()
        return result


@torch.compile
def KeLu(x:torch.Tensor, a:float = 3.5)->torch.tensor:
    return torch.where(x < -a, 0, torch.where(x > a, x, 0.5*x*(1+x/a+(1/torch.pi)*torch.sin(x*torch.pi/a))))


@torch.compile
def BCEWithLogitsLoss(num_classes:int = 1000,
                      label_smoothing:float = 0.1)->Callable[[torch.Tensor, torch.Tensor], torch.Tensor] :
    ## Target is of shape (B, num_classes), we shall do some label smoothing here!!!
    
    def loss(input:torch.Tensor, target:torch.Tensor)->torch.tensor:
        if target.dim() == 1:
            target = torch.nn.functional.one_hot(target, num_classes)
        ## Here we do some label smoothing
        target_smoothed = target*(1-label_smoothing) + label_smoothing/num_classes
    
        return torch.nn.functional.binary_cross_entropy_with_logits(input, target_smoothed)
    
    return loss

"""
BCEWithLogitsLoss()(torch.randn(1, 1000), torch.rand(1, 1000))

a = torch.zeros(1000)
a[0] = 1

a*(1-100/1000) + 100/1000
"""

class MeasureTime:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.stop = torch.cuda.Event(enable_timing=True)
        

    def __enter__(self):
        self.start.record()

    def __exit__(self, *args):
        # Record the stop time
        self.stop.record()
        torch.cuda.synchronize()  
        elapsed_time = self.start.elapsed_time(self.stop)
        print(f"Elapsed time: {elapsed_time/1000:.2f} seconds")

if __name__ == "__main__":
    pass