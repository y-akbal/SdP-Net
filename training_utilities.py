import torch
from torch.distributed import (
    all_reduce,
    ReduceOp,
)
import wandb




class distributed_loss_track:
    ## This is from one for all repo!!!
    def __init__(self, 
                 wandb_log:bool = True,
                 task:str = "Train"):

        self.temp_loss:float = 0
        self.counter:int = 0
        self.epoch:int = 0
        self.task = task

        self.log = wandb_log

    def update(self, loss):
        self.temp_loss += loss
        self.counter += 1
        
    def reset(self):
        self.temp_loss, self.counter = 0.0, 0
        self.epoch += 1
        
    def get_loss(self, 
                 additional_log = None):
        avg_loss = self.__get_avg_loss__()
        if self.log:
            wandb.log({f"Epoch_{self.epoch}_{self.task}_loss": avg_loss})
        return avg_loss

    def log_loss(self, additional_log = None)->None:
        self.get_loss(additional_log)
        self.reset()

    def __get_avg_loss__(self):
        ## we have very little number in the denominator
        ## to avoid overflow!!!
        try:
            self.all_reduce()
        except Exception:
            Warning("Buddy something wrong with you GPU!!!! We can not sync the values!!!!")
        return self.temp_loss / (self.counter+1e-5)

    def __all_reduce__(self):
            loss_tensor = torch.tensor(
            [self.temp_loss, self.counter], dtype=torch.float32
            ).cuda()
            ## We send the stuff to GPU and aggreage it !!!!
            all_reduce(loss_tensor, ReduceOp.SUM, async_op=False)
            ##  We then get back to cpu 
            self.temp_loss, self.counter = loss_tensor.tolist()

class track_accuracy:
    ##  This is class will be updated by its own worker,ß
    ##  At the end of the one epoch, the accuracies will be averaged!!! 
    ##  BTW all this stuff will be done on each GPU
    ##  At the end of the day the main worker will tell the result to W & B
    def __init__(self, 
                 task = "Validation", 
                 wandb_log = True):
        self.temp_acc:int = 0
        self.total_size:int = 0
        self.epoch:int = 0,
        self.task = task,
        self.wandb_log = wandb_log

    def update(self, 
               batch_accuracy:int, 
               batch_size:int):
        self.temp_acc += batch_accuracy
        self.total_size += batch_size

    def reset(self):
        self.temp_acc = 0
        self.total_size = 0
        self.epoch += 1
    
    def log(self):
        self.get_accuracy()
        self.reset()
    
    @property
    def accuracy(self):
        return self.temp_acc/self.total_size

    def get_accuracy(self):
        acc = self.__get_accuracy__()
        if self.wandb_log:
            wandb.log({f"Epoch_{self.epoch}_{self.task}_acc:{acc}"})
        return acc

    def __get_accuracy__(self):
        self.__all_reduce__()
        return self.temp_acc/self.total_size

    def __all_reduce__(self):
        loss_tensor = torch.tensor([self.temp_acc, self.total_size], dtype=torch.float32).cuda()
        all_reduce(loss_tensor, ReduceOp.SUM, async_op=False)
        self.temp_acc, self.total_size = loss_tensor.item()
        
"""
t = track_accuracy()
t.update(0.9)
t.accuracy
t.counter
t.reset()"""

