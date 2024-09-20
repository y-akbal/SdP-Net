import torch
from torch.distributed import (
    all_reduce,
    ReduceOp,
)
import wandb
import os

class distributed_loss_track:

    def __init__(self, 
                 wandb_log:bool = True,
                 task:str = "Train"):
        
        self.temp_loss:float = 0
        self.counter:int = 0
        self.epoch:int = 0
        self.task = task
        self.log = wandb_log


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

        if self.log:
                ## Only the main worker will log the loss -- else all workers will log the loss!!!
                wandb.log({f"{self.task}_{self.epoch}_loss":self.temp_loss/self.counter})
        loss = self.temp_loss/self.counter
        self.reset()
        return loss

        
class track_accuracy:
    def __init__(self, 
                 task = "Validation", 
                 wandb_log = True):
        self.temp_acc:int = 0
        self.total_size:int = 0
        self.epoch:int = 0
        self.task = task
        self.wandb_log = wandb_log

    def update(self, 
               batch_accuracy:int, 
               batch_size:int):
        self.temp_acc += batch_accuracy
        self.total_size += batch_size
    
    def __all_reduce__(self):
        loss_tensor = torch.tensor([self.temp_acc, self.total_size], dtype=torch.float32).cuda()
        all_reduce(loss_tensor, ReduceOp.SUM, async_op=False)
        self.temp_acc, self.total_size = loss_tensor.cpu().tolist()

    def reset(self):
        self.temp_acc = 0.0
        self.total_size = 0
        self.epoch += 1
    
    def log(self):
        try:
            self.__all_reduce__()
        except:
            print("Error in all reducing accuracy")

        if self.wandb_log:
            wandb.log({f"{self.task}_{self.epoch}_acc":self.temp_acc/self.total_size})
        acc = self.temp_acc/self.total_size
        self.reset()
        return acc


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

