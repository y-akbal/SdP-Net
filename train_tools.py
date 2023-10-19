import pickle
import os
import torch
from torch.distributed import (
    all_reduce,
    ReduceOp,
)

class track_accuracy:
    def __init__(self, initial_acc = 0.0):
        self.acc = initial_acc
        self.dist_acc = initial_acc
        self.counter = 1
    def update(self, batch_acc):
        self.counter += 1
        self.t += batch_acc
        self.acc += (batch_acc -  self.acc)/self.counter
    def reset(self):
        self.counter = 1
        self.acc = 0.0
    @property
    def accuracy(self):
        ### This is for logging purposses 
        ### should be called at the end of each epoch!!!
        ## This dude takes average of accuracies from difference processes
        self.__allreduce__()         
        return self.dist_acc

    def __allreduce__(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        loss_tensor = torch.tensor(
            [self.acc], device=device, dtype=torch.float32
        )
        all_reduce(loss_tensor, ReduceOp.AVG, async_op=False)
        self.dist_acc = loss_tensor.numpy()



class distributed_loss_track:
    ## This is from one for all repo!!!
    def __init__(self, project="at_net_pred", file_name: str = "loss.log"):
        self.project = project
        self.file_name = file_name
        self.temp_loss = 0
        self.counter = 1
        self.loss = []
        self.epoch = 0
        ## Bu kodu yazanlar ne güzel mühendislerdir, 
        ## onların supervisorları ne
        ## iyi supervisorlardır

    def update(self, loss, epoch = None):
        self.temp_loss += loss
        self.counter += 1
        if epoch is not None:
            self.epoch += epoch

    def reset(self):
        self.temp_loss, self.counter = 0, 0

    def get_avg_loss(self):
        return self.temp_loss / self.counter

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        loss_tensor = torch.tensor(
            [self.temp_loss, self.counter], device=device, dtype=torch.float32
        )
        all_reduce(loss_tensor, ReduceOp.SUM, async_op=False)
        self.temp_loss, self.counter = loss_tensor.tolist()
        self.loss.append(self.temp_loss)

    def save_log(self):
        dict_ = {f"epoch-{self.epoch}": self.temp_loss}
        with open(f"{self.epoch}_epoch" + self.file_name, mode="ab") as file:
            pickle.dump(dict_, file)

    @property
    def loss(self):
        return self.temp_loss


