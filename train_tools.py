import pickle
import os
import torch
from torch.distributed import (
    all_reduce,
    ReduceOp,
)
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location = loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)




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
        ## This dude takes average of accuracies from different processes
        self.__allreduce__()         
        return self.dist_acc

    def __allreduce__(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            loss_tensor = torch.tensor(
            [self.acc], device=device, dtype=torch.float32
             )
            all_reduce(loss_tensor, ReduceOp.AVG, async_op=False)
            self.dist_acc = loss_tensor.numpy()
        else:
            pass
        
        



class distributed_loss_track:
    ## This is from one for all repo!!!
    def __init__(self, project="at_net_pred", file_name: str = "loss.log"):
        self.project = project
        self.file_name = file_name
        self.temp_loss = 0
        self.counter = 1
        self.loss = []
        self.epoch = 0
        ## Bu kodu yazanlar ne güzel insanlardır, 
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
        ### no need to call this dude unless you really need!!!
        dict_ = {f"epoch-{self.epoch}": self.temp_loss}
        with open(f"{self.epoch}_epoch" + self.file_name, mode="ab") as file:
            pickle.dump(dict_, file)

    @property
    def loss(self):
        return self.temp_loss


