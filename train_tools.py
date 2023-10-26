import pickle
import os
import torch
from torch.distributed import (
    all_reduce,
    ReduceOp,
)
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        val_data: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        gpu_id: int,
        save_every: int,
        val_loss_logger=None,
        train_loss_logger=None,
        val_accuracy_logger=None,
        compile=False
        # tracker ## this dude is for tracking stuff
    ) -> None:
        self.gpu_id = gpu_id
        self.model_config = model.config
        self.model = model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[gpu_id], find_unused_parameters=True)

        if compile:
            self.model = torch.compile(self.model)
        ##
        self.train_data = train_data
        self.val_data = val_data
        ##
        self.optimizer = optimizer
        self.scheduler = scheduler
        ##
        self.save_every = save_every
        ##
        self.val_loss_logger = val_loss_logger
        self.train_loss_logger = train_loss_logger
        self.val_accuracy_logger = val_accuracy_logger
        ##
        self.autocast = torch.autocast
        self.scaler = torch.cuda.amp.GradScaler()
        try:
            self._load_checkpoint("checkpoint.pt")
            print("Started from where we stopped last time!!!")
        except Exception as e:
            print(f"There is a problem with loading the model weights and the problem is: {e}")
        
    def _run_batch(self, source, targets, i):
        ### All the things like low precision training will happen here dude!!!
        self.model.train() ## Model in train mode!!!
        self.optimizer.zero_grad()
        with self.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.model(source, task = None)
            loss = F.cross_entropy(output, targets)

        if i % 100 == 0:
            print(f"loss {loss.item()}, {i} batch, from gpu {self.gpu_id} ")
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        ### We log the loss ### 

 
    def _run_epoch(self, epoch, report_in_every = 100):
        # b_sz = len(next(iter(self.train_data))[0])
        if epoch % report_in_every == 0:
            print(f"[GPU{self.gpu_id}] Epoch {epoch}")
        self.train_data.sampler.set_epoch(epoch)

        for i, (source, targets) in enumerate(self.train_data):
            source = source.to(self.gpu_id, non_blocking=True)
            targets = targets.to(self.gpu_id, non_blocking=True)
            self._run_batch(source, targets, i)

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
               self._save_checkpoint(epoch)
            self.validate()

    def validate(self):
        if self.gpu_id == 0:
            print("Validation is started!!!")
        self.model.eval()
        with torch.no_grad():  ## block tracking gradients
            for source, targets, _ in self.val_data:
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                output = self.model(source, task = None)  
                loss = F.cross_entropy(output, targets)
                self.val_loss_logger.update(loss.item())
                accuracy = (output.argmax(-1) == targets).float().mean()
                #print(self.val_accuracy_logger.update(accuracy.item()))
                self.val_accuracy_logger.update(accuracy.item())

            self.val_loss_logger.all_reduce()
            self.val_accuracy_logger.all_reduce()
            if self.gpu_id == 0:
                print(self.val_loss_logger.get_avg_loss(), self.val_accuracy_logger.accuracy)
                
                self.val_loss_logger.reset()
                self.val_accuracy_logger.reset()    
    ## Some tools ## 
    def _load_checkpoint(self, checkpoint_file):
        model_dict = torch.load(checkpoint_file)
        ### Now the state dict are obtained below ###
        model_state_dict = model_dict["model_state_dict"]
        model_optimizer_state = model_dict["optimizer_state"]
        ### ---Let's load the model states--- ###
        #self.model = self.model.from_dict(model_config)
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(model_optimizer_state)
        print("Loaded the new model!!!!")
 
    def _save_checkpoint(self, epoch):
        ### This are the necessary steps to recover the model from the pickled file!!!
        model_weights = self.model.state_dict()
        model_config = self.model_config
        optimizer_state = self.optimizer.state_dict()

        checkpoint = {"model_state_dict":model_weights,
                      "model_config":model_config,
                      "optimizer_state":optimizer_state,
                      "epoch":epoch
                    }
        try:
            PATH = "checkpoint.pt"
            torch.save(checkpoint, PATH)        
            print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
        except Exception as exp:
            print(f"Something went wrong with {exp}, the training will start from begining!!!")
    



class distributed_loss_track:
    ## This is from one for all repo!!!
    def __init__(self, project="at_net_pred", file_name: str = "loss.log"):
        self.project = project
        self.file_name = file_name
        self.temp_loss = 0
        self.counter = 1
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
        ## we have very little number in the denominator
        ## to avoid overflow!!!
        return self.temp_loss / (self.counter+1e-4)

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            loss_tensor = torch.tensor(
            [self.temp_loss, self.counter], device=device, dtype=torch.float32
            )
            all_reduce(loss_tensor, ReduceOp.SUM, async_op=False)
            self.temp_loss, self.counter = loss_tensor.tolist()


    def save_log(self):
        ### no need to call this dude unless you really need!!!
        dict_ = {f"epoch-{self.epoch}": self.temp_loss}
        with open(f"{self.epoch}_epoch" + self.file_name, mode="ab") as file:
            pickle.dump(dict_, file)

    @property
    def loss(self):
        return self.temp_loss, self.counter 





class track_accuracy:
    def __init__(self, initial_acc = 0.1):
        self.temp_acc = initial_acc
        self.counter = 0

    def update(self, batch_acc):
        self.temp_acc += (batch_acc- self.temp_acc)/(self.counter+1)
        self.counter += 1

    def reset(self):
        self.counter = 0
        self.temp_acc = 0.0

    @property
    def accuracy(self):
        ### This is for logging purposses 
        ### should be called at the end of each epoch!!!
        ## This dude takes average of accuracies from different processes
        if self.counter != 0:
            return self.temp_acc
        return 1e-10

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        loss_tensor = torch.tensor(
        [self.temp_acc], device=device, dtype=torch.float32
             )
        all_reduce(loss_tensor, ReduceOp.AVG, async_op=False)
        self.temp_acc = loss_tensor.item()
