import pickle
import torch
from torch.distributed import (
    all_reduce,
    ReduceOp,
)
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import os
import wandb

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
        gradient_accumulation_steps:int,
        val_loss_logger = None,
        train_loss_logger = None,
        val_accuracy_logger = None,
        compile_model:bool =False,
        snapshot_name:str = "model.pt",
        snapshot_dir:str = "model",
        total_epochs:int = 300,
        use_ema_model:bool = False, 
        # tracker ## this dude is for tracking stuff
    ) -> None:
        self.gpu_id = gpu_id
        self.model_config = model.config
        self.model = model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[gpu_id])
        #
        if compile_model:
            self.model = torch.compile(self.model)
        #
        self.snapshot_dir = snapshot_dir
        self.snapshot_name = snapshot_name
        self.PATH = os.path.join(self.snapshot_dir, self.snapshot_name) if os.path.exists(self.snapshot_dir) else None
        #
        self.train_data = train_data
        self.val_data = val_data
        self.total_epochs = total_epochs
        #
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_accum_steps = gradient_accumulation_steps
        #
        self.save_every = save_every
        self.epoch = 0
        #
        self.val_loss_logger = val_loss_logger
        self.train_loss_logger = train_loss_logger
        self.val_accuracy_logger = val_accuracy_logger
        #  Mixed precision training
        self.autocast = torch.autocast
        self.scaler = torch.cuda.amp.GradScaler()
        # Recover from a checkpoint!
        try:
            self._load_checkpoint(self.PATH)
        except Exception as e:
            print(f"There is a problem with loading the model weights and the problem is: {e}")
        
    def _run_batch(self, 
                   source: torch.tensor, 
                   targets: torch.tensor, 
                   batch_enum = None)-> float:
        # -- Zero the grads on the graph -- #
        self.optimizer.zero_grad()

        with self.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.model(source)
            ## Use here binary-cross entropy loss instead of cross entropy loss
            loss = F.cross_entropy(output, 
                                   targets,
                                   label_smoothing=0.1)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        
        if (batch_enum+1)%self.grad_accum_steps == 0:
            # every 2 iterations we update the grads!!!
            self.scaler.step(self.optimizer)
            self.scaler.update()

        ### We return the batch loss for logging purposses ###
        return loss.item()
        

    def _run_epoch(self, epoch, report_in_every = 1):
        
        self.epoch = epoch
        
        if epoch % report_in_every == 0:
            print(f"[GPU{self.gpu_id}] Epoch {self.epoch}\n")
        ##
        self.train_data.sampler.set_epoch(self.epoch)
        ## 
        
        self.model.train() ## Model in train mode!!!
        
        for i, (source, targets) in enumerate(self.train_data):
            source, targets = source.to(self.gpu_id, non_blocking=False), targets.to(self.gpu_id, non_blocking=False)
            
            batch_loss = self._run_batch(source, targets, batch_enum = i)
            # log the batch loss onto local logger!!!
            self.train_loss_logger.update(batch_loss)
            ## prtint the loss
            if (self.gpu_id == 0) and i % 500 == 0:
                batch_loss = self.train_loss_logger.get_avg_loss()
                print(f"{i} Batch passed the average loss is {batch_loss}, lr is {self.scheduler.get_last_lr()}")
            ### -- ###

    def train(self):
                
        for epoch in range(self.epoch+1, self.total_epochs):
            
            """
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            """

            #init_start.record() ## How much time we spent!!!
            self._run_epoch(epoch)
            #init_end.record() ## let's record it now!!!
            self.train_loss_logger.log_loss()
            ##  Epoch is done take one step scheduler, 
            self.scheduler.step()
            #torch.cuda.synchronize() 
            #print(f"elapsed time: {init_start.elapsed_time(init_end) / 1000}secs")
            if self.gpu_id == 0 and epoch % self.save_every == 0:
               self._save_checkpoint()
            ## You gottta update the EMA model here!!!!
            self.validate()

    def validate(self):
        self.model.eval()
        if self.gpu_id == 0:
            print("Validation is started!!!")
        with torch.no_grad():  ## block tracking gradients
    
            for source, targets, _ in self.val_data:
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                output = self.model(source)  
                loss = F.cross_entropy(output, targets)
                accuracy = (output.argmax(-1) == targets).float().mean()
                self.val_loss_logger.update(loss.item())
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
        model_scheduler_state = model_dict["scheduler_state"]
        
        ### ---Let's load the model states--- ###
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(model_optimizer_state)
        self.scheduler.load_state_dict(model_scheduler_state)
        self.epoch = model_dict["epoch"]
        print(f"Loaded the new model saved at {self.PATH}, will continue training from {self.epoch} epoch")
 
    def _save_checkpoint(self):
          ### If dir does not exist, create it!!
        if not os.path.exists(self.snapshot_dir):
            os.mkdir(self.snapshot_dir)
        self.PATH = os.path.join(self.snapshot_dir, self.snapshot_name) 

        ### This are the necessary steps to recover the model from the pickled file!!!
        model_weights = self.model.state_dict()
        model_config = self.model_config
        optimizer_state = self.optimizer.state_dict()
        scheduler_state = self.scheduler.state_dict()
        checkpoint = {"model_state_dict":model_weights,
                      "model_config":model_config,
                      "optimizer_state":optimizer_state,
                      "scheduler_state":scheduler_state,
                      "epoch":self.epoch
                    }
        

        try:
            torch.save(checkpoint, self.PATH)        
            print(f"Epoch {self.epoch} | Training checkpoint saved at {self.PATH}")
        except Exception as exp:
            print(f"Something went wrong with {exp}, the training will start from begining!!!")



def return_scheduler_optimizer(model, **kwargs):
    ## -- ##
    #################
    ## ----------- ##
    ## -Optimizer- ##
    ## --config--- ##
    #################
    opt_kwargs = kwargs["optimizer_config"]
    optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)
    
    #################
    ## ----------- ##
    ## Scheduler   ##
    ## --config--- ##
    #################
    scheduler_kwargs = kwargs["scheduler_config"]
    ## -- ##
    ### Warm up step starts ### 
    scheduler0 = torch.optim.lr_scheduler.ConstantLR(optimizer, **scheduler_kwargs["constant_scheduler"])
    
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_kwargs["linear_scheduler"])
    ## Scheduler 2 ##
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_kwargs["cosine"])
#   end_of_warmup_steps = scheduler_kwargs["warm_up_steps"]
    ## combine those dudes ### 
    ms1 = scheduler_kwargs["constant_scheduler"]["total_iters"]
    ms2 = scheduler_kwargs["linear_scheduler"]["total_iters"]
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler0, scheduler1, scheduler2], milestones= [ms1, ms1+ms2])
    
    return optimizer, scheduler


class distributed_loss_track:
    ## This is from one for all repo!!!
    def __init__(self, 
                 wandb_log:bool = False,
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
    ##  This is class will be updated by its own worker,
    ##  At the end of the one epoch, the accuracies will be averaged!!! 
    ##  BTW all this stuff will be done on each GPU
    ##  At the end of the day the main worker will tell the result to W & B
    def __init__(self, wandb_log = False):
        self.temp_acc:int = 0
        self.total_size:int = 0
        self.epoch:int = 0
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

    def get_accuracy(self):
        acc = self.__get_accuracy__()
        if self.wandb_log:
            wandb.log({f"Epoch_{self.epoch}_acc:{acc}"})
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

