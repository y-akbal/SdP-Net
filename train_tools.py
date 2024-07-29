import pickle
import torch
from torch.distributed import (
    all_reduce,
    ReduceOp,
)
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import os

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
        self.snapshot_dir = snapshot_dir
        self.snapshot_name = snapshot_name
        self.PATH = os.path.join(self.snapshot_dir, self.snapshot_name) if os.path.exists(self.snapshot_dir) else None
        #
        if compile_model:
            self.model = torch.compile(self.model)
        ##
        self.train_data = train_data
        self.val_data = val_data
        self.total_epochs = total_epochs
        ##
        self.optimizer = optimizer
        self.scheduler = scheduler
        ##
        self.save_every = save_every
        self.epoch = 0
        ##
        self.val_loss_logger = val_loss_logger
        self.train_loss_logger = train_loss_logger
        self.val_accuracy_logger = val_accuracy_logger
        ##
        self.autocast = torch.autocast
        self.scaler = torch.cuda.amp.GradScaler()
        try:
            self._load_checkpoint(self.PATH)
        except Exception as e:
            print(f"There is a problem with loading the model weights and the problem is: {e}")
        
    def _run_batch(self, source, targets):
        ### All the things like low precision training will happen here dude!!!
        ## Accumulate some gradients here!!!
        
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
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        
        ### We log the loss ### 
        self.train_loss_logger.update(loss.item())
        ## update the loss


 
    def _run_epoch(self, epoch, report_in_every = 1):
        # b_sz = len(next(iter(self.train_data))[0])
        self.epoch = epoch
        if epoch % report_in_every == 0:
            print(f"[GPU{self.gpu_id}] Epoch {self.epoch}\n")
        self.train_data.sampler.set_epoch(self.epoch)
        ## 
        self.model.train() ## Model in train mode!!!
        
        for i, (source, targets) in enumerate(self.train_data):
            source = source.to(self.gpu_id, non_blocking=False)
            targets = targets.to(self.gpu_id, non_blocking=False)
            
            self._run_batch(source, targets)
            
            ### 
            ## sync the losses
            self.train_loss_logger.all_reduce()
            ## prtint the loss
            if (self.gpu_id == 0) and i % 500 == 0:
                batch_loss = self.train_loss_logger.get_avg_loss()
                print(f"{i} Batch passed the average loss is {batch_loss}, lr is {self.scheduler.get_last_lr()}")
            ### -- ###
            self.train_loss_logger.reset()       
        self.scheduler.step()



    def train(self):
                
        for epoch in range(self.epoch+1, self.total_epochs):
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            

            init_start.record() ## How much time we spent!!!
            self._run_epoch(epoch)
            init_end.record() ## let's record it now!!!

            torch.cuda.synchronize() 
            print(f"elapsed time: {init_start.elapsed_time(init_end) / 1000}secs")
            if self.gpu_id == 0 and epoch % self.save_every == 0:
               self._save_checkpoint()
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


"""
## place holder for the wandb_log class, a class to be used by wandb_logging...
class wandb_log:
    
    def __init__(self, user_name, password, **kwargs):
        ## init should be done here
        ## .. --- .. ##
        ## .. ...... ##
        ## ---...----##
        for keys, values in kwargs.items():
            vars(self)[keys] = values
        this includes logging in to wandb and doing some intial stuff...
    @classmethod
    def from_dict(**kwargs):
        pass
    
    def add_tracked(a:dict):
        pass

    def __call__(self, f):
        def wrapper(*args, **kwargs):
            print(args[0].a**2)
            return f(*args)
        return wrapper 

w = wandb_log(**{"temp_loss":0})


class d:
    def __init__(self, a):
        self.a = a
    @w
    def __call__(self, b):
        return (self.a)*b
    def w(self, f):
        return f

"""



class distributed_loss_track:
    ## This is from one for all repo!!!
    def __init__(self, project="at_net_pred", file_name: str = "loss.log"):
        self.project = project
        self.file_name = file_name
        self.temp_loss = 0
        self.counter = 0
        self.epoch = 0

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

"""
t = track_accuracy()
t.update(0.9)
t.accuracy
t.counter
t.reset()"""

