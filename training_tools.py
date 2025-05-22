import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import os
from torch import nn as nn
from typing import Any
import training_utilities as t

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
        val_loss_logger:Any = None,
        train_loss_logger:Any = None,
        val_accuracy_logger:Any = None,
        compile_model:bool = False,
        snapshot_name:str = "model.pt",
        snapshot_dir:str = "model",
        total_epochs:int = 300,
        report_every_epoch:int =1, 
        ema_decay:float = 0.999,
        use_cross_entropy:bool = True,
        teacher_model:Any = None,
        label_smoothing:float = 0.1
    ) -> None:
        self.gpu_id = gpu_id
        self.model_config = model.config
        self.model = model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[gpu_id])
        #
        if compile_model:
            self.model = torch.compile(self.model,  dynamic = True, fullgraph=False)
        if ema_decay > 0:
            self.ema_model = EMA_model(self.model, decay=ema_decay)
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
        self.report_in_every = report_every_epoch
        self.epoch = 0
        #
        self.val_loss_logger = val_loss_logger
        self.train_loss_logger = train_loss_logger
        self.val_accuracy_logger = val_accuracy_logger
        #  Mixed precision training
        self.autocast = torch.autocast
        self.scaler = torch.amp.GradScaler('cuda')
        # Recover from a checkpoint!
        try:
            self._load_checkpoint(self.PATH)
        except Exception as e:
            print(f"There is a problem with loading the model weights and the problem is: {e}")
        ## loss_fn
        if use_cross_entropy:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing = label_smoothing).to(gpu_id)
        else:
            self.loss_fn = t.BCEWithLogitsLoss(label_smoothing = label_smoothing, num_classes = 1000)

        
    def _run_batch(self, 
                   source: torch.tensor, 
                   targets: torch.tensor, 
                   batch_enum = None)-> float:
        # -- Zero the grads on the graph -- #
        self.optimizer.zero_grad(set_to_none=True)
        

        with self.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.model(source)
            ## Use here binary-cross entropy loss instead of cross entropy loss
            loss = self.loss_fn(output, 
                                   targets)

        self.scaler.scale(loss).backward()

        
        if (batch_enum+1)%self.grad_accum_steps == 0:
            # every 2 iterations we update the grads!!!
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        
        ### We return the batch loss for logging purposses ###
        return loss.item()


    def _run_epoch(self):

        init_start = torch.cuda.Event(enable_timing=True)
        init_end = torch.cuda.Event(enable_timing=True)
        for i, (source, targets) in enumerate(self.train_data):
            source, targets = source.to(self.gpu_id, non_blocking=True), targets.to(self.gpu_id, non_blocking=True)
            init_start.record()
            batch_loss = self._run_batch(source, targets, batch_enum = i)
            init_end.record()
            torch.cuda.synchronize()
            # log the batch loss onto local logger!!!
            # update ema
            if hasattr(self, "ema_model"):
                self.ema_model.update_parameters(self.model)
            self.train_loss_logger.update(batch_loss)
            ## print the loss
            if (self.gpu_id == 0) and i % 10 == 0:
                print(f"{i} Batch passed the average loss is {batch_loss}, lr is {self.scheduler.get_last_lr()} -- {init_start.elapsed_time(init_end) / 1000}secs to pass a batch!")
            ### -- ###

    def train(self):
                
        for epoch in range(self.epoch+1, self.total_epochs):
            
            """
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize() 
            print(f"elapsed time: {init_start.elapsed_time(init_end) / 1000}secs")
            """
            self.epoch = epoch
            if epoch % self.report_in_every == 0:
                print(f"[GPU{self.gpu_id}] Epoch {self.epoch}\n")
            ##
            self.train_data.sampler.set_epoch(self.epoch)
            ## 


            self.model.train() ## Model in train mode!!!
            #init_start.record() ## How much time we spent!!!
            self._run_epoch()
            # ## let's log the loss now!!!
            self.train_loss_logger.log()
            #init_end.record() 
            ##  Epoch is done take one step scheduler, 
            self.scheduler.step()
            #torch.cuda.synchronize() 
            #print(f"elapsed time: {init_start.elapsed_time(init_end) / 1000}secs")
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint()
                if hasattr(self, "ema_model"):
                    self.ema_model.save_ema_model()
            ## Let's validate
            self.validate()
            
            ## You gottta update the EMA model here!!!!

    def validate(self):
        self.model.eval()
        if self.gpu_id == 0:
            print("Validation is started!!!")
        with torch.no_grad():  ## block tracking gradients
    
            for source, targets in self.val_data:
                source, targets = source.to(self.gpu_id, non_blocking=True), targets.to(self.gpu_id, non_blocking=True)
                output = self.model(source)  
                
                loss = nn.CrossEntropyLoss()(output, targets).item()
                accuracy = (output.argmax(-1) == targets).sum().item()

                
                self.val_loss_logger.update(loss)
                self.val_accuracy_logger.update(batch_accuracy = accuracy, batch_size = targets.shape[0])
                #print(loss.item(), accuracy.item(), output.shape, self.val_accuracy_logger.accuracy, self.val_accuracy_logger.epoch)

                            
            val_loss = self.val_loss_logger.log()
            val_acc = self.val_accuracy_logger.log()   

        if self.gpu_id == 0:
            print(f"Epoch | {self.epoch} - validation loss is {val_loss}, accuracy is {val_acc}")
    
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


class TeacherModel:
    def __init__(self, 
                 model: nn.Module, 
                 compile_model:bool = False):
        if compile_model:
            self.model = torch.compile(model)  
        else:
            self.model = model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():  
            return self.model(x)  

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
        




class EMA_model:
    def __init__(self, 
                 model: nn.Module, 
                 decay: float = 0.999,
                 device = "cuda",
                 ema_model_name: str = "ema_model.pt"):
        self.ema_weights = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.decay = decay
        self.ema_model_name = ema_model_name
    def update_parameters(self, 
                          model: nn.Module)-> None:
        with torch.no_grad():
            for keys, values in model.state_dict().items():
                if "num_batches_tracked" or "running" in keys:
                    self.ema_weights[keys].copy_(values)
                self.ema_weights[keys] = self.decay*self.ema_weights[keys] + (1-self.decay)*values.detach()
    def state_dict(self):
        return self.ema_weights
    def save_ema_model(self):
        weights =  {k: v.cpu() for k, v in self.ema_weights.items()}
        torch.save(weights, self.ema_model_name)

        

