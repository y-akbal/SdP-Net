import os
import torch
import torch.distributed as dist
from torch.distributed import all_reduce, ReduceOp
from torch.distributed import init_process_group, destroy_process_group
from torch import multiprocessing as mp
import os
import numpy as np


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def main():
    ddp_setup()
    tensor = torch.randn(100,100).cuda()
    all_reduce(tensor, ReduceOp.AVG, async_op=False)
    print(tensor)
    print(int(os.environ["LOCAL_RANK"]))
    destroy_process_group()


if __name__ == "__main__":
   main()
 ## This is for torch run 
 ## do torchrun --standalone --nproc_per_node=2  test.py