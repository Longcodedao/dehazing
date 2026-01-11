import torch
import torch.distributed as dist
import os

def setup_ddp():
    """
    Initializes the distributed process group. 
    Expects environment variables (LOCAL_RANK) to be set by torchrun.
    """
    if not dist.is_available():
        raise RuntimeError("Torch distributed is not available on this system.")
        
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def clean_ddp():
    """Destroys the process group if it exists."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """
    Returns True if:
    1. We are in a standard single-GPU run (DDP not initialized).
    2. We are in DDP mode and this is Rank 0.
    """
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def reduce_tensor(tensor):
    """
    Reduces a tensor across all GPUs (averages it).
    If DDP is not initialized, returns the tensor as-is.
    """
    # If not running in DDP, average is just the value itself
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt