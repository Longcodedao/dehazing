import torch
import torch.distributed as dist
import os


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def clean_ddp():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0.0


def reduce_tensor(tensor):
    """Reduces a tensor across all GPUs (averages it) for metric logging."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
