import torch
from utils_ddp import is_main_process


class MemoryProfiler:
    def __init__(self, device):
        self.device = device
        self.last_allocated = 0
        self.last_reserved = 0

        # Reset peak stats at start
        torch.cuda.reset_peak_memory_stats(device)

    def _to_gb(self, bytes_val):
        return bytes_val / 1024**3

    def print_status(self, tag=""):
        # Force sync to get accurate reading
        torch.cuda.synchronize(self.device)

        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        max_allocated = torch.cuda.max_memory_allocated(self.device)

        delta_alloc = allocated - self.last_allocated

        if is_main_process():
            print(f"\n[MEM] --- {tag} ---")
            print(
                f"   Active Used: {self._to_gb(allocated):.2f} GB (Delta: {self._to_gb(delta_alloc):+.2f} GB)"
            )
            print(f"   Cache/Resrv: {self._to_gb(reserved):.2f} GB")
            print(f"   Peak So Far: {self._to_gb(max_allocated):.2f} GB")

        self.last_allocated = allocated
        self.last_reserved = reserved

    def inspect_model(self, model, name="Model"):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        total_size_mb = (param_size + buffer_size) / 1024**2

        if is_main_process():
            print(f"[INFO] {name} Theoretical Size: {total_size_mb:.2f} MB")
