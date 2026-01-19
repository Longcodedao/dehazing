import torch
from torch.cuda import memory_summary
from model import FM_PhysMamba_UNET

# --- CONFIG ---
DEVICE = "cuda"
INPUT_SIZE = (1, 3, 256, 256) # Batch size 1, 256x256 image
BASE_DIM = 64  # Your current dim (change to 64 if that's what crashed)

print(f"🔍 Profiling Model with Base Dim: {BASE_DIM} on {INPUT_SIZE}...")

# 1. Initialize Model
model = FM_PhysMamba_UNET(model_cfg_path="small", use_version=2).to(DEVICE)
model.eval() # Set to eval first to measure pure weights vs activations

# 2. Reset Max Memory Tracker
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

# 3. Create Dummy Data
x = torch.randn(INPUT_SIZE).to(DEVICE)
t = torch.randint(0, 1000, (INPUT_SIZE[0],)).to(DEVICE)

# 4. Profile FORWARD Pass
try:
    with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
        v_pred, t_map, A_pred = model(x, t)
        
        # Fake Loss for Backward profiling
        loss = v_pred.sum()
        loss.backward()

    # 5. Print Results
    print("\n--- 📊 Memory Breakdown (Top 10 Operations) ---")
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    # 6. Overall Stats
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n--- 📉 Peak Memory Usage: {peak_mem:.2f} GB ---")
    
except RuntimeError as e:
    print(f"\n❌ CRASHED during profiling: {e}")