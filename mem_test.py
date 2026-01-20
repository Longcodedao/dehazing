import torch
from model import FM_PhysMamba_UNET

# Try to import thop, if not available, instruct user
try:
    from thop import profile, clever_format
except ImportError:
    print("❌ Library 'thop' not found. Please run: pip install thop")
    exit()

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
INPUT_SIZE = (1, 3, 256, 256) # Batch size 1 is standard for GFLOPs calculation
BASE_DIM = 64 

print(f"🔍 Profiling Model GFLOPs & Params with Base Dim: {BASE_DIM}...")

# 1. Initialize Model
model = FM_PhysMamba_UNET(model_cfg_path="large", use_version=2).to(DEVICE)
model.eval() 

# 2. Create Dummy Data
x = torch.randn(INPUT_SIZE).to(DEVICE)
t = torch.randint(0, 1000, (INPUT_SIZE[0],)).to(DEVICE)

# 3. Profile GFLOPs and Params using THOP
# Note: thop measures MACs (Multiply-Accumulate Operations)
# Standard GFLOPs ≈ 2 * MACs
print("\n--- ⏳ Running Profiler (this may take a second) ---")

# custom_ops may be needed if Mamba has custom kernels, 
# but usually thop handles standard layers fine.
macs, params = profile(model, inputs=(x, t), verbose=False)

# 4. Convert and Format
gmacs = macs / 1e9
gflops = gmacs * 2 # Approximate FLOPs
params_m = params / 1e6

# 5. Print Results
print("\n" + "="*40)
print(f"   📊 MODEL COMPLEXITY REPORT")
print("="*40)
print(f"   Resolution:      {INPUT_SIZE[2]}x{INPUT_SIZE[3]}")
print(f"   Parameters:      {params_m:.2f} Million")
print(f"   MACs (GMACs):    {gmacs:.2f} G")
print(f"   GFLOPs (approx): {gflops:.2f} G")
print("="*40)

# 6. Sanity Check: PyTorch Native Parameter Count
# This double-checks the 'thop' result
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n[Sanity Check] Native Torch Count:")
print(f"   Total Params:     {total_params / 1e6:.2f} M")
print(f"   Trainable Params: {trainable_params / 1e6:.2f} M")
