from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# System Settings
# -----------------------------------------------------------------------------
# Matches self.device = cfg.DEVICE in trainer
_C.DEVICE = "cuda"
_C.NUM_WORKERS = 4
_C.SEED = 42
_C.LOG_DIR = "runs"
_C.PIN_MEMORY = False
_C.CHECKPOINT_DIR = "checkpoints"
_C.CHECKPOINT_INTERVAL = 5
# -----------------------------------------------------------------------------
# Data Paths and Checkpoints
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Root directory where all datasets are located
_C.DATA.DATASET_ROOT = "dataset"

# Specific paths relative to DATASET_ROOT
_C.DATA.RESIDE_INDOOR_PATH = "indoor-training-set"
_C.DATA.HAZE4K_PATH = "haze4k"

_C.DATA.TRAIN_RATIO = 0.8  # Train/Val split ratio for RESIDE_Indoor dataset

# -----------------------------------------------------------------------------
# Training Parameters
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.EPOCHS = 100
_C.TRAIN.RESUME_PATH = ""

# -----------------------------------------------------------------------------
# Optimization (AdamW)
# -----------------------------------------------------------------------------
_C.OPTIM = CN()
_C.OPTIM.LR = 2e-4
_C.OPTIM.WEIGHT_DECAY = 1e-4
_C.OPTIM.BETA1 = 0.9
_C.OPTIM.BETA2 = 0.999

# -----------------------------------------------------------------------------
# Scheduler (StepLR)
# -----------------------------------------------------------------------------
_C.SCHEDULER = CN()
_C.SCHEDULER.STEP_SIZE = 50  # Decay every 50 epochs
_C.SCHEDULER.GAMMA = 0.5  # Decay factor

# -----------------------------------------------------------------------------
# Loss Weights
# -----------------------------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.W_FLOW = 1.0  # Flow Matching (MSE of velocity)
_C.LOSS.W_PIXELS = 1.0  # Reconstruction (L1/L2)
_C.LOSS.W_PERC = 0.1  # Total Perceptual Loss weight
_C.LOSS.W_GEN = 0.01  # Adversarial Generator weight


# Perceptual Loss specific configuration
_C.LOSS.PERCEPTUAL = CN()
_C.LOSS.PERCEPTUAL.CONTENT = 1.0  # Weight for Feature Loss
_C.LOSS.PERCEPTUAL.STYLE = 1000.0  # Weight for Gram Matrix Style Loss
_C.LOSS.PERCEPTUAL.VGG_BACKBONE = "VGG16"


# -----------------------------------------------------------------------------
# Progressive Training Schedule (List of Stages)
# -----------------------------------------------------------------------------
# This list MUST be initialized as empty.
# It will be completely overwritten by the list defined in
# 'configs/train_cfgs/pretrain_schedule.yaml' when you call cfg.merge_from_file().
_C.SCHEDULE = []


# -----------------------------------------------------------------------------
# Config Helper
# -----------------------------------------------------------------------------
def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so we can't accidentally alter the global instance
    return _C.clone()

