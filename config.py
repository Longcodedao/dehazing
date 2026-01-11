from yacs.config import CfgNode as CN

_C = CN()

# --- System Settings ---
_C.DEVICE = "cuda"
_C.NUM_WORKERS = 4
_C.SEED = 42
_C.LOG_DIR = "runs"
_C.PIN_MEMORY = True
_C.CHECKPOINT_DIR = "checkpoints"
_C.CHECKPOINT_INTERVAL = 5

# --- Data Paths ---
_C.DATA = CN()
_C.DATA.DATASET_ROOT = "dataset"
_C.DATA.RESIDE_INDOOR_PATH = "indoor-training-set"
_C.DATA_RESIDE_INDOOR_SOTS_PATH = "reside-sots/indoor"
_C.DATA_RESIDE_OUTDOOR_SOTS_PATH = "reside-sots/outdoor"
_C.DATA.HAZE4K_PATH = "haze4k"
_C.DATA.O_HAZE = "o-haze"
_C.DATA.DENSE_HAZE = "dense-haze"
_C.DATA.TRAIN_RATIO = 0.8 

# --- Base Training Params ---
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.EPOCHS = 100
_C.TRAIN.RESUME_PATH = ""
_C.TRAIN.PATIENCE: 5 

# --- Optimization ---
_C.OPTIM = CN()
_C.OPTIM.LR = 5e-4
_C.OPTIM.WEIGHT_DECAY = 1e-4
_C.OPTIM.BETA1 = 0.9
_C.OPTIM.BETA2 = 0.999

# --- Scheduler ---
_C.SCHEDULER = CN()
_C.SCHEDULER.STEP_SIZE = 50 
_C.SCHEDULER.GAMMA = 0.5 
_C.SCHEDULER.WARMUP_EPOCHS = 10

# --- Loss Weights ---
_C.LOSS = CN()
_C.LOSS.W_FLOW = 1.0 
_C.LOSS.W_PERC = 0.1 
_C.LOSS.W_PHYS = 0.2
_C.LOSS.W_TV = 0.01
_C.LOSS.W_ATM = 0.01 

# --- Evaluation ---
_C.EVAL = CN()
_C.EVAL.EVAL_INTERVAL = 2
# I always set batch size = 1 in eval batch size
_C.EVAL.BATCH_SIZE = 16 

# --- Progressive Training Schedule ---
# This list is populated by the YAML file
_C.SCHEDULE = []

def get_cfg_defaults():
    return _C.clone()