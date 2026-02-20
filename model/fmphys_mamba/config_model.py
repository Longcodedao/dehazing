# config_model.py
from yacs.config import CfgNode as CN

def get_model_cfg_defaults():
    _C = CN()
    
    # Meta
    _C.VERSION = "small" # helpful for logging
    
    # Architecture Dimensions
    _C.BASE_DIM = 48
    _C.DIM_MULTS = [1, 2, 4, 8]
    _C.TIME_DIM_MULT = 4
    
    # Feature Flags
    _C.PHYSICS_GUIDED = True
    _C.USE_CHECKPOINT = False # Gradient checkpointing
    
    # Depth Control (BiMamba Blocks per stage)
    # Length should match len(DIM_MULTS) - 1
    _C.ENCODER_BLOCKS = [1, 1, 1] 
    _C.DECODER_BLOCKS = [1, 1, 1]
    
    return _C.clone()


