from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def get_scheduler(optimizer, cfg):
    # 1. Parse Config
    # Check cfg.TRAIN.EPOCHS or cfg.TRAIN.TOTAL_EPOCHS
    total_epochs = getattr(cfg.TRAIN, "EPOCHS", getattr(cfg.TRAIN, "TOTAL_EPOCHS", 100))

    warmup_epochs = getattr(cfg.SCHEDULER, "WARMUP_EPOCHS", 5)

    if warmup_epochs <= 0: 
        return CosineAnnealingLR(
            optimizer, 
            T_max=total_epochs, 
            eta_min=1e-6
        )

    # 2. Define Schedulers
    # Warmup: Linear increase from start_factor (1%) to 100%
    scheduler_warmup = LinearLR(
        optimizer, 
        start_factor=0.01, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )
    
    # Main: Cosine decay from 100% down to eta_min
    scheduler_cosine = CosineAnnealingLR(
        optimizer, 
        T_max=max(1, total_epochs - warmup_epochs), 
        eta_min=1e-6
    )

    # 3. Combine
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[scheduler_warmup, scheduler_cosine], 
        milestones=[warmup_epochs]
    )
    
    return scheduler