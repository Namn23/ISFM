""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""

from .cosine_lr import CosineLRScheduler
from .linear_lr import LinearWarmupMultiStepDecayLRScheduler


def create_scheduler(cfg, optimizer, len_lorder):
    if cfg.OPTIMIZER.SCHEDULER == "linear":
        last_epoch = cfg.MODEL.LAST_EPOCH
        lr_scheduler = LinearWarmupMultiStepDecayLRScheduler(
            optimizer, cfg.OPTIMIZER.WARMUP_STEPS, cfg.OPTIMIZER.WARMUP_RATE, cfg.OPTIMIZER.DECAY_RATE,
            cfg.OPTIMIZER.EPOCHS, cfg.OPTIMIZER.DECAY_EPOCHS, len_lorder,
            last_epoch=len_lorder * last_epoch - 1, override_lr=cfg.OPTIMIZER.OVERRIDE_LR)
    elif cfg.OPTIMIZER.SCHEDULER == "cosine":
        lr_min = 0.05 * cfg.OPTIMIZER.LR
        warmup_lr_init = 0.1 * cfg.OPTIMIZER.LR

        warmup_t = cfg.OPTIMIZER.WARMUP_ITER
        noise_range = None

        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=cfg.OPTIMIZER.EPOCHS,
            lr_min=lr_min,
            t_mul=1.,
            decay_rate=cfg.OPTIMIZER.DECAY_RATE, # 0.1
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct=0.67,
            noise_std=1.,
            noise_seed=42,
        )
    else:
        raise ValueError("wrong scheduler!")

    return lr_scheduler
