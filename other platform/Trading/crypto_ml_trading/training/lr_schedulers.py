"""
Learning Rate Schedulers
Factory for creating model-specific learning rate schedulers.
"""

from typing import Dict, Any, Optional
import math
import logging

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    _LRScheduler,
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau,
    LambdaLR,
    StepLR,
    ExponentialLR,
)

logger = logging.getLogger(__name__)


# Default scheduler configurations per model type
SCHEDULER_CONFIGS: Dict[str, Dict[str, Any]] = {
    'gru_attention': {
        'type': 'cosine_warm_restarts',
        'T_0': 10,  # Initial restart period (epochs)
        'T_mult': 2,  # Multiply period after each restart
        'eta_min': 1e-6,  # Minimum learning rate
    },
    'transformer': {
        'type': 'noam',  # Warmup + inverse sqrt decay (from "Attention is All You Need")
        'warmup_steps': 4000,
        'd_model': 512,  # Model dimension for scaling
    },
    'lstm': {
        'type': 'reduce_on_plateau',
        'patience': 5,
        'factor': 0.5,
        'min_lr': 1e-6,
        'mode': 'min',  # Reduce when loss stops decreasing
    },
    'bilstm': {
        'type': 'reduce_on_plateau',
        'patience': 5,
        'factor': 0.5,
        'min_lr': 1e-6,
        'mode': 'min',
    },
    'tcn': {
        'type': 'one_cycle',
        'max_lr_factor': 5,  # max_lr = initial_lr * factor
        'pct_start': 0.3,  # Fraction of cycle spent increasing LR
        'anneal_strategy': 'cos',
        'div_factor': 25,  # initial_lr = max_lr / div_factor
        'final_div_factor': 1e4,  # min_lr = max_lr / final_div_factor
    },
    'cnn_pattern': {
        'type': 'cosine_annealing',
        'T_max': 50,  # Maximum number of epochs
        'eta_min': 1e-6,
    },
    'tft': {
        'type': 'noam',
        'warmup_steps': 4000,
        'd_model': 256,
    },
    'ppo': {
        'type': 'linear_decay',
        'end_factor': 0.1,  # Final LR = initial * end_factor
    },
}


class NoamScheduler(_LRScheduler):
    """
    Noam learning rate scheduler from "Attention is All You Need".

    LR = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = self.d_model ** (-0.5) * min(
            step ** (-0.5),
            step * self.warmup_steps ** (-1.5)
        )
        return [base_lr * scale for base_lr in self.base_lrs]


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing with linear warmup.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            # Linear warmup
            warmup_factor = step / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


class LinearDecayScheduler(_LRScheduler):
    """
    Linear decay from initial LR to final LR.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        end_factor: float = 0.1,
        last_epoch: int = -1,
    ):
        self.total_steps = total_steps
        self.end_factor = end_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = min(self.last_epoch + 1, self.total_steps)
        factor = 1.0 - (1.0 - self.end_factor) * (step / self.total_steps)
        return [base_lr * factor for base_lr in self.base_lrs]


def create_scheduler(
    model_type: str,
    optimizer: Optimizer,
    total_steps: Optional[int] = None,
    total_epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    custom_config: Optional[Dict[str, Any]] = None,
) -> _LRScheduler:
    """
    Create appropriate learning rate scheduler for model type.

    Args:
        model_type: Model type identifier (e.g., 'gru_attention', 'transformer')
        optimizer: PyTorch optimizer
        total_steps: Total training steps (for step-based schedulers)
        total_epochs: Total training epochs (for epoch-based schedulers)
        steps_per_epoch: Steps per epoch (for converting epochs to steps)
        custom_config: Override default config with custom values

    Returns:
        PyTorch learning rate scheduler
    """
    # Get default config and merge with custom
    config = SCHEDULER_CONFIGS.get(model_type, {'type': 'reduce_on_plateau'}).copy()
    if custom_config:
        config.update(custom_config)

    scheduler_type = config.get('type', 'reduce_on_plateau')

    # Calculate total steps if needed
    if total_steps is None and total_epochs is not None and steps_per_epoch is not None:
        total_steps = total_epochs * steps_per_epoch

    logger.info(f"Creating {scheduler_type} scheduler for {model_type}")

    if scheduler_type == 'cosine_warm_restarts':
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get('T_0', 10),
            T_mult=config.get('T_mult', 2),
            eta_min=config.get('eta_min', 1e-6),
        )

    elif scheduler_type == 'cosine_annealing':
        T_max = config.get('T_max', total_epochs or 100)
        return CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=config.get('eta_min', 1e-6),
        )

    elif scheduler_type == 'noam':
        return NoamScheduler(
            optimizer,
            d_model=config.get('d_model', 512),
            warmup_steps=config.get('warmup_steps', 4000),
        )

    elif scheduler_type == 'warmup_cosine':
        if total_steps is None:
            raise ValueError("total_steps required for warmup_cosine scheduler")
        return WarmupCosineScheduler(
            optimizer,
            warmup_steps=config.get('warmup_steps', total_steps // 10),
            total_steps=total_steps,
            min_lr=config.get('min_lr', 1e-6),
        )

    elif scheduler_type == 'one_cycle':
        if total_steps is None:
            raise ValueError("total_steps required for one_cycle scheduler")

        base_lr = optimizer.defaults['lr']
        max_lr = base_lr * config.get('max_lr_factor', 5)

        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=config.get('pct_start', 0.3),
            anneal_strategy=config.get('anneal_strategy', 'cos'),
            div_factor=config.get('div_factor', 25),
            final_div_factor=config.get('final_div_factor', 1e4),
        )

    elif scheduler_type == 'linear_decay':
        if total_steps is None:
            raise ValueError("total_steps required for linear_decay scheduler")
        return LinearDecayScheduler(
            optimizer,
            total_steps=total_steps,
            end_factor=config.get('end_factor', 0.1),
        )

    elif scheduler_type == 'step':
        return StepLR(
            optimizer,
            step_size=config.get('step_size', 30),
            gamma=config.get('gamma', 0.1),
        )

    elif scheduler_type == 'exponential':
        return ExponentialLR(
            optimizer,
            gamma=config.get('gamma', 0.95),
        )

    else:  # Default: reduce_on_plateau
        return ReduceLROnPlateau(
            optimizer,
            mode=config.get('mode', 'min'),
            patience=config.get('patience', 5),
            factor=config.get('factor', 0.5),
            min_lr=config.get('min_lr', 1e-6),
            verbose=True,
        )


def get_scheduler_config(model_type: str) -> Dict[str, Any]:
    """
    Get the default scheduler configuration for a model type.

    Args:
        model_type: Model type identifier

    Returns:
        Dictionary with scheduler configuration
    """
    return SCHEDULER_CONFIGS.get(model_type, {'type': 'reduce_on_plateau'}).copy()


def is_epoch_based_scheduler(scheduler: _LRScheduler) -> bool:
    """
    Check if scheduler should be stepped per epoch (vs per batch).

    Args:
        scheduler: PyTorch scheduler

    Returns:
        True if scheduler should be stepped per epoch
    """
    epoch_based = (
        CosineAnnealingLR,
        CosineAnnealingWarmRestarts,
        StepLR,
        ExponentialLR,
        ReduceLROnPlateau,
    )
    return isinstance(scheduler, epoch_based)


def requires_metric(scheduler: _LRScheduler) -> bool:
    """
    Check if scheduler requires a metric value when stepping.

    Args:
        scheduler: PyTorch scheduler

    Returns:
        True if scheduler.step() requires a metric
    """
    return isinstance(scheduler, ReduceLROnPlateau)
