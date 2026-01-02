"""
GPU-Enhanced Training System for Crypto ML Trading.

Implements advanced training features including adaptive batch sizing,
learning rate scheduling, gradient accumulation, and mixed precision training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass
import logging
import time
import os
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.gpu_manager import GPUManager, get_gpu_manager
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for GPU-enhanced training."""
    # Basic settings
    epochs: int = 1000
    initial_batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    
    # GPU settings
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0
    memory_efficient: bool = True
    
    # Adaptive settings
    adaptive_batch_size: bool = True
    max_batch_size: int = 1024
    min_batch_size: int = 8
    
    # Learning rate schedule
    scheduler_type: str = 'cosine'  # 'cosine', 'plateau', 'cyclic', 'onecycle'
    warmup_epochs: int = 10
    min_lr: float = 1e-7
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 50
    min_delta: float = 0.0001
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_every: int = 10
    keep_best_only: bool = True
    
    # Monitoring
    log_interval: int = 10
    tensorboard: bool = True
    profile: bool = False


class CryptoDataset(Dataset):
    """GPU-optimized dataset for crypto trading data."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 sequence_length: int = 60, gpu_manager: Optional[GPUManager] = None):
        """
        Initialize dataset.
        
        Args:
            features: Feature array
            targets: Target array
            sequence_length: Sequence length for time series
            gpu_manager: GPU manager instance
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        self.gpu_manager = gpu_manager or get_gpu_manager()
        
        # Pre-compute valid indices
        self.valid_indices = list(range(len(features) - sequence_length))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        
        x = self.features[start_idx:end_idx]
        y = self.targets[end_idx - 1]
        
        return x, y


class GPUEnhancedTrainer:
    """
    Advanced trainer with GPU optimization and adaptive training.
    
    Features:
    - Automatic batch size optimization
    - Mixed precision training
    - Advanced learning rate scheduling
    - Gradient accumulation
    - Memory-efficient training
    - Multi-GPU support
    """
    
    def __init__(self, config: TrainingConfig, gpu_manager: Optional[GPUManager] = None):
        """
        Initialize GPU-enhanced trainer.
        
        Args:
            config: Training configuration
            gpu_manager: GPU manager instance
        """
        self.config = config
        self.gpu_manager = gpu_manager or get_gpu_manager()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.patience_counter = 0
        
        # Performance tracking
        self.training_history = {
            'loss': [], 'val_loss': [], 'lr': [], 'batch_size': [],
            'gpu_memory': [], 'epoch_time': []
        }
        
        # Setup directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard setup
        if config.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / 'tensorboard'))
        else:
            self.writer = None
    
    def train(self, model: nn.Module, train_data: Dataset, val_data: Dataset,
              optimizer_class: type = torch.optim.AdamW) -> nn.Module:
        """
        Train model with GPU enhancements.
        
        Args:
            model: PyTorch model
            train_data: Training dataset
            val_data: Validation dataset
            optimizer_class: Optimizer class to use
            
        Returns:
            Trained model
        """
        # Move model to GPU
        model = model.to(self.gpu_manager.device)
        
        # Enable gradient checkpointing if memory efficient
        if self.config.memory_efficient:
            self.gpu_manager.enable_gradient_checkpointing(model)
        
        # Optimize batch size
        if self.config.adaptive_batch_size:
            batch_size = self._optimize_batch_size(model, train_data)
        else:
            batch_size = self.config.initial_batch_size
        
        # Create data loaders with optimized settings
        train_loader = self._create_dataloader(train_data, batch_size, shuffle=True)
        val_loader = self._create_dataloader(val_data, batch_size, shuffle=False)
        
        # Setup optimizer
        optimizer = optimizer_class(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Setup learning rate scheduler
        scheduler = self._create_scheduler(optimizer, len(train_loader))
        
        # Setup loss function
        criterion = nn.MSELoss()
        
        # Training loop
        logger.info(f"Starting training with batch size: {batch_size}")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            train_loss = self._train_epoch(
                model, train_loader, criterion, optimizer, scheduler
            )
            
            # Validation phase
            val_loss = self._validate_epoch(model, val_loader, criterion)
            
            # Update learning rate
            if scheduler and self.config.scheduler_type == 'plateau':
                scheduler.step(val_loss)
            
            # Track metrics
            epoch_time = time.time() - epoch_start_time
            self._track_metrics(train_loss, val_loss, optimizer, batch_size, epoch_time)
            
            # Logging
            if epoch % self.config.log_interval == 0:
                self._log_progress(epoch, train_loss, val_loss, optimizer)
            
            # Checkpointing
            if epoch % self.config.save_every == 0 or val_loss < self.best_metric:
                self._save_checkpoint(model, optimizer, epoch, val_loss)
            
            # Early stopping
            if self._check_early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Adaptive batch size adjustment
            if self.config.adaptive_batch_size and epoch % 50 == 0 and epoch > 0:
                new_batch_size = self._adjust_batch_size(batch_size, train_loss)
                if new_batch_size != batch_size:
                    batch_size = new_batch_size
                    train_loader = self._create_dataloader(train_data, batch_size, shuffle=True)
                    val_loader = self._create_dataloader(val_data, batch_size, shuffle=False)
                    logger.info(f"Adjusted batch size to: {batch_size}")
        
        # Final cleanup
        self._finalize_training()
        
        return model
    
    def _optimize_batch_size(self, model: nn.Module, dataset: Dataset) -> int:
        """Optimize batch size based on GPU memory."""
        # Get sample input shape
        sample_x, sample_y = dataset[0]
        input_shape = sample_x.shape
        
        # Use GPU manager to find optimal batch size
        optimal_batch_size = self.gpu_manager.optimize_batch_size(
            model, input_shape,
            initial_batch_size=self.config.initial_batch_size,
            max_batch_size=self.config.max_batch_size
        )
        
        # Apply gradient accumulation adjustment
        if self.config.gradient_accumulation_steps > 1:
            optimal_batch_size = optimal_batch_size // self.config.gradient_accumulation_steps
        
        return max(self.config.min_batch_size, optimal_batch_size)
    
    def _create_dataloader(self, dataset: Dataset, batch_size: int, 
                          shuffle: bool = True) -> DataLoader:
        """Create optimized DataLoader."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=self.gpu_manager.is_gpu,
            prefetch_factor=2,
            persistent_workers=True
        )
    
    def _create_scheduler(self, optimizer: torch.optim.Optimizer, 
                         steps_per_epoch: int) -> Optional[Any]:
        """Create learning rate scheduler."""
        total_steps = steps_per_epoch * self.config.epochs
        warmup_steps = steps_per_epoch * self.config.warmup_epochs
        
        if self.config.scheduler_type == 'cosine':
            # Cosine annealing with warm restarts
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=50, T_mult=2, eta_min=self.config.min_lr
            )
        elif self.config.scheduler_type == 'plateau':
            # Reduce on plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10,
                min_lr=self.config.min_lr, verbose=True
            )
        elif self.config.scheduler_type == 'cyclic':
            # Cyclic learning rate
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=self.config.min_lr,
                max_lr=self.config.learning_rate, step_size_up=20,
                mode='triangular2'
            )
        elif self.config.scheduler_type == 'onecycle':
            # One cycle policy
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.config.learning_rate,
                total_steps=total_steps, pct_start=0.3,
                anneal_strategy='cos', div_factor=25
            )
        else:
            scheduler = None
        
        # Add warmup wrapper
        if warmup_steps > 0 and scheduler:
            from torch.optim.lr_scheduler import LambdaLR
            
            def warmup_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                return 1.0
            
            warmup_scheduler = LambdaLR(optimizer, warmup_lambda)
            
            class CombinedScheduler:
                def __init__(self, warmup, main):
                    self.warmup = warmup
                    self.main = main
                    self.step_count = 0
                
                def step(self, metric=None):
                    self.step_count += 1
                    if self.step_count <= warmup_steps:
                        self.warmup.step()
                    else:
                        if metric is not None and hasattr(self.main, 'step'):
                            self.main.step(metric)
                        elif hasattr(self.main, 'step'):
                            self.main.step()
            
            scheduler = CombinedScheduler(warmup_scheduler, scheduler)
        
        return scheduler
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader,
                    criterion: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: Optional[Any]) -> float:
        """Train one epoch with GPU optimization."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}', 
                   disable=not torch.cuda.is_available())
        
        optimizer.zero_grad()
        
        for batch_idx, (features, targets) in enumerate(pbar):
            # Move data to GPU
            features = self.gpu_manager.to_device(features)
            targets = self.gpu_manager.to_device(targets)
            
            # Mixed precision forward pass
            with self.gpu_manager.autocast(self.config.use_mixed_precision):
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            if self.config.use_mixed_precision:
                loss = self.gpu_manager.scale_loss(loss)
                loss.backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    if self.config.use_mixed_precision and self.gpu_manager.scaler:
                        self.gpu_manager.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.gradient_clipping
                    )
                
                # Optimizer step
                self.gpu_manager.optimizer_step(optimizer)
                optimizer.zero_grad()
                
                # Update scheduler
                if scheduler and self.config.scheduler_type in ['cyclic', 'onecycle']:
                    scheduler.step()
            
            # Track loss
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'gpu_mem': f'{self.gpu_manager.get_memory_used():.1f}GB'
            })
            
            self.global_step += 1
            
            # TensorBoard logging
            if self.writer and self.global_step % 100 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], 
                                     self.global_step)
        
        # Clear memory
        self.gpu_manager.clear_cache()
        
        return total_loss / num_batches
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader,
                       criterion: nn.Module) -> float:
        """Validate one epoch."""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                # Move data to GPU
                features = self.gpu_manager.to_device(features)
                targets = self.gpu_manager.to_device(targets)
                
                # Forward pass
                with self.gpu_manager.autocast(self.config.use_mixed_precision):
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        # Clear memory
        self.gpu_manager.clear_cache()
        
        return total_loss / num_batches
    
    def _adjust_batch_size(self, current_batch_size: int, loss: float) -> int:
        """Adaptively adjust batch size based on training progress."""
        memory_info = self.gpu_manager.get_memory_info()
        memory_usage = memory_info['percent']
        
        # If memory usage is low and loss is stable, increase batch size
        if memory_usage < 70 and loss < self.best_metric * 1.1:
            new_batch_size = min(
                int(current_batch_size * 1.5),
                self.config.max_batch_size
            )
        # If memory usage is high, decrease batch size
        elif memory_usage > 90:
            new_batch_size = max(
                int(current_batch_size * 0.75),
                self.config.min_batch_size
            )
        else:
            new_batch_size = current_batch_size
        
        # Round to nearest power of 2
        new_batch_size = 2 ** int(np.log2(new_batch_size))
        
        return new_batch_size
    
    def _track_metrics(self, train_loss: float, val_loss: float,
                      optimizer: torch.optim.Optimizer, batch_size: int,
                      epoch_time: float):
        """Track training metrics."""
        self.training_history['loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['lr'].append(optimizer.param_groups[0]['lr'])
        self.training_history['batch_size'].append(batch_size)
        self.training_history['gpu_memory'].append(self.gpu_manager.get_memory_used())
        self.training_history['epoch_time'].append(epoch_time)
    
    def _log_progress(self, epoch: int, train_loss: float, val_loss: float,
                     optimizer: torch.optim.Optimizer):
        """Log training progress."""
        lr = optimizer.param_groups[0]['lr']
        memory = self.gpu_manager.get_memory_info()
        
        logger.info(
            f"Epoch {epoch}/{self.config.epochs} - "
            f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
            f"LR: {lr:.2e} - GPU Memory: {memory['used']:.1f}/{memory['total']:.1f}GB"
        )
        
        if self.writer:
            self.writer.add_scalars('loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
    
    def _save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                        epoch: int, val_loss: float):
        """Save model checkpoint."""
        is_best = val_loss < self.best_metric
        if is_best:
            self.best_metric = val_loss
        
        if not self.config.keep_best_only or is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_metric': self.best_metric,
                'config': self.config.__dict__,
                'training_history': self.training_history
            }
            
            # Save checkpoint
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            
            if is_best:
                best_path = self.checkpoint_dir / 'best_model.pth'
                torch.save(checkpoint, best_path)
                logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check early stopping criteria."""
        if not self.config.early_stopping:
            return False
        
        if val_loss < self.best_metric - self.config.min_delta:
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.patience
    
    def _finalize_training(self):
        """Finalize training and save results."""
        # Save training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Plot training curves
        self._plot_training_curves()
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
        
        # Final GPU cleanup
        self.gpu_manager.clear_cache()
        
        logger.info("Training completed successfully")
    
    def _plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        
        # Learning rate
        axes[0, 1].plot(self.training_history['lr'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_yscale('log')
        
        # Batch size
        axes[1, 0].plot(self.training_history['batch_size'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Batch Size')
        axes[1, 0].set_title('Adaptive Batch Size')
        
        # GPU memory
        axes[1, 1].plot(self.training_history['gpu_memory'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('GPU Memory (GB)')
        axes[1, 1].set_title('GPU Memory Usage')
        
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'training_curves.png', dpi=300)
        plt.close()
    
    def load_checkpoint(self, model: nn.Module, checkpoint_path: str) -> nn.Module:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.gpu_manager.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        return model