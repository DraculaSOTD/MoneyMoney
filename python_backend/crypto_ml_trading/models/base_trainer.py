"""
Base Trainer Class for All Models

Provides common training functionality including:
- Optimizers (Adam, SGD, RMSprop, AdamW)
- Learning rate scheduling
- Gradient clipping
- Early stopping
- Model checkpointing
- Validation and metrics tracking
- Data loaders with batching
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union
from abc import ABC, abstractmethod
import time
import json
from pathlib import Path
from collections import defaultdict, deque


class BaseOptimizer:
    """Base optimizer class with common functionality."""
    
    def __init__(self, learning_rate: float = 0.001,
                 weight_decay: float = 0.0,
                 gradient_clip: float = 0.0):
        """
        Initialize optimizer.
        
        Args:
            learning_rate: Initial learning rate
            weight_decay: L2 regularization factor
            gradient_clip: Gradient clipping threshold (0 = no clipping)
        """
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.state = {}
        self.iteration = 0
        
    def clip_gradients(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Clip gradients by global norm."""
        if self.gradient_clip <= 0:
            return gradients
            
        # Calculate global norm
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)
        
        # Clip if necessary
        if total_norm > self.gradient_clip:
            clip_factor = self.gradient_clip / total_norm
            clipped_gradients = {}
            for name, grad in gradients.items():
                clipped_gradients[name] = grad * clip_factor
            return clipped_gradients
            
        return gradients
    
    def apply_weight_decay(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Apply L2 weight decay to gradient."""
        if self.weight_decay > 0:
            return grad + self.weight_decay * param
        return grad
    
    @abstractmethod
    def update(self, params: Dict[str, np.ndarray], 
               gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Update parameters with gradients."""
        pass


class AdamOptimizer(BaseOptimizer):
    """Adam optimizer with bias correction."""
    
    def __init__(self, learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8, **kwargs):
        super().__init__(learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
    def update(self, params: Dict[str, np.ndarray], 
               gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Update parameters using Adam."""
        self.iteration += 1
        
        # Clip gradients
        gradients = self.clip_gradients(gradients)
        
        updated_params = {}
        
        for name, param in params.items():
            if name not in gradients:
                updated_params[name] = param
                continue
                
            grad = gradients[name]
            
            # Apply weight decay
            grad = self.apply_weight_decay(param, grad)
            
            # Initialize state
            if name not in self.state:
                self.state[name] = {
                    'm': np.zeros_like(param),
                    'v': np.zeros_like(param)
                }
                
            # Update biased moments
            self.state[name]['m'] = self.beta1 * self.state[name]['m'] + (1 - self.beta1) * grad
            self.state[name]['v'] = self.beta2 * self.state[name]['v'] + (1 - self.beta2) * grad**2
            
            # Bias correction
            m_hat = self.state[name]['m'] / (1 - self.beta1**self.iteration)
            v_hat = self.state[name]['v'] / (1 - self.beta2**self.iteration)
            
            # Update parameters
            updated_params[name] = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
        return updated_params


class AdamWOptimizer(AdamOptimizer):
    """AdamW optimizer with decoupled weight decay."""
    
    def update(self, params: Dict[str, np.ndarray], 
               gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Update parameters using AdamW."""
        self.iteration += 1
        
        # Clip gradients
        gradients = self.clip_gradients(gradients)
        
        updated_params = {}
        
        for name, param in params.items():
            if name not in gradients:
                updated_params[name] = param
                continue
                
            grad = gradients[name]
            
            # Initialize state
            if name not in self.state:
                self.state[name] = {
                    'm': np.zeros_like(param),
                    'v': np.zeros_like(param)
                }
                
            # Update biased moments (without weight decay)
            self.state[name]['m'] = self.beta1 * self.state[name]['m'] + (1 - self.beta1) * grad
            self.state[name]['v'] = self.beta2 * self.state[name]['v'] + (1 - self.beta2) * grad**2
            
            # Bias correction
            m_hat = self.state[name]['m'] / (1 - self.beta1**self.iteration)
            v_hat = self.state[name]['v'] / (1 - self.beta2**self.iteration)
            
            # Update with decoupled weight decay
            updated_params[name] = param - self.learning_rate * (
                m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * param
            )
            
        return updated_params


class SGDOptimizer(BaseOptimizer):
    """SGD optimizer with momentum."""
    
    def __init__(self, learning_rate: float = 0.01,
                 momentum: float = 0.9, nesterov: bool = True, **kwargs):
        super().__init__(learning_rate, **kwargs)
        self.momentum = momentum
        self.nesterov = nesterov
        
    def update(self, params: Dict[str, np.ndarray], 
               gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Update parameters using SGD with momentum."""
        # Clip gradients
        gradients = self.clip_gradients(gradients)
        
        updated_params = {}
        
        for name, param in params.items():
            if name not in gradients:
                updated_params[name] = param
                continue
                
            grad = gradients[name]
            
            # Apply weight decay
            grad = self.apply_weight_decay(param, grad)
            
            # Initialize velocity
            if name not in self.state:
                self.state[name] = {'v': np.zeros_like(param)}
                
            # Update velocity
            self.state[name]['v'] = self.momentum * self.state[name]['v'] - self.learning_rate * grad
            
            # Update parameters
            if self.nesterov:
                updated_params[name] = param + self.momentum * self.state[name]['v'] - self.learning_rate * grad
            else:
                updated_params[name] = param + self.state[name]['v']
                
        return updated_params


class RMSpropOptimizer(BaseOptimizer):
    """RMSprop optimizer."""
    
    def __init__(self, learning_rate: float = 0.001,
                 alpha: float = 0.95, epsilon: float = 1e-8, **kwargs):
        super().__init__(learning_rate, **kwargs)
        self.alpha = alpha
        self.epsilon = epsilon
        
    def update(self, params: Dict[str, np.ndarray], 
               gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Update parameters using RMSprop."""
        # Clip gradients
        gradients = self.clip_gradients(gradients)
        
        updated_params = {}
        
        for name, param in params.items():
            if name not in gradients:
                updated_params[name] = param
                continue
                
            grad = gradients[name]
            
            # Apply weight decay
            grad = self.apply_weight_decay(param, grad)
            
            # Initialize cache
            if name not in self.state:
                self.state[name] = {'cache': np.zeros_like(param)}
                
            # Update cache
            self.state[name]['cache'] = (
                self.alpha * self.state[name]['cache'] + 
                (1 - self.alpha) * grad**2
            )
            
            # Update parameters
            updated_params[name] = param - self.learning_rate * grad / (
                np.sqrt(self.state[name]['cache']) + self.epsilon
            )
            
        return updated_params


class LearningRateScheduler:
    """Learning rate scheduler base class."""
    
    def __init__(self, initial_lr: float):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        
    def step(self, epoch: int, metrics: Optional[Dict] = None) -> float:
        """Update learning rate."""
        return self.current_lr


class CosineAnnealingScheduler(LearningRateScheduler):
    """Cosine annealing learning rate scheduler."""
    
    def __init__(self, initial_lr: float, total_epochs: int,
                 min_lr: float = 0.0, warmup_epochs: int = 0):
        super().__init__(initial_lr)
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        
    def step(self, epoch: int, metrics: Optional[Dict] = None) -> float:
        """Update learning rate with cosine annealing."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            self.current_lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * progress)
            )
            
        return self.current_lr


class StepLRScheduler(LearningRateScheduler):
    """Step learning rate scheduler."""
    
    def __init__(self, initial_lr: float, step_size: int, gamma: float = 0.1):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma
        
    def step(self, epoch: int, metrics: Optional[Dict] = None) -> float:
        """Update learning rate with step decay."""
        steps = epoch // self.step_size
        self.current_lr = self.initial_lr * (self.gamma ** steps)
        return self.current_lr


class ReduceLROnPlateauScheduler(LearningRateScheduler):
    """Reduce learning rate on plateau scheduler."""
    
    def __init__(self, initial_lr: float, factor: float = 0.1,
                 patience: int = 10, min_lr: float = 1e-6,
                 metric: str = 'val_loss', mode: str = 'min'):
        super().__init__(initial_lr)
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.metric = metric
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else -float('inf')
        self.counter = 0
        
    def step(self, epoch: int, metrics: Optional[Dict] = None) -> float:
        """Update learning rate based on plateau detection."""
        if metrics is None or self.metric not in metrics:
            return self.current_lr
            
        current_value = metrics[self.metric]
        
        if self.mode == 'min':
            improved = current_value < self.best_value
        else:
            improved = current_value > self.best_value
            
        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.counter = 0
            
        return self.current_lr


class BaseTrainer(ABC):
    """Base trainer class with common training functionality."""
    
    def __init__(self,
                 model,
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.0,
                 gradient_clip: float = 1.0,
                 batch_size: int = 32,
                 validation_split: float = 0.2,
                 early_stopping_patience: int = 10,
                 checkpoint_dir: Optional[str] = None):
        """
        Initialize base trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer name ('adam', 'adamw', 'sgd', 'rmsprop')
            learning_rate: Initial learning rate
            weight_decay: L2 regularization factor
            gradient_clip: Gradient clipping threshold
            batch_size: Training batch size
            validation_split: Validation data fraction
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('./checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer(
            optimizer, learning_rate, weight_decay, gradient_clip
        )
        
        # Training state
        self.epoch = 0
        self.best_val_metric = float('inf')
        self.patience_counter = 0
        self.training_history = defaultdict(list)
        
        # Metrics tracking
        self.train_metrics = defaultdict(lambda: deque(maxlen=100))
        self.val_metrics = defaultdict(lambda: deque(maxlen=100))
        
    def _create_optimizer(self, name: str, lr: float, 
                         weight_decay: float, gradient_clip: float) -> BaseOptimizer:
        """Create optimizer instance."""
        optimizers = {
            'adam': AdamOptimizer,
            'adamw': AdamWOptimizer,
            'sgd': SGDOptimizer,
            'rmsprop': RMSpropOptimizer
        }
        
        if name not in optimizers:
            raise ValueError(f"Unknown optimizer: {name}")
            
        return optimizers[name](
            learning_rate=lr,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip
        )
    
    @abstractmethod
    def prepare_data(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Prepare data for training."""
        pass
    
    @abstractmethod
    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Single training step."""
        pass
    
    @abstractmethod
    def validation_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Single validation step."""
        pass
    
    def create_data_loader(self, data: Dict[str, np.ndarray], 
                          shuffle: bool = True) -> List[Dict[str, np.ndarray]]:
        """Create batched data loader."""
        n_samples = len(next(iter(data.values())))
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
            
        batches = []
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = {}
            for key, values in data.items():
                batch[key] = values[batch_indices]
            batches.append(batch)
            
        return batches
    
    def train_epoch(self, train_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Train for one epoch."""
        # Create batches
        train_loader = self.create_data_loader(train_data, shuffle=True)
        
        epoch_metrics = defaultdict(float)
        n_batches = len(train_loader)
        
        for batch in train_loader:
            # Training step
            batch_metrics = self.train_step(batch)
            
            # Accumulate metrics
            for key, value in batch_metrics.items():
                epoch_metrics[key] += value
                self.train_metrics[key].append(value)
                
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
            
        return dict(epoch_metrics)
    
    def validate(self, val_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Validate model."""
        # Create batches
        val_loader = self.create_data_loader(val_data, shuffle=False)
        
        epoch_metrics = defaultdict(float)
        n_batches = len(val_loader)
        
        for batch in val_loader:
            # Validation step
            batch_metrics = self.validation_step(batch)
            
            # Accumulate metrics
            for key, value in batch_metrics.items():
                epoch_metrics[key] += value
                self.val_metrics[key].append(value)
                
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
            
        return dict(epoch_metrics)
    
    def train(self, 
              data: Union[np.ndarray, pd.DataFrame],
              epochs: int = 100,
              lr_scheduler: Optional[LearningRateScheduler] = None,
              callbacks: Optional[List[Callable]] = None,
              verbose: int = 1) -> Dict:
        """
        Train the model.
        
        Args:
            data: Training data
            epochs: Number of epochs
            lr_scheduler: Learning rate scheduler
            callbacks: List of callback functions
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        # Prepare data
        prepared_data = self.prepare_data(data)
        
        # Split into train/validation
        train_data, val_data = self._split_data(prepared_data)
        
        if verbose > 0:
            print(f"Training on {len(next(iter(train_data.values())))} samples")
            print(f"Validating on {len(next(iter(val_data.values())))} samples")
            print("-" * 60)
            
        # Training loop
        for epoch in range(epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Update learning rate
            if lr_scheduler:
                current_lr = lr_scheduler.step(epoch, self.training_history)
                self.optimizer.learning_rate = current_lr
                
            # Train
            train_metrics = self.train_epoch(train_data)
            
            # Validate
            val_metrics = self.validate(val_data)
            
            # Update history
            for key, value in train_metrics.items():
                self.training_history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.training_history[f'val_{key}'].append(value)
            self.training_history['learning_rate'].append(self.optimizer.learning_rate)
            
            # Check for improvement
            val_loss = val_metrics.get('loss', float('inf'))
            if val_loss < self.best_val_metric:
                self.best_val_metric = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pkl')
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                if verbose > 0:
                    print(f"\nEarly stopping at epoch {epoch}")
                break
                
            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, epoch, train_metrics, val_metrics)
                    
            # Print progress
            if verbose > 0 and epoch % max(1, epochs // 20) == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:3d}/{epochs} - {elapsed:.1f}s - "
                      f"loss: {train_metrics.get('loss', 0):.4f} - "
                      f"val_loss: {val_metrics.get('loss', 0):.4f}")
                      
        # Load best model
        self.load_checkpoint('best_model.pkl')
        
        if verbose > 0:
            print("-" * 60)
            print(f"Training completed. Best validation loss: {self.best_val_metric:.4f}")
            
        return dict(self.training_history)
    
    def _split_data(self, data: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:
        """Split data into train and validation sets."""
        n_samples = len(next(iter(data.values())))
        n_val = int(n_samples * self.validation_split)
        
        train_data = {}
        val_data = {}
        
        for key, values in data.items():
            train_data[key] = values[:-n_val]
            val_data[key] = values[-n_val:]
            
        return train_data, val_data
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_params': self.model.get_params() if hasattr(self.model, 'get_params') else None,
            'optimizer_state': self.optimizer.state,
            'best_val_metric': self.best_val_metric,
            'training_history': dict(self.training_history)
        }
        
        filepath = self.checkpoint_dir / filename
        np.savez_compressed(filepath, **checkpoint)
        
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        filepath = self.checkpoint_dir / filename
        if not filepath.exists():
            return
            
        checkpoint = np.load(filepath, allow_pickle=True)
        
        self.epoch = checkpoint['epoch'].item()
        if checkpoint['model_params'].item() and hasattr(self.model, 'set_params'):
            self.model.set_params(checkpoint['model_params'].item())
        self.optimizer.state = checkpoint['optimizer_state'].item()
        self.best_val_metric = checkpoint['best_val_metric'].item()
        self.training_history = defaultdict(list, checkpoint['training_history'].item())
        
    def plot_training_history(self, metrics: Optional[List[str]] = None):
        """Plot training history."""
        import matplotlib.pyplot as plt
        
        if metrics is None:
            metrics = ['loss']
            
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
        
        if n_metrics == 1:
            axes = [axes]
            
        for ax, metric in zip(axes, metrics):
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'
            
            if train_key in self.training_history:
                ax.plot(self.training_history[train_key], label='Train')
            if val_key in self.training_history:
                ax.plot(self.training_history[val_key], label='Validation')
                
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} History')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()