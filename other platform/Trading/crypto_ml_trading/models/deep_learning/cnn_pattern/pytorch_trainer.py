"""
PyTorch Trainer for CNN Pattern GPU Model.

Integrates with:
- admin_training.py training pipeline
- WebSocket progress broadcasting
- GPU memory management
- Mixed precision training
- Class-balanced loss
- Learning rate scheduling
- Early stopping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
from typing import Dict, Optional, Callable, Tuple, List, Any
import logging
from pathlib import Path
import time

import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils.gpu_manager import get_gpu_manager
from utils.logging_system import get_logger

logger = get_logger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Reduces loss for well-classified examples, focusing on hard negatives.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0,
                 class_weights: Optional[torch.Tensor] = None):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor
            gamma: Focusing parameter (higher = more focus on hard examples)
            class_weights: Optional per-class weights
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Model logits (batch, num_classes)
            targets: Ground truth labels (batch,)

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CNNPatternPyTorchTrainer:
    """
    PyTorch trainer for CNN Pattern GPU model.

    Features:
    - AdamW optimizer with weight decay
    - OneCycleLR or ReduceLROnPlateau scheduling
    - Mixed precision training (AMP)
    - Focal loss for class imbalance
    - Early stopping with best model restoration
    - Per-class metrics tracking
    - WebSocket progress callback support
    """

    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 weight_decay: float = 0.01,
                 use_mixed_precision: bool = True,
                 use_focal_loss: bool = True,
                 focal_gamma: float = 2.0,
                 label_smoothing: float = 0.1):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            learning_rate: Initial learning rate
            batch_size: Training batch size
            weight_decay: AdamW weight decay
            use_mixed_precision: Enable automatic mixed precision
            use_focal_loss: Use focal loss instead of cross entropy
            focal_gamma: Focal loss gamma parameter
            label_smoothing: Label smoothing factor
        """
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_mixed_precision = use_mixed_precision
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

        # GPU manager
        self.gpu_manager = get_gpu_manager()
        self.device = self.gpu_manager.device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler (will be set during training)
        self.scheduler = None

        # Loss function (will be set with class weights during training)
        self.criterion = None

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and self.gpu_manager.is_gpu else None

        # Training state
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0

        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"  Mixed precision: {use_mixed_precision}")
        logger.info(f"  Focal loss: {use_focal_loss}")

    def _compute_class_weights(self, labels: np.ndarray) -> torch.Tensor:
        """
        Compute inverse frequency class weights for balanced training.

        Args:
            labels: Training labels

        Returns:
            Class weights tensor
        """
        unique, counts = np.unique(labels, return_counts=True)
        n_samples = len(labels)
        n_classes = len(unique)

        # Inverse frequency weighting
        weights = n_samples / (n_classes * counts)

        # Normalize
        weights = weights / weights.sum() * n_classes

        # Create full weight tensor (handle missing classes)
        full_weights = torch.ones(self.model.num_classes, device=self.device)
        for i, class_idx in enumerate(unique):
            full_weights[class_idx] = weights[i]

        logger.info(f"Class weights computed: {dict(zip(unique, weights.round(2)))}")

        return full_weights

    def _create_loader(self, X: np.ndarray, y: np.ndarray,
                      shuffle: bool = True,
                      use_weighted_sampling: bool = False) -> DataLoader:
        """
        Create DataLoader from numpy arrays.

        Args:
            X: Feature array
            y: Label array
            shuffle: Whether to shuffle data
            use_weighted_sampling: Use weighted random sampling for balance

        Returns:
            DataLoader
        """
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).long()

        dataset = TensorDataset(X_tensor, y_tensor)

        sampler = None
        if use_weighted_sampling and shuffle:
            # Compute sample weights - need to handle sparse class indices
            unique, counts = np.unique(y, return_counts=True)

            # Create a weight lookup that maps class index -> weight
            # Use max label value + 1 to ensure all indices are valid
            max_class = max(int(np.max(y)) + 1, self.model.num_classes)
            class_weight_lookup = np.ones(max_class)

            # Set inverse frequency weights for classes that exist
            for cls_idx, count in zip(unique, counts):
                class_weight_lookup[cls_idx] = 1.0 / count

            # Now index safely - all label values will be valid indices
            sample_weights = class_weight_lookup[y]

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(y),
                replacement=True
            )
            shuffle = False  # Can't use both sampler and shuffle

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=0,  # Keep 0 for GPU training
            pin_memory=True if self.gpu_manager.is_gpu else False,
            drop_last=True if shuffle else False
        )

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 50,
              early_stopping_patience: int = 10,
              progress_callback: Optional[Callable] = None,
              use_weighted_sampling: bool = True) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training images (N, C, H, W)
            y_train: Training labels (N,)
            X_val: Validation images (optional)
            y_val: Validation labels (optional)
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            progress_callback: Async callback for progress updates
            use_weighted_sampling: Use weighted sampling for class balance

        Returns:
            Training history and final metrics
        """
        start_time = time.time()

        # Compute class weights
        class_weights = self._compute_class_weights(y_train)

        # Set loss function
        if self.use_focal_loss:
            self.criterion = FocalLoss(
                gamma=self.focal_gamma,
                class_weights=class_weights
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=self.label_smoothing
            )

        # Create data loaders
        train_loader = self._create_loader(X_train, y_train, shuffle=True,
                                          use_weighted_sampling=use_weighted_sampling)
        val_loader = self._create_loader(X_val, y_val, shuffle=False) if X_val is not None else None

        # Set up scheduler (OneCycleLR for better convergence)
        steps_per_epoch = len(train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate * 10,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy='cos'
        )

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0

        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Validation samples: {len(X_val) if X_val is not None else 0}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train epoch
            train_loss, train_acc = self._train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # Validate
            val_loss, val_acc = 0.0, 0.0
            if val_loader:
                val_loss, val_acc = self._validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if self.patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            epoch_time = time.time() - epoch_start

            # Log progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%} - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%} - "
                    f"Time: {epoch_time:.1f}s"
                )

            # Progress callback
            if progress_callback:
                progress = int((epoch + 1) / epochs * 100)
                try:
                    import asyncio
                    if asyncio.iscoroutinefunction(progress_callback):
                        # Schedule the coroutine
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(progress_callback(progress, epoch + 1, train_acc, val_acc))
                    else:
                        progress_callback(progress, epoch + 1, train_acc, val_acc)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

            # GPU memory cleanup periodically
            if (epoch + 1) % 10 == 0:
                self.gpu_manager.clear_cache()

        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)
            logger.info("Loaded best model from training")

        # Final metrics
        total_time = time.time() - start_time

        # Get per-class metrics
        class_metrics = {}
        if val_loader:
            class_metrics = self._compute_class_metrics(val_loader)

        results = {
            'history': history,
            'train_loss': history['train_loss'][-1],
            'train_accuracy': history['train_acc'][-1],
            'val_loss': history['val_loss'][-1] if history['val_loss'] else 0.0,
            'val_accuracy': history['val_acc'][-1] if history['val_acc'] else 0.0,
            'epochs_trained': len(history['train_loss']),
            'total_time': total_time,
            'best_val_loss': self.best_val_loss,
            'class_metrics': class_metrics
        }

        logger.info(f"Training complete in {total_time:.1f}s")
        logger.info(f"  Final train accuracy: {results['train_accuracy']:.2%}")
        logger.info(f"  Final val accuracy: {results['val_accuracy']:.2%}")

        return results

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            loader: Training data loader

        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    output = self.model(batch_x)
                    loss = self.criterion(output['logits'], batch_y)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(batch_x)
                loss = self.criterion(output['logits'], batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Update scheduler
            self.scheduler.step()

            total_loss += loss.item()
            correct += (output['predictions'] == batch_y).sum().item()
            total += batch_y.size(0)

        return total_loss / len(loader), correct / total

    def _validate(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.

        Args:
            loader: Validation data loader

        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = self.model(batch_x)
                        loss = self.criterion(output['logits'], batch_y)
                else:
                    output = self.model(batch_x)
                    loss = self.criterion(output['logits'], batch_y)

                total_loss += loss.item()
                correct += (output['predictions'] == batch_y).sum().item()
                total += batch_y.size(0)

        return total_loss / len(loader), correct / total

    def _compute_class_metrics(self, loader: DataLoader) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class precision, recall, and F1.

        Args:
            loader: Data loader

        Returns:
            Dictionary of per-class metrics
        """
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                output = self.model(batch_x)
                all_preds.extend(self.gpu_manager.to_numpy(output['predictions']))
                all_labels.extend(batch_y.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = {}
        unique_classes = np.unique(np.concatenate([all_preds, all_labels]))

        for class_idx in unique_classes:
            class_name = self.model.PATTERN_NAMES[class_idx] if class_idx < len(self.model.PATTERN_NAMES) else f"class_{class_idx}"

            tp = np.sum((all_preds == class_idx) & (all_labels == class_idx))
            fp = np.sum((all_preds == class_idx) & (all_labels != class_idx))
            fn = np.sum((all_preds != class_idx) & (all_labels == class_idx))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': int(np.sum(all_labels == class_idx))
            }

        return metrics

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.

        Args:
            X: Feature array
            y: Label array

        Returns:
            Evaluation metrics
        """
        loader = self._create_loader(X, y, shuffle=False)
        loss, accuracy = self._validate(loader)
        class_metrics = self._compute_class_metrics(loader)

        return {
            'loss': loss,
            'accuracy': accuracy,
            'class_metrics': class_metrics
        }
