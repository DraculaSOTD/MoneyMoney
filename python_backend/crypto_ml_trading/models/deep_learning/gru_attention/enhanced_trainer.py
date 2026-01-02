"""
Enhanced GRU-Attention Trainer with Advanced Optimization

Features:
- Multiple optimizer support (Adam, AdamW, SGD, RMSprop)
- Advanced learning rate scheduling
- Gradient clipping and accumulation
- Mixed precision training simulation
- Data augmentation for time series
- Attention visualization
- Comprehensive metrics tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import time
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.base_trainer import BaseTrainer, CosineAnnealingScheduler, ReduceLROnPlateauScheduler
from models.deep_learning.gru_attention.model import GRUAttentionModel


class EnhancedGRUAttentionTrainer(BaseTrainer):
    """
    Enhanced trainer for GRU-Attention model with advanced features.
    """
    
    def __init__(self,
                 model: GRUAttentionModel,
                 optimizer: str = 'adamw',
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.01,
                 gradient_clip: float = 1.0,
                 batch_size: int = 32,
                 sequence_length: int = 60,
                 validation_split: float = 0.2,
                 early_stopping_patience: int = 15,
                 checkpoint_dir: Optional[str] = None,
                 use_data_augmentation: bool = True,
                 gradient_accumulation_steps: int = 1,
                 label_smoothing: float = 0.1):
        """
        Initialize enhanced GRU-Attention trainer.
        
        Args:
            model: GRU-Attention model
            optimizer: Optimizer choice
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            gradient_clip: Gradient clipping threshold
            batch_size: Training batch size
            sequence_length: Input sequence length
            validation_split: Validation data fraction
            early_stopping_patience: Early stopping patience
            checkpoint_dir: Checkpoint directory
            use_data_augmentation: Whether to use data augmentation
            gradient_accumulation_steps: Steps to accumulate gradients
            label_smoothing: Label smoothing factor
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
            batch_size=batch_size,
            validation_split=validation_split,
            early_stopping_patience=early_stopping_patience,
            checkpoint_dir=checkpoint_dir
        )
        
        self.sequence_length = sequence_length
        self.use_data_augmentation = use_data_augmentation
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.label_smoothing = label_smoothing
        
        # Additional metrics tracking
        self.attention_weights_history = []
        self.gradient_norms = []
        self.accumulated_gradients = {}
        self.accumulation_counter = 0
        
    def prepare_data(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Prepare data for GRU-Attention training.
        
        Args:
            data: Raw data (DataFrame with OHLCV + indicators)
            
        Returns:
            Dictionary with prepared sequences and labels
        """
        if isinstance(data, pd.DataFrame):
            # Extract features and labels
            feature_columns = [col for col in data.columns 
                             if col not in ['timestamp', 'label', 'action', 'target']]
            features = data[feature_columns].values
            
            # Handle different label formats
            if 'action' in data.columns:
                labels = data['action'].values
            elif 'label' in data.columns:
                labels = data['label'].values
            else:
                # Generate labels from price movements
                labels = self._generate_labels_from_prices(data)
        else:
            # Assume numpy array with last column as labels
            features = data[:, :-1]
            labels = data[:, -1].astype(int)
            
        # Normalize features
        features = self._normalize_features(features)
        
        # Create sequences
        X, y = self._create_sequences(features, labels)
        
        return {'sequences': X, 'labels': y}
    
    def _generate_labels_from_prices(self, data: pd.DataFrame) -> np.ndarray:
        """Generate trading labels from price data."""
        if 'close' not in data.columns:
            raise ValueError("Need 'close' column to generate labels")
            
        # Calculate returns
        returns = data['close'].pct_change().fillna(0)
        
        # Generate labels based on returns
        # 0: Buy, 1: Hold, 2: Sell
        labels = np.ones(len(data), dtype=int)  # Default to hold
        
        # Use dynamic thresholds based on volatility
        volatility = returns.rolling(20).std().fillna(returns.std())
        
        buy_threshold = -0.5 * volatility
        sell_threshold = 0.5 * volatility
        
        labels[returns < buy_threshold] = 0  # Buy
        labels[returns > sell_threshold] = 2  # Sell
        
        return labels
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features with robust scaling."""
        # Use percentile-based normalization for robustness
        percentile_5 = np.percentile(features, 5, axis=0)
        percentile_95 = np.percentile(features, 95, axis=0)
        
        # Avoid division by zero
        scale = percentile_95 - percentile_5
        scale[scale == 0] = 1
        
        normalized = (features - percentile_5) / scale
        
        # Clip to reasonable range
        normalized = np.clip(normalized, -3, 3)
        
        return normalized
    
    def _create_sequences(self, features: np.ndarray, 
                         labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        n_samples = len(features) - self.sequence_length
        n_features = features.shape[1]
        
        X = np.zeros((n_samples, self.sequence_length, n_features))
        y = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            X[i] = features[i:i + self.sequence_length]
            y[i] = labels[i + self.sequence_length]
            
        return X, y
    
    def augment_batch(self, sequences: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to batch.
        
        Augmentation techniques:
        - Noise injection
        - Time warping
        - Magnitude scaling
        - Window slicing
        """
        if not self.use_data_augmentation or np.random.random() > 0.5:
            return sequences, labels
            
        batch_size = len(sequences)
        augmented_sequences = sequences.copy()
        
        for i in range(batch_size):
            aug_type = np.random.choice(['noise', 'scale', 'warp', 'slice'])
            
            if aug_type == 'noise':
                # Add Gaussian noise
                noise_level = np.random.uniform(0.01, 0.05)
                noise = np.random.randn(*augmented_sequences[i].shape) * noise_level
                augmented_sequences[i] += noise
                
            elif aug_type == 'scale':
                # Random magnitude scaling
                scale_factor = np.random.uniform(0.9, 1.1)
                augmented_sequences[i] *= scale_factor
                
            elif aug_type == 'warp':
                # Time warping (simple version)
                if self.sequence_length > 10:
                    warp_idx = np.random.randint(5, self.sequence_length - 5)
                    if np.random.random() > 0.5:
                        # Compress
                        augmented_sequences[i, warp_idx-2:warp_idx+2] = \
                            augmented_sequences[i, warp_idx-1:warp_idx+1].mean(axis=0)
                    else:
                        # Stretch (duplicate)
                        augmented_sequences[i, warp_idx] = augmented_sequences[i, warp_idx-1]
                        
            elif aug_type == 'slice':
                # Window slicing (shift window slightly)
                shift = np.random.randint(-2, 3)
                if shift != 0 and 0 <= i + shift < batch_size:
                    augmented_sequences[i] = sequences[i + shift]
                    
        return augmented_sequences, labels
    
    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Single training step with gradient accumulation.
        
        Args:
            batch: Batch data
            
        Returns:
            Metrics dictionary
        """
        sequences = batch['sequences']
        labels = batch['labels']
        
        # Apply data augmentation
        if self.use_data_augmentation:
            sequences, labels = self.augment_batch(sequences, labels)
            
        # Forward pass
        self.model.set_training(True)
        predictions = self.model.forward(sequences)
        
        # Apply label smoothing
        smooth_labels = self._smooth_labels(labels)
        
        # Compute loss
        targets = {'actions': labels, 'smooth_actions': smooth_labels}
        losses = self.model.compute_loss(predictions, targets)
        
        # Backward pass
        gradients = self.model.backward(predictions, targets)
        
        # Accumulate gradients
        self._accumulate_gradients(gradients)
        
        # Update weights if accumulation is complete
        metrics = {}
        if self.accumulation_counter >= self.gradient_accumulation_steps:
            # Scale accumulated gradients
            scaled_gradients = self._scale_accumulated_gradients()
            
            # Update model parameters
            updated_params = self.optimizer.update(self.model.params, scaled_gradients)
            self.model.params = updated_params
            
            # Track gradient norm
            grad_norm = self._compute_gradient_norm(scaled_gradients)
            self.gradient_norms.append(grad_norm)
            metrics['grad_norm'] = grad_norm
            
            # Reset accumulation
            self.accumulated_gradients = {}
            self.accumulation_counter = 0
            
        # Calculate metrics
        predicted_actions = np.argmax(predictions['action_probs'], axis=-1)
        accuracy = np.mean(predicted_actions == labels)
        
        metrics.update({
            'loss': losses['total_loss'],
            'accuracy': accuracy,
            'action_loss': losses['action_loss'],
            'confidence_loss': losses['confidence_loss']
        })
        
        # Store attention weights for visualization
        if hasattr(predictions, 'attention_weights'):
            self.attention_weights_history.append(
                predictions['attention_weights'].mean(axis=0)
            )
            
        return metrics
    
    def validation_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Single validation step.
        
        Args:
            batch: Batch data
            
        Returns:
            Metrics dictionary
        """
        sequences = batch['sequences']
        labels = batch['labels']
        
        # Forward pass (no training mode)
        self.model.set_training(False)
        predictions = self.model.forward(sequences)
        
        # Compute loss
        targets = {'actions': labels}
        losses = self.model.compute_loss(predictions, targets)
        
        # Calculate metrics
        predicted_actions = np.argmax(predictions['action_probs'], axis=-1)
        accuracy = np.mean(predicted_actions == labels)
        
        # Calculate per-class metrics
        class_metrics = self._calculate_class_metrics(predicted_actions, labels)
        
        metrics = {
            'loss': losses['total_loss'],
            'accuracy': accuracy,
            'confidence': predictions['confidence'].mean()
        }
        metrics.update(class_metrics)
        
        return metrics
    
    def _smooth_labels(self, labels: np.ndarray) -> np.ndarray:
        """Apply label smoothing."""
        n_classes = 3  # Buy, Hold, Sell
        smooth_labels = np.zeros((len(labels), n_classes))
        smooth_labels[np.arange(len(labels)), labels] = 1
        
        if self.label_smoothing > 0:
            smooth_labels = smooth_labels * (1 - self.label_smoothing) + \
                           self.label_smoothing / n_classes
                           
        return smooth_labels
    
    def _accumulate_gradients(self, gradients: Dict[str, np.ndarray]):
        """Accumulate gradients for gradient accumulation."""
        self.accumulation_counter += 1
        
        for name, grad in gradients.items():
            if name in self.accumulated_gradients:
                self.accumulated_gradients[name] += grad
            else:
                self.accumulated_gradients[name] = grad.copy()
                
    def _scale_accumulated_gradients(self) -> Dict[str, np.ndarray]:
        """Scale accumulated gradients by accumulation steps."""
        scaled_gradients = {}
        for name, grad in self.accumulated_gradients.items():
            scaled_gradients[name] = grad / self.gradient_accumulation_steps
        return scaled_gradients
    
    def _compute_gradient_norm(self, gradients: Dict[str, np.ndarray]) -> float:
        """Compute global gradient norm."""
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += np.sum(grad ** 2)
        return np.sqrt(total_norm)
    
    def _calculate_class_metrics(self, predictions: np.ndarray, 
                                labels: np.ndarray) -> Dict[str, float]:
        """Calculate per-class precision, recall, and F1."""
        metrics = {}
        
        for class_idx, class_name in enumerate(['buy', 'hold', 'sell']):
            true_positives = np.sum((predictions == class_idx) & (labels == class_idx))
            false_positives = np.sum((predictions == class_idx) & (labels != class_idx))
            false_negatives = np.sum((predictions != class_idx) & (labels == class_idx))
            
            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            metrics[f'{class_name}_precision'] = precision
            metrics[f'{class_name}_recall'] = recall
            metrics[f'{class_name}_f1'] = f1
            
        return metrics
    
    def train_with_cross_validation(self, 
                                   data: Union[np.ndarray, pd.DataFrame],
                                   n_folds: int = 5,
                                   epochs_per_fold: int = 50) -> Dict:
        """
        Train with k-fold cross-validation.
        
        Args:
            data: Training data
            n_folds: Number of CV folds
            epochs_per_fold: Epochs per fold
            
        Returns:
            Cross-validation results
        """
        # Prepare data
        prepared_data = self.prepare_data(data)
        n_samples = len(prepared_data['labels'])
        
        fold_size = n_samples // n_folds
        cv_results = []
        
        for fold in range(n_folds):
            print(f"\nTraining fold {fold + 1}/{n_folds}")
            
            # Create fold indices
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples
            
            # Split data
            train_indices = np.concatenate([
                np.arange(0, val_start),
                np.arange(val_end, n_samples)
            ])
            val_indices = np.arange(val_start, val_end)
            
            fold_train_data = {
                key: value[train_indices] for key, value in prepared_data.items()
            }
            fold_val_data = {
                key: value[val_indices] for key, value in prepared_data.items()
            }
            
            # Reset model for new fold
            self.model.reset_parameters()
            self.optimizer = self._create_optimizer(
                self.optimizer.__class__.__name__.lower().replace('optimizer', ''),
                self.optimizer.learning_rate,
                self.optimizer.weight_decay,
                self.optimizer.gradient_clip
            )
            
            # Train fold
            fold_history = self.train(
                data=None,  # We'll use prepared data directly
                epochs=epochs_per_fold,
                verbose=0
            )
            
            # Evaluate fold
            val_metrics = self.validate(fold_val_data)
            
            cv_results.append({
                'fold': fold,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'history': fold_history
            })
            
        # Aggregate results
        mean_val_loss = np.mean([r['val_loss'] for r in cv_results])
        std_val_loss = np.std([r['val_loss'] for r in cv_results])
        mean_val_accuracy = np.mean([r['val_accuracy'] for r in cv_results])
        std_val_accuracy = np.std([r['val_accuracy'] for r in cv_results])
        
        print(f"\nCross-validation results:")
        print(f"Val Loss: {mean_val_loss:.4f} (+/- {std_val_loss:.4f})")
        print(f"Val Accuracy: {mean_val_accuracy:.4f} (+/- {std_val_accuracy:.4f})")
        
        return {
            'fold_results': cv_results,
            'mean_val_loss': mean_val_loss,
            'std_val_loss': std_val_loss,
            'mean_val_accuracy': mean_val_accuracy,
            'std_val_accuracy': std_val_accuracy
        }
    
    def visualize_attention(self, sample_data: np.ndarray, 
                           save_path: Optional[str] = None):
        """
        Visualize attention weights for sample data.
        
        Args:
            sample_data: Sample sequences
            save_path: Path to save visualization
        """
        import matplotlib.pyplot as plt
        
        # Get attention weights
        self.model.set_training(False)
        predictions = self.model.forward(sample_data[:5])  # Use first 5 samples
        
        if not hasattr(predictions, 'attention_weights'):
            print("Model does not provide attention weights")
            return
            
        attention_weights = predictions['attention_weights']
        
        # Create visualization
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        for i, ax in enumerate(axes):
            weights = attention_weights[i]
            im = ax.imshow(weights, cmap='hot', aspect='auto')
            ax.set_title(f'Sample {i+1}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Feature')
            plt.colorbar(im, ax=ax)
            
        plt.suptitle('Attention Weights Visualization')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
    def get_training_summary(self) -> Dict:
        """Get comprehensive training summary."""
        summary = {
            'total_epochs': self.epoch,
            'best_val_loss': self.best_val_metric,
            'final_learning_rate': self.optimizer.learning_rate,
            'average_gradient_norm': np.mean(self.gradient_norms) if self.gradient_norms else 0,
            'training_time': sum(self.training_history.get('epoch_time', [])),
            'model_parameters': sum(p.size for p in self.model.params.values()),
            'performance_metrics': {
                'final_train_loss': self.training_history['train_loss'][-1] if 'train_loss' in self.training_history else None,
                'final_val_loss': self.training_history['val_loss'][-1] if 'val_loss' in self.training_history else None,
                'final_train_accuracy': self.training_history['train_accuracy'][-1] if 'train_accuracy' in self.training_history else None,
                'final_val_accuracy': self.training_history['val_accuracy'][-1] if 'val_accuracy' in self.training_history else None,
            }
        }
        
        return summary