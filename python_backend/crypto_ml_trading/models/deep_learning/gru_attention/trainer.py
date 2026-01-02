import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
from collections import deque
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from .model import GRUAttentionModel


class GRUAttentionTrainer:
    """
    Trainer for GRU-Attention model with crypto-specific optimizations.
    
    Features:
    - Mini-batch training with shuffling
    - Learning rate scheduling
    - Early stopping
    - Validation monitoring
    - Checkpointing
    - Performance tracking
    """
    
    def __init__(self, model: GRUAttentionModel,
                 batch_size: int = 32,
                 validation_split: float = 0.2,
                 patience: int = 10,
                 min_delta: float = 0.001):
        """
        Initialize trainer.
        
        Args:
            model: GRU-Attention model to train
            batch_size: Training batch size
            validation_split: Fraction of data for validation
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
        """
        self.model = model
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.min_delta = min_delta
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        # Performance tracking
        self.recent_losses = deque(maxlen=100)
        
    def prepare_data(self, features: np.ndarray, labels: np.ndarray,
                    sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training with sliding windows.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Target labels
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) ready for training
        """
        n_samples = len(features) - sequence_length
        n_features = features.shape[1]
        
        # Create sequences
        X = np.zeros((n_samples, sequence_length, n_features))
        y = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            X[i] = features[i:i + sequence_length]
            y[i] = labels[i + sequence_length]
            
        return X, y
    
    def train_test_split(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        Split data into train and validation sets.
        
        Args:
            X: Input sequences
            y: Target labels
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        n_samples = len(X)
        n_val = int(n_samples * self.validation_split)
        
        # Use last portion for validation (time series)
        X_train = X[:-n_val]
        X_val = X[-n_val:]
        y_train = y[:-n_val]
        y_val = y[-n_val:]
        
        return X_train, X_val, y_train, y_val
    
    def create_mini_batches(self, X: np.ndarray, y: np.ndarray,
                           shuffle: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create mini-batches for training.
        
        Args:
            X: Input data
            y: Target labels
            shuffle: Whether to shuffle data
            
        Returns:
            List of mini-batches
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
            
        batches = []
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            batches.append((batch_X, batch_y))
            
        return batches
    
    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            
        Returns:
            Dictionary with epoch metrics
        """
        self.model.set_training(True)
        
        # Create mini-batches
        batches = self.create_mini_batches(X_train, y_train, shuffle=True)
        
        epoch_losses = []
        epoch_accuracies = []
        
        for batch_X, batch_y in batches:
            # Forward pass
            predictions = self.model.forward(batch_X)
            
            # Compute loss
            targets = {'actions': batch_y}
            losses = self.model.compute_loss(predictions, targets)
            
            # Backward pass through the model
            gradients = self.model.backward(predictions, targets)
            
            # Update weights
            self.model.update_weights(gradients)
            
            # Track metrics
            epoch_losses.append(losses)
            
            # Calculate accuracy
            predicted_actions = np.argmax(predictions['action_probs'], axis=-1)
            accuracy = np.mean(predicted_actions == batch_y)
            epoch_accuracies.append(accuracy)
            
            # Update recent losses
            self.recent_losses.append(losses['total_loss'])
            
        return {
            'loss': np.mean([l['total_loss'] for l in epoch_losses]),
            'accuracy': np.mean(epoch_accuracies),
            'action_loss': np.mean([l['action_loss'] for l in epoch_losses]),
            'confidence_loss': np.mean([l['confidence_loss'] for l in epoch_losses])
        }
    
    
    def validate(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Args:
            X_val: Validation sequences
            y_val: Validation labels
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.set_training(False)
        
        # Create batches (no shuffle for validation)
        batches = self.create_mini_batches(X_val, y_val, shuffle=False)
        
        val_losses = []
        val_accuracies = []
        
        for batch_X, batch_y in batches:
            # Forward pass
            predictions = self.model.forward(batch_X)
            
            # Compute loss
            targets = {'actions': batch_y}
            losses = self.model.compute_loss(predictions, targets)
            
            val_losses.append(losses)
            
            # Calculate accuracy
            predicted_actions = np.argmax(predictions['action_probs'], axis=-1)
            accuracy = np.mean(predicted_actions == batch_y)
            val_accuracies.append(accuracy)
            
        return {
            'loss': np.mean([l['total_loss'] for l in val_losses]),
            'accuracy': np.mean(val_accuracies)
        }
    
    def train(self, X: np.ndarray, y: np.ndarray, 
             epochs: int = 100,
             verbose: int = 1) -> Dict:
        """
        Train the model.
        
        Args:
            X: Input sequences
            y: Target labels
            epochs: Number of epochs
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
            
        Returns:
            Training history
        """
        # Split data
        X_train, X_val, y_train, y_val = self.train_test_split(X, y)
        
        if verbose > 0:
            print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
            print(f"Batch size: {self.batch_size}, Epochs: {epochs}")
            print("-" * 70)
            
        # Training loop
        for epoch in range(epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(X_train, y_train)
            
            # Validate
            val_metrics = self.validate(X_val, y_val)
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_accuracy'].append(train_metrics['accuracy'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['learning_rate'].append(self.model.learning_rate)
            
            # Check for improvement
            if val_metrics['loss'] < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                # Save best model
                self.model.save_model('best_model.npz')
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.patience:
                if verbose > 0:
                    print(f"\nEarly stopping at epoch {epoch}")
                break
                
            # Learning rate decay
            if epoch > 0 and epoch % 10 == 0:
                self.model.learning_rate *= 0.9
                
            # Print progress
            if verbose > 0 and epoch % max(1, epochs // 20) == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:3d}/{epochs} - {elapsed:.1f}s - "
                      f"loss: {train_metrics['loss']:.4f} - "
                      f"acc: {train_metrics['accuracy']:.4f} - "
                      f"val_loss: {val_metrics['loss']:.4f} - "
                      f"val_acc: {val_metrics['accuracy']:.4f}")
                      
            if verbose > 1:
                print(f"  Action loss: {train_metrics['action_loss']:.4f}, "
                      f"Confidence loss: {train_metrics['confidence_loss']:.4f}")
                
        # Load best model
        self.model.load_model('best_model.npz')
        
        if verbose > 0:
            print("-" * 70)
            print(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")
            
        return self.training_history
    
    def predict_with_confidence(self, X: np.ndarray) -> pd.DataFrame:
        """
        Make predictions with confidence intervals.
        
        Args:
            X: Input sequences
            
        Returns:
            DataFrame with predictions and confidence
        """
        predictions = self.model.predict(X)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'action': predictions['actions'],
            'buy_prob': predictions['action_probs'][:, 0],
            'hold_prob': predictions['action_probs'][:, 1],
            'sell_prob': predictions['action_probs'][:, 2],
            'confidence': predictions['confidence'].flatten(),
            'prediction_entropy': -np.sum(
                predictions['action_probs'] * np.log(predictions['action_probs'] + 1e-10),
                axis=1
            )
        })
        
        # Add trading signals
        results['signal_strength'] = results['confidence'] * (
            results[['buy_prob', 'sell_prob']].max(axis=1) - results['hold_prob']
        )
        
        return results
    
    def analyze_predictions(self, X: np.ndarray, y_true: np.ndarray) -> Dict:
        """
        Analyze model predictions.
        
        Args:
            X: Input sequences
            y_true: True labels
            
        Returns:
            Dictionary with analysis metrics
        """
        predictions = self.model.predict(X)
        predicted_actions = np.argmax(predictions['action_probs'], axis=-1)
        
        # Confusion matrix
        confusion_matrix = np.zeros((3, 3))
        for true, pred in zip(y_true, predicted_actions):
            confusion_matrix[true, pred] += 1
            
        # Per-class metrics
        class_metrics = {}
        for i, action in enumerate(['buy', 'hold', 'sell']):
            true_positives = confusion_matrix[i, i]
            false_positives = confusion_matrix[:, i].sum() - true_positives
            false_negatives = confusion_matrix[i, :].sum() - true_positives
            
            precision = true_positives / (true_positives + false_positives + 1e-10)
            recall = true_positives / (true_positives + false_negatives + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            class_metrics[action] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
        # Overall metrics
        overall_accuracy = np.mean(predicted_actions == y_true)
        
        # Confidence analysis
        confidence_stats = {
            'mean_confidence': np.mean(predictions['confidence']),
            'confidence_when_correct': np.mean(
                predictions['confidence'][predicted_actions == y_true]
            ),
            'confidence_when_wrong': np.mean(
                predictions['confidence'][predicted_actions != y_true]
            )
        }
        
        return {
            'confusion_matrix': confusion_matrix,
            'class_metrics': class_metrics,
            'overall_accuracy': overall_accuracy,
            'confidence_stats': confidence_stats,
            'attention_analysis': self.model.get_attention_analysis(X[:10])  # Sample
        }