import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from .cnn_model import CNNPatternRecognizer
from .pattern_generator import PatternGenerator


class CNNPatternTrainer:
    """
    Trainer for CNN pattern recognition model.
    
    Features:
    - Pattern augmentation
    - Class balancing
    - Multi-scale training
    - Transfer learning simulation
    """
    
    def __init__(self, model: CNNPatternRecognizer,
                 learning_rate: float = 0.001,
                 batch_size: int = 32):
        """
        Initialize trainer.
        
        Args:
            model: CNN model to train
            learning_rate: Initial learning rate
            batch_size: Training batch size
        """
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Optimizer state (Adam)
        self.optimizer_state = self._init_optimizer()
        
    def _init_optimizer(self) -> Dict:
        """Initialize Adam optimizer state."""
        state = {}
        
        for param_name in self.model.params:
            if param_name.startswith('running'):  # Skip BN running stats
                continue
                
            state[f'm_{param_name}'] = np.zeros_like(self.model.params[param_name])
            state[f'v_{param_name}'] = np.zeros_like(self.model.params[param_name])
            
        state['t'] = 0  # Time step
        
        return state
    
    def prepare_training_data(self, data: pd.DataFrame,
                            pattern_labels: Optional[np.ndarray] = None,
                            window_size: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from OHLCV.
        
        Args:
            data: OHLCV data
            pattern_labels: Optional pre-labeled patterns
            window_size: Size of pattern window
            
        Returns:
            Tuple of (images, labels)
        """
        # Generate pattern images
        generator = PatternGenerator(
            image_size=self.model.image_size,
            methods=['gasf', 'gadf', 'rp']
        )
        
        images = generator.generate_pattern_images(data, window_size)
        
        # Generate or use provided labels
        if pattern_labels is None:
            labels = self._generate_synthetic_labels(data, window_size)
        else:
            labels = pattern_labels
            
        return images, labels
    
    def _generate_synthetic_labels(self, data: pd.DataFrame,
                                 window_size: int) -> np.ndarray:
        """Generate synthetic pattern labels based on price movements."""
        n_samples = len(data) - window_size + 1
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            window = data.iloc[i:i+window_size]
            future_price = data.iloc[i+window_size]['close'] if i+window_size < len(data) else window.iloc[-1]['close']
            current_price = window.iloc[-1]['close']
            
            # Calculate return
            ret = (future_price - current_price) / current_price
            
            # Classify based on return and volatility
            volatility = window['close'].pct_change().std()
            
            if ret > 2 * volatility:  # Strong bullish
                labels[i] = 1  # Bullish reversal
            elif ret < -2 * volatility:  # Strong bearish
                labels[i] = 2  # Bearish reversal
            elif abs(ret) < 0.5 * volatility:  # Low movement
                labels[i] = 4  # Consolidation
            else:
                labels[i] = 3  # Continuation
                
        return labels
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             epochs: int = 50,
             verbose: int = 1) -> Dict:
        """
        Train the CNN model.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of epochs
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        n_train = len(X_train)
        
        if verbose > 0:
            print(f"Training on {n_train} samples")
            if X_val is not None:
                print(f"Validating on {len(X_val)} samples")
            print("-" * 50)
            
        for epoch in range(epochs):
            start_time = time.time()
            
            # Shuffle training data
            indices = np.random.permutation(n_train)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Mini-batch training
            train_losses = []
            train_accs = []
            
            self.model.set_training(True)
            
            for i in range(0, n_train, self.batch_size):
                batch_X = X_train_shuffled[i:i+self.batch_size]
                batch_y = y_train_shuffled[i:i+self.batch_size]
                
                # Forward pass
                output = self.model.forward(batch_X)
                
                # Calculate loss
                loss = self._cross_entropy_loss(output['logits'], batch_y)
                train_losses.append(loss)
                
                # Calculate accuracy
                acc = np.mean(output['predictions'] == batch_y)
                train_accs.append(acc)
                
                # Backward pass (simplified gradient calculation)
                gradients = self._compute_gradients(batch_X, batch_y, output)
                
                # Update weights
                self._update_weights(gradients)
                
            # Validation
            if X_val is not None:
                self.model.set_training(False)
                val_output = self.model.forward(X_val)
                val_loss = self._cross_entropy_loss(val_output['logits'], y_val)
                val_acc = np.mean(val_output['predictions'] == y_val)
            else:
                val_loss = val_acc = 0
                
            # Update history
            self.history['train_loss'].append(np.mean(train_losses))
            self.history['train_acc'].append(np.mean(train_accs))
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate decay
            if epoch > 0 and epoch % 10 == 0:
                self.learning_rate *= 0.9
                
            # Print progress
            if verbose > 0 and epoch % max(1, epochs // 10) == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch}/{epochs} - {elapsed:.1f}s - "
                      f"loss: {self.history['train_loss'][-1]:.4f} - "
                      f"acc: {self.history['train_acc'][-1]:.4f}", end="")
                      
                if X_val is not None:
                    print(f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
                else:
                    print()
                    
        return self.history
    
    def _cross_entropy_loss(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Calculate cross-entropy loss."""
        n_samples = len(labels)
        
        # Convert to probabilities
        probs = self.model._softmax(logits)
        
        # Clip to prevent log(0)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        
        # One-hot encode labels
        y_one_hot = np.zeros((n_samples, self.model.num_classes))
        y_one_hot[np.arange(n_samples), labels] = 1
        
        # Calculate loss
        loss = -np.sum(y_one_hot * np.log(probs)) / n_samples
        
        return loss
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray,
                         output: Dict) -> Dict:
        """
        Compute gradients using proper backpropagation through the CNN.
        
        Args:
            X: Input images
            y: True labels
            output: Forward pass output
            
        Returns:
            Dictionary of gradients for all parameters
        """
        # Use the CNN's backward method for proper gradient computation
        gradients = self.model.backward(X, y, output)
        
        return gradients
    
    def _update_weights(self, gradients: Dict):
        """Update weights using Adam optimizer."""
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        
        self.optimizer_state['t'] += 1
        t = self.optimizer_state['t']
        
        for param_name in self.model.params:
            if param_name.startswith('running') or param_name not in gradients:
                continue
                
            # Get gradient
            grad = gradients[param_name]
            
            # Update biased first moment estimate
            self.optimizer_state[f'm_{param_name}'] = (
                beta1 * self.optimizer_state[f'm_{param_name}'] +
                (1 - beta1) * grad
            )
            
            # Update biased second raw moment estimate
            self.optimizer_state[f'v_{param_name}'] = (
                beta2 * self.optimizer_state[f'v_{param_name}'] +
                (1 - beta2) * grad**2
            )
            
            # Compute bias-corrected first moment estimate
            m_hat = self.optimizer_state[f'm_{param_name}'] / (1 - beta1**t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.optimizer_state[f'v_{param_name}'] / (1 - beta2**t)
            
            # Update parameters
            self.model.params[param_name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    def augment_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment training data.
        
        Args:
            X: Training images
            y: Training labels
            
        Returns:
            Augmented data
        """
        augmented_X = [X]
        augmented_y = [y]
        
        # Horizontal flip
        flipped = X[:, :, :, ::-1]
        augmented_X.append(flipped)
        augmented_y.append(y)
        
        # Add noise
        noise = X + np.random.randn(*X.shape) * 0.05
        augmented_X.append(noise)
        augmented_y.append(y)
        
        # Brightness adjustment
        bright = X * 1.1
        dark = X * 0.9
        augmented_X.append(bright)
        augmented_X.append(dark)
        augmented_y.append(y)
        augmented_y.append(y)
        
        return np.concatenate(augmented_X), np.concatenate(augmented_y)
    
    def evaluate_patterns(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model on pattern recognition.
        
        Args:
            X: Test images
            y: True labels
            
        Returns:
            Evaluation metrics
        """
        self.model.set_training(False)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Calculate metrics
        accuracy = np.mean(predictions == y)
        
        # Per-class metrics
        pattern_names = [
            'no_pattern',
            'bullish_reversal',
            'bearish_reversal',
            'continuation',
            'consolidation'
        ]
        
        class_metrics = {}
        for i in range(self.model.num_classes):
            mask = y == i
            if np.sum(mask) > 0:
                class_acc = np.mean(predictions[mask] == i)
                class_metrics[pattern_names[i]] = {
                    'accuracy': class_acc,
                    'support': np.sum(mask),
                    'avg_confidence': np.mean(probabilities[mask, i])
                }
                
        # Confusion matrix
        confusion_matrix = np.zeros((self.model.num_classes, self.model.num_classes))
        for true, pred in zip(y, predictions):
            confusion_matrix[true, pred] += 1
            
        return {
            'overall_accuracy': accuracy,
            'class_metrics': class_metrics,
            'confusion_matrix': confusion_matrix,
            'avg_confidence': np.mean(np.max(probabilities, axis=1))
        }