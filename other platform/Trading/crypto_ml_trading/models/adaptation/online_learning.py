"""
Online Learning Framework for Crypto Trading Models.

Implements various online learning algorithms for real-time model updates.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class OnlineLearner(ABC):
    """Abstract base class for online learning algorithms."""
    
    @abstractmethod
    def partial_fit(self, X: np.ndarray, y: np.ndarray, 
                   sample_weight: Optional[np.ndarray] = None):
        """Update model with new data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get current model parameters."""
        pass


class OnlineLinearRegression(OnlineLearner):
    """
    Online linear regression using stochastic gradient descent.
    
    Features:
    - Adaptive learning rate
    - L1/L2 regularization
    - Feature scaling
    """
    
    def __init__(self,
                 learning_rate: float = 0.01,
                 regularization: str = 'l2',
                 reg_strength: float = 0.01,
                 adaptive_lr: bool = True,
                 feature_scaling: bool = True):
        """
        Initialize online linear regression.
        
        Args:
            learning_rate: Initial learning rate
            regularization: Type of regularization ('l1', 'l2', 'none')
            reg_strength: Regularization strength
            adaptive_lr: Use adaptive learning rate
            feature_scaling: Scale features online
        """
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.reg_strength = reg_strength
        self.adaptive_lr = adaptive_lr
        self.feature_scaling = feature_scaling
        
        # Model parameters
        self.coef_ = None
        self.intercept_ = 0.0
        
        # Feature scaling parameters
        self.feature_mean_ = None
        self.feature_std_ = None
        self.n_samples_seen_ = 0
        
        # Adaptive learning rate parameters
        self.lr_schedule_ = []
        self.gradient_history_ = deque(maxlen=100)
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                   sample_weight: Optional[np.ndarray] = None):
        """Update model with new samples."""
        n_samples, n_features = X.shape
        
        # Initialize coefficients
        if self.coef_ is None:
            self.coef_ = np.zeros(n_features)
            self.feature_mean_ = np.zeros(n_features)
            self.feature_std_ = np.ones(n_features)
        
        # Update feature statistics
        if self.feature_scaling:
            self._update_feature_stats(X)
            X_scaled = self._scale_features(X)
        else:
            X_scaled = X
        
        # SGD updates
        for i in range(n_samples):
            xi = X_scaled[i]
            yi = y[i]
            weight = sample_weight[i] if sample_weight is not None else 1.0
            
            # Predict
            y_pred = np.dot(xi, self.coef_) + self.intercept_
            
            # Calculate error
            error = yi - y_pred
            
            # Calculate gradients
            grad_coef = -2 * error * xi * weight
            grad_intercept = -2 * error * weight
            
            # Add regularization gradient
            if self.regularization == 'l2':
                grad_coef += 2 * self.reg_strength * self.coef_
            elif self.regularization == 'l1':
                grad_coef += self.reg_strength * np.sign(self.coef_)
            
            # Update learning rate
            if self.adaptive_lr:
                current_lr = self._get_adaptive_lr(grad_coef)
            else:
                current_lr = self.learning_rate
            
            # Update parameters
            self.coef_ -= current_lr * grad_coef
            self.intercept_ -= current_lr * grad_intercept
            
            self.n_samples_seen_ += 1
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet")
        
        # Scale features
        if self.feature_scaling:
            X_scaled = self._scale_features(X)
        else:
            X_scaled = X
        
        return np.dot(X_scaled, self.coef_) + self.intercept_
    
    def _update_feature_stats(self, X: np.ndarray):
        """Update running statistics for feature scaling."""
        n_samples = X.shape[0]
        
        # Incremental mean update
        delta = X - self.feature_mean_
        self.feature_mean_ += np.mean(delta, axis=0)
        
        # Incremental variance update (Welford's algorithm)
        delta2 = X - self.feature_mean_
        self.feature_std_ = np.sqrt(
            (self.feature_std_**2 * (self.n_samples_seen_ - 1) + 
             np.sum(delta * delta2, axis=0)) / self.n_samples_seen_
        )
        
        # Prevent division by zero
        self.feature_std_ = np.maximum(self.feature_std_, 1e-8)
    
    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Scale features using running statistics."""
        return (X - self.feature_mean_) / self.feature_std_
    
    def _get_adaptive_lr(self, gradient: np.ndarray) -> float:
        """Calculate adaptive learning rate."""
        self.gradient_history_.append(np.sum(gradient**2))
        
        if len(self.gradient_history_) < 10:
            return self.learning_rate
        
        # AdaGrad-style learning rate
        grad_sum = np.mean(self.gradient_history_)
        adapted_lr = self.learning_rate / (1.0 + np.sqrt(grad_sum))
        
        # Learning rate decay
        decay_rate = 0.9999
        iteration = self.n_samples_seen_
        adapted_lr *= decay_rate ** (iteration / 1000)
        
        return max(adapted_lr, 1e-6)  # Minimum learning rate
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'coef': self.coef_.tolist() if self.coef_ is not None else None,
            'intercept': self.intercept_,
            'n_samples_seen': self.n_samples_seen_,
            'feature_mean': self.feature_mean_.tolist() if self.feature_mean_ is not None else None,
            'feature_std': self.feature_std_.tolist() if self.feature_std_ is not None else None
        }


class OnlineNeuralNetwork(OnlineLearner):
    """
    Online neural network with mini-batch updates.
    
    Features:
    - Single hidden layer
    - ReLU activation
    - Adam optimizer
    - Experience replay
    """
    
    def __init__(self,
                 hidden_size: int = 64,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 replay_buffer_size: int = 1000,
                 dropout_rate: float = 0.1):
        """
        Initialize online neural network.
        
        Args:
            hidden_size: Number of hidden units
            learning_rate: Learning rate
            batch_size: Mini-batch size
            replay_buffer_size: Size of experience replay buffer
            dropout_rate: Dropout probability
        """
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        
        # Network parameters
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        
        # Optimizer state (Adam)
        self.m_W1 = None
        self.v_W1 = None
        self.m_b1 = None
        self.v_b1 = None
        self.m_W2 = None
        self.v_W2 = None
        self.m_b2 = None
        self.v_b2 = None
        self.t = 0
        
        # Experience replay
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        
    def _initialize_weights(self, n_features: int, n_outputs: int):
        """Initialize network weights using Xavier initialization."""
        # Hidden layer
        limit1 = np.sqrt(6 / (n_features + self.hidden_size))
        self.W1 = np.random.uniform(-limit1, limit1, (n_features, self.hidden_size))
        self.b1 = np.zeros(self.hidden_size)
        
        # Output layer
        limit2 = np.sqrt(6 / (self.hidden_size + n_outputs))
        self.W2 = np.random.uniform(-limit2, limit2, (self.hidden_size, n_outputs))
        self.b2 = np.zeros(n_outputs)
        
        # Initialize Adam moments
        self.m_W1 = np.zeros_like(self.W1)
        self.v_W1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)
        self.m_W2 = np.zeros_like(self.W2)
        self.v_W2 = np.zeros_like(self.W2)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)
    
    def _forward(self, X: np.ndarray, training: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through network."""
        # Hidden layer
        z1 = np.dot(X, self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        
        # Dropout
        if training and self.dropout_rate > 0:
            mask = np.random.rand(*a1.shape) > self.dropout_rate
            a1 = a1 * mask / (1 - self.dropout_rate)
        
        # Output layer
        z2 = np.dot(a1, self.W2) + self.b2
        
        return z2, a1, z1
    
    def _backward(self, X: np.ndarray, y: np.ndarray, 
                  a1: np.ndarray, z1: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass to calculate gradients."""
        m = X.shape[0]
        
        # Output layer gradients
        y_pred, _, _ = self._forward(X, training=False)
        dz2 = (y_pred - y) / m
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0)
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (z1 > 0)  # ReLU gradient
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0)
        
        return {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2
        }
    
    def _adam_update(self, param: str, gradient: np.ndarray, 
                     m: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Adam optimizer update."""
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        
        # Update biased moments
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient**2
        
        # Bias correction
        m_hat = m / (1 - beta1**(self.t + 1))
        v_hat = v / (1 - beta2**(self.t + 1))
        
        # Update parameter
        param_update = self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        return param_update, m, v
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                   sample_weight: Optional[np.ndarray] = None):
        """Update network with new samples."""
        n_features = X.shape[1]
        n_outputs = 1 if len(y.shape) == 1 else y.shape[1]
        
        # Initialize weights if needed
        if self.W1 is None:
            self._initialize_weights(n_features, n_outputs)
        
        # Reshape y if needed
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Add to replay buffer
        for i in range(len(X)):
            weight = sample_weight[i] if sample_weight is not None else 1.0
            self.replay_buffer.append((X[i], y[i], weight))
        
        # Train on mini-batches
        if len(self.replay_buffer) >= self.batch_size:
            # Sample batch from replay buffer
            indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in indices]
            
            X_batch = np.array([b[0] for b in batch])
            y_batch = np.array([b[1] for b in batch])
            weights_batch = np.array([b[2] for b in batch])
            
            # Forward pass
            _, a1, z1 = self._forward(X_batch, training=True)
            
            # Backward pass
            gradients = self._backward(X_batch, y_batch, a1, z1)
            
            # Apply sample weights
            for key in gradients:
                if 'W' in key:
                    gradients[key] *= weights_batch.reshape(-1, 1)
                else:
                    gradients[key] *= weights_batch
            
            # Update parameters with Adam
            self.t += 1
            
            # Update W1
            update_W1, self.m_W1, self.v_W1 = self._adam_update(
                'W1', gradients['dW1'], self.m_W1, self.v_W1
            )
            self.W1 -= update_W1
            
            # Update b1
            update_b1, self.m_b1, self.v_b1 = self._adam_update(
                'b1', gradients['db1'], self.m_b1, self.v_b1
            )
            self.b1 -= update_b1
            
            # Update W2
            update_W2, self.m_W2, self.v_W2 = self._adam_update(
                'W2', gradients['dW2'], self.m_W2, self.v_W2
            )
            self.W2 -= update_W2
            
            # Update b2
            update_b2, self.m_b2, self.v_b2 = self._adam_update(
                'b2', gradients['db2'], self.m_b2, self.v_b2
            )
            self.b2 -= update_b2
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.W1 is None:
            raise ValueError("Model has not been fitted yet")
        
        y_pred, _, _ = self._forward(X, training=False)
        
        # Squeeze if single output
        if y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze()
        
        return y_pred
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'hidden_size': self.hidden_size,
            'n_parameters': (self.W1.size + self.b1.size + self.W2.size + self.b2.size) if self.W1 is not None else 0,
            'replay_buffer_size': len(self.replay_buffer),
            'training_steps': self.t
        }


class OnlineGradientBoostingRegressor(OnlineLearner):
    """
    Online gradient boosting using incremental trees.
    
    Features:
    - Incremental tree building
    - Adaptive learning rate
    - Feature importance tracking
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 subsample: float = 0.8,
                 min_samples_leaf: int = 20):
        """
        Initialize online gradient boosting.
        
        Args:
            n_estimators: Maximum number of trees
            learning_rate: Boosting learning rate
            max_depth: Maximum tree depth
            subsample: Subsample ratio for each tree
            min_samples_leaf: Minimum samples in leaf
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf
        
        # Trees and predictions
        self.trees_ = []
        self.feature_importances_ = None
        self.train_scores_ = []
        
        # Buffer for tree building
        self.buffer_X = deque(maxlen=1000)
        self.buffer_y = deque(maxlen=1000)
        self.buffer_residuals = deque(maxlen=1000)
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                   sample_weight: Optional[np.ndarray] = None):
        """Update model with new samples."""
        n_features = X.shape[1]
        
        # Initialize feature importances
        if self.feature_importances_ is None:
            self.feature_importances_ = np.zeros(n_features)
        
        # Add to buffer
        for i in range(len(X)):
            self.buffer_X.append(X[i])
            self.buffer_y.append(y[i])
            
            # Calculate residual
            if self.trees_:
                pred = self._predict_sample(X[i])
                residual = y[i] - pred
            else:
                residual = y[i]
            
            self.buffer_residuals.append(residual)
        
        # Build new tree if buffer is full
        if len(self.buffer_X) >= 100 and len(self.trees_) < self.n_estimators:
            self._build_tree()
    
    def _build_tree(self):
        """Build a new tree on buffered data."""
        X_buffer = np.array(self.buffer_X)
        residuals = np.array(self.buffer_residuals)
        
        # Subsample
        n_samples = len(X_buffer)
        subsample_size = int(n_samples * self.subsample)
        indices = np.random.choice(n_samples, subsample_size, replace=False)
        
        X_subsample = X_buffer[indices]
        residuals_subsample = residuals[indices]
        
        # Build simple decision tree (stub implementation)
        tree = self._build_decision_tree(X_subsample, residuals_subsample)
        self.trees_.append(tree)
        
        # Update residuals in buffer
        for i in range(len(self.buffer_X)):
            pred_update = self._tree_predict(tree, self.buffer_X[i])
            self.buffer_residuals[i] -= self.learning_rate * pred_update
        
        # Track training score
        train_pred = self.predict(X_buffer)
        train_score = 1 - np.mean((np.array(self.buffer_y) - train_pred)**2) / np.var(self.buffer_y)
        self.train_scores_.append(train_score)
    
    def _build_decision_tree(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Build a simple decision tree (placeholder implementation)."""
        # This is a simplified tree structure
        # In practice, would use a proper tree building algorithm
        
        tree = {
            'type': 'tree',
            'n_samples': len(X),
            'prediction': np.mean(y)
        }
        
        # Simple split on best feature
        if len(X) > self.min_samples_leaf * 2:
            best_feature, best_threshold, best_score = self._find_best_split(X, y)
            
            if best_feature is not None:
                left_mask = X[:, best_feature] <= best_threshold
                right_mask = ~left_mask
                
                tree['feature'] = best_feature
                tree['threshold'] = best_threshold
                tree['left'] = {'prediction': np.mean(y[left_mask]) if np.any(left_mask) else 0}
                tree['right'] = {'prediction': np.mean(y[right_mask]) if np.any(right_mask) else 0}
                
                # Update feature importance
                self.feature_importances_[best_feature] += best_score
        
        return tree
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """Find best split for tree node."""
        n_features = X.shape[1]
        best_score = 0
        best_feature = None
        best_threshold = None
        
        current_mse = np.mean((y - np.mean(y))**2)
        
        # Try each feature
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds[:-1]:  # Skip last
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # Calculate MSE reduction
                left_mse = np.mean((y[left_mask] - np.mean(y[left_mask]))**2) if np.any(left_mask) else 0
                right_mse = np.mean((y[right_mask] - np.mean(y[right_mask]))**2) if np.any(right_mask) else 0
                
                weighted_mse = (np.sum(left_mask) * left_mse + np.sum(right_mask) * right_mse) / len(y)
                score = current_mse - weighted_mse
                
                if score > best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_score
    
    def _tree_predict(self, tree: Dict[str, Any], x: np.ndarray) -> float:
        """Predict using a single tree."""
        if 'feature' not in tree:
            return tree['prediction']
        
        if x[tree['feature']] <= tree['threshold']:
            return tree['left']['prediction']
        else:
            return tree['right']['prediction']
    
    def _predict_sample(self, x: np.ndarray) -> float:
        """Predict for a single sample."""
        pred = 0.0
        for tree in self.trees_:
            pred += self.learning_rate * self._tree_predict(tree, x)
        return pred
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        predictions = np.zeros(len(X))
        
        for i, x in enumerate(X):
            predictions[i] = self._predict_sample(x)
        
        return predictions
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'n_trees': len(self.trees_),
            'buffer_size': len(self.buffer_X),
            'feature_importances': self.feature_importances_.tolist() if self.feature_importances_ is not None else None,
            'latest_train_score': self.train_scores_[-1] if self.train_scores_ else None
        }