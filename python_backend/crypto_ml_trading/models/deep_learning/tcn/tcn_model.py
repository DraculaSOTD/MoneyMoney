import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from utils.matrix_operations import MatrixOperations

logger = logging.getLogger(__name__)


class TCNModel:
    """
    Temporal Convolutional Network (TCN) implementation.
    
    TCNs use dilated causal convolutions to capture long-range dependencies
    in sequential data while maintaining computational efficiency.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        num_layers: int = 4,
        dilations: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        use_skip_connections: bool = True,
        activation: str = 'relu'
    ):
        # Remove base class init
        self.input_size = input_size
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dilations = dilations or [2**i for i in range(num_layers)]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_skip_connections = use_skip_connections
        self.activation_name = activation
        
        # Initialize layers
        self.conv_layers = []
        self.batch_norm_layers = []
        self.dropout_masks = []
        self.skip_connections = []
        
        # Output layer
        self.output_weight = None
        self.output_bias = None
        
        self._build_network()
        
    def _build_network(self):
        """Build the TCN architecture."""
        in_channels = self.input_size
        
        for i in range(self.num_layers):
            # Convolutional layer
            conv_weight = self._init_weights((self.kernel_size, in_channels, self.hidden_channels))
            conv_bias = np.zeros(self.hidden_channels)
            self.conv_layers.append({'weight': conv_weight, 'bias': conv_bias})
            
            # Batch normalization parameters
            bn_gamma = np.ones(self.hidden_channels)
            bn_beta = np.zeros(self.hidden_channels)
            bn_mean = np.zeros(self.hidden_channels)
            bn_var = np.ones(self.hidden_channels)
            self.batch_norm_layers.append({
                'gamma': bn_gamma, 'beta': bn_beta,
                'running_mean': bn_mean, 'running_var': bn_var
            })
            
            # Skip connection weights if input/output dimensions differ
            if self.use_skip_connections and in_channels != self.hidden_channels:
                skip_weight = self._init_weights((1, in_channels, self.hidden_channels))
                self.skip_connections.append(skip_weight)
            else:
                self.skip_connections.append(None)
            
            in_channels = self.hidden_channels
        
        # Output projection
        self.output_weight = self._init_weights((self.hidden_channels, 3))  # 3 actions
        self.output_bias = np.zeros(3)
        
    def _init_weights(self, shape):
        """Initialize weights using He initialization."""
        fan_in = np.prod(shape[:-1])
        std = np.sqrt(2.0 / fan_in)
        return np.random.normal(0, std, shape)
    
    def _dilated_conv1d(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray, 
                        dilation: int) -> np.ndarray:
        """
        Perform dilated causal convolution.
        
        Args:
            x: Input tensor (batch_size, seq_len, channels)
            weight: Conv weights (kernel_size, in_channels, out_channels)
            bias: Conv bias (out_channels,)
            dilation: Dilation rate
        """
        batch_size, seq_len, _ = x.shape
        kernel_size, _, out_channels = weight.shape
        
        # Calculate receptive field
        receptive_field = (kernel_size - 1) * dilation + 1
        
        # Pad for causal convolution
        padding = receptive_field - 1
        x_padded = np.pad(x, ((0, 0), (padding, 0), (0, 0)), mode='constant')
        
        # Output tensor
        output = np.zeros((batch_size, seq_len, out_channels))
        
        # Perform convolution
        for t in range(seq_len):
            for k in range(kernel_size):
                idx = t + k * dilation
                if idx < x_padded.shape[1]:
                    output[:, t, :] += np.dot(x_padded[:, idx, :], weight[k])
            output[:, t, :] += bias
        
        return output
    
    def _batch_norm(self, x: np.ndarray, bn_params: Dict[str, np.ndarray], 
                    training: bool = True) -> np.ndarray:
        """Apply batch normalization."""
        if training:
            mean = np.mean(x, axis=(0, 1), keepdims=True)
            var = np.var(x, axis=(0, 1), keepdims=True)
            
            # Update running statistics
            momentum = 0.9
            bn_params['running_mean'] = momentum * bn_params['running_mean'] + (1 - momentum) * mean.squeeze()
            bn_params['running_var'] = momentum * bn_params['running_var'] + (1 - momentum) * var.squeeze()
        else:
            mean = bn_params['running_mean'].reshape(1, 1, -1)
            var = bn_params['running_var'].reshape(1, 1, -1)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + 1e-8)
        
        # Scale and shift
        gamma = bn_params['gamma'].reshape(1, 1, -1)
        beta = bn_params['beta'].reshape(1, 1, -1)
        
        return gamma * x_norm + beta
    
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation_name == 'relu':
            return np.maximum(0, x)
        elif self.activation_name == 'tanh':
            return np.tanh(x)
        elif self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            return x
    
    def _dropout(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply dropout."""
        if training and self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape) / (1 - self.dropout_rate)
            return x * mask
        return x
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through TCN.
        
        Args:
            x: Input data (batch_size, seq_len, features)
            training: Whether in training mode
        
        Returns:
            Output predictions (batch_size, seq_len, 3)
        """
        batch_size, seq_len, _ = x.shape
        
        # Process through TCN layers
        hidden = x
        for i in range(self.num_layers):
            # Save input for skip connection
            residual = hidden
            
            # Dilated convolution
            hidden = self._dilated_conv1d(
                hidden, 
                self.conv_layers[i]['weight'],
                self.conv_layers[i]['bias'],
                self.dilations[i]
            )
            
            # Batch normalization
            hidden = self._batch_norm(hidden, self.batch_norm_layers[i], training)
            
            # Activation
            hidden = self._activation(hidden)
            
            # Dropout
            hidden = self._dropout(hidden, training)
            
            # Skip connection
            if self.use_skip_connections:
                if self.skip_connections[i] is not None:
                    # Project residual to match dimensions
                    residual = self._dilated_conv1d(
                        residual,
                        self.skip_connections[i],
                        np.zeros(self.hidden_channels),
                        dilation=1
                    )
                hidden = hidden + residual
        
        # Global pooling (take last timestep for now)
        hidden = hidden[:, -1, :]
        
        # Output projection
        output = np.dot(hidden, self.output_weight) + self.output_bias
        
        return output
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None) -> 'TCNModel':
        """
        Train the TCN model.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Target values (n_samples,)
            validation_data: Optional (X_val, y_val) tuple
        """
        # Validate and prepare data
        # Validate data
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")
        
        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Reshape for TCN (add time dimension)
        sequence_length = min(60, X.shape[0] // 10)  # Use last 60 timesteps or less
        X = self._create_sequences(X, sequence_length)
        y = y[-X.shape[0]:]  # Align targets
        
        # Convert targets to one-hot
        y_onehot = self._to_categorical(y)
        
        # Training loop
        n_batches = len(X) // self.batch_size
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            indices = np.random.permutation(len(X))
            
            for batch_idx in range(n_batches):
                batch_indices = indices[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
                X_batch = X[batch_indices]
                y_batch = y_onehot[batch_indices]
                
                # Forward pass
                predictions = self.forward(X_batch, training=True)
                
                # Compute loss (cross-entropy)
                loss = self._compute_loss(predictions, y_batch)
                epoch_loss += loss
                
                # Backward pass (simplified gradient descent)
                self._backward(X_batch, y_batch, predictions)
            
            # Log progress
            avg_loss = epoch_loss / n_batches
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def _create_sequences(self, X: np.ndarray, sequence_length: int) -> np.ndarray:
        """Create sequences from time series data."""
        n_samples = len(X) - sequence_length + 1
        n_features = X.shape[1]
        
        sequences = np.zeros((n_samples, sequence_length, n_features))
        for i in range(n_samples):
            sequences[i] = X[i:i + sequence_length]
        
        return sequences
    
    def _to_categorical(self, y: np.ndarray) -> np.ndarray:
        """Convert labels to one-hot encoding."""
        n_classes = 3
        n_samples = len(y)
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y.astype(int)] = 1
        return y_onehot
    
    def _compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        # Softmax
        exp_pred = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        softmax = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        
        # Cross-entropy
        loss = -np.sum(targets * np.log(softmax + 1e-8)) / len(predictions)
        return loss
    
    def _backward(self, X_batch: np.ndarray, y_batch: np.ndarray, predictions: np.ndarray):
        """Simplified backward pass with gradient descent."""
        # Compute gradients (simplified)
        batch_size = len(X_batch)
        
        # Softmax
        exp_pred = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        softmax = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        
        # Gradient of loss w.r.t. output
        d_output = (softmax - y_batch) / batch_size
        
        # Get last hidden state from forward pass
        # Process through TCN layers again to get hidden representation
        hidden = X_batch
        for i in range(self.num_layers):
            residual = hidden
            hidden = self._dilated_conv1d(hidden, self.conv_layers[i]['weight'], 
                                        self.conv_layers[i]['bias'], self.dilations[i])
            hidden = self._batch_norm(hidden, self.batch_norm_layers[i], training=False)
            hidden = self._activation(hidden)
            
            if self.use_skip_connections:
                if self.skip_connections[i] is not None:
                    residual = self._dilated_conv1d(residual, self.skip_connections[i],
                                                  np.zeros(self.hidden_channels), dilation=1)
                hidden = hidden + residual
        
        # Take last timestep
        hidden = hidden[:, -1, :]
        
        # Update output layer
        self.output_weight -= self.learning_rate * np.dot(hidden.T, d_output)
        self.output_bias -= self.learning_rate * np.sum(d_output, axis=0)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Create sequences
        sequence_length = min(60, X.shape[0])
        if X.shape[0] >= sequence_length:
            X_seq = self._create_sequences(X, sequence_length)
            
            # Forward pass
            predictions = self.forward(X_seq, training=False)
            
            # Apply softmax and get class predictions
            exp_pred = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
            softmax = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
            
            return np.argmax(softmax, axis=1)
        else:
            # If not enough data for sequence, return hold
            return np.ones(1, dtype=int) * 2
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = {
            'input_size': self.input_size,
            'hidden_channels': self.hidden_channels,
            'kernel_size': self.kernel_size,
            'num_layers': self.num_layers,
            'dilations': self.dilations,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'use_skip_connections': self.use_skip_connections,
            'activation': self.activation_name,
            'conv_layers': self.conv_layers,
            'batch_norm_layers': self.batch_norm_layers,
            'skip_connections': self.skip_connections,
            'output_weight': self.output_weight,
            'output_bias': self.output_bias
        }
        return params
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        self.is_fitted = True