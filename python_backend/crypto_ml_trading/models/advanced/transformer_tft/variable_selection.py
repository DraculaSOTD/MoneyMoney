"""
Variable Selection Networks for TFT.

Implements learnable feature selection with interpretability.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class VariableSelectionNetwork:
    """
    Variable Selection Network for automatic feature selection.
    
    Learns which features are most important for prediction at each time step,
    providing interpretability and improved performance.
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 160,
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = True):
        """
        Initialize Variable Selection Network.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Initialize parameters
        self.params = {}
        self._initialize_parameters()
        
        # Batch norm running statistics
        if use_batch_norm:
            self.bn_running_mean = np.zeros((hidden_size,))
            self.bn_running_var = np.ones((hidden_size,))
            self.bn_momentum = 0.1
        
    def _initialize_parameters(self):
        """Initialize network parameters."""
        # Gated Residual Network for context
        self.params['grn_W1'] = self._xavier_init((self.hidden_size, self.input_size))
        self.params['grn_b1'] = np.zeros((self.hidden_size, 1))
        
        self.params['grn_W2'] = self._xavier_init((self.hidden_size, self.hidden_size))
        self.params['grn_b2'] = np.zeros((self.hidden_size, 1))
        
        # Variable weights network
        self.params['var_W'] = self._xavier_init((self.input_size, self.hidden_size))
        self.params['var_b'] = np.zeros((self.input_size, 1))
        
        # Context integration
        self.params['context_W'] = self._xavier_init((self.hidden_size, self.input_size))
        self.params['context_b'] = np.zeros((self.hidden_size, 1))
        
        # Batch normalization parameters
        if self.use_batch_norm:
            self.params['bn_gamma'] = np.ones((self.hidden_size, 1))
            self.params['bn_beta'] = np.zeros((self.hidden_size, 1))
            
    def _xavier_init(self, shape: Tuple[int, int]) -> np.ndarray:
        """Xavier initialization."""
        fan_in, fan_out = shape[1], shape[0]
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    
    def forward(self, inputs: np.ndarray,
                context: Optional[np.ndarray] = None,
                training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through variable selection network.
        
        Args:
            inputs: Input features (batch_size, input_size) or (batch_size, seq_len, input_size)
            context: Optional context vector for selection
            training: Whether in training mode
            
        Returns:
            (selected_features, selection_weights)
        """
        # Handle both 2D and 3D inputs
        original_shape = inputs.shape
        if len(inputs.shape) == 3:
            batch_size, seq_len, input_size = inputs.shape
            inputs_2d = inputs.reshape(-1, input_size)
        else:
            batch_size, input_size = inputs.shape
            seq_len = 1
            inputs_2d = inputs
            
        # Gated Residual Network for context generation
        if context is None:
            context = self._generate_context(inputs_2d.T, training)
        
        # Generate variable weights
        weights = self._generate_variable_weights(context, training)
        
        # Apply variable selection
        selected_features = self._apply_selection(inputs_2d.T, weights, context)
        
        # Reshape back to original format
        if len(original_shape) == 3:
            selected_features = selected_features.T.reshape(batch_size, seq_len, self.hidden_size)
            weights = weights.T.reshape(batch_size, seq_len, self.input_size)
        else:
            selected_features = selected_features.T
            weights = weights.T
            
        return selected_features, weights
    
    def _generate_context(self, inputs: np.ndarray, training: bool) -> np.ndarray:
        """Generate context vector using GRN."""
        # First GRN layer
        hidden1 = self._elu(self.params['grn_W1'] @ inputs + self.params['grn_b1'])
        
        # Apply batch normalization
        if self.use_batch_norm:
            hidden1 = self._batch_norm(hidden1, training)
            
        # Apply dropout
        if training and self.dropout_rate > 0:
            dropout_mask = np.random.binomial(
                1, 1 - self.dropout_rate, hidden1.shape
            ) / (1 - self.dropout_rate)
            hidden1 = hidden1 * dropout_mask
            
        # Second GRN layer with gating
        hidden2 = self.params['grn_W2'] @ hidden1 + self.params['grn_b2']
        gate = self._sigmoid(hidden2)
        
        # Gated residual connection
        # Project input to hidden size for residual
        input_proj = self.params['context_W'] @ inputs + self.params['context_b']\n        context = gate * hidden1 + (1 - gate) * input_proj
        
        return context
    
    def _generate_variable_weights(self, context: np.ndarray, training: bool) -> np.ndarray:
        """Generate variable selection weights."""
        # Linear transformation
        logits = self.params['var_W'] @ context + self.params['var_b']
        
        # Apply softmax for selection weights
        weights = self._softmax(logits)
        
        return weights
    
    def _apply_selection(self, inputs: np.ndarray, weights: np.ndarray,
                        context: np.ndarray) -> np.ndarray:
        """Apply variable selection to inputs."""
        # Weighted input features
        weighted_inputs = inputs * weights
        
        # Combine with context through linear transformation
        selected = self.params['context_W'] @ weighted_inputs + self.params['context_b']
        
        # Add residual connection from context
        selected = selected + context
        
        return selected
    
    def _batch_norm(self, x: np.ndarray, training: bool) -> np.ndarray:
        """Batch normalization."""
        if training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=1, keepdims=True)
            batch_var = np.var(x, axis=1, keepdims=True)
            
            # Update running statistics
            self.bn_running_mean = (1 - self.bn_momentum) * self.bn_running_mean + \
                                  self.bn_momentum * batch_mean.flatten()
            self.bn_running_var = (1 - self.bn_momentum) * self.bn_running_var + \
                                 self.bn_momentum * batch_var.flatten()
            
            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + 1e-5)
        else:
            # Use running statistics
            running_mean = self.bn_running_mean.reshape(-1, 1)
            running_var = self.bn_running_var.reshape(-1, 1)
            x_norm = (x - running_mean) / np.sqrt(running_var + 1e-5)
            
        # Scale and shift
        return self.params['bn_gamma'] * x_norm + self.params['bn_beta']
    
    def _elu(self, x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """ELU activation function."""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def get_feature_importance(self, inputs: np.ndarray,
                             feature_names: Optional[List[str]] = None) -> Dict:
        """
        Get feature importance scores.
        
        Args:
            inputs: Input features
            feature_names: Optional names for features
            
        Returns:
            Feature importance analysis
        """
        # Get selection weights
        _, weights = self.forward(inputs, training=False)
        
        # Average weights across batch and time
        if len(weights.shape) == 3:
            avg_weights = np.mean(weights, axis=(0, 1))
        else:
            avg_weights = np.mean(weights, axis=0)
            
        # Create importance dictionary
        importance = {
            'feature_weights': avg_weights,
            'top_features': np.argsort(avg_weights)[::-1],
            'importance_scores': avg_weights / np.sum(avg_weights)
        }
        
        if feature_names:
            importance['feature_names'] = feature_names
            importance['ranked_features'] = [
                feature_names[i] for i in importance['top_features']
            ]
            
        return importance
    
    def analyze_temporal_importance(self, inputs: np.ndarray,
                                  feature_names: Optional[List[str]] = None) -> Dict:
        """
        Analyze how feature importance changes over time.
        
        Args:
            inputs: Input features (batch_size, seq_len, input_size)
            feature_names: Optional feature names
            
        Returns:
            Temporal importance analysis
        """
        if len(inputs.shape) != 3:
            raise ValueError("Temporal analysis requires 3D input (batch, seq_len, features)")
            
        # Get selection weights
        _, weights = self.forward(inputs, training=False)
        
        # Average across batch dimension
        temporal_weights = np.mean(weights, axis=0)  # (seq_len, input_size)
        
        # Calculate statistics
        importance_trend = np.mean(temporal_weights, axis=1)  # Overall importance per timestep
        feature_stability = np.std(temporal_weights, axis=0)  # Stability per feature
        
        analysis = {
            'temporal_weights': temporal_weights,
            'importance_trend': importance_trend,
            'feature_stability': feature_stability,
            'most_stable_features': np.argsort(feature_stability),
            'most_variable_features': np.argsort(feature_stability)[::-1]
        }
        
        if feature_names:
            analysis['feature_names'] = feature_names
            analysis['stable_feature_names'] = [
                feature_names[i] for i in analysis['most_stable_features'][:5]
            ]
            analysis['variable_feature_names'] = [
                feature_names[i] for i in analysis['most_variable_features'][:5]
            ]
            
        return analysis


class StaticVariableSelection:
    """
    Variable selection for static (non-temporal) features.
    
    Used for features that don't change over time but influence the entire sequence.
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 160,
                 num_selections: int = 10):
        """
        Initialize static variable selection.
        
        Args:
            input_size: Number of static features
            hidden_size: Hidden representation size
            num_selections: Number of top features to select
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_selections = min(num_selections, input_size)
        
        # Initialize parameters
        self.params = {}
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize parameters."""
        # Feature embedding
        self.params['embed_W'] = self._xavier_init((self.hidden_size, self.input_size))
        self.params['embed_b'] = np.zeros((self.hidden_size, 1))
        
        # Selection network
        self.params['select_W'] = self._xavier_init((self.input_size, self.hidden_size))
        self.params['select_b'] = np.zeros((self.input_size, 1))
        
        # Context generation
        self.params['context_W'] = self._xavier_init((self.hidden_size, self.hidden_size))
        self.params['context_b'] = np.zeros((self.hidden_size, 1))
        
    def _xavier_init(self, shape: Tuple[int, int]) -> np.ndarray:
        """Xavier initialization."""
        fan_in, fan_out = shape[1], shape[0]
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
        
    def forward(self, static_inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for static variable selection.
        
        Args:
            static_inputs: Static features (batch_size, input_size)
            
        Returns:
            (selected_context, selection_weights)
        """
        # Embed static features
        embedded = self._elu(
            self.params['embed_W'] @ static_inputs.T + self.params['embed_b']
        )
        
        # Generate selection weights
        selection_logits = self.params['select_W'] @ embedded + self.params['select_b']
        selection_weights = self._softmax(selection_logits)
        
        # Apply selection
        selected_features = static_inputs.T * selection_weights
        
        # Generate context vector
        context = self._elu(
            self.params['context_W'] @ embedded + self.params['context_b']
        )
        
        # Combine selected features with context
        enriched_context = context + self.params['embed_W'] @ selected_features
        
        return enriched_context.T, selection_weights.T
    
    def _elu(self, x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """ELU activation."""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation."""
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def get_selected_features(self, static_inputs: np.ndarray,
                            feature_names: Optional[List[str]] = None) -> Dict:
        """Get top selected features."""
        _, weights = self.forward(static_inputs)
        
        # Average across batch
        avg_weights = np.mean(weights, axis=0)
        
        # Get top features
        top_indices = np.argsort(avg_weights)[::-1][:self.num_selections]
        
        result = {
            'top_feature_indices': top_indices,
            'top_feature_weights': avg_weights[top_indices],
            'all_weights': avg_weights
        }
        
        if feature_names:
            result['top_feature_names'] = [feature_names[i] for i in top_indices]
            
        return result