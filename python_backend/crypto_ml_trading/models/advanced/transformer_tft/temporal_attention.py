"""
Temporal Self-Attention Module for TFT.

Implements multi-head temporal attention with interpretability features.
"""

import numpy as np
from typing import Tuple, Optional


class TemporalSelfAttention:
    """
    Multi-head temporal self-attention for time series data.
    
    Features:
    - Scaled dot-product attention
    - Multi-head attention with different representation subspaces
    - Temporal position encoding
    - Causal masking for autoregressive prediction
    """
    
    def __init__(self,
                 hidden_size: int = 160,
                 num_heads: int = 4,
                 dropout_rate: float = 0.1,
                 use_causal_mask: bool = True):
        """
        Initialize temporal attention.
        
        Args:
            hidden_size: Size of hidden representations
            num_heads: Number of attention heads
            dropout_rate: Dropout rate for regularization
            use_causal_mask: Whether to apply causal masking
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_causal_mask = use_causal_mask
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_dim = hidden_size // num_heads
        
        # Initialize parameters
        self.params = {}
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize attention parameters."""
        # Query, Key, Value projection matrices
        self.params['Wq'] = self._xavier_init((self.hidden_size, self.hidden_size))
        self.params['Wk'] = self._xavier_init((self.hidden_size, self.hidden_size))
        self.params['Wv'] = self._xavier_init((self.hidden_size, self.hidden_size))
        
        # Output projection
        self.params['Wo'] = self._xavier_init((self.hidden_size, self.hidden_size))
        
        # Layer normalization
        self.params['ln_gamma'] = np.ones((self.hidden_size, 1))
        self.params['ln_beta'] = np.zeros((self.hidden_size, 1))
        
        # Position encoding parameters
        self.params['position_embedding'] = self._init_position_encoding()
        
    def _xavier_init(self, shape: Tuple[int, int]) -> np.ndarray:
        """Xavier initialization."""
        fan_in, fan_out = shape[1], shape[0]
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    
    def _init_position_encoding(self, max_len: int = 1000) -> np.ndarray:
        """Initialize sinusoidal position encoding."""
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.hidden_size, 2) * 
                         -(np.log(10000.0) / self.hidden_size))
        
        pos_encoding = np.zeros((max_len, self.hidden_size))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return pos_encoding
    
    def forward(self, 
                query: np.ndarray,
                key: Optional[np.ndarray] = None,
                value: Optional[np.ndarray] = None,
                mask: Optional[np.ndarray] = None,
                return_attention: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Forward pass through temporal attention.
        
        Args:
            query: Query tensor (batch_size, seq_len, hidden_size)
            key: Key tensor (batch_size, seq_len, hidden_size), defaults to query
            value: Value tensor (batch_size, seq_len, hidden_size), defaults to query
            mask: Attention mask (batch_size, seq_len, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            (output, attention_weights)
        """
        batch_size, seq_len, _ = query.shape
        
        # Default to self-attention
        if key is None:
            key = query
        if value is None:
            value = query
            
        # Add position encoding
        query = self._add_position_encoding(query)
        key = self._add_position_encoding(key)
        
        # Project to Q, K, V
        Q = query @ self.params['Wq'].T  # (batch, seq_len, hidden)
        K = key @ self.params['Wk'].T    # (batch, seq_len, hidden)
        V = value @ self.params['Wv'].T  # (batch, seq_len, hidden)
        
        # Reshape for multi-head attention
        Q = self._reshape_for_multihead(Q)  # (batch, num_heads, seq_len, head_dim)
        K = self._reshape_for_multihead(K)
        V = self._reshape_for_multihead(V)
        
        # Scaled dot-product attention
        attention_output, attention_weights = self._scaled_dot_product_attention(
            Q, K, V, mask
        )
        
        # Concatenate heads
        attention_output = self._concatenate_heads(attention_output)
        
        # Output projection
        output = attention_output @ self.params['Wo'].T
        
        # Residual connection and layer norm
        output = self._layer_norm(output + query)
        
        if return_attention:
            return output, attention_weights
        else:
            return output, None
    
    def _add_position_encoding(self, x: np.ndarray) -> np.ndarray:
        """Add positional encoding to input."""
        batch_size, seq_len, hidden_size = x.shape
        
        # Get position encodings for this sequence length
        pos_enc = self.params['position_embedding'][:seq_len, :]
        
        # Add to input (broadcast across batch dimension)
        return x + pos_enc[np.newaxis, :, :]
    
    def _reshape_for_multihead(self, x: np.ndarray) -> np.ndarray:
        """Reshape tensor for multi-head attention."""
        batch_size, seq_len, hidden_size = x.shape
        
        # Reshape to (batch, seq_len, num_heads, head_dim)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch, num_heads, seq_len, head_dim)
        return x.transpose(0, 2, 1, 3)
    
    def _concatenate_heads(self, x: np.ndarray) -> np.ndarray:
        """Concatenate attention heads."""
        batch_size, num_heads, seq_len, head_dim = x.shape
        
        # Transpose to (batch, seq_len, num_heads, head_dim)
        x = x.transpose(0, 2, 1, 3)
        
        # Reshape to (batch, seq_len, hidden_size)
        return x.reshape(batch_size, seq_len, self.hidden_size)
    
    def _scaled_dot_product_attention(self,
                                    Q: np.ndarray,
                                    K: np.ndarray,
                                    V: np.ndarray,
                                    mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Scaled dot-product attention computation."""
        batch_size, num_heads, seq_len, head_dim = Q.shape
        
        # Compute attention scores
        scores = np.zeros((batch_size, num_heads, seq_len, seq_len))
        
        for b in range(batch_size):
            for h in range(num_heads):
                scores[b, h] = Q[b, h] @ K[b, h].T / np.sqrt(head_dim)
        
        # Apply causal mask if enabled
        if self.use_causal_mask:
            causal_mask = self._create_causal_mask(seq_len)
            scores = np.where(causal_mask, scores, -np.inf)
        
        # Apply additional mask if provided
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        
        # Apply softmax
        attention_weights = np.zeros_like(scores)
        for b in range(batch_size):
            for h in range(num_heads):
                attention_weights[b, h] = self._softmax(scores[b, h])
        
        # Apply dropout if training
        if hasattr(self, 'training') and self.training and self.dropout_rate > 0:
            dropout_mask = np.random.binomial(
                1, 1 - self.dropout_rate, attention_weights.shape
            ) / (1 - self.dropout_rate)
            attention_weights = attention_weights * dropout_mask
        
        # Apply attention to values
        output = np.zeros((batch_size, num_heads, seq_len, head_dim))
        for b in range(batch_size):
            for h in range(num_heads):
                output[b, h] = attention_weights[b, h] @ V[b, h]
        
        return output, attention_weights
    
    def _create_causal_mask(self, seq_len: int) -> np.ndarray:
        """Create causal mask for autoregressive prediction."""
        mask = np.tril(np.ones((seq_len, seq_len)))
        return mask.astype(bool)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax with numerical stability."""
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        x_norm = (x - mean) / np.sqrt(var + 1e-5)
        
        # Apply learnable parameters
        gamma = self.params['ln_gamma'].T
        beta = self.params['ln_beta'].T
        
        return x_norm * gamma + beta
    
    def get_attention_patterns(self,
                             query: np.ndarray,
                             key: Optional[np.ndarray] = None,
                             value: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Extract attention patterns for interpretability.
        
        Returns attention weights, head-specific patterns, and aggregated patterns.
        """
        _, attention_weights = self.forward(query, key, value, return_attention=True)
        
        # Average across batch and heads for overall pattern
        avg_attention = np.mean(attention_weights, axis=(0, 1))
        
        # Head-specific patterns
        head_patterns = np.mean(attention_weights, axis=0)
        
        # Temporal importance (how much each position attends to past)
        temporal_importance = np.mean(attention_weights, axis=(0, 1, 3))
        
        return {
            'attention_weights': attention_weights,
            'average_attention': avg_attention,
            'head_patterns': head_patterns,
            'temporal_importance': temporal_importance
        }
    
    def visualize_attention(self, attention_weights: np.ndarray,
                          timestamps: Optional[List] = None) -> Dict:
        """
        Prepare attention weights for visualization.
        
        Args:
            attention_weights: Attention weights (batch, heads, seq_len, seq_len)
            timestamps: Optional timestamps for labeling
            
        Returns:
            Dictionary with visualization data
        """
        # Average across batch and heads
        avg_attention = np.mean(attention_weights, axis=(0, 1))
        
        # Find most attended positions
        max_attention_per_query = np.argmax(avg_attention, axis=1)
        
        # Calculate attention entropy (diversity)
        attention_entropy = -np.sum(
            avg_attention * np.log(avg_attention + 1e-10), axis=1
        )
        
        visualization_data = {
            'attention_matrix': avg_attention,
            'max_attention_positions': max_attention_per_query,
            'attention_entropy': attention_entropy,
            'head_diversity': np.std(attention_weights, axis=1)
        }
        
        if timestamps:
            visualization_data['timestamps'] = timestamps
            
        return visualization_data