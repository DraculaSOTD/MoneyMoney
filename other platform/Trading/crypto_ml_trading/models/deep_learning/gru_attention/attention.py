import numpy as np
from typing import Tuple, Optional, Dict, Union
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from utils.matrix_operations import MatrixOperations


class ScaledDotProductAttention:
    """
    Scaled dot-product attention mechanism.
    
    Attention(Q, K, V) = softmax(QK^T / √d_k)V
    
    Optimized for time series with:
    - Causal masking for autoregressive predictions
    - Relative position encoding
    - Attention dropout
    """
    
    def __init__(self, d_model: int, dropout_rate: float = 0.1):
        """
        Initialize attention mechanism.
        
        Args:
            d_model: Dimension of the model
            dropout_rate: Dropout probability for attention weights
        """
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.scale = np.sqrt(d_model)
        self.training = True
        
    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None,
                return_attention: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor (batch_size, seq_len_q, d_model)
            key: Key tensor (batch_size, seq_len_k, d_model)
            value: Value tensor (batch_size, seq_len_v, d_model)
            mask: Attention mask (batch_size, seq_len_q, seq_len_k)
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor or tuple of (output, attention_weights)
        """
        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]
        
        # Compute attention scores
        scores = np.matmul(query, key.transpose(0, 2, 1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + (mask * -1e9)
            
        # Compute attention weights
        attention_weights = self._softmax(scores, axis=-1)
        
        # Apply dropout during training
        if self.training and self.dropout_rate > 0:
            attention_weights = self._dropout(attention_weights)
            
        # Apply attention to values
        output = np.matmul(attention_weights, value)
        
        if return_attention:
            return output, attention_weights
        return output
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Stable softmax implementation."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _dropout(self, x: np.ndarray) -> np.ndarray:
        """Apply dropout to attention weights."""
        if self.training:
            mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape)
            return x * mask / (1 - self.dropout_rate)
        return x
    
    def set_training(self, training: bool):
        """Set training mode."""
        self.training = training
    
    def backward(self, dout: np.ndarray, query: np.ndarray, key: np.ndarray, 
                value: np.ndarray, attention_weights: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass through scaled dot-product attention.
        
        Args:
            dout: Gradient w.r.t output (batch_size, seq_len_q, d_model)
            query: Query tensor from forward pass
            key: Key tensor from forward pass
            value: Value tensor from forward pass
            attention_weights: Attention weights from forward pass
            mask: Attention mask
            
        Returns:
            Tuple of (dquery, dkey, dvalue)
        """
        batch_size, seq_len_q, d_model = query.shape
        seq_len_k = key.shape[1]
        
        # Gradient through matrix multiplication with values
        # dout = attention_weights @ value
        dattention_weights = np.matmul(dout, value.transpose(0, 2, 1))
        dvalue = np.matmul(attention_weights.transpose(0, 2, 1), dout)
        
        # Gradient through softmax
        # attention_weights = softmax(scores)
        dscores = dattention_weights * attention_weights
        dscores -= attention_weights * np.sum(dattention_weights * attention_weights, 
                                              axis=-1, keepdims=True)
        
        # Apply mask gradient if mask was used
        if mask is not None:
            # Mask was applied as scores + (mask * -1e9)
            # The gradient just passes through where mask is 0
            dscores = dscores * (1 - mask)
        
        # Gradient through scaling
        dscores_scaled = dscores / self.scale
        
        # Gradient through matrix multiplication
        # scores = query @ key.T
        dquery = np.matmul(dscores_scaled, key)
        dkey = np.matmul(dscores_scaled.transpose(0, 2, 1), query)
        
        return dquery, dkey, dvalue


class MultiHeadAttention:
    """
    Multi-head attention mechanism for capturing different types of dependencies.
    
    Specifically designed for cryptocurrency time series:
    - Multiple attention heads for different time scales
    - Positional encoding for temporal awareness
    - Residual connections
    """
    
    def __init__(self, d_model: int, num_heads: int = 8,
                 dropout_rate: float = 0.1, use_bias: bool = True):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout probability
            use_bias: Whether to use bias in linear transformations
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        
        # Initialize weights
        self._initialize_weights()
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(self.d_head, dropout_rate)
        
        self.training = True
        
    def _initialize_weights(self):
        """Initialize projection weights."""
        # Xavier initialization
        scale = np.sqrt(2.0 / (self.d_model + self.d_head))
        
        # Query, Key, Value projections for all heads
        self.W_q = np.random.randn(self.d_model, self.d_model) * scale
        self.W_k = np.random.randn(self.d_model, self.d_model) * scale
        self.W_v = np.random.randn(self.d_model, self.d_model) * scale
        
        # Output projection
        self.W_o = np.random.randn(self.d_model, self.d_model) * scale
        
        if self.use_bias:
            self.b_q = np.zeros(self.d_model)
            self.b_k = np.zeros(self.d_model)
            self.b_v = np.zeros(self.d_model)
            self.b_o = np.zeros(self.d_model)
            
    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None,
                return_attention: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Forward pass through multi-head attention.
        
        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor
            value: Value tensor
            mask: Attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor or tuple with attention weights
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # Store inputs for backward pass
        self._forward_cache = {
            'query': query,
            'key': key,
            'value': value,
            'mask': mask
        }
        
        # Linear projections
        Q = self._linear(query, self.W_q, self.b_q if self.use_bias else None)
        K = self._linear(key, self.W_k, self.b_k if self.use_bias else None)
        V = self._linear(value, self.W_v, self.b_v if self.use_bias else None)
        
        # Reshape for multi-head attention
        Q_heads = self._split_heads(Q, batch_size)
        K_heads = self._split_heads(K, batch_size)
        V_heads = self._split_heads(V, batch_size)
        
        # Store projected values
        self._forward_cache['Q'] = Q_heads
        self._forward_cache['K'] = K_heads
        self._forward_cache['V'] = V_heads
        
        # Apply attention
        if mask is not None:
            # Expand mask for heads
            mask = np.expand_dims(mask, axis=1)
        
        # Process each head separately
        attention_outputs = []
        attention_weights_list = []
        
        for h in range(self.num_heads):
            output_h, weights_h = self.attention.forward(
                Q_heads[:, h], K_heads[:, h], V_heads[:, h],
                mask[:, 0] if mask is not None else None,
                return_attention=True
            )
            attention_outputs.append(output_h)
            attention_weights_list.append(weights_h)
        
        # Stack heads
        attention_output = np.stack(attention_outputs, axis=1)
        attention_weights = np.stack(attention_weights_list, axis=1)
        
        # Store attention outputs
        self._forward_cache['attention_output_heads'] = attention_output
        self._forward_cache['attention_weights'] = attention_weights
        
        # Concatenate heads
        attention_output_concat = self._concat_heads(attention_output, batch_size)
        self._forward_cache['attention_output'] = attention_output_concat
        
        # Final linear projection
        output = self._linear(attention_output_concat, self.W_o, 
                            self.b_o if self.use_bias else None)
        
        if return_attention:
            return output, attention_weights
        return output
    
    def _linear(self, x: np.ndarray, weight: np.ndarray,
                bias: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply linear transformation."""
        output = x @ weight
        if bias is not None:
            output += bias
        return output
    
    def _split_heads(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        """Split tensor into multiple heads."""
        seq_len = x.shape[1]
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_head)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq_len, d_head)
    
    def _concat_heads(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        """Concatenate attention heads."""
        x = x.transpose(0, 2, 1, 3)  # (batch, seq_len, heads, d_head)
        seq_len = x.shape[1]
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def set_training(self, training: bool):
        """Set training mode."""
        self.training = training
        self.attention.set_training(training)
        
    def get_params(self) -> Dict:
        """Get all parameters."""
        params = {
            'W_q': self.W_q,
            'W_k': self.W_k,
            'W_v': self.W_v,
            'W_o': self.W_o
        }
        
        if self.use_bias:
            params.update({
                'b_q': self.b_q,
                'b_k': self.b_k,
                'b_v': self.b_v,
                'b_o': self.b_o
            })
            
        return params
    
    def backward(self, dout: np.ndarray, query: np.ndarray, key: np.ndarray,
                value: np.ndarray, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                attention_output_heads: np.ndarray, attention_weights: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass through multi-head attention.
        
        Args:
            dout: Gradient w.r.t output (batch_size, seq_len, d_model)
            query, key, value: Original inputs
            Q, K, V: Projected and split queries, keys, values
            attention_output_heads: Output before concatenation
            attention_weights: Attention weights from forward
            mask: Attention mask
            
        Returns:
            Tuple of (dquery, dkey, dvalue)
        """
        batch_size, seq_len, d_model = query.shape
        
        # Gradient through output projection
        # output = attention_output @ W_o + b_o
        dattention_output = dout @ self.W_o.T
        self.dW_o = attention_output_heads.reshape(-1, d_model).T @ dout.reshape(-1, d_model)
        if self.use_bias:
            self.db_o = np.sum(dout.reshape(-1, d_model), axis=0)
        
        # Reshape back to heads
        dattention_output_heads = dattention_output.reshape(
            batch_size, seq_len, self.num_heads, self.d_head
        ).transpose(0, 2, 1, 3)
        
        # Backward through attention for each head
        dQ_heads = np.zeros_like(Q)
        dK_heads = np.zeros_like(K)
        dV_heads = np.zeros_like(V)
        
        for h in range(self.num_heads):
            dQ_h, dK_h, dV_h = self.attention.backward(
                dattention_output_heads[:, h],
                Q[:, h], K[:, h], V[:, h],
                attention_weights[:, h],
                mask[:, 0] if mask is not None else None
            )
            dQ_heads[:, h] = dQ_h
            dK_heads[:, h] = dK_h
            dV_heads[:, h] = dV_h
        
        # Concatenate head gradients
        dQ = self._concat_heads(dQ_heads, batch_size)
        dK = self._concat_heads(dK_heads, batch_size)
        dV = self._concat_heads(dV_heads, batch_size)
        
        # Backward through linear projections
        dquery = dQ @ self.W_q.T
        dkey = dK @ self.W_k.T
        dvalue = dV @ self.W_v.T
        
        # Compute weight gradients
        self.dW_q = query.reshape(-1, d_model).T @ dQ.reshape(-1, d_model)
        self.dW_k = key.reshape(-1, d_model).T @ dK.reshape(-1, d_model)
        self.dW_v = value.reshape(-1, d_model).T @ dV.reshape(-1, d_model)
        
        if self.use_bias:
            self.db_q = np.sum(dQ.reshape(-1, d_model), axis=0)
            self.db_k = np.sum(dK.reshape(-1, d_model), axis=0)
            self.db_v = np.sum(dV.reshape(-1, d_model), axis=0)

        # Clear forward cache to free memory after backward pass completes
        self._forward_cache = {}

        return dquery, dkey, dvalue


class TemporalAttention:
    """
    Temporal attention mechanism specifically for time series.
    
    Features:
    - Causal masking for future information
    - Exponential decay for distant time steps
    - Relative position encoding
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 1000,
                 decay_rate: float = 0.01):
        """
        Initialize temporal attention.
        
        Args:
            d_model: Model dimension
            max_seq_length: Maximum sequence length
            decay_rate: Exponential decay rate for temporal distance
        """
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.decay_rate = decay_rate
        
        # Initialize relative position embeddings
        self._init_position_embeddings()
        
    def _init_position_embeddings(self):
        """Initialize relative position embeddings."""
        # Create position encoding matrix
        self.pos_embedding = np.zeros((self.max_seq_length, self.d_model))
        
        for pos in range(self.max_seq_length):
            for i in range(0, self.d_model, 2):
                self.pos_embedding[pos, i] = np.sin(pos / (10000 ** (i / self.d_model)))
                if i + 1 < self.d_model:
                    self.pos_embedding[pos, i + 1] = np.cos(pos / (10000 ** (i / self.d_model)))
                    
    def create_causal_mask(self, seq_length: int) -> np.ndarray:
        """
        Create causal mask to prevent attending to future positions.
        
        Args:
            seq_length: Length of sequence
            
        Returns:
            Causal mask (seq_length, seq_length)
        """
        mask = np.triu(np.ones((seq_length, seq_length)), k=1)
        return mask
    
    def create_temporal_decay_mask(self, seq_length: int) -> np.ndarray:
        """
        Create temporal decay mask based on time distance.
        
        Args:
            seq_length: Length of sequence
            
        Returns:
            Decay mask (seq_length, seq_length)
        """
        positions = np.arange(seq_length)
        distance_matrix = np.abs(positions[:, None] - positions[None, :])
        decay_mask = np.exp(-self.decay_rate * distance_matrix)
        
        return decay_mask
    
    def add_positional_encoding(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (batch_size, seq_length, d_model)
            
        Returns:
            Input with positional encoding added
        """
        seq_length = x.shape[1]
        d_model = x.shape[2]
        
        # Only add positional encoding if dimensions match
        if d_model == self.d_model:
            return x + self.pos_embedding[:seq_length, :]
        else:
            # If dimensions don't match, return input unchanged
            # This can happen when input size differs from model dimension
            return x


class LocalAttention:
    """
    Local attention mechanism that focuses on nearby time steps.
    
    Efficient for long sequences by limiting attention window.
    """
    
    def __init__(self, d_model: int, window_size: int = 10,
                 num_heads: int = 4):
        """
        Initialize local attention.
        
        Args:
            d_model: Model dimension
            window_size: Size of local attention window
            num_heads: Number of attention heads
        """
        self.d_model = d_model
        self.window_size = window_size
        self.num_heads = num_heads
        
        # Multi-head attention for local window
        self.attention = MultiHeadAttention(d_model, num_heads)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply local attention.
        
        Args:
            x: Input tensor (batch_size, seq_length, d_model)
            
        Returns:
            Output with local attention applied
        """
        batch_size, seq_length, d_model = x.shape
        output = np.zeros_like(x)
        
        # Apply attention to each local window
        for i in range(seq_length):
            # Define window boundaries
            start = max(0, i - self.window_size // 2)
            end = min(seq_length, i + self.window_size // 2 + 1)
            
            # Extract local window
            local_x = x[:, start:end, :]
            
            # Apply attention within window
            local_output = self.attention.forward(
                local_x[:, -1:, :],  # Query is current position
                local_x,              # Keys are window
                local_x               # Values are window
            )
            
            output[:, i, :] = local_output[:, 0, :]
            
        return output