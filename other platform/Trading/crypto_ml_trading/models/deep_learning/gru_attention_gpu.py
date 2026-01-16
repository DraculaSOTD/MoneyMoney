"""
GPU-Accelerated GRU with Multi-Head Attention for Time Series.

PyTorch implementation of the custom GRU-Attention model with full GPU support,
mixed precision training, and optimized operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
import time
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.gpu_manager import GPUManager, get_gpu_manager
from utils.logging_system import get_logger

logger = get_logger(__name__)


class MultiHeadAttentionGPU(nn.Module):
    """
    GPU-optimized Multi-Head Attention mechanism.
    
    Features:
    - Efficient matrix operations on GPU
    - Mixed precision support
    - Memory-efficient attention computation
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize Multi-Head Attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttentionGPU, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)
            value: Value tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor and attention weights
        """
        batch_size, seq_len, _ = query.size()
        
        # Linear projections in batch from d_model => h x d_k
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear projection
        output = self.W_o(context)
        
        return output, attention_weights


class GRUCellGPU(nn.Module):
    """GPU-optimized GRU cell with custom implementation."""
    
    def __init__(self, input_size: int, hidden_size: int):
        """
        Initialize GRU cell.
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden state dimension
        """
        super(GRUCellGPU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gates
        self.W_ir = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.W_iz = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.W_in = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hn = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GRU cell.
        
        Args:
            x: Input tensor (batch_size, input_size)
            hidden: Hidden state (batch_size, hidden_size)
            
        Returns:
            New hidden state
        """
        # Reset gate
        r = torch.sigmoid(self.W_ir(x) + self.W_hr(hidden))
        
        # Update gate
        z = torch.sigmoid(self.W_iz(x) + self.W_hz(hidden))
        
        # New gate
        n = torch.tanh(self.W_in(x) + r * self.W_hn(hidden))
        
        # Update hidden state
        hidden_new = (1 - z) * n + z * hidden
        
        return hidden_new


class GRUAttentionGPU(nn.Module):
    """
    GPU-accelerated GRU with Multi-Head Attention.
    
    Features:
    - Fully GPU-optimized operations
    - Mixed precision training support
    - Efficient memory usage
    - Gradient checkpointing option
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 output_dim: int = 1,
                 dropout: float = 0.2,
                 bidirectional: bool = True,
                 use_layer_norm: bool = True,
                 gradient_checkpointing: bool = False):
        """
        Initialize GRU-Attention model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            n_heads: Number of attention heads
            n_layers: Number of GRU layers
            output_dim: Output dimension
            dropout: Dropout rate
            bidirectional: Use bidirectional GRU
            use_layer_norm: Apply layer normalization
            gradient_checkpointing: Enable gradient checkpointing
        """
        super(GRUAttentionGPU, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.use_layer_norm = use_layer_norm
        self.gradient_checkpointing = gradient_checkpointing
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GRU layers
        self.gru_layers = nn.ModuleList()
        for i in range(n_layers):
            layer_input_dim = hidden_dim if i == 0 else hidden_dim * (2 if bidirectional else 1)
            self.gru_layers.append(
                nn.GRU(
                    layer_input_dim,
                    hidden_dim,
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=dropout if i < n_layers - 1 else 0
                )
            )
        
        # Attention layers
        attention_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention_layers = nn.ModuleList([
            MultiHeadAttentionGPU(attention_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(attention_dim) for _ in range(n_layers * 2)
            ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(attention_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # GPU manager
        self.gpu_manager = get_gpu_manager()
        
        # Move to GPU
        self.to(self.gpu_manager.device)
        
        logger.info(f"GRU-Attention model initialized on {self.gpu_manager.device}")
    
    def forward(self, x: torch.Tensor, 
                hidden_states: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass of GRU-Attention model.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            hidden_states: Optional initial hidden states
            
        Returns:
            Output predictions
        """
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = self._init_hidden_states(batch_size)
        
        # Process through GRU and attention layers
        for i in range(self.n_layers):
            # GRU layer
            if self.gradient_checkpointing and self.training:
                x, hidden = torch.utils.checkpoint.checkpoint(
                    self._gru_forward, x, hidden_states[i], i
                )
            else:
                x, hidden = self._gru_forward(x, hidden_states[i], i)
            
            hidden_states[i] = hidden
            
            # Layer norm after GRU
            if self.use_layer_norm:
                x = self.layer_norms[i * 2](x)
            
            # Attention layer
            residual = x
            if self.gradient_checkpointing and self.training:
                attn_output, _ = torch.utils.checkpoint.checkpoint(
                    self.attention_layers[i], x, x, x
                )
            else:
                attn_output, _ = self.attention_layers[i](x, x, x)
            
            x = residual + self.dropout(attn_output)
            
            # Layer norm after attention
            if self.use_layer_norm:
                x = self.layer_norms[i * 2 + 1](x)
        
        # Output projection
        # Use last time step for prediction
        output = self.output_projection(x[:, -1, :])
        
        return output
    
    def _gru_forward(self, x: torch.Tensor, hidden: torch.Tensor, 
                     layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through GRU layer."""
        output, hidden = self.gru_layers[layer_idx](x, hidden)
        return output, hidden
    
    def _init_hidden_states(self, batch_size: int) -> List[torch.Tensor]:
        """Initialize hidden states for all layers."""
        hidden_states = []
        
        for i in range(self.n_layers):
            num_directions = 2 if self.bidirectional else 1
            hidden = torch.zeros(
                num_directions, batch_size, self.hidden_dim,
                device=self.gpu_manager.device
            )
            hidden_states.append(hidden)
        
        return hidden_states
    
    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            x: Input data
            
        Returns:
            Predictions as numpy array
        """
        self.eval()
        
        # Convert to tensor and move to GPU
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        x = self.gpu_manager.to_device(x)
        
        # Add batch dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            with self.gpu_manager.autocast():
                output = self.forward(x)
        
        # Convert to numpy
        return self.gpu_manager.to_numpy(output)
    
    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get attention weights for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            List of attention weights for each layer
        """
        self.eval()
        attention_weights = []
        
        with torch.no_grad():
            # Input projection
            x = self.input_projection(x)
            
            # Process through layers
            for i in range(self.n_layers):
                # GRU layer
                x, _ = self.gru_layers[i](x)
                
                # Layer norm
                if self.use_layer_norm:
                    x = self.layer_norms[i * 2](x)
                
                # Attention layer
                residual = x
                attn_output, attn_weights = self.attention_layers[i](x, x, x)
                x = residual + attn_output
                
                # Layer norm
                if self.use_layer_norm:
                    x = self.layer_norms[i * 2 + 1](x)
                
                attention_weights.append(attn_weights)
        
        return attention_weights
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def profile_performance(self, input_shape: Tuple[int, int, int], 
                          num_runs: int = 100) -> Dict[str, float]:
        """
        Profile model performance on GPU.
        
        Args:
            input_shape: Input shape (batch_size, seq_len, input_dim)
            num_runs: Number of runs for profiling
            
        Returns:
            Performance metrics
        """
        self.eval()
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape, device=self.gpu_manager.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.forward(dummy_input)
        
        self.gpu_manager.synchronize()
        
        # Time forward passes
        forward_times = []
        memory_usage = []
        
        for _ in range(num_runs):
            self.gpu_manager.clear_cache()
            start_memory = self.gpu_manager.get_memory_used()
            
            start_time = time.time()
            with torch.no_grad():
                with self.gpu_manager.autocast():
                    _ = self.forward(dummy_input)
            self.gpu_manager.synchronize()
            end_time = time.time()
            
            forward_times.append(end_time - start_time)
            memory_usage.append(self.gpu_manager.get_memory_used() - start_memory)
        
        return {
            "mean_forward_time": np.mean(forward_times),
            "std_forward_time": np.std(forward_times),
            "mean_memory_usage": np.mean(memory_usage),
            "throughput": input_shape[0] / np.mean(forward_times),
            "parameters": self.count_parameters(),
            "device": str(self.gpu_manager.device)
        }


def create_gru_attention_gpu(config: Dict[str, Any]) -> GRUAttentionGPU:
    """
    Factory function to create GRU-Attention model.
    
    Args:
        config: Model configuration
        
    Returns:
        GRU-Attention model
    """
    model = GRUAttentionGPU(
        input_dim=config.get('input_dim', 10),
        hidden_dim=config.get('hidden_dim', 128),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 3),
        output_dim=config.get('output_dim', 1),
        dropout=config.get('dropout', 0.2),
        bidirectional=config.get('bidirectional', True),
        use_layer_norm=config.get('use_layer_norm', True),
        gradient_checkpointing=config.get('gradient_checkpointing', False)
    )
    
    # Profile model
    profile = model.profile_performance((32, 60, config.get('input_dim', 10)))
    logger.info(f"Model profile: {profile}")
    
    return model