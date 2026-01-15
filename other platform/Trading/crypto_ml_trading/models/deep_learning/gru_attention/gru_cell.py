import numpy as np
from typing import Tuple, Optional, Dict, Union
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from utils.matrix_operations import MatrixOperations


class GRUCell:
    """
    Custom GRU (Gated Recurrent Unit) cell implementation from scratch.
    
    GRU equations:
    - Update gate: z_t = σ(W_z·[h_{t-1}, x_t])
    - Reset gate: r_t = σ(W_r·[h_{t-1}, x_t])
    - Candidate state: h̃_t = tanh(W·[r_t * h_{t-1}, x_t])
    - Hidden state: h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
    
    Optimized for cryptocurrency time series with:
    - Gradient clipping for stability
    - Layer normalization
    - Dropout for regularization
    """
    
    def __init__(self, input_size: int, hidden_size: int,
                 dropout_rate: float = 0.1, use_layer_norm: bool = True):
        """
        Initialize GRU cell.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            dropout_rate: Dropout probability
            use_layer_norm: Whether to use layer normalization
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        
        # Initialize weights
        self._initialize_weights()
        
        # Layer normalization parameters
        if use_layer_norm:
            self.ln_gamma = np.ones((3, hidden_size))
            self.ln_beta = np.zeros((3, hidden_size))
            
        # Training mode flag
        self.training = True
        
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        # Combined input and hidden size
        combined_size = self.input_size + self.hidden_size
        
        # Xavier initialization scale
        scale = np.sqrt(2.0 / (combined_size + self.hidden_size))
        
        # Weights for update gate
        self.W_z = np.random.randn(combined_size, self.hidden_size) * scale
        self.b_z = np.zeros(self.hidden_size)
        
        # Weights for reset gate
        self.W_r = np.random.randn(combined_size, self.hidden_size) * scale
        self.b_r = np.zeros(self.hidden_size)
        
        # Weights for candidate state
        self.W_h = np.random.randn(combined_size, self.hidden_size) * scale
        self.b_h = np.zeros(self.hidden_size)
        
        # Initialize biases for gates to help gradient flow
        self.b_z += 1.0  # Bias update gate to be initially open
        
    def forward(self, x: np.ndarray, h_prev: Optional[np.ndarray] = None,
                return_gates: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Forward pass through GRU cell.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size)
            return_gates: Whether to return gate values
            
        Returns:
            Hidden state or tuple of (hidden_state, gates_dict)
        """
        batch_size = x.shape[0]
        
        # Initialize hidden state if not provided
        if h_prev is None:
            h_prev = np.zeros((batch_size, self.hidden_size))
            
        # Concatenate input and previous hidden state
        combined = np.concatenate([h_prev, x], axis=1)
        
        # Cache for backward pass
        cache = {
            'h_prev': h_prev,
            'x': x,
            'combined': combined
        }
        
        # Update gate
        z_linear = combined @ self.W_z + self.b_z
        cache['z_pre_ln'] = z_linear.copy() if self.use_layer_norm else None
        if self.use_layer_norm:
            z_linear = self._layer_norm(z_linear, 0)
        z = self._sigmoid(z_linear)
        cache['z'] = z
        cache['z_linear'] = z_linear
        
        # Reset gate
        r_linear = combined @ self.W_r + self.b_r
        cache['r_pre_ln'] = r_linear.copy() if self.use_layer_norm else None
        if self.use_layer_norm:
            r_linear = self._layer_norm(r_linear, 1)
        r = self._sigmoid(r_linear)
        cache['r'] = r
        cache['r_linear'] = r_linear
        
        # Candidate hidden state
        combined_reset = np.concatenate([r * h_prev, x], axis=1)
        cache['combined_reset'] = combined_reset
        h_tilde_linear = combined_reset @ self.W_h + self.b_h
        cache['h_pre_ln'] = h_tilde_linear.copy() if self.use_layer_norm else None
        if self.use_layer_norm:
            h_tilde_linear = self._layer_norm(h_tilde_linear, 2)
        h_tilde = self._tanh(h_tilde_linear)
        cache['h_tilde'] = h_tilde
        cache['h_tilde_linear'] = h_tilde_linear
        
        # Apply dropout to candidate state during training
        if self.training and self.dropout_rate > 0:
            h_tilde_dropout, dropout_mask = self._dropout(h_tilde, return_mask=True)
            cache['dropout_mask'] = dropout_mask
            h_tilde = h_tilde_dropout
            
        # New hidden state
        h = (1 - z) * h_prev + z * h_tilde
        
        # Store cache for backward pass
        self._cache = cache
        
        if return_gates:
            gates = {
                'update_gate': z,
                'reset_gate': r,
                'candidate': h_tilde,
                'cache': cache
            }
            return h, gates
        
        return h
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Stable sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation."""
        return np.tanh(x)
    
    def _layer_norm(self, x: np.ndarray, gate_idx: int) -> np.ndarray:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor
            gate_idx: Index for gate-specific parameters (0=update, 1=reset, 2=candidate)
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + 1e-5)
        
        return self.ln_gamma[gate_idx] * x_norm + self.ln_beta[gate_idx]
    
    def _dropout(self, x: np.ndarray, return_mask: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Apply dropout during training."""
        if self.training:
            mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape)
            dropped = x * mask / (1 - self.dropout_rate)
            if return_mask:
                return dropped, mask
            return dropped
        if return_mask:
            return x, np.ones_like(x)
        return x
    
    def backward(self, dh_next: np.ndarray, cache: Dict) -> Dict:
        """
        Backward pass through GRU cell.
        
        Args:
            dh_next: Gradient of loss w.r.t. next hidden state
            cache: Cached values from forward pass
            
        Returns:
            Dictionary with gradients
        """
        # Unpack cache
        h_prev = cache['h_prev']
        x = cache['x']
        z = cache['z']
        r = cache['r']
        h_tilde = cache['h_tilde']
        combined = cache['combined']
        combined_reset = cache['combined_reset']
        
        # Apply dropout mask if it was used in forward pass
        if 'dropout_mask' in cache and self.training:
            dropout_mask = cache['dropout_mask']
            dh_next_for_tilde = dh_next * z
            dh_tilde = dh_next_for_tilde * dropout_mask / (1 - self.dropout_rate)
        else:
            dh_tilde = dh_next * z
            
        # Gradients of hidden state update
        dh_prev = dh_next * (1 - z)
        dz = dh_next * (h_tilde - h_prev)
        
        # Gradient through tanh
        dh_tilde_linear = dh_tilde * (1 - h_tilde**2)
        
        # Store gradients for layer norm update
        cache['dh_tilde_linear'] = dh_tilde_linear
        
        # Gradients for candidate weights
        self.dW_h = combined_reset.T @ dh_tilde_linear
        self.db_h = np.sum(dh_tilde_linear, axis=0)
        
        # Gradient through reset gate
        # Gradient w.r.t the reset gate part of combined_reset
        dcombined_reset = dh_tilde_linear @ self.W_h.T
        # Split gradient: first hidden_size elements are for r * h_prev
        dh_prev_reset = dcombined_reset[:, :self.hidden_size]
        dr = dh_prev_reset * h_prev
        dh_prev += dh_prev_reset * r
        
        # Gradient through sigmoid for reset gate
        dr_linear = dr * r * (1 - r)
        cache['dr_linear'] = dr_linear
        
        # Gradients for reset gate weights
        self.dW_r = combined.T @ dr_linear
        self.db_r = np.sum(dr_linear, axis=0)
        
        # Gradient through sigmoid for update gate
        dz_linear = dz * z * (1 - z)
        cache['dz_linear'] = dz_linear
        
        # Gradients for update gate weights
        self.dW_z = combined.T @ dz_linear
        self.db_z = np.sum(dz_linear, axis=0)
        
        # Gradient w.r.t. input
        # The weight matrices have shape (input_size + hidden_size, hidden_size)
        # We need the part corresponding to input: W[hidden_size:, :]
        dx = dz_linear @ self.W_z[self.hidden_size:, :].T
        dx += dr_linear @ self.W_r[self.hidden_size:, :].T
        dx += dh_tilde_linear @ self.W_h[self.hidden_size:, :].T
        
        # Update layer norm parameters if used
        if self.use_layer_norm:
            self._update_layer_norm_grads(cache)
            
        return {
            'dx': dx,
            'dh_prev': dh_prev,
            'dW_z': self.dW_z,
            'db_z': self.db_z,
            'dW_r': self.dW_r,
            'db_r': self.db_r,
            'dW_h': self.dW_h,
            'db_h': self.db_h
        }
    
    def _update_layer_norm_grads(self, cache: Dict):
        """Update layer normalization gradients."""
        # Retrieve cached values
        z_pre_ln = cache.get('z_pre_ln')
        r_pre_ln = cache.get('r_pre_ln')
        h_pre_ln = cache.get('h_pre_ln')
        
        # Gradients from the forward pass
        dz_linear = cache.get('dz_linear')
        dr_linear = cache.get('dr_linear')
        dh_tilde_linear = cache.get('dh_tilde_linear')
        
        # Update gate layer norm gradients
        if z_pre_ln is not None and dz_linear is not None:
            mean = np.mean(z_pre_ln, axis=-1, keepdims=True)
            var = np.var(z_pre_ln, axis=-1, keepdims=True)
            std = np.sqrt(var + 1e-5)
            x_norm = (z_pre_ln - mean) / std
            
            # Gradient w.r.t gamma
            self.dln_gamma = np.zeros_like(self.ln_gamma) if not hasattr(self, 'dln_gamma') else self.dln_gamma
            self.dln_gamma[0] += np.sum(dz_linear * x_norm, axis=0)
            
            # Gradient w.r.t beta
            self.dln_beta = np.zeros_like(self.ln_beta) if not hasattr(self, 'dln_beta') else self.dln_beta
            self.dln_beta[0] += np.sum(dz_linear, axis=0)
        
        # Reset gate layer norm gradients
        if r_pre_ln is not None and dr_linear is not None:
            mean = np.mean(r_pre_ln, axis=-1, keepdims=True)
            var = np.var(r_pre_ln, axis=-1, keepdims=True)
            std = np.sqrt(var + 1e-5)
            x_norm = (r_pre_ln - mean) / std
            
            self.dln_gamma[1] += np.sum(dr_linear * x_norm, axis=0)
            self.dln_beta[1] += np.sum(dr_linear, axis=0)
        
        # Candidate state layer norm gradients
        if h_pre_ln is not None and dh_tilde_linear is not None:
            mean = np.mean(h_pre_ln, axis=-1, keepdims=True)
            var = np.var(h_pre_ln, axis=-1, keepdims=True)
            std = np.sqrt(var + 1e-5)
            x_norm = (h_pre_ln - mean) / std
            
            self.dln_gamma[2] += np.sum(dh_tilde_linear * x_norm, axis=0)
            self.dln_beta[2] += np.sum(dh_tilde_linear, axis=0)
    
    def update_weights(self, gradients: Dict, learning_rate: float,
                      clip_value: float = 5.0):
        """
        Update weights using gradients.
        
        Args:
            gradients: Dictionary of gradients
            learning_rate: Learning rate
            clip_value: Gradient clipping threshold
        """
        # Clip gradients
        for key in ['dW_z', 'dW_r', 'dW_h']:
            if key in gradients:
                gradients[key] = np.clip(gradients[key], -clip_value, clip_value)
                
        # Update weights
        self.W_z -= learning_rate * gradients.get('dW_z', 0)
        self.b_z -= learning_rate * gradients.get('db_z', 0)
        self.W_r -= learning_rate * gradients.get('dW_r', 0)
        self.b_r -= learning_rate * gradients.get('db_r', 0)
        self.W_h -= learning_rate * gradients.get('dW_h', 0)
        self.b_h -= learning_rate * gradients.get('db_h', 0)
        
        # Update layer normalization parameters if used
        if self.use_layer_norm and hasattr(self, 'dln_gamma'):
            self.ln_gamma -= learning_rate * np.clip(self.dln_gamma, -clip_value, clip_value)
            self.ln_beta -= learning_rate * np.clip(self.dln_beta, -clip_value, clip_value)
            # Reset gradients
            self.dln_gamma = np.zeros_like(self.ln_gamma)
            self.dln_beta = np.zeros_like(self.ln_beta)
    
    def set_training(self, training: bool):
        """Set training mode."""
        self.training = training
    
    def get_params(self) -> Dict:
        """Get all parameters."""
        params = {
            'W_z': self.W_z,
            'b_z': self.b_z,
            'W_r': self.W_r,
            'b_r': self.b_r,
            'W_h': self.W_h,
            'b_h': self.b_h
        }
        
        if self.use_layer_norm:
            params['ln_gamma'] = self.ln_gamma
            params['ln_beta'] = self.ln_beta
            
        return params
    
    def set_params(self, params: Dict):
        """Set parameters from dictionary."""
        self.W_z = params['W_z']
        self.b_z = params['b_z']
        self.W_r = params['W_r']
        self.b_r = params['b_r']
        self.W_h = params['W_h']
        self.b_h = params['b_h']
        
        if self.use_layer_norm and 'ln_gamma' in params:
            self.ln_gamma = params['ln_gamma']
            self.ln_beta = params['ln_beta']


class MultiLayerGRU:
    """
    Multi-layer GRU implementation with dropout between layers.
    """
    
    def __init__(self, input_size: int, hidden_sizes: list,
                 dropout_rates: Optional[list] = None,
                 use_layer_norm: bool = True):
        """
        Initialize multi-layer GRU.
        
        Args:
            input_size: Input feature size
            hidden_sizes: List of hidden sizes for each layer
            dropout_rates: Dropout rates for each layer
            use_layer_norm: Whether to use layer normalization
        """
        self.num_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        
        if dropout_rates is None:
            dropout_rates = [0.1] * self.num_layers
            
        # Create GRU cells for each layer
        self.cells = []
        curr_input_size = input_size
        
        for i, (hidden_size, dropout) in enumerate(zip(hidden_sizes, dropout_rates)):
            cell = GRUCell(
                curr_input_size,
                hidden_size,
                dropout_rate=dropout,
                use_layer_norm=use_layer_norm
            )
            self.cells.append(cell)
            curr_input_size = hidden_size
            
    def forward(self, x: np.ndarray, hidden_states: Optional[list] = None) -> Tuple[np.ndarray, list]:
        """
        Forward pass through all layers.

        Args:
            x: Input sequence (batch_size, seq_length, input_size)
            hidden_states: Initial hidden states for each layer

        Returns:
            Tuple of (output, final_hidden_states)
        """
        batch_size, seq_length, _ = x.shape

        if hidden_states is None:
            hidden_states = [None] * self.num_layers

        # Clear caches at start of forward pass to prevent memory accumulation
        for cell in self.cells:
            cell._caches = []

        outputs = []
        layer_outputs = [[] for _ in range(self.num_layers)]
        
        # Process each time step
        for t in range(seq_length):
            x_t = x[:, t, :]
            
            # Pass through each layer
            for layer_idx, cell in enumerate(self.cells):
                h, gates = cell.forward(x_t, hidden_states[layer_idx], return_gates=True)
                hidden_states[layer_idx] = h
                layer_outputs[layer_idx].append(h)
                x_t = h  # Output becomes input to next layer
                
                # Store cache for this timestep (list cleared at start of forward)
                cell._caches.append(gates['cache'])
                
            outputs.append(x_t)
            
        # Stack outputs
        output = np.stack(outputs, axis=1)
        
        # Stack layer outputs for backward pass
        for layer_idx in range(self.num_layers):
            layer_outputs[layer_idx] = np.stack(layer_outputs[layer_idx], axis=1)
        
        self._layer_outputs = layer_outputs
        
        return output, hidden_states
    
    def set_training(self, training: bool):
        """Set training mode for all cells."""
        for cell in self.cells:
            cell.set_training(training)