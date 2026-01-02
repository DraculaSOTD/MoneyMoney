"""
Temporal Fusion Transformer (TFT) Model Implementation.

This module implements the core TFT architecture for multi-horizon time series forecasting
with interpretability features.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


class TemporalFusionTransformer:
    """
    Temporal Fusion Transformer for cryptocurrency price forecasting.
    
    Architecture components:
    - Variable Selection Networks for static and temporal features
    - LSTM encoder-decoder for temporal processing
    - Multi-head temporal self-attention
    - Gated Residual Networks (GRN)
    - Quantile output for probabilistic forecasting
    """
    
    def __init__(self,
                 n_encoder_steps: int = 168,  # 7 days of hourly data
                 n_prediction_steps: int = 24,  # 24 hour forecast
                 n_features: int = 10,
                 n_static_features: int = 5,
                 hidden_size: int = 160,
                 lstm_layers: int = 2,
                 num_attention_heads: int = 4,
                 dropout_rate: float = 0.1,
                 quantiles: List[float] = None):
        """
        Initialize TFT model.
        
        Args:
            n_encoder_steps: Number of past time steps
            n_prediction_steps: Number of future time steps to predict
            n_features: Number of time-varying features
            n_static_features: Number of static features
            hidden_size: Hidden layer size
            lstm_layers: Number of LSTM layers
            num_attention_heads: Number of attention heads
            dropout_rate: Dropout rate
            quantiles: List of quantiles for prediction (default: [0.1, 0.5, 0.9])
        """
        self.n_encoder_steps = n_encoder_steps
        self.n_prediction_steps = n_prediction_steps
        self.n_features = n_features
        self.n_static_features = n_static_features
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        
        # Initialize components
        self.params = {}
        self._initialize_parameters()
        
        # Training state
        self.training = True
        self.adam_params = {}
        
    def _initialize_parameters(self):
        """Initialize all model parameters."""
        # Variable selection networks
        self._init_variable_selection()
        
        # LSTM encoder-decoder
        self._init_lstm_layers()
        
        # Attention mechanism
        self._init_attention_layers()
        
        # Output layers
        self._init_output_layers()
        
        # Layer normalization parameters
        self._init_layer_norm()
        
    def _init_variable_selection(self):
        """Initialize variable selection networks."""
        # Static variable selection
        static_input_size = self.n_static_features
        self.params['static_vsn_grn1_W'] = self._xavier_init((self.hidden_size, static_input_size))
        self.params['static_vsn_grn1_b'] = np.zeros((self.hidden_size, 1))
        
        self.params['static_vsn_grn2_W'] = self._xavier_init((self.hidden_size, self.hidden_size))
        self.params['static_vsn_grn2_b'] = np.zeros((self.hidden_size, 1))
        
        # Variable weights
        self.params['static_vsn_weights_W'] = self._xavier_init((static_input_size, self.hidden_size))
        self.params['static_vsn_weights_b'] = np.zeros((static_input_size, 1))
        
        # Temporal variable selection (past)
        temporal_input_size = self.n_features
        self.params['temporal_vsn_past_W'] = self._xavier_init((self.hidden_size, temporal_input_size))
        self.params['temporal_vsn_past_b'] = np.zeros((self.hidden_size, 1))
        
        # Temporal variable selection (future)
        self.params['temporal_vsn_future_W'] = self._xavier_init((self.hidden_size, temporal_input_size))
        self.params['temporal_vsn_future_b'] = np.zeros((self.hidden_size, 1))
        
    def _init_lstm_layers(self):
        """Initialize LSTM encoder-decoder layers."""
        # Encoder LSTM
        for i in range(self.lstm_layers):
            if i == 0:
                input_size = self.hidden_size
            else:
                input_size = self.hidden_size
                
            # LSTM gates
            self.params[f'enc_lstm_{i}_Wi'] = self._xavier_init((self.hidden_size, input_size))
            self.params[f'enc_lstm_{i}_Wf'] = self._xavier_init((self.hidden_size, input_size))
            self.params[f'enc_lstm_{i}_Wo'] = self._xavier_init((self.hidden_size, input_size))
            self.params[f'enc_lstm_{i}_Wc'] = self._xavier_init((self.hidden_size, input_size))
            
            self.params[f'enc_lstm_{i}_Ui'] = self._xavier_init((self.hidden_size, self.hidden_size))
            self.params[f'enc_lstm_{i}_Uf'] = self._xavier_init((self.hidden_size, self.hidden_size))
            self.params[f'enc_lstm_{i}_Uo'] = self._xavier_init((self.hidden_size, self.hidden_size))
            self.params[f'enc_lstm_{i}_Uc'] = self._xavier_init((self.hidden_size, self.hidden_size))
            
            self.params[f'enc_lstm_{i}_bi'] = np.zeros((self.hidden_size, 1))
            self.params[f'enc_lstm_{i}_bf'] = np.zeros((self.hidden_size, 1))
            self.params[f'enc_lstm_{i}_bo'] = np.zeros((self.hidden_size, 1))
            self.params[f'enc_lstm_{i}_bc'] = np.zeros((self.hidden_size, 1))
            
        # Decoder LSTM (similar structure)
        for i in range(self.lstm_layers):
            # Decoder receives enriched features
            if i == 0:
                input_size = self.hidden_size
            else:
                input_size = self.hidden_size
                
            self.params[f'dec_lstm_{i}_Wi'] = self._xavier_init((self.hidden_size, input_size))
            self.params[f'dec_lstm_{i}_Wf'] = self._xavier_init((self.hidden_size, input_size))
            self.params[f'dec_lstm_{i}_Wo'] = self._xavier_init((self.hidden_size, input_size))
            self.params[f'dec_lstm_{i}_Wc'] = self._xavier_init((self.hidden_size, input_size))
            
            self.params[f'dec_lstm_{i}_Ui'] = self._xavier_init((self.hidden_size, self.hidden_size))
            self.params[f'dec_lstm_{i}_Uf'] = self._xavier_init((self.hidden_size, self.hidden_size))
            self.params[f'dec_lstm_{i}_Uo'] = self._xavier_init((self.hidden_size, self.hidden_size))
            self.params[f'dec_lstm_{i}_Uc'] = self._xavier_init((self.hidden_size, self.hidden_size))
            
            self.params[f'dec_lstm_{i}_bi'] = np.zeros((self.hidden_size, 1))
            self.params[f'dec_lstm_{i}_bf'] = np.zeros((self.hidden_size, 1))
            self.params[f'dec_lstm_{i}_bo'] = np.zeros((self.hidden_size, 1))
            self.params[f'dec_lstm_{i}_bc'] = np.zeros((self.hidden_size, 1))
            
    def _init_attention_layers(self):
        """Initialize multi-head attention layers."""
        # Attention projection matrices
        head_dim = self.hidden_size // self.num_attention_heads
        
        self.params['attention_Wq'] = self._xavier_init((self.hidden_size, self.hidden_size))
        self.params['attention_Wk'] = self._xavier_init((self.hidden_size, self.hidden_size))
        self.params['attention_Wv'] = self._xavier_init((self.hidden_size, self.hidden_size))
        self.params['attention_Wo'] = self._xavier_init((self.hidden_size, self.hidden_size))
        
        # Attention layer norm
        self.params['attention_ln_gamma'] = np.ones((self.hidden_size, 1))
        self.params['attention_ln_beta'] = np.zeros((self.hidden_size, 1))
        
    def _init_output_layers(self):
        """Initialize output layers for quantile predictions."""
        # Gated Residual Network before output
        self.params['output_grn_W1'] = self._xavier_init((self.hidden_size, self.hidden_size))
        self.params['output_grn_b1'] = np.zeros((self.hidden_size, 1))
        
        self.params['output_grn_W2'] = self._xavier_init((self.hidden_size, self.hidden_size))
        self.params['output_grn_b2'] = np.zeros((self.hidden_size, 1))
        
        # Output projections for each quantile
        for i, q in enumerate(self.quantiles):
            self.params[f'output_q{i}_W'] = self._xavier_init((1, self.hidden_size))
            self.params[f'output_q{i}_b'] = np.zeros((1, 1))
            
    def _init_layer_norm(self):
        """Initialize layer normalization parameters."""
        # Encoder layer norm
        self.params['enc_ln_gamma'] = np.ones((self.hidden_size, 1))
        self.params['enc_ln_beta'] = np.zeros((self.hidden_size, 1))
        
        # Decoder layer norm
        self.params['dec_ln_gamma'] = np.ones((self.hidden_size, 1))
        self.params['dec_ln_beta'] = np.zeros((self.hidden_size, 1))
        
    def _xavier_init(self, shape: Tuple[int, int]) -> np.ndarray:
        """Xavier initialization for weights."""
        fan_in, fan_out = shape[1], shape[0]
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    
    def forward(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Forward pass through TFT.
        
        Args:
            inputs: Dictionary containing:
                - 'temporal_inputs': (batch_size, n_encoder_steps + n_prediction_steps, n_features)
                - 'static_inputs': (batch_size, n_static_features)
                - 'known_future_mask': Binary mask for known future features
                
        Returns:
            Dictionary with quantile predictions and attention weights
        """
        batch_size = inputs['temporal_inputs'].shape[0]
        
        # Split temporal inputs
        past_inputs = inputs['temporal_inputs'][:, :self.n_encoder_steps, :]
        future_inputs = inputs['temporal_inputs'][:, self.n_encoder_steps:, :]
        
        # Variable selection
        static_encoding, static_weights = self._static_variable_selection(
            inputs['static_inputs']
        )
        
        past_features, past_weights = self._temporal_variable_selection(
            past_inputs, static_encoding, 'past'
        )
        
        future_features, future_weights = self._temporal_variable_selection(
            future_inputs, static_encoding, 'future'
        )
        
        # LSTM encoding
        encoder_outputs, encoder_states = self._lstm_encoder(past_features)
        
        # LSTM decoding with attention
        decoder_outputs, attention_weights = self._lstm_decoder_with_attention(
            future_features, encoder_outputs, encoder_states
        )
        
        # Generate quantile outputs
        quantile_outputs = self._generate_outputs(decoder_outputs)
        
        return {
            'predictions': quantile_outputs,
            'attention_weights': attention_weights,
            'static_weights': static_weights,
            'past_weights': past_weights,
            'future_weights': future_weights
        }
    
    def _static_variable_selection(self, static_inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply variable selection to static features."""
        # GRN for static features
        hidden = self._gated_residual_network(
            static_inputs.T,
            self.params['static_vsn_grn1_W'],
            self.params['static_vsn_grn1_b'],
            self.params['static_vsn_grn2_W'],
            self.params['static_vsn_grn2_b']
        )
        
        # Calculate variable weights
        weights = self._softmax(
            self.params['static_vsn_weights_W'] @ hidden + 
            self.params['static_vsn_weights_b']
        )
        
        # Apply weights
        selected_features = static_inputs * weights.T
        
        # Static encoding for temporal features
        static_encoding = hidden
        
        return static_encoding, weights
    
    def _temporal_variable_selection(self, temporal_inputs: np.ndarray,
                                   static_encoding: np.ndarray,
                                   selection_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Apply variable selection to temporal features."""
        batch_size, time_steps, n_features = temporal_inputs.shape
        
        # Choose parameters based on selection type
        if selection_type == 'past':
            W = self.params['temporal_vsn_past_W']
            b = self.params['temporal_vsn_past_b']
        else:
            W = self.params['temporal_vsn_future_W']
            b = self.params['temporal_vsn_future_b']
            
        # Process each time step
        selected_features = np.zeros((batch_size, time_steps, self.hidden_size))
        weights_all = np.zeros((batch_size, time_steps, n_features))
        
        for t in range(time_steps):
            # Combine temporal features with static encoding
            combined = np.concatenate([
                temporal_inputs[:, t, :],
                static_encoding.T
            ], axis=1)
            
            # Apply GRN
            hidden = self._apply_grn_layer(combined.T, W, b)
            
            # Calculate weights
            weights = self._softmax(hidden[:n_features, :])
            
            # Apply weights
            selected = temporal_inputs[:, t, :] * weights.T
            selected_features[:, t, :] = selected @ W[:, :n_features] + static_encoding.T
            weights_all[:, t, :] = weights.T
            
        return selected_features, weights_all
    
    def _lstm_encoder(self, inputs: np.ndarray) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """LSTM encoder for past inputs."""
        batch_size, time_steps, _ = inputs.shape
        
        # Initialize states
        states = []
        for i in range(self.lstm_layers):
            h = np.zeros((batch_size, self.hidden_size))
            c = np.zeros((batch_size, self.hidden_size))
            states.append((h, c))
            
        # Process each time step
        outputs = []
        
        for t in range(time_steps):
            x = inputs[:, t, :]
            
            # Through each LSTM layer
            for i in range(self.lstm_layers):
                h, c = states[i]
                h_new, c_new = self._lstm_cell(x, h, c, f'enc_lstm_{i}')
                states[i] = (h_new, c_new)
                x = h_new  # Input to next layer
                
            outputs.append(x)
            
        outputs = np.stack(outputs, axis=1)
        
        # Apply layer normalization
        outputs = self._layer_norm(outputs, 'enc')
        
        return outputs, states
    
    def _lstm_decoder_with_attention(self, future_inputs: np.ndarray,
                                   encoder_outputs: np.ndarray,
                                   encoder_states: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """LSTM decoder with temporal attention."""
        batch_size, future_steps, _ = future_inputs.shape
        
        # Initialize decoder states from encoder
        states = encoder_states.copy()
        
        # Process future steps
        outputs = []
        attention_weights_all = []
        
        for t in range(future_steps):
            x = future_inputs[:, t, :]
            
            # Apply attention to encoder outputs
            attended, attention_weights = self._temporal_attention(
                x, encoder_outputs
            )
            attention_weights_all.append(attention_weights)
            
            # Combine with attended features
            x = x + attended
            
            # Through decoder LSTM layers
            for i in range(self.lstm_layers):
                h, c = states[i]
                h_new, c_new = self._lstm_cell(x, h, c, f'dec_lstm_{i}')
                states[i] = (h_new, c_new)
                x = h_new
                
            outputs.append(x)
            
        outputs = np.stack(outputs, axis=1)
        attention_weights_all = np.stack(attention_weights_all, axis=1)
        
        # Apply layer normalization
        outputs = self._layer_norm(outputs, 'dec')
        
        return outputs, attention_weights_all
    
    def _temporal_attention(self, query: np.ndarray,
                          encoder_outputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply temporal self-attention."""
        batch_size = query.shape[0]
        seq_len = encoder_outputs.shape[1]
        
        # Project query
        q = query @ self.params['attention_Wq'].T  # (batch, hidden)
        
        # Project keys and values from encoder outputs
        k = encoder_outputs @ self.params['attention_Wk'].T  # (batch, seq_len, hidden)
        v = encoder_outputs @ self.params['attention_Wv'].T  # (batch, seq_len, hidden)
        
        # Scaled dot-product attention
        head_dim = self.hidden_size // self.num_attention_heads
        
        # Reshape for multi-head
        q = q.reshape(batch_size, self.num_attention_heads, head_dim)
        k = k.reshape(batch_size, seq_len, self.num_attention_heads, head_dim)
        v = v.reshape(batch_size, seq_len, self.num_attention_heads, head_dim)
        
        # Compute attention scores
        scores = np.zeros((batch_size, self.num_attention_heads, seq_len))
        for b in range(batch_size):
            for h in range(self.num_attention_heads):
                scores[b, h] = (q[b, h] @ k[b, :, h, :].T) / np.sqrt(head_dim)
                
        # Apply softmax
        attention_weights = np.zeros_like(scores)
        for b in range(batch_size):
            for h in range(self.num_attention_heads):
                attention_weights[b, h] = self._softmax(scores[b, h])
                
        # Apply attention weights
        attended = np.zeros((batch_size, self.hidden_size))
        for b in range(batch_size):
            for h in range(self.num_attention_heads):
                head_output = attention_weights[b, h] @ v[b, :, h, :]
                attended[b, h*head_dim:(h+1)*head_dim] = head_output
                
        # Output projection
        attended = attended @ self.params['attention_Wo'].T
        
        # Average attention weights across heads
        avg_weights = np.mean(attention_weights, axis=1)
        
        return attended, avg_weights
    
    def _lstm_cell(self, x: np.ndarray, h_prev: np.ndarray,
                   c_prev: np.ndarray, prefix: str) -> Tuple[np.ndarray, np.ndarray]:
        """LSTM cell computation."""
        # Input gate
        i = self._sigmoid(
            x @ self.params[f'{prefix}_Wi'].T +
            h_prev @ self.params[f'{prefix}_Ui'].T +
            self.params[f'{prefix}_bi'].T
        )
        
        # Forget gate
        f = self._sigmoid(
            x @ self.params[f'{prefix}_Wf'].T +
            h_prev @ self.params[f'{prefix}_Uf'].T +
            self.params[f'{prefix}_bf'].T
        )
        
        # Output gate
        o = self._sigmoid(
            x @ self.params[f'{prefix}_Wo'].T +
            h_prev @ self.params[f'{prefix}_Uo'].T +
            self.params[f'{prefix}_bo'].T
        )
        
        # Cell candidate
        c_tilde = np.tanh(
            x @ self.params[f'{prefix}_Wc'].T +
            h_prev @ self.params[f'{prefix}_Uc'].T +
            self.params[f'{prefix}_bc'].T
        )
        
        # New cell state
        c = f * c_prev + i * c_tilde
        
        # New hidden state
        h = o * np.tanh(c)
        
        # Apply dropout if training
        if self.training and self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, h.shape) / (1 - self.dropout_rate)
            h = h * mask
            
        return h, c
    
    def _gated_residual_network(self, x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                               W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
        """Gated Residual Network (GRN) layer."""
        # First linear layer with ELU activation
        hidden = self._elu(W1 @ x + b1)
        
        # Gating mechanism
        gate = self._sigmoid(W2 @ hidden + b2)
        
        # Residual connection
        output = gate * hidden + (1 - gate) * x
        
        return output
    
    def _apply_grn_layer(self, x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Apply simple GRN transformation."""
        return self._elu(W @ x + b)
    
    def _generate_outputs(self, decoder_outputs: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate quantile predictions from decoder outputs."""
        batch_size, time_steps, _ = decoder_outputs.shape
        
        # Apply output GRN
        outputs = decoder_outputs.reshape(-1, self.hidden_size).T
        
        hidden = self._gated_residual_network(
            outputs,
            self.params['output_grn_W1'],
            self.params['output_grn_b1'],
            self.params['output_grn_W2'],
            self.params['output_grn_b2']
        )
        
        # Generate predictions for each quantile
        quantile_predictions = {}
        
        for i, q in enumerate(self.quantiles):
            pred = self.params[f'output_q{i}_W'] @ hidden + self.params[f'output_q{i}_b']
            pred = pred.T.reshape(batch_size, time_steps)
            quantile_predictions[f'q{int(q*100)}'] = pred
            
        return quantile_predictions
    
    def _layer_norm(self, x: np.ndarray, prefix: str) -> np.ndarray:
        """Apply layer normalization."""
        # Compute mean and variance
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + 1e-5)
        
        # Scale and shift
        gamma = self.params[f'{prefix}_ln_gamma'].T
        beta = self.params[f'{prefix}_ln_beta'].T
        
        return x_norm * gamma + beta
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _elu(self, x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """ELU activation."""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Make predictions (inference mode)."""
        self.training = False
        predictions = self.forward(inputs)
        self.training = True
        return predictions
    
    def get_feature_importance(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract feature importance from variable selection networks."""
        outputs = self.forward(inputs)
        
        return {
            'static_importance': outputs['static_weights'],
            'temporal_past_importance': np.mean(outputs['past_weights'], axis=1),
            'temporal_future_importance': np.mean(outputs['future_weights'], axis=1),
            'attention_importance': np.mean(outputs['attention_weights'], axis=(1, 2))
        }