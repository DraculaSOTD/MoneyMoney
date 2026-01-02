import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from .gru_cell import MultiLayerGRU
from .attention import MultiHeadAttention, TemporalAttention
from utils.matrix_operations import MatrixOperations


class GRUAttentionModel:
    """
    GRU with Multi-Head Attention model for cryptocurrency price prediction.
    
    Architecture:
    - Multi-layer GRU for sequential feature extraction
    - Multi-head attention for capturing dependencies
    - Dual output heads: action probabilities + confidence scores
    
    Optimized for crypto trading with:
    - Handling of high-frequency 1-minute data
    - Robust to noise and volatility
    - Interpretable attention weights
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int] = [256, 128, 64],
                 num_attention_heads: int = 4,
                 num_classes: int = 3,  # buy, hold, sell
                 dropout_rates: Optional[List[float]] = None,
                 use_layer_norm: bool = True,
                 learning_rate: float = 0.001):
        """
        Initialize GRU-Attention model.
        
        Args:
            input_size: Number of input features
            hidden_sizes: Hidden dimensions for GRU layers
            num_attention_heads: Number of attention heads
            num_classes: Number of output classes
            dropout_rates: Dropout rates for each layer
            use_layer_norm: Whether to use layer normalization
            learning_rate: Initial learning rate
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_attention_heads = num_attention_heads
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Build model components
        self._build_model(dropout_rates, use_layer_norm)
        
        # Training state
        self.training = True
        self.train_step = 0
        self.best_loss = float('inf')
        
        # Optimizer state (Adam)
        self._init_optimizer()
        
    def _build_model(self, dropout_rates: Optional[List[float]], 
                    use_layer_norm: bool):
        """Build model architecture."""
        # GRU layers
        self.gru = MultiLayerGRU(
            self.input_size,
            self.hidden_sizes,
            dropout_rates=dropout_rates,
            use_layer_norm=use_layer_norm
        )
        
        # Attention mechanism
        self.attention = MultiHeadAttention(
            d_model=self.hidden_sizes[-1],
            num_heads=self.num_attention_heads,
            dropout_rate=0.1
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(
            d_model=self.hidden_sizes[-1]
        )
        
        # Output layers
        self._init_output_layers()
        
    def _init_output_layers(self):
        """Initialize output projection layers."""
        final_hidden_size = self.hidden_sizes[-1]
        
        # Optional: Additional layers for better predictions
        self.W_hidden = self._xavier_init(final_hidden_size, final_hidden_size // 2)
        self.b_hidden = np.zeros(final_hidden_size // 2)
        
        # Action prediction head (from hidden layer output)
        self.W_action = self._xavier_init(final_hidden_size // 2, self.num_classes)
        self.b_action = np.zeros(self.num_classes)
        
        # Confidence prediction head (from final output)
        self.W_confidence = self._xavier_init(final_hidden_size, 1)
        self.b_confidence = np.zeros(1)
        
    def _xavier_init(self, fan_in: int, fan_out: int) -> np.ndarray:
        """Xavier/Glorot initialization."""
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(fan_in, fan_out) * scale
    
    def _init_optimizer(self):
        """Initialize Adam optimizer state."""
        # Collect all parameters
        self.params = self._get_all_params()
        
        # Adam optimizer states
        self.m = {name: np.zeros_like(param) for name, param in self.params.items()}
        self.v = {name: np.zeros_like(param) for name, param in self.params.items()}
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        
    def _get_all_params(self) -> Dict[str, np.ndarray]:
        """Get all model parameters."""
        params = {}
        
        # GRU parameters
        for i, cell in enumerate(self.gru.cells):
            cell_params = cell.get_params()
            for name, param in cell_params.items():
                params[f'gru_layer{i}_{name}'] = param
                
        # Attention parameters
        attention_params = self.attention.get_params()
        for name, param in attention_params.items():
            params[f'attention_{name}'] = param
            
        # Output layer parameters
        params.update({
            'W_action': self.W_action,
            'b_action': self.b_action,
            'W_confidence': self.W_confidence,
            'b_confidence': self.b_confidence,
            'W_hidden': self.W_hidden,
            'b_hidden': self.b_hidden
        })
        
        return params
    
    def forward(self, x: np.ndarray, 
                return_attention: bool = False) -> Dict[str, np.ndarray]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (batch_size, seq_length, input_size)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with predictions and optional attention weights
        """
        batch_size, seq_length, _ = x.shape
        
        # Add positional encoding
        x_encoded = self.temporal_attention.add_positional_encoding(x)
        
        # Pass through GRU layers
        gru_output, hidden_states = self.gru.forward(x_encoded)
        
        # Apply temporal attention
        # Create causal mask for autoregressive prediction
        causal_mask = self.temporal_attention.create_causal_mask(seq_length)
        
        # Multi-head attention
        attention_output, attention_weights = self.attention.forward(
            gru_output, gru_output, gru_output,
            mask=causal_mask,
            return_attention=True
        )
        
        # Residual connection
        combined_output = gru_output + attention_output
        
        # Take last time step for prediction
        final_output = combined_output[:, -1, :]
        
        # Hidden layer with ReLU
        hidden = self._relu(final_output @ self.W_hidden + self.b_hidden)
        
        # Action logits
        action_logits = hidden @ self.W_action + self.b_action
        action_probs = self._softmax(action_logits)
        
        # Confidence score (sigmoid)
        confidence_logit = final_output @ self.W_confidence + self.b_confidence
        confidence = self._sigmoid(confidence_logit)
        
        results = {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'confidence': confidence,
            'hidden_states': hidden_states,
            'gru_output': gru_output,
            'attention_output': attention_output,
            'hidden': hidden,  # Store for backward pass
            'x_encoded': x_encoded  # Store for backward pass
        }
        
        if return_attention:
            results['attention_weights'] = attention_weights
            
        return results
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def predict(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions (inference mode).
        
        Args:
            x: Input data
            
        Returns:
            Predictions dictionary
        """
        # Set to evaluation mode
        self.set_training(False)
        
        # Forward pass
        outputs = self.forward(x, return_attention=True)
        
        # Get predicted actions
        predicted_actions = np.argmax(outputs['action_probs'], axis=-1)
        
        # Action mapping
        action_map = {0: 'buy', 1: 'hold', 2: 'sell'}
        actions = [action_map[a] for a in predicted_actions]
        
        predictions = {
            'actions': actions,
            'action_probs': outputs['action_probs'],
            'confidence': outputs['confidence'],
            'attention_weights': outputs['attention_weights']
        }
        
        # Set back to training mode
        self.set_training(True)
        
        return predictions
    
    def compute_loss(self, predictions: Dict[str, np.ndarray],
                    targets: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute loss for training.
        
        Args:
            predictions: Model predictions
            targets: Target values
            
        Returns:
            Dictionary with loss components
        """
        # Cross-entropy loss for actions
        action_loss = self._cross_entropy_loss(
            predictions['action_logits'],
            targets['actions']
        )
        
        # MSE loss for confidence (if we have confidence targets)
        if 'confidence' in targets:
            confidence_loss = np.mean(
                (predictions['confidence'] - targets['confidence'])**2
            )
        else:
            # Use action correctness as confidence target
            correct_actions = np.argmax(predictions['action_probs'], axis=-1) == targets['actions']
            confidence_target = correct_actions.astype(float).reshape(-1, 1)
            confidence_loss = np.mean(
                (predictions['confidence'] - confidence_target)**2
            )
            
        # L2 regularization
        l2_reg = 0.0
        for name, param in self.params.items():
            if 'W' in name:  # Only regularize weights, not biases
                l2_reg += np.sum(param**2)
        l2_loss = 0.0001 * l2_reg
        
        # Total loss
        total_loss = action_loss + 0.5 * confidence_loss + l2_loss
        
        return {
            'total_loss': total_loss,
            'action_loss': action_loss,
            'confidence_loss': confidence_loss,
            'l2_loss': l2_loss
        }
    
    def _cross_entropy_loss(self, logits: np.ndarray, 
                           labels: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        # One-hot encode labels
        batch_size = logits.shape[0]
        one_hot = np.zeros((batch_size, self.num_classes))
        one_hot[np.arange(batch_size), labels] = 1
        
        # Stable log-softmax
        log_probs = logits - np.max(logits, axis=-1, keepdims=True)
        log_probs -= np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))
        
        # Cross-entropy
        loss = -np.sum(one_hot * log_probs) / batch_size
        
        return loss
    
    def backward(self, predictions: Dict[str, np.ndarray], 
                targets: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Backward pass to compute gradients through the entire model.
        
        Args:
            predictions: Model predictions from forward pass
            targets: Target values
            
        Returns:
            Dictionary of gradients for all parameters
        """
        batch_size = predictions['action_logits'].shape[0]
        gradients = {}
        
        # Get cached values from forward pass
        gru_output = predictions['gru_output']
        attention_output = predictions['attention_output']
        hidden_states = predictions['hidden_states']
        
        # Gradient from action loss (cross-entropy)
        # Softmax + cross-entropy gradient
        action_probs = predictions['action_probs']
        daction_logits = action_probs.copy()
        daction_logits[np.arange(batch_size), targets['actions']] -= 1
        daction_logits /= batch_size
        
        # Gradient through hidden layer
        # hidden = relu(final_output @ W_hidden + b_hidden)
        dhidden = daction_logits @ self.W_action.T
        
        # ReLU gradient
        hidden_mask = predictions.get('hidden', np.zeros_like(dhidden)) > 0
        dhidden = dhidden * hidden_mask
        
        # Gradients for hidden layer
        final_output = attention_output[:, -1, :] + gru_output[:, -1, :]
        gradients['dW_hidden'] = final_output.T @ dhidden
        gradients['db_hidden'] = np.sum(dhidden, axis=0)
        
        # Gradient to final output
        dfinal_output = dhidden @ self.W_hidden.T
        
        # Add gradient from confidence loss
        if 'confidence' in targets:
            confidence_error = predictions['confidence'] - targets['confidence'].reshape(-1, 1)
        else:
            # Use action correctness as confidence target
            correct_actions = np.argmax(action_probs, axis=-1) == targets['actions']
            confidence_target = correct_actions.astype(float).reshape(-1, 1)
            confidence_error = predictions['confidence'] - confidence_target
            
        dconfidence = 2 * confidence_error / batch_size  # MSE gradient
        
        # Gradient through sigmoid for confidence
        confidence_sigmoid = predictions['confidence']
        dconfidence_logit = dconfidence * confidence_sigmoid * (1 - confidence_sigmoid)
        
        # Gradients for confidence head
        gradients['dW_confidence'] = final_output.T @ dconfidence_logit
        gradients['db_confidence'] = np.sum(dconfidence_logit, axis=0)
        
        # Add confidence gradient to final output
        dfinal_output += dconfidence_logit @ self.W_confidence.T
        
        # Gradients for action head
        gradients['dW_action'] = dhidden.T @ daction_logits
        gradients['db_action'] = np.sum(daction_logits, axis=0)
        
        # Split gradient between attention and GRU outputs (residual connection)
        dattention_output = np.zeros_like(attention_output)
        dattention_output[:, -1, :] = dfinal_output * 0.5  # Split equally
        dgru_output = np.zeros_like(gru_output)
        dgru_output[:, -1, :] = dfinal_output * 0.5
        
        # Backward through attention
        if hasattr(self.attention, '_forward_cache'):
            cache = self.attention._forward_cache
            dquery, dkey, dvalue = self.attention.backward(
                dattention_output,
                cache['query'], cache['key'], cache['value'],
                cache['Q'], cache['K'], cache['V'],
                cache['attention_output'], cache['attention_weights'],
                cache['mask']
            )
            
            # Since query, key, value are all from gru_output
            dgru_output += dquery + dkey + dvalue
            
            # Get attention gradients
            for name in ['dW_q', 'dW_k', 'dW_v', 'dW_o']:
                if hasattr(self.attention, name):
                    gradients[f'attention_{name}'] = getattr(self.attention, name)
            if self.attention.use_bias:
                for name in ['db_q', 'db_k', 'db_v', 'db_o']:
                    if hasattr(self.attention, name):
                        gradients[f'attention_{name}'] = getattr(self.attention, name)
        
        # Backward through GRU layers (BPTT)
        seq_length = gru_output.shape[1]
        
        # Initialize gradient accumulators for hidden states
        dh_next = []
        for layer_idx in range(len(self.gru.cells)):
            if layer_idx == len(self.gru.cells) - 1:
                # Last layer gets gradient from output
                dh_next.append(dgru_output[:, -1, :].copy())
            else:
                # Other layers start with zero gradient
                dh_next.append(np.zeros((batch_size, self.gru.hidden_sizes[layer_idx])))
        
        # Backpropagate through time
        for t in reversed(range(seq_length)):
            for layer_idx in reversed(range(len(self.gru.cells))):
                cell = self.gru.cells[layer_idx]
                
                # Get gradient from upper layer or output
                if layer_idx == len(self.gru.cells) - 1:
                    dh_t = dgru_output[:, t, :] if t < seq_length - 1 else dh_next[layer_idx]
                else:
                    dh_t = dh_next[layer_idx]
                
                # Get input to this layer
                if layer_idx == 0:
                    x_t = predictions.get('x_encoded', gru_output)[:, t, :]
                else:
                    # Output from previous layer at time t
                    if hasattr(self.gru, '_layer_outputs'):
                        x_t = self.gru._layer_outputs[layer_idx - 1][:, t, :]
                    else:
                        x_t = hidden_states[layer_idx - 1]
                
                # Backward through GRU cell using cached values from time t
                if hasattr(cell, '_caches') and t < len(cell._caches):
                    cache = cell._caches[t]
                    grads = cell.backward(dh_t, cache)
                    
                    # Accumulate weight gradients
                    for param_name in ['dW_z', 'db_z', 'dW_r', 'db_r', 'dW_h', 'db_h']:
                        grad_key = f'gru_layer{layer_idx}_{param_name[1:]}'
                        if grad_key not in gradients:
                            gradients[grad_key] = np.zeros_like(grads[param_name])
                        gradients[grad_key] += grads[param_name]
                    
                    # Update hidden state gradient for next timestep
                    if t > 0:
                        dh_next[layer_idx] = grads['dh_prev']
                    
                    # Pass gradient to lower layer
                    if layer_idx > 0:
                        dh_next[layer_idx - 1] += grads['dx']
        
        # Clear caches after backward pass
        for cell in self.gru.cells:
            if hasattr(cell, '_caches'):
                cell._caches = []
        
        # L2 regularization gradients
        l2_lambda = 0.0001
        for name, param in self.params.items():
            if 'W' in name and name in gradients:
                gradients[name] += l2_lambda * 2 * param
        
        return gradients
    
    def update_weights(self, gradients: Dict[str, np.ndarray]):
        """
        Update weights using Adam optimizer.
        
        Args:
            gradients: Parameter gradients
        """
        self.train_step += 1
        
        # Learning rate scheduling
        lr = self.learning_rate * np.sqrt(1 - self.beta2**self.train_step) / \
             (1 - self.beta1**self.train_step)
             
        # Map gradient names to parameter names
        grad_to_param_map = {}
        for grad_name in gradients:
            if grad_name.startswith('d'):
                # Direct gradient (e.g., dW_action -> W_action)
                param_name = grad_name[1:]
            else:
                # Already in correct format
                param_name = grad_name
            grad_to_param_map[grad_name] = param_name
             
        # Update each parameter
        for grad_name, grad in gradients.items():
            param_name = grad_to_param_map.get(grad_name, grad_name)
            
            if param_name in self.params:
                param = self.params[param_name]
                
                # Gradient clipping
                grad = np.clip(grad, -5.0, 5.0)
                
                # Adam updates
                self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
                self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * grad**2
                
                # Parameter update
                self.params[param_name] -= lr * self.m[param_name] / (np.sqrt(self.v[param_name]) + self.epsilon)
                
                # Update the actual parameter in the model
                if param_name in ['W_action', 'b_action', 'W_confidence', 'b_confidence', 'W_hidden', 'b_hidden']:
                    setattr(self, param_name, self.params[param_name])
                elif param_name.startswith('attention_'):
                    attr_name = param_name.replace('attention_', '')
                    setattr(self.attention, attr_name, self.params[param_name])
                elif param_name.startswith('gru_layer'):
                    # Extract layer index and parameter name
                    # Format: gru_layer0_W_z -> layer_idx=0, param_attr=W_z
                    parts = param_name.split('_', 2)  # Split into at most 3 parts
                    if len(parts) >= 3 and parts[0] == 'gru' and parts[1].startswith('layer'):
                        layer_idx = int(parts[1].replace('layer', ''))
                        param_attr = parts[2]
                        setattr(self.gru.cells[layer_idx], param_attr, self.params[param_name])
    
    def set_training(self, training: bool):
        """Set training mode."""
        self.training = training
        self.gru.set_training(training)
        self.attention.set_training(training)
        
    def save_model(self, filepath: str):
        """Save model parameters."""
        np.savez(filepath, **self.params)
        
    def load_model(self, filepath: str):
        """Load model parameters."""
        loaded = np.load(filepath)
        for name, param in loaded.items():
            if name in self.params:
                self.params[name][:] = param
                
    def get_attention_analysis(self, x: np.ndarray) -> Dict:
        """
        Analyze attention patterns for interpretability.
        
        Args:
            x: Input data
            
        Returns:
            Dictionary with attention analysis
        """
        predictions = self.predict(x)
        attention_weights = predictions['attention_weights']
        
        # Average attention across heads
        avg_attention = np.mean(attention_weights, axis=1)
        
        # Find most attended time steps
        most_attended = np.argsort(avg_attention[0, -1, :])[-10:]
        
        # Calculate attention entropy (concentration measure)
        entropy = -np.sum(avg_attention * np.log(avg_attention + 1e-10), axis=-1)
        
        return {
            'average_attention': avg_attention,
            'most_attended_steps': most_attended,
            'attention_entropy': entropy,
            'attention_concentration': 1 - entropy / np.log(avg_attention.shape[-1])
        }