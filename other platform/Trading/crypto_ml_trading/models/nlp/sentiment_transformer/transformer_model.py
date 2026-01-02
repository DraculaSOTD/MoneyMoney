import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


class SentimentTransformer:
    """
    Transformer model for cryptocurrency sentiment analysis.
    
    Features:
    - Multi-head self-attention
    - Positional encoding
    - Feed-forward networks
    - Layer normalization
    - Custom implementation without external dependencies
    """
    
    def __init__(self,
                 vocab_size: int = 10000,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 d_ff: int = 1024,
                 max_length: int = 128,
                 dropout_rate: float = 0.1):
        """
        Initialize transformer model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_length: Maximum sequence length
            dropout_rate: Dropout rate
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_length = max_length
        self.dropout_rate = dropout_rate
        
        # Check dimensions
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        
        # Initialize parameters
        self.params = {}
        self._initialize_parameters()
        
        # Training state
        self.is_training = True
        
    def _initialize_parameters(self):
        """Initialize model parameters."""
        # Embedding layers
        self.params['token_embedding'] = self._xavier_init((self.vocab_size, self.d_model))
        self.params['position_embedding'] = self._create_positional_encoding()
        
        # Transformer layers
        for layer in range(self.n_layers):
            # Multi-head attention
            self.params[f'layer_{layer}_mha_q'] = self._xavier_init((self.d_model, self.d_model))
            self.params[f'layer_{layer}_mha_k'] = self._xavier_init((self.d_model, self.d_model))
            self.params[f'layer_{layer}_mha_v'] = self._xavier_init((self.d_model, self.d_model))
            self.params[f'layer_{layer}_mha_o'] = self._xavier_init((self.d_model, self.d_model))
            
            # Layer norm 1
            self.params[f'layer_{layer}_ln1_gamma'] = np.ones((self.d_model,))
            self.params[f'layer_{layer}_ln1_beta'] = np.zeros((self.d_model,))
            
            # Feed-forward
            self.params[f'layer_{layer}_ff_w1'] = self._xavier_init((self.d_ff, self.d_model))
            self.params[f'layer_{layer}_ff_b1'] = np.zeros((self.d_ff,))
            self.params[f'layer_{layer}_ff_w2'] = self._xavier_init((self.d_model, self.d_ff))
            self.params[f'layer_{layer}_ff_b2'] = np.zeros((self.d_model,))
            
            # Layer norm 2
            self.params[f'layer_{layer}_ln2_gamma'] = np.ones((self.d_model,))
            self.params[f'layer_{layer}_ln2_beta'] = np.zeros((self.d_model,))
            
        # Output layers for sentiment classification
        self.params['output_w'] = self._xavier_init((3, self.d_model))  # 3 classes: negative, neutral, positive
        self.params['output_b'] = np.zeros((3,))
        
        # Sentiment regression head
        self.params['sentiment_w'] = self._xavier_init((1, self.d_model))
        self.params['sentiment_b'] = np.zeros((1,))
        
    def _xavier_init(self, shape: Tuple) -> np.ndarray:
        """Xavier weight initialization."""
        if len(shape) == 2:
            fan_in, fan_out = shape[1], shape[0]
        else:
            fan_in = fan_out = shape[0]
            
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    
    def _create_positional_encoding(self) -> np.ndarray:
        """Create sinusoidal positional encoding."""
        position = np.arange(self.max_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * 
                          -(np.log(10000.0) / self.d_model))
        
        pos_encoding = np.zeros((self.max_length, self.d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return pos_encoding
    
    def forward(self, input_ids: np.ndarray, 
                attention_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Forward pass through transformer.
        
        Args:
            input_ids: Token indices (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Dictionary with outputs
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeddings = self.params['token_embedding'][input_ids]
        
        # Add positional encoding
        position_embeddings = self.params['position_embedding'][:seq_len]
        embeddings = token_embeddings + position_embeddings
        
        # Scale embeddings
        embeddings *= np.sqrt(self.d_model)
        
        # Apply dropout
        if self.is_training:
            embeddings = self._dropout(embeddings, self.dropout_rate)
            
        # Pass through transformer layers
        hidden_states = embeddings
        attention_weights = []
        
        for layer in range(self.n_layers):
            hidden_states, attn_weights = self._transformer_layer(
                hidden_states, attention_mask, layer
            )
            attention_weights.append(attn_weights)
            
        # Global average pooling for sequence representation
        if attention_mask is not None:
            mask_expanded = attention_mask[:, :, np.newaxis]
            masked_hidden = hidden_states * mask_expanded
            sequence_lengths = np.sum(attention_mask, axis=1, keepdims=True)
            pooled_output = np.sum(masked_hidden, axis=1) / sequence_lengths
        else:
            pooled_output = np.mean(hidden_states, axis=1)
            
        # Classification head
        logits = pooled_output @ self.params['output_w'].T + self.params['output_b']
        
        # Sentiment score head (-1 to 1)
        sentiment_score = pooled_output @ self.params['sentiment_w'].T + self.params['sentiment_b']
        sentiment_score = np.tanh(sentiment_score)
        
        return {
            'logits': logits,
            'sentiment_score': sentiment_score,
            'hidden_states': hidden_states,
            'pooled_output': pooled_output,
            'attention_weights': attention_weights
        }
    
    def _transformer_layer(self, hidden_states: np.ndarray,
                          attention_mask: Optional[np.ndarray],
                          layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Single transformer layer."""
        # Multi-head attention
        attn_output, attn_weights = self._multi_head_attention(
            hidden_states, hidden_states, hidden_states,
            attention_mask, layer_idx
        )
        
        # Residual connection and layer norm
        hidden_states = self._layer_norm(
            hidden_states + attn_output,
            self.params[f'layer_{layer_idx}_ln1_gamma'],
            self.params[f'layer_{layer_idx}_ln1_beta']
        )
        
        # Feed-forward network
        ff_output = self._feed_forward(hidden_states, layer_idx)
        
        # Residual connection and layer norm
        hidden_states = self._layer_norm(
            hidden_states + ff_output,
            self.params[f'layer_{layer_idx}_ln2_gamma'],
            self.params[f'layer_{layer_idx}_ln2_beta']
        )
        
        return hidden_states, attn_weights
    
    def _multi_head_attention(self, query: np.ndarray, key: np.ndarray,
                             value: np.ndarray, attention_mask: Optional[np.ndarray],
                             layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-head attention mechanism."""
        batch_size, seq_len, _ = query.shape
        
        # Linear projections
        Q = query @ self.params[f'layer_{layer_idx}_mha_q'].T
        K = key @ self.params[f'layer_{layer_idx}_mha_k'].T
        V = value @ self.params[f'layer_{layer_idx}_mha_v'].T
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        attention_scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        
        # Apply attention mask
        if attention_mask is not None:
            # Expand mask for heads
            mask_expanded = attention_mask[:, np.newaxis, np.newaxis, :]
            attention_scores = attention_scores.masked_fill(mask_expanded == 0, -1e9)
            
        # Softmax
        attention_weights = self._softmax(attention_scores, axis=-1)
        
        # Apply dropout
        if self.is_training:
            attention_weights = self._dropout(attention_weights, self.dropout_rate)
            
        # Weighted sum
        attention_output = attention_weights @ V
        
        # Reshape and project
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.d_model
        )
        attention_output = attention_output @ self.params[f'layer_{layer_idx}_mha_o'].T
        
        return attention_output, attention_weights
    
    def _feed_forward(self, hidden_states: np.ndarray, layer_idx: int) -> np.ndarray:
        """Feed-forward network."""
        # First linear layer
        ff_output = hidden_states @ self.params[f'layer_{layer_idx}_ff_w1'].T
        ff_output += self.params[f'layer_{layer_idx}_ff_b1']
        
        # ReLU activation
        ff_output = np.maximum(0, ff_output)
        
        # Dropout
        if self.is_training:
            ff_output = self._dropout(ff_output, self.dropout_rate)
            
        # Second linear layer
        ff_output = ff_output @ self.params[f'layer_{layer_idx}_ff_w2'].T
        ff_output += self.params[f'layer_{layer_idx}_ff_b2']
        
        return ff_output
    
    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, 
                   beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax activation."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _dropout(self, x: np.ndarray, rate: float) -> np.ndarray:
        """Apply dropout."""
        if not self.is_training or rate == 0:
            return x
        mask = np.random.rand(*x.shape) > rate
        return x * mask / (1 - rate)
    
    def predict_sentiment(self, input_ids: np.ndarray,
                         attention_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Predict sentiment for input sequences.
        
        Args:
            input_ids: Token indices
            attention_mask: Attention mask
            
        Returns:
            Sentiment predictions
        """
        self.set_training(False)
        outputs = self.forward(input_ids, attention_mask)
        
        # Get class probabilities
        probs = self._softmax(outputs['logits'])
        
        # Get predicted classes
        predictions = np.argmax(probs, axis=-1)
        
        # Map to sentiment labels
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        sentiment_labels = [sentiment_map[p] for p in predictions]
        
        return {
            'labels': sentiment_labels,
            'probabilities': probs,
            'sentiment_scores': outputs['sentiment_score'],
            'confidence': np.max(probs, axis=-1)
        }
    
    def get_attention_analysis(self, input_ids: np.ndarray,
                             attention_mask: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze attention patterns.
        
        Args:
            input_ids: Token indices
            attention_mask: Attention mask
            
        Returns:
            Attention analysis
        """
        self.set_training(False)
        outputs = self.forward(input_ids, attention_mask)
        
        # Average attention across heads and layers
        attention_weights = outputs['attention_weights']
        avg_attention = np.mean([np.mean(attn, axis=1) for attn in attention_weights], axis=0)
        
        # Find most attended tokens
        batch_size, seq_len = input_ids.shape
        important_tokens = []
        
        for i in range(batch_size):
            # Get attention scores for CLS token to all other tokens
            cls_attention = avg_attention[i, 0, :]
            
            # Find top attended positions
            top_positions = np.argsort(cls_attention)[-5:][::-1]
            important_tokens.append(top_positions)
            
        return {
            'average_attention': avg_attention,
            'important_token_positions': important_tokens,
            'layer_attention_patterns': attention_weights
        }
    
    def set_training(self, is_training: bool):
        """Set training mode."""
        self.is_training = is_training
    
    def save_model(self, filepath: str):
        """Save model parameters."""
        np.savez(filepath, **self.params)
        
    def load_model(self, filepath: str):
        """Load model parameters."""
        data = np.load(filepath)
        for key in data.files:
            self.params[key] = data[key]