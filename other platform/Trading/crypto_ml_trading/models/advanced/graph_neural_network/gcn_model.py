"""
Graph Convolutional Network (GCN) Implementation.

Implements GCN for node classification and graph-level predictions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


class GraphConvolutionalLayer:
    """
    Single Graph Convolutional Layer.
    
    Implements the GCN layer: H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 activation: str = 'relu',
                 dropout_rate: float = 0.1,
                 use_bias: bool = True):
        """
        Initialize GCN layer.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            dropout_rate: Dropout rate for regularization
            use_bias: Whether to use bias term
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        
        # Initialize parameters
        self.W = self._xavier_init((output_dim, input_dim))
        if use_bias:
            self.b = np.zeros((output_dim, 1))
        else:
            self.b = None
            
        # Training state
        self.training = True
        
    def _xavier_init(self, shape: Tuple[int, int]) -> np.ndarray:
        """Xavier initialization for weights."""
        fan_in, fan_out = shape[1], shape[0]
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    
    def forward(self, 
                node_features: np.ndarray,
                adjacency_matrix: np.ndarray,
                normalized_laplacian: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through GCN layer.
        
        Args:
            node_features: Node features (n_nodes, input_dim)
            adjacency_matrix: Adjacency matrix (n_nodes, n_nodes)
            normalized_laplacian: Pre-computed normalized Laplacian
            
        Returns:
            Updated node features (n_nodes, output_dim)
        """
        # Transpose for matrix operations
        H = node_features.T  # (input_dim, n_nodes)
        
        # Compute normalized adjacency if not provided
        if normalized_laplacian is None:
            A_norm = self._normalize_adjacency(adjacency_matrix)
        else:
            A_norm = normalized_laplacian
            
        # Apply graph convolution: A_norm @ H @ W + b
        # H_new = A_norm @ H.T @ W.T + b
        H_transformed = self.W @ H  # (output_dim, n_nodes)
        
        if self.b is not None:
            H_transformed = H_transformed + self.b
            
        # Apply adjacency aggregation
        H_aggregated = H_transformed @ A_norm.T  # (output_dim, n_nodes)
        
        # Apply activation
        H_activated = self._apply_activation(H_aggregated)
        
        # Apply dropout during training
        if self.training and self.dropout_rate > 0:
            dropout_mask = np.random.binomial(
                1, 1 - self.dropout_rate, H_activated.shape
            ) / (1 - self.dropout_rate)
            H_activated = H_activated * dropout_mask
            
        return H_activated.T  # (n_nodes, output_dim)
    
    def _normalize_adjacency(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        Compute symmetric normalization of adjacency matrix.
        
        Returns: D^(-1/2) (A + I) D^(-1/2)
        """
        # Add self-loops
        A = adjacency_matrix + np.eye(adjacency_matrix.shape[0])
        
        # Compute degree matrix
        D = np.diag(np.sum(A, axis=1))
        
        # Compute D^(-1/2)
        D_inv_sqrt = np.zeros_like(D)
        for i in range(D.shape[0]):
            if D[i, i] > 0:
                D_inv_sqrt[i, i] = 1.0 / np.sqrt(D[i, i])
                
        # Symmetric normalization
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        
        return A_norm
    
    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)
        else:
            return x  # Linear activation


class GraphConvolutionalNetwork:
    """
    Multi-layer Graph Convolutional Network.
    
    Features:
    - Multi-layer GCN architecture
    - Node classification and graph-level prediction
    - Attention-based graph pooling
    - Residual connections
    - Batch normalization
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = None,
                 output_dim: int = 3,
                 num_classes: int = 3,
                 dropout_rate: float = 0.1,
                 use_residual: bool = True,
                 use_batch_norm: bool = True,
                 pooling_method: str = 'mean'):
        """
        Initialize GCN model.
        
        Args:
            input_dim: Input node feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Final node embedding dimension
            num_classes: Number of output classes
            dropout_rate: Dropout rate
            use_residual: Whether to use residual connections
            use_batch_norm: Whether to use batch normalization
            pooling_method: Graph pooling method ('mean', 'max', 'attention')
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        self.pooling_method = pooling_method
        
        # Build layer dimensions
        dims = [input_dim] + self.hidden_dims + [output_dim]
        
        # Initialize GCN layers
        self.gcn_layers = []
        for i in range(len(dims) - 1):
            layer = GraphConvolutionalLayer(
                input_dim=dims[i],
                output_dim=dims[i + 1],
                activation='relu' if i < len(dims) - 2 else 'linear',
                dropout_rate=dropout_rate
            )
            self.gcn_layers.append(layer)
            
        # Batch normalization layers
        if use_batch_norm:
            self.bn_layers = [BatchNormalization(dim) for dim in dims[1:]]
        else:
            self.bn_layers = [None] * len(dims[1:])
            
        # Classification layers
        self.classifier = MLPClassifier(
            input_dim=output_dim,
            hidden_dim=output_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        # Attention pooling
        if pooling_method == 'attention':
            self.attention_pooling = AttentionPooling(output_dim)
        else:
            self.attention_pooling = None
            
        # Training state
        self.training = True
        
    def forward(self, 
                node_features: np.ndarray,
                adjacency_matrix: np.ndarray,
                return_embeddings: bool = False) -> Dict[str, np.ndarray]:
        """
        Forward pass through GCN.
        
        Args:
            node_features: Node features (n_nodes, input_dim)
            adjacency_matrix: Adjacency matrix (n_nodes, n_nodes)
            return_embeddings: Whether to return node embeddings
            
        Returns:
            Dictionary with predictions and optionally embeddings
        """
        # Precompute normalized adjacency
        A_norm = self._normalize_adjacency(adjacency_matrix)
        
        # Forward through GCN layers
        h = node_features
        embeddings = [h]
        
        for i, (gcn_layer, bn_layer) in enumerate(zip(self.gcn_layers, self.bn_layers)):
            h_new = gcn_layer.forward(h, adjacency_matrix, A_norm)
            
            # Apply batch normalization
            if bn_layer is not None:
                h_new = bn_layer.forward(h_new, self.training)
                
            # Apply residual connection
            if (self.use_residual and 
                h.shape[1] == h_new.shape[1] and 
                i > 0):
                h_new = h_new + h
                
            h = h_new
            embeddings.append(h)
            
        # Final node embeddings
        node_embeddings = h
        
        # Graph-level pooling
        graph_embedding = self._pool_graph(node_embeddings, adjacency_matrix)
        
        # Node classification
        node_predictions = self.classifier.forward(node_embeddings)
        
        # Graph classification
        graph_prediction = self.classifier.forward(graph_embedding.reshape(1, -1))
        
        result = {
            'node_predictions': node_predictions,
            'graph_prediction': graph_prediction.flatten(),
            'graph_embedding': graph_embedding
        }
        
        if return_embeddings:
            result['node_embeddings'] = node_embeddings
            result['layer_embeddings'] = embeddings
            
        return result
    
    def _normalize_adjacency(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """Normalize adjacency matrix."""
        # Add self-loops
        A = adjacency_matrix + np.eye(adjacency_matrix.shape[0])
        
        # Compute degree matrix
        D = np.diag(np.sum(A, axis=1))
        
        # Compute D^(-1/2)
        D_inv_sqrt = np.zeros_like(D)
        for i in range(D.shape[0]):
            if D[i, i] > 0:
                D_inv_sqrt[i, i] = 1.0 / np.sqrt(D[i, i])
                
        # Symmetric normalization
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        
        return A_norm
    
    def _pool_graph(self, 
                   node_embeddings: np.ndarray,
                   adjacency_matrix: np.ndarray) -> np.ndarray:
        """Pool node embeddings to graph-level representation."""
        if self.pooling_method == 'mean':
            return np.mean(node_embeddings, axis=0)
        elif self.pooling_method == 'max':
            return np.max(node_embeddings, axis=0)
        elif self.pooling_method == 'sum':
            return np.sum(node_embeddings, axis=0)
        elif self.pooling_method == 'attention' and self.attention_pooling:
            return self.attention_pooling.forward(node_embeddings)
        else:
            return np.mean(node_embeddings, axis=0)
    
    def predict_node_types(self, 
                          node_features: np.ndarray,
                          adjacency_matrix: np.ndarray) -> np.ndarray:
        """Predict node types (wallet, exchange, whale, contract)."""
        self.training = False
        result = self.forward(node_features, adjacency_matrix)
        self.training = True
        
        # Convert to class predictions
        node_probs = self._softmax(result['node_predictions'])
        node_classes = np.argmax(node_probs, axis=1)
        
        return node_classes
    
    def predict_graph_signal(self,
                           node_features: np.ndarray,
                           adjacency_matrix: np.ndarray) -> Dict[str, float]:
        """Predict trading signal from graph structure."""
        self.training = False
        result = self.forward(node_features, adjacency_matrix)
        self.training = True
        
        # Interpret graph prediction as trading signal
        graph_probs = self._softmax(result['graph_prediction'].reshape(1, -1))[0]
        
        # Map to trading actions
        signal_mapping = ['sell', 'hold', 'buy']
        predicted_action = signal_mapping[np.argmax(graph_probs)]
        confidence = np.max(graph_probs)
        
        return {
            'action': predicted_action,
            'confidence': confidence,
            'action_probabilities': {
                'sell': graph_probs[0],
                'hold': graph_probs[1],
                'buy': graph_probs[2]
            },
            'graph_embedding': result['graph_embedding']
        }
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax with numerical stability."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def analyze_node_importance(self,
                               node_features: np.ndarray,
                               adjacency_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """Analyze importance of each node for graph prediction."""
        # Baseline prediction
        baseline_result = self.forward(node_features, adjacency_matrix)
        baseline_pred = baseline_result['graph_prediction']
        
        # Perturbation analysis
        n_nodes = node_features.shape[0]
        importance_scores = np.zeros(n_nodes)
        
        for i in range(n_nodes):
            # Create perturbed features (zero out node i)
            perturbed_features = node_features.copy()
            perturbed_features[i] = 0
            
            # Get prediction without node i
            perturbed_result = self.forward(perturbed_features, adjacency_matrix)
            perturbed_pred = perturbed_result['graph_prediction']
            
            # Calculate importance as prediction change
            importance_scores[i] = np.linalg.norm(baseline_pred - perturbed_pred)
            
        # Normalize importance scores
        if np.max(importance_scores) > 0:
            importance_scores /= np.max(importance_scores)
            
        # Rank nodes by importance
        importance_ranking = np.argsort(importance_scores)[::-1]
        
        return {
            'importance_scores': importance_scores,
            'importance_ranking': importance_ranking,
            'top_nodes': importance_ranking[:10],
            'baseline_prediction': baseline_pred
        }


class BatchNormalization:
    """Batch normalization for GCN layers."""
    
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones((num_features, 1))
        self.beta = np.zeros((num_features, 1))
        
        # Running statistics
        self.running_mean = np.zeros((num_features, 1))
        self.running_var = np.ones((num_features, 1))
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply batch normalization."""
        x = x.T  # (num_features, batch_size)
        
        if training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=1, keepdims=True)
            batch_var = np.var(x, axis=1, keepdims=True)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            # Use running statistics
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            
        # Scale and shift
        out = self.gamma * x_norm + self.beta
        
        return out.T  # (batch_size, num_features)


class MLPClassifier:
    """Multi-layer perceptron for node/graph classification."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 dropout_rate: float = 0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Initialize parameters
        self.W1 = self._xavier_init((hidden_dim, input_dim))
        self.b1 = np.zeros((hidden_dim, 1))
        
        self.W2 = self._xavier_init((num_classes, hidden_dim))
        self.b2 = np.zeros((num_classes, 1))
        
        # Training state
        self.training = True
        
    def _xavier_init(self, shape: Tuple[int, int]) -> np.ndarray:
        """Xavier initialization."""
        fan_in, fan_out = shape[1], shape[0]
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through MLP."""
        x = x.T  # (input_dim, batch_size)
        
        # Hidden layer
        h = np.maximum(0, self.W1 @ x + self.b1)  # ReLU activation
        
        # Apply dropout
        if self.training and self.dropout_rate > 0:
            dropout_mask = np.random.binomial(
                1, 1 - self.dropout_rate, h.shape
            ) / (1 - self.dropout_rate)
            h = h * dropout_mask
            
        # Output layer
        out = self.W2 @ h + self.b2
        
        return out.T  # (batch_size, num_classes)


class AttentionPooling:
    """Attention-based graph pooling."""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        
        # Attention parameters
        self.W_att = self._xavier_init((1, input_dim))
        self.b_att = np.zeros((1, 1))
        
    def _xavier_init(self, shape: Tuple[int, int]) -> np.ndarray:
        """Xavier initialization."""
        fan_in, fan_out = shape[1], shape[0]
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    
    def forward(self, node_embeddings: np.ndarray) -> np.ndarray:
        """Apply attention pooling."""
        # Compute attention scores
        x = node_embeddings.T  # (input_dim, num_nodes)
        att_scores = self.W_att @ x + self.b_att  # (1, num_nodes)
        
        # Apply softmax
        att_weights = self._softmax(att_scores)
        
        # Weighted average
        pooled = x @ att_weights.T  # (input_dim, 1)
        
        return pooled.flatten()
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)