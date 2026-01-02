"""
GraphSAGE (Graph Sample and Aggregate) Implementation.

Implements GraphSAGE for scalable graph learning with sampling.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


class GraphSAGELayer:
    """
    Single GraphSAGE layer with neighbor sampling and aggregation.
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 aggregator_type: str = 'mean',
                 normalize: bool = True,
                 dropout_rate: float = 0.1):
        """
        Initialize GraphSAGE layer.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            aggregator_type: Type of aggregator ('mean', 'max', 'lstm', 'pool')
            normalize: Whether to L2 normalize outputs
            dropout_rate: Dropout rate
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregator_type = aggregator_type
        self.normalize = normalize
        self.dropout_rate = dropout_rate
        
        # Initialize weight matrices
        self.W_self = self._xavier_init((output_dim, input_dim))
        self.W_neigh = self._xavier_init((output_dim, input_dim))
        self.b = np.zeros((output_dim, 1))
        
        # Aggregator-specific parameters
        if aggregator_type == 'lstm':
            self._init_lstm_aggregator()
        elif aggregator_type == 'pool':
            self._init_pooling_aggregator()
            
        # Training state
        self.training = True
        
    def _xavier_init(self, shape: Tuple[int, int]) -> np.ndarray:
        """Xavier initialization."""
        fan_in, fan_out = shape[1], shape[0]
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    
    def _init_lstm_aggregator(self):
        """Initialize LSTM aggregator parameters."""
        hidden_dim = self.output_dim
        
        # LSTM gates
        self.lstm_Wi = self._xavier_init((hidden_dim, self.input_dim))
        self.lstm_Wf = self._xavier_init((hidden_dim, self.input_dim))
        self.lstm_Wo = self._xavier_init((hidden_dim, self.input_dim))
        self.lstm_Wc = self._xavier_init((hidden_dim, self.input_dim))
        
        self.lstm_Ui = self._xavier_init((hidden_dim, hidden_dim))
        self.lstm_Uf = self._xavier_init((hidden_dim, hidden_dim))
        self.lstm_Uo = self._xavier_init((hidden_dim, hidden_dim))
        self.lstm_Uc = self._xavier_init((hidden_dim, hidden_dim))
        
        self.lstm_bi = np.zeros((hidden_dim, 1))
        self.lstm_bf = np.zeros((hidden_dim, 1))
        self.lstm_bo = np.zeros((hidden_dim, 1))
        self.lstm_bc = np.zeros((hidden_dim, 1))
        
    def _init_pooling_aggregator(self):
        """Initialize pooling aggregator parameters."""
        self.pool_W = self._xavier_init((self.output_dim, self.input_dim))
        self.pool_b = np.zeros((self.output_dim, 1))
        
    def forward(self, 
                node_features: np.ndarray,
                adjacency_matrix: np.ndarray,
                sample_sizes: List[int] = None) -> np.ndarray:
        """
        Forward pass through GraphSAGE layer.
        
        Args:
            node_features: Node features (n_nodes, input_dim)
            adjacency_matrix: Adjacency matrix (n_nodes, n_nodes)
            sample_sizes: Number of neighbors to sample for each node
            
        Returns:
            Updated node embeddings (n_nodes, output_dim)
        """
        n_nodes = node_features.shape[0]
        
        if sample_sizes is None:
            sample_sizes = [25] * n_nodes  # Default sampling
            
        # Sample neighbors for each node
        neighbor_samples = self._sample_neighbors(adjacency_matrix, sample_sizes)
        
        # Aggregate neighbor features
        aggregated_features = self._aggregate_neighbors(
            node_features, neighbor_samples
        )
        
        # Combine self and neighbor information
        self_features = node_features.T  # (input_dim, n_nodes)
        neigh_features = aggregated_features.T  # (input_dim, n_nodes)
        
        # Apply transformations
        self_transformed = self.W_self @ self_features
        neigh_transformed = self.W_neigh @ neigh_features
        
        # Combine and add bias
        combined = self_transformed + neigh_transformed + self.b
        
        # Apply activation (ReLU)
        activated = np.maximum(0, combined)
        
        # Apply dropout during training
        if self.training and self.dropout_rate > 0:
            dropout_mask = np.random.binomial(
                1, 1 - self.dropout_rate, activated.shape
            ) / (1 - self.dropout_rate)
            activated = activated * dropout_mask
            
        # L2 normalization
        if self.normalize:
            norms = np.linalg.norm(activated, axis=0, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            activated = activated / norms
            
        return activated.T  # (n_nodes, output_dim)
    
    def _sample_neighbors(self, 
                         adjacency_matrix: np.ndarray,
                         sample_sizes: List[int]) -> List[List[int]]:
        """Sample neighbors for each node."""
        n_nodes = adjacency_matrix.shape[0]
        neighbor_samples = []
        
        for node_id in range(n_nodes):
            # Get neighbors
            neighbors = np.where(adjacency_matrix[node_id] > 0)[0].tolist()
            neighbors = [n for n in neighbors if n != node_id]  # Remove self-loops
            
            # Sample neighbors
            sample_size = min(sample_sizes[node_id], len(neighbors))
            
            if sample_size > 0:
                sampled_neighbors = np.random.choice(
                    neighbors, size=sample_size, replace=False
                ).tolist()
            else:
                sampled_neighbors = []
                
            neighbor_samples.append(sampled_neighbors)
            
        return neighbor_samples
    
    def _aggregate_neighbors(self, 
                           node_features: np.ndarray,
                           neighbor_samples: List[List[int]]) -> np.ndarray:
        """Aggregate neighbor features using specified aggregator."""
        n_nodes = node_features.shape[0]
        aggregated = np.zeros((n_nodes, self.input_dim))
        
        for node_id, neighbors in enumerate(neighbor_samples):
            if not neighbors:
                # No neighbors, use zero aggregation
                continue
                
            neighbor_features = node_features[neighbors]  # (n_neighbors, input_dim)
            
            if self.aggregator_type == 'mean':
                aggregated[node_id] = np.mean(neighbor_features, axis=0)
            elif self.aggregator_type == 'max':
                aggregated[node_id] = np.max(neighbor_features, axis=0)
            elif self.aggregator_type == 'lstm':
                aggregated[node_id] = self._lstm_aggregate(neighbor_features)
            elif self.aggregator_type == 'pool':
                aggregated[node_id] = self._pool_aggregate(neighbor_features)
            else:
                # Default to mean
                aggregated[node_id] = np.mean(neighbor_features, axis=0)
                
        return aggregated
    
    def _lstm_aggregate(self, neighbor_features: np.ndarray) -> np.ndarray:
        """LSTM-based aggregation of neighbor features."""
        n_neighbors, input_dim = neighbor_features.shape
        hidden_dim = self.output_dim
        
        # Initialize LSTM state
        h = np.zeros((hidden_dim,))
        c = np.zeros((hidden_dim,))
        
        # Process each neighbor
        for i in range(n_neighbors):
            x = neighbor_features[i].reshape(-1, 1)  # (input_dim, 1)
            h_prev = h.reshape(-1, 1)  # (hidden_dim, 1)
            c_prev = c.reshape(-1, 1)  # (hidden_dim, 1)
            
            # LSTM gates
            i_gate = self._sigmoid(self.lstm_Wi @ x + self.lstm_Ui @ h_prev + self.lstm_bi)
            f_gate = self._sigmoid(self.lstm_Wf @ x + self.lstm_Uf @ h_prev + self.lstm_bf)
            o_gate = self._sigmoid(self.lstm_Wo @ x + self.lstm_Uo @ h_prev + self.lstm_bo)
            c_tilde = np.tanh(self.lstm_Wc @ x + self.lstm_Uc @ h_prev + self.lstm_bc)
            
            # Update states
            c = f_gate * c_prev + i_gate * c_tilde
            h = o_gate * np.tanh(c)
            
            h = h.flatten()
            c = c.flatten()
            
        return h[:self.input_dim]  # Truncate to input_dim
    
    def _pool_aggregate(self, neighbor_features: np.ndarray) -> np.ndarray:
        """Max pooling aggregation with learnable transformation."""
        # Apply learned transformation
        transformed = neighbor_features @ self.pool_W.T + self.pool_b.T
        
        # Apply activation
        activated = np.maximum(0, transformed)  # ReLU
        
        # Max pooling
        pooled = np.max(activated, axis=0)
        
        return pooled[:self.input_dim]  # Truncate to input_dim
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation with numerical stability."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class GraphSAGE:
    """
    GraphSAGE model for scalable graph learning.
    
    Features:
    - Multi-layer GraphSAGE architecture
    - Neighbor sampling for scalability
    - Multiple aggregation functions
    - Inductive learning capability
    - Node and graph-level predictions
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = None,
                 output_dim: int = 64,
                 num_classes: int = 3,
                 aggregator_types: List[str] = None,
                 sample_sizes: List[List[int]] = None,
                 dropout_rate: float = 0.1,
                 normalize_layers: List[bool] = None):
        """
        Initialize GraphSAGE model.
        
        Args:
            input_dim: Input node feature dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Final embedding dimension
            num_classes: Number of output classes
            aggregator_types: Aggregator type for each layer
            sample_sizes: Neighbor sample sizes for each layer
            dropout_rate: Dropout rate
            normalize_layers: Whether to normalize each layer
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Build layer dimensions
        dims = [input_dim] + self.hidden_dims + [output_dim]
        
        # Default aggregator types
        if aggregator_types is None:
            aggregator_types = ['mean'] * len(dims[:-1])
        self.aggregator_types = aggregator_types
        
        # Default sample sizes
        if sample_sizes is None:
            sample_sizes = [[25, 10]] * len(dims[:-1])  # Sample 25 neighbors, then 10\n        self.sample_sizes = sample_sizes\n        \n        # Default normalization\n        if normalize_layers is None:\n            normalize_layers = [True] * len(dims[:-1])\n        self.normalize_layers = normalize_layers\n        \n        # Initialize GraphSAGE layers\n        self.sage_layers = []\n        for i in range(len(dims) - 1):\n            layer = GraphSAGELayer(\n                input_dim=dims[i],\n                output_dim=dims[i + 1],\n                aggregator_type=aggregator_types[i],\n                normalize=normalize_layers[i],\n                dropout_rate=dropout_rate\n            )\n            self.sage_layers.append(layer)\n            \n        # Classification head\n        self.classifier = MLPClassifier(\n            input_dim=output_dim,\n            hidden_dim=output_dim,\n            num_classes=num_classes,\n            dropout_rate=dropout_rate\n        )\n        \n        # Training state\n        self.training = True\n        \n    def forward(self, \n                node_features: np.ndarray,\n                adjacency_matrix: np.ndarray,\n                target_nodes: Optional[List[int]] = None,\n                return_embeddings: bool = False) -> Dict[str, np.ndarray]:\n        \"\"\"\n        Forward pass through GraphSAGE.\n        \n        Args:\n            node_features: Node features (n_nodes, input_dim)\n            adjacency_matrix: Adjacency matrix (n_nodes, n_nodes)\n            target_nodes: Specific nodes to compute embeddings for\n            return_embeddings: Whether to return intermediate embeddings\n            \n        Returns:\n            Dictionary with predictions and embeddings\n        \"\"\"\n        # Forward through GraphSAGE layers\n        h = node_features\n        embeddings = [h]\n        \n        for i, sage_layer in enumerate(self.sage_layers):\n            # Use layer-specific sample sizes\n            layer_sample_sizes = None\n            if i < len(self.sample_sizes):\n                layer_sample_sizes = [self.sample_sizes[i][0]] * h.shape[0]\n                \n            h = sage_layer.forward(h, adjacency_matrix, layer_sample_sizes)\n            embeddings.append(h)\n            \n        # Final node embeddings\n        node_embeddings = h\n        \n        # If target nodes specified, extract their embeddings\n        if target_nodes is not None:\n            target_embeddings = node_embeddings[target_nodes]\n        else:\n            target_embeddings = node_embeddings\n            \n        # Node predictions\n        node_predictions = self.classifier.forward(target_embeddings)\n        \n        # Graph-level pooling\n        graph_embedding = self._pool_graph(node_embeddings)\n        graph_prediction = self.classifier.forward(graph_embedding.reshape(1, -1))\n        \n        result = {\n            'node_predictions': node_predictions,\n            'graph_prediction': graph_prediction.flatten(),\n            'node_embeddings': target_embeddings,\n            'graph_embedding': graph_embedding\n        }\n        \n        if return_embeddings:\n            result['layer_embeddings'] = embeddings\n            \n        return result\n    \n    def _pool_graph(self, node_embeddings: np.ndarray) -> np.ndarray:\n        \"\"\"Pool node embeddings to graph representation.\"\"\"\n        # Simple mean pooling\n        return np.mean(node_embeddings, axis=0)\n    \n    def minibatch_inference(self,\n                           node_features: np.ndarray,\n                           adjacency_matrix: np.ndarray,\n                           target_nodes: List[int],\n                           batch_size: int = 256) -> Dict[str, np.ndarray]:\n        \"\"\"\n        Efficient minibatch inference for large graphs.\n        \n        Args:\n            node_features: All node features\n            adjacency_matrix: Full adjacency matrix\n            target_nodes: Nodes to compute predictions for\n            batch_size: Batch size for processing\n            \n        Returns:\n            Predictions for target nodes\n        \"\"\"\n        self.training = False\n        \n        n_targets = len(target_nodes)\n        all_predictions = []\n        all_embeddings = []\n        \n        # Process in batches\n        for i in range(0, n_targets, batch_size):\n            batch_targets = target_nodes[i:i + batch_size]\n            \n            # Sample subgraph for this batch\n            subgraph_nodes, subgraph_adj, subgraph_features = self._sample_subgraph(\n                node_features, adjacency_matrix, batch_targets\n            )\n            \n            # Remap target nodes to subgraph indices\n            node_mapping = {old: new for new, old in enumerate(subgraph_nodes)}\n            batch_targets_remapped = [node_mapping[node] for node in batch_targets]\n            \n            # Forward pass on subgraph\n            result = self.forward(\n                subgraph_features,\n                subgraph_adj,\n                target_nodes=batch_targets_remapped\n            )\n            \n            all_predictions.append(result['node_predictions'])\n            all_embeddings.append(result['node_embeddings'])\n            \n        # Combine results\n        combined_predictions = np.vstack(all_predictions)\n        combined_embeddings = np.vstack(all_embeddings)\n        \n        self.training = True\n        \n        return {\n            'node_predictions': combined_predictions,\n            'node_embeddings': combined_embeddings\n        }\n    \n    def _sample_subgraph(self,\n                        node_features: np.ndarray,\n                        adjacency_matrix: np.ndarray,\n                        target_nodes: List[int],\n                        n_hops: int = 2) -> Tuple[List[int], np.ndarray, np.ndarray]:\n        \"\"\"Sample subgraph around target nodes.\"\"\"\n        # Start with target nodes\n        subgraph_nodes = set(target_nodes)\n        \n        # Expand by n_hops\n        current_nodes = set(target_nodes)\n        \n        for hop in range(n_hops):\n            next_nodes = set()\n            \n            for node in current_nodes:\n                # Sample neighbors\n                neighbors = np.where(adjacency_matrix[node] > 0)[0]\n                \n                # Limit number of neighbors per hop\n                max_neighbors = 25 if hop == 0 else 10\n                if len(neighbors) > max_neighbors:\n                    neighbors = np.random.choice(\n                        neighbors, size=max_neighbors, replace=False\n                    )\n                    \n                next_nodes.update(neighbors)\n                \n            subgraph_nodes.update(next_nodes)\n            current_nodes = next_nodes\n            \n        # Convert to sorted list\n        subgraph_nodes = sorted(list(subgraph_nodes))\n        \n        # Extract subgraph adjacency matrix\n        subgraph_adj = adjacency_matrix[np.ix_(subgraph_nodes, subgraph_nodes)]\n        \n        # Extract node features\n        subgraph_features = node_features[subgraph_nodes]\n        \n        return subgraph_nodes, subgraph_adj, subgraph_features\n    \n    def predict_trading_signal(self,\n                              node_features: np.ndarray,\n                              adjacency_matrix: np.ndarray,\n                              key_addresses: Optional[List[str]] = None) -> Dict[str, float]:\n        \"\"\"\n        Predict trading signal from graph structure.\n        \n        Args:\n            node_features: Node features\n            adjacency_matrix: Adjacency matrix\n            key_addresses: Important addresses to focus on\n            \n        Returns:\n            Trading signal prediction\n        \"\"\"\n        self.training = False\n        \n        # Forward pass\n        result = self.forward(node_features, adjacency_matrix)\n        \n        # Get graph-level prediction\n        graph_pred = result['graph_prediction']\n        graph_probs = self._softmax(graph_pred.reshape(1, -1))[0]\n        \n        # Map to trading actions\n        signal_mapping = ['sell', 'hold', 'buy']\n        predicted_action = signal_mapping[np.argmax(graph_probs)]\n        confidence = np.max(graph_probs)\n        \n        # Analyze key node influences if provided\n        key_influence = 0.0\n        if key_addresses:\n            # This would require address-to-node mapping\n            # For now, use node importance analysis\n            importance_analysis = self.analyze_node_influence(\n                node_features, adjacency_matrix\n            )\n            key_influence = np.mean(importance_analysis['importance_scores'][:10])\n            \n        self.training = True\n        \n        return {\n            'action': predicted_action,\n            'confidence': confidence,\n            'action_probabilities': {\n                'sell': graph_probs[0],\n                'hold': graph_probs[1], \n                'buy': graph_probs[2]\n            },\n            'key_node_influence': key_influence,\n            'graph_embedding': result['graph_embedding']\n        }\n    \n    def analyze_node_influence(self,\n                              node_features: np.ndarray,\n                              adjacency_matrix: np.ndarray) -> Dict[str, np.ndarray]:\n        \"\"\"Analyze node influence on graph prediction.\"\"\"\n        # Baseline prediction\n        baseline_result = self.forward(node_features, adjacency_matrix)\n        baseline_pred = baseline_result['graph_prediction']\n        \n        n_nodes = node_features.shape[0]\n        influence_scores = np.zeros(n_nodes)\n        \n        # Sample nodes for efficiency\n        sample_nodes = np.random.choice(\n            n_nodes, size=min(100, n_nodes), replace=False\n        )\n        \n        for node_id in sample_nodes:\n            # Remove node (set features to zero)\n            modified_features = node_features.copy()\n            modified_features[node_id] = 0\n            \n            # Get prediction without this node\n            modified_result = self.forward(modified_features, adjacency_matrix)\n            modified_pred = modified_result['graph_prediction']\n            \n            # Calculate influence\n            influence_scores[node_id] = np.linalg.norm(baseline_pred - modified_pred)\n            \n        # Normalize\n        if np.max(influence_scores) > 0:\n            influence_scores /= np.max(influence_scores)\n            \n        return {\n            'influence_scores': influence_scores,\n            'top_influential_nodes': np.argsort(influence_scores)[::-1][:20],\n            'baseline_prediction': baseline_pred\n        }\n    \n    def _softmax(self, x: np.ndarray) -> np.ndarray:\n        \"\"\"Softmax with numerical stability.\"\"\"\n        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))\n        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)\n    \n    def get_node_embeddings(self,\n                           node_features: np.ndarray,\n                           adjacency_matrix: np.ndarray,\n                           layer_idx: int = -1) -> np.ndarray:\n        \"\"\"Extract node embeddings from specific layer.\"\"\"\n        self.training = False\n        \n        result = self.forward(\n            node_features, adjacency_matrix, return_embeddings=True\n        )\n        \n        embeddings = result['layer_embeddings'][layer_idx]\n        \n        self.training = True\n        \n        return embeddings\n\n\nclass MLPClassifier:\n    \"\"\"Multi-layer perceptron classifier.\"\"\"\n    \n    def __init__(self,\n                 input_dim: int,\n                 hidden_dim: int,\n                 num_classes: int,\n                 dropout_rate: float = 0.1):\n        self.input_dim = input_dim\n        self.hidden_dim = hidden_dim\n        self.num_classes = num_classes\n        self.dropout_rate = dropout_rate\n        \n        # Initialize parameters\n        self.W1 = self._xavier_init((hidden_dim, input_dim))\n        self.b1 = np.zeros((hidden_dim, 1))\n        \n        self.W2 = self._xavier_init((num_classes, hidden_dim))\n        self.b2 = np.zeros((num_classes, 1))\n        \n        self.training = True\n        \n    def _xavier_init(self, shape: Tuple[int, int]) -> np.ndarray:\n        \"\"\"Xavier initialization.\"\"\"\n        fan_in, fan_out = shape[1], shape[0]\n        limit = np.sqrt(6 / (fan_in + fan_out))\n        return np.random.uniform(-limit, limit, shape)\n    \n    def forward(self, x: np.ndarray) -> np.ndarray:\n        \"\"\"Forward pass.\"\"\"\n        x = x.T  # (input_dim, batch_size)\n        \n        # Hidden layer\n        h = np.maximum(0, self.W1 @ x + self.b1)  # ReLU\n        \n        # Dropout\n        if self.training and self.dropout_rate > 0:\n            dropout_mask = np.random.binomial(\n                1, 1 - self.dropout_rate, h.shape\n            ) / (1 - self.dropout_rate)\n            h = h * dropout_mask\n            \n        # Output layer\n        out = self.W2 @ h + self.b2\n        \n        return out.T  # (batch_size, num_classes)"