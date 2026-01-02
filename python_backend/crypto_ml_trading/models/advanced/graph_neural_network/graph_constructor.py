"""
On-Chain Graph Constructor.

Builds transaction graphs from blockchain data for GNN analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import hashlib


@dataclass
class Transaction:
    """Represents a blockchain transaction."""
    tx_hash: str
    from_address: str
    to_address: str
    value: float
    timestamp: int
    gas_fee: float
    block_number: int
    token_transfers: List[Dict] = None
    
    def __post_init__(self):
        if self.token_transfers is None:
            self.token_transfers = []


@dataclass
class GraphNode:
    """Represents a node in the transaction graph."""
    address: str
    node_id: int
    features: np.ndarray
    node_type: str  # 'wallet', 'exchange', 'contract', 'whale'
    
    
@dataclass
class GraphEdge:
    """Represents an edge in the transaction graph."""
    from_node: int
    to_node: int
    features: np.ndarray
    edge_type: str  # 'transfer', 'contract_call', 'exchange'
    weight: float
    timestamp: int


class OnChainGraphConstructor:
    """
    Constructs transaction graphs from on-chain data.
    
    Features:
    - Multi-layered graph construction (ETH transfers, token transfers, contract calls)
    - Node feature engineering (balance, transaction patterns, centrality)
    - Edge feature engineering (value, frequency, recency)
    - Temporal graph snapshots
    - Address clustering and classification
    """
    
    def __init__(self,
                 min_transaction_value: float = 0.01,
                 max_nodes: int = 10000,
                 time_window_hours: int = 24,
                 include_contracts: bool = True):
        """
        Initialize graph constructor.
        
        Args:
            min_transaction_value: Minimum transaction value to include
            max_nodes: Maximum number of nodes in graph
            time_window_hours: Time window for graph construction
            include_contracts: Whether to include smart contracts
        """
        self.min_transaction_value = min_transaction_value
        self.max_nodes = max_nodes
        self.time_window_hours = time_window_hours
        self.include_contracts = include_contracts
        
        # Graph data structures
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.adjacency_matrix: Optional[np.ndarray] = None
        self.node_features: Optional[np.ndarray] = None
        self.edge_features: Optional[np.ndarray] = None
        
        # Address classification
        self.known_exchanges = set()
        self.known_whales = set()
        self.contract_addresses = set()
        
        # Statistics
        self.stats = defaultdict(int)
        
    def construct_graph(self, transactions: List[Transaction],
                       current_timestamp: int) -> Dict[str, np.ndarray]:
        """
        Construct transaction graph from list of transactions.
        
        Args:
            transactions: List of blockchain transactions
            current_timestamp: Current timestamp for temporal filtering
            
        Returns:
            Graph data dictionary
        """
        # Filter transactions by time window
        start_time = current_timestamp - (self.time_window_hours * 3600)
        filtered_txs = [
            tx for tx in transactions 
            if tx.timestamp >= start_time and tx.value >= self.min_transaction_value
        ]
        
        print(f"Constructing graph from {len(filtered_txs)} transactions")
        
        # Build address set and classify addresses
        self._classify_addresses(filtered_txs)
        
        # Create nodes
        self._create_nodes(filtered_txs)
        
        # Create edges
        self._create_edges(filtered_txs)
        
        # Limit graph size
        self._limit_graph_size()
        
        # Compute node features
        self._compute_node_features(filtered_txs)
        
        # Compute edge features
        self._compute_edge_features()
        
        # Build adjacency matrix
        self._build_adjacency_matrix()
        
        return self._get_graph_data()
    
    def _classify_addresses(self, transactions: List[Transaction]):
        """Classify addresses into different types."""
        address_stats = defaultdict(lambda: {
            'tx_count': 0,
            'total_value': 0,
            'unique_counterparts': set(),
            'is_contract': False
        })
        
        # Collect address statistics
        for tx in transactions:
            # From address
            address_stats[tx.from_address]['tx_count'] += 1
            address_stats[tx.from_address]['total_value'] += tx.value
            address_stats[tx.from_address]['unique_counterparts'].add(tx.to_address)
            
            # To address
            address_stats[tx.to_address]['tx_count'] += 1
            address_stats[tx.to_address]['total_value'] += tx.value
            address_stats[tx.to_address]['unique_counterparts'].add(tx.from_address)
            
            # Check for contract addresses (simplified heuristic)
            if tx.gas_fee > 0.01:  # High gas suggests contract interaction
                address_stats[tx.to_address]['is_contract'] = True
                
        # Classify addresses
        for address, stats in address_stats.items():
            # Exchange detection (high transaction count + many counterparts)
            if (stats['tx_count'] > 100 and 
                len(stats['unique_counterparts']) > 50):
                self.known_exchanges.add(address)
                
            # Whale detection (high transaction value)
            if stats['total_value'] > 1000:  # > 1000 ETH
                self.known_whales.add(address)
                
            # Contract detection
            if stats['is_contract']:
                self.contract_addresses.add(address)
                
        self.stats['exchanges'] = len(self.known_exchanges)
        self.stats['whales'] = len(self.known_whales)
        self.stats['contracts'] = len(self.contract_addresses)
        
    def _create_nodes(self, transactions: List[Transaction]):
        """Create graph nodes from unique addresses."""
        unique_addresses = set()
        
        for tx in transactions:
            unique_addresses.add(tx.from_address)
            unique_addresses.add(tx.to_address)
            
        # Remove contracts if not including them
        if not self.include_contracts:
            unique_addresses -= self.contract_addresses
            
        # Create nodes
        node_id = 0
        for address in unique_addresses:
            if len(self.nodes) >= self.max_nodes:
                break
                
            # Determine node type
            if address in self.known_exchanges:
                node_type = 'exchange'
            elif address in self.known_whales:
                node_type = 'whale'
            elif address in self.contract_addresses:
                node_type = 'contract'
            else:
                node_type = 'wallet'
                
            self.nodes[address] = GraphNode(
                address=address,
                node_id=node_id,
                features=np.zeros(10),  # Will be computed later
                node_type=node_type
            )
            node_id += 1
            
        self.stats['total_nodes'] = len(self.nodes)
        
    def _create_edges(self, transactions: List[Transaction]):
        """Create graph edges from transactions."""
        edge_aggregation = defaultdict(lambda: {
            'value_sum': 0,
            'tx_count': 0,
            'gas_sum': 0,
            'latest_timestamp': 0,
            'edge_type': 'transfer'
        })
        
        # Aggregate transactions between same address pairs
        for tx in transactions:
            if (tx.from_address not in self.nodes or 
                tx.to_address not in self.nodes):
                continue
                
            from_id = self.nodes[tx.from_address].node_id
            to_id = self.nodes[tx.to_address].node_id
            
            edge_key = (from_id, to_id)
            
            edge_aggregation[edge_key]['value_sum'] += tx.value
            edge_aggregation[edge_key]['tx_count'] += 1
            edge_aggregation[edge_key]['gas_sum'] += tx.gas_fee
            edge_aggregation[edge_key]['latest_timestamp'] = max(
                edge_aggregation[edge_key]['latest_timestamp'],
                tx.timestamp
            )
            
            # Determine edge type
            if tx.to_address in self.known_exchanges:
                edge_aggregation[edge_key]['edge_type'] = 'exchange'
            elif tx.to_address in self.contract_addresses:
                edge_aggregation[edge_key]['edge_type'] = 'contract_call'
                
        # Create edges
        for (from_id, to_id), data in edge_aggregation.items():
            edge = GraphEdge(
                from_node=from_id,
                to_node=to_id,
                features=np.array([
                    data['value_sum'],
                    data['tx_count'],
                    data['gas_sum'],
                    data['latest_timestamp']
                ]),
                edge_type=data['edge_type'],
                weight=data['value_sum'],
                timestamp=data['latest_timestamp']
            )
            self.edges.append(edge)
            
        self.stats['total_edges'] = len(self.edges)
        
    def _limit_graph_size(self):
        """Limit graph size by keeping most active nodes."""
        if len(self.nodes) <= self.max_nodes:
            return
            
        # Calculate node activity scores
        node_activity = defaultdict(float)
        
        for edge in self.edges:
            node_activity[edge.from_node] += edge.weight
            node_activity[edge.to_node] += edge.weight
            
        # Sort nodes by activity
        sorted_nodes = sorted(
            node_activity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Keep top nodes
        keep_node_ids = set([node_id for node_id, _ in sorted_nodes[:self.max_nodes]])
        
        # Filter nodes
        new_nodes = {}
        node_id_mapping = {}
        new_node_id = 0
        
        for address, node in self.nodes.items():
            if node.node_id in keep_node_ids:
                node_id_mapping[node.node_id] = new_node_id
                node.node_id = new_node_id
                new_nodes[address] = node
                new_node_id += 1
                
        self.nodes = new_nodes
        
        # Filter and remap edges
        new_edges = []
        for edge in self.edges:
            if (edge.from_node in node_id_mapping and 
                edge.to_node in node_id_mapping):
                edge.from_node = node_id_mapping[edge.from_node]
                edge.to_node = node_id_mapping[edge.to_node]
                new_edges.append(edge)
                
        self.edges = new_edges
        
    def _compute_node_features(self, transactions: List[Transaction]):
        """Compute comprehensive node features."""
        n_nodes = len(self.nodes)
        n_features = 15
        
        # Initialize feature matrix
        features = np.zeros((n_nodes, n_features))
        
        # Address to node_id mapping
        addr_to_id = {node.address: node.node_id for node in self.nodes.values()}
        
        # Compute transaction-based features
        for tx in transactions:
            if (tx.from_address not in addr_to_id or 
                tx.to_address not in addr_to_id):
                continue
                
            from_id = addr_to_id[tx.from_address]
            to_id = addr_to_id[tx.to_address]
            
            # Outgoing features for sender
            features[from_id, 0] += tx.value  # Total sent
            features[from_id, 1] += 1         # Outgoing tx count
            features[from_id, 2] += tx.gas_fee # Gas spent
            
            # Incoming features for receiver
            features[to_id, 3] += tx.value     # Total received
            features[to_id, 4] += 1            # Incoming tx count
            
        # Compute derived features
        for node_id in range(n_nodes):
            # Net flow
            features[node_id, 5] = features[node_id, 3] - features[node_id, 0]
            
            # Average transaction sizes
            if features[node_id, 1] > 0:
                features[node_id, 6] = features[node_id, 0] / features[node_id, 1]
            if features[node_id, 4] > 0:
                features[node_id, 7] = features[node_id, 3] / features[node_id, 4]
                
            # Activity ratio
            total_txs = features[node_id, 1] + features[node_id, 4]
            if total_txs > 0:
                features[node_id, 8] = features[node_id, 1] / total_txs
                
        # Compute graph centrality features
        centrality_features = self._compute_centrality_features()
        features[:, 9:12] = centrality_features
        
        # One-hot encode node types
        type_encoding = {
            'wallet': [1, 0, 0, 0],
            'exchange': [0, 1, 0, 0],
            'whale': [0, 0, 1, 0],
            'contract': [0, 0, 0, 1]
        }
        
        for node in self.nodes.values():
            encoding = type_encoding.get(node.node_type, [1, 0, 0, 0])
            features[node.node_id, 11:15] = encoding
            
        # Store features in nodes
        for node in self.nodes.values():
            node.features = features[node.node_id]
            
        self.node_features = features
        
    def _compute_centrality_features(self) -> np.ndarray:
        """Compute graph centrality measures."""
        n_nodes = len(self.nodes)
        centrality = np.zeros((n_nodes, 3))
        
        # Build adjacency matrix for centrality calculation
        adj_matrix = np.zeros((n_nodes, n_nodes))
        for edge in self.edges:
            adj_matrix[edge.from_node, edge.to_node] = edge.weight
            
        # Degree centrality
        out_degree = np.sum(adj_matrix > 0, axis=1)
        in_degree = np.sum(adj_matrix > 0, axis=0)
        centrality[:, 0] = out_degree + in_degree
        
        # Weighted degree centrality
        centrality[:, 1] = np.sum(adj_matrix, axis=1) + np.sum(adj_matrix, axis=0)
        
        # PageRank-style centrality (simplified)
        centrality[:, 2] = self._compute_pagerank(adj_matrix)
        
        # Normalize features
        for i in range(3):
            if np.max(centrality[:, i]) > 0:
                centrality[:, i] /= np.max(centrality[:, i])
                
        return centrality
    
    def _compute_pagerank(self, adj_matrix: np.ndarray, 
                         damping: float = 0.85, 
                         max_iter: int = 100) -> np.ndarray:
        """Compute PageRank centrality."""
        n = adj_matrix.shape[0]
        
        # Normalize adjacency matrix
        row_sums = np.sum(adj_matrix, axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = adj_matrix / row_sums[:, np.newaxis]
        
        # Initialize PageRank values
        pagerank = np.ones(n) / n
        
        # Power iteration
        for _ in range(max_iter):
            prev_pagerank = pagerank.copy()
            pagerank = (1 - damping) / n + damping * transition_matrix.T @ pagerank
            
            # Check convergence
            if np.allclose(pagerank, prev_pagerank, atol=1e-6):
                break
                
        return pagerank
    
    def _compute_edge_features(self):
        """Compute edge feature matrix."""
        if not self.edges:
            self.edge_features = np.array([]).reshape(0, 6)
            return
            
        n_edges = len(self.edges)
        features = np.zeros((n_edges, 6))
        
        current_time = max(edge.timestamp for edge in self.edges)
        
        for i, edge in enumerate(self.edges):
            # Basic features
            features[i, 0] = np.log1p(edge.weight)  # Log transaction value
            features[i, 1] = edge.features[1]       # Transaction count
            features[i, 2] = edge.features[2]       # Gas fees
            
            # Temporal features
            time_diff = current_time - edge.timestamp
            features[i, 3] = time_diff / 3600       # Hours since last transaction
            
            # Edge type encoding
            type_encoding = {
                'transfer': 0,
                'exchange': 1,
                'contract_call': 2
            }
            features[i, 4] = type_encoding.get(edge.edge_type, 0)
            
            # Normalized weight
            features[i, 5] = edge.weight
            
        # Normalize weight feature
        if np.max(features[:, 5]) > 0:
            features[:, 5] /= np.max(features[:, 5])
            
        self.edge_features = features
        
    def _build_adjacency_matrix(self):
        """Build graph adjacency matrix."""
        n_nodes = len(self.nodes)
        self.adjacency_matrix = np.zeros((n_nodes, n_nodes))
        
        for edge in self.edges:
            self.adjacency_matrix[edge.from_node, edge.to_node] = edge.weight
            
    def _get_graph_data(self) -> Dict[str, np.ndarray]:
        """Return graph data in standard format."""
        # Create edge index for GNN frameworks
        edge_index = np.array([[edge.from_node, edge.to_node] for edge in self.edges]).T
        
        return {
            'node_features': self.node_features,
            'edge_index': edge_index,
            'edge_features': self.edge_features,
            'adjacency_matrix': self.adjacency_matrix,
            'node_types': np.array([
                ['wallet', 'exchange', 'whale', 'contract'].index(node.node_type)
                for node in sorted(self.nodes.values(), key=lambda x: x.node_id)
            ]),
            'graph_stats': dict(self.stats)
        }
    
    def get_subgraph(self, center_addresses: List[str], 
                    hop_distance: int = 2) -> Dict[str, np.ndarray]:
        """Extract subgraph around specific addresses."""
        if not center_addresses:
            return self._get_graph_data()
            
        # Find center nodes
        center_node_ids = set()
        for address in center_addresses:
            if address in self.nodes:
                center_node_ids.add(self.nodes[address].node_id)
                
        if not center_node_ids:
            return self._get_graph_data()
            
        # Find nodes within hop distance
        subgraph_nodes = center_node_ids.copy()
        
        for hop in range(hop_distance):
            new_nodes = set()
            for edge in self.edges:
                if edge.from_node in subgraph_nodes:
                    new_nodes.add(edge.to_node)
                if edge.to_node in subgraph_nodes:
                    new_nodes.add(edge.from_node)
            subgraph_nodes.update(new_nodes)
            
        # Extract subgraph
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(subgraph_nodes))}
        
        # Filter nodes
        subgraph_node_features = []
        subgraph_node_types = []
        
        for old_id in sorted(subgraph_nodes):
            for node in self.nodes.values():
                if node.node_id == old_id:
                    subgraph_node_features.append(node.features)
                    subgraph_node_types.append(
                        ['wallet', 'exchange', 'whale', 'contract'].index(node.node_type)
                    )
                    break
                    
        # Filter edges
        subgraph_edges = []
        subgraph_edge_features = []
        
        for i, edge in enumerate(self.edges):
            if (edge.from_node in subgraph_nodes and 
                edge.to_node in subgraph_nodes):
                subgraph_edges.append([
                    node_mapping[edge.from_node],
                    node_mapping[edge.to_node]
                ])
                subgraph_edge_features.append(self.edge_features[i])
                
        return {
            'node_features': np.array(subgraph_node_features),
            'edge_index': np.array(subgraph_edges).T if subgraph_edges else np.array([]).reshape(2, 0),
            'edge_features': np.array(subgraph_edge_features) if subgraph_edge_features else np.array([]).reshape(0, 6),
            'node_types': np.array(subgraph_node_types),
            'center_nodes': [node_mapping[node_id] for node_id in center_node_ids if node_id in node_mapping]
        }
    
    def analyze_graph_properties(self) -> Dict[str, float]:
        """Analyze graph topological properties."""
        if not self.adjacency_matrix.size:
            return {}
            
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        
        # Basic properties
        density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
        
        # Degree distribution
        degrees = np.sum(self.adjacency_matrix > 0, axis=1) + np.sum(self.adjacency_matrix > 0, axis=0)
        avg_degree = np.mean(degrees)
        degree_std = np.std(degrees)
        
        # Clustering coefficient (simplified)
        clustering_coeff = self._compute_clustering_coefficient()
        
        # Connected components (simplified)
        n_components = self._count_connected_components()
        
        return {
            'num_nodes': n_nodes,
            'num_edges': n_edges,
            'density': density,
            'avg_degree': avg_degree,
            'degree_std': degree_std,
            'clustering_coefficient': clustering_coeff,
            'num_components': n_components,
            'avg_path_length': self._estimate_avg_path_length()
        }
    
    def _compute_clustering_coefficient(self) -> float:
        """Compute average clustering coefficient."""
        if len(self.nodes) < 3:
            return 0.0
            
        clustering_coeffs = []
        adj = (self.adjacency_matrix > 0).astype(int)
        
        for node_id in range(len(self.nodes)):
            neighbors = np.where(adj[node_id] + adj[:, node_id] > 0)[0]
            neighbors = neighbors[neighbors != node_id]
            
            if len(neighbors) < 2:
                continue
                
            # Count triangles
            triangles = 0
            possible_triangles = len(neighbors) * (len(neighbors) - 1) // 2
            
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if adj[neighbors[i], neighbors[j]] > 0:
                        triangles += 1
                        
            if possible_triangles > 0:
                clustering_coeffs.append(triangles / possible_triangles)
                
        return np.mean(clustering_coeffs) if clustering_coeffs else 0.0
    
    def _count_connected_components(self) -> int:
        """Count connected components using DFS."""
        if not self.nodes:
            return 0
            
        visited = set()
        components = 0
        adj = (self.adjacency_matrix > 0).astype(int)
        
        def dfs(node):
            visited.add(node)
            for neighbor in range(len(self.nodes)):
                if (neighbor not in visited and 
                    (adj[node, neighbor] > 0 or adj[neighbor, node] > 0)):
                    dfs(neighbor)
                    
        for node_id in range(len(self.nodes)):
            if node_id not in visited:
                dfs(node_id)
                components += 1
                
        return components
    
    def _estimate_avg_path_length(self) -> float:
        """Estimate average path length using sampling."""
        if len(self.nodes) < 2:
            return 0.0
            
        # Sample random node pairs and compute shortest paths
        n_samples = min(1000, len(self.nodes) * (len(self.nodes) - 1) // 2)
        path_lengths = []
        
        for _ in range(n_samples):
            source = np.random.randint(0, len(self.nodes))
            target = np.random.randint(0, len(self.nodes))
            
            if source != target:
                path_length = self._shortest_path_length(source, target)
                if path_length > 0:
                    path_lengths.append(path_length)
                    
        return np.mean(path_lengths) if path_lengths else float('inf')
    
    def _shortest_path_length(self, source: int, target: int) -> int:
        """Compute shortest path length using BFS."""
        if source == target:
            return 0
            
        visited = set([source])
        queue = [(source, 0)]
        adj = (self.adjacency_matrix > 0).astype(int)
        
        while queue:
            node, distance = queue.pop(0)
            
            for neighbor in range(len(self.nodes)):
                if (neighbor not in visited and 
                    (adj[node, neighbor] > 0 or adj[neighbor, node] > 0)):
                    if neighbor == target:
                        return distance + 1
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
                    
        return -1  # No path found