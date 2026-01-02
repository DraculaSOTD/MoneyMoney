"""
On-Chain Analyzer - Comprehensive Graph Analysis for Trading Signals.

Integrates all GNN components for complete on-chain analysis and trading signal generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.advanced.graph_neural_network.graph_constructor import OnChainGraphConstructor, Transaction
from models.advanced.graph_neural_network.gcn_model import GraphConvolutionalNetwork
from models.advanced.graph_neural_network.graphsage_model import GraphSAGE
from models.advanced.graph_neural_network.temporal_gnn import TemporalGraphNetwork, TemporalSnapshot


@dataclass
class OnChainSignal:
    """On-chain trading signal."""
    timestamp: int
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    signal_strength: float
    contributing_factors: Dict[str, float]
    graph_metrics: Dict[str, float]
    whale_activity: Dict[str, Any]
    exchange_flows: Dict[str, float]
    network_health: Dict[str, float]
    anomaly_score: float


class OnChainAnalyzer:
    """
    Comprehensive on-chain analyzer for cryptocurrency trading.
    
    Features:
    - Multi-model ensemble (GCN, GraphSAGE, Temporal GNN)
    - Real-time transaction graph analysis
    - Whale movement detection
    - Exchange flow analysis
    - Network health monitoring
    - Anomaly detection
    - Risk assessment
    """
    
    def __init__(self,
                 min_transaction_value: float = 0.01,
                 max_nodes: int = 5000,
                 time_window_hours: int = 24,
                 ensemble_weights: Optional[Dict[str, float]] = None):
        """
        Initialize on-chain analyzer.
        
        Args:
            min_transaction_value: Minimum transaction value to consider
            max_nodes: Maximum nodes in graph
            time_window_hours: Analysis time window
            ensemble_weights: Weights for model ensemble
        """
        self.min_transaction_value = min_transaction_value
        self.max_nodes = max_nodes
        self.time_window_hours = time_window_hours
        
        # Model ensemble weights
        self.ensemble_weights = ensemble_weights or {
            'gcn': 0.3,
            'graphsage': 0.4,
            'temporal_gnn': 0.3
        }
        
        # Initialize components
        self.graph_constructor = OnChainGraphConstructor(
            min_transaction_value=min_transaction_value,
            max_nodes=max_nodes,
            time_window_hours=time_window_hours
        )
        
        # Initialize models (will be configured after first graph)
        self.gcn_model = None
        self.graphsage_model = None
        self.temporal_gnn = None
        
        # Data storage
        self.transaction_history: List[Transaction] = []
        self.signal_history: List[OnChainSignal] = []
        self.graph_snapshots: List[Dict] = []
        
        # Analytics cache
        self.whale_addresses = set()
        self.exchange_addresses = set()
        self.suspicious_addresses = set()
        
        # Thresholds and parameters
        self.whale_threshold = 100.0  # ETH
        self.anomaly_threshold = 2.5  # Z-score
        self.confidence_threshold = 0.6
        
    def add_transactions(self, transactions: List[Transaction]):
        """Add new transactions for analysis."""
        self.transaction_history.extend(transactions)
        
        # Keep recent transactions only
        cutoff_time = datetime.now().timestamp() - (self.time_window_hours * 3600 * 7)  # 7 days
        self.transaction_history = [
            tx for tx in self.transaction_history 
            if tx.timestamp > cutoff_time
        ]
        
        # Update address classifications
        self._update_address_classifications(transactions)
        
    def _update_address_classifications(self, transactions: List[Transaction]):
        """Update whale and exchange address classifications."""
        address_volumes = {}
        
        for tx in transactions:
            # Track volumes
            if tx.from_address not in address_volumes:
                address_volumes[tx.from_address] = 0
            if tx.to_address not in address_volumes:
                address_volumes[tx.to_address] = 0
                
            address_volumes[tx.from_address] += tx.value
            address_volumes[tx.to_address] += tx.value
            
        # Classify whales
        for address, volume in address_volumes.items():
            if volume > self.whale_threshold:
                self.whale_addresses.add(address)
                
        # Simple exchange detection (high transaction count + volume)
        address_tx_counts = {}
        for tx in transactions:
            address_tx_counts[tx.from_address] = address_tx_counts.get(tx.from_address, 0) + 1
            address_tx_counts[tx.to_address] = address_tx_counts.get(tx.to_address, 0) + 1
            
        for address, tx_count in address_tx_counts.items():
            if (tx_count > 50 and 
                address_volumes.get(address, 0) > 500):
                self.exchange_addresses.add(address)
                
    def analyze_current_state(self, current_timestamp: Optional[int] = None) -> OnChainSignal:
        """
        Analyze current on-chain state and generate trading signal.
        
        Args:
            current_timestamp: Current timestamp (default: now)
            
        Returns:
            On-chain trading signal
        """
        if current_timestamp is None:
            current_timestamp = int(datetime.now().timestamp())
            
        # Filter recent transactions
        recent_txs = [
            tx for tx in self.transaction_history
            if tx.timestamp > current_timestamp - (self.time_window_hours * 3600)
        ]
        
        if len(recent_txs) < 10:
            return self._create_neutral_signal(current_timestamp, \"Insufficient transaction data\")\n            \n        # Construct transaction graph\n        graph_data = self.graph_constructor.construct_graph(recent_txs, current_timestamp)\n        \n        # Initialize models if needed\n        self._initialize_models(graph_data)\n        \n        # Generate predictions from each model\n        predictions = self._get_ensemble_predictions(graph_data)\n        \n        # Analyze specific patterns\n        whale_analysis = self._analyze_whale_activity(recent_txs, graph_data)\n        exchange_analysis = self._analyze_exchange_flows(recent_txs, graph_data)\n        network_analysis = self._analyze_network_health(graph_data)\n        anomaly_analysis = self._detect_anomalies(graph_data)\n        \n        # Combine analyses\n        signal = self._combine_analyses(\n            current_timestamp,\n            predictions,\n            whale_analysis,\n            exchange_analysis,\n            network_analysis,\n            anomaly_analysis\n        )\n        \n        # Store for history\n        self.signal_history.append(signal)\n        self.graph_snapshots.append(graph_data)\n        \n        # Update temporal GNN if available\n        if self.temporal_gnn:\n            snapshot = TemporalSnapshot(\n                timestamp=current_timestamp,\n                node_features=graph_data['node_features'],\n                edge_index=graph_data['edge_index'],\n                edge_features=graph_data['edge_features'],\n                adjacency_matrix=graph_data['adjacency_matrix'],\n                graph_stats=graph_data['graph_stats']\n            )\n            self.temporal_gnn.add_snapshot(snapshot)\n            \n        return signal\n    \n    def _initialize_models(self, graph_data: Dict[str, np.ndarray]):\n        \"\"\"Initialize models based on graph structure.\"\"\"\n        if graph_data['node_features'].shape[0] == 0:\n            return\n            \n        node_feature_dim = graph_data['node_features'].shape[1]\n        edge_feature_dim = graph_data['edge_features'].shape[1] if graph_data['edge_features'].shape[0] > 0 else 6\n        \n        # Initialize GCN\n        if self.gcn_model is None:\n            self.gcn_model = GraphConvolutionalNetwork(\n                input_dim=node_feature_dim,\n                hidden_dims=[64, 32],\n                output_dim=16,\n                num_classes=3,\n                dropout_rate=0.1\n            )\n            \n        # Initialize GraphSAGE\n        if self.graphsage_model is None:\n            self.graphsage_model = GraphSAGE(\n                input_dim=node_feature_dim,\n                hidden_dims=[64, 32],\n                output_dim=16,\n                num_classes=3,\n                aggregator_types=['mean', 'mean'],\n                dropout_rate=0.1\n            )\n            \n        # Initialize Temporal GNN\n        if self.temporal_gnn is None:\n            self.temporal_gnn = TemporalGraphNetwork(\n                node_feature_dim=node_feature_dim,\n                edge_feature_dim=edge_feature_dim,\n                hidden_dim=32,\n                temporal_dim=16,\n                num_snapshots=24\n            )\n            \n    def _get_ensemble_predictions(self, graph_data: Dict[str, np.ndarray]) -> Dict[str, Dict]:\n        \"\"\"Get predictions from all models.\"\"\"\n        predictions = {}\n        \n        if graph_data['node_features'].shape[0] == 0:\n            return {}\n            \n        # GCN prediction\n        if self.gcn_model:\n            try:\n                gcn_result = self.gcn_model.predict_graph_signal(\n                    graph_data['node_features'],\n                    graph_data['adjacency_matrix']\n                )\n                predictions['gcn'] = gcn_result\n            except Exception as e:\n                print(f\"GCN prediction error: {e}\")\n                \n        # GraphSAGE prediction\n        if self.graphsage_model:\n            try:\n                sage_result = self.graphsage_model.predict_trading_signal(\n                    graph_data['node_features'],\n                    graph_data['adjacency_matrix']\n                )\n                predictions['graphsage'] = sage_result\n            except Exception as e:\n                print(f\"GraphSAGE prediction error: {e}\")\n                \n        # Temporal GNN prediction\n        if self.temporal_gnn and len(self.temporal_gnn.snapshots) > 0:\n            try:\n                temporal_result = self.temporal_gnn.predict_trading_signal()\n                predictions['temporal_gnn'] = temporal_result\n            except Exception as e:\n                print(f\"Temporal GNN prediction error: {e}\")\n                \n        return predictions\n    \n    def _analyze_whale_activity(self, \n                               transactions: List[Transaction],\n                               graph_data: Dict[str, np.ndarray]) -> Dict[str, Any]:\n        \"\"\"Analyze whale movement patterns.\"\"\"\n        whale_activity = {\n            'large_transactions': [],\n            'whale_accumulation': 0.0,\n            'whale_distribution': 0.0,\n            'new_whale_activity': False,\n            'whale_sentiment': 'neutral'\n        }\n        \n        # Analyze large transactions\n        large_tx_threshold = self.whale_threshold\n        \n        accumulation = 0.0\n        distribution = 0.0\n        \n        for tx in transactions:\n            if tx.value > large_tx_threshold:\n                whale_activity['large_transactions'].append({\n                    'from': tx.from_address,\n                    'to': tx.to_address,\n                    'value': tx.value,\n                    'timestamp': tx.timestamp\n                })\n                \n                # Check if going to/from exchanges\n                if tx.to_address in self.exchange_addresses:\n                    distribution += tx.value  # Whale selling to exchange\n                elif tx.from_address in self.exchange_addresses:\n                    accumulation += tx.value  # Whale buying from exchange\n                    \n        whale_activity['whale_accumulation'] = accumulation\n        whale_activity['whale_distribution'] = distribution\n        \n        # Determine whale sentiment\n        net_flow = accumulation - distribution\n        if net_flow > large_tx_threshold:\n            whale_activity['whale_sentiment'] = 'bullish'\n        elif net_flow < -large_tx_threshold:\n            whale_activity['whale_sentiment'] = 'bearish'\n            \n        # Check for new whale activity\n        recent_whales = set()\n        recent_threshold = transactions[-1].timestamp - 3600  # Last hour\n        \n        for tx in transactions:\n            if tx.timestamp > recent_threshold and tx.value > large_tx_threshold:\n                if (tx.from_address not in self.whale_addresses and \n                    tx.to_address not in self.whale_addresses):\n                    whale_activity['new_whale_activity'] = True\n                    recent_whales.add(tx.from_address)\n                    recent_whales.add(tx.to_address)\n                    \n        whale_activity['new_whales'] = list(recent_whales)\n        \n        return whale_activity\n    \n    def _analyze_exchange_flows(self, \n                               transactions: List[Transaction],\n                               graph_data: Dict[str, np.ndarray]) -> Dict[str, float]:\n        \"\"\"Analyze flows to/from exchanges.\"\"\"\n        exchange_flows = {\n            'total_inflow': 0.0,\n            'total_outflow': 0.0,\n            'net_flow': 0.0,\n            'flow_ratio': 0.0,\n            'exchange_pressure': 'neutral'\n        }\n        \n        inflow = 0.0\n        outflow = 0.0\n        \n        for tx in transactions:\n            if tx.to_address in self.exchange_addresses:\n                inflow += tx.value\n            elif tx.from_address in self.exchange_addresses:\n                outflow += tx.value\n                \n        exchange_flows['total_inflow'] = inflow\n        exchange_flows['total_outflow'] = outflow\n        exchange_flows['net_flow'] = inflow - outflow\n        \n        if outflow > 0:\n            exchange_flows['flow_ratio'] = inflow / outflow\n        else:\n            exchange_flows['flow_ratio'] = float('inf') if inflow > 0 else 0\n            \n        # Determine exchange pressure\n        if exchange_flows['net_flow'] > self.whale_threshold:\n            exchange_flows['exchange_pressure'] = 'bearish'  # More inflow = selling pressure\n        elif exchange_flows['net_flow'] < -self.whale_threshold:\n            exchange_flows['exchange_pressure'] = 'bullish'  # More outflow = accumulation\n            \n        return exchange_flows\n    \n    def _analyze_network_health(self, graph_data: Dict[str, np.ndarray]) -> Dict[str, float]:\n        \"\"\"Analyze network health metrics.\"\"\"\n        network_health = {\n            'connectivity': 0.0,\n            'decentralization': 0.0,\n            'activity_level': 0.0,\n            'network_stress': 0.0,\n            'health_score': 0.0\n        }\n        \n        if graph_data['node_features'].shape[0] == 0:\n            return network_health\n            \n        stats = graph_data['graph_stats']\n        \n        # Connectivity (based on density and connected components)\n        connectivity = stats.get('density', 0) * (1.0 / max(1, stats.get('num_components', 1)))\n        network_health['connectivity'] = min(1.0, connectivity * 10)  # Scale to [0,1]\n        \n        # Decentralization (inverse of concentration)\n        if stats.get('num_nodes', 0) > 0:\n            avg_degree = stats.get('avg_degree', 0)\n            max_possible_degree = stats.get('num_nodes', 1) - 1\n            concentration = avg_degree / max(1, max_possible_degree)\n            network_health['decentralization'] = 1.0 - concentration\n        \n        # Activity level (normalized transaction count)\n        num_edges = stats.get('num_edges', 0)\n        num_nodes = stats.get('num_nodes', 1)\n        activity_ratio = num_edges / max(1, num_nodes)\n        network_health['activity_level'] = min(1.0, activity_ratio / 10)  # Normalize\n        \n        # Network stress (high clustering = low stress)\n        clustering_coeff = stats.get('clustering_coefficient', 0)\n        network_health['network_stress'] = 1.0 - clustering_coeff\n        \n        # Overall health score\n        network_health['health_score'] = np.mean([\n            network_health['connectivity'],\n            network_health['decentralization'],\n            network_health['activity_level'],\n            1.0 - network_health['network_stress']\n        ])\n        \n        return network_health\n    \n    def _detect_anomalies(self, graph_data: Dict[str, np.ndarray]) -> Dict[str, float]:\n        \"\"\"Detect anomalies in graph structure.\"\"\"\n        anomaly_analysis = {\n            'structural_anomaly': 0.0,\n            'volume_anomaly': 0.0,\n            'pattern_anomaly': 0.0,\n            'overall_anomaly_score': 0.0\n        }\n        \n        if len(self.graph_snapshots) < 5:  # Need history for comparison\n            return anomaly_analysis\n            \n        current_stats = graph_data['graph_stats']\n        \n        # Extract historical statistics\n        historical_stats = []\n        for snapshot in self.graph_snapshots[-10:]:  # Last 10 snapshots\n            historical_stats.append(snapshot['graph_stats'])\n            \n        # Structural anomaly detection\n        structural_metrics = ['num_nodes', 'num_edges', 'density']\n        structural_scores = []\n        \n        for metric in structural_metrics:\n            current_val = current_stats.get(metric, 0)\n            historical_vals = [s.get(metric, 0) for s in historical_stats]\n            \n            if len(historical_vals) > 1:\n                mean_val = np.mean(historical_vals)\n                std_val = np.std(historical_vals)\n                \n                if std_val > 0:\n                    z_score = abs(current_val - mean_val) / std_val\n                    structural_scores.append(z_score)\n                    \n        if structural_scores:\n            anomaly_analysis['structural_anomaly'] = np.mean(structural_scores)\n            \n        # Volume anomaly (based on total transaction volume)\n        if graph_data['edge_features'].shape[0] > 0:\n            current_volume = np.sum(graph_data['edge_features'][:, 0])  # Sum of log values\n            \n            historical_volumes = []\n            for snapshot in self.graph_snapshots[-10:]:\n                if snapshot['edge_features'].shape[0] > 0:\n                    vol = np.sum(snapshot['edge_features'][:, 0])\n                    historical_volumes.append(vol)\n                    \n            if len(historical_volumes) > 1:\n                vol_mean = np.mean(historical_volumes)\n                vol_std = np.std(historical_volumes)\n                \n                if vol_std > 0:\n                    vol_z_score = abs(current_volume - vol_mean) / vol_std\n                    anomaly_analysis['volume_anomaly'] = vol_z_score\n                    \n        # Pattern anomaly (simplified - based on clustering coefficient change)\n        current_clustering = current_stats.get('clustering_coefficient', 0)\n        historical_clustering = [s.get('clustering_coefficient', 0) for s in historical_stats]\n        \n        if len(historical_clustering) > 1:\n            clust_mean = np.mean(historical_clustering)\n            clust_std = np.std(historical_clustering)\n            \n            if clust_std > 0:\n                clust_z_score = abs(current_clustering - clust_mean) / clust_std\n                anomaly_analysis['pattern_anomaly'] = clust_z_score\n                \n        # Overall anomaly score\n        anomaly_scores = [\n            anomaly_analysis['structural_anomaly'],\n            anomaly_analysis['volume_anomaly'],\n            anomaly_analysis['pattern_anomaly']\n        ]\n        \n        valid_scores = [score for score in anomaly_scores if score > 0]\n        if valid_scores:\n            anomaly_analysis['overall_anomaly_score'] = np.mean(valid_scores)\n            \n        return anomaly_analysis\n    \n    def _combine_analyses(self,\n                         timestamp: int,\n                         predictions: Dict[str, Dict],\n                         whale_analysis: Dict[str, Any],\n                         exchange_analysis: Dict[str, float],\n                         network_analysis: Dict[str, float],\n                         anomaly_analysis: Dict[str, float]) -> OnChainSignal:\n        \"\"\"Combine all analyses into final signal.\"\"\"\n        \n        # Ensemble model predictions\n        action_votes = {'buy': 0, 'sell': 0, 'hold': 0}\n        confidence_scores = []\n        \n        for model_name, prediction in predictions.items():\n            if 'action' in prediction and 'confidence' in prediction:\n                weight = self.ensemble_weights.get(model_name, 0.33)\n                action_votes[prediction['action']] += weight\n                confidence_scores.append(prediction['confidence'] * weight)\n                \n        # Determine ensemble action\n        if action_votes['buy'] == action_votes['sell'] == action_votes['hold'] == 0:\n            ensemble_action = 'hold'\n            ensemble_confidence = 0.5\n        else:\n            ensemble_action = max(action_votes, key=action_votes.get)\n            ensemble_confidence = sum(confidence_scores) if confidence_scores else 0.5\n            \n        # Adjust based on whale activity\n        whale_adjustment = 0.0\n        if whale_analysis['whale_sentiment'] == 'bullish':\n            whale_adjustment = 0.1\n        elif whale_analysis['whale_sentiment'] == 'bearish':\n            whale_adjustment = -0.1\n            \n        # Adjust based on exchange flows\n        exchange_adjustment = 0.0\n        if exchange_analysis['exchange_pressure'] == 'bullish':\n            exchange_adjustment = 0.05\n        elif exchange_analysis['exchange_pressure'] == 'bearish':\n            exchange_adjustment = -0.05\n            \n        # Adjust based on network health\n        health_adjustment = (network_analysis['health_score'] - 0.5) * 0.1\n        \n        # Adjust based on anomalies\n        anomaly_adjustment = -min(0.2, anomaly_analysis['overall_anomaly_score'] * 0.05)\n        \n        # Calculate final signal strength\n        base_strength = action_votes[ensemble_action]\n        total_adjustment = whale_adjustment + exchange_adjustment + health_adjustment + anomaly_adjustment\n        \n        signal_strength = base_strength + total_adjustment\n        \n        # Final action determination\n        final_action = ensemble_action\n        if signal_strength > 0.6:\n            final_action = 'buy'\n        elif signal_strength < 0.4:\n            final_action = 'sell'\n        else:\n            final_action = 'hold'\n            \n        # Adjust confidence based on various factors\n        final_confidence = ensemble_confidence\n        \n        # Reduce confidence if high anomaly\n        if anomaly_analysis['overall_anomaly_score'] > self.anomaly_threshold:\n            final_confidence *= 0.7\n            \n        # Increase confidence if whale activity aligns\n        if ((final_action == 'buy' and whale_analysis['whale_sentiment'] == 'bullish') or\n            (final_action == 'sell' and whale_analysis['whale_sentiment'] == 'bearish')):\n            final_confidence *= 1.2\n            \n        # Cap confidence\n        final_confidence = min(1.0, max(0.0, final_confidence))\n        \n        # Contributing factors\n        contributing_factors = {\n            'model_ensemble': base_strength,\n            'whale_activity': whale_adjustment,\n            'exchange_flows': exchange_adjustment,\n            'network_health': health_adjustment,\n            'anomaly_impact': anomaly_adjustment\n        }\n        \n        return OnChainSignal(\n            timestamp=timestamp,\n            action=final_action,\n            confidence=final_confidence,\n            signal_strength=signal_strength,\n            contributing_factors=contributing_factors,\n            graph_metrics=network_analysis,\n            whale_activity=whale_analysis,\n            exchange_flows=exchange_analysis,\n            network_health=network_analysis,\n            anomaly_score=anomaly_analysis['overall_anomaly_score']\n        )\n    \n    def _create_neutral_signal(self, timestamp: int, reason: str) -> OnChainSignal:\n        \"\"\"Create neutral signal when analysis cannot be performed.\"\"\"\n        return OnChainSignal(\n            timestamp=timestamp,\n            action='hold',\n            confidence=0.5,\n            signal_strength=0.5,\n            contributing_factors={'reason': reason},\n            graph_metrics={},\n            whale_activity={},\n            exchange_flows={},\n            network_health={},\n            anomaly_score=0.0\n        )\n    \n    def get_signal_history(self, hours_back: int = 24) -> List[OnChainSignal]:\n        \"\"\"Get signal history for the last N hours.\"\"\"\n        cutoff_time = datetime.now().timestamp() - (hours_back * 3600)\n        return [\n            signal for signal in self.signal_history \n            if signal.timestamp > cutoff_time\n        ]\n    \n    def analyze_address(self, address: str) -> Dict[str, Any]:\n        \"\"\"Analyze specific address behavior.\"\"\"\n        address_analysis = {\n            'classification': 'unknown',\n            'total_volume': 0.0,\n            'transaction_count': 0,\n            'first_seen': None,\n            'last_seen': None,\n            'counterparties': set(),\n            'risk_score': 0.0,\n            'behavior_pattern': 'normal'\n        }\n        \n        # Classify address\n        if address in self.whale_addresses:\n            address_analysis['classification'] = 'whale'\n        elif address in self.exchange_addresses:\n            address_analysis['classification'] = 'exchange'\n        elif address in self.suspicious_addresses:\n            address_analysis['classification'] = 'suspicious'\n        else:\n            address_analysis['classification'] = 'regular'\n            \n        # Analyze transaction history\n        address_txs = [\n            tx for tx in self.transaction_history\n            if tx.from_address == address or tx.to_address == address\n        ]\n        \n        if address_txs:\n            address_analysis['transaction_count'] = len(address_txs)\n            address_analysis['first_seen'] = min(tx.timestamp for tx in address_txs)\n            address_analysis['last_seen'] = max(tx.timestamp for tx in address_txs)\n            \n            total_volume = 0.0\n            counterparties = set()\n            \n            for tx in address_txs:\n                total_volume += tx.value\n                if tx.from_address == address:\n                    counterparties.add(tx.to_address)\n                else:\n                    counterparties.add(tx.from_address)\n                    \n            address_analysis['total_volume'] = total_volume\n            address_analysis['counterparties'] = counterparties\n            \n            # Risk scoring (simplified)\n            risk_factors = 0\n            \n            # High volume in short time\n            if len(address_txs) > 0:\n                time_span = address_analysis['last_seen'] - address_analysis['first_seen']\n                if time_span > 0:\n                    volume_rate = total_volume / (time_span / 3600)  # ETH per hour\n                    if volume_rate > 10:  # High volume rate\n                        risk_factors += 1\n                        \n            # Many counterparties\n            if len(counterparties) > 100:\n                risk_factors += 1\n                \n            # Interaction with suspicious addresses\n            suspicious_interactions = len(counterparties.intersection(self.suspicious_addresses))\n            if suspicious_interactions > 0:\n                risk_factors += suspicious_interactions\n                \n            address_analysis['risk_score'] = min(1.0, risk_factors / 3.0)\n            \n            # Behavior pattern detection\n            if len(address_txs) > 10:\n                # Check for regular patterns\n                tx_intervals = []\n                sorted_txs = sorted(address_txs, key=lambda x: x.timestamp)\n                \n                for i in range(1, len(sorted_txs)):\n                    interval = sorted_txs[i].timestamp - sorted_txs[i-1].timestamp\n                    tx_intervals.append(interval)\n                    \n                if tx_intervals:\n                    interval_std = np.std(tx_intervals)\n                    interval_mean = np.mean(tx_intervals)\n                    \n                    # Regular pattern (low variance)\n                    if interval_std < interval_mean * 0.2:\n                        address_analysis['behavior_pattern'] = 'regular'\n                    # Burst pattern (high variance)\n                    elif interval_std > interval_mean * 2:\n                        address_analysis['behavior_pattern'] = 'burst'\n                        \n        return address_analysis\n    \n    def get_network_summary(self) -> Dict[str, Any]:\n        \"\"\"Get summary of current network state.\"\"\"\n        if not self.graph_snapshots:\n            return {}\n            \n        latest_snapshot = self.graph_snapshots[-1]\n        \n        summary = {\n            'timestamp': datetime.now().isoformat(),\n            'graph_stats': latest_snapshot['graph_stats'],\n            'total_addresses': len(self.whale_addresses) + len(self.exchange_addresses),\n            'whale_addresses': len(self.whale_addresses),\n            'exchange_addresses': len(self.exchange_addresses),\n            'suspicious_addresses': len(self.suspicious_addresses),\n            'recent_signals': len(self.signal_history[-24:]),  # Last 24 signals\n            'network_health': self._analyze_network_health(latest_snapshot)\n        }\n        \n        # Add trend analysis if enough history\n        if len(self.signal_history) > 10:\n            recent_signals = self.signal_history[-10:]\n            buy_signals = sum(1 for s in recent_signals if s.action == 'buy')\n            sell_signals = sum(1 for s in recent_signals if s.action == 'sell')\n            \n            summary['signal_trend'] = {\n                'buy_ratio': buy_signals / len(recent_signals),\n                'sell_ratio': sell_signals / len(recent_signals),\n                'avg_confidence': np.mean([s.confidence for s in recent_signals]),\n                'avg_anomaly_score': np.mean([s.anomaly_score for s in recent_signals])\n            }\n            \n        return summary"