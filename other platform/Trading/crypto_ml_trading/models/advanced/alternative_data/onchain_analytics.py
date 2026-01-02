"""
On-Chain Analytics for Cryptocurrency Trading.

Implements comprehensive blockchain data analysis and on-chain metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import hashlib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


@dataclass
class BlockchainTransaction:
    """Individual blockchain transaction data."""
    tx_hash: str
    timestamp: datetime
    from_address: str
    to_address: str
    amount: float
    fee: float
    block_height: int
    gas_used: Optional[int] = None
    gas_price: Optional[float] = None
    transaction_type: str = "transfer"  # transfer, contract, defi, etc.


@dataclass
class AddressMetrics:
    """Metrics for a blockchain address."""
    address: str
    timestamp: datetime
    balance: float
    transaction_count: int
    total_received: float
    total_sent: float
    first_seen: datetime
    last_active: datetime
    address_type: str  # whale, exchange, retail, smart_contract
    risk_score: float = 0.0


@dataclass
class NetworkMetrics:
    """Network-level blockchain metrics."""
    timestamp: datetime
    network: str
    hash_rate: Optional[float] = None
    difficulty: Optional[float] = None
    block_time: Optional[float] = None
    transaction_count: int = 0
    active_addresses: int = 0
    new_addresses: int = 0
    total_fees: float = 0.0
    mempool_size: Optional[int] = None
    network_value_transferred: float = 0.0


@dataclass
class WhaleMovement:
    """Large whale transaction movement."""
    timestamp: datetime
    transaction_hash: str
    from_address: str
    to_address: str
    amount: float
    usd_value: Optional[float]
    movement_type: str  # accumulation, distribution, exchange_inflow, exchange_outflow
    whale_category: str  # retail_whale, institutional, exchange
    market_impact_prediction: float = 0.0


@dataclass
class DeFiMetrics:
    """DeFi protocol metrics."""
    timestamp: datetime
    protocol: str
    total_value_locked: float
    trading_volume: float
    unique_users: int
    transaction_count: int
    governance_activity: float
    yield_rates: Dict[str, float] = field(default_factory=dict)
    liquidity_metrics: Dict[str, float] = field(default_factory=dict)


class OnChainAnalytics:
    """
    Advanced on-chain analytics for cryptocurrency trading.
    
    Features:
    - Transaction flow analysis
    - Whale movement tracking
    - Network health monitoring
    - DeFi protocol analytics
    - Exchange flow analysis
    - Address clustering and classification
    - Market impact prediction from on-chain data
    - Supply distribution analysis
    """
    
    def __init__(self,
                 networks: List[str] = None,
                 whale_threshold: float = 1000.0,
                 analysis_window: int = 1440):  # 24 hours in minutes
        """
        Initialize on-chain analytics.
        
        Args:
            networks: List of blockchain networks to analyze
            whale_threshold: Minimum amount to classify as whale transaction
            analysis_window: Analysis window in minutes
        """
        self.networks = networks or ['bitcoin', 'ethereum', 'binance_smart_chain', 'polygon']
        self.whale_threshold = whale_threshold
        self.analysis_window = analysis_window
        
        # Data storage
        self.transactions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))  # network -> transactions
        self.address_metrics: Dict[str, Dict[str, AddressMetrics]] = defaultdict(dict)  # network -> address -> metrics
        self.network_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))  # network -> metrics
        
        # Analysis results
        self.whale_movements: deque = deque(maxlen=1000)
        self.defi_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.flow_analysis: Dict[str, Dict] = defaultdict(dict)
        
        # Exchange addresses (would be populated from external sources)
        self.exchange_addresses: Dict[str, Set[str]] = defaultdict(set)
        self.known_addresses: Dict[str, Dict[str, str]] = defaultdict(dict)  # network -> address -> label
        
        # Analysis caches
        self.supply_distribution: Dict[str, Dict] = defaultdict(dict)
        self.network_health_scores: Dict[str, float] = defaultdict(float)
        
        # Initialize address classification
        self._initialize_address_classification()
    
    def _initialize_address_classification(self) -> None:
        """Initialize address classification system."""
        # This would be populated from external data sources in practice
        # For demo purposes, we'll use some patterns
        
        # Common exchange patterns (simplified)
        exchange_patterns = {
            'binance': ['1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa'],  # Example patterns
            'coinbase': ['3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy'],
            'kraken': ['1AJbsFZ64EpEfS5UAjAfcUG8pH8Jn3rn1F']
        }
        
        for exchange, addresses in exchange_patterns.items():
            for network in self.networks:
                self.exchange_addresses[network].update(addresses)
                for addr in addresses:
                    self.known_addresses[network][addr] = f"{exchange}_exchange"
    
    def add_transaction_data(self,
                           network: str,
                           transactions: List[BlockchainTransaction]) -> None:
        """
        Add blockchain transaction data.
        
        Args:
            network: Blockchain network name
            transactions: List of transaction data
        """
        for tx in transactions:
            self.transactions[network].append(tx)
            
            # Update address metrics
            self._update_address_metrics(network, tx)
            
            # Check for whale movements
            if tx.amount >= self.whale_threshold:
                self._analyze_whale_movement(network, tx)
            
            # Update flow analysis
            self._update_flow_analysis(network, tx)
    
    def add_network_metrics(self,
                          network: str,
                          metrics: NetworkMetrics) -> None:
        """
        Add network-level metrics.
        
        Args:
            network: Blockchain network name
            metrics: Network metrics data
        """
        self.network_metrics[network].append(metrics)
        
        # Update network health score
        self._calculate_network_health_score(network, metrics)
    
    def add_defi_metrics(self,
                        protocol: str,
                        metrics: DeFiMetrics) -> None:
        """
        Add DeFi protocol metrics.
        
        Args:
            protocol: DeFi protocol name
            metrics: DeFi metrics data
        """
        self.defi_metrics[protocol].append(metrics)
    
    def _update_address_metrics(self,
                              network: str,
                              tx: BlockchainTransaction) -> None:
        """Update metrics for addresses involved in transaction."""
        # Update sender metrics
        if tx.from_address not in self.address_metrics[network]:
            self.address_metrics[network][tx.from_address] = AddressMetrics(
                address=tx.from_address,
                timestamp=tx.timestamp,
                balance=0.0,
                transaction_count=0,
                total_received=0.0,
                total_sent=0.0,
                first_seen=tx.timestamp,
                last_active=tx.timestamp,
                address_type="unknown"
            )
        
        sender_metrics = self.address_metrics[network][tx.from_address]
        sender_metrics.transaction_count += 1
        sender_metrics.total_sent += tx.amount
        sender_metrics.balance -= (tx.amount + tx.fee)
        sender_metrics.last_active = tx.timestamp
        
        # Update receiver metrics
        if tx.to_address not in self.address_metrics[network]:
            self.address_metrics[network][tx.to_address] = AddressMetrics(
                address=tx.to_address,
                timestamp=tx.timestamp,
                balance=0.0,
                transaction_count=0,
                total_received=0.0,
                total_sent=0.0,
                first_seen=tx.timestamp,
                last_active=tx.timestamp,
                address_type="unknown"
            )
        
        receiver_metrics = self.address_metrics[network][tx.to_address]
        receiver_metrics.transaction_count += 1
        receiver_metrics.total_received += tx.amount
        receiver_metrics.balance += tx.amount
        receiver_metrics.last_active = tx.timestamp
        
        # Classify addresses
        self._classify_address(network, sender_metrics)
        self._classify_address(network, receiver_metrics)
    
    def _classify_address(self,
                         network: str,
                         metrics: AddressMetrics) -> None:
        """Classify address type based on behavior patterns."""
        address = metrics.address
        
        # Check if it's a known address
        if address in self.known_addresses[network]:
            metrics.address_type = self.known_addresses[network][address]
            return
        
        # Check if it's an exchange (high transaction volume, round numbers)
        if metrics.transaction_count > 1000:
            metrics.address_type = "exchange"
        
        # Check if it's a whale (high balance)
        elif metrics.balance > self.whale_threshold * 100:
            if metrics.transaction_count < 50:
                metrics.address_type = "whale_accumulator"
            else:
                metrics.address_type = "whale_trader"
        
        # Check if it's institutional (large, infrequent transactions)
        elif (metrics.total_received > self.whale_threshold * 10 and 
              metrics.transaction_count < 100):
            metrics.address_type = "institutional"
        
        # Check for smart contract patterns
        elif (address in self.exchange_addresses[network] or 
              self._is_smart_contract_pattern(address)):
            metrics.address_type = "smart_contract"
        
        # Default to retail
        else:
            metrics.address_type = "retail"
        
        # Calculate risk score
        metrics.risk_score = self._calculate_address_risk_score(metrics)
    
    def _is_smart_contract_pattern(self, address: str) -> bool:
        """Check if address follows smart contract patterns."""
        # Simplified pattern detection
        # In practice, this would use more sophisticated methods
        
        # Check for specific patterns in address
        if len(address) == 42 and address.startswith('0x'):  # Ethereum-style
            # Check for repeated patterns common in contract addresses
            hex_part = address[2:]
            if len(set(hex_part)) < 10:  # Low entropy suggests generated address
                return True
        
        return False
    
    def _calculate_address_risk_score(self, metrics: AddressMetrics) -> float:
        """Calculate risk score for an address."""
        risk_factors = []
        
        # High transaction velocity
        days_active = max(1, (metrics.last_active - metrics.first_seen).days)
        tx_per_day = metrics.transaction_count / days_active
        if tx_per_day > 100:
            risk_factors.append(0.3)
        
        # Large balance with low activity (potential for market manipulation)
        if metrics.balance > self.whale_threshold * 50 and metrics.transaction_count < 10:
            risk_factors.append(0.4)
        
        # Rapid accumulation pattern
        if metrics.total_received > metrics.total_sent * 10:
            risk_factors.append(0.2)
        
        # Exchange-like behavior but not labeled as exchange
        if (metrics.transaction_count > 500 and 
            metrics.address_type not in ['exchange', 'smart_contract']):
            risk_factors.append(0.5)
        
        return min(1.0, sum(risk_factors))
    
    def _analyze_whale_movement(self,
                              network: str,
                              tx: BlockchainTransaction) -> None:
        """Analyze whale transaction movement."""
        # Determine movement type
        movement_type = self._classify_movement_type(network, tx)
        
        # Determine whale category
        whale_category = self._classify_whale_category(network, tx)
        
        # Predict market impact
        market_impact = self._predict_market_impact(network, tx, movement_type)
        
        whale_movement = WhaleMovement(
            timestamp=tx.timestamp,
            transaction_hash=tx.tx_hash,
            from_address=tx.from_address,
            to_address=tx.to_address,
            amount=tx.amount,
            usd_value=None,  # Would be populated with price data
            movement_type=movement_type,
            whale_category=whale_category,
            market_impact_prediction=market_impact
        )
        
        self.whale_movements.append(whale_movement)
    
    def _classify_movement_type(self,
                              network: str,
                              tx: BlockchainTransaction) -> str:
        """Classify whale movement type."""
        from_addr = tx.from_address
        to_addr = tx.to_address
        
        # Check if addresses are exchanges
        from_is_exchange = from_addr in self.exchange_addresses[network]
        to_is_exchange = to_addr in self.exchange_addresses[network]
        
        if from_is_exchange and not to_is_exchange:
            return "exchange_outflow"
        elif not from_is_exchange and to_is_exchange:
            return "exchange_inflow"
        elif from_is_exchange and to_is_exchange:
            return "exchange_transfer"
        else:
            # Check accumulation vs distribution patterns
            from_metrics = self.address_metrics[network].get(from_addr)
            to_metrics = self.address_metrics[network].get(to_addr)
            
            if to_metrics and to_metrics.address_type in ['whale_accumulator', 'institutional']:
                return "accumulation"
            elif from_metrics and from_metrics.address_type in ['whale_trader']:
                return "distribution"
            else:
                return "transfer"
    
    def _classify_whale_category(self,
                               network: str,
                               tx: BlockchainTransaction) -> str:
        """Classify whale category."""
        from_metrics = self.address_metrics[network].get(tx.from_address)
        to_metrics = self.address_metrics[network].get(tx.to_address)
        
        # Check transaction amount relative to thresholds
        if tx.amount > self.whale_threshold * 100:
            return "institutional"
        elif tx.amount > self.whale_threshold * 10:
            return "large_whale"
        else:
            return "retail_whale"
    
    def _predict_market_impact(self,
                             network: str,
                             tx: BlockchainTransaction,
                             movement_type: str) -> float:
        """Predict market impact of whale movement."""
        # Simplified market impact model
        base_impact = min(1.0, tx.amount / (self.whale_threshold * 100))
        
        # Adjust based on movement type
        impact_multipliers = {
            'exchange_inflow': 0.8,  # Potential selling pressure
            'exchange_outflow': -0.5,  # Potential accumulation
            'accumulation': -0.3,  # Bullish
            'distribution': 0.7,  # Bearish
            'transfer': 0.1  # Neutral
        }
        
        multiplier = impact_multipliers.get(movement_type, 0.0)
        
        # Adjust for market conditions (simplified)
        recent_whale_activity = len([w for w in list(self.whale_movements)[-10:] 
                                   if (datetime.now() - w.timestamp).total_seconds() < 3600])
        
        activity_multiplier = 1.0 + (recent_whale_activity * 0.1)
        
        return base_impact * multiplier * activity_multiplier
    
    def _update_flow_analysis(self,
                            network: str,
                            tx: BlockchainTransaction) -> None:
        """Update transaction flow analysis."""
        if network not in self.flow_analysis:
            self.flow_analysis[network] = {
                'exchange_inflows': 0.0,
                'exchange_outflows': 0.0,
                'whale_accumulation': 0.0,
                'whale_distribution': 0.0,
                'defi_activity': 0.0,
                'retail_activity': 0.0
            }
        
        flow = self.flow_analysis[network]
        
        # Classify transaction flow
        movement_type = self._classify_movement_type(network, tx)
        
        if movement_type == 'exchange_inflow':
            flow['exchange_inflows'] += tx.amount
        elif movement_type == 'exchange_outflow':
            flow['exchange_outflows'] += tx.amount
        elif movement_type == 'accumulation':
            flow['whale_accumulation'] += tx.amount
        elif movement_type == 'distribution':
            flow['whale_distribution'] += tx.amount
        elif tx.transaction_type == 'defi':
            flow['defi_activity'] += tx.amount
        else:
            flow['retail_activity'] += tx.amount
    
    def _calculate_network_health_score(self,
                                      network: str,
                                      metrics: NetworkMetrics) -> None:
        """Calculate network health score."""
        score_components = []
        
        # Hash rate / security component
        if metrics.hash_rate is not None:
            # Normalized hash rate score (would need historical data for proper normalization)
            hash_rate_score = min(1.0, metrics.hash_rate / 100.0)  # Simplified
            score_components.append(hash_rate_score * 0.3)
        
        # Transaction activity component
        if metrics.transaction_count > 0:
            # Normalized transaction activity
            tx_activity_score = min(1.0, metrics.transaction_count / 100000.0)  # Simplified
            score_components.append(tx_activity_score * 0.25)
        
        # Address growth component
        if metrics.new_addresses > 0:
            address_growth_score = min(1.0, metrics.new_addresses / 10000.0)  # Simplified
            score_components.append(address_growth_score * 0.2)
        
        # Fee market health
        if metrics.total_fees > 0:
            # Moderate fees are healthy (not too high, not too low)
            avg_fee = metrics.total_fees / max(1, metrics.transaction_count)
            fee_health_score = 1.0 - min(1.0, avg_fee / 50.0)  # Simplified
            score_components.append(fee_health_score * 0.15)
        
        # Network efficiency (block time)
        if metrics.block_time is not None:
            # Closer to target block time is better
            target_block_time = 600  # 10 minutes for Bitcoin, adjust for other networks
            time_deviation = abs(metrics.block_time - target_block_time) / target_block_time
            efficiency_score = max(0.0, 1.0 - time_deviation)
            score_components.append(efficiency_score * 0.1)
        
        self.network_health_scores[network] = sum(score_components) if score_components else 0.5
    
    def get_whale_activity_summary(self,
                                 network: Optional[str] = None,
                                 hours: int = 24) -> Dict[str, Any]:
        """Get whale activity summary."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter whale movements
        if network:
            whale_movements = [w for w in self.whale_movements 
                             if w.timestamp >= cutoff_time]
        else:
            whale_movements = [w for w in self.whale_movements 
                             if w.timestamp >= cutoff_time]
        
        if not whale_movements:
            return {'total_movements': 0}
        
        # Aggregate metrics
        total_volume = sum(w.amount for w in whale_movements)
        
        movement_types = defaultdict(int)
        whale_categories = defaultdict(int)
        
        for movement in whale_movements:
            movement_types[movement.movement_type] += 1
            whale_categories[movement.whale_category] += 1
        
        # Market impact assessment
        total_impact = sum(w.market_impact_prediction for w in whale_movements)
        avg_impact = total_impact / len(whale_movements)
        
        return {
            'total_movements': len(whale_movements),
            'total_volume': total_volume,
            'average_amount': total_volume / len(whale_movements),
            'movement_types': dict(movement_types),
            'whale_categories': dict(whale_categories),
            'predicted_market_impact': avg_impact,
            'largest_movement': max(whale_movements, key=lambda x: x.amount).amount,
            'most_recent': whale_movements[-1].timestamp.isoformat() if whale_movements else None
        }
    
    def get_exchange_flow_analysis(self, network: str) -> Dict[str, Any]:
        """Get exchange flow analysis for a network."""
        if network not in self.flow_analysis:
            return {}
        
        flow = self.flow_analysis[network]
        
        # Calculate net flows
        net_exchange_flow = flow['exchange_inflows'] - flow['exchange_outflows']
        net_whale_flow = flow['whale_accumulation'] - flow['whale_distribution']
        
        # Flow ratios
        total_exchange_flow = flow['exchange_inflows'] + flow['exchange_outflows']
        inflow_ratio = flow['exchange_inflows'] / total_exchange_flow if total_exchange_flow > 0 else 0.5
        
        return {
            'exchange_inflows': flow['exchange_inflows'],
            'exchange_outflows': flow['exchange_outflows'],
            'net_exchange_flow': net_exchange_flow,
            'inflow_ratio': inflow_ratio,
            'whale_accumulation': flow['whale_accumulation'],
            'whale_distribution': flow['whale_distribution'],
            'net_whale_flow': net_whale_flow,
            'defi_activity': flow['defi_activity'],
            'retail_activity': flow['retail_activity'],
            'flow_sentiment': 'bearish' if net_exchange_flow > 0 else 'bullish'
        }
    
    def get_network_health_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get network health metrics for all networks."""
        health_metrics = {}
        
        for network in self.networks:
            if network in self.network_metrics and self.network_metrics[network]:
                latest_metrics = self.network_metrics[network][-1]
                health_score = self.network_health_scores.get(network, 0.0)
                
                health_metrics[network] = {
                    'health_score': health_score,
                    'transaction_count': latest_metrics.transaction_count,
                    'active_addresses': latest_metrics.active_addresses,
                    'new_addresses': latest_metrics.new_addresses,
                    'network_value_transferred': latest_metrics.network_value_transferred,
                    'avg_fee': (latest_metrics.total_fees / max(1, latest_metrics.transaction_count)),
                    'hash_rate': latest_metrics.hash_rate,
                    'block_time': latest_metrics.block_time
                }
        
        return health_metrics
    
    def get_address_analysis(self,
                           network: str,
                           address: str) -> Optional[Dict[str, Any]]:
        """Get detailed analysis for a specific address."""
        if (network not in self.address_metrics or 
            address not in self.address_metrics[network]):
            return None
        
        metrics = self.address_metrics[network][address]
        
        # Calculate additional metrics
        days_active = max(1, (metrics.last_active - metrics.first_seen).days)
        avg_tx_per_day = metrics.transaction_count / days_active
        
        net_flow = metrics.total_received - metrics.total_sent
        turnover_ratio = metrics.total_sent / max(metrics.balance, 1.0)
        
        return {
            'address': address,
            'address_type': metrics.address_type,
            'risk_score': metrics.risk_score,
            'balance': metrics.balance,
            'transaction_count': metrics.transaction_count,
            'total_received': metrics.total_received,
            'total_sent': metrics.total_sent,
            'net_flow': net_flow,
            'first_seen': metrics.first_seen.isoformat(),
            'last_active': metrics.last_active.isoformat(),
            'days_active': days_active,
            'avg_transactions_per_day': avg_tx_per_day,
            'turnover_ratio': turnover_ratio,
            'activity_level': self._classify_activity_level(avg_tx_per_day)
        }
    
    def _classify_activity_level(self, avg_tx_per_day: float) -> str:
        """Classify address activity level."""
        if avg_tx_per_day > 50:
            return "very_high"
        elif avg_tx_per_day > 10:
            return "high"
        elif avg_tx_per_day > 1:
            return "moderate"
        elif avg_tx_per_day > 0.1:
            return "low"
        else:
            return "very_low"
    
    def get_defi_analytics(self) -> Dict[str, Dict[str, Any]]:
        """Get DeFi protocol analytics."""
        defi_summary = {}
        
        for protocol, metrics_list in self.defi_metrics.items():
            if not metrics_list:
                continue
            
            latest_metrics = metrics_list[-1]
            
            # Calculate growth rates if we have historical data
            growth_rates = {}
            if len(metrics_list) > 1:
                prev_metrics = metrics_list[-2]
                
                growth_rates = {
                    'tvl_growth': ((latest_metrics.total_value_locked - prev_metrics.total_value_locked) / 
                                  max(prev_metrics.total_value_locked, 1)) * 100,
                    'volume_growth': ((latest_metrics.trading_volume - prev_metrics.trading_volume) / 
                                    max(prev_metrics.trading_volume, 1)) * 100,
                    'user_growth': ((latest_metrics.unique_users - prev_metrics.unique_users) / 
                                  max(prev_metrics.unique_users, 1)) * 100
                }
            
            defi_summary[protocol] = {
                'total_value_locked': latest_metrics.total_value_locked,
                'trading_volume': latest_metrics.trading_volume,
                'unique_users': latest_metrics.unique_users,
                'transaction_count': latest_metrics.transaction_count,
                'governance_activity': latest_metrics.governance_activity,
                'yield_rates': latest_metrics.yield_rates,
                'liquidity_metrics': latest_metrics.liquidity_metrics,
                'growth_rates': growth_rates,
                'timestamp': latest_metrics.timestamp.isoformat()
            }
        
        return defi_summary
    
    def detect_unusual_activity(self,
                              network: str,
                              lookback_hours: int = 6) -> List[Dict[str, Any]]:
        """Detect unusual on-chain activity."""
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        unusual_activities = []
        
        # Check for unusual whale movements
        recent_whales = [w for w in self.whale_movements 
                        if w.timestamp >= cutoff_time]
        
        if len(recent_whales) > 5:  # Threshold for unusual activity
            total_volume = sum(w.amount for w in recent_whales)
            unusual_activities.append({
                'type': 'high_whale_activity',
                'description': f'Detected {len(recent_whales)} whale movements totaling {total_volume:.2f}',
                'severity': 'medium',
                'timestamp': datetime.now().isoformat()
            })
        
        # Check for large exchange flows
        if network in self.flow_analysis:
            flow = self.flow_analysis[network]
            net_flow = flow['exchange_inflows'] - flow['exchange_outflows']
            
            if abs(net_flow) > self.whale_threshold * 50:
                direction = 'inflow' if net_flow > 0 else 'outflow'
                unusual_activities.append({
                    'type': f'large_exchange_{direction}',
                    'description': f'Large exchange {direction} detected: {abs(net_flow):.2f}',
                    'severity': 'high',
                    'timestamp': datetime.now().isoformat()
                })
        
        # Check network health degradation
        health_score = self.network_health_scores.get(network, 0.5)
        if health_score < 0.3:
            unusual_activities.append({
                'type': 'network_health_degradation',
                'description': f'Network health score dropped to {health_score:.2f}',
                'severity': 'high',
                'timestamp': datetime.now().isoformat()
            })
        
        return unusual_activities
    
    def get_supply_distribution_analysis(self, network: str) -> Dict[str, Any]:
        """Analyze supply distribution across address types."""
        if network not in self.address_metrics:
            return {}
        
        distribution = defaultdict(float)
        total_supply = 0.0
        
        for address, metrics in self.address_metrics[network].items():
            if metrics.balance > 0:
                distribution[metrics.address_type] += metrics.balance
                total_supply += metrics.balance
        
        # Calculate percentages
        distribution_pct = {}
        for addr_type, balance in distribution.items():
            distribution_pct[addr_type] = (balance / total_supply * 100) if total_supply > 0 else 0
        
        # Calculate concentration metrics
        balances = [metrics.balance for metrics in self.address_metrics[network].values() 
                   if metrics.balance > 0]
        
        if balances:
            balances.sort(reverse=True)
            
            # Top addresses concentration
            top_1_pct = sum(balances[:max(1, len(balances)//100)]) / total_supply * 100 if total_supply > 0 else 0
            top_10_pct = sum(balances[:max(1, len(balances)//10)]) / total_supply * 100 if total_supply > 0 else 0
            
            # Gini coefficient (simplified)
            gini = self._calculate_gini_coefficient(balances)
        else:
            top_1_pct = top_10_pct = gini = 0.0
        
        return {
            'total_supply': total_supply,
            'distribution_by_type': distribution_pct,
            'concentration_metrics': {
                'top_1_percent_holdings': top_1_pct,
                'top_10_percent_holdings': top_10_pct,
                'gini_coefficient': gini
            },
            'total_addresses': len(self.address_metrics[network]),
            'active_addresses': len([m for m in self.address_metrics[network].values() 
                                   if m.balance > 0])
        }
    
    def _calculate_gini_coefficient(self, balances: List[float]) -> float:
        """Calculate Gini coefficient for wealth distribution."""
        if not balances or len(balances) < 2:
            return 0.0
        
        balances = sorted(balances)
        n = len(balances)
        cumsum = np.cumsum(balances)
        
        return (n + 1 - 2 * sum((n + 1 - i) * x for i, x in enumerate(balances))) / (n * sum(balances))