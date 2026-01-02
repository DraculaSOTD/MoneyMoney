"""
Order Book Analysis for Market Microstructure.

Implements advanced order book analysis techniques for cryptocurrency markets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


@dataclass
class OrderBookLevel:
    """Represents a single level in the order book."""
    price: float
    quantity: float
    num_orders: int
    timestamp: datetime


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot."""
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    mid_price: float
    spread: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderBookImbalance:
    """Order book imbalance metrics."""
    timestamp: datetime
    price_imbalance: float
    volume_imbalance: float
    depth_imbalance: float
    order_count_imbalance: float
    microprice: float
    pressure_indicator: float


@dataclass
class LiquidityMetrics:
    """Liquidity analysis metrics."""
    bid_ask_spread: float
    effective_spread: float
    quoted_spread: float
    depth_at_bps: Dict[int, float]  # Depth at various basis points
    price_impact: Dict[float, float]  # Volume -> Price impact
    resilience_time: float
    kyle_lambda: float
    amihud_illiquidity: float


class OrderBookAnalyzer:
    """
    Advanced order book analysis for market microstructure.
    
    Features:
    - Real-time order book processing
    - Imbalance detection and analysis
    - Liquidity metrics calculation
    - Price discovery analysis
    - Market depth visualization
    - Order flow analysis
    - Microstructure noise filtering
    """
    
    def __init__(self,
                 max_levels: int = 20,
                 imbalance_window: int = 10,
                 tick_size: float = 0.01,
                 min_order_size: float = 0.001):
        """
        Initialize order book analyzer.
        
        Args:
            max_levels: Maximum order book levels to analyze
            imbalance_window: Window size for imbalance calculations
            tick_size: Minimum price increment
            min_order_size: Minimum order size to consider
        """
        self.max_levels = max_levels
        self.imbalance_window = imbalance_window
        self.tick_size = tick_size
        self.min_order_size = min_order_size
        
        # Order book state
        self.current_book: Optional[OrderBookSnapshot] = None
        self.book_history: deque = deque(maxlen=1000)
        self.imbalance_history: deque = deque(maxlen=500)
        
        # Metrics tracking
        self.liquidity_metrics: Dict[datetime, LiquidityMetrics] = {}
        self.price_levels: Dict[float, Dict] = defaultdict(dict)
        
        # Market microstructure features
        self.recent_trades: deque = deque(maxlen=1000)
        self.order_flow: deque = deque(maxlen=1000)
        
    def update_order_book(self, 
                         timestamp: datetime,
                         bids: List[Tuple[float, float, int]],  # price, quantity, orders
                         asks: List[Tuple[float, float, int]]) -> OrderBookSnapshot:
        """
        Update order book with new data.
        
        Args:
            timestamp: Update timestamp
            bids: List of (price, quantity, num_orders) for bids
            asks: List of (price, quantity, num_orders) for asks
            
        Returns:
            Updated order book snapshot
        """
        # Convert to OrderBookLevel objects
        bid_levels = []
        for price, qty, orders in sorted(bids, key=lambda x: x[0], reverse=True):
            if qty >= self.min_order_size:
                bid_levels.append(OrderBookLevel(price, qty, orders, timestamp))
        
        ask_levels = []
        for price, qty, orders in sorted(asks, key=lambda x: x[0]):
            if qty >= self.min_order_size:
                ask_levels.append(OrderBookLevel(price, qty, orders, timestamp))
        
        # Limit to max levels
        bid_levels = bid_levels[:self.max_levels]
        ask_levels = ask_levels[:self.max_levels]
        
        # Calculate mid price and spread
        if bid_levels and ask_levels:
            best_bid = bid_levels[0].price
            best_ask = ask_levels[0].price
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
        else:
            mid_price = 0.0
            spread = 0.0
        
        # Create snapshot
        snapshot = OrderBookSnapshot(
            timestamp=timestamp,
            bids=bid_levels,
            asks=ask_levels,
            mid_price=mid_price,
            spread=spread
        )
        
        # Update state
        self.current_book = snapshot
        self.book_history.append(snapshot)
        
        # Calculate imbalance
        imbalance = self._calculate_imbalance(snapshot)
        self.imbalance_history.append(imbalance)
        
        # Update liquidity metrics
        self._update_liquidity_metrics(snapshot)
        
        return snapshot
    
    def _calculate_imbalance(self, snapshot: OrderBookSnapshot) -> OrderBookImbalance:
        """Calculate order book imbalance metrics."""
        if not snapshot.bids or not snapshot.asks:
            return OrderBookImbalance(
                timestamp=snapshot.timestamp,
                price_imbalance=0.0,
                volume_imbalance=0.0,
                depth_imbalance=0.0,
                order_count_imbalance=0.0,
                microprice=snapshot.mid_price,
                pressure_indicator=0.0
            )
        
        # Get best levels
        best_bid = snapshot.bids[0]
        best_ask = snapshot.asks[0]
        
        # Price-weighted imbalance
        bid_volume = sum(level.quantity for level in snapshot.bids)
        ask_volume = sum(level.quantity for level in snapshot.asks)
        
        if bid_volume + ask_volume > 0:
            volume_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        else:
            volume_imbalance = 0.0
        
        # Depth imbalance (considering price levels)
        bid_depth = self._calculate_depth_at_distance(snapshot.bids, best_bid.price, 0.001)
        ask_depth = self._calculate_depth_at_distance(snapshot.asks, best_ask.price, 0.001)
        
        if bid_depth + ask_depth > 0:
            depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        else:
            depth_imbalance = 0.0
        
        # Order count imbalance
        bid_orders = sum(level.num_orders for level in snapshot.bids)
        ask_orders = sum(level.num_orders for level in snapshot.asks)
        
        if bid_orders + ask_orders > 0:
            order_count_imbalance = (bid_orders - ask_orders) / (bid_orders + ask_orders)
        else:
            order_count_imbalance = 0.0
        
        # Price imbalance (relative to recent mid price)
        if len(self.book_history) > 1:
            recent_mid = np.mean([book.mid_price for book in list(self.book_history)[-5:]])
            if recent_mid > 0:
                price_imbalance = (snapshot.mid_price - recent_mid) / recent_mid
            else:
                price_imbalance = 0.0
        else:
            price_imbalance = 0.0
        
        # Microprice calculation
        bid_qty = best_bid.quantity
        ask_qty = best_ask.quantity
        
        if bid_qty + ask_qty > 0:
            microprice = (best_ask.price * bid_qty + best_bid.price * ask_qty) / (bid_qty + ask_qty)
        else:
            microprice = snapshot.mid_price
        
        # Pressure indicator (combines multiple imbalances)
        pressure_indicator = (
            0.4 * volume_imbalance +
            0.3 * depth_imbalance +
            0.2 * order_count_imbalance +
            0.1 * price_imbalance
        )
        
        return OrderBookImbalance(
            timestamp=snapshot.timestamp,
            price_imbalance=price_imbalance,
            volume_imbalance=volume_imbalance,
            depth_imbalance=depth_imbalance,
            order_count_imbalance=order_count_imbalance,
            microprice=microprice,
            pressure_indicator=pressure_indicator
        )
    
    def _calculate_depth_at_distance(self, 
                                   levels: List[OrderBookLevel],
                                   reference_price: float,
                                   distance_pct: float) -> float:
        """Calculate cumulative depth within distance from reference price."""
        total_depth = 0.0
        
        for level in levels:
            price_distance = abs(level.price - reference_price) / reference_price
            if price_distance <= distance_pct:
                total_depth += level.quantity
        
        return total_depth
    
    def _update_liquidity_metrics(self, snapshot: OrderBookSnapshot) -> None:
        """Update liquidity metrics for current snapshot."""
        if not snapshot.bids or not snapshot.asks:
            return
        
        # Basic spread metrics
        best_bid = snapshot.bids[0].price
        best_ask = snapshot.asks[0].price
        mid_price = snapshot.mid_price
        
        bid_ask_spread = best_ask - best_bid
        relative_spread = bid_ask_spread / mid_price if mid_price > 0 else 0
        
        # Depth at various basis points
        depth_at_bps = {}
        for bps in [5, 10, 25, 50, 100]:
            distance = bps / 10000.0  # Convert basis points to decimal
            
            bid_depth = self._calculate_depth_at_distance(snapshot.bids, best_bid, distance)
            ask_depth = self._calculate_depth_at_distance(snapshot.asks, best_ask, distance)
            
            depth_at_bps[bps] = bid_depth + ask_depth
        
        # Price impact estimation
        price_impact = self._estimate_price_impact(snapshot)
        
        # Kyle's lambda (simplified estimation)
        kyle_lambda = self._estimate_kyle_lambda(snapshot)
        
        # Amihud illiquidity measure (requires trade data)
        amihud_illiquidity = self._calculate_amihud_illiquidity()
        
        metrics = LiquidityMetrics(
            bid_ask_spread=relative_spread,
            effective_spread=relative_spread,  # Simplified
            quoted_spread=relative_spread,
            depth_at_bps=depth_at_bps,
            price_impact=price_impact,
            resilience_time=0.0,  # Would need time series analysis
            kyle_lambda=kyle_lambda,
            amihud_illiquidity=amihud_illiquidity
        )
        
        self.liquidity_metrics[snapshot.timestamp] = metrics
    
    def _estimate_price_impact(self, snapshot: OrderBookSnapshot) -> Dict[float, float]:
        """Estimate price impact for different order sizes."""
        price_impact = {}
        
        if not snapshot.bids or not snapshot.asks:
            return price_impact
        
        # Test volumes as percentage of average volume
        test_volumes = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        # Estimate average volume from recent order book
        avg_volume = self._estimate_average_volume()
        
        for vol_pct in test_volumes:
            test_volume = vol_pct * avg_volume / 100.0
            
            # Calculate impact for buy order
            buy_impact = self._calculate_market_order_impact(
                snapshot.asks, test_volume, 'buy'
            )
            
            # Calculate impact for sell order  
            sell_impact = self._calculate_market_order_impact(
                snapshot.bids, test_volume, 'sell'
            )
            
            # Average impact
            avg_impact = (abs(buy_impact) + abs(sell_impact)) / 2
            price_impact[test_volume] = avg_impact
        
        return price_impact
    
    def _calculate_market_order_impact(self,
                                     levels: List[OrderBookLevel],
                                     volume: float,
                                     side: str) -> float:
        """Calculate price impact of a market order."""
        if not levels or volume <= 0:
            return 0.0
        
        initial_price = levels[0].price
        remaining_volume = volume
        weighted_price = 0.0
        executed_volume = 0.0
        
        for level in levels:
            if remaining_volume <= 0:
                break
            
            available_qty = level.quantity
            execute_qty = min(remaining_volume, available_qty)
            
            weighted_price += level.price * execute_qty
            executed_volume += execute_qty
            remaining_volume -= execute_qty
        
        if executed_volume > 0:
            avg_execution_price = weighted_price / executed_volume
            
            if side == 'buy':
                impact = (avg_execution_price - initial_price) / initial_price
            else:  # sell
                impact = (initial_price - avg_execution_price) / initial_price
                
            return impact
        
        return 0.0
    
    def _estimate_average_volume(self) -> float:
        """Estimate average order book volume."""
        if not self.book_history:
            return 1.0  # Default
        
        volumes = []
        for snapshot in self.book_history:
            total_volume = 0.0
            
            # Sum bid volume
            for level in snapshot.bids:
                total_volume += level.quantity
            
            # Sum ask volume
            for level in snapshot.asks:
                total_volume += level.quantity
            
            volumes.append(total_volume)
        
        return np.mean(volumes) if volumes else 1.0
    
    def _estimate_kyle_lambda(self, snapshot: OrderBookSnapshot) -> float:
        """Estimate Kyle's lambda (price impact parameter)."""
        if len(self.imbalance_history) < 10:
            return 0.0
        
        # Simplified Kyle's lambda estimation
        # In practice, this would use regression analysis
        
        recent_imbalances = list(self.imbalance_history)[-10:]
        price_changes = []
        order_flows = []
        
        for i, imbalance in enumerate(recent_imbalances):
            if i > 0:
                # Price change
                prev_book = list(self.book_history)[-(10-i)]
                current_book = list(self.book_history)[-(10-i-1)]
                
                if prev_book.mid_price > 0:
                    price_change = (current_book.mid_price - prev_book.mid_price) / prev_book.mid_price
                    price_changes.append(price_change)
                    order_flows.append(imbalance.volume_imbalance)
        
        if len(price_changes) > 3 and len(order_flows) > 3:
            # Simple correlation-based estimation
            correlation = np.corrcoef(order_flows, price_changes)[0, 1]
            if not np.isnan(correlation):
                return abs(correlation) * 0.01  # Scale appropriately
        
        return 0.0
    
    def _calculate_amihud_illiquidity(self) -> float:
        """Calculate Amihud illiquidity measure."""
        if len(self.recent_trades) < 10:
            return 0.0
        
        # Would need actual trade data
        # This is a placeholder implementation
        recent_spreads = []
        
        for snapshot in list(self.book_history)[-10:]:
            if snapshot.mid_price > 0:
                relative_spread = snapshot.spread / snapshot.mid_price
                recent_spreads.append(relative_spread)
        
        if recent_spreads:
            return np.mean(recent_spreads)
        
        return 0.0
    
    def get_current_imbalance(self) -> Optional[OrderBookImbalance]:
        """Get current order book imbalance."""
        if self.imbalance_history:
            return self.imbalance_history[-1]
        return None
    
    def get_imbalance_trend(self, window: int = 10) -> Dict[str, float]:
        """Get trend in order book imbalance."""
        if len(self.imbalance_history) < window:
            return {}
        
        recent_imbalances = list(self.imbalance_history)[-window:]
        
        trends = {}
        for metric in ['volume_imbalance', 'depth_imbalance', 'order_count_imbalance', 'pressure_indicator']:
            values = [getattr(imb, metric) for imb in recent_imbalances]
            
            if len(values) > 1:
                # Simple linear trend
                x = np.arange(len(values))
                trend = np.polyfit(x, values, 1)[0]
                trends[f'{metric}_trend'] = trend
                trends[f'{metric}_current'] = values[-1]
        
        return trends
    
    def get_liquidity_summary(self) -> Dict[str, Any]:
        """Get current liquidity summary."""
        if not self.current_book:
            return {}
        
        latest_metrics = None
        if self.liquidity_metrics:
            latest_timestamp = max(self.liquidity_metrics.keys())
            latest_metrics = self.liquidity_metrics[latest_timestamp]
        
        summary = {
            'timestamp': self.current_book.timestamp.isoformat(),
            'mid_price': self.current_book.mid_price,
            'spread': self.current_book.spread,
            'relative_spread': self.current_book.spread / self.current_book.mid_price if self.current_book.mid_price > 0 else 0,
            'bid_levels': len(self.current_book.bids),
            'ask_levels': len(self.current_book.asks)
        }
        
        if latest_metrics:
            summary['liquidity_metrics'] = {
                'depth_5bps': latest_metrics.depth_at_bps.get(5, 0),
                'depth_10bps': latest_metrics.depth_at_bps.get(10, 0),
                'depth_25bps': latest_metrics.depth_at_bps.get(25, 0),
                'kyle_lambda': latest_metrics.kyle_lambda,
                'amihud_illiquidity': latest_metrics.amihud_illiquidity
            }
        
        return summary
    
    def get_order_book_pressure(self) -> Dict[str, float]:
        """Get current order book pressure indicators."""
        if not self.imbalance_history:
            return {}
        
        current_imbalance = self.imbalance_history[-1]
        
        return {
            'volume_imbalance': current_imbalance.volume_imbalance,
            'depth_imbalance': current_imbalance.depth_imbalance,
            'order_count_imbalance': current_imbalance.order_count_imbalance,
            'pressure_indicator': current_imbalance.pressure_indicator,
            'microprice': current_imbalance.microprice,
            'mid_price': self.current_book.mid_price if self.current_book else 0,
            'microprice_edge': current_imbalance.microprice - (self.current_book.mid_price if self.current_book else 0)
        }
    
    def detect_liquidity_events(self) -> List[Dict[str, Any]]:
        """Detect significant liquidity events."""
        events = []
        
        if len(self.book_history) < 5:
            return events
        
        recent_books = list(self.book_history)[-5:]
        
        # Detect spread widening
        spreads = [book.spread for book in recent_books]
        current_spread = spreads[-1]
        avg_spread = np.mean(spreads[:-1])
        
        if current_spread > avg_spread * 2:
            events.append({
                'type': 'spread_widening',
                'severity': 'high' if current_spread > avg_spread * 3 else 'medium',
                'timestamp': recent_books[-1].timestamp,
                'current_spread': current_spread,
                'average_spread': avg_spread,
                'multiplier': current_spread / avg_spread if avg_spread > 0 else 0
            })
        
        # Detect depth depletion
        if self.liquidity_metrics:
            recent_metrics = list(self.liquidity_metrics.values())[-2:]
            
            if len(recent_metrics) == 2:
                current_depth = recent_metrics[-1].depth_at_bps.get(10, 0)
                prev_depth = recent_metrics[-2].depth_at_bps.get(10, 0)
                
                if prev_depth > 0 and current_depth < prev_depth * 0.5:
                    events.append({
                        'type': 'depth_depletion',
                        'severity': 'high' if current_depth < prev_depth * 0.3 else 'medium',
                        'timestamp': recent_books[-1].timestamp,
                        'current_depth': current_depth,
                        'previous_depth': prev_depth,
                        'depletion_ratio': current_depth / prev_depth
                    })
        
        # Detect extreme imbalance
        if self.imbalance_history:
            current_imbalance = self.imbalance_history[-1]
            
            if abs(current_imbalance.pressure_indicator) > 0.7:
                events.append({
                    'type': 'extreme_imbalance',
                    'severity': 'high' if abs(current_imbalance.pressure_indicator) > 0.9 else 'medium',
                    'timestamp': current_imbalance.timestamp,
                    'pressure_indicator': current_imbalance.pressure_indicator,
                    'direction': 'buy' if current_imbalance.pressure_indicator > 0 else 'sell'
                })
        
        return events
    
    def get_price_level_analysis(self) -> Dict[str, Any]:
        """Analyze key price levels and support/resistance."""
        if not self.current_book:
            return {}
        
        analysis = {
            'support_levels': [],
            'resistance_levels': [],
            'key_levels': []
        }
        
        # Analyze recent price levels
        recent_prices = []
        for snapshot in self.book_history:
            if snapshot.bids:
                recent_prices.extend([level.price for level in snapshot.bids[:3]])
            if snapshot.asks:
                recent_prices.extend([level.price for level in snapshot.asks[:3]])
        
        if recent_prices:
            # Find frequently occurring price levels
            price_counts = defaultdict(int)
            
            # Round prices to nearest tick
            for price in recent_prices:
                rounded_price = round(price / self.tick_size) * self.tick_size
                price_counts[rounded_price] += 1
            
            # Identify significant levels
            total_occurrences = sum(price_counts.values())
            threshold = max(3, total_occurrences * 0.05)  # At least 5% of occurrences
            
            current_mid = self.current_book.mid_price
            
            for price, count in price_counts.items():
                if count >= threshold:
                    level_info = {
                        'price': price,
                        'occurrences': count,
                        'relative_frequency': count / total_occurrences,
                        'distance_from_mid': (price - current_mid) / current_mid if current_mid > 0 else 0
                    }
                    
                    if price < current_mid:
                        analysis['support_levels'].append(level_info)
                    else:
                        analysis['resistance_levels'].append(level_info)
                    
                    analysis['key_levels'].append(level_info)
            
            # Sort by price
            analysis['support_levels'].sort(key=lambda x: x['price'], reverse=True)
            analysis['resistance_levels'].sort(key=lambda x: x['price'])
            analysis['key_levels'].sort(key=lambda x: x['relative_frequency'], reverse=True)
        
        return analysis