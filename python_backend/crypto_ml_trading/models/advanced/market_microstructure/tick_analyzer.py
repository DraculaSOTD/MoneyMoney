"""
Tick-by-Tick Data Analysis for Market Microstructure.

Implements high-frequency analysis of tick data for cryptocurrency markets.
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
class TickData:
    """Single tick data point."""
    timestamp: datetime
    price: float
    volume: float
    side: str  # 'buy', 'sell', 'unknown'
    trade_id: Optional[str] = None
    venue: Optional[str] = None


@dataclass
class MicrostructureMetrics:
    """High-frequency microstructure metrics."""
    timestamp: datetime
    price_discovery_ratio: float
    information_share: float
    effective_spread: float
    realized_spread: float
    price_impact: float
    order_flow_toxicity: float
    informed_trading_probability: float
    adverse_selection_cost: float


@dataclass
class FlowToxicity:
    """Order flow toxicity measures."""
    vpin: float  # Volume-Synchronized Probability of Informed Trading
    toxic_flow_ratio: float
    informed_volume_ratio: float
    flow_imbalance: float
    toxicity_trend: float


@dataclass
class PriceDiscoveryMetrics:
    """Price discovery analysis metrics."""
    information_leadership: Dict[str, float]  # By venue
    price_efficiency: float
    discovery_speed: float
    noise_to_signal_ratio: float
    microstructure_noise: float


class TickAnalyzer:
    """
    High-frequency tick data analysis for market microstructure.
    
    Features:
    - Tick-by-tick data processing
    - Order flow toxicity analysis
    - Price discovery measurement
    - Informed trading detection
    - Microstructure noise filtering
    - Trade classification (Lee-Ready algorithm)
    - High-frequency volatility estimation
    """
    
    def __init__(self,
                 tick_buffer_size: int = 10000,
                 analysis_window: int = 1000,
                 min_tick_interval: float = 0.001,
                 toxicity_window: int = 50):
        """
        Initialize tick analyzer.
        
        Args:
            tick_buffer_size: Maximum number of ticks to store
            analysis_window: Window size for rolling analysis
            min_tick_interval: Minimum time between ticks (seconds)
            toxicity_window: Window for toxicity calculations
        """
        self.tick_buffer_size = tick_buffer_size
        self.analysis_window = analysis_window
        self.min_tick_interval = min_tick_interval
        self.toxicity_window = toxicity_window
        
        # Tick data storage
        self.tick_data: deque = deque(maxlen=tick_buffer_size)
        self.venue_ticks: Dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))
        
        # Analysis results
        self.microstructure_metrics: deque = deque(maxlen=1000)
        self.toxicity_measures: deque = deque(maxlen=500)
        self.price_discovery_results: deque = deque(maxlen=200)
        
        # Trade classification
        self.classified_trades: deque = deque(maxlen=5000)
        self.buy_volume_buffer: deque = deque(maxlen=toxicity_window)
        self.sell_volume_buffer: deque = deque(maxlen=toxicity_window)
        
        # Price and quote data
        self.quote_midpoints: deque = deque(maxlen=2000)
        self.trade_prices: deque = deque(maxlen=2000)
        
    def add_tick(self, tick: TickData) -> None:
        """
        Add a new tick to the analyzer.
        
        Args:
            tick: Tick data to add
        """
        # Filter too frequent ticks
        if (self.tick_data and 
            (tick.timestamp - self.tick_data[-1].timestamp).total_seconds() < self.min_tick_interval):
            return
        
        # Store tick data
        self.tick_data.append(tick)
        
        if tick.venue:
            self.venue_ticks[tick.venue].append(tick)
        
        # Update trade classification
        classified_trade = self._classify_trade(tick)
        self.classified_trades.append(classified_trade)
        
        # Update volume buffers for toxicity analysis
        if classified_trade['side'] == 'buy':
            self.buy_volume_buffer.append(tick.volume)
            self.sell_volume_buffer.append(0)
        elif classified_trade['side'] == 'sell':
            self.buy_volume_buffer.append(0)
            self.sell_volume_buffer.append(tick.volume)
        else:
            # Unknown side, split evenly
            self.buy_volume_buffer.append(tick.volume / 2)
            self.sell_volume_buffer.append(tick.volume / 2)
        
        # Store price data
        self.trade_prices.append(tick.price)
        
        # Trigger analysis if enough data
        if len(self.tick_data) >= self.analysis_window // 10:
            self._update_analysis()
    
    def add_quote(self, timestamp: datetime, bid: float, ask: float) -> None:
        """
        Add quote data for spread analysis.
        
        Args:
            timestamp: Quote timestamp
            bid: Bid price
            ask: Ask price
        """
        midpoint = (bid + ask) / 2
        self.quote_midpoints.append({
            'timestamp': timestamp,
            'midpoint': midpoint,
            'spread': ask - bid,
            'bid': bid,
            'ask': ask
        })
    
    def _classify_trade(self, tick: TickData) -> Dict[str, Any]:
        """
        Classify trade direction using Lee-Ready algorithm.
        
        Args:
            tick: Tick data to classify
            
        Returns:
            Classified trade information
        """
        if tick.side in ['buy', 'sell']:
            # Already classified
            return {
                'timestamp': tick.timestamp,
                'price': tick.price,
                'volume': tick.volume,
                'side': tick.side,
                'confidence': 1.0
            }
        
        # Find closest quote
        closest_quote = self._find_closest_quote(tick.timestamp)
        
        if not closest_quote:
            # No quote data, use tick rule
            return self._apply_tick_rule(tick)
        
        # Apply quote rule
        mid_price = closest_quote['midpoint']
        
        if tick.price > mid_price:
            side = 'buy'
            confidence = 0.8
        elif tick.price < mid_price:
            side = 'sell'
            confidence = 0.8
        else:
            # Trade at midpoint, use tick rule
            tick_result = self._apply_tick_rule(tick)
            side = tick_result['side']
            confidence = tick_result['confidence'] * 0.5  # Lower confidence
        
        return {
            'timestamp': tick.timestamp,
            'price': tick.price,
            'volume': tick.volume,
            'side': side,
            'confidence': confidence
        }
    
    def _find_closest_quote(self, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Find quote closest to given timestamp."""
        if not self.quote_midpoints:
            return None
        
        closest_quote = None
        min_time_diff = float('inf')
        
        for quote in self.quote_midpoints:
            time_diff = abs((timestamp - quote['timestamp']).total_seconds())
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_quote = quote
        
        # Only use quote if it's within 5 seconds
        if min_time_diff <= 5.0:
            return closest_quote
        
        return None
    
    def _apply_tick_rule(self, tick: TickData) -> Dict[str, Any]:
        """Apply tick rule for trade classification."""
        if len(self.trade_prices) < 2:
            return {
                'timestamp': tick.timestamp,
                'price': tick.price,
                'volume': tick.volume,
                'side': 'unknown',
                'confidence': 0.1
            }
        
        # Compare with previous trade price
        prev_price = self.trade_prices[-2]
        
        if tick.price > prev_price:
            side = 'buy'
            confidence = 0.6
        elif tick.price < prev_price:
            side = 'sell'
            confidence = 0.6
        else:
            # No price change, look further back
            if len(self.trade_prices) >= 3:
                prev_prev_price = self.trade_prices[-3]
                if prev_price > prev_prev_price:
                    side = 'buy'
                    confidence = 0.4
                elif prev_price < prev_prev_price:
                    side = 'sell'
                    confidence = 0.4
                else:
                    side = 'unknown'
                    confidence = 0.1
            else:
                side = 'unknown'
                confidence = 0.1
        
        return {
            'timestamp': tick.timestamp,
            'price': tick.price,
            'volume': tick.volume,
            'side': side,
            'confidence': confidence
        }
    
    def _update_analysis(self) -> None:
        """Update microstructure analysis."""
        if len(self.tick_data) < 50:
            return
        
        # Calculate microstructure metrics
        metrics = self._calculate_microstructure_metrics()
        if metrics:
            self.microstructure_metrics.append(metrics)
        
        # Calculate toxicity measures
        toxicity = self._calculate_toxicity_measures()
        if toxicity:
            self.toxicity_measures.append(toxicity)
        
        # Price discovery analysis (less frequent)
        if len(self.tick_data) % 100 == 0:
            discovery = self._analyze_price_discovery()
            if discovery:
                self.price_discovery_results.append(discovery)
    
    def _calculate_microstructure_metrics(self) -> Optional[MicrostructureMetrics]:
        """Calculate high-frequency microstructure metrics."""
        if len(self.classified_trades) < 20:
            return None
        
        recent_trades = list(self.classified_trades)[-50:]
        recent_quotes = list(self.quote_midpoints)[-50:] if self.quote_midpoints else []
        
        timestamp = datetime.now()
        
        # Effective spread calculation
        effective_spread = self._calculate_effective_spread(recent_trades, recent_quotes)
        
        # Realized spread calculation
        realized_spread = self._calculate_realized_spread(recent_trades, recent_quotes)
        
        # Price impact
        price_impact = effective_spread - realized_spread
        
        # Order flow toxicity
        toxicity = self._calculate_order_flow_toxicity(recent_trades)
        
        # Informed trading probability
        informed_prob = self._calculate_informed_trading_probability(recent_trades)
        
        # Adverse selection cost
        adverse_selection = max(0, price_impact)
        
        # Price discovery ratio (simplified)
        discovery_ratio = self._calculate_price_discovery_ratio()
        
        # Information share (simplified)
        info_share = self._calculate_information_share()
        
        return MicrostructureMetrics(
            timestamp=timestamp,
            price_discovery_ratio=discovery_ratio,
            information_share=info_share,
            effective_spread=effective_spread,
            realized_spread=realized_spread,
            price_impact=price_impact,
            order_flow_toxicity=toxicity,
            informed_trading_probability=informed_prob,
            adverse_selection_cost=adverse_selection
        )
    
    def _calculate_effective_spread(self,
                                  trades: List[Dict[str, Any]],
                                  quotes: List[Dict[str, Any]]) -> float:
        """Calculate effective spread."""
        if not trades or not quotes:
            return 0.0
        
        effective_spreads = []
        
        for trade in trades[-20:]:  # Last 20 trades
            # Find closest quote
            closest_quote = None
            min_time_diff = float('inf')
            
            for quote in quotes:
                time_diff = abs((trade['timestamp'] - quote['timestamp']).total_seconds())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_quote = quote
            
            if closest_quote and min_time_diff <= 5.0:
                mid_price = closest_quote['midpoint']
                if mid_price > 0:
                    # Effective spread = 2 * |trade_price - mid_price| / mid_price
                    eff_spread = 2 * abs(trade['price'] - mid_price) / mid_price
                    effective_spreads.append(eff_spread)
        
        return np.mean(effective_spreads) if effective_spreads else 0.0
    
    def _calculate_realized_spread(self,
                                 trades: List[Dict[str, Any]],
                                 quotes: List[Dict[str, Any]]) -> float:
        """Calculate realized spread."""
        if not trades or not quotes or len(trades) < 2:
            return 0.0
        
        realized_spreads = []
        
        for i, trade in enumerate(trades[-20:]):  # Last 20 trades
            # Find future midpoint (5 seconds later)
            future_time = trade['timestamp'] + timedelta(seconds=5)
            
            future_quote = None
            min_time_diff = float('inf')
            
            for quote in quotes:
                if quote['timestamp'] >= future_time:
                    time_diff = (quote['timestamp'] - future_time).total_seconds()
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        future_quote = quote
            
            if future_quote:
                # Find current midpoint
                current_quote = None
                min_time_diff = float('inf')
                
                for quote in quotes:
                    time_diff = abs((trade['timestamp'] - quote['timestamp']).total_seconds())
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        current_quote = quote
                
                if current_quote and min_time_diff <= 2.0:
                    current_mid = current_quote['midpoint']
                    future_mid = future_quote['midpoint']
                    
                    if current_mid > 0:
                        # Realized spread considers price reversal
                        side_multiplier = 1 if trade['side'] == 'buy' else -1
                        realized_spread = 2 * side_multiplier * (trade['price'] - current_mid) * (future_mid - current_mid) / (current_mid ** 2)
                        realized_spreads.append(realized_spread)
        
        return np.mean(realized_spreads) if realized_spreads else 0.0
    
    def _calculate_order_flow_toxicity(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate order flow toxicity measure."""
        if len(trades) < 10:
            return 0.0
        
        # Simple toxicity measure based on price impact persistence
        price_impacts = []
        
        for i in range(1, min(len(trades), 20)):
            current_trade = trades[-i]
            prev_trade = trades[-i-1]
            
            if prev_trade['price'] > 0:
                price_change = (current_trade['price'] - prev_trade['price']) / prev_trade['price']
                
                # Weight by volume
                volume_weight = current_trade['volume']
                weighted_impact = price_change * volume_weight
                price_impacts.append(abs(weighted_impact))
        
        if price_impacts:
            # Toxicity as persistence of price impacts
            toxicity = np.mean(price_impacts) * 100  # Scale for readability
            return min(1.0, toxicity)
        
        return 0.0
    
    def _calculate_informed_trading_probability(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate probability of informed trading (PIN approximation)."""
        if len(trades) < 20:
            return 0.5  # Neutral
        
        # Count buy and sell trades
        buy_count = sum(1 for trade in trades[-20:] if trade['side'] == 'buy')
        sell_count = sum(1 for trade in trades[-20:] if trade['side'] == 'sell')
        total_count = buy_count + sell_count
        
        if total_count == 0:
            return 0.5
        
        # Simple imbalance-based approximation
        imbalance = abs(buy_count - sell_count) / total_count
        
        # Higher imbalance suggests more informed trading
        informed_prob = 0.5 + 0.3 * imbalance  # Scale to 0.5-0.8 range
        
        return min(1.0, informed_prob)
    
    def _calculate_price_discovery_ratio(self) -> float:
        """Calculate price discovery contribution ratio."""
        if len(self.trade_prices) < 20:
            return 0.5
        
        # Simplified: measure price change efficiency
        recent_prices = list(self.trade_prices)[-20:]
        
        # Calculate total price change
        total_change = abs(recent_prices[-1] - recent_prices[0])
        
        if total_change == 0:
            return 0.5
        
        # Calculate cumulative absolute changes
        cumulative_changes = sum(
            abs(recent_prices[i] - recent_prices[i-1])
            for i in range(1, len(recent_prices))
        )
        
        if cumulative_changes == 0:
            return 0.5
        
        # Efficiency ratio
        efficiency = total_change / cumulative_changes
        
        return min(1.0, efficiency)
    
    def _calculate_information_share(self) -> float:
        """Calculate information share metric."""
        # Simplified calculation for single venue
        # In multi-venue setting, this would compare venues
        
        if not self.venue_ticks or len(self.venue_ticks) <= 1:
            return 1.0  # Single venue gets full share
        
        # Calculate volume-weighted information share
        total_volume = 0
        venue_volumes = {}
        
        for venue, ticks in self.venue_ticks.items():
            recent_ticks = list(ticks)[-50:]
            venue_volume = sum(tick.volume for tick in recent_ticks)
            venue_volumes[venue] = venue_volume
            total_volume += venue_volume
        
        if total_volume == 0:
            return 1.0 / len(self.venue_ticks)  # Equal share
        
        # Return share of dominant venue
        max_volume = max(venue_volumes.values())
        return max_volume / total_volume
    
    def _calculate_toxicity_measures(self) -> Optional[FlowToxicity]:
        """Calculate comprehensive order flow toxicity measures."""
        if len(self.buy_volume_buffer) < self.toxicity_window:
            return None
        
        buy_volumes = np.array(list(self.buy_volume_buffer))
        sell_volumes = np.array(list(self.sell_volume_buffer))
        
        total_volume = buy_volumes + sell_volumes
        
        if np.sum(total_volume) == 0:
            return FlowToxicity(
                vpin=0.5,
                toxic_flow_ratio=0.0,
                informed_volume_ratio=0.0,
                flow_imbalance=0.0,
                toxicity_trend=0.0
            )
        
        # VPIN calculation
        volume_imbalance = np.abs(buy_volumes - sell_volumes)
        vpin = np.sum(volume_imbalance) / np.sum(total_volume)
        
        # Flow imbalance
        flow_imbalance = (np.sum(buy_volumes) - np.sum(sell_volumes)) / np.sum(total_volume)
        
        # Toxic flow ratio (simplified)
        # Based on volume concentration in large trades
        large_trade_threshold = np.percentile(total_volume[total_volume > 0], 80)
        large_trade_volume = np.sum(total_volume[total_volume > large_trade_threshold])
        toxic_flow_ratio = large_trade_volume / np.sum(total_volume)
        
        # Informed volume ratio (based on price impact correlation)
        informed_volume_ratio = min(1.0, vpin * 2)  # Simplified
        
        # Toxicity trend
        if len(self.toxicity_measures) > 0:
            recent_vpin = self.toxicity_measures[-1].vpin
            toxicity_trend = vpin - recent_vpin
        else:
            toxicity_trend = 0.0
        
        return FlowToxicity(
            vpin=vpin,
            toxic_flow_ratio=toxic_flow_ratio,
            informed_volume_ratio=informed_volume_ratio,
            flow_imbalance=flow_imbalance,
            toxicity_trend=toxicity_trend
        )
    
    def _analyze_price_discovery(self) -> Optional[PriceDiscoveryMetrics]:
        """Analyze price discovery process."""
        if len(self.tick_data) < 100:
            return None
        
        # Information leadership by venue
        info_leadership = {}
        if len(self.venue_ticks) > 1:
            for venue in self.venue_ticks:
                # Simplified: based on volume and price leadership
                venue_volume = sum(tick.volume for tick in list(self.venue_ticks[venue])[-50:])
                total_volume = sum(
                    sum(tick.volume for tick in list(venue_ticks)[-50:])
                    for venue_ticks in self.venue_ticks.values()
                )
                
                if total_volume > 0:
                    info_leadership[venue] = venue_volume / total_volume
        else:
            # Single venue
            venue_name = list(self.venue_ticks.keys())[0] if self.venue_ticks else 'default'
            info_leadership[venue_name] = 1.0
        
        # Price efficiency
        price_efficiency = self._calculate_price_discovery_ratio()
        
        # Discovery speed (inverse of price adjustment time)
        discovery_speed = self._calculate_discovery_speed()
        
        # Noise to signal ratio
        noise_ratio = self._calculate_noise_to_signal_ratio()
        
        # Microstructure noise
        microstructure_noise = self._estimate_microstructure_noise()
        
        return PriceDiscoveryMetrics(
            information_leadership=info_leadership,
            price_efficiency=price_efficiency,
            discovery_speed=discovery_speed,
            noise_to_signal_ratio=noise_ratio,
            microstructure_noise=microstructure_noise
        )
    
    def _calculate_discovery_speed(self) -> float:
        """Calculate price discovery speed."""
        if len(self.trade_prices) < 20:
            return 0.5
        
        # Measure how quickly prices adjust to new information
        # Simplified: inverse of autocorrelation in price changes
        
        recent_prices = np.array(list(self.trade_prices)[-50:])
        price_changes = np.diff(recent_prices) / recent_prices[:-1]
        
        if len(price_changes) < 2:
            return 0.5
        
        # Calculate autocorrelation at lag 1
        if np.std(price_changes) > 0:
            autocorr = np.corrcoef(price_changes[:-1], price_changes[1:])[0, 1]
            
            # Lower autocorrelation = faster discovery
            discovery_speed = max(0.0, 1.0 - abs(autocorr))
        else:
            discovery_speed = 0.5
        
        return discovery_speed
    
    def _calculate_noise_to_signal_ratio(self) -> float:
        """Calculate noise to signal ratio in prices."""
        if len(self.trade_prices) < 20:
            return 0.5
        
        recent_prices = np.array(list(self.trade_prices)[-50:])
        
        # Estimate signal as trend
        x = np.arange(len(recent_prices))
        trend = np.polyfit(x, recent_prices, 1)[0]
        trend_component = trend * x + recent_prices[0]
        
        # Estimate noise as deviations from trend
        noise = recent_prices - trend_component
        signal = trend_component - trend_component[0]
        
        noise_var = np.var(noise)
        signal_var = np.var(signal)
        
        if signal_var > 0:
            noise_ratio = noise_var / (noise_var + signal_var)
        else:
            noise_ratio = 1.0
        
        return min(1.0, noise_ratio)
    
    def _estimate_microstructure_noise(self) -> float:
        """Estimate microstructure noise level."""
        if len(self.trade_prices) < 10:
            return 0.0
        
        # Use realized variance decomposition
        recent_prices = np.array(list(self.trade_prices)[-30:])
        
        # First differences
        price_changes = np.diff(recent_prices)
        
        # Microstructure noise affects first-order autocorrelation
        if len(price_changes) > 1 and np.std(price_changes) > 0:
            first_autocorr = np.corrcoef(price_changes[:-1], price_changes[1:])[0, 1]
            
            # Negative autocorrelation indicates bid-ask bounce
            noise_level = max(0.0, -first_autocorr) * np.std(price_changes)
        else:
            noise_level = 0.0
        
        return noise_level
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current microstructure metrics."""
        metrics = {}
        
        if self.microstructure_metrics:
            latest_metrics = self.microstructure_metrics[-1]
            metrics['microstructure'] = {
                'timestamp': latest_metrics.timestamp.isoformat(),
                'effective_spread': latest_metrics.effective_spread,
                'realized_spread': latest_metrics.realized_spread,
                'price_impact': latest_metrics.price_impact,
                'order_flow_toxicity': latest_metrics.order_flow_toxicity,
                'informed_trading_probability': latest_metrics.informed_trading_probability,
                'price_discovery_ratio': latest_metrics.price_discovery_ratio
            }
        
        if self.toxicity_measures:
            latest_toxicity = self.toxicity_measures[-1]
            metrics['toxicity'] = {
                'vpin': latest_toxicity.vpin,
                'toxic_flow_ratio': latest_toxicity.toxic_flow_ratio,
                'flow_imbalance': latest_toxicity.flow_imbalance,
                'toxicity_trend': latest_toxicity.toxicity_trend
            }
        
        if self.price_discovery_results:
            latest_discovery = self.price_discovery_results[-1]
            metrics['price_discovery'] = {
                'price_efficiency': latest_discovery.price_efficiency,
                'discovery_speed': latest_discovery.discovery_speed,
                'noise_to_signal_ratio': latest_discovery.noise_to_signal_ratio,
                'microstructure_noise': latest_discovery.microstructure_noise,
                'information_leadership': latest_discovery.information_leadership
            }
        
        # Add basic statistics
        if self.tick_data:
            recent_ticks = list(self.tick_data)[-100:]
            metrics['basic_stats'] = {
                'tick_count': len(recent_ticks),
                'avg_volume': np.mean([tick.volume for tick in recent_ticks]),
                'price_volatility': np.std([tick.price for tick in recent_ticks]) / np.mean([tick.price for tick in recent_ticks]) if recent_ticks else 0,
                'trade_frequency': len(recent_ticks) / ((recent_ticks[-1].timestamp - recent_ticks[0].timestamp).total_seconds() + 1) if len(recent_ticks) > 1 else 0
            }
        
        return metrics
    
    def get_toxicity_alert_level(self) -> str:
        """Get current toxicity alert level."""
        if not self.toxicity_measures:
            return 'unknown'
        
        latest_toxicity = self.toxicity_measures[-1]
        
        # Combine multiple toxicity measures
        vpin = latest_toxicity.vpin
        toxic_ratio = latest_toxicity.toxic_flow_ratio
        flow_imbalance = abs(latest_toxicity.flow_imbalance)
        
        # Weighted toxicity score
        toxicity_score = (
            0.4 * vpin +
            0.3 * toxic_ratio +
            0.3 * flow_imbalance
        )
        
        if toxicity_score > 0.8:
            return 'critical'
        elif toxicity_score > 0.6:
            return 'high'
        elif toxicity_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def get_execution_quality_score(self) -> float:
        """Get overall execution quality score."""
        if not self.microstructure_metrics:
            return 0.5
        
        latest_metrics = self.microstructure_metrics[-1]
        
        # Components of execution quality
        spread_score = max(0.0, 1.0 - latest_metrics.effective_spread * 100)
        impact_score = max(0.0, 1.0 - latest_metrics.price_impact * 100)
        toxicity_score = 1.0 - latest_metrics.order_flow_toxicity
        discovery_score = latest_metrics.price_discovery_ratio
        
        # Weighted combination
        quality_score = (
            0.3 * spread_score +
            0.3 * impact_score +
            0.2 * toxicity_score +
            0.2 * discovery_score
        )
        
        return max(0.0, min(1.0, quality_score))