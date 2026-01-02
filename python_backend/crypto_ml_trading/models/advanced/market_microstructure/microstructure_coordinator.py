"""
Market Microstructure Coordinator.

Coordinates all microstructure analysis components and provides unified insights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.advanced.market_microstructure.order_book_analyzer import OrderBookAnalyzer, OrderBookSnapshot
from models.advanced.market_microstructure.liquidity_analyzer import LiquidityAnalyzer, LiquidityState
from models.advanced.market_microstructure.market_impact_model import MarketImpactModel, ImpactPrediction
from models.advanced.market_microstructure.tick_analyzer import TickAnalyzer, TickData


@dataclass
class MicrostructureSignal:
    """Unified microstructure signal."""
    timestamp: datetime
    signal_type: str
    strength: float
    confidence: float
    components: Dict[str, float]
    recommendations: List[str]
    risk_level: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketRegime:
    """Current market regime classification."""
    regime_type: str  # 'normal', 'stressed', 'illiquid', 'volatile', 'fragmented'
    confidence: float
    duration: float
    characteristics: Dict[str, float]
    transition_probability: Dict[str, float]


@dataclass
class ExecutionRecommendation:
    """Execution strategy recommendation."""
    strategy: str
    urgency: str
    order_size_limit: float
    venue_preferences: List[str]
    timing_recommendation: str
    expected_cost: float
    risk_assessment: Dict[str, float]
    reasoning: str


class MicrostructureCoordinator:
    """
    Coordinates all market microstructure analysis components.
    
    Features:
    - Unified microstructure signal generation
    - Market regime detection and classification
    - Execution strategy recommendations
    - Real-time monitoring and alerting
    - Cross-component analysis
    - Performance tracking
    """
    
    def __init__(self,
                 update_frequency: float = 1.0,
                 regime_window: int = 300,
                 signal_aggregation_method: str = 'weighted_ensemble'):
        """
        Initialize microstructure coordinator.
        
        Args:
            update_frequency: Update frequency in seconds
            regime_window: Window for regime detection in seconds
            signal_aggregation_method: Method for aggregating signals
        """
        self.update_frequency = update_frequency
        self.regime_window = regime_window
        self.signal_aggregation_method = signal_aggregation_method
        
        # Initialize components
        self.order_book_analyzer = OrderBookAnalyzer()
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.market_impact_model = MarketImpactModel()
        self.tick_analyzer = TickAnalyzer()
        
        # Signal and regime tracking
        self.microstructure_signals: deque = deque(maxlen=1000)
        self.market_regimes: deque = deque(maxlen=100)
        self.execution_recommendations: deque = deque(maxlen=200)
        
        # Component weights for signal aggregation
        self.component_weights = {
            'order_book': 0.25,
            'liquidity': 0.30,
            'market_impact': 0.25,
            'tick_analysis': 0.20
        }
        
        # Performance tracking
        self.component_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.recommendation_accuracy: deque = deque(maxlen=100)
        
        # Real-time coordination
        self.is_running = False
        self.coordination_thread = None
        self.data_lock = threading.Lock()
        
        # Current state
        self.current_regime: Optional[MarketRegime] = None
        self.current_signal: Optional[MicrostructureSignal] = None
        
    def update_market_data(self,
                          timestamp: datetime,
                          order_book_data: Optional[Dict] = None,
                          trade_data: Optional[List[Dict]] = None,
                          quote_data: Optional[List[Dict]] = None,
                          venue_data: Optional[List[Dict]] = None) -> MicrostructureSignal:
        """
        Update all components with new market data.
        
        Args:
            timestamp: Data timestamp
            order_book_data: Order book snapshot data
            trade_data: Recent trade data
            quote_data: Quote data
            venue_data: Multi-venue data
            
        Returns:
            Unified microstructure signal
        """
        with self.data_lock:
            # Update order book analyzer
            if order_book_data:
                bids = order_book_data.get('bids', [])
                asks = order_book_data.get('asks', [])
                self.order_book_analyzer.update_order_book(timestamp, bids, asks)
            
            # Update liquidity analyzer
            if trade_data or quote_data or venue_data:
                self.liquidity_analyzer.update_market_data(
                    timestamp,
                    trade_data or [],
                    quote_data or [],
                    venue_data
                )
            
            # Update market impact model
            if trade_data:
                market_state = self._extract_market_state(order_book_data, quote_data)
                self.market_impact_model.add_trade_data(timestamp, trade_data, market_state)
            
            # Update tick analyzer
            if trade_data:
                for trade in trade_data:
                    tick = TickData(
                        timestamp=trade.get('timestamp', timestamp),
                        price=trade.get('price', 0),
                        volume=trade.get('volume', 0),
                        side=trade.get('side', 'unknown'),
                        trade_id=trade.get('trade_id'),
                        venue=trade.get('venue')
                    )
                    self.tick_analyzer.add_tick(tick)
            
            if quote_data:
                for quote in quote_data:
                    self.tick_analyzer.add_quote(
                        quote.get('timestamp', timestamp),
                        quote.get('bid', 0),
                        quote.get('ask', 0)
                    )
        
        # Generate unified signal
        signal = self._generate_unified_signal(timestamp)
        
        # Update regime detection
        self._update_regime_detection(timestamp)
        
        # Generate execution recommendations
        self._update_execution_recommendations(timestamp)
        
        return signal
    
    def _extract_market_state(self,
                            order_book_data: Optional[Dict],
                            quote_data: Optional[List[Dict]]) -> Dict[str, Any]:
        """Extract market state for impact model."""
        market_state = {}
        
        if order_book_data:
            bids = order_book_data.get('bids', [])
            asks = order_book_data.get('asks', [])
            
            if bids and asks:
                best_bid = bids[0][0] if isinstance(bids[0], (list, tuple)) else bids[0]
                best_ask = asks[0][0] if isinstance(asks[0], (list, tuple)) else asks[0]
                
                market_state['mid_price'] = (best_bid + best_ask) / 2
                market_state['spread'] = best_ask - best_bid
                
                # Calculate depth
                bid_depth = sum(level[1] if isinstance(level, (list, tuple)) else level for level in bids[:5])
                ask_depth = sum(level[1] if isinstance(level, (list, tuple)) else level for level in asks[:5])
                market_state['depth'] = bid_depth + ask_depth
        
        if quote_data:
            recent_quotes = quote_data[-10:] if len(quote_data) > 10 else quote_data
            if recent_quotes:
                volumes = [q.get('volume', 0) for q in recent_quotes]
                market_state['volume'] = np.mean(volumes) if volumes else 0
                
                prices = [q.get('price', 0) for q in recent_quotes if q.get('price', 0) > 0]
                if len(prices) > 1:
                    returns = np.diff(prices) / prices[:-1]
                    market_state['volatility'] = np.std(returns)
        
        # Set defaults
        market_state.setdefault('mid_price', 0)
        market_state.setdefault('spread', 0.001)
        market_state.setdefault('depth', 100)
        market_state.setdefault('volume', 1000)
        market_state.setdefault('volatility', 0.02)
        
        return market_state
    
    def _generate_unified_signal(self, timestamp: datetime) -> MicrostructureSignal:
        """Generate unified microstructure signal."""
        components = {}
        confidence_scores = {}
        
        # Order book component
        ob_pressure = self.order_book_analyzer.get_order_book_pressure()
        if ob_pressure:
            ob_signal = self._normalize_order_book_signal(ob_pressure)
            components['order_book'] = ob_signal
            confidence_scores['order_book'] = 0.8  # Fixed confidence for now
        
        # Liquidity component
        liquidity_summary = self.liquidity_analyzer.get_liquidity_summary()
        if liquidity_summary.get('status') != 'no_data':
            liq_signal = self._normalize_liquidity_signal(liquidity_summary)
            components['liquidity'] = liq_signal
            confidence_scores['liquidity'] = liquidity_summary.get('quality_score', 0.5)
        
        # Market impact component
        if self.market_impact_model.is_calibrated:
            # Use a representative order size for signal generation
            test_order_size = 10.0  # Example size
            market_conditions = self._get_current_market_conditions()
            
            impact_pred = self.market_impact_model.predict_impact(
                test_order_size, market_conditions, model_ensemble=True
            )
            
            impact_signal = self._normalize_impact_signal(impact_pred)
            components['market_impact'] = impact_signal
            confidence_scores['market_impact'] = impact_pred.confidence
        
        # Tick analysis component
        tick_metrics = self.tick_analyzer.get_current_metrics()
        if tick_metrics:
            tick_signal = self._normalize_tick_signal(tick_metrics)
            components['tick_analysis'] = tick_signal
            
            # Calculate confidence based on data quality
            basic_stats = tick_metrics.get('basic_stats', {})
            tick_confidence = min(1.0, basic_stats.get('tick_count', 0) / 100.0)
            confidence_scores['tick_analysis'] = tick_confidence
        
        # Aggregate signals
        if components:
            aggregated_signal = self._aggregate_component_signals(components, confidence_scores)
            overall_confidence = np.mean(list(confidence_scores.values()))
        else:
            aggregated_signal = 0.0
            overall_confidence = 0.1
        
        # Determine signal type and recommendations
        signal_type, recommendations, risk_level = self._classify_signal(
            aggregated_signal, components
        )
        
        signal = MicrostructureSignal(
            timestamp=timestamp,
            signal_type=signal_type,
            strength=abs(aggregated_signal),
            confidence=overall_confidence,
            components=components,
            recommendations=recommendations,
            risk_level=risk_level,
            metadata={
                'component_weights': self.component_weights,
                'aggregation_method': self.signal_aggregation_method
            }
        )
        
        self.microstructure_signals.append(signal)
        self.current_signal = signal
        
        return signal
    
    def _normalize_order_book_signal(self, pressure: Dict[str, float]) -> float:
        """Normalize order book pressure to signal."""
        if not pressure:
            return 0.0
        
        # Combine pressure indicators
        volume_imbalance = pressure.get('volume_imbalance', 0)
        depth_imbalance = pressure.get('depth_imbalance', 0)
        pressure_indicator = pressure.get('pressure_indicator', 0)
        
        # Weighted combination
        signal = (
            0.4 * volume_imbalance +
            0.3 * depth_imbalance +
            0.3 * pressure_indicator
        )
        
        return np.clip(signal, -1.0, 1.0)
    
    def _normalize_liquidity_signal(self, liquidity_summary: Dict[str, Any]) -> float:
        """Normalize liquidity metrics to signal."""
        if not liquidity_summary or liquidity_summary.get('status') == 'no_data':
            return 0.0
        
        regime = liquidity_summary.get('regime', 'medium')
        quality_score = liquidity_summary.get('quality_score', 0.5)
        
        # Convert regime to signal
        regime_signals = {
            'high': 0.5,
            'medium': 0.0,
            'low': -0.3,
            'stressed': -0.8
        }
        
        regime_signal = regime_signals.get(regime, 0.0)
        
        # Adjust by quality score
        quality_adjustment = (quality_score - 0.5) * 0.5
        
        signal = regime_signal + quality_adjustment
        
        return np.clip(signal, -1.0, 1.0)
    
    def _normalize_impact_signal(self, impact_pred: ImpactPrediction) -> float:
        """Normalize market impact prediction to signal."""
        # Higher impact = negative signal for execution
        impact_signal = -min(1.0, impact_pred.predicted_impact * 50)
        
        # Adjust by confidence
        confidence_adjustment = (impact_pred.confidence - 0.5) * 0.2
        
        signal = impact_signal + confidence_adjustment
        
        return np.clip(signal, -1.0, 1.0)
    
    def _normalize_tick_signal(self, tick_metrics: Dict[str, Any]) -> float:
        """Normalize tick analysis metrics to signal."""
        if not tick_metrics:
            return 0.0
        
        signal_components = []
        
        # Microstructure metrics
        microstructure = tick_metrics.get('microstructure', {})
        if microstructure:
            # Lower spreads and impacts = positive signal
            spread_signal = -microstructure.get('effective_spread', 0) * 100
            impact_signal = -microstructure.get('price_impact', 0) * 100
            
            signal_components.extend([spread_signal, impact_signal])
        
        # Toxicity metrics
        toxicity = tick_metrics.get('toxicity', {})
        if toxicity:
            # Lower toxicity = positive signal
            vpin_signal = -toxicity.get('vpin', 0)
            toxic_flow_signal = -toxicity.get('toxic_flow_ratio', 0)
            
            signal_components.extend([vpin_signal, toxic_flow_signal])
        
        # Price discovery metrics
        discovery = tick_metrics.get('price_discovery', {})
        if discovery:
            # Higher efficiency = positive signal
            efficiency_signal = discovery.get('price_efficiency', 0.5) - 0.5
            speed_signal = discovery.get('discovery_speed', 0.5) - 0.5
            
            signal_components.extend([efficiency_signal, speed_signal])
        
        if signal_components:
            signal = np.mean(signal_components)
        else:
            signal = 0.0
        
        return np.clip(signal, -1.0, 1.0)
    
    def _aggregate_component_signals(self,
                                   components: Dict[str, float],
                                   confidences: Dict[str, float]) -> float:
        """Aggregate component signals."""
        if self.signal_aggregation_method == 'weighted_ensemble':
            total_weight = 0
            weighted_sum = 0
            
            for component, signal in components.items():
                weight = self.component_weights.get(component, 1.0)
                confidence = confidences.get(component, 0.5)
                
                # Adjust weight by confidence
                adjusted_weight = weight * confidence
                
                weighted_sum += signal * adjusted_weight
                total_weight += adjusted_weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        
        elif self.signal_aggregation_method == 'simple_average':
            return np.mean(list(components.values()))
        
        elif self.signal_aggregation_method == 'confidence_weighted':
            total_confidence = sum(confidences.values())
            if total_confidence > 0:
                return sum(
                    signal * confidences[component] / total_confidence
                    for component, signal in components.items()
                )
            else:
                return np.mean(list(components.values()))
        
        else:
            return np.mean(list(components.values()))
    
    def _classify_signal(self,
                        signal: float,
                        components: Dict[str, float]) -> Tuple[str, List[str], str]:
        """Classify signal and generate recommendations."""
        recommendations = []
        
        # Determine signal type
        if signal > 0.3:
            signal_type = 'favorable_execution'
            risk_level = 'low'
            recommendations.extend([
                'Favorable conditions for execution',
                'Consider increasing position sizes',
                'Good time for market orders'
            ])
        elif signal > 0.1:
            signal_type = 'neutral_execution'
            risk_level = 'medium'
            recommendations.extend([
                'Neutral market conditions',
                'Use standard execution strategies',
                'Monitor for changes'
            ])
        elif signal > -0.2:
            signal_type = 'cautious_execution'
            risk_level = 'medium'
            recommendations.extend([
                'Exercise caution in execution',
                'Consider smaller order sizes',
                'Use limit orders when possible'
            ])
        else:
            signal_type = 'adverse_execution'
            risk_level = 'high'
            recommendations.extend([
                'Adverse market conditions',
                'Defer non-urgent orders',
                'Use passive strategies only',
                'Monitor for improvement'
            ])
        
        # Add component-specific recommendations
        if 'order_book' in components and components['order_book'] < -0.5:
            recommendations.append('Order book showing significant imbalance')
        
        if 'liquidity' in components and components['liquidity'] < -0.5:
            recommendations.append('Low liquidity conditions detected')
        
        if 'market_impact' in components and components['market_impact'] < -0.5:
            recommendations.append('High market impact expected')
        
        if 'tick_analysis' in components and components['tick_analysis'] < -0.5:
            recommendations.append('Toxic order flow detected')
        
        return signal_type, recommendations, risk_level
    
    def _get_current_market_conditions(self) -> Dict[str, Any]:
        """Get current market conditions for impact model."""
        # Aggregate current conditions from all components
        conditions = {
            'mid_price': 0,
            'spread': 0.001,
            'depth': 100,
            'volume': 1000,
            'volatility': 0.02
        }
        
        # Get from order book
        if self.order_book_analyzer.current_book:
            book = self.order_book_analyzer.current_book
            conditions['mid_price'] = book.mid_price
            conditions['spread'] = book.spread
        
        # Get from liquidity analyzer
        liquidity_summary = self.liquidity_analyzer.get_liquidity_summary()
        if liquidity_summary.get('status') != 'no_data':
            metrics = liquidity_summary.get('metrics', {})
            if 'total_depth' in metrics:
                conditions['depth'] = metrics['total_depth']
        
        return conditions
    
    def _update_regime_detection(self, timestamp: datetime) -> None:
        """Update market regime detection."""
        if len(self.microstructure_signals) < 10:
            return
        
        # Analyze recent signals for regime classification
        recent_signals = list(self.microstructure_signals)[-20:]
        
        # Calculate regime indicators
        avg_signal = np.mean([s.strength for s in recent_signals])
        signal_volatility = np.std([s.strength for s in recent_signals])
        avg_confidence = np.mean([s.confidence for s in recent_signals])
        
        # Get component-specific indicators
        liquidity_summary = self.liquidity_analyzer.get_liquidity_summary()
        toxicity_level = self.tick_analyzer.get_toxicity_alert_level()
        
        # Classify regime
        regime_type = self._classify_market_regime(
            avg_signal, signal_volatility, avg_confidence,
            liquidity_summary, toxicity_level
        )
        
        # Calculate transition probabilities
        transition_probs = self._calculate_transition_probabilities()
        
        # Calculate regime characteristics
        characteristics = {
            'signal_strength': avg_signal,
            'signal_volatility': signal_volatility,
            'confidence_level': avg_confidence,
            'liquidity_quality': liquidity_summary.get('quality_score', 0.5),
            'toxicity_level': self._toxicity_level_to_numeric(toxicity_level)
        }
        
        # Calculate regime duration
        duration = 0.0
        if (self.current_regime and 
            self.current_regime.regime_type == regime_type):
            duration = (timestamp - self.market_regimes[-1].timestamp if self.market_regimes else timestamp).total_seconds()
        
        regime = MarketRegime(
            regime_type=regime_type,
            confidence=avg_confidence,
            duration=duration,
            characteristics=characteristics,
            transition_probability=transition_probs
        )
        
        self.market_regimes.append(regime)
        self.current_regime = regime
    
    def _classify_market_regime(self,
                              avg_signal: float,
                              signal_volatility: float,
                              avg_confidence: float,
                              liquidity_summary: Dict,
                              toxicity_level: str) -> str:
        """Classify current market regime."""
        # Get liquidity regime
        liquidity_regime = liquidity_summary.get('regime', 'medium')
        
        # Toxicity scoring
        toxicity_scores = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8,
            'critical': 1.0,
            'unknown': 0.5
        }
        toxicity_score = toxicity_scores.get(toxicity_level, 0.5)
        
        # Regime classification logic
        if toxicity_score > 0.8 or signal_volatility > 0.5:
            return 'stressed'
        elif liquidity_regime == 'stressed' or avg_signal < -0.5:
            return 'illiquid'
        elif signal_volatility > 0.3:
            return 'volatile'
        elif liquidity_summary.get('fragmentation_index', 0) > 0.7:
            return 'fragmented'
        else:
            return 'normal'
    
    def _toxicity_level_to_numeric(self, toxicity_level: str) -> float:
        """Convert toxicity level to numeric value."""
        levels = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8,
            'critical': 1.0,
            'unknown': 0.5
        }
        return levels.get(toxicity_level, 0.5)
    
    def _calculate_transition_probabilities(self) -> Dict[str, float]:
        """Calculate regime transition probabilities."""
        if len(self.market_regimes) < 5:
            return {'normal': 0.8, 'stressed': 0.1, 'illiquid': 0.05, 'volatile': 0.03, 'fragmented': 0.02}
        
        # Analyze recent regime transitions
        recent_regimes = [r.regime_type for r in list(self.market_regimes)[-10:]]
        
        # Simple transition probability estimation
        transition_counts = defaultdict(int)
        for regime in recent_regimes:
            transition_counts[regime] += 1
        
        total_count = sum(transition_counts.values())
        transition_probs = {
            regime: count / total_count
            for regime, count in transition_counts.items()
        }
        
        # Ensure all regimes have some probability
        all_regimes = ['normal', 'stressed', 'illiquid', 'volatile', 'fragmented']
        for regime in all_regimes:
            if regime not in transition_probs:
                transition_probs[regime] = 0.01
        
        # Normalize
        total_prob = sum(transition_probs.values())
        transition_probs = {
            regime: prob / total_prob
            for regime, prob in transition_probs.items()
        }
        
        return transition_probs
    
    def _update_execution_recommendations(self, timestamp: datetime) -> None:
        """Update execution strategy recommendations."""
        if not self.current_signal or not self.current_regime:
            return
        
        # Generate recommendations based on current conditions
        recommendation = self._generate_execution_recommendation(timestamp)
        
        self.execution_recommendations.append(recommendation)
    
    def _generate_execution_recommendation(self, timestamp: datetime) -> ExecutionRecommendation:
        """Generate execution strategy recommendation."""
        signal = self.current_signal
        regime = self.current_regime
        
        # Base recommendations on regime and signal
        if regime.regime_type == 'stressed':
            strategy = 'passive_only'
            urgency = 'low'
            order_size_limit = 0.1  # 10% of normal
            timing = 'defer_non_urgent'
            reasoning = 'Market is stressed, minimize market impact'
            
        elif regime.regime_type == 'illiquid':
            strategy = 'small_parcels'
            urgency = 'low'
            order_size_limit = 0.2
            timing = 'spread_over_time'
            reasoning = 'Low liquidity, use smaller order sizes'
            
        elif regime.regime_type == 'volatile':
            strategy = 'adaptive_timing'
            urgency = 'medium'
            order_size_limit = 0.5
            timing = 'wait_for_calm'
            reasoning = 'High volatility, time execution carefully'
            
        elif regime.regime_type == 'fragmented':
            strategy = 'multi_venue'
            urgency = 'medium'
            order_size_limit = 0.7
            timing = 'coordinate_venues'
            reasoning = 'Fragmented liquidity, spread across venues'
            
        else:  # normal
            if signal.strength > 0.3:
                strategy = 'aggressive'
                urgency = 'high'
                order_size_limit = 1.0
                timing = 'immediate'
                reasoning = 'Favorable conditions for execution'
            else:
                strategy = 'standard'
                urgency = 'medium'
                order_size_limit = 0.8
                timing = 'standard'
                reasoning = 'Normal market conditions'
        
        # Venue preferences (simplified)
        venue_preferences = self._generate_venue_preferences()
        
        # Expected cost estimation
        expected_cost = self._estimate_execution_cost(strategy, regime.regime_type)
        
        # Risk assessment
        risk_assessment = {
            'market_impact_risk': regime.characteristics.get('signal_strength', 0.5),
            'timing_risk': regime.characteristics.get('signal_volatility', 0.3),
            'liquidity_risk': 1.0 - regime.characteristics.get('liquidity_quality', 0.5),
            'toxicity_risk': regime.characteristics.get('toxicity_level', 0.5)
        }
        
        return ExecutionRecommendation(
            strategy=strategy,
            urgency=urgency,
            order_size_limit=order_size_limit,
            venue_preferences=venue_preferences,
            timing_recommendation=timing,
            expected_cost=expected_cost,
            risk_assessment=risk_assessment,
            reasoning=reasoning
        )
    
    def _generate_venue_preferences(self) -> List[str]:
        """Generate venue preferences based on current analysis."""
        # Simplified venue ranking
        # In practice, this would analyze venue-specific metrics
        
        if self.current_regime and self.current_regime.regime_type == 'fragmented':
            return ['venue_1', 'venue_2', 'venue_3']  # Spread across multiple venues
        else:
            return ['primary_venue', 'secondary_venue']  # Concentrate on top venues
    
    def _estimate_execution_cost(self, strategy: str, regime: str) -> float:
        """Estimate execution cost based on strategy and regime."""
        base_costs = {
            'passive_only': 0.002,
            'small_parcels': 0.003,
            'adaptive_timing': 0.004,
            'multi_venue': 0.005,
            'standard': 0.006,
            'aggressive': 0.008
        }
        
        regime_multipliers = {
            'normal': 1.0,
            'volatile': 1.3,
            'illiquid': 1.5,
            'fragmented': 1.2,
            'stressed': 2.0
        }
        
        base_cost = base_costs.get(strategy, 0.005)
        multiplier = regime_multipliers.get(regime, 1.0)
        
        return base_cost * multiplier
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get comprehensive current status."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'components': {
                'order_book': self.order_book_analyzer.get_liquidity_summary(),
                'liquidity': self.liquidity_analyzer.get_liquidity_summary(),
                'market_impact': self.market_impact_model.get_model_summary(),
                'tick_analysis': self.tick_analyzer.get_current_metrics()
            }
        }
        
        if self.current_signal:
            status['current_signal'] = {
                'type': self.current_signal.signal_type,
                'strength': self.current_signal.strength,
                'confidence': self.current_signal.confidence,
                'risk_level': self.current_signal.risk_level,
                'recommendations': self.current_signal.recommendations
            }
        
        if self.current_regime:
            status['current_regime'] = {
                'type': self.current_regime.regime_type,
                'confidence': self.current_regime.confidence,
                'duration': self.current_regime.duration,
                'characteristics': self.current_regime.characteristics
            }
        
        if self.execution_recommendations:
            latest_rec = self.execution_recommendations[-1]
            status['execution_recommendation'] = {
                'strategy': latest_rec.strategy,
                'urgency': latest_rec.urgency,
                'order_size_limit': latest_rec.order_size_limit,
                'expected_cost': latest_rec.expected_cost,
                'reasoning': latest_rec.reasoning
            }
        
        return status
    
    def start_coordination(self) -> None:
        """Start real-time coordination."""
        if self.is_running:
            return
        
        self.is_running = True
        self.coordination_thread = threading.Thread(
            target=self._coordination_loop,
            daemon=True
        )
        self.coordination_thread.start()
        print("Microstructure coordination started")
    
    def stop_coordination(self) -> None:
        """Stop real-time coordination."""
        self.is_running = False
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5)
        print("Microstructure coordination stopped")
    
    def _coordination_loop(self) -> None:
        """Main coordination loop."""
        while self.is_running:
            try:
                # Periodic maintenance and analysis
                self._update_component_performance()
                self._cleanup_old_data()
                
                time.sleep(self.update_frequency)
                
            except Exception as e:
                print(f"Error in coordination loop: {e}")
                time.sleep(self.update_frequency)
    
    def _update_component_performance(self) -> None:
        """Update component performance metrics."""
        # This would be implemented with actual performance tracking
        # For now, just placeholder
        pass
    
    def _cleanup_old_data(self) -> None:
        """Clean up old data from components."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Clean up old signals and regimes
        # (deques handle this automatically with maxlen)