"""
Liquidity Analysis for Market Microstructure.

Implements comprehensive liquidity analysis and fragmentation detection.
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
class LiquidityState:
    """Current liquidity state of the market."""
    timestamp: datetime
    total_depth: float
    effective_spread: float
    resilience_score: float
    fragmentation_index: float
    liquidity_ratio: float
    market_quality_score: float
    regime: str  # 'high', 'medium', 'low', 'stressed'


@dataclass
class LiquidityEvent:
    """Liquidity event detection."""
    timestamp: datetime
    event_type: str
    severity: str
    duration: float
    impact_score: float
    description: str
    metrics: Dict[str, float]


@dataclass
class VenueLiquidity:
    """Liquidity metrics for a specific venue/exchange."""
    venue_id: str
    timestamp: datetime
    depth: float
    spread: float
    volume_share: float
    price_impact: float
    execution_quality: float
    latency_score: float


class LiquidityAnalyzer:
    """
    Advanced liquidity analysis for cryptocurrency markets.
    
    Features:
    - Multi-dimensional liquidity measurement
    - Liquidity fragmentation analysis
    - Real-time liquidity monitoring
    - Venue-specific liquidity analysis
    - Liquidity regime detection
    - Market quality assessment
    - Execution cost analysis
    """
    
    def __init__(self,
                 depth_levels: List[int] = None,
                 fragmentation_threshold: float = 0.3,
                 resilience_window: int = 30,
                 quality_weights: Dict[str, float] = None):
        """
        Initialize liquidity analyzer.
        
        Args:
            depth_levels: Depth analysis levels in basis points
            fragmentation_threshold: Threshold for fragmentation detection
            resilience_window: Window for resilience calculation (seconds)
            quality_weights: Weights for market quality components
        """
        self.depth_levels = depth_levels or [5, 10, 25, 50, 100, 250]
        self.fragmentation_threshold = fragmentation_threshold
        self.resilience_window = resilience_window
        self.quality_weights = quality_weights or {
            'spread': 0.3,
            'depth': 0.25,
            'resilience': 0.2,
            'fragmentation': 0.15,
            'volatility': 0.1
        }
        
        # State tracking
        self.liquidity_history: deque = deque(maxlen=1000)
        self.venue_liquidity: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.liquidity_events: deque = deque(maxlen=200)
        
        # Market data
        self.trade_data: deque = deque(maxlen=2000)
        self.quote_data: deque = deque(maxlen=1000)
        self.price_history: deque = deque(maxlen=1000)
        
        # Analysis state
        self.current_regime = 'medium'
        self.fragmentation_factors: Dict[str, float] = {}
        self.liquidity_providers: Dict[str, Dict] = {}
        
    def update_market_data(self,
                          timestamp: datetime,
                          trades: List[Dict],
                          quotes: List[Dict],
                          venues: Optional[List[Dict]] = None) -> LiquidityState:
        """
        Update liquidity analysis with new market data.
        
        Args:
            timestamp: Update timestamp
            trades: Recent trade data
            quotes: Current quote data
            venues: Venue-specific data
            
        Returns:
            Current liquidity state
        """
        # Store market data
        self.trade_data.extend(trades)
        self.quote_data.extend(quotes)
        
        # Extract prices
        if trades:
            prices = [trade.get('price', 0) for trade in trades]
            self.price_history.extend(prices)
        
        # Update venue liquidity if provided
        if venues:
            self._update_venue_liquidity(timestamp, venues)
        
        # Calculate liquidity metrics
        liquidity_state = self._calculate_liquidity_state(timestamp)
        
        # Store state
        self.liquidity_history.append(liquidity_state)
        
        # Detect events
        events = self._detect_liquidity_events(liquidity_state)
        self.liquidity_events.extend(events)
        
        return liquidity_state
    
    def _update_venue_liquidity(self, timestamp: datetime, venues: List[Dict]) -> None:
        """Update venue-specific liquidity metrics."""
        total_volume = sum(venue.get('volume', 0) for venue in venues)
        
        for venue_data in venues:
            venue_id = venue_data.get('venue_id', 'unknown')
            
            # Calculate venue metrics
            venue_liquidity = VenueLiquidity(
                venue_id=venue_id,
                timestamp=timestamp,
                depth=venue_data.get('depth', 0),
                spread=venue_data.get('spread', 0),
                volume_share=venue_data.get('volume', 0) / total_volume if total_volume > 0 else 0,
                price_impact=venue_data.get('price_impact', 0),
                execution_quality=self._calculate_execution_quality(venue_data),
                latency_score=venue_data.get('latency_score', 0)
            )
            
            self.venue_liquidity[venue_id].append(venue_liquidity)
    
    def _calculate_execution_quality(self, venue_data: Dict) -> float:
        """Calculate execution quality score for a venue."""
        # Simplified execution quality calculation
        spread_component = 1.0 - min(1.0, venue_data.get('spread', 0) / 0.01)  # Normalize spread
        depth_component = min(1.0, venue_data.get('depth', 0) / 100.0)  # Normalize depth
        latency_component = 1.0 - min(1.0, venue_data.get('latency', 0) / 100.0)  # Normalize latency
        
        quality_score = (
            0.4 * spread_component +
            0.4 * depth_component +
            0.2 * latency_component
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def _calculate_liquidity_state(self, timestamp: datetime) -> LiquidityState:
        """Calculate current liquidity state."""
        # Calculate total depth across levels
        total_depth = self._calculate_total_depth()
        
        # Calculate effective spread
        effective_spread = self._calculate_effective_spread()
        
        # Calculate resilience score
        resilience_score = self._calculate_resilience_score()
        
        # Calculate fragmentation index
        fragmentation_index = self._calculate_fragmentation_index()
        
        # Calculate liquidity ratio
        liquidity_ratio = self._calculate_liquidity_ratio()
        
        # Calculate market quality score
        market_quality_score = self._calculate_market_quality_score(
            total_depth, effective_spread, resilience_score, fragmentation_index
        )
        
        # Determine liquidity regime
        regime = self._determine_liquidity_regime(market_quality_score)
        
        return LiquidityState(
            timestamp=timestamp,
            total_depth=total_depth,
            effective_spread=effective_spread,
            resilience_score=resilience_score,
            fragmentation_index=fragmentation_index,
            liquidity_ratio=liquidity_ratio,
            market_quality_score=market_quality_score,
            regime=regime
        )
    
    def _calculate_total_depth(self) -> float:
        """Calculate total market depth."""
        if not self.quote_data:
            return 0.0
        
        # Use most recent quote data
        recent_quotes = list(self.quote_data)[-10:]
        
        total_depth = 0.0
        for quote in recent_quotes:
            bid_depth = quote.get('bid_depth', 0)
            ask_depth = quote.get('ask_depth', 0)
            total_depth += bid_depth + ask_depth
        
        return total_depth / len(recent_quotes) if recent_quotes else 0.0
    
    def _calculate_effective_spread(self) -> float:
        """Calculate effective spread."""
        if not self.trade_data or not self.quote_data:
            return 0.0
        
        # Get recent trades and quotes
        recent_trades = [trade for trade in self.trade_data if 'price' in trade][-20:]
        recent_quotes = list(self.quote_data)[-10:]
        
        if not recent_trades or not recent_quotes:
            return 0.0
        
        effective_spreads = []
        
        for trade in recent_trades:
            trade_price = trade['price']
            trade_time = trade.get('timestamp', datetime.now())
            
            # Find closest quote
            closest_quote = min(
                recent_quotes,
                key=lambda q: abs((q.get('timestamp', datetime.now()) - trade_time).total_seconds())
            )
            
            mid_price = closest_quote.get('mid_price', trade_price)
            
            if mid_price > 0:
                # Calculate effective spread
                effective_spread = 2 * abs(trade_price - mid_price) / mid_price
                effective_spreads.append(effective_spread)
        
        return np.mean(effective_spreads) if effective_spreads else 0.0
    
    def _calculate_resilience_score(self) -> float:
        """Calculate market resilience score."""
        if len(self.price_history) < 20:
            return 0.5  # Default neutral score
        
        # Analyze price mean reversion
        recent_prices = list(self.price_history)[-50:]
        
        if len(recent_prices) < 10:
            return 0.5
        
        # Calculate returns
        returns = np.diff(recent_prices) / recent_prices[:-1]
        
        # Measure mean reversion
        # High resilience = quick return to equilibrium after shocks
        
        # Calculate autocorrelation at lag 1
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            
            # Negative autocorrelation indicates mean reversion (good resilience)
            resilience_score = max(0.0, min(1.0, 0.5 - autocorr))
        else:
            resilience_score = 0.5
        
        # Adjust based on volatility
        volatility = np.std(returns) if len(returns) > 1 else 0
        volatility_factor = max(0.5, min(1.5, 1.0 - volatility * 10))
        
        return resilience_score * volatility_factor
    
    def _calculate_fragmentation_index(self) -> float:
        """Calculate market fragmentation index."""
        if not self.venue_liquidity:
            return 0.0
        
        # Calculate venue concentration
        venue_volumes = []
        total_volume = 0
        
        for venue_id, venue_history in self.venue_liquidity.items():
            if venue_history:
                recent_volume = np.mean([v.volume_share for v in list(venue_history)[-5:]])
                venue_volumes.append(recent_volume)
                total_volume += recent_volume
        
        if not venue_volumes or total_volume == 0:
            return 0.0
        
        # Normalize volumes
        normalized_volumes = [v / total_volume for v in venue_volumes]
        
        # Calculate Herfindahl-Hirschman Index
        hhi = sum(v ** 2 for v in normalized_volumes)
        
        # Convert to fragmentation index (0 = concentrated, 1 = fragmented)
        fragmentation_index = 1.0 - hhi
        
        # Adjust for number of venues
        num_venues = len(venue_volumes)
        if num_venues > 1:
            max_fragmentation = 1.0 - (1.0 / num_venues)
            if max_fragmentation > 0:
                fragmentation_index = fragmentation_index / max_fragmentation
        
        return min(1.0, fragmentation_index)
    
    def _calculate_liquidity_ratio(self) -> float:
        """Calculate liquidity ratio (depth to volatility)."""
        if not self.price_history or len(self.price_history) < 10:
            return 0.0
        
        # Calculate recent volatility
        recent_prices = list(self.price_history)[-20:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns) if len(returns) > 1 else 0.01
        
        # Get current depth
        current_depth = self._calculate_total_depth()
        
        # Calculate ratio
        if volatility > 0:
            liquidity_ratio = current_depth / volatility
            # Normalize to 0-1 scale
            return min(1.0, liquidity_ratio / 1000.0)
        
        return 0.0
    
    def _calculate_market_quality_score(self,
                                      depth: float,
                                      spread: float,
                                      resilience: float,
                                      fragmentation: float) -> float:
        """Calculate overall market quality score."""
        # Normalize components
        depth_score = min(1.0, depth / 100.0)  # Normalize depth
        spread_score = max(0.0, 1.0 - spread * 100)  # Lower spread is better
        resilience_score = resilience
        fragmentation_score = max(0.0, 1.0 - fragmentation)  # Lower fragmentation is better
        
        # Calculate volatility score
        volatility_score = 0.5  # Default
        if len(self.price_history) > 10:
            recent_prices = list(self.price_history)[-20:]
            returns = np.diff(recent_prices) / recent_prices[:-1]
            volatility = np.std(returns) if len(returns) > 1 else 0
            volatility_score = max(0.0, min(1.0, 1.0 - volatility * 50))
        
        # Weighted combination
        quality_score = (
            self.quality_weights['depth'] * depth_score +
            self.quality_weights['spread'] * spread_score +
            self.quality_weights['resilience'] * resilience_score +
            self.quality_weights['fragmentation'] * fragmentation_score +
            self.quality_weights['volatility'] * volatility_score
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def _determine_liquidity_regime(self, quality_score: float) -> str:
        """Determine current liquidity regime."""
        if quality_score >= 0.8:
            return 'high'
        elif quality_score >= 0.6:
            return 'medium'
        elif quality_score >= 0.4:
            return 'low'
        else:
            return 'stressed'
    
    def _detect_liquidity_events(self, current_state: LiquidityState) -> List[LiquidityEvent]:
        """Detect significant liquidity events."""
        events = []
        
        if len(self.liquidity_history) < 5:
            return events
        
        recent_states = list(self.liquidity_history)[-5:]
        prev_state = recent_states[-2] if len(recent_states) > 1 else current_state
        
        # Detect liquidity crisis
        if (current_state.market_quality_score < 0.3 and 
            prev_state.market_quality_score > 0.5):
            events.append(LiquidityEvent(
                timestamp=current_state.timestamp,
                event_type='liquidity_crisis',
                severity='high',
                duration=0.0,  # Would be calculated based on event history
                impact_score=1.0 - current_state.market_quality_score,
                description='Sudden deterioration in market liquidity',
                metrics={
                    'quality_score': current_state.market_quality_score,
                    'depth': current_state.total_depth,
                    'spread': current_state.effective_spread
                }
            ))
        
        # Detect fragmentation increase
        if (current_state.fragmentation_index > self.fragmentation_threshold and
            current_state.fragmentation_index > prev_state.fragmentation_index * 1.5):
            events.append(LiquidityEvent(
                timestamp=current_state.timestamp,
                event_type='fragmentation_increase',
                severity='medium',
                duration=0.0,
                impact_score=current_state.fragmentation_index,
                description='Significant increase in market fragmentation',
                metrics={
                    'fragmentation_index': current_state.fragmentation_index,
                    'previous_fragmentation': prev_state.fragmentation_index
                }
            ))
        
        # Detect spread widening
        if (current_state.effective_spread > prev_state.effective_spread * 2 and
            current_state.effective_spread > 0.01):
            events.append(LiquidityEvent(
                timestamp=current_state.timestamp,
                event_type='spread_widening',
                severity='high' if current_state.effective_spread > 0.02 else 'medium',
                duration=0.0,
                impact_score=current_state.effective_spread * 100,
                description='Significant widening of effective spread',
                metrics={
                    'current_spread': current_state.effective_spread,
                    'previous_spread': prev_state.effective_spread
                }
            ))
        
        # Detect depth depletion
        if (current_state.total_depth < prev_state.total_depth * 0.5 and
            prev_state.total_depth > 10):
            events.append(LiquidityEvent(
                timestamp=current_state.timestamp,
                event_type='depth_depletion',
                severity='high',
                duration=0.0,
                impact_score=1.0 - (current_state.total_depth / prev_state.total_depth),
                description='Significant depletion of market depth',
                metrics={
                    'current_depth': current_state.total_depth,
                    'previous_depth': prev_state.total_depth
                }
            ))
        
        return events
    
    def get_liquidity_summary(self) -> Dict[str, Any]:
        """Get comprehensive liquidity summary."""
        if not self.liquidity_history:
            return {'status': 'no_data'}
        
        current_state = self.liquidity_history[-1]
        
        summary = {
            'timestamp': current_state.timestamp.isoformat(),
            'regime': current_state.regime,
            'quality_score': current_state.market_quality_score,
            'metrics': {
                'total_depth': current_state.total_depth,
                'effective_spread': current_state.effective_spread,
                'resilience_score': current_state.resilience_score,
                'fragmentation_index': current_state.fragmentation_index,
                'liquidity_ratio': current_state.liquidity_ratio
            }
        }
        
        # Add trend analysis
        if len(self.liquidity_history) >= 10:
            recent_scores = [state.market_quality_score for state in list(self.liquidity_history)[-10:]]
            trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            
            summary['trend'] = {
                'direction': 'improving' if trend > 0.01 else 'deteriorating' if trend < -0.01 else 'stable',
                'slope': trend,
                'volatility': np.std(recent_scores)
            }
        
        # Add venue analysis
        if self.venue_liquidity:
            venue_summary = {}
            for venue_id, venue_history in self.venue_liquidity.items():
                if venue_history:
                    recent_venue = list(venue_history)[-1]
                    venue_summary[venue_id] = {
                        'volume_share': recent_venue.volume_share,
                        'execution_quality': recent_venue.execution_quality,
                        'spread': recent_venue.spread,
                        'depth': recent_venue.depth
                    }
            
            summary['venues'] = venue_summary
        
        # Add recent events
        recent_events = [
            {
                'type': event.event_type,
                'severity': event.severity,
                'timestamp': event.timestamp.isoformat(),
                'description': event.description
            }
            for event in list(self.liquidity_events)[-5:]
        ]
        summary['recent_events'] = recent_events
        
        return summary
    
    def get_venue_comparison(self) -> Dict[str, Any]:
        """Compare liquidity across different venues."""
        if not self.venue_liquidity:
            return {}
        
        comparison = {}
        
        for venue_id, venue_history in self.venue_liquidity.items():
            if not venue_history:
                continue
            
            recent_data = list(venue_history)[-10:]
            
            venue_stats = {
                'avg_volume_share': np.mean([v.volume_share for v in recent_data]),
                'avg_execution_quality': np.mean([v.execution_quality for v in recent_data]),
                'avg_spread': np.mean([v.spread for v in recent_data]),
                'avg_depth': np.mean([v.depth for v in recent_data]),
                'consistency': 1.0 - np.std([v.execution_quality for v in recent_data])
            }
            
            comparison[venue_id] = venue_stats
        
        # Rank venues
        if comparison:
            # Rank by execution quality
            quality_ranking = sorted(
                comparison.items(),
                key=lambda x: x[1]['avg_execution_quality'],
                reverse=True
            )
            
            comparison['rankings'] = {
                'by_quality': [venue_id for venue_id, _ in quality_ranking],
                'by_volume': sorted(
                    comparison.keys(),
                    key=lambda x: comparison[x]['avg_volume_share'],
                    reverse=True
                ),
                'by_depth': sorted(
                    comparison.keys(),
                    key=lambda x: comparison[x]['avg_depth'],
                    reverse=True
                )
            }
        
        return comparison
    
    def calculate_execution_costs(self, 
                                order_size: float,
                                side: str = 'buy') -> Dict[str, float]:
        """Calculate estimated execution costs for an order."""
        if not self.liquidity_history:
            return {}
        
        current_state = self.liquidity_history[-1]
        
        # Base cost components
        spread_cost = current_state.effective_spread / 2  # Half spread
        
        # Market impact estimation
        market_impact = self._estimate_market_impact(order_size, current_state)
        
        # Timing risk (based on volatility)
        timing_risk = self._estimate_timing_risk()
        
        # Opportunity cost
        opportunity_cost = self._estimate_opportunity_cost(current_state)
        
        total_cost = spread_cost + market_impact + timing_risk + opportunity_cost
        
        return {
            'total_cost_bps': total_cost * 10000,  # Convert to basis points
            'spread_cost_bps': spread_cost * 10000,
            'market_impact_bps': market_impact * 10000,
            'timing_risk_bps': timing_risk * 10000,
            'opportunity_cost_bps': opportunity_cost * 10000,
            'liquidity_regime': current_state.regime,
            'confidence': current_state.market_quality_score
        }
    
    def _estimate_market_impact(self, order_size: float, state: LiquidityState) -> float:
        """Estimate market impact for given order size."""
        # Simple square-root model
        if state.total_depth > 0:
            impact = 0.01 * np.sqrt(order_size / state.total_depth)
        else:
            impact = 0.02  # Default high impact
        
        # Adjust for liquidity regime
        regime_multipliers = {
            'high': 0.7,
            'medium': 1.0,
            'low': 1.5,
            'stressed': 2.5
        }
        
        multiplier = regime_multipliers.get(state.regime, 1.0)
        return impact * multiplier
    
    def _estimate_timing_risk(self) -> float:
        """Estimate timing risk based on recent volatility."""
        if len(self.price_history) < 10:
            return 0.001  # Default low risk
        
        recent_prices = list(self.price_history)[-20:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns) if len(returns) > 1 else 0.001
        
        # Timing risk increases with volatility
        return min(0.01, volatility * 2)
    
    def _estimate_opportunity_cost(self, state: LiquidityState) -> float:
        """Estimate opportunity cost based on market conditions."""
        # Higher opportunity cost in low liquidity regimes
        regime_costs = {
            'high': 0.0001,
            'medium': 0.0002,
            'low': 0.0005,
            'stressed': 0.001
        }
        
        return regime_costs.get(state.regime, 0.0002)
    
    def get_optimal_execution_strategy(self, 
                                     order_size: float,
                                     urgency: str = 'medium') -> Dict[str, Any]:
        """Recommend optimal execution strategy."""
        if not self.liquidity_history:
            return {'strategy': 'wait', 'reason': 'insufficient_data'}
        
        current_state = self.liquidity_history[-1]
        execution_costs = self.calculate_execution_costs(order_size)
        
        strategy = {
            'recommended_strategy': '',
            'reasoning': '',
            'execution_costs': execution_costs,
            'market_conditions': {
                'regime': current_state.regime,
                'quality_score': current_state.market_quality_score,
                'fragmentation': current_state.fragmentation_index
            }
        }
        
        # Strategy logic based on conditions
        if current_state.regime == 'stressed':
            if urgency == 'high':
                strategy['recommended_strategy'] = 'aggressive_immediate'
                strategy['reasoning'] = 'Market stressed but high urgency requires immediate execution'
            else:
                strategy['recommended_strategy'] = 'wait_for_improvement'
                strategy['reasoning'] = 'Market stressed, wait for better conditions'
        
        elif current_state.regime == 'high':
            strategy['recommended_strategy'] = 'market_order'
            strategy['reasoning'] = 'High liquidity regime supports immediate execution'
        
        elif current_state.fragmentation_index > 0.7:
            strategy['recommended_strategy'] = 'venue_splitting'
            strategy['reasoning'] = 'High fragmentation suggests splitting across venues'
        
        else:
            if urgency == 'low':
                strategy['recommended_strategy'] = 'twap_passive'
                strategy['reasoning'] = 'Medium liquidity, low urgency supports passive strategy'
            else:
                strategy['recommended_strategy'] = 'twap_aggressive'
                strategy['reasoning'] = 'Medium liquidity with higher urgency'
        
        return strategy