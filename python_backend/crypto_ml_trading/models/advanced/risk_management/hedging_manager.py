"""
Advanced Hedging Manager for Portfolio Risk Management.

Implements sophisticated hedging strategies including delta hedging, volatility hedging,
tail risk hedging, and dynamic hedging with options and derivatives.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


@dataclass
class HedgingInstrument:
    """Hedging instrument definition."""
    instrument_id: str
    instrument_type: str  # option, future, swap, spot
    underlying_asset: str
    hedge_ratio: float
    cost: float
    effectiveness: float
    maturity: Optional[datetime] = None
    strike_price: Optional[float] = None
    option_type: Optional[str] = None  # call, put
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None


@dataclass
class HedgePosition:
    """Active hedge position."""
    position_id: str
    instrument: HedgingInstrument
    quantity: float
    entry_price: float
    entry_time: datetime
    target_hedge_ratio: float
    current_hedge_ratio: float
    pnl: float = 0.0
    effectiveness_score: float = 0.0
    status: str = "active"  # active, expired, closed


@dataclass
class HedgingStrategy:
    """Hedging strategy definition."""
    strategy_id: str
    strategy_name: str
    strategy_type: str  # delta_hedge, volatility_hedge, tail_hedge, basis_hedge
    target_assets: List[str]
    hedge_instruments: List[str]
    rebalancing_frequency: int  # hours
    max_hedge_ratio: float = 1.0
    min_hedge_ratio: float = 0.0
    cost_threshold: float = 0.01  # 1% of portfolio
    effectiveness_threshold: float = 0.7


@dataclass
class HedgingSignal:
    """Hedging recommendation signal."""
    timestamp: datetime
    signal_type: str  # hedge_increase, hedge_decrease, hedge_close, hedge_open
    target_asset: str
    recommended_instrument: str
    hedge_ratio: float
    urgency: str  # low, medium, high, critical
    rationale: List[str]
    expected_cost: float
    expected_effectiveness: float
    risk_reduction: float


class HedgingManager:
    """
    Advanced hedging manager for portfolio risk management.
    
    Features:
    - Delta hedging for directional risk
    - Volatility hedging for volatility risk
    - Tail risk hedging for extreme scenarios
    - Cross-asset hedging strategies
    - Dynamic hedge rebalancing
    - Cost-effectiveness optimization
    - Hedge performance monitoring
    - Multi-instrument hedging
    """
    
    def __init__(self,
                 max_hedge_budget: float = 0.05,  # 5% of portfolio
                 rebalancing_frequency: int = 4,  # hours
                 effectiveness_threshold: float = 0.6,
                 cost_efficiency_threshold: float = 0.1):
        """
        Initialize hedging manager.
        
        Args:
            max_hedge_budget: Maximum hedging budget as fraction of portfolio
            rebalancing_frequency: Hedge rebalancing frequency (hours)
            effectiveness_threshold: Minimum hedge effectiveness required
            cost_efficiency_threshold: Maximum cost per unit of risk reduction
        """
        self.max_hedge_budget = max_hedge_budget
        self.rebalancing_frequency = rebalancing_frequency
        self.effectiveness_threshold = effectiveness_threshold
        self.cost_efficiency_threshold = cost_efficiency_threshold
        
        # Hedging instruments database
        self.available_instruments: Dict[str, HedgingInstrument] = {}
        self.hedging_strategies: Dict[str, HedgingStrategy] = {}
        
        # Active hedging positions
        self.active_hedges: Dict[str, HedgePosition] = {}
        self.hedge_history: deque = deque(maxlen=1000)
        
        # Portfolio and market data
        self.portfolio_positions: Dict[str, float] = {}
        self.portfolio_value: float = 0.0
        self.asset_prices: Dict[str, float] = {}
        self.asset_volatilities: Dict[str, float] = {}
        self.correlations: Dict[Tuple[str, str], float] = {}
        
        # Risk metrics tracking
        self.portfolio_delta: Dict[str, float] = {}
        self.portfolio_gamma: Dict[str, float] = {}
        self.portfolio_vega: Dict[str, float] = {}
        self.risk_exposures: Dict[str, float] = {}
        
        # Hedging performance
        self.hedging_performance: deque = deque(maxlen=500)
        self.cost_tracking: Dict[str, float] = defaultdict(float)
        self.effectiveness_tracking: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Last rebalance time
        self.last_rebalance: Optional[datetime] = None
        
        # Initialize default instruments and strategies
        self._initialize_default_instruments()
        self._initialize_default_strategies()
    
    def _initialize_default_instruments(self) -> None:
        """Initialize default hedging instruments."""
        # Default instruments for common crypto assets
        default_instruments = [
            # Bitcoin hedging instruments
            HedgingInstrument(
                instrument_id="btc_put_hedge",
                instrument_type="option",
                underlying_asset="bitcoin",
                hedge_ratio=0.8,
                cost=0.02,
                effectiveness=0.85,
                option_type="put",
                delta=-0.4,
                gamma=0.01,
                vega=0.1,
                theta=-0.005
            ),
            
            # Ethereum hedging instruments
            HedgingInstrument(
                instrument_id="eth_put_hedge",
                instrument_type="option",
                underlying_asset="ethereum",
                hedge_ratio=0.75,
                cost=0.025,
                effectiveness=0.8,
                option_type="put",
                delta=-0.35,
                gamma=0.012,
                vega=0.12,
                theta=-0.006
            ),
            
            # Crypto index futures
            HedgingInstrument(
                instrument_id="crypto_index_future",
                instrument_type="future",
                underlying_asset="crypto_index",
                hedge_ratio=0.9,
                cost=0.001,
                effectiveness=0.9,
                delta=1.0
            ),
            
            # Volatility instruments
            HedgingInstrument(
                instrument_id="crypto_vol_swap",
                instrument_type="swap",
                underlying_asset="crypto_volatility",
                hedge_ratio=0.6,
                cost=0.015,
                effectiveness=0.7,
                vega=1.0
            ),
            
            # Stablecoin hedge
            HedgingInstrument(
                instrument_id="usdt_spot",
                instrument_type="spot",
                underlying_asset="usdt",
                hedge_ratio=1.0,
                cost=0.0001,
                effectiveness=0.95,
                delta=0.0
            )
        ]
        
        for instrument in default_instruments:
            self.available_instruments[instrument.instrument_id] = instrument
    
    def _initialize_default_strategies(self) -> None:
        """Initialize default hedging strategies."""
        strategies = [
            # Delta hedging strategy
            HedgingStrategy(
                strategy_id="delta_hedge_btc",
                strategy_name="Bitcoin Delta Hedging",
                strategy_type="delta_hedge",
                target_assets=["bitcoin", "btc"],
                hedge_instruments=["btc_put_hedge", "crypto_index_future"],
                rebalancing_frequency=6,
                max_hedge_ratio=0.8,
                cost_threshold=0.02
            ),
            
            # Volatility hedging strategy
            HedgingStrategy(
                strategy_id="vol_hedge_portfolio",
                strategy_name="Portfolio Volatility Hedging",
                strategy_type="volatility_hedge",
                target_assets=["bitcoin", "ethereum"],
                hedge_instruments=["crypto_vol_swap", "btc_put_hedge"],
                rebalancing_frequency=12,
                max_hedge_ratio=0.5,
                cost_threshold=0.025
            ),
            
            # Tail risk hedging
            HedgingStrategy(
                strategy_id="tail_risk_hedge",
                strategy_name="Tail Risk Protection",
                strategy_type="tail_hedge",
                target_assets=["bitcoin", "ethereum", "altcoins"],
                hedge_instruments=["btc_put_hedge", "eth_put_hedge"],
                rebalancing_frequency=24,
                max_hedge_ratio=0.3,
                cost_threshold=0.01,
                effectiveness_threshold=0.8
            )
        ]
        
        for strategy in strategies:
            self.hedging_strategies[strategy.strategy_id] = strategy
    
    def update_portfolio_data(self,
                            positions: Dict[str, float],
                            portfolio_value: float,
                            asset_prices: Dict[str, float],
                            asset_volatilities: Dict[str, float]) -> None:
        """
        Update portfolio data for hedging calculations.
        
        Args:
            positions: Portfolio positions {asset: weight}
            portfolio_value: Current portfolio value
            asset_prices: Asset prices {asset: price}
            asset_volatilities: Asset volatilities {asset: volatility}
        """
        self.portfolio_positions = positions.copy()
        self.portfolio_value = portfolio_value
        self.asset_prices = asset_prices.copy()
        self.asset_volatilities = asset_volatilities.copy()
        
        # Update risk exposures
        self._calculate_risk_exposures()
        
        # Check if rebalancing is needed
        if self._should_rebalance():
            self._rebalance_hedges()
    
    def _calculate_risk_exposures(self) -> None:
        """Calculate portfolio risk exposures."""
        self.risk_exposures.clear()
        self.portfolio_delta.clear()
        
        for asset, position in self.portfolio_positions.items():
            position_value = position * self.portfolio_value
            asset_volatility = self.asset_volatilities.get(asset, 0.02)
            
            # Calculate risk exposure (position * volatility)
            risk_exposure = abs(position_value) * asset_volatility
            self.risk_exposures[asset] = risk_exposure
            
            # Calculate delta exposure
            self.portfolio_delta[asset] = position_value
    
    def _should_rebalance(self) -> bool:
        """Check if hedge rebalancing is needed."""
        if not self.last_rebalance:
            return True
        
        time_since_rebalance = (datetime.now() - self.last_rebalance).total_seconds() / 3600
        return time_since_rebalance >= self.rebalancing_frequency
    
    def _rebalance_hedges(self) -> None:
        """Rebalance active hedges."""
        timestamp = datetime.now()
        
        # Evaluate current hedge effectiveness
        self._evaluate_hedge_effectiveness()
        
        # Generate new hedging signals
        hedging_signals = self._generate_hedging_signals()
        
        # Execute hedging actions
        for signal in hedging_signals:
            self._execute_hedging_signal(signal)
        
        self.last_rebalance = timestamp
    
    def generate_hedging_recommendations(self) -> List[HedgingSignal]:
        """Generate hedging recommendations for current portfolio."""
        return self._generate_hedging_signals()
    
    def _generate_hedging_signals(self) -> List[HedgingSignal]:
        """Generate hedging signals based on current portfolio state."""
        signals = []
        timestamp = datetime.now()
        
        # Analyze each hedging strategy
        for strategy_id, strategy in self.hedging_strategies.items():
            strategy_signals = self._analyze_strategy(strategy, timestamp)
            signals.extend(strategy_signals)
        
        # Prioritize signals by urgency and effectiveness
        signals.sort(key=lambda x: (x.urgency, -x.expected_effectiveness))
        
        return signals
    
    def _analyze_strategy(self, strategy: HedgingStrategy, timestamp: datetime) -> List[HedgingSignal]:
        """Analyze specific hedging strategy and generate signals."""
        signals = []
        
        if strategy.strategy_type == "delta_hedge":
            signals.extend(self._analyze_delta_hedging(strategy, timestamp))
        elif strategy.strategy_type == "volatility_hedge":
            signals.extend(self._analyze_volatility_hedging(strategy, timestamp))
        elif strategy.strategy_type == "tail_hedge":
            signals.extend(self._analyze_tail_risk_hedging(strategy, timestamp))
        
        return signals
    
    def _analyze_delta_hedging(self, strategy: HedgingStrategy, timestamp: datetime) -> List[HedgingSignal]:
        """Analyze delta hedging requirements."""
        signals = []
        
        for asset in strategy.target_assets:
            if asset not in self.portfolio_positions:
                continue
            
            # Calculate current delta exposure
            current_delta = self.portfolio_delta.get(asset, 0.0)
            
            # Check if hedging is needed
            delta_threshold = self.portfolio_value * 0.05  # 5% of portfolio
            
            if abs(current_delta) > delta_threshold:
                # Find best hedging instrument
                best_instrument = self._find_best_hedge_instrument(
                    asset, "delta_hedge", strategy.hedge_instruments
                )
                
                if best_instrument:
                    # Calculate required hedge ratio
                    required_hedge_ratio = min(
                        abs(current_delta) / self.portfolio_value,
                        strategy.max_hedge_ratio
                    )
                    
                    # Determine urgency
                    urgency = self._calculate_urgency(abs(current_delta), self.portfolio_value)
                    
                    signal = HedgingSignal(
                        timestamp=timestamp,
                        signal_type="hedge_open" if required_hedge_ratio > 0.1 else "hedge_increase",
                        target_asset=asset,
                        recommended_instrument=best_instrument.instrument_id,
                        hedge_ratio=required_hedge_ratio,
                        urgency=urgency,
                        rationale=[
                            f"Delta exposure {current_delta:.2f} exceeds threshold",
                            f"Recommended hedge ratio: {required_hedge_ratio:.2f}"
                        ],
                        expected_cost=best_instrument.cost * required_hedge_ratio * self.portfolio_value,
                        expected_effectiveness=best_instrument.effectiveness,
                        risk_reduction=required_hedge_ratio * best_instrument.effectiveness
                    )
                    signals.append(signal)
        
        return signals
    
    def _analyze_volatility_hedging(self, strategy: HedgingStrategy, timestamp: datetime) -> List[HedgingSignal]:
        """Analyze volatility hedging requirements."""
        signals = []
        
        # Calculate portfolio volatility exposure
        portfolio_vol_exposure = 0.0
        for asset in strategy.target_assets:
            if asset in self.portfolio_positions and asset in self.asset_volatilities:
                position_value = abs(self.portfolio_positions[asset]) * self.portfolio_value
                asset_vol = self.asset_volatilities[asset]
                portfolio_vol_exposure += position_value * asset_vol
        
        # Check if volatility hedging is needed
        vol_threshold = self.portfolio_value * 0.15  # 15% volatility exposure threshold
        
        if portfolio_vol_exposure > vol_threshold:
            # Find volatility hedging instruments
            vol_instruments = [
                inst for inst_id, inst in self.available_instruments.items()
                if inst_id in strategy.hedge_instruments and inst.vega is not None
            ]
            
            if vol_instruments:
                best_instrument = max(vol_instruments, key=lambda x: x.effectiveness / x.cost)
                
                required_hedge_ratio = min(
                    portfolio_vol_exposure / self.portfolio_value,
                    strategy.max_hedge_ratio
                )
                
                urgency = self._calculate_urgency(portfolio_vol_exposure, self.portfolio_value)
                
                signal = HedgingSignal(
                    timestamp=timestamp,
                    signal_type="hedge_open",
                    target_asset="portfolio_volatility",
                    recommended_instrument=best_instrument.instrument_id,
                    hedge_ratio=required_hedge_ratio,
                    urgency=urgency,
                    rationale=[
                        f"Portfolio volatility exposure {portfolio_vol_exposure:.2f} exceeds threshold",
                        f"Volatility hedging recommended"
                    ],
                    expected_cost=best_instrument.cost * required_hedge_ratio * self.portfolio_value,
                    expected_effectiveness=best_instrument.effectiveness,
                    risk_reduction=required_hedge_ratio * best_instrument.effectiveness
                )
                signals.append(signal)
        
        return signals
    
    def _analyze_tail_risk_hedging(self, strategy: HedgingStrategy, timestamp: datetime) -> List[HedgingSignal]:
        """Analyze tail risk hedging requirements."""
        signals = []
        
        # Calculate tail risk exposure (simplified)
        total_risk_exposure = sum(self.risk_exposures.values())
        tail_risk_threshold = self.portfolio_value * 0.1  # 10% tail risk threshold
        
        if total_risk_exposure > tail_risk_threshold:
            # Find tail risk hedging instruments (typically put options)
            tail_instruments = [
                inst for inst_id, inst in self.available_instruments.items()
                if (inst_id in strategy.hedge_instruments and 
                    inst.instrument_type == "option" and 
                    inst.option_type == "put")
            ]
            
            if tail_instruments:
                # Select multiple instruments for diversified tail protection
                for instrument in tail_instruments[:2]:  # Top 2 instruments
                    required_hedge_ratio = min(
                        total_risk_exposure / self.portfolio_value * 0.3,  # 30% of risk exposure
                        strategy.max_hedge_ratio
                    )
                    
                    if required_hedge_ratio > 0.05:  # Minimum 5% hedge ratio
                        urgency = "medium" if total_risk_exposure > tail_risk_threshold * 1.5 else "low"
                        
                        signal = HedgingSignal(
                            timestamp=timestamp,
                            signal_type="hedge_open",
                            target_asset="tail_risk",
                            recommended_instrument=instrument.instrument_id,
                            hedge_ratio=required_hedge_ratio,
                            urgency=urgency,
                            rationale=[
                                f"Total risk exposure {total_risk_exposure:.2f} exceeds tail risk threshold",
                                f"Tail risk protection recommended"
                            ],
                            expected_cost=instrument.cost * required_hedge_ratio * self.portfolio_value,
                            expected_effectiveness=instrument.effectiveness,
                            risk_reduction=required_hedge_ratio * instrument.effectiveness
                        )
                        signals.append(signal)
        
        return signals
    
    def _find_best_hedge_instrument(self,
                                  target_asset: str,
                                  hedge_type: str,
                                  allowed_instruments: List[str]) -> Optional[HedgingInstrument]:
        """Find best hedging instrument for target asset."""
        candidates = []
        
        for inst_id in allowed_instruments:
            if inst_id in self.available_instruments:
                instrument = self.available_instruments[inst_id]
                
                # Check if instrument is suitable for target asset
                if self._is_instrument_suitable(instrument, target_asset, hedge_type):
                    # Calculate cost-effectiveness score
                    score = instrument.effectiveness / max(instrument.cost, 0.001)
                    candidates.append((score, instrument))
        
        if candidates:
            # Return instrument with best cost-effectiveness score
            return max(candidates, key=lambda x: x[0])[1]
        
        return None
    
    def _is_instrument_suitable(self,
                              instrument: HedgingInstrument,
                              target_asset: str,
                              hedge_type: str) -> bool:
        """Check if instrument is suitable for hedging target asset."""
        # Asset compatibility
        if instrument.underlying_asset != target_asset:
            # Check for related assets (e.g., bitcoin/btc)
            asset_aliases = {
                'bitcoin': ['btc', 'bitcoin'],
                'ethereum': ['eth', 'ethereum'],
                'crypto_index': ['bitcoin', 'ethereum', 'btc', 'eth']
            }
            
            underlying_aliases = asset_aliases.get(instrument.underlying_asset, [instrument.underlying_asset])
            if target_asset.lower() not in [alias.lower() for alias in underlying_aliases]:
                return False
        
        # Hedge type compatibility
        if hedge_type == "delta_hedge" and instrument.delta is None:
            return False
        elif hedge_type == "volatility_hedge" and instrument.vega is None:
            return False
        
        return True
    
    def _calculate_urgency(self, exposure: float, portfolio_value: float) -> str:
        """Calculate urgency level for hedging signal."""
        exposure_ratio = abs(exposure) / portfolio_value
        
        if exposure_ratio > 0.3:
            return "critical"
        elif exposure_ratio > 0.2:
            return "high"
        elif exposure_ratio > 0.1:
            return "medium"
        else:
            return "low"
    
    def _execute_hedging_signal(self, signal: HedgingSignal) -> None:
        """Execute hedging signal (simulate hedge position creation)."""
        # In a real implementation, this would interface with trading systems
        
        if signal.recommended_instrument not in self.available_instruments:
            return
        
        instrument = self.available_instruments[signal.recommended_instrument]
        
        # Create hedge position
        position_id = f"hedge_{signal.target_asset}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate position size
        position_value = signal.hedge_ratio * self.portfolio_value
        quantity = position_value / self.asset_prices.get(signal.target_asset, 1.0)
        
        hedge_position = HedgePosition(
            position_id=position_id,
            instrument=instrument,
            quantity=quantity,
            entry_price=self.asset_prices.get(signal.target_asset, 0.0),
            entry_time=signal.timestamp,
            target_hedge_ratio=signal.hedge_ratio,
            current_hedge_ratio=signal.hedge_ratio,
            effectiveness_score=signal.expected_effectiveness
        )
        
        self.active_hedges[position_id] = hedge_position
        
        # Track cost
        self.cost_tracking[signal.target_asset] += signal.expected_cost
        
        print(f"Executed hedge: {signal.signal_type} for {signal.target_asset} "
              f"using {signal.recommended_instrument} (ratio: {signal.hedge_ratio:.2f})")
    
    def _evaluate_hedge_effectiveness(self) -> None:
        """Evaluate effectiveness of active hedges."""
        for position_id, hedge_position in self.active_hedges.items():
            # Calculate current effectiveness (simplified)
            target_asset = hedge_position.instrument.underlying_asset
            
            if target_asset in self.asset_prices:
                current_price = self.asset_prices[target_asset]
                price_change = (current_price - hedge_position.entry_price) / hedge_position.entry_price
                
                # Estimate hedge PnL based on instrument type
                if hedge_position.instrument.instrument_type == "option":
                    if hedge_position.instrument.option_type == "put":
                        # Put option gains when underlying falls
                        hedge_pnl = -price_change * hedge_position.quantity * hedge_position.instrument.delta
                    else:
                        # Call option gains when underlying rises
                        hedge_pnl = price_change * hedge_position.quantity * hedge_position.instrument.delta
                else:
                    # Future or forward
                    hedge_pnl = -price_change * hedge_position.quantity  # Opposite direction
                
                hedge_position.pnl = hedge_pnl
                
                # Calculate effectiveness score
                portfolio_loss = price_change * self.portfolio_positions.get(target_asset, 0) * self.portfolio_value
                if portfolio_loss != 0:
                    effectiveness = min(1.0, abs(hedge_pnl / portfolio_loss))
                    hedge_position.effectiveness_score = effectiveness
                    
                    # Store effectiveness tracking
                    self.effectiveness_tracking[target_asset].append(effectiveness)
    
    def close_hedge_position(self, position_id: str, reason: str = "manual") -> bool:
        """
        Close an active hedge position.
        
        Args:
            position_id: ID of position to close
            reason: Reason for closing
            
        Returns:
            True if position was closed successfully
        """
        if position_id not in self.active_hedges:
            return False
        
        hedge_position = self.active_hedges[position_id]
        hedge_position.status = "closed"
        
        # Move to history
        self.hedge_history.append(hedge_position)
        del self.active_hedges[position_id]
        
        print(f"Closed hedge position {position_id}: {reason}")
        return True
    
    def get_hedge_summary(self) -> Dict[str, Any]:
        """Get comprehensive hedging summary."""
        summary = {
            'active_hedges': len(self.active_hedges),
            'total_hedge_cost': sum(self.cost_tracking.values()),
            'hedge_budget_used': sum(self.cost_tracking.values()) / self.portfolio_value if self.portfolio_value > 0 else 0,
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None
        }
        
        # Active hedge details
        if self.active_hedges:
            active_details = []
            total_hedge_value = 0.0
            
            for position_id, hedge in self.active_hedges.items():
                hedge_value = hedge.quantity * self.asset_prices.get(hedge.instrument.underlying_asset, 0)
                total_hedge_value += hedge_value
                
                active_details.append({
                    'position_id': position_id,
                    'instrument': hedge.instrument.instrument_id,
                    'target_asset': hedge.instrument.underlying_asset,
                    'hedge_ratio': hedge.current_hedge_ratio,
                    'pnl': hedge.pnl,
                    'effectiveness': hedge.effectiveness_score
                })
            
            summary['active_hedge_details'] = active_details
            summary['total_hedge_value'] = total_hedge_value
        
        # Effectiveness analysis
        if self.effectiveness_tracking:
            effectiveness_summary = {}
            for asset, effectiveness_history in self.effectiveness_tracking.items():
                if effectiveness_history:
                    effectiveness_summary[asset] = {
                        'avg_effectiveness': np.mean(list(effectiveness_history)),
                        'recent_effectiveness': list(effectiveness_history)[-1] if effectiveness_history else 0,
                        'effectiveness_trend': 'improving' if len(effectiveness_history) > 1 and 
                                             list(effectiveness_history)[-1] > list(effectiveness_history)[-2] else 'stable'
                    }
            
            summary['effectiveness_analysis'] = effectiveness_summary
        
        return summary
    
    def get_hedging_recommendations(self) -> Dict[str, Any]:
        """Get current hedging recommendations."""
        signals = self.generate_hedging_recommendations()
        
        recommendations = {
            'immediate_actions': [],
            'monitoring_required': [],
            'cost_analysis': {},
            'risk_analysis': {}
        }
        
        # Categorize signals
        for signal in signals:
            if signal.urgency in ['critical', 'high']:
                recommendations['immediate_actions'].append({
                    'action': signal.signal_type,
                    'asset': signal.target_asset,
                    'instrument': signal.recommended_instrument,
                    'urgency': signal.urgency,
                    'cost': signal.expected_cost,
                    'rationale': signal.rationale
                })
            else:
                recommendations['monitoring_required'].append({
                    'asset': signal.target_asset,
                    'current_exposure': self.risk_exposures.get(signal.target_asset, 0),
                    'recommended_action': signal.signal_type
                })
        
        # Cost analysis
        total_recommended_cost = sum(s.expected_cost for s in signals)
        recommendations['cost_analysis'] = {
            'total_cost': total_recommended_cost,
            'cost_as_pct_portfolio': total_recommended_cost / self.portfolio_value if self.portfolio_value > 0 else 0,
            'within_budget': total_recommended_cost <= self.max_hedge_budget * self.portfolio_value
        }
        
        # Risk analysis
        total_risk_reduction = sum(s.risk_reduction for s in signals)
        recommendations['risk_analysis'] = {
            'total_risk_exposure': sum(self.risk_exposures.values()),
            'potential_risk_reduction': total_risk_reduction,
            'unhedged_risk': sum(self.risk_exposures.values()) - total_risk_reduction
        }
        
        return recommendations
    
    def add_custom_instrument(self, instrument: HedgingInstrument) -> None:
        """Add custom hedging instrument."""
        self.available_instruments[instrument.instrument_id] = instrument
        print(f"Added custom hedging instrument: {instrument.instrument_id}")
    
    def add_custom_strategy(self, strategy: HedgingStrategy) -> None:
        """Add custom hedging strategy."""
        self.hedging_strategies[strategy.strategy_id] = strategy
        print(f"Added custom hedging strategy: {strategy.strategy_id}")
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """Get hedging performance analysis."""
        if not self.hedge_history and not self.active_hedges:
            return {'status': 'no_hedge_data'}
        
        all_hedges = list(self.hedge_history) + list(self.active_hedges.values())
        
        analysis = {
            'total_hedges': len(all_hedges),
            'avg_effectiveness': np.mean([h.effectiveness_score for h in all_hedges if h.effectiveness_score > 0]),
            'total_hedge_pnl': sum(h.pnl for h in all_hedges),
            'cost_efficiency': {}
        }
        
        # Analyze by instrument type
        by_instrument = defaultdict(list)
        for hedge in all_hedges:
            by_instrument[hedge.instrument.instrument_type].append(hedge)
        
        for instrument_type, hedges in by_instrument.items():
            analysis['cost_efficiency'][instrument_type] = {
                'count': len(hedges),
                'avg_effectiveness': np.mean([h.effectiveness_score for h in hedges if h.effectiveness_score > 0]),
                'total_pnl': sum(h.pnl for h in hedges)
            }
        
        return analysis