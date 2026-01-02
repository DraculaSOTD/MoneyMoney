"""
Advanced Position Sizing System.

Implements sophisticated position sizing strategies including Kelly Criterion,
risk parity, and dynamic sizing based on market conditions.
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
class PositionSizeResult:
    """Result of position sizing calculation."""
    timestamp: datetime
    asset_symbol: str
    recommended_size: float
    max_size: float
    min_size: float
    risk_budget_used: float
    sizing_method: str
    confidence: float
    rationale: List[str]
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskBudget:
    """Risk budget allocation."""
    total_risk_budget: float
    asset_allocations: Dict[str, float]
    strategy_allocations: Dict[str, float]
    time_horizon_allocations: Dict[str, float]
    used_budget: float = 0.0
    available_budget: float = 0.0


@dataclass
class PositionConstraints:
    """Position sizing constraints."""
    max_position_size: float = 0.2  # 20% of portfolio
    min_position_size: float = 0.01  # 1% of portfolio
    max_sector_exposure: float = 0.4  # 40% to any sector
    max_single_asset_risk: float = 0.05  # 5% VaR per asset
    max_correlation_exposure: float = 0.6  # 60% to correlated assets
    liquidity_constraint: float = 0.1  # 10% of daily volume
    leverage_limit: float = 2.0  # Maximum 2x leverage


class PositionSizer:
    """
    Advanced position sizing system for cryptocurrency trading.
    
    Features:
    - Kelly Criterion optimization
    - Risk parity position sizing
    - Volatility-based sizing
    - Dynamic sizing based on market regime
    - Risk budget allocation
    - Correlation-aware sizing
    - Liquidity-constrained sizing
    - Multi-objective position optimization
    """
    
    def __init__(self,
                 default_risk_budget: float = 0.02,  # 2% portfolio risk per trade
                 sizing_method: str = "kelly_criterion",
                 lookback_period: int = 252,
                 rebalancing_frequency: int = 24):  # hours
        """
        Initialize position sizer.
        
        Args:
            default_risk_budget: Default risk budget per position
            sizing_method: Primary sizing method
            lookback_period: Historical data lookback period
            rebalancing_frequency: Position rebalancing frequency in hours
        """
        self.default_risk_budget = default_risk_budget
        self.sizing_method = sizing_method
        self.lookback_period = lookback_period
        self.rebalancing_frequency = rebalancing_frequency
        
        # Risk budget management
        self.risk_budget = RiskBudget(
            total_risk_budget=0.1,  # 10% total portfolio risk
            asset_allocations={},
            strategy_allocations={},
            time_horizon_allocations={}
        )
        
        # Position constraints
        self.constraints = PositionConstraints()
        
        # Historical data storage
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.return_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.volatility_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Position history
        self.position_history: deque = deque(maxlen=1000)
        self.sizing_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        
        # Market regime tracking
        self.current_regime: str = "normal"
        self.regime_adjustments: Dict[str, float] = {
            "bull": 1.2,
            "bear": 0.6,
            "volatile": 0.7,
            "sideways": 0.9,
            "normal": 1.0
        }
        
        # Correlation matrix
        self.correlation_matrix: Optional[np.ndarray] = None
        self.asset_symbols: List[str] = []
        
        # Available sizing methods
        self.sizing_methods = {
            "kelly_criterion": self._kelly_criterion_sizing,
            "risk_parity": self._risk_parity_sizing,
            "volatility_based": self._volatility_based_sizing,
            "fixed_risk": self._fixed_risk_sizing,
            "momentum_based": self._momentum_based_sizing,
            "mean_reversion": self._mean_reversion_sizing,
            "multi_objective": self._multi_objective_sizing
        }
    
    def add_price_data(self,
                      asset_symbol: str,
                      prices: List[float],
                      timestamps: List[datetime]) -> None:
        """
        Add price data for position sizing calculations.
        
        Args:
            asset_symbol: Asset symbol
            prices: Price series
            timestamps: Corresponding timestamps
        """
        for price, timestamp in zip(prices, timestamps):
            self.price_history[asset_symbol].append({
                'price': price,
                'timestamp': timestamp
            })
        
        # Calculate returns
        self._update_returns(asset_symbol)
        
        # Update volatility
        self._update_volatility(asset_symbol)
        
        # Update correlation matrix
        self._update_correlation_matrix()
    
    def _update_returns(self, asset_symbol: str) -> None:
        """Update return series for asset."""
        prices = [p['price'] for p in self.price_history[asset_symbol]]
        
        if len(prices) > 1:
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] != 0:
                    return_val = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(return_val)
            
            # Store recent returns
            if returns:
                for ret in returns[-10:]:  # Store last 10 returns
                    self.return_history[asset_symbol].append(ret)
    
    def _update_volatility(self, asset_symbol: str) -> None:
        """Update volatility estimates for asset."""
        returns = list(self.return_history[asset_symbol])
        
        if len(returns) > 10:
            # Calculate rolling volatility
            window_size = min(20, len(returns))
            recent_returns = returns[-window_size:]
            volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized
            
            self.volatility_history[asset_symbol].append(volatility)
    
    def _update_correlation_matrix(self) -> None:
        """Update correlation matrix between assets."""
        # Get common assets with sufficient data
        assets_with_data = [
            asset for asset in self.return_history
            if len(self.return_history[asset]) > 20
        ]
        
        if len(assets_with_data) < 2:
            return
        
        # Prepare return matrix
        min_length = min(len(self.return_history[asset]) for asset in assets_with_data)
        return_matrix = np.zeros((min_length, len(assets_with_data)))
        
        for i, asset in enumerate(assets_with_data):
            returns = list(self.return_history[asset])[-min_length:]
            return_matrix[:, i] = returns
        
        # Calculate correlation matrix
        self.correlation_matrix = np.corrcoef(return_matrix.T)
        self.asset_symbols = assets_with_data
    
    def calculate_position_size(self,
                               asset_symbol: str,
                               prediction: Dict[str, Any],
                               portfolio_value: float,
                               current_position: float = 0.0,
                               method: Optional[str] = None) -> PositionSizeResult:
        """
        Calculate optimal position size for asset.
        
        Args:
            asset_symbol: Asset to size
            prediction: Prediction data including expected return and confidence
            portfolio_value: Current portfolio value
            current_position: Current position size
            method: Sizing method to use (optional)
            
        Returns:
            Position sizing result
        """
        timestamp = datetime.now()
        method = method or self.sizing_method
        
        # Get sizing function
        sizing_func = self.sizing_methods.get(method, self._kelly_criterion_sizing)
        
        # Calculate recommended size
        size_result = sizing_func(
            asset_symbol, prediction, portfolio_value, current_position
        )
        
        # Apply constraints
        constrained_size = self._apply_constraints(
            asset_symbol, size_result.recommended_size, portfolio_value
        )
        
        # Update result with constraints
        size_result.recommended_size = constrained_size
        size_result.timestamp = timestamp
        size_result.sizing_method = method
        
        # Calculate risk budget usage
        risk_budget_used = self._calculate_risk_budget_usage(
            asset_symbol, constrained_size, portfolio_value
        )
        size_result.risk_budget_used = risk_budget_used
        
        # Store sizing decision
        self.position_history.append(size_result)
        
        return size_result
    
    def _kelly_criterion_sizing(self,
                               asset_symbol: str,
                               prediction: Dict[str, Any],
                               portfolio_value: float,
                               current_position: float) -> PositionSizeResult:
        """Kelly Criterion position sizing."""
        # Extract prediction parameters
        expected_return = prediction.get('expected_return', 0.0)
        confidence = prediction.get('confidence', 0.5)
        win_rate = prediction.get('win_rate', 0.5)
        
        # Get historical volatility
        volatility = self._get_asset_volatility(asset_symbol)
        if volatility == 0:
            volatility = 0.02  # Default 2% volatility
        
        # Kelly fraction calculation
        # Kelly = (bp - q) / b
        # where b = odds received on winning bet
        #       p = probability of winning
        #       q = probability of losing (1-p)
        
        # Adjust for confidence
        adjusted_win_rate = win_rate * confidence
        
        # Calculate expected odds based on expected return
        if expected_return > 0:
            # For positive expected returns
            b = abs(expected_return) / volatility  # Risk-adjusted odds
            kelly_fraction = (b * adjusted_win_rate - (1 - adjusted_win_rate)) / b
        else:
            # For negative expected returns, size is zero
            kelly_fraction = 0.0
        
        # Apply Kelly safety factor
        kelly_safety_factor = 0.25  # Use 25% of full Kelly
        kelly_fraction *= kelly_safety_factor
        
        # Convert to position size
        recommended_size = max(0.0, kelly_fraction)
        
        # Calculate max and min sizes
        max_size = self.constraints.max_position_size
        min_size = self.constraints.min_position_size if recommended_size > 0 else 0.0
        
        rationale = [
            f"Kelly Criterion with safety factor {kelly_safety_factor}",
            f"Expected return: {expected_return:.4f}",
            f"Adjusted win rate: {adjusted_win_rate:.4f}",
            f"Volatility: {volatility:.4f}"
        ]
        
        risk_metrics = {
            'volatility': volatility,
            'expected_return': expected_return,
            'kelly_fraction': kelly_fraction,
            'safety_factor': kelly_safety_factor
        }
        
        return PositionSizeResult(
            timestamp=datetime.now(),
            asset_symbol=asset_symbol,
            recommended_size=recommended_size,
            max_size=max_size,
            min_size=min_size,
            risk_budget_used=0.0,  # Will be calculated later
            sizing_method="kelly_criterion",
            confidence=confidence,
            rationale=rationale,
            risk_metrics=risk_metrics
        )
    
    def _risk_parity_sizing(self,
                           asset_symbol: str,
                           prediction: Dict[str, Any],
                           portfolio_value: float,
                           current_position: float) -> PositionSizeResult:
        """Risk parity position sizing."""
        # Get asset volatility
        asset_volatility = self._get_asset_volatility(asset_symbol)
        
        if asset_volatility == 0:
            asset_volatility = 0.02
        
        # Get portfolio volatility (average of all assets)
        all_volatilities = []
        for asset in self.volatility_history:
            if self.volatility_history[asset]:
                vol = list(self.volatility_history[asset])[-1]
                all_volatilities.append(vol)
        
        if all_volatilities:
            portfolio_volatility = np.mean(all_volatilities)
        else:
            portfolio_volatility = 0.02
        
        # Risk parity weight: inversely proportional to volatility
        risk_parity_weight = portfolio_volatility / asset_volatility
        
        # Normalize by number of assets
        n_assets = max(1, len(self.asset_symbols))
        target_weight = risk_parity_weight / n_assets
        
        # Apply prediction confidence
        confidence = prediction.get('confidence', 0.5)
        adjusted_weight = target_weight * confidence
        
        # Apply regime adjustment
        regime_adjustment = self.regime_adjustments.get(self.current_regime, 1.0)
        recommended_size = adjusted_weight * regime_adjustment
        
        rationale = [
            f"Risk parity targeting equal risk contribution",
            f"Asset volatility: {asset_volatility:.4f}",
            f"Portfolio volatility: {portfolio_volatility:.4f}",
            f"Regime adjustment: {regime_adjustment:.2f}"
        ]
        
        risk_metrics = {
            'asset_volatility': asset_volatility,
            'portfolio_volatility': portfolio_volatility,
            'risk_parity_weight': risk_parity_weight,
            'regime_adjustment': regime_adjustment
        }
        
        return PositionSizeResult(
            timestamp=datetime.now(),
            asset_symbol=asset_symbol,
            recommended_size=max(0.0, recommended_size),
            max_size=self.constraints.max_position_size,
            min_size=self.constraints.min_position_size if recommended_size > 0 else 0.0,
            risk_budget_used=0.0,
            sizing_method="risk_parity",
            confidence=confidence,
            rationale=rationale,
            risk_metrics=risk_metrics
        )
    
    def _volatility_based_sizing(self,
                                asset_symbol: str,
                                prediction: Dict[str, Any],
                                portfolio_value: float,
                                current_position: float) -> PositionSizeResult:
        """Volatility-based position sizing."""
        # Target risk per position
        target_risk = self.default_risk_budget
        
        # Get asset volatility
        asset_volatility = self._get_asset_volatility(asset_symbol)
        if asset_volatility == 0:
            asset_volatility = 0.02
        
        # Calculate position size for target risk
        # Position size = Target Risk / Asset Volatility
        base_size = target_risk / asset_volatility
        
        # Apply prediction confidence
        confidence = prediction.get('confidence', 0.5)
        confidence_adjusted_size = base_size * confidence
        
        # Apply expected return adjustment
        expected_return = prediction.get('expected_return', 0.0)
        if expected_return > 0:
            return_adjustment = min(2.0, 1.0 + expected_return * 10)  # Cap at 2x
        else:
            return_adjustment = max(0.0, 1.0 + expected_return * 10)  # Reduce for negative returns
        
        recommended_size = confidence_adjusted_size * return_adjustment
        
        rationale = [
            f"Volatility targeting {target_risk:.2%} risk",
            f"Asset volatility: {asset_volatility:.4f}",
            f"Confidence adjustment: {confidence:.2f}",
            f"Return adjustment: {return_adjustment:.2f}"
        ]
        
        risk_metrics = {
            'target_risk': target_risk,
            'asset_volatility': asset_volatility,
            'confidence_adjustment': confidence,
            'return_adjustment': return_adjustment
        }
        
        return PositionSizeResult(
            timestamp=datetime.now(),
            asset_symbol=asset_symbol,
            recommended_size=max(0.0, recommended_size),
            max_size=self.constraints.max_position_size,
            min_size=self.constraints.min_position_size if recommended_size > 0 else 0.0,
            risk_budget_used=0.0,
            sizing_method="volatility_based",
            confidence=confidence,
            rationale=rationale,
            risk_metrics=risk_metrics
        )
    
    def _fixed_risk_sizing(self,
                          asset_symbol: str,
                          prediction: Dict[str, Any],
                          portfolio_value: float,
                          current_position: float) -> PositionSizeResult:
        """Fixed risk position sizing."""
        # Use default risk budget
        risk_budget = self.default_risk_budget
        
        # Apply confidence scaling
        confidence = prediction.get('confidence', 0.5)
        adjusted_risk_budget = risk_budget * confidence
        
        # Apply regime adjustment
        regime_adjustment = self.regime_adjustments.get(self.current_regime, 1.0)
        final_size = adjusted_risk_budget * regime_adjustment
        
        rationale = [
            f"Fixed risk budget: {risk_budget:.2%}",
            f"Confidence scaling: {confidence:.2f}",
            f"Regime adjustment: {regime_adjustment:.2f}"
        ]
        
        risk_metrics = {
            'base_risk_budget': risk_budget,
            'confidence_scaling': confidence,
            'regime_adjustment': regime_adjustment
        }
        
        return PositionSizeResult(
            timestamp=datetime.now(),
            asset_symbol=asset_symbol,
            recommended_size=max(0.0, final_size),
            max_size=self.constraints.max_position_size,
            min_size=self.constraints.min_position_size if final_size > 0 else 0.0,
            risk_budget_used=0.0,
            sizing_method="fixed_risk",
            confidence=confidence,
            rationale=rationale,
            risk_metrics=risk_metrics
        )
    
    def _momentum_based_sizing(self,
                              asset_symbol: str,
                              prediction: Dict[str, Any],
                              portfolio_value: float,
                              current_position: float) -> PositionSizeResult:
        """Momentum-based position sizing."""
        # Calculate momentum score
        momentum_score = self._calculate_momentum_score(asset_symbol)
        
        # Base size from volatility targeting
        volatility_result = self._volatility_based_sizing(
            asset_symbol, prediction, portfolio_value, current_position
        )
        base_size = volatility_result.recommended_size
        
        # Apply momentum adjustment
        momentum_adjustment = 0.5 + momentum_score  # Range: [0, 1.5]
        momentum_adjusted_size = base_size * momentum_adjustment
        
        rationale = [
            f"Momentum-based sizing",
            f"Momentum score: {momentum_score:.4f}",
            f"Momentum adjustment: {momentum_adjustment:.2f}",
            f"Base size: {base_size:.4f}"
        ]
        
        risk_metrics = {
            'momentum_score': momentum_score,
            'momentum_adjustment': momentum_adjustment,
            'base_size': base_size
        }
        
        return PositionSizeResult(
            timestamp=datetime.now(),
            asset_symbol=asset_symbol,
            recommended_size=max(0.0, momentum_adjusted_size),
            max_size=self.constraints.max_position_size,
            min_size=self.constraints.min_position_size if momentum_adjusted_size > 0 else 0.0,
            risk_budget_used=0.0,
            sizing_method="momentum_based",
            confidence=prediction.get('confidence', 0.5),
            rationale=rationale,
            risk_metrics=risk_metrics
        )
    
    def _mean_reversion_sizing(self,
                              asset_symbol: str,
                              prediction: Dict[str, Any],
                              portfolio_value: float,
                              current_position: float) -> PositionSizeResult:
        """Mean reversion position sizing."""
        # Calculate mean reversion score
        reversion_score = self._calculate_mean_reversion_score(asset_symbol)
        
        # Base size from volatility targeting
        volatility_result = self._volatility_based_sizing(
            asset_symbol, prediction, portfolio_value, current_position
        )
        base_size = volatility_result.recommended_size
        
        # Apply mean reversion adjustment (inverse of momentum)
        reversion_adjustment = 1.5 - reversion_score  # Range: [0.5, 1.5]
        reversion_adjusted_size = base_size * reversion_adjustment
        
        rationale = [
            f"Mean reversion-based sizing",
            f"Reversion score: {reversion_score:.4f}",
            f"Reversion adjustment: {reversion_adjustment:.2f}",
            f"Base size: {base_size:.4f}"
        ]
        
        risk_metrics = {
            'reversion_score': reversion_score,
            'reversion_adjustment': reversion_adjustment,
            'base_size': base_size
        }
        
        return PositionSizeResult(
            timestamp=datetime.now(),
            asset_symbol=asset_symbol,
            recommended_size=max(0.0, reversion_adjusted_size),
            max_size=self.constraints.max_position_size,
            min_size=self.constraints.min_position_size if reversion_adjusted_size > 0 else 0.0,
            risk_budget_used=0.0,
            sizing_method="mean_reversion",
            confidence=prediction.get('confidence', 0.5),
            rationale=rationale,
            risk_metrics=risk_metrics
        )
    
    def _multi_objective_sizing(self,
                               asset_symbol: str,
                               prediction: Dict[str, Any],
                               portfolio_value: float,
                               current_position: float) -> PositionSizeResult:
        """Multi-objective position sizing combining multiple methods."""
        # Get sizes from different methods
        kelly_result = self._kelly_criterion_sizing(asset_symbol, prediction, portfolio_value, current_position)
        risk_parity_result = self._risk_parity_sizing(asset_symbol, prediction, portfolio_value, current_position)
        volatility_result = self._volatility_based_sizing(asset_symbol, prediction, portfolio_value, current_position)
        
        # Weight the methods
        method_weights = {
            'kelly': 0.4,
            'risk_parity': 0.3,
            'volatility': 0.3
        }
        
        # Combine sizes
        combined_size = (
            method_weights['kelly'] * kelly_result.recommended_size +
            method_weights['risk_parity'] * risk_parity_result.recommended_size +
            method_weights['volatility'] * volatility_result.recommended_size
        )
        
        # Combine confidence
        combined_confidence = np.mean([
            kelly_result.confidence,
            risk_parity_result.confidence,
            volatility_result.confidence
        ])
        
        rationale = [
            f"Multi-objective sizing combination",
            f"Kelly component: {kelly_result.recommended_size:.4f} (weight: {method_weights['kelly']})",
            f"Risk parity component: {risk_parity_result.recommended_size:.4f} (weight: {method_weights['risk_parity']})",
            f"Volatility component: {volatility_result.recommended_size:.4f} (weight: {method_weights['volatility']})"
        ]
        
        risk_metrics = {
            'kelly_size': kelly_result.recommended_size,
            'risk_parity_size': risk_parity_result.recommended_size,
            'volatility_size': volatility_result.recommended_size,
            'method_weights': method_weights
        }
        
        return PositionSizeResult(
            timestamp=datetime.now(),
            asset_symbol=asset_symbol,
            recommended_size=max(0.0, combined_size),
            max_size=self.constraints.max_position_size,
            min_size=self.constraints.min_position_size if combined_size > 0 else 0.0,
            risk_budget_used=0.0,
            sizing_method="multi_objective",
            confidence=combined_confidence,
            rationale=rationale,
            risk_metrics=risk_metrics
        )
    
    def _get_asset_volatility(self, asset_symbol: str) -> float:
        """Get current volatility estimate for asset."""
        if (asset_symbol in self.volatility_history and 
            self.volatility_history[asset_symbol]):
            return list(self.volatility_history[asset_symbol])[-1]
        
        # Fallback: calculate from returns
        if (asset_symbol in self.return_history and 
            len(self.return_history[asset_symbol]) > 5):
            returns = list(self.return_history[asset_symbol])[-20:]
            return np.std(returns) * np.sqrt(252)
        
        return 0.02  # Default volatility
    
    def _calculate_momentum_score(self, asset_symbol: str) -> float:
        """Calculate momentum score for asset."""
        if (asset_symbol not in self.price_history or 
            len(self.price_history[asset_symbol]) < 20):
            return 0.0
        
        prices = [p['price'] for p in list(self.price_history[asset_symbol])[-20:]]
        
        # Simple momentum: recent performance vs average
        short_term = np.mean(prices[-5:])
        long_term = np.mean(prices[-20:])
        
        if long_term != 0:
            momentum_score = (short_term - long_term) / long_term
            return np.clip(momentum_score, -0.5, 0.5)  # Normalize to [-0.5, 0.5]
        
        return 0.0
    
    def _calculate_mean_reversion_score(self, asset_symbol: str) -> float:
        """Calculate mean reversion score for asset."""
        momentum_score = self._calculate_momentum_score(asset_symbol)
        
        # Mean reversion is inverse of momentum
        return -momentum_score + 0.5  # Convert to [0, 1] range
    
    def _apply_constraints(self,
                          asset_symbol: str,
                          recommended_size: float,
                          portfolio_value: float) -> float:
        """Apply position sizing constraints."""
        constrained_size = recommended_size
        
        # Apply maximum position size constraint
        constrained_size = min(constrained_size, self.constraints.max_position_size)
        
        # Apply minimum position size constraint
        if constrained_size > 0:
            constrained_size = max(constrained_size, self.constraints.min_position_size)
        
        # Apply correlation constraint
        if self.correlation_matrix is not None and asset_symbol in self.asset_symbols:
            corr_adjustment = self._calculate_correlation_adjustment(asset_symbol)
            constrained_size *= corr_adjustment
        
        # Apply liquidity constraint (simplified)
        liquidity_adjustment = self._calculate_liquidity_adjustment(asset_symbol)
        constrained_size *= liquidity_adjustment
        
        # Apply risk budget constraint
        risk_budget_adjustment = self._calculate_risk_budget_adjustment(
            asset_symbol, constrained_size, portfolio_value
        )
        constrained_size *= risk_budget_adjustment
        
        return max(0.0, constrained_size)
    
    def _calculate_correlation_adjustment(self, asset_symbol: str) -> float:
        """Calculate adjustment based on correlation with existing positions."""
        if (self.correlation_matrix is None or 
            asset_symbol not in self.asset_symbols):
            return 1.0
        
        asset_idx = self.asset_symbols.index(asset_symbol)
        
        # Get correlations with other assets
        correlations = self.correlation_matrix[asset_idx, :]
        
        # Calculate average absolute correlation
        avg_correlation = np.mean(np.abs(correlations[correlations != 1.0]))
        
        # Reduce size for highly correlated assets
        if avg_correlation > 0.8:
            return 0.5
        elif avg_correlation > 0.6:
            return 0.7
        elif avg_correlation > 0.4:
            return 0.9
        else:
            return 1.0
    
    def _calculate_liquidity_adjustment(self, asset_symbol: str) -> float:
        """Calculate adjustment based on liquidity constraints."""
        # Simplified liquidity adjustment
        # In practice, would use actual order book data and trading volumes
        
        # Assume liquidity based on asset type
        major_assets = ['bitcoin', 'ethereum', 'btc', 'eth']
        
        if any(major in asset_symbol.lower() for major in major_assets):
            return 1.0  # No adjustment for major assets
        else:
            return 0.8  # 20% reduction for smaller assets
    
    def _calculate_risk_budget_adjustment(self,
                                        asset_symbol: str,
                                        position_size: float,
                                        portfolio_value: float) -> float:
        """Calculate adjustment based on risk budget constraints."""
        # Calculate risk usage for this position
        asset_volatility = self._get_asset_volatility(asset_symbol)
        position_risk = position_size * asset_volatility
        
        # Check against available risk budget
        available_budget = self.risk_budget.total_risk_budget - self.risk_budget.used_budget
        
        if position_risk <= available_budget:
            return 1.0  # No adjustment needed
        elif available_budget > 0:
            # Scale down to fit available budget
            return available_budget / position_risk
        else:
            return 0.0  # No budget available
    
    def _calculate_risk_budget_usage(self,
                                   asset_symbol: str,
                                   position_size: float,
                                   portfolio_value: float) -> float:
        """Calculate risk budget usage for position."""
        asset_volatility = self._get_asset_volatility(asset_symbol)
        return position_size * asset_volatility
    
    def update_risk_budget(self, new_budget: RiskBudget) -> None:
        """Update risk budget allocation."""
        self.risk_budget = new_budget
    
    def update_constraints(self, new_constraints: PositionConstraints) -> None:
        """Update position sizing constraints."""
        self.constraints = new_constraints
    
    def update_market_regime(self, regime: str) -> None:
        """Update current market regime."""
        if regime in self.regime_adjustments:
            self.current_regime = regime
    
    def get_portfolio_risk_allocation(self) -> Dict[str, Any]:
        """Get current portfolio risk allocation."""
        allocation = {
            'total_risk_budget': self.risk_budget.total_risk_budget,
            'used_budget': self.risk_budget.used_budget,
            'available_budget': self.risk_budget.total_risk_budget - self.risk_budget.used_budget,
            'asset_allocations': self.risk_budget.asset_allocations.copy(),
            'utilization_rate': self.risk_budget.used_budget / self.risk_budget.total_risk_budget
        }
        
        return allocation
    
    def get_sizing_performance_summary(self) -> Dict[str, Any]:
        """Get summary of sizing performance."""
        if not self.position_history:
            return {'status': 'no_data'}
        
        recent_positions = list(self.position_history)[-50:]
        
        # Analyze sizing methods used
        method_counts = defaultdict(int)
        method_performance = defaultdict(list)
        
        for position in recent_positions:
            method_counts[position.sizing_method] += 1
            # In practice, would calculate actual performance vs recommended size
            # For now, use confidence as proxy
            method_performance[position.sizing_method].append(position.confidence)
        
        summary = {
            'total_positions': len(recent_positions),
            'method_usage': dict(method_counts),
            'avg_confidence_by_method': {
                method: np.mean(scores) for method, scores in method_performance.items()
            },
            'avg_position_size': np.mean([p.recommended_size for p in recent_positions]),
            'risk_budget_utilization': np.mean([p.risk_budget_used for p in recent_positions])
        }
        
        return summary