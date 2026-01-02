import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.risk_management.kelly_criterion.kelly_optimizer import KellyOptimizer
from models.risk_management.value_at_risk.var_calculator import VaRCalculator


class RiskManager:
    """
    Comprehensive risk management system for cryptocurrency trading.
    
    Combines multiple risk management techniques:
    - Kelly Criterion for optimal position sizing
    - Value at Risk (VaR) for downside risk measurement
    - Maximum drawdown controls
    - Volatility-based position adjustments
    - Correlation-based portfolio risk
    - Dynamic risk limits
    
    Designed specifically for crypto markets with high volatility and fat tails.
    """
    
    def __init__(self, 
                 max_position_size: float = 0.2,
                 max_portfolio_risk: float = 0.05,
                 max_drawdown: float = 0.15,
                 var_confidence: float = 0.95,
                 kelly_fraction: float = 0.25):
        """
        Initialize risk manager.
        
        Args:
            max_position_size: Maximum size for any single position
            max_portfolio_risk: Maximum portfolio VaR as fraction of capital
            max_drawdown: Maximum acceptable drawdown
            var_confidence: Confidence level for VaR calculations
            kelly_fraction: Fraction of full Kelly to use
        """
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.max_drawdown = max_drawdown
        
        # Initialize components
        self.kelly_optimizer = KellyOptimizer(
            kelly_fraction=kelly_fraction,
            max_position=max_position_size
        )
        self.var_calculator = VaRCalculator(
            confidence_level=var_confidence
        )
        
        # Risk tracking
        self.current_positions = {}
        self.risk_metrics = {}
        self.position_history = []
        self.pnl_history = []
        
    def calculate_position_size(self, 
                              signal: Dict,
                              market_data: Dict,
                              portfolio_value: float) -> Dict:
        """
        Calculate optimal position size considering all risk factors.
        
        Args:
            signal: Trading signal with confidence and expected metrics
            market_data: Current market data (price, volume, volatility)
            portfolio_value: Total portfolio value
            
        Returns:
            Dictionary with position sizing details
        """
        # Extract signal information
        confidence = signal.get('confidence', 0.5)
        expected_return = signal.get('expected_return', 0.001)
        win_rate = signal.get('win_rate', 0.5)
        win_loss_ratio = signal.get('win_loss_ratio', 1.5)
        
        # Get market conditions
        current_volatility = market_data.get('volatility', 0.02)
        returns_history = market_data.get('returns_history', np.array([]))
        
        # Step 1: Kelly-based position size
        kelly_result = self.kelly_optimizer.calculate_position_size(
            capital=portfolio_value,
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
            volatility=current_volatility,
            confidence=confidence
        )
        
        base_position_size = kelly_result['adjusted_kelly']
        
        # Step 2: VaR constraint
        if len(returns_history) > 20:
            position_var = self._calculate_position_var(
                base_position_size,
                returns_history,
                portfolio_value
            )
            
            # Reduce position if VaR exceeds limit
            if position_var > self.max_portfolio_risk:
                var_adjustment = self.max_portfolio_risk / position_var
                base_position_size *= var_adjustment
            else:
                var_adjustment = 1.0
        else:
            position_var = base_position_size * current_volatility * 2.33  # Approximation
            var_adjustment = 1.0
            
        # Step 3: Correlation adjustment
        correlation_adjustment = self._calculate_correlation_adjustment(
            signal.get('asset', 'BTC'),
            base_position_size
        )
        
        # Step 4: Drawdown protection
        drawdown_adjustment = self._calculate_drawdown_adjustment()
        
        # Final position size
        final_position_size = (base_position_size * 
                             var_adjustment * 
                             correlation_adjustment * 
                             drawdown_adjustment)
        
        # Ensure within limits
        final_position_size = min(final_position_size, self.max_position_size)
        final_position_size = max(0, final_position_size)  # No negative positions for now
        
        # Calculate position details
        position_value = portfolio_value * final_position_size
        shares = position_value / market_data.get('price', 1)
        
        return {
            'position_size_pct': final_position_size,
            'position_value': position_value,
            'shares': shares,
            'kelly_size': kelly_result['adjusted_kelly'],
            'var_adjustment': var_adjustment,
            'correlation_adjustment': correlation_adjustment,
            'drawdown_adjustment': drawdown_adjustment,
            'position_var': position_var,
            'risk_budget_used': position_var / self.max_portfolio_risk,
            'confidence': confidence,
            'expected_sharpe': expected_return / current_volatility if current_volatility > 0 else 0
        }
    
    def _calculate_position_var(self, position_size: float,
                               returns: np.ndarray,
                               portfolio_value: float) -> float:
        """Calculate VaR for a position."""
        # Scale returns by position size
        position_returns = returns * position_size
        
        # Calculate VaR
        var_result = self.var_calculator.calculate_var(
            position_returns,
            method='historical'
        )
        
        return var_result['var_percent']
    
    def _calculate_correlation_adjustment(self, asset: str,
                                        position_size: float) -> float:
        """
        Adjust position size based on correlation with existing positions.
        
        Args:
            asset: Asset identifier
            position_size: Proposed position size
            
        Returns:
            Correlation adjustment factor
        """
        if not self.current_positions:
            return 1.0
            
        # Calculate correlation impact (simplified)
        # In practice, would use actual correlation matrix
        num_positions = len(self.current_positions)
        concentration_penalty = 1 - (num_positions * 0.05)  # 5% reduction per position
        
        return max(0.5, concentration_penalty)
    
    def _calculate_drawdown_adjustment(self) -> float:
        """
        Adjust position size based on current drawdown.
        
        Returns:
            Drawdown adjustment factor
        """
        if not self.pnl_history:
            return 1.0
            
        # Calculate current drawdown
        cumulative_pnl = np.cumsum(self.pnl_history)
        running_max = np.maximum.accumulate(cumulative_pnl)
        current_dd = (cumulative_pnl[-1] - running_max[-1]) / (running_max[-1] + 1e-8)
        
        # Reduce size as we approach max drawdown
        if abs(current_dd) > self.max_drawdown * 0.5:
            # Start reducing at 50% of max drawdown
            reduction = 1 - (abs(current_dd) - self.max_drawdown * 0.5) / (self.max_drawdown * 0.5)
            return max(0.2, reduction)  # Minimum 20% of normal size
            
        return 1.0
    
    def update_position(self, asset: str, position: Dict, pnl: float):
        """
        Update position tracking and risk metrics.
        
        Args:
            asset: Asset identifier
            position: Position details
            pnl: Realized PnL
        """
        # Update current positions
        if position['position_size_pct'] > 0:
            self.current_positions[asset] = position
        else:
            self.current_positions.pop(asset, None)
            
        # Track PnL
        self.pnl_history.append(pnl)
        
        # Update risk metrics
        self._update_risk_metrics()
        
    def _update_risk_metrics(self):
        """Update portfolio-wide risk metrics."""
        if not self.current_positions:
            self.risk_metrics = {
                'total_exposure': 0,
                'portfolio_var': 0,
                'max_position': 0,
                'num_positions': 0
            }
            return
            
        # Calculate total exposure
        total_exposure = sum(p['position_size_pct'] for p in self.current_positions.values())
        
        # Calculate portfolio VaR (simplified - assumes independence)
        portfolio_var = np.sqrt(sum(p['position_var']**2 for p in self.current_positions.values()))
        
        # Find largest position
        max_position = max(p['position_size_pct'] for p in self.current_positions.values())
        
        self.risk_metrics = {
            'total_exposure': total_exposure,
            'portfolio_var': portfolio_var,
            'max_position': max_position,
            'num_positions': len(self.current_positions),
            'current_drawdown': self._calculate_current_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if not self.pnl_history:
            return 0.0
            
        cumulative_pnl = np.cumsum(self.pnl_history)
        running_max = np.maximum.accumulate(cumulative_pnl)
        
        if running_max[-1] > 0:
            return (cumulative_pnl[-1] - running_max[-1]) / running_max[-1]
        return 0.0
    
    def _calculate_sharpe_ratio(self, lookback: int = 100) -> float:
        """Calculate rolling Sharpe ratio."""
        if len(self.pnl_history) < lookback:
            return 0.0
            
        recent_pnl = self.pnl_history[-lookback:]
        
        if np.std(recent_pnl) > 0:
            return np.mean(recent_pnl) / np.std(recent_pnl) * np.sqrt(252)
        return 0.0
    
    def check_risk_limits(self) -> Dict[str, bool]:
        """
        Check if any risk limits are breached.
        
        Returns:
            Dictionary of limit checks
        """
        checks = {
            'position_size_ok': all(p['position_size_pct'] <= self.max_position_size 
                                  for p in self.current_positions.values()),
            'portfolio_var_ok': self.risk_metrics.get('portfolio_var', 0) <= self.max_portfolio_risk,
            'drawdown_ok': abs(self.risk_metrics.get('current_drawdown', 0)) <= self.max_drawdown,
            'exposure_ok': self.risk_metrics.get('total_exposure', 0) <= 1.0
        }
        
        checks['all_limits_ok'] = all(checks.values())
        
        return checks
    
    def get_risk_report(self) -> Dict:
        """
        Generate comprehensive risk report.
        
        Returns:
            Dictionary with risk analysis
        """
        report = {
            'current_positions': self.current_positions,
            'risk_metrics': self.risk_metrics,
            'limit_checks': self.check_risk_limits(),
            'risk_capacity': {
                'var_capacity_used': self.risk_metrics.get('portfolio_var', 0) / self.max_portfolio_risk,
                'drawdown_capacity_used': abs(self.risk_metrics.get('current_drawdown', 0)) / self.max_drawdown,
                'position_capacity': self.risk_metrics.get('max_position', 0) / self.max_position_size
            }
        }
        
        # Add recommendations
        recommendations = []
        
        if report['risk_capacity']['var_capacity_used'] > 0.8:
            recommendations.append("High VaR usage - consider reducing positions")
            
        if report['risk_capacity']['drawdown_capacity_used'] > 0.7:
            recommendations.append("Approaching drawdown limit - reduce risk")
            
        if self.risk_metrics.get('sharpe_ratio', 0) < 0:
            recommendations.append("Negative Sharpe ratio - review strategy")
            
        report['recommendations'] = recommendations
        
        return report
    
    def calculate_stop_loss(self, entry_price: float,
                          position_size: float,
                          volatility: float,
                          method: str = 'atr') -> Dict:
        """
        Calculate stop loss levels.
        
        Args:
            entry_price: Entry price for position
            position_size: Position size as fraction
            volatility: Current volatility
            method: Stop loss method ('atr', 'percentage', 'var')
            
        Returns:
            Dictionary with stop loss details
        """
        if method == 'atr':
            # ATR-based stop (2-3x ATR)
            atr_multiplier = 2.5
            stop_distance = volatility * atr_multiplier
            stop_price = entry_price * (1 - stop_distance)
            
        elif method == 'percentage':
            # Fixed percentage stop
            stop_percentage = min(0.05, self.max_position_size * 0.25)  # Max 5% or 25% of position
            stop_price = entry_price * (1 - stop_percentage)
            stop_distance = stop_percentage
            
        elif method == 'var':
            # VaR-based stop
            var_stop = self.max_portfolio_risk / position_size
            stop_price = entry_price * (1 - var_stop)
            stop_distance = var_stop
            
        else:
            raise ValueError(f"Unknown stop loss method: {method}")
            
        # Calculate risk amount
        risk_amount = position_size * stop_distance
        
        return {
            'stop_price': stop_price,
            'stop_distance': stop_distance,
            'stop_distance_pct': stop_distance * 100,
            'risk_amount': risk_amount,
            'risk_reward_ratio': 1 / stop_distance if stop_distance > 0 else np.inf,
            'method': method
        }
    
    def calculate_take_profit(self, entry_price: float,
                            stop_loss: float,
                            risk_reward_ratio: float = 2.0) -> Dict:
        """
        Calculate take profit levels based on risk-reward ratio.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_reward_ratio: Desired risk-reward ratio
            
        Returns:
            Dictionary with take profit details
        """
        # Calculate risk distance
        risk_distance = abs(entry_price - stop_loss)
        
        # Calculate reward distance
        reward_distance = risk_distance * risk_reward_ratio
        
        # Take profit price
        take_profit_price = entry_price + reward_distance
        
        return {
            'take_profit_price': take_profit_price,
            'reward_distance': reward_distance,
            'reward_distance_pct': (reward_distance / entry_price) * 100,
            'risk_reward_ratio': risk_reward_ratio,
            'expected_value': (risk_reward_ratio - 1) / (risk_reward_ratio + 1)  # Assuming 50% win rate
        }
    
    def stress_test_portfolio(self, scenarios: Optional[Dict] = None) -> pd.DataFrame:
        """
        Stress test portfolio under various scenarios.
        
        Args:
            scenarios: Dictionary of stress scenarios
            
        Returns:
            DataFrame with stress test results
        """
        if scenarios is None:
            # Default crypto stress scenarios
            scenarios = {
                'flash_crash': {'market_move': -0.30, 'volatility_mult': 3.0},
                'black_swan': {'market_move': -0.50, 'volatility_mult': 5.0},
                'liquidity_crisis': {'market_move': -0.20, 'volatility_mult': 2.0, 'correlation': 0.9},
                'bull_run': {'market_move': 0.50, 'volatility_mult': 2.0},
                'normal_correction': {'market_move': -0.10, 'volatility_mult': 1.5}
            }
            
        results = []
        
        for scenario_name, scenario in scenarios.items():
            # Calculate portfolio impact
            portfolio_return = 0
            
            for asset, position in self.current_positions.items():
                position_return = position['position_size_pct'] * scenario['market_move']
                portfolio_return += position_return
                
            # Adjust for correlation in crisis
            if 'correlation' in scenario:
                # Higher correlation means less diversification benefit
                portfolio_return *= scenario['correlation']
                
            results.append({
                'scenario': scenario_name,
                'portfolio_return': portfolio_return,
                'portfolio_loss': -portfolio_return if portfolio_return < 0 else 0,
                'volatility_multiplier': scenario.get('volatility_mult', 1),
                'breaches_var': abs(portfolio_return) > self.risk_metrics.get('portfolio_var', 0),
                'breaches_max_loss': abs(portfolio_return) > self.max_drawdown
            })
            
        return pd.DataFrame(results)