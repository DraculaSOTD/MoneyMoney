import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy.optimize import minimize_scalar
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from utils.matrix_operations import MatrixOperations


class KellyOptimizer:
    """
    Kelly Criterion implementation for optimal position sizing in cryptocurrency trading.
    
    The Kelly Criterion maximizes the expected logarithmic growth of capital:
    f* = (p*b - q) / b
    
    where:
    - f* = optimal fraction of capital to bet
    - p = probability of winning
    - q = probability of losing (1-p)
    - b = odds (win/loss ratio)
    
    Enhanced for crypto trading with:
    - Fractional Kelly for risk reduction
    - Dynamic win rate estimation
    - Volatility adjustments
    - Maximum position limits
    - Multi-asset optimization
    """
    
    def __init__(self, kelly_fraction: float = 0.25, 
                 lookback_window: int = 100,
                 max_position: float = 0.2):
        """
        Initialize Kelly optimizer.
        
        Args:
            kelly_fraction: Fraction of full Kelly to use (0.25 = 25% Kelly)
            lookback_window: Window for calculating win rates
            max_position: Maximum position size allowed
        """
        self.kelly_fraction = kelly_fraction
        self.lookback_window = lookback_window
        self.max_position = max_position
        
        # Historical data storage
        self.trade_history = []
        self.position_history = []
        self.performance_metrics = {}
        
    def calculate_kelly_fraction(self, win_rate: float, 
                               win_loss_ratio: float,
                               confidence: float = 1.0) -> float:
        """
        Calculate basic Kelly fraction.
        
        Args:
            win_rate: Probability of winning (0-1)
            win_loss_ratio: Average win / average loss
            confidence: Confidence in estimates (0-1)
            
        Returns:
            Optimal fraction of capital to risk
        """
        if win_loss_ratio <= 0:
            return 0.0
            
        # Basic Kelly formula
        p = win_rate
        q = 1 - win_rate
        b = win_loss_ratio
        
        kelly = (p * b - q) / b
        
        # Adjust for confidence
        kelly *= confidence
        
        # Apply fractional Kelly
        kelly *= self.kelly_fraction
        
        # Apply maximum position limit
        kelly = min(kelly, self.max_position)
        
        # Never risk more than makes sense
        kelly = max(0, kelly)
        
        return kelly
    
    def estimate_win_metrics(self, returns: np.ndarray,
                           threshold: float = 0.0) -> Dict:
        """
        Estimate win rate and win/loss ratio from historical returns.
        
        Args:
            returns: Historical returns
            threshold: Threshold for considering a win
            
        Returns:
            Dictionary with win metrics
        """
        if len(returns) == 0:
            return {
                'win_rate': 0.5,
                'win_loss_ratio': 1.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'sample_size': 0
            }
            
        # Classify wins and losses
        wins = returns[returns > threshold]
        losses = returns[returns <= threshold]
        
        # Calculate metrics
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.5
        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 1.0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        
        return {
            'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sample_size': len(returns),
            'num_wins': len(wins),
            'num_losses': len(losses)
        }
    
    def calculate_position_size(self, capital: float,
                              win_rate: float,
                              win_loss_ratio: float,
                              volatility: float,
                              confidence: float = 1.0) -> Dict:
        """
        Calculate optimal position size with volatility adjustment.
        
        Args:
            capital: Total capital available
            win_rate: Estimated win rate
            win_loss_ratio: Win/loss ratio
            volatility: Current market volatility
            confidence: Confidence in predictions
            
        Returns:
            Dictionary with position sizing details
        """
        # Base Kelly fraction
        base_kelly = self.calculate_kelly_fraction(win_rate, win_loss_ratio, confidence)
        
        # Volatility adjustment
        # Reduce position size in high volatility
        target_volatility = 0.02  # 2% daily volatility target
        vol_scalar = min(1.0, target_volatility / volatility) if volatility > 0 else 1.0
        
        # Adjusted Kelly fraction
        adjusted_kelly = base_kelly * vol_scalar
        
        # Position size in currency units
        position_size = capital * adjusted_kelly
        
        # Risk metrics
        expected_growth = self._calculate_expected_growth(
            adjusted_kelly, win_rate, win_loss_ratio
        )
        
        return {
            'kelly_fraction': base_kelly,
            'adjusted_kelly': adjusted_kelly,
            'position_size': position_size,
            'volatility_scalar': vol_scalar,
            'expected_growth': expected_growth,
            'risk_amount': position_size,
            'risk_percentage': adjusted_kelly * 100
        }
    
    def _calculate_expected_growth(self, f: float, p: float, b: float) -> float:
        """
        Calculate expected logarithmic growth rate.
        
        Args:
            f: Fraction of capital to bet
            p: Win probability
            b: Win/loss ratio
            
        Returns:
            Expected growth rate
        """
        if f <= 0 or f >= 1:
            return 0.0
            
        # E[log(1 + f*X)] where X is the outcome
        expected_growth = p * np.log(1 + f * b) + (1 - p) * np.log(1 - f)
        
        return expected_growth
    
    def optimize_kelly_fraction(self, win_rate: float,
                              win_loss_ratio: float,
                              max_drawdown_limit: float = 0.2) -> float:
        """
        Optimize Kelly fraction considering maximum drawdown constraints.
        
        Args:
            win_rate: Win probability
            win_loss_ratio: Win/loss ratio
            max_drawdown_limit: Maximum acceptable drawdown
            
        Returns:
            Optimal Kelly fraction
        """
        # Calculate probability of hitting drawdown limit
        def drawdown_probability(f):
            if f <= 0 or f >= 1:
                return 1.0
                
            # Simplified calculation
            # In reality, would use more sophisticated methods
            consecutive_losses_needed = int(np.log(1 - max_drawdown_limit) / np.log(1 - f))
            prob_consecutive_losses = (1 - win_rate) ** consecutive_losses_needed
            
            return prob_consecutive_losses
        
        # Objective: maximize growth subject to drawdown constraint
        def objective(f):
            growth = self._calculate_expected_growth(f, win_rate, win_loss_ratio)
            dd_prob = drawdown_probability(f)
            
            # Penalize high drawdown probability
            if dd_prob > 0.05:  # 5% chance of hitting limit
                growth *= (1 - dd_prob)
                
            return -growth  # Negative for minimization
        
        # Optimize
        result = minimize_scalar(
            objective,
            bounds=(0, min(1.0, self.max_position)),
            method='bounded'
        )
        
        return result.x
    
    def calculate_multi_asset_kelly(self, 
                                  returns: pd.DataFrame,
                                  covariance: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Calculate Kelly weights for multiple assets.
        
        Args:
            returns: DataFrame of asset returns
            covariance: Covariance matrix (calculated if not provided)
            
        Returns:
            Series of optimal weights
        """
        assets = returns.columns
        n_assets = len(assets)
        
        # Calculate expected returns and covariance
        expected_returns = returns.mean()
        
        if covariance is None:
            covariance = returns.cov()
            
        # Kelly formula for multiple assets: f = C^(-1) * μ
        # where C is covariance matrix and μ is expected returns
        
        try:
            # Use robust matrix inversion
            cov_inv = MatrixOperations.stable_inverse(covariance.values)
            kelly_weights = cov_inv @ expected_returns.values
            
            # Scale to fractional Kelly
            kelly_weights *= self.kelly_fraction
            
            # Apply constraints
            kelly_weights = np.maximum(kelly_weights, 0)  # No short selling
            kelly_weights = np.minimum(kelly_weights, self.max_position)
            
            # Normalize if total exceeds 1
            if kelly_weights.sum() > 1:
                kelly_weights /= kelly_weights.sum()
                
        except:
            # Fallback to equal weights
            kelly_weights = np.ones(n_assets) / n_assets
            
        return pd.Series(kelly_weights, index=assets)
    
    def dynamic_kelly_adjustment(self, 
                               recent_performance: List[float],
                               base_kelly: float) -> float:
        """
        Dynamically adjust Kelly fraction based on recent performance.
        
        Args:
            recent_performance: Recent returns
            base_kelly: Base Kelly fraction
            
        Returns:
            Adjusted Kelly fraction
        """
        if len(recent_performance) < 10:
            return base_kelly
            
        recent_returns = np.array(recent_performance)
        
        # Calculate recent Sharpe ratio
        if np.std(recent_returns) > 0:
            recent_sharpe = np.mean(recent_returns) / np.std(recent_returns)
        else:
            recent_sharpe = 0
            
        # Adjust based on performance
        if recent_sharpe > 2:  # Excellent performance
            adjustment = 1.2
        elif recent_sharpe > 1:  # Good performance
            adjustment = 1.1
        elif recent_sharpe > 0:  # Positive performance
            adjustment = 1.0
        elif recent_sharpe > -1:  # Slightly negative
            adjustment = 0.8
        else:  # Poor performance
            adjustment = 0.5
            
        # Apply adjustment
        adjusted_kelly = base_kelly * adjustment
        
        # Ensure within bounds
        adjusted_kelly = max(0, min(adjusted_kelly, self.max_position))
        
        return adjusted_kelly
    
    def calculate_optimal_leverage(self, kelly_fraction: float,
                                 volatility: float,
                                 max_leverage: float = 3.0) -> float:
        """
        Calculate optimal leverage given Kelly fraction and volatility.
        
        Args:
            kelly_fraction: Base Kelly fraction
            volatility: Current volatility
            max_leverage: Maximum allowed leverage
            
        Returns:
            Optimal leverage
        """
        # Target volatility approach
        target_vol = 0.02  # 2% target
        
        if volatility > 0:
            optimal_leverage = (kelly_fraction * target_vol) / volatility
        else:
            optimal_leverage = 1.0
            
        # Apply constraints
        optimal_leverage = max(1.0, min(optimal_leverage, max_leverage))
        
        return optimal_leverage
    
    def risk_parity_allocation(self, 
                             volatilities: pd.Series,
                             correlations: pd.DataFrame) -> pd.Series:
        """
        Calculate risk parity allocation across assets.
        
        Args:
            volatilities: Asset volatilities
            correlations: Correlation matrix
            
        Returns:
            Risk parity weights
        """
        n_assets = len(volatilities)
        
        # Initial guess: inverse volatility weighting
        weights = 1 / volatilities
        weights /= weights.sum()
        
        # Iterative optimization for true risk parity
        for _ in range(100):
            # Calculate marginal risk contributions
            covariance = correlations * np.outer(volatilities, volatilities)
            portfolio_vol = np.sqrt(weights @ covariance @ weights)
            
            marginal_contrib = (covariance @ weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Update weights
            weights = contrib.sum() / (n_assets * marginal_contrib)
            weights /= weights.sum()
            
            # Check convergence
            if np.allclose(contrib, contrib.mean(), rtol=1e-4):
                break
                
        return pd.Series(weights, index=volatilities.index)
    
    def calculate_information_ratio_sizing(self, 
                                         alpha: float,
                                         tracking_error: float,
                                         confidence: float = 0.95) -> float:
        """
        Size positions based on information ratio.
        
        Args:
            alpha: Expected excess return
            tracking_error: Volatility of excess returns
            confidence: Confidence level
            
        Returns:
            Optimal position size
        """
        if tracking_error <= 0:
            return 0.0
            
        # Information ratio
        ir = alpha / tracking_error
        
        # Size based on IR and confidence
        # Higher IR = larger position
        base_size = np.tanh(ir / 2)  # Bounded between -1 and 1
        
        # Adjust for confidence
        position_size = base_size * confidence * self.kelly_fraction
        
        # Apply limits
        position_size = max(-self.max_position, min(position_size, self.max_position))
        
        return position_size
    
    def track_performance(self, position_size: float,
                         outcome: float,
                         timestamp: pd.Timestamp):
        """
        Track performance for analysis and adjustment.
        
        Args:
            position_size: Size of position taken
            outcome: Realized return
            timestamp: Time of trade
        """
        self.trade_history.append({
            'timestamp': timestamp,
            'position_size': position_size,
            'outcome': outcome,
            'pnl': position_size * outcome
        })
        
        # Update performance metrics
        if len(self.trade_history) >= 10:
            recent_trades = self.trade_history[-self.lookback_window:]
            outcomes = [t['outcome'] for t in recent_trades]
            
            self.performance_metrics = self.estimate_win_metrics(np.array(outcomes))
            self.performance_metrics['cumulative_return'] = np.sum([t['pnl'] for t in self.trade_history])
            self.performance_metrics['max_drawdown'] = self._calculate_max_drawdown()
            
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from trade history."""
        if not self.trade_history:
            return 0.0
            
        cumulative_pnl = np.cumsum([t['pnl'] for t in self.trade_history])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = (cumulative_pnl - running_max) / (running_max + 1e-8)
        
        return abs(np.min(drawdown))
    
    def get_position_summary(self) -> Dict:
        """Get summary of current positioning recommendations."""
        return {
            'kelly_fraction': self.kelly_fraction,
            'max_position': self.max_position,
            'performance_metrics': self.performance_metrics,
            'total_trades': len(self.trade_history),
            'recommended_confidence': self._calculate_confidence()
        }
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence based on recent performance."""
        if not self.performance_metrics:
            return 0.5
            
        # Base confidence on sample size and win rate consistency
        sample_size = self.performance_metrics.get('sample_size', 0)
        win_rate = self.performance_metrics.get('win_rate', 0.5)
        
        # More data = more confidence
        size_confidence = min(1.0, sample_size / 100)
        
        # Win rate away from 50% = more confidence
        win_confidence = 2 * abs(win_rate - 0.5)
        
        return size_confidence * win_confidence