import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from .hmm_model import RegimeDetector


class RegimeAnalyzer:
    """
    Advanced regime analysis for cryptocurrency markets.
    
    Features:
    - Regime persistence analysis
    - Transition probability estimation
    - Regime-specific strategy recommendations
    - Risk adjustment by regime
    - Performance attribution by regime
    """
    
    def __init__(self, detector: Optional[RegimeDetector] = None):
        """
        Initialize regime analyzer.
        
        Args:
            detector: Pre-fitted regime detector
        """
        self.detector = detector or RegimeDetector()
        self.regime_stats = {}
        self.transition_matrix = None
        self.regime_durations = {}
        
    def analyze_regimes(self, data: pd.DataFrame, 
                       price_col: str = 'close',
                       volume_col: str = 'volume') -> Dict:
        """
        Comprehensive regime analysis.
        
        Args:
            data: DataFrame with price and volume data
            price_col: Name of price column
            volume_col: Name of volume column
            
        Returns:
            Dictionary with regime analysis
        """
        # Calculate returns
        returns = data[price_col].pct_change().fillna(0).values
        volumes = data[volume_col].values if volume_col in data else None
        
        # Fit detector if not already fitted
        if self.detector.best_model is None:
            self.detector.fit(returns, volumes)
            
        # Predict regimes
        regimes = self.detector.predict_regime(returns, volumes)
        regime_probs = self.detector.predict_regime_proba(returns, volumes)
        
        # Calculate regime statistics
        self._calculate_regime_statistics(returns, regimes, data.index)
        
        # Calculate transition matrix
        self._calculate_transition_matrix(regimes)
        
        # Calculate regime durations
        self._calculate_regime_durations(regimes, data.index)
        
        # Detect regime changes
        regime_changes = self.detector.detect_regime_changes(returns, volumes)
        
        # Strategy recommendations
        recommendations = self._generate_strategy_recommendations(
            regimes[-1], regime_probs, len(returns)-1
        )
        
        return {
            'current_regime': regimes[-1],
            'regime_probability': {
                regime: probs[-1] 
                for regime, probs in regime_probs.items()
            },
            'regime_statistics': self.regime_stats,
            'transition_matrix': self.transition_matrix,
            'regime_durations': self.regime_durations,
            'recent_changes': regime_changes[-5:] if regime_changes else [],
            'recommendations': recommendations,
            'regime_history': pd.Series(regimes, index=data.index)
        }
    
    def _calculate_regime_statistics(self, returns: np.ndarray, 
                                   regimes: np.ndarray,
                                   index: pd.DatetimeIndex):
        """Calculate statistics for each regime."""
        unique_regimes = np.unique(regimes)
        
        for regime in unique_regimes:
            regime_mask = regimes == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) > 0:
                # Basic statistics
                stats = {
                    'mean_return': np.mean(regime_returns),
                    'volatility': np.std(regime_returns),
                    'sharpe_ratio': np.mean(regime_returns) / (np.std(regime_returns) + 1e-10) * np.sqrt(252),
                    'min_return': np.min(regime_returns),
                    'max_return': np.max(regime_returns),
                    'skewness': self._calculate_skewness(regime_returns),
                    'kurtosis': self._calculate_kurtosis(regime_returns),
                    'frequency': np.sum(regime_mask) / len(regimes),
                    'total_periods': np.sum(regime_mask)
                }
                
                # Risk metrics
                stats['var_95'] = np.percentile(regime_returns, 5)
                stats['cvar_95'] = np.mean(regime_returns[regime_returns <= stats['var_95']])
                
                # Drawdown analysis
                cumulative_returns = np.cumprod(1 + regime_returns) - 1
                stats['max_drawdown'] = self._calculate_max_drawdown(cumulative_returns)
                
                # Win rate
                stats['win_rate'] = np.mean(regime_returns > 0)
                
                self.regime_stats[regime] = stats
                
    def _calculate_transition_matrix(self, regimes: np.ndarray):
        """Calculate regime transition probability matrix."""
        unique_regimes = sorted(np.unique(regimes))
        n_regimes = len(unique_regimes)
        
        # Create mapping
        regime_to_idx = {regime: i for i, regime in enumerate(unique_regimes)}
        
        # Count transitions
        transition_counts = np.zeros((n_regimes, n_regimes))
        
        for i in range(1, len(regimes)):
            from_idx = regime_to_idx[regimes[i-1]]
            to_idx = regime_to_idx[regimes[i]]
            transition_counts[from_idx, to_idx] += 1
            
        # Normalize to get probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        
        self.transition_matrix = pd.DataFrame(
            transition_counts / row_sums,
            index=unique_regimes,
            columns=unique_regimes
        )
        
    def _calculate_regime_durations(self, regimes: np.ndarray,
                                  index: pd.DatetimeIndex):
        """Calculate duration statistics for each regime."""
        unique_regimes = np.unique(regimes)
        
        for regime in unique_regimes:
            durations = []
            current_duration = 0
            
            for i, r in enumerate(regimes):
                if r == regime:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                    current_duration = 0
                    
            if current_duration > 0:
                durations.append(current_duration)
                
            if durations:
                self.regime_durations[regime] = {
                    'mean_duration': np.mean(durations),
                    'max_duration': np.max(durations),
                    'min_duration': np.min(durations),
                    'std_duration': np.std(durations),
                    'current_duration': current_duration if regimes[-1] == regime else 0
                }
                
    def _generate_strategy_recommendations(self, current_regime: str,
                                         regime_probs: Dict[str, np.ndarray],
                                         current_idx: int) -> Dict:
        """Generate trading strategy recommendations based on regime."""
        recommendations = {
            'regime': current_regime,
            'confidence': regime_probs[current_regime][current_idx],
            'position_sizing': {},
            'risk_management': {},
            'strategy_selection': []
        }
        
        # Position sizing based on regime
        if current_regime == 'bull':
            recommendations['position_sizing'] = {
                'base_size': 1.0,
                'leverage': 1.5,
                'direction': 'long_bias',
                'volatility_adjustment': 0.8
            }
            recommendations['strategy_selection'] = [
                'trend_following', 'momentum', 'breakout'
            ]
        elif current_regime == 'bear':
            recommendations['position_sizing'] = {
                'base_size': 0.5,
                'leverage': 1.0,
                'direction': 'short_bias',
                'volatility_adjustment': 1.2
            }
            recommendations['strategy_selection'] = [
                'mean_reversion', 'short_momentum', 'defensive'
            ]
        else:  # sideways/neutral
            recommendations['position_sizing'] = {
                'base_size': 0.7,
                'leverage': 1.0,
                'direction': 'neutral',
                'volatility_adjustment': 1.0
            }
            recommendations['strategy_selection'] = [
                'mean_reversion', 'range_trading', 'arbitrage'
            ]
            
        # Risk management adjustments
        regime_stats = self.regime_stats.get(current_regime, {})
        
        recommendations['risk_management'] = {
            'stop_loss_multiplier': 1.5 if regime_stats.get('volatility', 0.02) > 0.03 else 1.0,
            'take_profit_multiplier': 2.0 if current_regime == 'bull' else 1.5,
            'max_drawdown_limit': 0.10 if current_regime == 'bear' else 0.15,
            'var_limit': abs(regime_stats.get('var_95', -0.02)) * 2
        }
        
        # Transition probability warnings
        transition_probs = self.transition_matrix.loc[current_regime] if self.transition_matrix is not None else None
        if transition_probs is not None:
            high_transition_regimes = transition_probs[transition_probs > 0.3].index.tolist()
            high_transition_regimes.remove(current_regime)  # Remove self-transition
            
            if high_transition_regimes:
                recommendations['warnings'] = {
                    'potential_transitions': high_transition_regimes,
                    'hedge_recommendation': True if 'bear' in high_transition_regimes else False
                }
                
        return recommendations
    
    def calculate_regime_performance(self, returns: np.ndarray,
                                   regimes: np.ndarray,
                                   positions: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Calculate performance metrics by regime.
        
        Args:
            returns: Asset returns
            regimes: Regime classifications
            positions: Optional position sizes
            
        Returns:
            DataFrame with performance by regime
        """
        if positions is None:
            positions = np.ones_like(returns)
            
        unique_regimes = np.unique(regimes)
        performance_data = []
        
        for regime in unique_regimes:
            regime_mask = regimes == regime
            regime_returns = returns[regime_mask]
            regime_positions = positions[regime_mask]
            
            # Calculate strategy returns
            strategy_returns = regime_returns * regime_positions
            
            if len(strategy_returns) > 0:
                # Calculate metrics
                total_return = np.prod(1 + strategy_returns) - 1
                avg_return = np.mean(strategy_returns)
                volatility = np.std(strategy_returns)
                sharpe = avg_return / volatility * np.sqrt(252) if volatility > 0 else 0
                
                # Win rate
                win_rate = np.mean(strategy_returns > 0)
                
                # Maximum drawdown
                cumulative = np.cumprod(1 + strategy_returns)
                max_dd = self._calculate_max_drawdown(cumulative - 1)
                
                performance_data.append({
                    'regime': regime,
                    'periods': np.sum(regime_mask),
                    'total_return': total_return,
                    'avg_return': avg_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'win_rate': win_rate,
                    'max_drawdown': max_dd,
                    'return_per_period': total_return / np.sum(regime_mask)
                })
                
        return pd.DataFrame(performance_data)
    
    def get_regime_specific_parameters(self, current_regime: str) -> Dict:
        """
        Get optimal parameters for current regime.
        
        Args:
            current_regime: Current market regime
            
        Returns:
            Dictionary of regime-specific parameters
        """
        params = {
            'lookback_period': 20,
            'volatility_window': 20,
            'momentum_period': 14,
            'rsi_period': 14,
            'volume_ma_period': 20
        }
        
        if current_regime == 'bull':
            params.update({
                'lookback_period': 30,
                'momentum_period': 20,
                'trend_strength_threshold': 0.6,
                'breakout_threshold': 1.5,
                'trailing_stop_multiplier': 2.0
            })
        elif current_regime == 'bear':
            params.update({
                'lookback_period': 15,
                'momentum_period': 10,
                'mean_reversion_threshold': 2.0,
                'oversold_threshold': 30,
                'stop_loss_multiplier': 1.5
            })
        else:  # sideways
            params.update({
                'lookback_period': 20,
                'bollinger_period': 20,
                'bollinger_std': 2.0,
                'range_breakout_threshold': 0.02,
                'mean_reversion_entry': 1.5
            })
            
        return params
    
    # Utility methods
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        if len(returns) < 3:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
        return np.mean(((returns - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns."""
        if len(returns) < 4:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
        return np.mean(((returns - mean) / std) ** 4) - 3
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        running_max = np.maximum.accumulate(cumulative_returns + 1)
        drawdown = (cumulative_returns + 1) / running_max - 1
        return np.min(drawdown)