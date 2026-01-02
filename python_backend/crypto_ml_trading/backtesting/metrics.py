import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats


class PerformanceMetrics:
    """
    Comprehensive performance metrics for trading strategy evaluation.
    
    Includes:
    - Return metrics
    - Risk metrics
    - Risk-adjusted returns
    - Drawdown analysis
    - Trade analysis
    - Statistical tests
    """
    
    @staticmethod
    def calculate_returns_metrics(returns: np.ndarray, 
                                periods_per_year: int = 252) -> Dict:
        """
        Calculate return-based metrics.
        
        Args:
            returns: Array of returns
            periods_per_year: Number of trading periods per year
            
        Returns:
            Dictionary of return metrics
        """
        if len(returns) == 0:
            return {}
            
        cumulative_returns = np.cumprod(1 + returns) - 1
        
        metrics = {
            'total_return': cumulative_returns[-1],
            'annualized_return': PerformanceMetrics._annualize_return(
                cumulative_returns[-1], len(returns), periods_per_year
            ),
            'average_return': np.mean(returns),
            'compound_annual_growth_rate': PerformanceMetrics._calculate_cagr(
                returns, periods_per_year
            ),
            'return_volatility': np.std(returns) * np.sqrt(periods_per_year),
            'downside_volatility': PerformanceMetrics._downside_deviation(
                returns, 0
            ) * np.sqrt(periods_per_year),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'best_day': np.max(returns),
            'worst_day': np.min(returns),
            'positive_days': np.sum(returns > 0),
            'negative_days': np.sum(returns < 0),
            'hit_rate': np.mean(returns > 0)
        }
        
        return metrics
    
    @staticmethod
    def calculate_risk_metrics(returns: np.ndarray,
                             confidence_levels: List[float] = [0.95, 0.99]) -> Dict:
        """
        Calculate risk metrics.
        
        Args:
            returns: Array of returns
            confidence_levels: VaR/CVaR confidence levels
            
        Returns:
            Dictionary of risk metrics
        """
        if len(returns) == 0:
            return {}
            
        metrics = {}
        
        # Value at Risk and Conditional VaR
        for conf in confidence_levels:
            var = np.percentile(returns, (1 - conf) * 100)
            cvar = np.mean(returns[returns <= var])
            
            metrics[f'var_{int(conf*100)}'] = var
            metrics[f'cvar_{int(conf*100)}'] = cvar
            
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        metrics['max_drawdown'] = np.min(drawdown)
        metrics['avg_drawdown'] = np.mean(drawdown[drawdown < 0]) if np.any(drawdown < 0) else 0
        
        # Drawdown duration
        dd_duration = PerformanceMetrics._calculate_drawdown_duration(drawdown)
        metrics['max_drawdown_duration'] = dd_duration['max_duration']
        metrics['avg_drawdown_duration'] = dd_duration['avg_duration']
        
        # Ulcer Index (measures downside volatility)
        metrics['ulcer_index'] = np.sqrt(np.mean(drawdown[drawdown < 0]**2)) if np.any(drawdown < 0) else 0
        
        return metrics
    
    @staticmethod
    def calculate_risk_adjusted_metrics(returns: np.ndarray,
                                      risk_free_rate: float = 0.02,
                                      periods_per_year: int = 252) -> Dict:
        """
        Calculate risk-adjusted return metrics.
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of trading periods per year
            
        Returns:
            Dictionary of risk-adjusted metrics
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return {}
            
        # Convert annual risk-free rate to period rate
        rf_period = risk_free_rate / periods_per_year
        excess_returns = returns - rf_period
        
        metrics = {
            'sharpe_ratio': np.mean(excess_returns) / np.std(returns) * np.sqrt(periods_per_year),
            'sortino_ratio': PerformanceMetrics._calculate_sortino_ratio(
                returns, rf_period, periods_per_year
            ),
            'calmar_ratio': PerformanceMetrics._calculate_calmar_ratio(
                returns, periods_per_year
            ),
            'omega_ratio': PerformanceMetrics._calculate_omega_ratio(
                returns, rf_period
            ),
            'information_ratio': PerformanceMetrics._calculate_information_ratio(
                returns, rf_period, periods_per_year
            )
        }
        
        return metrics
    
    @staticmethod
    def calculate_trade_metrics(trades: List[Dict]) -> Dict:
        """
        Calculate trade-level metrics.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary of trade metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'expectancy': 0
            }
            
        # Extract trade returns
        returns = [t.get('return_pct', 0) for t in trades]
        pnls = [t.get('pnl', 0) for t in trades]
        
        # Separate winners and losers
        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r <= 0]
        winning_pnls = [p for p in pnls if p > 0]
        losing_pnls = [p for p in pnls if p <= 0]
        
        # Calculate metrics
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
        
        gross_profit = sum(winning_pnls)
        gross_loss = abs(sum(losing_pnls))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        expectancy = np.mean(pnls) if pnls else 0
        
        # Trade duration
        durations = []
        for trade in trades:
            if 'entry_time' in trade and 'exit_time' in trade and trade['exit_time']:
                duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600
                durations.append(duration)
                
        metrics = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': avg_win / avg_loss if avg_loss > 0 else np.inf,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'expectancy_ratio': expectancy / avg_loss if avg_loss > 0 else 0,
            'avg_trade_duration': np.mean(durations) if durations else 0,
            'max_consecutive_wins': PerformanceMetrics._max_consecutive(winning_trades, True),
            'max_consecutive_losses': PerformanceMetrics._max_consecutive(losing_trades, False),
            'largest_win': max(pnls) if pnls else 0,
            'largest_loss': min(pnls) if pnls else 0
        }
        
        return metrics
    
    @staticmethod
    def calculate_rolling_metrics(returns: np.ndarray,
                                window: int = 252,
                                periods_per_year: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            returns: Array of returns
            window: Rolling window size
            periods_per_year: Number of trading periods per year
            
        Returns:
            DataFrame with rolling metrics
        """
        if len(returns) < window:
            return pd.DataFrame()
            
        # Convert to series for rolling calculations
        returns_series = pd.Series(returns)
        
        rolling_metrics = pd.DataFrame(index=returns_series.index[window-1:])
        
        # Rolling returns
        rolling_metrics['rolling_return'] = returns_series.rolling(window).apply(
            lambda x: np.prod(1 + x) - 1
        )
        
        # Rolling volatility
        rolling_metrics['rolling_volatility'] = returns_series.rolling(window).std() * np.sqrt(periods_per_year)
        
        # Rolling Sharpe
        rolling_metrics['rolling_sharpe'] = returns_series.rolling(window).apply(
            lambda x: np.mean(x) / np.std(x) * np.sqrt(periods_per_year) if np.std(x) > 0 else 0
        )
        
        # Rolling max drawdown
        rolling_metrics['rolling_max_drawdown'] = returns_series.rolling(window).apply(
            lambda x: PerformanceMetrics._calculate_max_drawdown_simple(x)
        )
        
        return rolling_metrics
    
    @staticmethod
    def statistical_tests(returns: np.ndarray,
                         benchmark_returns: Optional[np.ndarray] = None) -> Dict:
        """
        Perform statistical tests on returns.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            Dictionary of test results
        """
        tests = {}
        
        # Test for normality
        jarque_bera_stat, jarque_bera_p = stats.jarque_bera(returns)
        tests['jarque_bera'] = {
            'statistic': jarque_bera_stat,
            'p_value': jarque_bera_p,
            'is_normal': jarque_bera_p > 0.05
        }
        
        # Test for autocorrelation (Ljung-Box test approximation)
        acf_lag1 = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0
        tests['autocorrelation_lag1'] = acf_lag1
        
        # Test if returns are significantly different from zero
        t_stat, t_p_value = stats.ttest_1samp(returns, 0)
        tests['returns_significance'] = {
            't_statistic': t_stat,
            'p_value': t_p_value,
            'significant': t_p_value < 0.05
        }
        
        # Compare with benchmark if provided
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            # Test if strategy outperforms benchmark
            excess_returns = returns - benchmark_returns
            t_stat_excess, p_value_excess = stats.ttest_1samp(excess_returns, 0)
            
            tests['vs_benchmark'] = {
                't_statistic': t_stat_excess,
                'p_value': p_value_excess,
                'outperforms': t_stat_excess > 0 and p_value_excess < 0.05,
                'information_ratio': np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
            }
            
        return tests
    
    # Helper methods
    
    @staticmethod
    def _annualize_return(total_return: float, num_periods: int,
                         periods_per_year: int) -> float:
        """Annualize return."""
        if num_periods == 0:
            return 0
        years = num_periods / periods_per_year
        return (1 + total_return) ** (1 / years) - 1
    
    @staticmethod
    def _calculate_cagr(returns: np.ndarray, periods_per_year: int) -> float:
        """Calculate Compound Annual Growth Rate."""
        if len(returns) == 0:
            return 0
        total_return = np.prod(1 + returns) - 1
        years = len(returns) / periods_per_year
        return (1 + total_return) ** (1 / years) - 1
    
    @staticmethod
    def _downside_deviation(returns: np.ndarray, threshold: float = 0) -> float:
        """Calculate downside deviation."""
        downside_returns = returns[returns < threshold]
        if len(downside_returns) == 0:
            return 0
        return np.std(downside_returns)
    
    @staticmethod
    def _calculate_sortino_ratio(returns: np.ndarray, rf_rate: float,
                                periods_per_year: int) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - rf_rate
        downside_dev = PerformanceMetrics._downside_deviation(returns, rf_rate)
        if downside_dev == 0:
            return 0
        return np.mean(excess_returns) / downside_dev * np.sqrt(periods_per_year)
    
    @staticmethod
    def _calculate_calmar_ratio(returns: np.ndarray, periods_per_year: int) -> float:
        """Calculate Calmar ratio."""
        annual_return = PerformanceMetrics._calculate_cagr(returns, periods_per_year)
        max_dd = abs(PerformanceMetrics._calculate_max_drawdown_simple(returns))
        if max_dd == 0:
            return 0
        return annual_return / max_dd
    
    @staticmethod
    def _calculate_omega_ratio(returns: np.ndarray, threshold: float) -> float:
        """Calculate Omega ratio."""
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if np.sum(losses) == 0:
            return np.inf
        return np.sum(gains) / np.sum(losses)
    
    @staticmethod
    def _calculate_information_ratio(returns: np.ndarray, benchmark: float,
                                   periods_per_year: int) -> float:
        """Calculate Information ratio."""
        excess = returns - benchmark
        if np.std(excess) == 0:
            return 0
        return np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year)
    
    @staticmethod
    def _calculate_max_drawdown_simple(returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    @staticmethod
    def _calculate_drawdown_duration(drawdown: np.ndarray) -> Dict:
        """Calculate drawdown duration statistics."""
        in_drawdown = drawdown < 0
        
        # Find drawdown periods
        durations = []
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
                
        if current_duration > 0:
            durations.append(current_duration)
            
        return {
            'max_duration': max(durations) if durations else 0,
            'avg_duration': np.mean(durations) if durations else 0,
            'num_drawdowns': len(durations)
        }
    
    @staticmethod
    def _max_consecutive(trades: List, is_wins: bool) -> int:
        """Calculate maximum consecutive wins or losses."""
        if not trades:
            return 0
            
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if (is_wins and trade > 0) or (not is_wins and trade <= 0):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive