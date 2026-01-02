import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings


class GARCHUtils:
    """Utility functions for GARCH models and volatility analysis."""
    
    @staticmethod
    def test_arch_effects(returns: np.ndarray, lags: int = 10) -> Dict:
        """
        Test for ARCH effects in returns (heteroskedasticity).
        
        Args:
            returns: Return series
            lags: Number of lags to test
            
        Returns:
            Dictionary with test results
        """
        # Center returns
        centered_returns = returns - np.mean(returns)
        squared_returns = centered_returns**2
        
        # Calculate autocorrelations of squared returns
        acf_values = []
        for lag in range(1, lags + 1):
            if len(squared_returns) > lag:
                acf = np.corrcoef(squared_returns[:-lag], squared_returns[lag:])[0, 1]
                acf_values.append(acf)
            else:
                acf_values.append(0)
                
        # Ljung-Box test statistic (simplified)
        n = len(returns)
        lb_stat = n * (n + 2) * sum([(acf**2) / (n - i) 
                                     for i, acf in enumerate(acf_values, 1)])
        
        # Critical value approximation (would need chi2 for exact)
        critical_value = 2 * lags  # Rough approximation
        
        return {
            'has_arch_effects': lb_stat > critical_value,
            'ljung_box_statistic': lb_stat,
            'critical_value': critical_value,
            'squared_return_acf': acf_values,
            'recommendation': 'Use GARCH model' if lb_stat > critical_value 
                            else 'ARCH effects not significant'
        }
    
    @staticmethod
    def select_garch_order(returns: np.ndarray, max_p: int = 3,
                          max_q: int = 3, ic: str = 'bic') -> Tuple[int, int]:
        """
        Select optimal GARCH order using information criteria.
        
        Args:
            returns: Return series
            max_p: Maximum GARCH order
            max_q: Maximum ARCH order
            ic: Information criterion ('aic' or 'bic')
            
        Returns:
            Tuple of (p, q) optimal orders
        """
        from models.statistical.garch.garch_model import GARCH
        
        best_order = (1, 1)
        best_ic = np.inf
        results = []
        
        for p in range(1, max_p + 1):
            for q in range(1, max_q + 1):
                try:
                    model = GARCH(p=p, q=q, dist='t')
                    model.fit(returns)
                    
                    ic_value = model.aic if ic == 'aic' else model.bic
                    
                    results.append({
                        'order': (p, q),
                        'aic': model.aic,
                        'bic': model.bic,
                        'log_likelihood': model.log_likelihood
                    })
                    
                    if ic_value < best_ic:
                        best_ic = ic_value
                        best_order = (p, q)
                        
                except:
                    continue
                    
        return best_order
    
    @staticmethod
    def volatility_persistence(alpha: np.ndarray, beta: np.ndarray) -> Dict:
        """
        Calculate volatility persistence and half-life.
        
        Args:
            alpha: ARCH parameters
            beta: GARCH parameters
            
        Returns:
            Dictionary with persistence metrics
        """
        persistence = np.sum(alpha) + np.sum(beta)
        
        # Half-life of volatility shock
        if persistence < 1 and persistence > 0:
            half_life = np.log(0.5) / np.log(persistence)
        else:
            half_life = np.inf if persistence >= 1 else 0
            
        # Unconditional variance exists only if persistence < 1
        is_stationary = persistence < 1
        
        return {
            'persistence': persistence,
            'half_life': half_life,
            'is_stationary': is_stationary,
            'shock_duration': 'permanent' if persistence >= 1 else f'{half_life:.1f} periods'
        }
    
    @staticmethod
    def news_impact_curve(model, shock_range: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Calculate news impact curve showing how shocks affect volatility.
        
        Args:
            model: Fitted GARCH model
            shock_range: Range of shocks to consider
            
        Returns:
            DataFrame with news impact curve
        """
        if shock_range is None:
            shock_range = np.linspace(-3, 3, 61)
            
        # Get model parameters
        omega = model.omega
        alpha = model.alpha[0] if len(model.alpha) > 0 else 0
        
        # Calculate baseline variance (no shock)
        if hasattr(model, 'conditional_variance'):
            baseline_var = np.mean(model.conditional_variance)
        else:
            baseline_var = omega / (1 - alpha - np.sum(model.beta))
            
        # Calculate impact for each shock
        impacts = []
        for shock in shock_range:
            # Next period variance given shock
            next_var = omega + alpha * shock**2 + model.beta[0] * baseline_var
            
            impacts.append({
                'shock': shock,
                'variance': next_var,
                'volatility': np.sqrt(next_var),
                'variance_ratio': next_var / baseline_var
            })
            
        return pd.DataFrame(impacts)
    
    @staticmethod
    def calculate_volatility_ratios(returns: np.ndarray, 
                                  periods: List[int] = None) -> Dict:
        """
        Calculate volatility ratios for different time periods.
        
        Args:
            returns: Return series
            periods: List of periods to calculate volatility
            
        Returns:
            Dictionary with volatility ratios
        """
        if periods is None:
            periods = [5, 10, 30, 60]  # 5min to 1hour
            
        volatilities = {}
        
        for period in periods:
            if len(returns) >= period:
                # Calculate rolling volatility
                rolling_vol = pd.Series(returns).rolling(period).std()
                volatilities[f'vol_{period}'] = rolling_vol.iloc[-1]
            else:
                volatilities[f'vol_{period}'] = np.nan
                
        # Calculate ratios
        ratios = {}
        if 'vol_5' in volatilities and 'vol_60' in volatilities:
            ratios['short_long_ratio'] = volatilities['vol_5'] / volatilities['vol_60']
            
        if 'vol_10' in volatilities and 'vol_30' in volatilities:
            ratios['medium_ratio'] = volatilities['vol_10'] / volatilities['vol_30']
            
        return {
            'volatilities': volatilities,
            'ratios': ratios,
            'trend': 'increasing' if ratios.get('short_long_ratio', 1) > 1.1 else 
                    'decreasing' if ratios.get('short_long_ratio', 1) < 0.9 else 'stable'
        }
    
    @staticmethod
    def extreme_value_analysis(returns: np.ndarray, 
                             threshold_percentile: float = 95) -> Dict:
        """
        Analyze extreme values for tail risk assessment.
        
        Args:
            returns: Return series
            threshold_percentile: Percentile for extreme values
            
        Returns:
            Dictionary with extreme value statistics
        """
        # Calculate thresholds
        upper_threshold = np.percentile(returns, threshold_percentile)
        lower_threshold = np.percentile(returns, 100 - threshold_percentile)
        
        # Identify extremes
        upper_extremes = returns[returns > upper_threshold]
        lower_extremes = returns[returns < lower_threshold]
        
        # Calculate statistics
        stats = {
            'upper_threshold': upper_threshold,
            'lower_threshold': lower_threshold,
            'num_upper_extremes': len(upper_extremes),
            'num_lower_extremes': len(lower_extremes),
            'prob_upper_extreme': len(upper_extremes) / len(returns),
            'prob_lower_extreme': len(lower_extremes) / len(returns),
            'mean_upper_extreme': np.mean(upper_extremes) if len(upper_extremes) > 0 else 0,
            'mean_lower_extreme': np.mean(lower_extremes) if len(lower_extremes) > 0 else 0,
            'max_return': np.max(returns),
            'min_return': np.min(returns),
            'tail_ratio': len(lower_extremes) / len(upper_extremes) if len(upper_extremes) > 0 else np.inf
        }
        
        return stats
    
    @staticmethod
    def volatility_clustering_analysis(returns: np.ndarray, 
                                     window: int = 20) -> Dict:
        """
        Analyze volatility clustering patterns.
        
        Args:
            returns: Return series
            window: Window for calculating local volatility
            
        Returns:
            Dictionary with clustering metrics
        """
        # Calculate rolling volatility
        rolling_vol = pd.Series(returns).rolling(window).std()
        
        # Identify high and low volatility periods
        median_vol = rolling_vol.median()
        high_vol_periods = rolling_vol > median_vol * 1.5
        low_vol_periods = rolling_vol < median_vol * 0.67
        
        # Calculate runs (consecutive periods of high/low vol)
        high_vol_runs = []
        low_vol_runs = []
        
        current_run = 0
        in_high_vol = False
        
        for is_high in high_vol_periods:
            if is_high and not in_high_vol:
                in_high_vol = True
                current_run = 1
            elif is_high and in_high_vol:
                current_run += 1
            elif not is_high and in_high_vol:
                high_vol_runs.append(current_run)
                in_high_vol = False
                current_run = 0
                
        # Similar for low volatility
        current_run = 0
        in_low_vol = False
        
        for is_low in low_vol_periods:
            if is_low and not in_low_vol:
                in_low_vol = True
                current_run = 1
            elif is_low and in_low_vol:
                current_run += 1
            elif not is_low and in_low_vol:
                low_vol_runs.append(current_run)
                in_low_vol = False
                current_run = 0
                
        return {
            'median_volatility': median_vol,
            'pct_high_vol_periods': high_vol_periods.mean() * 100,
            'pct_low_vol_periods': low_vol_periods.mean() * 100,
            'avg_high_vol_duration': np.mean(high_vol_runs) if high_vol_runs else 0,
            'avg_low_vol_duration': np.mean(low_vol_runs) if low_vol_runs else 0,
            'max_high_vol_duration': max(high_vol_runs) if high_vol_runs else 0,
            'clustering_present': len(high_vol_runs) > 0 or len(low_vol_runs) > 0
        }
    
    @staticmethod
    def calculate_risk_metrics(returns: np.ndarray, 
                             volatility_forecast: np.ndarray,
                             confidence_levels: List[float] = None) -> Dict:
        """
        Calculate comprehensive risk metrics using GARCH volatility.
        
        Args:
            returns: Historical returns
            volatility_forecast: Forecasted volatility
            confidence_levels: VaR confidence levels
            
        Returns:
            Dictionary with risk metrics
        """
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]
            
        metrics = {}
        
        # Historical metrics
        metrics['historical_volatility'] = np.std(returns)
        metrics['historical_skewness'] = GARCHUtils._calculate_skewness(returns)
        metrics['historical_kurtosis'] = GARCHUtils._calculate_kurtosis(returns)
        
        # Downside risk metrics
        negative_returns = returns[returns < 0]
        metrics['downside_deviation'] = np.std(negative_returns) if len(negative_returns) > 0 else 0
        metrics['semi_variance'] = np.mean(negative_returns**2) if len(negative_returns) > 0 else 0
        
        # VaR and CVaR using GARCH forecast
        for conf in confidence_levels:
            z_score = np.abs(np.percentile(returns / np.std(returns), (1 - conf) * 100))
            metrics[f'var_{int(conf*100)}'] = z_score * volatility_forecast[0]
            
            # CVaR (expected shortfall)
            tail_returns = returns[returns <= -metrics[f'var_{int(conf*100)}']]
            metrics[f'cvar_{int(conf*100)}'] = -np.mean(tail_returns) if len(tail_returns) > 0 else metrics[f'var_{int(conf*100)}']
            
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        metrics['max_drawdown'] = np.min(drawdown)
        
        return metrics
    
    @staticmethod
    def _calculate_skewness(data: np.ndarray) -> float:
        """Calculate skewness."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    @staticmethod
    def _calculate_kurtosis(data: np.ndarray) -> float:
        """Calculate excess kurtosis."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3