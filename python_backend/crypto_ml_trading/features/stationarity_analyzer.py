"""
Stationarity analysis and transformation module.

Implements various stationarity tests and transformations to ensure
time series data is suitable for modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class StationarityAnalyzer:
    """
    Analyzes and transforms time series data for stationarity.
    
    Implements:
    - Augmented Dickey-Fuller test
    - KPSS test
    - Phillips-Perron test
    - Various transformation methods
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize stationarity analyzer.
        
        Args:
            significance_level: Significance level for hypothesis tests
        """
        self.significance_level = significance_level
        self.test_results = {}
        
    def adf_test(self, series: pd.Series, maxlag: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform Augmented Dickey-Fuller test for unit root.
        
        Args:
            series: Time series data
            maxlag: Maximum lag to use
            
        Returns:
            Test results dictionary
        """
        # Remove NaN values
        series_clean = series.dropna()
        
        if len(series_clean) < 20:
            logger.warning("Series too short for ADF test")
            return {'stationary': False, 'reason': 'insufficient_data'}
        
        try:
            # Manual implementation of ADF test
            y = series_clean.values
            n = len(y)
            
            # First differences
            dy = np.diff(y)
            
            # Lag selection (simplified)
            if maxlag is None:
                maxlag = int(12 * (n / 100) ** 0.25)
            
            # Create lagged differences
            lags = []
            for i in range(1, maxlag + 1):
                lag = pd.Series(dy).shift(i).fillna(0).values
                lags.append(lag[i:])
            
            # Dependent variable
            dy_t = dy[maxlag:]
            
            # Independent variables: y_{t-1} and lagged differences
            y_lag1 = y[maxlag:-1]
            
            # Build regression matrix
            if maxlag > 0:
                X = np.column_stack([y_lag1] + lags)
            else:
                X = y_lag1.reshape(-1, 1)
            
            # Add constant
            X = np.column_stack([np.ones(len(dy_t)), X])
            
            # OLS regression
            beta = np.linalg.lstsq(X, dy_t, rcond=None)[0]
            
            # Calculate t-statistic for unit root coefficient
            residuals = dy_t - X @ beta
            sigma2 = np.sum(residuals ** 2) / (len(dy_t) - X.shape[1])
            cov_matrix = sigma2 * np.linalg.inv(X.T @ X)
            se_beta = np.sqrt(np.diag(cov_matrix))
            
            adf_stat = beta[1] / se_beta[1]  # t-stat for y_{t-1} coefficient
            
            # Critical values (MacKinnon 1994 approximation)
            n_obs = len(series_clean)
            critical_values = {
                '1%': -3.43 - 5.83/n_obs - 29.5/n_obs**2,
                '5%': -2.86 - 2.78/n_obs - 9.8/n_obs**2,
                '10%': -2.57 - 1.62/n_obs - 3.7/n_obs**2
            }
            
            # Determine if stationary
            is_stationary = adf_stat < critical_values['5%']
            
            result = {
                'stationary': is_stationary,
                'test_statistic': adf_stat,
                'critical_values': critical_values,
                'p_value': self._calculate_mackinnon_p_value(adf_stat, n_obs),
                'n_lags': maxlag,
                'n_obs': n_obs
            }
            
        except Exception as e:
            logger.error(f"ADF test failed: {e}")
            result = {'stationary': False, 'error': str(e)}
        
        self.test_results['adf'] = result
        return result
    
    def _calculate_mackinnon_p_value(self, test_stat: float, n_obs: int) -> float:
        """
        Calculate approximate p-value using MacKinnon (1994) surface.
        Simplified version - actual implementation would use full tables.
        """
        # This is a simplified approximation
        # Full implementation would use MacKinnon critical value tables
        if test_stat > 0:
            return 1.0
        elif test_stat < -4:
            return 0.0
        else:
            # Linear approximation between critical values
            if test_stat < -2.86:
                return 0.05 * (test_stat + 4) / (-2.86 + 4)
            else:
                return 0.05 + 0.95 * (test_stat + 2.86) / 2.86
    
    def kpss_test(self, series: pd.Series, regression: str = 'c') -> Dict[str, Any]:
        """
        Perform KPSS test for stationarity.
        
        Args:
            series: Time series data
            regression: 'c' for constant, 'ct' for constant and trend
            
        Returns:
            Test results dictionary
        """
        series_clean = series.dropna()
        
        if len(series_clean) < 20:
            logger.warning("Series too short for KPSS test")
            return {'stationary': True, 'reason': 'insufficient_data'}
        
        try:
            y = series_clean.values
            n = len(y)
            
            # Demean or detrend
            if regression == 'c':
                # Constant only
                residuals = y - np.mean(y)
            else:
                # Constant and trend
                X = np.column_stack([np.ones(n), np.arange(n)])
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = y - X @ beta
            
            # Calculate cumulative sum of residuals
            cum_resid = np.cumsum(residuals)
            
            # Estimate long-run variance (simplified Bartlett kernel)
            lags = int(4 * (n/100)**(1/4))
            
            # Calculate autocovariances
            gamma = []
            for k in range(lags + 1):
                if k < n:
                    gamma_k = np.sum(residuals[:-k if k > 0 else None] * 
                                   residuals[k:]) / n
                    gamma.append(gamma_k)
            
            # Bartlett weights
            weights = 1 - np.arange(lags + 1) / (lags + 1)
            
            # Long-run variance estimate
            s2 = 2 * np.sum(weights * gamma) - gamma[0]
            
            # KPSS statistic
            kpss_stat = np.sum(cum_resid**2) / (n**2 * s2)
            
            # Critical values
            if regression == 'c':
                critical_values = {
                    '10%': 0.347,
                    '5%': 0.463,
                    '2.5%': 0.574,
                    '1%': 0.739
                }
            else:  # 'ct'
                critical_values = {
                    '10%': 0.119,
                    '5%': 0.146,
                    '2.5%': 0.176,
                    '1%': 0.216
                }
            
            # KPSS null hypothesis is stationarity
            is_stationary = kpss_stat < critical_values['5%']
            
            result = {
                'stationary': is_stationary,
                'test_statistic': kpss_stat,
                'critical_values': critical_values,
                'regression': regression,
                'n_obs': n
            }
            
        except Exception as e:
            logger.error(f"KPSS test failed: {e}")
            result = {'stationary': True, 'error': str(e)}
        
        self.test_results['kpss'] = result
        return result
    
    def pp_test(self, series: pd.Series) -> Dict[str, Any]:
        """
        Perform Phillips-Perron test for unit root.
        
        Args:
            series: Time series data
            
        Returns:
            Test results dictionary
        """
        series_clean = series.dropna()
        
        if len(series_clean) < 20:
            logger.warning("Series too short for PP test")
            return {'stationary': False, 'reason': 'insufficient_data'}
        
        try:
            y = series_clean.values
            n = len(y)
            
            # Regression: y_t = alpha + beta*y_{t-1} + u_t
            y_lag = y[:-1]
            y_t = y[1:]
            
            X = np.column_stack([np.ones(len(y_lag)), y_lag])
            beta = np.linalg.lstsq(X, y_t, rcond=None)[0]
            residuals = y_t - X @ beta
            
            # Calculate standard error
            sigma2 = np.sum(residuals**2) / (len(y_t) - 2)
            se_beta = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
            
            # Initial t-statistic
            t_stat = (beta[1] - 1) / se_beta
            
            # Newey-West adjustment for serial correlation
            lags = int(4 * (n/100)**(2/9))
            
            # Calculate long-run variance
            gamma0 = np.var(residuals)
            s2_lr = gamma0
            
            for k in range(1, lags + 1):
                weight = 1 - k / (lags + 1)
                gamma_k = np.cov(residuals[k:], residuals[:-k])[0, 1]
                s2_lr += 2 * weight * gamma_k
            
            # Adjusted statistic
            adjustment = n * (s2_lr - gamma0) / (2 * gamma0)
            pp_stat = t_stat - adjustment / (n * se_beta)
            
            # Use same critical values as ADF
            critical_values = {
                '1%': -3.43 - 5.83/n - 29.5/n**2,
                '5%': -2.86 - 2.78/n - 9.8/n**2,
                '10%': -2.57 - 1.62/n - 3.7/n**2
            }
            
            is_stationary = pp_stat < critical_values['5%']
            
            result = {
                'stationary': is_stationary,
                'test_statistic': pp_stat,
                'critical_values': critical_values,
                'n_obs': n
            }
            
        except Exception as e:
            logger.error(f"PP test failed: {e}")
            result = {'stationary': False, 'error': str(e)}
        
        self.test_results['pp'] = result
        return result
    
    def test_stationarity(self, series: pd.Series, 
                         tests: List[str] = ['adf', 'kpss']) -> Dict[str, Any]:
        """
        Run multiple stationarity tests.
        
        Args:
            series: Time series data
            tests: List of tests to run
            
        Returns:
            Combined test results
        """
        results = {}
        
        if 'adf' in tests:
            results['adf'] = self.adf_test(series)
        
        if 'kpss' in tests:
            results['kpss'] = self.kpss_test(series)
        
        if 'pp' in tests:
            results['pp'] = self.pp_test(series)
        
        # Determine overall stationarity
        stationary_count = sum(1 for test in results.values() 
                              if test.get('stationary', False))
        
        results['overall_stationary'] = stationary_count >= len(tests) / 2
        results['stationary_tests'] = stationary_count
        results['total_tests'] = len(tests)
        
        return results
    
    def make_stationary(self, series: pd.Series, 
                       method: str = 'auto') -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Transform series to achieve stationarity.
        
        Args:
            series: Time series data
            method: Transformation method ('auto', 'diff', 'log', 'detrend')
            
        Returns:
            Transformed series and transformation info
        """
        info = {'method': method, 'transformations': []}
        
        if method == 'auto':
            # Test original series
            if self.test_stationarity(series)['overall_stationary']:
                return series, info
            
            # Try differencing
            diff_series = series.diff().dropna()
            if self.test_stationarity(diff_series)['overall_stationary']:
                info['transformations'].append('first_difference')
                return diff_series, info
            
            # Try log transformation + differencing
            if (series > 0).all():
                log_series = np.log(series)
                log_diff_series = log_series.diff().dropna()
                if self.test_stationarity(log_diff_series)['overall_stationary']:
                    info['transformations'].extend(['log', 'first_difference'])
                    return log_diff_series, info
            
            # Try detrending
            detrended = self.detrend(series)
            if self.test_stationarity(detrended)['overall_stationary']:
                info['transformations'].append('detrend')
                return detrended, info
            
            # Second differencing as last resort
            diff2_series = series.diff().diff().dropna()
            info['transformations'].append('second_difference')
            return diff2_series, info
            
        elif method == 'diff':
            info['transformations'].append('first_difference')
            return series.diff().dropna(), info
            
        elif method == 'log':
            if (series > 0).all():
                info['transformations'].append('log')
                return np.log(series), info
            else:
                logger.warning("Cannot apply log transformation to non-positive values")
                return series, info
                
        elif method == 'detrend':
            info['transformations'].append('detrend')
            return self.detrend(series), info
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def detrend(self, series: pd.Series, method: str = 'linear') -> pd.Series:
        """
        Remove trend from series.
        
        Args:
            series: Time series data
            method: Detrending method ('linear', 'quadratic')
            
        Returns:
            Detrended series
        """
        series_clean = series.dropna()
        n = len(series_clean)
        
        if method == 'linear':
            # Linear trend
            X = np.column_stack([np.ones(n), np.arange(n)])
        elif method == 'quadratic':
            # Quadratic trend
            t = np.arange(n)
            X = np.column_stack([np.ones(n), t, t**2])
        else:
            raise ValueError(f"Unknown detrending method: {method}")
        
        # Fit trend
        y = series_clean.values
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        trend = X @ beta
        
        # Remove trend
        detrended = pd.Series(y - trend, index=series_clean.index)
        
        return detrended
    
    def inverse_transform(self, series: pd.Series, 
                         transform_info: Dict[str, Any],
                         original_series: Optional[pd.Series] = None) -> pd.Series:
        """
        Inverse transform to get back to original scale.
        
        Args:
            series: Transformed series
            transform_info: Information about transformations applied
            original_series: Original series (needed for some inverse transforms)
            
        Returns:
            Series in original scale
        """
        result = series.copy()
        transformations = transform_info.get('transformations', [])
        
        # Apply inverse transformations in reverse order
        for transform in reversed(transformations):
            if transform == 'first_difference':
                if original_series is not None:
                    # Integrate using original series as starting point
                    result = original_series.iloc[0] + result.cumsum()
                else:
                    logger.warning("Cannot inverse first difference without original series")
                    
            elif transform == 'second_difference':
                if original_series is not None:
                    # Double integration
                    result = result.cumsum().cumsum()
                    # Adjust to match original series
                    result = result + original_series.iloc[0]
                else:
                    logger.warning("Cannot inverse second difference without original series")
                    
            elif transform == 'log':
                result = np.exp(result)
                
            elif transform == 'detrend':
                logger.warning("Cannot inverse detrending without trend information")
        
        return result