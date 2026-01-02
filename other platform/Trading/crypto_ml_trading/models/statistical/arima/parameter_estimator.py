import numpy as np
from typing import Tuple, Optional, Dict
from statsmodels.tsa.stattools import acf, pacf
import warnings


class ARIMAParameterEstimator:
    """
    Tools for estimating ARIMA model parameters and diagnostics.
    """
    
    @staticmethod
    def estimate_differencing_order(data: np.ndarray, max_d: int = 2,
                                   method: str = 'adf') -> int:
        """
        Estimate the order of differencing needed for stationarity.
        
        Args:
            data: Time series data
            max_d: Maximum differencing order to test
            method: Test method ('adf' for Augmented Dickey-Fuller)
            
        Returns:
            Recommended differencing order
        """
        # Simple variance-based test for stationarity
        # (Full ADF test would require additional dependencies)
        
        best_d = 0
        min_variance_ratio = np.inf
        
        for d in range(max_d + 1):
            if d == 0:
                diff_data = data
            else:
                diff_data = np.diff(data, n=d)
            
            if len(diff_data) < 10:
                continue
            
            # Calculate rolling variance ratio
            window = min(len(diff_data) // 4, 30)
            if window < 3:
                continue
                
            rolling_mean = np.array([np.mean(diff_data[i:i+window]) 
                                   for i in range(len(diff_data)-window+1)])
            rolling_var = np.array([np.var(diff_data[i:i+window]) 
                                  for i in range(len(diff_data)-window+1)])
            
            # Variance of rolling statistics (lower is more stationary)
            mean_var = np.var(rolling_mean)
            var_var = np.var(rolling_var)
            variance_ratio = mean_var + var_var
            
            if variance_ratio < min_variance_ratio:
                min_variance_ratio = variance_ratio
                best_d = d
        
        return best_d
    
    @staticmethod
    def estimate_ar_order(data: np.ndarray, max_lag: int = 20,
                         ic: str = 'bic') -> int:
        """
        Estimate AR order using partial autocorrelation and information criteria.
        
        Args:
            data: Time series data (should be stationary)
            max_lag: Maximum lag to consider
            ic: Information criterion ('aic' or 'bic')
            
        Returns:
            Estimated AR order
        """
        n = len(data)
        max_lag = min(max_lag, n // 4)
        
        # Calculate PACF manually
        pacf_values = ARIMAParameterEstimator._calculate_pacf(data, max_lag)
        
        # Find significant lags (simplified without proper hypothesis testing)
        std_error = 1.96 / np.sqrt(n)  # 95% confidence
        significant_lags = np.where(np.abs(pacf_values[1:]) > std_error)[0] + 1
        
        if len(significant_lags) == 0:
            return 0
        
        # Use the last significant lag as initial estimate
        p_estimate = significant_lags[-1]
        
        # Refine using information criterion
        best_p = 0
        best_ic = np.inf
        
        for p in range(min(p_estimate + 3, max_lag + 1)):
            # Simple AR model estimation using Yule-Walker
            if p == 0:
                residual_variance = np.var(data)
            else:
                try:
                    ar_params = ARIMAParameterEstimator._yule_walker(data, p)
                    residuals = ARIMAParameterEstimator._calculate_ar_residuals(data, ar_params)
                    residual_variance = np.var(residuals)
                except:
                    continue
            
            # Calculate information criterion
            if ic == 'aic':
                ic_value = n * np.log(residual_variance) + 2 * (p + 1)
            else:  # bic
                ic_value = n * np.log(residual_variance) + np.log(n) * (p + 1)
            
            if ic_value < best_ic:
                best_ic = ic_value
                best_p = p
        
        return best_p
    
    @staticmethod
    def estimate_ma_order(data: np.ndarray, ar_order: int = 0,
                         max_lag: int = 20, ic: str = 'bic') -> int:
        """
        Estimate MA order using autocorrelation of residuals.
        
        Args:
            data: Time series data (should be stationary)
            ar_order: AR order (if already estimated)
            max_lag: Maximum lag to consider
            ic: Information criterion ('aic' or 'bic')
            
        Returns:
            Estimated MA order
        """
        n = len(data)
        max_lag = min(max_lag, n // 4)
        
        # Get residuals from AR model
        if ar_order > 0:
            ar_params = ARIMAParameterEstimator._yule_walker(data, ar_order)
            residuals = ARIMAParameterEstimator._calculate_ar_residuals(data, ar_params)
        else:
            residuals = data - np.mean(data)
        
        # Calculate ACF of residuals
        acf_values = ARIMAParameterEstimator._calculate_acf(residuals, max_lag)
        
        # Find significant lags
        std_error = 1.96 / np.sqrt(n)
        significant_lags = np.where(np.abs(acf_values[1:]) > std_error)[0] + 1
        
        if len(significant_lags) == 0:
            return 0
        
        # Use the last significant lag as initial estimate
        q_estimate = significant_lags[-1]
        
        return min(q_estimate, max_lag)
    
    @staticmethod
    def _calculate_acf(data: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate autocorrelation function."""
        n = len(data)
        data_centered = data - np.mean(data)
        c0 = np.dot(data_centered, data_centered) / n
        
        acf_values = np.zeros(max_lag + 1)
        acf_values[0] = 1.0
        
        for k in range(1, max_lag + 1):
            ck = np.dot(data_centered[:-k], data_centered[k:]) / n
            acf_values[k] = ck / c0
        
        return acf_values
    
    @staticmethod
    def _calculate_pacf(data: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate partial autocorrelation function using Yule-Walker."""
        pacf_values = np.zeros(max_lag + 1)
        pacf_values[0] = 1.0
        
        for k in range(1, max_lag + 1):
            # Fit AR(k) model and get the k-th coefficient
            try:
                ar_params = ARIMAParameterEstimator._yule_walker(data, k)
                pacf_values[k] = ar_params[-1]
            except:
                pacf_values[k] = 0.0
        
        return pacf_values
    
    @staticmethod
    def _yule_walker(data: np.ndarray, order: int) -> np.ndarray:
        """
        Solve Yule-Walker equations for AR parameters.
        
        Args:
            data: Time series data
            order: AR order
            
        Returns:
            AR parameters
        """
        # Calculate autocorrelations
        r = ARIMAParameterEstimator._calculate_acf(data, order) * np.var(data)
        
        # Build Toeplitz matrix
        R = np.zeros((order, order))
        for i in range(order):
            for j in range(order):
                R[i, j] = r[abs(i - j)]
        
        # Right-hand side
        r_vec = r[1:order+1]
        
        # Solve for AR parameters
        try:
            ar_params = np.linalg.solve(R, r_vec)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            ar_params = np.linalg.pinv(R) @ r_vec
        
        return ar_params
    
    @staticmethod
    def _calculate_ar_residuals(data: np.ndarray, ar_params: np.ndarray) -> np.ndarray:
        """Calculate residuals from AR model."""
        order = len(ar_params)
        n = len(data)
        residuals = np.zeros(n)
        
        data_centered = data - np.mean(data)
        
        for t in range(order, n):
            ar_pred = np.sum(ar_params * data_centered[t-order:t][::-1])
            residuals[t] = data_centered[t] - ar_pred
        
        return residuals[order:]
    
    @staticmethod
    def box_jenkins_identification(data: np.ndarray, max_p: int = 5,
                                 max_d: int = 2, max_q: int = 5) -> Dict:
        """
        Box-Jenkins methodology for ARIMA model identification.
        
        Args:
            data: Time series data
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
            
        Returns:
            Dictionary with suggested orders and diagnostics
        """
        # Step 1: Determine differencing order
        d = ARIMAParameterEstimator.estimate_differencing_order(data, max_d)
        
        # Apply differencing
        if d > 0:
            diff_data = np.diff(data, n=d)
        else:
            diff_data = data
        
        # Step 2: Identify AR order from PACF
        p = ARIMAParameterEstimator.estimate_ar_order(diff_data, max_p)
        
        # Step 3: Identify MA order from ACF
        q = ARIMAParameterEstimator.estimate_ma_order(diff_data, p, max_q)
        
        # Calculate ACF and PACF for diagnostics
        acf_values = ARIMAParameterEstimator._calculate_acf(diff_data, 20)
        pacf_values = ARIMAParameterEstimator._calculate_pacf(diff_data, 20)
        
        return {
            'suggested_order': (p, d, q),
            'differencing_order': d,
            'ar_order': p,
            'ma_order': q,
            'acf': acf_values.tolist(),
            'pacf': pacf_values.tolist(),
            'data_properties': {
                'mean': np.mean(diff_data),
                'variance': np.var(diff_data),
                'skewness': ARIMAParameterEstimator._calculate_skewness(diff_data),
                'kurtosis': ARIMAParameterEstimator._calculate_kurtosis(diff_data)
            }
        }
    
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
    
    @staticmethod
    def ljung_box_test(residuals: np.ndarray, lags: int = 10) -> Dict:
        """
        Ljung-Box test for residual autocorrelation.
        
        Args:
            residuals: Model residuals
            lags: Number of lags to test
            
        Returns:
            Test statistics and interpretation
        """
        n = len(residuals)
        acf_values = ARIMAParameterEstimator._calculate_acf(residuals, lags)
        
        # Calculate Q statistic
        Q = n * (n + 2) * np.sum([(acf_values[k]**2) / (n - k) 
                                  for k in range(1, lags + 1)])
        
        # Critical value (simplified - would need chi2 distribution)
        # Using approximation for 95% confidence
        critical_value = 1.96 * np.sqrt(lags)
        
        return {
            'statistic': Q,
            'critical_value': critical_value,
            'lags': lags,
            'reject_white_noise': Q > critical_value,
            'interpretation': 'Residuals show autocorrelation' if Q > critical_value 
                            else 'Residuals appear to be white noise'
        }