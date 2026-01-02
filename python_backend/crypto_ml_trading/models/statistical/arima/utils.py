import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
import warnings


class ARIMAUtils:
    """Utility functions for ARIMA models and time series analysis."""
    
    @staticmethod
    def check_stationarity(data: np.ndarray, window: Optional[int] = None) -> Dict:
        """
        Check stationarity of time series using rolling statistics.
        
        Args:
            data: Time series data
            window: Window size for rolling statistics (default: 1/4 of data length)
            
        Returns:
            Dictionary with stationarity indicators
        """
        n = len(data)
        if window is None:
            window = max(n // 4, 10)
        
        # Calculate rolling statistics
        rolling_mean = pd.Series(data).rolling(window=window).mean().dropna()
        rolling_std = pd.Series(data).rolling(window=window).std().dropna()
        
        # Trend test: Check if mean is changing significantly
        mean_trend = np.polyfit(range(len(rolling_mean)), rolling_mean, 1)[0]
        mean_trend_relative = abs(mean_trend) / np.mean(data) if np.mean(data) != 0 else 0
        
        # Variance test: Check if variance is changing
        std_cv = np.std(rolling_std) / np.mean(rolling_std) if np.mean(rolling_std) != 0 else 0
        
        # Simple stationarity decision
        is_stationary = mean_trend_relative < 0.01 and std_cv < 0.5
        
        return {
            'is_stationary': is_stationary,
            'mean_trend': mean_trend,
            'mean_trend_relative': mean_trend_relative,
            'std_coefficient_variation': std_cv,
            'recommendation': 'Data appears stationary' if is_stationary 
                            else 'Consider differencing or transformation'
        }
    
    @staticmethod
    def seasonal_decompose(data: np.ndarray, period: int, 
                          method: str = 'additive') -> Dict[str, np.ndarray]:
        """
        Simple seasonal decomposition.
        
        Args:
            data: Time series data
            period: Seasonal period
            method: 'additive' or 'multiplicative'
            
        Returns:
            Dictionary with trend, seasonal, and residual components
        """
        n = len(data)
        
        # Calculate trend using centered moving average
        if period % 2 == 0:
            # Even period: use 2x(period) moving average
            ma = pd.Series(data).rolling(window=period, center=True).mean()
            ma2 = ma.rolling(window=2, center=True).mean()
            trend = ma2.values
        else:
            # Odd period: use simple moving average
            trend = pd.Series(data).rolling(window=period, center=True).mean().values
        
        # Detrend
        if method == 'additive':
            detrended = data - trend
        else:  # multiplicative
            detrended = data / trend
            detrended[trend == 0] = 0
        
        # Calculate seasonal component
        seasonal = np.zeros(n)
        seasonal_pattern = np.zeros(period)
        
        for i in range(period):
            indices = np.arange(i, n, period)
            valid_indices = indices[~np.isnan(detrended[indices])]
            if len(valid_indices) > 0:
                if method == 'additive':
                    seasonal_pattern[i] = np.mean(detrended[valid_indices])
                else:
                    seasonal_pattern[i] = np.mean(detrended[valid_indices])
        
        # Normalize seasonal pattern
        if method == 'additive':
            seasonal_pattern -= np.mean(seasonal_pattern)
        else:
            seasonal_pattern /= np.mean(seasonal_pattern[seasonal_pattern != 0])
        
        # Apply seasonal pattern
        for i in range(n):
            seasonal[i] = seasonal_pattern[i % period]
        
        # Calculate residual
        if method == 'additive':
            residual = data - trend - seasonal
        else:
            residual = data / (trend * seasonal)
            residual[trend * seasonal == 0] = 0
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'seasonal_pattern': seasonal_pattern
        }
    
    @staticmethod
    def transform_data(data: np.ndarray, method: str = 'log') -> Tuple[np.ndarray, Dict]:
        """
        Transform data to achieve stationarity.
        
        Args:
            data: Time series data
            method: 'log', 'sqrt', 'box-cox', or 'none'
            
        Returns:
            Tuple of (transformed_data, transform_params)
        """
        transform_params = {'method': method}
        
        if method == 'log':
            # Add small constant to avoid log(0)
            min_val = np.min(data)
            if min_val <= 0:
                offset = abs(min_val) + 1e-8
                transform_params['offset'] = offset
                transformed = np.log(data + offset)
            else:
                transform_params['offset'] = 0
                transformed = np.log(data)
                
        elif method == 'sqrt':
            min_val = np.min(data)
            if min_val < 0:
                offset = abs(min_val)
                transform_params['offset'] = offset
                transformed = np.sqrt(data + offset)
            else:
                transform_params['offset'] = 0
                transformed = np.sqrt(data)
                
        elif method == 'box-cox':
            # Simplified Box-Cox (would need scipy for full implementation)
            # Find optimal lambda using log-likelihood
            min_val = np.min(data)
            if min_val <= 0:
                offset = abs(min_val) + 1e-8
                data_positive = data + offset
                transform_params['offset'] = offset
            else:
                data_positive = data
                transform_params['offset'] = 0
            
            # Try different lambda values
            lambdas = np.linspace(-2, 2, 41)
            log_likelihoods = []
            
            for lam in lambdas:
                if lam == 0:
                    transformed_temp = np.log(data_positive)
                else:
                    transformed_temp = (data_positive**lam - 1) / lam
                
                # Calculate log-likelihood (simplified)
                log_likelihood = -len(data) * np.log(np.std(transformed_temp))
                log_likelihoods.append(log_likelihood)
            
            # Select best lambda
            best_lambda = lambdas[np.argmax(log_likelihoods)]
            transform_params['lambda'] = best_lambda
            
            if best_lambda == 0:
                transformed = np.log(data_positive)
            else:
                transformed = (data_positive**best_lambda - 1) / best_lambda
                
        else:  # 'none'
            transformed = data.copy()
        
        return transformed, transform_params
    
    @staticmethod
    def inverse_transform(data: np.ndarray, transform_params: Dict) -> np.ndarray:
        """
        Inverse transform data back to original scale.
        
        Args:
            data: Transformed data
            transform_params: Parameters from transform_data
            
        Returns:
            Data in original scale
        """
        method = transform_params['method']
        
        if method == 'log':
            offset = transform_params.get('offset', 0)
            inverse = np.exp(data) - offset
            
        elif method == 'sqrt':
            offset = transform_params.get('offset', 0)
            inverse = data**2 - offset
            
        elif method == 'box-cox':
            lam = transform_params['lambda']
            offset = transform_params.get('offset', 0)
            
            if lam == 0:
                inverse = np.exp(data)
            else:
                inverse = (lam * data + 1)**(1/lam)
            
            inverse = inverse - offset
            
        else:  # 'none'
            inverse = data.copy()
        
        return inverse
    
    @staticmethod
    def create_lagged_features(data: np.ndarray, lags: List[int]) -> np.ndarray:
        """
        Create lagged features for time series.
        
        Args:
            data: Time series data
            lags: List of lag values
            
        Returns:
            Array with lagged features (n_samples - max_lag, n_features)
        """
        max_lag = max(lags)
        n = len(data)
        
        features = np.zeros((n - max_lag, len(lags)))
        
        for i, lag in enumerate(lags):
            features[:, i] = data[max_lag-lag:-lag if lag > 0 else None]
        
        return features
    
    @staticmethod
    def train_test_split_time_series(data: np.ndarray, test_size: float = 0.2,
                                   gap: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split time series data into train and test sets.
        
        Args:
            data: Time series data
            test_size: Proportion of data for testing
            gap: Number of periods to skip between train and test
            
        Returns:
            Tuple of (train_data, test_data)
        """
        n = len(data)
        test_periods = int(n * test_size)
        train_periods = n - test_periods - gap
        
        if train_periods <= 0:
            raise ValueError("Not enough data for train/test split with specified gap")
        
        train_data = data[:train_periods]
        test_data = data[train_periods + gap:]
        
        return train_data, test_data
    
    @staticmethod
    def expanding_window_cv(data: np.ndarray, min_train_size: int,
                           test_size: int = 1, gap: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create expanding window cross-validation splits.
        
        Args:
            data: Time series data
            min_train_size: Minimum training set size
            test_size: Size of each test set
            gap: Gap between train and test
            
        Returns:
            List of (train, test) tuples
        """
        n = len(data)
        splits = []
        
        for i in range(min_train_size, n - test_size - gap + 1):
            train = data[:i]
            test = data[i + gap:i + gap + test_size]
            splits.append((train, test))
        
        return splits
    
    @staticmethod
    def sliding_window_cv(data: np.ndarray, window_size: int,
                         test_size: int = 1, gap: int = 0,
                         step: int = 1) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create sliding window cross-validation splits.
        
        Args:
            data: Time series data
            window_size: Fixed training window size
            test_size: Size of each test set
            gap: Gap between train and test
            step: Step size for sliding window
            
        Returns:
            List of (train, test) tuples
        """
        n = len(data)
        splits = []
        
        for i in range(0, n - window_size - test_size - gap + 1, step):
            train = data[i:i + window_size]
            test = data[i + window_size + gap:i + window_size + gap + test_size]
            splits.append((train, test))
        
        return splits
    
    @staticmethod
    def calculate_prediction_intervals(point_forecast: np.ndarray,
                                     forecast_std: np.ndarray,
                                     alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals for forecasts.
        
        Args:
            point_forecast: Point predictions
            forecast_std: Standard deviation of forecasts
            alpha: Significance level (default 0.05 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Using normal distribution (would need scipy for t-distribution)
        z_score = 1.96 if alpha == 0.05 else 2.58 if alpha == 0.01 else 1.645
        
        lower = point_forecast - z_score * forecast_std
        upper = point_forecast + z_score * forecast_std
        
        return lower, upper
    
    @staticmethod
    def forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """
        Calculate forecast accuracy metrics.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary with various metrics
        """
        errors = actual - predicted
        abs_errors = np.abs(errors)
        pct_errors = errors / actual
        abs_pct_errors = np.abs(pct_errors)
        
        # Handle division by zero
        abs_pct_errors = abs_pct_errors[~np.isinf(abs_pct_errors)]
        pct_errors = pct_errors[~np.isinf(pct_errors)]
        
        metrics = {
            'mae': np.mean(abs_errors),
            'mse': np.mean(errors**2),
            'rmse': np.sqrt(np.mean(errors**2)),
            'mape': np.mean(abs_pct_errors) * 100 if len(abs_pct_errors) > 0 else np.nan,
            'bias': np.mean(errors),
            'directional_accuracy': np.mean((actual[1:] - actual[:-1]) * 
                                          (predicted[1:] - predicted[:-1]) > 0) * 100
        }
        
        return metrics