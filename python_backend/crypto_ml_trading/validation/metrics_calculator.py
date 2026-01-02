"""
Comprehensive Metrics Calculator for Model Validation.

Provides a unified interface for calculating various performance metrics
across different model types and tasks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, log_loss
)
import warnings


class ValidationMetricsCalculator:
    """
    Calculates comprehensive validation metrics for different model types.
    
    Features:
    - Regression metrics (MAE, RMSE, MAPE, R²)
    - Classification metrics (Accuracy, Precision, Recall, F1)
    - Financial metrics (Sharpe, Sortino, Calmar, Max Drawdown)
    - Statistical metrics (Correlation, Information Ratio)
    - Custom metrics for specific model types
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.metric_functions = {
            # Regression metrics
            'mae': self._calculate_mae,
            'mse': self._calculate_mse,
            'rmse': self._calculate_rmse,
            'mape': self._calculate_mape,
            'smape': self._calculate_smape,
            'r2': self._calculate_r2,
            'adjusted_r2': self._calculate_adjusted_r2,
            
            # Classification metrics
            'accuracy': self._calculate_accuracy,
            'precision': self._calculate_precision,
            'recall': self._calculate_recall,
            'f1_score': self._calculate_f1_score,
            'roc_auc': self._calculate_roc_auc,
            'log_loss': self._calculate_log_loss,
            
            # Financial metrics
            'sharpe_ratio': self._calculate_sharpe_ratio,
            'sortino_ratio': self._calculate_sortino_ratio,
            'calmar_ratio': self._calculate_calmar_ratio,
            'max_drawdown': self._calculate_max_drawdown,
            'var_95': self._calculate_var,
            'cvar_95': self._calculate_cvar,
            
            # Statistical metrics
            'correlation': self._calculate_correlation,
            'information_ratio': self._calculate_information_ratio,
            'tracking_error': self._calculate_tracking_error,
            'hit_rate': self._calculate_hit_rate,
            
            # Time series specific
            'directional_accuracy': self._calculate_directional_accuracy,
            'theil_u': self._calculate_theil_u,
            'mean_directional_accuracy': self._calculate_mean_directional_accuracy
        }
        
    def calculate_all_metrics(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            model_type: str = 'regression',
                            returns: Optional[np.ndarray] = None,
                            benchmark: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate all relevant metrics for given model type.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_type: Type of model ('regression', 'classification', 'financial')
            returns: Optional returns for financial metrics
            benchmark: Optional benchmark for comparison
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        # Ensure arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Select metrics based on model type
        if model_type == 'regression' or model_type == 'statistical':
            metric_names = [
                'mae', 'mse', 'rmse', 'mape', 'smape', 'r2',
                'correlation', 'directional_accuracy', 'theil_u'
            ]
        elif model_type == 'classification' or model_type == 'neural_network':
            # Check if binary or multiclass
            if y_pred.ndim > 1 and y_pred.shape[1] > 2:
                metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
            else:
                metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        elif model_type in ['financial', 'reinforcement_learning']:
            metric_names = [
                'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 
                'max_drawdown', 'var_95', 'cvar_95', 'hit_rate'
            ]
            if benchmark is not None:
                metric_names.extend(['information_ratio', 'tracking_error'])
        else:
            # Default to comprehensive set
            metric_names = [
                'mae', 'rmse', 'mape', 'correlation', 
                'sharpe_ratio', 'max_drawdown'
            ]
        
        # Calculate selected metrics
        for metric_name in metric_names:
            try:
                if metric_name in self.metric_functions:
                    if metric_name in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 
                                     'max_drawdown', 'var_95', 'cvar_95']:
                        # Financial metrics need returns
                        if returns is not None:
                            metrics[metric_name] = self.metric_functions[metric_name](returns)
                        else:
                            # Try to calculate returns from predictions
                            pseudo_returns = self._calculate_returns_from_predictions(y_true, y_pred)
                            metrics[metric_name] = self.metric_functions[metric_name](pseudo_returns)
                    elif metric_name in ['information_ratio', 'tracking_error']:
                        # Comparison metrics need benchmark
                        if returns is not None and benchmark is not None:
                            metrics[metric_name] = self.metric_functions[metric_name](returns, benchmark)
                    else:
                        # Standard metrics
                        metrics[metric_name] = self.metric_functions[metric_name](y_true, y_pred)
            except Exception as e:
                warnings.warn(f"Failed to calculate {metric_name}: {str(e)}")
                metrics[metric_name] = np.nan
        
        return metrics
    
    def calculate_custom_metrics(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               metric_names: List[str],
                               **kwargs) -> Dict[str, float]:
        """
        Calculate specific metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metric_names: List of metric names to calculate
            **kwargs: Additional arguments for specific metrics
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        for metric_name in metric_names:
            if metric_name in self.metric_functions:
                try:
                    metrics[metric_name] = self.metric_functions[metric_name](y_true, y_pred, **kwargs)
                except Exception as e:
                    warnings.warn(f"Failed to calculate {metric_name}: {str(e)}")
                    metrics[metric_name] = np.nan
            else:
                warnings.warn(f"Unknown metric: {metric_name}")
        
        return metrics
    
    # Regression Metrics
    
    def _calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    def _calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        return mean_squared_error(y_true, y_pred)
    
    def _calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error."""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if not np.any(mask):
            return 0.0
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    
    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared (Coefficient of Determination)."""
        return r2_score(y_true, y_pred)
    
    def _calculate_adjusted_r2(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             n_features: Optional[int] = None) -> float:
        """Adjusted R-squared."""
        r2 = r2_score(y_true, y_pred)
        n = len(y_true)
        
        if n_features is None:
            # Estimate from data if not provided
            n_features = 1
        
        if n - n_features - 1 <= 0:
            return r2
        
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        return adjusted_r2
    
    # Classification Metrics
    
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Classification Accuracy."""
        # Handle probability outputs
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        elif y_pred.ndim == 1 and not np.all(np.isin(y_pred, [0, 1])):
            y_pred = (y_pred > 0.5).astype(int)
        
        # Handle one-hot encoded true labels
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        
        return accuracy_score(y_true, y_pred)
    
    def _calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Precision (macro average for multiclass)."""
        # Handle probability outputs
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        elif y_pred.ndim == 1 and not np.all(np.isin(y_pred, [0, 1])):
            y_pred = (y_pred > 0.5).astype(int)
        
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        
        precision, _, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        return precision
    
    def _calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Recall (macro average for multiclass)."""
        # Handle probability outputs
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        elif y_pred.ndim == 1 and not np.all(np.isin(y_pred, [0, 1])):
            y_pred = (y_pred > 0.5).astype(int)
        
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        
        _, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        return recall
    
    def _calculate_f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """F1 Score (macro average for multiclass)."""
        # Handle probability outputs
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        elif y_pred.ndim == 1 and not np.all(np.isin(y_pred, [0, 1])):
            y_pred = (y_pred > 0.5).astype(int)
        
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        return f1
    
    def _calculate_roc_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ROC AUC Score (binary classification)."""
        try:
            # For probability predictions
            if y_pred.ndim == 1:
                return roc_auc_score(y_true, y_pred)
            elif y_pred.shape[1] == 2:
                return roc_auc_score(y_true, y_pred[:, 1])
            else:
                # Multiclass - use one-vs-rest
                return roc_auc_score(y_true, y_pred, multi_class='ovr')
        except:
            return np.nan
    
    def _calculate_log_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Logarithmic Loss."""
        try:
            return log_loss(y_true, y_pred)
        except:
            return np.nan
    
    # Financial Metrics
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Sharpe Ratio (annualized)."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # Assuming daily returns
        excess_returns = returns - risk_free_rate / 252
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, 
                               target_return: float = 0, 
                               risk_free_rate: float = 0.02) -> float:
        """Sortino Ratio (annualized)."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return np.inf if np.mean(excess_returns) > 0 else 0.0
        
        downside_deviation = np.std(downside_returns)
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
    
    def _calculate_calmar_ratio(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Calmar Ratio."""
        if len(returns) == 0:
            return 0.0
        
        # Annualized return
        total_return = np.prod(1 + returns) - 1
        years = len(returns) / periods_per_year
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Maximum drawdown
        max_dd = abs(self._calculate_max_drawdown(returns))
        
        if max_dd == 0:
            return np.inf if annual_return > 0 else 0.0
        
        return annual_return / max_dd
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Maximum Drawdown."""
        if len(returns) == 0:
            return 0.0
        
        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return np.min(drawdown)
    
    def _calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Value at Risk."""
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Conditional Value at Risk (Expected Shortfall)."""
        if len(returns) == 0:
            return 0.0
        
        var = self._calculate_var(returns, confidence)
        return np.mean(returns[returns <= var])
    
    # Statistical Metrics
    
    def _calculate_correlation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Pearson Correlation Coefficient."""
        if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
            return 0.0
        
        return np.corrcoef(y_true, y_pred)[0, 1]
    
    def _calculate_information_ratio(self, returns: np.ndarray, 
                                   benchmark_returns: np.ndarray) -> float:
        """Information Ratio."""
        if len(returns) != len(benchmark_returns):
            raise ValueError("Returns and benchmark must have same length")
        
        active_returns = returns - benchmark_returns
        
        if np.std(active_returns) == 0:
            return 0.0
        
        return np.mean(active_returns) / np.std(active_returns) * np.sqrt(252)
    
    def _calculate_tracking_error(self, returns: np.ndarray, 
                                benchmark_returns: np.ndarray) -> float:
        """Tracking Error (annualized)."""
        if len(returns) != len(benchmark_returns):
            raise ValueError("Returns and benchmark must have same length")
        
        active_returns = returns - benchmark_returns
        return np.std(active_returns) * np.sqrt(252)
    
    def _calculate_hit_rate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          threshold: float = 0.0) -> float:
        """Hit Rate (percentage of correct directional predictions)."""
        if len(y_true) < 2:
            return 0.0
        
        # Calculate returns/changes
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        # Hit rate
        return np.mean(true_direction == pred_direction)
    
    # Time Series Specific Metrics
    
    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Directional Accuracy for time series."""
        if len(y_true) < 2:
            return 0.0
        
        # Use changes instead of levels
        true_changes = np.diff(y_true)
        pred_changes = np.diff(y_pred)
        
        true_direction = np.sign(true_changes)
        pred_direction = np.sign(pred_changes)
        
        return np.mean(true_direction == pred_direction)
    
    def _calculate_theil_u(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Theil U Statistic (U2) for forecast accuracy."""
        if len(y_true) < 2:
            return 1.0
        
        # Calculate the forecast error
        mse_forecast = mean_squared_error(y_true[1:], y_pred[1:])
        
        # Calculate the no-change forecast error
        mse_no_change = mean_squared_error(y_true[1:], y_true[:-1])
        
        if mse_no_change == 0:
            return np.inf if mse_forecast > 0 else 0.0
        
        return np.sqrt(mse_forecast / mse_no_change)
    
    def _calculate_mean_directional_accuracy(self, y_true: np.ndarray, 
                                           y_pred: np.ndarray) -> float:
        """Mean Directional Accuracy (MDA)."""
        return self._calculate_directional_accuracy(y_true, y_pred)
    
    # Utility Methods
    
    def _calculate_returns_from_predictions(self, y_true: np.ndarray, 
                                          y_pred: np.ndarray) -> np.ndarray:
        """Calculate pseudo-returns from predictions for financial metrics."""
        if len(y_true) < 2:
            return np.array([])
        
        # Simple return calculation based on prediction accuracy
        # This is a simplified approach - actual implementation would depend on use case
        
        # Option 1: Use directional accuracy
        true_returns = np.diff(y_true) / y_true[:-1]
        pred_direction = np.sign(np.diff(y_pred))
        
        # Pseudo returns: actual returns when direction is correct, negative otherwise
        pseudo_returns = true_returns * pred_direction
        
        return pseudo_returns
    
    def calculate_residual_diagnostics(self, y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive residual diagnostics."""
        residuals = y_true - y_pred
        
        diagnostics = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'normality_test': stats.normaltest(residuals),
            'autocorrelation': self._calculate_autocorrelation(residuals),
            'heteroscedasticity': self._test_heteroscedasticity(residuals, y_pred)
        }
        
        return diagnostics
    
    def _calculate_autocorrelation(self, residuals: np.ndarray, max_lag: int = 10) -> Dict[str, float]:
        """Calculate autocorrelation of residuals."""
        acf_values = {}
        
        for lag in range(1, min(max_lag + 1, len(residuals) // 2)):
            if lag < len(residuals):
                acf_values[f'lag_{lag}'] = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
        
        return acf_values
    
    def _test_heteroscedasticity(self, residuals: np.ndarray, 
                                fitted_values: np.ndarray) -> Dict[str, Any]:
        """Test for heteroscedasticity in residuals."""
        # Breusch-Pagan test approximation
        squared_residuals = residuals ** 2
        
        # Regress squared residuals on fitted values
        X = np.column_stack([np.ones_like(fitted_values), fitted_values])
        
        try:
            beta = np.linalg.lstsq(X, squared_residuals, rcond=None)[0]
            predictions = X @ beta
            
            # R-squared of auxiliary regression
            ss_res = np.sum((squared_residuals - predictions) ** 2)
            ss_tot = np.sum((squared_residuals - np.mean(squared_residuals)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            # LM statistic
            n = len(residuals)
            lm_statistic = n * r_squared
            p_value = 1 - stats.chi2.cdf(lm_statistic, df=1)
            
            return {
                'statistic': lm_statistic,
                'p_value': p_value,
                'heteroscedastic': p_value < 0.05
            }
        except:
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'heteroscedastic': None
            }