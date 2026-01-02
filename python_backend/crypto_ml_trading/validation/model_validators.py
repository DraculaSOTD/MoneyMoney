"""
Model-Specific Validation Strategies.

Implements tailored validation approaches for different model types:
- Statistical models (ARIMA/GARCH)
- Neural Networks
- Reinforcement Learning
- Ensemble models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class StatisticalModelValidator:
    """
    Validator for statistical models (ARIMA, GARCH, etc.).
    
    Features:
    - Information criteria (AIC, BIC)
    - Residual diagnostics
    - Forecast accuracy metrics
    - Model stability tests
    """
    
    def __init__(self):
        """Initialize statistical model validator."""
        self.validation_results = {}
        
    def validate_arima(self,
                      model: Any,
                      train_data: np.ndarray,
                      test_data: np.ndarray,
                      forecast_horizons: List[int] = [1, 5, 10]) -> Dict[str, Any]:
        """
        Validate ARIMA model.
        
        Args:
            model: Fitted ARIMA model
            train_data: Training data
            test_data: Test data
            forecast_horizons: Forecast horizons to evaluate
            
        Returns:
            Dictionary of validation metrics
        """
        results = {
            'model_type': 'ARIMA',
            'timestamp': datetime.now(),
            'information_criteria': {},
            'residual_diagnostics': {},
            'forecast_accuracy': {},
            'stability_tests': {}
        }
        
        # Information criteria
        n_params = model.k_ar + model.k_ma + model.k_trend
        n_obs = len(train_data)
        
        # Get log likelihood (simplified - would need actual model implementation)
        residuals = self._get_residuals(model, train_data)
        log_likelihood = -0.5 * n_obs * (np.log(2 * np.pi) + np.log(np.var(residuals)) + 1)
        
        results['information_criteria']['AIC'] = 2 * n_params - 2 * log_likelihood
        results['information_criteria']['BIC'] = np.log(n_obs) * n_params - 2 * log_likelihood
        results['information_criteria']['AICc'] = results['information_criteria']['AIC'] + \
            (2 * n_params * (n_params + 1)) / (n_obs - n_params - 1)
        
        # Residual diagnostics
        results['residual_diagnostics'] = self._diagnose_residuals(residuals)
        
        # Forecast accuracy
        for horizon in forecast_horizons:
            forecast_metrics = self._evaluate_forecasts(
                model, train_data, test_data, horizon
            )
            results['forecast_accuracy'][f'h{horizon}'] = forecast_metrics
        
        # Stability tests
        results['stability_tests'] = self._test_parameter_stability(
            model, train_data
        )
        
        return results
    
    def validate_garch(self,
                      model: Any,
                      returns: np.ndarray,
                      test_returns: np.ndarray) -> Dict[str, Any]:
        """
        Validate GARCH model.
        
        Args:
            model: Fitted GARCH model
            returns: Training returns
            test_returns: Test returns
            
        Returns:
            Dictionary of validation metrics
        """
        results = {
            'model_type': 'GARCH',
            'timestamp': datetime.now(),
            'volatility_metrics': {},
            'likelihood_metrics': {},
            'forecast_evaluation': {}
        }
        
        # Standardized residuals
        conditional_volatility = self._get_conditional_volatility(model, returns)
        standardized_residuals = returns / conditional_volatility
        
        # Volatility clustering tests
        results['volatility_metrics']['arch_lm_test'] = self._arch_lm_test(
            standardized_residuals
        )
        
        # Likelihood-based metrics
        results['likelihood_metrics'] = {
            'log_likelihood': self._garch_log_likelihood(model, returns),
            'volatility_persistence': self._calculate_persistence(model)
        }
        
        # Forecast evaluation
        results['forecast_evaluation'] = self._evaluate_volatility_forecasts(
            model, returns, test_returns
        )
        
        return results
    
    def _get_residuals(self, model: Any, data: np.ndarray) -> np.ndarray:
        """Extract residuals from model."""
        # Simplified - actual implementation would depend on model type
        # This is a placeholder
        fitted_values = np.zeros_like(data)
        for i in range(len(data)):
            if i > 0:
                fitted_values[i] = 0.5 * data[i-1]  # Simple AR(1) example
        return data - fitted_values
    
    def _diagnose_residuals(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Perform residual diagnostics."""
        diagnostics = {}
        
        # Normality tests
        jb_stat, jb_pvalue = stats.jarque_bera(residuals)
        diagnostics['normality'] = {
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'is_normal': jb_pvalue > 0.05
        }
        
        # Autocorrelation tests
        diagnostics['autocorrelation'] = self._ljung_box_test(residuals)
        
        # Heteroscedasticity tests
        diagnostics['heteroscedasticity'] = self._breusch_pagan_test(residuals)
        
        # Distribution statistics
        diagnostics['distribution'] = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals)
        }
        
        return diagnostics
    
    def _ljung_box_test(self, residuals: np.ndarray, lags: int = 10) -> Dict[str, Any]:
        """Ljung-Box test for autocorrelation."""
        n = len(residuals)
        acf = np.array([np.corrcoef(residuals[:-i], residuals[i:])[0, 1] 
                       for i in range(1, lags + 1)])
        
        # Ljung-Box statistic
        lb_stat = n * (n + 2) * np.sum(acf**2 / (n - np.arange(1, lags + 1)))
        p_value = 1 - stats.chi2.cdf(lb_stat, lags)
        
        return {
            'statistic': lb_stat,
            'p_value': p_value,
            'no_autocorrelation': p_value > 0.05,
            'max_acf': np.max(np.abs(acf))
        }
    
    def _breusch_pagan_test(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Breusch-Pagan test for heteroscedasticity."""
        n = len(residuals)
        squared_residuals = residuals ** 2
        
        # Regress squared residuals on time trend
        X = np.column_stack([np.ones(n), np.arange(n)])
        beta = np.linalg.lstsq(X, squared_residuals, rcond=None)[0]
        fitted = X @ beta
        
        # Test statistic
        ssr = np.sum((squared_residuals - fitted) ** 2)
        sst = np.sum((squared_residuals - np.mean(squared_residuals)) ** 2)
        r_squared = 1 - ssr / sst
        
        lm_stat = n * r_squared
        p_value = 1 - stats.chi2.cdf(lm_stat, 1)
        
        return {
            'statistic': lm_stat,
            'p_value': p_value,
            'homoscedastic': p_value > 0.05
        }
    
    def _evaluate_forecasts(self,
                           model: Any,
                           train_data: np.ndarray,
                           test_data: np.ndarray,
                           horizon: int) -> Dict[str, float]:
        """Evaluate forecast accuracy."""
        n_forecasts = len(test_data) - horizon + 1
        forecasts = []
        actuals = []
        
        for i in range(n_forecasts):
            # Rolling forecast
            history = np.concatenate([train_data, test_data[:i]])
            
            # Generate forecast (simplified)
            forecast = self._generate_forecast(model, history, horizon)
            actual = test_data[i:i+horizon]
            
            forecasts.append(forecast)
            actuals.append(actual)
        
        forecasts = np.array(forecasts)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mae = mean_absolute_error(actuals.flatten(), forecasts.flatten())
        rmse = np.sqrt(mean_squared_error(actuals.flatten(), forecasts.flatten()))
        mape = np.mean(np.abs((actuals - forecasts) / (actuals + 1e-10))) * 100
        
        # Directional accuracy
        if horizon == 1:
            direction_accuracy = np.mean(
                np.sign(forecasts.flatten() - train_data[-1]) == 
                np.sign(actuals.flatten() - train_data[-1])
            )
        else:
            direction_accuracy = np.nan
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'direction_accuracy': direction_accuracy
        }
    
    def _generate_forecast(self, model: Any, history: np.ndarray, horizon: int) -> np.ndarray:
        """Generate forecast (placeholder)."""
        # Simplified forecast - actual implementation would use model's forecast method
        return np.random.normal(np.mean(history), np.std(history), horizon)
    
    def _test_parameter_stability(self, model: Any, data: np.ndarray) -> Dict[str, Any]:
        """Test parameter stability over time."""
        n = len(data)
        window_size = n // 3
        
        # Estimate parameters on different windows
        param_estimates = []
        
        for i in range(3):
            start = i * window_size
            end = min((i + 1) * window_size, n)
            window_data = data[start:end]
            
            # Estimate parameters (simplified)
            params = {'mean': np.mean(window_data), 'std': np.std(window_data)}
            param_estimates.append(params)
        
        # Check stability
        means = [p['mean'] for p in param_estimates]
        stds = [p['std'] for p in param_estimates]
        
        return {
            'mean_cv': np.std(means) / (np.mean(means) + 1e-10),
            'std_cv': np.std(stds) / (np.mean(stds) + 1e-10),
            'parameters_stable': np.std(means) / (np.mean(means) + 1e-10) < 0.1
        }
    
    def _arch_lm_test(self, residuals: np.ndarray, lags: int = 5) -> Dict[str, Any]:
        """ARCH LM test for volatility clustering."""
        squared_residuals = residuals ** 2
        n = len(squared_residuals)
        
        # Create lagged squared residuals
        X = np.ones((n - lags, lags + 1))
        for i in range(1, lags + 1):
            X[:, i] = squared_residuals[lags-i:-i]
        
        y = squared_residuals[lags:]
        
        # OLS regression
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        fitted = X @ beta
        residuals_reg = y - fitted
        
        # R-squared
        ssr = np.sum(residuals_reg ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ssr / sst
        
        # Test statistic
        lm_stat = (n - lags) * r_squared
        p_value = 1 - stats.chi2.cdf(lm_stat, lags)
        
        return {
            'statistic': lm_stat,
            'p_value': p_value,
            'no_arch_effects': p_value > 0.05
        }
    
    def _get_conditional_volatility(self, model: Any, returns: np.ndarray) -> np.ndarray:
        """Get conditional volatility from GARCH model."""
        # Placeholder - actual implementation would use model's methods
        return np.ones_like(returns) * np.std(returns)
    
    def _garch_log_likelihood(self, model: Any, returns: np.ndarray) -> float:
        """Calculate GARCH log-likelihood."""
        # Placeholder
        return -0.5 * len(returns) * (np.log(2 * np.pi) + np.log(np.var(returns)) + 1)
    
    def _calculate_persistence(self, model: Any) -> float:
        """Calculate volatility persistence."""
        # Placeholder - would sum ARCH and GARCH coefficients
        return 0.95
    
    def _evaluate_volatility_forecasts(self,
                                      model: Any,
                                      returns: np.ndarray,
                                      test_returns: np.ndarray) -> Dict[str, float]:
        """Evaluate volatility forecast accuracy."""
        # Generate volatility forecasts
        n_test = len(test_returns)
        volatility_forecasts = np.ones(n_test) * np.std(returns)  # Placeholder
        
        # Realized volatility (squared returns as proxy)
        realized_vol = test_returns ** 2
        
        # Metrics
        return {
            'mae': mean_absolute_error(realized_vol, volatility_forecasts),
            'rmse': np.sqrt(mean_squared_error(realized_vol, volatility_forecasts)),
            'correlation': np.corrcoef(realized_vol, volatility_forecasts)[0, 1]
        }


class NeuralNetworkValidator:
    """
    Validator for neural network models.
    
    Features:
    - Loss curve analysis
    - Overfitting detection
    - Gradient health checks
    - Activation statistics
    - Early stopping validation
    """
    
    def __init__(self):
        """Initialize neural network validator."""
        self.validation_results = {}
        
    def validate_training(self,
                         model: Any,
                         train_history: Dict[str, List[float]],
                         val_data: Tuple[np.ndarray, np.ndarray],
                         test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Validate neural network training.
        
        Args:
            model: Trained neural network
            train_history: Training history (losses, metrics)
            val_data: Validation data (X, y)
            test_data: Optional test data
            
        Returns:
            Dictionary of validation metrics
        """
        results = {
            'model_type': 'NeuralNetwork',
            'timestamp': datetime.now(),
            'convergence_analysis': {},
            'overfitting_metrics': {},
            'gradient_health': {},
            'generalization_metrics': {}
        }
        
        # Convergence analysis
        results['convergence_analysis'] = self._analyze_convergence(train_history)
        
        # Overfitting detection
        results['overfitting_metrics'] = self._detect_overfitting(train_history)
        
        # Gradient health (if available)
        if hasattr(model, 'get_gradients'):
            results['gradient_health'] = self._check_gradient_health(model)
        
        # Generalization metrics
        X_val, y_val = val_data
        results['generalization_metrics'] = self._evaluate_generalization(
            model, X_val, y_val, test_data
        )
        
        return results
    
    def validate_predictions(self,
                           model: Any,
                           X: np.ndarray,
                           y_true: np.ndarray,
                           task_type: str = 'classification') -> Dict[str, Any]:
        """
        Validate model predictions.
        
        Args:
            model: Neural network model
            X: Input data
            y_true: True labels
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary of prediction metrics
        """
        # Get predictions
        y_pred = model.predict(X)
        
        results = {
            'task_type': task_type,
            'basic_metrics': {},
            'confidence_analysis': {},
            'error_analysis': {}
        }
        
        if task_type == 'classification':
            results['basic_metrics'] = self._classification_metrics(y_true, y_pred)
            results['confidence_analysis'] = self._analyze_confidence(y_pred, y_true)
        else:
            results['basic_metrics'] = self._regression_metrics(y_true, y_pred)
            results['error_analysis'] = self._analyze_errors(y_true, y_pred)
        
        return results
    
    def _analyze_convergence(self, history: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze training convergence."""
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        
        if not train_loss:
            return {}
        
        analysis = {
            'converged': False,
            'convergence_epoch': None,
            'final_train_loss': train_loss[-1],
            'best_val_loss': min(val_loss) if val_loss else None,
            'loss_reduction': (train_loss[0] - train_loss[-1]) / train_loss[0]
        }
        
        # Check convergence (loss plateaued)
        if len(train_loss) > 10:
            recent_losses = train_loss[-10:]
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            
            analysis['converged'] = loss_std / loss_mean < 0.01
            
            # Find convergence epoch
            for i in range(10, len(train_loss)):
                window = train_loss[i-10:i]
                if np.std(window) / np.mean(window) < 0.01:
                    analysis['convergence_epoch'] = i
                    break
        
        return analysis
    
    def _detect_overfitting(self, history: Dict[str, List[float]]) -> Dict[str, Any]:
        """Detect overfitting from training history."""
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        
        if not train_loss or not val_loss:
            return {}
        
        # Calculate overfitting metrics
        gap = np.array(val_loss) - np.array(train_loss[:len(val_loss)])
        
        overfitting_metrics = {
            'train_val_gap': gap[-1] if len(gap) > 0 else 0,
            'max_gap': np.max(gap) if len(gap) > 0 else 0,
            'gap_trend': np.polyfit(range(len(gap)), gap, 1)[0] if len(gap) > 1 else 0,
            'is_overfitting': gap[-1] > 0.1 if len(gap) > 0 else False
        }
        
        # Early stopping analysis
        best_val_epoch = np.argmin(val_loss)
        overfitting_metrics['best_epoch'] = best_val_epoch
        overfitting_metrics['epochs_after_best'] = len(val_loss) - best_val_epoch - 1
        
        return overfitting_metrics
    
    def _check_gradient_health(self, model: Any) -> Dict[str, Any]:
        """Check gradient health metrics."""
        try:
            gradients = model.get_gradients()
            
            gradient_stats = {
                'mean_gradient': np.mean([np.mean(np.abs(g)) for g in gradients]),
                'max_gradient': np.max([np.max(np.abs(g)) for g in gradients]),
                'gradient_norm': np.sqrt(sum(np.sum(g**2) for g in gradients)),
                'vanishing_gradients': any(np.max(np.abs(g)) < 1e-7 for g in gradients),
                'exploding_gradients': any(np.max(np.abs(g)) > 1e3 for g in gradients)
            }
            
            return gradient_stats
        except:
            return {'error': 'Gradient information not available'}
    
    def _evaluate_generalization(self,
                               model: Any,
                               X_val: np.ndarray,
                               y_val: np.ndarray,
                               test_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Evaluate model generalization."""
        # Validation performance
        val_pred = model.predict(X_val)
        val_loss = self._calculate_loss(y_val, val_pred)
        
        generalization = {
            'validation_loss': val_loss,
            'validation_accuracy': self._calculate_accuracy(y_val, val_pred)
        }
        
        # Test performance if available
        if test_data is not None:
            X_test, y_test = test_data
            test_pred = model.predict(X_test)
            
            generalization['test_loss'] = self._calculate_loss(y_test, test_pred)
            generalization['test_accuracy'] = self._calculate_accuracy(y_test, test_pred)
            generalization['val_test_gap'] = abs(
                generalization['validation_loss'] - generalization['test_loss']
            )
        
        return generalization
    
    def _classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics."""
        # Handle probability outputs
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred_class = np.argmax(y_pred, axis=1)
        else:
            y_pred_class = (y_pred > 0.5).astype(int).flatten()
        
        # Ensure y_true is 1D
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        
        # Basic metrics
        accuracy = np.mean(y_pred_class == y_true)
        
        # Per-class metrics
        unique_classes = np.unique(y_true)
        precision_per_class = []
        recall_per_class = []
        
        for c in unique_classes:
            true_positives = np.sum((y_true == c) & (y_pred_class == c))
            false_positives = np.sum((y_true != c) & (y_pred_class == c))
            false_negatives = np.sum((y_true == c) & (y_pred_class != c))
            
            precision = true_positives / (true_positives + false_positives + 1e-10)
            recall = true_positives / (true_positives + false_negatives + 1e-10)
            
            precision_per_class.append(precision)
            recall_per_class.append(recall)
        
        return {
            'accuracy': accuracy,
            'mean_precision': np.mean(precision_per_class),
            'mean_recall': np.mean(recall_per_class),
            'f1_score': 2 * np.mean(precision_per_class) * np.mean(recall_per_class) / 
                       (np.mean(precision_per_class) + np.mean(recall_per_class) + 1e-10)
        }
    
    def _regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
        }
    
    def _analyze_confidence(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction confidence."""
        if y_pred.ndim == 1:
            confidences = np.abs(y_pred - 0.5) * 2  # For binary
        else:
            confidences = np.max(y_pred, axis=1)  # Max probability
        
        # Get predicted classes
        if y_pred.ndim > 1:
            y_pred_class = np.argmax(y_pred, axis=1)
        else:
            y_pred_class = (y_pred > 0.5).astype(int)
        
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        
        correct = y_pred_class == y_true
        
        return {
            'mean_confidence': np.mean(confidences),
            'confidence_when_correct': np.mean(confidences[correct]) if np.any(correct) else 0,
            'confidence_when_wrong': np.mean(confidences[~correct]) if np.any(~correct) else 0,
            'confidence_correlation': np.corrcoef(confidences, correct.astype(float))[0, 1]
        }
    
    def _analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction errors."""
        errors = y_true - y_pred
        
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'skewness': stats.skew(errors.flatten()),
            'kurtosis': stats.kurtosis(errors.flatten()),
            'error_quantiles': {
                'q05': np.percentile(errors, 5),
                'q25': np.percentile(errors, 25),
                'q50': np.percentile(errors, 50),
                'q75': np.percentile(errors, 75),
                'q95': np.percentile(errors, 95)
            }
        }
    
    def _calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate loss (MSE for simplicity)."""
        return mean_squared_error(y_true, y_pred)
    
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy."""
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            # Classification
            return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
        else:
            # Regression - use R²
            return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)


class ReinforcementLearningValidator:
    """
    Validator for reinforcement learning models.
    
    Features:
    - Cumulative reward analysis
    - Policy stability metrics
    - Exploration vs exploitation balance
    - Episode performance tracking
    """
    
    def __init__(self):
        """Initialize RL validator."""
        self.validation_results = {}
        
    def validate_training(self,
                         episode_rewards: List[float],
                         episode_lengths: List[int],
                         action_distribution: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Validate RL training process.
        
        Args:
            episode_rewards: Rewards per episode
            episode_lengths: Length of each episode
            action_distribution: Distribution of actions taken
            
        Returns:
            Dictionary of validation metrics
        """
        results = {
            'model_type': 'ReinforcementLearning',
            'timestamp': datetime.now(),
            'reward_analysis': {},
            'convergence_metrics': {},
            'exploration_metrics': {}
        }
        
        # Reward analysis
        results['reward_analysis'] = self._analyze_rewards(
            episode_rewards, episode_lengths
        )
        
        # Convergence analysis
        results['convergence_metrics'] = self._analyze_convergence_rl(
            episode_rewards
        )
        
        # Exploration analysis
        if action_distribution is not None:
            results['exploration_metrics'] = self._analyze_exploration(
                action_distribution
            )
        
        return results
    
    def validate_policy(self,
                       test_episodes: List[Dict[str, Any]],
                       baseline_performance: Optional[float] = None) -> Dict[str, Any]:
        """
        Validate learned policy on test episodes.
        
        Args:
            test_episodes: List of test episode data
            baseline_performance: Baseline to compare against
            
        Returns:
            Dictionary of policy validation metrics
        """
        results = {
            'performance_metrics': {},
            'risk_metrics': {},
            'consistency_metrics': {}
        }
        
        # Extract episode data
        rewards = [ep['total_reward'] for ep in test_episodes]
        returns = [ep.get('total_return', 0) for ep in test_episodes]
        drawdowns = [ep.get('max_drawdown', 0) for ep in test_episodes]
        
        # Performance metrics
        results['performance_metrics'] = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'sharpe_ratio': np.mean(rewards) / (np.std(rewards) + 1e-10),
            'win_rate': np.mean([r > 0 for r in rewards]),
            'best_episode': np.max(rewards),
            'worst_episode': np.min(rewards)
        }
        
        # Risk metrics
        results['risk_metrics'] = {
            'mean_drawdown': np.mean(drawdowns),
            'max_drawdown': np.max(drawdowns),
            'value_at_risk_95': np.percentile(rewards, 5),
            'conditional_var_95': np.mean([r for r in rewards if r <= np.percentile(rewards, 5)])
        }
        
        # Consistency metrics
        results['consistency_metrics'] = self._analyze_consistency(rewards)
        
        # Compare to baseline
        if baseline_performance is not None:
            results['vs_baseline'] = {
                'improvement': np.mean(rewards) - baseline_performance,
                'relative_improvement': (np.mean(rewards) - baseline_performance) / 
                                      (abs(baseline_performance) + 1e-10),
                'outperformance_rate': np.mean([r > baseline_performance for r in rewards])
            }
        
        return results
    
    def _analyze_rewards(self,
                        episode_rewards: List[float],
                        episode_lengths: List[int]) -> Dict[str, Any]:
        """Analyze episode rewards."""
        rewards_array = np.array(episode_rewards)
        lengths_array = np.array(episode_lengths)
        
        # Moving averages
        window_size = min(100, len(rewards_array) // 10)
        if window_size > 0:
            moving_avg = np.convolve(rewards_array, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
        else:
            moving_avg = rewards_array
        
        return {
            'total_episodes': len(episode_rewards),
            'mean_reward': np.mean(rewards_array),
            'std_reward': np.std(rewards_array),
            'mean_episode_length': np.mean(lengths_array),
            'reward_trend': np.polyfit(range(len(rewards_array)), rewards_array, 1)[0]
                           if len(rewards_array) > 1 else 0,
            'improvement_rate': (moving_avg[-1] - moving_avg[0]) / (moving_avg[0] + 1e-10)
                               if len(moving_avg) > 0 else 0,
            'reward_volatility': np.std(np.diff(rewards_array)) if len(rewards_array) > 1 else 0
        }
    
    def _analyze_convergence_rl(self, episode_rewards: List[float]) -> Dict[str, Any]:
        """Analyze RL convergence."""
        if len(episode_rewards) < 20:
            return {'converged': False, 'convergence_episode': None}
        
        rewards_array = np.array(episode_rewards)
        
        # Check for convergence using moving window
        window_size = 50
        converged = False
        convergence_episode = None
        
        for i in range(window_size, len(rewards_array) - window_size):
            window = rewards_array[i:i+window_size]
            
            # Check if variance is low relative to mean
            if np.std(window) / (abs(np.mean(window)) + 1e-10) < 0.1:
                converged = True
                convergence_episode = i
                break
        
        # Calculate stability metrics
        recent_rewards = rewards_array[-window_size:] if len(rewards_array) >= window_size else rewards_array
        
        return {
            'converged': converged,
            'convergence_episode': convergence_episode,
            'final_performance': np.mean(recent_rewards),
            'performance_stability': 1 - np.std(recent_rewards) / (abs(np.mean(recent_rewards)) + 1e-10),
            'episodes_to_convergence': convergence_episode if convergence_episode else len(rewards_array)
        }
    
    def _analyze_exploration(self, action_distribution: np.ndarray) -> Dict[str, Any]:
        """Analyze exploration behavior."""
        # action_distribution shape: (n_episodes, n_actions)
        
        # Calculate entropy over time
        entropy_per_episode = []
        for episode_actions in action_distribution:
            # Normalize to probabilities
            action_probs = episode_actions / (np.sum(episode_actions) + 1e-10)
            entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
            entropy_per_episode.append(entropy)
        
        entropy_array = np.array(entropy_per_episode)
        
        # Action diversity
        n_actions = action_distribution.shape[1]
        uniform_entropy = -np.log(1/n_actions)
        
        return {
            'mean_entropy': np.mean(entropy_array),
            'entropy_trend': np.polyfit(range(len(entropy_array)), entropy_array, 1)[0]
                           if len(entropy_array) > 1 else 0,
            'exploration_rate': np.mean(entropy_array) / uniform_entropy,
            'action_diversity': len(np.unique(np.argmax(action_distribution, axis=1))) / n_actions,
            'dominant_action_pct': np.max(np.mean(action_distribution, axis=0)) / 
                                  np.sum(np.mean(action_distribution, axis=0))
        }
    
    def _analyze_consistency(self, rewards: List[float]) -> Dict[str, Any]:
        """Analyze performance consistency."""
        rewards_array = np.array(rewards)
        
        # Rolling statistics
        window = min(20, len(rewards) // 5)
        if window > 1:
            rolling_means = [np.mean(rewards_array[i:i+window]) 
                           for i in range(len(rewards_array)-window+1)]
            rolling_stds = [np.std(rewards_array[i:i+window]) 
                          for i in range(len(rewards_array)-window+1)]
        else:
            rolling_means = rewards_array
            rolling_stds = [0] * len(rewards_array)
        
        return {
            'consistency_score': 1 - np.std(rolling_means) / (abs(np.mean(rolling_means)) + 1e-10),
            'volatility_trend': np.polyfit(range(len(rolling_stds)), rolling_stds, 1)[0]
                              if len(rolling_stds) > 1 else 0,
            'positive_episodes_pct': np.mean(rewards_array > 0),
            'consecutive_positives': self._max_consecutive(rewards_array > 0),
            'consecutive_negatives': self._max_consecutive(rewards_array <= 0)
        }
    
    def _max_consecutive(self, boolean_array: np.ndarray) -> int:
        """Find maximum consecutive True values."""
        max_consecutive = 0
        current_consecutive = 0
        
        for value in boolean_array:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive


class EnsembleValidator:
    """
    Validator for ensemble models.
    
    Features:
    - Model diversity analysis
    - Correlation between models
    - Weight optimization validation
    - Ensemble vs individual performance
    """
    
    def __init__(self):
        """Initialize ensemble validator."""
        self.validation_results = {}
        
    def validate_ensemble(self,
                         individual_predictions: Dict[str, np.ndarray],
                         ensemble_predictions: np.ndarray,
                         y_true: np.ndarray,
                         model_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Validate ensemble model.
        
        Args:
            individual_predictions: Predictions from each model
            ensemble_predictions: Combined ensemble predictions
            y_true: True values
            model_weights: Weights for each model
            
        Returns:
            Dictionary of ensemble validation metrics
        """
        results = {
            'model_type': 'Ensemble',
            'timestamp': datetime.now(),
            'diversity_metrics': {},
            'performance_comparison': {},
            'weight_analysis': {},
            'correlation_analysis': {}
        }
        
        # Diversity analysis
        results['diversity_metrics'] = self._analyze_diversity(
            individual_predictions
        )
        
        # Performance comparison
        results['performance_comparison'] = self._compare_performance(
            individual_predictions, ensemble_predictions, y_true
        )
        
        # Weight analysis
        if model_weights:
            results['weight_analysis'] = self._analyze_weights(
                model_weights, individual_predictions, y_true
            )
        
        # Correlation analysis
        results['correlation_analysis'] = self._analyze_correlations(
            individual_predictions
        )
        
        return results
    
    def _analyze_diversity(self, predictions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze prediction diversity among ensemble members."""
        # Convert predictions to matrix
        pred_matrix = np.array(list(predictions.values()))
        n_models, n_samples = pred_matrix.shape
        
        # Pairwise disagreement
        disagreements = []
        for i in range(n_models):
            for j in range(i+1, n_models):
                disagreement = np.mean(pred_matrix[i] != pred_matrix[j])
                disagreements.append(disagreement)
        
        # Variance across models
        prediction_variance = np.var(pred_matrix, axis=0)
        
        return {
            'mean_disagreement': np.mean(disagreements),
            'disagreement_std': np.std(disagreements),
            'mean_prediction_variance': np.mean(prediction_variance),
            'diversity_score': 1 - np.mean([
                np.corrcoef(pred_matrix[i], pred_matrix[j])[0, 1]
                for i in range(n_models)
                for j in range(i+1, n_models)
            ])
        }
    
    def _compare_performance(self,
                           individual_predictions: Dict[str, np.ndarray],
                           ensemble_predictions: np.ndarray,
                           y_true: np.ndarray) -> Dict[str, Any]:
        """Compare ensemble performance to individual models."""
        # Calculate metrics for each model
        individual_metrics = {}
        for model_name, predictions in individual_predictions.items():
            mae = mean_absolute_error(y_true, predictions)
            rmse = np.sqrt(mean_squared_error(y_true, predictions))
            individual_metrics[model_name] = {'mae': mae, 'rmse': rmse}
        
        # Ensemble metrics
        ensemble_mae = mean_absolute_error(y_true, ensemble_predictions)
        ensemble_rmse = np.sqrt(mean_squared_error(y_true, ensemble_predictions))
        
        # Best individual model
        best_individual_mae = min(m['mae'] for m in individual_metrics.values())
        best_individual_rmse = min(m['rmse'] for m in individual_metrics.values())
        
        return {
            'ensemble_mae': ensemble_mae,
            'ensemble_rmse': ensemble_rmse,
            'best_individual_mae': best_individual_mae,
            'best_individual_rmse': best_individual_rmse,
            'ensemble_improvement_mae': (best_individual_mae - ensemble_mae) / best_individual_mae,
            'ensemble_improvement_rmse': (best_individual_rmse - ensemble_rmse) / best_individual_rmse,
            'individual_metrics': individual_metrics,
            'ensemble_beats_all': all(
                ensemble_mae < m['mae'] for m in individual_metrics.values()
            )
        }
    
    def _analyze_weights(self,
                        weights: Dict[str, float],
                        predictions: Dict[str, np.ndarray],
                        y_true: np.ndarray) -> Dict[str, Any]:
        """Analyze model weights in ensemble."""
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate contribution of each model
        contributions = {}
        for model_name, weight in normalized_weights.items():
            if model_name in predictions:
                mae = mean_absolute_error(y_true, predictions[model_name])
                contributions[model_name] = {
                    'weight': weight,
                    'mae': mae,
                    'weight_mae_product': weight * mae
                }
        
        # Weight concentration
        weight_values = list(normalized_weights.values())
        weight_entropy = -np.sum([w * np.log(w + 1e-10) for w in weight_values])
        max_entropy = -np.log(1/len(weight_values))
        
        return {
            'normalized_weights': normalized_weights,
            'weight_concentration': 1 - weight_entropy / max_entropy,
            'highest_weight_model': max(normalized_weights, key=normalized_weights.get),
            'model_contributions': contributions,
            'effective_models': sum(1 for w in weight_values if w > 0.05)
        }
    
    def _analyze_correlations(self, predictions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze correlations between model predictions."""
        model_names = list(predictions.keys())
        n_models = len(model_names)
        
        # Calculate correlation matrix
        pred_matrix = np.array(list(predictions.values()))
        correlation_matrix = np.corrcoef(pred_matrix)
        
        # Extract correlations
        correlations = {}
        for i in range(n_models):
            for j in range(i+1, n_models):
                pair = f"{model_names[i]}_vs_{model_names[j]}"
                correlations[pair] = correlation_matrix[i, j]
        
        return {
            'mean_correlation': np.mean(list(correlations.values())),
            'min_correlation': min(correlations.values()),
            'max_correlation': max(correlations.values()),
            'correlation_pairs': correlations,
            'highly_correlated_pairs': [
                pair for pair, corr in correlations.items() if corr > 0.9
            ]
        }