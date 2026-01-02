"""
Real-Time Performance Monitor for ML Trading Models.

Implements comprehensive performance tracking, statistical validation,
and real-time monitoring of prediction accuracy and model effectiveness.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


@dataclass
class PredictionRecord:
    """Record of a model prediction and its outcome."""
    prediction_id: str
    model_id: str
    timestamp: datetime
    prediction_value: float
    confidence: float
    actual_value: Optional[float] = None
    error: Optional[float] = None
    absolute_error: Optional[float] = None
    squared_error: Optional[float] = None
    prediction_horizon: int = 1  # minutes
    asset_symbol: str = ""
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    validated: bool = False


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    timestamp: datetime
    model_id: str
    time_period: int  # minutes
    
    # Accuracy metrics
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error
    
    # Directional accuracy
    directional_accuracy: float
    hit_rate: float
    
    # Statistical metrics
    r_squared: float
    correlation: float
    sharpe_ratio: float
    
    # Confidence metrics
    confidence_calibration: float
    prediction_interval_coverage: float
    
    # Distribution metrics
    bias: float
    skewness: float
    kurtosis: float
    
    # Model-specific metrics
    information_ratio: float
    tracking_error: float
    max_drawdown: float
    
    # Metadata
    total_predictions: int
    validated_predictions: int
    coverage_ratio: float


@dataclass
class PerformanceAlert:
    """Performance-related alert."""
    timestamp: datetime
    alert_type: str
    severity: str  # low, medium, high, critical
    model_id: str
    metric_name: str
    current_value: float
    threshold_value: float
    description: str
    recommended_actions: List[str]


class PerformanceMonitor:
    """
    Real-time performance monitoring system for ML trading models.
    
    Features:
    - Real-time prediction tracking and validation
    - Comprehensive performance metrics calculation
    - Statistical significance testing
    - Performance degradation detection
    - Model comparison and ranking
    - Confidence calibration analysis
    - Rolling performance windows
    - Alert generation and notifications
    """
    
    def __init__(self,
                 max_predictions: int = 10000,
                 validation_window: int = 60,  # minutes
                 performance_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize performance monitor.
        
        Args:
            max_predictions: Maximum predictions to store
            validation_window: Window for performance calculation (minutes)
            performance_thresholds: Performance alert thresholds
        """
        self.max_predictions = max_predictions
        self.validation_window = validation_window
        
        # Performance thresholds
        self.performance_thresholds = performance_thresholds or {
            'min_accuracy': 0.55,  # 55% directional accuracy
            'max_mae': 0.05,  # 5% mean absolute error
            'min_correlation': 0.3,  # 30% correlation
            'min_sharpe': 0.5,  # 0.5 Sharpe ratio
            'max_tracking_error': 0.1,  # 10% tracking error
            'min_confidence_calibration': 0.7  # 70% calibration
        }
        
        # Prediction storage
        self.predictions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_predictions))
        self.prediction_lookup: Dict[str, PredictionRecord] = {}
        
        # Performance metrics history
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.model_rankings: deque = deque(maxlen=100)
        
        # Alert system
        self.performance_alerts: deque = deque(maxlen=200)
        self.alert_thresholds_breached: Dict[str, datetime] = {}
        
        # Statistical tracking
        self.statistical_tests: Dict[str, List] = defaultdict(list)
        self.benchmark_comparisons: Dict[str, Dict] = defaultdict(dict)
        
        # Real-time state
        self.last_validation_time: Dict[str, datetime] = {}
        self.model_status: Dict[str, str] = {}  # active, degraded, failed
        
        # Performance calculation methods
        self.metric_calculators = {
            'accuracy': self._calculate_accuracy_metrics,
            'statistical': self._calculate_statistical_metrics,
            'financial': self._calculate_financial_metrics,
            'confidence': self._calculate_confidence_metrics
        }
    
    def record_prediction(self,
                         model_id: str,
                         prediction_value: float,
                         confidence: float,
                         asset_symbol: str = "",
                         prediction_horizon: int = 1,
                         market_conditions: Optional[Dict[str, Any]] = None) -> str:
        """
        Record a new model prediction.
        
        Args:
            model_id: Identifier for the model
            prediction_value: Predicted value
            confidence: Confidence score (0-1)
            asset_symbol: Asset being predicted
            prediction_horizon: Prediction horizon in minutes
            market_conditions: Current market conditions
            
        Returns:
            Prediction ID for later validation
        """
        timestamp = datetime.now()
        prediction_id = f"{model_id}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
        
        prediction_record = PredictionRecord(
            prediction_id=prediction_id,
            model_id=model_id,
            timestamp=timestamp,
            prediction_value=prediction_value,
            confidence=confidence,
            asset_symbol=asset_symbol,
            prediction_horizon=prediction_horizon,
            market_conditions=market_conditions or {}
        )
        
        # Store prediction
        self.predictions[model_id].append(prediction_record)
        self.prediction_lookup[prediction_id] = prediction_record
        
        # Update model status
        if model_id not in self.model_status:
            self.model_status[model_id] = 'active'
        
        return prediction_id
    
    def validate_prediction(self,
                          prediction_id: str,
                          actual_value: float) -> bool:
        """
        Validate a prediction with actual outcome.
        
        Args:
            prediction_id: ID of prediction to validate
            actual_value: Actual observed value
            
        Returns:
            True if validation successful
        """
        if prediction_id not in self.prediction_lookup:
            return False
        
        prediction = self.prediction_lookup[prediction_id]
        
        # Calculate errors
        error = prediction.prediction_value - actual_value
        absolute_error = abs(error)
        squared_error = error ** 2
        
        # Update prediction record
        prediction.actual_value = actual_value
        prediction.error = error
        prediction.absolute_error = absolute_error
        prediction.squared_error = squared_error
        prediction.validated = True
        
        # Trigger performance calculation if enough time has passed
        model_id = prediction.model_id
        if (model_id not in self.last_validation_time or
            (datetime.now() - self.last_validation_time[model_id]).total_seconds() > 60):
            
            self._calculate_model_performance(model_id)
            self.last_validation_time[model_id] = datetime.now()
        
        return True
    
    def _calculate_model_performance(self, model_id: str) -> Optional[PerformanceMetrics]:
        """Calculate comprehensive performance metrics for a model."""
        if model_id not in self.predictions:
            return None
        
        # Get validated predictions within time window
        cutoff_time = datetime.now() - timedelta(minutes=self.validation_window)
        validated_predictions = [
            pred for pred in self.predictions[model_id]
            if pred.validated and pred.timestamp >= cutoff_time
        ]
        
        if len(validated_predictions) < 10:  # Minimum predictions required
            return None
        
        timestamp = datetime.now()
        
        # Calculate different metric categories
        accuracy_metrics = self._calculate_accuracy_metrics(validated_predictions)
        statistical_metrics = self._calculate_statistical_metrics(validated_predictions)
        financial_metrics = self._calculate_financial_metrics(validated_predictions)
        confidence_metrics = self._calculate_confidence_metrics(validated_predictions)
        
        # Combine all metrics
        performance_metrics = PerformanceMetrics(
            timestamp=timestamp,
            model_id=model_id,
            time_period=self.validation_window,
            
            # Accuracy metrics
            mae=accuracy_metrics['mae'],
            mse=accuracy_metrics['mse'],
            rmse=accuracy_metrics['rmse'],
            mape=accuracy_metrics['mape'],
            
            # Directional accuracy
            directional_accuracy=accuracy_metrics['directional_accuracy'],
            hit_rate=accuracy_metrics['hit_rate'],
            
            # Statistical metrics
            r_squared=statistical_metrics['r_squared'],
            correlation=statistical_metrics['correlation'],
            sharpe_ratio=statistical_metrics.get('sharpe_ratio', 0.0),
            
            # Confidence metrics
            confidence_calibration=confidence_metrics['calibration'],
            prediction_interval_coverage=confidence_metrics['interval_coverage'],
            
            # Distribution metrics
            bias=statistical_metrics['bias'],
            skewness=statistical_metrics['skewness'],
            kurtosis=statistical_metrics['kurtosis'],
            
            # Financial metrics
            information_ratio=financial_metrics.get('information_ratio', 0.0),
            tracking_error=financial_metrics.get('tracking_error', 0.0),
            max_drawdown=financial_metrics.get('max_drawdown', 0.0),
            
            # Coverage
            total_predictions=len(self.predictions[model_id]),
            validated_predictions=len(validated_predictions),
            coverage_ratio=len(validated_predictions) / len(self.predictions[model_id])
        )
        
        # Store performance metrics
        self.performance_history[model_id].append(performance_metrics)
        
        # Check for alert conditions
        self._check_performance_alerts(performance_metrics)
        
        return performance_metrics
    
    def _calculate_accuracy_metrics(self, predictions: List[PredictionRecord]) -> Dict[str, float]:
        """Calculate accuracy-based metrics."""
        if not predictions:
            return {}
        
        # Extract values
        predicted_values = np.array([p.prediction_value for p in predictions])
        actual_values = np.array([p.actual_value for p in predictions])
        errors = np.array([p.error for p in predictions])
        absolute_errors = np.array([p.absolute_error for p in predictions])
        
        # Basic accuracy metrics
        mae = np.mean(absolute_errors)
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        
        # MAPE (handle division by zero)
        actual_nonzero = actual_values[actual_values != 0]
        pred_nonzero = predicted_values[actual_values != 0]
        if len(actual_nonzero) > 0:
            mape = np.mean(np.abs((actual_nonzero - pred_nonzero) / actual_nonzero))
        else:
            mape = np.inf
        
        # Directional accuracy
        pred_directions = np.sign(predicted_values)
        actual_directions = np.sign(actual_values)
        directional_accuracy = np.mean(pred_directions == actual_directions)
        
        # Hit rate (predictions within certain threshold)
        threshold = 0.02  # 2% threshold
        hits = np.sum(absolute_errors <= threshold)
        hit_rate = hits / len(predictions)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'hit_rate': hit_rate
        }
    
    def _calculate_statistical_metrics(self, predictions: List[PredictionRecord]) -> Dict[str, float]:
        """Calculate statistical metrics."""
        if not predictions:
            return {}
        
        predicted_values = np.array([p.prediction_value for p in predictions])
        actual_values = np.array([p.actual_value for p in predictions])
        errors = np.array([p.error for p in predictions])
        
        # Correlation and R-squared
        if np.std(predicted_values) > 0 and np.std(actual_values) > 0:
            correlation = np.corrcoef(predicted_values, actual_values)[0, 1]
            r_squared = correlation ** 2
        else:
            correlation = 0.0
            r_squared = 0.0
        
        # Bias
        bias = np.mean(errors)
        
        # Distribution statistics
        skewness = self._calculate_skewness(errors)
        kurtosis = self._calculate_kurtosis(errors)
        
        # Calculate Sharpe-like ratio for predictions
        if np.std(errors) > 0:
            error_sharpe = -np.mean(errors) / np.std(errors)  # Lower error is better
        else:
            error_sharpe = 0.0
        
        return {
            'correlation': correlation,
            'r_squared': r_squared,
            'bias': bias,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'sharpe_ratio': error_sharpe
        }
    
    def _calculate_financial_metrics(self, predictions: List[PredictionRecord]) -> Dict[str, float]:
        """Calculate financial performance metrics."""
        if not predictions:
            return {}
        
        predicted_values = np.array([p.prediction_value for p in predictions])
        actual_values = np.array([p.actual_value for p in predictions])
        
        # Information ratio (prediction accuracy relative to volatility)
        prediction_errors = predicted_values - actual_values
        if np.std(prediction_errors) > 0:
            information_ratio = -np.mean(prediction_errors) / np.std(prediction_errors)
        else:
            information_ratio = 0.0
        
        # Tracking error
        tracking_error = np.std(prediction_errors)
        
        # Calculate maximum drawdown of prediction errors
        cumulative_errors = np.cumsum(prediction_errors)
        running_max = np.maximum.accumulate(cumulative_errors)
        drawdown = (cumulative_errors - running_max)
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        # Prediction profit/loss simulation
        prediction_returns = np.sign(predicted_values) * actual_values
        if np.std(prediction_returns) > 0:
            prediction_sharpe = np.mean(prediction_returns) / np.std(prediction_returns)
        else:
            prediction_sharpe = 0.0
        
        return {
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'max_drawdown': abs(max_drawdown),
            'prediction_sharpe': prediction_sharpe
        }
    
    def _calculate_confidence_metrics(self, predictions: List[PredictionRecord]) -> Dict[str, float]:
        """Calculate confidence and calibration metrics."""
        if not predictions:
            return {}
        
        confidences = np.array([p.confidence for p in predictions])
        absolute_errors = np.array([p.absolute_error for p in predictions])
        
        # Confidence calibration (correlation between confidence and accuracy)
        accuracy_scores = 1.0 - (absolute_errors / (np.mean(absolute_errors) + 1e-8))
        
        if np.std(confidences) > 0 and np.std(accuracy_scores) > 0:
            calibration = np.corrcoef(confidences, accuracy_scores)[0, 1]
        else:
            calibration = 0.0
        
        # Prediction interval coverage (simplified)
        # Assume prediction intervals based on confidence
        high_confidence_predictions = absolute_errors[confidences > 0.8]
        if len(high_confidence_predictions) > 0:
            interval_coverage = np.mean(high_confidence_predictions < np.median(absolute_errors))
        else:
            interval_coverage = 0.5
        
        # Overconfidence detection
        overconfidence_ratio = np.mean(confidences) - np.mean(accuracy_scores)
        
        return {
            'calibration': max(-1.0, min(1.0, calibration)),
            'interval_coverage': interval_coverage,
            'overconfidence_ratio': overconfidence_ratio
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) == 0:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return 0.0
        
        skewness = np.mean(((data - mean_val) / std_val) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if len(data) == 0:
            return 3.0  # Normal distribution kurtosis
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return 3.0
        
        kurtosis = np.mean(((data - mean_val) / std_val) ** 4)
        return kurtosis
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check for performance alert conditions."""
        alerts = []
        
        # Check directional accuracy
        if metrics.directional_accuracy < self.performance_thresholds['min_accuracy']:
            alerts.append(PerformanceAlert(
                timestamp=datetime.now(),
                alert_type="accuracy_degradation",
                severity="high",
                model_id=metrics.model_id,
                metric_name="directional_accuracy",
                current_value=metrics.directional_accuracy,
                threshold_value=self.performance_thresholds['min_accuracy'],
                description=f"Directional accuracy {metrics.directional_accuracy:.2%} below threshold",
                recommended_actions=["Retrain model", "Review feature engineering", "Check data quality"]
            ))
        
        # Check MAE
        if metrics.mae > self.performance_thresholds['max_mae']:
            alerts.append(PerformanceAlert(
                timestamp=datetime.now(),
                alert_type="error_increase",
                severity="medium",
                model_id=metrics.model_id,
                metric_name="mae",
                current_value=metrics.mae,
                threshold_value=self.performance_thresholds['max_mae'],
                description=f"Mean Absolute Error {metrics.mae:.4f} exceeds threshold",
                recommended_actions=["Review model parameters", "Check for regime changes"]
            ))
        
        # Check correlation
        if metrics.correlation < self.performance_thresholds['min_correlation']:
            alerts.append(PerformanceAlert(
                timestamp=datetime.now(),
                alert_type="correlation_drop",
                severity="high",
                model_id=metrics.model_id,
                metric_name="correlation",
                current_value=metrics.correlation,
                threshold_value=self.performance_thresholds['min_correlation'],
                description=f"Correlation {metrics.correlation:.3f} below threshold",
                recommended_actions=["Emergency model review", "Consider model replacement"]
            ))
        
        # Check confidence calibration
        if metrics.confidence_calibration < self.performance_thresholds['min_confidence_calibration']:
            alerts.append(PerformanceAlert(
                timestamp=datetime.now(),
                alert_type="calibration_issue",
                severity="medium",
                model_id=metrics.model_id,
                metric_name="confidence_calibration",
                current_value=metrics.confidence_calibration,
                threshold_value=self.performance_thresholds['min_confidence_calibration'],
                description=f"Confidence calibration {metrics.confidence_calibration:.3f} poor",
                recommended_actions=["Recalibrate confidence scores", "Review uncertainty estimation"]
            ))
        
        # Store alerts
        for alert in alerts:
            self.performance_alerts.append(alert)
            
            # Update model status based on severity
            if alert.severity in ['high', 'critical']:
                self.model_status[metrics.model_id] = 'degraded'
    
    def get_model_performance(self, model_id: str) -> Optional[PerformanceMetrics]:
        """Get latest performance metrics for a model."""
        if model_id not in self.performance_history:
            return None
        
        if not self.performance_history[model_id]:
            return None
        
        return list(self.performance_history[model_id])[-1]
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare performance across multiple models."""
        comparison = {
            'timestamp': datetime.now(),
            'models_compared': model_ids,
            'metrics_comparison': {},
            'rankings': {}
        }
        
        # Get latest metrics for each model
        model_metrics = {}
        for model_id in model_ids:
            latest_metrics = self.get_model_performance(model_id)
            if latest_metrics:
                model_metrics[model_id] = latest_metrics
        
        if not model_metrics:
            return comparison
        
        # Compare key metrics
        key_metrics = ['directional_accuracy', 'mae', 'correlation', 'sharpe_ratio']
        
        for metric in key_metrics:
            metric_values = {}
            for model_id, metrics in model_metrics.items():
                metric_values[model_id] = getattr(metrics, metric, 0.0)
            
            comparison['metrics_comparison'][metric] = metric_values
            
            # Rank models for this metric (higher is better for most metrics)
            if metric == 'mae':  # Lower is better for MAE
                ranked = sorted(metric_values.items(), key=lambda x: x[1])
            else:  # Higher is better
                ranked = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
            
            comparison['rankings'][metric] = [model_id for model_id, _ in ranked]
        
        # Overall ranking (simple average of ranks)
        model_scores = defaultdict(float)
        for metric_rankings in comparison['rankings'].values():
            for i, model_id in enumerate(metric_rankings):
                model_scores[model_id] += (len(metric_rankings) - i)
        
        overall_ranking = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        comparison['overall_ranking'] = [model_id for model_id, _ in overall_ranking]
        
        return comparison
    
    def get_performance_summary(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for model(s)."""
        summary = {
            'timestamp': datetime.now(),
            'monitoring_window_minutes': self.validation_window
        }
        
        if model_id:
            # Single model summary
            if model_id in self.performance_history and self.performance_history[model_id]:
                latest_metrics = list(self.performance_history[model_id])[-1]
                
                summary['model_id'] = model_id
                summary['status'] = self.model_status.get(model_id, 'unknown')
                summary['latest_metrics'] = {
                    'directional_accuracy': latest_metrics.directional_accuracy,
                    'mae': latest_metrics.mae,
                    'correlation': latest_metrics.correlation,
                    'sharpe_ratio': latest_metrics.sharpe_ratio,
                    'confidence_calibration': latest_metrics.confidence_calibration
                }
                summary['total_predictions'] = latest_metrics.total_predictions
                summary['validated_predictions'] = latest_metrics.validated_predictions
                
                # Recent alerts
                model_alerts = [alert for alert in self.performance_alerts 
                              if alert.model_id == model_id]
                summary['recent_alerts'] = len(model_alerts)
            else:
                summary['status'] = 'no_data'
        
        else:
            # All models summary
            all_models = list(self.model_status.keys())
            summary['total_models'] = len(all_models)
            summary['model_statuses'] = dict(self.model_status)
            
            # Active models with recent performance
            active_models = []
            for mid in all_models:
                if (mid in self.performance_history and 
                    self.performance_history[mid] and
                    self.model_status.get(mid) == 'active'):
                    active_models.append(mid)
            
            summary['active_models'] = len(active_models)
            summary['total_alerts'] = len(self.performance_alerts)
            
            # Best performing model
            if active_models:
                comparison = self.compare_models(active_models)
                if comparison.get('overall_ranking'):
                    summary['best_model'] = comparison['overall_ranking'][0]
        
        return summary
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent performance alerts."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.performance_alerts
            if alert.timestamp >= cutoff_time
        ]
        
        return [
            {
                'timestamp': alert.timestamp.isoformat(),
                'type': alert.alert_type,
                'severity': alert.severity,
                'model_id': alert.model_id,
                'metric': alert.metric_name,
                'current_value': alert.current_value,
                'threshold': alert.threshold_value,
                'description': alert.description,
                'actions': alert.recommended_actions
            }
            for alert in recent_alerts
        ]
    
    def run_statistical_tests(self, model_id: str) -> Dict[str, Any]:
        """Run statistical significance tests on model performance."""
        if model_id not in self.predictions:
            return {'error': 'No predictions found for model'}
        
        # Get validated predictions
        validated_predictions = [
            pred for pred in self.predictions[model_id]
            if pred.validated
        ]
        
        if len(validated_predictions) < 30:  # Minimum for statistical tests
            return {'error': 'Insufficient data for statistical tests'}
        
        errors = np.array([pred.error for pred in validated_predictions])
        
        # Test for bias (t-test against zero)
        from scipy import stats
        t_stat, p_value_bias = stats.ttest_1samp(errors, 0.0)
        
        # Test for normality of errors
        _, p_value_normality = stats.shapiro(errors[:5000])  # Limit for shapiro test
        
        # Test for autocorrelation in errors
        if len(errors) > 1:
            autocorr = np.corrcoef(errors[:-1], errors[1:])[0, 1]
        else:
            autocorr = 0.0
        
        # Test for heteroscedasticity (Breusch-Pagan test approximation)
        predicted_values = np.array([pred.prediction_value for pred in validated_predictions])
        if np.std(predicted_values) > 0:
            heteroscedasticity = np.corrcoef(predicted_values, np.abs(errors))[0, 1]
        else:
            heteroscedasticity = 0.0
        
        return {
            'model_id': model_id,
            'sample_size': len(validated_predictions),
            'bias_test': {
                't_statistic': t_stat,
                'p_value': p_value_bias,
                'significant_bias': p_value_bias < 0.05
            },
            'normality_test': {
                'p_value': p_value_normality,
                'errors_normal': p_value_normality > 0.05
            },
            'autocorrelation': {
                'correlation': autocorr,
                'significant_autocorr': abs(autocorr) > 0.1
            },
            'heteroscedasticity': {
                'correlation': heteroscedasticity,
                'significant_hetero': abs(heteroscedasticity) > 0.1
            }
        }
    
    def set_performance_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Update performance alert thresholds."""
        self.performance_thresholds.update(thresholds)
        print(f"Updated performance thresholds: {thresholds}")
    
    def reset_model_status(self, model_id: str) -> None:
        """Reset model status to active."""
        if model_id in self.model_status:
            self.model_status[model_id] = 'active'
            print(f"Reset status for model {model_id} to active")