"""
Comprehensive Model Monitoring System.

Tracks model performance, detects drift, and provides real-time alerts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from datetime import datetime, timedelta
from collections import deque
import json
import os

logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Real-time model monitoring with drift detection and performance tracking.
    
    Features:
    - Performance metric tracking
    - Data drift detection
    - Concept drift detection
    - Automated alerting
    - Performance degradation analysis
    - Model comparison
    """
    
    def __init__(
        self,
        model_name: str,
        metrics_to_track: List[str] = None,
        alert_thresholds: Dict[str, float] = None,
        window_size: int = 1000,
        drift_threshold: float = 0.05,
        performance_threshold: float = 0.1,
        save_dir: str = "monitoring_logs"
    ):
        """
        Initialize model monitor.
        
        Args:
            model_name: Name of the model to monitor
            metrics_to_track: List of metrics to track
            alert_thresholds: Thresholds for alerts
            window_size: Size of monitoring window
            drift_threshold: Threshold for drift detection
            performance_threshold: Threshold for performance degradation
            save_dir: Directory to save monitoring logs
        """
        self.model_name = model_name
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.save_dir = save_dir
        
        # Default metrics
        if metrics_to_track is None:
            metrics_to_track = [
                'accuracy', 'precision', 'recall', 'f1_score',
                'sharpe_ratio', 'max_drawdown', 'win_rate',
                'avg_return', 'volatility'
            ]
        self.metrics_to_track = metrics_to_track
        
        # Default alert thresholds
        if alert_thresholds is None:
            alert_thresholds = {
                'accuracy': 0.5,
                'sharpe_ratio': 1.0,
                'max_drawdown': -0.2,
                'win_rate': 0.45
            }
        self.alert_thresholds = alert_thresholds
        
        # Performance tracking
        self.performance_history = {
            metric: deque(maxlen=window_size) 
            for metric in metrics_to_track
        }
        self.timestamps = deque(maxlen=window_size)
        
        # Drift detection
        self.feature_distributions = {}
        self.prediction_distribution = deque(maxlen=window_size)
        self.drift_scores = deque(maxlen=window_size)
        
        # Alerts
        self.active_alerts = []
        self.alert_history = []
        
        # Statistics
        self.baseline_performance = {}
        self.monitoring_start_time = datetime.now()
        self.total_predictions = 0
        self.failed_predictions = 0
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
    def update_metrics(self, predictions: np.ndarray, 
                      true_values: np.ndarray,
                      features: Optional[np.ndarray] = None,
                      additional_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Update monitoring metrics.
        
        Args:
            predictions: Model predictions
            true_values: Actual values
            features: Input features (for drift detection)
            additional_metrics: Additional custom metrics
        """
        timestamp = datetime.now()
        self.timestamps.append(timestamp)
        self.total_predictions += len(predictions)
        
        # Calculate standard metrics
        metrics = self._calculate_metrics(predictions, true_values)
        
        # Add additional metrics
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Update history
        for metric_name, value in metrics.items():
            if metric_name in self.performance_history:
                self.performance_history[metric_name].append(value)
        
        # Update prediction distribution
        self.prediction_distribution.extend(predictions.flatten())
        
        # Check for drift if features provided
        if features is not None:
            drift_score = self._detect_drift(features, predictions)
            self.drift_scores.append(drift_score)
        
        # Check for alerts
        self._check_alerts(metrics, timestamp)
        
        # Log metrics
        self._log_metrics(metrics, timestamp)
    
    def _calculate_metrics(self, predictions: np.ndarray, 
                          true_values: np.ndarray) -> Dict[str, float]:
        """Calculate standard performance metrics."""
        metrics = {}
        
        # Classification metrics (assuming 3 classes: buy=0, hold=1, sell=2)
        if predictions.dtype in [np.int32, np.int64] or predictions.max() <= 2:
            # Accuracy
            metrics['accuracy'] = np.mean(predictions == true_values)
            
            # Per-class metrics
            for class_idx, class_name in enumerate(['buy', 'hold', 'sell']):
                mask = true_values == class_idx
                if mask.sum() > 0:
                    metrics[f'precision_{class_name}'] = np.mean(
                        predictions[predictions == class_idx] == class_idx
                    ) if (predictions == class_idx).sum() > 0 else 0
                    
                    metrics[f'recall_{class_name}'] = np.mean(
                        predictions[mask] == class_idx
                    )
                    
                    # F1 score
                    precision = metrics[f'precision_{class_name}']
                    recall = metrics[f'recall_{class_name}']
                    metrics[f'f1_{class_name}'] = (
                        2 * precision * recall / (precision + recall + 1e-8)
                    )
        
        # Trading metrics
        if len(predictions) > 1:
            # Calculate returns based on actions
            returns = self._calculate_trading_returns(predictions, true_values)
            
            if len(returns) > 0:
                metrics['avg_return'] = np.mean(returns)
                metrics['volatility'] = np.std(returns)
                metrics['sharpe_ratio'] = (
                    np.sqrt(252) * metrics['avg_return'] / (metrics['volatility'] + 1e-8)
                )
                
                # Win rate
                metrics['win_rate'] = np.mean(returns > 0)
                
                # Max drawdown
                cumulative_returns = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                metrics['max_drawdown'] = np.min(drawdowns)
        
        return metrics
    
    def _calculate_trading_returns(self, predictions: np.ndarray, 
                                 true_values: np.ndarray) -> np.ndarray:
        """Calculate returns from trading actions."""
        # Simplified return calculation
        # Assumes true_values are price changes or returns
        returns = []
        
        for i in range(len(predictions) - 1):
            if predictions[i] == 0:  # Buy
                returns.append(true_values[i+1])
            elif predictions[i] == 2:  # Sell
                returns.append(-true_values[i+1])
            # Hold (1) generates no return
        
        return np.array(returns)
    
    def _detect_drift(self, features: np.ndarray, 
                     predictions: np.ndarray) -> float:
        """
        Detect data and concept drift.
        
        Args:
            features: Current feature batch
            predictions: Current predictions
            
        Returns:
            Drift score (0-1)
        """
        drift_scores = []
        
        # Feature drift detection (Kolmogorov-Smirnov test)
        if hasattr(self, 'baseline_features') and self.baseline_features is not None:
            for i in range(min(features.shape[1], self.baseline_features.shape[1])):
                # KS test statistic
                sorted_current = np.sort(features[:, i])
                sorted_baseline = np.sort(self.baseline_features[:, i])
                
                # Calculate empirical CDFs
                cdf_current = np.arange(1, len(sorted_current) + 1) / len(sorted_current)
                cdf_baseline = np.arange(1, len(sorted_baseline) + 1) / len(sorted_baseline)
                
                # Interpolate to common points
                common_points = np.union1d(sorted_current, sorted_baseline)
                cdf_current_interp = np.interp(common_points, sorted_current, cdf_current)
                cdf_baseline_interp = np.interp(common_points, sorted_baseline, cdf_baseline)
                
                # KS statistic
                ks_stat = np.max(np.abs(cdf_current_interp - cdf_baseline_interp))
                drift_scores.append(ks_stat)
        
        # Prediction drift detection
        if len(self.prediction_distribution) > 100:
            recent_preds = list(self.prediction_distribution)[-100:]
            older_preds = list(self.prediction_distribution)[-1000:-100]
            
            if len(older_preds) > 0:
                # Compare distributions
                recent_dist = np.histogram(recent_preds, bins=20)[0] / len(recent_preds)
                older_dist = np.histogram(older_preds, bins=20)[0] / len(older_preds)
                
                # JS divergence
                m = 0.5 * (recent_dist + older_dist)
                js_div = 0.5 * np.sum(recent_dist * np.log(recent_dist / (m + 1e-8) + 1e-8))
                js_div += 0.5 * np.sum(older_dist * np.log(older_dist / (m + 1e-8) + 1e-8))
                
                drift_scores.append(js_div)
        
        return np.mean(drift_scores) if drift_scores else 0.0
    
    def _check_alerts(self, metrics: Dict[str, float], 
                     timestamp: datetime) -> None:
        """Check for alert conditions."""
        new_alerts = []
        
        # Check metric thresholds
        for metric_name, threshold in self.alert_thresholds.items():
            if metric_name in metrics:
                current_value = metrics[metric_name]
                
                # Different comparison for drawdown (negative is bad)
                if 'drawdown' in metric_name:
                    if current_value < threshold:
                        new_alerts.append({
                            'type': 'threshold_breach',
                            'metric': metric_name,
                            'value': current_value,
                            'threshold': threshold,
                            'timestamp': timestamp,
                            'severity': 'high'
                        })
                else:
                    if current_value < threshold:
                        new_alerts.append({
                            'type': 'threshold_breach',
                            'metric': metric_name,
                            'value': current_value,
                            'threshold': threshold,
                            'timestamp': timestamp,
                            'severity': 'medium'
                        })
        
        # Check for performance degradation
        for metric_name in ['accuracy', 'sharpe_ratio', 'win_rate']:
            if metric_name in self.performance_history:
                history = list(self.performance_history[metric_name])
                if len(history) > 100:
                    recent_avg = np.mean(history[-50:])
                    older_avg = np.mean(history[-200:-50])
                    
                    if older_avg > 0 and (older_avg - recent_avg) / older_avg > self.performance_threshold:
                        new_alerts.append({
                            'type': 'performance_degradation',
                            'metric': metric_name,
                            'recent_avg': recent_avg,
                            'older_avg': older_avg,
                            'degradation': (older_avg - recent_avg) / older_avg,
                            'timestamp': timestamp,
                            'severity': 'high'
                        })
        
        # Check for drift
        if len(self.drift_scores) > 0:
            recent_drift = np.mean(list(self.drift_scores)[-10:])
            if recent_drift > self.drift_threshold:
                new_alerts.append({
                    'type': 'drift_detected',
                    'drift_score': recent_drift,
                    'threshold': self.drift_threshold,
                    'timestamp': timestamp,
                    'severity': 'high'
                })
        
        # Update alerts
        self.active_alerts = new_alerts
        self.alert_history.extend(new_alerts)
        
        # Log alerts
        for alert in new_alerts:
            logger.warning(f"Alert for {self.model_name}: {alert}")
    
    def _log_metrics(self, metrics: Dict[str, float], 
                    timestamp: datetime) -> None:
        """Log metrics to file."""
        log_entry = {
            'model_name': self.model_name,
            'timestamp': timestamp.isoformat(),
            'metrics': metrics,
            'total_predictions': self.total_predictions,
            'active_alerts': len(self.active_alerts)
        }
        
        # Add drift score if available
        if len(self.drift_scores) > 0:
            log_entry['drift_score'] = list(self.drift_scores)[-1]
        
        # Write to log file
        log_file = os.path.join(self.save_dir, f"{self.model_name}_monitor.jsonl")
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def set_baseline(self, features: np.ndarray, 
                    performance_metrics: Dict[str, float]) -> None:
        """
        Set baseline for comparison.
        
        Args:
            features: Baseline feature distribution
            performance_metrics: Baseline performance metrics
        """
        self.baseline_features = features.copy()
        self.baseline_performance = performance_metrics.copy()
        
        logger.info(f"Baseline set for {self.model_name}")
    
    def get_performance_report(self, window: Optional[int] = None) -> Dict[str, Any]:
        """
        Get comprehensive performance report.
        
        Args:
            window: Number of recent observations to analyze
            
        Returns:
            Performance report dictionary
        """
        if window is None:
            window = min(1000, self.window_size)
        
        report = {
            'model_name': self.model_name,
            'monitoring_duration': str(datetime.now() - self.monitoring_start_time),
            'total_predictions': self.total_predictions,
            'failed_predictions': self.failed_predictions,
            'failure_rate': self.failed_predictions / (self.total_predictions + 1e-8)
        }
        
        # Recent performance metrics
        recent_metrics = {}
        for metric_name, history in self.performance_history.items():
            if len(history) > 0:
                recent_values = list(history)[-window:]
                recent_metrics[metric_name] = {
                    'mean': np.mean(recent_values),
                    'std': np.std(recent_values),
                    'min': np.min(recent_values),
                    'max': np.max(recent_values),
                    'trend': self._calculate_trend(recent_values)
                }
        report['recent_performance'] = recent_metrics
        
        # Compare to baseline
        if self.baseline_performance:
            comparison = {}
            for metric_name in self.baseline_performance:
                if metric_name in recent_metrics:
                    baseline_val = self.baseline_performance[metric_name]
                    current_val = recent_metrics[metric_name]['mean']
                    comparison[metric_name] = {
                        'baseline': baseline_val,
                        'current': current_val,
                        'change': (current_val - baseline_val) / (abs(baseline_val) + 1e-8)
                    }
            report['baseline_comparison'] = comparison
        
        # Drift analysis
        if len(self.drift_scores) > 0:
            drift_values = list(self.drift_scores)[-window:]
            report['drift_analysis'] = {
                'mean_drift': np.mean(drift_values),
                'max_drift': np.max(drift_values),
                'drift_trend': self._calculate_trend(drift_values),
                'drift_alerts': sum(1 for d in drift_values if d > self.drift_threshold)
            }
        
        # Alert summary
        recent_alerts = [a for a in self.alert_history 
                        if a['timestamp'] > datetime.now() - timedelta(hours=24)]
        
        alert_summary = {}
        for alert in recent_alerts:
            alert_type = alert['type']
            if alert_type not in alert_summary:
                alert_summary[alert_type] = 0
            alert_summary[alert_type] += 1
        
        report['alert_summary'] = alert_summary
        report['active_alerts'] = self.active_alerts
        
        return report
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 10:
            return 'insufficient_data'
        
        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Normalize by mean
        mean_val = np.mean(values)
        if abs(mean_val) > 1e-8:
            normalized_slope = slope / abs(mean_val)
            
            if normalized_slope > 0.01:
                return 'improving'
            elif normalized_slope < -0.01:
                return 'degrading'
            else:
                return 'stable'
        
        return 'stable'
    
    def export_monitoring_data(self, filepath: str) -> None:
        """Export monitoring data for analysis."""
        data = {
            'model_name': self.model_name,
            'metadata': {
                'start_time': self.monitoring_start_time.isoformat(),
                'export_time': datetime.now().isoformat(),
                'total_predictions': self.total_predictions,
                'window_size': self.window_size
            },
            'performance_history': {
                metric: list(values) 
                for metric, values in self.performance_history.items()
            },
            'timestamps': [ts.isoformat() for ts in self.timestamps],
            'drift_scores': list(self.drift_scores),
            'alert_history': self.alert_history,
            'baseline_performance': self.baseline_performance
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Monitoring data exported to {filepath}")