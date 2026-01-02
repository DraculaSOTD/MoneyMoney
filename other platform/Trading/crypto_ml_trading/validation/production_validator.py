"""
Production Validation and Monitoring for ML Trading Models.

Provides continuous validation, drift detection, and performance monitoring
for models deployed in production.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from scipy import stats
import logging


@dataclass
class ModelPerformanceSnapshot:
    """Snapshot of model performance at a point in time."""
    timestamp: datetime
    model_id: str
    predictions: int
    accuracy: float
    mae: float
    sharpe_ratio: float
    max_drawdown: float
    drift_score: float
    alerts: List[str] = field(default_factory=list)


@dataclass
class DriftAlert:
    """Alert for detected drift."""
    timestamp: datetime
    drift_type: str  # 'feature', 'prediction', 'performance'
    severity: str    # 'low', 'medium', 'high', 'critical'
    description: str
    metrics: Dict[str, float]
    recommended_actions: List[str]


class ProductionValidator:
    """
    Production validation and monitoring system.
    
    Features:
    - Real-time performance tracking
    - Data drift detection
    - Concept drift detection
    - Model performance degradation alerts
    - A/B testing framework
    - Automated retraining triggers
    """
    
    def __init__(self,
                 monitoring_window: int = 1000,
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 log_dir: str = "production_logs"):
        """
        Initialize production validator.
        
        Args:
            monitoring_window: Size of monitoring window (number of predictions)
            alert_thresholds: Custom alert thresholds
            log_dir: Directory for production logs
        """
        self.monitoring_window = monitoring_window
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'accuracy_drop': 0.05,      # 5% accuracy drop
            'mae_increase': 0.10,       # 10% MAE increase
            'drift_score': 0.3,         # Drift score threshold
            'prediction_shift': 0.2,    # Distribution shift threshold
            'feature_drift': 0.25,      # Feature drift threshold
            'performance_variance': 2.0  # Standard deviations
        }
        
        # Performance tracking
        self.performance_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=monitoring_window)
        )
        self.prediction_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=monitoring_window)
        )
        self.feature_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=monitoring_window)
        )
        
        # Baseline statistics
        self.baseline_stats: Dict[str, Dict[str, Any]] = {}
        self.current_stats: Dict[str, Dict[str, Any]] = {}
        
        # Alerts
        self.active_alerts: Dict[str, List[DriftAlert]] = defaultdict(list)
        self.alert_history: deque = deque(maxlen=1000)
        
        # A/B testing
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.log_dir / "production_validation.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def set_baseline(self,
                    model_id: str,
                    baseline_predictions: np.ndarray,
                    baseline_features: np.ndarray,
                    baseline_targets: Optional[np.ndarray] = None):
        """
        Set baseline statistics for drift detection.
        
        Args:
            model_id: Model identifier
            baseline_predictions: Baseline model predictions
            baseline_features: Baseline feature data
            baseline_targets: Baseline true values (if available)
        """
        self.logger.info(f"Setting baseline for model {model_id}")
        
        # Calculate baseline statistics
        baseline_stats = {
            'prediction_stats': self._calculate_distribution_stats(baseline_predictions),
            'feature_stats': self._calculate_feature_stats(baseline_features),
            'timestamp': datetime.now()
        }
        
        # If targets available, calculate performance baseline
        if baseline_targets is not None:
            baseline_stats['performance_stats'] = self._calculate_performance_stats(
                baseline_targets, baseline_predictions
            )
        
        self.baseline_stats[model_id] = baseline_stats
        
        # Initialize current stats
        self.current_stats[model_id] = {
            'n_predictions': 0,
            'last_update': datetime.now()
        }
        
    def validate_prediction(self,
                          model_id: str,
                          features: np.ndarray,
                          prediction: float,
                          true_value: Optional[float] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate a single prediction in production.
        
        Args:
            model_id: Model identifier
            features: Input features
            prediction: Model prediction
            true_value: True value (if available)
            metadata: Additional metadata
            
        Returns:
            Validation results and any alerts
        """
        # Store prediction
        self.prediction_history[model_id].append({
            'timestamp': datetime.now(),
            'prediction': prediction,
            'features': features,
            'true_value': true_value,
            'metadata': metadata or {}
        })
        
        # Update feature history
        self.feature_history[model_id].append(features)
        
        # Increment prediction count
        if model_id in self.current_stats:
            self.current_stats[model_id]['n_predictions'] += 1
        
        # Check for drift every N predictions
        check_interval = max(10, self.monitoring_window // 10)
        if self.current_stats[model_id]['n_predictions'] % check_interval == 0:
            drift_results = self._check_all_drifts(model_id)
            
            if drift_results['has_drift']:
                self._handle_drift_detection(model_id, drift_results)
            
            return {
                'validated': True,
                'drift_detected': drift_results['has_drift'],
                'drift_details': drift_results,
                'alerts': self.active_alerts.get(model_id, [])
            }
        
        return {
            'validated': True,
            'drift_detected': False,
            'alerts': []
        }
        
    def _check_all_drifts(self, model_id: str) -> Dict[str, Any]:
        """Check for all types of drift."""
        results = {
            'has_drift': False,
            'feature_drift': None,
            'prediction_drift': None,
            'performance_drift': None
        }
        
        if model_id not in self.baseline_stats:
            return results
        
        # Feature drift
        if self.feature_history[model_id]:
            results['feature_drift'] = self._check_feature_drift(model_id)
            if results['feature_drift']['is_drifted']:
                results['has_drift'] = True
        
        # Prediction drift
        if self.prediction_history[model_id]:
            results['prediction_drift'] = self._check_prediction_drift(model_id)
            if results['prediction_drift']['is_drifted']:
                results['has_drift'] = True
        
        # Performance drift (if true values available)
        results['performance_drift'] = self._check_performance_drift(model_id)
        if results['performance_drift'] and results['performance_drift']['is_drifted']:
            results['has_drift'] = True
        
        return results
        
    def _check_feature_drift(self, model_id: str) -> Dict[str, Any]:
        """Check for feature distribution drift."""
        baseline_stats = self.baseline_stats[model_id]['feature_stats']
        
        # Get recent features
        recent_features = np.array(list(self.feature_history[model_id]))
        if len(recent_features) < 10:
            return {'is_drifted': False, 'message': 'Insufficient data'}
        
        # Calculate current statistics
        current_stats = self._calculate_feature_stats(recent_features)
        
        # Compare distributions using multiple methods
        drift_scores = {}
        
        # 1. Kolmogorov-Smirnov test for each feature
        ks_scores = []
        for i in range(recent_features.shape[1]):
            if f'feature_{i}_values' in baseline_stats:
                baseline_vals = baseline_stats[f'feature_{i}_values']
                current_vals = recent_features[:, i]
                ks_stat, p_value = stats.ks_2samp(baseline_vals, current_vals)
                ks_scores.append(ks_stat)
                drift_scores[f'feature_{i}_ks'] = ks_stat
        
        # 2. Mean shift detection
        mean_shifts = []
        for i in range(recent_features.shape[1]):
            baseline_mean = baseline_stats.get(f'feature_{i}_mean', 0)
            baseline_std = baseline_stats.get(f'feature_{i}_std', 1)
            current_mean = current_stats.get(f'feature_{i}_mean', 0)
            
            if baseline_std > 0:
                z_score = abs(current_mean - baseline_mean) / baseline_std
                mean_shifts.append(z_score)
                drift_scores[f'feature_{i}_mean_shift'] = z_score
        
        # Overall drift score
        overall_drift_score = np.mean(ks_scores) if ks_scores else 0
        max_mean_shift = max(mean_shifts) if mean_shifts else 0
        
        is_drifted = (overall_drift_score > self.alert_thresholds['feature_drift'] or
                     max_mean_shift > self.alert_thresholds['performance_variance'])
        
        return {
            'is_drifted': is_drifted,
            'overall_drift_score': overall_drift_score,
            'max_mean_shift': max_mean_shift,
            'feature_scores': drift_scores,
            'message': 'Feature distribution drift detected' if is_drifted else 'No drift'
        }
        
    def _check_prediction_drift(self, model_id: str) -> Dict[str, Any]:
        """Check for prediction distribution drift."""
        baseline_stats = self.baseline_stats[model_id]['prediction_stats']
        
        # Get recent predictions
        recent_predictions = [
            p['prediction'] for p in self.prediction_history[model_id]
        ]
        
        if len(recent_predictions) < 10:
            return {'is_drifted': False, 'message': 'Insufficient data'}
        
        recent_predictions = np.array(recent_predictions)
        
        # Calculate current statistics
        current_stats = self._calculate_distribution_stats(recent_predictions)
        
        # KS test
        if 'values' in baseline_stats:
            ks_stat, p_value = stats.ks_2samp(
                baseline_stats['values'], 
                recent_predictions
            )
        else:
            ks_stat, p_value = 0, 1
        
        # Mean shift
        mean_shift = abs(current_stats['mean'] - baseline_stats['mean']) / (baseline_stats['std'] + 1e-10)
        
        # Distribution shape changes
        skew_change = abs(current_stats['skewness'] - baseline_stats['skewness'])
        kurtosis_change = abs(current_stats['kurtosis'] - baseline_stats['kurtosis'])
        
        is_drifted = (ks_stat > self.alert_thresholds['prediction_shift'] or
                     mean_shift > self.alert_thresholds['performance_variance'])
        
        return {
            'is_drifted': is_drifted,
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'mean_shift': mean_shift,
            'skew_change': skew_change,
            'kurtosis_change': kurtosis_change,
            'message': 'Prediction distribution drift detected' if is_drifted else 'No drift'
        }
        
    def _check_performance_drift(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Check for performance degradation."""
        # Get predictions with true values
        recent_results = [
            p for p in self.prediction_history[model_id]
            if p['true_value'] is not None
        ]
        
        if len(recent_results) < 10:
            return None
        
        predictions = np.array([r['prediction'] for r in recent_results])
        true_values = np.array([r['true_value'] for r in recent_results])
        
        # Calculate current performance
        current_performance = self._calculate_performance_stats(true_values, predictions)
        
        # Compare with baseline if available
        if 'performance_stats' not in self.baseline_stats[model_id]:
            return None
        
        baseline_performance = self.baseline_stats[model_id]['performance_stats']
        
        # Check for performance degradation
        accuracy_drop = baseline_performance['accuracy'] - current_performance['accuracy']
        mae_increase = (current_performance['mae'] - baseline_performance['mae']) / (baseline_performance['mae'] + 1e-10)
        
        is_drifted = (accuracy_drop > self.alert_thresholds['accuracy_drop'] or
                     mae_increase > self.alert_thresholds['mae_increase'])
        
        return {
            'is_drifted': is_drifted,
            'accuracy_drop': accuracy_drop,
            'mae_increase': mae_increase,
            'current_performance': current_performance,
            'baseline_performance': baseline_performance,
            'message': 'Performance degradation detected' if is_drifted else 'Performance stable'
        }
        
    def _handle_drift_detection(self, model_id: str, drift_results: Dict[str, Any]):
        """Handle detected drift by creating alerts and taking actions."""
        timestamp = datetime.now()
        
        # Create alerts for each type of drift
        if drift_results['feature_drift'] and drift_results['feature_drift']['is_drifted']:
            alert = DriftAlert(
                timestamp=timestamp,
                drift_type='feature',
                severity=self._determine_severity(drift_results['feature_drift']),
                description='Feature distribution has drifted significantly',
                metrics={
                    'drift_score': drift_results['feature_drift']['overall_drift_score'],
                    'max_mean_shift': drift_results['feature_drift']['max_mean_shift']
                },
                recommended_actions=[
                    'Review recent data quality',
                    'Check for upstream data pipeline issues',
                    'Consider model retraining with recent data'
                ]
            )
            self._add_alert(model_id, alert)
        
        if drift_results['prediction_drift'] and drift_results['prediction_drift']['is_drifted']:
            alert = DriftAlert(
                timestamp=timestamp,
                drift_type='prediction',
                severity=self._determine_severity(drift_results['prediction_drift']),
                description='Model predictions have shifted significantly',
                metrics={
                    'ks_statistic': drift_results['prediction_drift']['ks_statistic'],
                    'mean_shift': drift_results['prediction_drift']['mean_shift']
                },
                recommended_actions=[
                    'Investigate model behavior on recent data',
                    'Check for concept drift',
                    'Evaluate model recalibration'
                ]
            )
            self._add_alert(model_id, alert)
        
        if drift_results['performance_drift'] and drift_results['performance_drift']['is_drifted']:
            alert = DriftAlert(
                timestamp=timestamp,
                drift_type='performance',
                severity='high',  # Performance degradation is always high priority
                description='Model performance has degraded significantly',
                metrics={
                    'accuracy_drop': drift_results['performance_drift']['accuracy_drop'],
                    'mae_increase': drift_results['performance_drift']['mae_increase']
                },
                recommended_actions=[
                    'Immediate model review required',
                    'Consider reverting to previous model version',
                    'Initiate emergency retraining pipeline'
                ]
            )
            self._add_alert(model_id, alert)
            
    def _add_alert(self, model_id: str, alert: DriftAlert):
        """Add alert to active alerts and history."""
        self.active_alerts[model_id].append(alert)
        self.alert_history.append((model_id, alert))
        
        # Log alert
        self.logger.warning(
            f"Drift alert for model {model_id}: {alert.drift_type} drift detected "
            f"with {alert.severity} severity. {alert.description}"
        )
        
        # Save alert to file
        alert_file = self.log_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.json"
        alert_data = {
            'timestamp': alert.timestamp.isoformat(),
            'model_id': model_id,
            'drift_type': alert.drift_type,
            'severity': alert.severity,
            'description': alert.description,
            'metrics': alert.metrics,
            'recommended_actions': alert.recommended_actions
        }
        
        # Append to alerts file
        alerts = []
        if alert_file.exists():
            with open(alert_file, 'r') as f:
                alerts = json.load(f)
        
        alerts.append(alert_data)
        
        with open(alert_file, 'w') as f:
            json.dump(alerts, f, indent=2)
            
    def _determine_severity(self, drift_result: Dict[str, Any]) -> str:
        """Determine alert severity based on drift metrics."""
        if 'overall_drift_score' in drift_result:
            score = drift_result['overall_drift_score']
            if score > 0.5:
                return 'critical'
            elif score > 0.3:
                return 'high'
            elif score > 0.2:
                return 'medium'
            else:
                return 'low'
        
        if 'mean_shift' in drift_result:
            shift = drift_result['mean_shift']
            if shift > 3:
                return 'critical'
            elif shift > 2:
                return 'high'
            elif shift > 1:
                return 'medium'
            else:
                return 'low'
        
        return 'medium'
        
    def start_ab_test(self,
                     test_id: str,
                     model_a_id: str,
                     model_b_id: str,
                     traffic_split: float = 0.5,
                     min_samples: int = 100,
                     max_duration_hours: int = 168):  # 1 week
        """
        Start an A/B test between two models.
        
        Args:
            test_id: Unique test identifier
            model_a_id: ID of model A (control)
            model_b_id: ID of model B (treatment)
            traffic_split: Fraction of traffic to model B
            min_samples: Minimum samples before evaluation
            max_duration_hours: Maximum test duration
        """
        self.ab_tests[test_id] = {
            'model_a_id': model_a_id,
            'model_b_id': model_b_id,
            'traffic_split': traffic_split,
            'min_samples': min_samples,
            'max_duration_hours': max_duration_hours,
            'start_time': datetime.now(),
            'status': 'active',
            'results_a': [],
            'results_b': [],
            'metrics': {}
        }
        
        self.logger.info(
            f"Started A/B test {test_id}: {model_a_id} vs {model_b_id} "
            f"with {traffic_split:.0%} traffic to model B"
        )
        
    def record_ab_result(self,
                        test_id: str,
                        model_id: str,
                        prediction: float,
                        true_value: float,
                        metadata: Optional[Dict[str, Any]] = None):
        """Record result for A/B test."""
        if test_id not in self.ab_tests:
            return
        
        test = self.ab_tests[test_id]
        
        if test['status'] != 'active':
            return
        
        result = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'true_value': true_value,
            'error': abs(prediction - true_value),
            'metadata': metadata or {}
        }
        
        if model_id == test['model_a_id']:
            test['results_a'].append(result)
        elif model_id == test['model_b_id']:
            test['results_b'].append(result)
        
        # Check if test should be evaluated
        if (len(test['results_a']) + len(test['results_b']) >= test['min_samples'] or
            (datetime.now() - test['start_time']).total_seconds() > test['max_duration_hours'] * 3600):
            self._evaluate_ab_test(test_id)
            
    def _evaluate_ab_test(self, test_id: str):
        """Evaluate A/B test results."""
        test = self.ab_tests[test_id]
        
        if len(test['results_a']) < 10 or len(test['results_b']) < 10:
            test['status'] = 'insufficient_data'
            return
        
        # Extract metrics
        errors_a = np.array([r['error'] for r in test['results_a']])
        errors_b = np.array([r['error'] for r in test['results_b']])
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(errors_a, errors_b, equal_var=False)
        
        # Calculate performance metrics
        mae_a = np.mean(errors_a)
        mae_b = np.mean(errors_b)
        
        # Effect size
        pooled_std = np.sqrt((np.var(errors_a) + np.var(errors_b)) / 2)
        effect_size = (mae_a - mae_b) / pooled_std if pooled_std > 0 else 0
        
        # Determine winner
        if p_value < 0.05:
            if mae_a < mae_b:
                winner = test['model_a_id']
                improvement = (mae_b - mae_a) / mae_b
            else:
                winner = test['model_b_id']
                improvement = (mae_a - mae_b) / mae_a
        else:
            winner = 'no_significant_difference'
            improvement = 0
        
        test['status'] = 'completed'
        test['metrics'] = {
            'mae_a': mae_a,
            'mae_b': mae_b,
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'winner': winner,
            'improvement': improvement,
            'sample_size_a': len(test['results_a']),
            'sample_size_b': len(test['results_b'])
        }
        
        self.logger.info(
            f"A/B test {test_id} completed. Winner: {winner} "
            f"with {improvement:.1%} improvement (p={p_value:.4f})"
        )
        
    def get_model_health_report(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive health report for a model."""
        report = {
            'model_id': model_id,
            'timestamp': datetime.now(),
            'status': 'healthy',  # Will be updated based on checks
            'metrics': {},
            'alerts': [],
            'recommendations': []
        }
        
        # Check if model is being monitored
        if model_id not in self.current_stats:
            report['status'] = 'not_monitored'
            return report
        
        # Get recent performance
        recent_results = [
            p for p in self.prediction_history[model_id]
            if p['true_value'] is not None
        ]
        
        if recent_results:
            predictions = np.array([r['prediction'] for r in recent_results[-100:]])
            true_values = np.array([r['true_value'] for r in recent_results[-100:]])
            
            current_performance = self._calculate_performance_stats(true_values, predictions)
            report['metrics'] = current_performance
        
        # Active alerts
        if model_id in self.active_alerts:
            report['alerts'] = [
                {
                    'type': alert.drift_type,
                    'severity': alert.severity,
                    'description': alert.description,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in self.active_alerts[model_id]
            ]
            
            # Update status based on alerts
            severities = [alert.severity for alert in self.active_alerts[model_id]]
            if 'critical' in severities:
                report['status'] = 'critical'
            elif 'high' in severities:
                report['status'] = 'degraded'
            elif severities:
                report['status'] = 'warning'
        
        # Recommendations
        if report['status'] in ['critical', 'degraded']:
            report['recommendations'].append('Consider immediate model review')
            report['recommendations'].append('Prepare retraining pipeline')
        
        if report['status'] == 'warning':
            report['recommendations'].append('Monitor closely for further degradation')
            report['recommendations'].append('Schedule model evaluation')
        
        return report
        
    def _calculate_distribution_stats(self, values: np.ndarray) -> Dict[str, Any]:
        """Calculate distribution statistics."""
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
            'skewness': float(stats.skew(values)),
            'kurtosis': float(stats.kurtosis(values)),
            'values': values.tolist() if len(values) < 1000 else values[-1000:].tolist()
        }
        
    def _calculate_feature_stats(self, features: np.ndarray) -> Dict[str, Any]:
        """Calculate feature statistics."""
        stats_dict = {}
        
        n_features = features.shape[1] if features.ndim > 1 else 1
        
        for i in range(n_features):
            feature_values = features[:, i] if features.ndim > 1 else features
            
            stats_dict[f'feature_{i}_mean'] = float(np.mean(feature_values))
            stats_dict[f'feature_{i}_std'] = float(np.std(feature_values))
            stats_dict[f'feature_{i}_min'] = float(np.min(feature_values))
            stats_dict[f'feature_{i}_max'] = float(np.max(feature_values))
            
            # Store sample values for KS test
            if len(feature_values) < 1000:
                stats_dict[f'feature_{i}_values'] = feature_values.tolist()
            else:
                # Sample for large datasets
                indices = np.random.choice(len(feature_values), 1000, replace=False)
                stats_dict[f'feature_{i}_values'] = feature_values[indices].tolist()
        
        return stats_dict
        
    def _calculate_performance_stats(self, y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance statistics."""
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        
        # Basic metrics
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(errors**2))
        
        # Directional accuracy (for time series)
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            accuracy = np.mean(true_direction == pred_direction)
        else:
            accuracy = 0.0
        
        # Financial metrics (simplified)
        returns = np.diff(y_true) / y_true[:-1] if len(y_true) > 1 else np.array([])
        if len(returns) > 0:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
            
            # Max drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(np.min(drawdown))
        else:
            sharpe = 0.0
            max_drawdown = 0.0
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'accuracy': float(accuracy),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown)
        }
        
    def clear_alerts(self, model_id: str):
        """Clear active alerts for a model."""
        if model_id in self.active_alerts:
            self.active_alerts[model_id].clear()
            self.logger.info(f"Cleared alerts for model {model_id}")
            
    def export_monitoring_data(self, model_id: str, output_file: str):
        """Export monitoring data for analysis."""
        data = {
            'model_id': model_id,
            'export_timestamp': datetime.now().isoformat(),
            'baseline_stats': self.baseline_stats.get(model_id, {}),
            'current_stats': self.current_stats.get(model_id, {}),
            'recent_predictions': [
                {
                    'timestamp': p['timestamp'].isoformat(),
                    'prediction': p['prediction'],
                    'true_value': p['true_value'],
                    'metadata': p['metadata']
                }
                for p in list(self.prediction_history[model_id])[-1000:]
            ],
            'active_alerts': [
                {
                    'timestamp': alert.timestamp.isoformat(),
                    'type': alert.drift_type,
                    'severity': alert.severity,
                    'description': alert.description,
                    'metrics': alert.metrics
                }
                for alert in self.active_alerts.get(model_id, [])
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Exported monitoring data for model {model_id} to {output_file}")