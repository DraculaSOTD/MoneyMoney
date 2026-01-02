"""
Model Degradation Detector for Real-Time Trading Systems.

Implements sophisticated model degradation detection including performance decay,
concept drift, and automated retraining triggers.
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
class DegradationMetrics:
    """Metrics for model degradation assessment."""
    timestamp: datetime
    model_id: str
    
    # Performance degradation
    performance_decay_rate: float
    current_performance: float
    baseline_performance: float
    performance_ratio: float
    
    # Stability metrics
    prediction_stability: float
    confidence_stability: float
    variance_ratio: float
    
    # Drift indicators
    feature_drift_score: float
    prediction_drift_score: float
    error_drift_score: float
    
    # Degradation scores
    overall_degradation_score: float
    degradation_severity: str  # none, mild, moderate, severe, critical
    
    # Retraining recommendation
    requires_retraining: bool
    urgency_level: str  # low, medium, high, immediate
    degradation_reasons: List[str]
    recommended_actions: List[str]


@dataclass
class PerformanceWindow:
    """Performance tracking window."""
    window_id: str
    start_time: datetime
    end_time: datetime
    predictions_count: int
    average_error: float
    error_variance: float
    directional_accuracy: float
    confidence_calibration: float


@dataclass
class RetrainingTrigger:
    """Automated retraining trigger event."""
    timestamp: datetime
    model_id: str
    trigger_type: str  # performance, drift, schedule, manual
    trigger_reason: str
    severity: str
    metrics_snapshot: Dict[str, float]
    recommended_approach: str
    estimated_improvement: float


class ModelDegradationDetector:
    """
    Advanced model degradation detection system.
    
    Features:
    - Performance decay monitoring
    - Concept drift detection
    - Model stability analysis
    - Automated retraining triggers
    - Performance baseline tracking
    - Degradation pattern recognition
    - Multi-window performance analysis
    - Predictive degradation modeling
    """
    
    def __init__(self,
                 performance_window_size: int = 100,
                 degradation_thresholds: Optional[Dict[str, float]] = None,
                 retraining_config: Optional[Dict[str, Any]] = None):
        """
        Initialize model degradation detector.
        
        Args:
            performance_window_size: Size of performance tracking windows
            degradation_thresholds: Thresholds for degradation detection
            retraining_config: Configuration for retraining triggers
        """
        self.performance_window_size = performance_window_size
        
        # Degradation thresholds
        self.degradation_thresholds = degradation_thresholds or {
            'performance_decay_threshold': 0.1,  # 10% performance drop
            'drift_threshold': 0.3,
            'stability_threshold': 0.7,
            'variance_increase_threshold': 1.5,  # 50% variance increase
            'min_predictions_for_assessment': 50
        }
        
        # Retraining configuration
        self.retraining_config = retraining_config or {
            'auto_trigger_enabled': True,
            'min_time_between_retraining': 24,  # hours
            'performance_trigger_threshold': 0.15,  # 15% drop
            'drift_trigger_threshold': 0.5,
            'stability_trigger_threshold': 0.5
        }
        
        # Model tracking
        self.model_baselines: Dict[str, Dict[str, float]] = {}
        self.performance_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.current_windows: Dict[str, PerformanceWindow] = {}
        
        # Degradation tracking
        self.degradation_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.retraining_triggers: deque = deque(maxlen=100)
        self.last_retraining: Dict[str, datetime] = {}
        
        # Performance metrics storage
        self.prediction_errors: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.confidence_scores: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.feature_statistics: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=500)))
        
        # Degradation patterns
        self.degradation_patterns: Dict[str, List[str]] = {
            'sudden_drop': ['performance drops > 20% in single window'],
            'gradual_decay': ['consistent performance decline over multiple windows'],
            'volatility_increase': ['error variance increases significantly'],
            'confidence_miscalibration': ['confidence scores diverge from accuracy'],
            'feature_drift': ['input distribution changes detected']
        }
        
        # Model lifecycle tracking
        self.model_lifecycle: Dict[str, Dict[str, Any]] = {}
    
    def set_model_baseline(self,
                          model_id: str,
                          baseline_metrics: Dict[str, float]) -> None:
        """
        Set performance baseline for a model.
        
        Args:
            model_id: Model identifier
            baseline_metrics: Baseline performance metrics
        """
        self.model_baselines[model_id] = baseline_metrics.copy()
        
        # Initialize model lifecycle
        self.model_lifecycle[model_id] = {
            'creation_time': datetime.now(),
            'last_assessment': datetime.now(),
            'total_predictions': 0,
            'retraining_count': 0,
            'current_status': 'healthy'
        }
        
        print(f"Set baseline for model {model_id}: {baseline_metrics}")
    
    def track_prediction(self,
                        model_id: str,
                        prediction_error: float,
                        confidence: float,
                        features: Optional[Dict[str, float]] = None,
                        is_correct_direction: bool = True) -> None:
        """
        Track a model prediction for degradation monitoring.
        
        Args:
            model_id: Model identifier
            prediction_error: Prediction error (actual - predicted)
            confidence: Model confidence score
            features: Input features used
            is_correct_direction: Whether direction was predicted correctly
        """
        # Store metrics
        self.prediction_errors[model_id].append(prediction_error)
        self.confidence_scores[model_id].append(confidence)
        
        # Update feature statistics
        if features:
            for feature_name, value in features.items():
                self.feature_statistics[model_id][feature_name].append(value)
        
        # Update current window
        if model_id not in self.current_windows:
            self._start_new_window(model_id)
        
        window = self.current_windows[model_id]
        window.predictions_count += 1
        
        # Check if window is complete
        if window.predictions_count >= self.performance_window_size:
            self._complete_window(model_id)
            self._start_new_window(model_id)
    
    def _start_new_window(self, model_id: str) -> None:
        """Start a new performance tracking window."""
        window_id = f"{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_windows[model_id] = PerformanceWindow(
            window_id=window_id,
            start_time=datetime.now(),
            end_time=datetime.now(),
            predictions_count=0,
            average_error=0.0,
            error_variance=0.0,
            directional_accuracy=0.0,
            confidence_calibration=0.0
        )
    
    def _complete_window(self, model_id: str) -> None:
        """Complete current window and calculate metrics."""
        window = self.current_windows[model_id]
        window.end_time = datetime.now()
        
        # Get recent predictions for this window
        recent_errors = list(self.prediction_errors[model_id])[-window.predictions_count:]
        recent_confidences = list(self.confidence_scores[model_id])[-window.predictions_count:]
        
        if recent_errors:
            # Calculate window metrics
            window.average_error = np.mean(np.abs(recent_errors))
            window.error_variance = np.var(recent_errors)
            
            # Directional accuracy (simplified - assumes error sign indicates wrong direction)
            window.directional_accuracy = sum(1 for e in recent_errors if abs(e) < 0.02) / len(recent_errors)
            
            # Confidence calibration (correlation between confidence and accuracy)
            if recent_confidences and len(recent_confidences) == len(recent_errors):
                accuracies = [1.0 - min(abs(e), 1.0) for e in recent_errors]
                if np.std(recent_confidences) > 0 and np.std(accuracies) > 0:
                    window.confidence_calibration = np.corrcoef(recent_confidences, accuracies)[0, 1]
        
        # Store completed window
        self.performance_windows[model_id].append(window)
        
        # Update model lifecycle
        if model_id in self.model_lifecycle:
            self.model_lifecycle[model_id]['total_predictions'] += window.predictions_count
    
    def assess_degradation(self, model_id: str) -> DegradationMetrics:
        """
        Perform comprehensive degradation assessment for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Degradation metrics and recommendations
        """
        timestamp = datetime.now()
        
        # Check if we have enough data
        if (model_id not in self.performance_windows or 
            len(self.performance_windows[model_id]) < 2):
            
            return self._create_default_metrics(model_id, timestamp)
        
        # Get baseline
        baseline = self.model_baselines.get(model_id, {})
        if not baseline:
            # Use first window as baseline
            first_window = list(self.performance_windows[model_id])[0]
            baseline = {
                'average_error': first_window.average_error,
                'directional_accuracy': first_window.directional_accuracy,
                'error_variance': first_window.error_variance
            }
        
        # Calculate performance metrics
        recent_windows = list(self.performance_windows[model_id])[-5:]
        
        current_performance = np.mean([w.directional_accuracy for w in recent_windows])
        baseline_performance = baseline.get('directional_accuracy', 0.7)
        performance_ratio = current_performance / baseline_performance if baseline_performance > 0 else 0
        
        # Calculate performance decay rate
        if len(recent_windows) > 1:
            window_performances = [w.directional_accuracy for w in recent_windows]
            # Linear regression to find trend
            x = np.arange(len(window_performances))
            if np.std(x) > 0:
                slope = np.polyfit(x, window_performances, 1)[0]
                performance_decay_rate = -slope  # Positive if declining
            else:
                performance_decay_rate = 0.0
        else:
            performance_decay_rate = 0.0
        
        # Calculate stability metrics
        prediction_stability = self._calculate_prediction_stability(model_id)
        confidence_stability = self._calculate_confidence_stability(model_id)
        
        # Calculate variance ratio
        current_variance = np.mean([w.error_variance for w in recent_windows])
        baseline_variance = baseline.get('error_variance', current_variance)
        variance_ratio = current_variance / baseline_variance if baseline_variance > 0 else 1.0
        
        # Calculate drift scores
        feature_drift_score = self._calculate_feature_drift(model_id)
        prediction_drift_score = self._calculate_prediction_drift(model_id)
        error_drift_score = self._calculate_error_drift(model_id)
        
        # Calculate overall degradation score
        degradation_components = [
            max(0, 1 - performance_ratio),  # Performance drop
            performance_decay_rate * 10,  # Decay rate (scaled)
            max(0, 1 - prediction_stability),  # Instability
            max(0, variance_ratio - 1) / 2,  # Variance increase
            feature_drift_score,
            prediction_drift_score,
            error_drift_score
        ]
        
        overall_degradation_score = np.mean(degradation_components)
        
        # Determine severity
        degradation_severity = self._determine_degradation_severity(overall_degradation_score)
        
        # Check if retraining is required
        requires_retraining, urgency_level = self._check_retraining_requirement(
            model_id, overall_degradation_score, performance_ratio
        )
        
        # Identify degradation reasons
        degradation_reasons = self._identify_degradation_reasons(
            performance_decay_rate, variance_ratio, prediction_stability,
            feature_drift_score, prediction_drift_score
        )
        
        # Generate recommendations
        recommended_actions = self._generate_recommendations(
            degradation_severity, degradation_reasons, requires_retraining
        )
        
        # Create metrics
        metrics = DegradationMetrics(
            timestamp=timestamp,
            model_id=model_id,
            performance_decay_rate=performance_decay_rate,
            current_performance=current_performance,
            baseline_performance=baseline_performance,
            performance_ratio=performance_ratio,
            prediction_stability=prediction_stability,
            confidence_stability=confidence_stability,
            variance_ratio=variance_ratio,
            feature_drift_score=feature_drift_score,
            prediction_drift_score=prediction_drift_score,
            error_drift_score=error_drift_score,
            overall_degradation_score=overall_degradation_score,
            degradation_severity=degradation_severity,
            requires_retraining=requires_retraining,
            urgency_level=urgency_level,
            degradation_reasons=degradation_reasons,
            recommended_actions=recommended_actions
        )
        
        # Store assessment
        self.degradation_history[model_id].append(metrics)
        
        # Update model lifecycle
        if model_id in self.model_lifecycle:
            self.model_lifecycle[model_id]['last_assessment'] = timestamp
            self.model_lifecycle[model_id]['current_status'] = degradation_severity
        
        # Check for automatic retraining trigger
        if requires_retraining and self.retraining_config['auto_trigger_enabled']:
            self._trigger_retraining(model_id, metrics)
        
        return metrics
    
    def _create_default_metrics(self, model_id: str, timestamp: datetime) -> DegradationMetrics:
        """Create default metrics when insufficient data."""
        return DegradationMetrics(
            timestamp=timestamp,
            model_id=model_id,
            performance_decay_rate=0.0,
            current_performance=0.0,
            baseline_performance=0.0,
            performance_ratio=1.0,
            prediction_stability=1.0,
            confidence_stability=1.0,
            variance_ratio=1.0,
            feature_drift_score=0.0,
            prediction_drift_score=0.0,
            error_drift_score=0.0,
            overall_degradation_score=0.0,
            degradation_severity="none",
            requires_retraining=False,
            urgency_level="low",
            degradation_reasons=["Insufficient data for assessment"],
            recommended_actions=["Continue monitoring"]
        )
    
    def _calculate_prediction_stability(self, model_id: str) -> float:
        """Calculate prediction stability score."""
        if model_id not in self.prediction_errors:
            return 1.0
        
        recent_errors = list(self.prediction_errors[model_id])[-100:]
        if len(recent_errors) < 20:
            return 1.0
        
        # Calculate rolling standard deviation
        window_size = 10
        rolling_stds = []
        
        for i in range(window_size, len(recent_errors)):
            window_errors = recent_errors[i-window_size:i]
            rolling_stds.append(np.std(window_errors))
        
        if not rolling_stds:
            return 1.0
        
        # Stability is inverse of std variation
        std_variation = np.std(rolling_stds) / (np.mean(rolling_stds) + 1e-8)
        stability = max(0.0, 1.0 - std_variation)
        
        return stability
    
    def _calculate_confidence_stability(self, model_id: str) -> float:
        """Calculate confidence score stability."""
        if model_id not in self.confidence_scores:
            return 1.0
        
        recent_confidences = list(self.confidence_scores[model_id])[-100:]
        if len(recent_confidences) < 20:
            return 1.0
        
        # Check for sudden confidence changes
        confidence_changes = []
        for i in range(1, len(recent_confidences)):
            change = abs(recent_confidences[i] - recent_confidences[i-1])
            confidence_changes.append(change)
        
        if not confidence_changes:
            return 1.0
        
        # High stability means low average change
        avg_change = np.mean(confidence_changes)
        stability = max(0.0, 1.0 - avg_change * 10)  # Scale factor
        
        return stability
    
    def _calculate_feature_drift(self, model_id: str) -> float:
        """Calculate feature drift score."""
        if model_id not in self.feature_statistics:
            return 0.0
        
        drift_scores = []
        
        for feature_name, values in self.feature_statistics[model_id].items():
            if len(values) < 50:
                continue
            
            # Compare recent vs historical distribution
            mid_point = len(values) // 2
            historical = list(values)[:mid_point]
            recent = list(values)[mid_point:]
            
            if historical and recent:
                # Simple drift measure: difference in means normalized by std
                hist_mean = np.mean(historical)
                hist_std = np.std(historical)
                recent_mean = np.mean(recent)
                
                if hist_std > 0:
                    drift = abs(recent_mean - hist_mean) / hist_std
                    drift_scores.append(min(drift / 3, 1.0))  # Normalize to 0-1
        
        return np.mean(drift_scores) if drift_scores else 0.0
    
    def _calculate_prediction_drift(self, model_id: str) -> float:
        """Calculate prediction value drift."""
        if model_id not in self.prediction_errors:
            return 0.0
        
        errors = list(self.prediction_errors[model_id])
        if len(errors) < 50:
            return 0.0
        
        # Compare error distributions
        mid_point = len(errors) // 2
        historical_errors = errors[:mid_point]
        recent_errors = errors[mid_point:]
        
        # KL divergence approximation
        hist_mean = np.mean(historical_errors)
        hist_std = np.std(historical_errors)
        recent_mean = np.mean(recent_errors)
        recent_std = np.std(recent_errors)
        
        if hist_std > 0 and recent_std > 0:
            # Simplified KL divergence for normal distributions
            kl_div = np.log(recent_std / hist_std) + (hist_std**2 + (hist_mean - recent_mean)**2) / (2 * recent_std**2) - 0.5
            drift_score = min(abs(kl_div) / 2, 1.0)  # Normalize
        else:
            drift_score = 0.0
        
        return drift_score
    
    def _calculate_error_drift(self, model_id: str) -> float:
        """Calculate error pattern drift."""
        recent_windows = list(self.performance_windows[model_id])[-10:]
        
        if len(recent_windows) < 5:
            return 0.0
        
        # Check if error patterns are changing
        error_variances = [w.error_variance for w in recent_windows]
        avg_errors = [w.average_error for w in recent_windows]
        
        # Trend in error variance
        if len(error_variances) > 1:
            variance_trend = np.polyfit(range(len(error_variances)), error_variances, 1)[0]
            error_trend = np.polyfit(range(len(avg_errors)), avg_errors, 1)[0]
            
            # Combine trends
            drift_score = min(abs(variance_trend) * 10 + abs(error_trend) * 5, 1.0)
        else:
            drift_score = 0.0
        
        return drift_score
    
    def _determine_degradation_severity(self, score: float) -> str:
        """Determine degradation severity level."""
        if score < 0.1:
            return "none"
        elif score < 0.25:
            return "mild"
        elif score < 0.5:
            return "moderate"
        elif score < 0.75:
            return "severe"
        else:
            return "critical"
    
    def _check_retraining_requirement(self,
                                     model_id: str,
                                     degradation_score: float,
                                     performance_ratio: float) -> Tuple[bool, str]:
        """Check if model requires retraining."""
        # Check time since last retraining
        if model_id in self.last_retraining:
            hours_since_retraining = (datetime.now() - self.last_retraining[model_id]).total_seconds() / 3600
            if hours_since_retraining < self.retraining_config['min_time_between_retraining']:
                return False, "low"
        
        # Performance-based trigger
        if 1 - performance_ratio > self.retraining_config['performance_trigger_threshold']:
            return True, "high"
        
        # Degradation score trigger
        if degradation_score > 0.7:
            return True, "immediate"
        elif degradation_score > 0.5:
            return True, "high"
        elif degradation_score > 0.3:
            return True, "medium"
        
        return False, "low"
    
    def _identify_degradation_reasons(self,
                                    decay_rate: float,
                                    variance_ratio: float,
                                    stability: float,
                                    feature_drift: float,
                                    prediction_drift: float) -> List[str]:
        """Identify specific reasons for degradation."""
        reasons = []
        
        if decay_rate > 0.01:
            reasons.append("Consistent performance decline detected")
        
        if variance_ratio > 1.5:
            reasons.append("Increased prediction variance")
        
        if stability < 0.7:
            reasons.append("Unstable prediction patterns")
        
        if feature_drift > 0.3:
            reasons.append("Input feature distribution drift")
        
        if prediction_drift > 0.3:
            reasons.append("Prediction distribution shift")
        
        if not reasons:
            reasons.append("General model aging")
        
        return reasons
    
    def _generate_recommendations(self,
                                severity: str,
                                reasons: List[str],
                                requires_retraining: bool) -> List[str]:
        """Generate recommendations based on degradation analysis."""
        recommendations = []
        
        if requires_retraining:
            recommendations.append("Immediate model retraining recommended")
            
            if "feature distribution drift" in ' '.join(reasons):
                recommendations.append("Update feature engineering pipeline")
            
            if "performance decline" in ' '.join(reasons):
                recommendations.append("Consider model architecture changes")
        
        if severity in ["severe", "critical"]:
            recommendations.append("Implement fallback model strategy")
            recommendations.append("Increase monitoring frequency")
        
        if "variance" in ' '.join(reasons):
            recommendations.append("Review model regularization")
            recommendations.append("Check for data quality issues")
        
        if "unstable" in ' '.join(reasons):
            recommendations.append("Investigate recent data changes")
            recommendations.append("Consider ensemble approach for stability")
        
        return recommendations
    
    def _trigger_retraining(self, model_id: str, metrics: DegradationMetrics) -> None:
        """Trigger automatic retraining."""
        trigger = RetrainingTrigger(
            timestamp=datetime.now(),
            model_id=model_id,
            trigger_type="performance",
            trigger_reason="; ".join(metrics.degradation_reasons),
            severity=metrics.urgency_level,
            metrics_snapshot={
                'degradation_score': metrics.overall_degradation_score,
                'performance_ratio': metrics.performance_ratio,
                'variance_ratio': metrics.variance_ratio
            },
            recommended_approach="full_retrain" if metrics.degradation_severity == "critical" else "incremental_update",
            estimated_improvement=min(0.2, metrics.overall_degradation_score)
        )
        
        self.retraining_triggers.append(trigger)
        self.last_retraining[model_id] = datetime.now()
        
        print(f"Retraining triggered for model {model_id}: {trigger.trigger_reason}")
    
    def get_degradation_summary(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get degradation summary for model(s)."""
        summary = {
            'timestamp': datetime.now(),
            'monitoring_config': {
                'window_size': self.performance_window_size,
                'auto_retraining': self.retraining_config['auto_trigger_enabled']
            }
        }
        
        if model_id:
            # Single model summary
            if model_id in self.degradation_history and self.degradation_history[model_id]:
                recent_assessments = list(self.degradation_history[model_id])[-5:]
                latest = recent_assessments[-1]
                
                summary['model_id'] = model_id
                summary['current_status'] = latest.degradation_severity
                summary['degradation_score'] = latest.overall_degradation_score
                summary['requires_retraining'] = latest.requires_retraining
                summary['performance_ratio'] = latest.performance_ratio
                
                # Degradation trend
                if len(recent_assessments) > 1:
                    scores = [a.overall_degradation_score for a in recent_assessments]
                    trend = "worsening" if scores[-1] > scores[0] else "improving"
                    summary['degradation_trend'] = trend
                
                # Lifecycle info
                if model_id in self.model_lifecycle:
                    lifecycle = self.model_lifecycle[model_id]
                    summary['model_age_hours'] = (datetime.now() - lifecycle['creation_time']).total_seconds() / 3600
                    summary['total_predictions'] = lifecycle['total_predictions']
                    summary['retraining_count'] = lifecycle['retraining_count']
        else:
            # All models summary
            all_models = list(self.model_lifecycle.keys())
            summary['total_models'] = len(all_models)
            
            # Status distribution
            status_counts = defaultdict(int)
            for mid in all_models:
                if mid in self.model_lifecycle:
                    status = self.model_lifecycle[mid].get('current_status', 'unknown')
                    status_counts[status] += 1
            
            summary['status_distribution'] = dict(status_counts)
            
            # Models requiring retraining
            models_need_retraining = []
            for mid in all_models:
                if mid in self.degradation_history and self.degradation_history[mid]:
                    latest = list(self.degradation_history[mid])[-1]
                    if latest.requires_retraining:
                        models_need_retraining.append({
                            'model_id': mid,
                            'urgency': latest.urgency_level,
                            'score': latest.overall_degradation_score
                        })
            
            summary['models_requiring_retraining'] = models_need_retraining
            summary['total_retraining_triggers'] = len(self.retraining_triggers)
        
        return summary
    
    def get_retraining_history(self, hours: int = 168) -> List[Dict[str, Any]]:
        """Get retraining trigger history."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_triggers = [
            trigger for trigger in self.retraining_triggers
            if trigger.timestamp >= cutoff_time
        ]
        
        return [
            {
                'timestamp': trigger.timestamp.isoformat(),
                'model_id': trigger.model_id,
                'trigger_type': trigger.trigger_type,
                'reason': trigger.trigger_reason,
                'severity': trigger.severity,
                'recommended_approach': trigger.recommended_approach,
                'estimated_improvement': trigger.estimated_improvement
            }
            for trigger in recent_triggers
        ]
    
    def simulate_degradation(self, model_id: str, days: int = 30) -> Dict[str, Any]:
        """Simulate future degradation based on current trends."""
        if model_id not in self.degradation_history or len(self.degradation_history[model_id]) < 3:
            return {'error': 'Insufficient data for simulation'}
        
        recent_assessments = list(self.degradation_history[model_id])[-10:]
        
        # Extract degradation scores over time
        scores = [a.overall_degradation_score for a in recent_assessments]
        timestamps = [(a.timestamp - recent_assessments[0].timestamp).total_seconds() / 3600 for a in recent_assessments]
        
        # Fit trend
        if len(scores) > 1 and np.std(timestamps) > 0:
            # Linear regression
            slope, intercept = np.polyfit(timestamps, scores, 1)
            
            # Project forward
            future_hours = days * 24
            projected_score = intercept + slope * (timestamps[-1] + future_hours)
            projected_score = max(0.0, min(1.0, projected_score))  # Bound to [0, 1]
            
            # Determine when retraining will be needed
            retraining_threshold = 0.5
            if slope > 0 and projected_score < retraining_threshold:
                hours_to_retraining = (retraining_threshold - intercept) / slope
                days_to_retraining = hours_to_retraining / 24
            else:
                days_to_retraining = None
            
            return {
                'model_id': model_id,
                'current_score': scores[-1],
                'projected_score': projected_score,
                'degradation_rate_per_day': slope * 24,
                'days_to_retraining': days_to_retraining,
                'confidence': 'high' if len(scores) > 5 else 'medium'
            }
        else:
            return {'error': 'Unable to establish degradation trend'}