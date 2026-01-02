"""
Real-Time Prediction Validator for ML Trading Models.

Implements comprehensive prediction validation including sanity checks,
consistency validation, and cross-model agreement analysis.
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
class ValidationRule:
    """Validation rule definition."""
    rule_id: str
    rule_name: str
    rule_type: str  # range, consistency, agreement, statistical
    parameters: Dict[str, Any]
    severity: str  # warning, error, critical
    enabled: bool = True
    description: str = ""


@dataclass
class ValidationResult:
    """Result of prediction validation."""
    timestamp: datetime
    prediction_id: str
    model_id: str
    is_valid: bool
    validation_score: float  # 0-1 scale
    failed_rules: List[str]
    warnings: List[str]
    errors: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsistencyCheck:
    """Temporal consistency check result."""
    model_id: str
    timestamp: datetime
    consistency_score: float
    inconsistencies: List[Dict[str, Any]]
    trend_reversal_count: int
    volatility_spike_count: int
    outlier_count: int


class PredictionValidator:
    """
    Real-time prediction validation system.
    
    Features:
    - Range and bounds validation
    - Temporal consistency checking
    - Cross-model agreement analysis
    - Statistical outlier detection
    - Market regime validation
    - Prediction sanity checks
    - Real-time validation scoring
    - Custom validation rules
    """
    
    def __init__(self,
                 max_history: int = 1000,
                 consistency_window: int = 20,  # predictions
                 agreement_threshold: float = 0.3):
        """
        Initialize prediction validator.
        
        Args:
            max_history: Maximum predictions to store for validation
            consistency_window: Window for consistency checks
            agreement_threshold: Threshold for model agreement
        """
        self.max_history = max_history
        self.consistency_window = consistency_window
        self.agreement_threshold = agreement_threshold
        
        # Validation rules
        self.validation_rules: Dict[str, ValidationRule] = {}
        self._initialize_default_rules()
        
        # Prediction history for validation
        self.prediction_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.validation_results: deque = deque(maxlen=max_history * 2)
        
        # Cross-model tracking
        self.model_predictions: Dict[datetime, Dict[str, float]] = {}
        self.model_agreements: deque = deque(maxlen=500)
        
        # Consistency tracking
        self.consistency_scores: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.consistency_checks: deque = deque(maxlen=500)
        
        # Market regime tracking
        self.current_market_regime: str = "normal"
        self.regime_parameters: Dict[str, Dict[str, float]] = {
            'normal': {'volatility_range': (0.01, 0.05), 'return_range': (-0.1, 0.1)},
            'volatile': {'volatility_range': (0.05, 0.15), 'return_range': (-0.2, 0.2)},
            'extreme': {'volatility_range': (0.15, 0.5), 'return_range': (-0.5, 0.5)}
        }
        
        # Validation statistics
        self.validation_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    def _initialize_default_rules(self) -> None:
        """Initialize default validation rules."""
        rules = [
            # Range validation
            ValidationRule(
                rule_id="range_check",
                rule_name="Prediction Range Check",
                rule_type="range",
                parameters={'min_value': -0.5, 'max_value': 0.5},
                severity="error",
                description="Check if prediction is within reasonable range"
            ),
            
            # Volatility spike detection
            ValidationRule(
                rule_id="volatility_spike",
                rule_name="Volatility Spike Detection",
                rule_type="statistical",
                parameters={'max_std_dev': 3.0},
                severity="warning",
                description="Detect predictions that are statistical outliers"
            ),
            
            # Temporal consistency
            ValidationRule(
                rule_id="temporal_consistency",
                rule_name="Temporal Consistency Check",
                rule_type="consistency",
                parameters={'max_change_rate': 0.1, 'window': 5},
                severity="warning",
                description="Check for unrealistic changes between predictions"
            ),
            
            # Cross-model agreement
            ValidationRule(
                rule_id="model_agreement",
                rule_name="Cross-Model Agreement",
                rule_type="agreement",
                parameters={'min_agreement': 0.3, 'min_models': 2},
                severity="warning",
                description="Check if multiple models agree on direction"
            ),
            
            # Confidence validation
            ValidationRule(
                rule_id="confidence_bounds",
                rule_name="Confidence Bounds Check",
                rule_type="range",
                parameters={'min_confidence': 0.0, 'max_confidence': 1.0},
                severity="error",
                description="Ensure confidence scores are valid"
            ),
            
            # Market regime validation
            ValidationRule(
                rule_id="regime_consistency",
                rule_name="Market Regime Consistency",
                rule_type="consistency",
                parameters={'regime_change_threshold': 0.8},
                severity="warning",
                description="Check if prediction matches current market regime"
            )
        ]
        
        for rule in rules:
            self.validation_rules[rule.rule_id] = rule
    
    def validate_prediction(self,
                          model_id: str,
                          prediction_value: float,
                          confidence: float,
                          asset_symbol: str = "",
                          market_data: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate a new prediction in real-time.
        
        Args:
            model_id: Model identifier
            prediction_value: Predicted value
            confidence: Confidence score
            asset_symbol: Asset being predicted
            market_data: Current market conditions
            
        Returns:
            Validation result with score and any issues
        """
        timestamp = datetime.now()
        prediction_id = f"{model_id}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Store prediction
        prediction_data = {
            'timestamp': timestamp,
            'value': prediction_value,
            'confidence': confidence,
            'asset': asset_symbol,
            'market_data': market_data or {}
        }
        self.prediction_history[model_id].append(prediction_data)
        
        # Run validation rules
        failed_rules = []
        warnings = []
        errors = []
        validation_scores = []
        
        for rule_id, rule in self.validation_rules.items():
            if not rule.enabled:
                continue
            
            rule_result = self._apply_validation_rule(
                rule, model_id, prediction_value, confidence, prediction_data
            )
            
            if rule_result['passed']:
                validation_scores.append(rule_result['score'])
            else:
                failed_rules.append(rule_id)
                
                if rule.severity == 'warning':
                    warnings.append(rule_result['message'])
                elif rule.severity in ['error', 'critical']:
                    errors.append(rule_result['message'])
                
                validation_scores.append(rule_result['score'] * 0.5)  # Penalty for failure
        
        # Calculate overall validation score
        overall_score = np.mean(validation_scores) if validation_scores else 0.0
        is_valid = len(errors) == 0 and overall_score > 0.5
        
        # Create validation result
        result = ValidationResult(
            timestamp=timestamp,
            prediction_id=prediction_id,
            model_id=model_id,
            is_valid=is_valid,
            validation_score=overall_score,
            failed_rules=failed_rules,
            warnings=warnings,
            errors=errors,
            metadata={
                'prediction_value': prediction_value,
                'confidence': confidence,
                'asset_symbol': asset_symbol
            }
        )
        
        # Store result
        self.validation_results.append(result)
        
        # Update statistics
        self.validation_stats[model_id]['total'] += 1
        if is_valid:
            self.validation_stats[model_id]['valid'] += 1
        else:
            self.validation_stats[model_id]['invalid'] += 1
        
        # Store for cross-model analysis
        if timestamp not in self.model_predictions:
            self.model_predictions[timestamp] = {}
        self.model_predictions[timestamp][model_id] = prediction_value
        
        # Clean old timestamps
        cutoff_time = timestamp - timedelta(minutes=5)
        old_timestamps = [ts for ts in self.model_predictions if ts < cutoff_time]
        for ts in old_timestamps:
            del self.model_predictions[ts]
        
        return result
    
    def _apply_validation_rule(self,
                             rule: ValidationRule,
                             model_id: str,
                             prediction_value: float,
                             confidence: float,
                             prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a specific validation rule."""
        result = {
            'passed': True,
            'score': 1.0,
            'message': ''
        }
        
        if rule.rule_type == 'range':
            result = self._validate_range(rule, prediction_value, confidence)
        
        elif rule.rule_type == 'statistical':
            result = self._validate_statistical(rule, model_id, prediction_value)
        
        elif rule.rule_type == 'consistency':
            result = self._validate_consistency(rule, model_id, prediction_value)
        
        elif rule.rule_type == 'agreement':
            result = self._validate_agreement(rule, model_id, prediction_value, prediction_data['timestamp'])
        
        return result
    
    def _validate_range(self,
                       rule: ValidationRule,
                       prediction_value: float,
                       confidence: float) -> Dict[str, Any]:
        """Validate value is within acceptable range."""
        params = rule.parameters
        
        # Check prediction value range
        if 'min_value' in params and 'max_value' in params:
            if prediction_value < params['min_value'] or prediction_value > params['max_value']:
                return {
                    'passed': False,
                    'score': 0.0,
                    'message': f"Prediction {prediction_value:.4f} outside range [{params['min_value']}, {params['max_value']}]"
                }
        
        # Check confidence range
        if rule.rule_id == 'confidence_bounds':
            if confidence < params['min_confidence'] or confidence > params['max_confidence']:
                return {
                    'passed': False,
                    'score': 0.0,
                    'message': f"Confidence {confidence:.2f} outside valid range [0, 1]"
                }
        
        return {'passed': True, 'score': 1.0, 'message': ''}
    
    def _validate_statistical(self,
                            rule: ValidationRule,
                            model_id: str,
                            prediction_value: float) -> Dict[str, Any]:
        """Validate against statistical outliers."""
        if model_id not in self.prediction_history:
            return {'passed': True, 'score': 1.0, 'message': ''}
        
        recent_predictions = list(self.prediction_history[model_id])[-20:]
        if len(recent_predictions) < 5:
            return {'passed': True, 'score': 1.0, 'message': ''}
        
        recent_values = [p['value'] for p in recent_predictions[:-1]]  # Exclude current
        
        # Calculate statistics
        mean_val = np.mean(recent_values)
        std_val = np.std(recent_values)
        
        if std_val == 0:
            return {'passed': True, 'score': 1.0, 'message': ''}
        
        # Check if outlier
        z_score = abs((prediction_value - mean_val) / std_val)
        max_std_dev = rule.parameters.get('max_std_dev', 3.0)
        
        if z_score > max_std_dev:
            score = max(0.0, 1.0 - (z_score - max_std_dev) / max_std_dev)
            return {
                'passed': False,
                'score': score,
                'message': f"Statistical outlier detected: {z_score:.2f} standard deviations from mean"
            }
        
        return {'passed': True, 'score': 1.0, 'message': ''}
    
    def _validate_consistency(self,
                            rule: ValidationRule,
                            model_id: str,
                            prediction_value: float) -> Dict[str, Any]:
        """Validate temporal consistency."""
        if model_id not in self.prediction_history:
            return {'passed': True, 'score': 1.0, 'message': ''}
        
        recent_predictions = list(self.prediction_history[model_id])
        if len(recent_predictions) < 2:
            return {'passed': True, 'score': 1.0, 'message': ''}
        
        # Check rate of change
        window = min(rule.parameters.get('window', 5), len(recent_predictions) - 1)
        recent_values = [p['value'] for p in recent_predictions[-window:]]
        
        if rule.rule_id == 'temporal_consistency':
            max_change_rate = rule.parameters.get('max_change_rate', 0.1)
            
            for i in range(1, len(recent_values)):
                change_rate = abs(recent_values[i] - recent_values[i-1])
                if change_rate > max_change_rate:
                    score = max(0.0, 1.0 - (change_rate - max_change_rate) / max_change_rate)
                    return {
                        'passed': False,
                        'score': score,
                        'message': f"Rapid change detected: {change_rate:.4f} exceeds max rate {max_change_rate}"
                    }
        
        elif rule.rule_id == 'regime_consistency':
            # Check if prediction is consistent with market regime
            if self.current_market_regime in self.regime_parameters:
                regime_params = self.regime_parameters[self.current_market_regime]
                return_range = regime_params['return_range']
                
                if prediction_value < return_range[0] or prediction_value > return_range[1]:
                    return {
                        'passed': False,
                        'score': 0.5,
                        'message': f"Prediction inconsistent with {self.current_market_regime} market regime"
                    }
        
        return {'passed': True, 'score': 1.0, 'message': ''}
    
    def _validate_agreement(self,
                          rule: ValidationRule,
                          model_id: str,
                          prediction_value: float,
                          timestamp: datetime) -> Dict[str, Any]:
        """Validate cross-model agreement."""
        # Find predictions from other models at similar time
        similar_time_predictions = {}
        
        for ts, predictions in self.model_predictions.items():
            if abs((ts - timestamp).total_seconds()) < 60:  # Within 1 minute
                similar_time_predictions.update(predictions)
        
        # Remove current model
        other_predictions = {
            mid: val for mid, val in similar_time_predictions.items()
            if mid != model_id
        }
        
        if len(other_predictions) < rule.parameters.get('min_models', 2):
            return {'passed': True, 'score': 1.0, 'message': ''}
        
        # Check directional agreement
        current_direction = np.sign(prediction_value)
        other_directions = [np.sign(val) for val in other_predictions.values()]
        
        agreement_ratio = sum(d == current_direction for d in other_directions) / len(other_directions)
        min_agreement = rule.parameters.get('min_agreement', 0.3)
        
        if agreement_ratio < min_agreement:
            return {
                'passed': False,
                'score': agreement_ratio,
                'message': f"Low model agreement: {agreement_ratio:.2%} < {min_agreement:.2%}"
            }
        
        return {'passed': True, 'score': 1.0, 'message': ''}
    
    def check_consistency(self, model_id: str) -> ConsistencyCheck:
        """Perform comprehensive consistency check for a model."""
        timestamp = datetime.now()
        
        if model_id not in self.prediction_history:
            return ConsistencyCheck(
                model_id=model_id,
                timestamp=timestamp,
                consistency_score=1.0,
                inconsistencies=[],
                trend_reversal_count=0,
                volatility_spike_count=0,
                outlier_count=0
            )
        
        predictions = list(self.prediction_history[model_id])[-self.consistency_window:]
        if len(predictions) < 3:
            return ConsistencyCheck(
                model_id=model_id,
                timestamp=timestamp,
                consistency_score=1.0,
                inconsistencies=[],
                trend_reversal_count=0,
                volatility_spike_count=0,
                outlier_count=0
            )
        
        values = [p['value'] for p in predictions]
        timestamps = [p['timestamp'] for p in predictions]
        
        inconsistencies = []
        trend_reversal_count = 0
        volatility_spike_count = 0
        outlier_count = 0
        
        # Check for trend reversals
        for i in range(2, len(values)):
            prev_trend = np.sign(values[i-1] - values[i-2])
            curr_trend = np.sign(values[i] - values[i-1])
            
            if prev_trend != 0 and curr_trend != 0 and prev_trend != curr_trend:
                trend_reversal_count += 1
                inconsistencies.append({
                    'type': 'trend_reversal',
                    'timestamp': timestamps[i],
                    'index': i
                })
        
        # Check for volatility spikes
        rolling_std = []
        window = 5
        for i in range(window, len(values)):
            window_std = np.std(values[i-window:i])
            rolling_std.append(window_std)
        
        if rolling_std:
            median_std = np.median(rolling_std)
            for i, std_val in enumerate(rolling_std):
                if std_val > median_std * 2:
                    volatility_spike_count += 1
                    inconsistencies.append({
                        'type': 'volatility_spike',
                        'timestamp': timestamps[i + window],
                        'volatility': std_val
                    })
        
        # Check for outliers
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val > 0:
            for i, val in enumerate(values):
                z_score = abs((val - mean_val) / std_val)
                if z_score > 2.5:
                    outlier_count += 1
                    inconsistencies.append({
                        'type': 'outlier',
                        'timestamp': timestamps[i],
                        'value': val,
                        'z_score': z_score
                    })
        
        # Calculate consistency score
        total_issues = trend_reversal_count + volatility_spike_count + outlier_count
        consistency_score = max(0.0, 1.0 - (total_issues / len(predictions)))
        
        check_result = ConsistencyCheck(
            model_id=model_id,
            timestamp=timestamp,
            consistency_score=consistency_score,
            inconsistencies=inconsistencies,
            trend_reversal_count=trend_reversal_count,
            volatility_spike_count=volatility_spike_count,
            outlier_count=outlier_count
        )
        
        # Store result
        self.consistency_checks.append(check_result)
        self.consistency_scores[model_id].append(consistency_score)
        
        return check_result
    
    def analyze_model_agreement(self, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Analyze agreement across multiple models."""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Find recent predictions
        recent_predictions = {}
        for ts, predictions in self.model_predictions.items():
            if abs((ts - timestamp).total_seconds()) < 300:  # Within 5 minutes
                for model_id, value in predictions.items():
                    if model_id not in recent_predictions:
                        recent_predictions[model_id] = []
                    recent_predictions[model_id].append(value)
        
        if len(recent_predictions) < 2:
            return {
                'timestamp': timestamp,
                'models_analyzed': 0,
                'agreement_score': 0.0,
                'consensus_direction': None
            }
        
        # Calculate average predictions per model
        model_averages = {
            model_id: np.mean(values)
            for model_id, values in recent_predictions.items()
        }
        
        # Analyze directional agreement
        directions = [np.sign(avg) for avg in model_averages.values()]
        
        # Find consensus direction
        direction_counts = {}
        for d in directions:
            direction_counts[d] = direction_counts.get(d, 0) + 1
        
        consensus_direction = max(direction_counts, key=direction_counts.get)
        agreement_ratio = direction_counts[consensus_direction] / len(directions)
        
        # Calculate value dispersion
        all_values = list(model_averages.values())
        value_std = np.std(all_values) if len(all_values) > 1 else 0.0
        
        # Calculate pairwise correlations
        if len(recent_predictions) > 1:
            correlations = []
            model_ids = list(recent_predictions.keys())
            
            for i in range(len(model_ids)):
                for j in range(i + 1, len(model_ids)):
                    values1 = recent_predictions[model_ids[i]]
                    values2 = recent_predictions[model_ids[j]]
                    
                    # Match lengths
                    min_len = min(len(values1), len(values2))
                    if min_len > 1:
                        corr = np.corrcoef(values1[:min_len], values2[:min_len])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
            
            avg_correlation = np.mean(correlations) if correlations else 0.0
        else:
            avg_correlation = 0.0
        
        analysis = {
            'timestamp': timestamp,
            'models_analyzed': len(recent_predictions),
            'agreement_score': agreement_ratio,
            'consensus_direction': consensus_direction,
            'value_dispersion': value_std,
            'average_correlation': avg_correlation,
            'model_averages': model_averages,
            'unanimous': agreement_ratio == 1.0,
            'high_agreement': agreement_ratio >= 0.7
        }
        
        # Store agreement analysis
        self.model_agreements.append(analysis)
        
        return analysis
    
    def get_validation_summary(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get validation summary for model(s)."""
        summary = {
            'timestamp': datetime.now(),
            'total_validations': len(self.validation_results)
        }
        
        if model_id:
            # Single model summary
            if model_id in self.validation_stats:
                stats = self.validation_stats[model_id]
                valid_count = stats['valid']
                total_count = stats['total']
                
                summary['model_id'] = model_id
                summary['total_predictions'] = total_count
                summary['valid_predictions'] = valid_count
                summary['validation_rate'] = valid_count / total_count if total_count > 0 else 0.0
                
                # Recent consistency
                if model_id in self.consistency_scores and self.consistency_scores[model_id]:
                    recent_scores = list(self.consistency_scores[model_id])[-10:]
                    summary['recent_consistency'] = np.mean(recent_scores)
                    summary['consistency_trend'] = 'improving' if len(recent_scores) > 1 and recent_scores[-1] > recent_scores[0] else 'stable'
                
                # Failed rules analysis
                model_failures = [
                    result for result in self.validation_results
                    if result.model_id == model_id and not result.is_valid
                ]
                
                if model_failures:
                    rule_failure_counts = defaultdict(int)
                    for failure in model_failures:
                        for rule in failure.failed_rules:
                            rule_failure_counts[rule] += 1
                    
                    summary['common_failures'] = dict(rule_failure_counts)
        
        else:
            # All models summary
            all_models = list(self.validation_stats.keys())
            summary['total_models'] = len(all_models)
            
            # Overall validation rates
            total_valid = sum(stats['valid'] for stats in self.validation_stats.values())
            total_predictions = sum(stats['total'] for stats in self.validation_stats.values())
            
            summary['overall_validation_rate'] = total_valid / total_predictions if total_predictions > 0 else 0.0
            
            # Model rankings by validation rate
            model_rates = []
            for mid, stats in self.validation_stats.items():
                if stats['total'] > 0:
                    rate = stats['valid'] / stats['total']
                    model_rates.append((mid, rate))
            
            model_rates.sort(key=lambda x: x[1], reverse=True)
            summary['model_rankings'] = [mid for mid, _ in model_rates[:5]]
            
            # Recent agreement analysis
            if self.model_agreements:
                recent_agreements = list(self.model_agreements)[-10:]
                summary['average_agreement'] = np.mean([a['agreement_score'] for a in recent_agreements])
        
        return summary
    
    def add_custom_rule(self, rule: ValidationRule) -> None:
        """Add custom validation rule."""
        self.validation_rules[rule.rule_id] = rule
        print(f"Added custom validation rule: {rule.rule_name}")
    
    def update_market_regime(self, regime: str) -> None:
        """Update current market regime for validation."""
        if regime in self.regime_parameters:
            self.current_market_regime = regime
            print(f"Updated market regime to: {regime}")
    
    def get_recent_failures(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent validation failures."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_failures = [
            result for result in self.validation_results
            if not result.is_valid and result.timestamp >= cutoff_time
        ]
        
        return [
            {
                'timestamp': result.timestamp.isoformat(),
                'model_id': result.model_id,
                'validation_score': result.validation_score,
                'failed_rules': result.failed_rules,
                'errors': result.errors,
                'warnings': result.warnings
            }
            for result in recent_failures
        ]