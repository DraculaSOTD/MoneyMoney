"""
Data Quality Monitor for Real-Time Trading Systems.

Implements comprehensive data quality checks including completeness, accuracy,
timeliness, and drift detection for input features and market data.
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
class DataQualityMetrics:
    """Data quality metrics for a specific dataset."""
    timestamp: datetime
    dataset_name: str
    
    # Completeness metrics
    missing_rate: float
    null_rate: float
    completeness_score: float
    
    # Accuracy metrics
    outlier_rate: float
    invalid_rate: float
    accuracy_score: float
    
    # Consistency metrics
    duplicate_rate: float
    inconsistency_rate: float
    consistency_score: float
    
    # Timeliness metrics
    latency_ms: float
    staleness_rate: float
    timeliness_score: float
    
    # Overall quality
    overall_quality_score: float
    quality_grade: str  # A, B, C, D, F
    
    # Issues found
    quality_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DataDriftMetrics:
    """Metrics for data drift detection."""
    timestamp: datetime
    feature_name: str
    drift_type: str  # concept, covariate, prior
    
    # Statistical metrics
    drift_score: float
    p_value: float
    is_drifted: bool
    
    # Distribution metrics
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float
    
    # Change metrics
    mean_shift: float
    std_ratio: float
    distribution_distance: float
    
    # Metadata
    reference_period: Tuple[datetime, datetime]
    current_period: Tuple[datetime, datetime]
    samples_compared: int


@dataclass
class DataQualityAlert:
    """Alert for data quality issues."""
    timestamp: datetime
    alert_type: str
    severity: str  # low, medium, high, critical
    dataset_name: str
    description: str
    affected_features: List[str]
    impact_estimate: str
    recommended_actions: List[str]


class DataQualityMonitor:
    """
    Real-time data quality monitoring system.
    
    Features:
    - Missing data detection and handling
    - Outlier and anomaly detection
    - Data drift monitoring
    - Schema validation
    - Timeliness monitoring
    - Duplicate detection
    - Statistical profiling
    - Quality scoring and grading
    """
    
    def __init__(self,
                 max_history: int = 10000,
                 drift_window: int = 1000,
                 quality_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize data quality monitor.
        
        Args:
            max_history: Maximum data points to store
            drift_window: Window size for drift detection
            quality_thresholds: Quality thresholds for alerts
        """
        self.max_history = max_history
        self.drift_window = drift_window
        
        # Quality thresholds
        self.quality_thresholds = quality_thresholds or {
            'max_missing_rate': 0.05,  # 5% missing data
            'max_outlier_rate': 0.01,  # 1% outliers
            'max_latency_ms': 1000,  # 1 second latency
            'min_quality_score': 0.8,  # 80% minimum quality
            'drift_p_value': 0.05  # Statistical significance for drift
        }
        
        # Data storage
        self.data_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.feature_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Quality metrics tracking
        self.quality_history: deque = deque(maxlen=1000)
        self.drift_history: deque = deque(maxlen=500)
        self.quality_alerts: deque = deque(maxlen=200)
        
        # Schema definitions
        self.expected_schemas: Dict[str, Dict[str, type]] = {}
        self.feature_ranges: Dict[str, Dict[str, Tuple[float, float]]] = {}
        
        # Drift detection baselines
        self.reference_distributions: Dict[str, Dict[str, Any]] = {}
        self.drift_detectors: Dict[str, Any] = {}
        
        # Real-time statistics
        self.data_statistics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.latency_tracking: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Quality scoring weights
        self.quality_weights = {
            'completeness': 0.3,
            'accuracy': 0.3,
            'consistency': 0.2,
            'timeliness': 0.2
        }
    
    def check_data_quality(self,
                          dataset_name: str,
                          data: Dict[str, Any],
                          timestamp: Optional[datetime] = None) -> DataQualityMetrics:
        """
        Perform comprehensive data quality check.
        
        Args:
            dataset_name: Name of the dataset
            data: Data to check
            timestamp: Timestamp of data
            
        Returns:
            Data quality metrics
        """
        timestamp = timestamp or datetime.now()
        
        # Store data
        self.data_history[dataset_name].append({
            'timestamp': timestamp,
            'data': data.copy()
        })
        
        # Perform quality checks
        completeness_metrics = self._check_completeness(data)
        accuracy_metrics = self._check_accuracy(dataset_name, data)
        consistency_metrics = self._check_consistency(dataset_name, data)
        timeliness_metrics = self._check_timeliness(dataset_name, timestamp)
        
        # Calculate component scores
        completeness_score = 1.0 - completeness_metrics['missing_rate']
        accuracy_score = 1.0 - accuracy_metrics['outlier_rate'] - accuracy_metrics['invalid_rate']
        consistency_score = 1.0 - consistency_metrics['duplicate_rate'] - consistency_metrics['inconsistency_rate']
        timeliness_score = self._calculate_timeliness_score(timeliness_metrics)
        
        # Calculate overall quality score
        overall_score = (
            self.quality_weights['completeness'] * completeness_score +
            self.quality_weights['accuracy'] * accuracy_score +
            self.quality_weights['consistency'] * consistency_score +
            self.quality_weights['timeliness'] * timeliness_score
        )
        
        # Determine quality grade
        quality_grade = self._determine_quality_grade(overall_score)
        
        # Identify issues and recommendations
        quality_issues = self._identify_quality_issues(
            completeness_metrics, accuracy_metrics, consistency_metrics, timeliness_metrics
        )
        recommendations = self._generate_recommendations(quality_issues)
        
        # Create metrics object
        metrics = DataQualityMetrics(
            timestamp=timestamp,
            dataset_name=dataset_name,
            
            # Completeness
            missing_rate=completeness_metrics['missing_rate'],
            null_rate=completeness_metrics['null_rate'],
            completeness_score=completeness_score,
            
            # Accuracy
            outlier_rate=accuracy_metrics['outlier_rate'],
            invalid_rate=accuracy_metrics['invalid_rate'],
            accuracy_score=accuracy_score,
            
            # Consistency
            duplicate_rate=consistency_metrics['duplicate_rate'],
            inconsistency_rate=consistency_metrics['inconsistency_rate'],
            consistency_score=consistency_score,
            
            # Timeliness
            latency_ms=timeliness_metrics['latency_ms'],
            staleness_rate=timeliness_metrics['staleness_rate'],
            timeliness_score=timeliness_score,
            
            # Overall
            overall_quality_score=overall_score,
            quality_grade=quality_grade,
            quality_issues=quality_issues,
            recommendations=recommendations
        )
        
        # Store metrics
        self.quality_history.append(metrics)
        
        # Check for alerts
        self._check_quality_alerts(metrics)
        
        return metrics
    
    def _check_completeness(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Check data completeness."""
        total_fields = len(data)
        missing_fields = sum(1 for v in data.values() if v is None or v == '')
        null_fields = sum(1 for v in data.values() if v is None)
        
        return {
            'missing_rate': missing_fields / total_fields if total_fields > 0 else 0.0,
            'null_rate': null_fields / total_fields if total_fields > 0 else 0.0
        }
    
    def _check_accuracy(self, dataset_name: str, data: Dict[str, Any]) -> Dict[str, float]:
        """Check data accuracy."""
        outlier_count = 0
        invalid_count = 0
        total_fields = len(data)
        
        # Check each field
        for feature, value in data.items():
            if value is None:
                continue
            
            # Check against expected ranges
            if dataset_name in self.feature_ranges and feature in self.feature_ranges[dataset_name]:
                min_val, max_val = self.feature_ranges[dataset_name][feature]
                
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        outlier_count += 1
            
            # Check against expected schema
            if dataset_name in self.expected_schemas and feature in self.expected_schemas[dataset_name]:
                expected_type = self.expected_schemas[dataset_name][feature]
                if not isinstance(value, expected_type):
                    invalid_count += 1
            
            # Statistical outlier detection
            if feature in self.feature_profiles and isinstance(value, (int, float)):
                profile = self.feature_profiles[feature]
                if 'mean' in profile and 'std' in profile:
                    z_score = abs((value - profile['mean']) / (profile['std'] + 1e-8))
                    if z_score > 3:
                        outlier_count += 1
        
        return {
            'outlier_rate': outlier_count / total_fields if total_fields > 0 else 0.0,
            'invalid_rate': invalid_count / total_fields if total_fields > 0 else 0.0
        }
    
    def _check_consistency(self, dataset_name: str, data: Dict[str, Any]) -> Dict[str, float]:
        """Check data consistency."""
        # Check for duplicates in recent history
        recent_data = list(self.data_history[dataset_name])[-100:]
        duplicate_count = 0
        
        for historical in recent_data[:-1]:  # Exclude current
            if historical['data'] == data:
                duplicate_count += 1
                break
        
        # Check for logical inconsistencies (custom rules)
        inconsistency_count = 0
        
        # Example: Check if bid > ask (inconsistent)
        if 'bid' in data and 'ask' in data:
            if data['bid'] is not None and data['ask'] is not None:
                if data['bid'] > data['ask']:
                    inconsistency_count += 1
        
        # Example: Check if volume is negative
        if 'volume' in data and data['volume'] is not None:
            if data['volume'] < 0:
                inconsistency_count += 1
        
        total_checks = 2  # Number of consistency rules checked
        
        return {
            'duplicate_rate': 1.0 if duplicate_count > 0 else 0.0,
            'inconsistency_rate': inconsistency_count / total_checks if total_checks > 0 else 0.0
        }
    
    def _check_timeliness(self, dataset_name: str, timestamp: datetime) -> Dict[str, float]:
        """Check data timeliness."""
        # Calculate latency (simulated)
        processing_start = timestamp - timedelta(milliseconds=np.random.uniform(10, 500))
        latency_ms = (timestamp - processing_start).total_seconds() * 1000
        
        # Store latency
        self.latency_tracking[dataset_name].append(latency_ms)
        
        # Check staleness
        if dataset_name in self.data_history and len(self.data_history[dataset_name]) > 1:
            previous_data = list(self.data_history[dataset_name])[-2]
            time_diff = (timestamp - previous_data['timestamp']).total_seconds()
            
            # Expected update frequency (example: 1 minute)
            expected_frequency = 60  # seconds
            staleness_rate = max(0.0, (time_diff - expected_frequency) / expected_frequency)
        else:
            staleness_rate = 0.0
        
        return {
            'latency_ms': latency_ms,
            'staleness_rate': staleness_rate
        }
    
    def _calculate_timeliness_score(self, timeliness_metrics: Dict[str, float]) -> float:
        """Calculate timeliness score from metrics."""
        # Latency score (inverse relationship)
        max_latency = self.quality_thresholds['max_latency_ms']
        latency_score = max(0.0, 1.0 - timeliness_metrics['latency_ms'] / max_latency)
        
        # Staleness score (inverse relationship)
        staleness_score = max(0.0, 1.0 - timeliness_metrics['staleness_rate'])
        
        # Combined score
        return 0.7 * latency_score + 0.3 * staleness_score
    
    def _determine_quality_grade(self, score: float) -> str:
        """Determine quality grade from score."""
        if score >= 0.95:
            return 'A'
        elif score >= 0.85:
            return 'B'
        elif score >= 0.75:
            return 'C'
        elif score >= 0.65:
            return 'D'
        else:
            return 'F'
    
    def _identify_quality_issues(self,
                               completeness: Dict[str, float],
                               accuracy: Dict[str, float],
                               consistency: Dict[str, float],
                               timeliness: Dict[str, float]) -> List[str]:
        """Identify specific quality issues."""
        issues = []
        
        # Completeness issues
        if completeness['missing_rate'] > self.quality_thresholds['max_missing_rate']:
            issues.append(f"High missing data rate: {completeness['missing_rate']:.2%}")
        
        # Accuracy issues
        if accuracy['outlier_rate'] > self.quality_thresholds['max_outlier_rate']:
            issues.append(f"High outlier rate: {accuracy['outlier_rate']:.2%}")
        
        if accuracy['invalid_rate'] > 0:
            issues.append(f"Invalid data types detected: {accuracy['invalid_rate']:.2%}")
        
        # Consistency issues
        if consistency['duplicate_rate'] > 0:
            issues.append("Duplicate data detected")
        
        if consistency['inconsistency_rate'] > 0:
            issues.append(f"Data inconsistencies found: {consistency['inconsistency_rate']:.2%}")
        
        # Timeliness issues
        if timeliness['latency_ms'] > self.quality_thresholds['max_latency_ms']:
            issues.append(f"High latency: {timeliness['latency_ms']:.0f}ms")
        
        if timeliness['staleness_rate'] > 0.1:
            issues.append(f"Stale data detected: {timeliness['staleness_rate']:.2%} behind schedule")
        
        return issues
    
    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations based on issues."""
        recommendations = []
        
        for issue in issues:
            if "missing data" in issue:
                recommendations.append("Implement data imputation or improve data collection")
            elif "outlier" in issue:
                recommendations.append("Review outlier detection thresholds or data preprocessing")
            elif "invalid data" in issue:
                recommendations.append("Strengthen data validation at source")
            elif "duplicate" in issue:
                recommendations.append("Implement deduplication logic in data pipeline")
            elif "inconsistencies" in issue:
                recommendations.append("Add consistency validation rules")
            elif "latency" in issue:
                recommendations.append("Optimize data processing pipeline for lower latency")
            elif "stale data" in issue:
                recommendations.append("Increase data update frequency or check data sources")
        
        return list(set(recommendations))  # Remove duplicates
    
    def detect_drift(self,
                    feature_name: str,
                    current_values: np.ndarray,
                    reference_period: Optional[Tuple[datetime, datetime]] = None) -> DataDriftMetrics:
        """
        Detect data drift for a specific feature.
        
        Args:
            feature_name: Name of the feature
            current_values: Current feature values
            reference_period: Reference period for comparison
            
        Returns:
            Data drift metrics
        """
        timestamp = datetime.now()
        
        # Get reference distribution
        if feature_name not in self.reference_distributions:
            # Use first half of current values as reference
            mid_point = len(current_values) // 2
            reference_values = current_values[:mid_point]
            current_values = current_values[mid_point:]
        else:
            reference_values = self.reference_distributions[feature_name]['values']
        
        # Calculate statistics
        ref_mean = np.mean(reference_values)
        ref_std = np.std(reference_values)
        curr_mean = np.mean(current_values)
        curr_std = np.std(current_values)
        
        # Calculate drift metrics
        mean_shift = abs(curr_mean - ref_mean)
        std_ratio = curr_std / (ref_std + 1e-8)
        
        # Statistical tests
        # Kolmogorov-Smirnov test
        from scipy import stats
        ks_statistic, p_value = stats.ks_2samp(reference_values, current_values)
        
        # Determine drift type
        if mean_shift > ref_std * 0.5:
            drift_type = "covariate"  # Input distribution drift
        elif abs(std_ratio - 1.0) > 0.5:
            drift_type = "concept"  # Relationship drift
        else:
            drift_type = "none"
        
        # Create drift metrics
        metrics = DataDriftMetrics(
            timestamp=timestamp,
            feature_name=feature_name,
            drift_type=drift_type,
            drift_score=ks_statistic,
            p_value=p_value,
            is_drifted=p_value < self.quality_thresholds['drift_p_value'],
            reference_mean=ref_mean,
            current_mean=curr_mean,
            reference_std=ref_std,
            current_std=curr_std,
            mean_shift=mean_shift,
            std_ratio=std_ratio,
            distribution_distance=ks_statistic,
            reference_period=(datetime.now() - timedelta(days=7), datetime.now() - timedelta(days=1)),
            current_period=(datetime.now() - timedelta(days=1), datetime.now()),
            samples_compared=len(current_values)
        )
        
        # Store drift metrics
        self.drift_history.append(metrics)
        
        # Check for drift alerts
        if metrics.is_drifted:
            self._create_drift_alert(metrics)
        
        return metrics
    
    def _check_quality_alerts(self, metrics: DataQualityMetrics) -> None:
        """Check for quality alert conditions."""
        alerts = []
        
        # Overall quality alert
        if metrics.overall_quality_score < self.quality_thresholds['min_quality_score']:
            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                alert_type="low_quality",
                severity="high" if metrics.quality_grade == 'F' else "medium",
                dataset_name=metrics.dataset_name,
                description=f"Data quality below threshold: {metrics.overall_quality_score:.2f}",
                affected_features=[],
                impact_estimate="Model performance may be degraded",
                recommended_actions=metrics.recommendations
            ))
        
        # Missing data alert
        if metrics.missing_rate > self.quality_thresholds['max_missing_rate']:
            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                alert_type="missing_data",
                severity="high",
                dataset_name=metrics.dataset_name,
                description=f"High missing data rate: {metrics.missing_rate:.2%}",
                affected_features=[],
                impact_estimate="Incomplete model inputs",
                recommended_actions=["Check data sources", "Implement data recovery"]
            ))
        
        # Store alerts
        for alert in alerts:
            self.quality_alerts.append(alert)
    
    def _create_drift_alert(self, drift_metrics: DataDriftMetrics) -> None:
        """Create alert for detected drift."""
        severity = "high" if drift_metrics.drift_score > 0.5 else "medium"
        
        alert = DataQualityAlert(
            timestamp=datetime.now(),
            alert_type="data_drift",
            severity=severity,
            dataset_name=f"Feature: {drift_metrics.feature_name}",
            description=f"{drift_metrics.drift_type.capitalize()} drift detected (p-value: {drift_metrics.p_value:.4f})",
            affected_features=[drift_metrics.feature_name],
            impact_estimate="Model predictions may be unreliable",
            recommended_actions=[
                "Retrain model with recent data",
                "Review feature engineering",
                "Check for data source changes"
            ]
        )
        
        self.quality_alerts.append(alert)
    
    def profile_dataset(self, dataset_name: str, data_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create statistical profile of dataset."""
        profile = {
            'dataset_name': dataset_name,
            'sample_count': len(data_samples),
            'features': {}
        }
        
        # Extract all features
        all_features = set()
        for sample in data_samples:
            all_features.update(sample.keys())
        
        # Profile each feature
        for feature in all_features:
            values = []
            null_count = 0
            
            for sample in data_samples:
                if feature in sample:
                    if sample[feature] is not None:
                        values.append(sample[feature])
                    else:
                        null_count += 1
            
            # Create feature profile
            feature_profile = {
                'data_type': type(values[0]).__name__ if values else 'unknown',
                'null_rate': null_count / len(data_samples),
                'unique_count': len(set(values))
            }
            
            # Numerical statistics
            if values and isinstance(values[0], (int, float)):
                feature_profile.update({
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q1': np.percentile(values, 25),
                    'q3': np.percentile(values, 75)
                })
            
            # Categorical statistics
            elif values and isinstance(values[0], str):
                value_counts = defaultdict(int)
                for v in values:
                    value_counts[v] += 1
                
                feature_profile['top_values'] = dict(
                    sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                )
            
            profile['features'][feature] = feature_profile
            
            # Store profile for future use
            self.feature_profiles[feature] = feature_profile
        
        return profile
    
    def set_expected_schema(self, dataset_name: str, schema: Dict[str, type]) -> None:
        """Set expected schema for dataset."""
        self.expected_schemas[dataset_name] = schema
        print(f"Set expected schema for {dataset_name}")
    
    def set_feature_ranges(self, dataset_name: str, ranges: Dict[str, Tuple[float, float]]) -> None:
        """Set expected value ranges for features."""
        self.feature_ranges[dataset_name] = ranges
        print(f"Set feature ranges for {dataset_name}")
    
    def get_quality_summary(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """Get quality summary for dataset(s)."""
        summary = {
            'timestamp': datetime.now(),
            'monitoring_window': f"{self.max_history} records"
        }
        
        if dataset_name:
            # Single dataset summary
            dataset_metrics = [
                m for m in self.quality_history
                if m.dataset_name == dataset_name
            ]
            
            if dataset_metrics:
                recent_metrics = dataset_metrics[-10:]
                
                summary['dataset_name'] = dataset_name
                summary['total_checks'] = len(dataset_metrics)
                summary['average_quality'] = np.mean([m.overall_quality_score for m in recent_metrics])
                summary['current_grade'] = recent_metrics[-1].quality_grade if recent_metrics else 'N/A'
                summary['quality_trend'] = 'improving' if len(recent_metrics) > 1 and recent_metrics[-1].overall_quality_score > recent_metrics[0].overall_quality_score else 'stable'
                
                # Component scores
                summary['component_scores'] = {
                    'completeness': np.mean([m.completeness_score for m in recent_metrics]),
                    'accuracy': np.mean([m.accuracy_score for m in recent_metrics]),
                    'consistency': np.mean([m.consistency_score for m in recent_metrics]),
                    'timeliness': np.mean([m.timeliness_score for m in recent_metrics])
                }
        else:
            # All datasets summary
            all_datasets = set(m.dataset_name for m in self.quality_history)
            summary['total_datasets'] = len(all_datasets)
            summary['total_quality_checks'] = len(self.quality_history)
            summary['total_alerts'] = len(self.quality_alerts)
            
            # Overall quality distribution
            if self.quality_history:
                all_scores = [m.overall_quality_score for m in self.quality_history]
                summary['quality_distribution'] = {
                    'mean': np.mean(all_scores),
                    'std': np.std(all_scores),
                    'min': np.min(all_scores),
                    'max': np.max(all_scores)
                }
        
        # Recent alerts
        recent_alerts = list(self.quality_alerts)[-5:]
        summary['recent_alerts'] = [
            {
                'timestamp': alert.timestamp.isoformat(),
                'type': alert.alert_type,
                'severity': alert.severity,
                'dataset': alert.dataset_name,
                'description': alert.description
            }
            for alert in recent_alerts
        ]
        
        return summary
    
    def get_drift_analysis(self) -> Dict[str, Any]:
        """Get comprehensive drift analysis."""
        if not self.drift_history:
            return {'status': 'no_drift_analysis'}
        
        recent_drift = list(self.drift_history)[-20:]
        
        analysis = {
            'timestamp': datetime.now(),
            'total_drift_checks': len(self.drift_history),
            'drifted_features': []
        }
        
        # Analyze drifted features
        drifted = [d for d in recent_drift if d.is_drifted]
        
        if drifted:
            feature_drift = defaultdict(list)
            for d in drifted:
                feature_drift[d.feature_name].append(d)
            
            for feature, drift_list in feature_drift.items():
                latest_drift = drift_list[-1]
                analysis['drifted_features'].append({
                    'feature': feature,
                    'drift_type': latest_drift.drift_type,
                    'drift_score': latest_drift.drift_score,
                    'mean_shift': latest_drift.mean_shift,
                    'detection_time': latest_drift.timestamp.isoformat()
                })
        
        analysis['drift_rate'] = len(drifted) / len(recent_drift) if recent_drift else 0.0
        
        return analysis