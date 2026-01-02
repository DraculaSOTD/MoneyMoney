"""
Real-time Model Adaptation System.

Provides online learning and model adaptation capabilities for crypto trading models
to adapt to changing market conditions in real-time.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import queue
import logging
import pickle
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class AdaptationMetrics:
    """Metrics for model adaptation performance."""
    timestamp: datetime
    model_id: str
    adaptation_type: str
    performance_before: float
    performance_after: float
    drift_score: float
    samples_processed: int
    adaptation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""
    drift_detected: bool
    drift_type: str  # 'concept', 'covariate', 'prior'
    drift_score: float
    confidence: float
    affected_features: List[str]
    recommendation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DriftDetector(ABC):
    """Abstract base class for drift detection."""
    
    @abstractmethod
    def detect_drift(self, reference_data: np.ndarray, 
                    current_data: np.ndarray) -> DriftDetectionResult:
        """Detect drift between reference and current data."""
        pass
    
    @abstractmethod
    def get_detector_name(self) -> str:
        """Get detector name."""
        pass


class KolmogorovSmirnovDriftDetector(DriftDetector):
    """
    Drift detection using Kolmogorov-Smirnov test.
    
    Detects distribution changes in features.
    """
    
    def __init__(self, significance_level: float = 0.05,
                 min_samples: int = 100):
        """
        Initialize KS drift detector.
        
        Args:
            significance_level: Statistical significance level
            min_samples: Minimum samples for detection
        """
        self.significance_level = significance_level
        self.min_samples = min_samples
        
    def detect_drift(self, reference_data: np.ndarray, 
                    current_data: np.ndarray) -> DriftDetectionResult:
        """Detect drift using KS test."""
        if len(reference_data) < self.min_samples or len(current_data) < self.min_samples:
            return DriftDetectionResult(
                drift_detected=False,
                drift_type='none',
                drift_score=0.0,
                confidence=0.0,
                affected_features=[],
                recommendation='Insufficient data for drift detection'
            )
        
        # Calculate KS statistic for each feature
        n_features = reference_data.shape[1]
        ks_statistics = []
        p_values = []
        
        for i in range(n_features):
            ks_stat, p_value = self._ks_2samp(
                reference_data[:, i],
                current_data[:, i]
            )
            ks_statistics.append(ks_stat)
            p_values.append(p_value)
        
        # Determine drift
        drift_features = [i for i, p in enumerate(p_values) if p < self.significance_level]
        drift_score = np.mean(ks_statistics)
        
        drift_detected = len(drift_features) > 0
        confidence = 1.0 - np.mean(p_values) if drift_detected else np.mean(p_values)
        
        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type='covariate' if drift_detected else 'none',
            drift_score=drift_score,
            confidence=confidence,
            affected_features=[f'feature_{i}' for i in drift_features],
            recommendation='Retrain model' if drift_detected else 'Continue monitoring',
            metadata={
                'ks_statistics': ks_statistics,
                'p_values': p_values
            }
        )
    
    def _ks_2samp(self, data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
        """Two-sample Kolmogorov-Smirnov test (simplified)."""
        # Sort data
        data1_sorted = np.sort(data1)
        data2_sorted = np.sort(data2)
        
        # Calculate empirical CDFs
        n1, n2 = len(data1), len(data2)
        data_all = np.concatenate([data1_sorted, data2_sorted])
        
        # Calculate KS statistic
        cdf1 = np.searchsorted(data1_sorted, data_all, side='right') / n1
        cdf2 = np.searchsorted(data2_sorted, data_all, side='right') / n2
        
        ks_stat = np.max(np.abs(cdf1 - cdf2))
        
        # Approximate p-value
        en = np.sqrt(n1 * n2 / (n1 + n2))
        p_value = 2 * np.exp(-2 * en**2 * ks_stat**2)
        
        return ks_stat, p_value
    
    def get_detector_name(self) -> str:
        return "KolmogorovSmirnov"


class AdaptiveDriftDetector(DriftDetector):
    """
    Adaptive drift detection using multiple methods.
    
    Combines statistical tests with performance monitoring.
    """
    
    def __init__(self, window_size: int = 1000,
                 warning_level: float = 0.95,
                 drift_level: float = 0.99):
        """
        Initialize adaptive drift detector.
        
        Args:
            window_size: Size of reference window
            warning_level: Warning threshold
            drift_level: Drift threshold
        """
        self.window_size = window_size
        self.warning_level = warning_level
        self.drift_level = drift_level
        
        # ADWIN-like parameters
        self.delta = 0.002  # Confidence parameter
        self.min_window = 32
        
        # State
        self.reference_window = deque(maxlen=window_size)
        self.current_window = deque(maxlen=window_size // 2)
        
    def detect_drift(self, reference_data: np.ndarray,
                    current_data: np.ndarray) -> DriftDetectionResult:
        """Detect drift adaptively."""
        # Update windows
        for sample in reference_data:
            self.reference_window.append(sample)
        for sample in current_data:
            self.current_window.append(sample)
        
        if len(self.reference_window) < self.min_window or len(self.current_window) < self.min_window:
            return DriftDetectionResult(
                drift_detected=False,
                drift_type='none',
                drift_score=0.0,
                confidence=0.0,
                affected_features=[],
                recommendation='Collecting more data'
            )
        
        # Convert to arrays
        ref_array = np.array(self.reference_window)
        curr_array = np.array(self.current_window)
        
        # Calculate drift metrics
        n_features = ref_array.shape[1]
        drift_scores = []
        
        for i in range(n_features):
            # Mean difference
            ref_mean = np.mean(ref_array[:, i])
            curr_mean = np.mean(curr_array[:, i])
            mean_diff = abs(curr_mean - ref_mean)
            
            # Variance difference
            ref_var = np.var(ref_array[:, i])
            curr_var = np.var(curr_array[:, i])
            var_ratio = max(curr_var, ref_var) / max(min(curr_var, ref_var), 1e-10)
            
            # Combined score
            feature_drift = mean_diff + 0.5 * np.log(var_ratio)
            drift_scores.append(feature_drift)
        
        # Overall drift score
        drift_score = np.mean(drift_scores)
        max_drift = np.max(drift_scores)
        
        # Determine drift level
        if max_drift > self.drift_level:
            drift_detected = True
            drift_type = 'concept'
            confidence = 0.95
            recommendation = 'Immediate model update required'
        elif max_drift > self.warning_level:
            drift_detected = True
            drift_type = 'covariate'
            confidence = 0.7
            recommendation = 'Monitor closely, prepare for adaptation'
        else:
            drift_detected = False
            drift_type = 'none'
            confidence = 0.3
            recommendation = 'No drift detected, continue monitoring'
        
        # Find affected features
        threshold = self.warning_level
        affected_features = [f'feature_{i}' for i, score in enumerate(drift_scores) if score > threshold]
        
        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=drift_type,
            drift_score=drift_score,
            confidence=confidence,
            affected_features=affected_features,
            recommendation=recommendation,
            metadata={
                'feature_drift_scores': drift_scores,
                'max_drift': max_drift
            }
        )
    
    def get_detector_name(self) -> str:
        return "AdaptiveDrift"


class ModelAdapter:
    """
    Adapts models in real-time based on new data and drift detection.
    
    Features:
    - Online learning
    - Incremental updates
    - Performance tracking
    - Automatic rollback
    """
    
    def __init__(self,
                 base_model: Any,
                 adaptation_rate: float = 0.01,
                 buffer_size: int = 1000,
                 min_performance: float = 0.5):
        """
        Initialize model adapter.
        
        Args:
            base_model: Base model to adapt
            adaptation_rate: Learning rate for adaptation
            buffer_size: Size of data buffer
            min_performance: Minimum acceptable performance
        """
        self.base_model = base_model
        self.adaptation_rate = adaptation_rate
        self.buffer_size = buffer_size
        self.min_performance = min_performance
        
        # Adaptation state
        self.data_buffer = deque(maxlen=buffer_size)
        self.performance_history = deque(maxlen=100)
        self.adaptation_history = []
        
        # Model versions
        self.current_model = self._copy_model(base_model)
        self.best_model = self._copy_model(base_model)
        self.best_performance = 0.0
        
    def _copy_model(self, model: Any) -> Any:
        """Create a copy of the model."""
        try:
            return pickle.loads(pickle.dumps(model))
        except:
            logger.warning("Could not deep copy model, using reference")
            return model
    
    def adapt_online(self, X: np.ndarray, y: np.ndarray,
                    sample_weight: Optional[np.ndarray] = None) -> AdaptationMetrics:
        """
        Perform online adaptation with new data.
        
        Args:
            X: Feature data
            y: Target data
            sample_weight: Optional sample weights
            
        Returns:
            Adaptation metrics
        """
        start_time = datetime.now()
        
        # Add to buffer
        for i in range(len(X)):
            self.data_buffer.append((X[i], y[i], sample_weight[i] if sample_weight is not None else 1.0))
        
        # Calculate performance before adaptation
        if len(self.data_buffer) > 10:
            X_test, y_test, weights = zip(*list(self.data_buffer)[-100:])
            X_test = np.array(X_test)
            y_test = np.array(y_test)
            
            performance_before = self._evaluate_model(self.current_model, X_test, y_test)
        else:
            performance_before = 0.0
        
        # Perform adaptation
        if hasattr(self.current_model, 'partial_fit'):
            # Scikit-learn style incremental learning
            self.current_model.partial_fit(X, y, sample_weight=sample_weight)
        elif hasattr(self.current_model, 'fit'):
            # Retrain on buffered data
            if len(self.data_buffer) >= 100:
                X_train, y_train, weights = zip(*list(self.data_buffer))
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                weights = np.array(weights)
                
                # Weighted training with higher weight on recent samples
                time_weights = np.linspace(0.5, 1.0, len(weights))
                final_weights = weights * time_weights
                
                self.current_model.fit(X_train, y_train, sample_weight=final_weights)
        else:
            # Custom adaptation logic
            self._custom_adapt(X, y, sample_weight)
        
        # Evaluate after adaptation
        if len(self.data_buffer) > 10:
            performance_after = self._evaluate_model(self.current_model, X_test, y_test)
            self.performance_history.append(performance_after)
        else:
            performance_after = performance_before
        
        # Update best model if improved
        if performance_after > self.best_performance:
            self.best_model = self._copy_model(self.current_model)
            self.best_performance = performance_after
        
        # Rollback if performance degraded significantly
        if performance_after < self.min_performance and performance_after < performance_before * 0.9:
            logger.warning(f"Performance degraded from {performance_before:.3f} to {performance_after:.3f}, rolling back")
            self.current_model = self._copy_model(self.best_model)
            performance_after = self.best_performance
        
        # Calculate adaptation time
        adaptation_time = (datetime.now() - start_time).total_seconds()
        
        # Create metrics
        metrics = AdaptationMetrics(
            timestamp=datetime.now(),
            model_id=str(id(self.current_model)),
            adaptation_type='online',
            performance_before=performance_before,
            performance_after=performance_after,
            drift_score=abs(performance_after - performance_before),
            samples_processed=len(X),
            adaptation_time=adaptation_time,
            metadata={
                'buffer_size': len(self.data_buffer),
                'best_performance': self.best_performance
            }
        )
        
        self.adaptation_history.append(metrics)
        return metrics
    
    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model performance."""
        try:
            predictions = model.predict(X)
            
            # Classification accuracy or regression R²
            if len(np.unique(y)) < 10:  # Likely classification
                return np.mean(predictions == y)
            else:  # Regression
                ss_res = np.sum((y - predictions) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                return 1 - (ss_res / (ss_tot + 1e-10))
        except Exception as e:
            logger.error(f"Model evaluation error: {e}")
            return 0.0
    
    def _custom_adapt(self, X: np.ndarray, y: np.ndarray,
                     sample_weight: Optional[np.ndarray] = None):
        """Custom adaptation logic for non-standard models."""
        # Example: Simple parameter update for linear models
        if hasattr(self.current_model, 'coef_'):
            # Gradient descent update
            predictions = self.current_model.predict(X)
            errors = y - predictions
            
            # Weight errors
            if sample_weight is not None:
                errors = errors * sample_weight
            
            # Update coefficients
            gradient = -2 * X.T @ errors / len(X)
            self.current_model.coef_ -= self.adaptation_rate * gradient
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptation history."""
        if not self.adaptation_history:
            return {}
        
        recent_adaptations = self.adaptation_history[-10:]
        
        return {
            'total_adaptations': len(self.adaptation_history),
            'current_performance': self.performance_history[-1] if self.performance_history else 0.0,
            'best_performance': self.best_performance,
            'avg_adaptation_time': np.mean([m.adaptation_time for m in recent_adaptations]),
            'performance_trend': np.polyfit(
                range(len(self.performance_history[-20:])),
                list(self.performance_history)[-20:],
                1
            )[0] if len(self.performance_history) > 2 else 0.0,
            'samples_in_buffer': len(self.data_buffer)
        }


class RealTimeAdaptationSystem:
    """
    Complete real-time model adaptation system.
    
    Integrates drift detection, online learning, and performance monitoring.
    """
    
    def __init__(self,
                 models: Dict[str, Any],
                 drift_detectors: Optional[List[DriftDetector]] = None,
                 adaptation_interval: int = 100,
                 performance_threshold: float = 0.6):
        """
        Initialize adaptation system.
        
        Args:
            models: Dictionary of models to adapt
            drift_detectors: List of drift detectors
            adaptation_interval: Samples between adaptation checks
            performance_threshold: Minimum performance threshold
        """
        self.models = models
        self.drift_detectors = drift_detectors or [
            KolmogorovSmirnovDriftDetector(),
            AdaptiveDriftDetector()
        ]
        self.adaptation_interval = adaptation_interval
        self.performance_threshold = performance_threshold
        
        # Create adapters for each model
        self.adapters = {
            name: ModelAdapter(model, min_performance=performance_threshold)
            for name, model in models.items()
        }
        
        # System state
        self.sample_count = 0
        self.drift_history = defaultdict(list)
        self.performance_tracking = defaultdict(list)
        self.is_running = False
        
        # Processing queue
        self.data_queue = queue.Queue(maxsize=10000)
        self.adaptation_thread = None
        
        logger.info(f"Real-time adaptation system initialized with {len(models)} models")
    
    def start_adaptation(self):
        """Start real-time adaptation processing."""
        if self.is_running:
            logger.warning("Adaptation already running")
            return
        
        self.is_running = True
        self.adaptation_thread = threading.Thread(
            target=self._adaptation_loop,
            daemon=True
        )
        self.adaptation_thread.start()
        logger.info("Real-time adaptation started")
    
    def stop_adaptation(self):
        """Stop real-time adaptation."""
        self.is_running = False
        if self.adaptation_thread:
            self.adaptation_thread.join()
        logger.info("Real-time adaptation stopped")
    
    def _adaptation_loop(self):
        """Main adaptation processing loop."""
        batch = []
        
        while self.is_running:
            try:
                # Collect batch
                while len(batch) < self.adaptation_interval:
                    try:
                        data = self.data_queue.get(timeout=1.0)
                        batch.append(data)
                    except queue.Empty:
                        break
                
                if not batch:
                    continue
                
                # Process batch
                X = np.array([d[0] for d in batch])
                y = np.array([d[1] for d in batch])
                weights = np.array([d[2] if len(d) > 2 else 1.0 for d in batch])
                
                # Check for drift
                drift_results = self._check_drift(X, y)
                
                # Adapt models
                for model_name, adapter in self.adapters.items():
                    # Determine if adaptation needed
                    model_drifts = [r for r in drift_results if r['model'] == model_name]
                    
                    if model_drifts and any(d['result'].drift_detected for d in model_drifts):
                        # Drift detected, perform adaptation
                        logger.info(f"Drift detected for {model_name}, adapting...")
                        metrics = adapter.adapt_online(X, y, weights)
                        
                        # Track performance
                        self.performance_tracking[model_name].append({
                            'timestamp': metrics.timestamp,
                            'performance': metrics.performance_after,
                            'drift_score': metrics.drift_score
                        })
                    
                    elif self.sample_count % (self.adaptation_interval * 10) == 0:
                        # Periodic adaptation
                        metrics = adapter.adapt_online(X, y, weights)
                        
                        self.performance_tracking[model_name].append({
                            'timestamp': metrics.timestamp,
                            'performance': metrics.performance_after,
                            'drift_score': 0.0
                        })
                
                # Clear batch
                batch = []
                self.sample_count += len(X)
                
            except Exception as e:
                logger.error(f"Adaptation loop error: {e}")
                batch = []
    
    def _check_drift(self, X: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
        """Check for drift across all models and detectors."""
        results = []
        
        for model_name, adapter in self.adapters.items():
            # Get reference data from buffer
            if len(adapter.data_buffer) < 100:
                continue
            
            ref_data = []
            for i in range(min(500, len(adapter.data_buffer) - 100)):
                ref_data.append(adapter.data_buffer[i][0])
            
            if not ref_data:
                continue
            
            ref_array = np.array(ref_data)
            
            # Check with each detector
            for detector in self.drift_detectors:
                try:
                    result = detector.detect_drift(ref_array, X)
                    
                    drift_entry = {
                        'model': model_name,
                        'detector': detector.get_detector_name(),
                        'result': result,
                        'timestamp': datetime.now()
                    }
                    
                    results.append(drift_entry)
                    self.drift_history[model_name].append(drift_entry)
                    
                except Exception as e:
                    logger.error(f"Drift detection error for {model_name}: {e}")
        
        return results
    
    def add_data(self, X: np.ndarray, y: np.ndarray,
                sample_weight: Optional[np.ndarray] = None):
        """Add new data for adaptation."""
        # Add each sample to queue
        for i in range(len(X)):
            weight = sample_weight[i] if sample_weight is not None else 1.0
            try:
                self.data_queue.put((X[i], y[i], weight), timeout=1.0)
            except queue.Full:
                logger.warning("Adaptation queue full, dropping oldest samples")
                # Remove old samples
                try:
                    self.data_queue.get_nowait()
                    self.data_queue.put((X[i], y[i], weight), timeout=0.1)
                except:
                    pass
    
    def get_model(self, model_name: str) -> Any:
        """Get the current adapted model."""
        if model_name in self.adapters:
            return self.adapters[model_name].current_model
        return None
    
    def get_adaptation_report(self) -> Dict[str, Any]:
        """Get comprehensive adaptation report."""
        report = {
            'timestamp': datetime.now(),
            'total_samples': self.sample_count,
            'models': {}
        }
        
        for model_name, adapter in self.adapters.items():
            model_report = adapter.get_adaptation_summary()
            
            # Add drift information
            recent_drifts = [d for d in self.drift_history[model_name][-10:]]
            drift_rate = sum(1 for d in recent_drifts if d['result'].drift_detected) / max(len(recent_drifts), 1)
            
            model_report['drift_rate'] = drift_rate
            model_report['recent_drifts'] = [
                {
                    'detector': d['detector'],
                    'drift_type': d['result'].drift_type,
                    'score': d['result'].drift_score,
                    'timestamp': d['timestamp']
                }
                for d in recent_drifts if d['result'].drift_detected
            ]
            
            report['models'][model_name] = model_report
        
        return report
    
    def save_adapted_models(self, directory: str):
        """Save all adapted models."""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for model_name, adapter in self.adapters.items():
            # Save model
            model_path = os.path.join(directory, f"{model_name}_adapted.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(adapter.current_model, f)
            
            # Save adaptation history
            history_path = os.path.join(directory, f"{model_name}_history.json")
            history = {
                'adaptation_count': len(adapter.adaptation_history),
                'performance_history': list(adapter.performance_history),
                'best_performance': adapter.best_performance,
                'adaptations': [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'performance_before': m.performance_before,
                        'performance_after': m.performance_after,
                        'samples': m.samples_processed
                    }
                    for m in adapter.adaptation_history[-100:]
                ]
            }
            
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
        
        logger.info(f"Saved {len(self.adapters)} adapted models to {directory}")