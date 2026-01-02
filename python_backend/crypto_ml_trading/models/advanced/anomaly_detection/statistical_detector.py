"""
Statistical Anomaly Detection Methods.

Implements various statistical approaches for anomaly detection in trading data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import deque
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    is_anomaly: bool
    score: float
    threshold: float
    confidence: float
    method: str
    details: Dict


class StatisticalAnomalyDetector:
    """
    Statistical anomaly detection using multiple methods.
    
    Features:
    - Z-score based detection
    - Modified Z-score (using median)
    - Interquartile Range (IQR) method
    - Grubbs' test for outliers
    - Dixon's Q-test
    - Seasonal decomposition anomalies
    - Moving average based detection
    - Exponential smoothing anomalies
    """
    
    def __init__(self,
                 methods: List[str] = None,
                 z_threshold: float = 3.0,
                 iqr_factor: float = 1.5,
                 grubbs_alpha: float = 0.05,
                 window_size: int = 50,
                 seasonal_period: int = 24):
        """
        Initialize statistical anomaly detector.
        
        Args:
            methods: List of methods to use
            z_threshold: Z-score threshold
            iqr_factor: IQR multiplier for outlier detection
            grubbs_alpha: Significance level for Grubbs' test
            window_size: Window size for moving statistics
            seasonal_period: Period for seasonal decomposition
        """
        self.methods = methods or [
            'z_score', 'modified_z_score', 'iqr', 'moving_average'
        ]
        self.z_threshold = z_threshold
        self.iqr_factor = iqr_factor
        self.grubbs_alpha = grubbs_alpha
        self.window_size = window_size
        self.seasonal_period = seasonal_period
        
        # Historical data for streaming detection
        self.history = deque(maxlen=1000)
        self.statistics_cache = {}
        
    def detect_anomalies(self, 
                        data: Union[np.ndarray, pd.Series],
                        method: str = 'ensemble') -> List[AnomalyResult]:
        """
        Detect anomalies using specified method(s).
        
        Args:
            data: Time series data
            method: Detection method or 'ensemble' for multiple methods
            
        Returns:
            List of anomaly results
        """
        data = np.array(data) if not isinstance(data, np.ndarray) else data
        
        if method == 'ensemble':
            return self._ensemble_detection(data)
        else:
            return self._single_method_detection(data, method)
    
    def _ensemble_detection(self, data: np.ndarray) -> List[AnomalyResult]:
        """Ensemble anomaly detection using multiple methods."""
        all_results = []
        method_votes = np.zeros(len(data))
        method_scores = np.zeros(len(data))
        
        # Apply each method
        for method in self.methods:
            try:
                results = self._single_method_detection(data, method)
                
                for i, result in enumerate(results):
                    if result.is_anomaly:
                        method_votes[i] += 1
                    method_scores[i] += result.score
                    
            except Exception as e:
                print(f"Error in method {method}: {e}")
                continue
                
        # Combine results
        ensemble_results = []
        n_methods = len(self.methods)
        
        for i in range(len(data)):
            # Majority voting with score weighting
            vote_ratio = method_votes[i] / n_methods
            avg_score = method_scores[i] / n_methods
            
            is_anomaly = vote_ratio >= 0.5
            confidence = vote_ratio if is_anomaly else 1 - vote_ratio
            
            result = AnomalyResult(\n                is_anomaly=is_anomaly,\n                score=avg_score,\n                threshold=0.5,  # Majority threshold\n                confidence=confidence,\n                method='ensemble',\n                details={\n                    'vote_ratio': vote_ratio,\n                    'method_votes': method_votes[i],\n                    'total_methods': n_methods\n                }\n            )\n            ensemble_results.append(result)\n            \n        return ensemble_results\n    \n    def _single_method_detection(self, data: np.ndarray, method: str) -> List[AnomalyResult]:\n        \"\"\"Apply single anomaly detection method.\"\"\"\n        if method == 'z_score':\n            return self._z_score_detection(data)\n        elif method == 'modified_z_score':\n            return self._modified_z_score_detection(data)\n        elif method == 'iqr':\n            return self._iqr_detection(data)\n        elif method == 'grubbs':\n            return self._grubbs_test(data)\n        elif method == 'dixon':\n            return self._dixon_test(data)\n        elif method == 'moving_average':\n            return self._moving_average_detection(data)\n        elif method == 'exponential_smoothing':\n            return self._exponential_smoothing_detection(data)\n        elif method == 'seasonal':\n            return self._seasonal_decomposition_detection(data)\n        else:\n            raise ValueError(f\"Unknown method: {method}\")\n    \n    def _z_score_detection(self, data: np.ndarray) -> List[AnomalyResult]:\n        \"\"\"Z-score based anomaly detection.\"\"\"\n        mean = np.mean(data)\n        std = np.std(data)\n        \n        if std == 0:\n            # All values are the same\n            return [AnomalyResult(\n                is_anomaly=False,\n                score=0.0,\n                threshold=self.z_threshold,\n                confidence=1.0,\n                method='z_score',\n                details={'mean': mean, 'std': std}\n            ) for _ in range(len(data))]\n        \n        z_scores = np.abs((data - mean) / std)\n        \n        results = []\n        for i, z_score in enumerate(z_scores):\n            is_anomaly = z_score > self.z_threshold\n            confidence = min(1.0, z_score / self.z_threshold) if is_anomaly else min(1.0, 1 - z_score / self.z_threshold)\n            \n            result = AnomalyResult(\n                is_anomaly=is_anomaly,\n                score=z_score,\n                threshold=self.z_threshold,\n                confidence=confidence,\n                method='z_score',\n                details={'mean': mean, 'std': std, 'value': data[i]}\n            )\n            results.append(result)\n            \n        return results\n    \n    def _modified_z_score_detection(self, data: np.ndarray) -> List[AnomalyResult]:\n        \"\"\"Modified Z-score using median and MAD.\"\"\"\n        median = np.median(data)\n        mad = np.median(np.abs(data - median))\n        \n        if mad == 0:\n            # Use fallback standard deviation\n            mad = np.std(data) / 1.4826  # Conversion factor\n            \n        if mad == 0:\n            return [AnomalyResult(\n                is_anomaly=False,\n                score=0.0,\n                threshold=self.z_threshold,\n                confidence=1.0,\n                method='modified_z_score',\n                details={'median': median, 'mad': mad}\n            ) for _ in range(len(data))]\n        \n        modified_z_scores = 0.6745 * (data - median) / mad\n        abs_scores = np.abs(modified_z_scores)\n        \n        results = []\n        for i, score in enumerate(abs_scores):\n            is_anomaly = score > self.z_threshold\n            confidence = min(1.0, score / self.z_threshold) if is_anomaly else min(1.0, 1 - score / self.z_threshold)\n            \n            result = AnomalyResult(\n                is_anomaly=is_anomaly,\n                score=score,\n                threshold=self.z_threshold,\n                confidence=confidence,\n                method='modified_z_score',\n                details={'median': median, 'mad': mad, 'value': data[i]}\n            )\n            results.append(result)\n            \n        return results\n    \n    def _iqr_detection(self, data: np.ndarray) -> List[AnomalyResult]:\n        \"\"\"Interquartile Range (IQR) based detection.\"\"\"\n        q1 = np.percentile(data, 25)\n        q3 = np.percentile(data, 75)\n        iqr = q3 - q1\n        \n        lower_bound = q1 - self.iqr_factor * iqr\n        upper_bound = q3 + self.iqr_factor * iqr\n        \n        results = []\n        for i, value in enumerate(data):\n            is_anomaly = value < lower_bound or value > upper_bound\n            \n            # Calculate score as distance from bounds\n            if value < lower_bound:\n                score = (lower_bound - value) / (iqr + 1e-8)\n            elif value > upper_bound:\n                score = (value - upper_bound) / (iqr + 1e-8)\n            else:\n                # Distance to nearest bound\n                score = min(value - lower_bound, upper_bound - value) / (iqr + 1e-8)\n                \n            confidence = min(1.0, score) if is_anomaly else min(1.0, 1 - score)\n            \n            result = AnomalyResult(\n                is_anomaly=is_anomaly,\n                score=score,\n                threshold=0.0,  # IQR uses bounds rather than threshold\n                confidence=confidence,\n                method='iqr',\n                details={\n                    'q1': q1, 'q3': q3, 'iqr': iqr,\n                    'lower_bound': lower_bound, 'upper_bound': upper_bound,\n                    'value': value\n                }\n            )\n            results.append(result)\n            \n        return results\n    \n    def _grubbs_test(self, data: np.ndarray) -> List[AnomalyResult]:\n        \"\"\"Grubbs' test for outliers.\"\"\"\n        if len(data) < 3:\n            return [AnomalyResult(\n                is_anomaly=False,\n                score=0.0,\n                threshold=0.0,\n                confidence=1.0,\n                method='grubbs',\n                details={'error': 'Insufficient data'}\n            ) for _ in range(len(data))]\n        \n        n = len(data)\n        mean = np.mean(data)\n        std = np.std(data, ddof=1)\n        \n        if std == 0:\n            return [AnomalyResult(\n                is_anomaly=False,\n                score=0.0,\n                threshold=0.0,\n                confidence=1.0,\n                method='grubbs',\n                details={'mean': mean, 'std': std}\n            ) for _ in range(len(data))]\n        \n        # Critical value from t-distribution (approximation)\n        t_critical = self._get_grubbs_critical_value(n, self.grubbs_alpha)\n        \n        results = []\n        for i, value in enumerate(data):\n            # Grubbs statistic\n            g_score = abs(value - mean) / std\n            \n            is_anomaly = g_score > t_critical\n            confidence = min(1.0, g_score / t_critical) if is_anomaly else min(1.0, 1 - g_score / t_critical)\n            \n            result = AnomalyResult(\n                is_anomaly=is_anomaly,\n                score=g_score,\n                threshold=t_critical,\n                confidence=confidence,\n                method='grubbs',\n                details={'mean': mean, 'std': std, 'critical_value': t_critical, 'value': value}\n            )\n            results.append(result)\n            \n        return results\n    \n    def _get_grubbs_critical_value(self, n: int, alpha: float) -> float:\n        \"\"\"Approximate critical value for Grubbs' test.\"\"\"\n        # Simplified approximation\n        from math import sqrt, log\n        \n        if n <= 6:\n            # Use lookup table for small n\n            lookup = {3: 1.15, 4: 1.46, 5: 1.67, 6: 1.82}\n            return lookup.get(n, 2.0)\n        else:\n            # Approximation for larger n\n            t_val = 2.5  # Approximate t-value for alpha=0.05\n            return ((n - 1) / sqrt(n)) * sqrt(t_val**2 / (n - 2 + t_val**2))\n    \n    def _dixon_test(self, data: np.ndarray) -> List[AnomalyResult]:\n        \"\"\"Dixon's Q-test for outliers.\"\"\"\n        if len(data) < 3:\n            return [AnomalyResult(\n                is_anomaly=False,\n                score=0.0,\n                threshold=0.0,\n                confidence=1.0,\n                method='dixon',\n                details={'error': 'Insufficient data'}\n            ) for _ in range(len(data))]\n        \n        sorted_data = np.sort(data)\n        n = len(sorted_data)\n        \n        # Critical values for Dixon's test (approximation)\n        if n <= 7:\n            q_critical = 0.568\n        elif n <= 10:\n            q_critical = 0.481\n        elif n <= 13:\n            q_critical = 0.425\n        else:\n            q_critical = 0.384\n            \n        results = []\n        data_range = sorted_data[-1] - sorted_data[0]\n        \n        for i, value in enumerate(data):\n            # Find position in sorted array\n            pos = np.searchsorted(sorted_data, value)\n            \n            # Calculate Q statistic\n            if pos == 0:  # Smallest value\n                if n > 1:\n                    q_stat = (sorted_data[1] - sorted_data[0]) / data_range\n                else:\n                    q_stat = 0\n            elif pos == n:  # Largest value\n                if n > 1:\n                    q_stat = (sorted_data[-1] - sorted_data[-2]) / data_range\n                else:\n                    q_stat = 0\n            else:\n                # Not an extreme value\n                q_stat = 0\n                \n            is_anomaly = q_stat > q_critical\n            confidence = min(1.0, q_stat / q_critical) if is_anomaly else 1.0\n            \n            result = AnomalyResult(\n                is_anomaly=is_anomaly,\n                score=q_stat,\n                threshold=q_critical,\n                confidence=confidence,\n                method='dixon',\n                details={'critical_value': q_critical, 'range': data_range, 'value': value}\n            )\n            results.append(result)\n            \n        return results\n    \n    def _moving_average_detection(self, data: np.ndarray) -> List[AnomalyResult]:\n        \"\"\"Moving average based anomaly detection.\"\"\"\n        results = []\n        \n        for i in range(len(data)):\n            # Define window around current point\n            start_idx = max(0, i - self.window_size // 2)\n            end_idx = min(len(data), i + self.window_size // 2 + 1)\n            \n            window_data = data[start_idx:end_idx]\n            \n            # Exclude current point from statistics\n            if len(window_data) > 1:\n                current_excluded = np.concatenate([\n                    window_data[:i-start_idx],\n                    window_data[i-start_idx+1:]\n                ]) if 0 <= i-start_idx < len(window_data) else window_data\n                \n                if len(current_excluded) > 0:\n                    mean = np.mean(current_excluded)\n                    std = np.std(current_excluded)\n                else:\n                    mean = np.mean(window_data)\n                    std = np.std(window_data)\n            else:\n                mean = data[i]\n                std = 0\n                \n            if std > 0:\n                z_score = abs(data[i] - mean) / std\n                is_anomaly = z_score > self.z_threshold\n                confidence = min(1.0, z_score / self.z_threshold) if is_anomaly else min(1.0, 1 - z_score / self.z_threshold)\n            else:\n                z_score = 0\n                is_anomaly = False\n                confidence = 1.0\n                \n            result = AnomalyResult(\n                is_anomaly=is_anomaly,\n                score=z_score,\n                threshold=self.z_threshold,\n                confidence=confidence,\n                method='moving_average',\n                details={\n                    'window_mean': mean,\n                    'window_std': std,\n                    'window_size': len(window_data),\n                    'value': data[i]\n                }\n            )\n            results.append(result)\n            \n        return results\n    \n    def _exponential_smoothing_detection(self, data: np.ndarray, alpha: float = 0.3) -> List[AnomalyResult]:\n        \"\"\"Exponential smoothing based anomaly detection.\"\"\"\n        if len(data) == 0:\n            return []\n            \n        # Initialize\n        smoothed = [data[0]]\n        errors = [0]\n        \n        # Compute exponential smoothing\n        for i in range(1, len(data)):\n            prediction = smoothed[-1]\n            error = data[i] - prediction\n            errors.append(abs(error))\n            \n            # Update smoothed value\n            smoothed_value = alpha * data[i] + (1 - alpha) * smoothed[-1]\n            smoothed.append(smoothed_value)\n            \n        # Compute adaptive threshold\n        if len(errors) > 1:\n            error_mean = np.mean(errors)\n            error_std = np.std(errors)\n            threshold = error_mean + self.z_threshold * error_std\n        else:\n            threshold = 0\n            \n        results = []\n        for i in range(len(data)):\n            error = errors[i]\n            is_anomaly = error > threshold if threshold > 0 else False\n            \n            if threshold > 0:\n                score = error / threshold\n                confidence = min(1.0, score) if is_anomaly else min(1.0, 1 - score)\n            else:\n                score = 0\n                confidence = 1.0\n                \n            result = AnomalyResult(\n                is_anomaly=is_anomaly,\n                score=score,\n                threshold=threshold,\n                confidence=confidence,\n                method='exponential_smoothing',\n                details={\n                    'prediction': smoothed[i] if i < len(smoothed) else data[i],\n                    'error': error,\n                    'value': data[i]\n                }\n            )\n            results.append(result)\n            \n        return results\n    \n    def _seasonal_decomposition_detection(self, data: np.ndarray) -> List[AnomalyResult]:\n        \"\"\"Seasonal decomposition based anomaly detection.\"\"\"\n        if len(data) < 2 * self.seasonal_period:\n            # Fall back to z-score for insufficient data\n            return self._z_score_detection(data)\n            \n        # Simple seasonal decomposition\n        trend, seasonal, residual = self._decompose_time_series(data)\n        \n        # Detect anomalies in residual component\n        residual_results = self._z_score_detection(residual)\n        \n        # Modify results to include seasonal context\n        results = []\n        for i, res in enumerate(residual_results):\n            result = AnomalyResult(\n                is_anomaly=res.is_anomaly,\n                score=res.score,\n                threshold=res.threshold,\n                confidence=res.confidence,\n                method='seasonal',\n                details={\n                    'trend': trend[i],\n                    'seasonal': seasonal[i],\n                    'residual': residual[i],\n                    'original_value': data[i]\n                }\n            )\n            results.append(result)\n            \n        return results\n    \n    def _decompose_time_series(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:\n        \"\"\"Simple time series decomposition.\"\"\"\n        n = len(data)\n        \n        # Trend (moving average)\n        trend = np.zeros(n)\n        window = min(self.seasonal_period, n // 2)\n        \n        for i in range(n):\n            start = max(0, i - window // 2)\n            end = min(n, i + window // 2 + 1)\n            trend[i] = np.mean(data[start:end])\n            \n        # Seasonal component (average by position in cycle)\n        detrended = data - trend\n        seasonal = np.zeros(n)\n        \n        for i in range(n):\n            cycle_pos = i % self.seasonal_period\n            # Find all points at same position in cycle\n            same_pos = [detrended[j] for j in range(cycle_pos, n, self.seasonal_period)]\n            seasonal[i] = np.mean(same_pos) if same_pos else 0\n            \n        # Residual\n        residual = data - trend - seasonal\n        \n        return trend, seasonal, residual\n    \n    def detect_streaming(self, new_value: float) -> AnomalyResult:\n        \"\"\"Detect anomalies in streaming data.\"\"\"\n        self.history.append(new_value)\n        \n        if len(self.history) < 3:\n            return AnomalyResult(\n                is_anomaly=False,\n                score=0.0,\n                threshold=self.z_threshold,\n                confidence=1.0,\n                method='streaming',\n                details={'history_size': len(self.history)}\n            )\n            \n        # Use recent history for anomaly detection\n        recent_data = np.array(list(self.history))\n        \n        # Apply ensemble method on recent window\n        results = self._ensemble_detection(recent_data)\n        \n        # Return result for the latest point\n        return results[-1]\n    \n    def update_parameters(self, **kwargs):\n        \"\"\"Update detection parameters.\"\"\"\n        for key, value in kwargs.items():\n            if hasattr(self, key):\n                setattr(self, key, value)\n                \n    def get_statistics(self, data: np.ndarray) -> Dict:\n        \"\"\"Get comprehensive statistics about the data.\"\"\"\n        stats = {\n            'mean': float(np.mean(data)),\n            'median': float(np.median(data)),\n            'std': float(np.std(data)),\n            'mad': float(np.median(np.abs(data - np.median(data)))),\n            'min': float(np.min(data)),\n            'max': float(np.max(data)),\n            'q1': float(np.percentile(data, 25)),\n            'q3': float(np.percentile(data, 75)),\n            'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),\n            'skewness': float(self._calculate_skewness(data)),\n            'kurtosis': float(self._calculate_kurtosis(data)),\n            'range': float(np.max(data) - np.min(data))\n        }\n        \n        return stats\n    \n    def _calculate_skewness(self, data: np.ndarray) -> float:\n        \"\"\"Calculate skewness of data.\"\"\"\n        mean = np.mean(data)\n        std = np.std(data)\n        \n        if std == 0:\n            return 0.0\n            \n        n = len(data)\n        skewness = np.sum(((data - mean) / std) ** 3) / n\n        \n        return skewness\n    \n    def _calculate_kurtosis(self, data: np.ndarray) -> float:\n        \"\"\"Calculate kurtosis of data.\"\"\"\n        mean = np.mean(data)\n        std = np.std(data)\n        \n        if std == 0:\n            return 0.0\n            \n        n = len(data)\n        kurtosis = np.sum(((data - mean) / std) ** 4) / n - 3  # Excess kurtosis\n        \n        return kurtosis"