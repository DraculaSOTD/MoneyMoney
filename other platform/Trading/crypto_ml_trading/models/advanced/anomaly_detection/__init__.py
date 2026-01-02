"""
Anomaly Detection Module for Cryptocurrency Trading.

This module implements various anomaly detection algorithms for identifying
unusual patterns in market data, trading behavior, and system performance.

Key Features:
- Local Outlier Factor (LOF) for density-based anomaly detection
- Isolation Forest for ensemble-based anomaly detection
- Statistical anomaly detection methods
- Real-time anomaly monitoring
- Multi-dimensional anomaly scoring
"""

from models.advanced.anomaly_detection.lof_detector import LocalOutlierFactor
from models.advanced.anomaly_detection.isolation_forest import IsolationForest
from models.advanced.anomaly_detection.statistical_detector import StatisticalAnomalyDetector
from models.advanced.anomaly_detection.ensemble_detector import EnsembleAnomalyDetector
from models.advanced.anomaly_detection.realtime_monitor import RealtimeAnomalyMonitor

__all__ = [
    'LocalOutlierFactor',
    'IsolationForest',
    'StatisticalAnomalyDetector',
    'EnsembleAnomalyDetector',
    'RealtimeAnomalyMonitor'
]