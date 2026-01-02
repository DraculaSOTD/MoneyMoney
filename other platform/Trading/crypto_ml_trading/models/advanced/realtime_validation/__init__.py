"""
Real-Time Validation System for ML Trading Models.

This module implements comprehensive real-time validation capabilities including
model performance monitoring, prediction validation, data quality assessment,
and automated model retraining triggers.

Key Features:
- Real-time model performance tracking and validation
- Prediction accuracy monitoring with statistical tests
- Data drift detection and quality assessment
- Model degradation detection and alerting
- Automated retraining and model selection triggers
- Live backtesting and forward validation
- Performance attribution and error analysis
- Model confidence calibration and uncertainty quantification
"""

from models.advanced.realtime_validation.performance_monitor import PerformanceMonitor
from models.advanced.realtime_validation.prediction_validator import PredictionValidator
from models.advanced.realtime_validation.data_quality_monitor import DataQualityMonitor
from models.advanced.realtime_validation.model_degradation_detector import ModelDegradationDetector
from models.advanced.realtime_validation.validation_coordinator import ValidationCoordinator

__all__ = [
    'PerformanceMonitor',
    'PredictionValidator', 
    'DataQualityMonitor',
    'ModelDegradationDetector',
    'ValidationCoordinator'
]