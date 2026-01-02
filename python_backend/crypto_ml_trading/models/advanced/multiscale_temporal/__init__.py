"""
Multi-Scale Temporal Analysis Module for Cryptocurrency Trading.

This module implements multi-scale temporal analysis techniques for capturing
patterns across different time horizons in financial time series data.

Key Features:
- Wavelet decomposition for multi-resolution analysis
- Multi-scale trend detection and filtering
- Temporal pattern recognition across scales
- Hierarchical time series modeling
- Cross-scale correlation analysis
- Multi-horizon forecasting
"""

from models.advanced.multiscale_temporal.wavelet_analyzer import WaveletAnalyzer
from models.advanced.multiscale_temporal.multiscale_decomposition import MultiScaleDecomposition
from models.advanced.multiscale_temporal.temporal_patterns import TemporalPatternDetector
from models.advanced.multiscale_temporal.hierarchical_model import HierarchicalTemporalModel
from models.advanced.multiscale_temporal.scale_coordinator import ScaleCoordinator

__all__ = [
    'WaveletAnalyzer',
    'MultiScaleDecomposition',
    'TemporalPatternDetector',
    'HierarchicalTemporalModel',
    'ScaleCoordinator'
]