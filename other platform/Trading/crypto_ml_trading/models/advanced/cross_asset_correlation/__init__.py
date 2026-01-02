"""
Cross-Asset Correlation Analysis Module for Cryptocurrency Trading.

This module implements advanced cross-asset correlation analysis for capturing
relationships between different asset classes and their impact on cryptocurrency markets.

Key Features:
- Multi-asset correlation analysis and monitoring
- Dynamic correlation modeling with regime detection
- Cross-market spillover effects analysis
- Flight-to-quality and risk-on/risk-off detection
- Macro factor exposure analysis
- Portfolio correlation optimization
"""

from models.advanced.cross_asset_correlation.correlation_analyzer import CorrelationAnalyzer
from models.advanced.cross_asset_correlation.dynamic_correlation_model import DynamicCorrelationModel
from models.advanced.cross_asset_correlation.spillover_analyzer import SpilloverAnalyzer
from models.advanced.cross_asset_correlation.macro_factor_model import MacroFactorModel
from models.advanced.cross_asset_correlation.correlation_coordinator import CorrelationCoordinator

__all__ = [
    'CorrelationAnalyzer',
    'DynamicCorrelationModel', 
    'SpilloverAnalyzer',
    'MacroFactorModel',
    'CorrelationCoordinator'
]