"""
Temporal Fusion Transformer (TFT) for Multi-Horizon Cryptocurrency Forecasting

This module implements a state-of-the-art Transformer architecture specifically designed
for time series forecasting with interpretability features.

Key Features:
- Multi-horizon forecasting with confidence intervals
- Variable selection networks for feature importance
- Temporal self-attention for long-range dependencies
- Gated Residual Networks (GRN) for efficient learning
- Quantile regression for probabilistic predictions
"""

from models.advanced.transformer_tft.tft_model import TemporalFusionTransformer
from models.advanced.transformer_tft.temporal_attention import TemporalSelfAttention
from models.advanced.transformer_tft.variable_selection import VariableSelectionNetwork
from models.advanced.transformer_tft.quantile_loss import QuantileLoss
from models.advanced.transformer_tft.trainer import TFTTrainer
from models.advanced.transformer_tft.enhanced_tft_model import EnhancedTemporalFusionTransformer

__all__ = [
    'TemporalFusionTransformer',
    'TemporalSelfAttention',
    'VariableSelectionNetwork',
    'QuantileLoss',
    'TFTTrainer',
    'EnhancedTemporalFusionTransformer'
]