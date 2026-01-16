"""
CNN Pattern Recognition Module.

Provides CNN-based pattern recognition for cryptocurrency trading with:
- Original NumPy implementation (CNNPatternRecognizer)
- PyTorch GPU implementation with CBAM attention (CNNPatternGPU)
- Multi-timeframe pattern recognition (MultiTimeframeCNNPattern)
- Enhanced 12-class pattern detection (EnhancedPatternGenerator)
"""

# Original NumPy implementation
from .cnn_model import CNNPatternRecognizer
from .pattern_generator import PatternGenerator, EnhancedPatternGenerator
from .gaf_transformer import GAFTransformer
from .trainer import CNNPatternTrainer
from .enhanced_trainer import EnhancedCNNPatternTrainer

# PyTorch GPU implementation (optional - requires torch)
try:
    from .cnn_pattern_gpu import CNNPatternGPU, create_cnn_pattern_gpu
    from .pytorch_trainer import CNNPatternPyTorchTrainer, FocalLoss
    from .multi_timeframe_cnn import (
        MultiTimeframeCNNPattern,
        TimeframeEncoder,
        CrossTimeframeAttention,
        FeaturePyramidFusion,
        create_multi_timeframe_cnn
    )
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    CNNPatternGPU = None
    CNNPatternPyTorchTrainer = None
    MultiTimeframeCNNPattern = None


def create_cnn_pattern_model(use_gpu: bool = True, **kwargs):
    """
    Factory function to create CNN pattern model with automatic backend selection.

    Args:
        use_gpu: Whether to prefer GPU implementation
        **kwargs: Model configuration

    Returns:
        CNNPatternGPU if GPU available and use_gpu=True, else CNNPatternRecognizer
    """
    if use_gpu and PYTORCH_AVAILABLE:
        try:
            import torch
            if torch.cuda.is_available():
                return CNNPatternGPU(**kwargs)
        except:
            pass

    # Fallback to NumPy implementation
    return CNNPatternRecognizer(**kwargs)


__all__ = [
    # Original implementations
    'CNNPatternRecognizer',
    'PatternGenerator',
    'EnhancedPatternGenerator',
    'GAFTransformer',
    'CNNPatternTrainer',
    'EnhancedCNNPatternTrainer',

    # PyTorch implementations
    'CNNPatternGPU',
    'CNNPatternPyTorchTrainer',
    'FocalLoss',
    'MultiTimeframeCNNPattern',
    'TimeframeEncoder',
    'CrossTimeframeAttention',
    'FeaturePyramidFusion',

    # Factory functions
    'create_cnn_pattern_model',
    'create_cnn_pattern_gpu',
    'create_multi_timeframe_cnn',

    # Flags
    'PYTORCH_AVAILABLE'
]
