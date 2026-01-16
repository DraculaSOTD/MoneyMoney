"""
GPU-Accelerated CNN Pattern Recognition with CBAM Attention.

PyTorch implementation with:
- CBAM attention (channel + spatial)
- Multi-timeframe input support
- Mixed precision training
- GPU memory optimization
- 12 pattern classes for crypto trading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils.gpu_manager import GPUManager, get_gpu_manager
from utils.logging_system import get_logger

logger = get_logger(__name__)


class ChannelAttention(nn.Module):
    """
    Channel Attention Module from CBAM.

    Focuses on 'what' is meaningful given an input image.
    Uses both average-pooled and max-pooled features through a shared MLP.
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Initialize Channel Attention.

        Args:
            channels: Number of input channels
            reduction: Channel reduction ratio for MLP
        """
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP: channels -> channels//reduction -> channels
        reduced_channels = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of channel attention.

        Args:
            x: Input tensor (batch, channels, height, width)

        Returns:
            Attention-weighted tensor
        """
        b, c, _, _ = x.size()

        # Average pool branch
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.mlp(avg_out)

        # Max pool branch
        max_out = self.max_pool(x).view(b, c)
        max_out = self.mlp(max_out)

        # Combine with sigmoid
        attention = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)

        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module from CBAM.

    Focuses on 'where' is an informative part.
    Uses channel-wise pooled features through convolution.
    """

    def __init__(self, kernel_size: int = 7):
        """
        Initialize Spatial Attention.

        Args:
            kernel_size: Convolution kernel size (7 recommended)
        """
        super(SpatialAttention, self).__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of spatial attention.

        Args:
            x: Input tensor (batch, channels, height, width)

        Returns:
            Attention-weighted tensor
        """
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and convolve
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(combined))

        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Combines channel attention and spatial attention sequentially
    to refine feature maps both in channel and spatial dimensions.
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        """
        Initialize CBAM.

        Args:
            channels: Number of input channels
            reduction: Channel reduction ratio
            kernel_size: Spatial attention kernel size
        """
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CBAM.

        Args:
            x: Input tensor

        Returns:
            Refined tensor
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ConvBlockWithCBAM(nn.Module):
    """
    Convolutional block with optional CBAM attention.

    Structure: Conv2d -> BatchNorm -> ReLU -> CBAM (optional) -> MaxPool
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 use_cbam: bool = True,
                 pool_size: int = 2,
                 dropout: float = 0.0):
        """
        Initialize ConvBlock.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            use_cbam: Whether to use CBAM attention
            pool_size: Max pooling size
            dropout: Dropout rate
        """
        super(ConvBlockWithCBAM, self).__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels) if use_cbam else nn.Identity()
        self.pool = nn.MaxPool2d(pool_size, pool_size)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CNNPatternGPU(nn.Module):
    """
    GPU-accelerated CNN for pattern recognition with CBAM attention.

    Architecture:
    - 4 Conv blocks with CBAM: 32 -> 64 -> 128 -> 256 filters
    - Global Average Pooling (more robust than flatten)
    - FC layers: 256 -> 128 -> num_classes
    - Dropout for regularization

    Supports 12 crypto trading pattern classes.
    """

    # Pattern class names for reference
    PATTERN_NAMES = [
        'no_pattern',           # 0 - No recognizable pattern
        'double_bottom',        # 1 - Bullish reversal (W pattern)
        'double_top',           # 2 - Bearish reversal (M pattern)
        'head_shoulders',       # 3 - Bearish reversal (H&S)
        'inv_head_shoulders',   # 4 - Bullish reversal (Inverse H&S)
        'bull_flag',            # 5 - Bullish continuation
        'bear_flag',            # 6 - Bearish continuation
        'ascending_triangle',   # 7 - Bullish (rising lows, flat highs)
        'descending_triangle',  # 8 - Bearish (falling highs, flat lows)
        'symmetrical_triangle', # 9 - Neutral consolidation
        'cup_handle',           # 10 - Bullish (U-shape + handle)
        'consolidation'         # 11 - Range-bound sideways
    ]

    # Pattern to signal mapping
    PATTERN_SIGNALS = {
        0: 'hold',      # no_pattern
        1: 'buy',       # double_bottom
        2: 'sell',      # double_top
        3: 'sell',      # head_shoulders
        4: 'buy',       # inv_head_shoulders
        5: 'buy',       # bull_flag
        6: 'sell',      # bear_flag
        7: 'buy',       # ascending_triangle
        8: 'sell',      # descending_triangle
        9: 'hold',      # symmetrical_triangle
        10: 'buy',      # cup_handle
        11: 'hold'      # consolidation
    }

    def __init__(self,
                 input_channels: int = 6,
                 num_classes: int = 12,
                 image_size: int = 64,
                 use_cbam: bool = True,
                 dropout: float = 0.3,
                 base_filters: int = 32):
        """
        Initialize CNN Pattern GPU model.

        Args:
            input_channels: Number of input image channels (GAF + RP + OHLC)
            num_classes: Number of pattern classes (default 12)
            image_size: Input image size (default 64x64)
            use_cbam: Whether to use CBAM attention
            dropout: Dropout rate
            base_filters: Base number of filters (doubles each block)
        """
        super(CNNPatternGPU, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.use_cbam = use_cbam

        # Encoder blocks: progressively increase filters
        # Block 1: input -> 32 filters, 64x64 -> 32x32
        self.block1 = ConvBlockWithCBAM(
            input_channels, base_filters,
            kernel_size=5, use_cbam=use_cbam, dropout=dropout*0.5
        )

        # Block 2: 32 -> 64 filters, 32x32 -> 16x16
        self.block2 = ConvBlockWithCBAM(
            base_filters, base_filters*2,
            kernel_size=3, use_cbam=use_cbam, dropout=dropout*0.5
        )

        # Block 3: 64 -> 128 filters, 16x16 -> 8x8
        self.block3 = ConvBlockWithCBAM(
            base_filters*2, base_filters*4,
            kernel_size=3, use_cbam=use_cbam, dropout=dropout*0.5
        )

        # Block 4: 128 -> 256 filters, 8x8 -> 4x4
        self.block4 = ConvBlockWithCBAM(
            base_filters*4, base_filters*8,
            kernel_size=3, use_cbam=use_cbam, dropout=dropout*0.5
        )

        # Global Average Pooling (replaces flatten - more robust to input size)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        classifier_input = base_filters * 8  # 256
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # GPU manager
        self.gpu_manager = get_gpu_manager()

        # Move to GPU
        self.to(self.gpu_manager.device)

        # Initialize weights
        self._init_weights()

        logger.info(f"CNNPatternGPU initialized on {self.gpu_manager.device}")
        logger.info(f"  Input: {input_channels} channels, {image_size}x{image_size}")
        logger.info(f"  Output: {num_classes} classes")
        logger.info(f"  CBAM attention: {use_cbam}")
        logger.info(f"  Parameters: {self.count_parameters():,}")

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning logits, probabilities, and intermediate features.

        Args:
            x: Input tensor (batch, channels, height, width)

        Returns:
            Dictionary with logits, probabilities, predictions, confidence, features
        """
        # Encoder
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)

        # Global pooling
        pooled = self.global_pool(f4).view(x.size(0), -1)

        # Classification
        logits = self.classifier(pooled)
        probs = F.softmax(logits, dim=-1)

        return {
            'logits': logits,
            'probabilities': probs,
            'predictions': torch.argmax(probs, dim=-1),
            'confidence': torch.max(probs, dim=-1)[0],
            'features': {
                'block1': f1,
                'block2': f2,
                'block3': f3,
                'block4': f4,
                'pooled': pooled
            }
        }

    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Make predictions on input images.

        Args:
            x: Input images (numpy array or tensor)

        Returns:
            Predicted class indices as numpy array
        """
        self.eval()

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        x = self.gpu_manager.to_device(x)

        # Add batch dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(0)

        with torch.no_grad():
            with self.gpu_manager.autocast():
                output = self.forward(x)

        return self.gpu_manager.to_numpy(output['predictions'])

    def predict_with_confidence(self, x: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores.

        Args:
            x: Input images

        Returns:
            Tuple of (predictions, confidences)
        """
        self.eval()

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        x = self.gpu_manager.to_device(x)

        if x.dim() == 3:
            x = x.unsqueeze(0)

        with torch.no_grad():
            with self.gpu_manager.autocast():
                output = self.forward(x)

        predictions = self.gpu_manager.to_numpy(output['predictions'])
        confidences = self.gpu_manager.to_numpy(output['confidence'])

        return predictions, confidences

    def get_pattern_confidence(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Get confidence scores for each pattern class.

        Args:
            x: Single input image tensor

        Returns:
            Dictionary mapping pattern names to confidence scores
        """
        self.eval()

        if x.dim() == 3:
            x = x.unsqueeze(0)

        with torch.no_grad():
            output = self.forward(x)

        probs = self.gpu_manager.to_numpy(output['probabilities'])[0]

        return {
            name: float(probs[i])
            for i, name in enumerate(self.PATTERN_NAMES[:self.num_classes])
        }

    def get_trading_signal(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Get trading signal from pattern prediction.

        Args:
            x: Input image tensor

        Returns:
            Dictionary with pattern, signal, and confidence
        """
        predictions, confidences = self.predict_with_confidence(x)

        pattern_idx = int(predictions[0])
        pattern_name = self.PATTERN_NAMES[pattern_idx]
        signal = self.PATTERN_SIGNALS[pattern_idx]
        confidence = float(confidences[0])

        return {
            'pattern': pattern_name,
            'pattern_idx': pattern_idx,
            'signal': signal,
            'confidence': confidence,
            'all_probabilities': self.get_pattern_confidence(x)
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str):
        """
        Save model to file.

        Args:
            path: Save path (will add .pt extension if needed)
        """
        if not path.endswith('.pt'):
            path = path + '.pt'

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.state_dict(),
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'image_size': self.image_size,
            'use_cbam': self.use_cbam,
            'pattern_names': self.PATTERN_NAMES[:self.num_classes]
        }, path)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'CNNPatternGPU':
        """
        Load model from file.

        Args:
            path: Path to saved model

        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location='cpu')

        model = cls(
            input_channels=checkpoint.get('input_channels', 6),
            num_classes=checkpoint.get('num_classes', 12),
            image_size=checkpoint.get('image_size', 64),
            use_cbam=checkpoint.get('use_cbam', True)
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(model.gpu_manager.device)

        logger.info(f"Model loaded from {path}")
        return model

    def profile_performance(self, batch_size: int = 32,
                           num_runs: int = 100) -> Dict[str, float]:
        """
        Profile model performance on GPU.

        Args:
            batch_size: Batch size for profiling
            num_runs: Number of runs

        Returns:
            Performance metrics
        """
        import time

        self.eval()

        # Create dummy input
        dummy_input = torch.randn(
            batch_size, self.input_channels, self.image_size, self.image_size,
            device=self.gpu_manager.device
        )

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.forward(dummy_input)

        self.gpu_manager.synchronize()

        # Profile
        forward_times = []

        for _ in range(num_runs):
            self.gpu_manager.clear_cache()

            start_time = time.time()
            with torch.no_grad():
                with self.gpu_manager.autocast():
                    _ = self.forward(dummy_input)
            self.gpu_manager.synchronize()
            end_time = time.time()

            forward_times.append(end_time - start_time)

        return {
            "mean_forward_time": np.mean(forward_times),
            "std_forward_time": np.std(forward_times),
            "throughput": batch_size / np.mean(forward_times),
            "parameters": self.count_parameters(),
            "device": str(self.gpu_manager.device),
            "mixed_precision": self.gpu_manager.enable_mixed_precision
        }


def create_cnn_pattern_gpu(config: Dict[str, Any] = None) -> CNNPatternGPU:
    """
    Factory function to create CNN Pattern GPU model.

    Args:
        config: Model configuration dictionary

    Returns:
        CNNPatternGPU model
    """
    config = config or {}

    model = CNNPatternGPU(
        input_channels=config.get('input_channels', 6),
        num_classes=config.get('num_classes', 12),
        image_size=config.get('image_size', 64),
        use_cbam=config.get('use_cbam', True),
        dropout=config.get('dropout', 0.3),
        base_filters=config.get('base_filters', 32)
    )

    # Profile model
    profile = model.profile_performance(batch_size=32)
    logger.info(f"Model profile: throughput={profile['throughput']:.1f} samples/sec, "
               f"params={profile['parameters']:,}")

    return model
