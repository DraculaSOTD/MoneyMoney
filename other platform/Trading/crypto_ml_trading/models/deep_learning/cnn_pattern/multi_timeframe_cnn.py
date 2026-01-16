"""
Multi-Timeframe CNN Pattern Recognition.

Processes patterns from multiple timeframes (1m, 5m, 15m, 1h) and fuses
features using a Feature Pyramid Network (FPN) style architecture.

This enables the model to capture patterns at different time scales
simultaneously for more robust trading signal generation.
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
from utils.gpu_manager import get_gpu_manager
from utils.logging_system import get_logger

from .cnn_pattern_gpu import CBAM, ConvBlockWithCBAM, CNNPatternGPU

logger = get_logger(__name__)


class TimeframeEncoder(nn.Module):
    """
    Encoder for a single timeframe with CBAM attention.

    Produces multi-scale feature maps for later fusion.
    """

    def __init__(self,
                 input_channels: int = 6,
                 base_filters: int = 32,
                 use_cbam: bool = True,
                 dropout: float = 0.2):
        """
        Initialize timeframe encoder.

        Args:
            input_channels: Number of input channels
            base_filters: Base number of filters
            use_cbam: Use CBAM attention
            dropout: Dropout rate
        """
        super(TimeframeEncoder, self).__init__()

        # 3 conv blocks producing multi-scale features
        self.block1 = ConvBlockWithCBAM(
            input_channels, base_filters,
            kernel_size=5, use_cbam=use_cbam, dropout=dropout
        )  # -> 32x32

        self.block2 = ConvBlockWithCBAM(
            base_filters, base_filters*2,
            kernel_size=3, use_cbam=use_cbam, dropout=dropout
        )  # -> 16x16

        self.block3 = ConvBlockWithCBAM(
            base_filters*2, base_filters*4,
            kernel_size=3, use_cbam=use_cbam, dropout=dropout
        )  # -> 8x8

        self.output_channels = base_filters * 4

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning multi-scale features.

        Args:
            x: Input tensor (batch, channels, H, W)

        Returns:
            Dict with features at each scale
        """
        f1 = self.block1(x)  # (batch, 32, 32, 32)
        f2 = self.block2(f1)  # (batch, 64, 16, 16)
        f3 = self.block3(f2)  # (batch, 128, 8, 8)

        return {
            'scale1': f1,
            'scale2': f2,
            'scale3': f3
        }


class CrossTimeframeAttention(nn.Module):
    """
    Attention mechanism for fusing features across timeframes.

    Learns to weight the importance of each timeframe adaptively.
    """

    def __init__(self, channels: int, num_timeframes: int):
        """
        Initialize cross-timeframe attention.

        Args:
            channels: Feature channels
            num_timeframes: Number of timeframes
        """
        super(CrossTimeframeAttention, self).__init__()

        self.num_timeframes = num_timeframes

        # Global average pooling + FC to produce attention weights
        self.attention_fc = nn.Sequential(
            nn.Linear(channels * num_timeframes, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, num_timeframes),
            nn.Softmax(dim=1)
        )

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted fusion of timeframe features.

        Args:
            features: List of feature tensors from each timeframe

        Returns:
            Tuple of (fused features, attention weights)
        """
        batch_size = features[0].size(0)

        # Global average pool each feature map
        pooled = []
        for f in features:
            pooled.append(F.adaptive_avg_pool2d(f, 1).view(batch_size, -1))

        # Concatenate and compute attention
        concat = torch.cat(pooled, dim=1)
        attention_weights = self.attention_fc(concat)  # (batch, num_timeframes)

        # Apply attention weights
        fused = torch.zeros_like(features[0])
        for i, f in enumerate(features):
            weight = attention_weights[:, i].view(batch_size, 1, 1, 1)
            fused = fused + f * weight

        return fused, attention_weights


class FeaturePyramidFusion(nn.Module):
    """
    FPN-style feature fusion across timeframes and scales.

    Combines features from different timeframes using lateral connections
    and top-down pathway for multi-scale integration.
    """

    def __init__(self, channels: int = 128, num_timeframes: int = 4):
        """
        Initialize feature pyramid fusion.

        Args:
            channels: Number of channels for fusion
            num_timeframes: Number of timeframes
        """
        super(FeaturePyramidFusion, self).__init__()

        self.num_timeframes = num_timeframes
        self.channels = channels

        # Lateral connections to standardize channels
        self.lateral_conv = nn.Conv2d(channels, channels, kernel_size=1)

        # Cross-timeframe attention at each scale
        self.scale_attention = CrossTimeframeAttention(channels, num_timeframes)

        # Output projection
        self.output_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, timeframe_features: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Fuse features from multiple timeframes.

        Args:
            timeframe_features: Dict mapping timeframe -> scale features

        Returns:
            Fused feature tensor
        """
        # Get list of timeframes with features
        timeframes = list(timeframe_features.keys())

        # Extract scale3 features (highest level) from each timeframe
        scale3_features = [timeframe_features[tf]['scale3'] for tf in timeframes]

        # Apply cross-timeframe attention
        fused, attention = self.scale_attention(scale3_features)

        # Apply output projection
        output = self.output_conv(fused)

        return output


class MultiTimeframeCNNPattern(nn.Module):
    """
    Multi-timeframe CNN pattern recognizer.

    Processes patterns from multiple timeframes simultaneously and fuses
    them using cross-timeframe attention for robust pattern recognition.

    Features:
    - Separate encoder per timeframe
    - Cross-timeframe attention fusion
    - FPN-style multi-scale integration
    - GPU optimized with mixed precision
    """

    TIMEFRAMES = ['1m', '5m', '15m', '1h']

    PATTERN_NAMES = CNNPatternGPU.PATTERN_NAMES
    PATTERN_SIGNALS = CNNPatternGPU.PATTERN_SIGNALS

    def __init__(self,
                 input_channels: int = 6,
                 num_classes: int = 12,
                 image_size: int = 64,
                 timeframes: List[str] = None,
                 use_cbam: bool = True,
                 dropout: float = 0.3,
                 base_filters: int = 32):
        """
        Initialize multi-timeframe CNN.

        Args:
            input_channels: Number of input channels per timeframe
            num_classes: Number of output classes
            image_size: Input image size
            timeframes: List of timeframes to process
            use_cbam: Use CBAM attention
            dropout: Dropout rate
            base_filters: Base filter count
        """
        super(MultiTimeframeCNNPattern, self).__init__()

        self.timeframes = timeframes or self.TIMEFRAMES
        self.num_timeframes = len(self.timeframes)
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.image_size = image_size

        # Separate encoder for each timeframe
        self.encoders = nn.ModuleDict({
            tf: TimeframeEncoder(input_channels, base_filters, use_cbam, dropout)
            for tf in self.timeframes
        })

        # Feature fusion
        encoder_output_channels = base_filters * 4  # 128
        self.fusion = FeaturePyramidFusion(
            channels=encoder_output_channels,
            num_timeframes=self.num_timeframes
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(encoder_output_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # GPU manager
        self.gpu_manager = get_gpu_manager()
        self.to(self.gpu_manager.device)

        # Initialize weights
        self._init_weights()

        logger.info(f"MultiTimeframeCNNPattern initialized on {self.gpu_manager.device}")
        logger.info(f"  Timeframes: {self.timeframes}")
        logger.info(f"  Parameters: {self.count_parameters():,}")

    def _init_weights(self):
        """Initialize weights."""
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

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Forward pass with multi-timeframe inputs.

        Args:
            inputs: Dict mapping timeframe -> image tensor
                    e.g., {'1m': tensor, '5m': tensor, '15m': tensor, '1h': tensor}
                    Each tensor has shape (batch, channels, H, W)

        Returns:
            Dict with logits, probabilities, predictions, confidence, and features
        """
        # Encode each timeframe
        encoded_features = {}
        for tf in self.timeframes:
            if tf in inputs:
                encoded_features[tf] = self.encoders[tf](inputs[tf])

        # Handle case where not all timeframes are provided
        if len(encoded_features) == 0:
            raise ValueError("At least one timeframe must be provided")

        # If only one timeframe, use it directly without fusion
        if len(encoded_features) == 1:
            tf = list(encoded_features.keys())[0]
            fused = encoded_features[tf]['scale3']
        else:
            # Fuse features across timeframes
            fused = self.fusion(encoded_features)

        # Global pooling
        pooled = self.global_pool(fused).view(fused.size(0), -1)

        # Classification
        logits = self.classifier(pooled)
        probs = F.softmax(logits, dim=-1)

        return {
            'logits': logits,
            'probabilities': probs,
            'predictions': torch.argmax(probs, dim=-1),
            'confidence': torch.max(probs, dim=-1)[0],
            'timeframe_features': encoded_features,
            'fused_features': fused
        }

    def predict(self, inputs: Dict[str, Union[np.ndarray, torch.Tensor]]) -> np.ndarray:
        """
        Make predictions on multi-timeframe inputs.

        Args:
            inputs: Dict mapping timeframe -> images

        Returns:
            Predicted class indices
        """
        self.eval()

        # Convert to tensors and move to device
        tensor_inputs = {}
        for tf, data in inputs.items():
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            tensor_inputs[tf] = self.gpu_manager.to_device(data)

        with torch.no_grad():
            with self.gpu_manager.autocast():
                output = self.forward(tensor_inputs)

        return self.gpu_manager.to_numpy(output['predictions'])

    def predict_single_timeframe(self, x: Union[np.ndarray, torch.Tensor],
                                 timeframe: str = '1m') -> np.ndarray:
        """
        Predict using only a single timeframe (fallback mode).

        Args:
            x: Input images
            timeframe: Which timeframe these images represent

        Returns:
            Predictions
        """
        return self.predict({timeframe: x})

    def get_trading_signal(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Get trading signal from pattern prediction.

        Args:
            inputs: Multi-timeframe inputs

        Returns:
            Trading signal with pattern, signal type, and confidence
        """
        self.eval()

        with torch.no_grad():
            output = self.forward(inputs)

        pred_idx = int(output['predictions'][0])
        pattern_name = self.PATTERN_NAMES[pred_idx]
        signal = self.PATTERN_SIGNALS[pred_idx]
        confidence = float(output['confidence'][0])

        return {
            'pattern': pattern_name,
            'pattern_idx': pred_idx,
            'signal': signal,
            'confidence': confidence
        }

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str):
        """Save model to file."""
        if not path.endswith('.pt'):
            path = path + '.pt'

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.state_dict(),
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'image_size': self.image_size,
            'timeframes': self.timeframes,
            'pattern_names': self.PATTERN_NAMES[:self.num_classes]
        }, path)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'MultiTimeframeCNNPattern':
        """Load model from file."""
        checkpoint = torch.load(path, map_location='cpu')

        model = cls(
            input_channels=checkpoint.get('input_channels', 6),
            num_classes=checkpoint.get('num_classes', 12),
            image_size=checkpoint.get('image_size', 64),
            timeframes=checkpoint.get('timeframes', cls.TIMEFRAMES)
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(model.gpu_manager.device)

        logger.info(f"Model loaded from {path}")
        return model


def create_multi_timeframe_cnn(config: Dict[str, Any] = None) -> MultiTimeframeCNNPattern:
    """
    Factory function to create multi-timeframe CNN model.

    Args:
        config: Model configuration

    Returns:
        MultiTimeframeCNNPattern model
    """
    config = config or {}

    model = MultiTimeframeCNNPattern(
        input_channels=config.get('input_channels', 6),
        num_classes=config.get('num_classes', 12),
        image_size=config.get('image_size', 64),
        timeframes=config.get('timeframes', ['1m', '5m', '15m', '1h']),
        use_cbam=config.get('use_cbam', True),
        dropout=config.get('dropout', 0.3),
        base_filters=config.get('base_filters', 32)
    )

    return model
