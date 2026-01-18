"""
Time Series Data Augmentation
Augmentation techniques for financial time series data.
"""

from typing import Tuple, Optional, Union, List
from dataclasses import dataclass, field
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    # Gaussian noise
    noise_level: float = 0.01  # Standard deviation as fraction of data std
    noise_prob: float = 0.5  # Probability of applying noise

    # Magnitude scaling
    scale_range: Tuple[float, float] = (0.95, 1.05)
    scale_prob: float = 0.5

    # Mixup
    mixup_alpha: float = 0.2  # Beta distribution parameter
    mixup_prob: float = 0.3

    # CutMix temporal
    cutmix_prob: float = 0.3
    cutmix_min_ratio: float = 0.25  # Minimum cut length as fraction of sequence
    cutmix_max_ratio: float = 0.5  # Maximum cut length as fraction of sequence

    # Window slicing (random subsequence)
    window_slice_prob: float = 0.0  # Disabled by default
    window_slice_min_ratio: float = 0.8

    # Time warping (stretch/compress segments)
    time_warp_prob: float = 0.0  # Disabled by default - can distort patterns
    time_warp_sigma: float = 0.2

    # Feature dropout
    feature_dropout_prob: float = 0.1
    feature_dropout_rate: float = 0.1


class TimeSeriesAugmentation:
    """
    Data augmentation techniques for financial time series.

    Designed to be applied during training to improve model generalization.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize augmentation with configuration.

        Args:
            config: Augmentation configuration (uses defaults if None)
        """
        self.config = config or AugmentationConfig()

    def gaussian_noise(self, x: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise scaled by feature standard deviation.

        Args:
            x: Input array of shape (seq_len, n_features) or (batch, seq_len, n_features)

        Returns:
            Augmented array with same shape
        """
        # Calculate std per feature (last axis)
        std = np.std(x, axis=-2, keepdims=True) + 1e-8
        noise = np.random.normal(0, self.config.noise_level, x.shape) * std
        return x + noise

    def magnitude_scaling(self, x: np.ndarray) -> np.ndarray:
        """
        Apply random magnitude scaling to the entire sequence.

        Args:
            x: Input array

        Returns:
            Scaled array
        """
        scale = np.random.uniform(*self.config.scale_range)
        return x * scale

    def feature_wise_scaling(self, x: np.ndarray) -> np.ndarray:
        """
        Apply random scaling per feature (useful for multi-feature inputs).

        Args:
            x: Input array of shape (seq_len, n_features) or (batch, seq_len, n_features)

        Returns:
            Scaled array
        """
        n_features = x.shape[-1]
        scales = np.random.uniform(
            self.config.scale_range[0],
            self.config.scale_range[1],
            size=n_features
        )
        return x * scales

    def mixup(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        y1: Union[int, np.ndarray],
        y2: Union[int, np.ndarray],
    ) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
        """
        Mixup augmentation - interpolate between two samples.

        Args:
            x1: First input sample
            x2: Second input sample
            y1: First label (int for class, array for soft labels)
            y2: Second label

        Returns:
            Tuple of (mixed_x, mixed_y)
        """
        lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
        x_mixed = lam * x1 + (1 - lam) * x2

        # Handle different label types
        if isinstance(y1, (int, np.integer)) and isinstance(y2, (int, np.integer)):
            # For hard labels, create soft labels
            y_mixed = lam * y1 + (1 - lam) * y2
        else:
            y_mixed = lam * np.array(y1) + (1 - lam) * np.array(y2)

        return x_mixed, y_mixed

    def cutmix_temporal(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        y1: Union[int, np.ndarray],
        y2: Union[int, np.ndarray],
    ) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
        """
        CutMix for temporal sequences - replace a segment from x1 with x2.

        Args:
            x1: First input sample (seq_len, n_features)
            x2: Second input sample
            y1: First label
            y2: Second label

        Returns:
            Tuple of (mixed_x, mixed_y)
        """
        seq_len = x1.shape[0]

        # Determine cut length
        min_cut = int(seq_len * self.config.cutmix_min_ratio)
        max_cut = int(seq_len * self.config.cutmix_max_ratio)
        cut_len = np.random.randint(min_cut, max_cut + 1)

        # Random start position
        cut_start = np.random.randint(0, seq_len - cut_len + 1)

        # Create mixed sample
        x_mixed = x1.copy()
        x_mixed[cut_start:cut_start + cut_len] = x2[cut_start:cut_start + cut_len]

        # Mix labels proportionally
        lam = 1 - (cut_len / seq_len)

        if isinstance(y1, (int, np.integer)) and isinstance(y2, (int, np.integer)):
            y_mixed = lam * y1 + (1 - lam) * y2
        else:
            y_mixed = lam * np.array(y1) + (1 - lam) * np.array(y2)

        return x_mixed, y_mixed

    def window_slicing(self, x: np.ndarray) -> np.ndarray:
        """
        Extract a random window (subsequence) and resize to original length.

        Note: This can distort temporal patterns - use with caution.

        Args:
            x: Input array (seq_len, n_features)

        Returns:
            Resized subsequence
        """
        seq_len = x.shape[0]
        min_len = int(seq_len * self.config.window_slice_min_ratio)

        # Random window length and start
        window_len = np.random.randint(min_len, seq_len + 1)
        start = np.random.randint(0, seq_len - window_len + 1)

        # Extract window
        window = x[start:start + window_len]

        # Resize back to original length using linear interpolation
        indices = np.linspace(0, window_len - 1, seq_len)
        result = np.zeros_like(x)
        for i in range(x.shape[-1]):
            result[:, i] = np.interp(indices, np.arange(window_len), window[:, i])

        return result

    def feature_dropout(self, x: np.ndarray) -> np.ndarray:
        """
        Randomly zero out some features (like dropout but on features).

        Args:
            x: Input array (..., n_features)

        Returns:
            Array with some features zeroed
        """
        n_features = x.shape[-1]
        mask = np.random.random(n_features) > self.config.feature_dropout_rate
        return x * mask

    def __call__(
        self,
        x: np.ndarray,
        augment_prob: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply random augmentation to input.

        Args:
            x: Input array (seq_len, n_features) or (batch, seq_len, n_features)
            augment_prob: Override default augmentation probability

        Returns:
            Augmented array (same shape as input)
        """
        # Check if we should augment at all
        if augment_prob is not None:
            if np.random.random() > augment_prob:
                return x

        result = x.copy()

        # Apply noise
        if np.random.random() < self.config.noise_prob:
            result = self.gaussian_noise(result)

        # Apply magnitude scaling
        if np.random.random() < self.config.scale_prob:
            result = self.magnitude_scaling(result)

        # Apply feature dropout
        if np.random.random() < self.config.feature_dropout_prob:
            result = self.feature_dropout(result)

        return result


class TorchAugmentation:
    """
    PyTorch-compatible augmentation for use in DataLoader.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.augmenter = TimeSeriesAugmentation(config)
        self.config = config or AugmentationConfig()

    def augment_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mixup: bool = True,
        cutmix: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Augment a batch of samples with optional mixup/cutmix.

        Args:
            x: Batch tensor (batch, seq_len, n_features)
            y: Labels tensor (batch,) or (batch, n_classes)
            mixup: Whether to apply mixup
            cutmix: Whether to apply cutmix

        Returns:
            Tuple of (augmented_x, augmented_y)
        """
        device = x.device
        batch_size = x.shape[0]

        # Convert to numpy for augmentation
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()

        # Apply per-sample augmentations
        for i in range(batch_size):
            x_np[i] = self.augmenter(x_np[i], augment_prob=0.5)

        # Apply batch-level augmentations (mixup/cutmix)
        if mixup and np.random.random() < self.config.mixup_prob:
            indices = np.random.permutation(batch_size)
            x_np, y_np = self._batch_mixup(x_np, x_np[indices], y_np, y_np[indices])

        elif cutmix and np.random.random() < self.config.cutmix_prob:
            indices = np.random.permutation(batch_size)
            x_np, y_np = self._batch_cutmix(x_np, x_np[indices], y_np, y_np[indices])

        return torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)

    def _batch_mixup(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply mixup to entire batch."""
        lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
        x_mixed = lam * x1 + (1 - lam) * x2
        y_mixed = lam * y1 + (1 - lam) * y2
        return x_mixed, y_mixed

    def _batch_cutmix(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply cutmix to entire batch."""
        batch_size, seq_len, _ = x1.shape

        min_cut = int(seq_len * self.config.cutmix_min_ratio)
        max_cut = int(seq_len * self.config.cutmix_max_ratio)
        cut_len = np.random.randint(min_cut, max_cut + 1)
        cut_start = np.random.randint(0, seq_len - cut_len + 1)

        x_mixed = x1.copy()
        x_mixed[:, cut_start:cut_start + cut_len] = x2[:, cut_start:cut_start + cut_len]

        lam = 1 - (cut_len / seq_len)
        y_mixed = lam * y1 + (1 - lam) * y2

        return x_mixed, y_mixed


def create_augmentation(
    noise_level: float = 0.01,
    scale_range: Tuple[float, float] = (0.95, 1.05),
    mixup_alpha: float = 0.2,
    use_mixup: bool = True,
    use_cutmix: bool = True,
) -> TimeSeriesAugmentation:
    """
    Create augmentation with custom settings.

    Args:
        noise_level: Gaussian noise level
        scale_range: Min/max for magnitude scaling
        mixup_alpha: Mixup beta distribution parameter
        use_mixup: Whether to enable mixup
        use_cutmix: Whether to enable cutmix

    Returns:
        Configured TimeSeriesAugmentation instance
    """
    config = AugmentationConfig(
        noise_level=noise_level,
        scale_range=scale_range,
        mixup_alpha=mixup_alpha,
        mixup_prob=0.3 if use_mixup else 0.0,
        cutmix_prob=0.3 if use_cutmix else 0.0,
    )
    return TimeSeriesAugmentation(config)
