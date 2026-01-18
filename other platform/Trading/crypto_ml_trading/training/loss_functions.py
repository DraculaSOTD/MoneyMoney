"""
Custom Loss Functions
Trading-specific loss functions for improved model training.
"""

from typing import Optional, Union
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reduces the relative loss for well-classified examples, focusing more
    on hard, misclassified examples.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(
        self,
        alpha: Union[float, torch.Tensor] = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor. Can be scalar or tensor of shape (n_classes,)
            gamma: Focusing parameter. Higher = more focus on hard examples
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Model predictions (batch_size, n_classes) - logits
            targets: Ground truth labels (batch_size,)

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t

        # Apply alpha weighting if per-class
        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha[targets]
        else:
            alpha_t = self.alpha

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross-Entropy Loss.

    Instead of one-hot labels, uses soft labels:
    y_smooth = (1 - smoothing) * y_one_hot + smoothing / n_classes

    This prevents the model from becoming overconfident and improves
    generalization.
    """

    def __init__(
        self,
        n_classes: int,
        smoothing: float = 0.1,
        reduction: str = 'mean',
    ):
        """
        Initialize Label Smoothing Loss.

        Args:
            n_classes: Number of classes
            smoothing: Label smoothing factor (0-1)
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.n_classes = n_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.

        Args:
            pred: Model predictions (batch_size, n_classes) - logits
            target: Ground truth labels (batch_size,)

        Returns:
            Label smoothing loss value
        """
        log_probs = pred.log_softmax(dim=-1)

        # Create smoothed target distribution
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.n_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        # KL divergence (equivalent to cross-entropy with soft labels)
        loss = torch.sum(-true_dist * log_probs, dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss with class weights computed from data.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        n_classes: Optional[int] = None,
        reduction: str = 'mean',
    ):
        """
        Initialize Weighted CE Loss.

        Args:
            class_weights: Pre-computed class weights
            n_classes: Number of classes (for computing weights from labels)
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.class_weights = class_weights
        self.n_classes = n_classes
        self.reduction = reduction
        self._loss_fn = None

    def compute_weights_from_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute class weights inversely proportional to frequency.

        Args:
            labels: All training labels

        Returns:
            Tensor of class weights
        """
        unique, counts = torch.unique(labels, return_counts=True)
        weights = 1.0 / counts.float()
        weights = weights / weights.sum() * len(unique)  # Normalize

        # Handle missing classes
        full_weights = torch.ones(self.n_classes or unique.max().item() + 1)
        full_weights[unique] = weights

        return full_weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.

        Args:
            inputs: Model predictions (batch_size, n_classes)
            targets: Ground truth labels (batch_size,)

        Returns:
            Loss value
        """
        if self._loss_fn is None:
            self._loss_fn = nn.CrossEntropyLoss(
                weight=self.class_weights,
                reduction=self.reduction
            )

        return self._loss_fn(inputs, targets)


class TradingLoss(nn.Module):
    """
    Trading-aware loss function.

    Combines cross-entropy with a penalty for wrong direction predictions.
    This is particularly important for trading where direction matters more
    than exact price predictions.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        direction_weight: float = 0.3,
        confidence_weight: float = 0.1,
        focal_gamma: float = 0.0,  # 0 = no focal loss
        label_smoothing: float = 0.0,
        reduction: str = 'mean',
    ):
        """
        Initialize Trading Loss.

        Args:
            class_weights: Optional class weights for CE loss
            direction_weight: Weight for direction penalty
            confidence_weight: Weight for confidence calibration
            focal_gamma: Focal loss gamma (0 to disable)
            label_smoothing: Label smoothing factor
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.class_weights = class_weights
        self.direction_weight = direction_weight
        self.confidence_weight = confidence_weight
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        # Base loss
        if label_smoothing > 0:
            self.ce_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                reduction='none',
                label_smoothing=label_smoothing
            )
        else:
            self.ce_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                reduction='none'
            )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute trading loss.

        Args:
            pred: Model predictions (batch_size, n_classes) - logits
            target: Ground truth labels (batch_size,)
            returns: Actual returns for direction penalty (batch_size,)

        Returns:
            Loss value
        """
        # Base cross-entropy loss
        ce = self.ce_loss(pred, target)

        # Apply focal modulation if enabled
        if self.focal_gamma > 0:
            pt = torch.exp(-ce)
            ce = (1 - pt) ** self.focal_gamma * ce

        loss = ce

        # Direction penalty
        if returns is not None and self.direction_weight > 0:
            pred_direction = torch.argmax(pred, dim=1)

            # Assuming class 0 = down, 1 = neutral, 2 = up
            # Or for binary: 0 = down, 1 = up
            n_classes = pred.shape[1]

            if n_classes == 2:
                actual_direction = (returns > 0).long()
            else:
                # For 3-class: 0=down, 1=neutral, 2=up
                actual_direction = torch.where(
                    returns < -0.001,
                    torch.zeros_like(returns, dtype=torch.long),
                    torch.where(
                        returns > 0.001,
                        torch.full_like(returns, 2, dtype=torch.long),
                        torch.ones_like(returns, dtype=torch.long)
                    )
                )

            # Penalize wrong direction more when actual move is significant
            direction_wrong = (pred_direction != actual_direction).float()
            magnitude = returns.abs()
            direction_penalty = direction_wrong * (1 + magnitude * 10)  # Scale by magnitude

            loss = loss + self.direction_weight * direction_penalty

        # Confidence calibration penalty
        if self.confidence_weight > 0:
            probs = F.softmax(pred, dim=1)
            confidence = probs.max(dim=1)[0]
            correct = (torch.argmax(pred, dim=1) == target).float()

            # Penalize high confidence when wrong, low confidence when right
            calibration_penalty = torch.abs(confidence - correct)
            loss = loss + self.confidence_weight * calibration_penalty

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SoftmaxFocalLoss(nn.Module):
    """
    Focal Loss that works with soft targets (for mixup/cutmix).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss with soft targets.

        Args:
            inputs: Logits (batch_size, n_classes)
            targets: Soft targets (batch_size, n_classes) or hard (batch_size,)

        Returns:
            Loss value
        """
        # Handle hard labels
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=inputs.shape[-1]).float()

        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)

        # Focal weight: (1 - p)^gamma
        focal_weight = (1 - probs) ** self.gamma

        # Cross entropy with focal weighting
        loss = -focal_weight * targets * log_probs
        loss = loss.sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def create_loss_function(
    loss_type: str = 'cross_entropy',
    n_classes: int = 3,
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    direction_weight: float = 0.0,
) -> nn.Module:
    """
    Factory function to create loss functions.

    Args:
        loss_type: Type of loss ('cross_entropy', 'focal', 'label_smoothing',
                   'weighted', 'trading')
        n_classes: Number of classes
        class_weights: Optional class weights
        focal_gamma: Gamma for focal loss
        label_smoothing: Smoothing factor for label smoothing
        direction_weight: Weight for direction penalty in trading loss

    Returns:
        Configured loss function
    """
    if loss_type == 'focal':
        return FocalLoss(gamma=focal_gamma)

    elif loss_type == 'label_smoothing':
        return LabelSmoothingLoss(n_classes=n_classes, smoothing=label_smoothing)

    elif loss_type == 'weighted':
        return WeightedCrossEntropyLoss(class_weights=class_weights, n_classes=n_classes)

    elif loss_type == 'trading':
        return TradingLoss(
            class_weights=class_weights,
            direction_weight=direction_weight,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )

    elif loss_type == 'soft_focal':
        return SoftmaxFocalLoss(gamma=focal_gamma)

    else:  # Default cross_entropy
        return nn.CrossEntropyLoss(weight=class_weights)
