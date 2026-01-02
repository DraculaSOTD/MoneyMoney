"""
Enhanced CNN Pattern Recognition Trainer

Features:
- Advanced data augmentation for financial patterns
- Progressive training with curriculum learning
- Multi-scale pattern detection
- Mixup and CutMix augmentation
- Class-balanced sampling
- Pattern-specific metrics
- Visualization tools
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import time
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.base_trainer import BaseTrainer, CosineAnnealingScheduler, StepLRScheduler
from models.deep_learning.cnn_pattern.cnn_model import CNNPatternRecognizer
from models.deep_learning.cnn_pattern.pattern_generator import PatternGenerator


class EnhancedCNNPatternTrainer(BaseTrainer):
    """
    Enhanced trainer for CNN pattern recognition with advanced features.
    """
    
    def __init__(self,
                 model: CNNPatternRecognizer,
                 optimizer: str = 'adamw',
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.01,
                 gradient_clip: float = 1.0,
                 batch_size: int = 32,
                 validation_split: float = 0.2,
                 early_stopping_patience: int = 20,
                 checkpoint_dir: Optional[str] = None,
                 use_mixup: bool = True,
                 use_cutmix: bool = True,
                 class_balance: bool = True,
                 progressive_training: bool = True,
                 augmentation_probability: float = 0.5):
        """
        Initialize enhanced CNN pattern trainer.
        
        Args:
            model: CNN pattern recognition model
            optimizer: Optimizer choice
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            gradient_clip: Gradient clipping threshold
            batch_size: Training batch size
            validation_split: Validation data fraction
            early_stopping_patience: Early stopping patience
            checkpoint_dir: Checkpoint directory
            use_mixup: Whether to use Mixup augmentation
            use_cutmix: Whether to use CutMix augmentation
            class_balance: Whether to balance classes
            progressive_training: Whether to use curriculum learning
            augmentation_probability: Probability of applying augmentation
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
            batch_size=batch_size,
            validation_split=validation_split,
            early_stopping_patience=early_stopping_patience,
            checkpoint_dir=checkpoint_dir
        )
        
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.class_balance = class_balance
        self.progressive_training = progressive_training
        self.augmentation_probability = augmentation_probability
        
        # Pattern generator for creating images
        self.pattern_generator = PatternGenerator(
            image_size=model.image_size,
            methods=['gasf', 'gadf', 'rp', 'mtf']
        )
        
        # Class weights for balanced training
        self.class_weights = None
        
        # Progressive training state
        self.current_difficulty = 0.0
        self.difficulty_schedule = np.linspace(0.3, 1.0, 50)  # Gradually increase difficulty
        
        # Pattern-specific metrics
        self.pattern_confusion_matrix = None
        self.pattern_detection_rates = {}
        
    def prepare_data(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Prepare data for CNN pattern training.
        
        Args:
            data: Raw OHLCV data or pre-generated pattern images
            
        Returns:
            Dictionary with pattern images and labels
        """
        if isinstance(data, pd.DataFrame):
            # Generate pattern images from OHLCV data
            window_size = 60  # Default window size for patterns
            images, labels = self._generate_pattern_data(data, window_size)
        else:
            # Assume pre-generated images
            if len(data.shape) == 2:
                # Flat images, need to reshape
                n_samples = data.shape[0]
                image_size = int(np.sqrt(data.shape[1] // 3))  # Assume 3 channels
                images = data.reshape(n_samples, 3, image_size, image_size)
                labels = np.zeros(n_samples, dtype=int)  # Placeholder
            else:
                # Already shaped correctly
                images = data[:, :-1] if data.shape[1] > 3 else data
                labels = data[:, -1].astype(int) if data.shape[1] > 3 else np.zeros(len(data), dtype=int)
                
        # Calculate class weights if needed
        if self.class_balance:
            self.class_weights = self._calculate_class_weights(labels)
            
        return {'images': images, 'labels': labels}
    
    def _generate_pattern_data(self, data: pd.DataFrame, 
                              window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate pattern images and labels from OHLCV data."""
        # Generate images
        images = self.pattern_generator.generate_pattern_images(data, window_size)
        
        # Generate pattern labels
        labels = self._detect_patterns(data, window_size)
        
        return images, labels
    
    def _detect_patterns(self, data: pd.DataFrame, window_size: int) -> np.ndarray:
        """
        Detect chart patterns in price data.
        
        Pattern classes:
        0: No pattern
        1: Bullish reversal (double bottom, inverse H&S)
        2: Bearish reversal (double top, H&S)
        3: Continuation (flag, pennant)
        4: Consolidation (triangle, rectangle)
        """
        n_samples = len(data) - window_size + 1
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            window = data.iloc[i:i+window_size]
            
            # Simple pattern detection logic
            high = window['high'].values
            low = window['low'].values
            close = window['close'].values
            
            # Price change
            price_change = (close[-1] - close[0]) / close[0]
            volatility = np.std(close) / np.mean(close)
            
            # Detect patterns (simplified)
            if self._is_double_bottom(low, close):
                labels[i] = 1  # Bullish reversal
            elif self._is_double_top(high, close):
                labels[i] = 2  # Bearish reversal
            elif self._is_flag_pattern(high, low, close):
                labels[i] = 3  # Continuation
            elif volatility < 0.01:  # Low volatility
                labels[i] = 4  # Consolidation
            else:
                labels[i] = 0  # No pattern
                
        return labels
    
    def _is_double_bottom(self, low: np.ndarray, close: np.ndarray) -> bool:
        """Detect double bottom pattern."""
        # Find two local minima
        window_size = 5
        local_minima = []
        
        for i in range(window_size, len(low) - window_size):
            if low[i] == np.min(low[i-window_size:i+window_size+1]):
                local_minima.append(i)
                
        # Check if we have two similar lows
        if len(local_minima) >= 2:
            low1, low2 = low[local_minima[-2]], low[local_minima[-1]]
            if abs(low1 - low2) / low1 < 0.02:  # Within 2%
                # Check if price is rising after second bottom
                return close[-1] > close[local_minima[-1]]
                
        return False
    
    def _is_double_top(self, high: np.ndarray, close: np.ndarray) -> bool:
        """Detect double top pattern."""
        # Similar to double bottom but for highs
        window_size = 5
        local_maxima = []
        
        for i in range(window_size, len(high) - window_size):
            if high[i] == np.max(high[i-window_size:i+window_size+1]):
                local_maxima.append(i)
                
        if len(local_maxima) >= 2:
            high1, high2 = high[local_maxima[-2]], high[local_maxima[-1]]
            if abs(high1 - high2) / high1 < 0.02:
                return close[-1] < close[local_maxima[-1]]
                
        return False
    
    def _is_flag_pattern(self, high: np.ndarray, low: np.ndarray, 
                        close: np.ndarray) -> bool:
        """Detect flag/pennant continuation pattern."""
        # Check for strong move followed by consolidation
        mid_point = len(close) // 2
        
        # Strong move in first half
        first_half_change = abs(close[mid_point] - close[0]) / close[0]
        
        # Consolidation in second half
        second_half_volatility = np.std(close[mid_point:]) / np.mean(close[mid_point:])
        
        return first_half_change > 0.05 and second_half_volatility < 0.02
    
    def _calculate_class_weights(self, labels: np.ndarray) -> np.ndarray:
        """Calculate class weights for balanced training."""
        unique_classes, class_counts = np.unique(labels, return_counts=True)
        n_samples = len(labels)
        n_classes = len(unique_classes)
        
        # Calculate balanced weights
        weights = n_samples / (n_classes * class_counts)
        
        # Create weight array
        class_weights = np.ones(self.model.num_classes)
        for cls, weight in zip(unique_classes, weights):
            class_weights[cls] = weight
            
        return class_weights
    
    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Enhanced training step with advanced augmentation.
        
        Args:
            batch: Batch data
            
        Returns:
            Metrics dictionary
        """
        images = batch['images']
        labels = batch['labels']
        
        # Apply augmentation
        if np.random.random() < self.augmentation_probability:
            images, labels = self._apply_augmentation(images, labels)
            
        # Progressive training: adjust difficulty
        if self.progressive_training:
            images, labels = self._apply_curriculum_learning(images, labels)
            
        # Forward pass
        self.model.set_training(True)
        output = self.model.forward(images)
        
        # Compute weighted loss
        loss = self._compute_weighted_loss(output['logits'], labels)
        
        # Backward pass
        gradients = self.model.backward(images, labels, output)
        
        # Update parameters
        updated_params = self.optimizer.update(self.model.params, gradients)
        self.model.params = updated_params
        
        # Calculate metrics
        predictions = output['predictions']
        accuracy = np.mean(predictions == labels)
        
        # Pattern-specific metrics
        pattern_metrics = self._calculate_pattern_metrics(predictions, labels)
        
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'confidence': output['confidence'].mean()
        }
        metrics.update(pattern_metrics)
        
        return metrics
    
    def _apply_augmentation(self, images: np.ndarray, 
                           labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply advanced augmentation techniques."""
        batch_size = len(images)
        augmented_images = images.copy()
        augmented_labels = labels.copy()
        
        for i in range(batch_size):
            aug_choice = np.random.choice(['basic', 'mixup', 'cutmix', 'pattern'])
            
            if aug_choice == 'basic':
                # Basic augmentations
                augmented_images[i] = self._basic_augmentation(images[i])
                
            elif aug_choice == 'mixup' and self.use_mixup and i < batch_size - 1:
                # Mixup augmentation
                lambda_val = np.random.beta(1.0, 1.0)
                augmented_images[i] = lambda_val * images[i] + (1 - lambda_val) * images[i + 1]
                # Labels are handled differently for mixup (soft labels)
                
            elif aug_choice == 'cutmix' and self.use_cutmix and i < batch_size - 1:
                # CutMix augmentation
                augmented_images[i], _ = self._cutmix(images[i], images[i + 1])
                
            elif aug_choice == 'pattern':
                # Pattern-specific augmentation
                augmented_images[i] = self._pattern_augmentation(images[i], labels[i])
                
        return augmented_images, augmented_labels
    
    def _basic_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply basic image augmentations."""
        augmented = image.copy()
        
        # Random brightness
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            augmented = augmented * brightness
            
        # Random contrast
        if np.random.random() > 0.5:
            contrast = np.random.uniform(0.8, 1.2)
            mean = augmented.mean()
            augmented = (augmented - mean) * contrast + mean
            
        # Random noise
        if np.random.random() > 0.5:
            noise = np.random.randn(*augmented.shape) * 0.05
            augmented = augmented + noise
            
        # Random horizontal flip (for certain patterns)
        if np.random.random() > 0.5:
            augmented = augmented[:, :, ::-1]
            
        return np.clip(augmented, 0, 1)
    
    def _cutmix(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply CutMix augmentation."""
        h, w = image1.shape[-2:]
        
        # Sample lambda from beta distribution
        lam = np.random.beta(1.0, 1.0)
        
        # Sample random box
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        cx = np.random.randint(0, h)
        cy = np.random.randint(0, w)
        
        x1 = np.clip(cx - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_h // 2, 0, h)
        y1 = np.clip(cy - cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_w // 2, 0, w)
        
        # Apply CutMix
        mixed = image1.copy()
        mixed[:, x1:x2, y1:y2] = image2[:, x1:x2, y1:y2]
        
        # Adjust lambda based on actual box size
        lam = 1 - ((x2 - x1) * (y2 - y1)) / (h * w)
        
        return mixed, lam
    
    def _pattern_augmentation(self, image: np.ndarray, label: int) -> np.ndarray:
        """Apply pattern-specific augmentations."""
        augmented = image.copy()
        
        if label == 1:  # Bullish reversal
            # Enhance bottom features
            bottom_region = augmented[:, -augmented.shape[1]//3:, :]
            augmented[:, -augmented.shape[1]//3:, :] = bottom_region * 1.1
            
        elif label == 2:  # Bearish reversal
            # Enhance top features
            top_region = augmented[:, :augmented.shape[1]//3, :]
            augmented[:, :augmented.shape[1]//3, :] = top_region * 1.1
            
        elif label == 3:  # Continuation
            # Add trend emphasis
            for i in range(augmented.shape[1]):
                weight = i / augmented.shape[1]
                augmented[:, i, :] *= (0.9 + 0.2 * weight)
                
        elif label == 4:  # Consolidation
            # Reduce contrast to emphasize sideways movement
            mean = augmented.mean()
            augmented = (augmented - mean) * 0.8 + mean
            
        return augmented
    
    def _apply_curriculum_learning(self, images: np.ndarray, 
                                  labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply curriculum learning by adjusting sample difficulty."""
        if self.epoch >= len(self.difficulty_schedule):
            return images, labels
            
        current_difficulty = self.difficulty_schedule[self.epoch]
        
        # Filter samples based on difficulty
        # Easier samples: clear patterns (labels != 0)
        # Harder samples: no pattern or ambiguous
        
        if current_difficulty < 1.0:
            # Include only a portion of difficult samples
            easy_mask = labels != 0  # Patterns are easier
            hard_mask = labels == 0  # No pattern is harder
            
            # Keep all easy samples and some hard samples
            hard_keep_prob = current_difficulty
            hard_keep = np.random.random(np.sum(hard_mask)) < hard_keep_prob
            
            # Combine masks
            keep_mask = easy_mask.copy()
            keep_mask[hard_mask] = hard_keep
            
            # Filter data
            images = images[keep_mask]
            labels = labels[keep_mask]
            
        return images, labels
    
    def _compute_weighted_loss(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Compute class-weighted cross-entropy loss."""
        n_samples = len(labels)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # One-hot encode labels
        y_one_hot = np.zeros((n_samples, self.model.num_classes))
        y_one_hot[np.arange(n_samples), labels] = 1
        
        # Apply class weights
        if self.class_weights is not None:
            weights = self.class_weights[labels]
            loss = -np.sum(weights * np.sum(y_one_hot * np.log(probs + 1e-10), axis=1)) / n_samples
        else:
            loss = -np.sum(y_one_hot * np.log(probs + 1e-10)) / n_samples
            
        return loss
    
    def _calculate_pattern_metrics(self, predictions: np.ndarray, 
                                  labels: np.ndarray) -> Dict[str, float]:
        """Calculate pattern-specific metrics."""
        metrics = {}
        
        pattern_names = ['no_pattern', 'bullish_reversal', 'bearish_reversal',
                        'continuation', 'consolidation']
        
        for i, pattern in enumerate(pattern_names):
            mask = labels == i
            if np.sum(mask) > 0:
                pattern_acc = np.mean(predictions[mask] == i)
                metrics[f'{pattern}_accuracy'] = pattern_acc
                
                # Detection rate (recall)
                detection_rate = np.sum((predictions == i) & mask) / np.sum(mask)
                metrics[f'{pattern}_detection_rate'] = detection_rate
                self.pattern_detection_rates[pattern] = detection_rate
                
        return metrics
    
    def validation_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Validation step with comprehensive metrics.
        
        Args:
            batch: Batch data
            
        Returns:
            Metrics dictionary
        """
        images = batch['images']
        labels = batch['labels']
        
        # Forward pass
        self.model.set_training(False)
        output = self.model.forward(images)
        
        # Compute loss
        loss = self._compute_weighted_loss(output['logits'], labels)
        
        # Calculate metrics
        predictions = output['predictions']
        accuracy = np.mean(predictions == labels)
        
        # Update confusion matrix
        self._update_confusion_matrix(predictions, labels)
        
        # Pattern-specific metrics
        pattern_metrics = self._calculate_pattern_metrics(predictions, labels)
        
        # Calculate confidence calibration
        calibration_error = self._calculate_calibration_error(
            output['confidence'], predictions, labels
        )
        
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'confidence': output['confidence'].mean(),
            'calibration_error': calibration_error
        }
        metrics.update(pattern_metrics)
        
        return metrics
    
    def _update_confusion_matrix(self, predictions: np.ndarray, labels: np.ndarray):
        """Update pattern confusion matrix."""
        if self.pattern_confusion_matrix is None:
            self.pattern_confusion_matrix = np.zeros(
                (self.model.num_classes, self.model.num_classes)
            )
            
        for true, pred in zip(labels, predictions):
            self.pattern_confusion_matrix[true, pred] += 1
            
    def _calculate_calibration_error(self, confidence: np.ndarray,
                                   predictions: np.ndarray,
                                   labels: np.ndarray) -> float:
        """Calculate expected calibration error."""
        # Bin confidences
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        for i in range(n_bins):
            in_bin = (confidence > bin_boundaries[i]) & (confidence <= bin_boundaries[i + 1])
            
            if np.sum(in_bin) > 0:
                bin_accuracy = np.mean(predictions[in_bin] == labels[in_bin])
                bin_confidence = np.mean(confidence[in_bin])
                bin_weight = np.sum(in_bin) / len(confidence)
                
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
                
        return ece
    
    def visualize_patterns(self, sample_images: np.ndarray,
                          sample_labels: np.ndarray,
                          save_path: Optional[str] = None):
        """
        Visualize detected patterns.
        
        Args:
            sample_images: Sample pattern images
            sample_labels: True labels
            save_path: Path to save visualization
        """
        import matplotlib.pyplot as plt
        
        # Get predictions
        self.model.set_training(False)
        output = self.model.forward(sample_images[:20])  # Use first 20
        predictions = output['predictions']
        confidence = output['confidence']
        
        # Pattern names
        pattern_names = ['No Pattern', 'Bullish Rev', 'Bearish Rev',
                        'Continuation', 'Consolidation']
        
        # Create visualization
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            if i < len(sample_images):
                # Show first channel of pattern image
                img = sample_images[i, 0]  # First channel (e.g., GASF)
                im = ax.imshow(img, cmap='viridis', aspect='auto')
                
                # Add prediction info
                true_pattern = pattern_names[sample_labels[i]]
                pred_pattern = pattern_names[predictions[i]]
                conf = confidence[i]
                
                color = 'green' if predictions[i] == sample_labels[i] else 'red'
                ax.set_title(f'True: {true_pattern}\nPred: {pred_pattern} ({conf:.2f})',
                           color=color, fontsize=10)
                ax.axis('off')
            else:
                ax.axis('off')
                
        plt.suptitle('Pattern Recognition Results', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
    def get_pattern_analysis(self) -> Dict:
        """Get comprehensive pattern analysis."""
        analysis = {
            'pattern_detection_rates': self.pattern_detection_rates,
            'confusion_matrix': self.pattern_confusion_matrix.tolist() if self.pattern_confusion_matrix is not None else None,
            'class_weights': self.class_weights.tolist() if self.class_weights is not None else None,
            'training_summary': self.get_training_summary()
        }
        
        # Calculate pattern-specific precision/recall
        if self.pattern_confusion_matrix is not None:
            pattern_metrics = {}
            pattern_names = ['no_pattern', 'bullish_reversal', 'bearish_reversal',
                           'continuation', 'consolidation']
            
            for i, pattern in enumerate(pattern_names):
                tp = self.pattern_confusion_matrix[i, i]
                fp = np.sum(self.pattern_confusion_matrix[:, i]) - tp
                fn = np.sum(self.pattern_confusion_matrix[i, :]) - tp
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                pattern_metrics[pattern] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'support': int(np.sum(self.pattern_confusion_matrix[i, :]))
                }
                
            analysis['pattern_metrics'] = pattern_metrics
            
        return analysis