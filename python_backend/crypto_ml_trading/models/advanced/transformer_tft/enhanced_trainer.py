"""
Enhanced Temporal Fusion Transformer Trainer

Features:
- Advanced optimization with warm restarts
- Multi-horizon prediction with adaptive loss weighting
- Attention regularization and interpretability
- Online learning and adaptation
- Hyperparameter optimization
- Comprehensive validation metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
import time
from pathlib import Path
from collections import defaultdict, deque
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.base_trainer import BaseTrainer, CosineAnnealingScheduler, ReduceLROnPlateauScheduler
from models.advanced.transformer_tft.tft_model import TemporalFusionTransformer
from models.advanced.transformer_tft.quantile_loss import QuantileLoss, AdaptiveQuantileLoss


@dataclass
class EnhancedTFTConfig:
    """Enhanced configuration for TFT training."""
    # Model architecture
    n_encoder_steps: int = 168  # 7 days of hourly data
    n_prediction_steps: int = 24  # 1 day ahead
    n_features: int = 10
    n_static_features: int = 5
    hidden_size: int = 160
    lstm_layers: int = 2
    num_attention_heads: int = 4
    dropout_rate: float = 0.1
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    
    # Training parameters
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-6
    batch_size: int = 32
    max_epochs: int = 200
    early_stopping_patience: int = 20
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Advanced optimization
    optimizer: str = 'adamw'
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    lr_schedule: str = 'cosine_warm_restarts'
    warmup_epochs: int = 5
    t_0: int = 10  # First restart period
    t_mult: int = 2  # Period multiplication factor
    
    # Loss configuration
    use_adaptive_loss: bool = True
    attention_regularization: float = 0.01
    temporal_consistency_weight: float = 0.1
    multi_horizon_weights: Optional[List[float]] = None
    
    # Data augmentation
    use_data_augmentation: bool = True
    noise_level: float = 0.01
    mask_probability: float = 0.1
    
    # Online learning
    online_learning: bool = True
    online_batch_size: int = 16
    online_update_frequency: int = 100


class CosineWarmRestartsScheduler(LearningRateScheduler):
    """Cosine annealing with warm restarts."""
    
    def __init__(self, initial_lr: float, t_0: int = 10, t_mult: int = 2,
                 min_lr: float = 1e-6, warmup_epochs: int = 0):
        super().__init__(initial_lr)
        self.t_0 = t_0
        self.t_mult = t_mult
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.t_i = t_0
        self.t_cur = 0
        
    def step(self, epoch: int, metrics: Optional[Dict] = None) -> float:
        """Update learning rate with warm restarts."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            self.current_lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing with restarts
            if self.t_cur >= self.t_i:
                self.t_cur = 0
                self.t_i *= self.t_mult
                
            self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * self.t_cur / self.t_i)
            )
            self.t_cur += 1
            
        return self.current_lr


class EnhancedTFTTrainer(BaseTrainer):
    """
    Enhanced Temporal Fusion Transformer trainer with advanced features.
    """
    
    def __init__(self, config: EnhancedTFTConfig):
        """
        Initialize enhanced TFT trainer.
        
        Args:
            config: Enhanced training configuration
        """
        self.config = config
        
        # Initialize model
        self.model = TemporalFusionTransformer(
            n_encoder_steps=config.n_encoder_steps,
            n_prediction_steps=config.n_prediction_steps,
            n_features=config.n_features,
            n_static_features=config.n_static_features,
            hidden_size=config.hidden_size,
            lstm_layers=config.lstm_layers,
            num_attention_heads=config.num_attention_heads,
            dropout_rate=config.dropout_rate,
            quantiles=config.quantiles
        )
        
        super().__init__(
            model=self.model,
            optimizer=config.optimizer,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            gradient_clip=config.gradient_clip,
            batch_size=config.batch_size,
            validation_split=config.validation_split,
            early_stopping_patience=config.early_stopping_patience
        )
        
        # Initialize loss function
        if config.use_adaptive_loss:
            self.loss_fn = AdaptiveQuantileLoss(config.quantiles)
        else:
            self.loss_fn = QuantileLoss(config.quantiles)
            
        # Multi-horizon loss weights
        if config.multi_horizon_weights is None:
            # Exponentially decaying weights for future predictions
            self.horizon_weights = np.exp(-0.1 * np.arange(config.n_prediction_steps))
            self.horizon_weights /= self.horizon_weights.sum()
        else:
            self.horizon_weights = np.array(config.multi_horizon_weights)
            
        # Online learning buffer
        if config.online_learning:
            self.online_buffer = deque(maxlen=1000)
            self.online_update_counter = 0
            
        # Attention analysis
        self.attention_patterns = defaultdict(list)
        self.feature_importance_history = []
        
        # Validation metrics tracking
        self.validation_metrics = defaultdict(lambda: deque(maxlen=100))
        
    def prepare_data(self, data: Union[pd.DataFrame, Dict]) -> Dict[str, np.ndarray]:
        """
        Prepare data for TFT training.
        
        Args:
            data: Raw data (DataFrame or pre-processed dict)
            
        Returns:
            Dictionary with prepared sequences
        """
        if isinstance(data, pd.DataFrame):
            return self._prepare_dataframe(data)
        else:
            # Assume pre-processed dictionary
            return data
            
    def _prepare_dataframe(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Prepare DataFrame for TFT training."""
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
            
        # Identify feature types
        static_cols = []
        known_future_cols = ['hour', 'day_of_week', 'month']  # Calendar features
        observed_cols = []
        
        for col in df.columns:
            if col in ['timestamp', 'target', 'close']:
                continue
            elif col in known_future_cols:
                continue
            elif df[col].nunique() == 1:
                static_cols.append(col)
            else:
                observed_cols.append(col)
                
        # Extract target
        if 'target' in df.columns:
            target = df['target'].values
        elif 'close' in df.columns:
            # Use returns as target
            target = df['close'].pct_change().fillna(0).values
        else:
            raise ValueError("No target column found")
            
        # Normalize features
        normalized_df = df.copy()
        for col in observed_cols:
            if col in df.columns:
                # Robust normalization
                median = df[col].median()
                mad = (df[col] - median).abs().median()
                if mad > 0:
                    normalized_df[col] = (df[col] - median) / (1.48 * mad)
                else:
                    normalized_df[col] = 0
                    
        # Create sequences
        total_steps = self.config.n_encoder_steps + self.config.n_prediction_steps
        n_samples = len(df) - total_steps + 1
        
        # Initialize arrays
        temporal_inputs = []
        static_inputs = []
        targets = []
        known_future_masks = []
        
        for i in range(n_samples):
            # Temporal features (observed + known future)
            temporal_seq = normalized_df[observed_cols + known_future_cols].iloc[
                i:i + total_steps
            ].values
            temporal_inputs.append(temporal_seq)
            
            # Static features
            if static_cols:
                static_seq = normalized_df[static_cols].iloc[i].values
            else:
                static_seq = np.zeros(self.config.n_static_features)
            static_inputs.append(static_seq)
            
            # Target (only for prediction steps)
            target_seq = target[i + self.config.n_encoder_steps:i + total_steps]
            targets.append(target_seq)
            
            # Known future mask (1 for known features, 0 for unknown)
            mask = np.zeros((total_steps, len(observed_cols + known_future_cols)))
            # Mark known future features
            for j, col in enumerate(observed_cols + known_future_cols):
                if col in known_future_cols:
                    mask[:, j] = 1
            known_future_masks.append(mask)
            
        return {
            'temporal_inputs': np.array(temporal_inputs),
            'static_inputs': np.array(static_inputs),
            'targets': np.array(targets),
            'known_future_mask': np.array(known_future_masks)
        }
    
    def augment_batch(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply data augmentation to batch.
        
        Args:
            batch: Batch data
            
        Returns:
            Augmented batch
        """
        if not self.config.use_data_augmentation:
            return batch
            
        augmented = {}
        
        # Add noise to temporal inputs
        temporal = batch['temporal_inputs'].copy()
        if np.random.random() > 0.5:
            noise = np.random.randn(*temporal.shape) * self.config.noise_level
            temporal += noise
            
        # Random masking for robustness
        if np.random.random() > 0.5:
            mask = np.random.random(temporal.shape) > self.config.mask_probability
            temporal = temporal * mask
            
        augmented['temporal_inputs'] = temporal
        augmented['static_inputs'] = batch['static_inputs']
        augmented['targets'] = batch['targets']
        augmented['known_future_mask'] = batch['known_future_mask']
        
        return augmented
    
    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Enhanced training step with multi-objective optimization.
        
        Args:
            batch: Batch data
            
        Returns:
            Metrics dictionary
        """
        # Apply augmentation
        if self.config.use_data_augmentation:
            batch = self.augment_batch(batch)
            
        # Forward pass
        self.model.training = True
        outputs = self.model.forward({
            'temporal_inputs': batch['temporal_inputs'],
            'static_inputs': batch['static_inputs'],
            'known_future_mask': batch['known_future_mask']
        })
        
        # Compute multi-objective loss
        loss_components = self._compute_multi_objective_loss(
            outputs, batch['targets']
        )
        
        # Backward pass (simplified - would need proper implementation)
        gradients = self._compute_gradients(loss_components, outputs, batch)
        
        # Update parameters
        updated_params = self.optimizer.update(self.model.params, gradients)
        self.model.params = updated_params
        
        # Calculate metrics
        metrics = {
            'loss': loss_components['total_loss'],
            'quantile_loss': loss_components['quantile_loss'],
            'attention_reg': loss_components['attention_regularization'],
            'temporal_consistency': loss_components['temporal_consistency']
        }
        
        # Track attention patterns
        if 'attention_weights' in outputs:
            self._analyze_attention(outputs['attention_weights'])
            
        # Add to online buffer if enabled
        if self.config.online_learning:
            self.online_buffer.append({
                'batch': batch,
                'outputs': outputs,
                'loss': loss_components['total_loss']
            })
            
        return metrics
    
    def _compute_multi_objective_loss(self, outputs: Dict[str, np.ndarray],
                                     targets: np.ndarray) -> Dict[str, float]:
        """Compute multi-objective loss with various components."""
        loss_components = {}
        
        # Quantile loss with horizon weighting
        predictions = outputs['predictions']
        quantile_losses = []
        
        for t in range(self.config.n_prediction_steps):
            t_predictions = {q: predictions[q][:, t] for q in predictions}
            t_targets = targets[:, t]
            t_loss = self.loss_fn.compute_loss(t_predictions, t_targets)
            quantile_losses.append(t_loss['total_loss'] * self.horizon_weights[t])
            
        loss_components['quantile_loss'] = np.mean(quantile_losses)
        
        # Attention regularization (encourage sparse attention)
        if 'attention_weights' in outputs:
            attention_weights = outputs['attention_weights']
            # Entropy regularization
            entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-10), axis=-1)
            loss_components['attention_regularization'] = -self.config.attention_regularization * np.mean(entropy)
        else:
            loss_components['attention_regularization'] = 0
            
        # Temporal consistency (predictions should be smooth)
        if self.config.temporal_consistency_weight > 0:
            median_predictions = predictions[0.5]  # Median quantile
            temporal_diff = np.diff(median_predictions, axis=1)
            loss_components['temporal_consistency'] = (
                self.config.temporal_consistency_weight * np.mean(temporal_diff**2)
            )
        else:
            loss_components['temporal_consistency'] = 0
            
        # Total loss
        loss_components['total_loss'] = (
            loss_components['quantile_loss'] +
            loss_components['attention_regularization'] +
            loss_components['temporal_consistency']
        )
        
        return loss_components
    
    def _compute_gradients(self, loss_components: Dict[str, float],
                          outputs: Dict[str, np.ndarray],
                          batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute gradients (simplified implementation)."""
        # In practice, this would use automatic differentiation
        # Here we return placeholder gradients
        gradients = {}
        
        # Placeholder gradient computation
        for param_name, param_value in self.model.params.items():
            gradients[param_name] = np.random.randn(*param_value.shape) * 0.001
            
        return gradients
    
    def validation_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Enhanced validation step with comprehensive metrics.
        
        Args:
            batch: Batch data
            
        Returns:
            Metrics dictionary
        """
        # Forward pass
        self.model.training = False
        outputs = self.model.forward({
            'temporal_inputs': batch['temporal_inputs'],
            'static_inputs': batch['static_inputs'],
            'known_future_mask': batch['known_future_mask']
        })
        
        # Compute loss
        loss_components = self._compute_multi_objective_loss(
            outputs, batch['targets']
        )
        
        # Calculate additional validation metrics
        predictions = outputs['predictions']
        targets = batch['targets']
        
        # Prediction interval metrics
        interval_metrics = self._calculate_interval_metrics(predictions, targets)
        
        # Directional accuracy
        direction_accuracy = self._calculate_direction_accuracy(
            predictions[0.5], targets  # Use median prediction
        )
        
        # Feature importance
        if hasattr(self.model, 'get_feature_importance'):
            feature_importance = self.model.get_feature_importance({
                'temporal_inputs': batch['temporal_inputs'][:10],  # Sample
                'static_inputs': batch['static_inputs'][:10],
                'known_future_mask': batch['known_future_mask'][:10]
            })
            self.feature_importance_history.append(feature_importance)
            
        metrics = {
            'loss': loss_components['total_loss'],
            'quantile_loss': loss_components['quantile_loss'],
            'direction_accuracy': direction_accuracy
        }
        metrics.update(interval_metrics)
        
        # Track validation metrics
        for key, value in metrics.items():
            self.validation_metrics[key].append(value)
            
        return metrics
    
    def _calculate_interval_metrics(self, predictions: Dict[float, np.ndarray],
                                   targets: np.ndarray) -> Dict[str, float]:
        """Calculate prediction interval metrics."""
        metrics = {}
        
        # Coverage for different intervals
        intervals = [(0.1, 0.9), (0.25, 0.75)]
        
        for lower_q, upper_q in intervals:
            lower_pred = predictions[lower_q]
            upper_pred = predictions[upper_q]
            
            # Coverage: percentage of targets within interval
            coverage = np.mean((targets >= lower_pred) & (targets <= upper_pred))
            
            # Interval width
            width = np.mean(upper_pred - lower_pred)
            
            # Interval score (lower is better)
            alpha = upper_q - lower_q
            interval_score = width + (2 / alpha) * (
                (lower_pred - targets) * (targets < lower_pred) +
                (targets - upper_pred) * (targets > upper_pred)
            ).mean()
            
            interval_name = f"{int(alpha * 100)}%"
            metrics[f'coverage_{interval_name}'] = coverage
            metrics[f'interval_width_{interval_name}'] = width
            metrics[f'interval_score_{interval_name}'] = interval_score
            
        return metrics
    
    def _calculate_direction_accuracy(self, predictions: np.ndarray,
                                     targets: np.ndarray) -> float:
        """Calculate directional accuracy of predictions."""
        # For returns, check if sign matches
        pred_direction = np.sign(predictions)
        true_direction = np.sign(targets)
        
        # Exclude cases where true return is near zero
        significant_mask = np.abs(targets) > 0.0001
        
        if np.sum(significant_mask) > 0:
            accuracy = np.mean(pred_direction[significant_mask] == true_direction[significant_mask])
        else:
            accuracy = 0.5  # Random baseline
            
        return accuracy
    
    def _analyze_attention(self, attention_weights: np.ndarray):
        """Analyze attention patterns for interpretability."""
        # Average attention across batch and heads
        avg_attention = np.mean(attention_weights, axis=(0, 1))
        
        # Track which time steps get most attention
        temporal_importance = np.mean(avg_attention, axis=0)
        self.attention_patterns['temporal_importance'].append(temporal_importance)
        
        # Track feature attention if available
        if len(avg_attention.shape) > 2:
            feature_importance = np.mean(avg_attention, axis=1)
            self.attention_patterns['feature_importance'].append(feature_importance)
            
    def online_update(self, new_data: pd.DataFrame) -> Dict[str, float]:
        """
        Perform online learning update with new data.
        
        Args:
            new_data: New market data
            
        Returns:
            Update metrics
        """
        if not self.config.online_learning:
            return {}
            
        self.online_update_counter += 1
        
        # Only update at specified frequency
        if self.online_update_counter % self.config.online_update_frequency != 0:
            return {}
            
        # Prepare new data
        new_batch = self._prepare_dataframe(new_data)
        
        # Combine with buffered data
        if len(self.online_buffer) > 0:
            # Sample from buffer
            buffer_size = min(len(self.online_buffer), self.config.online_batch_size // 2)
            buffer_samples = np.random.choice(len(self.online_buffer), buffer_size, replace=False)
            
            # Combine batches
            combined_batch = self._combine_batches(
                new_batch,
                [self.online_buffer[i]['batch'] for i in buffer_samples]
            )
        else:
            combined_batch = new_batch
            
        # Reduce learning rate for online updates
        original_lr = self.optimizer.learning_rate
        self.optimizer.learning_rate *= 0.1
        
        # Perform update
        metrics = self.train_step(combined_batch)
        
        # Restore learning rate
        self.optimizer.learning_rate = original_lr
        
        print(f"Online update {self.online_update_counter}: Loss = {metrics['loss']:.4f}")
        
        return metrics
    
    def _combine_batches(self, new_batch: Dict[str, np.ndarray],
                        buffer_batches: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Combine new batch with buffered batches."""
        all_batches = [new_batch] + buffer_batches
        
        combined = {}
        for key in new_batch.keys():
            combined[key] = np.concatenate([b[key] for b in all_batches], axis=0)
            
        return combined
    
    def forecast(self, data: pd.DataFrame,
                n_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        Generate probabilistic forecasts with uncertainty.
        
        Args:
            data: Input data
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Forecast dictionary
        """
        # Prepare data
        prepared_data = self._prepare_dataframe(data)
        
        # Generate multiple forecasts with dropout
        all_forecasts = []
        
        for _ in range(n_samples):
            self.model.training = True  # Enable dropout
            outputs = self.model.forward({
                'temporal_inputs': prepared_data['temporal_inputs'],
                'static_inputs': prepared_data['static_inputs'],
                'known_future_mask': prepared_data['known_future_mask']
            })
            all_forecasts.append(outputs['predictions'])
            
        # Aggregate forecasts
        forecast_dict = {}
        
        # Mean and std for each quantile
        for q in self.config.quantiles:
            q_forecasts = np.array([f[q] for f in all_forecasts])
            forecast_dict[f'q{int(q*100)}_mean'] = np.mean(q_forecasts, axis=0)
            forecast_dict[f'q{int(q*100)}_std'] = np.std(q_forecasts, axis=0)
            
        # Point forecast (median)
        median_forecasts = np.array([f[0.5] for f in all_forecasts])
        forecast_dict['point_forecast'] = np.mean(median_forecasts, axis=0)
        forecast_dict['forecast_uncertainty'] = np.std(median_forecasts, axis=0)
        
        # Prediction intervals
        forecast_dict['lower_95'] = np.percentile(median_forecasts, 2.5, axis=0)
        forecast_dict['upper_95'] = np.percentile(median_forecasts, 97.5, axis=0)
        
        return forecast_dict
    
    def explain_predictions(self, data: pd.DataFrame,
                           target_idx: int = -1) -> Dict[str, np.ndarray]:
        """
        Explain predictions using attention weights and feature importance.
        
        Args:
            data: Input data
            target_idx: Index of sample to explain
            
        Returns:
            Explanation dictionary
        """
        # Prepare data
        prepared_data = self._prepare_dataframe(data)
        
        # Get single sample
        sample_data = {
            key: value[target_idx:target_idx+1] for key, value in prepared_data.items()
        }
        
        # Forward pass with attention
        self.model.training = False
        outputs = self.model.forward(sample_data)
        
        explanation = {
            'prediction': outputs['predictions'][0.5][0],  # Median prediction
            'prediction_intervals': {
                f'q{int(q*100)}': outputs['predictions'][q][0]
                for q in self.config.quantiles
            }
        }
        
        # Attention-based explanation
        if 'attention_weights' in outputs:
            attention = outputs['attention_weights'][0]  # First sample
            
            # Temporal importance
            temporal_importance = np.mean(attention, axis=0)
            explanation['temporal_importance'] = temporal_importance
            
            # Find most attended time steps
            top_timesteps = np.argsort(temporal_importance)[-5:]
            explanation['important_timesteps'] = top_timesteps
            
        # Feature importance from model
        if hasattr(self.model, 'get_feature_importance'):
            feature_importance = self.model.get_feature_importance(sample_data)
            explanation['feature_importance'] = feature_importance
            
        return explanation
    
    def hyperparameter_optimization(self, data: Union[pd.DataFrame, Dict],
                                  param_space: Dict[str, List],
                                  n_trials: int = 20) -> Dict:
        """
        Perform hyperparameter optimization.
        
        Args:
            data: Training data
            param_space: Parameter search space
            n_trials: Number of trials
            
        Returns:
            Best parameters and results
        """
        results = []
        
        for trial in range(n_trials):
            # Sample parameters
            trial_params = {}
            for param, values in param_space.items():
                if isinstance(values[0], (int, float)):
                    # Numeric parameter
                    trial_params[param] = np.random.choice(values)
                else:
                    # Categorical parameter
                    trial_params[param] = np.random.choice(values)
                    
            print(f"\nTrial {trial + 1}/{n_trials}")
            print(f"Parameters: {trial_params}")
            
            # Update config
            for param, value in trial_params.items():
                if hasattr(self.config, param):
                    setattr(self.config, param, value)
                    
            # Reinitialize model with new parameters
            self.model = TemporalFusionTransformer(
                n_encoder_steps=self.config.n_encoder_steps,
                n_prediction_steps=self.config.n_prediction_steps,
                n_features=self.config.n_features,
                n_static_features=self.config.n_static_features,
                hidden_size=trial_params.get('hidden_size', self.config.hidden_size),
                lstm_layers=trial_params.get('lstm_layers', self.config.lstm_layers),
                num_attention_heads=trial_params.get('num_attention_heads', self.config.num_attention_heads),
                dropout_rate=trial_params.get('dropout_rate', self.config.dropout_rate),
                quantiles=self.config.quantiles
            )
            
            # Train with reduced epochs for speed
            trial_history = self.train(
                data,
                epochs=50,
                verbose=0
            )
            
            # Record results
            val_loss = min(trial_history['val_loss'])
            results.append({
                'params': trial_params.copy(),
                'val_loss': val_loss,
                'history': trial_history
            })
            
            print(f"Validation loss: {val_loss:.4f}")
            
        # Find best parameters
        best_idx = np.argmin([r['val_loss'] for r in results])
        best_result = results[best_idx]
        
        print(f"\nBest parameters: {best_result['params']}")
        print(f"Best validation loss: {best_result['val_loss']:.4f}")
        
        return {
            'best_params': best_result['params'],
            'best_val_loss': best_result['val_loss'],
            'all_results': results
        }
    
    def get_training_summary(self) -> Dict:
        """Get comprehensive training summary."""
        summary = super().get_training_summary()
        
        # Add TFT-specific metrics
        if len(self.validation_metrics['coverage_80%']) > 0:
            summary['avg_coverage_80'] = np.mean(self.validation_metrics['coverage_80%'])
            summary['avg_interval_width_80'] = np.mean(self.validation_metrics['interval_width_80%'])
            
        if len(self.validation_metrics['direction_accuracy']) > 0:
            summary['avg_direction_accuracy'] = np.mean(self.validation_metrics['direction_accuracy'])
            
        # Attention analysis
        if 'temporal_importance' in self.attention_patterns:
            temporal_importance = np.mean(self.attention_patterns['temporal_importance'], axis=0)
            summary['most_important_timesteps'] = np.argsort(temporal_importance)[-5:].tolist()
            
        # Online learning stats
        if self.config.online_learning:
            summary['online_updates'] = self.online_update_counter
            summary['online_buffer_size'] = len(self.online_buffer)
            
        return summary