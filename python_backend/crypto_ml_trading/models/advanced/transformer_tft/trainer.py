"""
TFT Training Module with Advanced Optimization.

Implements training routines, validation, and hyperparameter optimization for TFT.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.advanced.transformer_tft.tft_model import TemporalFusionTransformer
from models.advanced.transformer_tft.quantile_loss import QuantileLoss, AdaptiveQuantileLoss
from utils.matrix_operations import MatrixOperations


@dataclass
class TFTConfig:
    """Configuration for TFT training."""
    # Model architecture
    n_encoder_steps: int = 168
    n_prediction_steps: int = 24
    n_features: int = 10
    n_static_features: int = 5
    hidden_size: int = 160
    lstm_layers: int = 2
    num_attention_heads: int = 4
    dropout_rate: float = 0.1
    quantiles: List[float] = None
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    
    # Optimization
    optimizer: str = 'adam'  # 'adam', 'sgd', 'rmsprop'
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    lr_schedule: str = 'cosine'  # 'cosine', 'step', 'plateau'
    
    # Loss configuration
    use_adaptive_loss: bool = True
    loss_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]


class TFTTrainer:
    """
    Temporal Fusion Transformer Trainer.
    
    Handles training, validation, and optimization of TFT models.
    """
    
    def __init__(self, config: TFTConfig):
        """
        Initialize TFT trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.optimizer_state = {}
        self.training_history = []
        self.best_model_params = None
        self.best_validation_loss = float('inf')
        
        # Initialize loss function
        if config.use_adaptive_loss:
            self.loss_fn = AdaptiveQuantileLoss(config.quantiles)
        else:
            self.loss_fn = QuantileLoss(config.quantiles)
            
    def prepare_data(self, data: pd.DataFrame,
                    target_column: str = 'close',
                    static_features: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Prepare data for TFT training.
        
        Args:
            data: Input DataFrame with time series data
            target_column: Name of target column
            static_features: List of static feature columns
            
        Returns:
            Prepared data dictionary
        """
        # Sort by timestamp
        data = data.sort_index()
        
        # Extract features
        feature_columns = [col for col in data.columns 
                         if col not in [target_column] + (static_features or [])]
        
        # Create sequences
        sequences = self._create_sequences(
            data, target_column, feature_columns, static_features
        )
        
        # Split into train/validation
        train_data, val_data = self._train_val_split(sequences)
        
        return {
            'train': train_data,
            'validation': val_data,
            'feature_names': feature_columns,
            'static_feature_names': static_features or []
        }
    
    def _create_sequences(self, data: pd.DataFrame,
                         target_column: str,
                         feature_columns: List[str],
                         static_features: Optional[List[str]]) -> Dict[str, np.ndarray]:
        """Create input sequences for TFT."""
        total_steps = self.config.n_encoder_steps + self.config.n_prediction_steps
        
        # Prepare temporal features
        temporal_data = data[feature_columns + [target_column]].values
        
        # Prepare static features (if any)
        if static_features:
            static_data = data[static_features].iloc[0].values  # Assume static
            static_data = np.tile(static_data, (len(data) - total_steps + 1, 1))
        else:
            static_data = np.zeros((len(data) - total_steps + 1, self.config.n_static_features))
        
        # Create sequences
        temporal_inputs = []
        targets = []
        
        for i in range(len(data) - total_steps + 1):
            # Input sequence
            seq = temporal_data[i:i + total_steps, :-1]  # Exclude target from features
            temporal_inputs.append(seq)
            
            # Target sequence (prediction horizon only)
            target_seq = temporal_data[i + self.config.n_encoder_steps:i + total_steps, -1]\n            targets.append(target_seq)\n            \n        return {\n            'temporal_inputs': np.array(temporal_inputs),\n            'static_inputs': static_data,\n            'targets': np.array(targets),\n            'known_future_mask': np.ones_like(np.array(temporal_inputs))  # Simplified\n        }\n    \n    def _train_val_split(self, sequences: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:\n        \"\"\"Split sequences into training and validation sets.\"\"\"\n        n_samples = len(sequences['targets'])\n        n_train = int(n_samples * (1 - self.config.validation_split))\n        \n        train_data = {key: val[:n_train] for key, val in sequences.items()}\n        val_data = {key: val[n_train:] for key, val in sequences.items()}\n        \n        return train_data, val_data\n    \n    def train(self, data: pd.DataFrame,\n             target_column: str = 'close',\n             static_features: Optional[List[str]] = None,\n             verbose: bool = True) -> Dict:\n        \"\"\"\n        Train TFT model.\n        \n        Args:\n            data: Training data\n            target_column: Target column name\n            static_features: Static feature columns\n            verbose: Whether to print training progress\n            \n        Returns:\n            Training results\n        \"\"\"\n        # Prepare data\n        prepared_data = self.prepare_data(data, target_column, static_features)\n        train_data = prepared_data['train']\n        val_data = prepared_data['validation']\n        \n        # Initialize model\n        self.model = TemporalFusionTransformer(\n            n_encoder_steps=self.config.n_encoder_steps,\n            n_prediction_steps=self.config.n_prediction_steps,\n            n_features=train_data['temporal_inputs'].shape[-1],\n            n_static_features=train_data['static_inputs'].shape[-1],\n            hidden_size=self.config.hidden_size,\n            lstm_layers=self.config.lstm_layers,\n            num_attention_heads=self.config.num_attention_heads,\n            dropout_rate=self.config.dropout_rate,\n            quantiles=self.config.quantiles\n        )\n        \n        # Initialize optimizer\n        self._initialize_optimizer()\n        \n        # Training loop\n        patience_counter = 0\n        \n        for epoch in range(self.config.max_epochs):\n            # Training step\n            train_metrics = self._train_epoch(train_data)\n            \n            # Validation step\n            val_metrics = self._validate_epoch(val_data)\n            \n            # Update learning rate\n            self._update_learning_rate(epoch, val_metrics['loss'])\n            \n            # Record metrics\n            epoch_metrics = {\n                'epoch': epoch,\n                'train_loss': train_metrics['loss'],\n                'val_loss': val_metrics['loss'],\n                'learning_rate': self._get_current_lr()\n            }\n            epoch_metrics.update({f'train_{k}': v for k, v in train_metrics.items()})\n            epoch_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})\n            \n            self.training_history.append(epoch_metrics)\n            \n            # Early stopping check\n            if val_metrics['loss'] < self.best_validation_loss:\n                self.best_validation_loss = val_metrics['loss']\n                self.best_model_params = self._copy_model_params()\n                patience_counter = 0\n            else:\n                patience_counter += 1\n                \n            if verbose and epoch % 10 == 0:\n                print(f\"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, \"\n                     f\"Val Loss: {val_metrics['loss']:.4f}\")\n                \n            if patience_counter >= self.config.early_stopping_patience:\n                if verbose:\n                    print(f\"Early stopping at epoch {epoch}\")\n                break\n                \n        # Restore best model\n        if self.best_model_params:\n            self.model.params = self.best_model_params\n            \n        # Final evaluation\n        final_metrics = self._validate_epoch(val_data)\n        \n        return {\n            'training_history': self.training_history,\n            'final_metrics': final_metrics,\n            'best_validation_loss': self.best_validation_loss,\n            'total_epochs': len(self.training_history)\n        }\n    \n    def _train_epoch(self, train_data: Dict[str, np.ndarray]) -> Dict[str, float]:\n        \"\"\"Train for one epoch.\"\"\"\n        total_loss = 0.0\n        total_samples = 0\n        batch_losses = []\n        \n        # Create batches\n        batches = self._create_batches(train_data, shuffle=True)\n        \n        for batch in batches:\n            # Forward pass\n            predictions = self.model.forward({\n                'temporal_inputs': batch['temporal_inputs'],\n                'static_inputs': batch['static_inputs'],\n                'known_future_mask': batch['known_future_mask']\n            })\n            \n            # Compute loss\n            loss_dict = self.loss_fn.compute_loss(\n                predictions['predictions'], \n                batch['targets']\n            )\n            \n            # Backward pass\n            self._backward_pass(loss_dict, predictions, batch)\n            \n            # Update parameters\n            self._update_parameters()\n            \n            # Track metrics\n            batch_loss = loss_dict['total_loss']\n            total_loss += batch_loss * len(batch['targets'])\n            total_samples += len(batch['targets'])\n            batch_losses.append(batch_loss)\n            \n        return {\n            'loss': total_loss / total_samples,\n            'batch_losses': batch_losses\n        }\n    \n    def _validate_epoch(self, val_data: Dict[str, np.ndarray]) -> Dict[str, float]:\n        \"\"\"Validate for one epoch.\"\"\"\n        total_loss = 0.0\n        total_samples = 0\n        all_predictions = []\n        all_targets = []\n        \n        # Create batches\n        batches = self._create_batches(val_data, shuffle=False)\n        \n        for batch in batches:\n            # Forward pass (no training)\n            predictions = self.model.predict({\n                'temporal_inputs': batch['temporal_inputs'],\n                'static_inputs': batch['static_inputs'],\n                'known_future_mask': batch['known_future_mask']\n            })\n            \n            # Compute loss\n            loss_dict = self.loss_fn.compute_loss(\n                predictions['predictions'], \n                batch['targets']\n            )\n            \n            # Track metrics\n            batch_loss = loss_dict['total_loss']\n            total_loss += batch_loss * len(batch['targets'])\n            total_samples += len(batch['targets'])\n            \n            # Store predictions for analysis\n            all_predictions.append(predictions['predictions'])\n            all_targets.append(batch['targets'])\n            \n        # Combine all predictions\n        combined_predictions = {}\n        for key in all_predictions[0].keys():\n            combined_predictions[key] = np.concatenate(\n                [pred[key] for pred in all_predictions], axis=0\n            )\n        combined_targets = np.concatenate(all_targets, axis=0)\n        \n        # Compute additional metrics\n        interval_metrics = self.loss_fn.evaluate_prediction_intervals(\n            combined_predictions, combined_targets\n        )\n        \n        metrics = {\n            'loss': total_loss / total_samples\n        }\n        metrics.update(interval_metrics)\n        \n        return metrics\n    \n    def _create_batches(self, data: Dict[str, np.ndarray], \n                       shuffle: bool = True) -> List[Dict[str, np.ndarray]]:\n        \"\"\"Create mini-batches from data.\"\"\"\n        n_samples = len(data['targets'])\n        \n        # Create indices\n        indices = np.arange(n_samples)\n        if shuffle:\n            np.random.shuffle(indices)\n            \n        # Create batches\n        batches = []\n        for i in range(0, n_samples, self.config.batch_size):\n            batch_indices = indices[i:i + self.config.batch_size]\n            \n            batch = {}\n            for key, values in data.items():\n                batch[key] = values[batch_indices]\n                \n            batches.append(batch)\n            \n        return batches\n    \n    def _initialize_optimizer(self):\n        \"\"\"Initialize optimizer state.\"\"\"\n        if self.config.optimizer == 'adam':\n            self.optimizer_state = {\n                'm': {},  # First moment estimates\n                'v': {},  # Second moment estimates\n                'beta1': 0.9,\n                'beta2': 0.999,\n                'epsilon': 1e-8,\n                't': 0  # Time step\n            }\n        elif self.config.optimizer == 'sgd':\n            self.optimizer_state = {\n                'momentum': {},\n                'momentum_rate': 0.9\n            }\n        elif self.config.optimizer == 'rmsprop':\n            self.optimizer_state = {\n                'cache': {},\n                'decay_rate': 0.95,\n                'epsilon': 1e-8\n            }\n            \n    def _backward_pass(self, loss_dict: Dict[str, float],\n                      predictions: Dict[str, np.ndarray],\n                      batch: Dict[str, np.ndarray]):\n        \"\"\"Compute gradients (simplified implementation).\"\"\"\n        # Compute loss gradients\n        loss_gradients = self.loss_fn.compute_gradients(\n            predictions['predictions'],\n            batch['targets']\n        )\n        \n        # Store gradients in model (simplified)\n        self.model.gradients = loss_gradients\n        \n    def _update_parameters(self):\n        \"\"\"Update model parameters using optimizer.\"\"\"\n        if not hasattr(self.model, 'gradients'):\n            return\n            \n        if self.config.optimizer == 'adam':\n            self._adam_update()\n        elif self.config.optimizer == 'sgd':\n            self._sgd_update()\n        elif self.config.optimizer == 'rmsprop':\n            self._rmsprop_update()\n            \n    def _adam_update(self):\n        \"\"\"Adam optimizer update.\"\"\"\n        self.optimizer_state['t'] += 1\n        t = self.optimizer_state['t']\n        \n        lr = self._get_current_lr()\n        beta1 = self.optimizer_state['beta1']\n        beta2 = self.optimizer_state['beta2']\n        epsilon = self.optimizer_state['epsilon']\n        \n        # Bias correction\n        lr_corrected = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)\n        \n        for param_name, param in self.model.params.items():\n            if param_name not in self.model.gradients:\n                continue\n                \n            grad = self.model.gradients[param_name]\n            \n            # Initialize moments if needed\n            if param_name not in self.optimizer_state['m']:\n                self.optimizer_state['m'][param_name] = np.zeros_like(param)\n                self.optimizer_state['v'][param_name] = np.zeros_like(param)\n                \n            # Update moments\n            self.optimizer_state['m'][param_name] = (\n                beta1 * self.optimizer_state['m'][param_name] + (1 - beta1) * grad\n            )\n            self.optimizer_state['v'][param_name] = (\n                beta2 * self.optimizer_state['v'][param_name] + (1 - beta2) * grad**2\n            )\n            \n            # Update parameters\n            update = lr_corrected * self.optimizer_state['m'][param_name] / (\n                np.sqrt(self.optimizer_state['v'][param_name]) + epsilon\n            )\n            \n            # Apply weight decay\n            if self.config.weight_decay > 0:\n                update += self.config.weight_decay * param\n                \n            # Gradient clipping\n            if self.config.gradient_clip > 0:\n                grad_norm = np.linalg.norm(update)\n                if grad_norm > self.config.gradient_clip:\n                    update = update * self.config.gradient_clip / grad_norm\n                    \n            self.model.params[param_name] -= update\n            \n    def _sgd_update(self):\n        \"\"\"SGD with momentum update.\"\"\"\n        lr = self._get_current_lr()\n        momentum_rate = self.optimizer_state['momentum_rate']\n        \n        for param_name, param in self.model.params.items():\n            if param_name not in self.model.gradients:\n                continue\n                \n            grad = self.model.gradients[param_name]\n            \n            # Initialize momentum if needed\n            if param_name not in self.optimizer_state['momentum']:\n                self.optimizer_state['momentum'][param_name] = np.zeros_like(param)\n                \n            # Update momentum\n            self.optimizer_state['momentum'][param_name] = (\n                momentum_rate * self.optimizer_state['momentum'][param_name] + \n                lr * grad\n            )\n            \n            # Update parameters\n            self.model.params[param_name] -= self.optimizer_state['momentum'][param_name]\n            \n    def _rmsprop_update(self):\n        \"\"\"RMSprop optimizer update.\"\"\"\n        lr = self._get_current_lr()\n        decay_rate = self.optimizer_state['decay_rate']\n        epsilon = self.optimizer_state['epsilon']\n        \n        for param_name, param in self.model.params.items():\n            if param_name not in self.model.gradients:\n                continue\n                \n            grad = self.model.gradients[param_name]\n            \n            # Initialize cache if needed\n            if param_name not in self.optimizer_state['cache']:\n                self.optimizer_state['cache'][param_name] = np.zeros_like(param)\n                \n            # Update cache\n            self.optimizer_state['cache'][param_name] = (\n                decay_rate * self.optimizer_state['cache'][param_name] + \n                (1 - decay_rate) * grad**2\n            )\n            \n            # Update parameters\n            update = lr * grad / (\n                np.sqrt(self.optimizer_state['cache'][param_name]) + epsilon\n            )\n            \n            self.model.params[param_name] -= update\n            \n    def _get_current_lr(self) -> float:\n        \"\"\"Get current learning rate based on schedule.\"\"\"\n        base_lr = self.config.learning_rate\n        \n        if self.config.lr_schedule == 'cosine':\n            # Cosine annealing\n            progress = len(self.training_history) / self.config.max_epochs\n            return base_lr * 0.5 * (1 + np.cos(np.pi * progress))\n        elif self.config.lr_schedule == 'step':\n            # Step decay\n            step_size = self.config.max_epochs // 3\n            decay_factor = 0.1\n            steps = len(self.training_history) // step_size\n            return base_lr * (decay_factor ** steps)\n        else:\n            return base_lr\n            \n    def _update_learning_rate(self, epoch: int, val_loss: float):\n        \"\"\"Update learning rate based on schedule.\"\"\"\n        if self.config.lr_schedule == 'plateau':\n            # Reduce on plateau\n            if len(self.training_history) > 10:\n                recent_losses = [h['val_loss'] for h in self.training_history[-10:]]\n                if all(loss >= val_loss * 0.999 for loss in recent_losses[-5:]):\n                    self.config.learning_rate *= 0.5\n                    \n    def _copy_model_params(self) -> Dict[str, np.ndarray]:\n        \"\"\"Create a deep copy of model parameters.\"\"\"\n        return {k: v.copy() for k, v in self.model.params.items()}\n    \n    def predict(self, data: pd.DataFrame,\n               target_column: str = 'close',\n               static_features: Optional[List[str]] = None) -> Dict[str, np.ndarray]:\n        \"\"\"\n        Make predictions using trained model.\n        \n        Args:\n            data: Input data\n            target_column: Target column name\n            static_features: Static feature columns\n            \n        Returns:\n            Predictions dictionary\n        \"\"\"\n        if self.model is None:\n            raise ValueError(\"Model must be trained first\")\n            \n        # Prepare data\n        prepared_data = self.prepare_data(data, target_column, static_features)\n        \n        # Make predictions\n        predictions = self.model.predict({\n            'temporal_inputs': prepared_data['train']['temporal_inputs'],\n            'static_inputs': prepared_data['train']['static_inputs'],\n            'known_future_mask': prepared_data['train']['known_future_mask']\n        })\n        \n        return predictions\n    \n    def get_feature_importance(self, data: pd.DataFrame,\n                             target_column: str = 'close',\n                             static_features: Optional[List[str]] = None) -> Dict:\n        \"\"\"\n        Extract feature importance from trained model.\n        \n        Args:\n            data: Input data\n            target_column: Target column name\n            static_features: Static feature columns\n            \n        Returns:\n            Feature importance analysis\n        \"\"\"\n        if self.model is None:\n            raise ValueError(\"Model must be trained first\")\n            \n        # Prepare data\n        prepared_data = self.prepare_data(data, target_column, static_features)\n        \n        # Get feature importance\n        importance = self.model.get_feature_importance({\n            'temporal_inputs': prepared_data['train']['temporal_inputs'][:100],  # Sample\n            'static_inputs': prepared_data['train']['static_inputs'][:100],\n            'known_future_mask': prepared_data['train']['known_future_mask'][:100]\n        })\n        \n        # Add feature names\n        importance['feature_names'] = prepared_data['feature_names']\n        importance['static_feature_names'] = prepared_data['static_feature_names']\n        \n        return importance\n    \n    def save_model(self, filepath: str):\n        \"\"\"Save trained model to file.\"\"\"\n        if self.model is None:\n            raise ValueError(\"No model to save\")\n            \n        save_data = {\n            'model_params': self.model.params,\n            'config': self.config,\n            'training_history': self.training_history\n        }\n        \n        np.savez_compressed(filepath, **save_data)\n        \n    def load_model(self, filepath: str):\n        \"\"\"Load trained model from file.\"\"\"\n        loaded_data = np.load(filepath, allow_pickle=True)\n        \n        # Reconstruct config\n        config_dict = loaded_data['config'].item()\n        self.config = TFTConfig(**config_dict)\n        \n        # Reconstruct model\n        self.model = TemporalFusionTransformer(\n            n_encoder_steps=self.config.n_encoder_steps,\n            n_prediction_steps=self.config.n_prediction_steps,\n            n_features=self.config.n_features,\n            n_static_features=self.config.n_static_features,\n            hidden_size=self.config.hidden_size,\n            lstm_layers=self.config.lstm_layers,\n            num_attention_heads=self.config.num_attention_heads,\n            dropout_rate=self.config.dropout_rate,\n            quantiles=self.config.quantiles\n        )\n        \n        # Load parameters\n        self.model.params = loaded_data['model_params'].item()\n        self.training_history = loaded_data['training_history'].tolist()"