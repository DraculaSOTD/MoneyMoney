import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import json
import os

from .tcn_model import TCNModel

logger = logging.getLogger(__name__)


class TCNTrainer:
    """
    Trainer for Temporal Convolutional Network model.
    Handles training, validation, and evaluation of TCN for trading signals.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TCN trainer.
        
        Args:
            config: Configuration dictionary with model and training parameters
        """
        self.config = config
        self.model = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_sharpe': [],
            'val_sharpe': []
        }
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def build_model(self, input_size: int) -> TCNModel:
        """
        Build TCN model with configuration.
        
        Args:
            input_size: Number of input features
            
        Returns:
            Initialized TCN model
        """
        model_config = self.config.get('model', {})
        
        self.model = TCNModel(
            input_size=input_size,
            hidden_channels=model_config.get('hidden_channels', 64),
            kernel_size=model_config.get('kernel_size', 3),
            num_layers=model_config.get('num_layers', 4),
            dilations=model_config.get('dilations', None),
            dropout_rate=model_config.get('dropout_rate', 0.2),
            learning_rate=model_config.get('learning_rate', 0.001),
            batch_size=model_config.get('batch_size', 32),
            epochs=model_config.get('epochs', 100),
            use_skip_connections=model_config.get('use_skip_connections', True),
            activation=model_config.get('activation', 'relu')
        )
        
        logger.info(f"Built TCN model with {model_config.get('num_layers', 4)} layers, "
                   f"kernel size {model_config.get('kernel_size', 3)}, hidden channels {model_config.get('hidden_channels', 64)}")
        
        return self.model
    
    def prepare_sequences(self, X: np.ndarray, y: np.ndarray, 
                         sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for TCN training.
        
        Args:
            X: Feature data
            y: Target labels
            sequence_length: Length of sequences
            
        Returns:
            Sequences and corresponding labels
        """
        n_samples = len(X) - sequence_length + 1
        n_features = X.shape[1]
        
        sequences = np.zeros((n_samples, sequence_length, n_features))
        labels = np.zeros(n_samples)
        
        for i in range(n_samples):
            sequences[i] = X[i:i + sequence_length]
            labels[i] = y[i + sequence_length - 1]
            
        return sequences, labels
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train TCN model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training results and metrics
        """
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        # Prepare sequences
        sequence_length = self.config.get('sequence_length', 60)
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train, sequence_length)
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val, sequence_length)
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None
        
        # Train model
        logger.info("Starting TCN training...")
        start_time = datetime.now()
        
        # Use model's fit method
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val) if X_val is not None else None)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate final performance
        train_metrics = self.evaluate(X_train, y_train)
        val_metrics = self.evaluate(X_val, y_val) if X_val is not None else None
        
        results = {
            'training_time': training_time,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'model_params': self.model.get_params(),
            'training_history': self.training_history
        }
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final train accuracy: {train_metrics['accuracy']:.4f}")
        if val_metrics:
            logger.info(f"Final val accuracy: {val_metrics['accuracy']:.4f}")
        
        return results
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature data
            y: True labels
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None or not self.model.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Handle case where predictions might be shorter due to sequence creation
        if len(predictions) < len(y):
            y = y[-len(predictions):]
        
        # Calculate metrics
        accuracy = np.mean(predictions == y)
        
        # Calculate per-class metrics
        classes = [0, 1, 2]  # buy, sell, hold
        precision = {}
        recall = {}
        f1_scores = {}
        
        for cls in classes:
            true_positive = np.sum((predictions == cls) & (y == cls))
            false_positive = np.sum((predictions == cls) & (y != cls))
            false_negative = np.sum((predictions != cls) & (y == cls))
            
            precision[cls] = true_positive / (true_positive + false_positive + 1e-8)
            recall[cls] = true_positive / (true_positive + false_negative + 1e-8)
            f1_scores[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls] + 1e-8)
        
        # Calculate trading metrics
        returns = self._calculate_returns(predictions, X)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        
        metrics = {
            'accuracy': accuracy,
            'precision_buy': precision[0],
            'precision_sell': precision[1],
            'precision_hold': precision[2],
            'recall_buy': recall[0],
            'recall_sell': recall[1],
            'recall_hold': recall[2],
            'f1_buy': f1_scores[0],
            'f1_sell': f1_scores[1],
            'f1_hold': f1_scores[2],
            'sharpe_ratio': sharpe_ratio,
            'total_predictions': len(predictions),
            'buy_signals': np.sum(predictions == 0),
            'sell_signals': np.sum(predictions == 1),
            'hold_signals': np.sum(predictions == 2)
        }
        
        return metrics
    
    def _calculate_returns(self, predictions: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Calculate returns based on predictions."""
        # Assuming last feature is price or return
        if X.shape[1] > 0:
            prices = X[-len(predictions):, -1]
            returns = np.diff(prices) / prices[:-1]
            
            # Apply trading signals
            # Buy = 0, Sell = 1, Hold = 2
            trading_returns = np.zeros(len(returns))
            position = 0  # 0 = no position, 1 = long, -1 = short
            
            for i in range(len(returns)):
                if i < len(predictions) - 1:
                    if predictions[i] == 0:  # Buy signal
                        position = 1
                    elif predictions[i] == 1:  # Sell signal
                        position = -1
                    # Hold keeps current position
                    
                    trading_returns[i] = position * returns[i]
            
            return trading_returns
        return np.zeros(len(predictions) - 1)
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if np.std(excess_returns) > 0:
            return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
        return 0.0
    
    def save_model(self, filepath: str):
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model_params': self.model.get_params(),
            'config': self.config,
            'training_history': self.training_history
        }
        
        np.savez(filepath, **model_data)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model."""
        data = np.load(filepath, allow_pickle=True)
        
        # Reconstruct model
        model_params = data['model_params'].item()
        self.config = data['config'].item()
        self.training_history = data['training_history'].item()
        
        # Build and set model parameters
        input_size = model_params['input_size']
        self.build_model(input_size)
        self.model.set_params(model_params)
        
        logger.info(f"Model loaded from {filepath}")