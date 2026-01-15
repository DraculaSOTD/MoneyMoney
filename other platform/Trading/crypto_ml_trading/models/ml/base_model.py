import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod
import json
import os
from pathlib import Path


class BaseMLModel(ABC):
    """Abstract base class for all ML trading models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.history = None
        self.scaler = None
        self.feature_names = []
        self.model_name = self.__class__.__name__
        
    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...], n_classes: int = 3):
        """
        Build the model architecture.
        
        Args:
            input_shape: Shape of input data
            n_classes: Number of output classes
        """
        pass
    
    @abstractmethod
    def prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare data for model input.
        
        Args:
            X: Input features
            y: Target labels (optional)
            
        Returns:
            Prepared (X, y) tuple
        """
        pass
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None, 
             y_val: Optional[np.ndarray] = None,
             **kwargs) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Training history dictionary
        """
        # Prepare data
        X_train, y_train = self.prepare_data(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            X_val, y_val = self.prepare_data(X_val, y_val)
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        # Build model if not already built
        if self.model is None:
            n_classes = len(np.unique(y_train))
            self.build_model(X_train.shape[1:], n_classes)
        
        # Default training parameters
        train_params = {
            'epochs': self.config.get('epochs', 100),
            'batch_size': self.config.get('batch_size', 32),
            'verbose': kwargs.get('verbose', 1),
            'validation_data': validation_data
        }
        
        # Update with any provided parameters
        train_params.update(kwargs)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            **train_params
        )
        
        return self.history.history
    
    def predict(self, X: np.ndarray, return_proba: bool = False) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            return_proba: Return probabilities instead of classes
            
        Returns:
            Predictions (classes or probabilities)
        """
        X, _ = self.prepare_data(X, None)
        
        if return_proba:
            return self.model.predict(X)
        else:
            proba = self.model.predict(X)
            return np.argmax(proba, axis=1)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics dictionary
        """
        X_test, y_test = self.prepare_data(X_test, y_test)
        
        # Get model metrics
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions for additional metrics
        y_pred = self.predict(X_test)
        y_proba = self.predict(X_test, return_proba=True)
        
        # Calculate additional metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Convert numeric predictions back to labels if needed
        label_map = {0: 'buy', 1: 'sell', 2: 'hold'}
        y_test_labels = [label_map.get(y, y) for y in y_test]
        y_pred_labels = [label_map.get(y, y) for y in y_pred]
        
        report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist()
        }
    
    def save_model(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: Directory to save model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = path / f"{self.model_name}.keras"
        self.model.save(model_path)
        
        # Save configuration
        config_path = path / f"{self.model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save training history if available
        if self.history:
            history_path = path / f"{self.model_name}_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.history.history, f, indent=2)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load model from disk.
        
        Args:
            path: Directory containing saved model
        """
        path = Path(path)
        
        # Load model
        model_path = path / f"{self.model_name}.keras"
        if model_path.exists():
            from tensorflow.keras.models import load_model
            self.model = load_model(model_path)
        
        # Load configuration
        config_path = path / f"{self.model_name}_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        print(f"Model loaded from {path}")
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model:
            from io import StringIO
            import sys

            old_stdout = sys.stdout
            sys.stdout = buffer = StringIO()

            self.model.summary()

            sys.stdout = old_stdout
            return buffer.getvalue()
        return "Model not built yet"

    def save(self, path: str):
        """Save model to disk (alias for save_model)."""
        self.save_model(path)

    def load(self, path: str):
        """Load model from disk (alias for load_model)."""
        self.load_model(path)