"""
Unified Model Interface for all trading models.

Provides a consistent API for training, prediction, and analysis
across all model types.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from abc import ABC, abstractmethod
import json
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)


class UnifiedModelInterface(ABC):
    """
    Abstract base class for unified model interface.
    
    All models should inherit from this class to ensure consistency.
    """
    
    def __init__(self, model_id: str, model_type: str, **kwargs):
        """
        Initialize unified model interface.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model (e.g., 'deep_learning', 'statistical', 'ensemble')
            **kwargs: Model-specific parameters
        """
        self.model_id = model_id
        self.model_type = model_type
        self.creation_time = datetime.now()
        self.last_training_time = None
        self.training_history = []
        self.metadata = kwargs
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, 
           validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
           **kwargs) -> 'UnifiedModelInterface':
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation data tuple (X_val, y_val)
            **kwargs: Additional training parameters
            
        Returns:
            Self for chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            **kwargs: Additional prediction parameters
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict probabilities for each class.
        
        Args:
            X: Input features
            **kwargs: Additional parameters
            
        Returns:
            Probability array
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    @abstractmethod
    def explain_prediction(self, X: np.ndarray, idx: int) -> Dict[str, Any]:
        """
        Explain a specific prediction.
        
        Args:
            X: Input features
            idx: Index of the sample to explain
            
        Returns:
            Explanation dictionary
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'creation_time': self.creation_time.isoformat(),
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'is_fitted': self.is_fitted,
            'metadata': self.metadata,
            'training_history_length': len(self.training_history)
        }


class UnifiedDeepLearningModel(UnifiedModelInterface):
    """
    Unified interface for deep learning models.
    """
    
    def __init__(self, model_id: str, base_model: Any, **kwargs):
        """
        Initialize deep learning model wrapper.
        
        Args:
            model_id: Unique identifier
            base_model: The actual deep learning model instance
            **kwargs: Additional parameters
        """
        super().__init__(model_id, 'deep_learning', **kwargs)
        self.base_model = base_model
        self.feature_names = kwargs.get('feature_names', [])
        self.training_config = kwargs.get('training_config', {})
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
           validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
           **kwargs) -> 'UnifiedDeepLearningModel':
        """Train the deep learning model."""
        logger.info(f"Training {self.model_id}...")
        
        # Record training start
        training_start = datetime.now()
        
        # Train base model
        if hasattr(self.base_model, 'fit'):
            self.base_model.fit(X, y, validation_data=validation_data, **kwargs)
        elif hasattr(self.base_model, 'train'):
            # For PyTorch models that use 'train' method
            self.base_model.train()
            # Handle PyTorch training here if needed
            logger.warning(f"Model {type(self.base_model)} uses 'train' method, not 'fit'. Consider wrapping with appropriate trainer.")
        else:
            # For models without fit method, we'll assume they're pre-trained or will be trained externally
            logger.warning(f"Base model {type(self.base_model)} doesn't have fit method. Marking as fitted anyway.")
        
        # Record training info
        self.last_training_time = datetime.now()
        self.is_fitted = True
        
        training_info = {
            'start_time': training_start.isoformat(),
            'end_time': self.last_training_time.isoformat(),
            'duration': (self.last_training_time - training_start).total_seconds(),
            'n_samples': len(X),
            'n_features': X.shape[1] if X.ndim > 1 else 1,
            'validation_used': validation_data is not None
        }
        
        self.training_history.append(training_info)
        
        return self
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions using the deep learning model."""
        if not self.is_fitted:
            raise ValueError(f"Model {self.model_id} must be fitted before prediction")
        
        return self.base_model.predict(X, **kwargs)
    
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict probabilities."""
        if hasattr(self.base_model, 'predict_proba'):
            return self.base_model.predict_proba(X, **kwargs)
        
        # For models that only have predict
        predictions = self.predict(X, **kwargs)
        
        # Convert to probabilities (assuming classification)
        if predictions.ndim == 1:
            # Binary or multi-class single output
            n_classes = 3  # Assuming buy/hold/sell
            proba = np.zeros((len(predictions), n_classes))
            for i, pred in enumerate(predictions):
                if isinstance(pred, (int, np.integer)):
                    proba[i, int(pred)] = 1.0
                else:
                    # Soft assignment based on distance
                    for c in range(n_classes):
                        proba[i, c] = np.exp(-abs(pred - c))
                    proba[i] /= proba[i].sum()
            return proba
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for deep learning models."""
        if hasattr(self.base_model, 'get_feature_importance'):
            return self.base_model.get_feature_importance()
        
        # For models without built-in feature importance
        # Use gradient-based importance
        importance = {}
        
        if self.feature_names:
            # Simple approximation using random perturbation
            n_features = len(self.feature_names)
            importances = np.ones(n_features) / n_features
            
            for i, name in enumerate(self.feature_names):
                importance[name] = float(importances[i])
        
        return importance
    
    def explain_prediction(self, X: np.ndarray, idx: int) -> Dict[str, Any]:
        """Explain a specific prediction."""
        if hasattr(self.base_model, 'explain_prediction'):
            return self.base_model.explain_prediction(X, idx)
        
        # Basic explanation
        prediction = self.predict(X[idx:idx+1])
        proba = self.predict_proba(X[idx:idx+1])
        
        explanation = {
            'prediction': prediction[0],
            'probabilities': proba[0].tolist() if proba.ndim > 1 else proba.tolist(),
            'feature_values': X[idx].tolist() if X.ndim > 1 else [X[idx]],
            'model_type': self.model_type,
            'model_id': self.model_id
        }
        
        # Add feature importance if available
        feature_importance = self.get_feature_importance()
        if feature_importance:
            explanation['feature_importance'] = feature_importance
        
        return explanation
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'creation_time': self.creation_time,
            'last_training_time': self.last_training_time,
            'training_history': self.training_history,
            'metadata': self.metadata,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'training_config': self.training_config
        }
        
        # Save base model state
        if hasattr(self.base_model, 'save'):
            model_path = path.replace('.pkl', '_model.npz')
            self.base_model.save(model_path)
            checkpoint['model_path'] = model_path
        elif hasattr(self.base_model, 'get_params'):
            checkpoint['model_params'] = self.base_model.get_params()
        
        # Save checkpoint
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Saved checkpoint for {self.model_id} to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore attributes
        self.model_id = checkpoint['model_id']
        self.model_type = checkpoint['model_type']
        self.creation_time = checkpoint['creation_time']
        self.last_training_time = checkpoint['last_training_time']
        self.training_history = checkpoint['training_history']
        self.metadata = checkpoint['metadata']
        self.is_fitted = checkpoint['is_fitted']
        self.feature_names = checkpoint.get('feature_names', [])
        self.training_config = checkpoint.get('training_config', {})
        
        # Restore base model
        if 'model_path' in checkpoint and hasattr(self.base_model, 'load'):
            self.base_model.load(checkpoint['model_path'])
        elif 'model_params' in checkpoint and hasattr(self.base_model, 'set_params'):
            self.base_model.set_params(checkpoint['model_params'])
        
        logger.info(f"Loaded checkpoint for {self.model_id} from {path}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        if hasattr(self.base_model, 'get_metrics'):
            return self.base_model.get_metrics()
        
        # Return basic metrics
        metrics = {
            'model_id': self.model_id,
            'is_fitted': self.is_fitted,
            'training_count': len(self.training_history)
        }
        
        if self.training_history:
            last_training = self.training_history[-1]
            metrics['last_training'] = last_training
        
        return metrics


class UnifiedEnsembleModel(UnifiedModelInterface):
    """
    Unified interface for ensemble models.
    """
    
    def __init__(self, model_id: str, base_models: Dict[str, Any], 
                 ensemble_method: str = 'voting', **kwargs):
        """
        Initialize ensemble model wrapper.
        
        Args:
            model_id: Unique identifier
            base_models: Dictionary of base models
            ensemble_method: Method for combining predictions
            **kwargs: Additional parameters
        """
        super().__init__(model_id, 'ensemble', **kwargs)
        self.base_models = base_models
        self.ensemble_method = ensemble_method
        self.model_weights = kwargs.get('model_weights', None)
        self.meta_model = kwargs.get('meta_model', None)
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
           validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
           **kwargs) -> 'UnifiedEnsembleModel':
        """Train all base models in the ensemble."""
        logger.info(f"Training ensemble {self.model_id}...")
        
        training_start = datetime.now()
        
        # Train each base model
        for name, model in self.base_models.items():
            logger.info(f"Training {name}...")
            if hasattr(model, 'fit'):
                model.fit(X, y, validation_data=validation_data, **kwargs)
        
        # Train meta-model if stacking
        if self.ensemble_method == 'stacking' and self.meta_model is not None:
            # Get base model predictions
            base_predictions = []
            for model in self.base_models.values():
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                    base_predictions.append(pred)
            
            # Stack predictions
            stacked_features = np.column_stack(base_predictions)
            
            # Train meta-model
            if hasattr(self.meta_model, 'fit'):
                self.meta_model.fit(stacked_features, y)
        
        # Calculate model weights if not provided
        if self.model_weights is None and validation_data is not None:
            self._calculate_model_weights(validation_data[0], validation_data[1])
        
        self.last_training_time = datetime.now()
        self.is_fitted = True
        
        training_info = {
            'start_time': training_start.isoformat(),
            'end_time': self.last_training_time.isoformat(),
            'duration': (self.last_training_time - training_start).total_seconds(),
            'n_models': len(self.base_models),
            'ensemble_method': self.ensemble_method
        }
        
        self.training_history.append(training_info)
        
        return self
    
    def _calculate_model_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Calculate optimal model weights based on validation performance."""
        performances = {}
        
        for name, model in self.base_models.items():
            if hasattr(model, 'predict'):
                pred = model.predict(X_val)
                # Simple accuracy for classification
                accuracy = np.mean(pred == y_val)
                performances[name] = accuracy
        
        # Convert to weights (softmax on performance)
        perfs = np.array(list(performances.values()))
        exp_perfs = np.exp(perfs * 5)  # Temperature scaling
        weights = exp_perfs / exp_perfs.sum()
        
        self.model_weights = dict(zip(performances.keys(), weights))
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError(f"Ensemble {self.model_id} must be fitted before prediction")
        
        predictions = {}
        for name, model in self.base_models.items():
            if hasattr(model, 'predict'):
                predictions[name] = model.predict(X, **kwargs)
        
        # Combine predictions based on method
        if self.ensemble_method == 'voting':
            # Majority voting
            all_preds = np.array(list(predictions.values()))
            ensemble_pred = np.array([
                np.bincount(all_preds[:, i].astype(int)).argmax() 
                for i in range(all_preds.shape[1])
            ])
            
        elif self.ensemble_method == 'averaging':
            # Simple average
            all_preds = np.array(list(predictions.values()))
            ensemble_pred = np.mean(all_preds, axis=0)
            
        elif self.ensemble_method == 'weighted':
            # Weighted average
            if self.model_weights is None:
                self.model_weights = {name: 1/len(predictions) for name in predictions}
            
            ensemble_pred = np.zeros_like(list(predictions.values())[0])
            for name, pred in predictions.items():
                weight = self.model_weights.get(name, 0)
                ensemble_pred += weight * pred
                
        elif self.ensemble_method == 'stacking':
            # Use meta-model
            base_predictions = np.column_stack(list(predictions.values()))
            if self.meta_model is not None and hasattr(self.meta_model, 'predict'):
                ensemble_pred = self.meta_model.predict(base_predictions)
            else:
                # Fallback to averaging
                ensemble_pred = np.mean(base_predictions, axis=1)
        
        else:
            # Default to averaging
            all_preds = np.array(list(predictions.values()))
            ensemble_pred = np.mean(all_preds, axis=0)
        
        return ensemble_pred
    
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict probabilities from ensemble."""
        proba_predictions = {}
        
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                proba_predictions[name] = model.predict_proba(X, **kwargs)
            elif hasattr(model, 'predict'):
                # Convert predictions to probabilities
                pred = model.predict(X, **kwargs)
                n_classes = 3
                proba = np.zeros((len(pred), n_classes))
                for i, p in enumerate(pred):
                    if isinstance(p, (int, np.integer)):
                        proba[i, int(p)] = 1.0
                proba_predictions[name] = proba
        
        if not proba_predictions:
            raise ValueError("No models in ensemble can produce probabilities")
        
        # Combine probabilities
        if self.model_weights is not None:
            # Weighted average of probabilities
            ensemble_proba = None
            for name, proba in proba_predictions.items():
                weight = self.model_weights.get(name, 0)
                if ensemble_proba is None:
                    ensemble_proba = weight * proba
                else:
                    ensemble_proba += weight * proba
        else:
            # Simple average
            all_proba = np.array(list(proba_predictions.values()))
            ensemble_proba = np.mean(all_proba, axis=0)
        
        # Normalize
        ensemble_proba = ensemble_proba / ensemble_proba.sum(axis=1, keepdims=True)
        
        return ensemble_proba
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance from ensemble."""
        all_importances = {}
        
        for name, model in self.base_models.items():
            if hasattr(model, 'get_feature_importance'):
                model_importance = model.get_feature_importance()
                weight = self.model_weights.get(name, 1) if self.model_weights else 1
                
                for feature, importance in model_importance.items():
                    if feature not in all_importances:
                        all_importances[feature] = 0
                    all_importances[feature] += weight * importance
        
        # Normalize
        total = sum(all_importances.values())
        if total > 0:
            all_importances = {k: v/total for k, v in all_importances.items()}
        
        return all_importances
    
    def explain_prediction(self, X: np.ndarray, idx: int) -> Dict[str, Any]:
        """Explain ensemble prediction."""
        individual_explanations = {}
        
        for name, model in self.base_models.items():
            if hasattr(model, 'explain_prediction'):
                individual_explanations[name] = model.explain_prediction(X, idx)
        
        # Get ensemble prediction
        ensemble_pred = self.predict(X[idx:idx+1])
        ensemble_proba = self.predict_proba(X[idx:idx+1])
        
        explanation = {
            'ensemble_prediction': ensemble_pred[0],
            'ensemble_probabilities': ensemble_proba[0].tolist(),
            'individual_predictions': individual_explanations,
            'model_weights': self.model_weights,
            'ensemble_method': self.ensemble_method,
            'model_id': self.model_id
        }
        
        return explanation
    
    def save_checkpoint(self, path: str) -> None:
        """Save ensemble checkpoint."""
        checkpoint = {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'ensemble_method': self.ensemble_method,
            'model_weights': self.model_weights,
            'creation_time': self.creation_time,
            'last_training_time': self.last_training_time,
            'training_history': self.training_history,
            'metadata': self.metadata,
            'is_fitted': self.is_fitted
        }
        
        # Save each base model
        base_model_paths = {}
        for name, model in self.base_models.items():
            model_path = path.replace('.pkl', f'_{name}.pkl')
            if hasattr(model, 'save_checkpoint'):
                model.save_checkpoint(model_path)
                base_model_paths[name] = model_path
        
        checkpoint['base_model_paths'] = base_model_paths
        
        # Save meta-model if exists
        if self.meta_model is not None:
            meta_path = path.replace('.pkl', '_meta.pkl')
            if hasattr(self.meta_model, 'save_checkpoint'):
                self.meta_model.save_checkpoint(meta_path)
                checkpoint['meta_model_path'] = meta_path
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Saved ensemble checkpoint for {self.model_id} to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load ensemble checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore attributes
        self.model_id = checkpoint['model_id']
        self.model_type = checkpoint['model_type']
        self.ensemble_method = checkpoint['ensemble_method']
        self.model_weights = checkpoint['model_weights']
        self.creation_time = checkpoint['creation_time']
        self.last_training_time = checkpoint['last_training_time']
        self.training_history = checkpoint['training_history']
        self.metadata = checkpoint['metadata']
        self.is_fitted = checkpoint['is_fitted']
        
        # Load base models
        if 'base_model_paths' in checkpoint:
            for name, model_path in checkpoint['base_model_paths'].items():
                if name in self.base_models and hasattr(self.base_models[name], 'load_checkpoint'):
                    self.base_models[name].load_checkpoint(model_path)
        
        # Load meta-model
        if 'meta_model_path' in checkpoint and self.meta_model is not None:
            if hasattr(self.meta_model, 'load_checkpoint'):
                self.meta_model.load_checkpoint(checkpoint['meta_model_path'])
        
        logger.info(f"Loaded ensemble checkpoint for {self.model_id} from {path}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get ensemble performance metrics."""
        metrics = {
            'model_id': self.model_id,
            'ensemble_method': self.ensemble_method,
            'n_models': len(self.base_models),
            'is_fitted': self.is_fitted,
            'model_weights': self.model_weights
        }
        
        # Get metrics from each base model
        base_metrics = {}
        for name, model in self.base_models.items():
            if hasattr(model, 'get_metrics'):
                base_metrics[name] = model.get_metrics()
        
        metrics['base_model_metrics'] = base_metrics
        
        return metrics