"""
Base Ensemble Framework for Hierarchical Architecture.

Implements the foundational ensemble system with model registration and coordination.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import abc
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


@dataclass
class ModelPrediction:
    """Individual model prediction with metadata."""
    model_id: str
    timestamp: datetime
    prediction: Union[float, np.ndarray]
    confidence: float
    horizon: int  # prediction horizon in minutes
    features_used: List[str]
    model_type: str
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsemblePrediction:
    """Ensemble prediction with component analysis."""
    timestamp: datetime
    prediction: Union[float, np.ndarray]
    confidence: float
    horizon: int
    component_predictions: List[ModelPrediction]
    weights: Dict[str, float]
    ensemble_method: str
    disagreement: float
    stability: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformance:
    """Model performance tracking."""
    model_id: str
    timestamp: datetime
    accuracy_metrics: Dict[str, float]
    prediction_errors: List[float]
    confidence_calibration: float
    computational_cost: float
    stability_score: float
    recent_performance: float
    long_term_performance: float


class BaseModel(abc.ABC):
    """Abstract base class for ensemble models."""
    
    def __init__(self, model_id: str, model_type: str):
        self.model_id = model_id
        self.model_type = model_type
        self.is_trained = False
        self.last_update = None
        
    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the model."""
        pass
    
    @abc.abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> ModelPrediction:
        """Make predictions."""
        pass
    
    @abc.abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'last_update': self.last_update.isoformat() if self.last_update else None
        }


class BaseEnsemble:
    """
    Base ensemble framework for hierarchical model combination.
    
    Features:
    - Model registration and lifecycle management
    - Multi-horizon prediction support
    - Dynamic weight calculation
    - Performance tracking and attribution
    - Uncertainty quantification
    - Model disagreement analysis
    """
    
    def __init__(self,
                 ensemble_id: str,
                 combination_method: str = "weighted_average",
                 performance_window: int = 100,
                 min_models_required: int = 2):
        """
        Initialize base ensemble.
        
        Args:
            ensemble_id: Unique identifier for ensemble
            combination_method: Method for combining predictions
            performance_window: Window for performance tracking
            min_models_required: Minimum models needed for ensemble prediction
        """
        self.ensemble_id = ensemble_id
        self.combination_method = combination_method
        self.performance_window = performance_window
        self.min_models_required = min_models_required
        
        # Model registry
        self.models: Dict[str, BaseModel] = {}
        self.model_weights: Dict[str, float] = {}
        self.model_performance: Dict[str, ModelPerformance] = {}
        
        # Prediction history
        self.prediction_history: deque = deque(maxlen=1000)
        self.component_predictions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        
        # Performance tracking
        self.ensemble_performance: deque = deque(maxlen=200)
        self.weight_history: deque = deque(maxlen=200)
        
        # Combination methods
        self.combination_methods = {
            'weighted_average': self._weighted_average,
            'rank_based': self._rank_based_combination,
            'bayesian_model_averaging': self._bayesian_model_averaging,
            'stacking': self._stacking_combination,
            'dynamic_selection': self._dynamic_selection
        }
        
        # State tracking
        self.last_prediction_time: Optional[datetime] = None
        self.prediction_horizons: List[int] = [5, 15, 30, 60, 240]  # minutes
    
    def register_model(self, model: BaseModel, initial_weight: float = 1.0) -> None:
        """
        Register a model with the ensemble.
        
        Args:
            model: Model instance to register
            initial_weight: Initial weight for the model
        """
        self.models[model.model_id] = model
        self.model_weights[model.model_id] = initial_weight
        
        # Initialize performance tracking
        self.model_performance[model.model_id] = ModelPerformance(
            model_id=model.model_id,
            timestamp=datetime.now(),
            accuracy_metrics={},
            prediction_errors=[],
            confidence_calibration=0.5,
            computational_cost=0.0,
            stability_score=0.5,
            recent_performance=0.5,
            long_term_performance=0.5
        )
        
        print(f"Registered model {model.model_id} with ensemble {self.ensemble_id}")
    
    def remove_model(self, model_id: str) -> None:
        """Remove a model from the ensemble."""
        if model_id in self.models:
            del self.models[model_id]
            del self.model_weights[model_id]
            del self.model_performance[model_id]
            print(f"Removed model {model_id} from ensemble {self.ensemble_id}")
    
    def predict(self,
                X: np.ndarray,
                horizon: int = 15,
                return_components: bool = False,
                **kwargs) -> EnsemblePrediction:
        """
        Generate ensemble prediction.
        
        Args:
            X: Input features
            horizon: Prediction horizon in minutes
            return_components: Whether to return component predictions
            
        Returns:
            Ensemble prediction with metadata
        """
        if len(self.models) < self.min_models_required:
            raise ValueError(f"Ensemble requires at least {self.min_models_required} models")
        
        timestamp = datetime.now()
        component_predictions = []
        
        # Collect predictions from all models
        for model_id, model in self.models.items():
            if model.is_trained:
                try:
                    start_time = datetime.now()
                    prediction = model.predict(X, horizon=horizon, **kwargs)
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    prediction.processing_time = processing_time
                    component_predictions.append(prediction)
                    
                    # Store component prediction
                    self.component_predictions[model_id].append(prediction)
                    
                except Exception as e:
                    print(f"Error getting prediction from model {model_id}: {e}")
                    continue
        
        if not component_predictions:
            raise RuntimeError("No models produced valid predictions")
        
        # Combine predictions
        ensemble_prediction = self._combine_predictions(
            component_predictions, horizon, timestamp
        )
        
        # Store prediction
        self.prediction_history.append(ensemble_prediction)
        self.last_prediction_time = timestamp
        
        # Update weights based on recent performance
        self._update_model_weights()
        
        return ensemble_prediction
    
    def _combine_predictions(self,
                           predictions: List[ModelPrediction],
                           horizon: int,
                           timestamp: datetime) -> EnsemblePrediction:
        """Combine individual model predictions into ensemble prediction."""
        method = self.combination_methods.get(
            self.combination_method, 
            self._weighted_average
        )
        
        return method(predictions, horizon, timestamp)
    
    def _weighted_average(self,
                         predictions: List[ModelPrediction],
                         horizon: int,
                         timestamp: datetime) -> EnsemblePrediction:
        """Weighted average combination method."""
        # Get weights for participating models
        weights = []
        values = []
        confidences = []
        
        for pred in predictions:
            weight = self.model_weights.get(pred.model_id, 1.0)
            weights.append(weight)
            
            if isinstance(pred.prediction, np.ndarray):
                values.append(pred.prediction)
            else:
                values.append(float(pred.prediction))
            
            confidences.append(pred.confidence)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Calculate weighted prediction
        if isinstance(values[0], np.ndarray):
            # Multi-dimensional prediction
            weighted_pred = np.average(values, weights=weights, axis=0)
        else:
            # Scalar prediction
            weighted_pred = np.average(values, weights=weights)
        
        # Calculate ensemble confidence
        weighted_confidence = np.average(confidences, weights=weights)
        
        # Calculate disagreement
        if len(values) > 1:
            if isinstance(values[0], np.ndarray):
                disagreement = np.mean([np.std([v[i] for v in values]) for i in range(len(values[0]))])
            else:
                disagreement = np.std(values)
        else:
            disagreement = 0.0
        
        # Calculate stability (based on weight distribution)
        stability = 1.0 - np.std(weights) if len(weights) > 1 else 1.0
        
        # Create weight dictionary
        weight_dict = {pred.model_id: w for pred, w in zip(predictions, weights)}
        
        return EnsemblePrediction(
            timestamp=timestamp,
            prediction=weighted_pred,
            confidence=weighted_confidence,
            horizon=horizon,
            component_predictions=predictions,
            weights=weight_dict,
            ensemble_method='weighted_average',
            disagreement=disagreement,
            stability=stability,
            metadata={'normalization_factor': np.sum(self.model_weights.values())}
        )
    
    def _rank_based_combination(self,
                              predictions: List[ModelPrediction],
                              horizon: int,
                              timestamp: datetime) -> EnsemblePrediction:
        """Rank-based combination method."""
        # Sort predictions by confidence
        sorted_predictions = sorted(predictions, key=lambda x: x.confidence, reverse=True)
        
        # Assign rank-based weights
        n = len(sorted_predictions)
        rank_weights = [(n - i) / (n * (n + 1) / 2) for i in range(n)]
        
        # Apply rank weights
        values = []
        confidences = []
        
        for pred, weight in zip(sorted_predictions, rank_weights):
            if isinstance(pred.prediction, np.ndarray):
                values.append(pred.prediction * weight)
            else:
                values.append(float(pred.prediction) * weight)
            
            confidences.append(pred.confidence * weight)
        
        # Combine
        if isinstance(values[0], np.ndarray):
            combined_pred = np.sum(values, axis=0)
        else:
            combined_pred = sum(values)
        
        combined_confidence = sum(confidences)
        
        # Calculate disagreement and stability
        pred_values = [p.prediction for p in predictions]
        if isinstance(pred_values[0], np.ndarray):
            disagreement = np.mean([np.std([v[i] for v in pred_values]) for i in range(len(pred_values[0]))])
        else:
            disagreement = np.std(pred_values)
        
        stability = 1.0 / (1.0 + disagreement)
        
        # Create weight dictionary
        weight_dict = {pred.model_id: w for pred, w in zip(sorted_predictions, rank_weights)}
        
        return EnsemblePrediction(
            timestamp=timestamp,
            prediction=combined_pred,
            confidence=combined_confidence,
            horizon=horizon,
            component_predictions=predictions,
            weights=weight_dict,
            ensemble_method='rank_based',
            disagreement=disagreement,
            stability=stability
        )
    
    def _bayesian_model_averaging(self,
                                predictions: List[ModelPrediction],
                                horizon: int,
                                timestamp: datetime) -> EnsemblePrediction:
        """Bayesian model averaging combination."""
        # Use model performance as prior probabilities
        priors = []
        values = []
        confidences = []
        
        for pred in predictions:
            perf = self.model_performance.get(pred.model_id)
            if perf:
                prior = perf.recent_performance
            else:
                prior = 0.5  # Default
            
            priors.append(prior)
            
            if isinstance(pred.prediction, np.ndarray):
                values.append(pred.prediction)
            else:
                values.append(float(pred.prediction))
            
            confidences.append(pred.confidence)
        
        # Normalize priors to get posterior weights
        priors = np.array(priors)
        weights = priors / np.sum(priors)
        
        # Bayesian averaging
        if isinstance(values[0], np.ndarray):
            bayesian_pred = np.average(values, weights=weights, axis=0)
        else:
            bayesian_pred = np.average(values, weights=weights)
        
        # Uncertainty from model disagreement and individual uncertainties
        uncertainty = np.sqrt(
            np.average([(1 - c) ** 2 for c in confidences], weights=weights) +
            np.var(values) if not isinstance(values[0], np.ndarray) else np.mean(np.var(values, axis=0))
        )
        
        bayesian_confidence = max(0.0, 1.0 - uncertainty)
        
        # Calculate disagreement and stability
        if isinstance(values[0], np.ndarray):
            disagreement = np.mean(np.std(values, axis=0))
        else:
            disagreement = np.std(values)
        
        stability = np.exp(-disagreement)  # Exponential decay with disagreement
        
        # Create weight dictionary
        weight_dict = {pred.model_id: w for pred, w in zip(predictions, weights)}
        
        return EnsemblePrediction(
            timestamp=timestamp,
            prediction=bayesian_pred,
            confidence=bayesian_confidence,
            horizon=horizon,
            component_predictions=predictions,
            weights=weight_dict,
            ensemble_method='bayesian_model_averaging',
            disagreement=disagreement,
            stability=stability,
            metadata={'uncertainty': uncertainty}
        )
    
    def _stacking_combination(self,
                            predictions: List[ModelPrediction],
                            horizon: int,
                            timestamp: datetime) -> EnsemblePrediction:
        """Stacking-based combination (simplified version)."""
        # For now, implement as performance-weighted average
        # In full implementation, would train a meta-model
        
        weights = []
        values = []
        confidences = []
        
        for pred in predictions:
            perf = self.model_performance.get(pred.model_id)
            if perf and perf.recent_performance > 0:
                weight = perf.recent_performance ** 2  # Square for more discrimination
            else:
                weight = 0.25  # Low default weight
            
            weights.append(weight)
            
            if isinstance(pred.prediction, np.ndarray):
                values.append(pred.prediction)
            else:
                values.append(float(pred.prediction))
            
            confidences.append(pred.confidence)
        
        # Normalize weights
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        # Stacked prediction
        if isinstance(values[0], np.ndarray):
            stacked_pred = np.average(values, weights=weights, axis=0)
        else:
            stacked_pred = np.average(values, weights=weights)
        
        stacked_confidence = np.average(confidences, weights=weights)
        
        # Calculate disagreement and stability
        if isinstance(values[0], np.ndarray):
            disagreement = np.mean(np.std(values, axis=0))
        else:
            disagreement = np.std(values)
        
        stability = np.mean(weights) / np.std(weights) if np.std(weights) > 0 else 1.0
        
        # Create weight dictionary
        weight_dict = {pred.model_id: w for pred, w in zip(predictions, weights)}
        
        return EnsemblePrediction(
            timestamp=timestamp,
            prediction=stacked_pred,
            confidence=stacked_confidence,
            horizon=horizon,
            component_predictions=predictions,
            weights=weight_dict,
            ensemble_method='stacking',
            disagreement=disagreement,
            stability=stability
        )
    
    def _dynamic_selection(self,
                         predictions: List[ModelPrediction],
                         horizon: int,
                         timestamp: datetime) -> EnsemblePrediction:
        """Dynamic model selection based on recent performance."""
        if not predictions:
            raise ValueError("No predictions to select from")
        
        # Select best performing model for current conditions
        best_pred = None
        best_score = -np.inf
        
        for pred in predictions:
            perf = self.model_performance.get(pred.model_id)
            if perf:
                # Score based on recent performance and confidence
                score = perf.recent_performance * pred.confidence
            else:
                score = pred.confidence
            
            if score > best_score:
                best_score = score
                best_pred = pred
        
        if best_pred is None:
            best_pred = predictions[0]  # Fallback
        
        # Create ensemble prediction with single model
        weight_dict = {pred.model_id: 1.0 if pred == best_pred else 0.0 for pred in predictions}
        
        return EnsemblePrediction(
            timestamp=timestamp,
            prediction=best_pred.prediction,
            confidence=best_pred.confidence,
            horizon=horizon,
            component_predictions=predictions,
            weights=weight_dict,
            ensemble_method='dynamic_selection',
            disagreement=0.0,  # Single model, no disagreement
            stability=1.0,
            metadata={'selected_model': best_pred.model_id, 'selection_score': best_score}
        )
    
    def _update_model_weights(self) -> None:
        """Update model weights based on recent performance."""
        # Get recent predictions for weight calculation
        recent_window = 20
        if len(self.prediction_history) < recent_window:
            return
        
        recent_predictions = list(self.prediction_history)[-recent_window:]
        
        # Calculate weight updates based on performance
        model_scores = defaultdict(list)
        
        for ensemble_pred in recent_predictions:
            for component_pred in ensemble_pred.component_predictions:
                model_id = component_pred.model_id
                
                # Score based on confidence and contribution to ensemble
                contribution_weight = ensemble_pred.weights.get(model_id, 0.0)
                score = component_pred.confidence * contribution_weight * ensemble_pred.stability
                
                model_scores[model_id].append(score)
        
        # Update weights with exponential moving average
        alpha = 0.1  # Learning rate
        
        for model_id in self.model_weights:
            if model_id in model_scores and model_scores[model_id]:
                avg_score = np.mean(model_scores[model_id])
                
                # Exponential moving average update
                current_weight = self.model_weights[model_id]
                new_weight = alpha * avg_score + (1 - alpha) * current_weight
                
                self.model_weights[model_id] = max(0.01, min(2.0, new_weight))  # Bounded
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for model_id in self.model_weights:
                self.model_weights[model_id] /= total_weight
        
        # Store weight history
        self.weight_history.append({
            'timestamp': datetime.now(),
            'weights': self.model_weights.copy()
        })
    
    def update_performance(self,
                         actual_values: np.ndarray,
                         prediction_timestamps: List[datetime]) -> None:
        """
        Update model performance based on actual outcomes.
        
        Args:
            actual_values: Actual observed values
            prediction_timestamps: Timestamps of predictions to evaluate
        """
        for timestamp, actual in zip(prediction_timestamps, actual_values):
            # Find matching predictions
            matching_predictions = [
                p for p in self.prediction_history
                if abs((p.timestamp - timestamp).total_seconds()) < 300  # 5 minute tolerance
            ]
            
            for ensemble_pred in matching_predictions:
                self._update_ensemble_performance(ensemble_pred, actual)
                self._update_component_performance(ensemble_pred, actual)
    
    def _update_ensemble_performance(self,
                                   prediction: EnsemblePrediction,
                                   actual: float) -> None:
        """Update ensemble-level performance metrics."""
        if isinstance(prediction.prediction, np.ndarray):
            error = np.mean(np.abs(prediction.prediction - actual))
        else:
            error = abs(prediction.prediction - actual)
        
        accuracy = max(0.0, 1.0 - error)
        
        performance_record = {
            'timestamp': prediction.timestamp,
            'accuracy': accuracy,
            'error': error,
            'confidence': prediction.confidence,
            'disagreement': prediction.disagreement,
            'stability': prediction.stability
        }
        
        self.ensemble_performance.append(performance_record)
    
    def _update_component_performance(self,
                                    ensemble_pred: EnsemblePrediction,
                                    actual: float) -> None:
        """Update individual model performance metrics."""
        for component_pred in ensemble_pred.component_predictions:
            model_id = component_pred.model_id
            
            if isinstance(component_pred.prediction, np.ndarray):
                error = np.mean(np.abs(component_pred.prediction - actual))
            else:
                error = abs(component_pred.prediction - actual)
            
            accuracy = max(0.0, 1.0 - error)
            
            # Update model performance
            if model_id in self.model_performance:
                perf = self.model_performance[model_id]
                
                # Add to error history
                perf.prediction_errors.append(error)
                if len(perf.prediction_errors) > self.performance_window:
                    perf.prediction_errors.pop(0)
                
                # Update performance metrics
                perf.accuracy_metrics['recent_accuracy'] = accuracy
                perf.recent_performance = np.mean([1.0 - e for e in perf.prediction_errors[-20:]])
                perf.long_term_performance = np.mean([1.0 - e for e in perf.prediction_errors])
                
                # Update confidence calibration
                confidence_error = abs(component_pred.confidence - accuracy)
                perf.confidence_calibration = 1.0 - confidence_error
                
                perf.timestamp = datetime.now()
    
    def get_model_rankings(self) -> List[Tuple[str, float]]:
        """Get current model rankings by performance."""
        rankings = []
        
        for model_id, perf in self.model_performance.items():
            # Combined score from recent performance and weight
            score = (perf.recent_performance * 0.7 + 
                    self.model_weights.get(model_id, 0.0) * 0.3)
            rankings.append((model_id, score))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get comprehensive ensemble summary."""
        summary = {
            'ensemble_id': self.ensemble_id,
            'num_models': len(self.models),
            'combination_method': self.combination_method,
            'last_prediction': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            'model_weights': self.model_weights.copy(),
            'model_rankings': self.get_model_rankings()
        }
        
        # Recent performance
        if self.ensemble_performance:
            recent_perf = list(self.ensemble_performance)[-10:]
            summary['recent_performance'] = {
                'avg_accuracy': np.mean([p['accuracy'] for p in recent_perf]),
                'avg_confidence': np.mean([p['confidence'] for p in recent_perf]),
                'avg_disagreement': np.mean([p['disagreement'] for p in recent_perf]),
                'avg_stability': np.mean([p['stability'] for p in recent_perf])
            }
        
        # Model information
        summary['models'] = {
            model_id: model.get_model_info() 
            for model_id, model in self.models.items()
        }
        
        return summary