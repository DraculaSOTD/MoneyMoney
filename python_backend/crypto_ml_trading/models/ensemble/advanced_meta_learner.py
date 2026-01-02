"""
Advanced Meta-Learner for Ensemble Models.

Enhances the basic meta-learner with dynamic weighting, online learning,
and market regime awareness.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


class AdvancedMetaLearner:
    """
    Advanced meta-learning system for combining multiple models.
    
    Features:
    - Dynamic weight adjustment based on recent performance
    - Market regime detection and regime-specific weights
    - Online learning with adaptive learning rates
    - Uncertainty quantification
    - Model confidence scoring
    """
    
    def __init__(
        self,
        base_models: Dict[str, Any],
        meta_model_type: str = 'weighted_average',
        lookback_window: int = 100,
        update_frequency: int = 10,
        regime_detection: bool = True,
        confidence_threshold: float = 0.6,
        learning_rate: float = 0.01,
        decay_factor: float = 0.95
    ):
        """
        Initialize advanced meta-learner.
        
        Args:
            base_models: Dictionary of base models
            meta_model_type: Type of meta-model ('weighted_average', 'stacking', 'blending')
            lookback_window: Window for performance evaluation
            update_frequency: How often to update weights
            regime_detection: Whether to use market regime detection
            confidence_threshold: Minimum confidence for predictions
            learning_rate: Learning rate for online updates
            decay_factor: Decay factor for older observations
        """
        self.base_models = base_models
        self.meta_model_type = meta_model_type
        self.lookback_window = lookback_window
        self.update_frequency = update_frequency
        self.regime_detection = regime_detection
        self.confidence_threshold = confidence_threshold
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        
        # Initialize model weights
        n_models = len(base_models)
        self.model_weights = {
            'global': np.ones(n_models) / n_models,
            'regime_specific': {}
        }
        
        # Performance tracking
        self.performance_history = {
            name: deque(maxlen=lookback_window) 
            for name in base_models.keys()
        }
        self.prediction_history = deque(maxlen=lookback_window)
        self.true_values = deque(maxlen=lookback_window)
        
        # Regime detection
        self.current_regime = 'normal'
        self.regime_history = deque(maxlen=lookback_window)
        self.regime_models = {}
        
        # Confidence scoring
        self.model_confidence = {name: 1.0 for name in base_models.keys()}
        self.ensemble_confidence = 1.0
        
        # Meta-model components
        self.stacking_model = None
        self.blending_weights = None
        
        # Statistics
        self.update_count = 0
        self.total_predictions = 0
        
    def detect_market_regime(self, features: np.ndarray, 
                           market_data: Optional[pd.DataFrame] = None) -> str:
        """
        Detect current market regime.
        
        Args:
            features: Current feature set
            market_data: Optional market data for regime detection
            
        Returns:
            Detected regime name
        """
        if not self.regime_detection:
            return 'normal'
        
        # Calculate regime indicators
        regime_features = self._calculate_regime_features(features, market_data)
        
        # Simple regime classification
        volatility = regime_features.get('volatility', 0.02)
        trend_strength = regime_features.get('trend_strength', 0)
        volume_ratio = regime_features.get('volume_ratio', 1.0)
        
        if volatility > 0.04:
            regime = 'high_volatility'
        elif abs(trend_strength) > 0.03:
            regime = 'trending'
        elif volume_ratio > 2.0:
            regime = 'high_volume'
        elif volatility < 0.01:
            regime = 'low_volatility'
        else:
            regime = 'normal'
        
        self.regime_history.append(regime)
        return regime
    
    def _calculate_regime_features(self, features: np.ndarray,
                                 market_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Calculate features for regime detection."""
        regime_features = {}
        
        if market_data is not None and len(market_data) > 20:
            # Volatility (using returns)
            returns = market_data['close'].pct_change().dropna()
            regime_features['volatility'] = returns.std()
            
            # Trend strength (using linear regression slope)
            x = np.arange(len(returns))
            if len(x) > 1:
                slope = np.polyfit(x, returns.values, 1)[0]
                regime_features['trend_strength'] = slope
            
            # Volume ratio (current vs average)
            if 'volume' in market_data.columns:
                current_volume = market_data['volume'].iloc[-1]
                avg_volume = market_data['volume'].mean()
                regime_features['volume_ratio'] = current_volume / (avg_volume + 1e-8)
        
        # Features-based indicators
        if features.ndim > 1 and features.shape[1] > 10:
            # Assuming certain feature positions
            regime_features['feature_volatility'] = np.std(features)
            regime_features['feature_mean'] = np.mean(features)
        
        return regime_features
    
    def update_model_weights(self, predictions: Dict[str, np.ndarray],
                           true_values: np.ndarray) -> None:
        """
        Update model weights based on recent performance.
        
        Args:
            predictions: Dictionary of model predictions
            true_values: Actual values
        """
        self.update_count += 1
        
        # Calculate performance metrics for each model
        for model_name, pred in predictions.items():
            if model_name in self.base_models:
                # Calculate various performance metrics
                mse = np.mean((pred - true_values) ** 2)
                mae = np.mean(np.abs(pred - true_values))
                directional_accuracy = np.mean(
                    np.sign(pred[1:] - pred[:-1]) == np.sign(true_values[1:] - true_values[:-1])
                )
                
                # Composite performance score
                performance_score = (
                    0.3 * (1 / (1 + mse)) +  # MSE component
                    0.3 * (1 / (1 + mae)) +  # MAE component
                    0.4 * directional_accuracy  # Directional component
                )
                
                self.performance_history[model_name].append(performance_score)
        
        # Update weights if enough history
        if self.update_count % self.update_frequency == 0:
            self._recalculate_weights()
    
    def _recalculate_weights(self) -> None:
        """Recalculate model weights based on performance history."""
        # Global weights
        model_names = list(self.base_models.keys())
        n_models = len(model_names)
        
        # Calculate average performance for each model
        avg_performance = np.zeros(n_models)
        for i, name in enumerate(model_names):
            if len(self.performance_history[name]) > 0:
                # Exponentially weighted average
                performances = list(self.performance_history[name])
                weights = np.array([self.decay_factor ** (len(performances) - i - 1) 
                                   for i in range(len(performances))])
                weights /= weights.sum()
                avg_performance[i] = np.dot(performances, weights)
            else:
                avg_performance[i] = 0.5  # Default
        
        # Convert to weights using softmax
        exp_performance = np.exp(avg_performance * 2)  # Temperature parameter
        self.model_weights['global'] = exp_performance / exp_performance.sum()
        
        # Regime-specific weights
        if self.regime_detection and len(self.regime_history) > 0:
            current_regime = self.regime_history[-1]
            
            # Get regime-specific performances
            regime_mask = np.array(self.regime_history) == current_regime
            if regime_mask.sum() > 10:  # Enough data for regime
                regime_performance = np.zeros(n_models)
                for i, name in enumerate(model_names):
                    perfs = np.array(list(self.performance_history[name]))
                    if len(perfs) >= len(regime_mask):
                        regime_perfs = perfs[-len(regime_mask):][regime_mask]
                        if len(regime_perfs) > 0:
                            regime_performance[i] = regime_perfs.mean()
                
                # Calculate regime-specific weights
                exp_regime_perf = np.exp(regime_performance * 2)
                self.model_weights['regime_specific'][current_regime] = (
                    exp_regime_perf / exp_regime_perf.sum()
                )
    
    def calculate_model_confidence(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate confidence scores for each model.
        
        Args:
            predictions: Model predictions
            
        Returns:
            Confidence scores for each model
        """
        confidence_scores = {}
        
        for model_name, pred in predictions.items():
            if model_name not in self.base_models:
                continue
            
            # Base confidence from performance history
            if len(self.performance_history[model_name]) > 0:
                recent_performance = np.mean(list(self.performance_history[model_name])[-20:])
                base_confidence = recent_performance
            else:
                base_confidence = 0.5
            
            # Prediction consistency
            if hasattr(pred, '__len__') and len(pred) > 1:
                pred_std = np.std(pred)
                consistency_score = 1 / (1 + pred_std)
            else:
                consistency_score = 1.0
            
            # Agreement with ensemble
            if len(predictions) > 1:
                all_preds = np.array(list(predictions.values()))
                ensemble_mean = all_preds.mean(axis=0)
                agreement = 1 - np.mean(np.abs(pred - ensemble_mean)) / (np.std(all_preds) + 1e-8)
                agreement_score = np.clip(agreement, 0, 1)
            else:
                agreement_score = 1.0
            
            # Combined confidence
            confidence = (
                0.5 * base_confidence +
                0.2 * consistency_score +
                0.3 * agreement_score
            )
            
            confidence_scores[model_name] = np.clip(confidence, 0, 1)
            self.model_confidence[model_name] = confidence_scores[model_name]
        
        return confidence_scores
    
    def predict(self, features: np.ndarray, 
               market_data: Optional[pd.DataFrame] = None,
               return_all: bool = False) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Make ensemble prediction.
        
        Args:
            features: Input features
            market_data: Optional market data
            return_all: Whether to return detailed results
            
        Returns:
            Predictions or detailed results dictionary
        """
        self.total_predictions += 1
        
        # Detect market regime
        current_regime = self.detect_market_regime(features, market_data)
        self.current_regime = current_regime
        
        # Get base model predictions
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(features)
                    base_predictions[name] = pred
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")
        
        if not base_predictions:
            raise ValueError("No base model predictions available")
        
        # Calculate model confidence
        confidence_scores = self.calculate_model_confidence(base_predictions)
        
        # Get appropriate weights
        if (self.regime_detection and 
            current_regime in self.model_weights['regime_specific']):
            weights = self.model_weights['regime_specific'][current_regime]
        else:
            weights = self.model_weights['global']
        
        # Apply confidence adjustment to weights
        model_names = list(self.base_models.keys())
        adjusted_weights = np.zeros(len(model_names))
        for i, name in enumerate(model_names):
            if name in confidence_scores:
                adjusted_weights[i] = weights[i] * confidence_scores[name]
        
        # Normalize weights
        if adjusted_weights.sum() > 0:
            adjusted_weights /= adjusted_weights.sum()
        else:
            adjusted_weights = weights
        
        # Combine predictions based on meta-model type
        if self.meta_model_type == 'weighted_average':
            ensemble_pred = self._weighted_average_prediction(
                base_predictions, adjusted_weights, model_names
            )
        elif self.meta_model_type == 'stacking':
            ensemble_pred = self._stacking_prediction(
                base_predictions, features
            )
        elif self.meta_model_type == 'blending':
            ensemble_pred = self._blending_prediction(
                base_predictions, adjusted_weights, model_names
            )
        else:
            ensemble_pred = self._weighted_average_prediction(
                base_predictions, adjusted_weights, model_names
            )
        
        # Calculate ensemble confidence
        self.ensemble_confidence = np.mean(list(confidence_scores.values()))
        
        # Store prediction for online learning
        self.prediction_history.append({
            'prediction': ensemble_pred,
            'base_predictions': base_predictions.copy(),
            'weights': adjusted_weights.copy(),
            'regime': current_regime,
            'confidence': self.ensemble_confidence,
            'timestamp': datetime.now()
        })
        
        if return_all:
            return {
                'prediction': ensemble_pred,
                'base_predictions': base_predictions,
                'weights': dict(zip(model_names, adjusted_weights)),
                'confidence_scores': confidence_scores,
                'ensemble_confidence': self.ensemble_confidence,
                'regime': current_regime,
                'model_weights': self.model_weights
            }
        
        return ensemble_pred
    
    def _weighted_average_prediction(self, predictions: Dict[str, np.ndarray],
                                   weights: np.ndarray,
                                   model_names: List[str]) -> np.ndarray:
        """Calculate weighted average prediction."""
        weighted_sum = None
        
        for i, name in enumerate(model_names):
            if name in predictions:
                pred = predictions[name]
                if weighted_sum is None:
                    weighted_sum = weights[i] * pred
                else:
                    weighted_sum += weights[i] * pred
        
        return weighted_sum
    
    def _stacking_prediction(self, base_predictions: Dict[str, np.ndarray],
                           original_features: np.ndarray) -> np.ndarray:
        """Stacking ensemble prediction."""
        # Concatenate base predictions
        pred_array = np.column_stack(list(base_predictions.values()))
        
        # Add original features (optional)
        if original_features.ndim == 1:
            stacking_features = np.hstack([pred_array, original_features.reshape(1, -1)])
        else:
            stacking_features = pred_array
        
        # Use simple linear combination if no stacking model trained
        if self.stacking_model is None:
            # Initialize with equal weights
            n_models = pred_array.shape[1]
            weights = np.ones(n_models) / n_models
            return np.dot(pred_array, weights)
        
        # Use trained stacking model
        return self.stacking_model.predict(stacking_features)
    
    def _blending_prediction(self, predictions: Dict[str, np.ndarray],
                           weights: np.ndarray,
                           model_names: List[str]) -> np.ndarray:
        """Blending ensemble with non-linear combination."""
        # Start with weighted average
        base_ensemble = self._weighted_average_prediction(predictions, weights, model_names)
        
        # Add non-linear adjustments based on agreement
        pred_values = np.array(list(predictions.values()))
        pred_std = np.std(pred_values, axis=0)
        
        # Reduce prediction when models disagree
        confidence_factor = 1 / (1 + pred_std)
        
        return base_ensemble * confidence_factor
    
    def online_update(self, true_value: np.ndarray) -> None:
        """
        Perform online update with most recent true value.
        
        Args:
            true_value: Actual observed value
        """
        if len(self.prediction_history) == 0:
            return
        
        # Get last prediction
        last_pred_info = self.prediction_history[-1]
        last_pred = last_pred_info['prediction']
        base_preds = last_pred_info['base_predictions']
        
        # Calculate errors
        ensemble_error = np.mean((last_pred - true_value) ** 2)
        
        # Update model weights using gradient descent
        model_names = list(self.base_models.keys())
        for i, name in enumerate(model_names):
            if name in base_preds:
                model_error = np.mean((base_preds[name] - true_value) ** 2)
                
                # Gradient of weight with respect to error
                gradient = 2 * (base_preds[name] - true_value) * (
                    base_preds[name] - last_pred
                )
                
                # Update weight
                self.model_weights['global'][i] -= (
                    self.learning_rate * np.mean(gradient)
                )
        
        # Normalize weights
        self.model_weights['global'] = np.maximum(
            self.model_weights['global'], 0.01
        )
        self.model_weights['global'] /= self.model_weights['global'].sum()
        
        # Store true value
        self.true_values.append(true_value)
        
        # Decay learning rate
        self.learning_rate *= 0.999
    
    def get_model_importance(self) -> Dict[str, float]:
        """
        Get current model importance scores.
        
        Returns:
            Dictionary of model importance scores
        """
        model_names = list(self.base_models.keys())
        importance = {}
        
        for i, name in enumerate(model_names):
            # Combine weight and confidence
            weight = self.model_weights['global'][i]
            confidence = self.model_confidence.get(name, 0.5)
            
            importance[name] = weight * confidence
        
        # Normalize
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v/total_importance for k, v in importance.items()}
        
        return importance
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        model_names = list(self.base_models.keys())
        
        summary = {
            'total_predictions': self.total_predictions,
            'current_regime': self.current_regime,
            'ensemble_confidence': self.ensemble_confidence,
            'model_weights': dict(zip(model_names, self.model_weights['global'])),
            'model_confidence': self.model_confidence.copy(),
            'model_importance': self.get_model_importance(),
            'regime_distribution': {}
        }
        
        # Calculate regime distribution
        if len(self.regime_history) > 0:
            regimes, counts = np.unique(list(self.regime_history), return_counts=True)
            summary['regime_distribution'] = dict(zip(regimes, counts / len(self.regime_history)))
        
        # Recent performance
        if len(self.performance_history[model_names[0]]) > 0:
            recent_perfs = {}
            for name in model_names:
                if len(self.performance_history[name]) > 0:
                    recent_perfs[name] = np.mean(list(self.performance_history[name])[-20:])
            summary['recent_performance'] = recent_perfs
        
        return summary