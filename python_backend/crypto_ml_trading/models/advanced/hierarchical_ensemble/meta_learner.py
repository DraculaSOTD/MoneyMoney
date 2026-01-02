"""
Meta-Learner for Hierarchical Ensemble Architecture.

Implements advanced meta-learning capabilities for optimal ensemble combination.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations
from models.advanced.hierarchical_ensemble.base_ensemble import ModelPrediction, EnsemblePrediction


@dataclass
class MetaFeatures:
    """Meta-features for meta-learning."""
    timestamp: datetime
    market_regime: str
    volatility: float
    trend_strength: float
    model_disagreement: float
    prediction_horizon: int
    feature_complexity: float
    data_quality: float
    ensemble_history: Dict[str, float] = field(default_factory=dict)


@dataclass
class MetaLearningResult:
    """Result from meta-learning optimization."""
    timestamp: datetime
    optimal_weights: Dict[str, float]
    predicted_performance: float
    confidence: float
    meta_features_used: List[str]
    optimization_method: str
    convergence_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationStrategy:
    """Strategy for ensemble adaptation."""
    strategy_name: str
    trigger_conditions: Dict[str, float]
    adaptation_actions: List[str]
    rollback_conditions: Dict[str, float]
    success_metrics: List[str]


class MetaLearner:
    """
    Advanced meta-learner for ensemble optimization.
    
    Features:
    - Online meta-learning for weight optimization
    - Market regime-aware adaptation
    - Multi-objective optimization
    - Catastrophic forgetting prevention
    - Real-time performance prediction
    - Ensemble configuration optimization
    """
    
    def __init__(self,
                 learning_rate: float = 0.01,
                 meta_window: int = 200,
                 adaptation_threshold: float = 0.05,
                 forgetting_factor: float = 0.95):
        """
        Initialize meta-learner.
        
        Args:
            learning_rate: Learning rate for weight updates
            meta_window: Window size for meta-learning
            adaptation_threshold: Threshold for triggering adaptation
            forgetting_factor: Factor for preventing catastrophic forgetting
        """
        self.learning_rate = learning_rate
        self.meta_window = meta_window
        self.adaptation_threshold = adaptation_threshold
        self.forgetting_factor = forgetting_factor
        
        # Meta-learning components
        self.meta_features_history: deque = deque(maxlen=1000)
        self.meta_targets: deque = deque(maxlen=1000)
        self.meta_learning_results: deque = deque(maxlen=500)
        
        # Weight optimization
        self.weight_gradients: Dict[str, np.ndarray] = {}
        self.weight_momentum: Dict[str, np.ndarray] = {}
        self.adaptive_learning_rates: Dict[str, float] = {}
        
        # Market regime tracking
        self.regime_weights: Dict[str, Dict[str, float]] = {}
        self.regime_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Adaptation strategies
        self.adaptation_strategies: List[AdaptationStrategy] = []
        self.active_adaptations: List[str] = []
        
        # Performance prediction
        self.performance_predictor_weights: np.ndarray = np.array([])
        self.meta_model_trained = False
        
        # Initialize adaptation strategies
        self._initialize_adaptation_strategies()
    
    def _initialize_adaptation_strategies(self) -> None:
        """Initialize predefined adaptation strategies."""
        strategies = [
            AdaptationStrategy(
                strategy_name="high_volatility_adaptation",
                trigger_conditions={"volatility": 0.05, "model_disagreement": 0.3},
                adaptation_actions=["increase_conservative_models", "reduce_momentum_models"],
                rollback_conditions={"volatility": 0.02, "performance_improvement": -0.1},
                success_metrics=["sharpe_ratio", "max_drawdown"]
            ),
            AdaptationStrategy(
                strategy_name="regime_change_adaptation", 
                trigger_conditions={"regime_confidence": 0.8, "regime_change": 1.0},
                adaptation_actions=["rebalance_weights", "activate_regime_models"],
                rollback_conditions={"regime_stability": 0.9},
                success_metrics=["prediction_accuracy", "regime_alignment"]
            ),
            AdaptationStrategy(
                strategy_name="poor_performance_adaptation",
                trigger_conditions={"ensemble_performance": 0.3, "performance_decline": 0.2},
                adaptation_actions=["reduce_worst_models", "increase_best_models", "emergency_rebalance"],
                rollback_conditions={"performance_improvement": 0.1},
                success_metrics=["accuracy_recovery", "confidence_stability"]
            )
        ]
        
        self.adaptation_strategies.extend(strategies)
    
    def extract_meta_features(self,
                            predictions: List[ModelPrediction],
                            market_data: Dict[str, Any],
                            ensemble_history: List[EnsemblePrediction]) -> MetaFeatures:
        """
        Extract meta-features for meta-learning.
        
        Args:
            predictions: Current model predictions
            market_data: Current market state data
            ensemble_history: Recent ensemble predictions
            
        Returns:
            Extracted meta-features
        """
        timestamp = datetime.now()
        
        # Market regime features
        market_regime = market_data.get('regime', 'unknown')
        volatility = market_data.get('volatility', 0.02)
        trend_strength = market_data.get('trend_strength', 0.5)
        
        # Model disagreement
        if len(predictions) > 1:
            pred_values = [p.prediction if not isinstance(p.prediction, np.ndarray) 
                          else np.mean(p.prediction) for p in predictions]
            model_disagreement = np.std(pred_values)
        else:
            model_disagreement = 0.0
        
        # Prediction horizon
        prediction_horizon = predictions[0].horizon if predictions else 15
        
        # Feature complexity (based on number of features used)
        if predictions:
            avg_features = np.mean([len(p.features_used) for p in predictions])
            feature_complexity = avg_features / 100.0  # Normalize
        else:
            feature_complexity = 0.5
        
        # Data quality (based on confidence scores)
        if predictions:
            data_quality = np.mean([p.confidence for p in predictions])
        else:
            data_quality = 0.5
        
        # Ensemble history features
        ensemble_hist_features = {}
        if ensemble_history:
            recent_history = ensemble_history[-10:]
            
            ensemble_hist_features = {
                'avg_ensemble_confidence': np.mean([e.confidence for e in recent_history]),
                'avg_disagreement': np.mean([e.disagreement for e in recent_history]),
                'avg_stability': np.mean([e.stability for e in recent_history]),
                'weight_entropy': self._calculate_weight_entropy(recent_history)
            }
        
        return MetaFeatures(
            timestamp=timestamp,
            market_regime=market_regime,
            volatility=volatility,
            trend_strength=trend_strength,
            model_disagreement=model_disagreement,
            prediction_horizon=prediction_horizon,
            feature_complexity=feature_complexity,
            data_quality=data_quality,
            ensemble_history=ensemble_hist_features
        )
    
    def _calculate_weight_entropy(self, ensemble_history: List[EnsemblePrediction]) -> float:
        """Calculate entropy of weight distributions."""
        if not ensemble_history:
            return 0.0
        
        # Get all unique model IDs
        all_models = set()
        for ens in ensemble_history:
            all_models.update(ens.weights.keys())
        
        # Calculate average weights
        avg_weights = {}
        for model_id in all_models:
            weights = [ens.weights.get(model_id, 0.0) for ens in ensemble_history]
            avg_weights[model_id] = np.mean(weights)
        
        # Calculate entropy
        weight_values = list(avg_weights.values())
        weight_values = np.array(weight_values)
        weight_values = weight_values / np.sum(weight_values) if np.sum(weight_values) > 0 else weight_values
        
        # Shannon entropy
        entropy = -np.sum([w * np.log(w + 1e-10) for w in weight_values if w > 0])
        
        return entropy
    
    def optimize_weights(self,
                        meta_features: MetaFeatures,
                        current_weights: Dict[str, float],
                        model_ids: List[str],
                        target_performance: Optional[float] = None) -> MetaLearningResult:
        """
        Optimize ensemble weights using meta-learning.
        
        Args:
            meta_features: Current meta-features
            current_weights: Current model weights
            model_ids: List of model identifiers
            target_performance: Target performance if available
            
        Returns:
            Meta-learning optimization result
        """
        timestamp = datetime.now()
        
        # Store meta-features for training
        self.meta_features_history.append(meta_features)
        
        if target_performance is not None:
            self.meta_targets.append(target_performance)
        
        # Choose optimization method based on available data
        if len(self.meta_features_history) > 50 and self.meta_targets:
            optimization_method = "gradient_based"
            optimal_weights, predicted_perf, confidence = self._gradient_based_optimization(
                meta_features, current_weights, model_ids
            )
        elif len(self.meta_features_history) > 20:
            optimization_method = "regime_based"
            optimal_weights, predicted_perf, confidence = self._regime_based_optimization(
                meta_features, current_weights, model_ids
            )
        else:
            optimization_method = "heuristic"
            optimal_weights, predicted_perf, confidence = self._heuristic_optimization(
                meta_features, current_weights, model_ids
            )
        
        # Apply forgetting factor to prevent catastrophic forgetting
        optimal_weights = self._apply_forgetting_factor(optimal_weights, current_weights)
        
        result = MetaLearningResult(
            timestamp=timestamp,
            optimal_weights=optimal_weights,
            predicted_performance=predicted_perf,
            confidence=confidence,
            meta_features_used=list(meta_features.__dict__.keys()),
            optimization_method=optimization_method,
            convergence_info={"iterations": 1, "converged": True}
        )
        
        self.meta_learning_results.append(result)
        
        return result
    
    def _gradient_based_optimization(self,
                                   meta_features: MetaFeatures,
                                   current_weights: Dict[str, float],
                                   model_ids: List[str]) -> Tuple[Dict[str, float], float, float]:
        """Gradient-based weight optimization."""
        # Prepare feature vector
        feature_vector = self._meta_features_to_vector(meta_features)
        
        # Train/update meta-model if needed
        if not self.meta_model_trained:
            self._train_meta_model()
        
        # Current weight vector
        weight_vector = np.array([current_weights.get(mid, 0.0) for mid in model_ids])
        
        # Predict performance gradient
        if len(self.performance_predictor_weights) == len(feature_vector):
            performance_gradient = self.performance_predictor_weights * feature_vector
            predicted_performance = np.dot(performance_gradient, weight_vector)
        else:
            predicted_performance = 0.5
            performance_gradient = np.zeros(len(weight_vector))
        
        # Calculate weight gradients
        weight_gradients = np.zeros(len(model_ids))
        
        for i, model_id in enumerate(model_ids):
            # Get or initialize gradient tracking
            if model_id not in self.weight_gradients:
                self.weight_gradients[model_id] = np.zeros(1)
                self.weight_momentum[model_id] = np.zeros(1)
                self.adaptive_learning_rates[model_id] = self.learning_rate
            
            # Simple gradient approximation
            # In practice, this would use more sophisticated methods
            if i < len(performance_gradient):
                gradient = performance_gradient[i]
            else:
                gradient = 0.0
            
            # Momentum update
            momentum = 0.9
            self.weight_momentum[model_id] = (momentum * self.weight_momentum[model_id] + 
                                            (1 - momentum) * gradient)
            
            # Adaptive learning rate
            lr = self.adaptive_learning_rates[model_id]
            weight_gradients[i] = lr * self.weight_momentum[model_id]
        
        # Update weights
        new_weights = weight_vector + weight_gradients
        
        # Apply constraints (non-negative, sum to 1)
        new_weights = np.maximum(new_weights, 0.01)  # Minimum weight
        new_weights = new_weights / np.sum(new_weights)  # Normalize
        
        # Convert back to dictionary
        optimal_weights = {model_id: float(w) for model_id, w in zip(model_ids, new_weights)}
        
        # Calculate confidence based on gradient magnitude
        confidence = min(1.0, 1.0 / (1.0 + np.linalg.norm(weight_gradients)))
        
        return optimal_weights, predicted_performance, confidence
    
    def _regime_based_optimization(self,
                                 meta_features: MetaFeatures,
                                 current_weights: Dict[str, float],
                                 model_ids: List[str]) -> Tuple[Dict[str, float], float, float]:
        """Regime-based weight optimization."""
        regime = meta_features.market_regime
        
        # Get or initialize regime-specific weights
        if regime not in self.regime_weights:
            self.regime_weights[regime] = current_weights.copy()
        
        # Update regime weights based on recent performance
        regime_weights = self.regime_weights[regime].copy()
        
        # Adjust weights based on meta-features
        volatility_factor = meta_features.volatility
        disagreement_factor = meta_features.model_disagreement
        
        for model_id in model_ids:
            current_weight = regime_weights.get(model_id, 1.0 / len(model_ids))
            
            # Adjust based on volatility (favor stable models in high volatility)
            if volatility_factor > 0.03:  # High volatility threshold
                # This would be customized based on model characteristics
                if "conservative" in model_id.lower() or "stable" in model_id.lower():
                    current_weight *= 1.2
                elif "momentum" in model_id.lower() or "trend" in model_id.lower():
                    current_weight *= 0.8
            
            # Adjust based on model disagreement
            if disagreement_factor > 0.2:  # High disagreement
                # Favor ensemble methods over individual models
                if "ensemble" in model_id.lower():
                    current_weight *= 1.1
            
            regime_weights[model_id] = current_weight
        
        # Normalize weights
        total_weight = sum(regime_weights.values())
        if total_weight > 0:
            regime_weights = {k: v/total_weight for k, v in regime_weights.items()}
        
        # Update stored regime weights
        self.regime_weights[regime] = regime_weights.copy()
        
        # Predicted performance based on regime history
        if regime in self.regime_performance and self.regime_performance[regime]:
            predicted_performance = np.mean(self.regime_performance[regime][-10:])
        else:
            predicted_performance = 0.5
        
        # Confidence based on regime stability
        confidence = 1.0 - meta_features.model_disagreement
        
        return regime_weights, predicted_performance, confidence
    
    def _heuristic_optimization(self,
                              meta_features: MetaFeatures,
                              current_weights: Dict[str, float],
                              model_ids: List[str]) -> Tuple[Dict[str, float], float, float]:
        """Heuristic-based weight optimization for cold start."""
        optimal_weights = current_weights.copy()
        
        # Simple heuristics based on meta-features
        data_quality = meta_features.data_quality
        disagreement = meta_features.model_disagreement
        
        # If high disagreement, favor ensemble methods
        if disagreement > 0.3:
            for model_id in model_ids:
                if "ensemble" in model_id.lower() or "average" in model_id.lower():
                    optimal_weights[model_id] = optimal_weights.get(model_id, 0.0) * 1.3
        
        # If low data quality, be more conservative
        if data_quality < 0.4:
            for model_id in model_ids:
                if "conservative" in model_id.lower() or "simple" in model_id.lower():
                    optimal_weights[model_id] = optimal_weights.get(model_id, 0.0) * 1.2
        
        # Normalize weights
        total_weight = sum(optimal_weights.values())
        if total_weight > 0:
            optimal_weights = {k: v/total_weight for k, v in optimal_weights.items()}
        else:
            # Equal weights fallback
            optimal_weights = {mid: 1.0/len(model_ids) for mid in model_ids}
        
        # Predicted performance (conservative estimate)
        predicted_performance = data_quality * 0.8
        
        # Confidence (lower for heuristic)
        confidence = 0.6
        
        return optimal_weights, predicted_performance, confidence
    
    def _apply_forgetting_factor(self,
                               new_weights: Dict[str, float],
                               old_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply forgetting factor to prevent catastrophic forgetting."""
        adjusted_weights = {}
        
        for model_id in new_weights:
            old_weight = old_weights.get(model_id, 0.0)
            new_weight = new_weights[model_id]
            
            # Exponential moving average
            adjusted_weight = (self.forgetting_factor * old_weight + 
                             (1 - self.forgetting_factor) * new_weight)
            
            adjusted_weights[model_id] = adjusted_weight
        
        # Renormalize
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def _train_meta_model(self) -> None:
        """Train the meta-model for performance prediction."""
        if len(self.meta_features_history) < 20 or len(self.meta_targets) < 20:
            return
        
        # Prepare training data
        X = []
        y = []
        
        min_len = min(len(self.meta_features_history), len(self.meta_targets))
        
        for i in range(min_len):
            feature_vector = self._meta_features_to_vector(self.meta_features_history[i])
            target = self.meta_targets[i]
            
            X.append(feature_vector)
            y.append(target)
        
        X = np.array(X)
        y = np.array(y)
        
        if X.shape[0] > 0 and X.shape[1] > 0:
            # Simple linear regression for meta-model
            try:
                # Add bias term
                X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
                
                # Solve normal equations
                self.performance_predictor_weights = np.linalg.lstsq(
                    X_with_bias, y, rcond=None
                )[0]
                
                self.meta_model_trained = True
                
            except Exception as e:
                print(f"Meta-model training failed: {e}")
                # Fallback to random weights
                self.performance_predictor_weights = np.random.randn(X.shape[1] + 1) * 0.1
    
    def _meta_features_to_vector(self, meta_features: MetaFeatures) -> np.ndarray:
        """Convert meta-features to numerical vector."""
        # One-hot encode regime
        regimes = ['bull', 'bear', 'sideways', 'volatile', 'unknown']
        regime_vector = [1.0 if meta_features.market_regime == r else 0.0 for r in regimes]
        
        # Numerical features
        numerical_features = [
            meta_features.volatility,
            meta_features.trend_strength,
            meta_features.model_disagreement,
            meta_features.prediction_horizon / 60.0,  # Normalize
            meta_features.feature_complexity,
            meta_features.data_quality
        ]
        
        # Ensemble history features
        history_features = [
            meta_features.ensemble_history.get('avg_ensemble_confidence', 0.5),
            meta_features.ensemble_history.get('avg_disagreement', 0.5),
            meta_features.ensemble_history.get('avg_stability', 0.5),
            meta_features.ensemble_history.get('weight_entropy', 0.0)
        ]
        
        # Combine all features
        feature_vector = regime_vector + numerical_features + history_features
        
        return np.array(feature_vector)
    
    def check_adaptation_triggers(self,
                                meta_features: MetaFeatures,
                                ensemble_performance: Dict[str, float]) -> List[str]:
        """Check if adaptation strategies should be triggered."""
        triggered_strategies = []
        
        for strategy in self.adaptation_strategies:
            should_trigger = True
            
            # Check all trigger conditions
            for condition, threshold in strategy.trigger_conditions.items():
                current_value = self._get_condition_value(condition, meta_features, ensemble_performance)
                
                if condition.endswith('_change') or condition == 'regime_change':
                    # For change conditions, trigger if value exceeds threshold
                    if current_value < threshold:
                        should_trigger = False
                        break
                elif 'performance' in condition:
                    # For performance conditions, trigger if below threshold
                    if current_value > threshold:
                        should_trigger = False
                        break
                else:
                    # For other conditions, trigger if above threshold
                    if current_value < threshold:
                        should_trigger = False
                        break
            
            if should_trigger and strategy.strategy_name not in self.active_adaptations:
                triggered_strategies.append(strategy.strategy_name)
                self.active_adaptations.append(strategy.strategy_name)
        
        return triggered_strategies
    
    def _get_condition_value(self,
                           condition: str,
                           meta_features: MetaFeatures,
                           ensemble_performance: Dict[str, float]) -> float:
        """Get current value for adaptation condition."""
        if condition == 'volatility':
            return meta_features.volatility
        elif condition == 'model_disagreement':
            return meta_features.model_disagreement
        elif condition == 'regime_confidence':
            # Would be calculated based on regime stability
            return 1.0 - meta_features.model_disagreement  # Simplified
        elif condition == 'regime_change':
            # Would be calculated based on regime transitions
            return 0.0  # Simplified - would track actual changes
        elif condition == 'ensemble_performance':
            return ensemble_performance.get('accuracy', 0.5)
        elif condition == 'performance_decline':
            # Would calculate performance change over time
            return 0.0  # Simplified
        else:
            return 0.0
    
    def apply_adaptation_strategy(self,
                                strategy_name: str,
                                current_weights: Dict[str, float],
                                model_ids: List[str]) -> Dict[str, float]:
        """Apply adaptation strategy to modify weights."""
        strategy = next((s for s in self.adaptation_strategies if s.strategy_name == strategy_name), None)
        
        if not strategy:
            return current_weights
        
        adapted_weights = current_weights.copy()
        
        for action in strategy.adaptation_actions:
            adapted_weights = self._apply_adaptation_action(action, adapted_weights, model_ids)
        
        return adapted_weights
    
    def _apply_adaptation_action(self,
                               action: str,
                               weights: Dict[str, float],
                               model_ids: List[str]) -> Dict[str, float]:
        """Apply specific adaptation action."""
        if action == "increase_conservative_models":
            for model_id in model_ids:
                if "conservative" in model_id.lower() or "stable" in model_id.lower():
                    weights[model_id] = weights.get(model_id, 0.0) * 1.5
        
        elif action == "reduce_momentum_models":
            for model_id in model_ids:
                if "momentum" in model_id.lower() or "trend" in model_id.lower():
                    weights[model_id] = weights.get(model_id, 0.0) * 0.7
        
        elif action == "rebalance_weights":
            # Move towards equal weights
            equal_weight = 1.0 / len(model_ids)
            for model_id in model_ids:
                current = weights.get(model_id, equal_weight)
                weights[model_id] = 0.7 * current + 0.3 * equal_weight
        
        elif action == "reduce_worst_models":
            # Reduce weights of bottom 25% performers
            # This would use actual performance metrics
            pass
        
        elif action == "emergency_rebalance":
            # Set to equal weights
            equal_weight = 1.0 / len(model_ids)
            for model_id in model_ids:
                weights[model_id] = equal_weight
        
        # Renormalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def update_performance_feedback(self,
                                  meta_learning_result: MetaLearningResult,
                                  actual_performance: float) -> None:
        """Update meta-learner with performance feedback."""
        # Store as target for future training
        self.meta_targets.append(actual_performance)
        
        # Update adaptive learning rates
        prediction_error = abs(meta_learning_result.predicted_performance - actual_performance)
        
        for model_id in meta_learning_result.optimal_weights:
            if model_id in self.adaptive_learning_rates:
                # Decrease learning rate if error is high, increase if low
                if prediction_error > 0.1:
                    self.adaptive_learning_rates[model_id] *= 0.95
                else:
                    self.adaptive_learning_rates[model_id] *= 1.05
                
                # Bound learning rates
                self.adaptive_learning_rates[model_id] = np.clip(
                    self.adaptive_learning_rates[model_id], 0.001, 0.1
                )
        
        # Retrain meta-model periodically
        if len(self.meta_targets) % 50 == 0:
            self.meta_model_trained = False  # Force retraining
    
    def get_meta_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive meta-learning summary."""
        summary = {
            'meta_model_trained': self.meta_model_trained,
            'training_samples': len(self.meta_targets),
            'active_adaptations': self.active_adaptations.copy(),
            'regime_weights': dict(self.regime_weights),
            'adaptive_learning_rates': self.adaptive_learning_rates.copy()
        }
        
        # Recent meta-learning performance
        if self.meta_learning_results:
            recent_results = list(self.meta_learning_results)[-10:]
            summary['recent_performance'] = {
                'avg_predicted_performance': np.mean([r.predicted_performance for r in recent_results]),
                'avg_confidence': np.mean([r.confidence for r in recent_results]),
                'optimization_methods': [r.optimization_method for r in recent_results]
            }
        
        return summary