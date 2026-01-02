"""
Ensemble Model Adaptation System.

Provides adaptive ensemble learning with dynamic weight adjustment
and model selection based on real-time performance.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from scipy.optimize import minimize
from scipy.special import softmax

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePerformance:
    """Performance metrics for ensemble members."""
    model_name: str
    timestamp: datetime
    accuracy: float
    recent_accuracy: float
    volatility: float
    weight: float
    active: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptiveEnsemble:
    """
    Adaptive ensemble that dynamically adjusts model weights.
    
    Features:
    - Dynamic weight optimization
    - Model selection based on performance
    - Diversity preservation
    - Online weight updates
    """
    
    def __init__(self,
                 models: Dict[str, Any],
                 window_size: int = 1000,
                 min_weight: float = 0.01,
                 diversity_weight: float = 0.1,
                 adaptation_rate: float = 0.1):
        """
        Initialize adaptive ensemble.
        
        Args:
            models: Dictionary of base models
            window_size: Performance evaluation window
            min_weight: Minimum weight for active models
            diversity_weight: Weight for diversity term
            adaptation_rate: Rate of weight updates
        """
        self.models = models
        self.window_size = window_size
        self.min_weight = min_weight
        self.diversity_weight = diversity_weight
        self.adaptation_rate = adaptation_rate
        
        # Initialize weights uniformly
        n_models = len(models)
        self.weights = {name: 1.0 / n_models for name in models}
        
        # Performance tracking
        self.predictions_history = defaultdict(lambda: deque(maxlen=window_size))
        self.targets_history = deque(maxlen=window_size)
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        
        # Model state
        self.active_models = set(models.keys())
        self.model_correlations = np.eye(n_models)
        
        logger.info(f"Adaptive ensemble initialized with {n_models} models")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make weighted ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Ensemble predictions
        """
        predictions = {}
        
        # Get predictions from active models
        for name in self.active_models:
            if name in self.models and self.weights[name] > self.min_weight:
                try:
                    pred = self.models[name].predict(X)
                    predictions[name] = pred
                except Exception as e:
                    logger.error(f"Prediction error for {name}: {e}")
        
        if not predictions:
            # Fallback to uniform ensemble if no predictions
            logger.warning("No valid predictions, using all models")
            for name, model in self.models.items():
                try:
                    predictions[name] = model.predict(X)
                except:
                    pass
        
        # Weighted average
        if predictions:
            # Stack predictions
            pred_array = np.array(list(predictions.values()))
            weights = np.array([self.weights[name] for name in predictions.keys()])
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Weighted prediction
            if len(pred_array.shape) == 3:  # Multi-output
                ensemble_pred = np.tensordot(weights, pred_array, axes=([0], [0]))
            else:
                ensemble_pred = np.average(pred_array, weights=weights, axis=0)
            
            return ensemble_pred
        else:
            logger.error("No valid predictions from ensemble")
            return np.zeros_like(X[:, 0])
    
    def update_weights(self, X: np.ndarray, y: np.ndarray,
                      immediate: bool = False) -> Dict[str, float]:
        """
        Update ensemble weights based on performance.
        
        Args:
            X: Input features
            y: True targets
            immediate: Whether to update immediately
            
        Returns:
            Updated weights
        """
        # Store predictions and targets
        for name in self.active_models:
            try:
                pred = self.models[name].predict(X)
                self.predictions_history[name].extend(pred.flatten())
            except Exception as e:
                logger.error(f"Prediction storage error for {name}: {e}")
        
        self.targets_history.extend(y.flatten())
        
        # Check if we have enough history
        if len(self.targets_history) < 100:
            return self.weights
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        # Update correlations
        self._update_correlations()
        
        # Optimize weights
        if immediate or len(self.targets_history) % 100 == 0:
            self.weights = self._optimize_weights(performance_metrics)
            
            # Update active models
            self._update_active_models(performance_metrics)
        
        return self.weights
    
    def _calculate_performance_metrics(self) -> Dict[str, EnsemblePerformance]:
        """Calculate performance metrics for each model."""
        metrics = {}
        targets = np.array(self.targets_history)
        
        for name in self.models.keys():
            if name not in self.predictions_history or len(self.predictions_history[name]) < 10:
                continue
            
            predictions = np.array(self.predictions_history[name])
            
            # Align predictions and targets
            min_len = min(len(predictions), len(targets))
            predictions = predictions[-min_len:]
            targets_aligned = targets[-min_len:]
            
            # Calculate accuracy (or other metrics)
            if len(np.unique(targets_aligned)) < 10:  # Classification
                accuracy = np.mean(predictions.round() == targets_aligned)
            else:  # Regression
                mse = np.mean((predictions - targets_aligned) ** 2)
                accuracy = 1.0 / (1.0 + mse)  # Convert to accuracy-like metric
            
            # Recent accuracy (last 20% of data)
            recent_size = max(10, min_len // 5)
            recent_accuracy = np.mean(predictions[-recent_size:].round() == targets_aligned[-recent_size:]) \
                if len(np.unique(targets_aligned)) < 10 else \
                1.0 / (1.0 + np.mean((predictions[-recent_size:] - targets_aligned[-recent_size:]) ** 2))
            
            # Volatility (standard deviation of rolling accuracy)
            window = min(50, min_len // 4)
            if window > 10:
                rolling_accuracies = []
                for i in range(window, min_len, 10):
                    window_acc = np.mean(predictions[i-window:i].round() == targets_aligned[i-window:i]) \
                        if len(np.unique(targets_aligned)) < 10 else \
                        1.0 / (1.0 + np.mean((predictions[i-window:i] - targets_aligned[i-window:i]) ** 2))
                    rolling_accuracies.append(window_acc)
                
                volatility = np.std(rolling_accuracies) if rolling_accuracies else 0.0
            else:
                volatility = 0.0
            
            metrics[name] = EnsemblePerformance(
                model_name=name,
                timestamp=datetime.now(),
                accuracy=accuracy,
                recent_accuracy=recent_accuracy,
                volatility=volatility,
                weight=self.weights.get(name, 0.0),
                active=name in self.active_models
            )
            
            # Store in history
            self.performance_history[name].append({
                'timestamp': datetime.now(),
                'accuracy': accuracy,
                'recent_accuracy': recent_accuracy
            })
        
        return metrics
    
    def _update_correlations(self):
        """Update model prediction correlations."""
        model_names = list(self.models.keys())
        n_models = len(model_names)
        correlations = np.eye(n_models)
        
        targets = np.array(self.targets_history)
        
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                if i >= j:  # Skip diagonal and lower triangle
                    continue
                
                if (name1 in self.predictions_history and 
                    name2 in self.predictions_history):
                    
                    pred1 = np.array(self.predictions_history[name1])
                    pred2 = np.array(self.predictions_history[name2])
                    
                    # Align lengths
                    min_len = min(len(pred1), len(pred2), len(targets))
                    if min_len > 10:
                        pred1 = pred1[-min_len:]
                        pred2 = pred2[-min_len:]
                        
                        # Calculate correlation of errors
                        targets_aligned = targets[-min_len:]
                        errors1 = pred1 - targets_aligned
                        errors2 = pred2 - targets_aligned
                        
                        if np.std(errors1) > 0 and np.std(errors2) > 0:
                            corr = np.corrcoef(errors1, errors2)[0, 1]
                            correlations[i, j] = corr
                            correlations[j, i] = corr
        
        self.model_correlations = correlations
    
    def _optimize_weights(self, performance_metrics: Dict[str, EnsemblePerformance]) -> Dict[str, float]:
        """Optimize ensemble weights using portfolio optimization approach."""
        model_names = list(performance_metrics.keys())
        n_models = len(model_names)
        
        if n_models == 0:
            return self.weights
        
        # Extract performance scores
        accuracies = np.array([performance_metrics[name].accuracy for name in model_names])
        recent_accuracies = np.array([performance_metrics[name].recent_accuracy for name in model_names])
        volatilities = np.array([performance_metrics[name].volatility for name in model_names])
        
        # Combined score (recent performance weighted more)
        scores = 0.7 * recent_accuracies + 0.3 * accuracies
        
        # Adjust for volatility (prefer stable models)
        volatility_penalty = 1.0 / (1.0 + volatilities)
        scores = scores * volatility_penalty
        
        # Diversity bonus based on correlations
        if hasattr(self, 'model_correlations'):
            # Get correlation subset for active models
            indices = [list(self.models.keys()).index(name) for name in model_names]
            corr_subset = self.model_correlations[np.ix_(indices, indices)]
            
            # Calculate diversity scores
            avg_correlations = np.mean(np.abs(corr_subset), axis=1) - np.diag(corr_subset)
            diversity_scores = 1.0 - avg_correlations
            
            # Add diversity bonus
            scores = scores + self.diversity_weight * diversity_scores
        
        # Optimization using softmax with temperature
        temperature = 2.0  # Higher = more uniform, lower = more concentrated
        raw_weights = softmax(scores / temperature)
        
        # Apply minimum weight threshold
        raw_weights[raw_weights < self.min_weight] = 0.0
        
        # Renormalize
        if raw_weights.sum() > 0:
            raw_weights = raw_weights / raw_weights.sum()
        else:
            raw_weights = np.ones(n_models) / n_models
        
        # Smooth weight updates
        new_weights = {}
        for i, name in enumerate(model_names):
            old_weight = self.weights.get(name, 1.0 / len(self.models))
            new_weight = raw_weights[i]
            
            # Exponential moving average
            smoothed_weight = (1 - self.adaptation_rate) * old_weight + self.adaptation_rate * new_weight
            new_weights[name] = smoothed_weight
        
        # Ensure non-active models have zero weight
        for name in self.models.keys():
            if name not in new_weights:
                new_weights[name] = 0.0
        
        return new_weights
    
    def _update_active_models(self, performance_metrics: Dict[str, EnsemblePerformance]):
        """Update set of active models based on performance."""
        # Calculate performance thresholds
        if performance_metrics:
            accuracies = [m.accuracy for m in performance_metrics.values()]
            mean_accuracy = np.mean(accuracies)
            min_acceptable = mean_accuracy * 0.8  # 80% of average
        else:
            min_acceptable = 0.5
        
        # Update active set
        new_active = set()
        
        for name, metrics in performance_metrics.items():
            # Keep if above threshold and weight is significant
            if (metrics.recent_accuracy > min_acceptable and 
                self.weights.get(name, 0) > self.min_weight):
                new_active.add(name)
            
            # Reactivate if showing improvement
            elif (name not in self.active_models and 
                  metrics.recent_accuracy > metrics.accuracy * 1.1):
                logger.info(f"Reactivating model {name} due to performance improvement")
                new_active.add(name)
        
        # Ensure minimum ensemble size
        if len(new_active) < 3 and len(performance_metrics) >= 3:
            # Keep top 3 by recent accuracy
            sorted_models = sorted(
                performance_metrics.items(),
                key=lambda x: x[1].recent_accuracy,
                reverse=True
            )
            for name, _ in sorted_models[:3]:
                new_active.add(name)
        
        self.active_models = new_active
    
    def add_model(self, name: str, model: Any):
        """Add a new model to the ensemble."""
        self.models[name] = model
        self.weights[name] = self.min_weight  # Start with minimum weight
        logger.info(f"Added model {name} to ensemble")
    
    def remove_model(self, name: str):
        """Remove a model from the ensemble."""
        if name in self.models:
            del self.models[name]
            if name in self.weights:
                del self.weights[name]
            if name in self.active_models:
                self.active_models.remove(name)
            
            # Renormalize weights
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                for model_name in self.weights:
                    self.weights[model_name] /= total_weight
            
            logger.info(f"Removed model {name} from ensemble")
    
    def get_ensemble_report(self) -> Dict[str, Any]:
        """Get comprehensive ensemble performance report."""
        performance_metrics = self._calculate_performance_metrics()
        
        report = {
            'timestamp': datetime.now(),
            'n_models': len(self.models),
            'n_active': len(self.active_models),
            'weights': self.weights.copy(),
            'model_performance': {}
        }
        
        # Add individual model performance
        for name, metrics in performance_metrics.items():
            report['model_performance'][name] = {
                'accuracy': metrics.accuracy,
                'recent_accuracy': metrics.recent_accuracy,
                'volatility': metrics.volatility,
                'weight': metrics.weight,
                'active': metrics.active
            }
        
        # Calculate ensemble metrics
        if performance_metrics:
            active_weights = [self.weights[name] for name in self.active_models if name in self.weights]
            
            report['ensemble_metrics'] = {
                'weight_concentration': 1.0 - np.exp(-np.sum(np.array(active_weights)**2)) if active_weights else 0.0,
                'avg_correlation': np.mean(self.model_correlations[np.triu_indices_from(self.model_correlations, k=1)]),
                'performance_spread': np.std([m.accuracy for m in performance_metrics.values()])
            }
        
        return report


class DynamicModelSelector:
    """
    Dynamically selects best models for current market conditions.
    
    Features:
    - Market regime detection
    - Model-regime mapping
    - Automatic model switching
    """
    
    def __init__(self,
                 model_pool: Dict[str, Any],
                 regime_window: int = 500,
                 switch_threshold: float = 0.7):
        """
        Initialize dynamic selector.
        
        Args:
            model_pool: Pool of available models
            regime_window: Window for regime detection
            switch_threshold: Confidence threshold for switching
        """
        self.model_pool = model_pool
        self.regime_window = regime_window
        self.switch_threshold = switch_threshold
        
        # Regime detection
        self.regime_features = deque(maxlen=regime_window)
        self.current_regime = 'normal'
        self.regime_history = deque(maxlen=100)
        
        # Model-regime performance
        self.regime_performance = defaultdict(lambda: defaultdict(list))
        
        # Active ensemble
        self.active_ensemble = AdaptiveEnsemble(
            models=self._select_initial_models(),
            window_size=regime_window
        )
        
        logger.info(f"Dynamic selector initialized with {len(model_pool)} models")
    
    def _select_initial_models(self) -> Dict[str, Any]:
        """Select initial model subset."""
        # Start with diverse set
        initial_size = min(5, len(self.model_pool))
        selected_names = list(self.model_pool.keys())[:initial_size]
        
        return {name: self.model_pool[name] for name in selected_names}
    
    def detect_regime(self, X: np.ndarray) -> str:
        """
        Detect current market regime.
        
        Args:
            X: Feature data
            
        Returns:
            Detected regime name
        """
        # Store features
        for row in X:
            self.regime_features.append(row)
        
        if len(self.regime_features) < 50:
            return self.current_regime
        
        # Calculate regime indicators
        features_array = np.array(self.regime_features)
        
        # Volatility regime
        returns = np.diff(features_array[:, 0])  # Assuming first feature is price
        volatility = np.std(returns[-50:])
        historical_vol = np.std(returns)
        
        # Trend regime  
        recent_mean = np.mean(features_array[-50:, 0])
        older_mean = np.mean(features_array[-100:-50, 0]) if len(features_array) > 100 else recent_mean
        trend_strength = (recent_mean - older_mean) / older_mean if older_mean != 0 else 0
        
        # Determine regime
        if volatility > historical_vol * 1.5:
            regime = 'high_volatility'
        elif volatility < historical_vol * 0.5:
            regime = 'low_volatility'
        elif trend_strength > 0.05:
            regime = 'bullish_trend'
        elif trend_strength < -0.05:
            regime = 'bearish_trend'
        else:
            regime = 'normal'
        
        # Store regime
        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': regime,
            'volatility': volatility,
            'trend': trend_strength
        })
        
        return regime
    
    def update_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Update active models based on regime and performance.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Active model set
        """
        # Detect current regime
        new_regime = self.detect_regime(X)
        
        # Update ensemble weights
        self.active_ensemble.update_weights(X, y)
        
        # Track regime performance
        ensemble_report = self.active_ensemble.get_ensemble_report()
        
        for model_name, perf in ensemble_report['model_performance'].items():
            self.regime_performance[new_regime][model_name].append(perf['recent_accuracy'])
        
        # Check if regime changed significantly
        if new_regime != self.current_regime:
            confidence = self._calculate_regime_confidence(new_regime)
            
            if confidence > self.switch_threshold:
                logger.info(f"Regime change detected: {self.current_regime} -> {new_regime}")
                self.current_regime = new_regime
                
                # Select best models for new regime
                best_models = self._select_best_models_for_regime(new_regime)
                
                # Update active ensemble
                self.active_ensemble = AdaptiveEnsemble(
                    models=best_models,
                    window_size=self.regime_window
                )
        
        return self.active_ensemble.models
    
    def _calculate_regime_confidence(self, regime: str) -> float:
        """Calculate confidence in regime detection."""
        if len(self.regime_history) < 5:
            return 0.0
        
        # Check consistency of recent regime detections
        recent_regimes = [r['regime'] for r in list(self.regime_history)[-10:]]
        regime_count = recent_regimes.count(regime)
        
        return regime_count / len(recent_regimes)
    
    def _select_best_models_for_regime(self, regime: str) -> Dict[str, Any]:
        """Select best performing models for given regime."""
        regime_perfs = self.regime_performance[regime]
        
        if not regime_perfs:
            # No historical data for regime, use current ensemble
            return self.active_ensemble.models
        
        # Calculate average performance per model
        model_scores = {}
        for model_name, accuracies in regime_perfs.items():
            if accuracies and model_name in self.model_pool:
                # Weight recent performance more
                weights = np.linspace(0.5, 1.0, len(accuracies))
                model_scores[model_name] = np.average(accuracies, weights=weights)
        
        # Select top performing models
        if model_scores:
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            n_select = min(5, len(sorted_models))
            
            selected = {}
            for model_name, score in sorted_models[:n_select]:
                selected[model_name] = self.model_pool[model_name]
                
            # Ensure minimum diversity
            if len(selected) < 3:
                # Add random models for diversity
                remaining = set(self.model_pool.keys()) - set(selected.keys())
                for model_name in list(remaining)[:3-len(selected)]:
                    selected[model_name] = self.model_pool[model_name]
            
            return selected
        else:
            return self._select_initial_models()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using active ensemble."""
        return self.active_ensemble.predict(X)
    
    def get_selector_report(self) -> Dict[str, Any]:
        """Get comprehensive selector report."""
        report = {
            'timestamp': datetime.now(),
            'current_regime': self.current_regime,
            'regime_confidence': self._calculate_regime_confidence(self.current_regime),
            'active_models': list(self.active_ensemble.models.keys()),
            'ensemble_report': self.active_ensemble.get_ensemble_report()
        }
        
        # Add regime performance summary
        regime_summary = {}
        for regime, model_perfs in self.regime_performance.items():
            regime_summary[regime] = {
                'n_observations': sum(len(perfs) for perfs in model_perfs.values()),
                'best_model': max(model_perfs.items(), key=lambda x: np.mean(x[1]) if x[1] else 0)[0] if model_perfs else None,
                'avg_performance': np.mean([np.mean(perfs) for perfs in model_perfs.values() if perfs]) if model_perfs else 0
            }
        
        report['regime_summary'] = regime_summary
        
        return report