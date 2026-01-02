"""
Ensemble Coordinator for Hierarchical Architecture.

Coordinates multiple ensemble levels and manages the hierarchical structure.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.advanced.hierarchical_ensemble.base_ensemble import BaseEnsemble, EnsemblePrediction
from models.advanced.hierarchical_ensemble.meta_learner import MetaLearner, MetaFeatures


@dataclass
class EnsembleLevel:
    """Definition of an ensemble level in the hierarchy."""
    level_id: str
    level_name: str
    ensemble_instances: List[BaseEnsemble]
    combination_method: str
    specialization: str  # time_horizon, market_regime, asset_class, etc.
    priority: int
    active: bool = True


@dataclass
class HierarchicalPrediction:
    """Hierarchical ensemble prediction with multi-level breakdown."""
    timestamp: datetime
    asset_symbol: str
    final_prediction: Union[float, np.ndarray]
    confidence: float
    horizon: int
    level_predictions: Dict[str, EnsemblePrediction]
    level_weights: Dict[str, float]
    aggregation_method: str
    uncertainty_bounds: Tuple[float, float]
    consensus_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HierarchyConfiguration:
    """Configuration for the hierarchical ensemble."""
    max_levels: int
    level_definitions: List[EnsembleLevel]
    inter_level_weights: Dict[str, float]
    coordination_method: str
    rebalancing_frequency: int  # minutes
    performance_window: int


class EnsembleCoordinator:
    """
    Coordinates hierarchical ensemble architecture.
    
    Features:
    - Multi-level ensemble coordination
    - Dynamic hierarchy management
    - Cross-level weight optimization
    - Performance attribution across levels
    - Real-time coordination and monitoring
    - Adaptive hierarchy reconfiguration
    """
    
    def __init__(self,
                 coordination_method: str = "weighted_consensus",
                 rebalancing_frequency: int = 60,
                 max_ensemble_levels: int = 4,
                 performance_attribution: bool = True):
        """
        Initialize ensemble coordinator.
        
        Args:
            coordination_method: Method for coordinating ensemble levels
            rebalancing_frequency: Frequency of weight rebalancing (minutes)
            max_ensemble_levels: Maximum number of hierarchy levels
            performance_attribution: Enable detailed performance attribution
        """
        self.coordination_method = coordination_method
        self.rebalancing_frequency = rebalancing_frequency
        self.max_ensemble_levels = max_ensemble_levels
        self.performance_attribution = performance_attribution
        
        # Hierarchy structure
        self.ensemble_levels: Dict[str, EnsembleLevel] = {}
        self.level_hierarchy: List[str] = []  # Ordered list of level IDs
        self.inter_level_weights: Dict[str, float] = {}
        
        # Meta-learner for coordination
        self.meta_learner = MetaLearner(
            learning_rate=0.01,
            meta_window=200,
            adaptation_threshold=0.05
        )
        
        # Prediction history
        self.hierarchical_predictions: deque = deque(maxlen=1000)
        self.level_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        
        # Coordination state
        self.last_rebalancing: Optional[datetime] = None
        self.coordination_active = False
        self.coordination_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.hierarchy_performance: deque = deque(maxlen=200)
        self.attribution_results: deque = deque(maxlen=100)
        
        # Initialize default hierarchy
        self._initialize_default_hierarchy()
    
    def _initialize_default_hierarchy(self) -> None:
        """Initialize default hierarchical structure."""
        # Level 1: Time Horizon Specialists
        level1 = EnsembleLevel(
            level_id="L1_time_horizon",
            level_name="Time Horizon Specialists",
            ensemble_instances=[],
            combination_method="weighted_average",
            specialization="time_horizon",
            priority=1
        )
        
        # Level 2: Market Regime Specialists
        level2 = EnsembleLevel(
            level_id="L2_market_regime",
            level_name="Market Regime Specialists", 
            ensemble_instances=[],
            combination_method="regime_weighted",
            specialization="market_regime",
            priority=2
        )
        
        # Level 3: Asset Class Specialists
        level3 = EnsembleLevel(
            level_id="L3_asset_class",
            level_name="Asset Class Specialists",
            ensemble_instances=[],
            combination_method="correlation_weighted",
            specialization="asset_class",
            priority=3
        )
        
        # Level 4: Meta-Ensemble
        level4 = EnsembleLevel(
            level_id="L4_meta",
            level_name="Meta-Ensemble",
            ensemble_instances=[],
            combination_method="meta_learned",
            specialization="meta",
            priority=4
        )
        
        # Register levels
        for level in [level1, level2, level3, level4]:
            self.register_ensemble_level(level)
    
    def register_ensemble_level(self, level: EnsembleLevel) -> None:
        """Register an ensemble level in the hierarchy."""
        self.ensemble_levels[level.level_id] = level
        
        # Maintain ordered hierarchy
        if level.level_id not in self.level_hierarchy:
            # Insert in priority order
            inserted = False
            for i, existing_id in enumerate(self.level_hierarchy):
                existing_level = self.ensemble_levels[existing_id]
                if level.priority < existing_level.priority:
                    self.level_hierarchy.insert(i, level.level_id)
                    inserted = True
                    break
            
            if not inserted:
                self.level_hierarchy.append(level.level_id)
        
        # Initialize weight
        self.inter_level_weights[level.level_id] = 1.0 / len(self.ensemble_levels)
        
        print(f"Registered ensemble level: {level.level_name}")
    
    def add_ensemble_to_level(self, level_id: str, ensemble: BaseEnsemble) -> None:
        """Add ensemble instance to a specific level."""
        if level_id in self.ensemble_levels:
            self.ensemble_levels[level_id].ensemble_instances.append(ensemble)
            print(f"Added ensemble {ensemble.ensemble_id} to level {level_id}")
        else:
            raise ValueError(f"Level {level_id} not found")
    
    def predict(self,
                X: np.ndarray,
                asset_symbol: str,
                horizon: int = 15,
                market_data: Optional[Dict[str, Any]] = None,
                **kwargs) -> HierarchicalPrediction:
        """
        Generate hierarchical ensemble prediction.
        
        Args:
            X: Input features
            asset_symbol: Asset symbol for prediction
            horizon: Prediction horizon in minutes
            market_data: Additional market context data
            
        Returns:
            Hierarchical prediction with multi-level breakdown
        """
        timestamp = datetime.now()
        market_data = market_data or {}
        
        # Collect predictions from each active level
        level_predictions = {}
        prediction_errors = []
        
        for level_id in self.level_hierarchy:
            level = self.ensemble_levels[level_id]
            
            if not level.active or not level.ensemble_instances:
                continue
            
            try:
                level_pred = self._get_level_prediction(
                    level, X, asset_symbol, horizon, market_data, **kwargs
                )
                level_predictions[level_id] = level_pred
                
            except Exception as e:
                print(f"Error in level {level_id}: {e}")
                prediction_errors.append(f"Level {level_id}: {str(e)}")
                continue
        
        if not level_predictions:
            raise RuntimeError("No ensemble levels produced valid predictions")
        
        # Extract meta-features for coordination
        all_component_predictions = []
        for level_pred in level_predictions.values():
            all_component_predictions.extend(level_pred.component_predictions)
        
        meta_features = self.meta_learner.extract_meta_features(
            all_component_predictions,
            market_data,
            list(self.hierarchical_predictions)[-20:] if self.hierarchical_predictions else []
        )
        
        # Optimize inter-level weights
        current_weights = self.inter_level_weights.copy()
        meta_result = self.meta_learner.optimize_weights(
            meta_features,
            current_weights,
            list(level_predictions.keys())
        )
        
        # Update inter-level weights
        self.inter_level_weights.update(meta_result.optimal_weights)
        
        # Combine level predictions
        final_prediction, confidence, uncertainty_bounds = self._combine_level_predictions(
            level_predictions, meta_result.optimal_weights
        )
        
        # Calculate consensus score
        consensus_score = self._calculate_consensus_score(level_predictions)
        
        # Create hierarchical prediction
        hierarchical_pred = HierarchicalPrediction(
            timestamp=timestamp,
            asset_symbol=asset_symbol,
            final_prediction=final_prediction,
            confidence=confidence,
            horizon=horizon,
            level_predictions=level_predictions,
            level_weights=meta_result.optimal_weights,
            aggregation_method=self.coordination_method,
            uncertainty_bounds=uncertainty_bounds,
            consensus_score=consensus_score,
            metadata={
                'meta_features': meta_features.__dict__,
                'prediction_errors': prediction_errors,
                'meta_optimization': meta_result.optimization_method
            }
        )
        
        # Store prediction
        self.hierarchical_predictions.append(hierarchical_pred)
        
        # Update level performance tracking
        for level_id, level_pred in level_predictions.items():
            self.level_performance[level_id].append({
                'timestamp': timestamp,
                'prediction': level_pred.prediction,
                'confidence': level_pred.confidence,
                'weight': meta_result.optimal_weights.get(level_id, 0.0)
            })
        
        return hierarchical_pred
    
    def _get_level_prediction(self,
                            level: EnsembleLevel,
                            X: np.ndarray,
                            asset_symbol: str,
                            horizon: int,
                            market_data: Dict[str, Any],
                            **kwargs) -> EnsemblePrediction:
        """Get prediction from a specific ensemble level."""
        if not level.ensemble_instances:
            raise ValueError(f"No ensemble instances in level {level.level_id}")
        
        # For now, use the first ensemble instance
        # In full implementation, would coordinate multiple ensembles per level
        ensemble = level.ensemble_instances[0]
        
        # Customize prediction based on level specialization
        level_kwargs = kwargs.copy()
        
        if level.specialization == "time_horizon":
            # Time horizon specialists might adjust based on horizon
            level_kwargs['horizon_specific'] = True
        
        elif level.specialization == "market_regime":
            # Market regime specialists adjust based on current regime
            current_regime = market_data.get('regime', 'unknown')
            level_kwargs['regime_context'] = current_regime
        
        elif level.specialization == "asset_class":
            # Asset class specialists adjust based on asset characteristics
            level_kwargs['asset_context'] = asset_symbol
        
        # Get prediction from ensemble
        prediction = ensemble.predict(X, horizon=horizon, **level_kwargs)
        
        return prediction
    
    def _combine_level_predictions(self,
                                 level_predictions: Dict[str, EnsemblePrediction],
                                 level_weights: Dict[str, float]) -> Tuple[Union[float, np.ndarray], float, Tuple[float, float]]:
        """Combine predictions from different levels."""
        if self.coordination_method == "weighted_consensus":
            return self._weighted_consensus(level_predictions, level_weights)
        elif self.coordination_method == "hierarchical_voting":
            return self._hierarchical_voting(level_predictions, level_weights)
        elif self.coordination_method == "uncertainty_weighted":
            return self._uncertainty_weighted(level_predictions, level_weights)
        else:
            # Default to weighted consensus
            return self._weighted_consensus(level_predictions, level_weights)
    
    def _weighted_consensus(self,
                          level_predictions: Dict[str, EnsemblePrediction],
                          level_weights: Dict[str, float]) -> Tuple[Union[float, np.ndarray], float, Tuple[float, float]]:
        """Weighted consensus combination method."""
        # Collect predictions and weights
        predictions = []
        weights = []
        confidences = []
        
        for level_id, pred in level_predictions.items():
            weight = level_weights.get(level_id, 0.0)
            
            if weight > 0:
                predictions.append(pred.prediction)
                weights.append(weight)
                confidences.append(pred.confidence)
        
        if not predictions:
            raise ValueError("No valid predictions with positive weights")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Weighted average
        if isinstance(predictions[0], np.ndarray):
            final_prediction = np.average(predictions, weights=weights, axis=0)
        else:
            final_prediction = np.average(predictions, weights=weights)
        
        # Weighted confidence
        final_confidence = np.average(confidences, weights=weights)
        
        # Calculate uncertainty bounds
        if isinstance(predictions[0], np.ndarray):
            prediction_std = np.std(predictions, axis=0)
            uncertainty = np.mean(prediction_std)
        else:
            uncertainty = np.std(predictions)
        
        # Uncertainty bounds (approximate 95% confidence interval)
        if isinstance(final_prediction, np.ndarray):
            uncertainty_bounds = (
                float(np.mean(final_prediction) - 1.96 * uncertainty),
                float(np.mean(final_prediction) + 1.96 * uncertainty)
            )
        else:
            uncertainty_bounds = (
                float(final_prediction - 1.96 * uncertainty),
                float(final_prediction + 1.96 * uncertainty)
            )
        
        return final_prediction, final_confidence, uncertainty_bounds
    
    def _hierarchical_voting(self,
                           level_predictions: Dict[str, EnsemblePrediction],
                           level_weights: Dict[str, float]) -> Tuple[Union[float, np.ndarray], float, Tuple[float, float]]:
        """Hierarchical voting combination method."""
        # Sort levels by priority
        sorted_levels = sorted(
            level_predictions.keys(),
            key=lambda x: self.ensemble_levels[x].priority
        )
        
        # Higher priority levels get more votes
        votes = []
        confidences = []
        
        for i, level_id in enumerate(sorted_levels):
            pred = level_predictions[level_id]
            weight = level_weights.get(level_id, 0.0)
            
            # Priority bonus
            priority_multiplier = len(sorted_levels) - i
            effective_weight = weight * priority_multiplier
            
            votes.append((pred.prediction, effective_weight))
            confidences.append(pred.confidence * effective_weight)
        
        # Calculate weighted result
        total_weight = sum(w for _, w in votes)
        
        if total_weight > 0:
            if isinstance(votes[0][0], np.ndarray):
                final_prediction = sum(pred * w for pred, w in votes) / total_weight
            else:
                final_prediction = sum(pred * w for pred, w in votes) / total_weight
            
            final_confidence = sum(confidences) / total_weight
        else:
            # Fallback
            first_pred = level_predictions[sorted_levels[0]]
            final_prediction = first_pred.prediction
            final_confidence = first_pred.confidence
        
        # Simple uncertainty bounds
        pred_values = [pred.prediction for pred in level_predictions.values()]
        if isinstance(pred_values[0], np.ndarray):
            uncertainty = np.mean(np.std(pred_values, axis=0))
        else:
            uncertainty = np.std(pred_values)
        
        if isinstance(final_prediction, np.ndarray):
            uncertainty_bounds = (
                float(np.mean(final_prediction) - uncertainty),
                float(np.mean(final_prediction) + uncertainty)
            )
        else:
            uncertainty_bounds = (
                float(final_prediction - uncertainty),
                float(final_prediction + uncertainty)
            )
        
        return final_prediction, final_confidence, uncertainty_bounds
    
    def _uncertainty_weighted(self,
                            level_predictions: Dict[str, EnsemblePrediction],
                            level_weights: Dict[str, float]) -> Tuple[Union[float, np.ndarray], float, Tuple[float, float]]:
        """Uncertainty-weighted combination method."""
        # Weight by inverse uncertainty (higher confidence gets more weight)
        predictions = []
        weights = []
        uncertainties = []
        
        for level_id, pred in level_predictions.items():
            base_weight = level_weights.get(level_id, 0.0)
            uncertainty = 1.0 - pred.confidence  # Convert confidence to uncertainty
            
            # Weight inversely proportional to uncertainty
            uncertainty_weight = 1.0 / (uncertainty + 1e-6)  # Add small epsilon
            combined_weight = base_weight * uncertainty_weight
            
            predictions.append(pred.prediction)
            weights.append(combined_weight)
            uncertainties.append(uncertainty)
        
        # Normalize weights
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        # Weighted combination
        if isinstance(predictions[0], np.ndarray):
            final_prediction = np.average(predictions, weights=weights, axis=0)
        else:
            final_prediction = np.average(predictions, weights=weights)
        
        # Uncertainty-weighted confidence
        final_confidence = 1.0 - np.average(uncertainties, weights=weights)
        
        # Uncertainty bounds based on prediction variance
        if isinstance(predictions[0], np.ndarray):
            prediction_var = np.average(
                [(pred - final_prediction) ** 2 for pred in predictions],
                weights=weights,
                axis=0
            )
            uncertainty = np.sqrt(np.mean(prediction_var))
        else:
            prediction_var = np.average(
                [(pred - final_prediction) ** 2 for pred in predictions],
                weights=weights
            )
            uncertainty = np.sqrt(prediction_var)
        
        if isinstance(final_prediction, np.ndarray):
            uncertainty_bounds = (
                float(np.mean(final_prediction) - 2 * uncertainty),
                float(np.mean(final_prediction) + 2 * uncertainty)
            )
        else:
            uncertainty_bounds = (
                float(final_prediction - 2 * uncertainty),
                float(final_prediction + 2 * uncertainty)
            )
        
        return final_prediction, final_confidence, uncertainty_bounds
    
    def _calculate_consensus_score(self, level_predictions: Dict[str, EnsemblePrediction]) -> float:
        """Calculate consensus score across ensemble levels."""
        if len(level_predictions) < 2:
            return 1.0
        
        # Collect prediction values
        pred_values = []
        for pred in level_predictions.values():
            if isinstance(pred.prediction, np.ndarray):
                pred_values.append(np.mean(pred.prediction))
            else:
                pred_values.append(float(pred.prediction))
        
        # Calculate coefficient of variation (inverse of consensus)
        if len(pred_values) > 1 and np.mean(pred_values) != 0:
            cv = np.std(pred_values) / abs(np.mean(pred_values))
            consensus_score = max(0.0, 1.0 - cv)
        else:
            consensus_score = 1.0
        
        return consensus_score
    
    def start_coordination(self) -> None:
        """Start real-time ensemble coordination."""
        if self.coordination_active:
            return
        
        self.coordination_active = True
        self.coordination_thread = threading.Thread(
            target=self._coordination_loop,
            daemon=True
        )
        self.coordination_thread.start()
        print("Hierarchical ensemble coordination started")
    
    def stop_coordination(self) -> None:
        """Stop real-time coordination."""
        self.coordination_active = False
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5)
        print("Hierarchical ensemble coordination stopped")
    
    def _coordination_loop(self) -> None:
        """Main coordination loop."""
        while self.coordination_active:
            try:
                current_time = datetime.now()
                
                # Check if rebalancing is needed
                if (self.last_rebalancing is None or 
                    (current_time - self.last_rebalancing).total_seconds() >= self.rebalancing_frequency * 60):
                    
                    self._perform_rebalancing()
                    self.last_rebalancing = current_time
                
                # Check for adaptation triggers
                self._check_adaptation_triggers()
                
                # Update performance metrics
                self._update_hierarchy_performance()
                
                # Sleep until next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"Error in coordination loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _perform_rebalancing(self) -> None:
        """Perform periodic rebalancing of inter-level weights."""
        if len(self.hierarchical_predictions) < 10:
            return
        
        print("Performing hierarchical ensemble rebalancing...")
        
        # Analyze recent performance by level
        recent_predictions = list(self.hierarchical_predictions)[-50:]
        level_performance_scores = defaultdict(list)
        
        for pred in recent_predictions:
            for level_id, level_pred in pred.level_predictions.items():
                # Score based on confidence and contribution
                weight = pred.level_weights.get(level_id, 0.0)
                score = level_pred.confidence * weight * pred.consensus_score
                level_performance_scores[level_id].append(score)
        
        # Update inter-level weights based on performance
        new_weights = {}
        total_score = 0.0
        
        for level_id in self.inter_level_weights:
            if level_id in level_performance_scores and level_performance_scores[level_id]:
                avg_score = np.mean(level_performance_scores[level_id])
                new_weights[level_id] = max(0.1, avg_score)  # Minimum weight
                total_score += new_weights[level_id]
            else:
                new_weights[level_id] = 0.1  # Default minimum
                total_score += 0.1
        
        # Normalize weights
        if total_score > 0:
            for level_id in new_weights:
                new_weights[level_id] /= total_score
        
        # Exponential moving average update
        alpha = 0.1
        for level_id in self.inter_level_weights:
            old_weight = self.inter_level_weights[level_id]
            new_weight = new_weights.get(level_id, old_weight)
            self.inter_level_weights[level_id] = alpha * new_weight + (1 - alpha) * old_weight
    
    def _check_adaptation_triggers(self) -> None:
        """Check for conditions requiring hierarchy adaptation."""
        if not self.hierarchical_predictions:
            return
        
        recent_pred = self.hierarchical_predictions[-1]
        
        # Extract meta-features from recent prediction
        meta_features = recent_pred.metadata.get('meta_features', {})
        
        if meta_features:
            # Reconstruct MetaFeatures object (simplified)
            meta_features_obj = MetaFeatures(
                timestamp=recent_pred.timestamp,
                market_regime=meta_features.get('market_regime', 'unknown'),
                volatility=meta_features.get('volatility', 0.02),
                trend_strength=meta_features.get('trend_strength', 0.5),
                model_disagreement=meta_features.get('model_disagreement', 0.0),
                prediction_horizon=recent_pred.horizon,
                feature_complexity=meta_features.get('feature_complexity', 0.5),
                data_quality=meta_features.get('data_quality', 0.5),
                ensemble_history=meta_features.get('ensemble_history', {})
            )
            
            # Check for adaptation triggers
            ensemble_performance = {
                'accuracy': recent_pred.confidence,
                'consensus': recent_pred.consensus_score
            }
            
            triggered_strategies = self.meta_learner.check_adaptation_triggers(
                meta_features_obj,
                ensemble_performance
            )
            
            # Apply triggered adaptations
            for strategy in triggered_strategies:
                print(f"Applying adaptation strategy: {strategy}")
                self._apply_hierarchy_adaptation(strategy)
    
    def _apply_hierarchy_adaptation(self, strategy_name: str) -> None:
        """Apply adaptation strategy to hierarchy."""
        # Apply strategy to inter-level weights
        adapted_weights = self.meta_learner.apply_adaptation_strategy(
            strategy_name,
            self.inter_level_weights,
            list(self.ensemble_levels.keys())
        )
        
        self.inter_level_weights.update(adapted_weights)
        
        # Could also modify level configurations, activation status, etc.
        if strategy_name == "high_volatility_adaptation":
            # Increase weight of conservative levels
            for level_id, level in self.ensemble_levels.items():
                if "conservative" in level.level_name.lower():
                    self.inter_level_weights[level_id] *= 1.2
        
        # Renormalize weights
        total_weight = sum(self.inter_level_weights.values())
        if total_weight > 0:
            for level_id in self.inter_level_weights:
                self.inter_level_weights[level_id] /= total_weight
    
    def _update_hierarchy_performance(self) -> None:
        """Update overall hierarchy performance metrics."""
        if len(self.hierarchical_predictions) < 5:
            return
        
        recent_predictions = list(self.hierarchical_predictions)[-20:]
        
        # Calculate hierarchy-level metrics
        avg_confidence = np.mean([p.confidence for p in recent_predictions])
        avg_consensus = np.mean([p.consensus_score for p in recent_predictions])
        
        # Weight distribution entropy
        weight_entropies = []
        for pred in recent_predictions:
            weights = list(pred.level_weights.values())
            weights = np.array(weights)
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
            
            entropy = -np.sum([w * np.log(w + 1e-10) for w in weights if w > 0])
            weight_entropies.append(entropy)
        
        avg_weight_entropy = np.mean(weight_entropies)
        
        # Store performance record
        performance_record = {
            'timestamp': datetime.now(),
            'avg_confidence': avg_confidence,
            'avg_consensus': avg_consensus,
            'weight_entropy': avg_weight_entropy,
            'active_levels': len([l for l in self.ensemble_levels.values() if l.active]),
            'total_predictions': len(recent_predictions)
        }
        
        self.hierarchy_performance.append(performance_record)
    
    def update_performance(self,
                         actual_values: np.ndarray,
                         prediction_timestamps: List[datetime]) -> None:
        """Update performance metrics with actual outcomes."""
        for timestamp, actual in zip(prediction_timestamps, actual_values):
            # Find matching hierarchical predictions
            matching_predictions = [
                p for p in self.hierarchical_predictions
                if abs((p.timestamp - timestamp).total_seconds()) < 300
            ]
            
            for hier_pred in matching_predictions:
                self._update_hierarchical_performance(hier_pred, actual)
    
    def _update_hierarchical_performance(self,
                                       prediction: HierarchicalPrediction,
                                       actual: float) -> None:
        """Update performance for a hierarchical prediction."""
        # Calculate prediction error
        if isinstance(prediction.final_prediction, np.ndarray):
            error = np.mean(np.abs(prediction.final_prediction - actual))
        else:
            error = abs(prediction.final_prediction - actual)
        
        accuracy = max(0.0, 1.0 - error)
        
        # Update meta-learner with feedback
        if prediction.metadata.get('meta_optimization'):
            # Create mock meta-learning result for feedback
            mock_result = type('MetaLearningResult', (), {
                'predicted_performance': prediction.confidence,
                'optimal_weights': prediction.level_weights
            })()
            
            self.meta_learner.update_performance_feedback(mock_result, accuracy)
        
        # Store attribution results
        if self.performance_attribution:
            attribution = {
                'timestamp': prediction.timestamp,
                'actual_performance': accuracy,
                'predicted_performance': prediction.confidence,
                'level_contributions': {},
                'level_errors': {}
            }
            
            for level_id, level_pred in prediction.level_predictions.items():
                weight = prediction.level_weights.get(level_id, 0.0)
                
                if isinstance(level_pred.prediction, np.ndarray):
                    level_error = np.mean(np.abs(level_pred.prediction - actual))
                else:
                    level_error = abs(level_pred.prediction - actual)
                
                level_accuracy = max(0.0, 1.0 - level_error)
                
                attribution['level_contributions'][level_id] = weight * level_accuracy
                attribution['level_errors'][level_id] = level_error
            
            self.attribution_results.append(attribution)
    
    def get_hierarchy_summary(self) -> Dict[str, Any]:
        """Get comprehensive hierarchy summary."""
        summary = {
            'coordination_method': self.coordination_method,
            'active_levels': len([l for l in self.ensemble_levels.values() if l.active]),
            'total_levels': len(self.ensemble_levels),
            'inter_level_weights': self.inter_level_weights.copy(),
            'level_hierarchy': self.level_hierarchy.copy()
        }
        
        # Level details
        summary['levels'] = {}
        for level_id, level in self.ensemble_levels.items():
            summary['levels'][level_id] = {
                'name': level.level_name,
                'specialization': level.specialization,
                'priority': level.priority,
                'active': level.active,
                'num_ensembles': len(level.ensemble_instances),
                'weight': self.inter_level_weights.get(level_id, 0.0)
            }
        
        # Recent performance
        if self.hierarchy_performance:
            recent_perf = list(self.hierarchy_performance)[-10:]
            summary['recent_performance'] = {
                'avg_confidence': np.mean([p['avg_confidence'] for p in recent_perf]),
                'avg_consensus': np.mean([p['avg_consensus'] for p in recent_perf]),
                'avg_weight_entropy': np.mean([p['weight_entropy'] for p in recent_perf])
            }
        
        # Meta-learner status
        summary['meta_learner'] = self.meta_learner.get_meta_learning_summary()
        
        return summary
    
    def get_performance_attribution(self, lookback_hours: int = 24) -> Dict[str, Any]:
        """Get performance attribution analysis."""
        if not self.attribution_results:
            return {}
        
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        recent_attributions = [
            attr for attr in self.attribution_results
            if attr['timestamp'] >= cutoff_time
        ]
        
        if not recent_attributions:
            return {}
        
        # Aggregate attribution by level
        level_contributions = defaultdict(list)
        level_errors = defaultdict(list)
        
        for attr in recent_attributions:
            for level_id, contribution in attr['level_contributions'].items():
                level_contributions[level_id].append(contribution)
            
            for level_id, error in attr['level_errors'].items():
                level_errors[level_id].append(error)
        
        # Calculate summary statistics
        attribution_summary = {}
        for level_id in level_contributions:
            contributions = level_contributions[level_id]
            errors = level_errors[level_id]
            
            attribution_summary[level_id] = {
                'avg_contribution': np.mean(contributions),
                'contribution_std': np.std(contributions),
                'avg_error': np.mean(errors),
                'error_std': np.std(errors),
                'num_predictions': len(contributions)
            }
        
        return {
            'attribution_period_hours': lookback_hours,
            'total_predictions': len(recent_attributions),
            'level_attributions': attribution_summary,
            'overall_accuracy': np.mean([attr['actual_performance'] for attr in recent_attributions])
        }