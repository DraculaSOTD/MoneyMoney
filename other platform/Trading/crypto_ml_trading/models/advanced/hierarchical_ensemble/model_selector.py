"""
Model Selection and Management for Hierarchical Ensemble.

Implements intelligent model selection, lifecycle management, and optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations
from models.advanced.hierarchical_ensemble.base_ensemble import BaseModel, ModelPrediction, ModelPerformance


@dataclass
class ModelCandidate:
    """Candidate model for selection."""
    model_id: str
    model_class: type
    model_config: Dict[str, Any]
    specialization: str
    expected_performance: float
    computational_cost: float
    memory_requirements: float
    training_time: float
    compatibility_score: float = 0.0


@dataclass
class SelectionCriteria:
    """Criteria for model selection."""
    performance_weight: float = 0.4
    diversity_weight: float = 0.3
    efficiency_weight: float = 0.2
    compatibility_weight: float = 0.1
    min_performance_threshold: float = 0.3
    max_computational_cost: float = 10.0  # seconds
    max_memory_mb: float = 1000.0


@dataclass
class ModelLifecycle:
    """Model lifecycle tracking."""
    model_id: str
    creation_time: datetime
    last_training: Optional[datetime] = None
    last_prediction: Optional[datetime] = None
    total_predictions: int = 0
    total_training_time: float = 0.0
    performance_history: List[float] = field(default_factory=list)
    status: str = "active"  # active, degraded, retired
    retirement_reason: Optional[str] = None


@dataclass
class SelectionResult:
    """Result of model selection process."""
    timestamp: datetime
    selected_models: List[str]
    rejected_models: List[str]
    selection_scores: Dict[str, float]
    diversity_metrics: Dict[str, float]
    resource_allocation: Dict[str, float]
    optimization_info: Dict[str, Any]


class ModelSelector:
    """
    Intelligent model selection and management system.
    
    Features:
    - Multi-criteria model selection
    - Diversity-aware ensemble composition
    - Performance-based model ranking
    - Resource-aware optimization
    - Lifecycle management and retirement
    - Dynamic model pool management
    - Cross-validation and selection validation
    """
    
    def __init__(self,
                 selection_criteria: Optional[SelectionCriteria] = None,
                 max_models_per_ensemble: int = 10,
                 min_models_per_ensemble: int = 3,
                 reselection_frequency: int = 24):  # hours
        """
        Initialize model selector.
        
        Args:
            selection_criteria: Criteria for model selection
            max_models_per_ensemble: Maximum models per ensemble
            min_models_per_ensemble: Minimum models per ensemble
            reselection_frequency: Frequency of model reselection (hours)
        """
        self.selection_criteria = selection_criteria or SelectionCriteria()
        self.max_models_per_ensemble = max_models_per_ensemble
        self.min_models_per_ensemble = min_models_per_ensemble
        self.reselection_frequency = reselection_frequency
        
        # Model registry and tracking
        self.model_candidates: Dict[str, ModelCandidate] = {}
        self.active_models: Dict[str, BaseModel] = {}
        self.model_lifecycles: Dict[str, ModelLifecycle] = {}
        self.model_performance_cache: Dict[str, ModelPerformance] = {}
        
        # Selection history
        self.selection_history: deque = deque(maxlen=100)
        self.performance_tracking: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        
        # Diversity tracking
        self.diversity_metrics: Dict[str, float] = {}
        self.model_correlations: Dict[Tuple[str, str], float] = {}
        
        # Resource management
        self.resource_usage: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.resource_limits: Dict[str, float] = {
            'total_memory_mb': 4000.0,
            'total_cpu_seconds': 60.0,
            'max_training_time': 300.0
        }
        
        # Last selection time
        self.last_selection: Optional[datetime] = None
        
        # Initialize with some default model types
        self._initialize_default_candidates()
    
    def _initialize_default_candidates(self) -> None:
        """Initialize default model candidates."""
        # This would be populated with actual model classes
        default_candidates = [
            ModelCandidate(
                model_id="linear_regression",
                model_class=type("LinearRegression", (), {}),
                model_config={"regularization": 0.01},
                specialization="linear_trends",
                expected_performance=0.6,
                computational_cost=1.0,
                memory_requirements=50.0,
                training_time=5.0
            ),
            ModelCandidate(
                model_id="random_forest",
                model_class=type("RandomForest", (), {}),
                model_config={"n_estimators": 100, "max_depth": 10},
                specialization="non_linear_patterns",
                expected_performance=0.7,
                computational_cost=5.0,
                memory_requirements=200.0,
                training_time=30.0
            ),
            ModelCandidate(
                model_id="lstm_network",
                model_class=type("LSTMNetwork", (), {}),
                model_config={"hidden_size": 64, "num_layers": 2},
                specialization="temporal_sequences",
                expected_performance=0.75,
                computational_cost=8.0,
                memory_requirements=400.0,
                training_time=120.0
            ),
            ModelCandidate(
                model_id="transformer_model",
                model_class=type("TransformerModel", (), {}),
                model_config={"n_heads": 8, "n_layers": 4},
                specialization="attention_patterns",
                expected_performance=0.8,
                computational_cost=15.0,
                memory_requirements=800.0,
                training_time=200.0
            ),
            ModelCandidate(
                model_id="ensemble_average",
                model_class=type("EnsembleAverage", (), {}),
                model_config={"combination_method": "mean"},
                specialization="stable_consensus",
                expected_performance=0.65,
                computational_cost=2.0,
                memory_requirements=100.0,
                training_time=10.0
            )
        ]
        
        for candidate in default_candidates:
            self.register_model_candidate(candidate)
    
    def register_model_candidate(self, candidate: ModelCandidate) -> None:
        """Register a new model candidate."""
        # Calculate compatibility score
        candidate.compatibility_score = self._calculate_compatibility_score(candidate)
        
        self.model_candidates[candidate.model_id] = candidate
        print(f"Registered model candidate: {candidate.model_id}")
    
    def _calculate_compatibility_score(self, candidate: ModelCandidate) -> float:
        """Calculate compatibility score with existing models."""
        if not self.active_models:
            return 1.0  # Perfect compatibility when no existing models
        
        compatibility_factors = []
        
        # Resource compatibility
        current_memory = sum(
            self.resource_usage.get(mid, {}).get('memory_mb', 0)
            for mid in self.active_models
        )
        memory_compatibility = max(0.0, 1.0 - (current_memory + candidate.memory_requirements) / self.resource_limits['total_memory_mb'])
        compatibility_factors.append(memory_compatibility)
        
        # Performance tier compatibility
        if self.model_performance_cache:
            avg_performance = np.mean([perf.recent_performance for perf in self.model_performance_cache.values()])
            perf_diff = abs(candidate.expected_performance - avg_performance)
            perf_compatibility = max(0.0, 1.0 - perf_diff)
            compatibility_factors.append(perf_compatibility)
        
        # Specialization diversity
        existing_specializations = set(
            self.model_candidates[mid].specialization 
            for mid in self.active_models 
            if mid in self.model_candidates
        )
        
        if candidate.specialization in existing_specializations:
            specialization_score = 0.5  # Some redundancy
        else:
            specialization_score = 1.0  # Good diversity
        
        compatibility_factors.append(specialization_score)
        
        return np.mean(compatibility_factors)
    
    def select_models(self,
                     ensemble_context: Dict[str, Any],
                     current_models: Optional[List[str]] = None,
                     force_reselection: bool = False) -> SelectionResult:
        """
        Select optimal models for ensemble.
        
        Args:
            ensemble_context: Context information for selection
            current_models: Currently active models
            force_reselection: Force reselection even if not due
            
        Returns:
            Selection result with chosen models
        """
        timestamp = datetime.now()
        
        # Check if reselection is needed
        if (not force_reselection and 
            self.last_selection and 
            (timestamp - self.last_selection).total_seconds() < self.reselection_frequency * 3600):
            
            # Return current selection if not time for reselection
            if current_models:
                return SelectionResult(
                    timestamp=timestamp,
                    selected_models=current_models,
                    rejected_models=[],
                    selection_scores={},
                    diversity_metrics={},
                    resource_allocation={},
                    optimization_info={'reason': 'not_due_for_reselection'}
                )
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Calculate selection scores for all candidates
        selection_scores = self._calculate_selection_scores(ensemble_context)
        
        # Perform multi-objective optimization
        selected_models, rejected_models = self._optimize_model_selection(
            selection_scores, ensemble_context
        )
        
        # Calculate diversity metrics
        diversity_metrics = self._calculate_diversity_metrics(selected_models)
        
        # Allocate resources
        resource_allocation = self._allocate_resources(selected_models)
        
        # Create selection result
        result = SelectionResult(
            timestamp=timestamp,
            selected_models=selected_models,
            rejected_models=rejected_models,
            selection_scores=selection_scores,
            diversity_metrics=diversity_metrics,
            resource_allocation=resource_allocation,
            optimization_info={
                'method': 'multi_objective_optimization',
                'criteria_weights': {
                    'performance': self.selection_criteria.performance_weight,
                    'diversity': self.selection_criteria.diversity_weight,
                    'efficiency': self.selection_criteria.efficiency_weight,
                    'compatibility': self.selection_criteria.compatibility_weight
                }
            }
        )
        
        # Store selection history
        self.selection_history.append(result)
        self.last_selection = timestamp
        
        # Update active models
        self._update_active_models(selected_models)
        
        return result
    
    def _calculate_selection_scores(self, ensemble_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate selection scores for all candidates."""
        scores = {}
        
        for model_id, candidate in self.model_candidates.items():
            # Performance score
            performance_score = self._get_performance_score(model_id, candidate, ensemble_context)
            
            # Diversity score
            diversity_score = self._get_diversity_score(model_id, candidate)
            
            # Efficiency score
            efficiency_score = self._get_efficiency_score(candidate)
            
            # Compatibility score
            compatibility_score = candidate.compatibility_score
            
            # Weighted combination
            total_score = (
                self.selection_criteria.performance_weight * performance_score +
                self.selection_criteria.diversity_weight * diversity_score +
                self.selection_criteria.efficiency_weight * efficiency_score +
                self.selection_criteria.compatibility_weight * compatibility_score
            )
            
            scores[model_id] = total_score
        
        return scores
    
    def _get_performance_score(self,
                             model_id: str,
                             candidate: ModelCandidate,
                             ensemble_context: Dict[str, Any]) -> float:
        """Get performance score for a model candidate."""
        # Use actual performance if available
        if model_id in self.model_performance_cache:
            perf = self.model_performance_cache[model_id]
            base_score = perf.recent_performance
        else:
            base_score = candidate.expected_performance
        
        # Adjust based on ensemble context
        context_multiplier = 1.0
        
        # Market regime adjustment
        market_regime = ensemble_context.get('market_regime', 'unknown')
        if market_regime in candidate.model_config.get('preferred_regimes', []):
            context_multiplier *= 1.2
        elif market_regime in candidate.model_config.get('weak_regimes', []):
            context_multiplier *= 0.8
        
        # Volatility adjustment
        volatility = ensemble_context.get('volatility', 0.02)
        if candidate.specialization == 'stable_consensus' and volatility > 0.05:
            context_multiplier *= 1.3  # Favor stable models in high volatility
        elif candidate.specialization == 'momentum_trading' and volatility < 0.01:
            context_multiplier *= 0.7  # Reduce momentum models in low volatility
        
        # Time horizon adjustment
        horizon = ensemble_context.get('prediction_horizon', 15)
        if candidate.specialization == 'temporal_sequences':
            if horizon > 60:  # Long horizon
                context_multiplier *= 1.2
            elif horizon < 5:  # Very short horizon
                context_multiplier *= 0.8
        
        return min(1.0, base_score * context_multiplier)
    
    def _get_diversity_score(self, model_id: str, candidate: ModelCandidate) -> float:
        """Get diversity score for a model candidate."""
        if not self.active_models:
            return 1.0  # Maximum diversity when no active models
        
        diversity_factors = []
        
        # Specialization diversity
        active_specializations = [
            self.model_candidates[mid].specialization 
            for mid in self.active_models 
            if mid in self.model_candidates
        ]
        
        specialization_diversity = 1.0 if candidate.specialization not in active_specializations else 0.3
        diversity_factors.append(specialization_diversity)
        
        # Model type diversity (based on class name)
        active_types = [
            self.model_candidates[mid].model_class.__name__ 
            for mid in self.active_models 
            if mid in self.model_candidates
        ]
        
        type_diversity = 1.0 if candidate.model_class.__name__ not in active_types else 0.4
        diversity_factors.append(type_diversity)
        
        # Configuration diversity
        config_diversity = self._calculate_config_diversity(candidate, self.active_models)
        diversity_factors.append(config_diversity)
        
        # Prediction correlation diversity (if available)
        if model_id in self.model_correlations:
            avg_correlation = np.mean([
                abs(self.model_correlations.get((model_id, other_id), 0.0))
                for other_id in self.active_models
                if (model_id, other_id) in self.model_correlations
            ])
            correlation_diversity = 1.0 - avg_correlation
            diversity_factors.append(correlation_diversity)
        
        return np.mean(diversity_factors)
    
    def _calculate_config_diversity(self,
                                  candidate: ModelCandidate,
                                  active_models: Dict[str, BaseModel]) -> float:
        """Calculate configuration diversity score."""
        if not active_models:
            return 1.0
        
        config_similarities = []
        
        for model_id in active_models:
            if model_id in self.model_candidates:
                other_candidate = self.model_candidates[model_id]
                
                # Compare configuration parameters
                candidate_params = set(candidate.model_config.keys())
                other_params = set(other_candidate.model_config.keys())
                
                # Jaccard similarity of parameter sets
                intersection = len(candidate_params.intersection(other_params))
                union = len(candidate_params.union(other_params))
                
                if union > 0:
                    param_similarity = intersection / union
                    config_similarities.append(param_similarity)
        
        if config_similarities:
            avg_similarity = np.mean(config_similarities)
            return 1.0 - avg_similarity
        else:
            return 1.0
    
    def _get_efficiency_score(self, candidate: ModelCandidate) -> float:
        """Get efficiency score for a model candidate."""
        # Normalize costs to 0-1 scale
        cost_factors = []
        
        # Computational cost efficiency
        max_cost = self.selection_criteria.max_computational_cost
        cost_efficiency = max(0.0, 1.0 - candidate.computational_cost / max_cost)
        cost_factors.append(cost_efficiency)
        
        # Memory efficiency
        max_memory = self.selection_criteria.max_memory_mb
        memory_efficiency = max(0.0, 1.0 - candidate.memory_requirements / max_memory)
        cost_factors.append(memory_efficiency)
        
        # Training time efficiency
        max_training_time = self.resource_limits['max_training_time']
        training_efficiency = max(0.0, 1.0 - candidate.training_time / max_training_time)
        cost_factors.append(training_efficiency)
        
        return np.mean(cost_factors)
    
    def _optimize_model_selection(self,
                                selection_scores: Dict[str, float],
                                ensemble_context: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Optimize model selection using multi-objective approach."""
        # Filter models that meet minimum criteria
        eligible_models = {
            model_id: score for model_id, score in selection_scores.items()
            if score >= self.selection_criteria.min_performance_threshold
        }
        
        if not eligible_models:
            # If no models meet criteria, select top performers anyway
            eligible_models = dict(sorted(
                selection_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:self.min_models_per_ensemble])
        
        # Sort by score
        sorted_models = sorted(eligible_models.items(), key=lambda x: x[1], reverse=True)
        
        # Greedy selection with diversity constraint
        selected_models = []
        rejected_models = []
        
        for model_id, score in sorted_models:
            if len(selected_models) >= self.max_models_per_ensemble:
                rejected_models.append(model_id)
                continue
            
            # Check resource constraints
            if self._check_resource_constraints(selected_models + [model_id]):
                # Check diversity constraints
                if self._check_diversity_constraints(selected_models + [model_id]):
                    selected_models.append(model_id)
                else:
                    rejected_models.append(model_id)
            else:
                rejected_models.append(model_id)
        
        # Ensure minimum number of models
        if len(selected_models) < self.min_models_per_ensemble:
            remaining_needed = self.min_models_per_ensemble - len(selected_models)
            additional_models = [m for m, _ in sorted_models[:remaining_needed] if m not in selected_models]
            selected_models.extend(additional_models[:remaining_needed])
            
            # Remove from rejected
            for model in additional_models:
                if model in rejected_models:
                    rejected_models.remove(model)
        
        return selected_models, rejected_models
    
    def _check_resource_constraints(self, model_ids: List[str]) -> bool:
        """Check if model combination meets resource constraints."""
        total_memory = 0.0
        total_cpu_cost = 0.0
        
        for model_id in model_ids:
            if model_id in self.model_candidates:
                candidate = self.model_candidates[model_id]
                total_memory += candidate.memory_requirements
                total_cpu_cost += candidate.computational_cost
        
        return (total_memory <= self.resource_limits['total_memory_mb'] and
                total_cpu_cost <= self.resource_limits['total_cpu_seconds'])
    
    def _check_diversity_constraints(self, model_ids: List[str]) -> bool:
        """Check if model combination meets diversity constraints."""
        if len(model_ids) <= 1:
            return True
        
        # Check specialization diversity
        specializations = [
            self.model_candidates[mid].specialization 
            for mid in model_ids 
            if mid in self.model_candidates
        ]
        
        unique_specializations = len(set(specializations))
        diversity_ratio = unique_specializations / len(specializations)
        
        # Require at least 50% diversity in specializations
        return diversity_ratio >= 0.5
    
    def _calculate_diversity_metrics(self, selected_models: List[str]) -> Dict[str, float]:
        """Calculate diversity metrics for selected models."""
        if len(selected_models) <= 1:
            return {'specialization_diversity': 1.0, 'type_diversity': 1.0, 'config_diversity': 1.0}
        
        # Specialization diversity
        specializations = [
            self.model_candidates[mid].specialization 
            for mid in selected_models 
            if mid in self.model_candidates
        ]
        spec_diversity = len(set(specializations)) / len(specializations) if specializations else 0.0
        
        # Model type diversity
        model_types = [
            self.model_candidates[mid].model_class.__name__ 
            for mid in selected_models 
            if mid in self.model_candidates
        ]
        type_diversity = len(set(model_types)) / len(model_types) if model_types else 0.0
        
        # Configuration diversity (average pairwise Jaccard distance)
        config_diversities = []
        for i, model1 in enumerate(selected_models):
            for model2 in selected_models[i+1:]:
                if model1 in self.model_candidates and model2 in self.model_candidates:
                    config1 = set(self.model_candidates[model1].model_config.keys())
                    config2 = set(self.model_candidates[model2].model_config.keys())
                    
                    intersection = len(config1.intersection(config2))
                    union = len(config1.union(config2))
                    
                    if union > 0:
                        diversity = 1.0 - (intersection / union)
                        config_diversities.append(diversity)
        
        config_diversity = np.mean(config_diversities) if config_diversities else 1.0
        
        return {
            'specialization_diversity': spec_diversity,
            'type_diversity': type_diversity,
            'config_diversity': config_diversity
        }
    
    def _allocate_resources(self, selected_models: List[str]) -> Dict[str, float]:
        """Allocate computational resources to selected models."""
        allocation = {}
        
        total_cost = sum(
            self.model_candidates[mid].computational_cost 
            for mid in selected_models 
            if mid in self.model_candidates
        )
        
        for model_id in selected_models:
            if model_id in self.model_candidates:
                candidate = self.model_candidates[model_id]
                
                # Proportional allocation based on computational cost
                if total_cost > 0:
                    allocation[model_id] = candidate.computational_cost / total_cost
                else:
                    allocation[model_id] = 1.0 / len(selected_models)
        
        return allocation
    
    def _update_active_models(self, selected_models: List[str]) -> None:
        """Update active models based on selection."""
        # Remove models that are no longer selected
        models_to_remove = [mid for mid in self.active_models if mid not in selected_models]
        for model_id in models_to_remove:
            if model_id in self.active_models:
                del self.active_models[model_id]
                # Update lifecycle
                if model_id in self.model_lifecycles:
                    self.model_lifecycles[model_id].status = "retired"
                    self.model_lifecycles[model_id].retirement_reason = "deselected"
        
        # Add new models (would instantiate actual model objects)
        for model_id in selected_models:
            if model_id not in self.active_models:
                # This would create actual model instances
                # For now, we'll create placeholder objects
                self.active_models[model_id] = type('PlaceholderModel', (), {
                    'model_id': model_id,
                    'is_trained': False
                })()
                
                # Initialize lifecycle tracking
                self.model_lifecycles[model_id] = ModelLifecycle(
                    model_id=model_id,
                    creation_time=datetime.now(),
                    status="active"
                )
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics for all models."""
        # This would collect actual performance data from models
        # For now, we'll update with simulated data
        
        for model_id in self.active_models:
            if model_id in self.performance_tracking:
                recent_scores = list(self.performance_tracking[model_id])[-10:]
                if recent_scores:
                    recent_performance = np.mean(recent_scores)
                    
                    # Update cached performance
                    if model_id not in self.model_performance_cache:
                        self.model_performance_cache[model_id] = ModelPerformance(
                            model_id=model_id,
                            timestamp=datetime.now(),
                            accuracy_metrics={},
                            prediction_errors=[],
                            confidence_calibration=0.5,
                            computational_cost=0.0,
                            stability_score=0.5,
                            recent_performance=recent_performance,
                            long_term_performance=recent_performance
                        )
                    else:
                        perf = self.model_performance_cache[model_id]
                        perf.recent_performance = recent_performance
                        perf.timestamp = datetime.now()
    
    def update_model_performance(self,
                               model_id: str,
                               performance_score: float,
                               prediction_error: float,
                               computational_time: float) -> None:
        """Update performance metrics for a specific model."""
        self.performance_tracking[model_id].append(performance_score)
        
        # Update resource usage
        self.resource_usage[model_id] = {
            'last_cpu_time': computational_time,
            'memory_mb': self.model_candidates.get(model_id, ModelCandidate('', None, {}, '', 0, 0, 100, 0)).memory_requirements
        }
        
        # Update lifecycle
        if model_id in self.model_lifecycles:
            lifecycle = self.model_lifecycles[model_id]
            lifecycle.last_prediction = datetime.now()
            lifecycle.total_predictions += 1
            lifecycle.performance_history.append(performance_score)
            
            # Check for performance degradation
            if len(lifecycle.performance_history) > 10:
                recent_avg = np.mean(lifecycle.performance_history[-5:])
                historical_avg = np.mean(lifecycle.performance_history[-10:-5])
                
                if recent_avg < historical_avg * 0.8:  # 20% degradation
                    lifecycle.status = "degraded"
    
    def get_model_recommendations(self,
                                ensemble_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get model recommendations for current context."""
        # Calculate scores for all candidates
        selection_scores = self._calculate_selection_scores(ensemble_context)
        
        # Get top recommendations
        top_models = sorted(selection_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        recommendations = {
            'top_models': [
                {
                    'model_id': model_id,
                    'score': score,
                    'specialization': self.model_candidates[model_id].specialization,
                    'expected_performance': self.model_candidates[model_id].expected_performance
                }
                for model_id, score in top_models
                if model_id in self.model_candidates
            ],
            'diversity_analysis': self._analyze_current_diversity(),
            'resource_analysis': self._analyze_resource_usage(),
            'lifecycle_analysis': self._analyze_model_lifecycles()
        }
        
        return recommendations
    
    def _analyze_current_diversity(self) -> Dict[str, Any]:
        """Analyze diversity of current model selection."""
        if not self.active_models:
            return {'status': 'no_active_models'}
        
        diversity_metrics = self._calculate_diversity_metrics(list(self.active_models.keys()))
        
        analysis = {
            'diversity_metrics': diversity_metrics,
            'diversity_score': np.mean(list(diversity_metrics.values())),
            'recommendations': []
        }
        
        # Generate recommendations
        if diversity_metrics['specialization_diversity'] < 0.6:
            analysis['recommendations'].append('Consider adding models with different specializations')
        
        if diversity_metrics['type_diversity'] < 0.5:
            analysis['recommendations'].append('Consider adding different model types')
        
        return analysis
    
    def _analyze_resource_usage(self) -> Dict[str, Any]:
        """Analyze current resource usage."""
        if not self.active_models:
            return {'status': 'no_active_models'}
        
        total_memory = sum(
            self.resource_usage.get(mid, {}).get('memory_mb', 0)
            for mid in self.active_models
        )
        
        total_cpu = sum(
            self.resource_usage.get(mid, {}).get('last_cpu_time', 0)
            for mid in self.active_models
        )
        
        analysis = {
            'memory_usage_mb': total_memory,
            'memory_utilization': total_memory / self.resource_limits['total_memory_mb'],
            'cpu_usage_seconds': total_cpu,
            'cpu_utilization': total_cpu / self.resource_limits['total_cpu_seconds'],
            'recommendations': []
        }
        
        # Generate recommendations
        if analysis['memory_utilization'] > 0.8:
            analysis['recommendations'].append('High memory usage - consider removing memory-intensive models')
        
        if analysis['cpu_utilization'] > 0.8:
            analysis['recommendations'].append('High CPU usage - consider optimizing model selection')
        
        return analysis
    
    def _analyze_model_lifecycles(self) -> Dict[str, Any]:
        """Analyze model lifecycle status."""
        active_count = len([l for l in self.model_lifecycles.values() if l.status == "active"])
        degraded_count = len([l for l in self.model_lifecycles.values() if l.status == "degraded"])
        retired_count = len([l for l in self.model_lifecycles.values() if l.status == "retired"])
        
        analysis = {
            'active_models': active_count,
            'degraded_models': degraded_count,
            'retired_models': retired_count,
            'total_models': len(self.model_lifecycles),
            'recommendations': []
        }
        
        # Generate recommendations
        if degraded_count > 0:
            analysis['recommendations'].append(f'{degraded_count} models showing performance degradation')
        
        if active_count < self.min_models_per_ensemble:
            analysis['recommendations'].append('Below minimum active models - consider adding more models')
        
        return analysis
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """Get comprehensive model selection summary."""
        summary = {
            'selection_criteria': {
                'performance_weight': self.selection_criteria.performance_weight,
                'diversity_weight': self.selection_criteria.diversity_weight,
                'efficiency_weight': self.selection_criteria.efficiency_weight,
                'compatibility_weight': self.selection_criteria.compatibility_weight
            },
            'model_counts': {
                'total_candidates': len(self.model_candidates),
                'active_models': len(self.active_models),
                'max_models': self.max_models_per_ensemble,
                'min_models': self.min_models_per_ensemble
            },
            'last_selection': self.last_selection.isoformat() if self.last_selection else None,
            'resource_limits': self.resource_limits.copy()
        }
        
        # Recent selection history
        if self.selection_history:
            recent_selections = list(self.selection_history)[-5:]
            summary['recent_selections'] = [
                {
                    'timestamp': sel.timestamp.isoformat(),
                    'selected_count': len(sel.selected_models),
                    'rejected_count': len(sel.rejected_models),
                    'optimization_method': sel.optimization_info.get('method', 'unknown')
                }
                for sel in recent_selections
            ]
        
        return summary