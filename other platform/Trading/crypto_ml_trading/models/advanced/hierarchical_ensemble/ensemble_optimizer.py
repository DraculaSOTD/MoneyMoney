"""
Ensemble Optimizer for Hierarchical Architecture.

Implements advanced optimization techniques for ensemble performance.
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


@dataclass
class OptimizationTarget:
    """Optimization target specification."""
    objective: str  # maximize_accuracy, minimize_error, maximize_sharpe, etc.
    weight: float
    constraints: Dict[str, Any] = field(default_factory=dict)
    target_value: Optional[float] = None


@dataclass
class OptimizationResult:
    """Result of ensemble optimization."""
    timestamp: datetime
    optimization_method: str
    objective_value: float
    optimal_weights: Dict[str, float]
    optimal_parameters: Dict[str, Any]
    convergence_info: Dict[str, Any]
    performance_improvement: float
    validation_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration for optimization."""
    parameter_name: str
    parameter_type: str  # continuous, discrete, categorical
    bounds: Tuple[Any, Any]
    current_value: Any
    importance: float = 1.0
    search_strategy: str = "random"  # random, grid, bayesian


@dataclass
class PerformanceProfile:
    """Performance profile for optimization."""
    timestamp: datetime
    accuracy_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    efficiency_metrics: Dict[str, float]
    stability_metrics: Dict[str, float]
    composite_score: float


class EnsembleOptimizer:
    """
    Advanced ensemble optimization system.
    
    Features:
    - Multi-objective optimization
    - Hyperparameter optimization
    - Weight optimization with constraints
    - Performance prediction and validation
    - Bayesian optimization
    - Genetic algorithm optimization
    - Real-time optimization adaptation
    - Cross-validation and robustness testing
    """
    
    def __init__(self,
                 optimization_frequency: int = 120,  # minutes
                 validation_split: float = 0.2,
                 max_optimization_time: float = 300.0,  # seconds
                 convergence_tolerance: float = 1e-6):
        """
        Initialize ensemble optimizer.
        
        Args:
            optimization_frequency: Frequency of optimization runs (minutes)
            validation_split: Fraction of data for validation
            max_optimization_time: Maximum time for optimization (seconds)
            convergence_tolerance: Convergence tolerance for optimization
        """
        self.optimization_frequency = optimization_frequency
        self.validation_split = validation_split
        self.max_optimization_time = max_optimization_time
        self.convergence_tolerance = convergence_tolerance
        
        # Optimization history
        self.optimization_results: deque = deque(maxlen=100)
        self.performance_profiles: deque = deque(maxlen=500)
        
        # Current optimization state
        self.current_targets: List[OptimizationTarget] = []
        self.hyperparameter_configs: Dict[str, HyperparameterConfig] = {}
        self.optimization_constraints: Dict[str, Any] = {}
        
        # Optimization methods
        self.optimization_methods = {
            'gradient_descent': self._gradient_descent_optimization,
            'bayesian': self._bayesian_optimization,
            'genetic_algorithm': self._genetic_algorithm_optimization,
            'particle_swarm': self._particle_swarm_optimization,
            'random_search': self._random_search_optimization,
            'grid_search': self._grid_search_optimization
        }
        
        # Performance tracking
        self.baseline_performance: Optional[PerformanceProfile] = None
        self.best_performance: Optional[PerformanceProfile] = None
        
        # Optimization state
        self.last_optimization: Optional[datetime] = None
        self.optimization_active = False
        
        # Initialize default targets
        self._initialize_default_targets()
        
        # Initialize default hyperparameters
        self._initialize_default_hyperparameters()
    
    def _initialize_default_targets(self) -> None:
        """Initialize default optimization targets."""
        default_targets = [
            OptimizationTarget(
                objective="maximize_accuracy",
                weight=0.4,
                constraints={"min_value": 0.5},
                target_value=0.8
            ),
            OptimizationTarget(
                objective="minimize_risk",
                weight=0.3,
                constraints={"max_drawdown": 0.15},
                target_value=0.05
            ),
            OptimizationTarget(
                objective="maximize_efficiency",
                weight=0.2,
                constraints={"max_cpu_time": 60.0},
                target_value=0.9
            ),
            OptimizationTarget(
                objective="maximize_stability",
                weight=0.1,
                constraints={"min_consistency": 0.7},
                target_value=0.85
            )
        ]
        
        self.current_targets = default_targets
    
    def _initialize_default_hyperparameters(self) -> None:
        """Initialize default hyperparameter configurations."""
        default_hyperparams = [
            HyperparameterConfig(
                parameter_name="learning_rate",
                parameter_type="continuous",
                bounds=(0.001, 0.1),
                current_value=0.01,
                importance=0.8
            ),
            HyperparameterConfig(
                parameter_name="ensemble_size",
                parameter_type="discrete",
                bounds=(3, 15),
                current_value=7,
                importance=0.6
            ),
            HyperparameterConfig(
                parameter_name="rebalancing_frequency",
                parameter_type="discrete", 
                bounds=(30, 480),  # 30 minutes to 8 hours
                current_value=120,
                importance=0.4
            ),
            HyperparameterConfig(
                parameter_name="combination_method",
                parameter_type="categorical",
                bounds=("weighted_average", "bayesian_averaging", "stacking"),
                current_value="weighted_average",
                importance=0.5
            )
        ]
        
        for config in default_hyperparams:
            self.hyperparameter_configs[config.parameter_name] = config
    
    def add_optimization_target(self, target: OptimizationTarget) -> None:
        """Add optimization target."""
        self.current_targets.append(target)
        print(f"Added optimization target: {target.objective}")
    
    def set_hyperparameter_config(self, config: HyperparameterConfig) -> None:
        """Set hyperparameter configuration."""
        self.hyperparameter_configs[config.parameter_name] = config
    
    def optimize_ensemble(self,
                         ensemble_data: Dict[str, Any],
                         method: str = "bayesian",
                         force_optimization: bool = False) -> OptimizationResult:
        """
        Optimize ensemble configuration and weights.
        
        Args:
            ensemble_data: Current ensemble data and performance
            method: Optimization method to use
            force_optimization: Force optimization even if not due
            
        Returns:
            Optimization result
        """
        timestamp = datetime.now()
        
        # Check if optimization is needed
        if (not force_optimization and 
            self.last_optimization and 
            (timestamp - self.last_optimization).total_seconds() < self.optimization_frequency * 60):
            
            # Return last optimization result
            if self.optimization_results:
                last_result = self.optimization_results[-1]
                return OptimizationResult(
                    timestamp=timestamp,
                    optimization_method=method,
                    objective_value=last_result.objective_value,
                    optimal_weights=last_result.optimal_weights,
                    optimal_parameters=last_result.optimal_parameters,
                    convergence_info={'reason': 'not_due_for_optimization'},
                    performance_improvement=0.0,
                    validation_score=last_result.validation_score
                )
        
        # Update baseline performance if needed
        if self.baseline_performance is None:
            self.baseline_performance = self._calculate_performance_profile(ensemble_data)
        
        # Select optimization method
        if method not in self.optimization_methods:
            method = "bayesian"  # Default fallback
        
        optimization_func = self.optimization_methods[method]
        
        # Run optimization
        try:
            result = optimization_func(ensemble_data, timestamp)
            
            # Validate result
            validation_score = self._validate_optimization_result(result, ensemble_data)
            result.validation_score = validation_score
            
            # Calculate performance improvement
            current_performance = self._calculate_performance_profile(ensemble_data)
            if self.baseline_performance:
                improvement = current_performance.composite_score - self.baseline_performance.composite_score
                result.performance_improvement = improvement
                
                # Update best performance
                if (self.best_performance is None or 
                    current_performance.composite_score > self.best_performance.composite_score):
                    self.best_performance = current_performance
            
            # Store result
            self.optimization_results.append(result)
            self.last_optimization = timestamp
            
            print(f"Optimization completed using {method}: improvement = {result.performance_improvement:.4f}")
            
            return result
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            # Return fallback result
            return OptimizationResult(
                timestamp=timestamp,
                optimization_method=method,
                objective_value=0.0,
                optimal_weights={},
                optimal_parameters={},
                convergence_info={'error': str(e)},
                performance_improvement=0.0,
                validation_score=0.0
            )
    
    def _gradient_descent_optimization(self,
                                     ensemble_data: Dict[str, Any],
                                     timestamp: datetime) -> OptimizationResult:
        """Gradient descent optimization."""
        # Extract current weights and parameters
        current_weights = ensemble_data.get('model_weights', {})
        current_params = ensemble_data.get('parameters', {})
        
        # Initialize optimization variables
        weight_vector = np.array(list(current_weights.values()))
        param_vector = np.array([current_params.get(p.parameter_name, p.current_value) 
                                for p in self.hyperparameter_configs.values()])
        
        # Optimization hyperparameters
        learning_rate = 0.01
        max_iterations = 100
        
        best_objective = float('-inf')
        best_weights = weight_vector.copy()
        best_params = param_vector.copy()
        
        for iteration in range(max_iterations):
            # Calculate objective function value
            objective_value = self._evaluate_objective_function(
                weight_vector, param_vector, ensemble_data
            )
            
            if objective_value > best_objective:
                best_objective = objective_value
                best_weights = weight_vector.copy()
                best_params = param_vector.copy()
            
            # Calculate gradients (numerical approximation)
            weight_gradients = self._calculate_weight_gradients(weight_vector, param_vector, ensemble_data)
            param_gradients = self._calculate_param_gradients(weight_vector, param_vector, ensemble_data)
            
            # Update variables
            weight_vector += learning_rate * weight_gradients
            param_vector += learning_rate * param_gradients
            
            # Apply constraints
            weight_vector = self._apply_weight_constraints(weight_vector)
            param_vector = self._apply_parameter_constraints(param_vector)
            
            # Check convergence
            if (np.linalg.norm(weight_gradients) < self.convergence_tolerance and
                np.linalg.norm(param_gradients) < self.convergence_tolerance):
                break
        
        # Convert back to dictionaries
        optimal_weights = {model_id: float(w) for model_id, w in 
                          zip(current_weights.keys(), best_weights)}
        
        optimal_parameters = {param.parameter_name: float(p) for param, p in 
                             zip(self.hyperparameter_configs.values(), best_params)}
        
        return OptimizationResult(
            timestamp=timestamp,
            optimization_method="gradient_descent",
            objective_value=best_objective,
            optimal_weights=optimal_weights,
            optimal_parameters=optimal_parameters,
            convergence_info={
                "iterations": iteration + 1,
                "converged": iteration < max_iterations - 1,
                "final_gradient_norm": float(np.linalg.norm(weight_gradients))
            },
            performance_improvement=0.0,  # Will be calculated later
            validation_score=0.0  # Will be calculated later
        )
    
    def _bayesian_optimization(self,
                             ensemble_data: Dict[str, Any],
                             timestamp: datetime) -> OptimizationResult:
        """Bayesian optimization (simplified implementation)."""
        # For a full implementation, would use Gaussian Process
        # This is a simplified version using random sampling with learning
        
        current_weights = ensemble_data.get('model_weights', {})
        current_params = ensemble_data.get('parameters', {})
        
        n_iterations = 50
        n_random_init = 10
        
        # Sample space definition
        weight_bounds = [(0.01, 1.0) for _ in current_weights]
        param_bounds = []
        
        for param_config in self.hyperparameter_configs.values():
            if param_config.parameter_type == "continuous":
                param_bounds.append(param_config.bounds)
            elif param_config.parameter_type == "discrete":
                param_bounds.append(param_config.bounds)
            else:  # categorical
                param_bounds.append((0, len(param_config.bounds) - 1))
        
        # Track best results
        best_objective = float('-inf')
        best_weights = None
        best_params = None
        
        # Sample collection
        weight_samples = []
        param_samples = []
        objective_values = []
        
        for iteration in range(n_iterations):
            if iteration < n_random_init:
                # Random initialization
                weight_sample = np.array([np.random.uniform(low, high) for low, high in weight_bounds])
                param_sample = np.array([np.random.uniform(low, high) for low, high in param_bounds])
            else:
                # Acquisition function (simplified - use best + noise)
                if best_weights is not None and best_params is not None:
                    noise_scale = 0.1 * (1.0 - iteration / n_iterations)  # Decreasing noise
                    weight_sample = best_weights + np.random.normal(0, noise_scale, len(best_weights))
                    param_sample = best_params + np.random.normal(0, noise_scale, len(best_params))
                    
                    # Ensure bounds
                    weight_sample = np.clip(weight_sample, [b[0] for b in weight_bounds], [b[1] for b in weight_bounds])
                    param_sample = np.clip(param_sample, [b[0] for b in param_bounds], [b[1] for b in param_bounds])
                else:
                    weight_sample = np.array([np.random.uniform(low, high) for low, high in weight_bounds])
                    param_sample = np.array([np.random.uniform(low, high) for low, high in param_bounds])
            
            # Apply constraints
            weight_sample = self._apply_weight_constraints(weight_sample)
            param_sample = self._apply_parameter_constraints(param_sample)
            
            # Evaluate objective
            objective_value = self._evaluate_objective_function(weight_sample, param_sample, ensemble_data)
            
            # Store samples
            weight_samples.append(weight_sample)
            param_samples.append(param_sample)
            objective_values.append(objective_value)
            
            # Update best
            if objective_value > best_objective:
                best_objective = objective_value
                best_weights = weight_sample.copy()
                best_params = param_sample.copy()
        
        # Convert results
        optimal_weights = {model_id: float(w) for model_id, w in 
                          zip(current_weights.keys(), best_weights)}
        
        optimal_parameters = {}
        for i, (param_name, param_config) in enumerate(self.hyperparameter_configs.items()):
            if param_config.parameter_type == "categorical":
                param_idx = int(round(best_params[i]))
                param_idx = max(0, min(param_idx, len(param_config.bounds) - 1))
                optimal_parameters[param_name] = param_config.bounds[param_idx]
            else:
                optimal_parameters[param_name] = float(best_params[i])
        
        return OptimizationResult(
            timestamp=timestamp,
            optimization_method="bayesian",
            objective_value=best_objective,
            optimal_weights=optimal_weights,
            optimal_parameters=optimal_parameters,
            convergence_info={
                "iterations": n_iterations,
                "best_iteration": np.argmax(objective_values),
                "improvement_over_random": best_objective - np.mean(objective_values[:n_random_init])
            },
            performance_improvement=0.0,
            validation_score=0.0
        )
    
    def _genetic_algorithm_optimization(self,
                                      ensemble_data: Dict[str, Any],
                                      timestamp: datetime) -> OptimizationResult:
        """Genetic algorithm optimization."""
        current_weights = ensemble_data.get('model_weights', {})
        current_params = ensemble_data.get('parameters', {})
        
        # GA parameters
        population_size = 50
        n_generations = 30
        mutation_rate = 0.1
        crossover_rate = 0.8
        elite_size = 5
        
        # Initialize population
        population = []
        
        for _ in range(population_size):
            # Random individual
            weights = np.random.dirichlet(np.ones(len(current_weights)))  # Sum to 1
            params = []
            
            for param_config in self.hyperparameter_configs.values():
                if param_config.parameter_type == "continuous":
                    param_val = np.random.uniform(*param_config.bounds)
                elif param_config.parameter_type == "discrete":
                    param_val = np.random.randint(*param_config.bounds)
                else:  # categorical
                    param_val = np.random.choice(len(param_config.bounds))
                
                params.append(param_val)
            
            params = np.array(params)
            individual = np.concatenate([weights, params])
            population.append(individual)
        
        population = np.array(population)
        n_weights = len(current_weights)
        
        best_individual = None
        best_fitness = float('-inf')
        
        for generation in range(n_generations):
            # Evaluate fitness
            fitness_scores = []
            
            for individual in population:
                weights = individual[:n_weights]
                params = individual[n_weights:]
                
                # Apply constraints
                weights = self._apply_weight_constraints(weights)
                params = self._apply_parameter_constraints(params)
                
                fitness = self._evaluate_objective_function(weights, params, ensemble_data)
                fitness_scores.append(fitness)
            
            fitness_scores = np.array(fitness_scores)
            
            # Track best
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > best_fitness:
                best_fitness = fitness_scores[best_idx]
                best_individual = population[best_idx].copy()
            
            # Selection (tournament selection)
            new_population = []
            
            # Keep elite
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            # Generate offspring
            while len(new_population) < population_size:
                # Parent selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                if np.random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if np.random.random() < mutation_rate:
                    child1 = self._mutate(child1, n_weights)
                if np.random.random() < mutation_rate:
                    child2 = self._mutate(child2, n_weights)
                
                new_population.extend([child1, child2])
            
            population = np.array(new_population[:population_size])
        
        # Extract best solution
        best_weights = best_individual[:n_weights]
        best_params = best_individual[n_weights:]
        
        optimal_weights = {model_id: float(w) for model_id, w in 
                          zip(current_weights.keys(), best_weights)}
        
        optimal_parameters = {}
        for i, (param_name, param_config) in enumerate(self.hyperparameter_configs.items()):
            if param_config.parameter_type == "categorical":
                param_idx = int(round(best_params[i]))
                param_idx = max(0, min(param_idx, len(param_config.bounds) - 1))
                optimal_parameters[param_name] = param_config.bounds[param_idx]
            else:
                optimal_parameters[param_name] = float(best_params[i])
        
        return OptimizationResult(
            timestamp=timestamp,
            optimization_method="genetic_algorithm",
            objective_value=best_fitness,
            optimal_weights=optimal_weights,
            optimal_parameters=optimal_parameters,
            convergence_info={
                "generations": n_generations,
                "population_size": population_size,
                "final_best_fitness": float(best_fitness)
            },
            performance_improvement=0.0,
            validation_score=0.0
        )
    
    def _particle_swarm_optimization(self,
                                   ensemble_data: Dict[str, Any],
                                   timestamp: datetime) -> OptimizationResult:
        """Particle Swarm Optimization."""
        current_weights = ensemble_data.get('model_weights', {})
        
        # PSO parameters
        n_particles = 30
        n_iterations = 50
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive parameter
        c2 = 1.5  # Social parameter
        
        # Problem dimensions
        n_weights = len(current_weights)
        n_params = len(self.hyperparameter_configs)
        n_dims = n_weights + n_params
        
        # Initialize particles
        particles = np.random.rand(n_particles, n_dims)
        velocities = np.random.rand(n_particles, n_dims) * 0.1
        
        # Initialize best positions
        personal_best = particles.copy()
        personal_best_fitness = np.full(n_particles, float('-inf'))
        
        global_best = None
        global_best_fitness = float('-inf')
        
        for iteration in range(n_iterations):
            for i in range(n_particles):
                # Extract weights and parameters
                weights = particles[i, :n_weights]
                params = particles[i, n_weights:]
                
                # Apply constraints
                weights = self._apply_weight_constraints(weights)
                params = self._apply_parameter_constraints(params)
                
                # Evaluate fitness
                fitness = self._evaluate_objective_function(weights, params, ensemble_data)
                
                # Update personal best
                if fitness > personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best[i] = particles[i].copy()
                
                # Update global best
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best = particles[i].copy()
            
            # Update velocities and positions
            for i in range(n_particles):
                r1, r2 = np.random.rand(2)
                
                cognitive = c1 * r1 * (personal_best[i] - particles[i])
                social = c2 * r2 * (global_best - particles[i])
                
                velocities[i] = w * velocities[i] + cognitive + social
                particles[i] += velocities[i]
                
                # Bound particles
                particles[i] = np.clip(particles[i], 0, 1)
        
        # Extract best solution
        best_weights = global_best[:n_weights]
        best_params = global_best[n_weights:]
        
        optimal_weights = {model_id: float(w) for model_id, w in 
                          zip(current_weights.keys(), best_weights)}
        
        optimal_parameters = {}
        for i, (param_name, param_config) in enumerate(self.hyperparameter_configs.items()):
            param_val = best_params[i]
            
            if param_config.parameter_type == "continuous":
                # Scale to actual bounds
                low, high = param_config.bounds
                optimal_parameters[param_name] = low + param_val * (high - low)
            elif param_config.parameter_type == "discrete":
                low, high = param_config.bounds
                optimal_parameters[param_name] = int(low + param_val * (high - low))
            else:  # categorical
                param_idx = int(param_val * len(param_config.bounds))
                param_idx = max(0, min(param_idx, len(param_config.bounds) - 1))
                optimal_parameters[param_name] = param_config.bounds[param_idx]
        
        return OptimizationResult(
            timestamp=timestamp,
            optimization_method="particle_swarm",
            objective_value=global_best_fitness,
            optimal_weights=optimal_weights,
            optimal_parameters=optimal_parameters,
            convergence_info={
                "iterations": n_iterations,
                "particles": n_particles,
                "final_best_fitness": float(global_best_fitness)
            },
            performance_improvement=0.0,
            validation_score=0.0
        )
    
    def _random_search_optimization(self,
                                  ensemble_data: Dict[str, Any],
                                  timestamp: datetime) -> OptimizationResult:
        """Random search optimization."""
        current_weights = ensemble_data.get('model_weights', {})
        
        n_samples = 100
        best_objective = float('-inf')
        best_weights = None
        best_params = None
        
        for _ in range(n_samples):
            # Random weights (Dirichlet distribution for sum-to-1 constraint)
            weights = np.random.dirichlet(np.ones(len(current_weights)))
            
            # Random parameters
            params = []
            for param_config in self.hyperparameter_configs.values():
                if param_config.parameter_type == "continuous":
                    param_val = np.random.uniform(*param_config.bounds)
                elif param_config.parameter_type == "discrete":
                    param_val = np.random.randint(*param_config.bounds)
                else:  # categorical
                    param_val = np.random.choice(len(param_config.bounds))
                
                params.append(param_val)
            
            params = np.array(params)
            
            # Evaluate objective
            objective_value = self._evaluate_objective_function(weights, params, ensemble_data)
            
            if objective_value > best_objective:
                best_objective = objective_value
                best_weights = weights.copy()
                best_params = params.copy()
        
        # Convert results
        optimal_weights = {model_id: float(w) for model_id, w in 
                          zip(current_weights.keys(), best_weights)}
        
        optimal_parameters = {}
        for i, (param_name, param_config) in enumerate(self.hyperparameter_configs.items()):
            if param_config.parameter_type == "categorical":
                param_idx = int(best_params[i])
                optimal_parameters[param_name] = param_config.bounds[param_idx]
            else:
                optimal_parameters[param_name] = float(best_params[i])
        
        return OptimizationResult(
            timestamp=timestamp,
            optimization_method="random_search",
            objective_value=best_objective,
            optimal_weights=optimal_weights,
            optimal_parameters=optimal_parameters,
            convergence_info={"samples": n_samples},
            performance_improvement=0.0,
            validation_score=0.0
        )
    
    def _grid_search_optimization(self,
                                ensemble_data: Dict[str, Any],
                                timestamp: datetime) -> OptimizationResult:
        """Grid search optimization."""
        # Simplified grid search (would be computationally expensive for many parameters)
        current_weights = ensemble_data.get('model_weights', {})
        
        # Create parameter grids
        weight_grid = np.linspace(0.1, 0.9, 5)  # 5 points per weight
        
        best_objective = float('-inf')
        best_weights = None
        best_params = None
        
        # Sample subset of grid for feasibility
        n_samples = 50
        
        for _ in range(n_samples):
            # Sample from grid
            weights = np.random.choice(weight_grid, len(current_weights))
            weights = weights / np.sum(weights)  # Normalize
            
            # Sample parameters from grid
            params = []
            for param_config in self.hyperparameter_configs.values():
                if param_config.parameter_type == "continuous":
                    low, high = param_config.bounds
                    grid_values = np.linspace(low, high, 5)
                    param_val = np.random.choice(grid_values)
                elif param_config.parameter_type == "discrete":
                    low, high = param_config.bounds
                    param_val = np.random.choice(range(low, high + 1))
                else:  # categorical
                    param_val = np.random.choice(len(param_config.bounds))
                
                params.append(param_val)
            
            params = np.array(params)
            
            # Evaluate objective
            objective_value = self._evaluate_objective_function(weights, params, ensemble_data)
            
            if objective_value > best_objective:
                best_objective = objective_value
                best_weights = weights.copy()
                best_params = params.copy()
        
        # Convert results
        optimal_weights = {model_id: float(w) for model_id, w in 
                          zip(current_weights.keys(), best_weights)}
        
        optimal_parameters = {}
        for i, (param_name, param_config) in enumerate(self.hyperparameter_configs.items()):
            if param_config.parameter_type == "categorical":
                param_idx = int(best_params[i])
                optimal_parameters[param_name] = param_config.bounds[param_idx]
            else:
                optimal_parameters[param_name] = float(best_params[i])
        
        return OptimizationResult(
            timestamp=timestamp,
            optimization_method="grid_search",
            objective_value=best_objective,
            optimal_weights=optimal_weights,
            optimal_parameters=optimal_parameters,
            convergence_info={"grid_samples": n_samples},
            performance_improvement=0.0,
            validation_score=0.0
        )
    
    def _evaluate_objective_function(self,
                                   weights: np.ndarray,
                                   params: np.ndarray,
                                   ensemble_data: Dict[str, Any]) -> float:
        """Evaluate multi-objective function."""
        total_objective = 0.0
        
        for target in self.current_targets:
            objective_value = self._evaluate_single_objective(
                target.objective, weights, params, ensemble_data
            )
            
            # Apply constraints
            if not self._check_objective_constraints(target, objective_value):
                return float('-inf')  # Infeasible solution
            
            total_objective += target.weight * objective_value
        
        return total_objective
    
    def _evaluate_single_objective(self,
                                 objective: str,
                                 weights: np.ndarray,
                                 params: np.ndarray,
                                 ensemble_data: Dict[str, Any]) -> float:
        """Evaluate single objective."""
        if objective == "maximize_accuracy":
            # Simulate accuracy based on weights and ensemble data
            base_accuracy = ensemble_data.get('base_accuracy', 0.6)
            weight_diversity = 1.0 - np.std(weights)  # Higher diversity = better
            return base_accuracy * weight_diversity
        
        elif objective == "minimize_risk":
            # Simulate risk metric
            weight_concentration = np.max(weights)  # Higher concentration = higher risk
            return 1.0 - weight_concentration
        
        elif objective == "maximize_efficiency":
            # Simulate efficiency based on computational complexity
            total_complexity = np.sum(weights * np.array([1.0, 2.0, 3.0][:len(weights)]))  # Simplified
            return 1.0 / (1.0 + total_complexity)
        
        elif objective == "maximize_stability":
            # Simulate stability based on weight distribution
            weight_entropy = -np.sum(weights * np.log(weights + 1e-10))
            max_entropy = np.log(len(weights))
            return weight_entropy / max_entropy if max_entropy > 0 else 0.0
        
        else:
            return 0.5  # Default value
    
    def _check_objective_constraints(self,
                                   target: OptimizationTarget,
                                   objective_value: float) -> bool:
        """Check if objective value meets constraints."""
        constraints = target.constraints
        
        if "min_value" in constraints and objective_value < constraints["min_value"]:
            return False
        
        if "max_value" in constraints and objective_value > constraints["max_value"]:
            return False
        
        return True
    
    def _apply_weight_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Apply constraints to weights."""
        # Ensure non-negative
        weights = np.maximum(weights, 0.01)
        
        # Normalize to sum to 1
        weights = weights / np.sum(weights)
        
        return weights
    
    def _apply_parameter_constraints(self, params: np.ndarray) -> np.ndarray:
        """Apply constraints to parameters."""
        constrained_params = params.copy()
        
        for i, (param_name, param_config) in enumerate(self.hyperparameter_configs.items()):
            if i < len(constrained_params):
                if param_config.parameter_type in ["continuous", "discrete"]:
                    low, high = param_config.bounds
                    constrained_params[i] = np.clip(constrained_params[i], low, high)
                else:  # categorical
                    constrained_params[i] = np.clip(constrained_params[i], 0, len(param_config.bounds) - 1)
        
        return constrained_params
    
    def _calculate_weight_gradients(self,
                                  weights: np.ndarray,
                                  params: np.ndarray,
                                  ensemble_data: Dict[str, Any]) -> np.ndarray:
        """Calculate gradients with respect to weights (numerical)."""
        epsilon = 1e-6
        gradients = np.zeros_like(weights)
        
        base_objective = self._evaluate_objective_function(weights, params, ensemble_data)
        
        for i in range(len(weights)):
            weights_plus = weights.copy()
            weights_plus[i] += epsilon
            weights_plus = self._apply_weight_constraints(weights_plus)
            
            objective_plus = self._evaluate_objective_function(weights_plus, params, ensemble_data)
            gradients[i] = (objective_plus - base_objective) / epsilon
        
        return gradients
    
    def _calculate_param_gradients(self,
                                 weights: np.ndarray,
                                 params: np.ndarray,
                                 ensemble_data: Dict[str, Any]) -> np.ndarray:
        """Calculate gradients with respect to parameters (numerical)."""
        epsilon = 1e-6
        gradients = np.zeros_like(params)
        
        base_objective = self._evaluate_objective_function(weights, params, ensemble_data)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            params_plus = self._apply_parameter_constraints(params_plus)
            
            objective_plus = self._evaluate_objective_function(weights, params_plus, ensemble_data)
            gradients[i] = (objective_plus - base_objective) / epsilon
        
        return gradients
    
    def _tournament_selection(self,
                            population: np.ndarray,
                            fitness_scores: np.ndarray,
                            tournament_size: int = 3) -> np.ndarray:
        """Tournament selection for genetic algorithm."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = fitness_scores[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover(self,
                  parent1: np.ndarray,
                  parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crossover operation for genetic algorithm."""
        crossover_point = np.random.randint(1, len(parent1))
        
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
        return child1, child2
    
    def _mutate(self,
               individual: np.ndarray,
               n_weights: int,
               mutation_strength: float = 0.1) -> np.ndarray:
        """Mutation operation for genetic algorithm."""
        mutated = individual.copy()
        
        # Mutate weights
        for i in range(n_weights):
            if np.random.random() < 0.1:  # 10% mutation probability per gene
                mutated[i] += np.random.normal(0, mutation_strength)
        
        # Mutate parameters
        for i in range(n_weights, len(individual)):
            if np.random.random() < 0.1:
                mutated[i] += np.random.normal(0, mutation_strength)
        
        return mutated
    
    def _calculate_performance_profile(self, ensemble_data: Dict[str, Any]) -> PerformanceProfile:
        """Calculate comprehensive performance profile."""
        timestamp = datetime.now()
        
        # Mock performance calculations (would use real data)
        accuracy_metrics = {
            'prediction_accuracy': ensemble_data.get('accuracy', 0.6),
            'directional_accuracy': ensemble_data.get('directional_accuracy', 0.65),
            'confidence_accuracy': ensemble_data.get('confidence_accuracy', 0.55)
        }
        
        risk_metrics = {
            'volatility': ensemble_data.get('volatility', 0.02),
            'max_drawdown': ensemble_data.get('max_drawdown', 0.1),
            'var_95': ensemble_data.get('var_95', 0.05)
        }
        
        efficiency_metrics = {
            'computational_efficiency': ensemble_data.get('computational_efficiency', 0.8),
            'memory_efficiency': ensemble_data.get('memory_efficiency', 0.7),
            'prediction_speed': ensemble_data.get('prediction_speed', 0.9)
        }
        
        stability_metrics = {
            'weight_stability': ensemble_data.get('weight_stability', 0.8),
            'prediction_consistency': ensemble_data.get('prediction_consistency', 0.7),
            'model_agreement': ensemble_data.get('model_agreement', 0.75)
        }
        
        # Calculate composite score
        composite_score = (
            0.4 * np.mean(list(accuracy_metrics.values())) +
            0.3 * (1.0 - np.mean(list(risk_metrics.values()))) +  # Lower risk is better
            0.2 * np.mean(list(efficiency_metrics.values())) +
            0.1 * np.mean(list(stability_metrics.values()))
        )
        
        return PerformanceProfile(
            timestamp=timestamp,
            accuracy_metrics=accuracy_metrics,
            risk_metrics=risk_metrics,
            efficiency_metrics=efficiency_metrics,
            stability_metrics=stability_metrics,
            composite_score=composite_score
        )
    
    def _validate_optimization_result(self,
                                    result: OptimizationResult,
                                    ensemble_data: Dict[str, Any]) -> float:
        """Validate optimization result using cross-validation."""
        # Simplified validation - would implement proper cross-validation
        
        # Check if weights are reasonable
        weights = list(result.optimal_weights.values())
        if any(w < 0 or w > 1 for w in weights):
            return 0.0
        
        if abs(sum(weights) - 1.0) > 0.01:  # Should sum to 1
            return 0.0
        
        # Check parameter bounds
        for param_name, param_value in result.optimal_parameters.items():
            if param_name in self.hyperparameter_configs:
                config = self.hyperparameter_configs[param_name]
                
                if config.parameter_type in ["continuous", "discrete"]:
                    low, high = config.bounds
                    if param_value < low or param_value > high:
                        return 0.5  # Partial validation failure
        
        # If optimization improved objective, give higher validation score
        if result.objective_value > 0:
            return min(1.0, result.objective_value)
        else:
            return 0.3
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary."""
        summary = {
            'optimization_frequency_minutes': self.optimization_frequency,
            'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None,
            'total_optimizations': len(self.optimization_results),
            'optimization_methods_available': list(self.optimization_methods.keys())
        }
        
        # Current targets
        summary['current_targets'] = [
            {
                'objective': target.objective,
                'weight': target.weight,
                'target_value': target.target_value
            }
            for target in self.current_targets
        ]
        
        # Hyperparameter configurations
        summary['hyperparameters'] = {
            param_name: {
                'type': config.parameter_type,
                'bounds': config.bounds,
                'current_value': config.current_value,
                'importance': config.importance
            }
            for param_name, config in self.hyperparameter_configs.items()
        }
        
        # Recent optimization results
        if self.optimization_results:
            recent_results = list(self.optimization_results)[-5:]
            summary['recent_optimizations'] = [
                {
                    'timestamp': result.timestamp.isoformat(),
                    'method': result.optimization_method,
                    'objective_value': result.objective_value,
                    'performance_improvement': result.performance_improvement,
                    'validation_score': result.validation_score
                }
                for result in recent_results
            ]
        
        # Performance tracking
        if self.best_performance:
            summary['best_performance'] = {
                'timestamp': self.best_performance.timestamp.isoformat(),
                'composite_score': self.best_performance.composite_score,
                'accuracy_metrics': self.best_performance.accuracy_metrics,
                'risk_metrics': self.best_performance.risk_metrics
            }
        
        return summary