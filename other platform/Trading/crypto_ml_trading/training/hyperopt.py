"""
Hyperparameter Optimizer
Optuna-based hyperparameter optimization for ML models.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Type
import logging
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from crypto_ml_trading.training.config import (
    TrainingConfig,
    HYPERPARAMETER_SPACES,
    get_hyperparameter_space
)
from crypto_ml_trading.training.walk_forward import WalkForwardValidator

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    optimization_history: List[Dict]
    study_summary: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': self.n_trials,
            'optimization_history': self.optimization_history[:20],  # Limit for JSON
            'study_summary': self.study_summary
        }


class HyperparameterOptimizer:
    """Optuna-based hyperparameter optimization."""

    def __init__(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Optional[TrainingConfig] = None,
        model_class: Optional[Type] = None,
        custom_search_space: Optional[Dict] = None
    ):
        """
        Initialize hyperparameter optimizer.

        Args:
            model_type: Type of model to optimize
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            config: Training configuration
            model_class: Model class to instantiate
            custom_search_space: Custom search space (overrides defaults)
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter optimization. Install with: pip install optuna")

        self.model_type = model_type
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.config = config or TrainingConfig()
        self.model_class = model_class

        # Get search space
        self.search_space = custom_search_space or get_hyperparameter_space(model_type)
        if not self.search_space:
            raise ValueError(f"No search space defined for model type: {model_type}")

        # For walk-forward validation if enabled
        self.walk_forward = WalkForwardValidator(n_folds=3)  # Use 3 folds for speed

    def optimize(
        self,
        n_trials: Optional[int] = None,
        timeout_hours: Optional[float] = None,
        objective_metric: str = 'accuracy',
        use_walk_forward: bool = False,
        callback: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Number of trials (uses config default if None)
            timeout_hours: Timeout in hours (uses config default if None)
            objective_metric: Metric to optimize ('accuracy', 'f1', 'sharpe')
            use_walk_forward: Whether to use walk-forward validation
            callback: Optional callback function(study, trial)

        Returns:
            OptimizationResult with best parameters
        """
        n_trials = n_trials or self.config.optuna_trials
        timeout = (timeout_hours or self.config.optuna_timeout_hours) * 3600

        logger.info(
            f"Starting hyperparameter optimization for {self.model_type}: "
            f"{n_trials} trials, {timeout/3600:.1f}h timeout"
        )

        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        # Create objective function
        def objective(trial: optuna.Trial) -> float:
            params = self._sample_params(trial)

            if use_walk_forward:
                return self._objective_walk_forward(params, objective_metric)
            else:
                return self._objective_simple(params, objective_metric)

        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[callback] if callback else None,
            show_progress_bar=True
        )

        # Collect results
        optimization_history = [
            {
                'trial': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state)
            }
            for t in study.trials
        ]

        # Study summary
        study_summary = {
            'best_trial': study.best_trial.number,
            'n_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'n_failed': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
            'duration_minutes': sum(
                (t.datetime_complete - t.datetime_start).total_seconds() / 60
                for t in study.trials
                if t.datetime_complete and t.datetime_start
            )
        }

        result = OptimizationResult(
            best_params=study.best_params,
            best_value=study.best_value,
            n_trials=len(study.trials),
            optimization_history=optimization_history,
            study_summary=study_summary
        )

        logger.info(
            f"Optimization complete: best_value={result.best_value:.4f}, "
            f"completed={study_summary['n_completed']}, pruned={study_summary['n_pruned']}"
        )
        logger.info(f"Best parameters: {result.best_params}")

        return result

    def _sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample parameters from search space.

        Args:
            trial: Optuna trial

        Returns:
            Dictionary of sampled parameters
        """
        params = {}

        for param_name, values in self.search_space.items():
            if isinstance(values, list):
                if all(isinstance(v, (int, float)) for v in values):
                    # Numeric values - use suggest_categorical or suggest_float
                    if all(isinstance(v, int) for v in values):
                        params[param_name] = trial.suggest_categorical(param_name, values)
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, values)
                elif all(isinstance(v, list) for v in values):
                    # List of lists (e.g., hidden_sizes)
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        [str(v) for v in values]  # Convert to string for Optuna
                    )
                    # Convert back to list
                    params[param_name] = eval(params[param_name])
                elif all(isinstance(v, tuple) for v in values):
                    # List of tuples
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        [str(v) for v in values]
                    )
                    params[param_name] = eval(params[param_name])
                else:
                    # Mixed or string values
                    params[param_name] = trial.suggest_categorical(param_name, values)
            elif isinstance(values, tuple) and len(values) == 2:
                # Range (min, max)
                if isinstance(values[0], int):
                    params[param_name] = trial.suggest_int(param_name, values[0], values[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, values[0], values[1])
            else:
                # Single value
                params[param_name] = values

        return params

    def _objective_simple(
        self,
        params: Dict[str, Any],
        objective_metric: str
    ) -> float:
        """
        Simple objective using validation set.

        Args:
            params: Model parameters
            objective_metric: Metric to optimize

        Returns:
            Objective value
        """
        try:
            # Create and train model
            if self.model_class:
                model = self.model_class(**params)
                model.fit(self.X_train, self.y_train)

                # Evaluate
                y_pred = model.predict(self.X_val)

                if objective_metric == 'accuracy':
                    return float(np.mean(y_pred == self.y_val))
                elif objective_metric == 'f1':
                    from sklearn.metrics import f1_score
                    return float(f1_score(self.y_val, y_pred, average='weighted'))
                else:
                    return float(np.mean(y_pred == self.y_val))
            else:
                # No model class provided, return dummy value
                logger.warning("No model class provided for optimization")
                return 0.5

        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0

    def _objective_walk_forward(
        self,
        params: Dict[str, Any],
        objective_metric: str
    ) -> float:
        """
        Objective using walk-forward validation.

        Args:
            params: Model parameters
            objective_metric: Metric to optimize

        Returns:
            Mean objective value across folds
        """
        if not self.model_class:
            logger.warning("No model class provided for optimization")
            return 0.5

        try:
            # Combine train and val for walk-forward
            X_combined = np.concatenate([self.X_train, self.X_val])
            y_combined = np.concatenate([self.y_train, self.y_val])

            result = self.walk_forward.validate(
                model_class=self.model_class,
                X=X_combined,
                y=y_combined,
                params=params
            )

            if objective_metric == 'accuracy':
                return result.mean_accuracy
            elif objective_metric == 'sharpe':
                return result.mean_sharpe
            else:
                return result.mean_accuracy

        except Exception as e:
            logger.warning(f"Walk-forward trial failed: {e}")
            return 0.0


def optimize_hyperparameters(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_class: Optional[Type] = None,
    n_trials: int = 50,
    timeout_hours: float = 4.0
) -> OptimizationResult:
    """
    Convenience function for hyperparameter optimization.

    Args:
        model_type: Type of model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_class: Model class
        n_trials: Number of trials
        timeout_hours: Timeout in hours

    Returns:
        OptimizationResult
    """
    optimizer = HyperparameterOptimizer(
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_class=model_class
    )

    return optimizer.optimize(
        n_trials=n_trials,
        timeout_hours=timeout_hours
    )
