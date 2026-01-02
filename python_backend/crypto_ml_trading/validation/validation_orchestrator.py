"""
Central Validation Orchestrator for ML Trading Models.

Coordinates validation across all model types and provides a unified
interface for comprehensive model validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .data_splitter import TimeSeriesDataSplitter, PurgedKFold, WalkForwardSplitter
from .cross_validators import (
    PurgedTimeSeriesCrossValidator,
    WalkForwardCrossValidator,
    BlockingTimeSeriesCrossValidator
)
from .model_validators import (
    StatisticalModelValidator,
    NeuralNetworkValidator,
    ReinforcementLearningValidator,
    EnsembleValidator
)
from .metrics_calculator import ValidationMetricsCalculator
from .significance_tester import StatisticalSignificanceTester
from .production_validator import ProductionValidator


class ValidationOrchestrator:
    """
    Central orchestrator for model validation.
    
    Features:
    - Unified validation interface for all model types
    - Automatic validation strategy selection
    - Parallel validation execution
    - Comprehensive reporting
    - Model comparison and selection
    """
    
    def __init__(self,
                 results_dir: str = "validation_results",
                 n_jobs: int = -1,
                 random_state: int = 42):
        """
        Initialize validation orchestrator.
        
        Args:
            results_dir: Directory to store validation results
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed for reproducibility
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.random_state = random_state
        
        # Initialize validators
        self.statistical_validator = StatisticalModelValidator()
        self.nn_validator = NeuralNetworkValidator()
        self.rl_validator = ReinforcementLearningValidator()
        self.ensemble_validator = EnsembleValidator()
        
        # Initialize utilities
        self.metrics_calculator = ValidationMetricsCalculator()
        self.significance_tester = StatisticalSignificanceTester()
        self.production_validator = ProductionValidator()
        
        # Validation strategies
        self.validation_strategies = {
            'statistical': self._validate_statistical_model,
            'neural_network': self._validate_neural_network,
            'reinforcement_learning': self._validate_rl_model,
            'ensemble': self._validate_ensemble_model
        }
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.results_dir / "validation.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def validate_model(self,
                      model: Any,
                      model_type: str,
                      X: np.ndarray,
                      y: np.ndarray,
                      timestamps: Optional[pd.Series] = None,
                      validation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate a single model.
        
        Args:
            model: Model to validate
            model_type: Type of model ('statistical', 'neural_network', etc.)
            X: Feature data
            y: Target data
            timestamps: Optional timestamps for time series
            validation_config: Optional validation configuration
            
        Returns:
            Dictionary of validation results
        """
        self.logger.info(f"Starting validation for {model_type} model")
        
        # Default configuration
        config = {
            'cv_strategy': 'purged_time_series',
            'n_splits': 5,
            'test_size': 0.2,
            'purge_days': 2,
            'embargo_days': 1,
            'metrics': ['mae', 'rmse', 'sharpe', 'max_drawdown'],
            'significance_level': 0.05
        }
        
        if validation_config:
            config.update(validation_config)
        
        # Prepare data splits
        splits = self._prepare_data_splits(X, y, timestamps, config)
        
        # Run model-specific validation
        if model_type not in self.validation_strategies:
            raise ValueError(f"Unknown model type: {model_type}")
        
        validation_results = self.validation_strategies[model_type](
            model, splits, config
        )
        
        # Add metadata
        validation_results['metadata'] = {
            'model_type': model_type,
            'validation_timestamp': datetime.now().isoformat(),
            'data_shape': X.shape,
            'config': config
        }
        
        # Save results
        self._save_results(validation_results, model_type)
        
        return validation_results
        
    def validate_multiple_models(self,
                               models: Dict[str, Tuple[Any, str]],
                               X: np.ndarray,
                               y: np.ndarray,
                               timestamps: Optional[pd.Series] = None,
                               validation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate multiple models in parallel.
        
        Args:
            models: Dictionary of {name: (model, type)} pairs
            X: Feature data
            y: Target data
            timestamps: Optional timestamps
            validation_config: Validation configuration
            
        Returns:
            Dictionary of all validation results
        """
        self.logger.info(f"Starting parallel validation for {len(models)} models")
        
        all_results = {}
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit validation tasks
            future_to_model = {
                executor.submit(
                    self.validate_model,
                    model,
                    model_type,
                    X,
                    y,
                    timestamps,
                    validation_config
                ): name
                for name, (model, model_type) in models.items()
            }
            
            # Collect results
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    all_results[model_name] = result
                    self.logger.info(f"Completed validation for {model_name}")
                except Exception as e:
                    self.logger.error(f"Validation failed for {model_name}: {str(e)}")
                    all_results[model_name] = {'error': str(e)}
        
        # Compare models
        comparison_results = self._compare_models(all_results)
        
        return {
            'individual_results': all_results,
            'comparison': comparison_results,
            'best_model': comparison_results.get('rankings', {}).get('overall', [None])[0]
        }
        
    def _prepare_data_splits(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           timestamps: Optional[pd.Series],
                           config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data splits based on configuration."""
        cv_strategy = config['cv_strategy']
        
        # Train/test split
        splitter = TimeSeriesDataSplitter(
            test_size=config.get('test_size', 0.2),
            gap_size=config.get('gap_size', 0),
            purge_days=config.get('purge_days', 2)
        )
        
        if config.get('use_validation', True):
            X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y, timestamps)
        else:
            X_train, X_test, y_train, y_test = splitter.split_train_test(X, y, timestamps)
            X_val, y_val = None, None
        
        # Cross-validation splits
        cv_splits = []
        
        if cv_strategy == 'purged_time_series':
            cv = PurgedTimeSeriesCrossValidator(
                n_splits=config['n_splits'],
                purge_days=config['purge_days'],
                embargo_days=config.get('embargo_days', 1)
            )
        elif cv_strategy == 'walk_forward':
            cv = WalkForwardCrossValidator(
                n_splits=config['n_splits'],
                test_window=config.get('test_window', 100),
                train_window=config.get('train_window', 500)
            )
        elif cv_strategy == 'blocking':
            cv = BlockingTimeSeriesCrossValidator(
                n_splits=config['n_splits'],
                block_size=config.get('block_size', 100)
            )
        else:
            raise ValueError(f"Unknown CV strategy: {cv_strategy}")
        
        # Generate CV folds
        for fold in cv.split(X_train, y_train, timestamps):
            cv_splits.append(fold)
        
        return {
            'train': (X_train, y_train),
            'validation': (X_val, y_val) if X_val is not None else None,
            'test': (X_test, y_test),
            'cv_splits': cv_splits,
            'timestamps': timestamps
        }
        
    def _validate_statistical_model(self,
                                  model: Any,
                                  splits: Dict[str, Any],
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate statistical model."""
        results = {
            'cv_results': [],
            'test_results': {},
            'model_diagnostics': {}
        }
        
        # Cross-validation
        for fold in splits['cv_splits']:
            X_train_cv = splits['train'][0][fold.train_indices]
            y_train_cv = splits['train'][1][fold.train_indices]
            X_val_cv = splits['train'][0][fold.val_indices]
            y_val_cv = splits['train'][1][fold.val_indices]
            
            # Fit model on CV training data
            model_cv = model.copy() if hasattr(model, 'copy') else model
            model_cv.fit(X_train_cv, y_train_cv)
            
            # Validate
            if hasattr(model_cv, 'model_type') and 'ARIMA' in str(model_cv.model_type):
                fold_results = self.statistical_validator.validate_arima(
                    model_cv, X_train_cv, X_val_cv
                )
            elif hasattr(model_cv, 'model_type') and 'GARCH' in str(model_cv.model_type):
                fold_results = self.statistical_validator.validate_garch(
                    model_cv, y_train_cv, y_val_cv
                )
            else:
                # Generic statistical validation
                fold_results = self._generic_statistical_validation(
                    model_cv, X_train_cv, y_train_cv, X_val_cv, y_val_cv
                )
            
            results['cv_results'].append(fold_results)
        
        # Test set validation
        X_test, y_test = splits['test']
        test_predictions = model.predict(X_test)
        
        results['test_results'] = self.metrics_calculator.calculate_all_metrics(
            y_test, test_predictions, model_type='statistical'
        )
        
        # Model diagnostics
        results['model_diagnostics'] = self._get_model_diagnostics(model)
        
        return results
        
    def _validate_neural_network(self,
                               model: Any,
                               splits: Dict[str, Any],
                               config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate neural network model."""
        results = {
            'training_validation': {},
            'cv_results': [],
            'test_results': {},
            'prediction_analysis': {}
        }
        
        # Training validation (if history available)
        if hasattr(model, 'history') or hasattr(model, 'training_history'):
            history = getattr(model, 'history', getattr(model, 'training_history', {}))
            results['training_validation'] = self.nn_validator.validate_training(
                model, history, splits.get('validation', splits['test'])
            )
        
        # Cross-validation
        for fold in splits['cv_splits']:
            X_train_cv = splits['train'][0][fold.train_indices]
            y_train_cv = splits['train'][1][fold.train_indices]
            X_val_cv = splits['train'][0][fold.val_indices]
            y_val_cv = splits['train'][1][fold.val_indices]
            
            # Evaluate on fold
            fold_results = self.nn_validator.validate_predictions(
                model, X_val_cv, y_val_cv,
                task_type=config.get('task_type', 'classification')
            )
            
            results['cv_results'].append(fold_results)
        
        # Test set validation
        X_test, y_test = splits['test']
        results['test_results'] = self.nn_validator.validate_predictions(
            model, X_test, y_test,
            task_type=config.get('task_type', 'classification')
        )
        
        # Additional prediction analysis
        test_predictions = model.predict(X_test)
        results['prediction_analysis'] = self.metrics_calculator.calculate_all_metrics(
            y_test, test_predictions, model_type='neural_network'
        )
        
        return results
        
    def _validate_rl_model(self,
                         model: Any,
                         splits: Dict[str, Any],
                         config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate reinforcement learning model."""
        results = {
            'training_validation': {},
            'policy_validation': {},
            'test_performance': {}
        }
        
        # Training validation (if available)
        if hasattr(model, 'episode_rewards'):
            results['training_validation'] = self.rl_validator.validate_training(
                model.episode_rewards,
                model.episode_lengths,
                getattr(model, 'action_distribution', None)
            )
        
        # Test the policy
        X_test, y_test = splits['test']
        
        # Run test episodes
        test_episodes = []
        for i in range(config.get('n_test_episodes', 100)):
            # Simulate episode (simplified)
            episode_result = self._run_test_episode(model, X_test, y_test)
            test_episodes.append(episode_result)
        
        results['policy_validation'] = self.rl_validator.validate_policy(
            test_episodes,
            baseline_performance=config.get('baseline_performance', 0)
        )
        
        # Calculate test performance metrics
        episode_returns = [ep['total_return'] for ep in test_episodes]
        results['test_performance'] = {
            'mean_return': np.mean(episode_returns),
            'sharpe_ratio': np.mean(episode_returns) / (np.std(episode_returns) + 1e-10),
            'max_drawdown': np.min([ep.get('max_drawdown', 0) for ep in test_episodes]),
            'win_rate': np.mean([ep['total_return'] > 0 for ep in test_episodes])
        }
        
        return results
        
    def _validate_ensemble_model(self,
                               model: Any,
                               splits: Dict[str, Any],
                               config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ensemble model."""
        results = {
            'ensemble_validation': {},
            'cv_results': [],
            'test_results': {}
        }
        
        X_test, y_test = splits['test']
        
        # Get individual model predictions
        if hasattr(model, 'models_'):
            individual_predictions = {}
            for name, sub_model in model.models_.items():
                individual_predictions[name] = sub_model.predict(X_test)
        else:
            # Fallback for custom ensemble implementations
            individual_predictions = self._get_individual_predictions(model, X_test)
        
        # Get ensemble predictions
        ensemble_predictions = model.predict(X_test)
        
        # Validate ensemble
        results['ensemble_validation'] = self.ensemble_validator.validate_ensemble(
            individual_predictions,
            ensemble_predictions,
            y_test,
            model_weights=getattr(model, 'weights_', None)
        )
        
        # Cross-validation
        for fold in splits['cv_splits']:
            X_val_cv = splits['train'][0][fold.val_indices]
            y_val_cv = splits['train'][1][fold.val_indices]
            
            fold_predictions = model.predict(X_val_cv)
            fold_metrics = self.metrics_calculator.calculate_all_metrics(
                y_val_cv, fold_predictions, model_type='ensemble'
            )
            
            results['cv_results'].append(fold_metrics)
        
        # Test metrics
        results['test_results'] = self.metrics_calculator.calculate_all_metrics(
            y_test, ensemble_predictions, model_type='ensemble'
        )
        
        return results
        
    def _generic_statistical_validation(self,
                                      model: Any,
                                      X_train: np.ndarray,
                                      y_train: np.ndarray,
                                      X_val: np.ndarray,
                                      y_val: np.ndarray) -> Dict[str, Any]:
        """Generic validation for statistical models."""
        # Make predictions
        val_predictions = model.predict(X_val)
        
        # Calculate residuals
        residuals = y_val - val_predictions
        
        # Basic diagnostics
        return {
            'metrics': {
                'mae': np.mean(np.abs(residuals)),
                'rmse': np.sqrt(np.mean(residuals**2)),
                'mape': np.mean(np.abs(residuals / (y_val + 1e-10))) * 100
            },
            'residual_stats': {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'skewness': self._calculate_skewness(residuals),
                'kurtosis': self._calculate_kurtosis(residuals)
            }
        }
        
    def _get_model_diagnostics(self, model: Any) -> Dict[str, Any]:
        """Extract model diagnostics."""
        diagnostics = {}
        
        # Try to get common attributes
        if hasattr(model, 'feature_importances_'):
            diagnostics['feature_importances'] = model.feature_importances_.tolist()
        
        if hasattr(model, 'coef_'):
            diagnostics['coefficients'] = model.coef_.tolist()
        
        if hasattr(model, 'params'):
            diagnostics['parameters'] = str(model.params)
        
        return diagnostics
        
    def _run_test_episode(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Run a test episode for RL model (simplified)."""
        # This is a placeholder - actual implementation would depend on the RL environment
        n_steps = min(1000, len(X))
        
        rewards = []
        actions = []
        
        for i in range(n_steps):
            # Get action from model
            if hasattr(model, 'predict_action'):
                action = model.predict_action(X[i:i+1])
            else:
                action = 0  # Default action
            
            # Calculate reward (simplified)
            if i < len(y) - 1:
                reward = action * (y[i+1] - y[i])  # Simple return
            else:
                reward = 0
            
            rewards.append(reward)
            actions.append(action)
        
        cumulative_returns = np.cumsum(rewards)
        
        return {
            'total_reward': np.sum(rewards),
            'total_return': cumulative_returns[-1] if len(cumulative_returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(cumulative_returns),
            'n_steps': n_steps,
            'actions': actions
        }
        
    def _get_individual_predictions(self, ensemble_model: Any, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract predictions from individual models in ensemble."""
        predictions = {}
        
        # Try different ensemble structures
        if hasattr(ensemble_model, 'estimators_'):
            for i, estimator in enumerate(ensemble_model.estimators_):
                predictions[f'model_{i}'] = estimator.predict(X)
        elif hasattr(ensemble_model, 'base_models'):
            for name, model in ensemble_model.base_models.items():
                predictions[name] = model.predict(X)
        else:
            # Fallback - just use ensemble prediction
            predictions['ensemble'] = ensemble_model.predict(X)
        
        return predictions
        
    def _compare_models(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare validation results across models."""
        comparison = {
            'summary_statistics': {},
            'rankings': {},
            'statistical_tests': {}
        }
        
        # Extract key metrics for comparison
        metrics_data = {}
        
        for model_name, model_results in results.items():
            if 'error' in model_results:
                continue
                
            # Extract test metrics
            test_results = model_results.get('test_results', {})
            
            metrics_data[model_name] = {
                'mae': test_results.get('mae', np.inf),
                'rmse': test_results.get('rmse', np.inf),
                'sharpe_ratio': test_results.get('sharpe_ratio', -np.inf),
                'max_drawdown': test_results.get('max_drawdown', -np.inf)
            }
        
        if not metrics_data:
            return comparison
        
        # Calculate summary statistics
        for metric in ['mae', 'rmse', 'sharpe_ratio', 'max_drawdown']:
            values = [m[metric] for m in metrics_data.values()]
            comparison['summary_statistics'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # Rank models
        for metric in ['mae', 'rmse', 'sharpe_ratio', 'max_drawdown']:
            if metric in ['mae', 'rmse', 'max_drawdown']:
                # Lower is better
                ranked = sorted(metrics_data.items(), key=lambda x: x[1][metric])
            else:
                # Higher is better
                ranked = sorted(metrics_data.items(), key=lambda x: x[1][metric], reverse=True)
            
            comparison['rankings'][metric] = [name for name, _ in ranked]
        
        # Overall ranking (simple average rank)
        model_ranks = {}
        for model_name in metrics_data:
            ranks = []
            for metric, ranking in comparison['rankings'].items():
                if model_name in ranking:
                    ranks.append(ranking.index(model_name) + 1)
            model_ranks[model_name] = np.mean(ranks) if ranks else np.inf
        
        comparison['rankings']['overall'] = sorted(model_ranks.items(), key=lambda x: x[1])
        comparison['rankings']['overall'] = [name for name, _ in comparison['rankings']['overall']]
        
        # Statistical significance tests
        if len(metrics_data) > 1:
            model_names = list(metrics_data.keys())
            
            # Pairwise tests
            for i, model1 in enumerate(model_names):
                for model2 in model_names[i+1:]:
                    # Extract predictions for statistical tests
                    # (This is simplified - would need actual prediction arrays)
                    test_result = {
                        'models': (model1, model2),
                        'significant_difference': False,  # Placeholder
                        'p_value': 0.5  # Placeholder
                    }
                    
                    comparison['statistical_tests'][f'{model1}_vs_{model2}'] = test_result
        
        return comparison
        
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
        
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 3
        return np.mean(((data - mean) / std) ** 4)
        
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
        
    def _save_results(self, results: Dict[str, Any], model_type: str):
        """Save validation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"{model_type}_validation_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_serializable = convert_numpy(results)
        
        with open(filename, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        self.logger.info(f"Results saved to {filename}")
        
    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive validation report."""
        report_lines = [
            "=" * 80,
            "MODEL VALIDATION REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        # Individual model results
        if 'individual_results' in results:
            report_lines.extend([
                "INDIVIDUAL MODEL RESULTS",
                "-" * 40,
            ])
            
            for model_name, model_results in results['individual_results'].items():
                if 'error' in model_results:
                    report_lines.append(f"{model_name}: ERROR - {model_results['error']}")
                    continue
                
                test_results = model_results.get('test_results', {})
                report_lines.extend([
                    f"\n{model_name}:",
                    f"  MAE: {test_results.get('mae', 'N/A'):.4f}",
                    f"  RMSE: {test_results.get('rmse', 'N/A'):.4f}",
                    f"  Sharpe Ratio: {test_results.get('sharpe_ratio', 'N/A'):.4f}",
                    f"  Max Drawdown: {test_results.get('max_drawdown', 'N/A'):.4f}",
                ])
        
        # Model comparison
        if 'comparison' in results:
            comparison = results['comparison']
            report_lines.extend([
                "\n" + "=" * 40,
                "MODEL COMPARISON",
                "-" * 40,
            ])
            
            # Rankings
            if 'rankings' in comparison:
                report_lines.append("\nModel Rankings:")
                for metric, ranking in comparison['rankings'].items():
                    if ranking:
                        report_lines.append(f"\n{metric}:")
                        for i, model in enumerate(ranking[:5], 1):  # Top 5
                            report_lines.append(f"  {i}. {model}")
        
        # Best model
        if 'best_model' in results:
            report_lines.extend([
                "\n" + "=" * 40,
                f"BEST MODEL: {results['best_model']}",
                "=" * 40,
            ])
        
        report = "\n".join(report_lines)
        
        # Save report
        report_file = self.results_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        return report