"""
Experiment Tracking System for ML models.

Provides comprehensive experiment tracking, hyperparameter logging,
and results comparison capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import hashlib
from pathlib import Path
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    experiment_id: str
    experiment_name: str
    model_type: str
    hyperparameters: Dict[str, Any]
    dataset_config: Dict[str, Any]
    training_config: Dict[str, Any]
    evaluation_config: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    description: str = ""
    parent_experiment_id: Optional[str] = None


@dataclass
class ExperimentRun:
    """Single run within an experiment."""
    run_id: str
    experiment_id: str
    start_time: datetime
    end_time: Optional[datetime]
    status: str  # running, completed, failed, aborted
    
    # Metrics
    train_metrics: Dict[str, List[float]] = field(default_factory=dict)
    val_metrics: Dict[str, List[float]] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    
    # System info
    system_info: Dict[str, Any] = field(default_factory=dict)
    runtime_seconds: Optional[float] = None
    
    # Artifacts
    model_checkpoint_path: Optional[str] = None
    logs_path: Optional[str] = None
    plots_path: Optional[str] = None
    
    # Notes
    notes: str = ""
    error_message: Optional[str] = None


@dataclass
class ExperimentResult:
    """Aggregated results from multiple runs."""
    experiment_id: str
    num_runs: int
    best_run_id: str
    
    # Aggregated metrics
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    best_metrics: Dict[str, float]
    
    # Statistical tests
    statistical_significance: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Hyperparameter importance
    hyperparameter_importance: Dict[str, float]
    
    # Recommendations
    recommendations: List[str]


class ExperimentTracker:
    """
    Comprehensive experiment tracking system.
    
    Features:
    - Experiment configuration management
    - Run tracking and logging
    - Metric aggregation and comparison
    - Hyperparameter optimization tracking
    - Visualization generation
    - Statistical analysis
    """
    
    def __init__(self, tracker_path: str = "experiment_tracker"):
        """
        Initialize experiment tracker.
        
        Args:
            tracker_path: Base path for experiment storage
        """
        self.tracker_path = Path(tracker_path)
        self.tracker_path.mkdir(parents=True, exist_ok=True)
        
        # Storage paths
        self.experiments_path = self.tracker_path / "experiments"
        self.runs_path = self.tracker_path / "runs"
        self.artifacts_path = self.tracker_path / "artifacts"
        self.plots_path = self.tracker_path / "plots"
        
        # Create directories
        self.experiments_path.mkdir(exist_ok=True)
        self.runs_path.mkdir(exist_ok=True)
        self.artifacts_path.mkdir(exist_ok=True)
        self.plots_path.mkdir(exist_ok=True)
        
        # Load existing data
        self.experiments = self._load_experiments()
        self.runs = self._load_runs()
        
        # Active tracking
        self.active_runs = {}
        
        logger.info(f"Experiment tracker initialized at {self.tracker_path}")
    
    def _load_experiments(self) -> Dict[str, ExperimentConfig]:
        """Load existing experiments."""
        experiments = {}
        
        for exp_file in self.experiments_path.glob("*.json"):
            try:
                with open(exp_file, 'r') as f:
                    exp_data = json.load(f)
                
                exp_config = ExperimentConfig(**exp_data)
                experiments[exp_config.experiment_id] = exp_config
                
            except Exception as e:
                logger.error(f"Failed to load experiment {exp_file}: {e}")
        
        return experiments
    
    def _load_runs(self) -> Dict[str, List[ExperimentRun]]:
        """Load existing runs grouped by experiment."""
        runs = defaultdict(list)
        
        for run_file in self.runs_path.glob("*.json"):
            try:
                with open(run_file, 'r') as f:
                    run_data = json.load(f)
                
                # Convert timestamps
                run_data['start_time'] = datetime.fromisoformat(run_data['start_time'])
                if run_data.get('end_time'):
                    run_data['end_time'] = datetime.fromisoformat(run_data['end_time'])
                
                run = ExperimentRun(**run_data)
                runs[run.experiment_id].append(run)
                
            except Exception as e:
                logger.error(f"Failed to load run {run_file}: {e}")
        
        return dict(runs)
    
    def create_experiment(self,
                         experiment_name: str,
                         model_type: str,
                         hyperparameters: Dict[str, Any],
                         dataset_config: Dict[str, Any],
                         training_config: Dict[str, Any],
                         evaluation_config: Dict[str, Any],
                         tags: Optional[List[str]] = None,
                         description: str = "",
                         parent_experiment_id: Optional[str] = None) -> str:
        """
        Create a new experiment.
        
        Args:
            experiment_name: Name of the experiment
            model_type: Type of model being tested
            hyperparameters: Model hyperparameters
            dataset_config: Dataset configuration
            training_config: Training configuration
            evaluation_config: Evaluation configuration
            tags: Optional tags
            description: Experiment description
            parent_experiment_id: Parent experiment for variants
            
        Returns:
            Experiment ID
        """
        # Generate experiment ID
        exp_id = self._generate_experiment_id(experiment_name, model_type)
        
        # Create experiment config
        exp_config = ExperimentConfig(
            experiment_id=exp_id,
            experiment_name=experiment_name,
            model_type=model_type,
            hyperparameters=hyperparameters,
            dataset_config=dataset_config,
            training_config=training_config,
            evaluation_config=evaluation_config,
            tags=tags or [],
            description=description,
            parent_experiment_id=parent_experiment_id
        )
        
        # Save experiment
        self._save_experiment(exp_config)
        self.experiments[exp_id] = exp_config
        
        logger.info(f"Created experiment {experiment_name} with ID {exp_id}")
        
        return exp_id
    
    def _generate_experiment_id(self, name: str, model_type: str) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{name}_{model_type}_{timestamp}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"exp_{timestamp}_{hash_suffix}"
    
    def _save_experiment(self, exp_config: ExperimentConfig):
        """Save experiment configuration."""
        exp_file = self.experiments_path / f"{exp_config.experiment_id}.json"
        
        exp_dict = exp_config.__dict__.copy()
        
        with open(exp_file, 'w') as f:
            json.dump(exp_dict, f, indent=2)
    
    def start_run(self, experiment_id: str, 
                  system_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new experiment run.
        
        Args:
            experiment_id: ID of the experiment
            system_info: System information (GPU, memory, etc.)
            
        Returns:
            Run ID
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Generate run ID
        run_id = self._generate_run_id(experiment_id)
        
        # Create run
        run = ExperimentRun(
            run_id=run_id,
            experiment_id=experiment_id,
            start_time=datetime.now(),
            end_time=None,
            status="running",
            system_info=system_info or {}
        )
        
        # Track active run
        self.active_runs[run_id] = run
        
        # Initialize run storage
        if experiment_id not in self.runs:
            self.runs[experiment_id] = []
        
        self.runs[experiment_id].append(run)
        
        logger.info(f"Started run {run_id} for experiment {experiment_id}")
        
        return run_id
    
    def _generate_run_id(self, experiment_id: str) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_number = len(self.runs.get(experiment_id, [])) + 1
        return f"{experiment_id}_run_{run_number}_{timestamp}"
    
    def log_metrics(self, run_id: str, 
                   train_metrics: Optional[Dict[str, float]] = None,
                   val_metrics: Optional[Dict[str, float]] = None,
                   test_metrics: Optional[Dict[str, float]] = None):
        """
        Log metrics for a run.
        
        Args:
            run_id: Run ID
            train_metrics: Training metrics for current epoch/step
            val_metrics: Validation metrics
            test_metrics: Test metrics (usually logged once)
        """
        if run_id not in self.active_runs:
            logger.warning(f"Run {run_id} not found in active runs")
            return
        
        run = self.active_runs[run_id]
        
        # Append to metric lists
        if train_metrics:
            for metric, value in train_metrics.items():
                if metric not in run.train_metrics:
                    run.train_metrics[metric] = []
                run.train_metrics[metric].append(value)
        
        if val_metrics:
            for metric, value in val_metrics.items():
                if metric not in run.val_metrics:
                    run.val_metrics[metric] = []
                run.val_metrics[metric].append(value)
        
        if test_metrics:
            run.test_metrics.update(test_metrics)
    
    def end_run(self, run_id: str, 
                status: str = "completed",
                notes: str = "",
                error_message: Optional[str] = None,
                model_checkpoint_path: Optional[str] = None):
        """
        End an experiment run.
        
        Args:
            run_id: Run ID
            status: Final status (completed, failed, aborted)
            notes: Optional notes
            error_message: Error message if failed
            model_checkpoint_path: Path to saved model
        """
        if run_id not in self.active_runs:
            logger.warning(f"Run {run_id} not found in active runs")
            return
        
        run = self.active_runs[run_id]
        
        # Update run
        run.end_time = datetime.now()
        run.status = status
        run.notes = notes
        run.error_message = error_message
        run.model_checkpoint_path = model_checkpoint_path
        
        # Calculate runtime
        if run.start_time and run.end_time:
            run.runtime_seconds = (run.end_time - run.start_time).total_seconds()
        
        # Save run
        self._save_run(run)
        
        # Remove from active runs
        del self.active_runs[run_id]
        
        logger.info(f"Ended run {run_id} with status {status}")
    
    def _save_run(self, run: ExperimentRun):
        """Save run data."""
        run_file = self.runs_path / f"{run.run_id}.json"
        
        run_dict = run.__dict__.copy()
        
        # Convert timestamps
        run_dict['start_time'] = run.start_time.isoformat()
        if run.end_time:
            run_dict['end_time'] = run.end_time.isoformat()
        
        with open(run_file, 'w') as f:
            json.dump(run_dict, f, indent=2)
    
    def compare_experiments(self, 
                           experiment_ids: List[str],
                           metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: Specific metrics to compare
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for exp_id in experiment_ids:
            if exp_id not in self.experiments:
                continue
            
            exp_config = self.experiments[exp_id]
            exp_runs = self.runs.get(exp_id, [])
            
            if not exp_runs:
                continue
            
            # Get best run
            completed_runs = [r for r in exp_runs if r.status == "completed"]
            if not completed_runs:
                continue
            
            # Find best run by primary metric
            primary_metric = metrics[0] if metrics else "accuracy"
            best_run = max(
                completed_runs,
                key=lambda r: r.test_metrics.get(primary_metric, 0)
            )
            
            # Compile comparison data
            row = {
                'experiment_id': exp_id,
                'experiment_name': exp_config.experiment_name,
                'model_type': exp_config.model_type,
                'num_runs': len(exp_runs),
                'best_run_id': best_run.run_id
            }
            
            # Add hyperparameters
            for hp_name, hp_value in exp_config.hyperparameters.items():
                row[f"hp_{hp_name}"] = hp_value
            
            # Add metrics
            for metric_name, metric_value in best_run.test_metrics.items():
                if metrics is None or metric_name in metrics:
                    row[metric_name] = metric_value
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def analyze_experiment(self, experiment_id: str) -> ExperimentResult:
        """
        Analyze all runs of an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Aggregated experiment results
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp_runs = self.runs.get(experiment_id, [])
        completed_runs = [r for r in exp_runs if r.status == "completed"]
        
        if not completed_runs:
            raise ValueError(f"No completed runs for experiment {experiment_id}")
        
        # Aggregate metrics
        all_metrics = defaultdict(list)
        
        for run in completed_runs:
            for metric, value in run.test_metrics.items():
                all_metrics[metric].append(value)
        
        # Calculate statistics
        mean_metrics = {
            metric: np.mean(values) 
            for metric, values in all_metrics.items()
        }
        
        std_metrics = {
            metric: np.std(values) 
            for metric, values in all_metrics.items()
        }
        
        # Find best run
        primary_metric = list(all_metrics.keys())[0] if all_metrics else "accuracy"
        best_run = max(
            completed_runs,
            key=lambda r: r.test_metrics.get(primary_metric, 0)
        )
        
        best_metrics = best_run.test_metrics
        
        # Confidence intervals (95%)
        confidence_intervals = {}
        for metric, values in all_metrics.items():
            if len(values) > 1:
                mean = np.mean(values)
                std_err = np.std(values) / np.sqrt(len(values))
                ci_lower = mean - 1.96 * std_err
                ci_upper = mean + 1.96 * std_err
                confidence_intervals[metric] = (ci_lower, ci_upper)
        
        # Hyperparameter importance (simplified)
        hyperparameter_importance = self._analyze_hyperparameter_importance(
            experiment_id, primary_metric
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            mean_metrics, std_metrics, completed_runs
        )
        
        return ExperimentResult(
            experiment_id=experiment_id,
            num_runs=len(completed_runs),
            best_run_id=best_run.run_id,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            best_metrics=best_metrics,
            statistical_significance={},  # Would require baseline
            confidence_intervals=confidence_intervals,
            hyperparameter_importance=hyperparameter_importance,
            recommendations=recommendations
        )
    
    def _analyze_hyperparameter_importance(self, 
                                         experiment_id: str,
                                         target_metric: str) -> Dict[str, float]:
        """Analyze hyperparameter importance."""
        # Simplified version - would use more sophisticated methods in production
        importance = {}
        
        exp_config = self.experiments[experiment_id]
        
        # For now, return uniform importance
        for hp_name in exp_config.hyperparameters:
            importance[hp_name] = 1.0 / len(exp_config.hyperparameters)
        
        return importance
    
    def _generate_recommendations(self,
                                mean_metrics: Dict[str, float],
                                std_metrics: Dict[str, float],
                                runs: List[ExperimentRun]) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        # Check for high variance
        for metric, std in std_metrics.items():
            mean = mean_metrics.get(metric, 1)
            cv = std / (abs(mean) + 1e-8)
            
            if cv > 0.2:
                recommendations.append(
                    f"High variance in {metric} (CV={cv:.2f}). "
                    f"Consider more runs or regularization."
                )
        
        # Check for consistent failures
        failed_runs = [r for r in runs if r.status == "failed"]
        if len(failed_runs) > len(runs) * 0.3:
            recommendations.append(
                f"High failure rate ({len(failed_runs)}/{len(runs)}). "
                f"Review error logs and system requirements."
            )
        
        # Check runtime
        runtimes = [r.runtime_seconds for r in runs if r.runtime_seconds]
        if runtimes and np.mean(runtimes) > 3600:
            recommendations.append(
                f"Long average runtime ({np.mean(runtimes)/60:.1f} min). "
                f"Consider optimization or smaller datasets for testing."
            )
        
        return recommendations
    
    def plot_experiment_history(self, 
                              experiment_id: str,
                              metrics: Optional[List[str]] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Plot training history for an experiment.
        
        Args:
            experiment_id: Experiment ID
            metrics: Specific metrics to plot
            save_path: Path to save plot
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp_runs = self.runs.get(experiment_id, [])
        completed_runs = [r for r in exp_runs if r.status == "completed"]
        
        if not completed_runs:
            logger.warning(f"No completed runs for experiment {experiment_id}")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot metrics for each run
        for i, (metric_type, ax) in enumerate(zip(['train', 'val'], axes[:2])):
            for run in completed_runs[:5]:  # Limit to 5 runs for clarity
                metric_dict = getattr(run, f"{metric_type}_metrics", {})
                
                if metrics:
                    plot_metrics = [m for m in metrics if m in metric_dict]
                else:
                    plot_metrics = list(metric_dict.keys())[:4]
                
                for metric in plot_metrics:
                    if metric in metric_dict:
                        values = metric_dict[metric]
                        ax.plot(values, label=f"{run.run_id[-8:]}_{metric}")
            
            ax.set_title(f"{metric_type.capitalize()} Metrics")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # Box plot of final metrics
        final_metrics = defaultdict(list)
        for run in completed_runs:
            for metric, value in run.test_metrics.items():
                if metrics is None or metric in metrics:
                    final_metrics[metric].append(value)
        
        if final_metrics:
            ax = axes[2]
            data_to_plot = [final_metrics[m] for m in final_metrics]
            ax.boxplot(data_to_plot, labels=list(final_metrics.keys()))
            ax.set_title("Test Metrics Distribution")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
        
        # Runtime distribution
        ax = axes[3]
        runtimes = [r.runtime_seconds/60 for r in completed_runs if r.runtime_seconds]
        if runtimes:
            ax.hist(runtimes, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title("Runtime Distribution")
            ax.set_xlabel("Runtime (minutes)")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f"Experiment: {self.experiments[experiment_id].experiment_name}")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def get_best_hyperparameters(self, 
                                experiment_id: str,
                                metric: str = "accuracy") -> Dict[str, Any]:
        """
        Get best hyperparameters from an experiment.
        
        Args:
            experiment_id: Experiment ID
            metric: Metric to optimize
            
        Returns:
            Best hyperparameters
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp_runs = self.runs.get(experiment_id, [])
        completed_runs = [r for r in exp_runs if r.status == "completed"]
        
        if not completed_runs:
            raise ValueError(f"No completed runs for experiment {experiment_id}")
        
        # Find best run
        best_run = max(
            completed_runs,
            key=lambda r: r.test_metrics.get(metric, 0)
        )
        
        # Return hyperparameters from experiment config
        return self.experiments[experiment_id].hyperparameters
    
    def export_results(self, 
                      experiment_ids: List[str],
                      output_path: str,
                      format: str = "csv"):
        """
        Export experiment results.
        
        Args:
            experiment_ids: List of experiment IDs
            output_path: Output file path
            format: Export format (csv, json)
        """
        if format == "csv":
            df = self.compare_experiments(experiment_ids)
            df.to_csv(output_path, index=False)
        
        elif format == "json":
            results = {}
            for exp_id in experiment_ids:
                if exp_id in self.experiments:
                    try:
                        result = self.analyze_experiment(exp_id)
                        results[exp_id] = {
                            'config': self.experiments[exp_id].__dict__,
                            'results': result.__dict__
                        }
                    except Exception as e:
                        logger.error(f"Failed to analyze {exp_id}: {e}")
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported results to {output_path}")