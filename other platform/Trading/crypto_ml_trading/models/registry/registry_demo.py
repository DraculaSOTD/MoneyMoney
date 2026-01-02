"""
Demonstration of the Model Registry and Version Control System.

Shows how to use all components together for comprehensive model management.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import json
from pathlib import Path

# Import registry components
from model_registry import ModelRegistry, ModelStatus, ModelType
from version_control import ModelVersionControl
from model_catalog import ModelCatalog, ModelFilter, ModelRanking
from experiment_tracker import ExperimentTracker


class MockModel:
    """Mock model for demonstration."""
    def __init__(self, model_type: str, params: dict):
        self.model_type = model_type
        self.params = params
        self.weights = np.random.randn(100, 50)
    
    def predict(self, X):
        return np.random.randn(len(X))


def demonstrate_model_registry():
    """Demonstrate model registry functionality."""
    print("\n" + "="*80)
    print("MODEL REGISTRY DEMONSTRATION")
    print("="*80)
    
    # Initialize registry
    registry = ModelRegistry(registry_path="demo_registry")
    
    # Register multiple models
    models_data = [
        {
            "name": "LSTM_Predictor_v1",
            "type": ModelType.DEEP_LEARNING,
            "description": "LSTM model for price prediction",
            "architecture": {"layers": 3, "units": 128, "dropout": 0.2},
            "performance": {"sharpe_ratio": 1.8, "accuracy": 0.65, "max_drawdown": -0.12}
        },
        {
            "name": "DQN_Trader",
            "type": ModelType.REINFORCEMENT_LEARNING,
            "description": "Deep Q-Network for trading decisions",
            "architecture": {"hidden_layers": [256, 128], "learning_rate": 0.001},
            "performance": {"sharpe_ratio": 2.1, "accuracy": 0.58, "max_drawdown": -0.18}
        },
        {
            "name": "Ensemble_Meta",
            "type": ModelType.ENSEMBLE,
            "description": "Meta-ensemble combining multiple strategies",
            "architecture": {"n_models": 5, "combination": "weighted"},
            "performance": {"sharpe_ratio": 2.5, "accuracy": 0.68, "max_drawdown": -0.10}
        }
    ]
    
    registered_models = []
    
    for model_data in models_data:
        # Create mock model
        model = MockModel(model_data["type"].value, model_data["architecture"])
        
        # Register model
        model_id = registry.register_model(
            model=model,
            model_name=model_data["name"],
            model_type=model_data["type"],
            author="demo_user",
            description=model_data["description"],
            architecture=model_data["architecture"],
            hyperparameters={"batch_size": 32, "epochs": 100},
            training_config={"optimizer": "adam", "loss": "mse"},
            performance_metrics=model_data["performance"],
            validation_metrics={"val_loss": 0.05, "val_accuracy": 0.63},
            tags=["production_ready", "high_frequency"],
            asset_classes=["crypto", "forex"],
            time_horizons=["5m", "15m", "1h"]
        )
        
        registered_models.append(model_id)
        print(f"\nRegistered model: {model_data['name']} (ID: {model_id})")
    
    # Search models
    print("\n--- Searching for Deep Learning Models ---")
    dl_models = registry.search_models(
        model_type=ModelType.DEEP_LEARNING,
        status=ModelStatus.DEVELOPMENT
    )
    for model in dl_models:
        print(f"Found: {model.model_name} - Sharpe: {model.performance_metrics.get('sharpe_ratio', 0):.2f}")
    
    # Update model status
    print(f"\n--- Promoting {registered_models[0]} to STAGING ---")
    registry.update_model_status(registered_models[0], ModelStatus.STAGING)
    
    # Compare models
    print("\n--- Model Comparison ---")
    comparison_df = registry.compare_models(registered_models[:2])
    print(comparison_df[['model_name', 'model_type', 'sharpe_ratio', 'accuracy']].to_string())
    
    # Export report
    report_path = "demo_registry/registry_report.json"
    registry.export_registry_report(report_path)
    print(f"\nRegistry report exported to: {report_path}")
    
    return registry, registered_models


def demonstrate_version_control():
    """Demonstrate version control functionality."""
    print("\n" + "="*80)
    print("VERSION CONTROL DEMONSTRATION")
    print("="*80)
    
    # Initialize version control
    vcs = ModelVersionControl(vcs_path="demo_vcs")
    
    # Create initial model
    model = MockModel("deep_learning", {"layers": 2})
    
    # First commit
    print("\n--- Creating Initial Commit ---")
    commit1 = vcs.commit(
        model=model,
        model_id="demo_model_001",
        author="developer1",
        message="Initial model implementation",
        parameters={"learning_rate": 0.001, "batch_size": 32},
        architecture={"layers": 2, "units": 64},
        performance={"accuracy": 0.60, "loss": 0.40}
    )
    print(f"Commit created: {commit1}")
    
    # Create a branch for experimentation
    print("\n--- Creating Feature Branch ---")
    vcs.create_branch(
        branch_name="feature/improved_architecture",
        description="Testing deeper architecture",
        author="developer2"
    )
    
    # Switch to new branch
    vcs.checkout("feature/improved_architecture")
    
    # Make changes and commit
    model.params["layers"] = 3
    model.weights = np.random.randn(150, 50)  # Simulate weight changes
    
    commit2 = vcs.commit(
        model=model,
        model_id="demo_model_001",
        author="developer2",
        message="Increased model depth to 3 layers",
        parameters={"learning_rate": 0.0005, "batch_size": 64},
        architecture={"layers": 3, "units": 128},
        performance={"accuracy": 0.65, "loss": 0.35},
        tags=["experimental"]
    )
    
    # Create diff
    print("\n--- Diff Between Commits ---")
    diff = vcs.diff(commit1, commit2)
    print(f"Summary: {diff.summary}")
    print(f"Performance changes: {diff.perf_changes}")
    
    # Tag a version
    vcs.tag("v1.0.0", commit2)
    print(f"\nTagged commit {commit2} as v1.0.0")
    
    # Get history
    print("\n--- Commit History ---")
    history = vcs.get_history(limit=5)
    for commit in history:
        print(f"{commit.commit_id[:8]} - {commit.message} ({commit.author})")
    
    # Performance history
    print("\n--- Performance History ---")
    perf_df = vcs.get_performance_history("accuracy")
    print(perf_df[['commit_id', 'timestamp', 'accuracy']].to_string())
    
    return vcs


def demonstrate_model_catalog():
    """Demonstrate model catalog functionality."""
    print("\n" + "="*80)
    print("MODEL CATALOG DEMONSTRATION")
    print("="*80)
    
    # Use the registry from previous demo
    registry, model_ids = demonstrate_model_registry()
    
    # Initialize catalog
    catalog = ModelCatalog(registry)
    
    # Search with filters
    print("\n--- Advanced Search ---")
    filters = ModelFilter(
        model_types=[ModelType.DEEP_LEARNING, ModelType.ENSEMBLE],
        min_performance={"sharpe_ratio": 1.5},
        tags=["production_ready"]
    )
    
    search_results = catalog.search(
        query="prediction",
        filters=filters,
        limit=10
    )
    
    print(f"Found {len(search_results)} models matching criteria:")
    for model in search_results:
        print(f"  - {model.model_name}: Sharpe {model.performance_metrics.get('sharpe_ratio', 0):.2f}")
    
    # Get recommendations
    print("\n--- Model Recommendations ---")
    recommendations = catalog.recommend_models(
        use_case="high frequency crypto trading",
        requirements={
            "min_sharpe": 1.5,
            "max_drawdown": -0.20,
            "assets": ["crypto"],
            "production_only": False
        },
        limit=3
    )
    
    for rec in recommendations:
        print(f"\n{rec.model_name} (Score: {rec.score})")
        print(f"  Pros: {', '.join(rec.pros)}")
        print(f"  Cons: {', '.join(rec.cons) if rec.cons else 'None'}")
        print(f"  Reasons: {', '.join(rec.reasons)}")
    
    # Ensemble suggestions
    print("\n--- Ensemble Suggestions ---")
    base_models = model_ids[:2]  # Use first two models
    
    suggestions = catalog.create_ensemble_suggestions(
        base_models=base_models,
        target_performance={"sharpe_ratio": 3.0, "accuracy": 0.70},
        max_models=5
    )
    
    for sug in suggestions:
        print(f"\nSuggested: {sug['model_name']}")
        print(f"  Score: {sug['score']}")
        print(f"  Improves: {', '.join(sug['improves'])}")
        print(f"  Adds diversity: {sug['adds_diversity']}")
    
    # Export catalog
    catalog.export_catalog("demo_registry/model_catalog.md", format="markdown")
    print("\nCatalog exported to: demo_registry/model_catalog.md")
    
    return catalog


def demonstrate_experiment_tracker():
    """Demonstrate experiment tracking functionality."""
    print("\n" + "="*80)
    print("EXPERIMENT TRACKER DEMONSTRATION")
    print("="*80)
    
    # Initialize tracker
    tracker = ExperimentTracker(tracker_path="demo_experiments")
    
    # Create experiment
    print("\n--- Creating Experiment ---")
    exp_id = tracker.create_experiment(
        experiment_name="LSTM_Hyperparameter_Search",
        model_type="LSTM",
        hyperparameters={
            "hidden_units": 128,
            "num_layers": 3,
            "dropout": 0.2,
            "learning_rate": 0.001
        },
        dataset_config={
            "asset": "BTC/USDT",
            "timeframe": "5m",
            "train_size": 10000,
            "val_size": 2000
        },
        training_config={
            "epochs": 50,
            "batch_size": 32,
            "optimizer": "adam"
        },
        evaluation_config={
            "metrics": ["accuracy", "sharpe_ratio", "max_drawdown"],
            "test_size": 2000
        },
        tags=["hyperparameter_search", "lstm", "crypto"],
        description="Grid search for optimal LSTM architecture"
    )
    print(f"Created experiment: {exp_id}")
    
    # Run multiple trials
    print("\n--- Running Experiment Trials ---")
    hyperparameter_grid = [
        {"hidden_units": 64, "dropout": 0.1},
        {"hidden_units": 128, "dropout": 0.2},
        {"hidden_units": 256, "dropout": 0.3}
    ]
    
    for i, hp_variant in enumerate(hyperparameter_grid):
        print(f"\nTrial {i+1}: {hp_variant}")
        
        # Start run
        run_id = tracker.start_run(
            experiment_id=exp_id,
            system_info={"gpu": "NVIDIA RTX 3080", "memory": "32GB"}
        )
        
        # Simulate training
        for epoch in range(5):  # Shortened for demo
            # Training metrics
            train_metrics = {
                "loss": 0.5 - epoch * 0.05 + np.random.uniform(-0.02, 0.02),
                "accuracy": 0.5 + epoch * 0.05 + np.random.uniform(-0.02, 0.02)
            }
            
            # Validation metrics
            val_metrics = {
                "val_loss": 0.48 - epoch * 0.04 + np.random.uniform(-0.03, 0.03),
                "val_accuracy": 0.52 + epoch * 0.04 + np.random.uniform(-0.03, 0.03)
            }
            
            tracker.log_metrics(run_id, train_metrics, val_metrics)
            time.sleep(0.1)  # Simulate training time
        
        # Final test metrics
        test_metrics = {
            "accuracy": 0.65 + i * 0.02 + np.random.uniform(-0.05, 0.05),
            "sharpe_ratio": 1.5 + i * 0.2 + np.random.uniform(-0.1, 0.1),
            "max_drawdown": -0.15 + i * 0.02 + np.random.uniform(-0.02, 0.02)
        }
        
        tracker.log_metrics(run_id, test_metrics=test_metrics)
        
        # End run
        tracker.end_run(
            run_id,
            status="completed",
            notes=f"Completed trial with {hp_variant}",
            model_checkpoint_path=f"models/lstm_trial_{i+1}.pkl"
        )
    
    # Analyze experiment
    print("\n--- Analyzing Experiment Results ---")
    results = tracker.analyze_experiment(exp_id)
    
    print(f"\nExperiment Summary:")
    print(f"  Total runs: {results.num_runs}")
    print(f"  Best run: {results.best_run_id}")
    print(f"\nMean metrics:")
    for metric, value in results.mean_metrics.items():
        std = results.std_metrics.get(metric, 0)
        print(f"  {metric}: {value:.3f} ± {std:.3f}")
    
    print(f"\nBest metrics:")
    for metric, value in results.best_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"\nRecommendations:")
    for rec in results.recommendations:
        print(f"  - {rec}")
    
    # Compare experiments (if we had multiple)
    print("\n--- Experiment Comparison ---")
    comparison_df = tracker.compare_experiments([exp_id])
    print(comparison_df[['experiment_name', 'model_type', 'num_runs']].to_string())
    
    # Plot history
    plot_path = "demo_experiments/experiment_history.png"
    tracker.plot_experiment_history(exp_id, save_path=plot_path)
    print(f"\nExperiment history plot saved to: {plot_path}")
    
    # Export results
    tracker.export_results([exp_id], "demo_experiments/results.json", format="json")
    print("Results exported to: demo_experiments/results.json")
    
    return tracker


def demonstrate_integration():
    """Demonstrate integrated workflow."""
    print("\n" + "="*80)
    print("INTEGRATED WORKFLOW DEMONSTRATION")
    print("="*80)
    
    # Initialize all components
    registry = ModelRegistry(registry_path="demo_integrated")
    vcs = ModelVersionControl(vcs_path="demo_integrated/vcs")
    catalog = ModelCatalog(registry)
    tracker = ExperimentTracker(tracker_path="demo_integrated/experiments")
    
    print("\n--- Step 1: Experiment and Track ---")
    # Create and run experiment
    exp_id = tracker.create_experiment(
        experiment_name="Integrated_Model_Development",
        model_type="ensemble",
        hyperparameters={"n_models": 3, "combination": "weighted"},
        dataset_config={"asset": "BTC/USDT", "timeframe": "15m"},
        training_config={"epochs": 10},
        evaluation_config={"metrics": ["sharpe_ratio", "accuracy"]}
    )
    
    run_id = tracker.start_run(exp_id)
    
    # Simulate training
    for epoch in range(3):
        tracker.log_metrics(
            run_id,
            train_metrics={"loss": 0.3 - epoch * 0.05},
            val_metrics={"val_loss": 0.35 - epoch * 0.04}
        )
    
    tracker.log_metrics(
        run_id,
        test_metrics={"sharpe_ratio": 2.3, "accuracy": 0.67}
    )
    
    # Create model
    model = MockModel("ensemble", {"n_models": 3})
    
    tracker.end_run(run_id, model_checkpoint_path="models/ensemble_best.pkl")
    
    print("\n--- Step 2: Register Model ---")
    # Register in model registry
    model_id = registry.register_model(
        model=model,
        model_name="Integrated_Ensemble_v1",
        model_type=ModelType.ENSEMBLE,
        author="integrated_demo",
        description="Ensemble from integrated workflow",
        architecture={"n_models": 3, "combination": "weighted"},
        hyperparameters=tracker.get_best_hyperparameters(exp_id),
        training_config={"epochs": 10},
        performance_metrics={"sharpe_ratio": 2.3, "accuracy": 0.67},
        validation_metrics={"val_accuracy": 0.65},
        experiment_id=exp_id
    )
    print(f"Registered model: {model_id}")
    
    print("\n--- Step 3: Version Control ---")
    # Create initial commit
    commit_id = vcs.commit(
        model=model,
        model_id=model_id,
        author="integrated_demo",
        message="Initial ensemble model from experiments",
        parameters={"n_models": 3},
        architecture={"combination": "weighted"},
        performance={"sharpe_ratio": 2.3, "accuracy": 0.67}
    )
    print(f"Created commit: {commit_id}")
    
    # Tag release
    vcs.tag("v1.0.0-beta", commit_id)
    
    print("\n--- Step 4: Catalog and Discover ---")
    # Search in catalog
    recommendations = catalog.recommend_models(
        use_case="ensemble trading",
        requirements={"min_sharpe": 2.0}
    )
    
    if recommendations:
        print(f"Found our model in recommendations: {recommendations[0].model_name}")
    
    print("\n--- Step 5: Promote to Production ---")
    # Update status
    registry.update_model_status(model_id, ModelStatus.STAGING)
    print("Model promoted to STAGING")
    
    # After validation...
    registry.update_model_status(model_id, ModelStatus.PRODUCTION)
    print("Model promoted to PRODUCTION")
    
    print("\n" + "="*80)
    print("INTEGRATED WORKFLOW COMPLETE")
    print("="*80)
    
    return registry, vcs, catalog, tracker


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("MODEL REGISTRY AND VERSION CONTROL SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Individual component demos
    demonstrate_model_registry()
    demonstrate_version_control()
    demonstrate_model_catalog()
    demonstrate_experiment_tracker()
    
    # Integrated workflow
    demonstrate_integration()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nAll components demonstrated successfully!")
    print("Check the demo directories for generated artifacts.")


if __name__ == "__main__":
    main()