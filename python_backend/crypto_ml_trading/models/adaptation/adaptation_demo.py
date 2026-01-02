"""
Real-time Model Adaptation Demo.

Demonstrates the complete adaptation system with drift detection,
online learning, and ensemble adaptation.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from real_time_adapter import (
    RealTimeAdaptationSystem, 
    KolmogorovSmirnovDriftDetector,
    AdaptiveDriftDetector
)
from ensemble_adapter import AdaptiveEnsemble, DynamicModelSelector
from online_learning import (
    OnlineLinearRegression,
    OnlineNeuralNetwork,
    OnlineGradientBoostingRegressor
)


def generate_synthetic_data(n_samples: int, n_features: int = 10, 
                          regime: str = 'normal') -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic market data with different regimes."""
    X = np.random.randn(n_samples, n_features)
    
    # Base signal
    signal = np.sum(X[:, :3], axis=1)
    
    if regime == 'normal':
        # Normal market conditions
        noise = np.random.randn(n_samples) * 0.5
        y = signal + noise
        
    elif regime == 'trending':
        # Trending market
        trend = np.linspace(0, 5, n_samples)
        noise = np.random.randn(n_samples) * 0.3
        y = signal + trend + noise
        
    elif regime == 'volatile':
        # High volatility
        noise = np.random.randn(n_samples) * 2.0
        volatility_clusters = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 2
        y = signal + noise + volatility_clusters
        
    elif regime == 'regime_change':
        # Sudden regime change
        y = np.zeros(n_samples)
        y[:n_samples//2] = signal[:n_samples//2] + np.random.randn(n_samples//2) * 0.5
        y[n_samples//2:] = -signal[n_samples//2:] + np.random.randn(n_samples//2) * 0.5
        
    else:
        y = signal + np.random.randn(n_samples) * 0.5
    
    return X, y


def demonstrate_online_learning():
    """Demonstrate online learning algorithms."""
    print("=" * 80)
    print("ONLINE LEARNING DEMONSTRATION")
    print("=" * 80)
    
    # Create online learners
    learners = {
        'Linear': OnlineLinearRegression(learning_rate=0.01, adaptive_lr=True),
        'Neural': OnlineNeuralNetwork(hidden_size=32, learning_rate=0.001),
        'GradientBoosting': OnlineGradientBoostingRegressor(n_estimators=50)
    }
    
    # Generate data stream
    n_batches = 50
    batch_size = 100
    
    performance_history = {name: [] for name in learners}
    
    print("\nTraining online learners...")
    
    for batch in range(n_batches):
        # Generate batch with possible regime change
        if batch < 20:
            regime = 'normal'
        elif batch < 35:
            regime = 'trending'
        else:
            regime = 'volatile'
        
        X_batch, y_batch = generate_synthetic_data(batch_size, regime=regime)
        
        # Train and evaluate each learner
        for name, learner in learners.items():
            # Train
            learner.partial_fit(X_batch[:80], y_batch[:80])
            
            # Test
            try:
                predictions = learner.predict(X_batch[80:])
                mse = np.mean((predictions - y_batch[80:])**2)
                performance_history[name].append(mse)
            except:
                performance_history[name].append(np.nan)
        
        if batch % 10 == 0:
            print(f"Batch {batch}: Regime = {regime}")
            for name in learners:
                if performance_history[name]:
                    recent_mse = np.nanmean(performance_history[name][-10:])
                    print(f"  {name}: MSE = {recent_mse:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for name, history in performance_history.items():
        plt.plot(history, label=name)
    
    plt.axvline(x=20, color='r', linestyle='--', alpha=0.5, label='Regime change 1')
    plt.axvline(x=35, color='r', linestyle='--', alpha=0.5, label='Regime change 2')
    plt.xlabel('Batch')
    plt.ylabel('MSE')
    plt.title('Online Learning Performance')
    plt.legend()
    plt.show()
    
    # Print final parameters
    print("\nFinal Model Parameters:")
    for name, learner in learners.items():
        print(f"\n{name}:")
        params = learner.get_params()
        for key, value in params.items():
            if isinstance(value, list) and len(value) > 5:
                print(f"  {key}: [array of length {len(value)}]")
            else:
                print(f"  {key}: {value}")


def demonstrate_drift_detection():
    """Demonstrate drift detection."""
    print("\n" + "=" * 80)
    print("DRIFT DETECTION DEMONSTRATION")
    print("=" * 80)
    
    # Create drift detectors
    detectors = [
        KolmogorovSmirnovDriftDetector(significance_level=0.05),
        AdaptiveDriftDetector(window_size=500)
    ]
    
    # Generate reference data
    X_ref, _ = generate_synthetic_data(1000, regime='normal')
    
    print("\nTesting different scenarios...")
    
    # Test scenarios
    scenarios = [
        ('normal', 'No drift (same distribution)'),
        ('trending', 'Covariate drift (trending)'),
        ('volatile', 'Volatility change'),
        ('regime_change', 'Concept drift')
    ]
    
    for regime, description in scenarios:
        print(f"\n{description}:")
        
        X_current, _ = generate_synthetic_data(500, regime=regime)
        
        for detector in detectors:
            result = detector.detect_drift(X_ref, X_current)
            
            print(f"\n  {detector.get_detector_name()}:")
            print(f"    Drift detected: {result.drift_detected}")
            print(f"    Drift type: {result.drift_type}")
            print(f"    Drift score: {result.drift_score:.4f}")
            print(f"    Confidence: {result.confidence:.2%}")
            if result.affected_features:
                print(f"    Affected features: {len(result.affected_features)}")


def demonstrate_ensemble_adaptation():
    """Demonstrate adaptive ensemble."""
    print("\n" + "=" * 80)
    print("ADAPTIVE ENSEMBLE DEMONSTRATION")
    print("=" * 80)
    
    # Create base models
    base_models = {
        'linear_1': OnlineLinearRegression(learning_rate=0.01),
        'linear_2': OnlineLinearRegression(learning_rate=0.05, regularization='l1'),
        'neural_1': OnlineNeuralNetwork(hidden_size=16),
        'neural_2': OnlineNeuralNetwork(hidden_size=64),
        'gb': OnlineGradientBoostingRegressor(n_estimators=20)
    }
    
    # Create adaptive ensemble
    ensemble = AdaptiveEnsemble(
        models=base_models,
        window_size=500,
        diversity_weight=0.2
    )
    
    print("\nTraining adaptive ensemble...")
    
    # Training loop
    n_batches = 30
    weight_history = {name: [] for name in base_models}
    ensemble_performance = []
    
    for batch in range(n_batches):
        # Vary data characteristics
        if batch < 10:
            X, y = generate_synthetic_data(100, regime='normal')
        elif batch < 20:
            X, y = generate_synthetic_data(100, regime='trending')
        else:
            X, y = generate_synthetic_data(100, regime='volatile')
        
        # Train base models
        for model in base_models.values():
            model.partial_fit(X[:80], y[:80])
        
        # Make ensemble prediction
        ensemble_pred = ensemble.predict(X[80:])
        ensemble_mse = np.mean((ensemble_pred - y[80:])**2)
        ensemble_performance.append(ensemble_mse)
        
        # Update weights
        weights = ensemble.update_weights(X[80:], y[80:], immediate=True)
        
        # Record weights
        for name in base_models:
            weight_history[name].append(weights[name])
        
        if batch % 5 == 0:
            print(f"\nBatch {batch}:")
            print(f"  Ensemble MSE: {ensemble_mse:.4f}")
            print("  Model weights:")
            for name, weight in weights.items():
                print(f"    {name}: {weight:.3f}")
    
    # Plot weight evolution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    for name, history in weight_history.items():
        plt.plot(history, label=name)
    plt.axvline(x=10, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=20, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Batch')
    plt.ylabel('Weight')
    plt.title('Ensemble Weight Evolution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(ensemble_performance)
    plt.axvline(x=10, color='r', linestyle='--', alpha=0.5, label='Regime change')
    plt.axvline(x=20, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Batch')
    plt.ylabel('MSE')
    plt.title('Ensemble Performance')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print final report
    report = ensemble.get_ensemble_report()
    print("\nFinal Ensemble Report:")
    print(f"  Active models: {report['n_active']}/{report['n_models']}")
    print("  Final weights:")
    for name, weight in report['weights'].items():
        print(f"    {name}: {weight:.3f}")
    
    if 'ensemble_metrics' in report:
        print("  Ensemble metrics:")
        for metric, value in report['ensemble_metrics'].items():
            print(f"    {metric}: {value:.3f}")


def demonstrate_real_time_adaptation():
    """Demonstrate complete real-time adaptation system."""
    print("\n" + "=" * 80)
    print("REAL-TIME ADAPTATION SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Create models
    models = {
        'baseline': OnlineLinearRegression(learning_rate=0.01),
        'adaptive': OnlineLinearRegression(learning_rate=0.05, adaptive_lr=True),
        'neural': OnlineNeuralNetwork(hidden_size=32)
    }
    
    # Create adaptation system
    adaptation_system = RealTimeAdaptationSystem(
        models=models,
        adaptation_interval=50,
        performance_threshold=0.6
    )
    
    print("\nStarting real-time adaptation...")
    adaptation_system.start_adaptation()
    
    # Simulate real-time data stream
    n_iterations = 20
    batch_size = 50
    
    for i in range(n_iterations):
        # Generate data with drift
        if i < 7:
            X, y = generate_synthetic_data(batch_size, regime='normal')
        elif i < 14:
            X, y = generate_synthetic_data(batch_size, regime='regime_change')
        else:
            X, y = generate_synthetic_data(batch_size, regime='volatile')
        
        # Add data to adaptation system
        adaptation_system.add_data(X, y)
        
        # Small delay to simulate real-time
        time.sleep(0.1)
        
        if i % 5 == 0:
            print(f"\nIteration {i}:")
            print(f"  Total samples: {adaptation_system.sample_count}")
            
            # Get adapted models and test
            for name in models:
                adapted_model = adaptation_system.get_model(name)
                if adapted_model and hasattr(adapted_model, 'predict'):
                    try:
                        # Test on new data
                        X_test, y_test = generate_synthetic_data(20)
                        predictions = adapted_model.predict(X_test)
                        mse = np.mean((predictions - y_test)**2)
                        print(f"  {name} MSE: {mse:.4f}")
                    except:
                        pass
    
    # Stop adaptation
    adaptation_system.stop_adaptation()
    
    # Get final report
    final_report = adaptation_system.get_adaptation_report()
    
    print("\n" + "=" * 50)
    print("FINAL ADAPTATION REPORT")
    print("=" * 50)
    
    print(f"\nTotal samples processed: {final_report['total_samples']}")
    
    for model_name, model_report in final_report['models'].items():
        print(f"\n{model_name.upper()}:")
        print(f"  Current performance: {model_report.get('current_performance', 'N/A')}")
        print(f"  Best performance: {model_report.get('best_performance', 'N/A')}")
        print(f"  Drift rate: {model_report.get('drift_rate', 0):.2%}")
        print(f"  Total adaptations: {model_report.get('total_adaptations', 0)}")
        
        if model_report.get('recent_drifts'):
            print("  Recent drifts:")
            for drift in model_report['recent_drifts'][-3:]:
                print(f"    - {drift['drift_type']} (score: {drift['score']:.3f})")
    
    # Save adapted models
    import os
    save_dir = 'adapted_models'
    os.makedirs(save_dir, exist_ok=True)
    adaptation_system.save_adapted_models(save_dir)
    print(f"\nAdapted models saved to {save_dir}/")


def demonstrate_dynamic_selection():
    """Demonstrate dynamic model selection."""
    print("\n" + "=" * 80)
    print("DYNAMIC MODEL SELECTION DEMONSTRATION")
    print("=" * 80)
    
    # Create model pool
    model_pool = {
        'trend_follower': OnlineLinearRegression(learning_rate=0.1),
        'mean_reverter': OnlineLinearRegression(learning_rate=0.01, regularization='l2'),
        'pattern_detector': OnlineNeuralNetwork(hidden_size=64),
        'volatility_adapter': OnlineNeuralNetwork(hidden_size=32, dropout_rate=0.2),
        'ensemble_tree': OnlineGradientBoostingRegressor(n_estimators=30)
    }
    
    # Create dynamic selector
    selector = DynamicModelSelector(
        model_pool=model_pool,
        regime_window=200,
        switch_threshold=0.7
    )
    
    print("\nSimulating market regimes...")
    
    # Simulate different market regimes
    regime_sequence = [
        ('normal', 200),
        ('trending', 200),
        ('volatile', 200),
        ('normal', 200)
    ]
    
    regime_history = []
    active_models_history = []
    
    for regime, duration in regime_sequence:
        print(f"\nMarket regime: {regime}")
        
        for _ in range(duration // 20):
            # Generate data
            X, y = generate_synthetic_data(20, regime=regime)
            
            # Update selector
            active_models = selector.update_models(X, y)
            
            # Make predictions
            predictions = selector.predict(X)
            
            # Record history
            regime_history.append(selector.current_regime)
            active_models_history.append(list(active_models.keys()))
        
        # Print active models
        print(f"  Active models: {list(active_models.keys())}")
    
    # Get final report
    selector_report = selector.get_selector_report()
    
    print("\n" + "=" * 50)
    print("DYNAMIC SELECTION REPORT")
    print("=" * 50)
    
    print(f"\nCurrent regime: {selector_report['current_regime']}")
    print(f"Regime confidence: {selector_report['regime_confidence']:.2%}")
    print(f"Active models: {selector_report['active_models']}")
    
    if 'regime_summary' in selector_report:
        print("\nRegime Performance Summary:")
        for regime, summary in selector_report['regime_summary'].items():
            print(f"\n{regime}:")
            print(f"  Observations: {summary['n_observations']}")
            print(f"  Best model: {summary['best_model']}")
            print(f"  Avg performance: {summary['avg_performance']:.3f}")


def main():
    """Run all demonstrations."""
    print("REAL-TIME MODEL ADAPTATION DEMONSTRATION")
    print("=" * 80)
    
    demonstrations = [
        ("Online Learning", demonstrate_online_learning),
        ("Drift Detection", demonstrate_drift_detection),
        ("Ensemble Adaptation", demonstrate_ensemble_adaptation),
        ("Real-time Adaptation", demonstrate_real_time_adaptation),
        ("Dynamic Selection", demonstrate_dynamic_selection)
    ]
    
    for name, demo_func in demonstrations:
        print(f"\n{'='*80}")
        print(f"Running: {name}")
        print('='*80)
        
        try:
            demo_func()
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()
        
        print("\nPress Enter to continue...")
        input()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Import for type hints
    from typing import Tuple
    
    # Run demonstration
    main()