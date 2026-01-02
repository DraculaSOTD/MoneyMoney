"""
Demonstration of the Comprehensive Model Validation System.

Shows how to use the validation framework with different model types
and validation strategies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import validation components
from validation import (
    ValidationOrchestrator,
    TimeSeriesDataSplitter,
    PurgedTimeSeriesCrossValidator,
    WalkForwardCrossValidator,
    StatisticalModelValidator,
    NeuralNetworkValidator,
    ProductionValidator
)

# Import some example models (you'll need to adjust based on your actual models)
from models.statistical.arima import ARIMAModel
from models.deep_learning.gru_attention import GRUAttentionModel
from models.ensemble.meta_learner import MetaLearner


def generate_sample_data(n_samples: int = 10000) -> tuple:
    """Generate sample financial time series data for demonstration."""
    # Create synthetic time series with trend and seasonality
    time = np.arange(n_samples)
    trend = 0.0001 * time
    seasonal = 0.1 * np.sin(2 * np.pi * time / 100)
    noise = 0.05 * np.random.randn(n_samples)
    
    # Price series
    price = 100 * np.exp(trend + seasonal + noise)
    
    # Create features
    features = np.column_stack([
        price,                                      # Price
        pd.Series(price).pct_change().fillna(0),   # Returns
        pd.Series(price).rolling(10).mean().fillna(price),  # MA10
        pd.Series(price).rolling(50).mean().fillna(price),  # MA50
        pd.Series(price).rolling(20).std().fillna(0.1),     # Volatility
        np.random.randn(n_samples)                 # Random feature
    ])
    
    # Create targets (next period return)
    returns = pd.Series(price).pct_change().fillna(0)
    targets = returns.shift(-1).fillna(0).values
    
    # Create timestamps
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=n_samples//1440),
        periods=n_samples,
        freq='T'
    )
    
    return features, targets, timestamps


def demonstrate_data_splitting():
    """Demonstrate time series aware data splitting."""
    print("\n" + "="*80)
    print("DEMONSTRATING DATA SPLITTING")
    print("="*80)
    
    # Generate data
    X, y, timestamps = generate_sample_data(5000)
    
    # 1. Basic time series split
    splitter = TimeSeriesDataSplitter(
        test_size=0.2,
        validation_size=0.1,
        gap_size=10,
        purge_days=2
    )
    
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y, timestamps)
    
    print(f"\nBasic Time Series Split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Gap between sets: 10 periods")
    print(f"Purge period: 2 days")
    
    # 2. Purged K-Fold cross-validation
    from validation.data_splitter import PurgedKFold
    
    pkf = PurgedKFold(n_splits=5, purge_days=2, embargo_days=1)
    
    print(f"\nPurged K-Fold Cross-Validation (5 folds):")
    for i, (train_idx, test_idx) in enumerate(pkf.split(X, y, timestamps)):
        print(f"Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    # 3. Walk-forward validation
    from validation.data_splitter import WalkForwardSplitter
    
    wf = WalkForwardSplitter(
        train_window=1000,
        test_window=200,
        step_size=100
    )
    
    print(f"\nWalk-Forward Validation:")
    for i, (train_idx, test_idx) in enumerate(wf.split(X)):
        if i < 3:  # Show first 3 windows
            print(f"Window {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
    print("...")


def demonstrate_cross_validation():
    """Demonstrate advanced cross-validation strategies."""
    print("\n" + "="*80)
    print("DEMONSTRATING CROSS-VALIDATION STRATEGIES")
    print("="*80)
    
    # Generate data
    X, y, timestamps = generate_sample_data(5000)
    
    # 1. Purged Time Series CV
    cv = PurgedTimeSeriesCrossValidator(
        n_splits=5,
        purge_days=2,
        embargo_days=1,
        max_train_size=2000
    )
    
    print("\nPurged Time Series Cross-Validation:")
    for fold in cv.split(X, y, timestamps):
        print(f"Fold {fold.fold_id}: "
              f"Train={len(fold.train_indices)} samples "
              f"({fold.train_start.strftime('%Y-%m-%d')} to {fold.train_end.strftime('%Y-%m-%d')}), "
              f"Val={len(fold.val_indices)} samples "
              f"({fold.val_start.strftime('%Y-%m-%d')} to {fold.val_end.strftime('%Y-%m-%d')})")
    
    # 2. Walk-Forward CV
    wf_cv = WalkForwardCrossValidator(
        n_splits=10,
        train_window=1000,
        test_window=100,
        expanding=True
    )
    
    print("\nWalk-Forward Cross-Validation (Expanding Window):")
    fold_count = 0
    for fold in wf_cv.split(X, y, timestamps):
        if fold_count < 3:  # Show first 3 folds
            print(f"Fold {fold.fold_id}: "
                  f"Train={len(fold.train_indices)}, Val={len(fold.val_indices)}")
        fold_count += 1
    print(f"... (Total {fold_count} folds)")


def demonstrate_model_validation():
    """Demonstrate model-specific validation."""
    print("\n" + "="*80)
    print("DEMONSTRATING MODEL-SPECIFIC VALIDATION")
    print("="*80)
    
    # Generate data
    X, y, timestamps = generate_sample_data(5000)
    
    # Initialize validation orchestrator
    orchestrator = ValidationOrchestrator(
        results_dir="validation_results",
        n_jobs=4
    )
    
    # Example 1: Validate a statistical model (mock)
    print("\n1. Statistical Model Validation:")
    
    # Create mock ARIMA model
    class MockARIMAModel:
        def __init__(self):
            self.k_ar = 2
            self.k_ma = 1
            self.k_trend = 1
            
        def fit(self, X, y):
            return self
            
        def predict(self, X):
            # Simple mock prediction
            return np.random.normal(0, 0.01, len(X))
        
        def copy(self):
            return MockARIMAModel()
    
    arima_model = MockARIMAModel()
    
    validation_config = {
        'cv_strategy': 'purged_time_series',
        'n_splits': 3,
        'test_size': 0.2,
        'metrics': ['mae', 'rmse', 'directional_accuracy']
    }
    
    # Validate single model
    results = orchestrator.validate_model(
        model=arima_model,
        model_type='statistical',
        X=X,
        y=y,
        timestamps=timestamps,
        validation_config=validation_config
    )
    
    print("Validation completed!")
    print(f"Test MAE: {results.get('test_results', {}).get('mae', 'N/A'):.4f}")
    print(f"Results saved to: {orchestrator.results_dir}")
    
    # Example 2: Validate multiple models
    print("\n2. Multiple Model Validation:")
    
    # Create mock models
    class MockNNModel:
        def __init__(self, name):
            self.name = name
            self.history = {'train_loss': [0.1, 0.08, 0.06], 'val_loss': [0.12, 0.10, 0.09]}
            
        def predict(self, X):
            return np.random.normal(0, 0.01, len(X))
        
        def fit(self, X, y):
            return self
    
    models = {
        'arima': (MockARIMAModel(), 'statistical'),
        'neural_net_1': (MockNNModel('nn1'), 'neural_network'),
        'neural_net_2': (MockNNModel('nn2'), 'neural_network')
    }
    
    # Validate all models
    all_results = orchestrator.validate_multiple_models(
        models=models,
        X=X,
        y=y,
        timestamps=timestamps,
        validation_config=validation_config
    )
    
    print(f"Validated {len(models)} models")
    print(f"Best model: {all_results.get('best_model', 'N/A')}")
    
    # Generate report
    report = orchestrator.generate_validation_report(all_results)
    print("\nValidation report generated")


def demonstrate_production_validation():
    """Demonstrate production validation and monitoring."""
    print("\n" + "="*80)
    print("DEMONSTRATING PRODUCTION VALIDATION")
    print("="*80)
    
    # Initialize production validator
    prod_validator = ProductionValidator(
        monitoring_window=1000,
        alert_thresholds={
            'accuracy_drop': 0.05,
            'mae_increase': 0.10,
            'drift_score': 0.3
        }
    )
    
    # Generate baseline data
    X_baseline, y_baseline, _ = generate_sample_data(2000)
    baseline_predictions = y_baseline + np.random.normal(0, 0.01, len(y_baseline))
    
    # Set baseline
    prod_validator.set_baseline(
        model_id='model_001',
        baseline_predictions=baseline_predictions,
        baseline_features=X_baseline,
        baseline_targets=y_baseline
    )
    
    print("Baseline statistics set for model_001")
    
    # Simulate production predictions with drift
    print("\nSimulating production predictions...")
    
    # Normal predictions
    for i in range(500):
        features = X_baseline[i] + np.random.normal(0, 0.01, X_baseline.shape[1])
        prediction = baseline_predictions[i] + np.random.normal(0, 0.01)
        
        result = prod_validator.validate_prediction(
            model_id='model_001',
            features=features,
            prediction=prediction,
            true_value=y_baseline[i] if i % 10 == 0 else None  # True values available occasionally
        )
    
    print("500 normal predictions validated")
    
    # Introduce drift
    print("\nIntroducing feature drift...")
    for i in range(500, 600):
        # Shift features significantly
        features = X_baseline[i] + np.random.normal(0.5, 0.1, X_baseline.shape[1])
        prediction = baseline_predictions[i] + np.random.normal(0.2, 0.05)
        
        result = prod_validator.validate_prediction(
            model_id='model_001',
            features=features,
            prediction=prediction,
            true_value=y_baseline[i] if i % 10 == 0 else None
        )
        
        if result['drift_detected']:
            print(f"Drift detected at prediction {i}!")
            print(f"Alerts: {len(result['alerts'])}")
            break
    
    # Get model health report
    health_report = prod_validator.get_model_health_report('model_001')
    print(f"\nModel Health Status: {health_report['status']}")
    print(f"Active Alerts: {len(health_report['alerts'])}")
    
    # Demonstrate A/B testing
    print("\n3. A/B Testing:")
    
    # Start A/B test
    prod_validator.start_ab_test(
        test_id='test_001',
        model_a_id='model_001',
        model_b_id='model_002',
        traffic_split=0.5,
        min_samples=50
    )
    
    # Simulate A/B test results
    for i in range(60):
        # Model A results
        pred_a = y_baseline[i] + np.random.normal(0, 0.02)
        prod_validator.record_ab_result(
            test_id='test_001',
            model_id='model_001',
            prediction=pred_a,
            true_value=y_baseline[i]
        )
        
        # Model B results (slightly better)
        pred_b = y_baseline[i] + np.random.normal(0, 0.015)
        prod_validator.record_ab_result(
            test_id='test_001',
            model_id='model_002',
            prediction=pred_b,
            true_value=y_baseline[i]
        )
    
    # Check test results
    if 'test_001' in prod_validator.ab_tests:
        test_results = prod_validator.ab_tests['test_001']['metrics']
        if test_results:
            print(f"A/B Test Winner: {test_results.get('winner', 'N/A')}")
            print(f"Improvement: {test_results.get('improvement', 0):.1%}")
            print(f"P-value: {test_results.get('p_value', 'N/A'):.4f}")


def demonstrate_statistical_significance():
    """Demonstrate statistical significance testing."""
    print("\n" + "="*80)
    print("DEMONSTRATING STATISTICAL SIGNIFICANCE TESTING")
    print("="*80)
    
    from validation import StatisticalSignificanceTester
    
    # Generate predictions from two models
    n_samples = 1000
    y_true = np.random.normal(0, 1, n_samples)
    
    # Model 1: Good model
    predictions_1 = y_true + np.random.normal(0, 0.1, n_samples)
    
    # Model 2: Slightly worse model
    predictions_2 = y_true + np.random.normal(0, 0.15, n_samples)
    
    # Initialize tester
    tester = StatisticalSignificanceTester(significance_level=0.05)
    
    # Compare two models
    print("\nComparing Two Models:")
    results = tester.compare_two_models(y_true, predictions_1, predictions_2)
    
    for test_name, result in results.items():
        if hasattr(result, 'p_value'):
            print(f"\n{result.test_name}:")
            print(f"  Statistic: {result.statistic:.4f}")
            print(f"  P-value: {result.p_value:.4f}")
            print(f"  Significant: {result.significant}")
            if result.effect_size:
                print(f"  Effect size: {result.effect_size:.4f}")
            print(f"  Interpretation: {result.interpretation}")
    
    # Compare multiple models
    print("\n\nComparing Multiple Models:")
    
    predictions_3 = y_true + np.random.normal(0, 0.2, n_samples)
    
    predictions_dict = {
        'model_1': predictions_1,
        'model_2': predictions_2,
        'model_3': predictions_3
    }
    
    multi_results = tester.compare_multiple_models(y_true, predictions_dict)
    
    print(f"\nOmnibus Test: {multi_results['omnibus_test'].test_name}")
    print(f"P-value: {multi_results['omnibus_test'].p_value:.4f}")
    print(f"Significant: {multi_results['omnibus_test'].significant}")
    
    print("\nModel Rankings:")
    for metric, ranking in multi_results['rankings'].items():
        if metric != 'overall':
            print(f"{metric}: {ranking}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL VALIDATION SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Run demonstrations
    demonstrate_data_splitting()
    demonstrate_cross_validation()
    demonstrate_model_validation()
    demonstrate_production_validation()
    demonstrate_statistical_significance()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nThe validation system provides:")
    print("- Time series aware data splitting with purging and embargo")
    print("- Multiple cross-validation strategies (purged, walk-forward, blocking)")
    print("- Model-specific validation for all model types")
    print("- Statistical significance testing for model comparison")
    print("- Production monitoring with drift detection")
    print("- A/B testing framework for model selection")
    print("- Comprehensive reporting and alerting")
    
    print("\nCheck the 'validation_results' directory for detailed results and reports.")


if __name__ == "__main__":
    main()