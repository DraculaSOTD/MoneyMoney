"""
Demonstration of enhanced model capabilities.

Shows how to use the advanced meta-learner, model monitoring,
and unified interface together.
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ensemble.advanced_meta_learner import AdvancedMetaLearner
from models.monitoring.model_monitor import ModelMonitor
from models.unified_interface import UnifiedDeepLearningModel, UnifiedEnsembleModel
from models.deep_learning.gru_attention import GRUAttentionModel
from models.deep_learning.tcn import TCNModel
from models.reinforcement import DRQNAgent
from features import EnhancedFeatureEngineering

import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_market_data(n_samples: int = 2000) -> pd.DataFrame:
    """Generate realistic market data with regime changes."""
    np.random.seed(42)
    
    # Create different market regimes
    regime_lengths = [500, 800, 400, 300]
    regimes = ['normal', 'trending', 'high_volatility', 'low_volatility']
    
    data_segments = []
    current_price = 100
    
    for regime, length in zip(regimes, regime_lengths):
        if regime == 'normal':
            returns = np.random.normal(0.0001, 0.02, length)
        elif regime == 'trending':
            returns = np.random.normal(0.001, 0.015, length)
        elif regime == 'high_volatility':
            returns = np.random.normal(0, 0.04, length)
        else:  # low_volatility
            returns = np.random.normal(0, 0.01, length)
        
        prices = current_price * np.cumprod(1 + returns)
        current_price = prices[-1]
        
        segment_data = {
            'close': prices,
            'returns': returns,
            'regime': [regime] * length
        }
        data_segments.append(pd.DataFrame(segment_data))
    
    # Combine segments
    df = pd.concat(data_segments, ignore_index=True)
    
    # Add OHLCV data
    df['open'] = df['close'] * (1 + np.random.normal(0, 0.001, len(df)))
    df['high'] = df['close'] * (1 + np.abs(np.random.normal(0, 0.002, len(df))))
    df['low'] = df['close'] * (1 - np.abs(np.random.normal(0, 0.002, len(df))))
    df['volume'] = np.random.lognormal(10, 0.5, len(df))
    
    return df


def create_base_models(input_size: int) -> Dict[str, Any]:
    """Create a suite of base models."""
    models = {}
    
    # GRU with Attention
    gru_model = GRUAttentionModel(
        input_size=input_size,
        hidden_sizes=[128, 64],
        num_attention_heads=4,
        num_classes=3
    )
    models['gru_attention'] = UnifiedDeepLearningModel(
        'gru_attention_v1', gru_model, feature_names=[f'feature_{i}' for i in range(input_size)]
    )
    
    # TCN
    tcn_model = TCNModel(
        input_size=input_size,
        hidden_channels=64,
        kernel_size=3,
        num_layers=4,
        epochs=50
    )
    models['tcn'] = UnifiedDeepLearningModel(
        'tcn_v1', tcn_model, feature_names=[f'feature_{i}' for i in range(input_size)]
    )
    
    # Note: DRQN requires special handling due to its RL nature
    # For this demo, we'll use just the deep learning models
    
    return models


def main():
    """Run enhanced models demonstration."""
    logger.info("Starting enhanced models demonstration...")
    
    # Generate market data
    logger.info("Generating market data with regime changes...")
    df = generate_market_data(3000)
    
    # Feature engineering
    logger.info("Creating features with stationarity analysis...")
    feature_engineer = EnhancedFeatureEngineering()
    
    # Prepare data for modeling
    modeling_data = feature_engineer.prepare_for_modeling(
        df, target_type='classification', lookahead=1
    )
    
    X = modeling_data['features'].values
    y = modeling_data['target'].values
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create base models
    logger.info("Creating base models...")
    base_models = create_base_models(X_train.shape[1])
    
    # Train base models
    logger.info("Training base models...")
    for name, model in base_models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train, validation_data=(X_val, y_val))
    
    # Create ensemble with advanced meta-learner
    logger.info("Creating advanced ensemble...")
    ensemble = UnifiedEnsembleModel(
        'advanced_ensemble_v1',
        base_models,
        ensemble_method='weighted'
    )
    
    # Create advanced meta-learner
    meta_learner = AdvancedMetaLearner(
        base_models,
        meta_model_type='weighted_average',
        lookback_window=100,
        regime_detection=True,
        learning_rate=0.01
    )
    
    # Initialize model monitors
    monitors = {}
    for name in base_models.keys():
        monitors[name] = ModelMonitor(
            model_name=name,
            window_size=500
        )
    
    ensemble_monitor = ModelMonitor(
        model_name='ensemble',
        window_size=500
    )
    
    # Simulate online prediction and monitoring
    logger.info("\n" + "="*60)
    logger.info("ONLINE PREDICTION AND MONITORING")
    logger.info("="*60)
    
    # Process test data in batches (simulating real-time)
    batch_size = 50
    n_batches = len(X_test) // batch_size
    
    all_predictions = []
    all_ensemble_predictions = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        X_batch = X_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]
        
        # Get market data for regime detection
        market_batch = df.iloc[train_size+val_size+start_idx:train_size+val_size+end_idx]
        
        # Make predictions with all models
        batch_predictions = {}
        for name, model in base_models.items():
            pred = model.predict(X_batch)
            batch_predictions[name] = pred
            
            # Update monitor
            monitors[name].update_metrics(pred, y_batch, X_batch)
        
        # Meta-learner prediction with regime detection
        meta_result = meta_learner.predict(
            X_batch, market_data=market_batch, return_all=True
        )
        
        ensemble_pred = meta_result['prediction']
        all_ensemble_predictions.extend(ensemble_pred)
        
        # Update ensemble monitor
        ensemble_monitor.update_metrics(ensemble_pred, y_batch, X_batch)
        
        # Online learning
        if batch_idx > 0:
            # Update with previous batch's true values
            prev_start = (batch_idx-1) * batch_size
            prev_end = prev_start + batch_size
            meta_learner.online_update(y_test[prev_start:prev_end])
        
        # Log progress every 10 batches
        if batch_idx % 10 == 0:
            logger.info(f"\nBatch {batch_idx}/{n_batches}")
            logger.info(f"Current regime: {meta_result['regime']}")
            logger.info(f"Ensemble confidence: {meta_result['ensemble_confidence']:.3f}")
            
            # Show model weights
            weights = meta_result['weights']
            logger.info("Model weights:")
            for name, weight in weights.items():
                logger.info(f"  {name}: {weight:.3f}")
    
    # Final performance analysis
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE ANALYSIS")
    logger.info("="*60)
    
    # Get performance reports
    for name, monitor in monitors.items():
        report = monitor.get_performance_report()
        logger.info(f"\n{name} Performance:")
        if 'recent_performance' in report:
            for metric, stats in report['recent_performance'].items():
                if metric in ['accuracy', 'sharpe_ratio', 'win_rate']:
                    logger.info(f"  {metric}: {stats['mean']:.3f} (±{stats['std']:.3f})")
    
    # Ensemble performance
    ensemble_report = ensemble_monitor.get_performance_report()
    logger.info(f"\nEnsemble Performance:")
    if 'recent_performance' in ensemble_report:
        for metric, stats in ensemble_report['recent_performance'].items():
            if metric in ['accuracy', 'sharpe_ratio', 'win_rate']:
                logger.info(f"  {metric}: {stats['mean']:.3f} (±{stats['std']:.3f})")
    
    # Meta-learner summary
    meta_summary = meta_learner.get_performance_summary()
    logger.info(f"\nMeta-Learner Summary:")
    logger.info(f"Total predictions: {meta_summary['total_predictions']}")
    logger.info(f"Regime distribution: {meta_summary['regime_distribution']}")
    logger.info(f"Model importance: {meta_summary['model_importance']}")
    
    # Drift analysis
    logger.info("\n" + "="*60)
    logger.info("DRIFT ANALYSIS")
    logger.info("="*60)
    
    for name, monitor in monitors.items():
        report = monitor.get_performance_report()
        if 'drift_analysis' in report:
            drift = report['drift_analysis']
            logger.info(f"{name} - Mean drift: {drift['mean_drift']:.3f}, "
                       f"Max drift: {drift['max_drift']:.3f}")
    
    # Feature importance from ensemble
    logger.info("\n" + "="*60)
    logger.info("FEATURE IMPORTANCE")
    logger.info("="*60)
    
    feature_importance = ensemble.get_feature_importance()
    if feature_importance:
        # Show top 10 features
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)[:10]
        for feature, importance in sorted_features:
            logger.info(f"  {feature}: {importance:.3f}")
    
    # Example prediction explanation
    logger.info("\n" + "="*60)
    logger.info("PREDICTION EXPLANATION EXAMPLE")
    logger.info("="*60)
    
    # Explain first test prediction
    explanation = ensemble.explain_prediction(X_test, 0)
    logger.info(f"Sample prediction explanation:")
    logger.info(f"  Ensemble prediction: {explanation['ensemble_prediction']}")
    logger.info(f"  Probabilities: {explanation['ensemble_probabilities']}")
    logger.info(f"  Model weights: {explanation['model_weights']}")
    
    # Save models and monitoring data
    logger.info("\n" + "="*60)
    logger.info("SAVING RESULTS")
    logger.info("="*60)
    
    # Save ensemble checkpoint
    ensemble.save_checkpoint('enhanced_ensemble_checkpoint.pkl')
    logger.info("Saved ensemble checkpoint")
    
    # Export monitoring data
    for name, monitor in monitors.items():
        monitor.export_monitoring_data(f'{name}_monitoring.json')
    ensemble_monitor.export_monitoring_data('ensemble_monitoring.json')
    logger.info("Exported monitoring data")
    
    logger.info("\nEnhanced models demonstration completed!")
    
    # Clean up
    import glob
    for file in glob.glob('*.pkl') + glob.glob('*.json'):
        if os.path.exists(file):
            os.remove(file)


if __name__ == "__main__":
    main()