"""
Demo script for Temporal Convolutional Network (TCN) model.
Shows how to use TCN for cryptocurrency trading signals.
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.deep_learning.tcn import TCNModel, TCNTrainer
from data.data_loader import DataLoader
from features.enhanced_technical_indicators import EnhancedTechnicalIndicators
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 1000) -> tuple:
    """Generate synthetic data for testing."""
    np.random.seed(42)
    
    # Generate price data with trend and noise
    time = np.arange(n_samples)
    trend = 0.01 * time
    seasonal = 10 * np.sin(2 * np.pi * time / 100)
    noise = np.random.normal(0, 2, n_samples)
    price = 100 + trend + seasonal + noise
    
    # Generate volume
    volume = np.random.lognormal(10, 1, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1h'),
        'open': price + np.random.normal(0, 0.5, n_samples),
        'high': price + np.abs(np.random.normal(0, 1, n_samples)),
        'low': price - np.abs(np.random.normal(0, 1, n_samples)),
        'close': price,
        'volume': volume
    })
    
    # Generate labels (0: buy, 1: sell, 2: hold)
    returns = df['close'].pct_change().fillna(0)
    labels = np.where(returns > 0.001, 0,  # Buy
                     np.where(returns < -0.001, 1,  # Sell
                             2))  # Hold
    
    return df, labels[1:]  # Skip first label due to pct_change


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """Prepare features from price data."""
    features_dict = {}
    
    # Price-based features
    features_dict['returns'] = df['close'].pct_change().fillna(0)
    features_dict['log_returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    
    # Simple moving averages
    for period in [5, 10, 20]:
        features_dict[f'sma_{period}'] = df['close'].rolling(window=period).mean().fillna(df['close'])
        features_dict[f'sma_ratio_{period}'] = (df['close'] / features_dict[f'sma_{period}']).fillna(1)
    
    # Volatility
    features_dict['volatility_20'] = features_dict['returns'].rolling(window=20).std().fillna(0)
    
    # Price position
    features_dict['high_low_ratio'] = (df['high'] / df['low']).fillna(1)
    features_dict['close_open_ratio'] = (df['close'] / df['open']).fillna(1)
    
    # Volume features
    features_dict['volume_ratio'] = (df['volume'] / df['volume'].rolling(window=20).mean()).fillna(1)
    
    # Convert to DataFrame and then to numpy array
    features_df = pd.DataFrame(features_dict)
    
    # Fill any remaining NaN values
    features_df = features_df.ffill().fillna(0)
    
    return features_df.values


def main():
    """Run TCN demo."""
    logger.info("Starting TCN demo...")
    
    # Generate or load data
    logger.info("Generating synthetic data...")
    df, labels = generate_synthetic_data(2000)
    
    # Prepare features
    logger.info("Calculating technical indicators...")
    features = prepare_features(df)
    
    # Remove rows with NaN from feature calculation
    min_length = min(len(features), len(labels))
    features = features[:min_length]
    labels = labels[:min_length]
    
    # Split data
    train_size = int(0.8 * len(features))
    X_train = features[:train_size]
    y_train = labels[:train_size]
    X_test = features[train_size:]
    y_test = labels[train_size:]
    
    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    logger.info(f"Number of features: {X_train.shape[1]}")
    
    # Configure TCN
    config = {
        'model': {
            'hidden_channels': 64,
            'kernel_size': 3,
            'num_layers': 4,
            'dilations': [1, 2, 4, 8],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'use_skip_connections': True,
            'activation': 'relu'
        },
        'sequence_length': 30
    }
    
    # Initialize trainer
    trainer = TCNTrainer(config)
    
    # Train model
    logger.info("Training TCN model...")
    results = trainer.train(X_train, y_train, X_test, y_test)
    
    # Print results
    logger.info("\n=== Training Results ===")
    logger.info(f"Training time: {results['training_time']:.2f} seconds")
    
    if results['train_metrics']:
        logger.info("\nTraining Metrics:")
        for key, value in results['train_metrics'].items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    if results['val_metrics']:
        logger.info("\nValidation Metrics:")
        for key, value in results['val_metrics'].items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    # Make predictions on test set
    logger.info("\nMaking predictions on test set...")
    predictions = trainer.model.predict(X_test)
    
    # Analyze predictions
    unique, counts = np.unique(predictions, return_counts=True)
    logger.info("\nPrediction Distribution:")
    action_map = {0: 'Buy', 1: 'Sell', 2: 'Hold'}
    for action, count in zip(unique, counts):
        percentage = count / len(predictions) * 100
        logger.info(f"  {action_map.get(action, action)}: {count} ({percentage:.1f}%)")
    
    # Save model
    model_path = "tcn_model_demo.npz"
    trainer.save_model(model_path)
    logger.info(f"\nModel saved to {model_path}")
    
    # Test loading
    logger.info("\nTesting model loading...")
    new_trainer = TCNTrainer(config)
    new_trainer.load_model(model_path)
    
    # Verify loaded model works
    loaded_predictions = new_trainer.model.predict(X_test[:10])
    logger.info(f"Loaded model predictions (first 10): {loaded_predictions}")
    
    # Clean up
    os.remove(model_path)
    logger.info("\nDemo completed successfully!")


if __name__ == "__main__":
    main()