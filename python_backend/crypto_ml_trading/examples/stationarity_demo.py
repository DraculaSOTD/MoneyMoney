"""
Demo script for enhanced feature engineering with stationarity analysis.
Shows how to create stationary features for better model performance.
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import StationarityAnalyzer, EnhancedFeatureEngineering
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample cryptocurrency data with trends."""
    np.random.seed(42)
    
    # Time index
    time = np.arange(n_samples)
    
    # Generate price with trend (non-stationary)
    trend = 0.05 * time
    seasonal = 10 * np.sin(2 * np.pi * time / 100)
    noise = np.random.normal(0, 2, n_samples)
    price = 100 + trend + seasonal + noise
    
    # Generate volume
    volume_trend = 1000 + 0.1 * time
    volume_noise = np.random.lognormal(0, 0.3, n_samples)
    volume = volume_trend * volume_noise
    
    # Create OHLC data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1h'),
        'open': price + np.random.normal(0, 0.5, n_samples),
        'high': price + np.abs(np.random.normal(0, 1, n_samples)),
        'low': price - np.abs(np.random.normal(0, 1, n_samples)),
        'close': price,
        'volume': volume
    })
    
    return df


def test_stationarity(series: pd.Series, name: str) -> None:
    """Test and display stationarity results."""
    analyzer = StationarityAnalyzer()
    
    logger.info(f"\n=== Testing stationarity for: {name} ===")
    
    # Run tests
    results = analyzer.test_stationarity(series, tests=['adf', 'kpss', 'pp'])
    
    # Display results
    for test_name, test_result in results.items():
        if test_name == 'overall_stationary':
            logger.info(f"Overall stationary: {test_result}")
        elif test_name in ['stationary_tests', 'total_tests']:
            continue
        else:
            if isinstance(test_result, dict):
                stationary = test_result.get('stationary', 'N/A')
                stat = test_result.get('test_statistic', 'N/A')
                logger.info(f"{test_name.upper()} Test: Stationary={stationary}, "
                           f"Test Statistic={stat:.4f if isinstance(stat, (int, float)) else stat}")


def plot_transformations(original: pd.Series, transformed: pd.Series, 
                        title: str, filename: str) -> None:
    """Plot original vs transformed series."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Original series
    ax1.plot(original.index, original.values, 'b-', alpha=0.7)
    ax1.set_title(f'Original {title}')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)
    
    # Transformed series
    ax2.plot(transformed.index, transformed.values, 'g-', alpha=0.7)
    ax2.set_title(f'Transformed {title} (Stationary)')
    ax2.set_ylabel('Value')
    ax2.set_xlabel('Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    logger.info(f"Saved plot: {filename}")


def main():
    """Run stationarity demo."""
    logger.info("Starting stationarity analysis demo...")
    
    # Generate sample data
    logger.info("Generating sample cryptocurrency data...")
    df = generate_sample_data(2000)
    
    # Initialize analyzers
    stationarity_analyzer = StationarityAnalyzer()
    feature_engineer = EnhancedFeatureEngineering()
    
    # Test stationarity of raw price data
    logger.info("\n" + "="*60)
    logger.info("PART 1: Testing Raw Data Stationarity")
    logger.info("="*60)
    
    test_stationarity(df['close'], "Close Price")
    test_stationarity(df['volume'], "Volume")
    
    # Transform to achieve stationarity
    logger.info("\n" + "="*60)
    logger.info("PART 2: Applying Transformations")
    logger.info("="*60)
    
    # Transform close price
    close_transformed, close_info = stationarity_analyzer.make_stationary(
        df['close'], method='auto'
    )
    logger.info(f"\nClose price transformations: {close_info['transformations']}")
    test_stationarity(close_transformed, "Transformed Close Price")
    
    # Transform volume
    volume_transformed, volume_info = stationarity_analyzer.make_stationary(
        df['volume'], method='auto'
    )
    logger.info(f"\nVolume transformations: {volume_info['transformations']}")
    test_stationarity(volume_transformed, "Transformed Volume")
    
    # Create plots
    plot_transformations(df['close'], close_transformed, 
                        'Close Price', 'close_price_transformation.png')
    plot_transformations(df['volume'], volume_transformed,
                        'Volume', 'volume_transformation.png')
    
    # Create comprehensive features
    logger.info("\n" + "="*60)
    logger.info("PART 3: Enhanced Feature Engineering")
    logger.info("="*60)
    
    feature_result = feature_engineer.create_all_features(df, apply_stationarity=True)
    features = feature_result['features']
    info = feature_result['info']
    
    logger.info(f"\nCreated {info['n_features']} features")
    logger.info(f"Stationary features: {len(info['stationary_features'])}")
    logger.info(f"Non-stationary features requiring transformation: "
               f"{info['n_features'] - len(info['stationary_features'])}")
    
    # Show sample features
    logger.info("\nSample features created:")
    for i, feature in enumerate(features.columns[:10]):
        logger.info(f"  {i+1}. {feature}")
    
    # Test some key features for stationarity
    logger.info("\n" + "="*60)
    logger.info("PART 4: Testing Key Features")
    logger.info("="*60)
    
    key_features = ['return_1', 'rsi_14', 'macd_histogram', 'volatility_20']
    for feature in key_features:
        if feature in features.columns:
            test_stationarity(features[feature].dropna(), feature)
    
    # Prepare for modeling
    logger.info("\n" + "="*60)
    logger.info("PART 5: Preparing Data for Modeling")
    logger.info("="*60)
    
    modeling_data = feature_engineer.prepare_for_modeling(
        df, target_type='classification', lookahead=1
    )
    
    logger.info(f"\nModeling data prepared:")
    logger.info(f"  Samples: {modeling_data['n_samples']}")
    logger.info(f"  Features: {modeling_data['n_features']}")
    logger.info(f"  Target type: {modeling_data['target_type']}")
    logger.info(f"  Lookahead: {modeling_data['lookahead']} periods")
    
    # Show target distribution
    target = modeling_data['target']
    target_counts = target.value_counts().sort_index()
    logger.info(f"\nTarget distribution:")
    logger.info(f"  Buy (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(target)*100:.1f}%)")
    logger.info(f"  Hold (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(target)*100:.1f}%)")
    logger.info(f"  Sell (2): {target_counts.get(2, 0)} ({target_counts.get(2, 0)/len(target)*100:.1f}%)")
    
    # Create feature importance plot (based on correlation with target)
    logger.info("\n" + "="*60)
    logger.info("PART 6: Feature Analysis")
    logger.info("="*60)
    
    # Calculate absolute correlation with target
    feature_importance = features.corrwith(target).abs().sort_values(ascending=False)
    
    # Plot top features
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = feature_importance.head(20)
    top_features.plot(kind='barh', ax=ax)
    ax.set_xlabel('Absolute Correlation with Target')
    ax.set_title('Top 20 Features by Correlation')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.close()
    logger.info("Saved feature importance plot: feature_importance.png")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info("1. Raw price data is typically non-stationary (has trends)")
    logger.info("2. Transformations like differencing can achieve stationarity")
    logger.info("3. Return-based features are naturally stationary")
    logger.info("4. Technical indicators vary in stationarity")
    logger.info("5. Proper feature engineering improves model performance")
    
    logger.info("\nDemo completed successfully!")
    
    # Clean up plots
    for file in ['close_price_transformation.png', 'volume_transformation.png', 
                 'feature_importance.png']:
        if os.path.exists(file):
            os.remove(file)


if __name__ == "__main__":
    main()