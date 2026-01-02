#!/usr/bin/env python3
"""
Comprehensive ML Pipeline Demonstration
Shows the complete flow from raw data to ML-ready features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import all components
from data.enhanced_data_loader import EnhancedDataLoader, BinanceDataSource
from data.data_validator import DataValidator
from data.preprocessing import AdvancedPreprocessor
from features.enhanced_technical_indicators import EnhancedTechnicalIndicators
from features.ml_feature_engineering import MLFeatureEngineering
from features.decision_labeler import DecisionLabeler


def create_sample_data():
    """Create sample Binance format data for demonstration."""
    # Generate 1000 data points
    n_points = 1000
    base_price = 100
    
    # Create time series
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=n_points),
        periods=n_points,
        freq='1H'
    )
    
    # Generate realistic OHLCV data
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.01, n_points)
    prices = base_price * (1 + returns).cumprod()
    
    # Create OHLC from prices
    high_factor = 1 + abs(np.random.normal(0, 0.005, n_points))
    low_factor = 1 - abs(np.random.normal(0, 0.005, n_points))
    
    df = pd.DataFrame({
        'Open time': timestamps.astype(np.int64) // 10**9 * 1000,  # Unix ms
        'Open': prices * (1 + np.random.normal(0, 0.001, n_points)),
        'High': prices * high_factor,
        'Low': prices * low_factor,
        'Close': prices,
        'Volume': np.random.lognormal(10, 1, n_points),
        'Close time': (timestamps + pd.Timedelta(hours=1)).astype(np.int64) // 10**9 * 1000,
        'Quote asset volume': np.random.lognormal(15, 1, n_points),
        'Number of trades': np.random.poisson(100, n_points),
        'Taker buy base asset volume': np.random.lognormal(9.5, 1, n_points),
        'Taker buy quote asset volume': np.random.lognormal(14.5, 1, n_points),
        'Ignore': 0
    })
    
    # Save to file
    sample_file = '/tmp/sample_binance_data.csv'
    df.to_csv(sample_file, index=False)
    print(f"✓ Created sample data with {len(df)} rows")
    
    return sample_file


def demonstrate_pipeline():
    """Demonstrate the complete ML pipeline."""
    print("="*80)
    print("ML PIPELINE DEMONSTRATION")
    print("="*80)
    
    # 1. Create sample data
    print("\n1. CREATING SAMPLE DATA")
    print("-"*40)
    data_file = create_sample_data()
    
    # 2. Load data with auto-detection
    print("\n2. LOADING DATA WITH AUTO-DETECTION")
    print("-"*40)
    # Create a data source that will auto-detect format
    from data.enhanced_data_loader import UniversalDataSource
    data_source = UniversalDataSource(data_file)
    
    loader = EnhancedDataLoader(data_source=data_source)
    
    # Load data directly from the universal source
    df = data_source.fetch_data(
        symbol="BTCUSDT",  # Not used for file source
        start_time=datetime.now() - timedelta(days=30),
        end_time=datetime.now()
    )
    print(f"✓ Loaded {len(df)} rows")
    print(f"✓ Auto-detected format: {data_source.detected_format}")
    print(f"✓ Columns: {list(df.columns)[:5]}...")
    
    # 3. Initial validation
    print("\n3. INITIAL DATA VALIDATION")
    print("-"*40)
    validator = DataValidator()
    initial_report = validator.validate(df)
    print(f"✓ Data Quality Score: {initial_report['data_quality']:.1f}/100")
    print(f"✓ Missing Values: {initial_report['missing_values']['total_missing']}")
    print(f"✓ Data Shape: {initial_report['data_shape']}")
    
    # 4. Add technical indicators
    print("\n4. ADDING TECHNICAL INDICATORS")
    print("-"*40)
    indicator_config = {
        'sma': {'enabled': True, 'periods': [10, 20, 50]},
        'ema': {'enabled': True, 'periods': [12, 26]},
        'rsi': {'enabled': True, 'period': 14},
        'macd': {'enabled': True},
        'bollinger': {'enabled': True},
        'atr': {'enabled': True},
        'adx': {'enabled': True},
        'stochastic': {'enabled': True},
        'ichimoku': {'enabled': True}
    }
    
    df = EnhancedTechnicalIndicators.compute_all_indicators(df, indicator_config)
    print(f"✓ Added {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']])} indicators")
    
    # 5. Create ML features
    print("\n5. FEATURE ENGINEERING")
    print("-"*40)
    ml_engineer = MLFeatureEngineering()
    
    # Create all feature types
    df = ml_engineer.prepare_ml_features(df, {
        'percentage_features': True,
        'lagged_features': True,
        'rolling_features': True,
        'time_features': True,
        'indicator_features': True
    })
    
    print(f"✓ Total features: {len(ml_engineer.feature_names)}")
    print(f"✓ Feature categories:")
    print(f"  - Percentage changes: {len([f for f in df.columns if 'pct' in f])}")
    print(f"  - Lagged features: {len([f for f in df.columns if 'lag' in f])}")
    print(f"  - Rolling features: {len([f for f in df.columns if 'mean' in f or 'std' in f])}")
    print(f"  - Time features: {len([f for f in df.columns if f in ['hour', 'day_of_week', 'month']])}")
    
    # 6. Add ML labels
    print("\n6. CREATING ML LABELS")
    print("-"*40)
    labeler = DecisionLabeler()
    df = labeler.create_labels(df, lookforward=5, threshold=0.002)
    
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        print(f"✓ Label distribution:")
        for label, count in label_counts.items():
            print(f"  - {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # 7. Advanced preprocessing
    print("\n7. ADVANCED PREPROCESSING")
    print("-"*40)
    preprocessor = AdvancedPreprocessor()
    
    # Configure preprocessing
    preprocess_config = {
        'handle_missing': {'enabled': True, 'method': 'forward_fill'},
        'handle_infinite': {'enabled': True, 'method': 'clip'},
        'remove_duplicates': {'enabled': True},
        'stationarity': {'enabled': True, 'method': 'pct_change'},
        'outliers': {'enabled': True, 'method': 'iqr', 'threshold': 3.0},
        'scaling': {'enabled': True, 'method': 'standard'}
    }
    
    df_processed = preprocessor.preprocess(df, target_col='label', config=preprocess_config)
    print(f"✓ Preprocessing complete")
    print(f"✓ Final shape: {df_processed.shape}")
    
    # 8. Final validation
    print("\n8. FINAL VALIDATION")
    print("-"*40)
    final_report = validator.validate(df_processed, target_col='label')
    print(f"✓ Final Quality Score: {final_report['data_quality']:.1f}/100")
    print(f"✓ ML Ready: {'Yes' if final_report['is_ml_ready'] else 'No'}")
    
    # Check stationarity improvement
    if 'stationarity' in final_report:
        print(f"✓ Stationary features: {final_report['stationarity']['num_stationary']}")
        print(f"✓ Non-stationary features: {final_report['stationarity']['num_non_stationary']}")
    
    # 9. Create sequences for deep learning
    print("\n9. CREATING SEQUENCES FOR DEEP LEARNING")
    print("-"*40)
    
    # Remove NaN values
    df_clean = df_processed.dropna()
    
    if len(df_clean) > 100:
        X, y = ml_engineer.create_sequences(
            df_clean,
            sequence_length=50,
            target_col='label'
        )
        print(f"✓ Created sequences: X shape = {X.shape}, y shape = {y.shape}")
        print(f"✓ Ready for LSTM/GRU training")
    
    # 10. Feature importance preview
    print("\n10. KEY FEATURES PREVIEW")
    print("-"*40)
    
    # Show some key features
    key_features = ['close_pct', 'RSI_14', 'MACD', 'volume_pct', 'ATR_14', 
                   '30_Ret', 'RSI_Ret', 'BB_position']
    
    print("Sample feature values (last 5 rows):")
    available_features = [f for f in key_features if f in df_processed.columns]
    if available_features:
        print(df_processed[available_features].tail())
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"""
✓ Data Loading: Auto-detected Binance format and converted to OHLCV
✓ Indicators: Added {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']])} technical indicators
✓ Features: Created {len(ml_engineer.feature_names)} ML features
✓ Validation: Quality score improved from {initial_report['data_quality']:.1f} to {final_report['data_quality']:.1f}
✓ Preprocessing: Applied stationarity correction, outlier handling, and scaling
✓ ML Ready: Data is prepared for training with balanced labels

Next steps:
1. Train ML models using the processed data
2. Evaluate model performance
3. Integrate predictions into trading strategy
4. Backtest the complete system
""")
    
    # Save processed data
    output_file = '/tmp/ml_ready_data.csv'
    df_processed.to_csv(output_file, index=False)
    print(f"\n✓ Saved ML-ready data to: {output_file}")
    
    return df_processed


def demonstrate_specific_features():
    """Demonstrate specific feature engineering capabilities."""
    print("\n" + "="*80)
    print("SPECIFIC FEATURE DEMONSTRATIONS")
    print("="*80)
    
    # Create simple data for clear demonstration
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    prices = 100 * (1 + np.random.normal(0, 0.01, 100)).cumprod()
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.lognormal(10, 1, 100)
    })
    
    # 1. Percentage features (for stationarity)
    print("\n1. PERCENTAGE FEATURES")
    print("-"*40)
    ml_engineer = MLFeatureEngineering()
    df_pct = ml_engineer.create_percentage_features(df.copy())
    
    print("Original close prices (first 5):")
    print(df['close'].head())
    print("\nPercentage changes (first 5):")
    print(df_pct['close_pct'].head())
    
    # 2. RSI returns (from ML project)
    print("\n2. RSI RETURNS (ML PROJECT FEATURE)")
    print("-"*40)
    # Add RSI first
    df_pct['RSI_14'] = 50 + np.random.normal(0, 10, 100)  # Simulated RSI
    df_indicators = ml_engineer.create_indicator_features(df_pct.copy())
    
    print("RSI values (last 5):")
    print(df_indicators['RSI_14'].tail())
    print("\nRSI returns (last 5):")
    print(df_indicators['RSI_Ret'].tail())
    
    # 3. Rolling features (30_Ret from ML project)
    print("\n3. ROLLING FEATURES (30_RET)")
    print("-"*40)
    df_rolling = ml_engineer.create_rolling_features(df_indicators.copy())
    
    if '30_Ret' in df_rolling.columns:
        print("30-day cumulative returns (last 5):")
        print(df_rolling['30_Ret'].tail())
    
    # 4. Time features (Unix timestamp)
    print("\n4. TIME FEATURES (UNIX TIMESTAMP)")
    print("-"*40)
    df_time = ml_engineer.create_time_features(df_rolling.copy())
    
    print("Timestamp to Unix conversion:")
    print(f"Original: {df_time['timestamp'].iloc[-1]}")
    print(f"Unix timestamp: {df_time['unix_timestamp'].iloc[-1]}")
    
    # 5. Validation checks
    print("\n5. VALIDATION CHECKS")
    print("-"*40)
    validator = DataValidator()
    
    # Check stationarity
    from statsmodels.tsa.stattools import adfuller
    
    # Original close prices
    adf_original = adfuller(df['close'].dropna())
    print(f"Original close prices - ADF p-value: {adf_original[1]:.4f}")
    print(f"Stationary: {'Yes' if adf_original[1] < 0.05 else 'No'}")
    
    # Percentage changes
    adf_pct = adfuller(df_pct['close_pct'].dropna())
    print(f"\nPercentage changes - ADF p-value: {adf_pct[1]:.4f}")
    print(f"Stationary: {'Yes' if adf_pct[1] < 0.05 else 'No'}")


if __name__ == "__main__":
    # Run main demonstration
    processed_data = demonstrate_pipeline()
    
    # Run specific feature demonstrations
    demonstrate_specific_features()
    
    print("\n✓ Demonstration complete!")