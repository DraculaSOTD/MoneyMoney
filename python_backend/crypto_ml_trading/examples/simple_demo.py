#!/usr/bin/env python3
"""
Simple demonstration of the integrated ML pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import components
from data.enhanced_data_loader import BinanceDataSource
from data.data_validator import DataValidator
from data.preprocessing import AdvancedPreprocessor
from features.enhanced_technical_indicators import EnhancedTechnicalIndicators
from features.ml_feature_engineering import MLFeatureEngineering
from features.decision_labeler import DecisionLabeler


def create_sample_binance_data():
    """Create sample data in Binance format."""
    n = 500
    base_price = 100
    
    # Generate realistic price movement
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.01, n)
    prices = base_price * (1 + returns).cumprod()
    
    # Create OHLC
    timestamps = pd.date_range(end=datetime.now(), periods=n, freq='1h')
    
    df = pd.DataFrame({
        'Open time': timestamps.astype(np.int64) // 10**9 * 1000,
        'Open': prices * (1 + np.random.normal(0, 0.001, n)),
        'High': prices * (1 + abs(np.random.normal(0, 0.005, n))),
        'Low': prices * (1 - abs(np.random.normal(0, 0.005, n))),
        'Close': prices,
        'Volume': np.random.lognormal(10, 1, n),
        'Close time': (timestamps + pd.Timedelta(hours=1)).astype(np.int64) // 10**9 * 1000,
        'Quote asset volume': np.random.lognormal(15, 1, n),
        'Number of trades': np.random.poisson(100, n),
        'Taker buy base asset volume': np.random.lognormal(9.5, 1, n),
        'Taker buy quote asset volume': np.random.lognormal(14.5, 1, n),
        'Ignore': 0
    })
    
    return df


def main():
    print("="*60)
    print("INTEGRATED ML PIPELINE DEMONSTRATION")
    print("="*60)
    
    # 1. Create sample data
    print("\n1. Creating sample data...")
    df_binance = create_sample_binance_data()
    print(f"✓ Created {len(df_binance)} rows of Binance format data")
    
    # 2. Convert to standard format
    print("\n2. Converting Binance format to OHLCV...")
    data_source = BinanceDataSource('')  # Empty path, we'll use df directly
    df = data_source._convert_binance_to_standard(df_binance)
    print(f"✓ Converted to standard format: {df.columns.tolist()}")
    
    # 3. Add technical indicators
    print("\n3. Adding technical indicators...")
    df = EnhancedTechnicalIndicators.compute_all_indicators(df)
    indicator_count = len([c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])
    print(f"✓ Added {indicator_count} technical indicators")
    
    # 4. Create ML features
    print("\n4. Engineering ML features...")
    ml_eng = MLFeatureEngineering()
    df = ml_eng.prepare_ml_features(df)
    print(f"✓ Created {len(ml_eng.feature_names)} ML features")
    
    # 5. Add labels
    print("\n5. Creating ML labels...")
    labeler = DecisionLabeler()
    df = labeler.create_labels(df, lookforward=5)
    if 'label' in df.columns:
        print(f"✓ Label distribution: {df['label'].value_counts().to_dict()}")
    
    # 6. Validate data
    print("\n6. Validating data quality...")
    validator = DataValidator()
    report = validator.validate(df)
    print(f"✓ Data quality score: {report['data_quality']:.1f}/100")
    print(f"✓ Missing values: {report['missing_values']['total_missing']}")
    
    # 7. Preprocess data
    print("\n7. Preprocessing data...")
    preprocessor = AdvancedPreprocessor()
    df_processed = preprocessor.preprocess(df, target_col='label')
    print(f"✓ Final shape: {df_processed.shape}")
    
    # 8. Final validation
    print("\n8. Final validation...")
    final_report = validator.validate(df_processed)
    print(f"✓ Final quality score: {final_report['data_quality']:.1f}/100")
    print(f"✓ ML ready: {final_report['is_ml_ready']}")
    
    # 9. Show sample features
    print("\n9. Sample processed features:")
    key_features = ['close_pct', 'RSI_14', 'volume_pct', 'MACD', '30_Ret']
    available = [f for f in key_features if f in df_processed.columns]
    if available:
        print(df_processed[available].tail(3).round(4))
    
    print("\n✓ Pipeline demonstration complete!")
    print(f"✓ Data is ready for ML training with {df_processed.shape[0]} samples and {df_processed.shape[1]} features")
    
    return df_processed


if __name__ == "__main__":
    df_final = main()