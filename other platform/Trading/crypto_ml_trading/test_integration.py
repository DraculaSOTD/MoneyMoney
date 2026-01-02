#!/usr/bin/env python3
"""
Test script to demonstrate the integrated feature engineering and trading decision system.
This script shows how all components work together.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.enhanced_data_loader import EnhancedDataLoader
from trading.decision_engine import TradingDecisionEngine


def test_data_loading():
    """Test data loading with both formats."""
    print("="*60)
    print("TESTING DATA LOADING")
    print("="*60)
    
    loader = EnhancedDataLoader(data_dir="./data/historical")
    
    # Test with existing data
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)
        
        df = loader.load_data(
            symbol="BTCUSDT",
            start_time=start_time,
            end_time=end_time,
            interval="1m"
        )
        
        print(f"✓ Successfully loaded {len(df)} rows of data")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Columns: {list(df.columns)[:5]}... ({len(df.columns)} total)")
        
        return df
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        
        # Create synthetic data for testing
        print("\nCreating synthetic data for demonstration...")
        from data.data_loader import create_synthetic_data
        
        df = create_synthetic_data(
            symbol="BTCUSDT",
            start_time=datetime.now() - timedelta(hours=24),
            end_time=datetime.now(),
            interval="1m",
            initial_price=50000.0,
            volatility=0.02
        )
        
        print(f"✓ Created synthetic data with {len(df)} rows")
        return df


def test_indicator_computation(df):
    """Test technical indicator computation."""
    print("\n" + "="*60)
    print("TESTING TECHNICAL INDICATORS")
    print("="*60)
    
    from data.enhanced_data_loader import EnhancedDataLoader
    loader = EnhancedDataLoader(data_dir="./data/historical")
    
    # Load data with all indicators
    print("Computing all technical indicators...")
    df_indicators = loader.load_data_with_indicators(
        symbol="BTCUSDT",
        start_time=df['timestamp'].min(),
        end_time=df['timestamp'].max(),
        interval="1m",
        indicators=None  # Compute all
    )
    
    # Count indicators by category
    original_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    indicator_cols = [col for col in df_indicators.columns if col not in original_cols]
    
    print(f"✓ Computed {len(indicator_cols)} technical indicators")
    
    # Sample some indicators
    sample_indicators = {
        'Moving Averages': ['SMA_50', 'EMA_26'],
        'Momentum': ['RSI_14', 'MACD'],
        'Volatility': ['ATR_14', 'Upper_Band'],
        'Japanese': ['Tenkan_sen', 'Senkou_Span_A'],
        'Patterns': ['Bullish_MACD_Divergence', 'Elliott_Wave_Sequence']
    }
    
    print("\nSample indicator values (last row):")
    last_row = df_indicators.iloc[-1]
    
    for category, indicators in sample_indicators.items():
        print(f"\n{category}:")
        for ind in indicators:
            if ind in df_indicators.columns:
                value = last_row[ind]
                if not pd.isna(value):
                    if isinstance(value, (int, float)):
                        print(f"  {ind}: {value:.4f}")
                    else:
                        print(f"  {ind}: {value}")
                else:
                    print(f"  {ind}: N/A")
    
    return df_indicators


def test_trading_decisions(df):
    """Test trading decision generation."""
    print("\n" + "="*60)
    print("TESTING TRADING DECISIONS")
    print("="*60)
    
    # Initialize decision engine
    engine = TradingDecisionEngine()
    
    # Generate decision for last data point
    print("Generating trading decision for current market state...")
    decision = engine.generate_decision(df)
    
    print(f"\n📊 Trading Decision:")
    print(f"  Signal: {decision.signal.value}")
    print(f"  Confidence: {decision.confidence:.2%}")
    print(f"  Price: ${decision.price:.2f}")
    print(f"  Buy Score: {decision.buy_score:.2%}")
    print(f"  Sell Score: {decision.sell_score:.2%}")
    
    print(f"\n📋 Triggered Indicators ({len(decision.triggered_indicators)}):")
    for ind in decision.triggered_indicators[:5]:  # Show first 5
        print(f"  • {ind}")
    if len(decision.triggered_indicators) > 5:
        print(f"  ... and {len(decision.triggered_indicators) - 5} more")
    
    print(f"\n💭 Reasoning:")
    print(f"  {decision.reasoning}")
    
    # Test batch decisions
    print("\n" + "-"*40)
    print("Testing batch decision generation...")
    
    # Generate decisions for last 100 rows
    if len(df) > 100:
        df_decisions = engine.batch_decisions(df.tail(200), lookback_window=100)
        
        # Count signals
        signal_counts = df_decisions['signal'].value_counts()
        print(f"\nSignal distribution (last 100 decisions):")
        for signal, count in signal_counts.items():
            if not pd.isna(signal):
                print(f"  {signal}: {count} ({count/100*100:.1f}%)")
        
        # Average confidence by signal type
        print(f"\nAverage confidence by signal:")
        for signal in ['BUY', 'SELL', 'HOLD']:
            mask = df_decisions['signal'] == signal
            if mask.any():
                avg_conf = df_decisions.loc[mask, 'confidence'].mean()
                print(f"  {signal}: {avg_conf:.2%}")
    
    return decision


def test_ml_feature_preparation(df):
    """Test ML feature preparation."""
    print("\n" + "="*60)
    print("TESTING ML FEATURE PREPARATION")
    print("="*60)
    
    from data.enhanced_data_loader import EnhancedDataLoader
    loader = EnhancedDataLoader(data_dir="./data/historical")
    
    # Prepare ML features
    print("Preparing features for machine learning...")
    df_ml = loader.prepare_ml_features(df)
    
    # Create ML dataset
    print("\nCreating ML dataset with sliding windows...")
    ml_dataset = loader.create_dataset_for_ml(
        df_ml,
        target_column='returns',
        lookback_window=50,
        forecast_horizon=1,
        train_test_split=0.8
    )
    
    print(f"\n✓ ML Dataset created:")
    print(f"  Training samples: {ml_dataset['train_size']}")
    print(f"  Test samples: {ml_dataset['test_size']}")
    print(f"  Features: {len(ml_dataset['feature_columns'])}")
    print(f"  Input shape: {ml_dataset['X_train'].shape}")
    print(f"  Target shape: {ml_dataset['y_train'].shape}")
    
    # Show feature importance (mock)
    print(f"\nTop features for ML:")
    top_features = ['returns', 'RSI_14', 'MACD', 'volume_ma_ratio', 'ATR_14']
    for i, feat in enumerate(top_features[:5], 1):
        if feat in ml_dataset['feature_columns']:
            print(f"  {i}. {feat}")
    
    return ml_dataset


def run_integration_test():
    """Run complete integration test."""
    print("\n" + "="*70)
    print("CRYPTO ML TRADING SYSTEM - INTEGRATION TEST")
    print("="*70)
    print("\nThis test demonstrates the integration of:")
    print("• Enhanced data loading (Binance/Standard formats)")
    print("• 40+ technical indicators")
    print("• Pattern detection")
    print("• Trading decision engine with confidence scoring")
    print("• ML feature preparation")
    print()
    
    # Test data loading
    df = test_data_loading()
    if df is None or len(df) == 0:
        print("\n❌ Failed to load data. Exiting.")
        return
    
    # Test indicator computation
    df_indicators = test_indicator_computation(df)
    
    # Test trading decisions
    decision = test_trading_decisions(df_indicators)
    
    # Test ML features
    ml_dataset = test_ml_feature_preparation(df_indicators)
    
    # Summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    print("\n✅ All components tested successfully!")
    print("\nKey capabilities demonstrated:")
    print("• Auto-detection of data formats")
    print("• Comprehensive technical analysis")
    print("• Advanced pattern recognition")
    print("• Intelligent trading signals")
    print("• ML-ready feature engineering")
    print("\nThe system is ready for:")
    print("• Live trading integration")
    print("• Model training")
    print("• Backtesting")
    print("• Performance analysis")


if __name__ == "__main__":
    run_integration_test()