#!/usr/bin/env python3
"""
Demonstration of enhanced feature engineering capabilities.
Shows how to load data in multiple formats and compute comprehensive technical indicators.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from data.enhanced_data_loader import EnhancedDataLoader
from features.enhanced_technical_indicators import EnhancedTechnicalIndicators


def demonstrate_data_loading():
    """Demonstrate loading data from different formats."""
    print("=== Data Loading Demonstration ===\n")
    
    # Initialize enhanced data loader
    loader = EnhancedDataLoader(data_dir="./data/historical")
    
    # Load standard format data
    print("1. Loading standard OHLCV format...")
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        
        df_standard = loader.load_data(
            symbol="BTCUSDT",
            start_time=start_time,
            end_time=end_time,
            interval="1m"
        )
        print(f"   Loaded {len(df_standard)} rows")
        print(f"   Columns: {list(df_standard.columns)}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # Demonstrate Binance format loading
    print("2. Loading Binance format (auto-detection)...")
    print("   The loader automatically detects and converts Binance format")
    print("   Original columns: Open time, Open, High, Low, Close, Volume, etc.")
    print("   Converted to: timestamp, open, high, low, close, volume\n")
    
    return df_standard if 'df_standard' in locals() else None


def demonstrate_technical_indicators(df):
    """Demonstrate comprehensive technical indicator computation."""
    print("=== Technical Indicators Demonstration ===\n")
    
    if df is None or len(df) == 0:
        print("No data available for demonstration")
        return None
    
    # Compute all indicators
    print("Computing comprehensive technical indicators...")
    df_with_indicators = EnhancedTechnicalIndicators.compute_all_indicators(df)
    
    # Show indicator categories
    indicator_categories = {
        'Moving Averages': ['SMA_12', 'SMA_26', 'SMA_50', 'EMA_12', 'EMA_26'],
        'MACD': ['MACD', 'Signal_Line', 'MACD_Histogram'],
        'Momentum': ['RSI_14', '%K', '%D', 'Momentum_14'],
        'Volatility': ['ATR_14', 'Upper_Band', 'Middle_Band', 'Lower_Band', 'BB_Width'],
        'Trend': ['Parabolic_SAR', 'ADX_14', '+DI_14', '-DI_14'],
        'Japanese': ['Tenkan_sen', 'Kijun_sen', 'Senkou_Span_A', 'Senkou_Span_B'],
        'Support/Resistance': ['Pivot_Point', 'Support_1', 'Resistance_1'],
        'Volume': ['CMF_20'],
        'Pattern Detection': ['Bullish_MACD_Divergence', 'Bearish_RSI_Divergence', 'Elliott_Wave_Sequence']
    }
    
    print("\nComputed indicators by category:")
    for category, indicators in indicator_categories.items():
        available = [ind for ind in indicators if ind in df_with_indicators.columns]
        print(f"\n{category}:")
        for ind in available:
            # Get last non-null value
            last_value = df_with_indicators[ind].dropna().iloc[-1] if len(df_with_indicators[ind].dropna()) > 0 else np.nan
            print(f"  - {ind}: {last_value:.4f}" if not np.isnan(last_value) else f"  - {ind}: N/A")
    
    return df_with_indicators


def demonstrate_feature_engineering(df):
    """Demonstrate ML feature preparation."""
    print("\n=== Feature Engineering Demonstration ===\n")
    
    if df is None or len(df) == 0:
        print("No data available for demonstration")
        return None
    
    # Initialize enhanced loader for feature prep
    from data.enhanced_data_loader import EnhancedDataLoader
    loader = EnhancedDataLoader(data_dir="./data/historical")
    
    # Prepare ML features
    print("Preparing features for machine learning...")
    df_ml = loader.prepare_ml_features(df, feature_config={
        'price_features': True,
        'volume_features': True,
        'technical_features': True,
        'time_features': True,
        'lag_features': True,
        'interaction_features': False
    })
    
    # Show new feature categories
    feature_categories = {
        'Price Features': ['returns', 'log_returns', 'price_range', 'price_change', 'price_position'],
        'Volume Features': ['volume_change', 'volume_ma_ratio'],
        'Time Features': ['hour', 'day_of_week', 'is_weekend'],
        'Lag Features': ['returns_lag_1', 'returns_lag_5', 'volume_lag_1']
    }
    
    print("\nML-ready features:")
    for category, features in feature_categories.items():
        available = [feat for feat in features if feat in df_ml.columns]
        if available:
            print(f"\n{category}:")
            for feat in available[:3]:  # Show first 3
                print(f"  - {feat}")
    
    print(f"\nTotal features: {len(df_ml.columns)}")
    print(f"Dataset shape: {df_ml.shape}")
    
    return df_ml


def demonstrate_trading_signals(df):
    """Demonstrate trading signal generation."""
    print("\n=== Trading Signal Generation ===\n")
    
    if df is None or len(df) == 0:
        print("No data available for demonstration")
        return
    
    # Simple rule-based signals
    buy_conditions = 0
    sell_conditions = 0
    
    # Check last row for signal conditions
    last_row = df.iloc[-1]
    
    print("Checking trading conditions (last data point):")
    print(f"Timestamp: {df['timestamp'].iloc[-1] if 'timestamp' in df.columns else 'N/A'}")
    print(f"Close Price: {last_row['close']:.2f}")
    
    # Buy conditions
    print("\nBuy Conditions:")
    if 'RSI_14' in df.columns and not pd.isna(last_row['RSI_14']):
        if last_row['RSI_14'] < 30:
            print("  ✓ RSI < 30 (Oversold)")
            buy_conditions += 1
        else:
            print(f"  ✗ RSI = {last_row['RSI_14']:.2f} (Not oversold)")
    
    if 'MACD' in df.columns and 'Signal_Line' in df.columns:
        if not pd.isna(last_row['MACD']) and not pd.isna(last_row['Signal_Line']):
            if last_row['MACD'] > last_row['Signal_Line']:
                print("  ✓ MACD > Signal Line (Bullish crossover)")
                buy_conditions += 1
            else:
                print("  ✗ MACD < Signal Line")
    
    if 'close' in df.columns and 'Lower_Band' in df.columns:
        if not pd.isna(last_row['Lower_Band']):
            if last_row['close'] < last_row['Lower_Band']:
                print("  ✓ Price below Lower Bollinger Band")
                buy_conditions += 1
            else:
                print("  ✗ Price above Lower Bollinger Band")
    
    # Sell conditions
    print("\nSell Conditions:")
    if 'RSI_14' in df.columns and not pd.isna(last_row['RSI_14']):
        if last_row['RSI_14'] > 70:
            print("  ✓ RSI > 70 (Overbought)")
            sell_conditions += 1
        else:
            print(f"  ✗ RSI = {last_row['RSI_14']:.2f} (Not overbought)")
    
    if 'close' in df.columns and 'Upper_Band' in df.columns:
        if not pd.isna(last_row['Upper_Band']):
            if last_row['close'] > last_row['Upper_Band']:
                print("  ✓ Price above Upper Bollinger Band")
                sell_conditions += 1
            else:
                print("  ✗ Price below Upper Bollinger Band")
    
    # Decision
    print(f"\nSignal Summary:")
    print(f"Buy conditions met: {buy_conditions}")
    print(f"Sell conditions met: {sell_conditions}")
    
    if buy_conditions > sell_conditions and buy_conditions >= 2:
        print("📈 SIGNAL: BUY")
    elif sell_conditions > buy_conditions and sell_conditions >= 2:
        print("📉 SIGNAL: SELL")
    else:
        print("⏸️  SIGNAL: HOLD")


def visualize_indicators(df):
    """Create a simple visualization of price and indicators."""
    print("\n=== Creating Visualization ===\n")
    
    if df is None or len(df) < 100:
        print("Insufficient data for visualization")
        return
    
    # Use last 500 points for clarity
    df_plot = df.tail(500)
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Price and Bollinger Bands
    ax1 = axes[0]
    ax1.plot(df_plot.index, df_plot['close'], label='Close', color='black', linewidth=1)
    if 'Upper_Band' in df_plot.columns:
        ax1.plot(df_plot.index, df_plot['Upper_Band'], label='Upper BB', color='red', alpha=0.5)
        ax1.plot(df_plot.index, df_plot['Lower_Band'], label='Lower BB', color='green', alpha=0.5)
        ax1.fill_between(df_plot.index, df_plot['Upper_Band'], df_plot['Lower_Band'], alpha=0.1)
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.set_title('Price with Bollinger Bands')
    
    # MACD
    ax2 = axes[1]
    if 'MACD' in df_plot.columns and 'Signal_Line' in df_plot.columns:
        ax2.plot(df_plot.index, df_plot['MACD'], label='MACD', color='blue')
        ax2.plot(df_plot.index, df_plot['Signal_Line'], label='Signal', color='red')
        ax2.bar(df_plot.index, df_plot['MACD'] - df_plot['Signal_Line'], alpha=0.3, label='Histogram')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_ylabel('MACD')
    ax2.legend(loc='upper left')
    
    # RSI
    ax3 = axes[2]
    if 'RSI_14' in df_plot.columns:
        ax3.plot(df_plot.index, df_plot['RSI_14'], label='RSI(14)', color='purple')
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
        ax3.set_ylim(0, 100)
    ax3.set_ylabel('RSI')
    ax3.legend(loc='upper left')
    
    # Volume
    ax4 = axes[3]
    ax4.bar(df_plot.index, df_plot['volume'], alpha=0.5, color='gray')
    ax4.set_ylabel('Volume')
    ax4.set_xlabel('Time Period')
    
    plt.tight_layout()
    
    # Save plot
    output_path = './examples/indicator_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


def main():
    """Run the complete demonstration."""
    print("="*60)
    print("ENHANCED FEATURE ENGINEERING DEMONSTRATION")
    print("="*60)
    print()
    
    # Load data
    df = demonstrate_data_loading()
    
    if df is not None and len(df) > 0:
        # Compute indicators
        df_indicators = demonstrate_technical_indicators(df)
        
        # Prepare ML features
        df_ml = demonstrate_feature_engineering(df_indicators)
        
        # Generate trading signals
        demonstrate_trading_signals(df_indicators)
        
        # Create visualization
        visualize_indicators(df_indicators)
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Auto-detection of data formats (Binance/Standard)")
        print("✓ 40+ technical indicators computed")
        print("✓ Pattern detection (divergences, Elliott waves)")
        print("✓ Japanese indicators (Ichimoku cloud)")
        print("✓ ML feature preparation")
        print("✓ Trading signal generation")
        print("✓ Visualization capabilities")
    else:
        print("\nNo data available. Please ensure data files exist in ./data/historical/")
        print("Expected format: SYMBOL_INTERVAL.csv (e.g., BTCUSDT_1m.csv)")


if __name__ == "__main__":
    main()