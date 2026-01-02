"""
Test Data Loader
================

Test script to verify data loading and indicator computation works with real BTCUSDT data.
"""

import sys
from pathlib import Path

# Add crypto_ml_trading to path
sys.path.append(str(Path(__file__).parent / 'crypto_ml_trading'))

from services.data_loader import DataLoader, load_btcusdt_data, get_available_indicators

def main():
    print("=" * 80)
    print("DATA LOADER TEST - Using Real BTCUSDT Data")
    print("=" * 80)

    # Test 1: Load raw BTCUSDT data from CSV
    print("\n[Test 1] Loading BTCUSDT data from CSV...")
    loader = DataLoader()

    try:
        df_raw = loader.load_from_csv('BTCUSDT', limit=100)
        print(f"✓ Loaded {len(df_raw)} candles")
        print(f"  Columns: {', '.join(df_raw.columns.tolist())}")
        print(f"  Date range: {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")
        print(f"\n  Sample data (first 3 rows):")
        print(df_raw.head(3))
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return

    # Test 2: Compute all indicators
    print("\n\n[Test 2] Computing technical indicators...")
    try:
        df_with_indicators = loader.compute_indicators(df_raw)
        print(f"✓ Computed indicators successfully")
        print(f"  Total columns: {len(df_with_indicators.columns)}")
        print(f"  Original OHLCV: 6 columns")
        print(f"  Indicators added: {len(df_with_indicators.columns) - 6} columns")

        # List all indicators
        base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        indicators = [col for col in df_with_indicators.columns if col not in base_cols]

        print(f"\n  All {len(indicators)} indicators:")
        for i, ind in enumerate(indicators, 1):
            print(f"    {i:2d}. {ind}")

    except Exception as e:
        print(f"✗ Error computing indicators: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 3: Get feature info
    print("\n\n[Test 3] Feature information...")
    try:
        feature_info = loader.get_feature_info(df_with_indicators)
        print(f"✓ Feature info retrieved:")
        print(f"  Total features: {feature_info['total_features']}")
        print(f"  Total rows: {feature_info['total_rows']}")
        print(f"  Features with NaN: {feature_info['features_with_nan']}")
        print(f"  Max NaN count: {feature_info['max_nan_count']}")
        print(f"  Memory usage: {feature_info['memory_usage_mb']} MB")
    except Exception as e:
        print(f"✗ Error getting feature info: {e}")

    # Test 4: Prepare for inference
    print("\n\n[Test 4] Preparing data for ML inference...")
    try:
        feature_array, feature_names = loader.prepare_for_inference(df_with_indicators)
        print(f"✓ Data prepared for inference:")
        print(f"  Array shape: {feature_array.shape}")
        print(f"  Number of features: {len(feature_names)}")
        print(f"  Number of samples: {feature_array.shape[0]}")
        print(f"\n  Sample feature values (last row):")
        sample = feature_array[-1]
        for i, (name, value) in enumerate(zip(feature_names[:10], sample[:10])):
            print(f"    {name}: {value:.4f}")
        if len(feature_names) > 10:
            print(f"    ... and {len(feature_names) - 10} more features")

    except Exception as e:
        print(f"✗ Error preparing for inference: {e}")
        import traceback
        traceback.print_exc()

    # Test 5: Quick load utility
    print("\n\n[Test 5] Testing quick load utility...")
    try:
        df_quick = load_btcusdt_data(limit=50, with_indicators=True)
        print(f"✓ Quick load successful:")
        print(f"  Loaded {len(df_quick)} candles with {len(df_quick.columns)} columns")
    except Exception as e:
        print(f"✗ Error with quick load: {e}")

    # Test 6: Get available indicators list
    print("\n\n[Test 6] Getting list of available indicators...")
    try:
        indicators_list = get_available_indicators()
        print(f"✓ Retrieved {len(indicators_list)} available indicators")
        print("\n  Categorized indicators:")

        # Categorize
        ma_indicators = [i for i in indicators_list if any(x in i for x in ['SMA', 'EMA', 'WMA', 'VWMA'])]
        macd_indicators = [i for i in indicators_list if 'MACD' in i or 'EMA' in i]
        oscillators = [i for i in indicators_list if any(x in i for x in ['RSI', '%', 'Stochastic'])]
        volatility = [i for i in indicators_list if any(x in i for x in ['BB', 'ATR', 'Bollinger'])]
        trend = [i for i in indicators_list if any(x in i for x in ['SAR', 'ADX', 'DI', 'Ichimoku', 'Tenkan', 'Kijun', 'Senkou', 'Chikou'])]
        volume_ind = [i for i in indicators_list if 'CMF' in i or 'Volume' in i]
        support_res = [i for i in indicators_list if any(x in i for x in ['PP', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3', 'Support', 'Resistance', 'Pivot'])]
        divergence = [i for i in indicators_list if 'Divergence' in i]
        elliott = [i for i in indicators_list if 'Elliott' in i or 'Wave' in i]
        other = [i for i in indicators_list if i not in ma_indicators + macd_indicators + oscillators + volatility + trend + volume_ind + support_res + divergence + elliott]

        print(f"\n  Moving Averages ({len(ma_indicators)}): {', '.join(ma_indicators)}")
        print(f"  MACD Related ({len(macd_indicators)}): {', '.join(macd_indicators)}")
        print(f"  Oscillators ({len(oscillators)}): {', '.join(oscillators)}")
        print(f"  Volatility ({len(volatility)}): {', '.join(volatility)}")
        print(f"  Trend ({len(trend)}): {', '.join(trend)}")
        print(f"  Volume ({len(volume_ind)}): {', '.join(volume_ind)}")
        print(f"  Support/Resistance ({len(support_res)}): {', '.join(support_res)}")
        print(f"  Divergences ({len(divergence)}): {', '.join(divergence)}")
        print(f"  Elliott Waves ({len(elliott)}): {', '.join(elliott)}")
        if other:
            print(f"  Other ({len(other)}): {', '.join(other)}")

    except Exception as e:
        print(f"✗ Error getting indicators list: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("DATA LOADER TEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
