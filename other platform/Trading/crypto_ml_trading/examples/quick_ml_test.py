#!/usr/bin/env python3
"""Quick ML pipeline test with minimal data"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch

# Import components
from training.unified_ml_pipeline import UnifiedMLPipeline
from data.enhanced_data_loader import BinanceDataSource

def create_minimal_data():
    """Create minimal data for quick testing."""
    n = 500  # Minimal samples
    timestamps = pd.date_range(end=datetime.now(), periods=n, freq='1h')
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    return pd.DataFrame({
        'Open time': timestamps.astype(np.int64) // 10**9 * 1000,
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.lognormal(10, 1, n),
        'Close time': (timestamps + pd.Timedelta(hours=1)).astype(np.int64) // 10**9 * 1000,
        'Quote asset volume': np.random.lognormal(15, 1, n),
        'Number of trades': np.random.poisson(100, n),
        'Taker buy base asset volume': np.random.lognormal(9.5, 1, n),
        'Taker buy quote asset volume': np.random.lognormal(14.5, 1, n),
        'Ignore': 0
    })

# Test configuration - minimal for quick test
config = {
    'data': {
        'sequence_length': 20,  # Very short sequences
        'lookforward': 3,
        'threshold': 0.001,
        'train_split': 0.6,
        'val_split': 0.2,
        'test_split': 0.2
    },
    'models': {
        'lstm': {
            'enabled': True,
            'hidden_size': 16,  # Very small
            'num_layers': 1,
            'dropout': 0.1,
            'bidirectional': False
        },
        'gru': {'enabled': False},  # Disable others for quick test
        'cnn_lstm': {'enabled': False}
    },
    'training': {
        'batch_size': 16,
        'epochs': 5,  # Very few epochs
        'learning_rate': 0.001,
        'early_stopping_patience': 2,
        'device': 'cpu'  # Force CPU
    },
    'preprocessing': {
        'handle_missing': {'enabled': True, 'method': 'forward_fill'},
        'handle_infinite': {'enabled': True, 'method': 'clip'},
        'stationarity': {'enabled': True, 'method': 'pct_change'},
        'outliers': {'enabled': False},  # Skip for speed
        'scaling': {'enabled': True, 'method': 'standard'}
    },
    'indicators': {
        'sma': {'enabled': True, 'periods': [10, 20]},
        'rsi': {'enabled': True, 'period': 14},
        'macd': {'enabled': True},
        # Disable others for speed
        'ema': {'enabled': False},
        'bollinger': {'enabled': False},
        'atr': {'enabled': False},
        'adx': {'enabled': False},
        'stochastic': {'enabled': False},
        'ichimoku': {'enabled': False},
        'parabolic_sar': {'enabled': False},
        'fibonacci': {'enabled': False}
    }
}

print("ML Pipeline Quick Test")
print("="*40)
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {config['training']['device']}")

# Create data
print("\nCreating test data...")
df_binance = create_minimal_data()
print(f"✓ Created {len(df_binance)} samples")

# Convert to standard format
from data.enhanced_data_loader import BinanceDataSource
data_source = BinanceDataSource('')
df = data_source._convert_binance_to_standard(df_binance)
print(f"✓ Converted to standard format")

# Create and run pipeline
print("\nRunning minimal pipeline...")
pipeline = UnifiedMLPipeline()
pipeline.config = config

try:
    results = pipeline.run_pipeline(df)
    print("\n✓ Pipeline test successful!")
    
    # Print quick results
    for model_name, model_results in results.items():
        acc = model_results['evaluation']['classification_report']['accuracy']
        print(f"  {model_name}: {acc:.3f} accuracy")
        
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()