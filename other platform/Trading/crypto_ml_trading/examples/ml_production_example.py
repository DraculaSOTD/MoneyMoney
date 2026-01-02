#!/usr/bin/env python3
"""
ML Production Trading Example
Shows how to use trained models for live trading decisions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
import json
from datetime import datetime, timedelta
from pathlib import Path

# Import components
from data.enhanced_data_loader import EnhancedDataLoader, UniversalDataSource
from strategies.ml_strategy import MLTradingStrategy, EnsembleMLStrategy
from models.risk_management.risk_manager import RiskManager


class MLProductionTrader:
    """
    Production-ready ML trader that loads trained models and makes predictions.
    """
    
    def __init__(self, model_path: str, config_path: str, preprocessor_path: str = None):
        """
        Initialize production trader.
        
        Args:
            model_path: Path to saved model (.pth file)
            config_path: Path to configuration JSON
            preprocessor_path: Path to saved preprocessor (optional)
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Load preprocessor if provided
        if preprocessor_path and os.path.exists(preprocessor_path):
            import pickle
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
        else:
            from data.preprocessing import AdvancedPreprocessor
            self.preprocessor = AdvancedPreprocessor()
        
        # Create strategy
        self.strategy = MLTradingStrategy(
            model=self.model,
            model_name=self.config.get('model_name', 'lstm'),
            feature_config=self.config.get('features', {}),
            risk_config=self.config.get('risk', {}),
            preprocessor=self.preprocessor,
            sequence_length=self.config.get('sequence_length', 50)
        )
        
        # Risk manager
        self.risk_manager = RiskManager(self.config.get('risk', {}))
        
        print(f"✓ ML Production Trader initialized")
        print(f"  Model: {self.config.get('model_name', 'unknown')}")
        print(f"  Sequence length: {self.config.get('sequence_length', 50)}")
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained model from file."""
        # Import model architectures
        from models.ml.pytorch_models import BiLSTMModel, GRUModel, CNNLSTMModel
        
        # Get model configuration
        model_type = self.config.get('model_name', 'lstm')
        model_config = self.config.get('model_config', {})
        
        # Create model instance
        if model_type == 'lstm':
            model = BiLSTMModel(**model_config)
        elif model_type == 'gru':
            model = GRUModel(**model_config)
        elif model_type == 'cnn_lstm':
            model = CNNLSTMModel(**model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        return model
    
    def get_trading_decision(self, 
                           historical_data: pd.DataFrame,
                           current_positions: dict = None,
                           account_balance: float = 10000) -> dict:
        """
        Get trading decision based on current market data.
        
        Args:
            historical_data: Recent OHLCV data (at least sequence_length + indicator lookback)
            current_positions: Current open positions
            account_balance: Current account balance
            
        Returns:
            Trading decision with action, confidence, position size, etc.
        """
        # Ensure we have enough data
        min_required = self.config.get('sequence_length', 50) + 200  # Extra for indicators
        if len(historical_data) < min_required:
            return {
                'action': 'hold',
                'confidence': 0.0,
                'reason': f'Insufficient data: {len(historical_data)} rows, need {min_required}'
            }
        
        # Get ML signal
        account_info = {'balance': account_balance}
        signal = self.strategy.generate_signal(
            historical_data,
            current_positions or {},
            account_info
        )
        
        # Enhance with risk management
        if signal['action'] != 'hold':
            # Get detailed position sizing
            position_info = self.risk_manager.calculate_position_size(
                signal_strength=signal['confidence'],
                expected_return=signal.get('expected_return', 0.02),
                win_rate=signal.get('win_rate', 0.5),
                current_price=historical_data['close'].iloc[-1],
                account_balance=account_balance,
                existing_positions=current_positions or {}
            )
            
            signal.update(position_info)
        
        # Add metadata
        signal['timestamp'] = datetime.now()
        signal['data_end'] = historical_data.index[-1] if hasattr(historical_data.index, '__getitem__') else str(historical_data.index[-1])
        signal['model'] = self.config.get('model_name', 'unknown')
        
        return signal
    
    def should_close_position(self, position: dict, current_price: float) -> Tuple[bool, str]:
        """
        Check if a position should be closed.
        
        Args:
            position: Position information
            current_price: Current market price
            
        Returns:
            Tuple of (should_close, reason)
        """
        # Check stop loss
        if position.get('stop_loss') and current_price <= position['stop_loss']:
            return True, 'stop_loss_hit'
        
        # Check take profit
        if position.get('take_profit') and current_price >= position['take_profit']:
            return True, 'take_profit_hit'
        
        # Check time-based exit (optional)
        if 'entry_time' in position:
            holding_time = (datetime.now() - position['entry_time']).total_seconds() / 3600
            max_holding_hours = self.config.get('max_holding_hours', 168)  # 1 week default
            
            if holding_time > max_holding_hours:
                return True, 'max_holding_time'
        
        return False, ''


def demonstrate_production_trading():
    """Demonstrate production ML trading setup."""
    print("="*60)
    print("ML PRODUCTION TRADING DEMONSTRATION")
    print("="*60)
    
    # 1. Create mock trained model and config
    print("\n1. Setting up mock production environment...")
    
    # Create config
    config = {
        'model_name': 'lstm',
        'sequence_length': 50,
        'model_config': {
            'input_size': 50,  # Number of features
            'hidden_size': 64,
            'num_layers': 2,
            'num_classes': 3,
            'dropout': 0.2
        },
        'features': {
            'indicators': {
                'sma': {'enabled': True, 'periods': [10, 20, 50]},
                'rsi': {'enabled': True, 'period': 14},
                'macd': {'enabled': True}
            }
        },
        'risk': {
            'max_position_size': 0.2,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04
        }
    }
    
    # Save config
    config_path = 'ml_production_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create and save mock model
    from models.ml.pytorch_models import BiLSTMModel
    model = BiLSTMModel(**config['model_config'])
    model_path = 'mock_lstm_model.pth'
    torch.save(model.state_dict(), model_path)
    
    print(f"✓ Created mock config: {config_path}")
    print(f"✓ Created mock model: {model_path}")
    
    # 2. Initialize production trader
    print("\n2. Initializing production trader...")
    trader = MLProductionTrader(model_path, config_path)
    
    # 3. Generate sample market data
    print("\n3. Generating sample market data...")
    from examples.simple_demo import create_sample_binance_data
    from data.enhanced_data_loader import BinanceDataSource
    
    df_binance = create_sample_binance_data()
    data_source = BinanceDataSource('')
    df_market = data_source._convert_binance_to_standard(df_binance)
    
    print(f"✓ Generated {len(df_market)} hours of market data")
    
    # 4. Simulate trading decisions
    print("\n4. Making trading decisions...")
    
    # Simulate multiple time points
    current_positions = {}
    account_balance = 10000
    
    for i in range(5):
        # Get data up to current point
        current_idx = 300 + i * 10  # Start from 300 to have enough history
        historical_data = df_market.iloc[:current_idx].copy()
        
        print(f"\n--- Time Point {i+1} ---")
        print(f"Current price: ${historical_data['close'].iloc[-1]:.2f}")
        print(f"Account balance: ${account_balance:.2f}")
        print(f"Open positions: {len(current_positions)}")
        
        # Get trading decision
        decision = trader.get_trading_decision(
            historical_data,
            current_positions,
            account_balance
        )
        
        print(f"\nDecision: {decision['action'].upper()}")
        print(f"Confidence: {decision.get('confidence', 0):.2%}")
        
        if decision['action'] != 'hold':
            print(f"Position size: ${decision.get('position_size', 0):.2f}")
            print(f"Stop loss: ${decision.get('stop_loss', 0):.2f}")
            print(f"Take profit: ${decision.get('take_profit', 0):.2f}")
            
            # Simulate position entry
            if decision['action'] == 'buy' and not current_positions:
                current_positions['main'] = {
                    'entry_price': historical_data['close'].iloc[-1],
                    'size': decision.get('position_size', 1000),
                    'stop_loss': decision.get('stop_loss'),
                    'take_profit': decision.get('take_profit'),
                    'entry_time': datetime.now()
                }
                account_balance -= current_positions['main']['size']
                print(f"\n✓ Opened position at ${current_positions['main']['entry_price']:.2f}")
        
        # Check if should close positions
        if current_positions:
            for pos_id, position in list(current_positions.items()):
                should_close, reason = trader.should_close_position(
                    position,
                    historical_data['close'].iloc[-1]
                )
                
                if should_close:
                    # Calculate P&L
                    pnl = (historical_data['close'].iloc[-1] - position['entry_price']) * \
                          (position['size'] / position['entry_price'])
                    
                    print(f"\n✓ Closing position: {reason}")
                    print(f"  P&L: ${pnl:.2f} ({pnl/position['size']*100:.2%})")
                    
                    account_balance += position['size'] + pnl
                    del current_positions[pos_id]
    
    # 5. Performance summary
    print("\n" + "="*60)
    print("TRADING SESSION SUMMARY")
    print("="*60)
    print(f"Final balance: ${account_balance:.2f}")
    print(f"Total P&L: ${account_balance - 10000:.2f} ({(account_balance/10000 - 1)*100:.2%})")
    
    if hasattr(trader.strategy, 'get_performance_summary'):
        perf = trader.strategy.get_performance_summary()
        print(f"\nModel Performance:")
        print(f"  Predictions made: {perf['total_predictions']}")
        print(f"  Average confidence: {perf['avg_confidence']:.2%}")
    
    # Cleanup
    os.remove(config_path)
    os.remove(model_path)
    print("\n✓ Cleaned up temporary files")


def create_model_deployment_package():
    """Create a deployment package for production use."""
    print("\n" + "="*60)
    print("CREATING MODEL DEPLOYMENT PACKAGE")
    print("="*60)
    
    deployment_dir = Path('ml_deployment')
    deployment_dir.mkdir(exist_ok=True)
    
    # 1. Create deployment README
    readme_content = """# ML Trading Model Deployment

## Quick Start

```python
from ml_production_trader import MLProductionTrader

# Initialize trader
trader = MLProductionTrader(
    model_path='models/best_lstm_model.pth',
    config_path='config/production_config.json'
)

# Get trading decision
decision = trader.get_trading_decision(
    historical_data=df_ohlcv,
    current_positions={},
    account_balance=10000
)

print(f"Action: {decision['action']}")
print(f"Confidence: {decision['confidence']:.2%}")
```

## Configuration

The config file should include:
- model_name: Type of model (lstm, gru, cnn_lstm)
- sequence_length: Input sequence length
- model_config: Model architecture parameters
- features: Feature engineering configuration
- risk: Risk management parameters

## Model Files

Place trained model files (.pth) in the models/ directory.

## Requirements

- torch
- pandas
- numpy
- scikit-learn

## API Endpoints (for REST API deployment)

POST /predict
- Input: JSON with OHLCV data
- Output: Trading decision

GET /model/info
- Returns model configuration and status

## Monitoring

Monitor these metrics:
- Prediction latency
- Model confidence distribution
- Prediction accuracy (if ground truth available)
- Resource usage (CPU/Memory)
"""
    
    with open(deployment_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    
    # 2. Create requirements file
    requirements = """torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.10.0
statsmodels>=0.14.0"""
    
    with open(deployment_dir / 'requirements.txt', 'w') as f:
        f.write(requirements)
    
    # 3. Create example config
    example_config = {
        'model_name': 'lstm',
        'sequence_length': 50,
        'model_config': {
            'input_size': 100,
            'hidden_size': 128,
            'num_layers': 2,
            'num_classes': 3,
            'dropout': 0.2
        },
        'features': {
            'indicators': {
                'sma': {'enabled': True, 'periods': [10, 20, 50]},
                'ema': {'enabled': True, 'periods': [12, 26]},
                'rsi': {'enabled': True, 'period': 14},
                'macd': {'enabled': True},
                'bollinger': {'enabled': True},
                'atr': {'enabled': True}
            },
            'percentage_features': True,
            'lagged_features': True,
            'rolling_features': True
        },
        'risk': {
            'max_position_size': 0.2,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'max_correlation': 0.7,
            'target_volatility': 0.15
        },
        'preprocessing': {
            'handle_missing': {'enabled': True, 'method': 'forward_fill'},
            'stationarity': {'enabled': True, 'method': 'pct_change'},
            'scaling': {'enabled': True, 'method': 'standard'}
        }
    }
    
    (deployment_dir / 'config').mkdir(exist_ok=True)
    with open(deployment_dir / 'config' / 'example_config.json', 'w') as f:
        json.dump(example_config, f, indent=2)
    
    print(f"✓ Created deployment package in {deployment_dir}/")
    print(f"  - README.md")
    print(f"  - requirements.txt")
    print(f"  - config/example_config.json")
    
    return deployment_dir


if __name__ == "__main__":
    # Run demonstration
    demonstrate_production_trading()
    
    # Create deployment package
    create_model_deployment_package()
    
    print("\n✓ Production example complete!")