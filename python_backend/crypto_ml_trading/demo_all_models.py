"""
Demonstration of all implemented models in the Multi-Model ML Trading System.

This script shows how each model works and how they can be integrated.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all models
from data.data_loader import create_synthetic_data
from features.technical_indicators import TechnicalIndicators
from features.market_microstructure import MarketMicrostructureFeatures

# Statistical Models
from models.statistical.arima.arima_model import ARIMA, AutoARIMA
from models.statistical.garch.garch_model import GARCH

# Machine Learning Models
from models.unsupervised.hmm.hmm_model import RegimeDetector
from models.unsupervised.hmm.trainer import HMMTrainer

# Deep Learning Models
from models.deep_learning.gru_attention.model import GRUAttentionModel
from models.deep_learning.gru_attention.trainer import GRUAttentionTrainer
from models.deep_learning.cnn_pattern.cnn_model import CNNPatternRecognizer
from models.deep_learning.cnn_pattern.pattern_generator import PatternGenerator
from models.deep_learning.cnn_pattern.trainer import CNNPatternTrainer

# Reinforcement Learning
from models.reinforcement.ppo.ppo_agent import PPOAgent
from models.reinforcement.ppo.trading_env import TradingEnvironment
from models.reinforcement.ppo.trainer import PPOTrainer

# Risk Management
from models.risk_management.risk_manager import RiskManager

# Backtesting
from backtesting.engine import BacktestEngine, BacktestConfig
from backtesting.walk_forward import WalkForwardAnalysis
from backtesting.metrics import PerformanceMetrics


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title} ")
    print("=" * 70 + "\n")


def demo_data_preparation():
    """Demonstrate data loading and feature engineering."""
    print_section("DATA PREPARATION AND FEATURE ENGINEERING")
    
    # Create synthetic data for demonstration
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    print("Creating synthetic cryptocurrency data...")
    data = create_synthetic_data(
        symbol="BTCUSDT",
        start_time=start_time,
        end_time=end_time,
        interval="1m"
    )
    
    print(f"Generated {len(data)} data points")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    
    # Calculate technical indicators
    print("\nCalculating technical indicators...")
    data = TechnicalIndicators.calculate_all_indicators(data)
    
    # Calculate market microstructure features
    print("Calculating market microstructure features...")
    data = MarketMicrostructureFeatures.calculate_all_microstructure_features(data)
    
    print(f"\nTotal features: {len(data.columns)}")
    print("Sample features:", list(data.columns)[:10])
    
    return data


def demo_arima_model(data: pd.DataFrame):
    """Demonstrate ARIMA model."""
    print_section("ARIMA MODEL - TIME SERIES FORECASTING")
    
    # Prepare data
    returns = data['returns'].dropna().values[-1000:]  # Use last 1000 points
    
    # Auto ARIMA selection
    print("Running AutoARIMA to find optimal parameters...")
    auto_arima = AutoARIMA(max_p=3, max_d=1, max_q=3)
    arima_model = auto_arima.fit(returns)
    
    print(f"Selected model: ARIMA{auto_arima.best_params}")
    print(f"AIC: {auto_arima.best_aic:.4f}")
    print(f"BIC: {auto_arima.best_bic:.4f}")
    
    # Make predictions
    forecast, lower, upper = arima_model.predict(steps=60, return_conf_int=True)
    
    print(f"\n60-minute ahead forecast:")
    print(f"Mean return: {np.mean(forecast):.6f}")
    print(f"Volatility: {np.std(forecast):.6f}")
    print(f"95% CI range: [{np.mean(lower):.6f}, {np.mean(upper):.6f}]")
    
    return arima_model


def demo_garch_model(data: pd.DataFrame):
    """Demonstrate GARCH model."""
    print_section("GARCH MODEL - VOLATILITY MODELING")
    
    # Prepare data
    returns = data['returns'].dropna().values[-1000:]
    
    # Fit GARCH model
    print("Fitting GARCH(1,1) model...")
    garch = GARCH(p=1, q=1, dist='t')
    garch.fit(returns)
    
    # Model summary
    summary = garch.summary()
    print(f"\nModel: {summary['model']}")
    print(f"Persistence: {summary['persistence']:.4f}")
    print(f"Unconditional volatility: {np.sqrt(summary['unconditional_variance']):.4%}")
    
    # Volatility forecast
    vol_forecast = garch.forecast(steps=60)
    print(f"\nVolatility forecast (next hour):")
    print(f"Current: {vol_forecast['volatility'][0]:.4%}")
    print(f"Average: {np.mean(vol_forecast['volatility']):.4%}")
    print(f"Max: {np.max(vol_forecast['volatility']):.4%}")
    
    # Risk metrics
    var_95, cvar_95 = garch.calculate_var(confidence_level=0.95)
    print(f"\nRisk metrics:")
    print(f"VaR (95%): {var_95:.2%}")
    print(f"CVaR (95%): {cvar_95:.2%}")
    
    return garch


def demo_hmm_regime_detection(data: pd.DataFrame):
    """Demonstrate HMM for regime detection."""
    print_section("HIDDEN MARKOV MODEL - REGIME DETECTION")
    
    # Initialize and train HMM
    print("Training HMM for market regime detection...")
    hmm_trainer = HMMTrainer(n_states=None, feature_set='standard')
    
    # Train with limited data for demo
    results = hmm_trainer.train(data.tail(5000), validation_split=0.2, cv_folds=2)
    
    print(f"\nDetected {results['n_states']} market regimes")
    print(f"Current regime: {results['regime_analysis']['current_regime']}")
    
    # Print regime statistics
    print("\nRegime characteristics:")
    for regime, stats in results['regime_analysis']['regime_statistics'].items():
        print(f"\n{regime.upper()}:")
        print(f"  Frequency: {stats['frequency']:.1%}")
        print(f"  Mean return: {stats['mean_return']*100:.3f}%")
        print(f"  Volatility: {stats['volatility']*100:.1f}%")
        print(f"  Sharpe: {stats['sharpe_ratio']:.2f}")
    
    return hmm_trainer


def demo_gru_attention(data: pd.DataFrame):
    """Demonstrate GRU with Attention model."""
    print_section("GRU WITH ATTENTION - DEEP LEARNING")
    
    # Initialize model
    print("Initializing GRU-Attention model...")
    gru_model = GRUAttentionModel(
        input_size=20,  # Number of features
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_actions=3  # Buy, Hold, Sell
    )
    
    # Prepare sample data
    n_samples = 100
    sequence_length = 60
    n_features = 20
    
    # Create dummy training data
    X = np.random.randn(n_samples, sequence_length, n_features)
    y = np.random.randint(0, 3, n_samples)  # Random actions
    
    print(f"Model architecture:")
    print(f"  Input: ({sequence_length}, {n_features})")
    print(f"  Hidden layers: {gru_model.num_layers}")
    print(f"  Hidden size: {gru_model.hidden_size}")
    print(f"  Attention heads: {gru_model.num_heads}")
    print(f"  Output actions: {gru_model.num_actions}")
    
    # Quick training demo
    trainer = GRUAttentionTrainer(gru_model, batch_size=16)
    
    print("\nTraining for 5 epochs (demo)...")
    history = trainer.train(X, y, epochs=5, verbose=0)
    
    print(f"Final training accuracy: {history['train_accuracy'][-1]:.2%}")
    
    # Make predictions
    test_sequence = np.random.randn(1, sequence_length, n_features)
    predictions = gru_model.predict(test_sequence)
    
    print(f"\nSample prediction:")
    print(f"  Action: {['Buy', 'Hold', 'Sell'][predictions['actions'][0]]}")
    print(f"  Confidence: {predictions['confidence'][0]:.2%}")
    
    return gru_model


def demo_cnn_pattern_recognition(data: pd.DataFrame):
    """Demonstrate CNN for pattern recognition."""
    print_section("CNN PATTERN RECOGNITION - CHART PATTERNS")
    
    # Initialize pattern generator
    print("Generating pattern images from price data...")
    pattern_gen = PatternGenerator(image_size=64, methods=['gasf', 'gadf'])
    
    # Generate some pattern images
    pattern_images = pattern_gen.generate_pattern_images(data.tail(200), window_size=60)
    print(f"Generated {len(pattern_images)} pattern images")
    print(f"Image shape: {pattern_images[0].shape}")
    
    # Initialize CNN model
    cnn_model = CNNPatternRecognizer(
        input_channels=pattern_images.shape[1],
        num_classes=5,
        image_size=64
    )
    
    print(f"\nCNN Architecture:")
    print(f"  Input channels: {cnn_model.input_channels}")
    print(f"  Pattern classes: {cnn_model.num_classes}")
    print(f"  Conv layers: 3")
    print(f"  FC layers: 3")
    
    # Detect candlestick patterns
    patterns = pattern_gen.create_candlestick_patterns(data.tail(500))
    
    print(f"\nDetected candlestick patterns:")
    for pattern_name, pattern_data in patterns.items():
        if len(pattern_data) > 0:
            print(f"  {pattern_name}: {len(pattern_data)} instances")
    
    # Make prediction on sample image
    if len(pattern_images) > 0:
        sample_image = pattern_images[0:1]
        pattern_confidence = cnn_model.get_pattern_confidence(sample_image)
        
        print(f"\nPattern analysis (untrained model):")
        for pattern, conf in pattern_confidence.items():
            print(f"  {pattern}: {conf:.2%}")
    
    return cnn_model


def demo_ppo_reinforcement_learning(data: pd.DataFrame):
    """Demonstrate PPO reinforcement learning."""
    print_section("PPO REINFORCEMENT LEARNING - ADAPTIVE TRADING")
    
    # Create trading environment
    print("Setting up trading environment...")
    env = TradingEnvironment(
        data=data.tail(1000),
        initial_capital=100000,
        lookback_window=60
    )
    
    print(f"Environment configuration:")
    print(f"  Initial capital: ${env.initial_capital:,.0f}")
    print(f"  Commission rate: {env.commission_rate:.1%}")
    print(f"  Max position: {env.max_position:.0%}")
    print(f"  Observation shape: {env.observation_shape}")
    
    # Initialize PPO agent
    ppo_agent = PPOAgent(
        state_dim=env.observation_shape,
        action_dim=env.action_dim,
        hidden_size=128
    )
    
    print(f"\nPPO Agent architecture:")
    print(f"  Actor network: 2 hidden layers (128 units)")
    print(f"  Critic network: 2 hidden layers (128 units)")
    print(f"  Action space: continuous (position, stop_loss, take_profit)")
    
    # Quick training demo
    trainer = PPOTrainer(
        agent=ppo_agent,
        env=env,
        n_steps=128,
        batch_size=32,
        n_epochs=3
    )
    
    print("\nTraining for 5 iterations (demo)...")
    history = trainer.train(n_iterations=5, verbose=0)
    
    # Evaluate
    print("\nEvaluating trained agent...")
    metrics = trainer.evaluate(n_episodes=3)
    
    print(f"Evaluation results:")
    print(f"  Mean return: {metrics['mean_return']:.2%}")
    print(f"  Win rate: {metrics['win_rate']:.1%}")
    print(f"  Avg trades: {metrics['mean_trades']:.0f}")
    
    return ppo_agent, env


def demo_risk_management(data: pd.DataFrame):
    """Demonstrate risk management system."""
    print_section("RISK MANAGEMENT SYSTEM")
    
    # Initialize risk manager
    risk_manager = RiskManager(
        max_position_size=0.20,
        max_portfolio_risk=0.05,
        max_drawdown=0.15,
        var_confidence=0.95,
        kelly_fraction=0.25
    )
    
    print("Risk management parameters:")
    print(f"  Max position size: {risk_manager.max_position_size:.0%}")
    print(f"  Max portfolio risk (VaR): {risk_manager.max_portfolio_risk:.0%}")
    print(f"  Max drawdown: {risk_manager.max_drawdown:.0%}")
    print(f"  Kelly fraction: {risk_manager.kelly_fraction:.0%}")
    
    # Example signal
    signal = {
        'action': 'buy',
        'confidence': 0.75,
        'expected_return': 0.02,
        'win_rate': 0.60,
        'win_loss_ratio': 1.5
    }
    
    # Market data
    market_data = {
        'price': data.iloc[-1]['close'],
        'volatility': 0.03,
        'returns_history': data['returns'].dropna().values[-100:]
    }
    
    # Calculate position size
    position = risk_manager.calculate_position_size(
        signal=signal,
        market_data=market_data,
        portfolio_value=100000
    )
    
    print(f"\nPosition sizing for BUY signal:")
    print(f"  Signal confidence: {signal['confidence']:.0%}")
    print(f"  Kelly optimal size: {position['kelly_size']:.1%}")
    print(f"  Risk-adjusted size: {position['risk_adjusted_size']:.1%}")
    print(f"  Final position size: {position['position_size_pct']:.1%}")
    print(f"  Position VaR: {position['position_var']:.1%}")
    print(f"  Risk budget used: {position['risk_budget_used']:.0%}")
    
    return risk_manager


def demo_backtesting(data: pd.DataFrame):
    """Demonstrate backtesting framework."""
    print_section("BACKTESTING FRAMEWORK")
    
    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005,
        max_position_size=0.2
    )
    
    # Create simple signal generator
    def simple_signal_generator(data, positions):
        """Simple momentum-based signal generator."""
        if len(data) < 20:
            return {'action': 'hold', 'confidence': 0}
            
        # Calculate momentum
        returns = data['close'].pct_change().dropna()
        momentum = returns.rolling(10).mean().iloc[-1]
        
        if momentum > 0.001:
            return {'action': 'buy', 'confidence': 0.7}
        elif momentum < -0.001:
            return {'action': 'sell', 'confidence': 0.7}
        else:
            return {'action': 'hold', 'confidence': 0.5}
    
    # Run backtest
    print("Running backtest with simple momentum strategy...")
    engine = BacktestEngine(config)
    results = engine.run(data.tail(1000), simple_signal_generator)
    
    print(f"\nBacktest Results:")
    print(f"  Total return: {results['total_return']:.2%}")
    print(f"  Sharpe ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Max drawdown: {results['max_drawdown']:.2%}")
    print(f"  Number of trades: {results['num_trades']}")
    print(f"  Win rate: {results['win_rate']:.1%}")
    print(f"  Profit factor: {results['profit_factor']:.2f}")
    
    # Performance metrics
    metrics_calc = PerformanceMetrics()
    detailed_metrics = metrics_calc.calculate_returns_metrics(
        np.array(results['equity_curve'][1:]) / np.array(results['equity_curve'][:-1]) - 1
    )
    
    print(f"\nDetailed Performance Metrics:")
    print(f"  Annualized return: {detailed_metrics['annualized_return']:.2%}")
    print(f"  Return volatility: {detailed_metrics['return_volatility']:.2%}")
    print(f"  Downside volatility: {detailed_metrics['downside_volatility']:.2%}")
    print(f"  Best day: {detailed_metrics['best_day']:.2%}")
    print(f"  Worst day: {detailed_metrics['worst_day']:.2%}")
    
    return results


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print(" MULTI-MODEL MACHINE LEARNING TRADING SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # Prepare data
    data = demo_data_preparation()
    
    # Statistical Models
    arima_model = demo_arima_model(data)
    garch_model = demo_garch_model(data)
    
    # Machine Learning Models
    hmm_trainer = demo_hmm_regime_detection(data)
    
    # Deep Learning Models
    gru_model = demo_gru_attention(data)
    cnn_model = demo_cnn_pattern_recognition(data)
    
    # Reinforcement Learning
    ppo_agent, env = demo_ppo_reinforcement_learning(data)
    
    # Risk Management
    risk_manager = demo_risk_management(data)
    
    # Backtesting
    backtest_results = demo_backtesting(data)
    
    print_section("DEMONSTRATION COMPLETE")
    
    print("Summary of implemented models:")
    print("\n1. Statistical Models:")
    print("   - ARIMA: Time series forecasting")
    print("   - GARCH: Volatility modeling and VaR")
    
    print("\n2. Machine Learning:")
    print("   - HMM: Market regime detection")
    
    print("\n3. Deep Learning:")
    print("   - GRU-Attention: Sequential pattern learning")
    print("   - CNN: Chart pattern recognition")
    
    print("\n4. Reinforcement Learning:")
    print("   - PPO: Adaptive trading agent")
    
    print("\n5. Risk Management:")
    print("   - Kelly Criterion + VaR position sizing")
    print("   - Dynamic stop-loss and take-profit")
    
    print("\n6. Backtesting:")
    print("   - Event-driven backtesting engine")
    print("   - Walk-forward analysis")
    print("   - Comprehensive performance metrics")
    
    print("\nAll models implemented from scratch without external ML dependencies!")
    print("Ready for production deployment with proper training and optimization.")


if __name__ == "__main__":
    main()