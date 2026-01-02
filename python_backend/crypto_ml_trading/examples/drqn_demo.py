"""
Demo script for Deep Recurrent Q-Network (DRQN) agent.
Shows how to use DRQN for cryptocurrency trading with partial observability.
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.reinforcement import DRQNAgent, DRQNTradingEnvironment, DRQNTrainer
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_crypto_data(n_samples: int = 2000) -> pd.DataFrame:
    """Generate synthetic cryptocurrency data."""
    np.random.seed(42)
    
    # Generate base price with trend
    time = np.arange(n_samples)
    trend = 0.0001 * time  # Slight upward trend
    
    # Add multiple frequency components for realism
    seasonal_daily = 5 * np.sin(2 * np.pi * time / 96)  # 24-hour cycle (15min bars)
    seasonal_weekly = 3 * np.sin(2 * np.pi * time / 672)  # Weekly cycle
    
    # Add volatility clusters
    volatility = np.ones(n_samples)
    for i in range(10):
        start = np.random.randint(0, n_samples - 100)
        volatility[start:start+100] *= np.random.uniform(1.5, 3.0)
    
    # Generate price with realistic noise
    noise = np.random.normal(0, 1, n_samples) * volatility
    price = 100 * np.exp(trend) + seasonal_daily + seasonal_weekly + noise
    
    # Ensure positive prices
    price = np.maximum(price, 10)
    
    # Generate OHLC data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='15min'),
        'open': price + np.random.normal(0, 0.1, n_samples),
        'high': price + np.abs(np.random.normal(0, 0.5, n_samples)),
        'low': price - np.abs(np.random.normal(0, 0.5, n_samples)),
        'close': price,
        'volume': np.random.lognormal(10, 0.5, n_samples)
    })
    
    # Add technical indicators
    df['returns'] = df['close'].pct_change().fillna(0)
    df['sma_20'] = df['close'].rolling(20).mean().fillna(df['close'])
    df['sma_50'] = df['close'].rolling(50).mean().fillna(df['close'])
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['volatility'] = df['returns'].rolling(20).std().fillna(0)
    
    # Normalize features
    for col in ['sma_20', 'sma_50']:
        df[col] = df[col] / df['close']
    
    df['rsi'] = df['rsi'] / 100  # Normalize RSI to [0, 1]
    
    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)  # Neutral RSI


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for DRQN."""
    feature_columns = [
        'returns', 'volatility', 'rsi',
        'sma_20', 'sma_50'
    ]
    
    # Ensure all features exist
    for col in feature_columns:
        if col not in df.columns:
            logger.warning(f"Feature {col} not found in data")
    
    # Select available features
    available_features = [col for col in feature_columns if col in df.columns]
    
    return df[['close'] + available_features].copy()


def main():
    """Run DRQN demo."""
    logger.info("Starting DRQN trading demo...")
    
    # Generate synthetic data
    logger.info("Generating synthetic cryptocurrency data...")
    df = generate_crypto_data(3000)
    
    # Prepare features
    features_df = prepare_features(df)
    
    # Remove NaN values
    features_df = features_df.dropna()
    
    # Split data
    train_size = int(0.7 * len(features_df))
    val_size = int(0.15 * len(features_df))
    
    train_data = features_df[:train_size].copy()
    val_data = features_df[train_size:train_size + val_size].copy()
    test_data = features_df[train_size + val_size:].copy()
    
    logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Configure DRQN
    config = {
        'agent': {
            'hidden_size': 128,
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'batch_size': 32,
            'sequence_length': 10,
            'buffer_size': 10000
        },
        'environment': {
            'initial_balance': 10000,
            'position_size': 0.1,
            'transaction_cost': 0.001,
            'stop_loss': 0.05,
            'take_profit': 0.10,
            'max_holding_period': 100,
            'reward_scaling': 100.0
        },
        'target_update_freq': 10,
        'max_episode_length': 1000
    }
    
    # Initialize trainer
    trainer = DRQNTrainer(config)
    
    # Train agent
    logger.info("Training DRQN agent...")
    results = trainer.train(
        train_data=train_data,
        val_data=val_data,
        episodes=50,  # Reduced for demo
        save_freq=10,
        checkpoint_dir="drqn_checkpoints"
    )
    
    # Print training results
    logger.info("\n=== Training Results ===")
    logger.info(f"Best episode: {results['best_episode']}")
    logger.info(f"Best portfolio value: ${results['best_portfolio_value']:.2f}")
    logger.info(f"Final portfolio value: ${results['final_portfolio_value']:.2f}")
    logger.info(f"Total trades: {results['total_trades']}")
    logger.info(f"Win rate: {results['win_rate']:.2%}")
    logger.info(f"Max drawdown: {results['max_drawdown']:.2%}")
    
    if results['val_results']:
        logger.info("\n=== Validation Results ===")
        val_results = results['val_results']
        logger.info(f"Portfolio value: ${val_results['portfolio_value']:.2f}")
        logger.info(f"Total return: {val_results['total_return']:.2%}")
        logger.info(f"Sharpe ratio: {val_results['sharpe_ratio']:.3f}")
        logger.info(f"Action distribution - Buy: {val_results['action_distribution']['buy']:.2%}, "
                   f"Sell: {val_results['action_distribution']['sell']:.2%}, "
                   f"Hold: {val_results['action_distribution']['hold']:.2%}")
    
    # Evaluate on test data
    logger.info("\n=== Test Set Evaluation ===")
    test_results = trainer.evaluate(test_data, render=False)
    
    logger.info(f"Portfolio value: ${test_results['portfolio_value']:.2f}")
    logger.info(f"Total return: {test_results['total_return']:.2%}")
    logger.info(f"Total trades: {test_results['total_trades']}")
    logger.info(f"Win rate: {test_results['win_rate']:.2%}")
    logger.info(f"Max drawdown: {test_results['max_drawdown']:.2%}")
    logger.info(f"Sharpe ratio: {test_results['sharpe_ratio']:.3f}")
    logger.info(f"Action distribution - Buy: {test_results['action_distribution']['buy']:.2%}, "
               f"Sell: {test_results['action_distribution']['sell']:.2%}, "
               f"Hold: {test_results['action_distribution']['hold']:.2%}")
    
    # Plot training progress
    plot_training_progress(results['training_history'])
    
    logger.info("\nDemo completed successfully!")
    
    # Clean up
    import shutil
    if os.path.exists("drqn_checkpoints"):
        shutil.rmtree("drqn_checkpoints")


def plot_training_progress(history: dict):
    """Plot training progress (text-based for terminal)."""
    logger.info("\n=== Training Progress ===")
    
    # Show last 10 episodes
    n_show = min(10, len(history['episode_rewards']))
    
    logger.info("Last 10 episodes:")
    for i in range(-n_show, 0):
        episode = len(history['episode_rewards']) + i
        reward = history['episode_rewards'][i]
        portfolio = history['portfolio_values'][i]
        win_rate = history['win_rates'][i]
        epsilon = history['epsilons'][i]
        
        logger.info(f"Episode {episode}: Reward={reward:.2f}, "
                   f"Portfolio=${portfolio:.2f}, "
                   f"Win Rate={win_rate:.2%}, "
                   f"Epsilon={epsilon:.3f}")


if __name__ == "__main__":
    main()