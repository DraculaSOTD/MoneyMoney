#!/usr/bin/env python3
"""
Reinforcement Learning Training Demo
Shows PPO training on trading environment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Import components
from environments.trading_gym_env import TradingGymEnvironment
from training.ppo_trainer import PPOTrainer
from data.enhanced_data_loader import BinanceDataSource
from examples.simple_demo import create_sample_binance_data


def prepare_training_data(n_samples: int = 10000):
    """Prepare data for RL training."""
    print("Preparing training data...")
    
    # Generate more realistic data with trends and volatility
    np.random.seed(42)
    
    # Create time series with multiple regimes
    timestamps = pd.date_range(end=datetime.now(), periods=n_samples, freq='1h')
    
    # Generate price with different market regimes
    prices = []
    current_price = 100
    
    for i in range(n_samples):
        # Market regime (trending, ranging, volatile)
        regime_length = 500
        regime = (i // regime_length) % 3
        
        if regime == 0:  # Trending up
            drift = 0.0002
            volatility = 0.008
        elif regime == 1:  # Ranging
            drift = 0
            volatility = 0.005
        else:  # Volatile
            drift = -0.0001
            volatility = 0.015
        
        # Add some market events
        if np.random.random() < 0.001:  # 0.1% chance of event
            shock = np.random.choice([-1, 1]) * np.random.uniform(0.02, 0.05)
        else:
            shock = 0
        
        # Update price
        return_ = np.random.normal(drift, volatility) + shock
        current_price *= (1 + return_)
        prices.append(current_price)
    
    prices = np.array(prices)
    
    # Create OHLCV data
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1.5, n_samples)
    })
    
    # Add volume spikes during price movements
    price_changes = np.abs(df['close'].pct_change())
    volume_multiplier = 1 + price_changes * 50
    df['volume'] *= volume_multiplier.fillna(1)
    
    print(f"✓ Generated {len(df)} hours of training data")
    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"  Volatility: {df['close'].pct_change().std() * 100:.2f}%")
    
    return df


def create_environments(train_data: pd.DataFrame, test_data: pd.DataFrame):
    """Create training and testing environments."""
    print("\nCreating environments...")
    
    # Environment configuration
    env_config = {
        'initial_balance': 10000,
        'commission': 0.001,
        'slippage': 0.0005,
        'reward_type': 'sharpe',  # sharpe, simple, risk_adjusted
        'lookback_window': 50,
        'discrete_actions': False,  # Continuous actions
        'feature_config': {
            'indicators': {
                'sma': {'enabled': True, 'periods': [10, 20, 50]},
                'ema': {'enabled': True, 'periods': [12, 26]},
                'rsi': {'enabled': True, 'period': 14},
                'macd': {'enabled': True},
                'bollinger': {'enabled': True},
                'atr': {'enabled': True}
            }
        },
        'risk_config': {
            'max_position_size': 1.0,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04
        }
    }
    
    # Create environments
    train_env = TradingGymEnvironment(train_data, **env_config)
    test_env = TradingGymEnvironment(test_data, **env_config)
    
    print(f"✓ Created environments")
    print(f"  Observation space: {train_env.observation_space.shape}")
    print(f"  Action space: {train_env.action_space}")
    
    return train_env, test_env


def train_ppo_agent(train_env: TradingGymEnvironment, 
                   test_env: TradingGymEnvironment,
                   total_timesteps: int = 100000):
    """Train PPO agent on trading environment."""
    print("\nTraining PPO agent...")
    
    # PPO configuration
    ppo_config = {
        # Network
        'hidden_size': 256,
        'num_layers': 2,
        'use_lstm': True,
        
        # PPO hyperparameters
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_param': 0.2,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        
        # Training
        'num_steps': 2048,
        'num_epochs': 10,
        'batch_size': 64,
        
        # Saving and evaluation
        'save_freq': 10,
        'eval_freq': 5
    }
    
    # Create trainer
    trainer = PPOTrainer(train_env, config=ppo_config)
    
    # Train agent
    trainer.train(total_timesteps=total_timesteps, eval_env=test_env)
    
    print(f"✓ Training complete")
    print(f"  Total episodes: {trainer.episode_count}")
    print(f"  Best reward: {trainer.best_reward:.2f}")
    
    return trainer


def evaluate_agent(trainer: PPOTrainer, test_env: TradingGymEnvironment, 
                  num_episodes: int = 10):
    """Evaluate trained agent and visualize results."""
    print("\nEvaluating trained agent...")
    
    # Evaluate
    eval_stats = trainer.evaluate(test_env, num_episodes=num_episodes)
    
    print(f"\nEvaluation Results ({num_episodes} episodes):")
    print(f"  Average Return: {eval_stats['avg_return']:.2f}% ± {eval_stats['std_return']:.2f}%")
    print(f"  Sharpe Ratio: {eval_stats['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {eval_stats['max_drawdown']:.2f}%")
    print(f"  Win Rate: {eval_stats['win_rate']:.2%}")
    
    # Run one episode for detailed visualization
    print("\nRunning detailed evaluation episode...")
    
    obs = test_env.reset()
    hidden = None
    done = False
    
    episode_data = {
        'prices': [],
        'positions': [],
        'portfolio_values': [],
        'actions': [],
        'rewards': []
    }
    
    while not done:
        # Get action
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
        with torch.no_grad():
            action, _, _, hidden = trainer.actor_critic.get_action(
                obs_tensor, hidden, deterministic=True
            )
        
        action_np = action.cpu().numpy().squeeze()
        
        # Store data
        episode_data['prices'].append(test_env.state.current_price)
        episode_data['positions'].append(test_env.state.position)
        episode_data['portfolio_values'].append(test_env.state.portfolio_value)
        episode_data['actions'].append(action_np)
        
        # Step
        obs, reward, done, info = test_env.step(action_np)
        episode_data['rewards'].append(reward)
    
    # Get final statistics
    final_stats = test_env.get_episode_statistics()
    
    return eval_stats, episode_data, final_stats


def visualize_results(episode_data: dict, final_stats: dict, 
                     trainer: PPOTrainer, save_path: str = 'rl_results.png'):
    """Visualize training and evaluation results."""
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('PPO Trading Agent Results', fontsize=16)
    
    # 1. Training progress
    ax1 = axes[0, 0]
    if trainer.training_stats:
        stats_df = pd.DataFrame(trainer.training_stats)
        ax1.plot(stats_df['timestep'], stats_df['avg_reward'], label='Episode Reward')
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Average Episode Reward')
        ax1.set_title('Training Progress')
        ax1.grid(True, alpha=0.3)
    
    # 2. Price and positions
    ax2 = axes[0, 1]
    prices = episode_data['prices']
    positions = episode_data['positions']
    
    ax2.plot(prices, label='Price', color='black', alpha=0.7)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(positions, label='Position', color='blue', alpha=0.5)
    ax2_twin.fill_between(range(len(positions)), positions, alpha=0.2, color='blue')
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Price ($)')
    ax2_twin.set_ylabel('Position')
    ax2.set_title('Price and Trading Positions')
    ax2.grid(True, alpha=0.3)
    
    # 3. Portfolio value
    ax3 = axes[1, 0]
    portfolio_values = episode_data['portfolio_values']
    returns = pd.Series(portfolio_values).pct_change().fillna(0)
    
    ax3.plot(portfolio_values, label='Portfolio Value')
    initial_value = portfolio_values[0]
    ax3.axhline(initial_value, color='red', linestyle='--', label='Initial Value')
    
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Portfolio Value ($)')
    ax3.set_title('Portfolio Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Action distribution
    ax4 = axes[1, 1]
    actions = np.array(episode_data['actions'])
    
    if len(actions.shape) > 1 and actions.shape[1] > 1:
        # Continuous actions
        ax4.hist(actions[:, 0], bins=30, alpha=0.7, label='Position Size')
        ax4.set_xlabel('Action Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Action Distribution')
        ax4.legend()
    else:
        # Discrete actions
        unique, counts = np.unique(actions, return_counts=True)
        ax4.bar(['Sell', 'Hold', 'Buy'][:len(unique)], counts)
        ax4.set_xlabel('Action')
        ax4.set_ylabel('Count')
        ax4.set_title('Action Distribution')
    
    # 5. Reward distribution
    ax5 = axes[2, 0]
    rewards = episode_data['rewards']
    ax5.hist(rewards, bins=50, alpha=0.7, color='green')
    ax5.axvline(np.mean(rewards), color='red', linestyle='--', 
                label=f'Mean: {np.mean(rewards):.3f}')
    ax5.set_xlabel('Reward')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Reward Distribution')
    ax5.legend()
    
    # 6. Performance summary
    ax6 = axes[2, 1]
    ax6.axis('off')
    
    summary_text = f"""Performance Summary:
    
Total Return: {final_stats['total_return']:.2f}%
Sharpe Ratio: {final_stats['sharpe_ratio']:.2f}
Max Drawdown: {final_stats['max_drawdown']:.2f}%
Number of Trades: {final_stats['num_trades']}
Win Rate: {final_stats['win_rate']:.2%}
Final Portfolio: ${final_stats['final_portfolio_value']:.2f}
Total P&L: ${final_stats['total_pnl']:.2f}
    """
    
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
            verticalalignment='center', fontsize=12, family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {save_path}")
    
    return fig


def compare_with_baseline(test_env: TradingGymEnvironment):
    """Compare RL agent with baseline strategies."""
    print("\nComparing with baseline strategies...")
    
    results = {}
    
    # Random agent
    print("  Testing random agent...")
    obs = test_env.reset()
    done = False
    while not done:
        action = test_env.action_space.sample()
        obs, reward, done, info = test_env.step(action)
    
    results['random'] = test_env.get_episode_statistics()
    
    # Buy and hold
    print("  Testing buy-and-hold...")
    obs = test_env.reset()
    obs, reward, done, info = test_env.step(np.array([1.0, 0.02, 0.04]))  # Full buy
    while not done:
        obs, reward, done, info = test_env.step(np.array([1.0, 0.02, 0.04]))  # Hold
    
    results['buy_hold'] = test_env.get_episode_statistics()
    
    return results


def main():
    """Run complete RL training demonstration."""
    print("="*80)
    print("REINFORCEMENT LEARNING TRADING DEMONSTRATION")
    print("="*80)
    
    # Create directories
    Path('rl_models').mkdir(exist_ok=True)
    Path('rl_results').mkdir(exist_ok=True)
    
    # 1. Prepare data
    all_data = prepare_training_data(n_samples=10000)
    
    # Split data
    train_size = int(len(all_data) * 0.8)
    train_data = all_data.iloc[:train_size].copy()
    test_data = all_data.iloc[train_size:].copy()
    
    print(f"\nData split:")
    print(f"  Training: {len(train_data)} samples")
    print(f"  Testing: {len(test_data)} samples")
    
    # 2. Create environments
    train_env, test_env = create_environments(train_data, test_data)
    
    # 3. Train PPO agent
    # Note: Reduced timesteps for demonstration
    trainer = train_ppo_agent(train_env, test_env, total_timesteps=50000)
    
    # 4. Evaluate agent
    eval_stats, episode_data, final_stats = evaluate_agent(trainer, test_env)
    
    # 5. Compare with baselines
    baseline_results = compare_with_baseline(test_env)
    
    print("\nComparison with Baselines:")
    print(f"{'Strategy':<15} {'Return':<10} {'Sharpe':<10} {'Max DD':<10}")
    print("-"*45)
    
    # PPO results
    print(f"{'PPO Agent':<15} {final_stats['total_return']:>8.2f}% "
          f"{final_stats['sharpe_ratio']:>9.2f} {final_stats['max_drawdown']:>8.2f}%")
    
    # Baseline results
    for name, stats in baseline_results.items():
        print(f"{name.upper():<15} {stats['total_return']:>8.2f}% "
              f"{stats['sharpe_ratio']:>9.2f} {stats['max_drawdown']:>8.2f}%")
    
    # 6. Visualize results
    visualize_results(episode_data, final_stats, trainer)
    
    # 7. Save results
    results_summary = {
        'training_config': trainer.config,
        'eval_stats': eval_stats,
        'final_stats': final_stats,
        'baseline_comparison': baseline_results,
        'environment_config': {
            'observation_shape': train_env.observation_space.shape,
            'action_shape': train_env.action_space.shape if hasattr(train_env.action_space, 'shape') else train_env.action_space.n,
            'reward_type': train_env.reward_type
        }
    }
    
    with open('rl_results/training_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print("\n✓ Results saved to rl_results/")
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    
    return trainer, eval_stats, baseline_results


if __name__ == "__main__":
    # Check dependencies
    try:
        import torch
        import gym
        import matplotlib
        matplotlib.use('Agg')
        print(f"PyTorch version: {torch.__version__}")
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install: pip install torch gym matplotlib")
        sys.exit(1)
    
    # Run demonstration
    trainer, eval_stats, baseline_results = main()