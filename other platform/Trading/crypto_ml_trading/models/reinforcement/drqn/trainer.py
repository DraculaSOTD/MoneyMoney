import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import os
from tqdm import tqdm

from .drqn_agent import DRQNAgent
from .trading_env import TradingEnvironment

logger = logging.getLogger(__name__)


class DRQNTrainer:
    """
    Trainer for Deep Recurrent Q-Network (DRQN).
    
    Handles training loop, evaluation, and model persistence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DRQN trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.agent = None
        self.env = None
        
        # Training history
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'portfolio_values': [],
            'win_rates': [],
            'losses': [],
            'epsilons': []
        }
        
    def build_agent(self, state_size: int) -> DRQNAgent:
        """Build DRQN agent with configuration."""
        agent_config = self.config.get('agent', {})
        
        self.agent = DRQNAgent(
            state_size=state_size,
            action_size=agent_config.get('action_size', 3),
            hidden_size=agent_config.get('hidden_size', 128),
            learning_rate=agent_config.get('learning_rate', 0.001),
            gamma=agent_config.get('gamma', 0.99),
            epsilon=agent_config.get('epsilon', 1.0),
            epsilon_decay=agent_config.get('epsilon_decay', 0.995),
            epsilon_min=agent_config.get('epsilon_min', 0.01),
            batch_size=agent_config.get('batch_size', 32),
            sequence_length=agent_config.get('sequence_length', 10),
            buffer_size=agent_config.get('buffer_size', 10000)
        )
        
        logger.info(f"Built DRQN agent with state_size={state_size}, "
                   f"hidden_size={agent_config.get('hidden_size', 128)}")
        
        return self.agent
    
    def build_environment(self, data: pd.DataFrame) -> TradingEnvironment:
        """Build trading environment with configuration."""
        env_config = self.config.get('environment', {})
        
        self.env = TradingEnvironment(
            data=data,
            initial_balance=env_config.get('initial_balance', 10000),
            position_size=env_config.get('position_size', 0.1),
            transaction_cost=env_config.get('transaction_cost', 0.001),
            stop_loss=env_config.get('stop_loss', 0.05),
            take_profit=env_config.get('take_profit', 0.10),
            max_holding_period=env_config.get('max_holding_period', 100),
            reward_scaling=env_config.get('reward_scaling', 100.0)
        )
        
        logger.info("Built trading environment")
        
        return self.env
    
    def train(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None,
              episodes: int = 100, save_freq: int = 10,
              checkpoint_dir: str = "checkpoints") -> Dict[str, Any]:
        """
        Train DRQN agent.
        
        Args:
            train_data: Training data
            val_data: Validation data
            episodes: Number of training episodes
            save_freq: Frequency of saving checkpoints
            checkpoint_dir: Directory for checkpoints
            
        Returns:
            Training results
        """
        # Ensure required columns exist
        required_columns = ['close']
        for col in required_columns:
            if col not in train_data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Build environment and agent
        self.env = self.build_environment(train_data)
        state = self.env.reset()
        state_size = len(state)
        self.agent = self.build_agent(state_size)
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training metrics
        best_portfolio_value = 0
        best_episode = 0
        
        logger.info(f"Starting DRQN training for {episodes} episodes...")
        
        # Training loop
        for episode in tqdm(range(episodes), desc="Training Progress"):
            # Reset environment and agent state
            state = self.env.reset()
            self.agent.reset_state()
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Select action
                action = self.agent.act(state, training=True)
                
                # Execute action
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                if len(self.agent.memory) > self.agent.batch_size:
                    loss = self.agent.replay()
                    if loss is not None:
                        self.training_history['losses'].append(loss)
                
                # Update state
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # Limit episode length
                if episode_length > self.config.get('max_episode_length', 1000):
                    done = True
            
            # Update target network periodically
            if episode % self.config.get('target_update_freq', 10) == 0:
                self.agent.update_target_network()
            
            # Record episode metrics
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['portfolio_values'].append(info['portfolio_value'])
            self.training_history['win_rates'].append(info['win_rate'])
            self.training_history['epsilons'].append(self.agent.epsilon)
            
            # Check for best performance
            if info['portfolio_value'] > best_portfolio_value:
                best_portfolio_value = info['portfolio_value']
                best_episode = episode
                
                # Save best model
                best_model_path = os.path.join(checkpoint_dir, 'best_model.npz')
                self.agent.save(best_model_path)
            
            # Log progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-10:])
                avg_portfolio = np.mean(self.training_history['portfolio_values'][-10:])
                avg_win_rate = np.mean(self.training_history['win_rates'][-10:])
                
                logger.info(f"Episode {episode}: Avg Reward={avg_reward:.2f}, "
                           f"Avg Portfolio=${avg_portfolio:.2f}, "
                           f"Win Rate={avg_win_rate:.2%}, "
                           f"Epsilon={self.agent.epsilon:.3f}")
            
            # Save checkpoint
            if episode % save_freq == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_ep{episode}.npz')
                self.save_checkpoint(checkpoint_path, episode)
        
        # Evaluate on validation data if provided
        val_results = None
        if val_data is not None:
            logger.info("Evaluating on validation data...")
            val_results = self.evaluate(val_data)
        
        # Prepare results
        results = {
            'best_episode': best_episode,
            'best_portfolio_value': best_portfolio_value,
            'final_portfolio_value': self.training_history['portfolio_values'][-1],
            'total_trades': self.env.total_trades,
            'win_rate': self.env.winning_trades / (self.env.total_trades + 1e-8),
            'max_drawdown': self.env.max_drawdown,
            'training_history': self.training_history,
            'val_results': val_results
        }
        
        logger.info(f"Training completed. Best portfolio value: ${best_portfolio_value:.2f} "
                   f"at episode {best_episode}")
        
        return results
    
    def evaluate(self, data: pd.DataFrame, render: bool = False) -> Dict[str, Any]:
        """
        Evaluate agent on data.
        
        Args:
            data: Evaluation data
            render: Whether to render environment
            
        Returns:
            Evaluation metrics
        """
        if self.agent is None:
            raise ValueError("Agent must be trained before evaluation")
        
        # Create evaluation environment
        eval_env = self.build_environment(data)
        
        # Run evaluation episode
        state = eval_env.reset()
        self.agent.reset_state()
        
        episode_reward = 0
        done = False
        actions_taken = []
        
        while not done:
            # Select action (no exploration)
            action = self.agent.act(state, training=False)
            actions_taken.append(action)
            
            # Execute action
            next_state, reward, done, info = eval_env.step(action)
            
            if render:
                eval_env.render()
            
            state = next_state
            episode_reward += reward
        
        # Calculate action distribution
        action_counts = np.bincount(actions_taken, minlength=3)
        action_distribution = action_counts / len(actions_taken)
        
        results = {
            'total_reward': episode_reward,
            'portfolio_value': info['portfolio_value'],
            'total_return': (info['portfolio_value'] - eval_env.initial_balance) / eval_env.initial_balance,
            'total_trades': info['total_trades'],
            'win_rate': info['win_rate'],
            'max_drawdown': info['max_drawdown'],
            'action_distribution': {
                'buy': action_distribution[0],
                'sell': action_distribution[1],
                'hold': action_distribution[2]
            },
            'sharpe_ratio': self._calculate_sharpe_ratio(eval_env)
        }
        
        return results
    
    def _calculate_sharpe_ratio(self, env: TradingEnvironment) -> float:
        """Calculate Sharpe ratio from environment history."""
        if env.current_step < 2:
            return 0.0
        
        # Extract returns
        returns = []
        for i in range(1, env.current_step):
            if i < len(env.data):
                prev_price = env.data.iloc[i-1]['close']
                curr_price = env.data.iloc[i]['close']
                ret = (curr_price - prev_price) / prev_price
                returns.append(ret)
        
        if len(returns) == 0:
            return 0.0
        
        returns = np.array(returns)
        
        # Calculate Sharpe ratio (assuming daily returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        sharpe = np.sqrt(252) * mean_return / std_return
        
        return sharpe
    
    def save_checkpoint(self, filepath: str, episode: int):
        """Save training checkpoint."""
        checkpoint = {
            'episode': episode,
            'training_history': self.training_history,
            'config': self.config
        }
        
        # Save agent separately
        agent_path = filepath.replace('.npz', '_agent.npz')
        self.agent.save(agent_path)
        
        # Save checkpoint data
        np.savez(filepath, **checkpoint)
        
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        # Load checkpoint data
        checkpoint = np.load(filepath, allow_pickle=True)
        
        self.training_history = checkpoint['training_history'].item()
        self.config = checkpoint['config'].item()
        
        # Load agent
        agent_path = filepath.replace('.npz', '_agent.npz')
        if self.agent is None:
            # Need to build agent first
            state_size = self.config.get('state_size', 10)  # Default
            self.agent = self.build_agent(state_size)
        
        self.agent.load(agent_path)
        
        logger.info(f"Checkpoint loaded from {filepath}")