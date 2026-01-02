import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import deque
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from models.reinforcement.ppo.ppo_agent import PPOAgent
from models.reinforcement.ppo.trading_env import TradingEnvironment


class PPOTrainer:
    """
    Trainer for PPO agent in cryptocurrency trading.
    
    Features:
    - Episode collection and batching
    - Curriculum learning
    - Performance tracking
    - Model checkpointing
    - Live trading mode
    """
    
    def __init__(self,
                 agent: PPOAgent,
                 env: TradingEnvironment,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95):
        """
        Initialize PPO trainer.
        
        Args:
            agent: PPO agent
            env: Trading environment
            n_steps: Steps per update
            batch_size: Mini-batch size
            n_epochs: Epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
        """
        self.agent = agent
        self.env = env
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Training statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'portfolio_values': []
        }
        
        # Best model tracking
        self.best_reward = -float('inf')
        self.best_sharpe = -float('inf')
        
    def collect_rollout(self) -> Dict[str, np.ndarray]:
        """
        Collect experience by running policy in environment.
        
        Returns:
            Dictionary of collected experience
        """
        # Storage
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        # Reset if needed
        if hasattr(self, 'last_state'):
            state = self.last_state
        else:
            state = self.env.reset()
            
        for step in range(self.n_steps):
            # Get action from policy
            action, action_info = self.agent.get_action(state)
            
            # Step environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(action_info['value'])
            log_probs.append(action_info['log_prob'])
            dones.append(done)
            
            state = next_state
            
            if done:
                # Episode finished
                episode_info = self.env.get_episode_summary()
                self.episode_rewards.append(episode_info['total_return'])
                self.episode_lengths.append(self.env.state.step)
                
                # Reset environment
                state = self.env.reset()
                
        # Store last state for next rollout
        self.last_state = state
        
        # Convert to arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        values = np.array(values)
        log_probs = np.array(log_probs)
        dones = np.array(dones)
        
        # Compute advantages
        next_values = np.append(values[1:], action_info['value'])
        advantages = self.agent.compute_gae(
            rewards, values, next_values, dones,
            self.gamma, self.gae_lambda
        )
        
        # Compute returns
        returns = advantages + values
        
        return {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'advantages': advantages,
            'returns': returns,
            'values': values
        }
    
    def train(self, n_iterations: int, save_freq: int = 10,
             verbose: int = 1) -> Dict:
        """
        Train PPO agent.
        
        Args:
            n_iterations: Number of training iterations
            save_freq: Frequency of model saving
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if verbose > 0:
            print(f"Starting PPO training for {n_iterations} iterations")
            print(f"Environment: {self.n_steps} steps per iteration")
            print("-" * 60)
            
        for iteration in range(n_iterations):
            start_time = time.time()
            
            # Collect experience
            rollout_data = self.collect_rollout()
            
            # Update policy
            update_info = self.agent.update(
                rollout_data['states'],
                rollout_data['actions'],
                rollout_data['log_probs'],
                rollout_data['advantages'],
                rollout_data['returns'],
                epochs=self.n_epochs,
                batch_size=self.batch_size
            )
            
            # Update history
            self.training_history['episode_rewards'].extend(list(self.episode_rewards)[-10:])
            self.training_history['episode_lengths'].extend(list(self.episode_lengths)[-10:])
            self.training_history['actor_losses'].append(update_info['actor_loss'])
            self.training_history['critic_losses'].append(update_info['critic_loss'])
            
            # Calculate statistics
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards)
                mean_length = np.mean(self.episode_lengths)
                
                # Calculate Sharpe ratio from recent episodes
                if len(self.episode_rewards) > 10:
                    episode_returns = np.array(list(self.episode_rewards)[-20:])
                    sharpe = np.mean(episode_returns) / (np.std(episode_returns) + 1e-6) * np.sqrt(252)
                else:
                    sharpe = 0
                    
                # Track best model
                if mean_reward > self.best_reward:
                    self.best_reward = mean_reward
                    self.agent.save_model('best_reward_model.npz')
                    
                if sharpe > self.best_sharpe:
                    self.best_sharpe = sharpe
                    self.agent.save_model('best_sharpe_model.npz')
                    
                # Save periodic checkpoint
                if (iteration + 1) % save_freq == 0:
                    self.agent.save_model(f'checkpoint_{iteration+1}.npz')
                    
                # Print progress
                if verbose > 0 and iteration % max(1, n_iterations // 20) == 0:
                    elapsed = time.time() - start_time
                    print(f"Iteration {iteration}/{n_iterations} - {elapsed:.1f}s")
                    print(f"  Mean Return: {mean_reward:.4f}")
                    print(f"  Mean Episode Length: {mean_length:.0f}")
                    print(f"  Sharpe Ratio: {sharpe:.2f}")
                    print(f"  Actor Loss: {update_info['actor_loss']:.4f}")
                    print(f"  Critic Loss: {update_info['critic_loss']:.4f}")
                    print(f"  Entropy: {update_info['entropy']:.4f}")
                    
        if verbose > 0:
            print("-" * 60)
            print(f"Training completed!")
            print(f"Best Return: {self.best_reward:.4f}")
            print(f"Best Sharpe: {self.best_sharpe:.2f}")
            
        return self.training_history
    
    def evaluate(self, n_episodes: int = 10, 
                render: bool = False) -> Dict[str, float]:
        """
        Evaluate trained agent.
        
        Args:
            n_episodes: Number of evaluation episodes
            render: Whether to render environment
            
        Returns:
            Evaluation metrics
        """
        episode_returns = []
        episode_lengths = []
        episode_sharpes = []
        episode_max_dds = []
        episode_trades = []
        
        for episode in range(n_episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                # Get action (deterministic for evaluation)
                action, _ = self.agent.get_action(state, deterministic=True)
                
                # Step environment
                state, reward, done, info = self.env.step(action)
                
                if render:
                    self.env.render()
                    
            # Get episode summary
            summary = self.env.get_episode_summary()
            episode_returns.append(summary['total_return'])
            episode_lengths.append(self.env.state.step)
            episode_sharpes.append(summary['sharpe_ratio'])
            episode_max_dds.append(summary['max_drawdown'])
            episode_trades.append(summary['num_trades'])
            
        # Calculate statistics
        metrics = {
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'mean_sharpe': np.mean(episode_sharpes),
            'mean_max_dd': np.mean(episode_max_dds),
            'mean_trades': np.mean(episode_trades),
            'win_rate': np.mean([r > 0 for r in episode_returns])
        }
        
        return metrics
    
    def generate_trading_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for live/paper trading.
        
        Args:
            market_data: Recent market data
            
        Returns:
            DataFrame with trading signals
        """
        # Create temporary environment with market data
        temp_env = TradingEnvironment(
            data=market_data,
            initial_capital=self.env.initial_capital,
            lookback_window=self.env.lookback_window
        )
        
        # Generate signals
        signals = []
        state = temp_env.reset()
        
        for i in range(len(market_data) - temp_env.lookback_window):
            # Get action from policy
            action, action_info = self.agent.get_action(state, deterministic=True)
            
            # Create signal
            signal = {
                'timestamp': temp_env.state.timestamp,
                'price': temp_env.state.current_price,
                'position_target': float(action[0]),
                'stop_loss': float(action[1]),
                'take_profit': float(action[2]),
                'confidence': float(action_info['value']),
                'action_mean': action_info['mean'].tolist(),
                'action_std': action_info['std'].tolist()
            }
            
            signals.append(signal)
            
            # Step environment
            state, _, done, _ = temp_env.step(action)
            
            if done:
                break
                
        return pd.DataFrame(signals)
    
    def adapt_to_live_market(self, recent_data: pd.DataFrame,
                           n_adaptation_steps: int = 100) -> Dict[str, float]:
        """
        Fine-tune agent on recent market data.
        
        Args:
            recent_data: Recent market data
            n_adaptation_steps: Number of adaptation steps
            
        Returns:
            Adaptation metrics
        """
        # Create environment with recent data
        adapt_env = TradingEnvironment(
            data=recent_data,
            initial_capital=self.env.initial_capital,
            lookback_window=self.env.lookback_window
        )
        
        # Store original environment
        original_env = self.env
        self.env = adapt_env
        
        # Reduce learning rate for fine-tuning
        original_lr = self.agent.learning_rate
        self.agent.learning_rate *= 0.1
        
        # Fine-tune
        print("Adapting to recent market conditions...")
        adaptation_history = self.train(
            n_iterations=n_adaptation_steps,
            save_freq=n_adaptation_steps + 1,  # Don't save during adaptation
            verbose=0
        )
        
        # Restore original settings
        self.env = original_env
        self.agent.learning_rate = original_lr
        
        # Evaluate adaptation
        metrics = self.evaluate(n_episodes=5)
        
        print(f"Adaptation complete. New performance:")
        print(f"  Mean Return: {metrics['mean_return']:.4f}")
        print(f"  Sharpe Ratio: {metrics['mean_sharpe']:.2f}")
        
        return metrics
    
    def analyze_policy(self, market_states: np.ndarray) -> Dict:
        """
        Analyze learned policy behavior.
        
        Args:
            market_states: Sample market states
            
        Returns:
            Policy analysis
        """
        actions = []
        values = []
        action_stds = []
        
        for state in market_states:
            action, info = self.agent.get_action(state, deterministic=False)
            actions.append(action)
            values.append(info['value'])
            action_stds.append(info['std'])
            
        actions = np.array(actions)
        values = np.array(values)
        action_stds = np.array(action_stds)
        
        analysis = {
            'mean_position': np.mean(actions[:, 0]),
            'std_position': np.std(actions[:, 0]),
            'mean_stop_loss': np.mean(actions[:, 1]),
            'mean_take_profit': np.mean(actions[:, 2]),
            'mean_value': np.mean(values),
            'std_value': np.std(values),
            'mean_action_uncertainty': np.mean(action_stds),
            'position_distribution': {
                'long_bias': np.mean(actions[:, 0] > 0.1),
                'short_bias': np.mean(actions[:, 0] < -0.1),
                'neutral': np.mean(np.abs(actions[:, 0]) <= 0.1)
            }
        }
        
        return analysis