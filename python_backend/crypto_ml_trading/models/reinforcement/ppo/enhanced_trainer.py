"""
Enhanced PPO Trainer for Cryptocurrency Trading

Features:
- Advanced PPO implementation with clipping and entropy bonus
- Adaptive exploration with temperature scaling
- Multi-environment parallel training
- Experience replay buffer with prioritization
- Advanced reward shaping and normalization
- Comprehensive performance tracking
- Live market adaptation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
import time
from pathlib import Path
import multiprocessing as mp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.base_trainer import BaseOptimizer, AdamOptimizer
from models.reinforcement.ppo.ppo_agent import PPOAgent
from models.reinforcement.ppo.trading_env import TradingEnvironment


class ExperienceBuffer:
    """Advanced experience replay buffer with prioritization."""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        """
        Initialize prioritized experience buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
    def add(self, experience: Dict, td_error: Optional[float] = None):
        """Add experience with priority."""
        priority = (abs(td_error) + 1e-6) ** self.alpha if td_error else 1.0
        
        self.buffer.append(experience)
        self.priorities.append(priority)
        
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-6) ** self.alpha
            
    def __len__(self):
        return len(self.buffer)


class ParallelEnvironment:
    """Parallel environment wrapper for multi-env training."""
    
    def __init__(self, env_fn, n_envs: int = 4):
        """
        Initialize parallel environments.
        
        Args:
            env_fn: Function to create environment
            n_envs: Number of parallel environments
        """
        self.n_envs = n_envs
        self.envs = [env_fn() for _ in range(n_envs)]
        self.states = None
        
    def reset(self) -> np.ndarray:
        """Reset all environments."""
        self.states = np.array([env.reset() for env in self.envs])
        return self.states
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments."""
        results = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            state, reward, done, info = env.step(action)
            
            if done:
                state = env.reset()
                
            results.append((state, reward, done, info))
            
        states = np.array([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = [r[3] for r in results]
        
        self.states = states
        return states, rewards, dones, infos
    
    def get_episode_summaries(self) -> List[Dict]:
        """Get episode summaries from all environments."""
        return [env.get_episode_summary() if hasattr(env, 'get_episode_summary') else {} 
                for env in self.envs]


class EnhancedPPOTrainer:
    """
    Enhanced PPO trainer with advanced features for crypto trading.
    """
    
    def __init__(self,
                 agent: PPOAgent,
                 env: TradingEnvironment,
                 n_steps: int = 2048,
                 n_epochs: int = 10,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 learning_rate: float = 3e-4,
                 n_envs: int = 4,
                 use_experience_replay: bool = True,
                 use_reward_normalization: bool = True,
                 adaptive_exploration: bool = True):
        """
        Initialize enhanced PPO trainer.
        
        Args:
            agent: PPO agent
            env: Trading environment
            n_steps: Steps per rollout
            n_epochs: Epochs per update
            batch_size: Mini-batch size
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clipping range
            entropy_coef: Entropy coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Max gradient norm
            learning_rate: Learning rate
            n_envs: Number of parallel environments
            use_experience_replay: Whether to use experience replay
            use_reward_normalization: Whether to normalize rewards
            adaptive_exploration: Whether to use adaptive exploration
        """
        self.agent = agent
        self.env = env
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.use_experience_replay = use_experience_replay
        self.use_reward_normalization = use_reward_normalization
        self.adaptive_exploration = adaptive_exploration
        
        # Initialize parallel environments
        self.n_envs = n_envs
        if n_envs > 1:
            self.parallel_env = ParallelEnvironment(
                lambda: TradingEnvironment(
                    data=env.data,
                    initial_capital=env.initial_capital,
                    lookback_window=env.lookback_window
                ),
                n_envs=n_envs
            )
        else:
            self.parallel_env = None
            
        # Initialize optimizer
        self.optimizer = AdamOptimizer(
            learning_rate=learning_rate,
            gradient_clip=max_grad_norm
        )
        
        # Experience replay buffer
        if use_experience_replay:
            self.experience_buffer = ExperienceBuffer(capacity=100000)
        else:
            self.experience_buffer = None
            
        # Reward normalization
        self.reward_stats = {
            'mean': 0.0,
            'std': 1.0,
            'count': 0
        }
        
        # Adaptive exploration
        self.exploration_temperature = 1.0
        self.temperature_schedule = np.linspace(1.0, 0.1, 100)
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_stats = defaultdict(lambda: deque(maxlen=1000))
        self.best_episode_reward = -float('inf')
        self.best_sharpe_ratio = -float('inf')
        
    def collect_rollout(self) -> Dict[str, np.ndarray]:
        """
        Collect experience using parallel environments.
        
        Returns:
            Rollout data dictionary
        """
        if self.parallel_env:
            return self._collect_parallel_rollout()
        else:
            return self._collect_single_rollout()
            
    def _collect_single_rollout(self) -> Dict[str, np.ndarray]:
        """Collect rollout from single environment."""
        # Storage
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        # Initialize
        state = self.env.state.get_observation() if hasattr(self.env, 'state') else self.env.reset()
        
        for step in range(self.n_steps):
            # Get action with exploration
            if self.adaptive_exploration:
                action, action_info = self.agent.get_action(
                    state, 
                    temperature=self.exploration_temperature
                )
            else:
                action, action_info = self.agent.get_action(state)
                
            # Step environment
            next_state, reward, done, info = self.env.step(action)
            
            # Normalize reward if enabled
            if self.use_reward_normalization:
                reward = self._normalize_reward(reward)
                
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(action_info['value'])
            log_probs.append(action_info['log_prob'])
            dones.append(done)
            
            # Add to experience buffer
            if self.experience_buffer:
                experience = {
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done,
                    'value': action_info['value'],
                    'log_prob': action_info['log_prob']
                }
                self.experience_buffer.add(experience)
                
            state = next_state
            
            if done:
                # Track episode statistics
                episode_info = self.env.get_episode_summary()
                self.episode_rewards.append(episode_info['total_return'])
                self.episode_lengths.append(self.env.state.step)
                
                # Check for best performance
                if episode_info['total_return'] > self.best_episode_reward:
                    self.best_episode_reward = episode_info['total_return']
                    self.agent.save_model('best_reward_model.npz')
                    
                if episode_info.get('sharpe_ratio', 0) > self.best_sharpe_ratio:
                    self.best_sharpe_ratio = episode_info['sharpe_ratio']
                    self.agent.save_model('best_sharpe_model.npz')
                    
                # Reset environment
                state = self.env.reset()
                
        # Convert to arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        values = np.array(values)
        log_probs = np.array(log_probs)
        dones = np.array(dones)
        
        # Compute advantages and returns
        advantages, returns = self._compute_gae(rewards, values, dones)
        
        return {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'advantages': advantages,
            'returns': returns,
            'values': values
        }
    
    def _collect_parallel_rollout(self) -> Dict[str, np.ndarray]:
        """Collect rollout from parallel environments."""
        # Storage
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        # Initialize
        current_states = self.parallel_env.reset()
        
        for step in range(self.n_steps // self.n_envs):
            # Get actions for all environments
            env_actions = []
            env_values = []
            env_log_probs = []
            
            for state in current_states:
                if self.adaptive_exploration:
                    action, action_info = self.agent.get_action(
                        state,
                        temperature=self.exploration_temperature
                    )
                else:
                    action, action_info = self.agent.get_action(state)
                    
                env_actions.append(action)
                env_values.append(action_info['value'])
                env_log_probs.append(action_info['log_prob'])
                
            env_actions = np.array(env_actions)
            
            # Step all environments
            next_states, env_rewards, env_dones, env_infos = self.parallel_env.step(env_actions)
            
            # Normalize rewards
            if self.use_reward_normalization:
                env_rewards = np.array([self._normalize_reward(r) for r in env_rewards])
                
            # Store experiences
            states.extend(current_states)
            actions.extend(env_actions)
            rewards.extend(env_rewards)
            values.extend(env_values)
            log_probs.extend(env_log_probs)
            dones.extend(env_dones)
            
            # Track completed episodes
            for i, (done, info) in enumerate(zip(env_dones, env_infos)):
                if done and 'episode_summary' in info:
                    summary = info['episode_summary']
                    self.episode_rewards.append(summary['total_return'])
                    self.episode_lengths.append(summary['episode_length'])
                    
            current_states = next_states
            
        # Convert to arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        values = np.array(values)
        log_probs = np.array(log_probs)
        dones = np.array(dones)
        
        # Reshape for parallel environments
        n_samples = len(states)
        
        # Compute advantages and returns
        advantages, returns = self._compute_gae(rewards, values, dones)
        
        return {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'advantages': advantages,
            'returns': returns,
            'values': values
        }
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        # Update running statistics
        self.reward_stats['count'] += 1
        delta = reward - self.reward_stats['mean']
        self.reward_stats['mean'] += delta / self.reward_stats['count']
        delta2 = reward - self.reward_stats['mean']
        self.reward_stats['std'] = np.sqrt(
            (self.reward_stats['std']**2 * (self.reward_stats['count'] - 1) + delta * delta2) / 
            self.reward_stats['count']
        )
        
        # Normalize
        if self.reward_stats['std'] > 0:
            normalized = (reward - self.reward_stats['mean']) / (self.reward_stats['std'] + 1e-8)
        else:
            normalized = reward
            
        return np.clip(normalized, -10, 10)
    
    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray, 
                    dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Rewards array
            values: Value predictions
            dones: Done flags
            
        Returns:
            Advantages and returns
        """
        n_steps = len(rewards)
        advantages = np.zeros(n_steps)
        
        # Get next value prediction
        if hasattr(self, 'last_state'):
            _, last_value_info = self.agent.get_action(self.last_state)
            next_value = last_value_info['value']
        else:
            next_value = 0
            
        # Compute GAE
        gae = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            
        returns = advantages + values
        
        return advantages, returns
    
    def update_policy(self, rollout_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Update policy using PPO algorithm.
        
        Args:
            rollout_data: Collected rollout data
            
        Returns:
            Update statistics
        """
        # Prepare data
        states = rollout_data['states']
        actions = rollout_data['actions']
        old_log_probs = rollout_data['log_probs']
        advantages = rollout_data['advantages']
        returns = rollout_data['returns']
        old_values = rollout_data['values']
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training statistics
        update_stats = defaultdict(float)
        n_updates = 0
        
        # Multiple epochs
        for epoch in range(self.n_epochs):
            # Create mini-batches
            indices = np.random.permutation(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # Forward pass
                batch_values, batch_log_probs, batch_entropy = self.agent.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Compute losses
                losses = self._compute_losses(
                    batch_values, batch_log_probs, batch_entropy,
                    batch_old_values, batch_old_log_probs,
                    batch_advantages, batch_returns
                )
                
                # Backward pass
                gradients = self.agent.compute_gradients(losses, batch_states, batch_actions)
                
                # Update parameters
                updated_params = self.optimizer.update(self.agent.params, gradients)
                self.agent.params = updated_params
                
                # Track statistics
                for key, value in losses.items():
                    update_stats[key] += value
                n_updates += 1
                
        # Average statistics
        for key in update_stats:
            update_stats[key] /= n_updates
            self.training_stats[key].append(update_stats[key])
            
        return dict(update_stats)
    
    def _compute_losses(self, values: np.ndarray, log_probs: np.ndarray,
                       entropy: np.ndarray, old_values: np.ndarray,
                       old_log_probs: np.ndarray, advantages: np.ndarray,
                       returns: np.ndarray) -> Dict[str, float]:
        """Compute PPO losses."""
        # Policy loss (clipped)
        ratio = np.exp(log_probs - old_log_probs)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * np.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -np.mean(np.minimum(policy_loss_1, policy_loss_2))
        
        # Value loss (clipped)
        value_pred_clipped = old_values + np.clip(
            values - old_values, -self.clip_range, self.clip_range
        )
        value_loss_1 = (values - returns) ** 2
        value_loss_2 = (value_pred_clipped - returns) ** 2
        value_loss = 0.5 * np.mean(np.maximum(value_loss_1, value_loss_2))
        
        # Entropy bonus
        entropy_loss = -np.mean(entropy)
        
        # Total loss
        total_loss = (
            policy_loss + 
            self.value_coef * value_loss + 
            self.entropy_coef * entropy_loss
        )
        
        # Additional metrics
        clip_fraction = np.mean(np.abs(ratio - 1) > self.clip_range)
        explained_variance = 1 - np.var(returns - values) / (np.var(returns) + 1e-8)
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'clip_fraction': clip_fraction,
            'explained_variance': explained_variance
        }
    
    def train(self, n_iterations: int, save_freq: int = 10,
             eval_freq: int = 10, verbose: int = 1) -> Dict:
        """
        Train PPO agent.
        
        Args:
            n_iterations: Number of training iterations
            save_freq: Model save frequency
            eval_freq: Evaluation frequency
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        training_history = defaultdict(list)
        
        if verbose > 0:
            print(f"Starting PPO training for {n_iterations} iterations")
            print(f"Parallel environments: {self.n_envs}")
            print("-" * 60)
            
        for iteration in range(n_iterations):
            start_time = time.time()
            
            # Update exploration temperature
            if self.adaptive_exploration and iteration < len(self.temperature_schedule):
                self.exploration_temperature = self.temperature_schedule[iteration]
                
            # Collect rollout
            rollout_data = self.collect_rollout()
            
            # Update policy
            update_stats = self.update_policy(rollout_data)
            
            # Add experience replay updates if enabled
            if self.experience_buffer and len(self.experience_buffer) > self.batch_size:
                replay_stats = self._update_from_replay()
                update_stats.update(replay_stats)
                
            # Track statistics
            for key, value in update_stats.items():
                training_history[key].append(value)
                
            # Episode statistics
            if len(self.episode_rewards) > 0:
                training_history['mean_episode_reward'].append(np.mean(self.episode_rewards))
                training_history['mean_episode_length'].append(np.mean(self.episode_lengths))
                
            # Evaluation
            if iteration % eval_freq == 0:
                eval_stats = self.evaluate(n_episodes=5)
                for key, value in eval_stats.items():
                    training_history[f'eval_{key}'].append(value)
                    
            # Save model
            if iteration % save_freq == 0:
                self.agent.save_model(f'checkpoint_{iteration}.npz')
                
            # Print progress
            if verbose > 0 and iteration % max(1, n_iterations // 20) == 0:
                elapsed = time.time() - start_time
                self._print_progress(iteration, n_iterations, elapsed, update_stats)
                
        # Save final model
        self.agent.save_model('final_model.npz')
        
        if verbose > 0:
            print("-" * 60)
            print("Training completed!")
            print(f"Best episode reward: {self.best_episode_reward:.4f}")
            print(f"Best Sharpe ratio: {self.best_sharpe_ratio:.2f}")
            
        return dict(training_history)
    
    def _update_from_replay(self) -> Dict[str, float]:
        """Update from experience replay buffer."""
        # Sample from buffer
        experiences, indices, weights = self.experience_buffer.sample(
            self.batch_size, beta=0.4
        )
        
        # Prepare batch
        states = np.array([e['state'] for e in experiences])
        actions = np.array([e['action'] for e in experiences])
        rewards = np.array([e['reward'] for e in experiences])
        next_states = np.array([e['next_state'] for e in experiences])
        dones = np.array([e['done'] for e in experiences])
        old_values = np.array([e['value'] for e in experiences])
        old_log_probs = np.array([e['log_prob'] for e in experiences])
        
        # Compute TD errors for priority update
        next_values = []
        for next_state in next_states:
            _, value_info = self.agent.get_action(next_state)
            next_values.append(value_info['value'])
        next_values = np.array(next_values)
        
        td_errors = rewards + self.gamma * next_values * (1 - dones) - old_values
        
        # Update priorities
        self.experience_buffer.update_priorities(indices, td_errors)
        
        # Compute advantages
        advantages = td_errors
        returns = rewards + self.gamma * next_values * (1 - dones)
        
        # Forward pass
        values, log_probs, entropy = self.agent.evaluate_actions(states, actions)
        
        # Compute weighted losses
        ratio = np.exp(log_probs - old_log_probs)
        policy_loss = -np.mean(weights * advantages * ratio)
        value_loss = np.mean(weights * (values - returns) ** 2)
        
        return {
            'replay_policy_loss': policy_loss,
            'replay_value_loss': value_loss,
            'replay_td_error': np.mean(np.abs(td_errors))
        }
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate agent performance.
        
        Args:
            n_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        episode_sharpes = []
        episode_drawdowns = []
        win_rates = []
        
        for episode in range(n_episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                # Get action (deterministic)
                action, _ = self.agent.get_action(state, deterministic=True)
                state, reward, done, info = self.env.step(action)
                
            # Get episode metrics
            summary = self.env.get_episode_summary()
            episode_rewards.append(summary['total_return'])
            episode_lengths.append(self.env.state.step)
            episode_sharpes.append(summary.get('sharpe_ratio', 0))
            episode_drawdowns.append(summary.get('max_drawdown', 0))
            win_rates.append(summary.get('win_rate', 0))
            
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_sharpe': np.mean(episode_sharpes),
            'mean_drawdown': np.mean(episode_drawdowns),
            'mean_win_rate': np.mean(win_rates),
            'mean_episode_length': np.mean(episode_lengths)
        }
    
    def _print_progress(self, iteration: int, total_iterations: int,
                       elapsed: float, stats: Dict[str, float]):
        """Print training progress."""
        print(f"\nIteration {iteration}/{total_iterations} - {elapsed:.1f}s")
        
        if len(self.episode_rewards) > 0:
            print(f"  Episode Reward: {np.mean(self.episode_rewards):.4f} "
                  f"(+/- {np.std(self.episode_rewards):.4f})")
                  
        print(f"  Policy Loss: {stats.get('policy_loss', 0):.4f}")
        print(f"  Value Loss: {stats.get('value_loss', 0):.4f}")
        print(f"  Entropy: {-stats.get('entropy_loss', 0):.4f}")
        print(f"  Clip Fraction: {stats.get('clip_fraction', 0):.3f}")
        print(f"  Explained Variance: {stats.get('explained_variance', 0):.3f}")
        
        if self.adaptive_exploration:
            print(f"  Exploration Temperature: {self.exploration_temperature:.3f}")
            
    def adapt_to_market(self, recent_data: pd.DataFrame,
                       adaptation_steps: int = 50) -> Dict[str, float]:
        """
        Adapt agent to recent market conditions.
        
        Args:
            recent_data: Recent market data
            adaptation_steps: Number of adaptation steps
            
        Returns:
            Adaptation metrics
        """
        # Create temporary environment with recent data
        adapt_env = TradingEnvironment(
            data=recent_data,
            initial_capital=self.env.initial_capital,
            lookback_window=self.env.lookback_window
        )
        
        # Store original environment
        original_env = self.env
        self.env = adapt_env
        
        # Reduce learning rate for adaptation
        original_lr = self.optimizer.learning_rate
        self.optimizer.learning_rate *= 0.1
        
        # Increase exploration for adaptation
        original_temp = self.exploration_temperature
        self.exploration_temperature = 0.5
        
        print("Adapting to recent market conditions...")
        
        # Fine-tune
        adaptation_history = self.train(
            n_iterations=adaptation_steps,
            save_freq=adaptation_steps + 1,
            eval_freq=adaptation_steps + 1,
            verbose=0
        )
        
        # Restore original settings
        self.env = original_env
        self.optimizer.learning_rate = original_lr
        self.exploration_temperature = original_temp
        
        # Evaluate adaptation
        eval_metrics = self.evaluate(n_episodes=5)
        
        print(f"Adaptation complete:")
        print(f"  Mean Return: {eval_metrics['mean_reward']:.4f}")
        print(f"  Sharpe Ratio: {eval_metrics['mean_sharpe']:.2f}")
        print(f"  Max Drawdown: {eval_metrics['mean_drawdown']:.2%}")
        
        return eval_metrics
    
    def get_training_summary(self) -> Dict:
        """Get comprehensive training summary."""
        summary = {
            'total_episodes': len(self.episode_rewards) * self.n_envs if self.parallel_env else len(self.episode_rewards),
            'best_episode_reward': self.best_episode_reward,
            'best_sharpe_ratio': self.best_sharpe_ratio,
            'final_learning_rate': self.optimizer.learning_rate,
            'exploration_temperature': self.exploration_temperature,
            'buffer_size': len(self.experience_buffer) if self.experience_buffer else 0,
            'performance_metrics': {
                'recent_mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'recent_std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0,
                'recent_mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0
            }
        }
        
        # Add latest training statistics
        for key, values in self.training_stats.items():
            if len(values) > 0:
                summary[f'final_{key}'] = values[-1]
                summary[f'mean_{key}'] = np.mean(values)
                
        return summary