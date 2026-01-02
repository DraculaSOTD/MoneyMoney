"""
PPO (Proximal Policy Optimization) Trainer for Trading
Implements PPO algorithm with trading-specific enhancements
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
import logging
from tqdm import tqdm
import json
from pathlib import Path

# Import environment
from environments.trading_gym_env import TradingGymEnvironment

logger = logging.getLogger(__name__)


class TradingActorCritic(nn.Module):
    """
    Actor-Critic network for PPO trading agent.
    Handles both continuous and discrete action spaces.
    """
    
    def __init__(self,
                 observation_shape: Tuple[int, ...],
                 action_dim: int,
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 discrete_actions: bool = False,
                 use_lstm: bool = True):
        """
        Initialize Actor-Critic network.
        
        Args:
            observation_shape: Shape of observations
            action_dim: Dimension of action space
            hidden_size: Hidden layer size
            num_layers: Number of hidden layers
            discrete_actions: Whether to use discrete actions
            use_lstm: Whether to use LSTM for temporal processing
        """
        super().__init__()
        
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.discrete_actions = discrete_actions
        self.use_lstm = use_lstm
        
        # Feature extraction
        if len(observation_shape) == 2:  # (sequence_len, features)
            input_size = observation_shape[1]
            
            if use_lstm:
                self.feature_extractor = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.1 if num_layers > 1 else 0
                )
                feature_size = hidden_size
            else:
                # Flatten for MLP
                self.feature_extractor = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(np.prod(observation_shape), hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                feature_size = hidden_size
        else:
            raise ValueError(f"Unsupported observation shape: {observation_shape}")
        
        # Shared layers
        self.shared_net = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head
        if discrete_actions:
            self.actor_head = nn.Linear(hidden_size, action_dim)
        else:
            # Continuous actions: output mean and log_std
            self.actor_mean = nn.Linear(hidden_size, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic head
        self.critic_head = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Forward pass through the network.
        
        Args:
            obs: Observations
            hidden: Hidden state for LSTM
            
        Returns:
            action_probs/dist, value, hidden_state
        """
        # Feature extraction
        if self.use_lstm and isinstance(self.feature_extractor, nn.LSTM):
            if hidden is None:
                features, hidden = self.feature_extractor(obs)
                features = features[:, -1, :]  # Take last timestep
            else:
                features, hidden = self.feature_extractor(obs, hidden)
                features = features[:, -1, :]
        else:
            features = self.feature_extractor(obs)
            hidden = None
        
        # Shared processing
        shared_features = self.shared_net(features)
        
        # Actor output
        if self.discrete_actions:
            action_logits = self.actor_head(shared_features)
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
        else:
            action_mean = self.actor_mean(shared_features)
            action_std = self.actor_log_std.exp().expand_as(action_mean)
            dist = Normal(action_mean, action_std)
        
        # Critic output
        value = self.critic_head(shared_features)
        
        return dist, value, hidden
    
    def get_action(self, obs: torch.Tensor, 
                   hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                   deterministic: bool = False):
        """Get action from the policy."""
        dist, value, hidden = self.forward(obs, hidden)
        
        if deterministic:
            if self.discrete_actions:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.mean
        else:
            action = dist.sample()
        
        action_log_prob = dist.log_prob(action)
        
        return action, action_log_prob, value, hidden
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for PPO update."""
        dist, values, _ = self.forward(obs)
        
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        
        return values, action_log_probs, dist_entropy


class PPOTrainer:
    """
    PPO trainer for trading environments.
    Implements PPO with trading-specific enhancements.
    """
    
    def __init__(self,
                 env: TradingGymEnvironment,
                 config: Optional[Dict] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize PPO trainer.
        
        Args:
            env: Trading environment
            config: Training configuration
            device: Training device
        """
        self.env = env
        self.device = device
        self.config = config or self._default_config()
        
        # Create networks
        obs_shape = env.observation_space.shape
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_dim = env.action_space.n
            discrete = True
        else:
            action_dim = env.action_space.shape[0]
            discrete = False
        
        self.actor_critic = TradingActorCritic(
            observation_shape=obs_shape,
            action_dim=action_dim,
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            discrete_actions=discrete,
            use_lstm=self.config['use_lstm']
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=self.config['learning_rate'],
            eps=1e-5
        )
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.best_reward = -float('inf')
        
        # Logging
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_stats = []
    
    def _default_config(self) -> Dict:
        """Default PPO configuration."""
        return {
            # Network
            'hidden_size': 256,
            'num_layers': 2,
            'use_lstm': True,
            
            # PPO
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
            'num_envs': 1,  # Can be increased for parallel training
            
            # Exploration
            'exploration_fraction': 0.1,
            'exploration_final_eps': 0.01,
            
            # Saving
            'save_freq': 10,
            'eval_freq': 5
        }
    
    def collect_rollouts(self) -> Dict[str, torch.Tensor]:
        """Collect rollouts from the environment."""
        rollout_buffer = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        obs = self.env.reset()
        hidden = None
        
        for _ in range(self.config['num_steps']):
            # Convert to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get action
            with torch.no_grad():
                action, log_prob, value, hidden = self.actor_critic.get_action(
                    obs_tensor, hidden
                )
            
            # Step environment
            action_np = action.cpu().numpy().squeeze()
            next_obs, reward, done, info = self.env.step(action_np)
            
            # Store transition
            rollout_buffer['observations'].append(obs)
            rollout_buffer['actions'].append(action_np)
            rollout_buffer['rewards'].append(reward)
            rollout_buffer['values'].append(value.cpu().numpy().squeeze())
            rollout_buffer['log_probs'].append(log_prob.cpu().numpy().squeeze())
            rollout_buffer['dones'].append(done)
            
            obs = next_obs
            
            if done:
                # Log episode statistics
                episode_stats = self.env.get_episode_statistics()
                self.episode_rewards.append(episode_stats['total_return'])
                self.episode_lengths.append(self.env.current_step)
                self.episode_count += 1
                
                # Reset
                obs = self.env.reset()
                hidden = None
        
        # Convert to tensors
        for key in rollout_buffer:
            rollout_buffer[key] = np.array(rollout_buffer[key])
        
        # Compute returns and advantages
        rollout_buffer = self._compute_returns_and_advantages(rollout_buffer)
        
        return rollout_buffer
    
    def _compute_returns_and_advantages(self, rollout_buffer: Dict) -> Dict:
        """Compute returns and GAE advantages."""
        rewards = rollout_buffer['rewards']
        values = rollout_buffer['values']
        dones = rollout_buffer['dones']
        
        # Bootstrap value if not done
        with torch.no_grad():
            last_obs = torch.FloatTensor(
                self.env._get_observation()
            ).unsqueeze(0).to(self.device)
            _, last_value, _ = self.actor_critic(last_obs)
            last_value = last_value.cpu().numpy().squeeze()
        
        # Compute GAE
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
                next_done = dones[t]
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]
            
            delta = rewards[t] + self.config['gamma'] * next_value * (1 - next_done) - values[t]
            gae = delta + self.config['gamma'] * self.config['gae_lambda'] * (1 - next_done) * gae
            
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        rollout_buffer['advantages'] = advantages
        rollout_buffer['returns'] = returns
        
        return rollout_buffer
    
    def update(self, rollout_buffer: Dict) -> Dict[str, float]:
        """Update policy using PPO."""
        # Convert to tensors
        obs = torch.FloatTensor(rollout_buffer['observations']).to(self.device)
        actions = torch.FloatTensor(rollout_buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_buffer['log_probs']).to(self.device)
        advantages = torch.FloatTensor(rollout_buffer['advantages']).to(self.device)
        returns = torch.FloatTensor(rollout_buffer['returns']).to(self.device)
        
        # Training statistics
        pg_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []
        
        # PPO epochs
        for epoch in range(self.config['num_epochs']):
            # Create random batches
            indices = np.random.permutation(len(obs))
            
            for start in range(0, len(obs), self.config['batch_size']):
                end = start + self.config['batch_size']
                batch_indices = indices[start:end]
                
                # Get batch
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                values, log_probs, entropy = self.actor_critic.evaluate_actions(
                    batch_obs, batch_actions
                )
                
                # Policy loss (PPO clip)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config['clip_param'],
                    1.0 + self.config['clip_param']
                ) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values = values.squeeze()
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.config['value_loss_coef'] * value_loss +
                    self.config['entropy_coef'] * entropy_loss
                )
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(),
                    self.config['max_grad_norm']
                )
                self.optimizer.step()
                
                # Statistics
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                # Clip fraction
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1) > self.config['clip_param']).float()
                ).item()
                clip_fractions.append(clip_fraction)
        
        # Update global step
        self.global_step += len(rollout_buffer['observations'])
        
        return {
            'policy_loss': np.mean(pg_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'clip_fraction': np.mean(clip_fractions)
        }
    
    def train(self, total_timesteps: int, 
              eval_env: Optional[TradingGymEnvironment] = None):
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Total training timesteps
            eval_env: Environment for evaluation
        """
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        
        num_updates = total_timesteps // self.config['num_steps']
        
        for update in tqdm(range(num_updates), desc="PPO Training"):
            # Collect rollouts
            rollout_buffer = self.collect_rollouts()
            
            # Update policy
            update_stats = self.update(rollout_buffer)
            
            # Log statistics
            if self.episode_rewards:
                avg_reward = np.mean(self.episode_rewards)
                avg_length = np.mean(self.episode_lengths)
                
                stats = {
                    'timestep': self.global_step,
                    'episode': self.episode_count,
                    'avg_reward': avg_reward,
                    'avg_length': avg_length,
                    **update_stats
                }
                
                self.training_stats.append(stats)
                
                # Print progress
                if update % 10 == 0:
                    logger.info(
                        f"Update {update}/{num_updates} | "
                        f"Timestep {self.global_step} | "
                        f"Avg Reward: {avg_reward:.2f} | "
                        f"Policy Loss: {update_stats['policy_loss']:.4f}"
                    )
            
            # Evaluate
            if eval_env and update % self.config['eval_freq'] == 0:
                eval_stats = self.evaluate(eval_env, num_episodes=5)
                logger.info(f"Evaluation - Avg Return: {eval_stats['avg_return']:.2f}")
                
                # Save best model
                if eval_stats['avg_return'] > self.best_reward:
                    self.best_reward = eval_stats['avg_return']
                    self.save(f'best_ppo_model.pth')
            
            # Save checkpoint
            if update % self.config['save_freq'] == 0:
                self.save(f'ppo_checkpoint_{update}.pth')
    
    def evaluate(self, env: TradingGymEnvironment, 
                 num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the trained agent."""
        episode_rewards = []
        episode_lengths = []
        episode_stats = []
        
        for _ in range(num_episodes):
            obs = env.reset()
            hidden = None
            done = False
            
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, _, _, hidden = self.actor_critic.get_action(
                        obs_tensor, hidden, deterministic=True
                    )
                
                action_np = action.cpu().numpy().squeeze()
                obs, reward, done, info = env.step(action_np)
            
            # Get episode statistics
            stats = env.get_episode_statistics()
            episode_rewards.append(stats['total_return'])
            episode_lengths.append(env.current_step)
            episode_stats.append(stats)
        
        return {
            'avg_return': np.mean(episode_rewards),
            'std_return': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'sharpe_ratio': np.mean([s['sharpe_ratio'] for s in episode_stats]),
            'max_drawdown': np.mean([s['max_drawdown'] for s in episode_stats]),
            'win_rate': np.mean([s['win_rate'] for s in episode_stats])
        }
    
    def save(self, path: str):
        """Save the model and training state."""
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'best_reward': self.best_reward,
            'config': self.config
        }, path)
        
        # Save training statistics
        stats_path = Path(path).with_suffix('.json')
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
    
    def load(self, path: str):
        """Load the model and training state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.episode_count = checkpoint['episode_count']
        self.best_reward = checkpoint['best_reward']
        
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])


# Import gym after defining the trainer to avoid circular imports
import gym