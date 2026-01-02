import numpy as np
from typing import Dict, Tuple, Optional, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


class PPOAgent:
    """
    Proximal Policy Optimization agent for cryptocurrency trading.
    
    Features:
    - Actor-Critic architecture
    - Continuous action space
    - PPO clipping for stable updates
    - GAE (Generalized Advantage Estimation)
    - Custom implementation without external dependencies
    """
    
    def __init__(self,
                 state_dim: Tuple[int, ...],
                 action_dim: int,
                 hidden_size: int = 256,
                 learning_rate: float = 3e-4,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Observation space dimensions
            action_dim: Action space dimension
            hidden_size: Hidden layer size
            learning_rate: Learning rate
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Flatten state dimension for network input
        self.input_dim = np.prod(state_dim)
        
        # Initialize networks
        self.actor_params = self._init_actor_network()
        self.critic_params = self._init_critic_network()
        
        # Initialize optimizers
        self.actor_optimizer = self._init_optimizer(self.actor_params)
        self.critic_optimizer = self._init_optimizer(self.critic_params)
        
    def _init_actor_network(self) -> Dict[str, np.ndarray]:
        """Initialize actor network parameters."""
        params = {}
        
        # Input layer
        params['W1'] = self._xavier_init((self.hidden_size, self.input_dim))
        params['b1'] = np.zeros((self.hidden_size, 1))
        
        # Hidden layer
        params['W2'] = self._xavier_init((self.hidden_size, self.hidden_size))
        params['b2'] = np.zeros((self.hidden_size, 1))
        
        # Output layer (mean and log_std for continuous actions)
        params['W_mean'] = self._xavier_init((self.action_dim, self.hidden_size))
        params['b_mean'] = np.zeros((self.action_dim, 1))
        
        params['W_log_std'] = self._xavier_init((self.action_dim, self.hidden_size))
        params['b_log_std'] = np.zeros((self.action_dim, 1))
        
        return params
    
    def _init_critic_network(self) -> Dict[str, np.ndarray]:
        """Initialize critic network parameters."""
        params = {}
        
        # Input layer
        params['W1'] = self._xavier_init((self.hidden_size, self.input_dim))
        params['b1'] = np.zeros((self.hidden_size, 1))
        
        # Hidden layer
        params['W2'] = self._xavier_init((self.hidden_size, self.hidden_size))
        params['b2'] = np.zeros((self.hidden_size, 1))
        
        # Output layer (state value)
        params['W3'] = self._xavier_init((1, self.hidden_size))
        params['b3'] = np.zeros((1, 1))
        
        return params
    
    def _xavier_init(self, shape: Tuple[int, int]) -> np.ndarray:
        """Xavier weight initialization."""
        fan_in, fan_out = shape[1], shape[0]
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    
    def _init_optimizer(self, params: Dict[str, np.ndarray]) -> Dict:
        """Initialize Adam optimizer state."""
        optimizer_state = {
            'm': {},
            'v': {},
            't': 0
        }
        
        for param_name in params:
            optimizer_state['m'][param_name] = np.zeros_like(params[param_name])
            optimizer_state['v'][param_name] = np.zeros_like(params[param_name])
            
        return optimizer_state
    
    def forward_actor(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through actor network.
        
        Args:
            states: Batch of states
            
        Returns:
            Tuple of (action means, action log_stds)
        """
        # Flatten states
        batch_size = states.shape[0]
        states_flat = states.reshape(batch_size, -1)
        
        # Layer 1
        h1 = self._relu(states_flat @ self.actor_params['W1'].T + self.actor_params['b1'].T)
        
        # Layer 2  
        h2 = self._relu(h1 @ self.actor_params['W2'].T + self.actor_params['b2'].T)
        
        # Output
        means = h2 @ self.actor_params['W_mean'].T + self.actor_params['b_mean'].T
        log_stds = h2 @ self.actor_params['W_log_std'].T + self.actor_params['b_log_std'].T
        
        # Clip log_stds for stability
        log_stds = np.clip(log_stds, -2, 2)
        
        return means, log_stds
    
    def forward_critic(self, states: np.ndarray) -> np.ndarray:
        """
        Forward pass through critic network.
        
        Args:
            states: Batch of states
            
        Returns:
            State values
        """
        # Flatten states
        batch_size = states.shape[0]
        states_flat = states.reshape(batch_size, -1)
        
        # Layer 1
        h1 = self._relu(states_flat @ self.critic_params['W1'].T + self.critic_params['b1'].T)
        
        # Layer 2
        h2 = self._relu(h1 @ self.critic_params['W2'].T + self.critic_params['b2'].T)
        
        # Output
        values = h2 @ self.critic_params['W3'].T + self.critic_params['b3'].T
        
        return values.squeeze()
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Sample action from policy.
        
        Args:
            state: Current state
            deterministic: Use deterministic policy
            
        Returns:
            Tuple of (action, action_info)
        """
        # Add batch dimension if needed
        if state.ndim == len(self.state_dim):
            state = state[np.newaxis, ...]
            
        # Get action distribution
        means, log_stds = self.forward_actor(state)
        
        if deterministic:
            actions = means
        else:
            # Sample from Gaussian
            stds = np.exp(log_stds)
            actions = means + stds * np.random.randn(*means.shape)
            
        # Compute log probability
        log_probs = self._gaussian_log_prob(actions, means, log_stds)
        
        # Get value estimate
        values = self.forward_critic(state)
        
        # Squeeze batch dimension
        actions = actions.squeeze(0)
        log_probs = log_probs.squeeze(0)
        values = values.squeeze()
        
        action_info = {
            'log_prob': log_probs,
            'value': values,
            'mean': means.squeeze(0),
            'std': np.exp(log_stds).squeeze(0)
        }
        
        return actions, action_info
    
    def _gaussian_log_prob(self, actions: np.ndarray, means: np.ndarray,
                          log_stds: np.ndarray) -> np.ndarray:
        """Calculate log probability of actions under Gaussian policy."""
        stds = np.exp(log_stds)
        
        # Log probability of Gaussian
        log_probs = -0.5 * (
            ((actions - means) / stds) ** 2 +
            2 * log_stds +
            np.log(2 * np.pi)
        )
        
        # Sum over action dimensions
        return np.sum(log_probs, axis=-1)
    
    def compute_gae(self, rewards: np.ndarray, values: np.ndarray,
                   next_values: np.ndarray, dones: np.ndarray,
                   gamma: float = 0.99, lam: float = 0.95) -> np.ndarray:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Episode rewards
            values: State value estimates
            next_values: Next state value estimates
            dones: Episode termination flags
            gamma: Discount factor
            lam: GAE lambda parameter
            
        Returns:
            Advantages
        """
        n_steps = len(rewards)
        advantages = np.zeros(n_steps)
        last_advantage = 0
        
        for t in reversed(range(n_steps)):
            if dones[t]:
                next_value = 0
            else:
                next_value = next_values[t]
                
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = last_advantage = delta + gamma * lam * last_advantage * (1 - dones[t])
            
        return advantages
    
    def update(self, states: np.ndarray, actions: np.ndarray,
              old_log_probs: np.ndarray, advantages: np.ndarray,
              returns: np.ndarray, epochs: int = 10,
              batch_size: int = 64) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Log probabilities from old policy
            advantages: Advantage estimates
            returns: Return estimates
            epochs: Number of update epochs
            batch_size: Mini-batch size
            
        Returns:
            Dictionary of losses
        """
        n_samples = len(states)
        indices = np.arange(n_samples)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for epoch in range(epochs):
            # Shuffle data
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                means, log_stds = self.forward_actor(batch_states)
                values = self.forward_critic(batch_states)
                
                # Calculate new log probabilities
                new_log_probs = self._gaussian_log_prob(batch_actions, means, log_stds)
                
                # Calculate ratio
                ratio = np.exp(new_log_probs - batch_old_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                # Actor loss
                actor_loss = -np.mean(np.minimum(surr1, surr2))
                
                # Value loss
                value_loss = np.mean((values - batch_returns) ** 2)
                
                # Entropy bonus
                entropy = self._gaussian_entropy(log_stds)
                
                # Total loss
                total_loss = (actor_loss + 
                            self.value_coef * value_loss - 
                            self.entropy_coef * entropy)
                
                # Compute gradients (simplified - using numerical gradients)
                actor_grads = self._compute_actor_gradients(
                    batch_states, batch_actions, batch_advantages,
                    batch_old_log_probs
                )
                critic_grads = self._compute_critic_gradients(
                    batch_states, batch_returns
                )
                
                # Update parameters
                self._update_parameters(self.actor_params, actor_grads,
                                      self.actor_optimizer)
                self._update_parameters(self.critic_params, critic_grads,
                                      self.critic_optimizer)
                
                total_actor_loss += actor_loss
                total_critic_loss += value_loss
                total_entropy += entropy
                
        n_updates = epochs * (n_samples // batch_size)
        
        return {
            'actor_loss': total_actor_loss / n_updates,
            'critic_loss': total_critic_loss / n_updates,
            'entropy': total_entropy / n_updates
        }
    
    def _gaussian_entropy(self, log_stds: np.ndarray) -> float:
        """Calculate entropy of Gaussian distribution."""
        return np.mean(0.5 + 0.5 * np.log(2 * np.pi) + log_stds)
    
    def _compute_actor_gradients(self, states: np.ndarray, actions: np.ndarray,
                               advantages: np.ndarray,
                               old_log_probs: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute actor gradients using proper backpropagation."""
        batch_size = states.shape[0]
        states_flat = states.reshape(batch_size, -1)
        
        # Forward pass with intermediate activations
        # Layer 1
        z1 = states_flat @ self.actor_params['W1'].T + self.actor_params['b1'].T
        h1 = self._relu(z1)
        
        # Layer 2
        z2 = h1 @ self.actor_params['W2'].T + self.actor_params['b2'].T
        h2 = self._relu(z2)
        
        # Output
        means = h2 @ self.actor_params['W_mean'].T + self.actor_params['b_mean'].T
        log_stds = h2 @ self.actor_params['W_log_std'].T + self.actor_params['b_log_std'].T
        log_stds = np.clip(log_stds, -2, 2)
        
        # Calculate new log probabilities and ratio
        new_log_probs = self._gaussian_log_prob(actions, means, log_stds)
        ratio = np.exp(new_log_probs - old_log_probs)
        clipped_ratio = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        # PPO objective gradient
        use_clipped = (advantages >= 0) * (ratio > 1 + self.clip_epsilon) + \
                     (advantages < 0) * (ratio < 1 - self.clip_epsilon)
        
        # Gradient of log probability w.r.t. means and log_stds
        stds = np.exp(log_stds)
        
        # For unclipped samples
        grad_log_prob_means = (actions - means) / (stds ** 2)
        grad_log_prob_log_stds = ((actions - means) ** 2 / (stds ** 2) - 1)
        
        # Apply advantage weighting and clipping mask
        grad_multiplier = advantages[:, np.newaxis] * ratio[:, np.newaxis] * (~use_clipped[:, np.newaxis])
        
        # Gradients for output layer
        grad_means = grad_multiplier * grad_log_prob_means / batch_size
        grad_log_stds = grad_multiplier * grad_log_prob_log_stds / batch_size
        
        # Add entropy gradient for log_stds
        grad_log_stds += self.entropy_coef / batch_size
        
        # Backpropagate through output layer
        gradients = {}
        gradients['W_mean'] = grad_means.T @ h2
        gradients['b_mean'] = grad_means.T.sum(axis=1, keepdims=True)
        gradients['W_log_std'] = grad_log_stds.T @ h2
        gradients['b_log_std'] = grad_log_stds.T.sum(axis=1, keepdims=True)
        
        # Gradient for h2
        grad_h2 = grad_means @ self.actor_params['W_mean'] + grad_log_stds @ self.actor_params['W_log_std']
        
        # Backpropagate through layer 2
        grad_z2 = grad_h2 * (z2 > 0)  # ReLU gradient
        gradients['W2'] = grad_z2.T @ h1
        gradients['b2'] = grad_z2.T.sum(axis=1, keepdims=True)
        
        # Gradient for h1
        grad_h1 = grad_z2 @ self.actor_params['W2']
        
        # Backpropagate through layer 1
        grad_z1 = grad_h1 * (z1 > 0)  # ReLU gradient
        gradients['W1'] = grad_z1.T @ states_flat
        gradients['b1'] = grad_z1.T.sum(axis=1, keepdims=True)
        
        return gradients
    
    def _compute_critic_gradients(self, states: np.ndarray,
                                returns: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute critic gradients using proper backpropagation."""
        batch_size = states.shape[0]
        states_flat = states.reshape(batch_size, -1)
        
        # Forward pass with intermediate activations
        # Layer 1
        z1 = states_flat @ self.critic_params['W1'].T + self.critic_params['b1'].T
        h1 = self._relu(z1)
        
        # Layer 2
        z2 = h1 @ self.critic_params['W2'].T + self.critic_params['b2'].T
        h2 = self._relu(z2)
        
        # Output
        values = (h2 @ self.critic_params['W3'].T + self.critic_params['b3'].T).squeeze()
        
        # Value loss gradient
        grad_values = 2 * (values - returns) / batch_size
        
        # Backpropagate through output layer
        gradients = {}
        grad_values = grad_values.reshape(-1, 1)
        gradients['W3'] = grad_values.T @ h2
        gradients['b3'] = grad_values.T.sum(axis=1, keepdims=True)
        
        # Gradient for h2
        grad_h2 = grad_values @ self.critic_params['W3']
        
        # Backpropagate through layer 2
        grad_z2 = grad_h2 * (z2 > 0)  # ReLU gradient
        gradients['W2'] = grad_z2.T @ h1
        gradients['b2'] = grad_z2.T.sum(axis=1, keepdims=True)
        
        # Gradient for h1
        grad_h1 = grad_z2 @ self.critic_params['W2']
        
        # Backpropagate through layer 1
        grad_z1 = grad_h1 * (z1 > 0)  # ReLU gradient
        gradients['W1'] = grad_z1.T @ states_flat
        gradients['b1'] = grad_z1.T.sum(axis=1, keepdims=True)
        
        return gradients
    
    def _update_parameters(self, params: Dict[str, np.ndarray],
                         gradients: Dict[str, np.ndarray],
                         optimizer_state: Dict):
        """Update parameters using Adam optimizer."""
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        
        optimizer_state['t'] += 1
        t = optimizer_state['t']
        
        for param_name in params:
            if param_name not in gradients:
                continue
                
            # Update moments
            optimizer_state['m'][param_name] = (
                beta1 * optimizer_state['m'][param_name] +
                (1 - beta1) * gradients[param_name]
            )
            optimizer_state['v'][param_name] = (
                beta2 * optimizer_state['v'][param_name] +
                (1 - beta2) * gradients[param_name] ** 2
            )
            
            # Bias correction
            m_hat = optimizer_state['m'][param_name] / (1 - beta1 ** t)
            v_hat = optimizer_state['v'][param_name] / (1 - beta2 ** t)
            
            # Update parameters
            params[param_name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def save_model(self, filepath: str):
        """Save model parameters."""
        save_dict = {
            'actor_params': self.actor_params,
            'critic_params': self.critic_params,
            'actor_optimizer': self.actor_optimizer,
            'critic_optimizer': self.critic_optimizer
        }
        np.savez(filepath, **save_dict)
        
    def load_model(self, filepath: str):
        """Load model parameters."""
        data = np.load(filepath, allow_pickle=True)
        self.actor_params = data['actor_params'].item()
        self.critic_params = data['critic_params'].item()
        self.actor_optimizer = data['actor_optimizer'].item()
        self.critic_optimizer = data['critic_optimizer'].item()