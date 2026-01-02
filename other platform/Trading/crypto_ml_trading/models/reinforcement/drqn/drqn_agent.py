import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Experience replay buffer for DRQN."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


class LSTMCell:
    """Custom LSTM cell implementation."""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights
        self.W_f = self._init_weights(input_size + hidden_size, hidden_size)  # Forget gate
        self.W_i = self._init_weights(input_size + hidden_size, hidden_size)  # Input gate
        self.W_c = self._init_weights(input_size + hidden_size, hidden_size)  # Candidate
        self.W_o = self._init_weights(input_size + hidden_size, hidden_size)  # Output gate
        
        # Biases
        self.b_f = np.ones(hidden_size) * 0.1  # Forget gate bias (initialized to 1)
        self.b_i = np.zeros(hidden_size)
        self.b_c = np.zeros(hidden_size)
        self.b_o = np.zeros(hidden_size)
        
    def _init_weights(self, in_size: int, out_size: int) -> np.ndarray:
        """Initialize weights using Xavier initialization."""
        return np.random.randn(in_size, out_size) * np.sqrt(2.0 / in_size)
    
    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through LSTM cell.
        
        Args:
            x: Input (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size)
            c_prev: Previous cell state (batch_size, hidden_size)
            
        Returns:
            h: New hidden state
            c: New cell state
        """
        # Concatenate input and previous hidden state
        combined = np.concatenate([x, h_prev], axis=1)
        
        # Gates
        f_t = self._sigmoid(np.dot(combined, self.W_f) + self.b_f)  # Forget gate
        i_t = self._sigmoid(np.dot(combined, self.W_i) + self.b_i)  # Input gate
        c_tilde = np.tanh(np.dot(combined, self.W_c) + self.b_c)    # Candidate values
        o_t = self._sigmoid(np.dot(combined, self.W_o) + self.b_o)  # Output gate
        
        # Update cell state
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Update hidden state
        h_t = o_t * np.tanh(c_t)
        
        return h_t, c_t
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class DRQNAgent:
    """
    Deep Recurrent Q-Network (DRQN) Agent.
    
    Uses LSTM to handle partial observability in trading environments.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 3,  # buy, sell, hold
        hidden_size: int = 128,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 32,
        sequence_length: int = 10,
        buffer_size: int = 10000
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        
        # LSTM cell
        self.lstm = LSTMCell(state_size, hidden_size)
        
        # Q-value output layer
        self.W_q = self._init_weights(hidden_size, action_size)
        self.b_q = np.zeros(action_size)
        
        # Target network (for stability)
        self.target_lstm = LSTMCell(state_size, hidden_size)
        self.target_W_q = self.W_q.copy()
        self.target_b_q = self.b_q.copy()
        self._copy_weights_to_target()
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Hidden state memory for episodes
        self.h = None
        self.c = None
        
        # Training metrics
        self.losses = []
        
    def _init_weights(self, in_size: int, out_size: int) -> np.ndarray:
        """Initialize weights."""
        return np.random.randn(in_size, out_size) * np.sqrt(2.0 / in_size)
    
    def _copy_weights_to_target(self):
        """Copy weights from main network to target network."""
        # Copy LSTM weights
        self.target_lstm.W_f = self.lstm.W_f.copy()
        self.target_lstm.W_i = self.lstm.W_i.copy()
        self.target_lstm.W_c = self.lstm.W_c.copy()
        self.target_lstm.W_o = self.lstm.W_o.copy()
        self.target_lstm.b_f = self.lstm.b_f.copy()
        self.target_lstm.b_i = self.lstm.b_i.copy()
        self.target_lstm.b_c = self.lstm.b_c.copy()
        self.target_lstm.b_o = self.lstm.b_o.copy()
        
        # Copy Q-network weights
        self.target_W_q = self.W_q.copy()
        self.target_b_q = self.b_q.copy()
    
    def reset_state(self):
        """Reset LSTM hidden state for new episode."""
        self.h = np.zeros((1, self.hidden_size))
        self.c = np.zeros((1, self.hidden_size))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if self.h is None:
            self.reset_state()
        
        # Epsilon-greedy action selection
        if training and np.random.random() < self.epsilon:
            return np.random.choice(self.action_size)
        
        # Reshape state
        state = state.reshape(1, -1)
        
        # Forward through LSTM
        self.h, self.c = self.lstm.forward(state, self.h, self.c)
        
        # Get Q-values
        q_values = np.dot(self.h, self.W_q) + self.b_q
        
        return np.argmax(q_values[0])
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self) -> Optional[float]:
        """
        Train the model on a batch of experiences.
        
        Returns:
            Average loss for the batch
        """
        if len(self.memory) < self.batch_size:
            return None
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        # Initialize hidden states
        h = np.zeros((self.batch_size, self.hidden_size))
        c = np.zeros((self.batch_size, self.hidden_size))
        target_h = np.zeros((self.batch_size, self.hidden_size))
        target_c = np.zeros((self.batch_size, self.hidden_size))
        
        # Forward pass through LSTM
        h, c = self.lstm.forward(states, h, c)
        
        # Get current Q-values
        q_values = np.dot(h, self.W_q) + self.b_q
        current_q = q_values[np.arange(self.batch_size), actions]
        
        # Forward pass through target network
        target_h, target_c = self.target_lstm.forward(next_states, target_h, target_c)
        next_q_values = np.dot(target_h, self.target_W_q) + self.target_b_q
        
        # Calculate target Q-values
        max_next_q = np.max(next_q_values, axis=1)
        target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Calculate loss (MSE)
        loss = np.mean((current_q - target_q) ** 2)
        self.losses.append(loss)
        
        # Backpropagation (simplified gradient descent)
        # Gradient of loss w.r.t Q-values
        d_q = 2 * (current_q - target_q) / self.batch_size
        
        # Update Q-network weights
        d_W_q = np.zeros_like(self.W_q)
        d_b_q = np.zeros_like(self.b_q)
        
        for i in range(self.batch_size):
            d_W_q[:, actions[i]] += h[i] * d_q[i]
            d_b_q[actions[i]] += d_q[i]
        
        # Apply gradients
        self.W_q -= self.learning_rate * d_W_q
        self.b_q -= self.learning_rate * d_b_q
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def update_target_network(self):
        """Update target network with current network weights."""
        self._copy_weights_to_target()
    
    def save(self, filepath: str):
        """Save model weights."""
        model_data = {
            'lstm_W_f': self.lstm.W_f,
            'lstm_W_i': self.lstm.W_i,
            'lstm_W_c': self.lstm.W_c,
            'lstm_W_o': self.lstm.W_o,
            'lstm_b_f': self.lstm.b_f,
            'lstm_b_i': self.lstm.b_i,
            'lstm_b_c': self.lstm.b_c,
            'lstm_b_o': self.lstm.b_o,
            'W_q': self.W_q,
            'b_q': self.b_q,
            'epsilon': self.epsilon,
            'losses': self.losses
        }
        np.savez(filepath, **model_data)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model weights."""
        data = np.load(filepath)
        
        self.lstm.W_f = data['lstm_W_f']
        self.lstm.W_i = data['lstm_W_i']
        self.lstm.W_c = data['lstm_W_c']
        self.lstm.W_o = data['lstm_W_o']
        self.lstm.b_f = data['lstm_b_f']
        self.lstm.b_i = data['lstm_b_i']
        self.lstm.b_c = data['lstm_b_c']
        self.lstm.b_o = data['lstm_b_o']
        self.W_q = data['W_q']
        self.b_q = data['b_q']
        self.epsilon = float(data['epsilon'])
        self.losses = list(data['losses'])
        
        # Update target network
        self._copy_weights_to_target()
        
        logger.info(f"Model loaded from {filepath}")