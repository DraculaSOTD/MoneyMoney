import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class TradingEnvironment:
    """
    Trading environment for DRQN agent.
    
    Features:
    - Position management (long only for simplicity)
    - Transaction costs
    - Risk management (stop-loss, take-profit)
    - Reward shaping for better learning
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000,
        position_size: float = 0.1,  # 10% of portfolio per trade
        transaction_cost: float = 0.001,  # 0.1% per trade
        stop_loss: float = 0.05,  # 5% stop loss
        take_profit: float = 0.10,  # 10% take profit
        max_holding_period: int = 100,  # Maximum steps to hold a position
        reward_scaling: float = 100.0
    ):
        self.data = data
        self.initial_balance = initial_balance
        self.position_size = position_size
        self.transaction_cost = transaction_cost
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_holding_period = max_holding_period
        self.reward_scaling = reward_scaling
        
        # State variables
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0: no position, 1: long position
        self.entry_price = 0
        self.entry_step = 0
        self.portfolio_value = self.initial_balance
        self.last_portfolio_value = self.initial_balance
        
        # Trading statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.max_drawdown = 0
        self.peak_value = self.initial_balance
        
        # Action tracking
        self.last_action = 2  # Start with hold
        self.consecutive_losses = 0
        self.last_trade_step = 0
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation including market data and position info.
        
        Returns:
            Normalized observation vector
        """
        if self.current_step >= len(self.data):
            return np.zeros(self._get_observation_size())
        
        # Get market features from data
        market_features = self.data.iloc[self.current_step].values
        
        # Add position and portfolio features
        position_features = np.array([
            self.position,  # Current position
            self.balance / self.initial_balance,  # Normalized balance
            self.portfolio_value / self.initial_balance,  # Normalized portfolio value
            (self.current_step - self.entry_step) / 100 if self.position else 0,  # Holding period
            (self.data.iloc[self.current_step]['close'] - self.entry_price) / self.entry_price if self.position and self.entry_price > 0 else 0,  # Unrealized P&L
            self.winning_trades / (self.total_trades + 1),  # Win rate
            self.consecutive_losses / 10,  # Normalized consecutive losses
        ])
        
        # Combine features
        observation = np.concatenate([market_features, position_features])
        
        # Normalize
        observation = observation / (np.linalg.norm(observation) + 1e-8)
        
        return observation
    
    def _get_observation_size(self) -> int:
        """Get size of observation vector."""
        return len(self.data.columns) + 7  # Market features + position features
    
    def _calculate_reward(self, action: int, prev_portfolio_value: float) -> float:
        """
        Calculate reward with multiple components.
        
        Args:
            action: Action taken
            prev_portfolio_value: Portfolio value before action
            
        Returns:
            Shaped reward
        """
        reward = 0
        
        # Base reward: change in portfolio value
        portfolio_change = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        reward += portfolio_change * self.reward_scaling
        
        # Transaction cost penalty
        if action in [0, 1] and self.last_action != action:
            reward -= self.transaction_cost * self.reward_scaling
        
        # Risk management rewards
        if self.position == 0:
            # Reward for closing position with profit
            if hasattr(self, 'last_trade_return') and self.last_trade_return > 0:
                reward += self.last_trade_return * self.reward_scaling * 0.5
            # Reward for avoiding losses
            elif hasattr(self, 'last_trade_return') and self.last_trade_return < -self.stop_loss:
                reward += 0.1 * self.reward_scaling  # Small reward for cutting losses
        
        # Holding period penalty (encourage decisive action)
        if self.position == 1:
            holding_period = self.current_step - self.entry_step
            if holding_period > self.max_holding_period:
                reward -= 0.01 * self.reward_scaling
        
        # Win rate bonus
        if self.total_trades > 10:
            win_rate = self.winning_trades / self.total_trades
            if win_rate > 0.6:
                reward += 0.02 * self.reward_scaling
        
        # Consecutive losses penalty
        if self.consecutive_losses > 3:
            reward -= 0.05 * self.reward_scaling * self.consecutive_losses
        
        # Overtrading penalty
        if self.current_step - self.last_trade_step < 5 and action in [0, 1]:
            reward -= 0.03 * self.reward_scaling
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action in environment.
        
        Args:
            action: 0=buy, 1=sell, 2=hold
            
        Returns:
            observation, reward, done, info
        """
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, {}
        
        prev_portfolio_value = self.portfolio_value
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute action
        if action == 0 and self.position == 0:  # Buy
            # Open long position
            position_value = self.balance * self.position_size
            shares = position_value / current_price
            
            # Apply transaction cost
            cost = position_value * self.transaction_cost
            self.balance -= position_value + cost
            
            self.position = 1
            self.entry_price = current_price
            self.entry_step = self.current_step
            self.shares = shares
            self.last_trade_step = self.current_step
            
        elif action == 1 and self.position == 1:  # Sell
            # Close long position
            position_value = self.shares * current_price
            
            # Apply transaction cost
            cost = position_value * self.transaction_cost
            self.balance += position_value - cost
            
            # Calculate trade return
            self.last_trade_return = (current_price - self.entry_price) / self.entry_price
            self.total_profit += position_value - (self.shares * self.entry_price)
            
            # Update statistics
            self.total_trades += 1
            if self.last_trade_return > 0:
                self.winning_trades += 1
                self.consecutive_losses = 0
            else:
                self.losing_trades += 1
                self.consecutive_losses += 1
            
            # Reset position
            self.position = 0
            self.entry_price = 0
            self.shares = 0
            self.last_trade_step = self.current_step
        
        # Force exit on stop-loss or take-profit
        if self.position == 1:
            current_return = (current_price - self.entry_price) / self.entry_price
            holding_period = self.current_step - self.entry_step
            
            if (current_return <= -self.stop_loss or 
                current_return >= self.take_profit or
                holding_period >= self.max_holding_period):
                # Force sell
                self.step(1)
                
        # Update portfolio value
        if self.position == 1:
            self.portfolio_value = self.balance + self.shares * current_price
        else:
            self.portfolio_value = self.balance
        
        # Update max drawdown
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Calculate reward
        reward = self._calculate_reward(action, prev_portfolio_value)
        
        # Update state
        self.last_action = action
        self.last_portfolio_value = self.portfolio_value
        self.current_step += 1
        
        # Check if done
        done = (self.current_step >= len(self.data) - 1 or 
                self.balance <= 0 or
                self.portfolio_value <= self.initial_balance * 0.5)  # 50% loss stops episode
        
        # Prepare info
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'position': self.position,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / (self.total_trades + 1e-8),
            'total_profit': self.total_profit,
            'max_drawdown': self.max_drawdown
        }
        
        return self._get_observation(), reward, done, info
    
    def render(self):
        """Display current state."""
        if self.current_step < len(self.data):
            current_price = self.data.iloc[self.current_step]['close']
            logger.info(f"Step: {self.current_step}, Price: {current_price:.2f}, "
                       f"Portfolio: ${self.portfolio_value:.2f}, Position: {self.position}")